"""
Step 6: Three-class scoring with balanced sampling.

Rating Class
  4-5  POSITIVE
  3    NEUTRAL
  1-2  NEGATIVE

Instead of reading the first N rows (which under-represents NEUTRAL and NEGATIVE),
this pulls a BALANCED group from the whole 152K-review file: a roughly equal
number (default ~50) from each class, chosen with a fixed random seed so the
same set comes up every run. The label is derived ONLY from the rating, compared
against the LLM's POSITIVE/NEUTRAL/NEGATIVE prediction (which never sees the
rating). Reports 3-class accuracy, a 3x3 confusion matrix, per-class metrics,
and explicitly whether the NEUTRAL class holds up or collapses into others.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from collections import Counter

from classifier import classify_with_emotion, client, LABELS


def rating_to_class(rating: float) -> str:
    if rating >= 4:
        return "POSITIVE"
    if rating == 3:
        return "NEUTRAL"
    return "NEGATIVE"


def load_all(path: str):
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            rec["_idx"] = i
            yield rec


def build_balanced_sample(path: str, per_class: int, seed: int,
                          cache_file: str) -> list[dict]:
    """Reproducibly sample ~per_class rows from each rating class."""
    # Bucket every review by class.
    buckets: dict[str, list[int]] = {c: [] for c in LABELS}
    for rec in load_all(path):
        buckets[rating_to_class(rec["rating"])].append(rec["_idx"])

    rng = random.Random(seed)
    # Try to load cached chosen indices (so re-runs reuse the exact same set).
    try:
        with open(cache_file) as f:
            cached = [int(x) for x in f.read().split()]
        chosen_all = cached
    except FileNotFoundError:
        chosen = {c: rng.sample(idx, min(per_class, len(idx))) for c, idx in buckets.items()}
        chosen_all = sorted(v for vals in chosen.values() for v in vals)
        with open(cache_file, "w") as f:
            f.write(" ".join(str(x) for x in chosen_all))

    chosen_set = set(chosen_all)
    picked = [
        {**rec, "_idx": rec["_idx"]}
        for rec in load_all(path)
        if rec["_idx"] in chosen_set
    ]
    # Order by class then by index for stable reading.
    picked.sort(key=lambda r: (rating_to_class(r["rating"]), r["_idx"]))
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default="Gift_Cards.jsonl.gz")
    ap.add_argument("--cache", default="step6_sample.txt")
    ap.add_argument("--out", default="step6_scores.csv")
    args = ap.parse_args()

    rows = build_balanced_sample(args.data, args.per_class, args.seed, args.cache)
    class_counts = Counter(rating_to_class(r["rating"]) for r in rows)
    print(f"Balanced sample loaded: {len(rows)} rows  "
          f"({', '.join(f'{c}:{class_counts[c]}' for c in LABELS)})")

    cl = client()
    recs = []
    for i, r in enumerate(rows, 1):
        true_label = rating_to_class(r["rating"])
        pred_label, emotion = classify_with_emotion(r.get("title"), r.get("text"), cl=cl)
        recs.append({
            "row": r["_idx"],
            "rating": r["rating"],
            "true_label": true_label,
            "pred_label": pred_label,
            "emotion": emotion,
            "correct": pred_label == true_label,
            "title": (r.get("title") or ""),
            "text": (r.get("text") or "")[:200],
        })
        print(f"[{i:{2}}/{len(rows)}] row={r['_idx']:>7} rating={r['rating']} "
              f"true={true_label:8s} pred={pred_label:8s} "
              f"{'OK' if pred_label == true_label else 'MISS'}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        for r in recs:
            w.writerow(r)

    # ---- Summary ----
    n = len(recs)
    correct = sum(1 for r in recs if r["correct"])
    print("\n" + "=" * 62)
    print(f"Step 6 — three-class, balanced (per-class={args.per_class}, seed={args.seed})")
    print(f"Rows: {n}  (balanced mix: {dict(class_counts)})")
    print(f"Saved to: {args.out}")
    print("=" * 62)

    print(f"\nOverall accuracy: {correct/n:.1%} ({correct}/{n})")

    print("\nPer-class accuracy:")
    for c in LABELS:
        members = [r for r in recs if r["true_label"] == c]
        if not members:
            print(f"  {c:8s}: (none in sample)")
            continue
        hit = sum(1 for r in members if r["correct"])
        print(f"  {c:8s}: {hit}/{len(members)} = {hit/len(members):.1%}   (of {len(members)} true {c})")

    # Confusion matrix
    print("\nConfusion matrix (rows=true label, cols=predicted):")
    hdr = f"{'':16s}" + "".join(f"{c:>11s}" for c in LABELS)
    print(hdr)
    for tc in LABELS:
        line = f"{tc + ' (true)':16s}"
        for pc in LABELS:
            line += f"{sum(1 for r in recs if r['true_label']==tc and r['pred_label']==pc):>11d}"
        print(line)

    # Where do NEUTRAL predictions pile up? and NEUTRAL truths go?
    print("\nNEUTRAL focus (the question):")
    neut_true = [r for r in recs if r["true_label"] == "NEUTRAL"]
    neut_pred = [r for r in recs if r["pred_label"] == "NEUTRAL"]
    if neut_true:
        dist = Counter(r["pred_label"] for r in neut_true)
        rec = sum(1 for r in neut_true if r["correct"])
        print(f"  True NEUTRAL ({len(neut_true)}) predicted as: {dict(dist)}  => kept NEUTRAL {rec}/{len(neut_true)} ({rec/len(neut_true):.0%})")
    if neut_pred:
        src = Counter(r["true_label"] for r in neut_pred)
        print(f"  Predicted NEUTRAL ({len(neut_pred)}) came from true: {dict(src)}")

    # Balanced vs imbalanced comparison
    total_true = Counter(r["true_label"] for r in recs)
    total_pred = Counter(r["pred_label"] for r in recs)
    print(f"\nTrue-label mix (balanced sample): {dict(total_true)}")
    print(f"Pred-label mix:                   {dict(total_pred)}")
    majority = max(total_true, key=total_true.get)
    print(f"\nMajority-only baseline = {total_true[majority]/n:.1%} (always '{majority}')")
    print(f"Model beats baseline by {(correct/n - total_true[majority]/n):+.1%}")


if __name__ == "__main__":
    main()
