"""
Step 2: score a 100-row batch of Amazon reviews.

Pipeline:
  1. Load the Gift_Cards dataset, build a reproducible random sample of N rows.
  2. For each sampled row, call the LLM classifier on title + text ONLY.
     The star rating is NEVER sent to the model — it is reserved for checking.
  3. Ground truth derived from the rating: rating >= 4 -> POSITIVE, else NEGATIVE.
  4. Write full per-row results to CSV/JSON as evidence, then print a summary:
     overall agreement, how often it's right on each class, the confusion
     matrix, and the specific reviews it got wrong.

CLI:
  source .venv/bin/activate
  OPENAI_BASE_URL=<url> OPENAI_MODEL=<model> OPENAI_API_KEY=<key> \
      python score_batch.py [--n 100] [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import sys
from datetime import datetime, timezone

from classifier import classify_review, client


def load_rows(path: str):
    """Yield every review dict as-is."""
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def sample_rows(path: str, n: int, seed: int, cache_file: str):
    """Sample n rows reproducibly; cache the sampled indices so a re-run scores
    the exact same reviews (good for re-checking without changing the batch)."""
    # First pass: count rows.
    total = 0
    for _ in load_rows(path):
        total += 1

    try:
        with open(cache_file) as f:
            idxs = [int(x) for x in f.read().split()]
    except FileNotFoundError:
        rng = random.Random(seed)
        idxs = sorted(rng.sample(range(total), n))
        with open(cache_file, "w") as f:
            f.write(" ".join(str(i) for i in idxs))

    picked = {i: rec for i, rec in enumerate(load_rows(path)) if i in set(idxs)}
    # Preserve row order by index.
    return [picked[i] for i in idxs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default="Gift_Cards.jsonl.gz")
    ap.add_argument("--samples", default="batch_samples.txt")
    ap.add_argument("--out", default="batch_scores.csv")
    args = ap.parse_args()

    print(f"Loading/sampling {args.n} rows (seed={args.seed}) ...")
    rows = sample_rows(args.data, args.n, args.seed, args.samples)

    cl = client()
    results = []
    for i, r in enumerate(rows, 1):
        true_label = "POSITIVE" if r["rating"] >= 4 else "NEGATIVE"
        got = None
        error = None
        try:
            got = classify_review(r.get("title"), r.get("text"), cl=cl)
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
        results.append({
            "row": i,
            "rating": r["rating"],
            "title": r.get("title") or "",
            "text": (r.get("text") or "")[:200],
            "true_label": true_label,
            "pred_label": got,
            "correct": (got == true_label) if got else "",
            "error": error or "",
        })
        print(f"[{i}/{args.n}] rating={r['rating']} true={true_label} pred={got}")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for row in results:
            w.writerow({
                "row": row["row"], "rating": row["rating"],
                "title": row["title"], "text": row["text"],
                "true_label": row["true_label"], "pred_label": row["pred_label"],
                "correct": row["correct"],
                "error": row["error"] if row["error"] else "",
            })

    # ---- Summarize ----
    scored = [r for r in results if r["pred_label"] in ("POSITIVE", "NEGATIVE")]
    total = len(results)
    print("\n" + "=" * 60)
    print(f"Batch size: {total}  (scored: {len(scored)}, errors: {total - len(scored)})")
    print(f"Written to: {args.out}")
    print("=" * 60)

    if not scored:
        print("No rows scored — nothing to summarise.")
        sys.exit(1)

    correct = sum(1 for r in scored if r["correct"])
    acc = correct / len(scored)
    print(f"\nOverall agreement with rating: {acc:.1%} ({correct}/{len(scored)})")

    # Class composition (ground truth) to expose the skew.
    pos_n = sum(1 for r in scored if r["true_label"] == "POSITIVE")
    neg_n = len(scored) - pos_n
    print(f"\nGround-truth class mix in this batch: "
          f"{pos_n} POSITIVE ({pos_n/len(scored):.1%}), "
          f"{neg_n} NEGATIVE ({neg_n/len(scored):.1%})")

    print("\nPer-class accuracy (against rating):")
    for cls in ("POSITIVE", "NEGATIVE"):
        members = [r for r in scored if r["true_label"] == cls]
        if not members:
            print(f"  {cls:8s}: (no members in batch)")
            continue
        hit = sum(1 for r in members if r["correct"])
        print(f"  {cls:8s}: {hit}/{len(members)} = {hit/len(members):.1%}")

    # Confusion matrix
    print("\nConfusion matrix (rows=true label, cols=predicted):")
    header = f"{'':16s}{'pred POS':>10s}{'pred NEG':>10s}"
    print(header)
    for true_cls in ("POSITIVE", "NEGATIVE"):
        pp = sum(1 for r in scored if r["true_label"] == true_cls and r["pred_label"] == "POSITIVE")
        pn = sum(1 for r in scored if r["true_label"] == true_cls and r["pred_label"] == "NEGATIVE")
        print(f"{true_cls + ' (true)':16s}{pp:>10d}{pn:>10d}")

    # Wrong reviews, with full evidence.
    wrong = [r for r in scored if not r["correct"]]
    print(f"\nReviews the model got WRONG ({len(wrong)}):")
    for r in wrong:
        print(f"  row {r['row']} | rating={r['rating']} true={r['true_label']} "
              f"pred={r['pred_label']}")
        print(f"     title: {r['title'][:80]!r}")
        print(f"     text : {r['text'][:120]!r}")

    # Majority-baseline sanity check: predicts the majority class every time.
    maj_cls = "POSITIVE" if pos_n >= neg_n else "NEGATIVE"
    maj_acc = (pos_n if maj_cls == "POSITIVE" else neg_n) / len(scored)
    print(f"\nSanity / skew guardrail:")
    print(f"  Always-predict-{maj_cls} baseline accuracy = {maj_acc:.1%}")
    print(f"  Model beats baseline by {acc - maj_acc:+.1%}")
    print(f"  (Beats baseline only if model is genuinely better than just "
          f"guessing the majority class.)")


if __name__ == "__main__":
    main()
