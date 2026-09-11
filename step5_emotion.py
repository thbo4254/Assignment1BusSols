"""
Step 5: Primary-emotion detection — compare two independent methods.

For the same 100-review batch used in Steps 2-3:
  * Method A (LLM): the model returns both sentiment and a primary emotion
    (one of the 8 NRC emotion categories) from title+text only.
  * Method B (NRC word list): derives the primary emotion by scoring each
    review's words against the NRC emotion lexicon — no model call at all.

We keep both, then measure how often the two agree on the primary emotion,
break the agreement down, surface the divergences, and (for context) check
each method's agreement with the rating-derived sentiment. Evidence is saved
to CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter

from classifier import classify_with_emotion, client, EMOTIONS
from nrc_emotion import load_scorer
from score_batch import sample_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default="Gift_Cards.jsonl.gz")
    ap.add_argument("--samples", default="batch_samples.txt")
    ap.add_argument("--out", default="step5_emotion.csv")
    args = ap.parse_args()

    rows = sample_rows(args.data, args.n, args.seed, args.samples)
    cl = client()
    nrc = load_scorer()

    recs = []
    for i, r in enumerate(rows, 1):
        true_label = "POSITIVE" if r["rating"] >= 4 else "NEGATIVE"
        llm_label, llm_emo = classify_with_emotion(r.get("title"), r.get("text"), cl=cl)
        nrc_emo = nrc.primary_emotion(r.get("title"), r.get("text"))
        nrc_scores = dict(nrc.score(r.get("title"), r.get("text")))
        recs.append({
            "row": i,
            "rating": r["rating"],
            "true_label": true_label,
            "title": r.get("title") or "",
            "text": (r.get("text") or "")[:200],
            "llm_label": llm_label,
            "llm_emotion": llm_emo,
            "nrc_emotion": nrc_emo or "",
            "nrc_scores": json.dumps(nrc_scores),
            "agree": (llm_emo == nrc_emo),
            "llm_label_correct": (llm_label == true_label),
        })
        print(f"[{i}/{args.n}] rating={r['rating']} true={true_label} "
              f"llm=({llm_label},{llm_emo}) nrc={nrc_emo or '-'} "
              f"agree={'Y' if llm_emo == nrc_emo else 'n'}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        for r in recs:
            w.writerow(r)

    # ---- Summary ----
    n = len(recs)
    print("\n" + "=" * 62)
    print(f"Step 5 — primary-emotion comparison ({n} reviews)")
    print(f"Saved to: {args.out}")
    print("=" * 62)

    # 1. Does the NRC method find anything?
    nrc_any = sum(1 for r in recs if r["nrc_emotion"])
    print(f"\nNRC found an emotion in {nrc_any}/{n} reviews "
          f"({nrc_any/n:.1%}); {n - nrc_any} had no emotion-bearing words.")

    # 2. Agreement between the two methods (only where NRC found an emotion)
    paired = [r for r in recs if r["nrc_emotion"]]
    agree = sum(1 for r in paired if r["agree"])
    print(f"\nEmotion agreement (LLM vs NRC), of {len(paired)} reviews NRC could score:")
    print(f"  Exact same primary emotion: {agree}/{len(paired)} = {agree/len(paired):.1%}")

    # 3. Agreement broken down by LLM label
    for lab in ("POSITIVE", "NEGATIVE"):
        sub = [r for r in paired if r["llm_label"] == lab]
        if not sub:
            continue
        a = sum(1 for r in sub if r["agree"])
        print(f"    {lab:8s}: {a}/{len(sub)} = {a/len(sub):.1%}")

    # 4. Where the LLM picked which emotion, vs NRC
    print("\n  LLM emotion distribution:", dict(Counter(r["llm_emotion"] for r in recs)))
    print("  NRC  emotion distribution:", dict(Counter(r["nrc_emotion"] for r in recs)))

    # 5. Agreement-to-sentiment sanity (context): LLM sentiment vs rating
    llm_sent_acc = sum(1 for r in recs if r["llm_label_correct"]) / n
    print(f"\n  Context — LLM label vs rating: {llm_sent_acc:.1%} (same as Step 2)")

    # 6. Divergences: show the llm emotion vs nrc emotion table where they differ
    diffs = [r for r in paired if not r["agree"]]
    print(f"\nDivergences ({len(diffs)}):  LLM_emotion → NRC_emotion (from score {nrc_emotion_max(r)})")
    from collections import defaultdict
    grid = defaultdict(int)
    for r in diffs:
        grid[(r["llm_emotion"], r["nrc_emotion"])] += 1
    for (le, ne), c in sorted(grid.items(), key=lambda kv: -kv[1]):
        print(f"    {le:12s} → {ne:12s}  ×{c}")

    # 7. A few concrete examples
    print("\n  Three sample divergences:")
    shown = 0
    for r in diffs:
        if shown >= 3:
            break
        print(f"    row {r['row']} rating={r['rating']} | LLM:{r['llm_emotion']} "
              f"NRC:{r['nrc_emotion']}  ({r['title'][:40]!r})")
        shown += 1


def nrc_emotion_max(r) -> str:
    try:
        d = json.loads(r["nrc_scores"])
        if not d:
            return "(no score)"
        top = max(d.items(), key=lambda kv: kv[1])
        return f"{top[0]}={top[1]}"
    except Exception:
        return "?"


if __name__ == "__main__":
    main()
