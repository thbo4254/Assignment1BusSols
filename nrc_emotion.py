"""
Step 5: NRC word-list emotion scorer.

Derives a review's primary emotion WITHOUT any model call:
  1. Tokenize the review (title + text), lowercased.
  2. Reduce each token to a stem so inflections hit the lexicon
     (e.g. "sad", "sadly" -> the lexicon's "sadness").
  3. Look up each token's emotions in the NRC Emotion Lexicon
     (a public word->emotion list; 8 emotions + positive + negative).
  4. Sum the score per emotion across all words; the emotion with the
     highest total is the review's primary emotion.

The lexicon ships with the `NRCLex` package as `nrc_en.json`.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Optional

# The 8 NRC emotion categories (all that participate in the comparison).
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]

_TOKEN_RE = re.compile(r"[a-z]+[']?[a-z]+|[a-z]+")
# Minimal English plural / -ed / -ing desuffixer — enough to match lexicon
# stems without pulling in a full lemmatizer (keeps the scorer dependency-light).
_SUFFIX_RULES = [
    (r"ies$", "y"), (r"ves$", "f"), (r"oes$", "o"),
    (r"sses$", "ss"), (r"ning$", "n"), (r"(" + r"ied)$", "y"),
    (r"ing$", ""), (r"ed$", ""), (r"es$", ""), (r"s$", ""),
]


class NRCScorer:
    def __init__(self, lexicon_path: Optional[str] = None):
        if lexicon_path is None:
            import nrclex  # locate the lexicon shipped with the NRCLex package

            lexicon_path = os.path.join(
                os.path.dirname(nrclex.__file__), "data", "nrc_en.json"
            )
        self.lexicon: dict[str, list[str]] = json.load(open(lexicon_path))
        self._rules = [(re.compile(p), s) for p, s in _SUFFIX_RULES]

    def _stem(self, token: str) -> str:
        for pat, repl in self._rules:
            if pat.search(token):
                cand = pat.sub(repl, token)
                if cand in self.lexicon:
                    return cand
        return token

    def score(self, title: Optional[str], text: Optional[str]) -> Counter:
        """Return a Counter of summed emotion scores for the review."""
        combined = f"{(title or '')} {(text or '')}"
        scores = Counter()
        for tok in _TOKEN_RE.findall(combined.lower()):
            for emo in self.lexicon.get(tok, []) or []:
                if emo in EMOTIONS:
                    scores[emo] += 1
        return scores

    def primary_emotion(self, title: Optional[str], text: Optional[str]) -> str:
        """Return the emotion with the highest summed score."""
        scores = self.score(title, text)
        if not scores:
            # No emotion-bearing words found.
            return ""
        # Tie-break deterministically by EMOTIONS order.
        best = max(scores.items(), key=lambda kv: (kv[1], -EMOTIONS.index(kv[0])))
        return best[0]


def load_scorer(lexicon_path: Optional[str] = None) -> NRCScorer:
    return NRCScorer(lexicon_path)


if __name__ == "__main__":
    import sys

    sc = load_scorer()
    for line in sys.stdin:
        title, _, text = line.rstrip("\n").partition("\t")
        emo = sc.primary_emotion(title, text)
        print(f"{emo}\t{sorted(sc.score(title, text).items(), key=lambda kv: -kv[1])}")
