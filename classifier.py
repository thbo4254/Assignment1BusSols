"""
Reusable binary sentiment classifier for Amazon review title+text.

Classifies a review as POSITIVE or NEGATIVE using an LLM reached through an
OpenAI-compatible endpoint. The prompt does NOT see or rely on the star rating.

Emotion component: the model first reads the emotional tone of the review
(happy, angry, frustrated, grateful, etc.) and uses that tone as an intermediate
Input when deciding POSITIVE / NEGATIVE.

Output is machine-readable: the model is asked to reply with a JSON object
carrying a single `label` of "POSITIVE" or "NEGATIVE". We wrap that in a
parser that is tolerant of stray text/whitespace so callers can read the
answer back programmatically.
"""

from __future__ import annotations

import json
import os
import re
from typing import Literal

import openai

Label = Literal["POSITIVE", "NEGATIVE"]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert Amazon-review sentiment analyst. You will be given a product \
review consisting of a title and a body. Your job is to classify the overall \
sentiment of the review as either POSITIVE or NEGATIVE.

Important rules:
1. Judge ONLY the title and body text supplied to you. Do NOT invent or assume \
any star rating, price, or product facts that are not written in the review.
2. First assess the emotional tone of the review — e.g. whether it conveys \
happiness, satisfaction, gratitude, excitement, relief, neutrality, \
disappointment, frustration, anger, or regret. Use that tone to guide your \
verdict, but don't stop there: weigh what the reviewer is actually saying.
3. A POSITIVE review recommends, praises, or is satisfied with the product or \
experience. A NEGATIVE review complains, warns against purchase, reports a \
defective/failing product, or expresses clear dissatisfaction.
4. The review is about the *reviewer's experience*, not a third party's. A \
review that merely describes someone else being happy is still judged by its \
own tone and content.
5. Edge cases — handle these as follows:
   - Conflicting title and text: the body text generally carries more weight \
than the title, but judge the review as a whole and pick the dominant sentiment.
   - Terse reviews ("Great!", "Do not buy!!"): give clear emotional one-liners \
their plain meaning.
   - Angry but ultimately approving reviews: if the buyer is satisfied overall, \
classify POSITIVE even if the tone is gruff; anger about the product or a failed \
purchase is NEGATIVE.
   - Neutral / mixed / undecided reviews MUST still be resolved to POSITIVE or \
NEGATIVE — there is no third option, so lean on the dominant directional signal.

Respond with ONLY a single JSON object in exactly this format, and nothing else.
The "label" is the sentiment verdict; the "emotion" is the single primary emotion
you detect most strongly in the review, chosen from exactly this set:
anger, anticipation, disgust, fear, joy, sadness, surprise, trust.
{"label": "POSITIVE", "emotion": "joy"}   or   {"label": "NEGATIVE", "emotion": "fear"}

Always include both keys, even if the emotion is weak — pick the strongest of the
eight. If truly undecided, fall back to "trust" for positive or "fear" for negative."""


def build_user_prompt(title: str | None, text: str | None) -> str:
    """Compose the review-then-ask prompt for one review."""
    title = (title or "").strip()
    text = (text or "").strip()
    if not title and not text:
        # Degenerate case: nothing to judge. Keep it visible to the model.
        review_block = "(no title / text supplied)"
    else:
        review_block = ""
        if title:
            review_block += f"Title: {title}\n"
        if text:
            review_block += f"Body: {text}\n"
        review_block = review_block.rstrip()

    return (
        "Classify the sentiment of the following Amazon review and detect its "
        "primary emotion. Consider the emotional tone of both the title and the "
        "body, then decide the overall sentiment (POSITIVE or NEGATIVE) and the "
        "single strongest emotion chosen from: anger, anticipation, disgust, "
        "fear, joy, sadness, surprise, trust.\n\n"
        f"{review_block}\n\n"
        'Reply with a single JSON object, e.g. {"label": "POSITIVE", '
        '"emotion": "joy"} or {"label": "NEGATIVE", "emotion": "fear"} — '
        "include BOTH keys."
    )


# ---------------------------------------------------------------------------
# Configuration (OpenAI-compatible endpoint) — fill in / export to point the
# client at the class endpoint.
# ---------------------------------------------------------------------------

def client() -> openai.OpenAI:
    """Build a client from config; reads env vars so no secrets are hardcoded."""
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))
    return openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Prediction + parsing
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(r'"label"\s*:\s*"?(POSITIVE|NEGATIVE)"?', re.IGNORECASE)
_EMOTION_RE = re.compile(r'"emotion"\s*:\s*"([a-zA-Z]+)"')
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness",
            "surprise", "trust"]


def classify_review(title: str | None, text: str | None, *, cl=None, model: str | None = None) -> Label:
    """Classify a single review. Returns 'POSITIVE' or 'NEGATIVE'."""
    return classify_with_emotion(title, text, cl=cl, model=model)[0]


def classify_with_emotion(title: str | None, text: str | None, *, cl=None,
                          model: str | None = None) -> tuple[Label, str]:
    """Classify a review and detect its primary emotion — (label, emotion)."""
    cl = cl or client()
    model = model or MODEL
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(title, text)},
    ]
    resp = cl.chat.completions.create(model=model, messages=messages, temperature=0)
    answer = resp.choices[0].message.content or ""
    return parse_label(answer), parse_emotion(answer)


def parse_emotion(raw: str) -> str:
    """Extract the primary emotion from a model reply, tolerating prose."""
    raw = raw.strip()
    m = _EMOTION_RE.search(raw)
    if m:
        return m.group(1).lower()
    # Fallback: any NRC emotion word appearing in the reply.
    found = [e for e in EMOTIONS if re.search(rf"\b{e}\b", raw, re.IGNORECASE)]
    if found:
        # Return the one that appears last in the reply (most likely the stated answer).
        last = max(found, key=lambda e: raw.lower().rfind(e))
        return last
    raise ValueError(f"Could not parse an emotion from model reply: {raw!r}")


def parse_label(raw: str) -> Label:
    """Programmatically read POSITIVE/NEGATIVE out of a model reply."""
    raw = raw.strip()
    # Prefer an explicit JSON label.
    m = _LABEL_RE.search(raw)
    if m:
        return m.group(1).upper()
    # Fallback: bare token anywhere in the reply.
    tokens = re.findall(r"\b(POSITIVE|NEGATIVE)\b", raw, re.IGNORECASE)
    if tokens:
        return tokens[-1].upper()
    raise ValueError(f"Could not parse a POSITIVE/NEGATIVE label from model reply: {raw!r}")


# ---------------------------------------------------------------------------
# CLI: classify a file of reviews, or a one-off review given on the command line.
# ---------------------------------------------------------------------------

def _load_reviews(path: str, limit: int | None = None):
    import gzip

    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            yield json.loads(line)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Classify Amazon reviews as POSITIVE/NEGATIVE.")
    parser.add_argument("file", nargs="?", help="Path to (optionally .gz) JSONL of reviews; else pass --title/--text.")
    parser.add_argument("--title", help="Review title (single-review mode).")
    parser.add_argument("--text", help="Review body (single-review mode).")
    parser.add_argument("--limit", type=int, default=None, help="Only read this many reviews from file.")
    args = parser.parse_args(argv)

    cl = client()
    if args.title is not None or args.text is not None:
        label = classify_review(args.title, args.text, cl=cl)
        print(label)
        return

    if not args.file:
        parser.error("provide a file, or --title/--text")
    for rec in _load_reviews(args.file, args.limit):
        label = classify_review(rec.get("title"), rec.get("text"), cl=cl)
        print(f"{rec.get('rating','?')}\t{rec.get('asin','?')}\t{label}")


if __name__ == "__main__":
    main()
