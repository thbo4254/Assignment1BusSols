"""
Quick spot-check for the Step 1 classifier.

Sends a small hand-picked set of obviously POSITIVE and obviously NEGATIVE
reviews (title + text only — no rating is exposed to the model) through the
real endpoint and prints, for each, the TRUE label and the model's label.

Usage:
    source .venv/bin/activate
    OPENAI_BASE_URL=<class url> OPENAI_API_KEY=<key> OPENAI_MODEL=<model> \
        python spot_check.py
"""

from __future__ import annotations

import sys

from classifier import classify_review, client

# (true_label, title, text)
CASES = [
    # --- Obviously POSITIVE ---
    (
        "POSITIVE",
        "Great Gift Pack",
        "Think it is a great gift. Was a little worried after reading the "
        "negative reviews about cards not being activated. Went out to "
        "Starbucks site and it worked fine.",
    ),
    (
        "POSITIVE",
        "I always love to get a gift card",
        "Whether for someone else or for myself, sometimes it is hard to know "
        "what to 'get' a person and you want to give them something. A gift "
        "card is always a safe, appreciated choice.",
    ),
    (
        "POSITIVE",
        "Good!",
        "It's a gift card. Think it's a great gift.",
    ),
    # --- Obviously NEGATIVE ---
    (
        "NEGATIVE",
        "Unable to use",
        "Do not buy!!!!! Cards came unable to be activated. I spent over 4 "
        "hours on the phone between Amazon and Mastercard. Amazon won't help.",
    ),
    (
        "NEGATIVE",
        "BAD BAD BAD",
        "Deserves 0 stars! The gift card can only be used to pay for things "
        "inside the app and that was not made clear. Total waste of money.",
    ),
    (
        "NEGATIVE",
        "Came ripped and bent.",
        "I had ordered this for the holder and it came bent and ripped. I have "
        "to give it today so I have no choice but to give it like that. Very "
        "disappointed.",
    ),
    # --- Edge cases ---
    (
        "NEGATIVE",  # terse / angry one-liner
        "Do not buy!!",
        "Card never worked. Scam.",
    ),
    (
        "POSITIVE",  # terse approving one-liner
        "good deal....",
        "Easy to get the card",
    ),
    (
        "POSITIVE",  # conflicting title (negative) vs positive body
        "Was worried",
        "I was really nervous this would be a scam but it arrived on time and "
        "activated instantly. Everything worked perfectly. Very happy.",
    ),
    (
        "NEGATIVE",  # favourable title but angry/failed-purchase body
        "Looks nice",
        "Looks nice on the outside but the code inside was already redeemed. "
        "Absolutely furious, this is a total rip-off.",
    ),
]


def main() -> None:
    cl = client()
    print(f"Base URL:    {cl.base_url}")
    print(f"Model:       {__import__('os').getenv('OPENAI_MODEL','gpt-4o-mini')}\n")
    correct = 0
    for true_label, title, text in CASES:
        try:
            got = classify_review(title, text, cl=cl)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR  true={true_label:8s} title={title!r:28s} -> {type(e).__name__}: {e}")
            continue
        ok = got == true_label
        correct += int(ok)
        mark = "OK " if ok else "MISS"
        print(f"{mark}  true={true_label:8s} pred={got:8s} | {title!r}")
    print(f"\n{correct}/{len(CASES)} spot-check cases correct.")
    sys.exit(0 if correct == len(CASES) else 1)


if __name__ == "__main__":
    main()
