"""Prepare email text and extract six numerical features for spam detection.

Expected input: data/emails.csv with columns:
    text, label, original_id (original_id is strongly recommended)

Labels may be spam/legitimate (or ham), or 1/0. Each variation of the same
starting email should share the same original_id so leakage can be detected.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "emails.csv"
DEFAULT_OUTPUT = ROOT / "data" / "features.csv"

WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

MONEY_TERMS = {
    "money", "cash", "dollar", "dollars", "usd", "payment", "prize",
    "reward", "winnings", "bonus", "refund", "credit", "gift", "payout",
}
URGENCY_TERMS = {
    "urgent", "immediately", "now", "today", "asap", "expires", "expiring",
    "deadline", "act", "limited", "hurry", "final", "soon", "rightaway",
}
LABELS = {
    "spam": 1, "junk": 1, "1": 1,
    "legitimate": 0, "legit": 0, "ham": 0, "not spam": 0, "0": 0,
}


def normalize_label(value: object) -> int:
    key = str(value).strip().lower()
    if key not in LABELS:
        raise ValueError(
            f"Unknown label {value!r}. Use spam/legitimate or 1/0."
        )
    return LABELS[key]


def extract_features(text: object) -> dict[str, int]:
    email = "" if pd.isna(text) else str(text)
    words = WORD_RE.findall(email)
    lowered = [word.lower() for word in words]
    caps = [word for word in words if len(word) > 1 and word.isalpha() and word.isupper()]
    return {
        "word_count": len(words),
        "exclamations": email.count("!"),
        "money_words": sum(word in MONEY_TERMS for word in lowered),
        "all_caps": len(caps),
        "link_count": len(URL_RE.findall(email)),
        "urgency_words": sum(word in URGENCY_TERMS for word in lowered),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract numerical email features.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="CSV containing email text and labels")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Where to save the feature CSV")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Could not find {args.input}. Run `python generate_dataset.py` first, or "
            "place the instructor-provided labelled template at data/emails.csv."
        )

    df = pd.read_csv(args.input)
    # Accept common alternatives for the text column.
    text_column = next(
        (name for name in ("text", "email_text", "body", "message", "email")
         if name in df.columns), None
    )
    if text_column is None or "label" not in df.columns:
        raise ValueError(
            "Input CSV must include a text column (text/email_text/body/message/email) "
            "and a label column."
        )
    if df[text_column].isna().any():
        raise ValueError("Some email text cells are blank. Please fill or remove them.")

    df = df.copy()
    df["text"] = df[text_column].astype(str)
    df["label"] = df["label"].apply(normalize_label)
    if "original_id" not in df.columns:
        # This fallback permits a basic run, but cannot identify variations.
        df["original_id"] = [f"email_{i + 1}" for i in range(len(df))]
        print("WARNING: original_id was not provided; near-copy grouping may be limited.")
    df["original_id"] = df["original_id"].fillna(" ").astype(str).str.strip()
    if (df["original_id"] == "").any():
        raise ValueError("original_id contains blank values. Give every email a source ID.")

    feature_frame = pd.DataFrame([extract_features(value) for value in df["text"]])
    result = pd.concat(
        [df[["original_id", "text", "label"]].reset_index(drop=True), feature_frame],
        axis=1,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    counts = result["label"].map({1: "spam", 0: "legitimate"}).value_counts()
    print(f"Prepared {len(result)} emails -> {args.output}")
    print(f"Spam: {int((result['label'] == 1).sum())}; "
          f"legitimate: {int((result['label'] == 0).sum())}")
    print("Features:", ", ".join(feature_frame.columns))
    if len(result) != 16:
        print("NOTE: The assignment template describes 16 original labelled emails; "
              "check that your input includes the intended rows/variations.")


if __name__ == "__main__":
    main()
