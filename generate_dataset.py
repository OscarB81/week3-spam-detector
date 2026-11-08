"""Create small, clearly labelled email variations from 16 seed examples.

This is a student-created fallback dataset, not an instructor-provided template.
Replace data/original_emails.csv with the instructor's file if it becomes available.
Each original and its variations retain the same original_id for leakage checks.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEED_PATH = ROOT / "data" / "original_emails.csv"
OUTPUT_PATH = ROOT / "data" / "emails.csv"


def make_variations(text: str) -> list[str]:
    """Return the original plus three minor edits that keep the message meaning."""
    variants = [text]
    variants.append("Hello, " + text)
    variants.append(text + " Thank you.")
    if text.endswith("."):
        variants.append(text[:-1] + "!")
    else:
        variants.append(text + "!")
    return variants


def main() -> None:
    if not SEED_PATH.exists():
        raise FileNotFoundError(f"Missing seed dataset: {SEED_PATH}")

    seeds = pd.read_csv(SEED_PATH)
    required = {"original_id", "label", "text"}
    if not required.issubset(seeds.columns):
        raise ValueError(f"Seed file must contain columns: {sorted(required)}")
    if len(seeds) != 16:
        raise ValueError(f"Expected 16 original emails; found {len(seeds)}.")
    if seeds["label"].astype(str).str.lower().value_counts().to_dict() not in (
        {"spam": 8, "legitimate": 8}, {"legitimate": 8, "spam": 8}
    ):
        raise ValueError("Seed dataset must contain exactly 8 spam and 8 legitimate emails.")

    rows = []
    for row in seeds.itertuples(index=False):
        for version, text in enumerate(make_variations(str(row.text)), start=1):
            rows.append({
                "original_id": str(row.original_id),
                "variation": version - 1,
                "label": str(row.label).lower(),
                "text": text,
            })
    result = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Created {len(result)} email rows from {len(seeds)} originals: {OUTPUT_PATH}")
    print("Each original has four versions (the original plus three small variations).")
    print("Labels:", result["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
