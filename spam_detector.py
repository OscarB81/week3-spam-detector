"""Train, evaluate, and stress-test a small email spam classifier."""
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(__file__).resolve().parent
FEATURES_PATH = ROOT / "data" / "features.csv"
ARTIFACTS = ROOT / "reports"
FEATURES = [
    "word_count", "exclamations", "money_words", "all_caps",
    "link_count", "urgency_words",
]

# Two spam examples written to avoid obvious spam signals, plus a legitimate
# message designed to trigger the filter. Edit these and explain the results.
ATTACK_EMAILS = [
    {
        "attack_name": "Disguised promotional message",
        "text": "Hello Jordan, I thought you might find this update useful. "
                "Your account has a new member benefit available to review at "
                "https://member-update.example. Please take a look when convenient.",
        "intended_class": "spam",
    },
    {
        "attack_name": "Low-pressure reward message",
        "text": "A note about your recent participation: a reward is ready in your "
                "online profile. Sign in through the usual account page to see the "
                "details. This message is for your records.",
        "intended_class": "spam",
    },
    {
        "attack_name": "Legitimate urgent meeting notice",
        "text": "URGENT: Today's project meeting has moved to 2 PM. Please review "
                "the agenda now and reply ASAP if you cannot attend!!!",
        "intended_class": "legitimate",
    },
]


def find_leakage(train: pd.DataFrame, test: pd.DataFrame, threshold: float = 0.85):
    """Return exact source-ID overlaps and highly similar train/test text pairs."""
    shared_ids = sorted(set(train["original_id"]) & set(test["original_id"]))
    similar_pairs = []
    for test_index, test_row in test.iterrows():
        test_text = str(test_row["text"]).lower()
        for train_index, train_row in train.iterrows():
            train_text = str(train_row["text"]).lower()
            similarity = SequenceMatcher(None, train_text, test_text).ratio()
            if similarity >= threshold:
                similar_pairs.append({
                    "train_row": int(train_index),
                    "test_row": int(test_index),
                    "train_original_id": train_row["original_id"],
                    "test_original_id": test_row["original_id"],
                    "similarity": round(similarity, 3),
                })
    return shared_ids, similar_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the spam detector.")
    parser.add_argument("--input", type=Path, default=FEATURES_PATH)
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError("Run `python prepare_data.py` first to create data/features.csv.")
    data = pd.read_csv(args.input)
    required = set(FEATURES + ["label", "text", "original_id"])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Feature file is missing columns: {sorted(missing)}")
    if data["label"].nunique() != 2:
        raise ValueError("The dataset must contain both spam (1) and legitimate (0) labels.")

    X = data[FEATURES]
    y = data["label"].astype(int)
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, data.index, test_size=args.test_size, random_state=42, stratify=y
    )
    train_rows = data.loc[train_idx]
    test_rows = data.loc[test_idx]

    # Scaling puts count-based features on comparable scales before logistic regression.
    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    matrix = [[int(tn), int(fp)], [int(fn), int(tp)]]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_title("Spam classifier — confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], ["Legitimate", "Spam"])
    ax.set_yticks([0, 1], ["Legitimate", "Spam"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row][col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Logistic-regression coefficients show each standardized feature's direction
    # and relative influence in this particular trained model.
    classifier = model.named_steps["classifier"]
    weights = pd.DataFrame({
        "feature": FEATURES,
        "weight": classifier.coef_[0],
    })
    weights["absolute_weight"] = weights["weight"].abs()
    weights = weights.sort_values("absolute_weight", ascending=False)
    weights.to_csv(ARTIFACTS / "feature_weights.csv", index=False)

    test_results = test_rows[["original_id", "text", "label"]].copy()
    test_results["predicted_label"] = predictions
    test_results.to_csv(ARTIFACTS / "test_predictions.csv", index=False)

    # Test the three deliberately tricky emails using the same feature extraction.
    from prepare_data import extract_features
    attack_features = pd.DataFrame([extract_features(item["text"]) for item in ATTACK_EMAILS])
    attack_predictions = model.predict(attack_features[FEATURES])
    attack_probabilities = model.predict_proba(attack_features[FEATURES])[:, 1]
    attack_results = pd.DataFrame({
        "attack_name": [item["attack_name"] for item in ATTACK_EMAILS],
        "intended_class": [item["intended_class"] for item in ATTACK_EMAILS],
        "predicted_class": ["spam" if value == 1 else "legitimate" for value in attack_predictions],
        "spam_probability": attack_probabilities.round(3),
        "text": [item["text"] for item in ATTACK_EMAILS],
    })
    attack_results.to_csv(ARTIFACTS / "attack_results.csv", index=False)

    shared_ids, similar_pairs = find_leakage(train_rows, test_rows)
    pd.DataFrame(similar_pairs).to_csv(ARTIFACTS / "similarity_pairs.csv", index=False)
    joblib.dump(model, ARTIFACTS / "spam_model.joblib")

    print("\n=== Test-set performance ===")
    print(f"Accuracy:  {accuracy:.3f} ({accuracy:.1%})")
    print(f"Precision: {precision:.3f} ({precision:.1%})")
    print(f"Recall:    {recall:.3f} ({recall:.1%})")
    print("\nConfusion matrix (rows=actual [legitimate, spam]; columns=predicted [legitimate, spam]):")
    print(f"{matrix}")
    print(f"Spam correctly caught (true positives): {tp}")
    print(f"Legitimate correctly passed (true negatives): {tn}")
    print(f"Legitimate wrongly marked spam (false positives): {fp}")
    print(f"Spam wrongly let through (false negatives): {fn}")

    print("\n=== Feature weights (largest absolute weight first) ===")
    print(weights[["feature", "weight"]].to_string(index=False))
    print("\n=== Adversarial email tests ===")
    print(attack_results[["attack_name", "intended_class", "predicted_class", "spam_probability"]].to_string(index=False))

    print("\n=== Train/test overlap check ===")
    print(f"Shared original_id values: {shared_ids if shared_ids else 'None found'}")
    print(f"Near-copy pairs at similarity >= 0.85: {len(similar_pairs)}")
    if similar_pairs:
        print(pd.DataFrame(similar_pairs).to_string(index=False))
    else:
        print("No near-copy pairs crossed the 0.85 similarity threshold.")

    print("\nSaved outputs in reports/: confusion_matrix.png, feature_weights.csv, "
          "test_predictions.csv, attack_results.csv, similarity_pairs.csv, spam_model.joblib")


if __name__ == "__main__":
    main()
