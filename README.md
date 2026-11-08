# Week 3 — Spam Detector

A small supervised-learning project that classifies email examples using six engineered numerical features.

## Important dataset note

The assignment screenshots refer to an instructor-provided template of 16 labelled emails, but no template or download link was visible in the materials available to me. To keep the project moving, this repository includes a **student-created fallback dataset**: 16 example emails (8 spam and 8 legitimate), plus three small variations of each. This is not represented as the instructor's original template. Replace `data/original_emails.csv` with the instructor's source examples if they are later provided, then rerun the generator.

## 1. Generate the data

```bash
python generate_dataset.py
```

This reads `data/original_emails.csv` and creates `data/emails.csv` with 64 rows. Each original and its three variations share the same `original_id`, allowing the train/test leakage check to find related messages across the split.

## 2. Install dependencies

```bash
python -m pip install pandas numpy matplotlib scikit-learn joblib
```

## 3. Prepare features

```bash
python prepare_data.py
```

This creates `data/features.csv` containing the four required features plus `link_count` and `urgency_words`.

## 4. Train and evaluate

```bash
python spam_detector.py
```

The script prints accuracy, precision, recall, all four confusion-matrix counts, feature weights, three adversarial test results, and a train/test near-copy check. It saves generated results under `reports/`.

## Feature definitions

| Feature | What it counts |
|---|---|
| `word_count` | Number of word tokens in the email |
| `exclamations` | Number of `!` characters |
| `money_words` | Matches to a small list of money/reward-related terms |
| `all_caps` | All-uppercase alphabetic words with at least two letters |
| `link_count` | URLs beginning with `http://`, `https://`, or `www.` |
| `urgency_words` | Matches to a list of urgency/pressure terms |

These are simple signals rather than definitive indicators of spam.

## GitHub submission

Commit the source files, `data/original_emails.csv`, the generated `data/emails.csv`, and `analysis_report.md` unless your instructor directs otherwise. Do not commit virtual environments, caches, or generated model binaries unless asked. Replace report placeholders with the results from your own run, and state clearly that the included dataset is a fallback created because the instructor template was not available.
