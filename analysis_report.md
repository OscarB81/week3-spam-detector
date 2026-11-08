# Spam Detector — Analysis Report

> **Dataset note:** The assignment instructions referred to an instructor-provided template of 16 labelled emails, but no template or download link was available in the course materials I could locate. To complete a working version, I created a clearly labelled fallback dataset of 16 sample emails (8 spam and 8 legitimate) and generated three minor variations of each. These results describe that fallback dataset, not an instructor-provided dataset. If the official template is later supplied, replace `data/original_emails.csv`, rerun the scripts, and update this report.

## 1. Features and feature engineering

The model receives six numerical features rather than the full email text. The four required features are `word_count`, `exclamations`, `money_words`, and `all_caps`. I added `link_count` because unsolicited messages may direct readers to unfamiliar websites, and `urgency_words` because spam often pressures readers to act quickly. These are only signals, not proof that an email is spam; legitimate messages can contain links or urgent language too.

## 2. Training and test split

I split the 64 email rows into 75% training data and 25% test data, using a fixed random state and stratification so both labels were represented. The training set is used to fit the model. The test set is held aside to estimate how well the trained model handles examples it did not directly train on. Testing on training examples would measure performance on familiar data rather than the model's ability to generalize.

## 3. Model and results

I used a logistic regression classifier with standardized features. Standardization puts numerical features on comparable scales before the model learns its weights.

- Test accuracy: **100% (16 of 16 test rows correct)**
- Precision: **100%**
- Recall: **100%**

### Confusion matrix

Rows are the actual labels and columns are the predicted labels:

| Actual / predicted | Legitimate | Spam |
|---|---:|---:|
| Legitimate | **8 (true negatives)** | **0 (false positives)** |
| Spam | **0 (false negatives)** | **8 (true positives)** |

Precision means that, among the messages the model marked as spam, the share that really were spam was 100% in this test. Recall means that, among all spam messages in the test set, the share the model caught was 100%.

## 4. Which error matters more?

For a security-focused filter, I consider a false negative more serious. A false negative is a spam or phishing message that reaches the inbox, where a person might click a malicious link or provide sensitive information. A false positive is also serious because a legitimate message could be hidden in the spam folder. I would prioritize high recall but keep suspected messages in a reviewable spam or quarantine folder rather than automatically deleting them. This allows the filter to catch more suspicious messages while giving users a way to recover legitimate mail.

## 5. What the model learned

The largest absolute feature weight was **`money_words` (1.627)**, followed by `urgency_words` (1.218), `word_count` (-0.977), `all_caps` (0.690), `link_count` (0.214), and `exclamations` (0.161). Positive weights push the prediction toward spam, while negative weights push it toward legitimate mail. The strong positive money-word weight makes sense for this particular sample because several spam examples mention prizes, cash, rewards, payments, or refunds. The negative word-count weight is less intuitive and may be an accidental pattern in this small dataset: the legitimate examples in this sample tend to be longer. It should not be treated as a general rule that longer emails are legitimate.

## 6. Adversarial tests

I tested three deliberately tricky emails using the same features and trained model:

| Test email | Intended label | Model prediction | Result |
|---|---|---|---|
| Disguised promotional message | Spam | Legitimate | Fooled the model |
| Low-pressure reward message | Spam | Legitimate | Fooled the model |
| Legitimate urgent meeting notice | Legitimate | Spam | Fooled the model |

The two spam examples avoided the obvious patterns the model learned, such as frequent money and urgency terms. The legitimate meeting notice included uppercase and urgency language, which made it look suspicious according to the limited feature set. These tests show how a classifier can be fooled when it relies on a few counts instead of understanding message meaning and context.

## 7. Data quality and train/test leakage

Each of the 16 seed messages was expanded into four closely related versions, and each group was given the same `original_id`. The random row-level split placed **11 original IDs** in both the training and test sets. The text-similarity check also found **38 train/test pairs** with similarity of at least 0.85.

This means the model was tested on near-copies of messages it had already seen during training. Its 100% test accuracy is therefore likely inflated and does not show how well it would classify genuinely new email. A stronger evaluation would split by `original_id`, ensuring that every version of one original message stays entirely in either training or test data. The dataset is also very small and manually created, so it may not represent the variety of real spam and legitimate email.

## 8. Would I trust this filter on real email?

I would treat this model as a learning experiment, not as a production email filter. Its perfect test score is undermined by the near-duplicate leakage, and it failed all three deliberately difficult examples. It only sees six manually selected counts and cannot understand the full meaning of a message, sender reputation, context, or whether a link is actually safe. Before trusting it on real email, I would need a much larger and more varied labelled dataset, a train/test split grouped by original message, testing on recent unseen messages, monitoring for changing spam tactics, and a way for users to report mistakes.
