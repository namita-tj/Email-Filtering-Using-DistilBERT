"""Datasets and splits, shared by training and evaluation.

Main dataset: Enron-Spam (Metsis et al., 2006), as published on Hugging Face by
SetFit: about 31.7k training and 2k test emails with subject and body.

    train       90% of the published train split, used to fit the model
    validation  10% of the published train split, used for early metrics and
                to choose the confidence band
    test        the published test split, used only by evaluation/run.py

Keeping the test split out of threshold selection means the reported KPIs are
measured on emails the system never tuned on.

Out-of-distribution test: a modern phishing-email dataset (Kaggle "Phishing
Email Detection", mirrored on Hugging Face). It is only ever used for testing,
to check whether a model trained on 2000s spam still catches today's phishing.
"""

MAIN_DATASET = "SetFit/enron_spam"
PHISHING_DATASET = "zefang-liu/phishing-email-dataset"
SEED = 42
VALIDATION_SHARE = 0.1
MAX_BODY_CHARS = 20_000  # same limit as the API; a few source emails are megabytes long
PLACEHOLDER_TEXTS = {"", "empty", "null", "nan", "none"}

# Keep these imports below the constants: they are light, and the dataset
# library itself is only imported inside the loaders.
from app.preprocessing import combine_subject_and_body  # noqa: E402


def prepare_enron(example: dict) -> dict:
    subject = example.get("subject") or ""
    body = (example.get("message") or "")[:MAX_BODY_CHARS]
    return {
        "subject": subject,
        "body": body,
        "combined_text": combine_subject_and_body(subject, body),
        "label": int(example["label"]),
    }


def prepare_phishing(example: dict) -> dict:
    body = (example.get("Email Text") or "")[:MAX_BODY_CHARS]
    return {
        "subject": "",
        "body": body,
        "combined_text": combine_subject_and_body("", body),
        "label": 1 if example.get("Email Type") == "Phishing Email" else 0,
    }


def is_usable(example: dict) -> bool:
    """Drops emails with no real content, such as empty bodies or "null" placeholders."""
    raw = f"{example.get('subject', '')} {example.get('body', '')}".strip().lower()
    return raw not in PLACEHOLDER_TEXTS and bool(example.get("combined_text"))


def load_splits():
    """Returns a DatasetDict with "train", "validation" and "test"."""
    from datasets import DatasetDict, load_dataset

    raw = load_dataset(MAIN_DATASET)
    columns = raw["train"].column_names
    prepared = raw.map(prepare_enron, remove_columns=columns).filter(is_usable)
    split = prepared["train"].train_test_split(test_size=VALIDATION_SHARE, seed=SEED)
    return DatasetDict(train=split["train"], validation=split["test"], test=prepared["test"])


def load_phishing_test():
    """The phishing dataset as one test split with the same columns as `load_splits`."""
    from datasets import load_dataset

    raw = load_dataset(PHISHING_DATASET)["train"]
    return raw.map(prepare_phishing, remove_columns=raw.column_names).filter(is_usable)
