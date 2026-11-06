"""Loads the dataset and makes the same train/validation split everywhere."""

from app.preprocessing import combine_subject_and_body

DATASET = "TrainingDataPro/email-spam-classification"
SEED = 42


def _prepare(example):
    example["combined_text"] = combine_subject_and_body(example["title"], example["text"])
    example["label"] = 1 if example["type"] == "spam" else 0
    return example


def load_splits():
    """Returns a DatasetDict with "train" and "test" (validation) splits."""
    from datasets import load_dataset

    dataset = load_dataset(DATASET).map(_prepare)
    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.2, seed=SEED)
    return dataset
