"""TF-IDF + logistic regression baseline on the same splits as DistilBERT.

A strong classical baseline shows how much the transformer actually adds. It
runs in about a minute on a laptop CPU and plugs into the same decision chain
and KPIs as the real model.

    python -m training.baseline                       # data from Hugging Face
    python -m training.baseline --enron-csv FILE.csv  # data from the Enron-Spam CSV on GitHub

The CSV option uses enron_spam_data.csv from github.com/MWiechmann/enron_spam_data,
the source of the Hugging Face version, and splits it the way SetFit's
prepare.py does (shuffle with seed 42, first 2,000 emails as test).

Writes reports/baseline.json.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from app.pipeline import SpamFilter
from evaluation.metrics import classification_metrics
from evaluation.system import run_system
from training.data import SEED, VALIDATION_SHARE, is_usable, prepare_enron
from training.thresholds import choose_thresholds

TEST_SIZE = 2000


class TfidfBaseline:
    """Word and bigram TF-IDF features with a balanced logistic regression."""

    def __init__(self, max_features: int = 200_000, c: float = 10.0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True
        )
        self.model = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")

    def fit(self, texts: list[str], labels: list[int]) -> "TfidfBaseline":
        self.model.fit(self.vectorizer.fit_transform(texts), labels)
        return self

    def spam_probabilities(self, texts: list[str]) -> list[float]:
        return self.model.predict_proba(self.vectorizer.transform(texts))[:, 1].tolist()

    def spam_probability(self, text: str) -> float:
        return self.spam_probabilities([text])[0]


def splits_from_csv(path: Path) -> dict[str, list[dict]]:
    """Enron-Spam CSV -> train/validation/test lists with the columns of training.data."""
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as f:
        rows = [
            {"subject": r["Subject"], "message": r["Message"], "label": int(r["Spam/Ham"] == "spam")}
            for r in csv.DictReader(f)
        ]
    random.Random(SEED).shuffle(rows)
    test_raw, train_raw = rows[:TEST_SIZE], rows[TEST_SIZE:]

    def prepare(items):
        return [p for p in (prepare_enron(r) for r in items) if is_usable(p)]

    train, test = prepare(train_raw), prepare(test_raw)
    random.Random(SEED).shuffle(train)
    n_val = int(len(train) * VALIDATION_SHARE)
    return {"train": train[n_val:], "validation": train[:n_val], "test": test}


def splits_from_hub() -> dict[str, list[dict]]:
    from training.data import load_splits

    return {name: list(split) for name, split in load_splits().items()}


def evaluate_baseline(splits: dict[str, list[dict]], target_precision: float = 0.99) -> dict:
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    baseline = TfidfBaseline().fit([r["combined_text"] for r in train], [r["label"] for r in train])

    val_probs = baseline.spam_probabilities([r["combined_text"] for r in validation])
    val_labels = [r["label"] for r in validation]
    thresholds = choose_thresholds(val_probs, val_labels, target=target_precision)

    test_labels = [r["label"] for r in test]
    test_probs = baseline.spam_probabilities([r["combined_text"] for r in test])
    system = run_system(
        SpamFilter(baseline, llm=None, low=thresholds.low, high=thresholds.high),
        [(r["subject"], r["body"]) for r in test],
        test_labels,
    )
    return {
        "model": "TF-IDF (1-2 grams) + logistic regression",
        "sizes": {name: len(rows) for name, rows in splits.items()},
        "validation": classification_metrics(val_labels, [int(p >= 0.5) for p in val_probs]),
        "test_model_alone": classification_metrics(test_labels, [int(p >= 0.5) for p in test_probs]),
        "thresholds": thresholds.to_dict(),
        "test_system_without_llm": system,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--enron-csv", type=Path, help="Use the Enron-Spam CSV instead of Hugging Face")
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    splits = splits_from_csv(args.enron_csv) if args.enron_csv else splits_from_hub()
    print({name: len(rows) for name, rows in splits.items()})
    results = evaluate_baseline(splits)

    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "baseline.json").write_text(json.dumps(results, indent=2))

    test = results["test_model_alone"]
    system = results["test_system_without_llm"]
    print(f"Test, model alone: accuracy {test['accuracy']:.4f}, F1 {test['f1']:.4f}, "
          f"precision {test['precision']:.4f}, recall {test['recall']:.4f}, "
          f"false positive rate {test['false_positive_rate']:.4f}")
    print(f"Confidence band: {results['thresholds']['low']} - {results['thresholds']['high']}, "
          f"uncertain share on test: {system['routing']['rules_share']:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
