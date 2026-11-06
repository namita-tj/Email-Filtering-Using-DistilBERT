import csv
import json

import pytest

from training import baseline
from training.baseline import TfidfBaseline, evaluate_baseline, splits_from_csv

SPAM = ["win a free prize now", "cheap meds online now", "claim your free prize", "free money win big"]
HAM = [
    "meeting notes for monday", "lunch with the team today",
    "project report attached", "see you at the meeting",
]


def _rows(n):
    rows = []
    for i in range(n):
        spam = i % 2 == 0
        words = (SPAM if spam else HAM)[i % 4]
        rows.append(
            {"subject": f"msg {i}", "body": words, "combined_text": f"msg {words}", "label": int(spam)}
        )
    return rows


def test_baseline_learns_simple_signal():
    rows = _rows(40)
    model = TfidfBaseline(c=10).fit([r["combined_text"] for r in rows], [r["label"] for r in rows])
    probs = model.spam_probabilities(["claim your free prize", "project report attached"])
    assert probs[0] > 0.5 > probs[1]
    assert model.spam_probability("win a free prize now") > 0.5


def test_splits_from_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "TEST_SIZE", 10)
    path = tmp_path / "enron.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Message ID", "Subject", "Message", "Spam/Ham", "Date"])
        for i in range(60):
            spam = i % 2 == 1
            body = "free prize" if spam else "team meeting\nnotes"
            writer.writerow([i, f"subject {i}", body, "spam" if spam else "ham", "2000-01-01"])
        writer.writerow([60, "", "", "ham", "2000-01-01"])  # empty email, dropped
    splits = splits_from_csv(path)
    assert len(splits["test"]) + len(splits["train"]) + len(splits["validation"]) == 60
    assert len(splits["validation"]) == int((len(splits["train"]) + len(splits["validation"])) * 0.1)
    assert set(splits["train"][0]) == {"subject", "body", "combined_text", "label"}


def test_evaluate_baseline_end_to_end(tmp_path):
    splits = {"train": _rows(60), "validation": _rows(30), "test": _rows(20)}
    results = evaluate_baseline(splits, target_precision=0.9)
    assert results["sizes"] == {"train": 60, "validation": 30, "test": 20}
    assert results["test_model_alone"]["accuracy"] == pytest.approx(1.0)
    assert results["test_system_without_llm"]["emails"] == 20
    json.dumps(results)  # must be serialisable for reports/baseline.json
