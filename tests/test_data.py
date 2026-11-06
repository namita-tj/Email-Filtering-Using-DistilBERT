import json

from app.classifier import DEFAULT_MAX_LENGTH, trained_max_length
from training.data import MAX_BODY_CHARS, is_usable, prepare_enron, prepare_phishing


def test_prepare_enron():
    row = prepare_enron({"subject": "Re: <b>Q3</b> report", "message": "See http://x.io now!", "label": 1})
    assert row == {
        "subject": "Re: <b>Q3</b> report",
        "body": "See http://x.io now!",
        "combined_text": "Re Q report See now",
        "label": 1,
    }


def test_prepare_enron_handles_missing_fields_and_huge_bodies():
    row = prepare_enron({"subject": None, "message": "a" * (MAX_BODY_CHARS + 50), "label": 0})
    assert row["subject"] == ""
    assert len(row["body"]) == MAX_BODY_CHARS


def test_prepare_phishing_labels():
    assert prepare_phishing({"Email Text": "Verify now", "Email Type": "Phishing Email"})["label"] == 1
    safe = prepare_phishing({"Email Text": "Lunch?", "Email Type": "Safe Email"})
    assert (safe["label"], safe["subject"], safe["combined_text"]) == (0, "", "Lunch")


def test_placeholder_and_empty_emails_are_dropped():
    assert not is_usable(prepare_phishing({"Email Text": "empty", "Email Type": "Phishing Email"}))
    assert not is_usable(prepare_phishing({"Email Text": " null ", "Email Type": "Phishing Email"}))
    assert not is_usable(prepare_phishing({"Email Text": None, "Email Type": "Safe Email"}))
    assert not is_usable(prepare_enron({"subject": "", "message": "!!! ???", "label": 1}))
    assert is_usable(prepare_enron({"subject": "Hi", "message": "", "label": 0}))


def test_serving_uses_training_max_length(tmp_path):
    assert trained_max_length(tmp_path) == DEFAULT_MAX_LENGTH
    (tmp_path / "training_config.json").write_text(json.dumps({"max_length": 256}))
    assert trained_max_length(tmp_path) == 256
