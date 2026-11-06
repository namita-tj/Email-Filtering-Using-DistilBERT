import json
import random
import xml.etree.ElementTree as ET

import pytest

from app.explain import WordAttribution as W
from evaluation.faithfulness import score_example, summarize, top_indices
from evaluation.render import render_svg
from evaluation.report import to_markdown
from evaluation.run import pick_example, stratified_sample
from evaluation.shortcut_audit import aggregate, audit

SPAM_WORDS = {"win", "prize", "verify"}


def toy_predict(texts):
    """Spam probability grows with the share of spam words, like a simple real model."""
    probs = []
    for text in texts:
        words = text.split()
        hits = sum(w in SPAM_WORDS for w in words)
        probs.append(0.1 + 0.8 * hits / max(len(SPAM_WORDS), 1))
    return probs


EMAIL = [W("win", 0.5), W("a", 0.0), W("prize", 0.4), W("today", -0.05), W("verify", 0.3)]


def test_top_indices_follow_predicted_class():
    assert top_indices(EMAIL, "spam", 2) == [0, 2]
    assert top_indices(EMAIL, "ham", 1) == [3]


def test_faithful_explanation_beats_random():
    result = score_example(EMAIL, "spam", toy_predict, k=3, rng=random.Random(0))
    assert result.comprehensiveness == pytest.approx(0.8)
    assert result.sufficiency == pytest.approx(0.0)
    assert result.comprehensiveness > result.random_comprehensiveness


def test_unfaithful_explanation_is_caught():
    backwards = [W(w.word, -w.weight) for w in EMAIL]
    result = score_example(backwards, "spam", toy_predict, k=2, rng=random.Random(0))
    assert result.comprehensiveness == pytest.approx(0.0)
    assert result.comprehensiveness < result.random_comprehensiveness


def test_k_larger_than_email_is_capped():
    result = score_example(EMAIL[:2], "spam", toy_predict, k=5, rng=random.Random(0))
    assert result.k == 2


def test_summarize():
    rng = random.Random(0)
    results = [score_example(EMAIL, "spam", toy_predict, 3, rng) for _ in range(3)]
    summary = summarize(results)
    assert summary["examples"] == 3
    assert summary["beats_random_share"] == 1.0
    assert summarize([]) == {}


def _audit_examples():
    spam = [([W("prize", 0.4), W("the", 0.3), W("acme", 0.5)], 1) for _ in range(6)]
    ham = [([W("meeting", -0.4), W("the", -0.1), W("prize", 0.1)], 0) for _ in range(6)]
    return spam + ham


def test_aggregate_counts_each_email_once():
    stats = aggregate([([W("win", 0.2), W("win", 0.4)], 1)])
    assert stats["win"].occurrences == 2
    assert stats["win"].spam_emails == 1
    assert stats["win"].mean_weight == pytest.approx(0.3)


def test_audit_flags_shortcut_candidates():
    report = audit(aggregate(_audit_examples()), min_emails=5, top_n=10)
    flags = {c["word"]: c["flag"] for c in report["shortcut_candidates"]}
    assert flags == {"acme": "class-exclusive", "the": "non-content", "meeting": "class-exclusive"}
    assert report["top_spam_words"][0]["word"] == "acme"
    assert report["top_ham_words"][0]["word"] == "meeting"
    assert report["words_analysed"] == 4


def test_render_svg_is_valid_and_escaped():
    svg = render_svg([W("win", 0.5), W("<b>", -0.2), W("now", 0.0)], "spam", 0.97, note="Test & note")
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "&lt;b&gt;" in svg
    assert "spam probability 0.97" in svg
    assert "Test &amp; note" in svg


def test_render_wraps_long_emails():
    svg = render_svg([W(f"word{i}", 0.1) for i in range(60)], "spam", 0.9)
    assert int(ET.fromstring(svg).get("height")) > 300


def test_report_markdown():
    stats = aggregate(_audit_examples())
    results = {
        "samples": 12, "spam": 6, "ham": 6, "steps": 32,
        "model": {"accuracy": 0.95, "uncertain_share": 0.08, "mean_convergence_delta": 0.01},
        "faithfulness": {"5": {"comprehensiveness": 0.6, "random_comprehensiveness": 0.1,
                               "sufficiency": 0.05, "beats_random_share": 0.9}},
        "shortcut_audit": {**audit(stats), "min_emails": 5},
    }
    md = to_markdown(json.loads(json.dumps(results)))
    assert "| Accuracy | 95.0% |" in md
    assert "| 5 | +0.6000 | +0.1000 | +0.0500 | 90.0% |" in md
    assert "| acme | 6 | +0.5000 | 100.0% | class-exclusive |" in md


def test_stratified_sample_balances_classes():
    labels = [1] * 30 + [0] * 70
    picked = stratified_sample(labels, 20, random.Random(0))
    assert sum(labels[i] for i in picked) == 10
    assert len(picked) == 20


def test_pick_example_prefers_readable_confident_spam():
    words = [W("w", 0.1)] * 20
    rows = [
        {"label": 1, "pred": "spam", "p": 0.99, "words": [W("w", 0.1)] * 5},
        {"label": 1, "pred": "spam", "p": 0.95, "words": words},
        {"label": 0, "pred": "ham", "p": 0.01, "words": words},
    ]
    assert pick_example(rows)["p"] == 0.95
    assert pick_example([]) is None
