import random

import pytest

from app.explain import Explanation, WordAttribution
from app.llm import LLMResult
from app.pipeline import SpamFilter
from evaluation import kpis
from evaluation.explainability import run_explainability
from evaluation.metrics import classification_metrics, llm_cost, percentile
from evaluation.system import run_system

# ---------- metrics ----------


def test_classification_metrics():
    m = classification_metrics([1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 1])
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 4, 1)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == pytest.approx(1 / 5)


def test_metrics_without_positives_do_not_divide_by_zero():
    m = classification_metrics([0, 0], [0, 0])
    assert (m["precision"], m["recall"], m["f1"], m["accuracy"]) == (0.0, 0.0, 0.0, 1.0)


def test_metrics_length_mismatch():
    with pytest.raises(ValueError):
        classification_metrics([1], [1, 0])


def test_percentile():
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 50) == 30
    assert percentile(values, 95) == pytest.approx(48)
    assert percentile([], 95) == 0.0
    assert percentile([7], 95) == 7


def test_llm_cost():
    assert llm_cost(1_000_000, 100_000, 0.15, 0.60) == pytest.approx(0.21)


# ---------- system evaluation ----------


class ScriptedModel:
    """Spam probability is looked up from the email subject."""

    def __init__(self, probs):
        self.probs = probs

    def spam_probability(self, text):
        return self.probs[text.split()[0]]


class CountingLLM:
    def __init__(self, answers):
        self.answers = answers
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def classify(self, subject, body):
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += 200
        self.usage["completion_tokens"] += 20
        return LLMResult(self.answers[subject], "checked")


def _scenario(llm):
    # a, b: confident and right. c: uncertain spam the model gets wrong. d: uncertain ham.
    model = ScriptedModel({"a": 0.95, "b": 0.05, "c": 0.45, "d": 0.55})
    emails = [("a", "x"), ("b", "x"), ("c", "x"), ("d", "x")]
    labels = [1, 0, 1, 0]
    return run_system(SpamFilter(model, llm, low=0.3, high=0.8), emails, labels)


def test_system_with_helpful_llm():
    result = _scenario(CountingLLM({"c": "spam", "d": "ham"}))
    assert result["model_alone"]["accuracy"] == 0.5
    assert result["hybrid"]["accuracy"] == 1.0
    assert result["routing"] == {"model_share": 0.5, "llm_share": 0.5, "rules_share": 0.0}
    assert result["uncertain_emails"] == {
        "count": 2, "model_accuracy": 0.0, "hybrid_accuracy": 1.0, "uplift": 1.0,
    }
    assert result["llm_usage"]["calls"] == 2
    assert result["llm_usage"]["cost_per_1000_emails_usd"] == pytest.approx(
        llm_cost(400, 40, 0.15, 0.60) / 4 * 1000, abs=1e-4
    )
    assert result["latency"]["llm"]["count"] == 2


def test_system_without_llm_uses_rules():
    result = _scenario(None)
    assert result["llm_enabled"] is False
    assert result["routing"]["rules_share"] == 0.5
    assert result["llm_usage"]["calls"] == 0


def test_system_length_mismatch():
    with pytest.raises(ValueError):
        run_system(SpamFilter(ScriptedModel({})), [("a", "b")], [])


# ---------- explainability runner ----------


class ToyExplainer:
    SPAM = {"win", "prize"}

    def spam_probabilities(self, texts):
        return [0.9 if self.SPAM & set(t.split()) else 0.1 for t in texts]

    def explain(self, text, steps=32):
        words = [WordAttribution(w, 0.5 if w in self.SPAM else -0.01) for w in text.split()]
        return Explanation("integrated_gradients", self.spam_probabilities([text])[0], words, 0.001)


def test_run_explainability_end_to_end():
    spam = [f"win a prize today number {i} for you and your family and friends now" for i in range(6)]
    ham = [f"meeting notes for the team number {i} please read before friday morning" for i in range(6)]
    results, svg = run_explainability(ToyExplainer(), spam + ham, [1] * 6 + [0] * 6,
                                      low=0.3, high=0.8, rng=random.Random(0))
    assert results["samples"] == 12
    assert results["model"]["accuracy"] == 1.0
    # Spam explanations beat random; this toy model has no ham evidence to remove.
    assert results["faithfulness"]["5"]["beats_random_share"] == 0.5
    assert results["faithfulness"]["5"]["comprehensiveness"] > results["faithfulness"]["5"][
        "random_comprehensiveness"
    ]
    assert svg is not None and "win" in svg


def test_run_explainability_with_no_words():
    class Empty(ToyExplainer):
        def explain(self, text, steps=32):
            return Explanation("integrated_gradients", 0.5, [], 0.0)

    assert run_explainability(Empty(), ["x"], [1], 0.3, 0.8) == ({}, None)


# ---------- KPIs ----------

SYSTEM = {
    "llm_enabled": True,
    "hybrid": {"false_positive_rate": 0.004, "recall": 0.97, "f1": 0.96},
    "uncertain_emails": {"uplift": 0.12},
    "routing": {"llm_share": 0.08, "rules_share": 0.0},
    "latency": {"model": {"count": 90, "p95_ms": 120.0}, "llm": {"count": 8, "p95_ms": 4200.0}},
    "llm_usage": {"cost_per_1000_emails_usd": 0.02},
}
XAI = {
    "model": {"mean_convergence_delta": 0.01},
    "faithfulness": {"5": {"comprehensiveness": 0.55, "random_comprehensiveness": 0.05,
                           "sufficiency": 0.08, "beats_random_share": 0.91}},
    "shortcut_audit": {"shortcut_candidates": [{"word": "acme"}, {"word": "the"}]},
}


def test_collect_and_evaluate_all_kpis():
    values = kpis.collect({"f1_spam": 0.97}, SYSTEM, XAI)
    assert values["faithfulness_gap"] == pytest.approx(0.5)
    assert values["shortcut_candidates"] == 2

    rows = {r["key"]: r for r in kpis.evaluate(values)}
    assert len(rows) == len(kpis.KPIS)
    assert rows["false_positive_rate"]["status"] == "pass"
    assert rows["latency_p95_llm"]["status"] == "miss"
    assert rows["shortcut_candidates"]["status"] == "review"
    assert rows["false_positive_rate"]["display"] == "0.4%"
    assert rows["llm_uplift"]["display"] == "+12.0 pp"
    assert rows["cost_per_1000"]["display"] == "$0.0200"
    assert kpis.any_missed(list(rows.values()))


def test_missing_results_are_not_measured():
    rows = kpis.evaluate(kpis.collect(None, None, None))
    assert {r["status"] for r in rows} == {"not measured"}
    assert not kpis.any_missed(rows)


def test_llm_kpis_skipped_when_llm_off():
    system = {**SYSTEM, "llm_enabled": False,
              "latency": {"model": {"count": 5, "p95_ms": 50.0}, "llm": {"count": 0, "p95_ms": 0.0}}}
    values = kpis.collect(None, system, None)
    assert values["cost_per_1000"] is None
    assert values["latency_p95_llm"] is None


def test_kpi_markdown():
    md = kpis.to_markdown(kpis.evaluate(kpis.collect({"f1_spam": 0.97}, SYSTEM, XAI)))
    assert md.startswith("# KPI report")
    assert "| Quality | Legitimate emails marked as spam | ≤ 1.0% | 0.4% | pass |" in md
    assert "**13 pass · 1 miss · 1 to review · 0 not measured**" in md
