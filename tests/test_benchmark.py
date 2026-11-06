from types import SimpleNamespace

from training.benchmark import llm_approach, run_benchmark, to_markdown

SPAM = ["win a free prize now", "cheap meds online now", "claim your free prize", "free money win big"]
HAM = [
    "meeting notes for monday",
    "lunch with the team today",
    "project report attached",
    "see you at the meeting",
]


def _rows(n, offset=0):
    rows = []
    for i in range(offset, offset + n):
        spam = i % 2 == 0
        words = (SPAM if spam else HAM)[i % 4]
        rows.append(
            {"subject": f"msg {i}", "body": words, "combined_text": f"msg {words}", "label": int(spam)}
        )
    return rows


class FakeLLM:
    """Says spam when it sees 'free'; fails on one email to exercise error handling."""

    def __init__(self):
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def classify(self, subject, body):
        if subject == "msg 1001":
            raise RuntimeError("timeout")
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += 100
        self.usage["completion_tokens"] += 10
        return SimpleNamespace(label="spam" if "free" in body else "ham")


def test_classical_models_learn_the_toy_task():
    results = run_benchmark(_rows(80), {"Toy": _rows(20, offset=1000)}, llm_samples=10)
    names = [a["name"] for a in results["approaches"]]
    assert len(names) == 3 and any("Naive Bayes" in n for n in names)
    for a in results["approaches"]:
        assert a["full"]["Toy"]["n"] == 20
        assert a["full"]["Toy"]["accuracy"] >= 0.9
        assert a["llm_subset"]["Toy"]["n"] == 10
        assert a["full"]["Toy"]["ms_per_email"] >= 0


def test_llm_runs_on_subset_with_cost_and_failures_counted_as_ham():
    results = run_benchmark(_rows(40), {"Toy": _rows(20, offset=1000)}, llm=FakeLLM(), llm_samples=20)
    llm = results["approaches"][-1]
    assert llm["kind"] == "zero-shot LLM" and llm["full"] is None
    assert llm["llm_subset"]["Toy"]["n"] == 20
    assert llm["usage"]["calls"] == 19  # one call failed
    assert llm["usage"]["cost_usd"] > 0


def test_markdown_has_both_tables_and_cost():
    results = run_benchmark(_rows(40), {"Toy": _rows(10, offset=1000)}, llm=FakeLLM(), llm_samples=5)
    md = to_markdown(results)
    assert md.count("| Approach |") == 2
    assert "GPT-4o-mini" in md and "LLM cost" in md


def test_llm_approach_predicts_labels():
    rows = _rows(4, offset=0)
    assert llm_approach(FakeLLM()).predict(rows) == [1, 0, 1, 0]
