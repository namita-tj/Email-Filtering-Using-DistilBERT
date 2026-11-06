import pytest

from app.llm import LLMError, LLMResult


class FakeModel:
    """Returns a fixed spam probability and records what it was asked."""

    def __init__(self, probability: float):
        self.probability = probability
        self.calls: list[str] = []

    def spam_probability(self, text: str) -> float:
        self.calls.append(text)
        return self.probability


class FakeLLM:
    def __init__(self, label: str = "spam", reason: str = "looks like phishing", fail=False):
        self.label = label
        self.reason = reason
        self.fail = fail
        self.calls = 0

    def classify(self, subject: str, body: str) -> LLMResult:
        self.calls += 1
        if self.fail:
            raise LLMError("timeout")
        return LLMResult(self.label, self.reason)


@pytest.fixture
def fake_model():
    return FakeModel


@pytest.fixture
def fake_llm():
    return FakeLLM


class FakeExplainableModel(FakeModel):
    """Adds an `explain` method that returns fixed word weights."""

    def explain(self, text: str, steps: int = 32):
        from app.explain import Explanation, WordAttribution

        words = [WordAttribution(w, 0.0) for w in text.split()]
        if words:
            words[0] = WordAttribution(words[0].word, 0.4)
        if len(words) > 1:
            words[-1] = WordAttribution(words[-1].word, -0.2)
        return Explanation("integrated_gradients", self.probability, words, 0.01)


@pytest.fixture
def fake_explainable_model():
    return FakeExplainableModel
