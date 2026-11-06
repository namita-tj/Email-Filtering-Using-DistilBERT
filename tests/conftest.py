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
