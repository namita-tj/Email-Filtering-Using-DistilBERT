"""The decision chain: model first, LLM when the model is unsure, rules as the last resort."""

import logging
from dataclasses import dataclass
from typing import Protocol

from app.llm import LLMError
from app.preprocessing import combine_subject_and_body
from app.rules import classify_with_rules

logger = logging.getLogger(__name__)


class SpamModel(Protocol):
    def spam_probability(self, text: str) -> float: ...


@dataclass(frozen=True)
class Prediction:
    label: str
    source: str
    spam_probability: float
    reason: str | None = None


class SpamFilter:
    def __init__(self, model: SpamModel, llm=None, low: float = 0.3, high: float = 0.8):
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("Thresholds must satisfy 0 <= low <= high <= 1")
        self.model = model
        self.llm = llm
        self.low = low
        self.high = high

    def classify(self, subject: str, body: str) -> Prediction:
        p = self.model.spam_probability(combine_subject_and_body(subject, body))

        if p >= self.high:
            return Prediction("spam", "model", p)
        if p <= self.low:
            return Prediction("ham", "model", p)

        if self.llm is not None:
            try:
                result = self.llm.classify(subject, body)
                return Prediction(result.label, "llm", p, result.reason)
            except LLMError as exc:
                logger.warning("LLM fallback failed, using rules: %s", exc)

        rules = classify_with_rules(f"{subject}\n{body}")
        return Prediction(rules.label, "rules", p, rules.reason)

    @property
    def can_explain(self) -> bool:
        return callable(getattr(self.model, "explain", None))

    def explain(self, subject: str, body: str, steps: int = 32):
        """Word attributions for the model's spam probability (Integrated Gradients)."""
        if not self.can_explain:
            raise NotImplementedError("The loaded model does not support explanations")
        return self.model.explain(combine_subject_and_body(subject, body), steps=steps)
