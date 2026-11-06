"""Turns token-level attributions into word-level explanations.

The gradient computation lives in `app/classifier.py` (it needs TensorFlow).
Everything here is plain Python so it can be tested and reused by the
evaluation scripts.

Sign convention: a positive weight pushed the email towards spam, a negative
weight towards ham.
"""

from dataclasses import asdict, dataclass

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[MASK]"}


@dataclass(frozen=True)
class WordAttribution:
    word: str
    weight: float


@dataclass(frozen=True)
class Explanation:
    method: str
    spam_probability: float
    words: list[WordAttribution]
    convergence_delta: float

    def top(self, k: int) -> list[WordAttribution]:
        """The k words with the largest effect in either direction."""
        return sorted(self.words, key=lambda w: abs(w.weight), reverse=True)[:k]

    def to_dict(self, top_k: int | None = None) -> dict:
        words = self.top(top_k) if top_k else self.words
        return {
            "method": self.method,
            "spam_probability": round(self.spam_probability, 4),
            "convergence_delta": round(self.convergence_delta, 4),
            "words": [asdict(w) for w in words],
        }


def merge_wordpieces(tokens: list[str], scores: list[float]) -> list[WordAttribution]:
    """Joins WordPiece sub-tokens ("verif", "##y") into words and sums their scores.

    Special tokens are dropped. Word order is preserved, so the result can be
    rejoined with spaces to rebuild the model input.
    """
    if len(tokens) != len(scores):
        raise ValueError("tokens and scores must have the same length")

    words: list[list] = []
    for token, score in zip(tokens, scores, strict=True):
        if token in SPECIAL_TOKENS:
            continue
        if token.startswith("##") and words:
            words[-1][0] += token[2:]
            words[-1][1] += score
        else:
            words.append([token.removeprefix("##"), score])
    return [WordAttribution(word, round(float(weight), 6)) for word, weight in words]
