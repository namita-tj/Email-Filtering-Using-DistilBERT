"""Checks whether explanations reflect what the model actually relies on.

Deletion tests from the ERASER benchmark (DeYoung et al., 2020):

- comprehensiveness: remove the top-k words. A faithful explanation makes the
  predicted class's probability drop a lot.
- sufficiency: keep only the top-k words. A faithful explanation keeps the
  probability close to the original, so the drop is small.

Both are compared with removing k random words, the baseline an explanation
has to beat.
"""

import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from statistics import mean

from app.explain import WordAttribution

# Takes a batch of texts, returns their spam probabilities.
PredictFn = Callable[[list[str]], list[float]]


@dataclass(frozen=True)
class FaithfulnessResult:
    k: int
    comprehensiveness: float
    sufficiency: float
    random_comprehensiveness: float

    def to_dict(self) -> dict:
        return asdict(self)


def _class_prob(p_spam: float, label: str) -> float:
    return p_spam if label == "spam" else 1.0 - p_spam


def top_indices(words: Sequence[WordAttribution], label: str, k: int) -> list[int]:
    """Indices of the k words that pushed most strongly towards `label`."""
    sign = 1.0 if label == "spam" else -1.0
    order = sorted(range(len(words)), key=lambda i: sign * words[i].weight, reverse=True)
    return order[:k]


def _join(words: Sequence[WordAttribution], keep: set[int]) -> str:
    return " ".join(w.word for i, w in enumerate(words) if i in keep)


def score_example(
    words: Sequence[WordAttribution],
    label: str,
    predict: PredictFn,
    k: int,
    rng: random.Random,
    random_repeats: int = 5,
) -> FaithfulnessResult:
    """Deletion tests for one email. `label` is the model's predicted class."""
    n = len(words)
    k = min(k, n)
    everything = set(range(n))
    top = set(top_indices(words, label, k))

    texts = [
        _join(words, everything),
        _join(words, everything - top),
        _join(words, top),
    ]
    for _ in range(random_repeats):
        texts.append(_join(words, everything - set(rng.sample(range(n), k))))

    probs = [_class_prob(p, label) for p in predict(texts)]
    full, without_top, only_top, randoms = probs[0], probs[1], probs[2], probs[3:]

    return FaithfulnessResult(
        k=k,
        comprehensiveness=full - without_top,
        sufficiency=full - only_top,
        random_comprehensiveness=mean(full - r for r in randoms) if randoms else 0.0,
    )


def summarize(results: Sequence[FaithfulnessResult]) -> dict:
    """Means over all emails, plus how often the explanation beat the random baseline."""
    if not results:
        return {}
    return {
        "examples": len(results),
        "comprehensiveness": round(mean(r.comprehensiveness for r in results), 4),
        "random_comprehensiveness": round(mean(r.random_comprehensiveness for r in results), 4),
        "sufficiency": round(mean(r.sufficiency for r in results), 4),
        "beats_random_share": round(
            mean(r.comprehensiveness > r.random_comprehensiveness for r in results), 4
        ),
    }
