"""Chooses the confidence band in which the model's answer is not trusted.

Emails whose spam probability falls strictly between `low` and `high` are sent
to the LLM fallback. Outside the band the model decides alone. Both edges are
picked on the validation set so that the model's own decisions reach a target
precision (for spam) and negative predictive value (for ham).
"""

from dataclasses import asdict, dataclass

DEFAULT_LOW = 0.3
DEFAULT_HIGH = 0.8


@dataclass(frozen=True)
class Thresholds:
    low: float
    high: float
    target: float
    uncertain_share: float
    calibrated: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _share_between(probs: list[float], low: float, high: float) -> float:
    if not probs:
        return 0.0
    return sum(low < p < high for p in probs) / len(probs)


def choose_thresholds(
    probs: list[float],
    labels: list[int],
    target: float = 0.99,
    min_support: int = 10,
) -> Thresholds:
    """Returns the widest trusted ranges that still meet `target`.

    `probs` are spam probabilities, `labels` are 1 for spam and 0 for ham.
    Falls back to the defaults when the validation set cannot support the target.
    """
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have the same length")

    pairs = list(zip(probs, labels, strict=True))
    candidates = sorted(set(probs))

    high = None
    for t in candidates:
        chosen = [y for p, y in pairs if p >= t]
        if len(chosen) >= min_support and sum(chosen) / len(chosen) >= target:
            high = t
            break

    low = None
    for t in reversed(candidates):
        chosen = [y for p, y in pairs if p <= t]
        if len(chosen) >= min_support and chosen.count(0) / len(chosen) >= target:
            low = t
            break

    calibrated = low is not None and high is not None
    if not calibrated:
        low, high = DEFAULT_LOW, DEFAULT_HIGH
    elif low >= high:
        # The model is confident everywhere: no uncertain band is needed.
        low = high = 0.5

    return Thresholds(
        low=round(low, 4),
        high=round(high, 4),
        target=target,
        uncertain_share=round(_share_between(probs, low, high), 4),
        calibrated=calibrated,
    )
