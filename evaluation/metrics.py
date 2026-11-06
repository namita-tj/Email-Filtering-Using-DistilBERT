"""Plain-Python metrics, so they are easy to test and need no extra libraries."""

import math
from collections.abc import Sequence


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Binary metrics with spam = 1 as the positive class."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred, strict=True))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred, strict=True))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred, strict=True))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred, strict=True))

    def ratio(a: int, b: int) -> float:
        return a / b if b else 0.0

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    return {
        "n": len(y_true),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": ratio(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": ratio(2 * precision * recall, precision + recall) if precision + recall else 0.0,
        "false_positive_rate": ratio(fp, fp + tn),
    }


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 100]. Returns 0.0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def llm_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> float:
    return (
        prompt_tokens * input_price_per_million + completion_tokens * output_price_per_million
    ) / 1_000_000
