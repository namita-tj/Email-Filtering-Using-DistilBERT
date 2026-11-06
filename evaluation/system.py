"""End-to-end evaluation of the hybrid filter: model -> LLM -> rules.

Runs every email through the same `SpamFilter` the API uses and measures what
users and operators care about: quality, how often the LLM is needed, whether
it actually helps, latency and cost.
"""

import time
from collections import Counter
from collections.abc import Sequence

from app.pipeline import SpamFilter
from evaluation.metrics import classification_metrics, llm_cost, percentile

# gpt-4o-mini list prices in USD per 1M tokens at the time of writing.
# Check current pricing and pass --input-price / --output-price to override.
DEFAULT_INPUT_PRICE = 0.15
DEFAULT_OUTPUT_PRICE = 0.60


def _latency_summary(values_ms: Sequence[float]) -> dict:
    return {
        "count": len(values_ms),
        "p50_ms": round(percentile(values_ms, 50), 1),
        "p95_ms": round(percentile(values_ms, 95), 1),
    }


def run_system(
    spam_filter: SpamFilter,
    emails: Sequence[tuple[str, str]],
    labels: Sequence[int],
    input_price: float = DEFAULT_INPUT_PRICE,
    output_price: float = DEFAULT_OUTPUT_PRICE,
) -> dict:
    """`emails` are (subject, body) pairs, `labels` are 1 for spam and 0 for ham."""
    if len(emails) != len(labels):
        raise ValueError("emails and labels must have the same length")

    hybrid_pred, model_pred, sources = [], [], []
    latency_by_source: dict[str, list[float]] = {"model": [], "llm": [], "rules": []}

    for subject, body in emails:
        start = time.perf_counter()
        prediction = spam_filter.classify(subject, body)
        elapsed_ms = (time.perf_counter() - start) * 1000

        hybrid_pred.append(1 if prediction.label == "spam" else 0)
        model_pred.append(1 if prediction.spam_probability >= 0.5 else 0)
        sources.append(prediction.source)
        latency_by_source[prediction.source].append(elapsed_ms)

    n = len(emails)
    counts = Counter(sources)
    routed = [i for i, s in enumerate(sources) if s != "model"]

    def accuracy_on(indices: list[int], predictions: list[int]) -> float | None:
        if not indices:
            return None
        return sum(predictions[i] == labels[i] for i in indices) / len(indices)

    model_on_routed = accuracy_on(routed, model_pred)
    hybrid_on_routed = accuracy_on(routed, hybrid_pred)

    usage = dict(getattr(spam_filter.llm, "usage", None) or
                 {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
    cost = llm_cost(usage["prompt_tokens"], usage["completion_tokens"], input_price, output_price)

    return {
        "emails": n,
        "thresholds": {"low": spam_filter.low, "high": spam_filter.high},
        "llm_enabled": spam_filter.llm is not None,
        "model_alone": classification_metrics(labels, model_pred),
        "hybrid": classification_metrics(labels, hybrid_pred),
        "routing": {
            "model_share": counts["model"] / n if n else 0.0,
            "llm_share": counts["llm"] / n if n else 0.0,
            "rules_share": counts["rules"] / n if n else 0.0,
        },
        "uncertain_emails": {
            "count": len(routed),
            "model_accuracy": model_on_routed,
            "hybrid_accuracy": hybrid_on_routed,
            "uplift": (
                hybrid_on_routed - model_on_routed
                if routed and model_on_routed is not None and hybrid_on_routed is not None
                else None
            ),
        },
        "latency": {source: _latency_summary(v) for source, v in latency_by_source.items()},
        "llm_usage": {
            **usage,
            "input_price_per_million": input_price,
            "output_price_per_million": output_price,
            "total_cost_usd": round(cost, 6),
            "cost_per_1000_emails_usd": round(cost / n * 1000, 4) if n else 0.0,
        },
    }
