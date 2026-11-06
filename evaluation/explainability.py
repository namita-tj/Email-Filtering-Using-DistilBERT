"""Explainability evaluation: IG completeness, ERASER faithfulness, shortcut audit."""

import random
from collections.abc import Sequence
from statistics import mean

from evaluation.faithfulness import score_example, summarize
from evaluation.render import render_svg
from evaluation.shortcut_audit import aggregate, audit

K_VALUES = (1, 3, 5)
MIN_EMAILS = 5


def stratified_sample(labels: Sequence[int], n: int, rng: random.Random) -> list[int]:
    spam = [i for i, y in enumerate(labels) if y == 1]
    ham = [i for i, y in enumerate(labels) if y == 0]
    half = n // 2
    take_spam = min(len(spam), half)
    take_ham = min(len(ham), n - take_spam)
    return sorted(rng.sample(spam, take_spam) + rng.sample(ham, take_ham))


def pick_example(rows: list[dict]) -> dict | None:
    """A correctly classified spam email of readable length, the most confident one."""
    good = [r for r in rows if r["label"] == 1 and r["pred"] == "spam" and 12 <= len(r["words"]) <= 60]
    if not good:
        good = [r for r in rows if r["label"] == 1 and r["pred"] == "spam"]
    return max(good, key=lambda r: r["p"], default=None)


def run_explainability(
    classifier,
    texts: Sequence[str],
    labels: Sequence[int],
    low: float,
    high: float,
    steps: int = 32,
    rng: random.Random | None = None,
) -> tuple[dict, str | None]:
    """Returns the results and, if a good example exists, the README SVG.

    `classifier` needs `spam_probabilities(texts)` and `explain(text, steps)`.
    `texts` are already cleaned model inputs.
    """
    rng = rng or random.Random(42)
    probs = classifier.spam_probabilities(list(texts))
    rows, per_k = [], {k: [] for k in K_VALUES}
    for n, (text, label, p) in enumerate(zip(texts, labels, probs, strict=True), start=1):
        pred = "spam" if p >= 0.5 else "ham"
        explanation = classifier.explain(text, steps=steps)
        if not explanation.words:
            continue
        rows.append({"label": label, "pred": pred, "p": p, "words": explanation.words,
                     "delta": explanation.convergence_delta})
        for k in K_VALUES:
            per_k[k].append(score_example(explanation.words, pred, classifier.spam_probabilities, k, rng))
        if n % 20 == 0:
            print(f"  explained {n}/{len(texts)}")

    if not rows:
        return {}, None

    stats = aggregate((r["words"], r["label"]) for r in rows)
    results = {
        "samples": len(rows),
        "spam": sum(r["label"] for r in rows),
        "ham": sum(1 - r["label"] for r in rows),
        "steps": steps,
        "model": {
            "accuracy": mean((r["pred"] == "spam") == (r["label"] == 1) for r in rows),
            "uncertain_share": mean(low < r["p"] < high for r in rows),
            "thresholds": {"low": low, "high": high},
            "mean_convergence_delta": mean(r["delta"] for r in rows),
        },
        "faithfulness": {str(k): summarize(v) for k, v in per_k.items()},
        "shortcut_audit": {**audit(stats, min_emails=MIN_EMAILS), "min_emails": MIN_EMAILS},
    }

    example = pick_example(rows)
    svg = None
    if example:
        svg = render_svg(example["words"], example["pred"], example["p"],
                         note="Test email from the Enron-Spam dataset.")
    return results, svg
