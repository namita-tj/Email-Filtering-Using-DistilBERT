"""Explainability evaluation on the validation set.

Run from the repository root after training:

    python -m evaluation.run --samples 200

Writes:
    reports/evaluation.json        all numbers
    reports/evaluation.md          readable report (linked from the README)
    docs/explanation_example.svg   highlighted example email for the README
"""

import argparse
import json
import random
from pathlib import Path
from statistics import mean

from app.classifier import DistilBertClassifier
from app.config import Settings, load_thresholds
from app.model_store import ensure_model
from evaluation.faithfulness import score_example, summarize
from evaluation.render import render_svg
from evaluation.report import to_markdown
from evaluation.shortcut_audit import aggregate, audit
from training.data import load_splits

K_VALUES = (1, 3, 5)
MIN_EMAILS = 5


def stratified_sample(labels: list[int], n: int, rng: random.Random) -> list[int]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--steps", type=int, default=32, help="Integrated Gradients steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args()

    settings = Settings.from_env()
    ensure_model(settings.model_dir, settings.s3_bucket, settings.s3_prefix)
    low, high = load_thresholds(settings)
    classifier = DistilBertClassifier(settings.model_dir)

    validation = load_splits()["test"]
    rng = random.Random(args.seed)
    indices = stratified_sample(validation["label"], args.samples, rng)
    texts = [validation[i]["combined_text"] for i in indices]
    labels = [validation[i]["label"] for i in indices]

    probs = classifier.spam_probabilities(texts)
    rows, per_k = [], {k: [] for k in K_VALUES}
    for n, (text, label, p) in enumerate(zip(texts, labels, probs, strict=True), start=1):
        pred = "spam" if p >= 0.5 else "ham"
        explanation = classifier.explain(text, steps=args.steps)
        if not explanation.words:
            continue
        rows.append({"label": label, "pred": pred, "p": p, "words": explanation.words,
                     "delta": explanation.convergence_delta})
        for k in K_VALUES:
            per_k[k].append(
                score_example(explanation.words, pred, classifier.spam_probabilities, k, rng)
            )
        if n % 20 == 0:
            print(f"  explained {n}/{len(texts)}")

    stats = aggregate((r["words"], r["label"]) for r in rows)
    results = {
        "samples": len(rows),
        "spam": sum(r["label"] for r in rows),
        "ham": sum(1 - r["label"] for r in rows),
        "steps": args.steps,
        "model": {
            "accuracy": mean((r["pred"] == "spam") == (r["label"] == 1) for r in rows),
            "uncertain_share": mean(low < r["p"] < high for r in rows),
            "thresholds": {"low": low, "high": high},
            "mean_convergence_delta": mean(r["delta"] for r in rows),
        },
        "faithfulness": {str(k): summarize(v) for k, v in per_k.items()},
        "shortcut_audit": {**audit(stats, min_emails=MIN_EMAILS), "min_emails": MIN_EMAILS},
    }

    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "evaluation.json").write_text(json.dumps(results, indent=2))
    (args.reports / "evaluation.md").write_text(to_markdown(results))

    example = pick_example(rows)
    if example:
        args.docs.mkdir(parents=True, exist_ok=True)
        svg = render_svg(example["words"], example["pred"], example["p"],
                         note="Validation email from the TrainingDataPro spam dataset.")
        (args.docs / "explanation_example.svg").write_text(svg)

    five = results["faithfulness"]["5"]
    print(f"\nAccuracy on sample: {results['model']['accuracy']:.1%}")
    print(f"Top-5 comprehensiveness {five['comprehensiveness']:+.3f} "
          f"vs random {five['random_comprehensiveness']:+.3f}")
    print(f"Shortcut candidates: {len(results['shortcut_audit']['shortcut_candidates'])}")
    print(f"Reports written to {args.reports}/ and {args.docs}/")


if __name__ == "__main__":
    main()
