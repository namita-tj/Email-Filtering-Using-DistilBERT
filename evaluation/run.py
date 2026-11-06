"""Full evaluation: system quality, explainability and the KPI report.

Run from the repository root after training:

    python -m evaluation.run                   # Enron-Spam test split
    python -m evaluation.run --phishing        # also test on modern phishing emails
    python -m evaluation.run --no-llm          # model and rules only, no API cost
    python -m evaluation.run --fail-on-miss    # exit code 1 if a KPI misses its target

Writes to reports/:
    system.json, system_phishing.json, explainability.json, explainability.md,
    kpis.json, kpis.md
and docs/explanation_example.svg for the README.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from app.classifier import DistilBertClassifier
from app.config import Settings, load_thresholds
from app.llm import build_llm_classifier
from app.model_store import ensure_model
from app.pipeline import SpamFilter
from evaluation import kpis
from evaluation.explainability import run_explainability, stratified_sample
from evaluation.report import to_markdown as explainability_markdown
from evaluation.system import DEFAULT_INPUT_PRICE, DEFAULT_OUTPUT_PRICE, run_system
from training.data import load_phishing_test, load_splits


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _sample(n_total: int, n: int, rng: random.Random) -> list[int]:
    return sorted(rng.sample(range(n_total), n)) if 0 < n < n_total else list(range(n_total))


def _system_on(split, indices, spam_filter, args) -> dict:
    return run_system(
        spam_filter,
        [(split[i]["subject"], split[i]["body"]) for i in indices],
        [split[i]["label"] for i in indices],
        input_price=args.input_price,
        output_price=args.output_price,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--system-samples", type=int, default=0,
                        help="Test emails for the system evaluation; 0 = all 2,000")
    parser.add_argument("--xai-samples", type=int, default=200,
                        help="Emails for the explainability evaluation (balanced spam/ham)")
    parser.add_argument("--phishing", action="store_true",
                        help="Also test on the modern phishing dataset (out of distribution)")
    parser.add_argument("--phishing-samples", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=32, help="Integrated Gradients steps")
    parser.add_argument("--no-llm", action="store_true", help="Skip the LLM fallback")
    parser.add_argument("--skip-system", action="store_true")
    parser.add_argument("--skip-explainability", action="store_true")
    parser.add_argument("--input-price", type=float, default=DEFAULT_INPUT_PRICE,
                        help="LLM input price, USD per 1M tokens")
    parser.add_argument("--output-price", type=float, default=DEFAULT_OUTPUT_PRICE,
                        help="LLM output price, USD per 1M tokens")
    parser.add_argument("--fail-on-miss", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    ensure_model(settings.model_dir, settings.s3_bucket, settings.s3_prefix)
    low, high = load_thresholds(settings)
    classifier = DistilBertClassifier(settings.model_dir)
    test = load_splits()["test"]
    rng = random.Random(args.seed)
    args.reports.mkdir(parents=True, exist_ok=True)

    llm = None if args.no_llm or args.skip_system else build_llm_classifier(settings)
    spam_filter = SpamFilter(classifier, llm, low=low, high=high)

    system = None
    if not args.skip_system:
        indices = _sample(len(test), args.system_samples, rng)
        print(f"System evaluation on {len(indices)} Enron-Spam test emails "
              f"(LLM: {'on' if llm else 'off'})...")
        system = _system_on(test, indices, spam_filter, args)
        (args.reports / "system.json").write_text(json.dumps(system, indent=2))

    phishing = None
    if args.phishing and not args.skip_system:
        phishing_split = load_phishing_test()
        indices = _sample(len(phishing_split), args.phishing_samples, rng)
        print(f"Out-of-distribution test on {len(indices)} phishing-dataset emails...")
        # A fresh filter, so the LLM cost of this test is counted separately.
        fresh_llm = None if args.no_llm else build_llm_classifier(settings)
        phishing = _system_on(phishing_split, indices,
                              SpamFilter(classifier, fresh_llm, low=low, high=high), args)
        (args.reports / "system_phishing.json").write_text(json.dumps(phishing, indent=2))

    explainability = None
    if not args.skip_explainability:
        indices = stratified_sample(test["label"], args.xai_samples, rng)
        print(f"Explainability evaluation on {len(indices)} emails...")
        explainability, svg = run_explainability(
            classifier,
            [test[i]["combined_text"] for i in indices],
            [test[i]["label"] for i in indices],
            low, high, steps=args.steps, rng=rng,
        )
        if explainability:
            (args.reports / "explainability.json").write_text(json.dumps(explainability, indent=2))
            (args.reports / "explainability.md").write_text(explainability_markdown(explainability))
        if svg:
            args.docs.mkdir(parents=True, exist_ok=True)
            (args.docs / "explanation_example.svg").write_text(svg)

    # Reuse earlier results for any part skipped this time.
    system = system or _read_json(args.reports / "system.json")
    phishing = phishing or _read_json(args.reports / "system_phishing.json")
    explainability = explainability or _read_json(args.reports / "explainability.json")
    training = _read_json(settings.model_dir / "metrics.json")

    rows = kpis.evaluate(kpis.collect(training, system, explainability, phishing))
    (args.reports / "kpis.json").write_text(json.dumps(rows, indent=2))
    (args.reports / "kpis.md").write_text(kpis.to_markdown(rows))

    print()
    for r in rows:
        print(f"  [{r['status']:>12}] {r['name']}: {r['display']} (target {r['target']})")
    print(f"\nReports written to {args.reports}/")
    return 1 if args.fail_on_miss and kpis.any_missed(rows) else 0


if __name__ == "__main__":
    sys.exit(main())
