"""Full evaluation: system quality, explainability and the KPI report.

Run from the repository root after training:

    python -m evaluation.run                   # everything
    python -m evaluation.run --no-llm          # model and rules only, no API cost
    python -m evaluation.run --fail-on-miss    # exit code 1 if a KPI misses its target

Writes to reports/:
    system.json, explainability.json, explainability.md, kpis.json, kpis.md
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
from training.data import load_splits


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--system-samples", type=int, default=0,
                        help="Emails for the system evaluation; 0 = whole validation set")
    parser.add_argument("--xai-samples", type=int, default=200,
                        help="Emails for the explainability evaluation (balanced spam/ham)")
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
    validation = load_splits()["test"]
    rng = random.Random(args.seed)
    args.reports.mkdir(parents=True, exist_ok=True)

    system = None
    if not args.skip_system:
        n = len(validation["label"])
        indices = sorted(rng.sample(range(n), args.system_samples)) \
            if 0 < args.system_samples < n else list(range(n))
        llm = None if args.no_llm else build_llm_classifier(settings)
        spam_filter = SpamFilter(classifier, llm, low=low, high=high)
        print(f"System evaluation on {len(indices)} emails (LLM: {'on' if llm else 'off'})...")
        system = run_system(
            spam_filter,
            [(validation[i]["title"], validation[i]["text"]) for i in indices],
            [validation[i]["label"] for i in indices],
            input_price=args.input_price,
            output_price=args.output_price,
        )
        (args.reports / "system.json").write_text(json.dumps(system, indent=2))

    explainability = None
    if not args.skip_explainability:
        indices = stratified_sample(validation["label"], args.xai_samples, rng)
        print(f"Explainability evaluation on {len(indices)} emails...")
        explainability, svg = run_explainability(
            classifier,
            [validation[i]["combined_text"] for i in indices],
            [validation[i]["label"] for i in indices],
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
    explainability = explainability or _read_json(args.reports / "explainability.json")
    training = _read_json(settings.model_dir / "metrics.json")

    rows = kpis.evaluate(kpis.collect(training, system, explainability))
    (args.reports / "kpis.json").write_text(json.dumps(rows, indent=2))
    (args.reports / "kpis.md").write_text(kpis.to_markdown(rows))

    print()
    for r in rows:
        print(f"  [{r['status']:>12}] {r['name']}: {r['display']} (target {r['target']})")
    print(f"\nReports written to {args.reports}/")
    return 1 if args.fail_on_miss and kpis.any_missed(rows) else 0


if __name__ == "__main__":
    sys.exit(main())
