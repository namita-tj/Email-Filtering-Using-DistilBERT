"""Benchmark: classical models, fine-tuned DistilBERT and a zero-shot LLM on the same emails.

Every approach is scored on the Enron-Spam test split (in distribution) and on the
same 2,000 modern phishing-dataset emails that evaluation/run.py uses (out of
distribution). The zero-shot LLM is expensive, so it runs on a random subset of
each test set; every other model is also scored on that subset for a fair comparison.

    python -m training.benchmark                        # all approaches (LLM if configured)
    python -m training.benchmark --no-llm               # skip the LLM
    python -m training.benchmark --enron-csv FILE.csv --no-phishing --no-distilbert   # CPU-only check

Writes reports/benchmark.json and reports/benchmark.md.
"""

import argparse
import json
import random
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from evaluation.metrics import classification_metrics, llm_cost

SEED = 42
PHISHING_SAMPLES = 2000  # same sample as evaluation/run.py --phishing
GPT_4O_MINI_PRICES = (0.15, 0.60)  # USD per million input / output tokens


# ── Approaches ────────────────────────────────────────────────────────────────
@dataclass
class Approach:
    name: str
    kind: str  # "classical", "fine-tuned transformer", "zero-shot LLM"
    predict: Callable[[list[dict]], list[int]]  # rows -> 0/1 labels
    train_seconds: float = 0.0


def _texts(rows: list[dict]) -> list[str]:
    return [r["combined_text"] for r in rows]


def classical_approaches(train: list[dict]) -> list[Approach]:
    """Naive Bayes, TF-IDF + logistic regression and a linear SVM, fitted on `train`."""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    from training.baseline import TfidfBaseline

    texts, labels = _texts(train), [r["label"] for r in train]
    approaches = []

    def fit(name, model):
        start = time.perf_counter()
        model.fit(texts, labels)
        seconds = time.perf_counter() - start
        approaches.append(
            Approach(
                name, "classical", lambda rows, m=model: [int(p) for p in m.predict(_texts(rows))], seconds
            )
        )

    fit(
        "Naive Bayes (word + bigram counts)",
        make_pipeline(
            CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000), MultinomialNB(alpha=0.1)
        ),
    )
    fit(
        "Linear SVM (TF-IDF)",
        make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True),
            LinearSVC(C=1.0, class_weight="balanced"),
        ),
    )

    start = time.perf_counter()
    lr = TfidfBaseline().fit(texts, labels)
    approaches.append(
        Approach(
            "Logistic regression (TF-IDF), project baseline",
            "classical",
            lambda rows: [int(p >= 0.5) for p in lr.spam_probabilities(_texts(rows))],
            time.perf_counter() - start,
        )
    )
    return approaches


def distilbert_approach(model_dir: Path) -> Approach:
    from app.classifier import DistilBertClassifier

    clf = DistilBertClassifier(model_dir)
    return Approach(
        "DistilBERT, fine-tuned (this project)",
        "fine-tuned transformer",
        lambda rows: [int(p >= 0.5) for p in clf.spam_probabilities(_texts(rows), batch_size=32)],
    )


def llm_approach(llm, workers: int = 8) -> Approach:
    """Zero-shot: the project's LLM fallback prompt applied to every email."""

    def classify(row: dict) -> int | None:
        try:
            return int(llm.classify(row["subject"], row["body"]).label == "spam")
        except Exception:  # noqa: BLE001 - counted as a failure, not a crash
            return None

    def predict(rows: list[dict]) -> list[int]:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(classify, rows))
        failures = sum(r is None for r in results)
        if failures:
            print(f"  LLM: {failures} of {len(rows)} calls failed; counted as 'ham'")
        return [r if r is not None else 0 for r in results]

    return Approach("GPT-4o-mini, zero-shot (no training)", "zero-shot LLM", predict)


# ── Scoring ───────────────────────────────────────────────────────────────────
def score(approach: Approach, rows: list[dict]) -> dict:
    start = time.perf_counter()
    predictions = approach.predict(rows)
    seconds = time.perf_counter() - start
    metrics = classification_metrics([r["label"] for r in rows], predictions)
    metrics["ms_per_email"] = round(1000 * seconds / max(len(rows), 1), 3)
    return metrics


def run_benchmark(
    train: list[dict],
    test_sets: dict[str, list[dict]],
    model_dir: Path | None = None,
    llm=None,
    llm_samples: int = 200,
    seed: int = SEED,
) -> dict:
    approaches = classical_approaches(train)
    if model_dir is not None:
        approaches.append(distilbert_approach(model_dir))

    rng = random.Random(seed)
    subsets = {name: rng.sample(rows, min(llm_samples, len(rows))) for name, rows in test_sets.items()}

    results: dict = {
        "test_sets": {n: len(r) for n, r in test_sets.items()},
        "llm_subset_size": {n: len(r) for n, r in subsets.items()},
        "approaches": [],
    }
    for approach in approaches:
        print(f"Scoring {approach.name}...")
        entry = {
            "name": approach.name,
            "kind": approach.kind,
            "train_seconds": round(approach.train_seconds, 1),
            "full": {n: score(approach, rows) for n, rows in test_sets.items()},
            "llm_subset": {n: score(approach, rows) for n, rows in subsets.items()},
        }
        results["approaches"].append(entry)

    if llm is not None:
        approach = llm_approach(llm)
        print(f"Scoring {approach.name} on {sum(len(s) for s in subsets.values())} emails...")
        entry = {
            "name": approach.name,
            "kind": approach.kind,
            "train_seconds": 0.0,
            "full": None,
            "llm_subset": {n: score(approach, rows) for n, rows in subsets.items()},
        }
        usage = llm.usage
        cost = llm_cost(usage["prompt_tokens"], usage["completion_tokens"], *GPT_4O_MINI_PRICES)
        n_emails = sum(len(s) for s in subsets.values())
        entry["usage"] = {
            **usage,
            "cost_usd": round(cost, 4),
            "cost_per_1000_emails_usd": round(1000 * cost / max(n_emails, 1), 4),
        }
        results["approaches"].append(entry)
    return results


# ── Report ────────────────────────────────────────────────────────────────────
def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def to_markdown(results: dict) -> str:
    names = list(results["test_sets"])
    lines = ["# Benchmark", ""]
    sizes = ", ".join(f"{n}: {k:,}" for n, k in results["test_sets"].items())
    lines += [f"Full test sets ({sizes} emails). Spam cut-off 0.5 for every model.", ""]

    header = (
        "| Approach | Type | "
        + " | ".join(f"{n}: F1 | {n}: legit flagged | {n}: spam caught" for n in names)
        + " | ms / email |"
    )
    lines += [header, "| " + " | ".join(["---"] * 2 + ["---:"] * (3 * len(names) + 1)) + " |"]
    for a in results["approaches"]:
        if a["full"] is None:
            continue
        cells = []
        for n in names:
            m = a["full"][n]
            cells += [f"{m['f1']:.3f}", _pct(m["false_positive_rate"]), _pct(m["recall"])]
        ms = a["full"][names[0]]["ms_per_email"]
        lines.append(f"| {a['name']} | {a['kind']} | " + " | ".join(cells) + f" | {ms:.2f} |")

    subset = ", ".join(f"{n}: {k}" for n, k in results["llm_subset_size"].items())
    lines += ["", f"## Same random subset for every approach, including the LLM ({subset} emails)", ""]
    lines += [
        header.replace(" | ms / email |", " |"),
        "| " + " | ".join(["---"] * 2 + ["---:"] * (3 * len(names))) + " |",
    ]
    for a in results["approaches"]:
        cells = []
        for n in names:
            m = a["llm_subset"][n]
            cells += [f"{m['f1']:.3f}", _pct(m["false_positive_rate"]), _pct(m["recall"])]
        lines.append(f"| {a['name']} | {a['kind']} | " + " | ".join(cells) + " |")

    llm = next((a for a in results["approaches"] if "usage" in a), None)
    if llm:
        u = llm["usage"]
        lines += [
            "",
            f"LLM cost: ${u['cost_usd']:.4f} for {u['calls']} calls "
            f"(${u['cost_per_1000_emails_usd']:.4f} per 1,000 emails).",
        ]
    return "\n".join(lines) + "\n"


# ── Data ──────────────────────────────────────────────────────────────────────
def phishing_sample(n: int = PHISHING_SAMPLES, seed: int = SEED) -> list[dict]:
    """The same emails evaluation/run.py scores: first draw of Random(seed)."""
    from training.data import load_phishing_test

    split = load_phishing_test()
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(split)), n)) if 0 < n < len(split) else range(len(split))
    return [split[i] for i in idx]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--enron-csv", type=Path, help="Enron-Spam CSV instead of Hugging Face")
    parser.add_argument("--no-phishing", action="store_true")
    parser.add_argument("--no-distilbert", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--llm-samples", type=int, default=200, help="LLM emails per test set")
    parser.add_argument("--model-dir", type=Path, default=Path("spam_detection_model"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    if args.enron_csv:
        from training.baseline import splits_from_csv

        splits = splits_from_csv(args.enron_csv)
    else:
        from training.baseline import splits_from_hub

        splits = splits_from_hub()

    test_sets = {"Enron": splits["test"]}
    if not args.no_phishing:
        test_sets["Modern phishing"] = phishing_sample()

    llm = None
    if not args.no_llm:
        from app.config import Settings
        from app.llm import build_llm_classifier

        llm = build_llm_classifier(Settings.from_env())
        if llm is None:
            print("No LLM configured (LLM_PROVIDER=none): skipping the zero-shot LLM.")

    results = run_benchmark(
        splits["train"],
        test_sets,
        model_dir=None if args.no_distilbert else args.model_dir,
        llm=llm,
        llm_samples=args.llm_samples,
    )
    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "benchmark.json").write_text(json.dumps(results, indent=2))
    markdown = to_markdown(results)
    (args.reports / "benchmark.md").write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
