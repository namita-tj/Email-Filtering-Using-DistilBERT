"""The project's KPIs, their targets, and a pass/miss report.

Targets are starting points for a production spam filter and should be
revisited after the first full evaluation run. The most important one is the
false positive rate: a legitimate email lost in the spam folder costs far more
than a spam email that slips through.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class KPI:
    key: str
    group: str
    name: str
    target: float | None  # None = tracked, reviewed by hand
    higher_is_better: bool
    unit: str  # "%", "pp", "ms", "usd" or ""
    why: str

    def status(self, value: float | None) -> str:
        if value is None:
            return "not measured"
        if self.target is None:
            return "review"
        ok = value >= self.target if self.higher_is_better else value <= self.target
        return "pass" if ok else "miss"

    def format(self, value: float | None) -> str:
        if value is None:
            return "–"
        if self.unit == "%":
            return f"{value * 100:.1f}%"
        if self.unit == "pp":
            return f"{value * 100:+.1f} pp"
        if self.unit == "ms":
            return f"{value:.0f} ms"
        if self.unit == "usd":
            return f"${value:.4f}"
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    def format_target(self) -> str:
        if self.target is None:
            return "review"
        sign = "≥" if self.higher_is_better else "≤"
        return f"{sign} {self.format(self.target)}"


KPIS: tuple[KPI, ...] = (
    # Quality of the full system, the numbers a user feels.
    KPI("false_positive_rate", "Quality", "Legitimate emails marked as spam", 0.01, False, "%",
        "Losing a real email is the costliest error."),
    KPI("spam_recall", "Quality", "Spam caught (recall)", 0.95, True, "%",
        "Share of spam that never reaches the inbox."),
    KPI("f1", "Quality", "F1 score, spam", 0.95, True, "%",
        "Balance of precision and recall."),
    KPI("model_f1", "Quality", "F1 score, model alone", 0.95, True, "%",
        "The fine-tuned model must be strong without help."),
    # Does the hybrid design pay off?
    KPI("llm_uplift", "Hybrid design", "Accuracy gain from the LLM on uncertain emails", 0.0, True, "pp",
        "The LLM must beat the model where it is routed, or it is wasted cost."),
    KPI("llm_share", "Hybrid design", "Emails sent to the LLM", 0.15, False, "%",
        "Keeps cost and latency low: most emails are decided by the model."),
    KPI("rules_share", "Hybrid design", "Emails falling back to rules", 0.01, False, "%",
        "High values mean the LLM is failing or not configured."),
    # Operations.
    KPI("latency_p95_model", "Operations", "Latency p95, model path", 300.0, False, "ms",
        "Most requests take this path."),
    KPI("latency_p95_llm", "Operations", "Latency p95, LLM path", 3000.0, False, "ms",
        "Uncertain emails can wait a little longer."),
    KPI("cost_per_1000", "Operations", "LLM cost per 1,000 emails", 0.10, False, "usd",
        "Running cost of the fallback."),
    # Are the explanations trustworthy?
    KPI("ig_convergence_delta", "Explainability", "IG completeness error (mean)", 0.05, False, "",
        "Attributions must add up to the model's change in probability."),
    KPI("faithfulness_gap", "Explainability", "Top-5 comprehensiveness minus random", 0.10, True, "",
        "Removing the explained words must hurt far more than removing random ones."),
    KPI("beats_random_share", "Explainability", "Explanations beating random (top-5)", 0.80, True, "%",
        "Faithful for most emails, not just on average."),
    KPI("sufficiency", "Explainability", "Top-5 sufficiency (probability drop)", 0.20, False, "",
        "The explained words alone should nearly keep the prediction."),
    KPI("shortcut_candidates", "Explainability", "Shortcut words to review", None, False, "",
        "Each flagged word is checked by hand before release."),
    # Does it generalise beyond the training data? Tracked, no target: the honest
    # answer for a model trained on 2000s spam is what this measures.
    KPI("phishing_recall", "Robustness", "Modern phishing caught (out of distribution)", None, True, "%",
        "Spam has changed since the Enron era; this shows how much that matters."),
    KPI("phishing_false_positive_rate", "Robustness", "Safe modern emails marked as spam", None, False,
        "%", "False alarms on present-day legitimate email."),
)


def _get(data: dict | None, *path):
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def collect(
    training: dict | None,
    system: dict | None,
    explainability: dict | None,
    phishing: dict | None = None,
) -> dict:
    """Pulls each KPI's value out of the result files. Missing parts become None."""
    faith5 = _get(explainability, "faithfulness", "5")
    comp, rand = _get(faith5, "comprehensiveness"), _get(faith5, "random_comprehensiveness")
    candidates = _get(explainability, "shortcut_audit", "shortcut_candidates")
    return {
        "false_positive_rate": _get(system, "hybrid", "false_positive_rate"),
        "spam_recall": _get(system, "hybrid", "recall"),
        "f1": _get(system, "hybrid", "f1"),
        "model_f1": _get(training, "f1_spam"),
        "llm_uplift": _get(system, "uncertain_emails", "uplift"),
        "llm_share": _get(system, "routing", "llm_share"),
        "rules_share": _get(system, "routing", "rules_share"),
        "latency_p95_model": _get(system, "latency", "model", "p95_ms")
        if _get(system, "latency", "model", "count") else None,
        "latency_p95_llm": _get(system, "latency", "llm", "p95_ms")
        if _get(system, "latency", "llm", "count") else None,
        "cost_per_1000": _get(system, "llm_usage", "cost_per_1000_emails_usd")
        if _get(system, "llm_enabled") else None,
        "ig_convergence_delta": _get(explainability, "model", "mean_convergence_delta"),
        "faithfulness_gap": comp - rand if comp is not None and rand is not None else None,
        "beats_random_share": _get(faith5, "beats_random_share"),
        "sufficiency": _get(faith5, "sufficiency"),
        "shortcut_candidates": len(candidates) if candidates is not None else None,
        "phishing_recall": _get(phishing, "hybrid", "recall"),
        "phishing_false_positive_rate": _get(phishing, "hybrid", "false_positive_rate"),
    }


def evaluate(values: dict) -> list[dict]:
    return [
        {
            "key": kpi.key,
            "group": kpi.group,
            "name": kpi.name,
            "target": kpi.format_target(),
            "value": values.get(kpi.key),
            "display": kpi.format(values.get(kpi.key)),
            "status": kpi.status(values.get(kpi.key)),
            "why": kpi.why,
        }
        for kpi in KPIS
    ]


def to_markdown(rows: list[dict]) -> str:
    counts = {s: sum(r["status"] == s for r in rows) for s in ("pass", "miss", "review", "not measured")}
    lines = [
        "# KPI report",
        "",
        f"**{counts['pass']} pass · {counts['miss']} miss · {counts['review']} to review · "
        f"{counts['not measured']} not measured**",
        "",
        "| Group | KPI | Target | Measured | Status |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(f"| {r['group']} | {r['name']} | {r['target']} | {r['display']} | {r['status']} |")
    lines += ["", "## Why these KPIs", ""]
    lines += [f"- **{r['name']}:** {r['why']}" for r in rows]
    return "\n".join(lines) + "\n"


def any_missed(rows: list[dict]) -> bool:
    return any(r["status"] == "miss" for r in rows)
