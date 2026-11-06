"""Turns evaluation results into a short Markdown report."""


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _word_table(rows: list[dict], with_flag: bool = False) -> str:
    if not rows:
        return "_None found._\n"
    header = "| Word | Emails | Mean weight | Share in spam |" + (" Flag |" if with_flag else "")
    sep = "| --- | ---: | ---: | ---: |" + (" --- |" if with_flag else "")
    lines = [header, sep]
    for r in rows:
        line = f"| {r['word']} | {r['emails']} | {r['mean_weight']:+.4f} | {_pct(r['spam_share'])} |"
        if with_flag:
            line += f" {r['flag']} |"
        lines.append(line)
    return "\n".join(lines) + "\n"


def to_markdown(results: dict) -> str:
    model = results["model"]
    lines = [
        "# Explainability evaluation",
        "",
        f"{results['samples']} validation emails ({results['spam']} spam, {results['ham']} ham), "
        f"Integrated Gradients with {results['steps']} steps.",
        "",
        "## Model on this sample",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Accuracy | {_pct(model['accuracy'])} |",
        f"| Share in the uncertain band (sent to the LLM) | {_pct(model['uncertain_share'])} |",
        f"| Mean IG convergence error | {model['mean_convergence_delta']:.4f} |",
        "",
        "## Faithfulness (ERASER deletion tests)",
        "",
        "Drop in the predicted class's probability. Higher comprehensiveness and lower "
        "sufficiency mean a more faithful explanation.",
        "",
        "| Top-k words | Comprehensiveness | Random words removed | Sufficiency "
        "| Beats random |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for k, s in sorted(results["faithfulness"].items(), key=lambda item: int(item[0])):
        lines.append(
            f"| {k} | {s['comprehensiveness']:+.4f} | {s['random_comprehensiveness']:+.4f} "
            f"| {s['sufficiency']:+.4f} | {_pct(s['beats_random_share'])} |"
        )

    audit = results["shortcut_audit"]
    lines += [
        "",
        "## Shortcut audit",
        "",
        f"{audit['words_analysed']} words appearing in at least {audit.get('min_emails', 5)} emails.",
        "",
        "### Candidates to review",
        "",
        _word_table(audit["shortcut_candidates"], with_flag=True),
        "### Words pushing towards spam",
        "",
        _word_table(audit["top_spam_words"][:10]),
        "### Words pushing towards ham",
        "",
        _word_table(audit["top_ham_words"][:10]),
    ]
    return "\n".join(lines)
