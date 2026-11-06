# KPI report

**9 pass · 3 miss · 3 to review · 2 not measured**

| Group | KPI | Target | Measured | Status |
| --- | --- | ---: | ---: | --- |
| Quality | Legitimate emails marked as spam | ≤ 1.0% | 0.9% | pass |
| Quality | Spam caught (recall) | ≥ 95.0% | 99.2% | pass |
| Quality | F1 score, spam | ≥ 95.0% | 99.2% | pass |
| Quality | F1 score, model alone | ≥ 95.0% | 99.5% | pass |
| Hybrid design | Accuracy gain from the LLM on uncertain emails | ≥ +0.0 pp | – | not measured |
| Hybrid design | Emails sent to the LLM | ≤ 15.0% | 0.0% | pass |
| Hybrid design | Emails falling back to rules | ≤ 1.0% | 0.0% | pass |
| Operations | Latency p95, model path | ≤ 300 ms | 165 ms | pass |
| Operations | Latency p95, LLM path | ≤ 3000 ms | – | not measured |
| Operations | LLM cost per 1,000 emails | ≤ $0.1000 | $0.0000 | pass |
| Explainability | IG completeness error (mean) | ≤ 0.050 | 0.159 | miss |
| Explainability | Top-5 comprehensiveness minus random | ≥ 0.100 | 0.009 | miss |
| Explainability | Explanations beating random (top-5) | ≥ 80.0% | 57.0% | miss |
| Explainability | Top-5 sufficiency (probability drop) | ≤ 0.200 | 0.092 | pass |
| Explainability | Shortcut words to review | review | 12 | review |
| Robustness | Modern phishing caught (out of distribution) | review | 99.6% | review |
| Robustness | Safe modern emails marked as spam | review | 36.6% | review |

## Why these KPIs

- **Legitimate emails marked as spam:** Losing a real email is the costliest error.
- **Spam caught (recall):** Share of spam that never reaches the inbox.
- **F1 score, spam:** Balance of precision and recall.
- **F1 score, model alone:** The fine-tuned model must be strong without help.
- **Accuracy gain from the LLM on uncertain emails:** The LLM must beat the model where it is routed, or it is wasted cost.
- **Emails sent to the LLM:** Keeps cost and latency low: most emails are decided by the model.
- **Emails falling back to rules:** High values mean the LLM is failing or not configured.
- **Latency p95, model path:** Most requests take this path.
- **Latency p95, LLM path:** Uncertain emails can wait a little longer.
- **LLM cost per 1,000 emails:** Running cost of the fallback.
- **IG completeness error (mean):** Attributions must add up to the model's change in probability.
- **Top-5 comprehensiveness minus random:** Removing the explained words must hurt far more than removing random ones.
- **Explanations beating random (top-5):** Faithful for most emails, not just on average.
- **Top-5 sufficiency (probability drop):** The explained words alone should nearly keep the prediction.
- **Shortcut words to review:** Each flagged word is checked by hand before release.
- **Modern phishing caught (out of distribution):** Spam has changed since the Enron era; this shows how much that matters.
- **Safe modern emails marked as spam:** False alarms on present-day legitimate email.
