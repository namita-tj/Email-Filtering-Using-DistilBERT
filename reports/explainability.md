# Explainability evaluation

200 validation emails (100 spam, 100 ham), Integrated Gradients with 32 steps.

## Model on this sample

| Metric | Value |
| --- | ---: |
| Accuracy | 99.5% |
| Share in the uncertain band (sent to the LLM) | 0.0% |
| Mean IG convergence error | 0.1589 |

## Faithfulness (ERASER deletion tests)

Drop in the predicted class's probability. Higher comprehensiveness and lower sufficiency mean a more faithful explanation.

| Top-k words | Comprehensiveness | Random words removed | Sufficiency | Beats random |
| ---: | ---: | ---: | ---: | ---: |
| 1 | +0.0062 | +0.0006 | +0.1838 | 49.0% |
| 3 | +0.0115 | +0.0032 | +0.1332 | 60.0% |
| 5 | +0.0185 | +0.0096 | +0.0917 | 57.0% |

## Shortcut audit

610 words appearing in at least 5 emails.

### Candidates to review

| Word | Emails | Mean weight | Share in spam | Flag |
| --- | ---: | ---: | ---: | --- |
| enron | 46 | -0.1425 | 0.0% | class-exclusive |
| daren | 7 | -0.1270 | 0.0% | class-exclusive |
| hourahead | 5 | -0.0849 | 0.0% | class-exclusive |
| hpl | 5 | -0.0565 | 0.0% | class-exclusive |
| vince | 18 | -0.0524 | 0.0% | class-exclusive |
| kaminski | 13 | -0.0511 | 0.0% | class-exclusive |
| schedules | 5 | -0.0429 | 0.0% | class-exclusive |
| meeting | 14 | -0.0284 | 0.0% | class-exclusive |
| portland | 5 | -0.0282 | 0.0% | class-exclusive |
| october | 6 | -0.0261 | 0.0% | class-exclusive |
| contract | 5 | -0.0204 | 0.0% | class-exclusive |
| questions | 8 | -0.0200 | 0.0% | class-exclusive |

### Words pushing towards spam

| Word | Emails | Mean weight | Share in spam |
| --- | ---: | ---: | ---: |
| julie | 5 | +0.0195 | 40.0% |
| north | 5 | +0.0085 | 20.0% |
| higher | 5 | +0.0032 | 60.0% |
| say | 5 | +0.0025 | 60.0% |
| technology | 9 | +0.0020 | 77.8% |
| stock | 8 | +0.0011 | 75.0% |
| offers | 5 | +0.0011 | 60.0% |
| agent | 5 | +0.0010 | 60.0% |
| call | 7 | +0.0008 | 14.3% |
| page | 5 | +0.0005 | 40.0% |

### Words pushing towards ham

| Word | Emails | Mean weight | Share in spam |
| --- | ---: | ---: | ---: |
| enron | 46 | -0.1425 | 0.0% |
| daren | 7 | -0.1270 | 0.0% |
| hourahead | 5 | -0.0849 | 0.0% |
| hpl | 5 | -0.0565 | 0.0% |
| vince | 18 | -0.0524 | 0.0% |
| kaminski | 13 | -0.0511 | 0.0% |
| schedules | 5 | -0.0429 | 0.0% |
| schedule | 8 | -0.0365 | 12.5% |
| march | 5 | -0.0291 | 20.0% |
| meeting | 14 | -0.0284 | 0.0% |
