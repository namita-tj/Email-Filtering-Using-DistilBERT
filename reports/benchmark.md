# Benchmark

Full test sets (Enron: 2,000, Modern phishing: 2,000 emails). Spam cut-off 0.5 for every model.

| Approach | Type | Enron: F1 | Enron: legit flagged | Enron: spam caught | Modern phishing: F1 | Modern phishing: legit flagged | Modern phishing: spam caught | ms / email |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive Bayes (word + bigram counts) | classical | 0.990 | 0.8% | 98.7% | 0.839 | 23.2% | 99.2% | 0.56 |
| Linear SVM (TF-IDF) | classical | 0.989 | 1.6% | 99.4% | 0.830 | 25.2% | 99.6% | 0.36 |
| Logistic regression (TF-IDF), project baseline | classical | 0.986 | 1.9% | 99.1% | 0.844 | 22.6% | 99.5% | 0.35 |
| DistilBERT, fine-tuned (this project) | fine-tuned transformer | 0.992 | 0.9% | 99.2% | 0.768 | 37.4% | 99.6% | 10.36 |

## Same random subset for every approach, including the LLM (Enron: 200, Modern phishing: 200 emails)

| Approach | Type | Enron: F1 | Enron: legit flagged | Enron: spam caught | Modern phishing: F1 | Modern phishing: legit flagged | Modern phishing: spam caught |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive Bayes (word + bigram counts) | classical | 0.986 | 3.1% | 100.0% | 0.857 | 20.7% | 98.7% |
| Linear SVM (TF-IDF) | classical | 0.990 | 2.1% | 100.0% | 0.862 | 19.8% | 98.7% |
| Logistic regression (TF-IDF), project baseline | classical | 0.981 | 3.1% | 99.0% | 0.872 | 18.2% | 98.7% |
| DistilBERT, fine-tuned (this project) | fine-tuned transformer | 0.986 | 2.1% | 99.0% | 0.768 | 38.0% | 98.7% |
| Qwen2.5-3B-Instruct, zero-shot (local, no training) | zero-shot LLM (local) | 0.817 | 46.9% | 99.0% | 0.626 | 74.4% | 97.5% |

The LLM ran locally on the GPU: no API cost (400 emails).
