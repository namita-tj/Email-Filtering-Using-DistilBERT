"""DistilBERT spam classifier and its Integrated Gradients explainer.

TensorFlow is imported lazily so the rest of the service (and its tests) can run
without it.
"""

from pathlib import Path

from app.explain import Explanation, merge_wordpieces


class DistilBertClassifier:
    def __init__(self, model_dir: Path | str, max_length: int = 512, tokenizer=None, model=None):
        import tensorflow as tf

        self._tf = tf
        if tokenizer is None or model is None:
            from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = TFAutoModelForSequenceClassification.from_pretrained(model_dir)
        self._tokenizer = tokenizer
        self._model = model
        self._max_length = max_length

    def _encode(self, text: str):
        return self._tokenizer(
            text, return_tensors="tf", truncation=True, max_length=self._max_length
        )

    def spam_probability(self, text: str) -> float:
        logits = self._model(**self._encode(text)).logits
        return float(self._tf.nn.softmax(logits, axis=-1).numpy()[0][1])

    def spam_probabilities(self, texts: list[str], batch_size: int = 16) -> list[float]:
        """Batched version, used by the evaluation scripts."""
        tf = self._tf
        probs: list[float] = []
        for start in range(0, len(texts), batch_size):
            batch = self._tokenizer(
                texts[start:start + batch_size],
                return_tensors="tf",
                truncation=True,
                max_length=self._max_length,
                padding=True,
            )
            logits = self._model(**batch).logits
            probs.extend(tf.nn.softmax(logits, axis=-1).numpy()[:, 1].tolist())
        return probs

    def explain(self, text: str, steps: int = 32, batch_size: int = 16) -> Explanation:
        """Integrated Gradients on the input embeddings.

        Baseline: the same sequence with every word replaced by [PAD], keeping
        [CLS] and [SEP]. Attributions are for the spam probability and are
        summed over the embedding dimension, then merged into words.
        """
        tf = self._tf
        encoded = self._encode(text)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        embedding_matrix = self._model.get_input_embeddings().weight
        special = {self._tokenizer.cls_token_id, self._tokenizer.sep_token_id}
        ids = input_ids.numpy()[0].tolist()
        baseline_ids = [i if i in special else self._tokenizer.pad_token_id for i in ids]

        x = tf.gather(embedding_matrix, input_ids)  # (1, seq, dim)
        baseline = tf.gather(embedding_matrix, tf.constant([baseline_ids]))
        diff = x - baseline

        alphas = tf.linspace(0.0, 1.0, steps + 1)
        grads = []
        for start in range(0, steps + 1, batch_size):
            alpha = tf.reshape(alphas[start:start + batch_size], (-1, 1, 1))
            interpolated = baseline + alpha * diff
            mask = tf.repeat(attention_mask, tf.shape(interpolated)[0], axis=0)
            with tf.GradientTape() as tape:
                tape.watch(interpolated)
                logits = self._model(input_ids=None, inputs_embeds=interpolated, attention_mask=mask).logits
                spam = tf.nn.softmax(logits, axis=-1)[:, 1]
            grads.append(tape.gradient(spam, interpolated))
        grads = tf.concat(grads, axis=0)

        # Trapezoidal rule over the path, then attribution = (x - baseline) * mean gradient.
        avg_grads = tf.reduce_mean((grads[:-1] + grads[1:]) / 2.0, axis=0, keepdims=True)
        token_scores = tf.reduce_sum(diff * avg_grads, axis=-1).numpy()[0].tolist()

        def prob(embeds):
            logits = self._model(input_ids=None, inputs_embeds=embeds, attention_mask=attention_mask).logits
            return float(tf.nn.softmax(logits, axis=-1).numpy()[0][1])

        p_input, p_baseline = prob(x), prob(baseline)
        # IG completeness: attributions should sum to p(input) - p(baseline).
        delta = abs(sum(token_scores) - (p_input - p_baseline))

        tokens = self._tokenizer.convert_ids_to_tokens(ids)
        return Explanation(
            method="integrated_gradients",
            spam_probability=p_input,
            words=merge_wordpieces(tokens, token_scores),
            convergence_delta=delta,
        )
