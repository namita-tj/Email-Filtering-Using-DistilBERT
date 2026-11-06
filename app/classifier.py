"""DistilBERT spam classifier. TensorFlow is imported lazily so the rest of the
service (and its tests) can run without it."""

from pathlib import Path


class DistilBertClassifier:
    def __init__(self, model_dir: Path, max_length: int = 512):
        import tensorflow as tf
        from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

        self._tf = tf
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = TFAutoModelForSequenceClassification.from_pretrained(model_dir)
        self._max_length = max_length

    def spam_probability(self, text: str) -> float:
        inputs = self._tokenizer(
            text, return_tensors="tf", truncation=True, max_length=self._max_length
        )
        logits = self._model(**inputs).logits
        probs = self._tf.nn.softmax(logits, axis=-1).numpy()[0]
        return float(probs[1])
