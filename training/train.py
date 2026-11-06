"""Fine-tunes DistilBERT for spam detection.

Run from the repository root:  python -m training.train

Dataset: Enron-Spam (see training/data.py). The model is fitted on the train
split; metrics and the confidence band come from the validation split. The test
split is left for evaluation/run.py.

Settings (environment variables):
  MAX_LENGTH   tokens per email (default 256; covers most emails, and is
               much faster than 512)
  EPOCHS       default 2; DistilBERT converges quickly on this task
  TRAIN_LIMIT  use only the first N training emails, for a quick trial run

Writes to MODEL_DIR (default: spam_detection_model/):
  config.json, tf_model.h5, tokenizer files  -> loaded by the API
  training_config.json                        -> max length, so serving matches training
  metrics.json                                -> validation results
  thresholds.json                             -> confidence band for the LLM fallback
"""

import json
import os
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Hugging Face TF models need Keras 2 (the tf_keras package); TensorFlow 2.16+ defaults to Keras 3.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification  # noqa: E402

from training.data import SEED, load_splits  # noqa: E402
from training.thresholds import choose_thresholds  # noqa: E402

MODEL_DIR = Path(os.getenv("MODEL_DIR", "spam_detection_model"))
BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
BATCH_SIZE = 16
EPOCHS = int(os.getenv("EPOCHS", "2"))
TRAIN_LIMIT = int(os.getenv("TRAIN_LIMIT", "0"))
LEARNING_RATE = 5e-5
TARGET_PRECISION = 0.99


dataset = load_splits()
if TRAIN_LIMIT:
    dataset["train"] = dataset["train"].shuffle(seed=SEED).select(
        range(min(TRAIN_LIMIT, len(dataset["train"])))
    )
print({name: len(split) for name, split in dataset.items()})

labels = [ex["label"] for ex in dataset["train"]]
weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(weights))

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


def tokenize(examples):
    return tokenizer(
        examples["combined_text"], padding="max_length", truncation=True, max_length=MAX_LENGTH
    )


tokenized = dataset.map(tokenize, batched=True)
tf_train = tokenized["train"].shuffle(seed=SEED).to_tf_dataset(
    columns=["input_ids", "attention_mask"], label_cols=["label"], batch_size=BATCH_SIZE, shuffle=True
)
tf_val = tokenized["validation"].to_tf_dataset(
    columns=["input_ids", "attention_mask"], label_cols=["label"], batch_size=BATCH_SIZE, shuffle=False
)

model = TFAutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

print("Training model...")
model.fit(tf_train, validation_data=tf_val, epochs=EPOCHS, class_weight=class_weights)

logits = model.predict(tf_val).logits
spam_probs = tf.nn.softmax(logits, axis=-1).numpy()[:, 1]
y_pred = (spam_probs >= 0.5).astype(int)
y_true = np.concatenate([y for _, y in tf_val], axis=0)

print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=["ham", "spam"]))

thresholds = choose_thresholds(spam_probs.tolist(), y_true.tolist(), target=TARGET_PRECISION)
print(
    f"\nConfidence band: {thresholds.low} < p < {thresholds.high} "
    f"({thresholds.uncertain_share:.1%} of validation emails go to the LLM fallback)"
)

metrics = {
    "dataset": "SetFit/enron_spam",
    "train_size": len(dataset["train"]),
    "validation_size": int(len(y_true)),
    "accuracy": float((y_pred == y_true).mean()),
    "f1_spam": float(f1_score(y_true, y_pred)),
    "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    "report": classification_report(
        y_true, y_pred, target_names=["ham", "spam"], output_dict=True
    ),
}

MODEL_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
(MODEL_DIR / "training_config.json").write_text(
    json.dumps({"base_model": BASE_MODEL, "max_length": MAX_LENGTH, "epochs": EPOCHS}, indent=2)
)
(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
(MODEL_DIR / "thresholds.json").write_text(json.dumps(thresholds.to_dict(), indent=2))

print(f"\nSaved model, metrics and thresholds to {MODEL_DIR}/")
