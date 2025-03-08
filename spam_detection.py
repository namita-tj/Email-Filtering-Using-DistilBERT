import nltk
import re
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Disable oneDNN warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ✅ NLTK Setup
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# ✅ Text Cleaning
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars
    return text.strip()

# ✅ Combine Title and Text
def combine_title_text(example):
    example["combined_text"] = clean_text(example["title"]) + " " + clean_text(example["text"])
    return example

# ✅ Load Dataset
dataset = load_dataset("TrainingDataPro/email-spam-classification")
dataset = dataset.map(combine_title_text)

# ✅ Encode Labels (spam = 1, ham = 0)
def encode_labels(example):
    example["label"] = 1 if example["type"] == "spam" else 0
    return example

dataset = dataset.map(encode_labels)

# ✅ Split Dataset into Train and Validation
if "validation" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)  # 80% train, 20% validation

# ✅ Handle Class Imbalance
labels = [ex["label"] for ex in dataset["train"]]
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weights = {i: w for i, w in enumerate(class_weights)}

# ✅ Tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_data(examples):
    return tokenizer(
        examples["combined_text"],
        padding="max_length",
        truncation=True,
        max_length=512  # Fixed sequence length
    )

tokenized_dataset = dataset.map(tokenize_data, batched=True)

# ✅ Prepare TensorFlow Datasets
train_dataset = tokenized_dataset["train"].shuffle(seed=42)
val_dataset = tokenized_dataset["test"].shuffle(seed=42)  # Use "test" split as validation

tf_train_dataset = train_dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    batch_size=16,
    shuffle=True
)

tf_val_dataset = val_dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    batch_size=16,
    shuffle=False
)

# ✅ Load Model
model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# ✅ Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# ✅ Train Model
print("Training model...")
history = model.fit(
    tf_train_dataset,
    validation_data=tf_val_dataset,
    epochs=5,
    class_weight=class_weights
)

# ✅ Evaluate Model
print("\nEvaluation Results:")
results = model.evaluate(tf_val_dataset)
print(f"Validation Loss: {results[0]:.4f}, Validation Accuracy: {results[1]:.4f}")

# ✅ Generate Predictions
y_pred = model.predict(tf_val_dataset).logits.argmax(axis=1)
y_true = np.concatenate([y for x, y in tf_val_dataset], axis=0)

# ✅ Confusion Matrix and Classification Report
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["ham", "spam"]))

# ✅ Save Model
# Save in TensorFlow's recommended format instead of HDF5 (.h5)
model.save("spam_detection_model", save_format="tf")

print("\nModel saved successfully!")

# Verify files
import os

model_dir = "spam_detection_model"
if os.path.exists(model_dir):
    print("Files in model directory:", os.listdir(model_dir))
else:
    print("Model directory not found.")