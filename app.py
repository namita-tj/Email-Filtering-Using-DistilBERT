import boto3
import os
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from flask import Flask, request, jsonify

# AWS S3 Setup
S3_BUCKET_NAME = "spam-detection-model-storage"
MODEL_DIR = "spam_detection_model"

# Ensure directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model from S3 if not available locally
if not os.path.exists(f"{MODEL_DIR}/tf_model.h5"):
    print("Downloading model from S3...")
    s3 = boto3.client("s3")

    for obj in s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=MODEL_DIR)["Contents"]:
        file_path = obj["Key"]
        local_path = os.path.join(MODEL_DIR, file_path.split("/")[-1])
        s3.download_file(S3_BUCKET_NAME, file_path, local_path)

    print("Model downloaded successfully!")

# Load TensorFlow model and tokenizer
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Spam Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        email_text = data.get("text", "")

        if not email_text:
            return jsonify({"error": "No email text provided"}), 400

        # Tokenize and classify
        inputs = tokenizer(email_text, return_tensors="tf", padding="max_length", truncation=True, max_length=512)
        logits = model(inputs["input_ids"]).logits
        prediction = tf.argmax(logits, axis=1).numpy()[0]

        return jsonify({"prediction": "spam" if prediction == 1 else "ham"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)