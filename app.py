from flask import Flask, request, jsonify
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load trained spam detection model
model = TFAutoModelForSequenceClassification.from_pretrained("spam_detection_model")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


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


# Run Flask app locally
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
