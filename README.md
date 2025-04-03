# 📧 Email Filtering System using DistilBERT

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)
![Transformers](https://img.shields.io/badge/🤗_Transformers-4.30%2B-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green)

## 🌟 Overview
A hybrid spam detection system combining:
- **DistilBERT** for deep learning classification
- **Rule-based checks** for common spam patterns
- **GPT-3.5 fallback** for low-confidence predictions
- **FastAPI** REST endpoint for production deployment

## 🛠️ Core Components

### 1. Model Training (`model.py`)

Key Features:
- Uses DistilBERT (lightweight BERT variant)
- Processes email text with regex cleaning
- Handles class imbalance with sklearn's class_weight
- Saves TensorFlow SavedModel format


### 2. Prediction API (app.py)
Key Features:
- FastAPI endpoint with /predict route
- Hybrid decision system:
  1. DistilBERT primary prediction
  2. GPT-3.5 fallback for uncertain cases
  3. Rule-based keyword matching as backup
- Cached LLM queries with @lru_cache
- Detailed logging and health checks

  ## 📦 Installation
```bash
git clone https://github.com/your-repo/Email-Filtering-Using-DistilBERT
cd Email-Filtering-Using-DistilBERT
pip install -r requirements.txt
```

## 🚀 Usage
Training the Model
```bash
python model.py
  ```

Running the API
```bash
python app.py
```
Then visit http://localhost:8080/docs for interactive docs.

## 🔧 Configuration
Add to .env:
```ini
OPENAI_API_KEY=sk-your-key-here  # For GPT-3.5 fallback
```

## 📂 File Structure
```
.
├── app.py                    # FastAPI application
├── model.py         # Model training script
├── spam_detection_model/     # Saved TensorFlow model
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 📜 License
MIT License - For research and educational use only.
