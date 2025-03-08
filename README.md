# 📧 Email Spam Detection API

## 🚀 Introduction
This project is an **AI-powered email spam detection API** that classifies emails as **spam** or **ham** using **BERT-based NLP models**.  
It is deployed as a **Flask REST API** on **AWS EC2** and loads the trained model from **AWS S3**.

## 🛠️ Features
- **Real-time spam detection** with a REST API.
- **Uses pre-trained BERT-based Transformer model**.
- **Flask API with JSON request-response structure**.
- **AWS EC2 deployment with model storage in S3**.
- **Supports Docker & Gunicorn for production deployment**.

---


---

## 🖥️ **Installation**
### 🔹 Prerequisites
- Python 3.8+
- `pip` installed
- AWS CLI configured (for model loading)
- Virtual environment (recommended)

### 🔹 Setup

# Clone the repository
git clone https://github.com/your-username/EmailSpamDetection.git
cd EmailSpamDetection

# Create and activate virtual environment
python -m venv myenv
source myenv/bin/activate  # (On Windows use `myenv\Scripts\activate`)

# Install dependencies
pip install -r requirements.txt


