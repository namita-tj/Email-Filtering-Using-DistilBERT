# 📧 Email Spam Detection Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)
![AWS](https://img.shields.io/badge/AWS-S3%2C_EC2-yellow)
![NLP](https://img.shields.io/badge/NLP-spaCy%2Fnltk-green)

## 🌟 Overview
A machine learning system that classifies emails as spam or ham (non-spam) using:
- **Natural Language Processing (NLP)** for text preprocessing
- **Supervised learning** (Logistic Regression/Random Forest)
- **AWS S3** for scalable model storage
- **Git** for version control (without large file bloat)

## 🛠️ Tools Used
| Category       | Tools                                                                 |
|----------------|-----------------------------------------------------------------------|
| Core ML        | `scikit-learn`, `nltk`, `spaCy`, `pandas`, `numpy`                   |
| Cloud Storage  | **AWS S3** (`spam-detection-model-storage`)                          |
| Development   | `Python 3.8+`, `Jupyter Notebook`, `VS Code`                         |
| Version Control| `Git` (with `.gitignore` for models)                                 |

## 🔧 Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure AWS Credentials 
```bash
aws configure
```
Or set environment variables:
```
export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
export AWS_DEFAULT_REGION="us-east-1"
```

## 📂 Downloading the Model
Model path: s3://spam-detection-model-storage/spam_detection_model/

### Option 1: AWS CLI
```bash
aws s3 cp s3://spam-detection-model-storage/spam_detection_model/latest_model.pkl ./models/
```

### Option 2: Python (boto3)
```bash
import boto3
s3 = boto3.client('s3')
s3.download_file(
    Bucket="spam-detection-model-storage",
    Key="spam_detection_model/latest_model.pkl",
    Filename="./models/latest_model.pkl"
)
```

### Option 3: Pre-Signed URL
Run this to generate a temporary link:
```bash
import boto3
s3 = boto3.client('s3')
url = s3.generate_presigned_url(
    'get_object',
    Params={
        'Bucket': 'spam-detection-model-storage',
        'Key': 'spam_detection_model/latest_model.pkl'
    },
    ExpiresIn=604800  # 7 days
)
print("Download URL:", url)
```

## 🔄 Updating the Model
```bash
aws s3 cp ./new_model.pkl s3://spam-detection-model-storage/spam_detection_model/latest_model.pkl
```

## 📁 S3 Structure
```bash
spam_detection_model/
├── latest_model.pkl
├── v1.0_model.pkl
└── training_data.csv
```

## 🔒 Access
Requires s3:GetObject permissions

Add this to your .gitignore:
```bash
/models/
```

## 🚀 Deployment
Reference directly in AWS services:
```bash
model_path = "s3://spam-detection-model-storage/spam_detection_model/latest_model.pkl"
```
