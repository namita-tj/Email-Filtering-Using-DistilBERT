# Spam Detection API

A Flask-based API for detecting spam emails using a pre-trained Transformer model from Hugging Face.

## Features
- Classifies emails as "spam" or "ham".
- Easy-to-use RESTful API.
- Downloads the model from AWS S3 if not available locally.

## Setup

### Prerequisites
- Python 3.8 or higher
- AWS CLI (for S3 access, if needed)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-detection.git
   cd spam-detection
