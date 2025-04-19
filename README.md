# Email Classification System

This project implements an email classification system for a company's support team. The system categorizes incoming support emails into predefined categories while ensuring that personal information (PII) is masked before processing. The trained BERT Model has an accuracy of 78.77%.

## Features

- Email classification using BERT
- PII masking using Named Entity Recognition (NER) and regex patterns
- FastAPI-based API for email classification
- Supports various PII entity types (full name, email, phone number, etc.)

## Setup Instructions

### Prerequisites

- Python 3.7+
- PyTorch
- Transformers library
- Spacy
- FastAPI

### Installation

1. Clone this repository:
```
git clone https://github.com/ArjunKanthimath/Email_Classifier.git
cd Email_Classifier
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Download the SpaCy English model:
```
python -m spacy download en_core_web_sm
```

### Training the Model

1. Prepare your training data in a CSV file with 'email' and 'type' columns.
2. Run the training script:
```python
from models import EmailClassifier

classifier = EmailClassifier()
classifier.train('path/to/your/data.csv', epochs=4, batch_size=16)
```

### Running the API

```
python app.py
```

The API will be available at `http://localhost:8000`.

## API Documentation

### Classify Email

**Endpoint:** `POST /classify`

**Request Body:**
```json
{
    "email_body": "Your email text here"
}
```

**Response:**
```json
{
    "input_email_body": "Original email text",
    "list_of_masked_entities": [
        {
            "position": [start_index, end_index],
            "classification": "entity_type",
            "entity": "original_entity_value"
        }
    ],
    "masked_email": "Masked email text",
    "category_of_the_email": "Predicted category"
}
```

### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
    "status": "ok",
    "model_loaded": true
}
```

## Deployment on Hugging Face Spaces

1. Create a new Space on Hugging Face with Docker template
2. Upload the project files to the Space
3. Set up the Space to run the FastAPI application

## File Structure

- `app.py`: FastAPI application
- `models.py`: Email classification model
- `utils.py`: PII masking utilities
- `requirements.txt`: Required packages
