from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import logging

from models import EmailClassifier
from utils import mask_email

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Email Classification API",
              description="API for classifying and masking PII in support emails")

# Initialize classifier
classifier = EmailClassifier()
model_loaded = classifier.load_model()
if not model_loaded:
    logger.warning("Model not loaded. Please train or load a model before making predictions.")

class EmailRequest(BaseModel):
    """Request model for email classification"""
    email_body: str

class EmailResponse(BaseModel):
    """Response model for email classification"""
    input_email_body: str
    list_of_masked_entities: List[Dict[str, Any]]
    masked_email: str
    category_of_the_email: str

@app.post("/classify", response_model=EmailResponse)
async def classify_email(request: EmailRequest):
    """
    Classify email and mask PII.

    Parameters:
    - email_body: The email text to classify

    Returns:
    - input_email_body: Original email text
    - list_of_masked_entities: List of detected PII entities
    - masked_email: Email with PII masked
    - category_of_the_email: Predicted email category
    """
    try:
        # Check if email body is provided
        if not request.email_body or len(request.email_body.strip()) == 0:
            raise HTTPException(status_code=400, detail="Email body cannot be empty")

        # Mask PII entities
        masked_email, entities = mask_email(request.email_body)

        # Classify the masked email
        category = classifier.predict(masked_email)

        # Format the response
        response = {
            "input_email_body": request.email_body,
            "list_of_masked_entities": entities,
            "masked_email": masked_email,
            "category_of_the_email": category
        }

        return response

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model_loaded:
        return {"status": "ok", "model_loaded": True}
    else:
        return {"status": "warning", "model_loaded": False,
                "message": "Model not loaded. Please train or load a model."}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
