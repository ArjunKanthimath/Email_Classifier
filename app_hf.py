from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
from pathlib import Path
import torch
import json
import sys

from models import EmailClassifier
from utils import PiiMasker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Email Classification API",
    description="API for classifying and masking PII in support emails"
)

# Setup templates for the web interface
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Initialize classifier with proper device handling
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = EmailClassifier(device=device, model_dir="./model_data")
model_loaded = classifier.load_model()
if not model_loaded:
    logger.warning("Model not loaded. This is expected if you need to train first.")

# Initialize PII masker
try:
    pii_masker = PiiMasker()
except Exception as e:
    logger.error(f"Failed to initialize PII masker: {e}")
    raise RuntimeError("Failed to initialize PII masker") from e

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
    """Classify email and mask PII"""
    try:
        if not model_loaded and not classifier.load_model():
            raise HTTPException(status_code=500, detail="Model not loaded or trained")

        if not request.email_body or len(request.email_body.strip()) == 0:
            raise HTTPException(status_code=400, detail="Email body cannot be empty")

        masked_email, entities = pii_masker.extract_masked_entities(request.email_body)
        logger.info(f"Entities returned from mask_email: {entities}")
        category = classifier.predict(masked_email)

        # Safely format the response
        safe_entities = [
            {
                "position": [
                    entity.get("start_index", 0),
                    entity.get("end_index", 0)
                ],
                "classification": entity.get("entity_type", "unknown"),
                "entity": entity.get("entity_value", "")
            } for entity in entities
        ]

        return {
            "input_email_body": request.email_body,
            "list_of_masked_entities": safe_entities,
            "masked_email": masked_email,
            "category_of_the_email": category
        }

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the web interface"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "show_results": False,
        "device": str(device)
    })

@app.post("/", response_class=HTMLResponse)
async def process_email(request: Request, email_body: str = Form(...)):
    """Process email from web form"""
    try:
        if not model_loaded and not classifier.load_model():
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Model not loaded or trained",
                "show_results": False
            })

        if not email_body or len(email_body.strip()) == 0:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Email body cannot be empty",
                "show_results": False
            })

        masked_email, entities = pii_masker.extract_masked_entities(email_body)
        logger.info(f"Entities returned from mask_email: {entities}")
        category = classifier.predict(masked_email)

        # Safely format the JSON response
        formatted_json = {
            "input_email_body": email_body,
            "list_of_masked_entities": [
                {
                    "position": [
                        entity.get("start_index", 0),
                        entity.get("end_index", 0)
                    ],
                    "classification": entity.get("entity_type", "unknown"),
                    "entity": entity.get("entity_value", "")
                } for entity in entities
            ],
            "masked_email": masked_email,
            "category_of_the_email": category
        }

        # Pretty print the JSON
        pretty_json = json.dumps(formatted_json, indent=2, ensure_ascii=False)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "input_email_body": email_body,
            "masked_email": masked_email,
            "category": category,
            "entities": entities,
            "show_results": True,
            "device": str(device),
            "formatted_json": pretty_json
        })

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Error processing request: {str(e)}",
            "show_results": False
        })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok" if model_loaded else "warning",
        "model_loaded": model_loaded,
        "device": str(device),
        "message": "" if model_loaded else "Model not loaded. Please train or load a model."
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"}
    )
