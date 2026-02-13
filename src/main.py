from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from typing import List
import logging
import os

from src.model import load_model, preprocess_image, predict_image

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Image Classification API",
    description="Production-ready ML inference service"
)

@app.on_event("startup")
async def startup():
    load_model()
    logger.info("Model loaded successfully")

class PredictionResponse(BaseModel):
    class_label: str
    probabilities: List[float]

@app.get("/health", status_code=200)
def health():
    return {"status": "ok", "message": "API is healthy and model is loaded."}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files (JPEG/PNG) are allowed."
        )

    try:
        image_bytes = await file.read()
        processed = preprocess_image(image_bytes)
        result = predict_image(processed)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal server error")
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from typing import List
import logging
import os

from src.model import load_model, preprocess_image, predict_image

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Image Classification API",
    description="Production-ready ML inference service"
)

@app.on_event("startup")
async def startup():
    load_model()
    logger.info("Model loaded successfully")

class PredictionResponse(BaseModel):
    class_label: str
    probabilities: List[float]

@app.get("/health", status_code=200)
def health():
    return {"status": "ok", "message": "API is healthy and model is loaded."}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files (JPEG/PNG) are allowed."
        )

    try:
        image_bytes = await file.read()
        processed = preprocess_image(image_bytes)
        result = predict_image(processed)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal server error")
