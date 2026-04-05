"""
pneumonia detection api - Fastapi backend
uses a ViT model fine tuned on chest X-rays from huggingface hub.
model : nickmuchi/vit-finetuned-chest-xray-pneumonia
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import time
import torch
from transformers import pipeline
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title='Pneumonia Detection API',
    description='Deep learning-based chest X-ray analysis for pneumonia detection',
    version='1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_ID = 'nickmuchi/vit-finetuned-chest-xray-pneumonia'
classifier = None
model_load_time_ms = None


@app.on_event('startup')
async def load_model():
    global classifier, model_load_time_ms
    logger.info(f'Loading model: {MODEL_ID}')
    t0 = time.perf_counter()
    try:
        classifier = pipeline('image-classification', model=MODEL_ID)
        model_load_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(f'Model loaded in {model_load_time_ms} ms')
    except Exception as e:
        logger.error(f'Failed to load model: {e}')
        raise RuntimeError(f'Model load failed: {e}')


class ScoreItem(BaseModel):
    label: str
    score: float

class PredictionResult(BaseModel):
    label: str
    confidence: float
    all_scores: List[ScoreItem]
    inference_time_ms: float
    model_id: str
    verdict: str
    risk_level: str
    recommendation: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_id: str
    device: str
    model_load_time_ms: Optional[float]


def map_risk(label: str, confidence: float):
    is_pneumonia = 'PNEUMONIA' in label.upper()
    if is_pneumonia:
        if confidence >= 0.85:
            return (
                'Pneumonia detected — Please consult a physician immediately',
                'HIGH',
                'Seek urgent medical attention. A radiologist review is strongly advised.'
            )
        else:
            return (
                'Possible Pneumonia — Further evaluation is needed',
                'MEDIUM',
                'Schedule a follow-up with your doctor for clinical confirmation.'
            )
    return (
        'Normal — No signs of pneumonia detected',
        'LOW',
        'No immediate action required. Maintain routine health check-ups.'
    )


@app.get('/health', response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status='ok',
        model_loaded=classifier is not None,
        model_id=MODEL_ID,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model_load_time_ms=model_load_time_ms
    )


@app.post('/predict', response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    if classifier is None:
        raise HTTPException(status_code=503, detail='Model not loaded yet. Try again shortly')
    if file.content_type not in ('image/jpeg', 'image/png', 'image/jpg', 'image/webp'):
        raise HTTPException(status_code=400, detail='Only JPEG/PNG images are supported')
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail='Could not process uploaded image')

    t0 = time.perf_counter()
    try:
        results = classifier(image, top_k=None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Inference error: {e}')
    elapsed_ms = (time.perf_counter() - t0) * 1000

    results = sorted(results, key=lambda x: x['score'], reverse=True)
    top = results[0]
    verdict, risk_level, recommendation = map_risk(top['label'], top['score'])

    return PredictionResult(
        label=top['label'],
        confidence=top['score'],
        all_scores=[{'label': r['label'], 'score': r['score']} for r in results],
        inference_time_ms=round(elapsed_ms, 2),
        model_id=MODEL_ID,
        verdict=verdict,
        risk_level=risk_level,
        recommendation=recommendation
    )


@app.get('/model-info')
async def model_info():
    return {
        'model_id': MODEL_ID,
        'architecture': 'Vision Transformer (ViT)',
        'task': 'Image Classification - Chest X-Ray',
        'classes': ['NORMAL', 'PNEUMONIA'],
        'source': 'Hugging Face'
    }