from fastapi import APIRouter, Request
from app.core.config import settings
from app.core.cls import My_model, drop_punctuation, normilise_text
from app.s3_client import S3Client
from pydantic import BaseModel
import torch

router = APIRouter()

s3_client = S3Client(
        settings.S3_ACCESS_KEY_ID,
        settings.S3_SECRET_ACCESS_KEY,
        settings.PUBLIC_URL,
        default_bucket_name='ads-classifier'
        )

m = My_model(404, 70, 70)

s3_client.download_model()

class ProcessingRequest(BaseModel):
    input: str

@router.post("/process")
async def predict_ads_category(request: Request, text: ProcessingRequest):
    res = request.app.model.predict(text.input, drop_punctuation, normilise_text, request.app.vectorizer)
    return res
