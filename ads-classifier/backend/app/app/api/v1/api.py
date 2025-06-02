from fastapi import APIRouter
from app.api.v1.endpoints import (
    weather, predict  
)

api_router = APIRouter()
api_router.include_router(weather.router, prefix="/weather", tags=["weather"])
api_router.include_router(predict.router, prefix="", tags=['predict'])