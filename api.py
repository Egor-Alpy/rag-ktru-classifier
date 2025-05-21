from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import uvicorn
import os
import json
from datetime import datetime

from classifier import ProductClassifier
from config import API_HOST, API_PORT, MAX_CONCURRENT_REQUESTS
from logging_config import setup_logging

logger = setup_logging("api")

app = FastAPI(
    title="KTRU Product Classifier API",
    description="API для классификации товаров по кодам КТРУ с использованием RAG",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить запросы со всех источников
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы
    allow_headers=["*"],  # Разрешить все заголовки
)

# Семафор для ограничения количества одновременных запросов
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


# Модели данных
class ProductRequest(BaseModel):
    """Модель для запроса классификации товара."""
    product: Dict[str, Any]


class ClassificationResponse(BaseModel):
    """Модель для ответа с результатом классификации."""
    ktru_code: str
    processing_time: float


# Инициализация классификатора
product_classifier = ProductClassifier()


@app.get("/")
async def read_root():
    """Корневой маршрут API."""
    return {
        "message": "KTRU Product Classifier API",
        "status": "active",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Проверка работоспособности API."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/classify", response_model=ClassificationResponse)
async def classify_product(request: ProductRequest):
    """
    Классифицирует товар по коду КТРУ.

    Принимает JSON-описание товара и возвращает код КТРУ или "код не найден".
    """
    start_time = datetime.now().timestamp()

    try:
        async with semaphore:
            # Классифицируем товар
            result = product_classifier.classify_product(request.product)
    except Exception as e:
        logger.error(f"Error classifying product: {e}")
        raise HTTPException(status_code=500, detail=f"Error during classification: {str(e)}")

    end_time = datetime.now().timestamp()
    processing_time = end_time - start_time

    return ClassificationResponse(
        ktru_code=result,
        processing_time=processing_time
    )


@app.get("/stats")
async def get_stats():
    """Возвращает статистику работы API."""
    # Здесь можно добавить сбор и вывод разной статистики
    return {
        "active_requests": MAX_CONCURRENT_REQUESTS - semaphore._value,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "vector_db_documents": product_classifier.vector_store.count_documents(),
        "cached_results": len(product_classifier.cache)
    }


def start_api():
    """Запускает API сервер."""
    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    start_api()