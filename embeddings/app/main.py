from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger

from app.model_loader import EmbeddingModelLoader
from app.config import Settings

# Загрузка настроек
settings = Settings()

# Инициализация загрузчика модели
embedding_loader = EmbeddingModelLoader(
    model_id=settings.embedding_model_id,
    device=settings.embedding_device
)

app = FastAPI(
    title="Embedding Service",
    description="Внутренний сервис для создания эмбеддингов",
    version="1.0.0"
)

class EmbeddingRequest(BaseModel):
    text: str

class BatchEmbeddingRequest(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    embedding: List[float]

class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

@app.on_event("startup")
async def startup_event():
    """Загружает модель при запуске приложения"""
    logger.info(f"Загрузка embedding модели {settings.embedding_model_id}...")
    await embedding_loader.load_model()
    logger.info("Модель успешно загружена")

@app.post("/embed", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    """Создает векторное представление текста"""
    try:
        embedding = await embedding_loader.get_embedding(request.text)
        return EmbeddingResponse(embedding=embedding)
    except Exception as e:
        logger.error(f"Ошибка при создании эмбеддинга: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed_batch", response_model=BatchEmbeddingResponse)
async def create_batch_embedding(request: BatchEmbeddingRequest):
    """Создает векторные представления для пакета текстов"""
    try:
        embeddings = await embedding_loader.get_embeddings_batch(request.texts)
        return BatchEmbeddingResponse(embeddings=embeddings)
    except Exception as e:
        logger.error(f"Ошибка при создании пакета эмбеддингов: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy" if embedding_loader.is_model_loaded() else "loading",
        "model_id": settings.embedding_model_id
    }