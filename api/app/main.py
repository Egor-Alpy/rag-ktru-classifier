from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import Settings
from app.models.sku import SKUModel, ClassificationResult
from app.services.classifier import KTRUClassifier

# Загрузка конфигурации
settings = Settings()

# Инициализация сервисов
classifier = KTRUClassifier(
    qdrant_host=settings.qdrant_host,
    qdrant_port=settings.qdrant_port,
    qdrant_collection=settings.qdrant_collection,
    embedding_service_url=settings.embedding_service_url,
    llm_service_url=settings.llm_service_url
)

app = FastAPI(
    title="KTRU Classifier API",
    description="API для классификации товаров по кодам КТРУ",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/classify", response_model=ClassificationResult)
async def classify_sku(sku: SKUModel):
    """Классифицирует товар по кодам КТРУ"""
    try:
        result = await classifier.classify(sku)
        return result
    except Exception as e:
        logger.error(f"Ошибка при классификации товара: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Проверка состояния API"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
    