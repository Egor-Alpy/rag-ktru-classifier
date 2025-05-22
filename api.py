from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import time
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from classifier import classify_sku
from config import API_HOST, API_PORT

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Создание приложения FastAPI
app = FastAPI(title="KTRU Classification API")

# Добавление поддержки CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить запросы от всех источников (в производстве укажите конкретные домены)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Модели Pydantic
class Attribute(BaseModel):
    attr_name: str
    attr_value: str


class PriceItem(BaseModel):
    qnt: int
    discount: float
    price: float


class SupplierOffer(BaseModel):
    price: List[PriceItem]
    stock: str
    delivery_time: str
    package_info: str
    purchase_url: str


class Supplier(BaseModel):
    dealer_id: str
    supplier_name: str
    supplier_tel: str
    supplier_address: str
    supplier_description: str
    supplier_offers: List[SupplierOffer]


class SKUItem(BaseModel):
    title: str
    description: Optional[str] = None
    article: Optional[str] = None
    brand: Optional[str] = None
    country_of_origin: Optional[str] = None
    warranty_months: Optional[str] = None
    category: Optional[str] = None
    created_at: Optional[str] = None
    attributes: Optional[List[Attribute]] = None
    suppliers: Optional[List[Supplier]] = None


class ClassificationResponse(BaseModel):
    ktru_code: str
    ktru_title: Optional[str] = None  # Добавлено поле для названия КТРУ
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_time: float


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware для логирования запросов и измерения времени обработки"""
    start_time = time.time()

    # Логируем входящий запрос
    logger.info(f"Входящий запрос: {request.method} {request.url}")

    # Обрабатываем запрос
    response = await call_next(request)

    # Вычисляем время обработки
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    # Логируем ответ
    logger.info(f"Запрос обработан за {process_time:.4f} сек., статус: {response.status_code}")

    return response


@app.post("/classify", response_model=ClassificationResponse)
async def classify_item(sku: SKUItem):
    """Endpoint для классификации товара по КТРУ"""
    start_time = time.time()

    try:
        logger.info(f"Запрос на классификацию товара: {sku.title}")

        # Вызываем функцию классификации
        result = classify_sku(sku.dict())

        # Вычисляем время обработки
        processing_time = time.time() - start_time

        # Обрабатываем результат классификации
        if isinstance(result, dict):
            # Новый формат возврата с кодом и названием
            ktru_code = result.get('ktru_code', 'код не найден')
            ktru_title = result.get('ktru_title', None)
            confidence = result.get('confidence', 1.0 if ktru_code != 'код не найден' else 0.0)
        elif isinstance(result, str):
            # Старый формат (обратная совместимость)
            ktru_code = result
            ktru_title = None
            confidence = 1.0 if result != "код не найден" else 0.0
        else:
            # Неожиданный формат
            ktru_code = "код не найден"
            ktru_title = None
            confidence = 0.0

        # Формируем ответ
        return ClassificationResponse(
            ktru_code=ktru_code,
            ktru_title=ktru_title,
            confidence=confidence,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Endpoint для проверки работоспособности API"""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/status")
async def system_status():
    """Endpoint для проверки состояния всей системы"""
    from qdrant_client import QdrantClient
    from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION

    status = {
        "api": "healthy",
        "qdrant": "unknown",
        "collections": {},
        "models": "unknown",
        "timestamp": time.time()
    }

    try:
        # Проверка Qdrant
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Получаем информацию о коллекциях
        collections = qdrant_client.get_collections()
        status["qdrant"] = "healthy"

        for collection in collections.collections:
            collection_info = qdrant_client.get_collection(collection.name)
            collection_stats = qdrant_client.count(collection.name)

            status["collections"][collection.name] = {
                "vectors_count": collection_stats.count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance.name,
                "status": collection_info.status.name
            }

        # Проверяем, есть ли наша основная коллекция
        if QDRANT_COLLECTION in status["collections"]:
            status["ktru_loaded"] = status["collections"][QDRANT_COLLECTION]["vectors_count"] > 0
        else:
            status["ktru_loaded"] = False

    except Exception as e:
        status["qdrant"] = f"error: {str(e)}"
        logger.error(f"Ошибка при проверке состояния Qdrant: {e}")

    try:
        # Проверяем модели
        from embedding import embedding_model
        if embedding_model and embedding_model.model:
            status["models"] = "loaded"
        else:
            status["models"] = "not_loaded"
    except Exception as e:
        status["models"] = f"error: {str(e)}"

    return status


@app.get("/collections")
async def get_collections_info():
    """Подробная информация о коллекциях"""
    from qdrant_client import QdrantClient
    from config import QDRANT_HOST, QDRANT_PORT

    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = qdrant_client.get_collections()

        detailed_info = {}

        for collection in collections.collections:
            collection_info = qdrant_client.get_collection(collection.name)
            collection_stats = qdrant_client.count(collection.name)

            # Получаем примеры записей
            try:
                sample_points = qdrant_client.scroll(
                    collection_name=collection.name,
                    limit=3,
                    with_payload=True,
                    with_vectors=False
                )
                samples = [point.payload for point in sample_points[0]] if sample_points[0] else []
            except:
                samples = []

            detailed_info[collection.name] = {
                "vectors_count": collection_stats.count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance.name,
                "status": collection_info.status.name,
                "optimizer_status": collection_info.optimizer_status,
                "samples": samples
            }

        return {"collections": detailed_info, "total_collections": len(detailed_info)}

    except Exception as e:
        logger.error(f"Ошибка при получении информации о коллекциях: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения данных: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Глобальный обработчик исключений"""
    logger.error(f"Необработанное исключение: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Внутренняя ошибка сервера", "detail": str(exc)}
    )


if __name__ == "__main__":
    try:
        logger.info(f"Запуск API-сервиса на {API_HOST}:{API_PORT}")
        uvicorn.run(app, host=API_HOST, port=API_PORT)
    except Exception as e:
        logger.error(f"Ошибка при запуске API-сервиса: {e}")