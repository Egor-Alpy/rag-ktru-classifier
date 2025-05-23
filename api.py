"""
FastAPI сервис для KTRU классификатора
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import time
import logging
import uvicorn

from config import API_HOST, API_PORT
from classifier import classify_product
from vector_db import vector_db
from embeddings import embedding_manager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создание приложения
app = FastAPI(
    title="KTRU Classification API",
    description="API для классификации товаров по КТРУ с использованием RAG",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Модели данных
class Attribute(BaseModel):
    attr_name: str
    attr_value: str


class ProductRequest(BaseModel):
    title: str = Field(..., description="Название товара")
    description: Optional[str] = Field(None, description="Описание товара")
    category: Optional[str] = Field(None, description="Категория товара")
    brand: Optional[str] = Field(None, description="Бренд")
    attributes: Optional[List[Attribute]] = Field(None, description="Атрибуты товара")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Ноутбук ASUS X515EA",
                "description": "Портативный компьютер для офисной работы",
                "category": "Компьютеры",
                "brand": "ASUS",
                "attributes": [
                    {"attr_name": "Процессор", "attr_value": "Intel Core i5"},
                    {"attr_name": "ОЗУ", "attr_value": "8 ГБ"}
                ]
            }
        }


class ClassificationResponse(BaseModel):
    ktru_code: str = Field(..., description="Код КТРУ или 'код не найден'")
    ktru_title: Optional[str] = Field(None, description="Название КТРУ")
    confidence: float = Field(..., description="Уверенность классификации (0-1)")
    processing_time: float = Field(..., description="Время обработки в секундах")


class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, str]


class StatusResponse(BaseModel):
    api: str
    vector_db: str
    embeddings: str
    classifier: str
    statistics: Optional[Dict] = None


# Endpoints
@app.get("/", tags=["General"])
async def root():
    """Корневой endpoint"""
    return {
        "message": "KTRU Classification API",
        "version": "2.0.0",
        "endpoints": {
            "classify": "/classify",
            "health": "/health",
            "status": "/status",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Проверка здоровья сервиса"""
    components = {}

    # Проверка векторной БД
    try:
        if vector_db.health_check():
            components["vector_db"] = "healthy"
        else:
            components["vector_db"] = "unhealthy"
    except:
        components["vector_db"] = "error"

    # Проверка эмбеддингов
    try:
        info = embedding_manager.get_model_info()
        components["embeddings"] = f"healthy (model: {info['model_name']})"
    except:
        components["embeddings"] = "error"

    # Проверка классификатора
    try:
        from classifier import classifier
        if classifier:
            components["classifier"] = "healthy"
            if classifier.llm:
                components["llm"] = "loaded"
            else:
                components["llm"] = "not_loaded"
        else:
            components["classifier"] = "error"
    except:
        components["classifier"] = "error"

    # Определяем общий статус
    all_healthy = all(
        "healthy" in status or status == "loaded"
        for status in components.values()
    )

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="2.0.0",
        components=components
    )


@app.get("/status", response_model=StatusResponse, tags=["Monitoring"])
async def system_status():
    """Детальный статус системы"""
    status = StatusResponse(
        api="healthy",
        vector_db="unknown",
        embeddings="unknown",
        classifier="unknown"
    )

    # Статус векторной БД
    try:
        stats = vector_db.get_statistics()
        status.vector_db = stats.get('status', 'unknown')
        status.statistics = {
            'total_vectors': stats.get('total_vectors', 0),
            'categories': len(stats.get('categories', {}))
        }
    except Exception as e:
        status.vector_db = f"error: {str(e)}"

    # Статус эмбеддингов
    try:
        info = embedding_manager.get_model_info()
        status.embeddings = f"loaded ({info['model_name']})"
    except Exception as e:
        status.embeddings = f"error: {str(e)}"

    # Статус классификатора
    try:
        from classifier import classifier
        if classifier:
            status.classifier = "loaded"
            if classifier.llm:
                status.classifier += " (with LLM)"
            else:
                status.classifier += " (without LLM)"
        else:
            status.classifier = "not_initialized"
    except Exception as e:
        status.classifier = f"error: {str(e)}"

    return status


@app.post("/classify", response_model=ClassificationResponse, tags=["Classification"])
async def classify_item(product: ProductRequest):
    """
    Классификация товара по КТРУ

    Принимает данные о товаре и возвращает код КТРУ с уверенностью.
    """
    start_time = time.time()

    try:
        logger.info(f"Запрос на классификацию: {product.title}")

        # Преобразуем в словарь для классификатора
        product_data = product.model_dump()

        # Классифицируем
        result = classify_product(product_data)

        # Время обработки
        processing_time = time.time() - start_time

        logger.info(
            f"Результат: {result['ktru_code']} "
            f"(уверенность: {result['confidence']:.3f}, время: {processing_time:.2f}с)"
        )

        return ClassificationResponse(
            ktru_code=result['ktru_code'],
            ktru_title=result.get('ktru_title'),
            confidence=result['confidence'],
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Ошибка при классификации: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_classify", tags=["Classification"])
async def batch_classify(products: List[ProductRequest]):
    """
    Пакетная классификация товаров

    Принимает список товаров и возвращает результаты классификации.
    """
    start_time = time.time()
    results = []

    try:
        logger.info(f"Пакетная классификация {len(products)} товаров")

        for product in products:
            try:
                product_data = product.model_dump()
                result = classify_product(product_data)

                results.append({
                    "input_title": product.title,
                    "ktru_code": result['ktru_code'],
                    "ktru_title": result.get('ktru_title'),
                    "confidence": result['confidence'],
                    "status": "success"
                })

            except Exception as e:
                results.append({
                    "input_title": product.title,
                    "ktru_code": "ошибка",
                    "ktru_title": None,
                    "confidence": 0.0,
                    "status": "error",
                    "error": str(e)
                })

        total_time = time.time() - start_time

        return {
            "total_items": len(products),
            "successful": sum(1 for r in results if r['status'] == 'success'),
            "failed": sum(1 for r in results if r['status'] == 'error'),
            "total_time": total_time,
            "avg_time_per_item": total_time / len(products) if products else 0,
            "results": results
        }

    except Exception as e:
        logger.error(f"Ошибка при пакетной классификации: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search", tags=["Search"])
async def search_ktru(
        query: str,
        limit: int = 10
):
    """
    Поиск KTRU кодов по запросу

    Выполняет векторный поиск по базе KTRU.
    """
    try:
        results = vector_db.search(query, top_k=limit)

        return {
            "query": query,
            "count": len(results),
            "results": [
                {
                    "ktru_code": r['payload'].get('ktru_code'),
                    "title": r['payload'].get('title'),
                    "score": r['score']
                }
                for r in results
            ]
        }

    except Exception as e:
        logger.error(f"Ошибка при поиске: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Запуск сервера
if __name__ == "__main__":
    logger.info(f"Запуск API сервера на {API_HOST}:{API_PORT}")
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )
