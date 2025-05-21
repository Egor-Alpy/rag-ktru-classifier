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

        # Формируем ответ
        if result == "код не найден":
            return ClassificationResponse(
                ktru_code="код не найден",
                confidence=0.0,
                processing_time=processing_time
            )
        else:
            return ClassificationResponse(
                ktru_code=result,
                confidence=1.0,
                processing_time=processing_time
            )

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Endpoint для проверки работоспособности API"""
    return {"status": "healthy", "version": "1.0.0"}


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