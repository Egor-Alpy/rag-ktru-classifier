from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import os

from app.embedding import EmbeddingModel
from app.database import VectorDatabase


# Определение моделей данных для API
class ProductAttribute(BaseModel):
    attr_name: str
    attr_value: str


class ProductSupplierOffer(BaseModel):
    price: List[Dict[str, Any]]
    stock: str
    delivery_time: str
    package_info: str
    purchase_url: str


class ProductSupplier(BaseModel):
    dealer_id: str
    supplier_name: str
    supplier_tel: str
    supplier_address: str
    supplier_description: str
    supplier_offers: List[ProductSupplierOffer]


class Product(BaseModel):
    id: Dict[str, str] = Field(..., alias="_id")
    title: str
    description: str
    article: str
    brand: str
    country_of_origin: str
    warranty_months: str
    category: str
    created_at: str
    attributes: List[ProductAttribute] = []
    suppliers: List[ProductSupplier] = []


class KTRUMatch(BaseModel):
    ktru_code: str
    title: str
    unit: Optional[str] = None
    confidence: float
    attributes: Optional[List[Dict[str, Any]]] = None


class MatchRequest(BaseModel):
    product: Product


class MatchResponse(BaseModel):
    found: bool
    matches: List[KTRUMatch] = []
    message: str


# Инициализация приложения FastAPI
app = FastAPI(
    title="KTRU Matching API",
    description="API для сопоставления товаров с кодами КТРУ",
    version="1.0.0"
)

# Глобальные переменные для моделей
embedding_model = None
vector_db = None


# Инициализация моделей при запуске приложения
@app.on_event("startup")
async def startup_event():
    global embedding_model, vector_db

    # Инициализация модели эмбеддингов
    model_name = os.getenv("EMBEDDING_MODEL", "ai-forever/ru-en-RoSBERTa")
    embedding_model = EmbeddingModel(model_name=model_name)

    # Инициализация векторной базы данных
    persist_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    vector_db = VectorDatabase(
        persist_directory=persist_dir,
        embedding_model_name=model_name
    )


@app.post("/match", response_model=MatchResponse)
async def match_product(request: MatchRequest):
    """Сопоставление товара с кодами КТРУ"""
    global embedding_model, vector_db

    # Проверка инициализации моделей
    if not embedding_model or not vector_db:
        raise HTTPException(
            status_code=500,
            detail="Модели не инициализированы. Пожалуйста, проверьте журналы сервера."
        )

    try:
        # Получаем эмбеддинг товара
        product_dict = request.product.dict()
        product_embedding = embedding_model.get_product_embedding(product_dict)

        # Подготавливаем фильтр по атрибутам (если необходимо)
        filter_metadata = None
        # Здесь можно добавить логику для создания фильтра на основе атрибутов товара

        # Ищем похожие коды КТРУ
        similar_items = vector_db.find_similar_ktru(
            product_embedding=product_embedding,
            filter_metadata=filter_metadata,
            top_k=5  # Возвращаем топ-5 совпадений
        )

        # Если ничего не найдено
        if not similar_items:
            return MatchResponse(
                found=False,
                matches=[],
                message="Подходящие коды КТРУ не найдены"
            )

        # Подготавливаем ответ
        matches = []
        for item in similar_items:
            # Преобразуем дистанцию в показатель уверенности (0-1)
            # В Chroma меньшая дистанция = большее сходство
            distance = item.get("distance", 1.0)
            confidence = 1.0 - min(distance, 1.0)

            # Добавляем совпадение в результаты
            matches.append(KTRUMatch(
                ktru_code=item["ktru_code"],
                title=item["metadata"].get("title", ""),
                unit=item["metadata"].get("unit", ""),
                confidence=confidence,
                attributes=None  # Здесь можно добавить извлечение атрибутов из метаданных
            ))

        return MatchResponse(
            found=True,
            matches=matches,
            message=f"Найдено {len(matches)} подходящих кодов КТРУ"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при сопоставлении товара: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Проверка работоспособности API"""
    global embedding_model, vector_db

    # Проверка инициализации моделей
    models_initialized = embedding_model is not None and vector_db is not None

    return {
        "status": "healthy" if models_initialized else "not_ready",
        "embedding_model": "initialized" if embedding_model else "not_initialized",
        "vector_db": "initialized" if vector_db else "not_initialized"
    }