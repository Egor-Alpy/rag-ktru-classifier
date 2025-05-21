from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger

from app.model_loader import ModelLoader
from app.config import Settings

# Загрузка настроек
settings = Settings()

# Инициализация загрузчика модели
model_loader = ModelLoader(
    model_id=settings.model_id,
    model_revision=settings.model_revision,
    quantization=settings.model_quantization
)

app = FastAPI(
    title="LLM Service",
    description="Внутренний сервис для генерации текста с помощью LLM",
    version="1.0.0"
)


class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50


class GenerationResponse(BaseModel):
    text: str


@app.on_event("startup")
async def startup_event():
    """Загружает модель при запуске приложения"""
    logger.info(f"Загрузка LLM модели {settings.model_id}...")
    await model_loader.load_model()
    logger.info("Модель успешно загружена")


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Генерирует текст с помощью LLM на основе промпта"""
    try:
        text = await model_loader.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )

        return GenerationResponse(text=text)
    except Exception as e:
        logger.error(f"Ошибка при генерации текста: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy" if model_loader.is_model_loaded() else "loading",
        "model_id": settings.model_id
    }