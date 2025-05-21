from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения"""
    # Qdrant
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "ktru_vectors"

    # Сервисы
    embedding_service_url: str = "http://embeddings:8080"
    llm_service_url: str = "http://llm:8081"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # RAG
    search_top_k: int = 5
    similarity_threshold: float = 0.6

    class Config:
        env_file = ".env"