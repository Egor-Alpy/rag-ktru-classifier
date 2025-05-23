import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # LLM Configuration (Local only)
    llm_provider: str = "ollama"  # Только локальный провайдер
    llm_model: str = os.getenv("LLM_MODEL", "mistral")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Embedding Configuration (Local model)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    embedding_batch_size: int = 32

    # Vector Store Configuration
    vector_store_type: str = os.getenv("VECTOR_STORE_TYPE", "qdrant")
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "ktru_codes")

    # API Configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_workers: int = int(os.getenv("API_WORKERS", "4"))

    # Redis Configuration
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_ttl: int = int(os.getenv("REDIS_TTL", "3600"))
    use_cache: bool = os.getenv("USE_CACHE", "false").lower() == "true"

    # Paths
    project_root: Path = Path(__file__).parent
    data_dir: Path = project_root / "data"
    log_dir: Path = project_root / "logs"

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "ktru_rag.log")

    # Classification thresholds
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.95"))
    max_candidates: int = int(os.getenv("MAX_CANDIDATES", "10"))

    class Config:
        env_file = ".env"


settings = Settings()