from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки для сервиса эмбеддингов"""
    embedding_model_id: str = "intfloat/multilingual-e5-large"
    embedding_device: str = "cuda"  # cuda или cpu

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # Allow extra fields to be provided without validation errors
    }