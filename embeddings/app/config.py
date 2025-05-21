from typing import Optional
from pydantic-settings import BaseSettings


class Settings(BaseSettings):
    """Настройки для сервиса эмбеддингов"""
    embedding_model_id: str = "intfloat/multilingual-e5-large"
    embedding_device: str = "cuda"  # cuda или cpu

    class Config:
        env_file = ".env"