from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Настройки LLM сервиса"""
    model_id: str = "Saiga/saiga_mistral_7b"
    model_revision: str = "main"
    model_quantization: str = "4bit"  # Варианты: "none", "4bit", "8bit"
    device: str = "cuda"

    class Config:
        env_file = ".env"