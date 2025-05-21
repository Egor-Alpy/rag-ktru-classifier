from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки LLM сервиса"""
    model_id: str = "mistralai/Mistral-7B-v0.1"  # Changed to public model
    model_revision: str = "main"
    model_quantization: str = "4bit"  # Варианты: "none", "4bit", "8bit"
    device: str = "cuda"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # Allow extra fields to be provided without validation errors
        "protected_namespaces": ("settings_",)  # Change protected namespace from 'model_' to 'settings_'
    }