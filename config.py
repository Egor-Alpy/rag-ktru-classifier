"""
Конфигурация для RAG KTRU классификатора
Оптимизирована для RunPod с 24GB VRAM
"""

import os
from pathlib import Path

# Базовые пути
BASE_DIR = Path("/workspace/rag-ktru-classifier")
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
QDRANT_STORAGE = BASE_DIR / "qdrant_storage"

# Создание директорий
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, QDRANT_STORAGE]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Настройки Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "ktru_vectors"

# Настройки данных
KTRU_JSON_PATH = DATA_DIR / "ktru_data.json"

# Настройки моделей для эмбеддингов
# Используем multilingual-e5-base для баланса качества и скорости
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
VECTOR_SIZE = 768  # Размерность для e5-base
BATCH_SIZE = 64  # Увеличиваем для GPU

# Настройки LLM - используем квантизированную версию для эффективности
LLM_MODEL = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"  # 4-bit квантизация
LLM_MAX_LENGTH = 2048
LLM_TEMPERATURE = 0.1  # Низкая для детерминированности
LLM_TOP_P = 0.95
LLM_TOP_K = 50

# Параметры поиска и классификации
SEARCH_TOP_K = 50  # Количество кандидатов из векторного поиска
RERANK_TOP_K = 10  # Количество кандидатов после ре-ранжирования
MIN_CONFIDENCE = 0.85  # Минимальный порог уверенности

# API настройки
API_HOST = "0.0.0.0"
API_PORT = 8000

# Оптимизация производительности
USE_GPU = True
DEVICE = "cuda" if USE_GPU else "cpu"
NUM_WORKERS = 4
USE_CACHE = True
CACHE_SIZE = 10000

# Логирование
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Веса для комбинирования методов
WEIGHTS = {
    "vector_similarity": 0.4,  # Векторное сходство
    "keyword_match": 0.3,  # Совпадение ключевых слов
    "fuzzy_match": 0.2,  # Нечеткое совпадение
    "category_match": 0.1  # Совпадение категорий
}

# Специальные правила для категорий
CATEGORY_MAPPINGS = {
    # Компьютерная техника
    "компьютер": ["26.20.11", "26.20.13", "26.20.14"],
    "ноутбук": ["26.20.11", "26.20.13"],
    "монитор": ["26.20.17"],
    "принтер": ["26.20.16", "28.23.22"],
    "клавиатура": ["26.20.16"],
    "мышь": ["26.20.16"],

    # Канцелярские товары
    "ручка": ["32.99.12", "32.99.13"],
    "карандаш": ["32.99.15"],
    "маркер": ["32.99.12"],
    "степлер": ["25.99.23"],
    "скрепки": ["25.93.18"],

    # Бумажная продукция
    "бумага": ["17.12.14", "17.23.12"],
    "картон": ["17.12.42"],
    "конверт": ["17.23.12"],

    # Мебель
    "стол": ["31.01.11", "31.09.11"],
    "стул": ["31.01.11", "31.09.12"],
    "кресло": ["31.01.12", "31.09.12"],
    "шкаф": ["31.01.13", "31.09.13"],
}

# Стоп-слова для фильтрации
STOP_WORDS = {
    'и', 'в', 'на', 'с', 'по', 'для', 'или', 'под', 'над', 'все',
    'при', 'без', 'до', 'из', 'к', 'от', 'об', 'о', 'у', 'за'
}