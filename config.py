import os
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Базовые пути
current_dir = os.getcwd()
if current_dir.endswith('rag-ktru-classifier'):
    BASE_DIR = current_dir
elif os.path.exists('/workspace/rag-ktru-classifier'):
    BASE_DIR = "/workspace/rag-ktru-classifier"
else:
    BASE_DIR = current_dir

print(f"Используется базовая директория: {BASE_DIR}")

# Директории
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
QDRANT_STORAGE = os.path.join(BASE_DIR, "qdrant_storage")

# Создание директорий
for dir_path in [MODELS_DIR, DATA_DIR, LOGS_DIR, QDRANT_STORAGE]:
    os.makedirs(dir_path, exist_ok=True)

# Настройки Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "ktru_codes")

# Настройки MongoDB
MONGO_EXTERNAL_URI = os.getenv("MONGO_EXTERNAL_URI", "mongodb://external_mongodb_server:27017/")
MONGO_LOCAL_URI = os.getenv("MONGO_LOCAL_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "TenderDB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "ktru")

MONGO_EXT_HOST = os.getenv("MONGO_EXT_HOST", "")
MONGO_EXT_PORT = os.getenv("MONGO_EXT_PORT", "")
MONGO_EXT_USERNAME = os.getenv("MONGO_EXT_USERNAME", "")
MONGO_EXT_PASS = os.getenv("MONGO_EXT_PASS", "")
MONGO_EXT_AUTHMEC = os.getenv("MONGO_EXT_AUTHMEC", "")
MONGO_EXT_AUTHSOURCE = os.getenv("MONGO_EXT_AUTHSOURCE", "")

# Настройки JSON
KTRU_JSON_PATH = os.getenv("KTRU_JSON_PATH", os.path.join(DATA_DIR, "ktru_data.json"))
ENABLE_JSON_FALLBACK = os.getenv("ENABLE_JSON_FALLBACK", "true").lower() == "true"

# ОПТИМИЗИРОВАННЫЕ НАСТРОЙКИ МОДЕЛЕЙ ДЛЯ ТОЧНОСТИ

# Модель эмбеддингов - используем лучшую для русского языка
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "ai-forever/sbert_large_nlu_ru")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))  # Размерность для sbert_large_nlu_ru
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))  # Оптимальный размер батча

# LLM модель для финальной классификации
LLM_BASE_MODEL = os.getenv("LLM_BASE_MODEL", "Open-Orca/Mistral-7B-OpenOrca")
LLM_ADAPTER_MODEL = os.getenv("LLM_ADAPTER_MODEL", "IlyaGusev/saiga_mistral_7b_lora")

# Настройки API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ПАРАМЕТРЫ ГЕНЕРАЦИИ - ОПТИМИЗИРОВАНЫ ДЛЯ ТОЧНОСТИ
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))  # Очень низкая для детерминированности
TOP_P = float(os.getenv("TOP_P", "0.9"))  # Консервативный отбор токенов
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.15"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "50"))  # Достаточно для КТРУ кода

# ПАРАМЕТРЫ ПОИСКА - УВЕЛИЧЕНЫ ДЛЯ ЛУЧШЕГО ОТБОРА
TOP_K = int(os.getenv("TOP_K", "50"))  # Больше кандидатов для анализа

# НОВЫЕ ПАРАМЕТРЫ ДЛЯ УЛУЧШЕННОЙ КЛАССИФИКАЦИИ

# Веса для комбинирования методов поиска
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.7"))  # Вес поиска по ключевым словам
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.3"))  # Вес векторного поиска
TITLE_WEIGHT = float(os.getenv("TITLE_WEIGHT", "0.5"))  # Бонус за совпадение названий

# Пороги уверенности для разных типов совпадений
EXACT_MATCH_CONFIDENCE = float(os.getenv("EXACT_MATCH_CONFIDENCE", "0.98"))  # Точное совпадение
HIGH_MATCH_CONFIDENCE = float(os.getenv("HIGH_MATCH_CONFIDENCE", "0.95"))  # Высокое совпадение
MEDIUM_MATCH_CONFIDENCE = float(os.getenv("MEDIUM_MATCH_CONFIDENCE", "0.90"))  # Среднее совпадение
LOW_MATCH_CONFIDENCE = float(os.getenv("LOW_MATCH_CONFIDENCE", "0.85"))  # Низкое совпадение

# Пороги для различных метрик
KEYWORD_SCORE_THRESHOLD = float(os.getenv("KEYWORD_SCORE_THRESHOLD", "2.0"))  # Порог для ключевых слов
TITLE_SIMILARITY_THRESHOLD = float(os.getenv("TITLE_SIMILARITY_THRESHOLD", "0.7"))  # Порог схожести названий
VECTOR_SCORE_THRESHOLD = float(os.getenv("VECTOR_SCORE_THRESHOLD", "0.8"))  # Порог векторной схожести

# Настройки обработки текста
MIN_KEYWORD_LENGTH = int(os.getenv("MIN_KEYWORD_LENGTH", "3"))  # Минимальная длина ключевого слова
MAX_DESCRIPTION_LENGTH = int(os.getenv("MAX_DESCRIPTION_LENGTH", "200"))  # Максимальная длина описания

# Режим классификации
CLASSIFICATION_MODE = os.getenv("CLASSIFICATION_MODE", "hybrid")  # hybrid, keywords_first, vectors_first

# Настройки для категорий товаров
ENABLE_CATEGORY_MAPPING = os.getenv("ENABLE_CATEGORY_MAPPING", "true").lower() == "true"
CATEGORY_BOOST_FACTOR = float(os.getenv("CATEGORY_BOOST_FACTOR", "2.0"))  # Усиление для известных категорий

# Логирование и отладка
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_TOP_CANDIDATES = int(os.getenv("LOG_TOP_CANDIDATES", "5"))  # Количество кандидатов для логирования
ENABLE_DEBUG_MODE = os.getenv("ENABLE_DEBUG_MODE", "false").lower() == "true"

# Кэширование
ENABLE_SEARCH_CACHE = os.getenv("ENABLE_SEARCH_CACHE", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # Время жизни кэша в секундах

# Производительность
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))  # Количество воркеров для параллельной обработки
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))  # Таймаут запроса в секундах


# Валидация конфигурации
def validate_config():
    """Проверка корректности конфигурации"""
    errors = []

    # Проверка весов
    if abs(KEYWORD_WEIGHT + VECTOR_WEIGHT - 1.0) > 0.01:
        errors.append("KEYWORD_WEIGHT + VECTOR_WEIGHT должны давать 1.0")

    # Проверка порогов
    if not (0 < LOW_MATCH_CONFIDENCE < MEDIUM_MATCH_CONFIDENCE < HIGH_MATCH_CONFIDENCE < EXACT_MATCH_CONFIDENCE <= 1.0):
        errors.append("Пороги уверенности должны быть упорядочены по возрастанию")

    # Проверка режима классификации
    if CLASSIFICATION_MODE not in ["hybrid", "keywords_first", "vectors_first"]:
        errors.append(f"Неизвестный режим классификации: {CLASSIFICATION_MODE}")

    if errors:
        print("⚠️ Ошибки конфигурации:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("Некорректная конфигурация")
    else:
        print("✅ Конфигурация валидна")


# Выполняем валидацию при импорте
validate_config()