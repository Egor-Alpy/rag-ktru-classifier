import os
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла, если он существует
load_dotenv()

# Определяем базовую директорию проекта
current_dir = os.getcwd()
if current_dir.endswith('rag-ktru-classifier'):
    BASE_DIR = current_dir
elif os.path.exists('/workspace/rag-ktru-classifier'):
    BASE_DIR = "/workspace/rag-ktru-classifier"
else:
    BASE_DIR = current_dir

print(f"Используется базовая директория: {BASE_DIR}")

# Пути
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
QDRANT_STORAGE = os.path.join(BASE_DIR, "qdrant_storage")

# Создание директорий, если они не существуют
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

# Настройки JSON fallback
KTRU_JSON_PATH = os.getenv("KTRU_JSON_PATH", os.path.join(DATA_DIR, "ktru_data.json"))
ENABLE_JSON_FALLBACK = os.getenv("ENABLE_JSON_FALLBACK", "true").lower() == "true"

# Настройки моделей - ОПТИМИЗИРОВАНО для максимальной точности
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "ai-forever/sbert_large_nlu_ru")  # Лучшая модель для русского языка
LLM_BASE_MODEL = os.getenv("LLM_BASE_MODEL", "Open-Orca/Mistral-7B-OpenOrca")
LLM_ADAPTER_MODEL = os.getenv("LLM_ADAPTER_MODEL", "IlyaGusev/saiga_mistral_7b_lora")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))  # Размерность для sbert_large_nlu_ru
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))     # Уменьшен для большей модели

# Настройки API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Параметры генерации - МАКСИМАЛЬНАЯ ТОЧНОСТЬ
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))    # Очень низкая для детерминированности
TOP_P = float(os.getenv("TOP_P", "0.9"))               # Строгий отбор токенов
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.15"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "50"))  # Достаточно для кода КТРУ

# Параметры поиска - УВЕЛИЧЕНЫ для точности
TOP_K = int(os.getenv("TOP_K", "100"))  # Много кандидатов для тщательного анализа

# Пороги точности - ПОВЫШЕНЫ для требования 95%+
SIMILARITY_THRESHOLD_HIGH = float(os.getenv("SIMILARITY_THRESHOLD_HIGH", "0.95"))   # Очень высокая схожесть
SIMILARITY_THRESHOLD_MEDIUM = float(os.getenv("SIMILARITY_THRESHOLD_MEDIUM", "0.90")) # Высокая схожесть
SIMILARITY_THRESHOLD_LOW = float(os.getenv("SIMILARITY_THRESHOLD_LOW", "0.85"))     # Минимальный порог

# Настройки классификации
CLASSIFICATION_CONFIDENCE_THRESHOLD = float(os.getenv("CLASSIFICATION_CONFIDENCE_THRESHOLD", "0.95"))  # 95% уверенность
ENABLE_ATTRIBUTE_MATCHING = os.getenv("ENABLE_ATTRIBUTE_MATCHING", "true").lower() == "true"
ATTRIBUTE_WEIGHT = float(os.getenv("ATTRIBUTE_WEIGHT", "0.4"))  # Увеличен вес атрибутов

