import os
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла, если он существует
load_dotenv()

# Пути
BASE_DIR = "/workspace"
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

# Настройки моделей
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "cointegrated/rubert-tiny2")
LLM_BASE_MODEL = os.getenv("LLM_BASE_MODEL", "Open-Orca/Mistral-7B-OpenOrca")
LLM_ADAPTER_MODEL = os.getenv("LLM_ADAPTER_MODEL", "IlyaGusev/saiga_mistral_7b_lora")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "312"))  # Размерность векторов rubert-tiny2
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))     # Размер пакета для обработки данных

# Настройки API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Параметры генерации
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.15"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "100"))
TOP_K = int(os.getenv("TOP_K", "5"))  # Количество похожих КТРУ для поиска