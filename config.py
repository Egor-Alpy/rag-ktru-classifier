import os
from dotenv import load_dotenv

# Загрузка переменных окружения из .env
load_dotenv()

# Настройки Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "cointegrated/rubert-tiny2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "cointegrated/rubert-tiny2-sentence")

# Настройки MongoDB
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "ktru_db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "ktru_items")
KTRU_SYNC_INTERVAL = int(os.getenv("KTRU_SYNC_INTERVAL", 3600))  # в секундах

# Настройки Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "ktru_vectors"
VECTOR_DIM = 384  # Размерность вектора для rubert-tiny2-sentence

# Настройки API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Настройки обработки
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", 100))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.95))

# Настройки индексации
CHUNK_SIZE = 512  # Размер текстового чанка
CHUNK_OVERLAP = 128  # Перекрытие чанков
INDEX_REFRESH_INTERVAL = int(os.getenv("INDEX_REFRESH_INTERVAL", 86400))  # в секундах (24 часа)

# Настройки логирования
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")