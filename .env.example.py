# Hugging Face
HF_TOKEN=your_hugging_face_token

# MongoDB
MONGO_URI=mongodb://username:password@mongodb.example.com:27017/
MONGO_DB=ktru_db
MONGO_COLLECTION=ktru_items
KTRU_SYNC_INTERVAL=3600

# Qdrant
QDRANT_URL=http://localhost:6333

# Models
MODEL_NAME=cointegrated/rubert-tiny2
EMBEDDING_MODEL=cointegrated/rubert-tiny2-sentence

# API
API_HOST=0.0.0.0
API_PORT=8000

# Processing
BATCH_SIZE=32
MAX_CONCURRENT_REQUESTS=100
CONFIDENCE_THRESHOLD=0.95

# Indexing
INDEX_REFRESH_INTERVAL=86400

# Logging
LOG_LEVEL=INFO