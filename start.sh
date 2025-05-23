#!/bin/bash

# RAG KTRU Classifier - Startup Script
# Оптимизирован для RunPod

set -e

echo "🚀 RAG KTRU Classifier - Starting System"
echo "========================================"

# Проверка окружения
if [ -f /workspace/rag-ktru-classifier ]; then
    cd /workspace/rag-ktru-classifier
else
    cd "$(dirname "$0")"
fi

# Создание необходимых директорий
echo "📁 Creating directories..."
mkdir -p data models logs qdrant_storage

# Проверка Python
echo "🐍 Checking Python..."
python --version

# Установка зависимостей если нужно
if ! python -c "import torch" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Проверка наличия Qdrant
QDRANT_PID=""
if ! curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo "🔄 Starting Qdrant..."

    # Загрузка Qdrant если нет
    if [ ! -f "./qdrant" ]; then
        echo "📥 Downloading Qdrant..."
        curl -L https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-unknown-linux-gnu.tar.gz -o qdrant.tar.gz
        tar -xzf qdrant.tar.gz
        rm qdrant.tar.gz
        chmod +x qdrant
    fi

    # Создание конфига Qdrant
    if [ ! -f "./config.yaml" ]; then
        cat > ./config.yaml << EOL
storage:
  storage_path: ./qdrant_storage
service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334
log_level: INFO
EOL
    fi

    # Запуск Qdrant
    nohup ./qdrant --config-path ./config.yaml > ./logs/qdrant.log 2>&1 &
    QDRANT_PID=$!
    echo "✅ Qdrant started (PID: $QDRANT_PID)"

    # Ждем запуска
    echo "⏳ Waiting for Qdrant..."
    sleep 5
else
    echo "✅ Qdrant already running"
fi

# Проверка данных KTRU
echo "📊 Checking KTRU data..."
if [ ! -f "./data/ktru_data.json" ]; then
    echo "⚠️  KTRU data not found!"
    echo "Creating sample data..."
    python load_data.py --create-sample
    echo ""
    echo "❗ Please add your KTRU data to ./data/ktru_data.json"
    echo "   Then run: python load_data.py"
else
    # Проверяем загружены ли данные
    VECTOR_COUNT=$(python -c "
from vector_db import vector_db
stats = vector_db.get_statistics()
print(stats.get('total_vectors', 0))
" 2>/dev/null || echo "0")

    if [ "$VECTOR_COUNT" -eq "0" ]; then
        echo "📥 Loading KTRU data..."
        python load_data.py --json-file ./data/ktru_data.json
    else
        echo "✅ KTRU data loaded ($VECTOR_COUNT vectors)"
    fi
fi

# Запуск API
echo "🌐 Starting API server..."
nohup python api.py > ./logs/api.log 2>&1 &
API_PID=$!
echo "✅ API started (PID: $API_PID)"

# Ждем запуска API
sleep 5

# Проверка статуса
echo ""
echo "🔍 Checking system status..."
curl -s http://localhost:8000/health | python -m json.tool || echo "⚠️ API not responding"

echo ""
echo "✅ System is ready!"
echo "========================================"
echo "📍 Endpoints:"
echo "   - API: http://localhost:8000"
echo "   - Docs: http://localhost:8000/docs"
echo "   - Qdrant: http://localhost:6333"
echo ""
echo "📋 Available commands:"
echo "   - Test system: python test_system.py"
echo "   - Load data: python load_data.py --json-file your_ktru.json"
echo "   - Check logs: tail -f logs/*.log"
echo ""
echo "Press Ctrl+C to stop all services"

# Функция для остановки сервисов
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    [ ! -z "$API_PID" ] && kill $API_PID 2>/dev/null
    [ ! -z "$QDRANT_PID" ] && kill $QDRANT_PID 2>/dev/null
    echo "Goodbye!"
    exit 0
}

# Обработка сигналов
trap cleanup SIGINT SIGTERM

# Бесконечный цикл
while true; do
    sleep 60

    # Проверка что API работает
    if ! kill -0 $API_PID 2>/dev/null; then
        echo "⚠️ API stopped, restarting..."
        nohup python api.py > ./logs/api.log 2>&1 &
        API_PID=$!
    fi
done