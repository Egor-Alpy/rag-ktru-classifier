#!/bin/bash

# Получаем текущую директорию
CURRENT_DIR=$(pwd)
PROJECT_DIR="/workspace/rag-ktru-classifier"

# Переходим в директорию проекта, если не в ней
if [ "$CURRENT_DIR" != "$PROJECT_DIR" ]; then
    echo "🔄 Переход в директорию проекта: $PROJECT_DIR"
    cd "$PROJECT_DIR" || {
        echo "❌ Ошибка: Не удалось перейти в $PROJECT_DIR"
        exit 1
    }
fi

# Создание необходимых директорий
mkdir -p logs qdrant_storage models data

echo "🚀 Запуск сервисов классификации КТРУ из $PROJECT_DIR..."

# Проверка наличия Qdrant в проекте
if [ ! -f "./qdrant" ]; then
    echo "📥 Qdrant не найден. Загружаем..."
    curl -L https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-unknown-linux-gnu.tar.gz -o qdrant.tar.gz
    tar -xzf qdrant.tar.gz
    rm qdrant.tar.gz
    chmod +x qdrant
fi

# Добавить проверку наличия curl
if ! command -v curl &> /dev/null; then
    echo "📦 curl не установлен, устанавливаем..."
    apt-get update && apt-get install -y curl
fi

# Проверяем, что Python зависимости установлены
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Python зависимости не установлены. Устанавливаем..."
    pip install -r requirements.txt
fi

# Функция для проверки состояния Qdrant
check_qdrant_status() {
    local max_attempts=30
    local attempt=1

    echo "⏳ Ожидание запуска Qdrant..."

    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
            echo "✅ Qdrant успешно запущен!"

            # Получаем информацию о коллекциях
            echo "📊 Проверка состояния векторной БД..."
            python3 -c "
import requests
import json

try:
    response = requests.get('http://localhost:6333/collections')
    if response.status_code == 200:
        data = response.json()
        collections = data.get('result', {}).get('collections', [])

        if collections:
            print(f'📚 Найдено коллекций: {len(collections)}')

            for collection in collections:
                name = collection.get('name', 'unknown')
                try:
                    # Получаем статистику коллекции
                    count_response = requests.get(f'http://localhost:6333/collections/{name}/points/count')
                    if count_response.status_code == 200:
                        count_data = count_response.json()
                        count = count_data.get('result', {}).get('count', 0)
                        print(f'   - {name}: {count:,} записей')
                    else:
                        print(f'   - {name}: статистика недоступна')
                except:
                    print(f'   - {name}: ошибка получения статистики')
        else:
            print('📭 Коллекции не найдены')
    else:
        print('❌ Ошибка получения списка коллекций')
except Exception as e:
    print(f'❌ Ошибка при проверке коллекций: {e}')
" 2>/dev/null || echo "⚠️  Не удалось получить детальную информацию о коллекциях"
            return 0
        fi

        echo "Попытка $attempt/$max_attempts..."
        sleep 2
        ((attempt++))
    done

    echo "❌ Ошибка: Qdrant не запустился в течение 60 секунд."
    return 1
}

# Функция для проверки API
check_api_status() {
    local max_attempts=10
    local attempt=1

    echo "⏳ Проверка API..."

    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ API успешно запущен!"

            # Получаем статус системы
            echo "📊 Проверка состояния системы..."
            python3 -c "
import requests
import json

try:
    response = requests.get('http://localhost:8000/status', timeout=10)
    if response.status_code == 200:
        data = response.json()

        print(f'🔧 Статус компонентов:')
        print(f'   - API: {data.get(\"api\", \"unknown\")}')
        print(f'   - Qdrant: {data.get(\"qdrant\", \"unknown\")}')
        print(f'   - Модели: {data.get(\"models\", \"unknown\")}')
        print(f'   - КТРУ загружено: {data.get(\"ktru_loaded\", False)}')

        collections = data.get('collections', {})
        if collections:
            print(f'📚 Коллекции:')
            for name, info in collections.items():
                count = info.get('vectors_count', 0)
                size = info.get('vector_size', 0)
                print(f'   - {name}: {count:,} векторов (размерность {size})')
        else:
            print('📭 Коллекции не найдены в статусе')
    else:
        print(f'⚠️  Статус API недоступен: {response.status_code}')
except Exception as e:
    print(f'⚠️  Ошибка при получении статуса: {e}')
" 2>/dev/null || echo "⚠️  Не удалось получить детальную информацию о статусе"

            return 0
        fi

        echo "Попытка $attempt/$max_attempts..."
        sleep 3
        ((attempt++))
    done

    echo "❌ API не отвечает"
    return 1
}

# Проверка доступности Qdrant
QDRANT_RUNNING=$(curl -s http://localhost:6333/collections > /dev/null && echo "yes" || echo "no")

if [ "$QDRANT_RUNNING" = "no" ]; then
    echo "🔄 Запуск Qdrant..."

    # Проверяем наличие config.yaml, если нет - создаем базовый
    if [ ! -f "./config.yaml" ]; then
        echo "📝 Создание базовой конфигурации Qdrant..."
        cat > ./config.yaml << EOL
storage:
  storage_path: $PROJECT_DIR/qdrant_storage
service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334
log_level: INFO
EOL
    fi

    nohup ./qdrant --config-path ./config.yaml > ./logs/qdrant.log 2>&1 &
    QDRANT_PID=$!

    # Проверка запуска Qdrant с детальной информацией
    if ! check_qdrant_status; then
        echo "❌ Критическая ошибка: Qdrant не запустился"
        echo "📋 Логи Qdrant:"
        tail -20 ./logs/qdrant.log
        exit 1
    fi
else
    echo "✅ Qdrant уже запущен."
    check_qdrant_status
fi

# Создаем .env файл, если нужно
if [ ! -f "./.env" ]; then
    echo "📝 Создание .env файла..."
    cat > ./.env << EOL
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=ktru_codes
MONGO_EXTERNAL_URI=${MONGO_EXTERNAL_URI:-mongodb://external_mongodb_server:27017/}
MONGO_LOCAL_URI=mongodb://localhost:27017/
MONGO_DB_NAME=${MONGO_DB_NAME:-ktru_database}
MONGO_COLLECTION=${MONGO_COLLECTION:-ktru_collection}
API_HOST=0.0.0.0
API_PORT=8000
EMBEDDING_MODEL=cointegrated/rubert-tiny2
LLM_BASE_MODEL=Open-Orca/Mistral-7B-OpenOrca
LLM_ADAPTER_MODEL=IlyaGusev/saiga_mistral_7b_lora
VECTOR_SIZE=312
BATCH_SIZE=32
TEMPERATURE=0.1
TOP_P=0.95
REPETITION_PENALTY=1.15
MAX_NEW_TOKENS=100
TOP_K=5
EOL
fi

# Запуск синхронизации с MongoDB в фоновом режиме
echo "🔄 Попытка запуска синхронизации с MongoDB..."
nohup python ./mongodb_sync.py > ./logs/mongodb_sync.log 2>&1 &
SYNC_PID=$!
echo "✅ Процесс синхронизации запущен с PID: $SYNC_PID"

# Запуск API сервиса
echo "🔄 Запуск API-сервиса..."
nohup python ./api.py > ./logs/api.log 2>&1 &
API_PID=$!
echo "✅ API-сервис запущен с PID: $API_PID"

# Ждем запуска API и проверяем статус
sleep 5

if ! check_api_status; then
    echo "❌ Критическая ошибка: API не запустился"
    echo "📋 Логи API:"
    tail -20 ./logs/api.log
    exit 1
fi

# Обработка сигналов завершения
trap 'echo "🛑 Завершение работы..."; kill $SYNC_PID $API_PID 2>/dev/null; exit' SIGINT SIGTERM

echo ""
echo "🎉 Все сервисы успешно запущены!"
echo "=================================================="
echo "📍 Endpoints:"
echo "   🏥 Health check: http://localhost:8000/health"
echo "   📊 System status: http://localhost:8000/status"
echo "   📚 Collections:   http://localhost:8000/collections"
echo "   🤖 Classify:      http://localhost:8000/classify"
echo "   🔍 Qdrant:        http://localhost:6333"
echo ""
echo "📁 Логи в директории: $PROJECT_DIR/logs/"
echo "🔧 Для проверки системы: cd $PROJECT_DIR && python system_status.py"
echo ""
echo "Для завершения нажмите Ctrl+C"

# Бесконечный цикл для поддержания контейнера активным
while true; do
    # Проверка, что процессы все еще работают
    if ! ps -p $SYNC_PID > /dev/null 2>&1; then
        echo "$(date): ⚠️  Процесс синхронизации остановлен. Перезапуск..."
        nohup python ./mongodb_sync.py > ./logs/mongodb_sync.log 2>&1 &
        SYNC_PID=$!
    fi

    if ! ps -p $API_PID > /dev/null 2>&1; then
        echo "$(date): ⚠️  API-сервис остановлен. Перезапуск..."
        nohup python ./api.py > ./logs/api.log 2>&1 &
        API_PID=$!

        # Проверяем что API снова заработал
        sleep 5
        check_api_status
    fi

    sleep 60
done