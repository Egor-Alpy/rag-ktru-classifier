#!/bin/bash

# Создание лог-директории
mkdir -p /workspace/logs

echo "Запуск сервисов классификации КТРУ..."

# Проверка доступности Qdrant
QDRANT_RUNNING=$(curl -s http://localhost:6333/collections > /dev/null && echo "yes" || echo "no")

if [ "$QDRANT_RUNNING" = "no" ]; then
    echo "Запуск Qdrant..."
    cd /workspace
    ./qdrant --storage-path /workspace/qdrant_storage > /workspace/logs/qdrant.log 2>&1 &

    # Проверка запуска Qdrant
    echo "Ожидание запуска Qdrant..."
    COUNTER=0
    while [ $COUNTER -lt 30 ]; do
        sleep 2
        QDRANT_RUNNING=$(curl -s http://localhost:6333/collections > /dev/null && echo "yes" || echo "no")
        if [ "$QDRANT_RUNNING" = "yes" ]; then
            echo "Qdrant успешно запущен!"
            break
        fi
        COUNTER=$((COUNTER+1))
    done

    if [ "$QDRANT_RUNNING" = "no" ]; then
        echo "Ошибка: Qdrant не запустился в течение 60 секунд. Проверьте логи."
        exit 1
    fi
else
    echo "Qdrant уже запущен."
fi

# Создаем .env файл, если нужно
if [ ! -f "/workspace/.env" ]; then
    echo "Создание .env файла..."
    cat > /workspace/.env << EOL
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=ktru_codes
MONGO_EXTERNAL_URI=${MONGO_EXTERNAL_URI:-mongodb://external_mongodb_server:27017/}
MONGO_LOCAL_URI=mongodb://localhost:27017/
MONGO_DB_NAME=${MONGO_DB_NAME:-ktru_database}
MONGO_COLLECTION=${MONGO_COLLECTION:-ktru_collection}
API_HOST=0.0.0.0
API_PORT=8000
EOL
fi

# Запуск синхронизации с MongoDB в фоновом режиме
echo "Запуск синхронизации с MongoDB..."
python /workspace/mongodb_sync.py > /workspace/logs/mongodb_sync.log 2>&1 &
SYNC_PID=$!
echo "Процесс синхронизации запущен с PID: $SYNC_PID"

# Запуск API сервиса
echo "Запуск API-сервиса..."
python /workspace/api.py > /workspace/logs/api.log 2>&1 &
API_PID=$!
echo "API-сервис запущен с PID: $API_PID"

# Обработка сигналов завершения
trap 'kill $SYNC_PID $API_PID; exit' SIGINT SIGTERM

echo "Все сервисы запущены. Для мониторинга используйте логи в директории /workspace/logs/"

# Бесконечный цикл для поддержания контейнера активным
while true; do
    # Проверка, что процессы все еще работают
    if ! ps -p $SYNC_PID > /dev/null; then
        echo "Процесс синхронизации остановлен. Перезапуск..."
        python /workspace/mongodb_sync.py > /workspace/logs/mongodb_sync.log 2>&1 &
        SYNC_PID=$!
    fi

    if ! ps -p $API_PID > /dev/null; then
        echo "API-сервис остановлен. Перезапуск..."
        python /workspace/api.py > /workspace/logs/api.log 2>&1 &
        API_PID=$!
    fi

    sleep 60
done