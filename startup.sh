#!/bin/bash

# Активация виртуального окружения, если необходимо
# source venv/bin/activate

# Установка переменных окружения из .env файла, если файл существует
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Проверка наличия токена Hugging Face
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set, some models may not be available or will have limited access"
fi

# Проверка подключения к MongoDB
if [ -z "$MONGO_URI" ]; then
    echo "Error: MONGO_URI not set, aborting startup"
    exit 1
fi

# Запуск основного скрипта
echo "Starting KTRU Product Classification System..."
python main.py