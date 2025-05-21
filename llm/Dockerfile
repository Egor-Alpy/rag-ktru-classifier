# llm/Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Установка дополнительных пакетов
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копирование кода
COPY app/ ./app/

# Запуск LLM сервиса
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8081"]