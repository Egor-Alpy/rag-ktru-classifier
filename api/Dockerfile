# api/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY app/ ./app/
COPY scripts/ ./scripts/

# Создание директории для данных
RUN mkdir -p /app/data

# Запуск API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]