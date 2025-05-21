# Базовый образ с CUDA 11.8 и PyTorch
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Установка Qdrant
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Скачивание и установка Qdrant
RUN mkdir -p /qdrant/storage && \
    wget https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-unknown-linux-gnu.tar.gz -O qdrant.tar.gz && \
    tar -xzf qdrant.tar.gz -C /usr/local/bin && \
    rm qdrant.tar.gz

# Установка зависимостей для всех сервисов
COPY api/requirements.txt api-requirements.txt
COPY embeddings/requirements.txt embedding-requirements.txt
COPY llm/requirements.txt llm-requirements.txt

RUN pip install --no-cache-dir -r api-requirements.txt && \
    pip install --no-cache-dir -r embedding-requirements.txt && \
    pip install --no-cache-dir -r llm-requirements.txt && \
    pip install --no-cache-dir pydantic-settings==2.0.0

# Копирование кода всех сервисов
COPY api/ ./api/
COPY embeddings/ ./embeddings/
COPY llm/ ./llm/
COPY scripts/ ./scripts/

# Создание директорий для данных
RUN mkdir -p /app/data

# Создание файлов __init__.py для корректной работы модулей
RUN touch /app/api/__init__.py /app/api/app/__init__.py \
    /app/embeddings/__init__.py /app/embeddings/app/__init__.py \
    /app/llm/__init__.py /app/llm/app/__init__.py

# Копирование конфигурации .env
COPY .env /app/.env
COPY .env /app/api/.env
COPY .env /app/embeddings/.env
COPY .env /app/llm/.env

# Конфигурация supervisor для управления процессами
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Открываем необходимые порты
EXPOSE 6333 6334 8000 8080 8081

# Запуск supervisor для управления всеми сервисами
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]