# Базовый образ с CUDA 11.8 и PyTorch
FROM pytorch/pytorch:2.0.1-cuda11.8.0-cudnn8-runtime

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
    pip install --no-cache-dir -r llm-requirements.txt

# Копирование кода всех сервисов
COPY api/ ./api/
COPY embeddings/ ./embeddings/
COPY llm/ ./llm/
COPY scripts/ ./scripts/

# Создание директорий для данных
RUN mkdir -p /app/data

# Конфигурация supervisor для управления процессами
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Переопределение URL-адресов в конфигурационных файлах для работы с localhost
RUN sed -i 's|http://embeddings:8080|http://localhost:8080|g' api/app/config.py && \
    sed -i 's|http://llm:8081|http://localhost:8081|g' api/app/config.py && \
    sed -i 's|qdrant_host: str = "qdrant"|qdrant_host: str = "localhost"|g' api/app/config.py

# Открываем необходимые порты
EXPOSE 6333 6334 8000 8080 8081

# Запуск supervisor для управления всеми сервисами
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]