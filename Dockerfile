FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /workspace

# Клонирование репозитория (или копирование, если билдится локально)
# COPY . /workspace/rag-ktru-classifier/

# Создание необходимых директорий
RUN mkdir -p /workspace/rag-ktru-classifier/models \
             /workspace/rag-ktru-classifier/data \
             /workspace/rag-ktru-classifier/logs \
             /workspace/rag-ktru-classifier/qdrant_storage

# Копирование файлов зависимостей
COPY requirements.txt /workspace/rag-ktru-classifier/

# Установка Python зависимостей
RUN cd /workspace/rag-ktru-classifier && pip install --no-cache-dir -r requirements.txt

# Загрузка и установка Qdrant в папку проекта
RUN cd /workspace/rag-ktru-classifier && \
    curl -L https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-unknown-linux-gnu.tar.gz -o qdrant.tar.gz && \
    tar -xzf qdrant.tar.gz && \
    rm qdrant.tar.gz && \
    chmod +x qdrant

# Копирование скриптов и конфигурации
COPY *.py /workspace/rag-ktru-classifier/
COPY config.yaml /workspace/rag-ktru-classifier/
COPY start.sh /workspace/rag-ktru-classifier/
COPY supervisord.conf /workspace/rag-ktru-classifier/
RUN chmod +x /workspace/rag-ktru-classifier/start.sh

# Установка рабочей директории в папку проекта
WORKDIR /workspace/rag-ktru-classifier

# Открытие портов
EXPOSE 8000 6333

# Запуск сервисов через start.sh (который теперь адаптирован для локальной папки)
CMD ["./start.sh"]