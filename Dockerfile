FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0

# Установка системных зависимостей
RUN apt-get install -y \
    git \
    wget \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /workspace

# Создание необходимых директорий
RUN mkdir -p /workspace/models /workspace/data /workspace/logs /workspace/qdrant_storage

# Копирование файлов зависимостей
COPY requirements.txt /workspace/

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Загрузка и установка Qdrant
RUN curl -L https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-unknown-linux-gnu.tar.gz -o qdrant.tar.gz \
    && tar -xzf qdrant.tar.gz -C /workspace \
    && rm qdrant.tar.gz


# Копирование скриптов и конфигурации
COPY *.py /workspace/
COPY start.sh /workspace/
COPY supervisord.conf /etc/supervisor/conf.d/
RUN chmod +x /workspace/start.sh

# Открытие портов
EXPOSE 8000 6333 27017

# Запуск сервисов через Supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]