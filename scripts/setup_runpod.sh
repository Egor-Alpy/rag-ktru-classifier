#!/bin/bash
# scripts/setup_runpod.sh

# Проверка и установка Docker
if ! command -v docker &> /dev/null; then
    echo "Docker не установлен. Установка Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

# Проверка и установка docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose не установлен. Установка Docker Compose..."
    curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# Настройка Docker для работы с NVIDIA GPU
echo "Настройка Docker для работы с NVIDIA GPU..."
cat <<EOF > /etc/docker/daemon.json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
EOF

# Перезапуск Docker
systemctl restart docker

echo "Настройка RunPod успешно завершена."