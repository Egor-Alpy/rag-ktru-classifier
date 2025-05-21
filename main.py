import uvicorn
import os
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# Конфигурация сервера
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
WORKERS = int(os.getenv("WORKERS", "1"))

if __name__ == "__main__":
    print(f"Запуск API сервера на {HOST}:{PORT} с {WORKERS} рабочими процессами...")
    uvicorn.run(
        "app.api:app",
        host=HOST,
        port=PORT,
        workers=WORKERS
    )