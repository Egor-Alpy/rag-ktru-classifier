import time
import json
import os
from datetime import datetime
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from process_ktru_data import process_ktru_data
from process_ktru_json import process_ktru_json
from config import (
    MONGO_EXTERNAL_URI, MONGO_LOCAL_URI, MONGO_DB_NAME, MONGO_COLLECTION,
    KTRU_JSON_PATH, ENABLE_JSON_FALLBACK
)

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def connect_to_mongodb(uri, db_name=None):
    """Подключение к MongoDB"""
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Проверка подключения
        client.admin.command('ping')
        logger.info(f"Успешное подключение к MongoDB: {uri}")

        if db_name:
            db = client[db_name]
            return client, db
        return client, None

    except ConnectionFailure as e:
        logger.error(f"Не удалось подключиться к MongoDB {uri}: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Ошибка при подключении к MongoDB {uri}: {e}")
        return None, None


def load_ktru_from_json():
    """Загрузка данных КТРУ из JSON файла как fallback"""
    try:
        if not os.path.exists(KTRU_JSON_PATH):
            logger.error(f"JSON файл не найден: {KTRU_JSON_PATH}")
            return False

        logger.info(f"Загрузка данных КТРУ из JSON файла: {KTRU_JSON_PATH}")

        # Используем функцию из process_ktru_json.py
        process_ktru_json(KTRU_JSON_PATH)

        logger.info("Данные КТРУ успешно загружены из JSON файла")
        return True

    except Exception as e:
        logger.error(f"Ошибка при загрузке данных из JSON: {e}")
        return False


def check_json_file_status():
    """Проверка состояния JSON файла"""
    if not os.path.exists(KTRU_JSON_PATH):
        logger.warning(f"JSON файл не найден: {KTRU_JSON_PATH}")
        return False

    try:
        file_stat = os.stat(KTRU_JSON_PATH)
        file_size = file_stat.st_size
        file_modified = datetime.fromtimestamp(file_stat.st_mtime)

        logger.info(f"JSON файл найден:")
        logger.info(f"  - Путь: {KTRU_JSON_PATH}")
        logger.info(f"  - Размер: {file_size / (1024 * 1024):.2f} МБ")
        logger.info(f"  - Последнее изменение: {file_modified}")

        # Проверяем что файл не пустой
        if file_size < 100:
            logger.warning("JSON файл слишком мал, возможно поврежден")
            return False

        return True

    except Exception as e:
        logger.error(f"Ошибка при проверке JSON файла: {e}")
        return False


def sync_ktru_data():
    """Синхронизация данных КТРУ с внешнего сервера или из JSON"""
    try:
        logger.info("Запуск синхронизации данных КТРУ")

        # Сначала пытаемся подключиться к внешней MongoDB
        external_client, external_db = connect_to_mongodb(MONGO_EXTERNAL_URI, MONGO_DB_NAME)

        if not external_client or not external_db:
            logger.warning("Не удалось подключиться к внешней MongoDB")

            # Проверяем, включен ли JSON fallback
            if not ENABLE_JSON_FALLBACK:
                logger.error("JSON fallback отключен в конфигурации")
                return False

            logger.info("Переключение на загрузку данных из JSON файла")

            # Проверяем наличие и состояние JSON файла
            if not check_json_file_status():
                logger.error("JSON файл недоступен или поврежден")
                return False

            # Загружаем данные из JSON
            return load_ktru_from_json()

        # Если удалось подключиться к внешней MongoDB, продолжаем обычную синхронизацию
        external_ktru = external_db[MONGO_COLLECTION]

        # Подключение к локальной MongoDB
        local_client, local_db = connect_to_mongodb(MONGO_LOCAL_URI, MONGO_DB_NAME)
        if not local_client or not local_db:
            logger.error("Не удалось подключиться к локальной MongoDB")
            external_client.close()

            # Если локальная MongoDB недоступна, но есть JSON, используем его
            if ENABLE_JSON_FALLBACK and check_json_file_status():
                logger.info("Локальная MongoDB недоступна, используем JSON fallback")
                return load_ktru_from_json()

            return False

        local_ktru = local_db[MONGO_COLLECTION]

        # Проверяем наличие коллекции sync_info, создаем если не существует
        if "sync_info" not in local_db.list_collection_names():
            local_db.create_collection("sync_info")

        # Получаем время последней синхронизации
        sync_info = local_db["sync_info"].find_one({"type": "ktru_sync"})
        last_sync = sync_info["last_sync"] if sync_info else datetime.min

        logger.info(f"Последняя синхронизация: {last_sync}")

        # Получаем новые или обновленные записи
        query = {"updated_at": {"$gt": last_sync}}
        new_records = external_ktru.find(query)

        # Подсчитываем количество записей для обновления
        count_to_update = external_ktru.count_documents(query)
        logger.info(f"Найдено {count_to_update} новых/обновленных записей КТРУ")

        if count_to_update == 0:
            logger.info("Нет новых данных для синхронизации")
            external_client.close()
            local_client.close()
            return True

        # Обновляем локальную базу данных
        updated_count = 0

        for record in new_records:
            local_ktru.replace_one(
                {"ktru_code": record["ktru_code"]},
                record,
                upsert=True
            )
            updated_count += 1

            if updated_count % 1000 == 0:
                logger.info(f"Обновлено {updated_count}/{count_to_update} записей")

        # Обновляем время синхронизации
        current_time = datetime.now()
        local_db["sync_info"].update_one(
            {"type": "ktru_sync"},
            {"$set": {"last_sync": current_time}},
            upsert=True
        )

        logger.info(f"Синхронизировано {updated_count} записей КТРУ")

        # Закрываем соединения
        external_client.close()
        local_client.close()

        # Если были обновления, запускаем процесс обновления векторной базы
        if updated_count > 0:
            logger.info("Запуск обновления векторной базы данных")
            process_ktru_data()

        return True

    except Exception as e:
        logger.error(f"Ошибка при синхронизации данных КТРУ: {e}")

        # При любой ошибке пытаемся использовать JSON fallback
        if ENABLE_JSON_FALLBACK and check_json_file_status():
            logger.info("Ошибка синхронизации, пытаемся использовать JSON fallback")
            return load_ktru_from_json()

        return False


def run_sync_loop(interval=3600):
    """Запуск цикла синхронизации с заданным интервалом"""
    logger.info(f"Запуск цикла синхронизации данных КТРУ с интервалом {interval} секунд")
    logger.info(f"JSON fallback {'включен' if ENABLE_JSON_FALLBACK else 'отключен'}")

    if ENABLE_JSON_FALLBACK:
        logger.info(f"Путь к JSON файлу: {KTRU_JSON_PATH}")

    # Проверяем наличие JSON файла при запуске
    if ENABLE_JSON_FALLBACK:
        check_json_file_status()

    while True:
        try:
            sync_success = sync_ktru_data()

            if sync_success:
                logger.info(f"Синхронизация успешно завершена. Следующая через {interval} секунд.")
            else:
                logger.warning(f"Синхронизация завершилась с ошибками. Повтор через {interval} секунд.")

        except Exception as e:
            logger.error(f"Необработанная ошибка в цикле синхронизации: {e}")

        # Ждем следующего цикла
        time.sleep(interval)


if __name__ == "__main__":
    run_sync_loop()