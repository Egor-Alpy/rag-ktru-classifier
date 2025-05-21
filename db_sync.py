import pymongo
import time
from datetime import datetime, timedelta
from pymongo import MongoClient
from typing import List, Dict, Any, Optional
import json

from config import (
    MONGO_URI, MONGO_DB, MONGO_COLLECTION,
    KTRU_SYNC_INTERVAL
)
from logging_config import setup_logging

logger = setup_logging("db_sync")


class MongoSync:
    def __init__(self):
        """Инициализация класса для синхронизации с MongoDB."""
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[MONGO_DB]
        self.collection = self.db[MONGO_COLLECTION]
        self.last_sync_time = datetime.now() - timedelta(days=365)  # Начальное значение - год назад

        # Создаем индекс по полю updated_at для быстрого поиска обновленных записей
        self.collection.create_index("updated_at")

        logger.info(f"MongoDB connection initialized to {MONGO_URI}, database: {MONGO_DB}")

    def get_updated_documents(self) -> List[Dict[str, Any]]:
        """Получение всех документов, обновленных с момента последней синхронизации."""
        query = {"updated_at": {"$gt": self.last_sync_time}}

        try:
            documents = list(self.collection.find(query))
            logger.info(f"Found {len(documents)} updated documents since {self.last_sync_time}")
            return documents
        except Exception as e:
            logger.error(f"Error fetching documents: {e}")
            return []

    def sync(self) -> List[Dict[str, Any]]:
        """Синхронизация данных из MongoDB и обновление времени последней синхронизации."""
        updated_docs = self.get_updated_documents()
        self.last_sync_time = datetime.now()
        return updated_docs

    def get_document_by_ktru_code(self, ktru_code: str) -> Optional[Dict[str, Any]]:
        """Получение документа по коду КТРУ."""
        try:
            document = self.collection.find_one({"ktru_code": ktru_code})
            return document
        except Exception as e:
            logger.error(f"Error fetching document by ktru_code {ktru_code}: {e}")
            return None

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Получение всех документов из коллекции."""
        try:
            documents = list(self.collection.find())
            logger.info(f"Fetched all {len(documents)} documents from collection")
            return documents
        except Exception as e:
            logger.error(f"Error fetching all documents: {e}")
            return []


def run_sync_job():
    """Запуск задачи синхронизации с MongoDB."""
    mongo_sync = MongoSync()

    while True:
        try:
            logger.info("Starting MongoDB synchronization...")
            updated_docs = mongo_sync.sync()
            logger.info(f"Synchronized {len(updated_docs)} documents from MongoDB")

            # TODO: Здесь должен быть код для обновления Qdrant
            # Он будет добавлен позже в indexer.py

            logger.info(f"Sleeping for {KTRU_SYNC_INTERVAL} seconds until next sync")
            time.sleep(KTRU_SYNC_INTERVAL)
        except Exception as e:
            logger.error(f"Error during synchronization: {e}")
            logger.info("Retrying in 60 seconds...")
            time.sleep(60)  # Короткая пауза перед повторной попыткой


if __name__ == "__main__":
    run_sync_job()