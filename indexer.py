from typing import List, Dict, Any
import time

from db_sync import MongoSync
from text_processing import create_document_chunks
from embedding import Embedder
from vector_store import QdrantStore
from config import INDEX_REFRESH_INTERVAL
from logging_config import setup_logging

logger = setup_logging("indexer")


class Indexer:
    def __init__(self):
        """
        Инициализация класса для индексации документов КТРУ.
        """
        self.mongo_sync = MongoSync()
        self.embedder = Embedder()
        self.vector_store = QdrantStore()
        logger.info("Indexer initialized")

    def index_documents(self, documents: List[Dict[str, Any]], reindex: bool = False):
        """
        Индексирует документы КТРУ в векторной базе данных.
        """
        if not documents:
            logger.warning("No documents to index")
            return

        logger.info(f"Indexing {len(documents)} documents")

        for document in documents:
            ktru_code = document.get('ktru_code', '')
            if not ktru_code:
                logger.warning(f"Skipping document without ktru_code: {document}")
                continue

            # Если требуется переиндексация, удаляем старые данные
            if reindex:
                self.vector_store.delete_by_ktru_code(ktru_code)

            # Создаем чанки для документа
            chunks = create_document_chunks(document)

            if not chunks:
                logger.warning(f"No chunks created for document with ktru_code: {ktru_code}")
                continue

            # Создаем эмбеддинги для чанков
            chunk_texts = [chunk.get('text', '') for chunk in chunks]
            embeddings = self.embedder.create_embeddings(chunk_texts)

            # Добавляем чанки и эмбеддинги в векторное хранилище
            self.vector_store.add_documents(chunks, embeddings)

            logger.debug(f"Indexed document with ktru_code: {ktru_code}, created {len(chunks)} chunks")

    def index_all_documents(self):
        """
        Индексирует все документы из MongoDB.
        """
        logger.info("Starting indexing of all documents")
        documents = self.mongo_sync.get_all_documents()
        logger.info(f"Found {len(documents)} documents to index")
        self.index_documents(documents, reindex=True)
        logger.info("Completed indexing of all documents")

    def update_index(self):
        """
        Обновляет индекс для измененных документов.
        """
        logger.info("Updating index for changed documents")
        updated_documents = self.mongo_sync.sync()
        self.index_documents(updated_documents, reindex=True)
        logger.info(f"Completed updating index for {len(updated_documents)} documents")


def run_indexer_job():
    """
    Запуск задачи индексации документов КТРУ.
    """
    indexer = Indexer()

    # Первоначальная индексация всех документов
    logger.info("Starting initial indexing of all documents")
    indexer.index_all_documents()

    # Периодическое обновление индекса
    while True:
        try:
            logger.info(f"Sleeping for {INDEX_REFRESH_INTERVAL} seconds until next index update")
            time.sleep(INDEX_REFRESH_INTERVAL)

            logger.info("Starting index update")
            indexer.update_index()
            logger.info("Completed index update")
        except Exception as e:
            logger.error(f"Error during index update: {e}")
            logger.info("Retrying in 60 seconds...")
            time.sleep(60)  # Короткая пауза перед повторной попыткой


if __name__ == "__main__":
    run_indexer_job()