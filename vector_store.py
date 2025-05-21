from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from config import QDRANT_URL, QDRANT_COLLECTION, VECTOR_DIM
from logging_config import setup_logging

logger = setup_logging("vector_store")


class QdrantStore:
    def __init__(self):
        """Инициализация класса для работы с Qdrant."""
        self.client = QdrantClient(url=QDRANT_URL)
        self._ensure_collection()
        logger.info(f"Qdrant connection initialized to {QDRANT_URL}")

    def _ensure_collection(self):
        """Проверяет существование коллекции и создает ее при необходимости."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if QDRANT_COLLECTION not in collection_names:
            logger.info(f"Creating collection {QDRANT_COLLECTION}")
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

            # Создаем индексы для метаданных
            self.client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="ktru_code",
                field_schema="keyword"
            )
            self.client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="title",
                field_schema="text"
            )
            logger.info(f"Collection {QDRANT_COLLECTION} created successfully with indexes")
        else:
            logger.info(f"Collection {QDRANT_COLLECTION} already exists")

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Добавляет документы и их эмбеддинги в Qdrant.
        """
        if not documents or not embeddings:
            logger.warning("No documents or embeddings to add")
            return

        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point_id = doc.get("ktru_code", "") + "_" + str(doc.get("chunk_id", 0))
            point_id = str(hash(point_id))  # Преобразуем в строковый хеш для уникальности

            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": doc.get("text", ""),
                    "ktru_code": doc.get("ktru_code", ""),
                    "title": doc.get("title", ""),
                    "chunk_id": doc.get("chunk_id", 0)
                }
            ))

        # Добавляем точки в коллекцию
        self.client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )

        logger.info(f"Added {len(points)} document chunks to Qdrant")

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Ищет наиболее релевантные документы по вектору запроса.
        """
        search_result = self.client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k
        )

        results = []
        for scored_point in search_result:
            results.append({
                "score": scored_point.score,
                "ktru_code": scored_point.payload.get("ktru_code", ""),
                "title": scored_point.payload.get("title", ""),
                "text": scored_point.payload.get("text", ""),
                "chunk_id": scored_point.payload.get("chunk_id", 0)
            })

        logger.debug(f"Found {len(results)} relevant documents for query")
        return results

    def delete_by_ktru_code(self, ktru_code: str):
        """
        Удаляет все документы с указанным кодом КТРУ из коллекции.
        """
        self.client.delete(
            collection_name=QDRANT_COLLECTION,
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="ktru_code",
                        match=models.MatchValue(value=ktru_code)
                    )
                ]
            )
        )
        logger.info(f"Deleted all documents with ktru_code {ktru_code} from Qdrant")

    def count_documents(self) -> int:
        """
        Возвращает количество документов в коллекции.
        """
        collection_info = self.client.get_collection(collection_name=QDRANT_COLLECTION)
        return collection_info.vectors_count