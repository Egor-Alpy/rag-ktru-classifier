"""
Модуль для работы с векторной базой данных Qdrant
Оптимизирован для эффективного поиска и индексации
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue, SearchRequest,
    OptimizersConfigDiff, HnswConfigDiff
)

from config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
    VECTOR_SIZE, BATCH_SIZE, SEARCH_TOP_K
)
from embeddings import embedding_manager

logger = logging.getLogger(__name__)


class VectorDatabase:
    """Менеджер векторной базы данных"""

    def __init__(self):
        """Инициализация подключения к Qdrant"""
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = QDRANT_COLLECTION
        self.vector_size = embedding_manager.vector_size

        # Создаем или проверяем коллекцию
        self._setup_collection()

    def _setup_collection(self):
        """Создание или проверка коллекции"""
        try:
            # Проверяем существование коллекции
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            if not collection_exists:
                logger.info(f"Создание коллекции {self.collection_name}")

                # Создаем коллекцию с оптимизированными параметрами
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    ),
                    # Оптимизация для быстрого поиска
                    hnsw_config=HnswConfigDiff(
                        m=16,
                        ef_construct=100,
                        full_scan_threshold=10000
                    ),
                    optimizers_config=OptimizersConfigDiff(
                        default_segment_number=5,
                        indexing_threshold=20000
                    )
                )

                # Создаем индексы для полей
                self._create_indexes()

                logger.info(f"✅ Коллекция {self.collection_name} создана")
            else:
                # Проверяем параметры существующей коллекции
                collection_info = self.client.get_collection(self.collection_name)
                logger.info(f"✅ Коллекция {self.collection_name} существует")
                logger.info(f"   Векторов: {collection_info.points_count}")
                logger.info(f"   Размерность: {collection_info.config.params.vectors.size}")

        except Exception as e:
            logger.error(f"Ошибка при настройке коллекции: {e}")
            raise

    def _create_indexes(self):
        """Создание индексов для ускорения поиска"""
        try:
            # Индекс для кода KTRU
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="ktru_code",
                field_schema="keyword"
            )

            # Индекс для категории
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="category",
                field_schema="keyword"
            )

            logger.info("✅ Индексы созданы")

        except Exception as e:
            logger.warning(f"Не удалось создать индексы (возможно уже существуют): {e}")

    def load_ktru_from_json(self, json_path: Path, recreate: bool = False):
        """Загрузка данных KTRU из JSON файла"""
        if recreate:
            logger.info("Пересоздание коллекции...")
            self.client.delete_collection(self.collection_name)
            self._setup_collection()

        logger.info(f"Загрузка данных из {json_path}")

        # Загружаем JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            ktru_data = json.load(f)

        logger.info(f"Загружено {len(ktru_data)} записей KTRU")

        # Обрабатываем батчами
        total_batches = (len(ktru_data) + BATCH_SIZE - 1) // BATCH_SIZE

        with tqdm(total=len(ktru_data), desc="Индексация KTRU") as pbar:
            for batch_idx in range(0, len(ktru_data), BATCH_SIZE):
                batch = ktru_data[batch_idx:batch_idx + BATCH_SIZE]

                # Подготавливаем тексты для эмбеддингов
                texts = []
                points = []

                for idx, item in enumerate(batch):
                    # Готовим текст для эмбеддинга
                    text = embedding_manager.prepare_ktru_text(item)
                    texts.append(text)

                    # Извлекаем категорию из кода
                    ktru_code = item.get('ktru_code', '')
                    category = ktru_code.split('.')[0] if ktru_code else ''

                    # Готовим payload
                    payload = {
                        'ktru_code': ktru_code,
                        'title': item.get('title', ''),
                        'description': item.get('description', ''),
                        'unit': item.get('unit', ''),
                        'category': category,
                        'keywords': item.get('keywords', []),
                        'attributes': item.get('attributes', []),
                        '_text': text[:500]  # Для отладки
                    }

                    points.append({
                        'id': batch_idx + idx,
                        'payload': payload
                    })

                # Создаем эмбеддинги
                embeddings = embedding_manager.encode_batch(texts)

                # Создаем точки для Qdrant
                qdrant_points = []
                for point, embedding in zip(points, embeddings):
                    qdrant_points.append(
                        PointStruct(
                            id=point['id'],
                            vector=embedding.tolist(),
                            payload=point['payload']
                        )
                    )

                # Загружаем в Qdrant
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=qdrant_points
                )

                pbar.update(len(batch))

        # Оптимизируем коллекцию после загрузки
        logger.info("Оптимизация коллекции...")
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=OptimizersConfigDiff(
                indexing_threshold=1000
            )
        )

        # Статистика
        collection_info = self.client.get_collection(self.collection_name)
        logger.info(f"✅ Загрузка завершена. Векторов в базе: {collection_info.points_count}")

    def search(self, query_text: str, top_k: int = SEARCH_TOP_K,
               filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Векторный поиск по запросу"""
        # Создаем эмбеддинг запроса
        query_embedding = embedding_manager.encode_single(query_text)

        # Формируем фильтр если нужно
        query_filter = None
        if filter_dict:
            conditions = []
            for field, value in filter_dict.items():
                conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=value)
                    )
                )
            query_filter = Filter(must=conditions)

        # Выполняем поиск
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True
        )

        # Преобразуем результаты
        output = []
        for result in results:
            output.append({
                'score': result.score,
                'payload': result.payload,
                'id': result.id
            })

        return output

    def search_by_code_prefix(self, code_prefix: str, limit: int = 100) -> List[Dict]:
        """Поиск по префиксу кода KTRU"""
        # Используем scroll для получения всех записей с префиксом
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="ktru_code",
                        match=MatchValue(value=code_prefix)
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        return [{'id': r.id, 'payload': r.payload} for r in records]

    def get_by_code(self, ktru_code: str) -> Optional[Dict]:
        """Получение записи по точному коду KTRU"""
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="ktru_code",
                        match=MatchValue(value=ktru_code)
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False
        )

        if records:
            return {'id': records[0].id, 'payload': records[0].payload}
        return None

    def get_statistics(self) -> Dict:
        """Получение статистики по коллекции"""
        try:
            collection_info = self.client.get_collection(self.collection_name)

            # Подсчет по категориям
            categories = {}
            offset = None

            while True:
                records, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=["category"],
                    with_vectors=False
                )

                for record in records:
                    cat = record.payload.get('category', 'unknown')
                    categories[cat] = categories.get(cat, 0) + 1

                if offset is None:
                    break

            return {
                'total_vectors': collection_info.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'categories': categories,
                'status': 'healthy'
            }

        except Exception as e:
            logger.error(f"Ошибка при получении статистики: {e}")
            return {'status': 'error', 'error': str(e)}

    def health_check(self) -> bool:
        """Проверка доступности векторной БД"""
        try:
            self.client.get_collections()
            return True
        except:
            return False


# Глобальный экземпляр
vector_db = VectorDatabase()