#!/usr/bin/env python3

import os
import json
import argparse
import asyncio
import httpx
from tqdm import tqdm
from typing import List, Dict, Any
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models


def parse_args():
    """Парсит аргументы командной строки"""
    parser = argparse.ArgumentParser(description="Загрузка данных КТРУ в Qdrant")
    parser.add_argument(
        "--data-file",
        default=os.environ.get("KTRU_DATA_FILE", "/app/data/ktru_data.json"),
        help="Путь к файлу данных КТРУ"
    )
    parser.add_argument(
        "--mongo-uri",
        default=os.environ.get("MONGO_URI", "mongodb://username:password@hostname:27017/"),
        help="URI для подключения к MongoDB"
    )
    parser.add_argument(
        "--mongo-db",
        default=os.environ.get("MONGO_DB", "ktru_db"),
        help="Имя базы данных MongoDB"
    )
    parser.add_argument(
        "--mongo-collection",
        default=os.environ.get("MONGO_COLLECTION", "ktru_data"),
        help="Имя коллекции MongoDB"
    )
    parser.add_argument(
        "--embedding-url",
        default=os.environ.get("EMBEDDING_SERVICE_URL", "http://embeddings:8080"),
        help="URL сервиса эмбеддингов"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", "32")),
        help="Размер пакета для загрузки"
    )

    return parser.parse_args()


def load_ktru_data(mongo_uri: str, mongo_db: str, mongo_collection: str) -> List[Dict[str, Any]]:
    """Загружает данные КТРУ из MongoDB"""
    try:
        from pymongo import MongoClient

        logger.info(f"Подключение к MongoDB {mongo_uri}, база {mongo_db}, коллекция {mongo_collection}")
        client = MongoClient(mongo_uri)
        db = client[mongo_db]
        collection = db[mongo_collection]

        # Загрузка всех документов
        data = list(collection.find({}))
        logger.info(f"Загружено {len(data)} записей КТРУ из MongoDB")

        # MongoDB добавляет поле _id, которое не нужно нам в Qdrant
        for item in data:
            if '_id' in item:
                del item['_id']

        return data
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных КТРУ из MongoDB: {e}")
        raise


def prepare_ktru_text(ktru_item: Dict[str, Any]) -> str:
    """Подготавливает текстовое представление записи КТРУ для эмбеддинга"""
    text_parts = [
        f"Название: {ktru_item.get('title', '')}",
    ]

    description = ktru_item.get('description', '')
    if description:
        text_parts.append(f"Описание: {description}")

    # Добавляем атрибуты, если они есть
    attributes = ktru_item.get('attributes', [])
    if attributes:
        attrs = []
        for attr in attributes:
            attr_name = attr.get('attr_name', '')
            attr_values = attr.get('attr_values', [])
            values_str = ', '.join([str(v.get('value', '')) for v in attr_values])
            attrs.append(f"{attr_name}: {values_str}")

        if attrs:
            text_parts.append(f"Атрибуты: {'; '.join(attrs)}")

    return " ".join(text_parts)


async def get_embeddings_batch(
        client: httpx.AsyncClient,
        texts: List[str]
) -> List[List[float]]:
    """Получает эмбеддинги для пакета текстов"""
    response = await client.post("/embed_batch", json={"texts": texts})
    response.raise_for_status()
    return response.json()["embeddings"]


async def process_data(args):
    # Загрузка данных из MongoDB
    ktru_data = load_ktru_data(args.mongo_uri, args.mongo_db, args.mongo_collection)

    # Инициализация клиентов
    qdrant_client = QdrantClient(host=args.host, port=args.port)
    embedding_client = httpx.AsyncClient(base_url=args.embedding_url, timeout=60.0)

    try:
        # Проверка соединения с сервисом эмбеддингов
        health_response = await embedding_client.get("/health")
        health_response.raise_for_status()

        # Обработка данных пакетами
        batch_size = args.batch_size
        num_batches = (len(ktru_data) + batch_size - 1) // batch_size

        with tqdm(total=len(ktru_data), desc="Загрузка данных в Qdrant") as pbar:
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(ktru_data))
                batch_data = ktru_data[start_idx:end_idx]

                # Подготовка текстов для эмбеддингов
                texts = [prepare_ktru_text(item) for item in batch_data]

                # Получение эмбеддингов
                embeddings = await get_embeddings_batch(embedding_client, texts)

                # Подготовка точек для загрузки в Qdrant
                points = []
                for idx, (item, embedding) in enumerate(zip(batch_data, embeddings)):
                    point_id = start_idx + idx

                    # Подготовка данных для сохранения
                    payload = {
                        "ktru_code": item.get("ktru_code", ""),
                        "title": item.get("title", ""),
                        "description": item.get("description", ""),
                        "attributes": item.get("attributes", []),
                        "version": item.get("version", ""),
                        "updated_at": item.get("updated_at", "")
                    }

                    points.append(models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    ))

                # Загрузка данных в Qdrant
                qdrant_client.upsert(
                    collection_name=args.collection,
                    points=points
                )

                pbar.update(len(batch_data))

        # Проверка количества загруженных документов
        collection_info = qdrant_client.get_collection(args.collection)
        logger.info(f"Всего загружено {collection_info.vectors_count} документов в коллекцию {args.collection}")

    finally:
        await embedding_client.aclose()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(process_data(args))