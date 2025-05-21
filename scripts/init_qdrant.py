#!/usr/bin/env python3

import os
import argparse
from qdrant_client import QdrantClient
from qdrant_client.http import models
from loguru import logger


def parse_args():
    """Парсит аргументы командной строки"""
    parser = argparse.ArgumentParser(description="Инициализация коллекции в Qdrant")
    parser.add_argument(
        "--host",
        default=os.environ.get("QDRANT_HOST", "qdrant"),
        help="Хост Qdrant"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("QDRANT_PORT", "6333")),
        help="Порт Qdrant"
    )
    parser.add_argument(
        "--collection",
        default=os.environ.get("QDRANT_COLLECTION", "ktru_vectors"),
        help="Имя коллекции"
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=int(os.environ.get("VECTOR_SIZE", "1024")),
        help="Размерность векторов"
    )

    return parser.parse_args()


def init_collection(host, port, collection_name, vector_size):
    """Инициализирует коллекцию в Qdrant"""
    client = QdrantClient(host=host, port=port)

    # Проверяем существование коллекции
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if collection_name in collection_names:
        logger.info(f"Коллекция {collection_name} уже существует")
        return

    # Создаем коллекцию
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=10000,
            memmap_threshold=50000
        )
    )

    # Создаем индексы для фильтрации
    client.create_payload_index(
        collection_name=collection_name,
        field_name="ktru_code",
        field_schema=models.PayloadSchemaType.KEYWORD
    )

    logger.info(f"Коллекция {collection_name} успешно создана с размерностью {vector_size}")


if __name__ == "__main__":
    args = parse_args()

    logger.info(f"Инициализация коллекции {args.collection} в Qdrant ({args.host}:{args.port})")
    init_collection(args.host, args.port, args.collection, args.vector_size)