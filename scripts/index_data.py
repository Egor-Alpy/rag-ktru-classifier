#!/usr/bin/env python3
"""
Script to index KTRU data into vector store
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from services.embeddings import get_embedding_service
from services.vector_store import get_vector_store


def load_ktru_data(data_path: Path) -> List[Dict[str, Any]]:
    """Load KTRU data from file"""
    if data_path.suffix == ".json":
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")


def create_embeddings_batch(
        texts: List[str],
        batch_size: int = 32
) -> List[Any]:
    """Create embeddings in batches"""
    embedding_service = get_embedding_service()
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embedding_service.encode(batch_texts)
        embeddings.extend(batch_embeddings)

    return embeddings


def main():
    """Main indexing function"""
    # Load KTRU data
    data_path = settings.data_dir / "ktru_codes.json"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run download_ktru.py first")
        return

    logger.info(f"Loading KTRU data from {data_path}")
    ktru_data = load_ktru_data(data_path)
    logger.info(f"Loaded {len(ktru_data)} KTRU codes")

    # Initialize services
    embedding_service = get_embedding_service()
    vector_store = get_vector_store()

    # Prepare texts for embedding
    texts = []
    for item in ktru_data:
        # Combine relevant fields for embedding
        text_parts = [
            f"Код КТРУ: {item['code']}",
            f"Наименование: {item['name']}",
        ]

        if item.get("description"):
            text_parts.append(f"Описание: {item['description']}")

        text = " ".join(text_parts)
        texts.append(text)

    # Create embeddings
    logger.info("Creating embeddings...")
    embeddings = create_embeddings_batch(texts, batch_size=32)

    # Index into vector store
    logger.info("Indexing into vector store...")
    count = vector_store.add_ktru_codes(
        codes=ktru_data,
        embeddings=embeddings,
        batch_size=100
    )

    logger.info(f"Successfully indexed {count} KTRU codes")

    # Verify indexing
    info = vector_store.get_collection_info()
    logger.info(f"Collection info: {info}")


if __name__ == "__main__":
    main()