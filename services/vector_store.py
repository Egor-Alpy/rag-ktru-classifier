from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchRequest, ScoredPoint
)
from loguru import logger
from config import settings
import uuid


class VectorStoreService:
    """Service for managing vector storage with Qdrant"""

    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.collection_name = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure collection exists with proper configuration"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection {self.collection_name} created")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise

    def add_ktru_codes(
            self,
            codes: List[Dict[str, Any]],
            embeddings: np.ndarray,
            batch_size: int = 100
    ) -> int:
        """
        Add KTRU codes to vector store

        Args:
            codes: List of KTRU code information
            embeddings: Numpy array of embeddings
            batch_size: Batch size for insertion

        Returns:
            Number of codes added
        """
        points = []

        for i, (code, embedding) in enumerate(zip(codes, embeddings)):
            point_id = str(uuid.uuid4())

            # Prepare payload
            payload = {
                "code": code.get("code"),
                "name": code.get("name"),
                "description": code.get("description", ""),
                "parent_code": code.get("parent_code", ""),
                "level": code.get("level", 0),
                "okpd2_code": code.get("okpd2_code", ""),
                "full_text": f"{code.get('code')} {code.get('name')} {code.get('description', '')}"
            }

            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)

            # Insert in batches
            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                points = []

        # Insert remaining points
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

        logger.info(f"Added {len(codes)} KTRU codes to vector store")
        return len(codes)

    def search(
            self,
            query_embedding: np.ndarray,
            top_k: int = 10,
            filters: Optional[Dict[str, Any]] = None,
            score_threshold: Optional[float] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar KTRU codes

        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            filters: Optional filters
            score_threshold: Minimum similarity score

        Returns:
            List of (payload, score) tuples
        """
        search_params = {
            "collection_name": self.collection_name,
            "query_vector": query_embedding.tolist(),
            "limit": top_k
        }

        # Add filters if provided
        if filters:
            filter_conditions = []

            if "level" in filters:
                filter_conditions.append(
                    FieldCondition(
                        key="level",
                        match=MatchValue(value=filters["level"])
                    )
                )

            if "parent_code" in filters:
                filter_conditions.append(
                    FieldCondition(
                        key="parent_code",
                        match=MatchValue(value=filters["parent_code"])
                    )
                )

            if filter_conditions:
                search_params["query_filter"] = Filter(
                    must=filter_conditions
                )

        results = self.client.search(**search_params)

        # Filter by score threshold if provided
        output = []
        for result in results:
            if score_threshold is None or result.score >= score_threshold:
                output.append((result.payload, result.score))

        return output

    def get_by_code(self, ktru_code: str) -> Optional[Dict[str, Any]]:
        """Get KTRU information by code"""
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="code",
                        match=MatchValue(value=ktru_code)
                    )
                ]
            ),
            limit=1
        )[0]

        if results:
            return results[0].payload
        return None

    def delete_collection(self):
        """Delete the entire collection"""
        self.client.delete_collection(self.collection_name)
        logger.warning(f"Deleted collection: {self.collection_name}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": info.name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "segments_count": info.segments_count,
            "status": info.status
        }


# Global instance
vector_store = None


def get_vector_store() -> VectorStoreService:
    """Get or create vector store instance"""
    global vector_store
    if vector_store is None:
        vector_store = VectorStoreService()
    return vector_store