from typing import Dict, Any, List, Optional
import time
from loguru import logger
from models.schemas import (
    ProductInfo, ClassificationResult, ClassificationStatus,
    ClassificationCandidate, KTRUCode
)
from services.embeddings import get_embedding_service
from services.vector_store import get_vector_store
from services.llm_service import get_llm_service
from config import settings
import numpy as np


class ClassifierService:
    """Main classification service"""

    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store()
        self.llm_service = get_llm_service()
        logger.info("Classifier service initialized")

    def classify(self, product: ProductInfo) -> ClassificationResult:
        """
        Classify product to KTRU code

        Args:
            product: Product information

        Returns:
            Classification result
        """
        start_time = time.time()

        try:
            # Step 1: Create product embedding
            product_dict = product.model_dump()
            product_embedding = self.embedding_service.encode_product(product_dict)

            # Step 2: Search for similar KTRU codes
            candidates = self.vector_store.search(
                query_embedding=product_embedding,
                top_k=settings.max_candidates,
                score_threshold=0.5
            )

            if not candidates:
                return ClassificationResult(
                    status=ClassificationStatus.NOT_FOUND,
                    code=None,
                    confidence=0.0,
                    candidates=[],
                    reasoning="Не найдено подходящих кодов КТРУ",
                    processing_time=time.time() - start_time
                )

            # Step 3: Use LLM for final classification
            llm_result = self.llm_service.classify_product(
                product_dict,
                candidates
            )

            # Step 4: Prepare classification candidates
            classification_candidates = []
            for ktru_info, score in candidates[:5]:  # Top 5 for response
                ktru_code = KTRUCode(
                    code=ktru_info["code"],
                    name=ktru_info["name"],
                    description=ktru_info.get("description", ""),
                    parent_code=ktru_info.get("parent_code"),
                    level=ktru_info.get("level", 0),
                    okpd2_code=ktru_info.get("okpd2_code")
                )

                classification_candidates.append(
                    ClassificationCandidate(
                        ktru_code=ktru_code,
                        score=float(score),
                        reasoning=None
                    )
                )

            # Step 5: Determine final result
            selected_code = llm_result.get("code")
            confidence = llm_result.get("confidence", 0)
            reasoning = llm_result.get("reasoning", "")

            if selected_code and confidence >= settings.confidence_threshold:
                status = ClassificationStatus.SUCCESS
            elif selected_code and confidence > 0:
                status = ClassificationStatus.LOW_CONFIDENCE
            else:
                status = ClassificationStatus.NOT_FOUND
                selected_code = None

            return ClassificationResult(
                status=status,
                code=selected_code,
                confidence=confidence,
                candidates=classification_candidates,
                reasoning=reasoning,
                processing_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Classification error: {e}")
            return ClassificationResult(
                status=ClassificationStatus.ERROR,
                code=None,
                confidence=0.0,
                candidates=[],
                reasoning=f"Ошибка классификации: {str(e)}",
                processing_time=time.time() - start_time
            )

    def search_ktru(
            self,
            query: str,
            top_k: int = 10,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search KTRU codes by query

        Args:
            query: Search query
            top_k: Number of results
            filters: Optional filters

        Returns:
            List of KTRU codes with scores
        """
        # Create query embedding
        query_embedding = self.embedding_service.encode(query)[0]

        # Search in vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )

        # Format results
        output = []
        for ktru_info, score in results:
            output.append({
                "code": ktru_info["code"],
                "name": ktru_info["name"],
                "description": ktru_info.get("description", ""),
                "score": float(score),
                "level": ktru_info.get("level", 0),
                "parent_code": ktru_info.get("parent_code", "")
            })

        return output


# Global instance
classifier_service = None


def get_classifier_service() -> ClassifierService:
    """Get or create classifier service instance"""
    global classifier_service
    if classifier_service is None:
        classifier_service = ClassifierService()
    return classifier_service