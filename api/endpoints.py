from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, Dict, Any
from models.schemas import (
    ProductInfo, ClassificationResult, HealthCheck,
    SearchRequest, IndexingRequest
)
from services.classifier import get_classifier_service
from services.vector_store import get_vector_store
from services.embeddings import get_embedding_service
from loguru import logger
import numpy as np

router = APIRouter()


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Check health status of all services"""
    services_status = {}

    # Check vector store
    try:
        vector_store = get_vector_store()
        info = vector_store.get_collection_info()
        services_status["vector_store"] = True
    except Exception as e:
        logger.error(f"Vector store health check failed: {e}")
        services_status["vector_store"] = False

    # Check embedding service
    try:
        embedding_service = get_embedding_service()
        test_embedding = embedding_service.encode("test")
        services_status["embedding_service"] = len(test_embedding) > 0
    except Exception as e:
        logger.error(f"Embedding service health check failed: {e}")
        services_status["embedding_service"] = False

    # Check LLM service
    try:
        from services.llm_service import get_llm_service
        llm_service = get_llm_service()
        services_status["llm_service"] = llm_service.model is not None
    except Exception as e:
        logger.error(f"LLM service health check failed: {e}")
        services_status["llm_service"] = False

    all_healthy = all(services_status.values())

    return HealthCheck(
        status="healthy" if all_healthy else "degraded",
        services=services_status
    )


@router.post("/classify", response_model=ClassificationResult)
async def classify_product(product: ProductInfo = Body(...)):
    """
    Classify product to KTRU code

    Args:
        product: Product information

    Returns:
        Classification result with KTRU code and confidence
    """
    try:
        classifier = get_classifier_service()
        result = classifier.classify(product)

        logger.info(
            f"Classification completed: {result.status.value} - "
            f"Code: {result.code} - Confidence: {result.confidence:.2f}"
        )

        return result

    except Exception as e:
        logger.error(f"Classification endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_ktru(request: SearchRequest = Body(...)):
    """
    Search KTRU codes by query

    Args:
        request: Search request with query and parameters

    Returns:
        List of matching KTRU codes
    """
    try:
        classifier = get_classifier_service()
        results = classifier.search_ktru(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )

        return {"results": results, "count": len(results)}

    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ktru/{code}")
async def get_ktru_code(code: str):
    """
    Get KTRU code information

    Args:
        code: KTRU code

    Returns:
        KTRU code information
    """
    try:
        vector_store = get_vector_store()
        ktru_info = vector_store.get_by_code(code)

        if not ktru_info:
            raise HTTPException(status_code=404, detail="KTRU code not found")

        return ktru_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get KTRU code error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index")
async def index_ktru_codes(request: IndexingRequest = Body(...)):
    """
    Index new KTRU codes (admin endpoint)

    Args:
        request: Indexing request with KTRU codes

    Returns:
        Indexing result
    """
    try:
        embedding_service = get_embedding_service()
        vector_store = get_vector_store()

        # Convert to dictionaries
        codes_data = [code.model_dump() for code in request.codes]

        # Create embeddings
        texts = []
        for code in codes_data:
            text = f"{code['code']} {code['name']} {code.get('description', '')}"
            texts.append(text)

        embeddings = embedding_service.encode(texts, show_progress=True)

        # Add to vector store
        count = vector_store.add_ktru_codes(codes_data, embeddings)

        return {
            "indexed": count,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_statistics():
    """Get system statistics"""
    try:
        vector_store = get_vector_store()
        info = vector_store.get_collection_info()

        return {
            "ktru_codes_count": info["points_count"],
            "collection_status": info["status"],
            "vector_dimension": settings.embedding_dimension,
            "embedding_model": settings.embedding_model,
            "llm_model": f"{settings.llm_provider}/{settings.llm_model}"
        }

    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))