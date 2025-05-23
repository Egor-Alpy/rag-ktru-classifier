from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ClassificationStatus(str, Enum):
    SUCCESS = "success"
    NOT_FOUND = "not_found"
    LOW_CONFIDENCE = "low_confidence"
    ERROR = "error"


class ProductInfo(BaseModel):
    """Input product information for classification"""
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    article: Optional[str] = Field(None, description="Product article/SKU")
    brand: Optional[str] = Field(None, description="Brand name")
    category: Optional[str] = Field(None, description="Product category")
    attributes: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Блокнот Пластик-Металлик А7 80 листов синий в клетку на спирали",
                "description": "",
                "article": "445816",
                "brand": "Нет данных",
                "category": "Блокноты",
                "attributes": []
            }
        }


class KTRUCode(BaseModel):
    """KTRU code information"""
    code: str = Field(..., description="KTRU code in format XX.XX.XX.XXX-XXXXXXXX")
    name: str = Field(..., description="KTRU code name")
    description: Optional[str] = Field(None, description="KTRU code description")
    parent_code: Optional[str] = Field(None, description="Parent KTRU code")
    level: int = Field(..., description="Hierarchy level")
    okpd2_code: Optional[str] = Field(None, description="Related OKPD2 code")


class ClassificationCandidate(BaseModel):
    """Single classification candidate"""
    ktru_code: KTRUCode
    score: float = Field(..., ge=0, le=1, description="Similarity score")
    reasoning: Optional[str] = Field(None, description="LLM reasoning")


class ClassificationResult(BaseModel):
    """Classification result"""
    status: ClassificationStatus
    code: Optional[str] = Field(None, description="Selected KTRU code")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    candidates: List[ClassificationCandidate] = Field(default_factory=list)
    reasoning: Optional[str] = Field(None, description="Classification reasoning")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthCheck(BaseModel):
    """Health check response"""
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, bool] = Field(default_factory=dict)


class IndexingRequest(BaseModel):
    """Request for indexing new KTRU codes"""
    codes: List[KTRUCode]
    update_existing: bool = False


class SearchRequest(BaseModel):
    """Search request for KTRU codes"""
    query: str
    top_k: int = Field(10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None