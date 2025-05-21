from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, Field


class Attribute(BaseModel):
    attr_name: str
    attr_value: str


class SKUModel(BaseModel):
    """Модель данных для товара (SKU)"""
    title: str
    description: Optional[str] = None
    article: Optional[str] = None
    brand: Optional[str] = None
    country_of_origin: Optional[str] = None
    category: Optional[str] = None
    attributes: Optional[List[Attribute]] = []

    class Config:
        schema_extra = {
            "example": {
                "title": "Бумага туалетная \"Мягкий знак\" 1-слойная",
                "description": "Туалетная бумага «Мягкий знак» 1 сл, 1 рул, 100 % целлюлоза, 54 м.",
                "article": "100170",
                "brand": "Мягкий знак",
                "country_of_origin": "Россия",
                "category": "Туалетная бумага",
                "attributes": [
                    {"attr_name": "Тип", "attr_value": "Бытовая"},
                    {"attr_name": "Количество слоев", "attr_value": "1"}
                ]
            }
        }


class KTRUMatch(BaseModel):
    """Модель данных для найденного кода КТРУ"""
    ktru_code: str
    title: str
    description: Optional[str] = None
    score: float
    attributes: Optional[List[Dict[str, Any]]] = []


class ClassificationResult(BaseModel):
    """Результат классификации товара"""
    sku: SKUModel
    matched_ktru: Optional[KTRUMatch] = None
    alternative_matches: List[KTRUMatch] = []
    has_match: bool = False
    explanation: str