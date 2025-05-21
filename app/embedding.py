from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Any


class EmbeddingModel:
    """Класс для работы с эмбеддингами на основе языковой модели"""

    def __init__(self, model_name: str = "ai-forever/ru-en-RoSBERTa"):
        """Инициализация модели для создания эмбеддингов"""
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_ktru_embedding(self, ktru_data: Dict[str, Any]) -> List[float]:
        """Создание эмбеддинга для записи КТРУ"""
        # Объединяем релевантные поля в один текст
        text_parts = [
            f"classification: {ktru_data['title']}",
            ktru_data.get('description', '')
        ]

        # Добавляем информацию об атрибутах
        for attr in ktru_data.get('attributes', []):
            attr_name = attr.get('attr_name', '')
            attr_values = []

            for val in attr.get('attr_values', []):
                value = val.get('value', '')
                unit = val.get('value_unit', '')
                if value and unit:
                    attr_values.append(f"{value} {unit}")
                elif value:
                    attr_values.append(value)

            if attr_name and attr_values:
                attr_text = f"{attr_name}: {', '.join(attr_values)}"
                text_parts.append(attr_text)

        # Объединяем все в единый текст
        full_text = " ".join([part for part in text_parts if part])

        # Создаем эмбеддинг
        embedding = self.model.encode(full_text)
        return embedding.tolist()

    def get_product_embedding(self, product_data: Dict[str, Any]) -> List[float]:
        """Создание эмбеддинга для товара"""
        # Объединяем релевантные поля в один текст
        text_parts = [
            f"classification: {product_data['title']}",
            product_data.get('description', '')
        ]

        # Добавляем информацию об атрибутах
        for attr in product_data.get('attributes', []):
            attr_name = attr.get('attr_name', '')
            attr_value = attr.get('attr_value', '')

            if attr_name and attr_value:
                attr_text = f"{attr_name}: {attr_value}"
                text_parts.append(attr_text)

        # Объединяем все в единый текст
        full_text = " ".join([part for part in text_parts if part])

        # Создаем эмбеддинг
        embedding = self.model.encode(full_text)
        return embedding.tolist()