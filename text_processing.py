from typing import List, Dict, Any, Tuple
import re
import json
from config import CHUNK_SIZE, CHUNK_OVERLAP
from logging_config import setup_logging

logger = setup_logging("text_processing")


def clean_text(text: str) -> str:
    """Очистка текста от лишних символов."""
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    # Удаление HTML тегов
    text = re.sub(r'<[^>]+>', '', text)
    return text


def create_document_chunks(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Разбивает документ КТРУ на чанки для индексации.
    Возвращает список словарей, каждый из которых содержит текст чанка и метаданные.
    """
    chunks = []

    # Создаем полный текст документа
    doc_text = []

    # Добавляем основные поля
    doc_text.append(f"Код КТРУ: {document.get('ktru_code', '')}")
    doc_text.append(f"Название: {document.get('title', '')}")

    # Добавляем ключевые слова, если есть
    keywords = document.get('keywords', [])
    if keywords:
        doc_text.append(f"Ключевые слова: {', '.join(keywords)}")

    # Добавляем описание, если есть
    description = document.get('description', '')
    if description:
        doc_text.append(f"Описание: {description}")

    # Добавляем единицу измерения
    unit = document.get('unit', '')
    if unit:
        doc_text.append(f"Единица измерения: {unit}")

    # Добавляем атрибуты, если есть
    attributes = document.get('attributes', [])
    if attributes:
        doc_text.append("Атрибуты:")
        for attr in attributes:
            attr_name = attr.get('attr_name', '')
            attr_values = []

            for val in attr.get('attr_values', []):
                value = val.get('value', '')
                unit = val.get('value_unit', '')
                if unit:
                    attr_values.append(f"{value} {unit}")
                else:
                    attr_values.append(value)

            if attr_values:
                doc_text.append(f"  {attr_name}: {', '.join(attr_values)}")

    # Объединяем всё в один текст
    full_text = "\n".join(doc_text)
    full_text = clean_text(full_text)

    # Разбиваем на чанки с перекрытием
    text_length = len(full_text)

    if text_length <= CHUNK_SIZE:
        # Если текст меньше размера чанка, сохраняем его целиком
        chunks.append({
            "text": full_text,
            "ktru_code": document.get('ktru_code', ''),
            "title": document.get('title', ''),
            "chunk_id": 0
        })
    else:
        # Разбиваем на чанки с перекрытием
        for i in range(0, text_length, CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_text = full_text[i:min(i + CHUNK_SIZE, text_length)]
            if len(chunk_text) < 50:  # Пропускаем слишком маленькие чанки
                continue

            chunks.append({
                "text": chunk_text,
                "ktru_code": document.get('ktru_code', ''),
                "title": document.get('title', ''),
                "chunk_id": len(chunks)
            })

    logger.debug(f"Created {len(chunks)} chunks for document {document.get('ktru_code', '')}")
    return chunks


def format_product_text(product: Dict[str, Any]) -> str:
    """
    Форматирует информацию о товаре для передачи в промпт.
    """
    product_text = []

    # Добавляем название
    product_text.append(f"Название: {product.get('title', '')}")

    # Добавляем описание, если есть
    description = product.get('description', '')
    if description:
        product_text.append(f"Описание: {description}")

    # Добавляем артикул, если есть
    article = product.get('article', '')
    if article:
        product_text.append(f"Артикул: {article}")

    # Добавляем бренд, если есть
    brand = product.get('brand', '')
    if brand:
        product_text.append(f"Бренд: {brand}")

    # Добавляем страну происхождения, если есть
    country = product.get('country_of_origin', '')
    if country and country != "Нет данных":
        product_text.append(f"Страна происхождения: {country}")

    # Добавляем гарантию, если есть
    warranty = product.get('warranty_months', '')
    if warranty and warranty != "Нет данных":
        product_text.append(f"Гарантия: {warranty}")

    # Добавляем категорию, если есть
    category = product.get('category', '')
    if category:
        product_text.append(f"Категория: {category}")

    # Добавляем атрибуты, если есть
    attributes = product.get('attributes', [])
    if attributes:
        product_text.append("Атрибуты:")
        for attr in attributes:
            name = attr.get('attr_name', '')
            value = attr.get('attr_value', '')
            if name and value:
                product_text.append(f"  {name}: {value}")

    # Добавляем информацию о поставщиках, если есть
    suppliers = product.get('suppliers', [])
    if suppliers:
        product_text.append("Поставщики:")
        for supplier in suppliers:
            supplier_name = supplier.get('supplier_name', '')
            if supplier_name:
                product_text.append(f"  Название: {supplier_name}")

    return "\n".join(product_text)