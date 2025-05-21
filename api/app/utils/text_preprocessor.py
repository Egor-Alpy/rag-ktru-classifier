### api/app/utils/text_preprocessor.py
import re
from typing import List


def preprocess_text(text: str) -> str:
    """Предобработка текста для улучшения качества эмбеддингов"""
    if not text:
        return ""

    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text)

    # Удаление специальных символов
    text = re.sub(r'[^\w\s\-.,;:()№«»"]', ' ', text)

    # Приведение к нижнему регистру
    text = text.lower().strip()

    return text


def split_into_chunks(text: str, max_chunk_size: int = 1000) -> List[str]:
    """Разбивает длинный текст на чанки"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        # +1 для учета пробела
        if current_length + len(word) + 1 > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks