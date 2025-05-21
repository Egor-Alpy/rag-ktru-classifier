import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional


class VectorDatabase:
    """Интерфейс для работы с векторной базой данных Chroma"""

    def __init__(
            self,
            persist_directory: str = "./chroma_db",
            embedding_model_name: str = "ai-forever/ru-en-RoSBERTa",
            collection_name: str = "ktru_codes"
    ):
        """Инициализация клиента векторной базы данных"""
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        self.collection_name = collection_name

        # Создаем или получаем коллекцию
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Найдена существующая коллекция '{collection_name}' с {self.collection.count()} записями")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Создана новая коллекция '{collection_name}'")

    def add_ktru_records(self, records: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """Добавление записей КТРУ в векторную базу данных"""
        total_records = len(records)
        print(f"Добавление {total_records} записей КТРУ в базу данных...")

        # Обрабатываем по батчам для эффективности
        for i in range(0, total_records, batch_size):
            end_idx = min(i + batch_size, total_records)
            batch = records[i:end_idx]

            ids = [record["ktru_code"] for record in batch]
            documents = [f"{record['title']}. {record.get('description', '')}" for record in batch]
            embeddings = None  # Будет автоматически вычислено через embedding_function
            metadatas = []

            for record in batch:
                # Подготавливаем метаданные для фильтрации
                metadata = {
                    "ktru_code": record["ktru_code"],
                    "title": record["title"],
                    "unit": record.get("unit", ""),
                }

                # Атрибуты добавляем в плоском виде для возможности фильтрации
                for attr in record.get("attributes", []):
                    attr_name = attr.get("attr_name", "")
                    attr_values = []

                    for val in attr.get("attr_values", []):
                        value = val.get("value", "")
                        if value:
                            attr_values.append(value)

                    if attr_name and attr_values:
                        metadata[f"attr_{attr_name}"] = ", ".join(attr_values)

                metadatas.append(metadata)

            # Добавляем батч в коллекцию
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

            print(f"Добавлено {end_idx}/{total_records} записей...")

    def find_similar_ktru(
            self,
            product_embedding: List[float],
            filter_metadata: Optional[Dict] = None,
            top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Поиск похожих кодов КТРУ по эмбеддингу товара"""
        # Подготавливаем фильтр, если он передан
        where_filter = None
        if filter_metadata and isinstance(filter_metadata, dict):
            # Преобразуем фильтр в формат, который понимает Chroma
            where_filter = {}
            for key, value in filter_metadata.items():
                if isinstance(value, list):
                    # Для списков используем оператор $in
                    where_filter[key] = {"$in": value}
                else:
                    # Для одиночных значений используем прямое сравнение
                    where_filter[key] = value

        # Выполняем поиск
        results = self.collection.query(
            query_embeddings=[product_embedding],
            where=where_filter,
            n_results=top_k
        )

        # Форматируем результаты
        formatted_results = []
        if results and "ids" in results and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "ktru_code": results["ids"][0][i],
                    "document": results["documents"][0][i] if "documents" in results else "",
                    "metadata": results["metadatas"][0][i] if "metadatas" in results else {},
                    "distance": results["distances"][0][i] if "distances" in results else None
                })

        return formatted_results