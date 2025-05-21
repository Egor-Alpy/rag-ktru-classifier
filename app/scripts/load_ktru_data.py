import json
import argparse
import os
from tqdm import tqdm
from .app.embedding import EmbeddingModel
from .app.database import VectorDatabase


def load_ktru_data(file_path):
    """Загрузка данных КТРУ из файла"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def process_ktru_data(data_path, db_path, model_name, batch_size=100):
    """Обработка данных КТРУ и загрузка в векторную БД"""
    print(f"Загрузка данных КТРУ из {data_path}...")
    ktru_data = load_ktru_data(data_path)

    print(f"Инициализация модели эмбеддингов {model_name}...")
    # Инициализация векторной базы данных
    vector_db = VectorDatabase(
        persist_directory=db_path,
        embedding_model_name=model_name
    )

    # Загрузка данных в векторную БД
    print(f"Загрузка {len(ktru_data)} записей КТРУ в векторную базу данных...")
    vector_db.add_ktru_records(ktru_data, batch_size=batch_size)

    print("Загрузка данных КТРУ завершена успешно.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Загрузка данных КТРУ в векторную базу данных")
    parser.add_argument("--data-path", type=str, required=True, help="Путь к файлу с данными КТРУ")
    parser.add_argument("--db-path", type=str, default="./chroma_db", help="Путь для сохранения векторной БД")
    parser.add_argument("--model-name", type=str, default="ai-forever/ru-en-RoSBERTa",
                        help="Название модели эмбеддингов")
    parser.add_argument("--batch-size", type=int, default=100, help="Размер батча для обработки")

    args = parser.parse_args()

    process_ktru_data(args.data_path, args.db_path, args.model_name, args.batch_size)