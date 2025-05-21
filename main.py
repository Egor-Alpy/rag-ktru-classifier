import argparse
import multiprocessing
import time
import os
import sys

from db_sync import run_sync_job
from indexer import run_indexer_job
from api import start_api
from logging_config import setup_logging

logger = setup_logging("main")


def parse_args():
    """Парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(description="KTRU Product Classification System")
    parser.add_argument("--skip-sync", action="store_true", help="Skip MongoDB synchronization")
    parser.add_argument("--skip-index", action="store_true", help="Skip document indexing")
    parser.add_argument("--skip-api", action="store_true", help="Skip API server")
    return parser.parse_args()


def main():
    """
    Основная функция для запуска системы классификации товаров по КТРУ.
    """
    args = parse_args()

    processes = []

    # Запуск синхронизации с MongoDB
    if not args.skip_sync:
        logger.info("Starting MongoDB synchronization process")
        sync_process = multiprocessing.Process(target=run_sync_job)
        sync_process.start()
        processes.append(sync_process)

    # Запуск индексации документов
    if not args.skip_index:
        logger.info("Starting document indexing process")
        index_process = multiprocessing.Process(target=run_indexer_job)
        index_process.start()
        processes.append(index_process)

    # Запуск API сервера
    if not args.skip_api:
        logger.info("Starting API server")
        api_process = multiprocessing.Process(target=start_api)
        api_process.start()
        processes.append(api_process)

    # Ожидание завершения всех процессов
    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        logger.info("Stopping all processes...")
        for process in processes:
            process.terminate()

        # Ожидание завершения процессов
        for process in processes:
            process.join()

        logger.info("All processes stopped")


if __name__ == "__main__":
    main()