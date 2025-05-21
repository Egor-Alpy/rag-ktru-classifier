import logging
import sys
from config import LOG_LEVEL


def setup_logging(name):
    """Настраивает логирование с указанным именем."""
    logger = logging.getLogger(name)

    # Установка уровня логирования
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Настройка обработчика для вывода в консоль
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger