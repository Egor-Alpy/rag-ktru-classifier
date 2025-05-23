#!/usr/bin/env python3
"""
Script to download and parse KTRU data from official sources
"""

import json
import csv
import requests
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
import pandas as pd
from tqdm import tqdm

# Sample KTRU data structure (in real implementation, download from official source)
SAMPLE_KTRU_DATA = [
    {
        "code": "01.11.11.111-00000001",
        "name": "Пшеница мягкая",
        "description": "Пшеница мягкая озимая или яровая",
        "parent_code": "01.11.11.111",
        "level": 4,
        "okpd2_code": "01.11.11.111"
    },
    {
        "code": "01.11.11.111-00000002",
        "name": "Пшеница твердая",
        "description": "Пшеница твердая (дурум)",
        "parent_code": "01.11.11.111",
        "level": 4,
        "okpd2_code": "01.11.11.111"
    },
    {
        "code": "17.12.14.110-00000001",
        "name": "Бумага для печати",
        "description": "Бумага для печати офсетная",
        "parent_code": "17.12.14.110",
        "level": 4,
        "okpd2_code": "17.12.14.110"
    },
    {
        "code": "17.12.14.119-00000001",
        "name": "Бумага писчая",
        "description": "Бумага писчая и тетрадная",
        "parent_code": "17.12.14.119",
        "level": 4,
        "okpd2_code": "17.12.14.119"
    },
    {
        "code": "17.23.13.191-00000001",
        "name": "Блокноты",
        "description": "Блокноты на спирали",
        "parent_code": "17.23.13.191",
        "level": 4,
        "okpd2_code": "17.23.13.191"
    },
    {
        "code": "17.23.13.191-00000002",
        "name": "Блокноты на клею",
        "description": "Блокноты, скрепленные клеем",
        "parent_code": "17.23.13.191",
        "level": 4,
        "okpd2_code": "17.23.13.191"
    },
    {
        "code": "17.23.13.192-00000001",
        "name": "Тетради школьные",
        "description": "Тетради школьные 12-96 листов",
        "parent_code": "17.23.13.192",
        "level": 4,
        "okpd2_code": "17.23.13.192"
    },
    {
        "code": "17.23.13.193-00000001",
        "name": "Тетради общие",
        "description": "Тетради общие более 96 листов",
        "parent_code": "17.23.13.193",
        "level": 4,
        "okpd2_code": "17.23.13.193"
    },
    {
        "code": "20.41.31.110-00000001",
        "name": "Мыло туалетное твердое",
        "description": "Мыло туалетное в виде брусков",
        "parent_code": "20.41.31.110",
        "level": 4,
        "okpd2_code": "20.41.31.110"
    },
    {
        "code": "20.41.31.120-00000001",
        "name": "Мыло жидкое",
        "description": "Мыло жидкое во флаконах",
        "parent_code": "20.41.31.120",
        "level": 4,
        "okpd2_code": "20.41.31.120"
    },
    {
        "code": "27.20.23.110-00000001",
        "name": "Батарейки щелочные AA",
        "description": "Батарейки щелочные типоразмера AA (LR6)",
        "parent_code": "27.20.23.110",
        "level": 4,
        "okpd2_code": "27.20.23.110"
    },
    {
        "code": "27.20.23.110-00000002",
        "name": "Батарейки щелочные AAA",
        "description": "Батарейки щелочные типоразмера AAA (LR03)",
        "parent_code": "27.20.23.110",
        "level": 4,
        "okpd2_code": "27.20.23.110"
    },
    {
        "code": "27.20.23.120-00000001",
        "name": "Батарейки литиевые",
        "description": "Батарейки литиевые различных типоразмеров",
        "parent_code": "27.20.23.120",
        "level": 4,
        "okpd2_code": "27.20.23.120"
    },
    {
        "code": "28.23.12.110-00000001",
        "name": "Калькуляторы настольные",
        "description": "Калькуляторы настольные с питанием от сети/батареек",
        "parent_code": "28.23.12.110",
        "level": 4,
        "okpd2_code": "28.23.12.110"
    },
    {
        "code": "28.23.12.110-00000002",
        "name": "Калькуляторы карманные",
        "description": "Калькуляторы карманные на батарейках",
        "parent_code": "28.23.12.110",
        "level": 4,
        "okpd2_code": "28.23.12.110"
    },
    {
        "code": "32.99.12.110-00000001",
        "name": "Ручки шариковые",
        "description": "Ручки шариковые с синими чернилами",
        "parent_code": "32.99.12.110",
        "level": 4,
        "okpd2_code": "32.99.12.110"
    },
    {
        "code": "32.99.12.120-00000001",
        "name": "Ручки гелевые",
        "description": "Ручки гелевые различных цветов",
        "parent_code": "32.99.12.120",
        "level": 4,
        "okpd2_code": "32.99.12.120"
    },
    {
        "code": "32.99.13.110-00000001",
        "name": "Карандаши чернографитные",
        "description": "Карандаши чернографитные различной твердости",
        "parent_code": "32.99.13.110",
        "level": 4,
        "okpd2_code": "32.99.13.110"
    },
    {
        "code": "32.99.13.120-00000001",
        "name": "Карандаши цветные",
        "description": "Карандаши цветные наборы 6-36 цветов",
        "parent_code": "32.99.13.120",
        "level": 4,
        "okpd2_code": "32.99.13.120"
    },
    {
        "code": "32.99.15.110-00000001",
        "name": "Маркеры перманентные",
        "description": "Маркеры перманентные для различных поверхностей",
        "parent_code": "32.99.15.110",
        "level": 4,
        "okpd2_code": "32.99.15.110"
    }
]


def download_ktru_data(output_dir: Path) -> List[Dict[str, Any]]:
    """
    Download KTRU data from official source

    In production, this would:
    1. Download from https://zakupki.gov.ru or official API
    2. Parse XML/JSON/CSV format
    3. Extract all KTRU codes with full information
    """
    logger.info("Downloading KTRU data...")

    # For demo purposes, we'll use sample data
    # In production, implement actual download logic here

    ktru_data = SAMPLE_KTRU_DATA

    # Enrich with additional fields
    for item in ktru_data:
        # Add searchable text field
        item["search_text"] = f"{item['code']} {item['name']} {item['description']}"

        # Add category from code
        code_parts = item["code"].split(".")
        if len(code_parts) >= 2:
            item["category_code"] = f"{code_parts[0]}.{code_parts[1]}"
        else:
            item["category_code"] = code_parts[0]

    logger.info(f"Downloaded {len(ktru_data)} KTRU codes")
    return ktru_data


def save_ktru_data(ktru_data: List[Dict[str, Any]], output_dir: Path):
    """Save KTRU data to files"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON with the correct filename
    json_path = output_dir / "ktru_data.json"  # Изменено с ktru_codes.json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ktru_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to {json_path}")

    # Save as CSV
    csv_path = output_dir / "ktru_data.csv"  # Изменено с ktru_codes.csv
    df = pd.DataFrame(ktru_data)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"Saved to {csv_path}")


def main():
    """Main function"""
    from config import settings

    output_dir = settings.data_dir

    # Download KTRU data
    ktru_data = download_ktru_data(output_dir)

    # Save to files
    save_ktru_data(ktru_data, output_dir)

    # Print statistics
    logger.info("\nKTRU Data Statistics:")
    logger.info(f"Total codes: {len(ktru_data)}")

    # Group by level
    levels = {}
    for item in ktru_data:
        level = item.get("level", 0)
        levels[level] = levels.get(level, 0) + 1

    for level, count in sorted(levels.items()):
        logger.info(f"Level {level}: {count} codes")


if __name__ == "__main__":
    main()