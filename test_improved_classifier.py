#!/usr/bin/env python3
"""
Скрипт для тестирования улучшенного классификатора КТРУ
"""

import sys
import os
import time
import json
import logging
from typing import List, Dict
import requests

# Настройка путей
project_dir = "/workspace/rag-ktru-classifier"
if os.path.exists(project_dir) and os.getcwd() != project_dir:
    os.chdir(project_dir)
    sys.path.insert(0, project_dir)

# Импортируем улучшенный классификатор
from classifier_v2 import UtilityKtruClassifier
from config_v2 import API_HOST, API_PORT

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestRunner:
    def __init__(self):
        """Инициализация тестового раннера"""
        self.classifier = UtilityKtruClassifier()
        self.test_results = []

    def create_test_cases(self):
        """Создание тестовых кейсов для различных категорий товаров"""
        return [
            # Компьютерная техника
            {
                "category": "Компьютерная техника",
                "test_name": "Ноутбук ASUS",
                "data": {
                    "title": "Ноутбук ASUS X515EA",
                    "description": "Портативный компьютер для офисной работы",
                    "brand": "ASUS",
                    "attributes": [
                        {"attr_name": "Процессор", "attr_value": "Intel Core i5"},
                        {"attr_name": "Оперативная память", "attr_value": "8 ГБ"},
                        {"attr_name": "Диагональ экрана", "attr_value": "15.6 дюймов"}
                    ]
                },
                "expected_code_prefix": "26.20"
            },
            {
                "category": "Компьютерная техника",
                "test_name": "Компьютер персональный",
                "data": {
                    "title": "Компьютер персональный настольный",
                    "description": "Системный блок для офиса",
                    "attributes": [
                        {"attr_name": "Тип", "attr_value": "Настольный ПК"},
                        {"attr_name": "Процессор", "attr_value": "AMD Ryzen 5"}
                    ]
                },
                "expected_code_prefix": "26.20"
            },
            {
                "category": "Компьютерная техника",
                "test_name": "Монитор Dell",
                "data": {
                    "title": "Монитор Dell 24 дюйма",
                    "description": "Монитор для компьютера с разрешением Full HD",
                    "brand": "Dell",
                    "attributes": [
                        {"attr_name": "Диагональ", "attr_value": "24 дюйма"},
                        {"attr_name": "Разрешение", "attr_value": "1920x1080"}
                    ]
                },
                "expected_code_prefix": "26.20"
            },

            # Канцелярские товары
            {
                "category": "Канцелярские товары",
                "test_name": "Ручка шариковая",
                "data": {
                    "title": "Ручка шариковая синяя",
                    "description": "Канцелярская принадлежность для письма",
                    "attributes": [
                        {"attr_name": "Цвет чернил", "attr_value": "синий"},
                        {"attr_name": "Тип", "attr_value": "шариковая"}
                    ]
                },
                "expected_code_prefix": "32.99"
            },
            {
                "category": "Канцелярские товары",
                "test_name": "Карандаш чернографитный",
                "data": {
                    "title": "Карандаш чернографитный HB",
                    "description": "Простой карандаш для письма и рисования",
                    "attributes": [
                        {"attr_name": "Твердость", "attr_value": "HB"},
                        {"attr_name": "Материал корпуса", "attr_value": "дерево"}
                    ]
                },
                "expected_code_prefix": "32.99"
            },
            {
                "category": "Канцелярские товары",
                "test_name": "Степлер офисный",
                "data": {
                    "title": "Степлер металлический №24/6",
                    "description": "Степлер для скрепления документов",
                    "attributes": [
                        {"attr_name": "Размер скоб", "attr_value": "24/6"},
                        {"attr_name": "Материал", "attr_value": "металл"}
                    ]
                },
                "expected_code_prefix": "25.99"
            },

            # Бумажная продукция
            {
                "category": "Бумажная продукция",
                "test_name": "Бумага А4",
                "data": {
                    "title": "Бумага офисная А4 80 г/м2",
                    "description": "Бумага для печати и копирования",
                    "attributes": [
                        {"attr_name": "Формат", "attr_value": "А4"},
                        {"attr_name": "Плотность", "attr_value": "80 г/м2"},
                        {"attr_name": "Белизна", "attr_value": "146%"}
                    ]
                },
                "expected_code_prefix": "17.12"
            },
            {
                "category": "Бумажная продукция",
                "test_name": "Туалетная бумага",
                "data": {
                    "title": "Бумага туалетная \"Мягкий знак\" 1-слойная",
                    "description": "Туалетная бумага бытовая",
                    "brand": "Мягкий знак",
                    "attributes": [
                        {"attr_name": "Количество слоев", "attr_value": "1"},
                        {"attr_name": "Цвет", "attr_value": "белый"},
                        {"attr_name": "Тип", "attr_value": "бытовая"}
                    ]
                },
                "expected_code_prefix": "17.22"
            },

            # Мебель
            {
                "category": "Мебель",
                "test_name": "Стол письменный",
                "data": {
                    "title": "Стол письменный офисный",
                    "description": "Стол для рабочего места",
                    "attributes": [
                        {"attr_name": "Материал", "attr_value": "ЛДСП"},
                        {"attr_name": "Размер", "attr_value": "120x60 см"},
                        {"attr_name": "Назначение", "attr_value": "офисный"}
                    ]
                },
                "expected_code_prefix": "31.01"
            },
            {
                "category": "Мебель",
                "test_name": "Стул офисный",
                "data": {
                    "title": "Стул офисный эргономичный",
                    "description": "Кресло для офиса с подлокотниками",
                    "attributes": [
                        {"attr_name": "Тип", "attr_value": "офисное кресло"},
                        {"attr_name": "Материал обивки", "attr_value": "ткань"},
                        {"attr_name": "Регулировка высоты", "attr_value": "есть"}
                    ]
                },
                "expected_code_prefix": "31.01"
            },

            # Медицинские товары
            {
                "category": "Медицинские товары",
                "test_name": "Маска медицинская",
                "data": {
                    "title": "Маска медицинская одноразовая",
                    "description": "Маска трехслойная нестерильная",
                    "attributes": [
                        {"attr_name": "Количество слоев", "attr_value": "3"},
                        {"attr_name": "Тип", "attr_value": "одноразовая"},
                        {"attr_name": "Стерильность", "attr_value": "нестерильная"}
                    ]
                },
                "expected_code_prefix": "32.50"
            },

            # Принтеры и МФУ
            {
                "category": "Оргтехника",
                "test_name": "Принтер лазерный",
                "data": {
                    "title": "Принтер лазерный HP LaserJet",
                    "description": "Принтер для черно-белой печати",
                    "brand": "HP",
                    "attributes": [
                        {"attr_name": "Тип печати", "attr_value": "лазерная"},
                        {"attr_name": "Цветность", "attr_value": "черно-белая"},
                        {"attr_name": "Формат", "attr_value": "A4"}
                    ]
                },
                "expected_code_prefix": "26.20"
            }
        ]

    def run_test(self, test_case: Dict) -> Dict:
        """Запуск одного теста"""
        start_time = time.time()

        try:
            # Классификация через улучшенный классификатор
            result = self.classifier.classify_sku(test_case['data'])

            processing_time = time.time() - start_time

            # Проверка результата
            ktru_code = result.get('ktru_code', 'код не найден')
            ktru_title = result.get('ktru_title', '')
            confidence = result.get('confidence', 0.0)

            # Определение корректности
            is_correct = False
            if ktru_code != "код не найден":
                is_correct = ktru_code.startswith(test_case['expected_code_prefix'])

            test_result = {
                'test_name': test_case['test_name'],
                'category': test_case['category'],
                'input_title': test_case['data']['title'],
                'expected_prefix': test_case['expected_code_prefix'],
                'actual_code': ktru_code,
                'actual_title': ktru_title,
                'confidence': confidence,
                'is_correct': is_correct,
                'processing_time': processing_time,
                'error': None
            }

            return test_result

        except Exception as e:
            logger.error(f"Ошибка при тестировании: {e}")
            return {
                'test_name': test_case['test_name'],
                'category': test_case['category'],
                'input_title': test_case['data']['title'],
                'expected_prefix': test_case['expected_code_prefix'],
                'actual_code': 'ошибка',
                'actual_title': None,
                'confidence': 0.0,
                'is_correct': False,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }

    def run_all_tests(self):
        """Запуск всех тестов"""
        logger.info("🚀 Запуск тестирования улучшенного классификатора КТРУ")
        logger.info("=" * 80)

        test_cases = self.create_test_cases()
        self.test_results = []

        # Статистика по категориям
        category_stats = {}

        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n🧪 Тест {i}/{len(test_cases)}: {test_case['test_name']}")
            logger.info(f"   Категория: {test_case['category']}")
            logger.info(f"   Входные данные: {test_case['data']['title']}")

            result = self.run_test(test_case)
            self.test_results.append(result)

            # Обновляем статистику по категориям
            category = result['category']
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'correct': 0}
            category_stats[category]['total'] += 1
            if result['is_correct']:
                category_stats[category]['correct'] += 1

            # Вывод результата
            status_icon = "✅" if result['is_correct'] else "❌"
            logger.info(f"   {status_icon} Результат: {result['actual_code']}")
            if result['actual_title']:
                logger.info(f"   📋 Название КТРУ: {result['actual_title']}")
            logger.info(f"   🎯 Уверенность: {result['confidence']:.3f}")
            logger.info(f"   ⏱️  Время: {result['processing_time']:.2f}с")

            if result['error']:
                logger.error(f"   ⚠️  Ошибка: {result['error']}")

        # Итоговая статистика
        self._print_summary(category_stats)

        return self.test_results

    def _print_summary(self, category_stats: Dict):
        """Вывод итоговой статистики"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 ИТОГОВАЯ СТАТИСТИКА")
        logger.info("=" * 80)

        total_tests = sum(stats['total'] for stats in category_stats.values())
        total_correct = sum(stats['correct'] for stats in category_stats.values())
        overall_accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0

        # Статистика по категориям
        logger.info("\n📈 Точность по категориям:")
        for category, stats in sorted(category_stats.items()):
            accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            logger.info(f"   {category:.<30} {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")

        # Общая статистика
        logger.info(f"\n🎯 Общая точность: {total_correct}/{total_tests} ({overall_accuracy:.1f}%)")

        # Анализ ошибок
        errors = [r for r in self.test_results if not r['is_correct']]
        if errors:
            logger.info(f"\n❌ Неправильно классифицировано: {len(errors)}")
            for error in errors[:5]:  # Показываем первые 5 ошибок
                logger.info(f"   - {error['input_title']}")
                logger.info(f"     Ожидалось: {error['expected_prefix']}*, получено: {error['actual_code']}")

        # Производительность
        avg_time = sum(r['processing_time'] for r in self.test_results) / len(self.test_results)
        logger.info(f"\n⏱️  Среднее время классификации: {avg_time:.2f}с")

        # Вывод для достижения цели 95%+
        if overall_accuracy >= 95:
            logger.info("\n🎉 ЦЕЛЬ ДОСТИГНУТА! Точность классификации превышает 95%!")
        else:
            logger.info(f"\n⚠️  До цели 95% не хватает {95 - overall_accuracy:.1f}%")

    def test_api_endpoint(self):
        """Тест API эндпоинта с новым классификатором"""
        logger.info("\n🌐 Тестирование API эндпоинта")

        try:
            # Простой тест
            test_data = {
                "title": "Ноутбук Dell Inspiron 15",
                "description": "Портативный компьютер для работы и учебы",
                "attributes": [
                    {"attr_name": "Процессор", "attr_value": "Intel Core i7"},
                    {"attr_name": "ОЗУ", "attr_value": "16 ГБ"}
                ]
            }

            response = requests.post(
                f"http://{API_HOST}:{API_PORT}/classify",
                json=test_data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"   ✅ API работает корректно")
                logger.info(f"   Код: {result.get('ktru_code')}")
                logger.info(f"   Название: {result.get('ktru_title')}")
                logger.info(f"   Уверенность: {result.get('confidence')}")
            else:
                logger.error(f"   ❌ API вернул ошибку: {response.status_code}")

        except Exception as e:
            logger.error(f"   ❌ Ошибка при тестировании API: {e}")

    def save_results(self, filename="test_results.json"):
        """Сохранение результатов тестирования"""
        filepath = os.path.join(BASE_DIR, "logs", filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_tests': len(self.test_results),
                'correct': sum(1 for r in self.test_results if r['is_correct']),
                'accuracy': sum(1 for r in self.test_results if r['is_correct']) / len(self.test_results) * 100,
                'results': self.test_results
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"\n💾 Результаты сохранены в {filepath}")


def main():
    """Основная функция"""
    runner = TestRunner()

    # Запуск тестов
    results = runner.run_all_tests()

    # Тест API
    runner.test_api_endpoint()

    # Сохранение результатов
    runner.save_results()

    # Возврат кода завершения
    accuracy = sum(1 for r in results if r['is_correct']) / len(results) * 100
    return 0 if accuracy >= 95 else 1


if __name__ == "__main__":
    # Добавляем путь к проекту
    if os.path.exists("/workspace/rag-ktru-classifier"):
        BASE_DIR = "/workspace/rag-ktru-classifier"
    else:
        BASE_DIR = os.getcwd()

    exit_code = main()
    sys.exit(exit_code)