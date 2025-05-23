"""
Скрипт для тестирования RAG KTRU классификатора
"""

import time
import logging
import json
from typing import List, Dict, Tuple
import requests
from tabulate import tabulate

from config import API_HOST, API_PORT
from classifier import classifier
from vector_db import vector_db

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemTester:
    """Класс для тестирования системы"""

    def __init__(self):
        self.test_results = []
        self.api_url = f"http://{API_HOST}:{API_PORT}"

    def create_test_cases(self) -> List[Dict]:
        """Создание тестовых случаев"""
        return [
            # Компьютерная техника
            {
                "category": "Компьютеры",
                "title": "Ноутбук ASUS X515EA-BQ1189",
                "data": {
                    "title": "Ноутбук ASUS X515EA-BQ1189",
                    "description": "15.6 дюймов, Intel Core i3-1115G4, 8 ГБ DDR4, 256 ГБ SSD",
                    "category": "Ноутбуки",
                    "brand": "ASUS",
                    "attributes": [
                        {"attr_name": "Процессор", "attr_value": "Intel Core i3-1115G4"},
                        {"attr_name": "ОЗУ", "attr_value": "8 ГБ"},
                        {"attr_name": "Накопитель", "attr_value": "256 ГБ SSD"},
                        {"attr_name": "Диагональ экрана", "attr_value": "15.6 дюймов"}
                    ]
                },
                "expected_category": "26.20"
            },
            {
                "category": "Компьютеры",
                "title": "Компьютер Dell OptiPlex 3090",
                "data": {
                    "title": "Компьютер Dell OptiPlex 3090 SFF",
                    "description": "Системный блок, Intel Core i5-10505, 8GB, 256GB SSD",
                    "category": "Системные блоки",
                    "brand": "Dell",
                    "attributes": [
                        {"attr_name": "Форм-фактор", "attr_value": "SFF"},
                        {"attr_name": "Процессор", "attr_value": "Intel Core i5-10505"}
                    ]
                },
                "expected_category": "26.20"
            },

            # Канцелярские товары
            {
                "category": "Канцелярия",
                "title": "Ручка шариковая BIC Round Stic",
                "data": {
                    "title": "Ручка шариковая BIC Round Stic синяя",
                    "description": "Одноразовая шариковая ручка с прозрачным корпусом",
                    "category": "Письменные принадлежности",
                    "brand": "BIC",
                    "attributes": [
                        {"attr_name": "Цвет чернил", "attr_value": "Синий"},
                        {"attr_name": "Толщина линии", "attr_value": "0.7 мм"},
                        {"attr_name": "Тип", "attr_value": "Шариковая"}
                    ]
                },
                "expected_category": "32.99"
            },
            {
                "category": "Канцелярия",
                "title": "Карандаши чернографитные Koh-i-Noor",
                "data": {
                    "title": "Карандаши чернографитные Koh-i-Noor 1500 HB",
                    "description": "Набор простых карандашей для черчения и рисования",
                    "category": "Карандаши",
                    "brand": "Koh-i-Noor",
                    "attributes": [
                        {"attr_name": "Твердость", "attr_value": "HB"},
                        {"attr_name": "Количество в упаковке", "attr_value": "12 шт"}
                    ]
                },
                "expected_category": "32.99"
            },

            # Бумажная продукция
            {
                "category": "Бумага",
                "title": "Бумага SvetoCopy A4",
                "data": {
                    "title": "Бумага офисная SvetoCopy A4 80 г/м2 500 листов",
                    "description": "Белая бумага для копирования и печати",
                    "category": "Бумага офисная",
                    "brand": "SvetoCopy",
                    "attributes": [
                        {"attr_name": "Формат", "attr_value": "A4"},
                        {"attr_name": "Плотность", "attr_value": "80 г/м2"},
                        {"attr_name": "Белизна", "attr_value": "146%"},
                        {"attr_name": "Количество листов", "attr_value": "500"}
                    ]
                },
                "expected_category": "17.12"
            },

            # Мебель
            {
                "category": "Мебель",
                "title": "Стол письменный IKEA MICKE",
                "data": {
                    "title": "Стол письменный IKEA MICKE 105x50 см",
                    "description": "Компактный письменный стол с выдвижным ящиком",
                    "category": "Столы офисные",
                    "brand": "IKEA",
                    "attributes": [
                        {"attr_name": "Ширина", "attr_value": "105 см"},
                        {"attr_name": "Глубина", "attr_value": "50 см"},
                        {"attr_name": "Материал", "attr_value": "ЛДСП"},
                        {"attr_name": "Цвет", "attr_value": "Белый"}
                    ]
                },
                "expected_category": "31.01"
            },

            # Принтеры
            {
                "category": "Оргтехника",
                "title": "Принтер HP LaserJet Pro M15w",
                "data": {
                    "title": "Принтер лазерный HP LaserJet Pro M15w",
                    "description": "Компактный лазерный принтер с Wi-Fi",
                    "category": "Принтеры",
                    "brand": "HP",
                    "attributes": [
                        {"attr_name": "Технология печати", "attr_value": "Лазерная"},
                        {"attr_name": "Цветность", "attr_value": "Черно-белая"},
                        {"attr_name": "Формат", "attr_value": "A4"},
                        {"attr_name": "Скорость печати", "attr_value": "18 стр/мин"}
                    ]
                },
                "expected_category": "26.20"
            }
        ]

    def test_direct_classification(self):
        """Тестирование прямой классификации"""
        logger.info("\n" + "=" * 60)
        logger.info("ТЕСТИРОВАНИЕ ПРЯМОЙ КЛАССИФИКАЦИИ")
        logger.info("=" * 60)

        test_cases = self.create_test_cases()
        results = []

        for test_case in test_cases:
            logger.info(f"\nТест: {test_case['title']}")

            start_time = time.time()

            try:
                # Классификация
                result = classifier.classify(test_case['data'])

                processing_time = time.time() - start_time

                # Проверка результата
                is_correct = result.ktru_code.startswith(test_case['expected_category'])

                test_result = {
                    'test_name': test_case['title'],
                    'category': test_case['category'],
                    'expected': test_case['expected_category'],
                    'actual_code': result.ktru_code,
                    'actual_title': result.ktru_title[:50] + '...' if result.ktru_title else '',
                    'confidence': result.confidence,
                    'method': result.method,
                    'correct': '✅' if is_correct else '❌',
                    'time': processing_time
                }

                results.append(test_result)

                logger.info(f"  Результат: {result.ktru_code}")
                logger.info(f"  Название: {result.ktru_title}")
                logger.info(f"  Уверенность: {result.confidence:.3f}")
                logger.info(f"  Метод: {result.method}")
                logger.info(f"  Время: {processing_time:.2f}с")
                logger.info(f"  Статус: {'✅ Корректно' if is_correct else '❌ Некорректно'}")

            except Exception as e:
                logger.error(f"  Ошибка: {e}")
                results.append({
                    'test_name': test_case['title'],
                    'category': test_case['category'],
                    'expected': test_case['expected_category'],
                    'actual_code': 'ОШИБКА',
                    'actual_title': str(e)[:50],
                    'confidence': 0.0,
                    'method': 'error',
                    'correct': '❌',
                    'time': time.time() - start_time
                })

        # Вывод таблицы результатов
        self._print_results_table(results)

        return results

    def test_api_classification(self):
        """Тестирование классификации через API"""
        logger.info("\n" + "=" * 60)
        logger.info("ТЕСТИРОВАНИЕ API КЛАССИФИКАЦИИ")
        logger.info("=" * 60)

        # Проверка доступности API
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code != 200:
                logger.error("API недоступен")
                return []
        except Exception as e:
            logger.error(f"Не удалось подключиться к API: {e}")
            return []

        test_cases = self.create_test_cases()
        results = []

        for test_case in test_cases:
            logger.info(f"\nТест API: {test_case['title']}")

            start_time = time.time()

            try:
                # Отправляем запрос
                response = requests.post(
                    f"{self.api_url}/classify",
                    json=test_case['data'],
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    # Проверка результата
                    is_correct = result['ktru_code'].startswith(test_case['expected_category'])

                    test_result = {
                        'test_name': test_case['title'],
                        'category': test_case['category'],
                        'expected': test_case['expected_category'],
                        'actual_code': result['ktru_code'],
                        'actual_title': result.get('ktru_title', '')[:50] + '...' if result.get('ktru_title') else '',
                        'confidence': result['confidence'],
                        'correct': '✅' if is_correct else '❌',
                        'api_time': result['processing_time'],
                        'total_time': time.time() - start_time
                    }

                    results.append(test_result)

                    logger.info(f"  Результат: {result['ktru_code']}")
                    logger.info(f"  Уверенность: {result['confidence']:.3f}")
                    logger.info(f"  Время API: {result['processing_time']:.2f}с")
                    logger.info(f"  Статус: {'✅ Корректно' if is_correct else '❌ Некорректно'}")

                else:
                    logger.error(f"  Ошибка API: {response.status_code}")

            except Exception as e:
                logger.error(f"  Ошибка: {e}")

        # Вывод таблицы результатов
        if results:
            headers = ['Тест', 'Категория', 'Ожидаемый', 'Полученный код', 'Уверенность', 'Статус', 'Время']
            table_data = [
                [
                    r['test_name'][:30],
                    r['category'],
                    r['expected'],
                    r['actual_code'],
                    f"{r['confidence']:.3f}",
                    r['correct'],
                    f"{r['api_time']:.2f}с"
                ]
                for r in results
            ]

            print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))

        return results

    def test_vector_search(self):
        """Тестирование векторного поиска"""
        logger.info("\n" + "=" * 60)
        logger.info("ТЕСТИРОВАНИЕ ВЕКТОРНОГО ПОИСКА")
        logger.info("=" * 60)

        test_queries = [
            "ноутбук для работы",
            "ручка для письма",
            "бумага для принтера а4",
            "стол офисный",
            "компьютер персональный"
        ]

        for query in test_queries:
            logger.info(f"\nПоиск: '{query}'")

            try:
                results = vector_db.search(query, top_k=5)

                logger.info(f"Найдено {len(results)} результатов:")
                for i, result in enumerate(results[:3], 1):
                    logger.info(
                        f"  {i}. {result['payload']['ktru_code']} | "
                        f"Score: {result['score']:.3f} | "
                        f"{result['payload']['title'][:50]}..."
                    )

            except Exception as e:
                logger.error(f"  Ошибка поиска: {e}")

    def _print_results_table(self, results: List[Dict]):
        """Вывод таблицы с результатами"""
        if not results:
            return

        # Подготовка данных для таблицы
        headers = ['Тест', 'Категория', 'Ожидаемый', 'Полученный код', 'Уверенность', 'Метод', 'Статус', 'Время']
        table_data = []

        for r in results:
            table_data.append([
                r['test_name'][:30],
                r['category'],
                r['expected'],
                r['actual_code'][:20],
                f"{r['confidence']:.3f}",
                r.get('method', 'N/A')[:15],
                r['correct'],
                f"{r['time']:.2f}с"
            ])

        print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))

        # Статистика
        total = len(results)
        correct = sum(1 for r in results if r['correct'] == '✅')
        accuracy = (correct / total * 100) if total > 0 else 0

        print(f"\nСтатистика:")
        print(f"  Всего тестов: {total}")
        print(f"  Успешных: {correct}")
        print(f"  Точность: {accuracy:.1f}%")

        # Статистика по категориям
        categories = {}
        for r in results:
            cat = r['category']
            if cat not in categories:
                categories[cat] = {'total': 0, 'correct': 0}
            categories[cat]['total'] += 1
            if r['correct'] == '✅':
                categories[cat]['correct'] += 1

        print("\nТочность по категориям:")
        for cat, stats in categories.items():
            cat_accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {cat}: {stats['correct']}/{stats['total']} ({cat_accuracy:.1f}%)")

    def run_all_tests(self):
        """Запуск всех тестов"""
        logger.info("🚀 ЗАПУСК ПОЛНОГО ТЕСТИРОВАНИЯ СИСТЕМЫ")

        # Проверка компонентов
        logger.info("\nПроверка компонентов:")

        # Векторная БД
        try:
            stats = vector_db.get_statistics()
            logger.info(f"✅ Векторная БД: {stats['total_vectors']:,} векторов")
        except:
            logger.error("❌ Векторная БД недоступна")
            return

        # Классификатор
        try:
            if classifier.llm:
                logger.info("✅ Классификатор: инициализирован (с LLM)")
            else:
                logger.info("✅ Классификатор: инициализирован (без LLM)")
        except:
            logger.error("❌ Классификатор не инициализирован")
            return

        # Запуск тестов
        all_results = []

        # 1. Прямая классификация
        direct_results = self.test_direct_classification()
        all_results.extend([('direct', r) for r in direct_results])

        # 2. API классификация
        api_results = self.test_api_classification()
        all_results.extend([('api', r) for r in api_results])

        # 3. Векторный поиск
        self.test_vector_search()

        # Итоговая статистика
        logger.info("\n" + "=" * 60)
        logger.info("ИТОГОВАЯ СТАТИСТИКА")
        logger.info("=" * 60)

        direct_correct = sum(1 for t, r in all_results if t == 'direct' and r['correct'] == '✅')
        direct_total = sum(1 for t, r in all_results if t == 'direct')

        api_correct = sum(1 for t, r in all_results if t == 'api' and r['correct'] == '✅')
        api_total = sum(1 for t, r in all_results if t == 'api')

        if direct_total > 0:
            logger.info(
                f"Прямая классификация: {direct_correct}/{direct_total} ({direct_correct / direct_total * 100:.1f}%)")

        if api_total > 0:
            logger.info(f"API классификация: {api_correct}/{api_total} ({api_correct / api_total * 100:.1f}%)")

        total_correct = direct_correct + api_correct
        total_tests = direct_total + api_total

        if total_tests > 0:
            overall_accuracy = total_correct / total_tests * 100
            logger.info(f"\nОбщая точность: {total_correct}/{total_tests} ({overall_accuracy:.1f}%)")

            if overall_accuracy >= 85:
                logger.info("🎉 ЦЕЛЬ ДОСТИГНУТА! Точность >= 85%")
            else:
                logger.info(f"⚠️ До цели не хватает {85 - overall_accuracy:.1f}%")


def main():
    """Основная функция"""
    tester = SystemTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()