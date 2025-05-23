#!/usr/bin/env python3
"""
Script to evaluate classification performance
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from loguru import logger
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.schemas import ProductInfo, ClassificationStatus
from services.classifier import get_classifier_service

# Test cases for evaluation
TEST_CASES = [
    {
        "product": {
            "title": "Блокнот на спирали А5 80 листов в клетку",
            "category": "Канцелярские товары",
            "description": "Блокнот на металлической спирали"
        },
        "expected_code": "17.23.13.191-00000001"
    },
    {
        "product": {
            "title": "Ручка шариковая синяя 0.7мм",
            "category": "Письменные принадлежности",
            "brand": "BIC"
        },
        "expected_code": "32.99.12.110-00000001"
    },
    {
        "product": {
            "title": "Батарейки алкалиновые AA 4шт",
            "category": "Элементы питания",
            "description": "Щелочные батарейки типа AA"
        },
        "expected_code": "27.20.23.110-00000001"
    },
    {
        "product": {
            "title": "Калькулятор настольный 12-разрядный",
            "category": "Вычислительная техника",
            "brand": "Citizen"
        },
        "expected_code": "28.23.12.110-00000001"
    },
    {
        "product": {
            "title": "Бумага офисная А4 80г/м2 500 листов",
            "category": "Бумажная продукция",
            "description": "Бумага для офисной техники"
        },
        "expected_code": "17.12.14.110-00000001"
    }
]


def evaluate_classification(
        test_cases: List[Dict[str, Any]],
        classifier_service
) -> Dict[str, Any]:
    """Evaluate classification performance"""
    predictions = []
    expected = []
    confidences = []
    processing_times = []

    logger.info(f"Evaluating {len(test_cases)} test cases...")

    for test_case in tqdm(test_cases):
        # Create product info
        product = ProductInfo(**test_case["product"])
        expected_code = test_case["expected_code"]

        # Classify
        result = classifier_service.classify(product)

        # Collect results
        predicted_code = result.code if result.status == ClassificationStatus.SUCCESS else None
        predictions.append(predicted_code)
        expected.append(expected_code)
        confidences.append(result.confidence)
        processing_times.append(result.processing_time)

        # Log individual result
        is_correct = predicted_code == expected_code
        logger.info(
            f"Product: {product.title[:50]}... | "
            f"Expected: {expected_code} | "
            f"Predicted: {predicted_code} | "
            f"Confidence: {result.confidence:.2f} | "
            f"Correct: {is_correct}"
        )

    # Calculate metrics
    accuracy = accuracy_score(expected, predictions)

    # For precision/recall, handle None predictions
    predictions_binary = [1 if p == e else 0 for p, e in zip(predictions, expected)]
    expected_binary = [1] * len(expected)

    precision, recall, f1, _ = precision_recall_fscore_support(
        expected_binary, predictions_binary, average='binary'
    )

    avg_confidence = sum(confidences) / len(confidences)
    avg_processing_time = sum(processing_times) / len(processing_times)

    results = {
        "total_cases": len(test_cases),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "avg_confidence": avg_confidence,
        "avg_processing_time": avg_processing_time,
        "predictions": list(zip(expected, predictions, confidences))
    }

    return results


def save_evaluation_results(results: Dict[str, Any], output_path: Path):
    """Save evaluation results"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output_path}")


def main():
    """Main evaluation function"""
    # Initialize classifier
    classifier_service = get_classifier_service()

    # Run evaluation
    results = evaluate_classification(TEST_CASES, classifier_service)

    # Print results
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Total test cases: {results['total_cases']}")
    logger.info(f"Accuracy: {results['accuracy']:.2%}")
    logger.info(f"Precision: {results['precision']:.2%}")
    logger.info(f"Recall: {results['recall']:.2%}")
    logger.info(f"F1 Score: {results['f1_score']:.2%}")
    logger.info(f"Average confidence: {results['avg_confidence']:.2f}")
    logger.info(f"Average processing time: {results['avg_processing_time']:.2f}s")

    # Save results
    output_path = Path("evaluation_results.json")
    save_evaluation_results(results, output_path)


if __name__ == "__main__":
    main()