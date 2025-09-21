#!/usr/bin/env python3
"""
Example script demonstrating evaluation utilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.dataset_loader import ensure_wtq_data
from utils.splits_loader import load_wtq_test_questions_with_tables
from utils.eval_utils import (
    denotation_accuracy,
    normalize_token,
    split_prediction
)

def main():
    print("=" * 60)
    print("Evaluation Utilities Example")
    print("=" * 60)

    # Load a few examples for demonstration
    print("\n1. Loading test examples...")
    data_dir = ensure_wtq_data()
    examples = load_wtq_test_questions_with_tables(data_dir, limit=5)

    print(f"Loaded {len(examples)} examples for evaluation demo")

    # Demonstrate token normalization
    print(f"\n2. Token normalization examples:")
    test_tokens = [
        "New York",
        "1,000.50",
        "$500",
        "75%",
        "  extra  spaces  ",
        '"quoted"',
        "mixed-CASE"
    ]

    for token in test_tokens:
        normalized = normalize_token(token)
        print(f"   '{token}' -> '{normalized}'")

    # Demonstrate prediction splitting
    print(f"\n3. Prediction splitting examples:")
    test_predictions = [
        "New York|NYC|Manhattan",
        "New York, NYC, Manhattan",
        "Single answer",
        "1,000|2,000|3,000",
        "yes|no|maybe"
    ]

    for pred in test_predictions:
        for gold_count in [1, 3]:
            split_result = split_prediction(pred, gold_count)
            print(f"   '{pred}' (gold_count={gold_count}) -> {split_result}")

    # Simulate predictions and compute accuracy
    print(f"\n4. Simulating predictions and computing accuracy:")

    # Create some test predictions (simulate model outputs)
    predictions = []
    gold_answers = []

    for i, example in enumerate(examples):
        # Simulate different prediction scenarios
        if i == 0:
            # Perfect match
            pred = example['answers'][0]
        elif i == 1:
            # Alternative correct answer
            pred = "some alternative"  # This will be wrong
        elif i == 2:
            # Multiple answers in pipe format
            pred = "|".join(example['answers'][:2])
        elif i == 3:
            # Wrong answer
            pred = "incorrect answer"
        else:
            # Partially correct
            pred = example['answers'][0] if example['answers'] else "no answer"

        predictions.append([pred])
        gold_answers.append(example['answers'])

        print(f"\n   Example {i+1}:")
        print(f"     Question: {example['question']}")
        print(f"     Gold: {example['answers']}")
        print(f"     Predicted: {pred}")

    # Compute accuracy
    print(f"\n5. Computing denotation accuracy:")
    accuracy = denotation_accuracy(gold_answers, predictions)
    print(f"   Accuracy: {accuracy:.2%} ({sum(1 for g, p in zip(gold_answers, predictions) if set(g) == set(p))}/{len(gold_answers)})")

    # Show detailed comparison
    print(f"\n6. Detailed comparison:")
    correct = 0
    for i, (gold, pred) in enumerate(zip(gold_answers, predictions)):
        gold_set = {normalize_token(g) for g in gold}
        pred_set = {normalize_token(p) for p in pred}
        is_correct = gold_set == pred_set
        if is_correct:
            correct += 1

        print(f"   Example {i+1}: {'✅' if is_correct else '❌'}")
        print(f"     Gold normalized: {sorted(gold_set)}")
        print(f"     Pred normalized: {sorted(pred_set)}")

    print(f"\n   Final accuracy: {correct}/{len(gold_answers)} ({correct/len(gold_answers):.2%})")

    # Test edge cases
    print(f"\n7. Edge case testing:")
    edge_cases = [
        {"gold": ["1,000"], "pred": ["1000"], "desc": "Number normalization"},
        {"gold": ["75%"], "pred": ["75"], "desc": "Percentage normalization"},
        {"gold": ["New York"], "pred": ["new york"], "desc": "Case normalization"},
        {"gold": ["  spaced  "], "pred": ["spaced"], "desc": "Space normalization"},
        {"gold": ['"quoted"'], "pred": ["quoted"], "desc": "Quote normalization"},
    ]

    for case in edge_cases:
        accuracy = denotation_accuracy([case["gold"]], [case["pred"]])
        print(f"   {case['desc']}: {'✅' if accuracy == 1.0 else '❌'} (accuracy: {accuracy:.2%})")

    print("\n" + "=" * 60)
    print("Evaluation utilities example completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()