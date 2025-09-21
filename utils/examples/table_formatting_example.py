#!/usr/bin/env python3
"""
Example script demonstrating table formatting utilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.dataset_loader import ensure_wtq_data
from utils.splits_loader import load_wtq_test_questions_with_tables
from utils.table_utils import (
    format_table_token_efficient,
    serialize_table_for_prompt,
    human_table_preview
)

def main():
    print("=" * 60)
    print("Table Formatting Example")
    print("=" * 60)

    # Load a few examples
    print("\n1. Loading test examples...")
    data_dir = ensure_wtq_data()
    examples = load_wtq_test_questions_with_tables(data_dir, limit=3)

    for i, example in enumerate(examples):
        print(f"\n{'='*40}")
        print(f"Example {i+1}")
        print(f"{'='*40}")

        print(f"\nQuestion: {example['question']}")
        print(f"Expected Answers: {example['answers']}")
        print(f"Table Name: {example['table_name']}")

        table = example['table']

        # Human-readable preview
        print(f"\n2. Human-readable preview:")
        human_preview = human_table_preview(table, n=3)
        print(human_preview)

        # Traditional serialization
        print(f"\n3. Traditional serialization:")
        traditional = serialize_table_for_prompt(table, row_limit=5, col_limit=8)
        print(traditional)
        print(f"   Length: {len(traditional)} characters")

        # Token-efficient formatting
        print(f"\n4. Token-efficient formatting:")
        efficient = format_table_token_efficient(
            table,
            question=example['question'],
            delimiter="|",
            max_rows=10
        )
        print(efficient)
        print(f"   Length: {len(efficient)} characters")

        # Show savings
        savings = len(traditional) - len(efficient)
        savings_pct = (savings / len(traditional)) * 100
        print(f"\n   Token savings: {savings} characters ({savings_pct:.1f}%)")

    # Demonstrate question-based filtering
    print(f"\n{'='*40}")
    print("Question-based filtering example")
    print(f"{'='*40}")

    # Use a specific example to show filtering
    sample_table = {
        "header": ["Name", "Age", "City", "Country", "Year", "Population"],
        "rows": [
            ["New York", 394, "USA", "2020", "8336817"],
            ["Los Angeles", 379, "USA", "2020", "3979576"],
            ["Chicago", 271, "USA", "2020", "2693976"],
            ["Houston", 232, "USA", "2020", "2320268"],
            ["Phoenix", 168, "USA", "2020", "1680992"],
            ["Philadelphia", 158, "USA", "2020", "1584064"],
            ["San Antonio", 157, "USA", "2020", "1571158"],
            ["San Diego", 142, "USA", "2020", "1423851"],
            ["Dallas", 134, "USA", "2020", "1343573"],
            ["San Jose", 103, "USA", "2020", "1030119"],
            ["Austin", 98, "USA", "2020", "978908"],
            ["Jacksonville", 91, "USA", "2020", "911507"],
            ["Fort Worth", 90, "USA", "2020", "895008"],
            ["Columbus", 89, "USA", "2020", "892533"],
            ["Charlotte", 88, "USA", "2020", "885708"]
        ]
    }

    # Without question filtering
    print(f"\n5. Without question filtering:")
    no_filter = format_table_token_efficient(
        sample_table,
        question=None,
        delimiter="|",
        max_rows=5
    )
    print(no_filter)
    print(f"   Length: {len(no_filter)} characters")

    # With question filtering
    print(f"\n6. With question filtering (What cities have population over 1 million?):")
    with_filter = format_table_token_efficient(
        sample_table,
        question="What cities have population over 1 million?",
        delimiter="|",
        max_rows=5
    )
    print(with_filter)
    print(f"   Length: {len(with_filter)} characters")

    print("\n" + "=" * 60)
    print("Table formatting example completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()