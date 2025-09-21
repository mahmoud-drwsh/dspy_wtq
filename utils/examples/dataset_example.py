#!/usr/bin/env python3
"""
Example script demonstrating WTQ dataset loading utilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.dataset_loader import (
    ensure_wtq_data,
    is_wtq_data_available,
    get_wtq_data_path,
    get_wtq_root_dir
)
from utils.splits_loader import (
    load_wtq_test_questions_with_tables,
    get_split_summary,
    load_wtq_splits_with_tables
)

def main():
    print("=" * 60)
    print("WTQ Dataset Loading Example")
    print("=" * 60)

    # Check initial data availability
    print("\n1. Checking initial data availability:")
    initial_available = is_wtq_data_available()
    print(f"   Data initially available: {initial_available}")

    # Ensure data is available
    print("\n2. Ensuring WTQ data is available...")
    data_dir = ensure_wtq_data()
    print(f"   Data directory: {data_dir}")

    # Check root directory
    root_dir = get_wtq_root_dir(data_dir)
    print(f"   Root directory: {root_dir}")

    # Get split summary
    print("\n3. Getting split summary:")
    summary = get_split_summary(data_dir)
    print(f"   Available splits: {summary}")

    # Load test examples with tables
    print("\n4. Loading test examples with tables (limit=3):")
    test_examples = load_wtq_test_questions_with_tables(data_dir, limit=3)
    print(f"   Loaded {len(test_examples)} test examples")

    for i, example in enumerate(test_examples):
        print(f"\n   Example {i+1}:")
        print(f"     ID: {example['id']}")
        print(f"     Question: {example['question']}")
        print(f"     Answers: {example['answers']}")
        print(f"     Table name: {example['table_name']}")

        # Show table preview
        table = example['table']
        print(f"     Table header: {table['header']}")
        print(f"     Table rows: {len(table['rows'])}")
        if table['rows']:
            print(f"     Sample row: {table['rows'][0]}")

    # Load all splits with tables (small limit)
    print("\n5. Loading all splits with tables (limit=2 each):")
    splits = load_wtq_splits_with_tables(
        data_dir=data_dir,
        limit={'train': 2, 'validation': 2, 'test': 2}
    )

    for split_name, examples in splits.items():
        print(f"   {split_name}: {len(examples)} examples")
        if examples:
            print(f"     Sample question: {examples[0]['question']}")

    print("\n" + "=" * 60)
    print("Dataset loading example completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()