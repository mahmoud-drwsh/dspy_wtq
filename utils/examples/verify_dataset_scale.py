#!/usr/bin/env python3
"""
Script to verify the actual scale of the WTQ dataset.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.dataset_loader import ensure_wtq_data
from utils.splits_loader import load_wtq_splits, get_split_summary

def verify_dataset_scale():
    print("=" * 60)
    print("Verifying WTQ Dataset Scale")
    print("=" * 60)

    # Ensure data is available
    print("\n1. Ensuring WTQ data is available...")
    data_dir = ensure_wtq_data()
    print(f"   Data directory: {data_dir}")

    # Get split summary using existing function
    print("\n2. Getting split summary using get_split_summary():")
    summary = get_split_summary(data_dir)
    total_from_summary = sum(summary.values())
    print(f"   Split summary: {summary}")
    print(f"   Total from summary: {total_from_summary:,}")

    # Load all splits to verify
    print("\n3. Loading all splits to verify counts:")
    splits = load_wtq_splits(data_dir)

    detailed_counts = {}
    for split_name, examples in splits.items():
        count = len(examples)
        detailed_counts[split_name] = count
        print(f"   {split_name}: {count:,} examples")

        # Show sample questions for verification
        if examples:
            print(f"     Sample question: {examples[0]['question']}")
            print(f"     Sample answer: {examples[0]['answers']}")

    total_manual = sum(detailed_counts.values())
    print(f"\n   Total (manual count): {total_manual:,}")

    # Compare counts
    print(f"\n4. Verification results:")
    print(f"   Summary total: {total_from_summary:,}")
    print(f"   Manual total: {total_manual:,}")
    print(f"   Match: {'✅' if total_from_summary == total_manual else '❌'}")

    # Calculate breakdown
    print(f"\n5. Dataset breakdown:")
    for split_name, count in detailed_counts.items():
        percentage = (count / total_manual) * 100 if total_manual > 0 else 0
        print(f"   {split_name}: {count:,} examples ({percentage:.1f}%)")

    print(f"\n   Grand total: {total_manual:,} examples")

    # Verify against known WTQ dataset sizes
    print(f"\n6. Comparison with known WTQ dataset:")
    expected_total = 22033  # The number from the README
    print(f"   Expected total: {expected_total:,}")
    print(f"   Actual total: {total_manual:,}")
    print(f"   Difference: {abs(expected_total - total_manual):,}")
    print(f"   Match: {'✅' if total_manual == expected_total else '❌'}")

    return detailed_counts, total_manual

if __name__ == "__main__":
    counts, total = verify_dataset_scale()