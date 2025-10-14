"""
Utility to identify and filter duplicate test items from digital twin evaluations.

Background:
-----------
In the 70/30 digital twin split, 16 test items are exact duplicates of training items
(same participant, same message, same ratings). This occurs because 22 participants
rated some messages multiple times, and random splitting placed duplicates in both
train and test sets.

This script provides:
1. Function to identify these 16 duplicate items
2. Function to filter results excluding duplicates
3. Statistics on impact

Usage:
------
from filter_duplicates import get_duplicate_signatures, filter_clean_results

# Get duplicate signatures
duplicates = get_duplicate_signatures()

# Filter results
clean_results = filter_clean_results(all_results, duplicates)
"""

import json
from typing import Set, Tuple, Dict, Any

def load_split_data():
    """Load train and test splits for 70/30 digital twin."""
    with open('data_splits/canonical/train_digital_twin_7030.json') as f:
        train = json.load(f)
    with open('data_splits/canonical/test_digital_twin_7030.json') as f:
        test = json.load(f)
    return train, test


def get_duplicate_signatures() -> Set[Tuple[str, str, Tuple]]:
    """
    Get signatures of the 16 duplicate items that appear in both train and test.

    Returns:
        Set of tuples: (response_id, message_lowercase, ratings_tuple)
    """
    train, test = load_split_data()

    # Create signatures for train items
    train_sigs = set()
    for item in train:
        if isinstance(item, dict):
            sig = (
                item['response_id'],
                item['input_message'].strip().lower(),
                tuple(sorted(item.get('ratings', {}).items()))
            )
            train_sigs.add(sig)

    # Find duplicates in test
    duplicate_sigs = set()
    for item in test:
        if isinstance(item, dict):
            sig = (
                item['response_id'],
                item['input_message'].strip().lower(),
                tuple(sorted(item.get('ratings', {}).items()))
            )
            if sig in train_sigs:
                duplicate_sigs.add(sig)

    return duplicate_sigs


def is_duplicate(result_item: Dict[str, Any], duplicate_sigs: Set[Tuple]) -> bool:
    """
    Check if a result item is one of the known duplicates.

    Args:
        result_item: Dictionary containing 'response_id', 'input_message', and 'ground_truth_*' fields
        duplicate_sigs: Set of duplicate signatures from get_duplicate_signatures()

    Returns:
        True if this item is a duplicate, False otherwise
    """
    if not isinstance(result_item, dict):
        return False

    response_id = result_item.get('response_id')
    message = result_item.get('input_message', '').strip().lower()

    # Reconstruct ratings from ground truth fields
    ratings = {}
    for domain in ['content', 'design', 'coping', 'quitting']:
        gt_field = f'ground_truth_{domain}'
        if gt_field in result_item:
            ratings[domain] = result_item[gt_field]

    if not ratings:
        return False

    sig = (response_id, message, tuple(sorted(ratings.items())))
    return sig in duplicate_sigs


def filter_clean_results(results: Dict[str, Dict], duplicate_sigs: Set[Tuple] = None) -> Dict[str, Dict]:
    """
    Filter out duplicate items from results dictionary.

    Args:
        results: Dictionary of result items (e.g., loaded from evaluation JSON)
        duplicate_sigs: Set of duplicate signatures (if None, will compute automatically)

    Returns:
        Dictionary with duplicates removed
    """
    if duplicate_sigs is None:
        duplicate_sigs = get_duplicate_signatures()

    clean_results = {}
    removed_count = 0

    for qid, result in results.items():
        if not is_duplicate(result, duplicate_sigs):
            clean_results[qid] = result
        else:
            removed_count += 1

    print(f"Filtered {removed_count} duplicate items from {len(results)} total items")
    print(f"Clean dataset: {len(clean_results)} items ({len(clean_results)/len(results)*100:.1f}% of original)")

    return clean_results


def get_duplicate_stats():
    """Print statistics about the duplicates."""
    duplicates = get_duplicate_signatures()

    print("=" * 80)
    print("DUPLICATE ITEMS STATISTICS")
    print("=" * 80)
    print(f"\nTotal duplicates found: {len(duplicates)}")
    print(f"Affected participants: {len(set(sig[0] for sig in duplicates))}")
    print("\nThese items appear in BOTH train and test for the 70/30 digital twin split.")
    print("Impact: ~5% of test set (16/323 items)")
    print("\nRecommendation: Exclude these items when computing metrics to avoid")
    print("                memorization bias in digital twin and hybrid models.")
    print("=" * 80)


if __name__ == '__main__':
    # Example usage
    get_duplicate_stats()

    # Show example duplicates
    duplicates = get_duplicate_signatures()
    print("\nExample duplicates (first 3):")
    for idx, (response_id, message, ratings) in enumerate(list(duplicates)[:3]):
        print(f"\n{idx + 1}. Participant: {response_id}")
        print(f"   Message: {message[:80]}...")
        print(f"   Ratings: {dict(ratings)}")
