import csv
import time
import random
from itertools import combinations
from collections import Counter
import numpy as np

def load_groceries_csv(filepath: str):
    transactions = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            items = sorted(set(item.strip() for item in row if item.strip()))
            if items:
                transactions.append(items)
    return transactions

def generate_synthetic_transactions(
    num_transactions: int = 50000,
    num_items: int = 500,
    avg_length: int = 6,
    distribution: str = "skewed",
    zipf_alpha: float = 1.15,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    item_universe = np.arange(1, num_items + 1)

    if distribution == "uniform":
        probs = np.full(num_items, 1.0 / num_items)
    elif distribution == "skewed":
        ranks = np.arange(1, num_items + 1, dtype=np.float64)
        weights = 1.0 / (ranks ** zipf_alpha)
        probs = weights / weights.sum()
    else:
        raise ValueError("distribution must be 'uniform' or 'skewed'")

    lengths = rng.poisson(lam=avg_length, size=num_transactions)
    lengths = np.clip(lengths, 1, min(num_items, 18))

    transactions = []
    for length in lengths:
        basket = rng.choice(item_universe, size=int(length), replace=False, p=probs)
        transactions.append(sorted(basket.tolist()))
    return transactions

def build_candidate_stream(transactions, max_len: int = 3, absent_seed: int = 42, build_absent: bool = True):
    unique_items = sorted({item for t in transactions for item in t})
    item_to_id = {item: idx + 1 for idx, item in enumerate(unique_items)}

    def encode(comb):
        key = 0
        for item in comb:
            key = (key << 16) | item_to_id[item]
        return key

    exact_counts = Counter()
    stream_keys = []
    key_to_itemset = {}
    updates_by_len = {k: 0 for k in range(1, max_len + 1)}
    unique_by_len = {k: set() for k in range(1, max_len + 1)}

    t0 = time.perf_counter()
    for t in transactions:
        for k in range(1, max_len + 1):
            for comb in combinations(t, k):
                key = encode(comb)
                exact_counts[key] += 1
                stream_keys.append(key)
                updates_by_len[k] += 1
                unique_by_len[k].add(key)
                if key not in key_to_itemset:
                    key_to_itemset[key] = comb
    build_time = time.perf_counter() - t0

    stream_arr = np.array(stream_keys, dtype=np.uint64)
    present_keys = np.array(list(exact_counts.keys()), dtype=np.uint64)
    true_freqs = np.array([exact_counts[k] for k in present_keys], dtype=np.float64)

    absent_keys = np.array([], dtype=np.uint64)
    if build_absent and len(unique_items) >= 3:
        rng = random.Random(absent_seed)
        present_set = set(exact_counts.keys())
        absent_set = set()
        target = len(present_keys)
        attempts = 0
        while len(absent_set) < target and attempts < target * 15:
            comb = tuple(sorted(rng.sample(unique_items, 3)))
            k = encode(comb)
            if k not in present_set:
                absent_set.add(k)
            attempts += 1
        absent_keys = np.array(list(absent_set), dtype=np.uint64)

    return {
        "n_txns": len(transactions),
        "n_items": len(unique_items),
        "item_to_id": item_to_id,
        "encode_fn": encode,
        "stream": stream_arr,
        "present_keys": present_keys,
        "absent_keys": absent_keys,
        "true_freqs": true_freqs,
        "exact_counts": exact_counts,
        "key_to_itemset": key_to_itemset,
        "updates_by_len": updates_by_len,
        "unique_by_len": {k: len(v) for k, v in unique_by_len.items()},
        "exact_build_time": build_time,
        "candidate_storage_bytes": len(present_keys) * 8,
        "exact_dict_min_bytes": len(present_keys) * 12,
    }
