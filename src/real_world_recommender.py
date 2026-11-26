import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import load_groceries_csv, build_candidate_stream
from frequency_estimators import CountMinSketch, CountSketch
from membership_filters import CuckooFilter

class RetailBasketRecommender:
    def __init__(self, memory_kb: int = 256, estimator: str = "cms", min_support: float = 0.01, min_confidence: float = 0.30):
        self.memory_bytes = memory_kb * 1024
        self.estimator_name = estimator
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rules_df = pd.DataFrame()
        self.n_txns = 0
        self.sketch_mem_kb = 0.0

    def fit(self, data: dict, use_two_pass_recount: bool = True, seed: int = 0):
        self.n_txns = data["n_txns"]
        stream = data["stream"]
        present_keys = data["present_keys"]
        true_freqs = data["true_freqs"]
        key_to_itemset = data["key_to_itemset"]

        self.gatekeeper = CuckooFilter(capacity=len(present_keys), bucket_size=4, fp_bits=12, seed=seed)
        self.gatekeeper.insert_batch(present_keys)

        if self.estimator_name == "cms":
            self.sketch = CountMinSketch(self.memory_bytes, d=5, seed=seed)
        else:
            self.sketch = CountSketch(self.memory_bytes, d=5, seed=seed)

        self.sketch.update_batch(stream)
        self.sketch_mem_kb = self.sketch.memory_bytes() / 1024.0
        est_counts = self.sketch.query_batch(present_keys)

        min_cnt = self.min_support * self.n_txns
        survivor_indices = np.where(est_counts >= min_cnt)[0]

        freq_map = {}
        for idx in survivor_indices:
            k = int(present_keys[idx])
            itemset = key_to_itemset[k]
            cnt = float(true_freqs[idx]) if use_two_pass_recount else float(est_counts[idx])
            if cnt >= min_cnt:
                freq_map[itemset] = cnt

        rules = []
        for itemset, sup_cnt in freq_map.items():
            if len(itemset) < 2:
                continue
            sup_xy = sup_cnt / self.n_txns
            for i in range(len(itemset)):
                consequent = (itemset[i],)
                antecedent = tuple(x for j, x in enumerate(itemset) if j != i)
                if antecedent in freq_map and consequent in freq_map:
                    sup_x = freq_map[antecedent] / self.n_txns
                    sup_y = freq_map[consequent] / self.n_txns
                    conf = sup_xy / sup_x
                    lift = conf / sup_y if sup_y > 0 else 0.0
                    if conf >= self.min_confidence:
                        rules.append({
                            "Antecedent": " + ".join(antecedent),
                            "Consequent": consequent[0],
                            "Support_pct": round(sup_xy * 100, 2),
                            "Confidence_pct": round(conf * 100, 2),
                            "Lift": round(lift, 2),
                            "Estimated_Count": int(sup_cnt),
                        })

        self.rules_df = pd.DataFrame(rules).sort_values(by=["Lift", "Confidence_pct"], ascending=False).reset_index(drop=True)
        return {
            "total_candidates": len(present_keys),
            "sketch_survivors": len(survivor_indices),
            "pruned_pct": round((1 - len(survivor_indices) / len(present_keys)) * 100, 2),
            "verified_frequent_itemsets": len(freq_map),
            "rules_generated": len(self.rules_df),
        }

    def recommend(self, cart_items: list, top_k: int = 5) -> pd.DataFrame:
        cart_set = set(cart_items)
        matched = []
        for _, row in self.rules_df.iterrows():
            ant_set = set(row["Antecedent"].split(" + "))
            if ant_set.issubset(cart_set) and row["Consequent"] not in cart_set:
                matched.append(row)
        if not matched:
            return pd.DataFrame(columns=["Consequent", "Antecedent", "Confidence_pct", "Lift", "Support_pct"])
        df_m = pd.DataFrame(matched).sort_values(by=["Lift", "Confidence_pct"], ascending=False)
        return df_m.drop_duplicates(subset=["Consequent"]).head(top_k)[
            ["Consequent", "Antecedent", "Confidence_pct", "Lift", "Support_pct"]
        ]

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "groceries.csv")
    txns = load_groceries_csv(data_path)
    data = build_candidate_stream(txns, max_len=3, build_absent=False)

    engine = RetailBasketRecommender(memory_kb=256, estimator="cms", min_support=0.01, min_confidence=0.30)
    stats = engine.fit(data, use_two_pass_recount=True)
    print("Production Pipeline Stats (256 KB CMS + 2-Pass Verification):", stats)
    print("\nTop 10 Association Rules Discovered:")
    print(engine.rules_df.head(10).to_string(index=False))

    sample_cart = ["citrus fruit", "other vegetables"]
    print(f"\nLive Cross-Sell Recommendations for Cart {sample_cart}:")
    print(engine.recommend(sample_cart).to_string(index=False))
