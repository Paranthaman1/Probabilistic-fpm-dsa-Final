import csv
import math
import os
import time
from collections import Counter
from itertools import combinations
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Live Smart-Market PDS Streamer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #0f766e 100%);
        padding: 1.25rem 1.6rem;
        border-radius: 14px;
        color: #f8fafc;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
    }
    .main-banner h1 {
        margin: 0;
        font-size: 1.72rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
    }
    .main-banner p {
        margin: 0.35rem 0 0 0;
        font-size: 0.94rem;
        color: #cbd5e1;
    }
    .live-badge {
        display: inline-block;
        background: #ef4444;
        color: white;
        font-weight: 700;
        font-size: 0.72rem;
        padding: 2px 9px;
        border-radius: 999px;
        margin-right: 8px;
        vertical-align: middle;
        letter-spacing: 0.06em;
    }
    .paused-badge {
        display: inline-block;
        background: #64748b;
        color: white;
        font-weight: 700;
        font-size: 0.72rem;
        padding: 2px 9px;
        border-radius: 999px;
        margin-right: 8px;
        vertical-align: middle;
        letter-spacing: 0.06em;
    }
    .shelf-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #0d9488;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.06);
    }
    .shelf-card-warn {
        border-left-color: #f59e0b;
        background: #fffbeb;
    }
    .shelf-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }
    .shelf-sub {
        font-size: 0.84rem;
        color: #475569;
    }
    .pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        margin-right: 5px;
        margin-top: 4px;
    }
    .pill-lift { background: #dcfce7; color: #166534; }
    .pill-conf { background: #dbeafe; color: #1e40af; }
    .pill-supp { background: #f1f5f9; color: #334155; }
    .pill-fp   { background: #fee2e2; color: #991b1b; }
    .receipt-box {
        background: #0f172a;
        color: #e2e8f0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.80rem;
        padding: 0.85rem 1rem;
        border-radius: 12px;
        line-height: 1.55;
        border: 1px solid #334155;
    }
    .receipt-id { color: #38bdf8; font-weight: 700; }
    .receipt-items { color: #a7f3d0; }
    .kpi-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .kpi-label { font-size: 0.75rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .kpi-val { font-size: 1.45rem; color: #0f172a; font-weight: 800; margin-top: 2px; }
    .kpi-sub { font-size: 0.75rem; color: #0d9488; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

ITEM_EMOJIS = {
    "whole milk": "🥛",
    "other vegetables": "🥦",
    "rolls/buns": "🥐",
    "soda": "🥤",
    "yogurt": "🥣",
    "bottled water": "💧",
    "root vegetables": "🥕",
    "tropical fruit": "🍍",
    "shopping bags": "🛍️",
    "sausage": "🌭",
    "pastry": "🧁",
    "citrus fruit": "🍊",
    "bottled beer": "🍺",
    "newspapers": "📰",
    "canned beer": "🥫",
    "pip fruit": "🍎",
    "fruit/vegetable juice": "🧃",
    "whipped/sour cream": "🍦",
    "brown bread": "🍞",
    "domestic eggs": "🥚",
    "frankfurter": "🌭",
    "margarine": "🧈",
    "coffee": "☕",
    "pork": "🥩",
    "butter": "🧈",
    "curd": "🧀",
    "beef": "🥩",
    "napkins": "🧻",
    "chocolate": "🍫",
    "frozen vegetables": "❄️",
    "chicken": "🍗",
    "white bread": "🍞",
    "cream cheese": "🧀",
    "waffles": "🧇",
    "salty snack": "🥨",
    "long life bakery product": "🥖",
    "dessert": "🍮",
    "sugar": "🧂",
    "UHT-milk": "🥛",
    "berries": "🍓",
    "hamburger meat": "🍔",
    "hygiene articles": "🧼",
    "onions": "🧅",
    "specialty chocolate": "🍫",
    "candy": "🍬",
    "frozen meals": "🍱",
    "misc. beverages": "🥤",
    "oil": "🫒",
    "butter milk": "🥛",
    "specialty bar": "🍫",
    "ham": "🥓",
    "meat": "🥩",
    "ice cream": "🍨",
    "hard cheese": "🧀",
    "sliced cheese": "🧀",
    "cat food": "🐱",
    "grapes": "🍇",
}

def item_label(name: str) -> str:
    return f"{ITEM_EMOJIS.get(name, '🛒')} {name}"

def mix_hash_vec(keys: np.ndarray, seed: int) -> np.ndarray:
    h = (keys ^ np.uint64(seed)) * np.uint64(0xBF58476D1CE4E5B9)
    h = (h ^ (h >> np.uint64(30))) * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(31))

class CountMinSketch:

    def __init__(self, memory_bytes: int, d: int = 5, seed: int = 0):
        self.d = d
        self.w = max(1, memory_bytes // (4 * d))
        self.table = np.zeros((self.d, self.w), dtype=np.uint32)
        self.seeds = np.random.default_rng(seed).integers(1, 2**62, size=self.d, dtype=np.uint64)

    def update_batch(self, stream: np.ndarray):
        if len(stream) == 0:
            return
        for r in range(self.d):
            cols = (mix_hash_vec(stream, self.seeds[r]) % np.uint64(self.w)).astype(np.int64)
            self.table[r] += np.bincount(cols, minlength=self.w).astype(np.uint32)

    def query_batch(self, keys: np.ndarray) -> np.ndarray:
        if len(keys) == 0:
            return np.array([], dtype=np.float64)
        est = np.full(len(keys), np.iinfo(np.uint32).max, dtype=np.uint32)
        for r in range(self.d):
            cols = (mix_hash_vec(keys, self.seeds[r]) % np.uint64(self.w)).astype(np.int64)
            est = np.minimum(est, self.table[r, cols])
        return est.astype(np.float64)

    def memory_bytes(self) -> int:
        return int(self.table.nbytes)

class CountSketch:

    def __init__(self, memory_bytes: int, d: int = 5, seed: int = 0):
        self.d = d
        self.w = max(1, memory_bytes // (4 * d))
        self.table = np.zeros((self.d, self.w), dtype=np.int32)
        rng = np.random.default_rng(seed)
        self.col_seeds = rng.integers(1, 2**62, size=self.d, dtype=np.uint64)
        self.sign_seeds = rng.integers(1, 2**62, size=self.d, dtype=np.uint64)

    def update_batch(self, stream: np.ndarray):
        if len(stream) == 0:
            return
        for r in range(self.d):
            cols = (mix_hash_vec(stream, self.col_seeds[r]) % np.uint64(self.w)).astype(np.int64)
            signs = np.where((mix_hash_vec(stream, self.sign_seeds[r]) & np.uint64(1)) == 0, 1.0, -1.0)
            self.table[r] += np.bincount(cols, weights=signs, minlength=self.w).astype(np.int32)

    def query_batch(self, keys: np.ndarray) -> np.ndarray:
        if len(keys) == 0:
            return np.array([], dtype=np.float64)
        ests = np.zeros((self.d, len(keys)), dtype=np.int32)
        for r in range(self.d):
            cols = (mix_hash_vec(keys, self.col_seeds[r]) % np.uint64(self.w)).astype(np.int64)
            signs = np.where((mix_hash_vec(keys, self.sign_seeds[r]) & np.uint64(1)) == 0, 1, -1)
            ests[r] = self.table[r, cols] * signs
        return np.median(ests, axis=0)

    def memory_bytes(self) -> int:
        return int(self.table.nbytes)

class CountingBloomEstimator:

    def __init__(self, memory_bytes: int, k: int = 5, seed: int = 0):
        self.k = k
        self.m = max(1, memory_bytes // 4)
        self.counters = np.zeros(self.m, dtype=np.uint32)
        self.seeds = np.random.default_rng(seed).integers(1, 2**62, size=self.k, dtype=np.uint64)

    def update_batch(self, stream: np.ndarray):
        if len(stream) == 0:
            return
        for s in self.seeds:
            pos = (mix_hash_vec(stream, s) % np.uint64(self.m)).astype(np.int64)
            self.counters += np.bincount(pos, minlength=self.m).astype(np.uint32)

    def query_batch(self, keys: np.ndarray) -> np.ndarray:
        if len(keys) == 0:
            return np.array([], dtype=np.float64)
        est = np.full(len(keys), np.iinfo(np.uint32).max, dtype=np.uint32)
        for s in self.seeds:
            pos = (mix_hash_vec(keys, s) % np.uint64(self.m)).astype(np.int64)
            est = np.minimum(est, self.counters[pos])
        return est.astype(np.float64)

    def memory_bytes(self) -> int:
        return int(self.counters.nbytes)

class BloomGatekeeper:

    def __init__(self, n: int = 150000, p: float = 0.01, seed: int = 0):
        self.m = int(math.ceil(-(n * math.log(p)) / (math.log(2) ** 2)))
        self.k = max(1, int(round((self.m / n) * math.log(2))))
        self.bits = np.zeros((self.m + 7) // 8, dtype=np.uint8)
        self.seeds = np.random.default_rng(seed).integers(1, 2**62, size=self.k, dtype=np.uint64)

    def insert_batch(self, keys: np.ndarray):
        if len(keys) == 0:
            return
        for s in self.seeds:
            pos = mix_hash_vec(keys, s) % np.uint64(self.m)
            np.bitwise_or.at(
                self.bits,
                pos >> np.uint64(3),
                (np.uint8(1) << (pos & np.uint64(7)).astype(np.uint8)),
            )

    def memory_bytes(self) -> int:
        return int(self.bits.nbytes)

def find_groceries_path() -> Optional[str]:
    candidates = [
        "groceries.csv",
        os.path.join("data", "groceries.csv"),
        os.path.join("Probabilistic_FPM_DSA_Project", "data", "groceries.csv"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "groceries.csv"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "groceries.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

@st.cache_data(show_spinner="Preparing real-time POS transaction stream...")
def prepare_dataset(csv_path: Optional[str], uploaded_bytes: Optional[bytes], max_len: int = 3):
    transactions = []
    if uploaded_bytes is not None:
        lines = uploaded_bytes.decode("utf-8", errors="ignore").splitlines()
        reader = csv.reader(lines)
        for row in reader:
            items = sorted(set(x.strip() for x in row if x.strip()))
            if items:
                transactions.append(items)
    elif csv_path is not None:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                items = sorted(set(x.strip() for x in row if x.strip()))
                if items:
                    transactions.append(items)
    else:
        rng = np.random.default_rng(42)
        catalog = list(ITEM_EMOJIS.keys())
        ranks = np.arange(1, len(catalog) + 1, dtype=np.float64)
        probs = 1.0 / (ranks**1.15)
        probs /= probs.sum()
        lengths = np.clip(rng.poisson(4.5, size=5000), 1, 12)
        for length in lengths:
            basket = rng.choice(catalog, size=int(length), replace=False, p=probs)
            transactions.append(sorted(basket.tolist()))

    unique_items = sorted({item for t in transactions for item in t})
    item_to_id = {item: idx + 1 for idx, item in enumerate(unique_items)}
    id_to_item = {idx + 1: item for idx, item in enumerate(unique_items)}

    encoded_txns = []
    key_to_itemset = {}
    for t in transactions:
        ids = [item_to_id[x] for x in t]
        keys = []
        for k in range(1, max_len + 1):
            for comb in combinations(ids, k):
                key = 0
                for cid in comb:
                    key = (key << 16) | cid
                keys.append(key)
                if key not in key_to_itemset:
                    key_to_itemset[key] = tuple(id_to_item[cid] for cid in comb)
        encoded_txns.append(np.array(keys, dtype=np.uint64))

    return {
        "transactions": transactions,
        "encoded_txns": encoded_txns,
        "unique_items": unique_items,
        "item_to_id": item_to_id,
        "key_to_itemset": key_to_itemset,
        "source_name": (
            "Uploaded CSV"
            if uploaded_bytes
            else (os.path.basename(csv_path) if csv_path else "Synthetic Supermarket Stream")
        ),
    }

def make_sketch(algo: str, budget_kb: int, depth: int, seed: int):
    mem_bytes = budget_kb * 1024
    if algo == "Count-Min Sketch (CMS)":
        return CountMinSketch(mem_bytes, d=depth, seed=seed)
    elif algo == "Count Sketch (CS)":
        return CountSketch(mem_bytes, d=depth, seed=seed)
    else:
        return CountingBloomEstimator(mem_bytes, k=depth, seed=seed)

def init_stream_state(algo: str, budget_kb: int, depth: int, seed: int, max_len: int):
    st.session_state.cfg = (algo, budget_kb, depth, seed, max_len)
    st.session_state.sketch = make_sketch(algo, budget_kb, depth, seed)
    st.session_state.cms_cmp = CountMinSketch(budget_kb * 1024, d=depth, seed=seed)
    st.session_state.cs_cmp = CountSketch(budget_kb * 1024, d=depth, seed=seed)
    st.session_state.bloom_gate = BloomGatekeeper(n=150000, p=0.01, seed=seed)
    st.session_state.exact_counts = Counter()
    st.session_state.stream_cursor = 0
    st.session_state.total_txns_processed = 0
    st.session_state.total_updates = 0
    st.session_state.total_ingest_time = 0.0
    st.session_state.recent_receipts = []
    st.session_state.history = []
    st.session_state.is_running = False

def ingest_next_batch(dataset: dict, batch_size: int, theta: float, loop_stream: bool):
    txns = dataset["transactions"]
    encoded = dataset["encoded_txns"]
    n_total = len(txns)

    start = st.session_state.stream_cursor
    if start >= n_total:
        if loop_stream:
            start = 0
            st.session_state.stream_cursor = 0
        else:
            st.session_state.is_running = False
            return

    end = min(start + batch_size, n_total)
    batch_chunks = encoded[start:end]
    flat_stream = np.concatenate(batch_chunks) if batch_chunks else np.array([], dtype=np.uint64)

    t0 = time.perf_counter()
    st.session_state.sketch.update_batch(flat_stream)
    dt = time.perf_counter() - t0
    st.session_state.total_ingest_time += dt

    st.session_state.cms_cmp.update_batch(flat_stream)
    st.session_state.cs_cmp.update_batch(flat_stream)
    st.session_state.bloom_gate.insert_batch(flat_stream)

    st.session_state.exact_counts.update(flat_stream.tolist())

    batch_len = end - start
    st.session_state.stream_cursor = end
    st.session_state.total_txns_processed += batch_len
    st.session_state.total_updates += len(flat_stream)

    base_receipt_no = st.session_state.total_txns_processed - batch_len
    recent = []
    for offset, i in enumerate(range(max(start, end - 6), end)):
        receipt_num = base_receipt_no + (i - start) + 1
        recent.append((receipt_num, txns[i]))
    st.session_state.recent_receipts = recent[::-1]

    if end >= n_total and not loop_stream:
        st.session_state.is_running = False

    n_seen_txns = st.session_state.total_txns_processed
    seen_keys = np.fromiter(st.session_state.exact_counts.keys(), dtype=np.uint64)
    true_vals = np.fromiter(st.session_state.exact_counts.values(), dtype=np.float64)
    est_vals = st.session_state.sketch.query_batch(seen_keys)

    min_cnt = theta * n_seen_txns
    true_f = true_vals >= min_cnt
    est_f = est_vals >= min_cnt
    tp = int(np.sum(true_f & est_f))
    fp = int(np.sum((~true_f) & est_f))
    fn = int(np.sum(true_f & (~est_f)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    mae = float(np.mean(np.abs((est_vals - true_vals) / max(1, n_seen_txns))))

    v_prec = 1.0 if tp > 0 else 0.0
    v_f1 = (2 * v_prec * rec / (v_prec + rec)) if (v_prec + rec) > 0 else 0.0

    st.session_state.history.append({
        "txns": n_seen_txns,
        "updates": st.session_state.total_updates,
        "sketch_kb": st.session_state.sketch.memory_bytes() / 1024.0,
        "exact_kb": (len(seen_keys) * 12) / 1024.0,
        "precision": prec * 100.0,
        "recall": rec * 100.0,
        "f1": f1 * 100.0,
        "pass2_f1": v_f1 * 100.0,
        "mae": mae,
        "survivors": int(np.sum(est_f)),
        "true_freq": int(np.sum(true_f)),
    })

def mine_live_rules(dataset: dict, theta: float, min_conf: float, use_pass2: bool):
    n_txns = st.session_state.total_txns_processed
    if n_txns == 0 or len(st.session_state.exact_counts) == 0:
        return pd.DataFrame(), pd.DataFrame(), {}

    seen_keys = np.fromiter(st.session_state.exact_counts.keys(), dtype=np.uint64)
    true_vals = np.fromiter(st.session_state.exact_counts.values(), dtype=np.float64)
    est_vals = st.session_state.sketch.query_batch(seen_keys)

    min_cnt = theta * n_txns
    survivor_idx = np.where(est_vals >= min_cnt)[0]
    true_freq_mask = true_vals >= min_cnt

    key_to_itemset = dataset["key_to_itemset"]
    freq_map = {}
    is_false_positive_map = {}
    top_items = []

    for idx in survivor_idx:
        k = int(seen_keys[idx])
        itemset = key_to_itemset[k]
        t_cnt = float(true_vals[idx])
        e_cnt = float(est_vals[idx])
        used_cnt = t_cnt if use_pass2 else e_cnt
        if used_cnt >= min_cnt:
            freq_map[itemset] = used_cnt
            is_false_positive_map[itemset] = t_cnt < min_cnt
            top_items.append({
                "Itemset": " + ".join(itemset),
                "Len": len(itemset),
                "Sketch Est.": int(round(e_cnt)),
                "Exact Count": int(t_cnt),
                "Error": int(round(e_cnt - t_cnt)),
                "Support (%)": round(used_cnt / n_txns * 100, 2),
                "Status": "⚠️ False Positive" if (not use_pass2 and t_cnt < min_cnt) else "✅ True Frequent",
            })

    df_items = pd.DataFrame(top_items)
    if not df_items.empty:
        df_items = df_items.sort_values(by=["Support (%)", "Len"], ascending=[False, True]).reset_index(drop=True)

    rules = []
    for itemset, sup_cnt in freq_map.items():
        if len(itemset) < 2:
            continue
        sup_xy = sup_cnt / n_txns
        for i in range(len(itemset)):
            consequent = (itemset[i],)
            antecedent = tuple(x for j, x in enumerate(itemset) if j != i)
            if antecedent in freq_map and consequent in freq_map:
                sup_x = freq_map[antecedent] / n_txns
                sup_y = freq_map[consequent] / n_txns
                conf = min(1.0, sup_xy / sup_x) if sup_x > 0 else 0.0
                lift = (conf / sup_y) if sup_y > 0 else 0.0
                if conf >= min_conf and lift >= 1.15:
                    rules.append({
                        "Antecedent": " + ".join(antecedent),
                        "Consequent": consequent[0],
                        "Support (%)": round(sup_xy * 100, 2),
                        "Confidence (%)": round(conf * 100, 1),
                        "Lift": round(lift, 2),
                        "Count": int(round(sup_cnt)),
                        "Is_FP": is_false_positive_map.get(itemset, False),
                    })

    df_rules = pd.DataFrame(rules)
    if not df_rules.empty:
        df_rules = df_rules.sort_values(by=["Lift", "Confidence (%)"], ascending=False).reset_index(drop=True)

    est_f = est_vals >= min_cnt
    tp = int(np.sum(true_freq_mask & est_f))
    fp = int(np.sum((~true_freq_mask) & est_f))
    fn = int(np.sum(true_freq_mask & (~est_f)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    p2_f1 = (2 * 1.0 * rec / (1.0 + rec)) if tp > 0 else 0.0

    summary = {
        "seen_candidates": len(seen_keys),
        "survivors": len(survivor_idx),
        "pruned_pct": (1.0 - len(survivor_idx) / max(1, len(seen_keys))) * 100.0,
        "true_freq": int(np.sum(true_freq_mask)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec * 100.0,
        "recall": rec * 100.0,
        "f1": f1 * 100.0,
        "pass2_f1": p2_f1 * 100.0,
        "mae": float(np.mean(np.abs((est_vals - true_vals) / max(1, n_txns)))),
    }
    return df_items, df_rules, summary

def build_supermarket_aisle_svg(df_rules: pd.DataFrame) -> str:
    zones = []
    used_pairs = set()
    if not df_rules.empty:
        for _, r in df_rules.iterrows():
            a = r["Antecedent"].split(" + ")[0]
            b = r["Consequent"]
            pair_key = tuple(sorted([a, b]))
            if pair_key not in used_pairs:
                used_pairs.add(pair_key)
                zones.append((a, b, r["Lift"], r["Confidence (%)"], r["Is_FP"]))
            if len(zones) >= 6:
                break

    if not zones:
        return (
            '<div style="background:#f8fafc;border:1px dashed #cbd5e1;border-radius:12px;'
            'padding:2rem;text-align:center;color:#64748b;">'
            'Start the live stream to auto-arrange supermarket shelves based on detected co-purchase bundles!</div>'
        )

    cards_svg = []
    coords = [(18, 52), (310, 52), (602, 52), (18, 178), (310, 178), (602, 178)]
    aisle_colors = ["#0d9488", "#2563eb", "#7c3aed", "#d97706", "#059669", "#dc2626"]

    for idx, (a, b, lift, conf, is_fp) in enumerate(zones):
        x, y = coords[idx]
        col = "#dc2626" if is_fp else aisle_colors[idx % len(aisle_colors)]
        bg = "#fef2f2" if is_fp else "#ffffff"
        badge = "NOISE / FP" if is_fp else f"LIFT {lift:.2f}x"
        a_txt = (a[:15] + "..") if len(a) > 17 else a
        b_txt = (b[:15] + "..") if len(b) > 17 else b
        ea = ITEM_EMOJIS.get(a, "🛒")
        eb = ITEM_EMOJIS.get(b, "🛒")

        cards_svg.append(
            f"""
            <g transform="translate({x},{y})">
              <rect width="272" height="106" rx="12" fill="{bg}" stroke="{col}" stroke-width="2"/>
              <rect width="272" height="26" rx="10" fill="{col}"/>
              <text x="12" y="17" fill="#ffffff" font-size="11.5" font-weight="700" font-family="sans-serif">AISLE ZONE {idx+1} • KEEP ADJACENT</text>
              <text x="260" y="17" fill="#ffffff" font-size="10.5" font-weight="700" text-anchor="end" font-family="sans-serif">{badge}</text>

              <rect x="10" y="38" width="114" height="54" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>
              <text x="67" y="60" text-anchor="middle" font-size="18">{ea}</text>
              <text x="67" y="80" text-anchor="middle" fill="#0f172a" font-size="10.5" font-weight="700" font-family="sans-serif">{a_txt}</text>

              <text x="136" y="69" text-anchor="middle" fill="{col}" font-size="16" font-weight="800">⇄</text>

              <rect x="148" y="38" width="114" height="54" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>
              <text x="205" y="60" text-anchor="middle" font-size="18">{eb}</text>
              <text x="205" y="80" text-anchor="middle" fill="#0f172a" font-size="10.5" font-weight="700" font-family="sans-serif">{b_txt}</text>
            </g>
            """
        )

    svg = f"""
    <div style="width:100%;overflow-x:auto;background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;padding:10px;">
      <svg viewBox="0 0 892 302" width="100%" xmlns="http://www.w3.org/2000/svg">
        <rect x="0" y="0" width="892" height="302" rx="12" fill="#f8fafc"/>
        <text x="18" y="29" fill="#0f172a" font-size="15" font-weight="800" font-family="sans-serif">
          🏪 LIVE AUTO-OPTIMIZED SUPERMARKET FLOORPLAN — PLACE LINKED PRODUCTS SIDE-BY-SIDE
        </text>
        <text x="874" y="29" fill="#64748b" font-size="11" text-anchor="end" font-family="sans-serif">
          Updated in real time from Fixed-Memory Sketch
        </text>
        {''.join(cards_svg)}
      </svg>
    </div>
    """
    return svg

st.sidebar.title("⚙️ Stream & Memory Controls")

default_csv = find_groceries_path()
uploaded_csv = None
if default_csv is None:
    st.sidebar.warning("`groceries.csv` not found in folder — using synthetic fallback or upload below:")
    uploaded_csv = st.sidebar.file_uploader("Upload groceries.csv", type=["csv"])

algo_choice = st.sidebar.selectbox(
    "Probabilistic Algorithm",
    ["Count-Min Sketch (CMS)", "Count Sketch (CS)", "Counting Bloom Variant"],
    index=0,
    help="Choose which fixed-memory sketch runs in the live POS backend.",
)

budget_kb = st.sidebar.select_slider(
    "🔒 Fixed Memory Budget (KB)",
    options=[16, 32, 64, 128, 256, 512, 1024],
    value=256,
    help="Strict physical RAM allocated to the sketch array (.nbytes). Try lowering to 32 or 64 KB to watch collision noise appear live!",
)

depth_d = st.sidebar.select_slider("Hash Rows / Depth (d)", options=[3, 5, 7], value=5)
theta_pct = st.sidebar.slider("Min Support Threshold (θ %)", 0.5, 5.0, 1.0, step=0.5)
min_conf_pct = st.sidebar.slider("Min Rule Confidence (%)", 15, 60, 30, step=5)
use_pass2 = st.sidebar.toggle(
    "Enable Pass-2 Exact Recount of Survivors",
    value=False,
    help="Turn OFF to see raw sketch errors (e.g. false positives at 64 KB). Turn ON to show how CMS + Pass-2 reaches 100% F1!",
)

st.sidebar.divider()
st.sidebar.subheader("🌊 Continuous Stream Settings")
batch_size = st.sidebar.select_slider("Baskets per Stream Tick", options=[100, 250, 500, 1000], value=250)
tick_delay = st.sidebar.slider("Stream Speed (delay sec)", 0.05, 0.80, 0.20, step=0.05)
loop_stream = st.sidebar.checkbox("Loop stream continuously", value=False)
seed_val = st.sidebar.number_input("Hash Seed", min_value=0, max_value=99, value=0)

dataset = prepare_dataset(
    default_csv,
    uploaded_csv.getvalue() if uploaded_csv is not None else None,
    max_len=3,
)

desired_cfg = (algo_choice, budget_kb, depth_d, int(seed_val), 3)
if "cfg" not in st.session_state or st.session_state.cfg != desired_cfg:
    init_stream_state(*desired_cfg)

status_html = (
    '<span class="live-badge">● LIVE STREAMING</span>'
    if st.session_state.is_running
    else '<span class="paused-badge">❚❚ READY / PAUSED</span>'
)

st.markdown(
    f"""
    <div class="main-banner">
      <h1>{status_html} Smart-Market Real-Time Pattern & Shelf Placement Engine</h1>
      <p>
        Backend: <b>{algo_choice}</b> locked at <b>{budget_kb} KB fixed RAM</b> (d={depth_d}) &nbsp;|&nbsp;
        Dataset: <b>{dataset['source_name']}</b> ({len(dataset['transactions']):,} baskets) &nbsp;|&nbsp;
        Mode: <b>{'Two-Pass Verified' if use_pass2 else 'Single-Pass Raw Sketch'}</b>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

total_txns = len(dataset["transactions"])

b1, b2, b3, b4, b5 = st.columns([1.3, 1.2, 1.3, 1.5, 1.1])
if b1.button("▶️ Start Live Stream", type="primary", use_container_width=True):
    if st.session_state.stream_cursor >= total_txns and not loop_stream:
        init_stream_state(*desired_cfg)
    st.session_state.is_running = True
    st.rerun()

if b2.button("⏸️ Pause", use_container_width=True):
    st.session_state.is_running = False
    st.rerun()

if b3.button("⏭️ Step +1 Batch", use_container_width=True):
    st.session_state.is_running = False
    ingest_next_batch(dataset, batch_size, theta_pct / 100.0, loop_stream)
    st.rerun()

if b4.button("⚡ Stream All Now", use_container_width=True):
    st.session_state.is_running = False
    if st.session_state.stream_cursor >= total_txns:
        init_stream_state(*desired_cfg)
    remaining = total_txns - st.session_state.stream_cursor
    chunk = max(1, remaining // 10)
    while st.session_state.stream_cursor < total_txns:
        ingest_next_batch(dataset, chunk, theta_pct / 100.0, False)
    st.rerun()

if b5.button("🔄 Reset", use_container_width=True):
    init_stream_state(*desired_cfg)
    st.rerun()

if st.session_state.is_running:
    ingest_next_batch(dataset, batch_size, theta_pct / 100.0, loop_stream)

progress = min(1.0, st.session_state.stream_cursor / max(1, total_txns))
st.progress(
    progress,
    text=(
        f"Streamed {st.session_state.total_txns_processed:,} baskets "
        f"({st.session_state.stream_cursor:,} / {total_txns:,} in current pass)  •  "
        f"{st.session_state.total_updates:,} itemset updates ingested"
    ),
)

df_items, df_rules, summary = mine_live_rules(
    dataset, theta_pct / 100.0, min_conf_pct / 100.0, use_pass2
)

actual_kb = st.session_state.sketch.memory_bytes() / 1024.0
exact_kb = (len(st.session_state.exact_counts) * 12) / 1024.0
throughput = (
    st.session_state.total_updates / st.session_state.total_ingest_time
    if st.session_state.total_ingest_time > 0
    else 0.0
)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Fixed Sketch RAM</div>'
        f'<div class="kpi-val">{actual_kb:.1f} KB</div>'
        f'<div class="kpi-sub">vs {exact_kb:.1f} KB Exact Table</div></div>',
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Stream Throughput</div>'
        f'<div class="kpi-val">{throughput/1e6:.2f}M/s</div>'
        f'<div class="kpi-sub">{st.session_state.total_updates:,} updates</div></div>',
        unsafe_allow_html=True,
    )
with k3:
    f1_show = summary.get("pass2_f1", 0.0) if use_pass2 else summary.get("f1", 0.0)
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{"Pass-2" if use_pass2 else "Pass-1"} F1-Score</div>'
        f'<div class="kpi-val">{f1_show:.1f}%</div>'
        f'<div class="kpi-sub">Prec {summary.get("precision", 0):.1f}% • Rec {summary.get("recall", 0):.1f}%</div></div>',
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Noise Pruned</div>'
        f'<div class="kpi-val">{summary.get("pruned_pct", 0):.2f}%</div>'
        f'<div class="kpi-sub">{summary.get("survivors", 0):,} / {summary.get("seen_candidates", 0):,} left</div></div>',
        unsafe_allow_html=True,
    )
with k5:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Placement Rules</div>'
        f'<div class="kpi-val">{len(df_rules)}</div>'
        f'<div class="kpi-sub">Support MAE: {summary.get("mae", 0):.5f}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

col_floor, col_pos = st.columns([2.15, 1.0])

with col_floor:
    st.markdown(build_supermarket_aisle_svg(df_rules), unsafe_allow_html=True)

with col_pos:
    st.markdown("#### 🧾 Live POS Checkout Stream")
    if st.session_state.recent_receipts:
        lines = []
        for rid, items in st.session_state.recent_receipts:
            preview = ", ".join(item_label(x) for x in items[:4])
            more = f" +{len(items)-4} more" if len(items) > 4 else ""
            lines.append(f'<span class="receipt-id">#{rid:04d}</span> <span class="receipt-items">{preview}{more}</span>')
        st.markdown(f'<div class="receipt-box">{"<br>".join(lines)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="receipt-box">Waiting for checkout stream...<br>Click <b>▶️ Start Live Stream</b> above!</div>',
            unsafe_allow_html=True,
        )

    st.caption(
        f"🛡️ Bloom Gatekeeper: {st.session_state.bloom_gate.memory_bytes()/1024:.1f} KB • "
        f"Exact Ground Truth: {summary.get('true_freq', 0)} frequent sets"
    )

st.write("")

t_shelf, t_charts, t_cmp, t_cart = st.tabs([
    "📍 Live Shelf Placement Recommendations",
    "📈 Real-Time Memory O(1) vs O(|C|) & Accuracy",
    "⚔️ Live CMS vs. Count Sketch Race",
    "🛍️ Interactive Customer Cart Recommender",
])

with t_shelf:
    left_s, right_s = st.columns([1.15, 1.0])
    with left_s:
        st.markdown("#### 🎯 Actionable Merchandising Directives (*Keep Near Each Other*)")
        if not df_rules.empty:
            for idx, r in df_rules.head(6).iterrows():
                card_cls = "shelf-card shelf-card-warn" if r["Is_FP"] else "shelf-card"
                ant_pretty = " + ".join(item_label(x) for x in r["Antecedent"].split(" + "))
                cons_pretty = item_label(r["Consequent"])
                fp_pill = '<span class="pill pill-fp">⚠️ Sketch Collision FP</span>' if r["Is_FP"] else ""
                st.markdown(
                    f"""
                    <div class="{card_cls}">
                      <div class="shelf-title">#{idx+1} Place {cons_pretty} next to {ant_pretty}</div>
                      <div class="shelf-sub">
                        Shoppers buying <b>{r['Antecedent']}</b> buy <b>{r['Consequent']}</b>
                        <b>{r['Lift']:.2f}×</b> more often than random chance.
                      </div>
                      <div>
                        <span class="pill pill-lift">Lift {r['Lift']:.2f}×</span>
                        <span class="pill pill-conf">Conf {r['Confidence (%)']:.1f}%</span>
                        <span class="pill pill-supp">Support {r['Support (%)']:.2f}% ({r['Count']} txns)</span>
                        {fp_pill}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Stream more transactions (or lower Min Support / Confidence) to discover strong shelf placement bundles.")

    with right_s:
        st.markdown("#### 🔥 Live Detected Frequent Itemsets (Sketch vs. Exact)")
        if not df_items.empty:
            st.dataframe(df_items.head(15), use_container_width=True, hide_index=True, height=430)
        else:
            st.info("No frequent itemsets detected yet — press Start Live Stream.")

with t_charts:
    if len(st.session_state.history) >= 2:
        df_h = pd.DataFrame(st.session_state.history)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

        axes[0].plot(df_h["txns"], df_h["exact_kb"], color="#dc2626", lw=2.5, label="Exact Compact Table — O(|C|) Growing")
        axes[0].plot(df_h["txns"], df_h["sketch_kb"], color="#0d9488", lw=2.8, linestyle="-", label=f"{algo_choice.split(' (')[0]} — Fixed {actual_kb:.0f} KB O(1)")
        axes[0].fill_between(
            df_h["txns"],
            df_h["sketch_kb"],
            df_h["exact_kb"],
            where=(df_h["exact_kb"] >= df_h["sketch_kb"]),
            color="#fecaca",
            alpha=0.35,
            label="Memory Saved",
        )
        axes[0].set_xlabel("Transactions Streamed (N)")
        axes[0].set_ylabel("Memory Footprint (KB)")
        axes[0].set_title("Live Space Complexity: Fixed O(1) vs. Unbounded O(|C|)")
        axes[0].grid(True, linestyle="--", alpha=0.4)
        axes[0].legend(fontsize=8.5)

        axes[1].plot(df_h["txns"], df_h["f1"], color="#2563eb", lw=2.4, label="Pass-1 F1 (%)")
        axes[1].plot(df_h["txns"], df_h["pass2_f1"], color="#16a34a", lw=2.2, linestyle="--", label="Pass-2 Verified F1 (%)")
        axes[1].plot(df_h["txns"], df_h["precision"], color="#f59e0b", lw=1.6, alpha=0.85, label="Pass-1 Precision (%)")
        axes[1].plot(df_h["txns"], df_h["recall"], color="#7c3aed", lw=1.6, linestyle=":", label="Recall (%)")
        axes[1].set_ylim(-2, 105)
        axes[1].set_xlabel("Transactions Streamed (N)")
        axes[1].set_ylabel("Score (%)")
        axes[1].set_title(f"Live Mining Quality at Fixed {budget_kb} KB (θ={theta_pct:.1f}%)")
        axes[1].grid(True, linestyle="--", alpha=0.4)
        axes[1].legend(fontsize=8.2)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Stream at least 2 batches to watch the live O(1) vs O(|C|) memory and accuracy curves draw in real time.")

with t_cmp:
    st.markdown(f"#### ⚔️ Live Head-to-Head at Exact Equal Memory ({budget_kb} KB, d={depth_d})")
    if st.session_state.total_txns_processed > 0:
        seen_keys = np.fromiter(st.session_state.exact_counts.keys(), dtype=np.uint64)
        true_vals = np.fromiter(st.session_state.exact_counts.values(), dtype=np.float64)
        n_tx = st.session_state.total_txns_processed
        min_cnt = (theta_pct / 100.0) * n_tx
        true_f = true_vals >= min_cnt

        cmp_rows = []
        for cname, sk_obj in [
            ("Count-Min Sketch (CMS)", st.session_state.cms_cmp),
            ("Count Sketch (CS)", st.session_state.cs_cmp),
        ]:
            est = sk_obj.query_batch(seen_keys)
            est_f = est >= min_cnt
            tp = int(np.sum(true_f & est_f))
            fp = int(np.sum((~true_f) & est_f))
            fn = int(np.sum(true_f & (~est_f)))
            p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            p2 = (2 * 1.0 * r / (1.0 + r)) if tp > 0 else 0.0
            mae = float(np.mean(np.abs((est - true_vals) / n_tx)))
            cmp_rows.append({
                "Algorithm": cname,
                "Fixed RAM (KB)": round(sk_obj.memory_bytes() / 1024, 1),
                "Support MAE": round(mae, 5),
                "Pass-1 Prec (%)": round(p * 100, 2),
                "Recall (%)": round(r * 100, 2),
                "Pass-1 F1 (%)": round(f1 * 100, 2),
                "Pass-2 Verified F1 (%)": round(p2 * 100, 2),
                "False Positives": fp,
                "Missed Patterns (FN)": fn,
            })
        st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
        st.caption(
            "💡 Try sliding Fixed Memory down to **64 KB** in the sidebar: watch **Count Sketch** crush CMS in Pass-1 F1 "
            "(due to signed ±1 noise cancellation), while **CMS** keeps **0 Missed Patterns (100% Recall)** for Pass-2 Verification!"
        )
    else:
        st.info("Start the stream to compare Count-Min Sketch vs. Count Sketch side-by-side at your chosen RAM budget.")

with t_cart:
    st.markdown("#### 🛍️ Live Checkout Cross-Sell Simulator (*What should we suggest to this shopper?*)")
    default_cart = [x for x in ["citrus fruit", "other vegetables"] if x in dataset["unique_items"]]
    cart = st.multiselect(
        "Pick items currently in the customer's shopping basket:",
        dataset["unique_items"],
        default=default_cart,
    )
    if cart and not df_rules.empty:
        cart_set = set(cart)
        matched = []
        for _, r in df_rules.iterrows():
            ant = set(r["Antecedent"].split(" + "))
            if ant.issubset(cart_set) and r["Consequent"] not in cart_set:
                matched.append(r)
        if matched:
            df_m = pd.DataFrame(matched).drop_duplicates(subset=["Consequent"]).head(5)
            for _, r in df_m.iterrows():
                st.success(
                    f"✨ Recommend **{item_label(r['Consequent'])}** — "
                    f"Lift **{r['Lift']:.2f}×** | Confidence **{r['Confidence (%)']:.1f}%** "
                    f"(matched `{r['Antecedent']}`)"
                )
        else:
            st.warning("No matching rule yet at current thresholds — try streaming more baskets or adding staples like `whole milk` or `yogurt`.")
    elif not cart:
        st.info("Select one or more items above to test live recommendations.")

if st.session_state.is_running:
    time.sleep(tick_delay)
    st.rerun()
