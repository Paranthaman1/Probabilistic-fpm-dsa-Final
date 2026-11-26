import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from data_utils import load_groceries_csv, generate_synthetic_transactions, build_candidate_stream
from membership_filters import BloomFilter, CountingBloomFilter, CuckooFilter
from frequency_estimators import CountMinSketch, CountSketch, CountingBloomEstimator, LossyCounting
from metrics import compute_mining_metrics, evaluate_two_pass_verification
from real_world_recommender import RetailBasketRecommender

SEEDS = list(range(10))
TABLES_DIR = os.path.join(ROOT_DIR, "results", "tables")
PLOTS_DIR = os.path.join(ROOT_DIR, "results", "plots")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
})

def run_experiment_a(data: dict):
    present_keys = data["present_keys"]
    absent_keys = data["absent_keys"]
    n = len(present_keys)
    rows = []

    for p_target in [0.1, 0.01, 0.001]:
        bf_fprs, bf_ins, bf_q, bf_mem, bf_k = [], [], [], 0, 0
        cbf_fprs, cbf_ins, cbf_q, cbf_del, cbf_mem, cbf_ovf = [], [], [], [], 0, []

        for s in SEEDS:
            bf = BloomFilter(n, p_target, seed=s)
            t0 = time.perf_counter()
            bf.insert_batch(present_keys)
            bf_ins.append(n / (time.perf_counter() - t0))
            t0 = time.perf_counter()
            res_abs = bf.query_batch(absent_keys)
            bf_q.append(len(absent_keys) / (time.perf_counter() - t0))
            bf_fprs.append(np.mean(res_abs))
            bf_mem, bf_k = bf.memory_bytes(), bf.k

            cbf = CountingBloomFilter(n, p_target, counter_bits=8, seed=s)
            t0 = time.perf_counter()
            cbf.insert_batch(present_keys)
            cbf_ins.append(n / (time.perf_counter() - t0))
            t0 = time.perf_counter()
            res_abs_c = cbf.query_batch(absent_keys)
            cbf_q.append(len(absent_keys) / (time.perf_counter() - t0))
            cbf_fprs.append(np.mean(res_abs_c))
            cbf_ovf.append(cbf.overflows)
            t0 = time.perf_counter()
            cbf.delete_batch(present_keys[:10000])
            cbf_del.append(10000 / (time.perf_counter() - t0))
            cbf_mem = cbf.memory_bytes()

        rows.append({
            "Structure": f"Bloom Filter (p={p_target}, k={bf_k})",
            "Family": "Bloom Filter",
            "Memory_KB": round(bf_mem / 1024, 2),
            "Insertions_per_s": round(float(np.mean(bf_ins))),
            "Queries_per_s": round(float(np.mean(bf_q))),
            "Deletes_per_s": "N/A",
            "FPR_mean_pct": round(float(np.mean(bf_fprs)) * 100, 4),
            "FPR_pct": f"{np.mean(bf_fprs)*100:.3f} ± {np.std(bf_fprs)*100:.3f}",
            "FNR_pct": "0.000",
            "Failures_or_Overflows": "0",
        })
        rows.append({
            "Structure": f"Counting Bloom (p={p_target}, 8-bit)",
            "Family": "Counting Bloom (8-bit)",
            "Memory_KB": round(cbf_mem / 1024, 2),
            "Insertions_per_s": round(float(np.mean(cbf_ins))),
            "Queries_per_s": round(float(np.mean(cbf_q))),
            "Deletes_per_s": round(float(np.mean(cbf_del))),
            "FPR_mean_pct": round(float(np.mean(cbf_fprs)) * 100, 4),
            "FPR_pct": f"{np.mean(cbf_fprs)*100:.3f} ± {np.std(cbf_fprs)*100:.3f}",
            "FNR_pct": "0.000",
            "Failures_or_Overflows": f"{np.mean(cbf_ovf):.1f}",
        })

    for fp_bits in [8, 12, 16]:
        cf_fprs, cf_fnrs, cf_ins, cf_q, cf_del, cf_fail, cf_mem = [], [], [], [], [], [], 0
        for s in SEEDS:
            cf = CuckooFilter(n, bucket_size=4, fp_bits=fp_bits, max_kicks=150, seed=s)
            t0 = time.perf_counter()
            cf.insert_batch(present_keys)
            cf_ins.append(n / (time.perf_counter() - t0))
            t0 = time.perf_counter()
            res_abs_cf = cf.query_batch(absent_keys)
            cf_q.append(len(absent_keys) / (time.perf_counter() - t0))
            res_pres_cf = cf.query_batch(present_keys)
            cf_fprs.append(np.mean(res_abs_cf))
            cf_fnrs.append(1.0 - np.mean(res_pres_cf))
            cf_fail.append(cf.failures)
            t0 = time.perf_counter()
            cf.delete_batch(present_keys[:10000])
            cf_del.append(10000 / (time.perf_counter() - t0))
            cf_mem = cf.memory_bytes()

        rows.append({
            "Structure": f"Cuckoo Filter (b=4, fp={fp_bits}-bit)",
            "Family": "Cuckoo Filter (b=4)",
            "Memory_KB": round(cf_mem / 1024, 2),
            "Insertions_per_s": round(float(np.mean(cf_ins))),
            "Queries_per_s": round(float(np.mean(cf_q))),
            "Deletes_per_s": round(float(np.mean(cf_del))),
            "FPR_mean_pct": round(float(np.mean(cf_fprs)) * 100, 4),
            "FPR_pct": f"{np.mean(cf_fprs)*100:.3f} ± {np.std(cf_fprs)*100:.3f}",
            "FNR_pct": f"{np.mean(cf_fnrs)*100:.3f}",
            "Failures_or_Overflows": f"{np.mean(cf_fail):.1f}",
        })

    stress_rows = []
    for b_size in [2, 4]:
        for max_kicks in [100, 500]:
            sub_keys = present_keys[:128000]
            cf = CuckooFilter(capacity=131072, bucket_size=b_size, fp_bits=12, max_kicks=max_kicks, seed=0)
            cf.insert_batch(sub_keys)
            stress_rows.append({
                "Bucket_Size": b_size,
                "Max_Kicks": max_kicks,
                "Total_Slots": cf.num_buckets * cf.bucket_size,
                "Attempted_Inserts": 128000,
                "Achieved_Load_Factor_pct": round(cf.load_factor() * 100, 2),
                "Insertion_Failures": cf.failures,
            })

    df_a = pd.DataFrame(rows)
    df_stress = pd.DataFrame(stress_rows)
    df_a.to_csv(os.path.join(TABLES_DIR, "experiment_a_membership.csv"), index=False)
    df_stress.to_csv(os.path.join(TABLES_DIR, "experiment_a_cuckoo_stress.csv"), index=False)
    return df_a, df_stress

def run_experiment_b(data: dict, theta: float = 0.01):
    stream = data["stream"]
    present_keys = data["present_keys"]
    true_freqs = data["true_freqs"]
    n_txns = data["n_txns"]
    aux_kb = data["candidate_storage_bytes"] / 1024.0

    budgets_kb = [64, 256, 1024, 4096]
    rows = []

    for b_kb in budgets_kb:
        mem_bytes = b_kb * 1024
        for name, cls in [
            ("Count-Min Sketch (d=5)", CountMinSketch),
            ("Count Sketch (d=5)", CountSketch),
            ("Counting Bloom Variant (k=5)", CountingBloomEstimator),
        ]:
            precs, recs, f1s, maes, maxs, upds, qrys, vf1s, vrec, pruned = [], [], [], [], [], [], [], [], [], []
            actual_mem = 0
            for s in SEEDS:
                sk = cls(mem_bytes, 5, seed=s)
                t0 = time.perf_counter()
                sk.update_batch(stream)
                upds.append(len(stream) / (time.perf_counter() - t0))

                t0 = time.perf_counter()
                est = sk.query_batch(present_keys)
                qrys.append(len(present_keys) / (time.perf_counter() - t0))

                m = compute_mining_metrics(est, true_freqs, n_txns, theta=theta)
                v = evaluate_two_pass_verification(est, true_freqs, n_txns, theta=theta)
                precs.append(m["precision"])
                recs.append(m["recall"])
                f1s.append(m["f1"])
                maes.append(m["mae_support"])
                maxs.append(m["max_support_error"])
                vf1s.append(v["verified_f1"])
                vrec.append(v["verified_recall"])
                pruned.append(v["candidates_pruned_pct"])
                actual_mem = sk.memory_bytes()

            rows.append({
                "Budget_KB": b_kb,
                "Estimator": name,
                "Sketch_KB": round(actual_mem / 1024, 1),
                "Total_Pipeline_KB": round(actual_mem / 1024 + aux_kb, 1),
                "Updates_per_s": round(float(np.mean(upds))),
                "Queries_per_s": round(float(np.mean(qrys))),
                "MAE_mean": float(np.mean(maes)),
                "Support_MAE": f"{np.mean(maes):.5f} ± {np.std(maes):.5f}",
                "Max_Error": round(float(np.mean(maxs)), 4),
                "Prec_mean": float(np.mean(precs)) * 100,
                "Rec_mean": float(np.mean(recs)) * 100,
                "F1_mean": float(np.mean(f1s)) * 100,
                "Precision_pct": f"{np.mean(precs)*100:.2f} ± {np.std(precs)*100:.2f}",
                "Recall_pct": f"{np.mean(recs)*100:.2f} ± {np.std(recs)*100:.2f}",
                "F1_pct": f"{np.mean(f1s)*100:.2f} ± {np.std(f1s)*100:.2f}",
                "Pass2_Recall_pct": round(float(np.mean(vrec)) * 100, 2),
                "Pass2_F1_pct": round(float(np.mean(vf1s)) * 100, 2),
                "Candidates_Pruned_pct": round(float(np.mean(pruned)), 2),
            })

    depth_rows = []
    for d_val in [3, 5, 7]:
        for name, cls in [
            ("Count-Min Sketch", CountMinSketch),
            ("Count Sketch", CountSketch),
            ("Counting Bloom Variant", CountingBloomEstimator),
        ]:
            f1s, maes = [], []
            for s in SEEDS:
                sk = cls(256 * 1024, d_val, seed=s)
                sk.update_batch(stream)
                est = sk.query_batch(present_keys)
                m = compute_mining_metrics(est, true_freqs, n_txns, theta=theta)
                f1s.append(m["f1"] * 100)
                maes.append(m["mae_support"])
            depth_rows.append({
                "Depth_or_Hashes": d_val,
                "Estimator": name,
                "F1_mean_pct": round(float(np.mean(f1s)), 2),
                "F1_std_pct": round(float(np.std(f1s)), 2),
                "MAE_mean": round(float(np.mean(maes)), 5),
            })

    thresh_rows = []
    for th in [0.005, 0.01, 0.02, 0.05]:
        for name, cls in [
            ("Count-Min Sketch (d=5)", CountMinSketch),
            ("Count Sketch (d=5)", CountSketch),
            ("Counting Bloom Variant (k=5)", CountingBloomEstimator),
        ]:
            precs, recs, f1s = [], [], []
            for s in SEEDS:
                sk = cls(256 * 1024, 5, seed=s)
                sk.update_batch(stream)
                est = sk.query_batch(present_keys)
                m = compute_mining_metrics(est, true_freqs, n_txns, theta=th)
                precs.append(m["precision"] * 100)
                recs.append(m["recall"] * 100)
                f1s.append(m["f1"] * 100)
            thresh_rows.append({
                "Threshold_pct": th * 100,
                "Estimator": name,
                "Precision_pct": round(float(np.mean(precs)), 2),
                "Recall_pct": round(float(np.mean(recs)), 2),
                "F1_pct": round(float(np.mean(f1s)), 2),
            })

    lc_rows = []
    for eps in [0.005, 0.001, 0.0005, 0.0002]:
        lc = LossyCounting(eps=eps)
        t0 = time.perf_counter()
        lc.update_stream(stream)
        upd_rate = len(stream) / (time.perf_counter() - t0)
        est = lc.query_batch(present_keys)
        m = compute_mining_metrics(est, true_freqs, n_txns, theta=theta)
        lc_rows.append({
            "Epsilon": eps,
            "Peak_Entries": lc.max_entries,
            "Peak_Mem_KB": round(lc.peak_memory_bytes() / 1024, 1),
            "Updates_per_s": round(upd_rate),
            "Support_MAE": round(m["mae_support"], 5),
            "Precision_pct": round(m["precision"] * 100, 2),
            "Recall_pct": round(m["recall"] * 100, 2),
            "F1_pct": round(m["f1"] * 100, 2),
        })

    df_b = pd.DataFrame(rows)
    df_depth = pd.DataFrame(depth_rows)
    df_thresh = pd.DataFrame(thresh_rows)
    df_lc = pd.DataFrame(lc_rows)

    df_b.to_csv(os.path.join(TABLES_DIR, "experiment_b_estimators.csv"), index=False)
    df_depth.to_csv(os.path.join(TABLES_DIR, "experiment_b_depth_tuning.csv"), index=False)
    df_thresh.to_csv(os.path.join(TABLES_DIR, "experiment_b_threshold_sweep.csv"), index=False)
    df_lc.to_csv(os.path.join(TABLES_DIR, "experiment_b_lossy_counting.csv"), index=False)
    return df_b, df_depth, df_thresh, df_lc

def run_experiment_c_synthetic_scaling():
    rows = []
    for dist in ["uniform", "skewed"]:
        txns = generate_synthetic_transactions(num_transactions=50000, num_items=300, avg_length=5, distribution=dist, seed=42)
        sdata = build_candidate_stream(txns, max_len=2, build_absent=False)
        th = 0.01 if dist == "skewed" else 0.005
        for name, cls in [("Count-Min Sketch (d=5)", CountMinSketch), ("Count Sketch (d=5)", CountSketch)]:
            f1s, maes, upds = [], [], []
            for s in range(5):
                sk = cls(256 * 1024, 5, seed=s)
                t0 = time.perf_counter()
                sk.update_batch(sdata["stream"])
                upds.append(len(sdata["stream"]) / (time.perf_counter() - t0))
                est = sk.query_batch(sdata["present_keys"])
                m = compute_mining_metrics(est, sdata["true_freqs"], sdata["n_txns"], theta=th)
                f1s.append(m["f1"] * 100)
                maes.append(m["mae_support"])
            rows.append({
                "Dataset_Type": f"Synthetic 50K ({dist.title()})",
                "N_Transactions": sdata["n_txns"],
                "Stream_Updates_M": len(sdata["stream"]),
                "Unique_Candidates": len(sdata["present_keys"]),
                "Estimator": name,
                "Budget_KB": 256,
                "Updates_per_s": round(float(np.mean(upds))),
                "Support_MAE": round(float(np.mean(maes)), 5),
                "F1_pct": round(float(np.mean(f1s)), 2),
            })

    scale_rows = []
    for n_tx in [10000, 50000, 100000]:
        txns = generate_synthetic_transactions(num_transactions=n_tx, num_items=300, avg_length=5, distribution="skewed", seed=42)
        sdata = build_candidate_stream(txns, max_len=2, build_absent=False)
        for name, cls in [("Count-Min Sketch (d=5)", CountMinSketch), ("Count Sketch (d=5)", CountSketch)]:
            sk = cls(256 * 1024, 5, seed=0)
            t0 = time.perf_counter()
            sk.update_batch(sdata["stream"])
            elapsed = time.perf_counter() - t0
            est = sk.query_batch(sdata["present_keys"])
            m = compute_mining_metrics(est, sdata["true_freqs"], sdata["n_txns"], theta=0.01)
            scale_rows.append({
                "N_Transactions": n_tx,
                "Stream_Updates_M": len(sdata["stream"]),
                "Unique_Candidates": len(sdata["present_keys"]),
                "Estimator": name,
                "Update_Time_s": round(elapsed, 4),
                "Updates_per_s": round(len(sdata["stream"]) / elapsed),
                "Support_MAE": round(m["mae_support"], 5),
                "F1_pct": round(m["f1"] * 100, 2),
            })

    df_dist = pd.DataFrame(rows)
    df_scale = pd.DataFrame(scale_rows)
    df_dist.to_csv(os.path.join(TABLES_DIR, "experiment_c_uniform_vs_skewed.csv"), index=False)
    df_scale.to_csv(os.path.join(TABLES_DIR, "experiment_c_scaling.csv"), index=False)
    return df_dist, df_scale

def run_real_world_demo(data: dict):
    cms_engine = RetailBasketRecommender(memory_kb=256, estimator="cms", min_support=0.01, min_confidence=0.30)
    cms_stats = cms_engine.fit(data, use_two_pass_recount=True, seed=0)

    cs_engine = RetailBasketRecommender(memory_kb=256, estimator="cs", min_support=0.01, min_confidence=0.30)
    cs_stats = cs_engine.fit(data, use_two_pass_recount=True, seed=0)

    cms_engine.rules_df.to_csv(os.path.join(TABLES_DIR, "real_world_association_rules.csv"), index=False)

    sample_carts = [
        ["citrus fruit", "other vegetables"],
        ["tropical fruit", "yogurt"],
        ["rolls/buns", "sausage"],
    ]
    rec_rows = []
    for cart in sample_carts:
        recs = cms_engine.recommend(cart, top_k=3)
        for _, r in recs.iterrows():
            rec_rows.append({
                "Customer_Cart": " + ".join(cart),
                "Recommended_Item": r["Consequent"],
                "Matched_Rule": r["Antecedent"],
                "Confidence_pct": r["Confidence_pct"],
                "Lift": r["Lift"],
                "Support_pct": r["Support_pct"],
            })
    df_recs = pd.DataFrame(rec_rows)
    df_recs.to_csv(os.path.join(TABLES_DIR, "real_world_cart_recommendations.csv"), index=False)
    return cms_stats, cs_stats, cms_engine.rules_df, df_recs

def generate_all_plots(df_a, df_stress, df_b, df_depth, df_thresh, df_scale, rules_df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {"Bloom Filter": "#1f77b4", "Counting Bloom (8-bit)": "#d62728", "Cuckoo Filter (b=4)": "#2ca02c"}
    markers = {"Bloom Filter": "o", "Counting Bloom (8-bit)": "s", "Cuckoo Filter (b=4)": "^"}

    for fam in ["Bloom Filter", "Counting Bloom (8-bit)", "Cuckoo Filter (b=4)"]:
        sub = df_a[df_a["Family"] == fam].sort_values("Memory_KB")
        axes[0].plot(sub["Memory_KB"], sub["FPR_mean_pct"], marker=markers[fam], lw=2.2, color=colors[fam], label=fam)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Actual Memory Footprint (KB)")
    axes[0].set_ylabel("Empirical False-Positive Rate (%) [Log]")
    axes[0].set_title("Exp A: Space vs. False-Positive Rate (149,229 Keys)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    short_labels = [
        "Bloom\n(p=0.1)", "CBF 8b\n(p=0.1)", "Bloom\n(p=0.01)", "CBF 8b\n(p=0.01)",
        "Bloom\n(p=0.001)", "CBF 8b\n(p=0.001)", "Cuckoo\n(8b fp)", "Cuckoo\n(12b fp)", "Cuckoo\n(16b fp)"
    ]
    bar_colors = [colors[f] for f in df_a["Family"]]
    axes[1].bar(range(len(df_a)), df_a["Queries_per_s"] / 1e6, color=bar_colors, alpha=0.88)
    axes[1].set_xticks(range(len(df_a)))
    axes[1].set_xticklabels(short_labels, fontsize=7.8)
    axes[1].set_ylabel("Query Throughput (Million Queries / sec)")
    axes[1].set_title("Exp A: Membership Query Throughput")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig1_exp_a_membership.png"), dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    budgets = sorted(df_b["Budget_KB"].unique())
    styles = [
        ("Count-Min Sketch (d=5)", "o", "#1f77b4"),
        ("Count Sketch (d=5)", "s", "#ff7f0e"),
        ("Counting Bloom Variant (k=5)", "^", "#2ca02c"),
    ]
    for name, mk, col in styles:
        sub = df_b[df_b["Estimator"] == name]
        axes[0].plot(sub["Budget_KB"], sub["F1_mean"], marker=mk, lw=2.2, color=col, label=name)
        axes[1].plot(sub["Budget_KB"], sub["MAE_mean"], marker=mk, lw=2.2, color=col, label=name)
        axes[2].plot(sub["Budget_KB"], sub["Prec_mean"], marker=mk, lw=2.2, color=col, label=f"{name.split(' (')[0]} Prec")
        axes[2].plot(sub["Budget_KB"], sub["Rec_mean"], marker=mk, lw=1.6, linestyle="--", color=col, alpha=0.75, label=f"{name.split(' (')[0]} Rec")

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{k} KB" for k in budgets])
        ax.set_xlabel("Structure Memory Budget (Log Scale)")
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].set_ylabel("F1-Score (%) at θ = 1.0%")
    axes[0].set_title("Exp B: F1-Score vs. Memory Budget")
    axes[0].legend()

    axes[1].set_yscale("log")
    axes[1].set_ylabel("Support MAE (Log Scale)")
    axes[1].set_title("Exp B: Support Error (MAE) vs. Memory")
    axes[1].legend()

    axes[2].set_ylabel("Score (%)")
    axes[2].set_title("Exp B: Precision (Solid) vs. Recall (Dashed)")
    axes[2].legend(fontsize=7.2, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig2_exp_b_memory_tradeoffs.png"), dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for name, mk, col in [
        ("Count-Min Sketch", "o", "#1f77b4"),
        ("Count Sketch", "s", "#ff7f0e"),
        ("Counting Bloom Variant", "^", "#2ca02c"),
    ]:
        sub = df_depth[df_depth["Estimator"] == name]
        axes[0].plot(sub["Depth_or_Hashes"], sub["F1_mean_pct"], marker=mk, lw=2.2, color=col, label=name)
    axes[0].set_xticks([3, 5, 7])
    axes[0].set_xlabel("Number of Rows / Hash Functions (d or k)")
    axes[0].set_ylabel("F1-Score (%) at 256 KB")
    axes[0].set_title("Parameter Tuning: Row Depth (256 KB)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    for name, mk, col in styles:
        sub = df_thresh[df_thresh["Estimator"] == name]
        axes[1].plot(sub["Threshold_pct"], sub["F1_pct"], marker=mk, lw=2.2, color=col, label=name)
    axes[1].set_xticks([0.5, 1.0, 2.0, 5.0])
    axes[1].set_xticklabels(["0.5%", "1.0%", "2.0%", "5.0%"])
    axes[1].set_xlabel("Minimum Support Threshold (θ)")
    axes[1].set_ylabel("F1-Score (%) at 256 KB")
    axes[1].set_title("Threshold Sensitivity (256 KB)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    sub_256 = df_b[df_b["Budget_KB"] == 256]
    x = np.arange(len(sub_256))
    w = 0.35
    axes[2].bar(x - w / 2, sub_256["F1_mean"], w, label="Pass 1 (Sketch Only)", color="#5b84b1")
    axes[2].bar(x + w / 2, sub_256["Pass2_F1_pct"], w, label="Pass 2 (Exact Verification)", color="#2ca02c")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["CMS\n(d=5)", "Count Sketch\n(d=5)", "CBF Variant\n(k=5)"])
    axes[2].set_ylim(75, 102)
    axes[2].set_ylabel("F1-Score (%)")
    axes[2].set_title("1-Pass vs. 2-Pass Verification (256 KB)")
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.5)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig3_exp_b_tuning_and_2pass.png"), dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for name, mk, col in [("Count-Min Sketch (d=5)", "o", "#1f77b4"), ("Count Sketch (d=5)", "s", "#ff7f0e")]:
        sub = df_scale[df_scale["Estimator"] == name]
        axes[0].plot(sub["N_Transactions"] / 1000, sub["Update_Time_s"], marker=mk, lw=2.2, color=col, label=name)
    axes[0].set_xticks([10, 50, 100])
    axes[0].set_xticklabels(["10K", "50K", "100K"])
    axes[0].set_xlabel("Dataset Scale (Transactions)")
    axes[0].set_ylabel("Stream Ingestion Time (seconds)")
    axes[0].set_title("Exp C: Linear Stream Scaling (10K → 100K Txns)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    top_rules = rules_df.head(8).iloc[::-1]
    labels = [f"{r['Antecedent']} → {r['Consequent']}" for _, r in top_rules.iterrows()]
    axes[1].barh(labels, top_rules["Lift"], color="#3b7a57", alpha=0.88)
    axes[1].set_xlabel("Association Rule Lift (Strength of Co-Purchase)")
    axes[1].set_title("Real-World Output: Top Grocery Rules (256 KB CMS)")
    axes[1].grid(True, axis="x", linestyle="--", alpha=0.5)
    for i, (_, r) in enumerate(top_rules.iterrows()):
        axes[1].text(r["Lift"] + 0.04, i, f"{r['Lift']:.2f}x ({r['Confidence_pct']:.0f}% conf)", va="center", fontsize=8)
    axes[1].set_xlim(0, top_rules["Lift"].max() + 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig4_scaling_and_realworld_rules.png"), dpi=160)
    plt.close()

def main():
    print("Loading groceries.csv and building ground truth...")
    txns = load_groceries_csv(os.path.join(ROOT_DIR, "data", "groceries.csv"))
    data = build_candidate_stream(txns, max_len=3)
    print(f"Dataset: {data['n_txns']:,} txns | {data['n_items']} items | {len(data['stream']):,} updates | {len(data['present_keys']):,} unique candidates")

    print("\nRunning Experiment A (Membership Filters)...")
    df_a, df_stress = run_experiment_a(data)

    print("Running Experiment B (Frequency Estimators, Depth Tuning, Threshold Sweep, Lossy Counting)...")
    df_b, df_depth, df_thresh, df_lc = run_experiment_b(data, theta=0.01)

    print("Running Experiment C (Synthetic Uniform vs Skewed & 10K-100K Scaling)...")
    df_dist, df_scale = run_experiment_c_synthetic_scaling()

    print("Running Real-World Retail Cross-Sell Engine Demo...")
    cms_stats, cs_stats, rules_df, df_recs = run_real_world_demo(data)

    print("Generating publication-quality plots...")
    generate_all_plots(df_a, df_stress, df_b, df_depth, df_thresh, df_scale, rules_df)

    print("\n=== SUMMARY OUTPUTS ===")
    print("\n[Experiment B: Depth Tuning at 256 KB]")
    print(df_depth.to_string(index=False))
    print("\n[Experiment C: Uniform vs Skewed (50K Txns, 256 KB)]")
    print(df_dist.to_string(index=False))
    print("\n[Experiment C: Scaling (10K -> 100K Skewed Txns, 256 KB)]")
    print(df_scale.to_string(index=False))
    print("\n[Real-World Engine Stats (256 KB + 2-Pass Verification)]")
    print("CMS Pipeline:", cms_stats)
    print("CS Pipeline:", cs_stats)
    print("\n[Top 8 Real-World Association Rules]")
    print(rules_df.head(8).to_string(index=False))
    print("\n[Sample Live Cart Cross-Sell Recommendations]")
    print(df_recs.to_string(index=False))

if __name__ == "__main__":
    main()
