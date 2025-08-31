
import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.stats import ks_2samp, entropy
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# ------------- Utility -------------

def js_divergence(p: np.ndarray, q: np.ndarray, bins: int = 30) -> float:
    """Jensen–Shannon divergence between two 1D samples using histograms."""
    p_hist, bin_edges = np.histogram(p[~np.isnan(p)], bins=bins, density=True)
    q_hist, _ = np.histogram(q[~np.isnan(q)], bins=bin_edges, density=True)
    # Smooth to avoid zeros
    eps = 1e-12
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()
    m = 0.5 * (p_hist + q_hist)
    return 0.5 * (entropy(p_hist, m) + entropy(q_hist, m))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_fig(path: Path, tight: bool = True):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

def infer_os_from_path(path: Path) -> Optional[int]:
    """Infer oversampling percent from path or filename."""
    s = str(path)
    m = re.search(r'(\D|^)(\d{1,2})\s*%(\D|$)', s)
    if m:
        return int(m.group(2))
    m = re.search(r'(\D|^)(\d{1,2})\s*pct(\D|$)', s, re.IGNORECASE)
    if m:
        return int(m.group(2))
    m = re.search(r'os[_\-]?(\d{1,2})(\D|$)', s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

def infer_disease_from_path(path: Path, diseases: List[str]) -> Optional[str]:
    s = str(path).lower()
    for d in diseases:
        if d.lower() in s:
            return d
    return None

# ------------- Metrics -------------

def compute_basic_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    metrics = {}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob) if y_prob is not None else None

    metrics["n"] = int(len(y_true))
    metrics["prevalence"] = float(np.mean(y_true)) if len(y_true) else np.nan
    metrics["accuracy"] = accuracy_score(y_true, y_pred) if len(y_true) else np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    if y_prob is not None and len(np.unique(y_true))>1:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["roc_auc"] = np.nan
        try:
            metrics["brier"] = brier_score_loss(y_true, y_prob)
        except Exception:
            metrics["brier"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
        metrics["brier"] = np.nan
    # confusion
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        # e.g., only one class present
        tn=fp=fn=tp=np.nan
    metrics.update({"tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn)})
    return metrics

def selection_rate(y_pred) -> float:
    y_pred = np.asarray(y_pred)
    if len(y_pred)==0:
        return np.nan
    return float(np.mean(y_pred))

def true_positive_rate(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pos = y_true == 1
    if pos.sum()==0:
        return np.nan
    return float(np.mean(y_pred[pos]))

def false_positive_rate(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    neg = y_true == 0
    if neg.sum()==0:
        return np.nan
    return float(np.mean(y_pred[neg]))

def disparate_impact_ratio(y_pred, sensitive) -> float:
    """Ratio of positive prediction rate: (max group SR) / (min group SR)."""
    df = pd.DataFrame({"y_pred": y_pred, "s": sensitive})
    rates = df.groupby("s")["y_pred"].mean()
    if len(rates)==0 or rates.min()==0:
        return np.nan
    return float(rates.max() / rates.min())

def demographic_parity_difference(y_pred, sensitive) -> float:
    df = pd.DataFrame({"y_pred": y_pred, "s": sensitive})
    rates = df.groupby("s")["y_pred"].mean()
    if len(rates)==0:
        return np.nan
    return float(rates.max() - rates.min())

def equal_opportunity_difference(y_true, y_pred, sensitive) -> float:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "s": sensitive})
    tprs = df.groupby("s").apply(lambda g: true_positive_rate(g["y_true"], g["y_pred"]))
    tprs = tprs.dropna()
    if len(tprs)==0:
        return np.nan
    return float(tprs.max() - tprs.min())

# ------------- Subgroup analysis -------------

def subgroup_table(df: pd.DataFrame, y_true_col: str, y_pred_col: str, y_prob_col: Optional[str], group_col: str) -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby(group_col):
        y_true = sub[y_true_col].values
        y_pred = sub[y_pred_col].values
        y_prob = sub[y_prob_col].values if (y_prob_col and y_prob_col in sub) else None
        met = compute_basic_metrics(y_true, y_pred, y_prob)
        met["group"] = g
        met["selection_rate"] = selection_rate(y_pred)
        met["tpr"] = true_positive_rate(y_true, y_pred)
        met["fpr"] = false_positive_rate(y_true, y_pred)
        rows.append(met)
    return pd.DataFrame(rows)

# ------------- Calibration plots -------------

def plot_calibration(y_true, y_prob, out_path: Path, title: str):
    if y_prob is None:
        return
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    except Exception:
        return
    plt.figure(figsize=(5,4))
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    save_fig(out_path)

def plot_group_bars(df: pd.DataFrame, value_col: str, group_col: str, out_path: Path, title: str):
    plt.figure(figsize=(6,4))
    groups = df[group_col].astype(str).values
    vals = df[value_col].values
    x = np.arange(len(groups))
    plt.bar(x, vals)
    plt.xticks(x, groups, rotation=30, ha="right")
    plt.ylabel(value_col)
    plt.title(title)
    save_fig(out_path)

def plot_os_trend(os_levels: List[int], values: List[float], out_path: Path, title: str, ylabel: str):
    plt.figure(figsize=(6,4))
    x = np.arange(len(os_levels))
    plt.plot(x, values, marker="o")
    plt.xticks(x, [str(o) for o in os_levels])
    plt.xlabel("Oversampling (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    save_fig(out_path)

# ------------- Distribution shift -------------

def compare_os_distributions(dfs_by_os: Dict[int, pd.DataFrame], features: List[str]) -> pd.DataFrame:
    """Pairwise KS and JS across OS for each feature; returns long-form table."""
    rows = []
    levels = sorted(dfs_by_os.keys())
    for f in features:
        for i in range(len(levels)):
            for j in range(i+1, len(levels)):
                a = levels[i]; b = levels[j]
                x = dfs_by_os[a][f].values if f in dfs_by_os[a] else np.array([])
                y = dfs_by_os[b][f].values if f in dfs_by_os[b] else np.array([])
                if len(x)==0 or len(y)==0:
                    ks_stat = js = np.nan
                    ks_p = np.nan
                else:
                    ks = ks_2samp(x, y)
                    ks_stat, ks_p = ks.statistic, ks.pvalue
                    try:
                        js = js_divergence(x, y)
                    except Exception:
                        js = np.nan
                rows.append({"feature": f, "os_a": a, "os_b": b, "ks_stat": ks_stat, "ks_p": ks_p, "js_divergence": js})
    return pd.DataFrame(rows)

# ------------- File discovery -------------

def discover_prediction_files(data_root: Path, diseases: List[str], os_levels: List[int], file_pattern: Optional[str]) -> Dict[str, Dict[int, List[Path]]]:
    """Return dict[disease][os] -> list of files."""
    result = {d: {o: [] for o in os_levels} for d in diseases}
    regex = re.compile(file_pattern) if file_pattern else None
    for root, _, files in os.walk(data_root):
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue
            p = Path(root) / fn
            if regex and not regex.search(str(p)):
                continue
            d = infer_disease_from_path(p, diseases)
            o = infer_os_from_path(p)
            if d is None or o is None or o not in os_levels:
                continue
            result[d][o].append(p)
    return result

def choose_best_file(paths: List[Path]) -> Optional[Path]:
    """If multiple candidates exist, pick the shortest path name (heuristic)."""
    if not paths:
        return None
    return sorted(paths, key=lambda x: len(str(x)))[0]

# ------------- Thresholds -------------

def load_thresholds(thresholds_json: Optional[str]) -> Dict[str, float]:
    if not thresholds_json:
        return {}
    with open(thresholds_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    # keys like "mastitis:25" -> float
    out = {}
    for k, v in data.items():
        try:
            out[str(k).lower()] = float(v)
        except Exception:
            pass
    return out

def get_threshold(thresholds_map: Dict[str, float], disease: str, os_level: int, default: float = 0.5) -> float:
    key = f"{disease.lower()}:{os_level}"
    return thresholds_map.get(key, default)

# ------------- Main audit -------------

def audit(args):
    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    plots_dir = outdir / "plots"
    ensure_dir(plots_dir)
    summary_dir = outdir / "summary"
    ensure_dir(summary_dir)

    thresholds = load_thresholds(args.thresholds_json)

    files = discover_prediction_files(data_root, args.diseases, args.oversampling, args.file_pattern)
    # Prepare holders
    overall_rows = []
    os_trend_rows = []
    fairness_group_tables = {col: [] for col in args.sensitive_cols}
    dist_shift_tables = {d: None for d in args.diseases}
    dfs_by_disease_os = {d: {} for d in args.diseases}

    # Load all data once
    for d in args.diseases:
        for o in args.oversampling:
            p = choose_best_file(files[d][o])
            if p is None:
                print(f"[WARN] No file for {d} at OS {o}%.")
                continue
            df = pd.read_csv(p)
            # Prepare y_pred if missing and y_prob present
            if "y_pred" not in df.columns and "y_prob" in df.columns:
                thr = get_threshold(thresholds, d, o, default=args.default_threshold)
                df["y_pred"] = (df["y_prob"] >= thr).astype(int)
            if "y_true" not in df.columns or "y_pred" not in df.columns:
                print(f"[WARN] Missing y_true or y_pred in {p}. Skipping metrics.")
            dfs_by_disease_os[d][o] = df

    # Per disease + OS metrics and plots
    for d in args.diseases:
        disease_os_levels = sorted([o for o in args.oversampling if o in dfs_by_disease_os[d]])
        # distribution shift across OS for features
        if args.features:
            dist_tbl = compare_os_distributions(
                {o: dfs_by_disease_os[d][o] for o in disease_os_levels},
                args.features
            )
            if dist_tbl is not None and not dist_tbl.empty:
                dist_tbl["disease"] = d
                dist_shift_tables[d] = dist_tbl

        # trend placeholders
        trend_acc = []; trend_f1 = []; trend_auc = []; trend_prevalence = []

        for o in disease_os_levels:
            df = dfs_by_disease_os[d][o]
            # Basic metrics
            if "y_true" in df and "y_pred" in df:
                y_true = df["y_true"].values
                y_pred = df["y_pred"].values
                y_prob = df["y_prob"].values if "y_prob" in df else None
                met = compute_basic_metrics(y_true, y_pred, y_prob)
                met.update({"disease": d, "oversampling": o})
                overall_rows.append(met)
                trend_acc.append(met["accuracy"])
                trend_f1.append(met["f1"])
                trend_auc.append(met["roc_auc"])
                trend_prevalence.append(met["prevalence"])

                # Calibration
                cal_path = plots_dir / f"calibration__{d}__os{o}.png"
                plot_calibration(y_true, y_prob, cal_path, f"Calibration: {d} @ OS {o}%")

                # Subgroup fairness
                for col in args.sensitive_cols:
                    if col in df.columns:
                        sub_tbl = subgroup_table(df, "y_true", "y_pred", "y_prob" if "y_prob" in df else None, col)
                        sub_tbl["disease"] = d
                        sub_tbl["oversampling"] = o
                        sub_tbl["group_col"] = col
                        fairness_group_tables[col].append(sub_tbl)

                        # group plots
                        for metric_name in ["selection_rate", "tpr", "fpr", "accuracy", "f1"]:
                            if metric_name in sub_tbl.columns:
                                outp = plots_dir / f"{metric_name}__by_{col}__{d}__os{o}.png"
                                plot_group_bars(sub_tbl, metric_name, "group", outp, f"{metric_name} by {col}: {d} @ OS {o}%")

                # OS trend plots (collect; plot once per disease)
            else:
                print(f"[WARN] Skipping metrics for {d} OS {o}% (no y_true/y_pred).")

        # Trend plots
        if disease_os_levels:
            # Fill NaNs with np.nan; lengths match oversampling levels present
            outp = plots_dir / f"trend_accuracy__{d}.png"
            plot_os_trend(disease_os_levels, trend_acc, outp, f"Accuracy vs OS: {d}", "Accuracy")
            outp = plots_dir / f"trend_f1__{d}.png"
            plot_os_trend(disease_os_levels, trend_f1, outp, f"F1 vs OS: {d}", "F1")
            outp = plots_dir / f"trend_auc__{d}.png"
            plot_os_trend(disease_os_levels, trend_auc, outp, f"ROC AUC vs OS: {d}", "ROC AUC")
            outp = plots_dir / f"trend_prev__{d}.png"
            plot_os_trend(disease_os_levels, trend_prevalence, outp, f"Prevalence vs OS: {d}", "Prevalence")

            # Add to trend table
            for idx, o in enumerate(disease_os_levels):
                os_trend_rows.append({
                    "disease": d,
                    "oversampling": o,
                    "accuracy": trend_acc[idx] if idx < len(trend_acc) else np.nan,
                    "f1": trend_f1[idx] if idx < len(trend_f1) else np.nan,
                    "roc_auc": trend_auc[idx] if idx < len(trend_auc) else np.nan,
                    "prevalence": trend_prevalence[idx] if idx < len(trend_prevalence) else np.nan,
                })

    # Write summaries
    if overall_rows:
        overall_df = pd.DataFrame(overall_rows)
        overall_df.to_csv(summary_dir / "overall_metrics.csv", index=False)

    if os_trend_rows:
        os_trend_df = pd.DataFrame(os_trend_rows)
        os_trend_df.to_csv(summary_dir / "os_trend_metrics.csv", index=False)

    for col, tables in fairness_group_tables.items():
        if tables:
            fair_df = pd.concat(tables, ignore_index=True)
            # fairness deltas per OS (DP diff, EO diff, DIR) computed here
            # group-level stats already in table; compute deltas per OS
            rows = []
            for (d, o), sub in fair_df.groupby(["disease", "oversampling"]):
                # Build arrays
                # For DP diff and DIR we need y_pred by group; we only have metrics per group not raw preds
                # We compute DP diff & DIR using group selection_rate values directly.
                try:
                    sr = sub.set_index("group")["selection_rate"].dropna()
                    dp_diff = sr.max() - sr.min() if not sr.empty else np.nan
                    dir_ratio = (sr.max() / sr.min()) if (len(sr)>0 and sr.min()>0) else np.nan
                except Exception:
                    dp_diff = np.nan; dir_ratio = np.nan
                # EO diff using TPR column
                try:
                    tpr = sub.set_index("group")["tpr"].dropna()
                    eo_diff = tpr.max() - tpr.min() if not tpr.empty else np.nan
                except Exception:
                    eo_diff = np.nan
                rows.append({"disease": d, "oversampling": o, "group_col": col, "dp_diff": dp_diff, "eo_diff": eo_diff, "dir_ratio": dir_ratio})
            pd.DataFrame(rows).to_csv(summary_dir / f"fairness_by_group__{col}.csv", index=False)
            fair_df.to_csv(summary_dir / f"group_metrics_table__{col}.csv", index=False)

    for d, tbl in dist_shift_tables.items():
        if tbl is not None and not tbl.empty:
            tbl.to_csv(summary_dir / f"distribution_shift__{d}.csv", index=False)

    # Report
    report_path = outdir / "bias_audit_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Bias & Fairness Audit Summary\n\n")
        if overall_rows:
            f.write("## Overall metrics (saved to `summary/overall_metrics.csv`)\n\n")
        if os_trend_rows:
            f.write("## OS trends (saved to `summary/os_trend_metrics.csv`)\n\n")
        if any(v is not None for v in dist_shift_tables.values()):
            f.write("## Distribution shifts (saved per disease as `summary/distribution_shift__{d}.csv`)\n\n")
        if any(len(v)>0 for v in fairness_group_tables.values()):
            f.write("## Fairness by subgroup (saved as `summary/fairness_by_group__{col}.csv`)\n\n")
        f.write("Plots are in the `plots/` folder.\n")

    print(f"✅ Audit complete. Outputs saved to: {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Bias & Fairness Audit for Disease Prediction")
    parser.add_argument("--data-root", type=str, required=True, help="Root folder containing per-OS prediction CSVs")
    parser.add_argument("--outdir", type=str, required=True, help="Folder to save audit outputs")
    parser.add_argument("--diseases", nargs="+", default=["mastitis","oestrus","lameness","calving"], help="Diseases to include")
    parser.add_argument("--oversampling", nargs="+", type=int, default=[0,5,10,15,20,25], help="Oversampling levels (%)")
    parser.add_argument("--sensitive-cols", nargs="+", default=["time_bucket","days_window"], help="Subgroup columns")
    parser.add_argument("--features", nargs="*", default=["EAT","REST","IN_ALLEYS","ACTIVITY_LEVEL"], help="Feature columns for shift tests")
    parser.add_argument("--thresholds-json", type=str, default=None, help="Path to thresholds JSON (keys like 'mastitis:25')")
    parser.add_argument("--default-threshold", type=float, default=0.5, help="Default threshold when JSON not provided")
    parser.add_argument("--file-pattern", type=str, default=None, help="Optional regex to restrict file discovery")
    args = parser.parse_args()
    audit(args)
