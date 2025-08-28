
import argparse
import os
import re
import sys
import numpy as np
import pandas as pd

# --- Lazy imports for optional libs ---
def _lazy_import_joblib():
    try:
        import joblib
        return joblib
    except Exception:
        return None

def _lazy_import_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except Exception:
        return None

def _lazy_import_xgboost():
    try:
        import xgboost as xgb
        return xgb
    except Exception:
        return None

def _lazy_import_torch():
    try:
        import torch
        return torch
    except Exception:
        return None

# --- Model IO ---
def load_model(path, model_name_hint=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".pkl", ".joblib"]:
        joblib = _lazy_import_joblib()
        if not joblib:
            raise RuntimeError("joblib not installed. pip install joblib")
        return ("sklearn_like", joblib.load(path))
    if ext == ".txt" and (model_name_hint or "").lower().startswith("lgbm"):
        lgb = _lazy_import_lightgbm()
        if not lgb:
            raise RuntimeError("lightgbm not installed. pip install lightgbm")
        return ("lightgbm_booster", lgb.Booster(model_file=path))
    if ext == ".json" and (model_name_hint or "").lower().startswith(("xgb","xgboost")):
        xgb = _lazy_import_xgboost()
        if not xgb:
            raise RuntimeError("xgboost not installed. pip install xgboost")
        booster = xgb.Booster()
        booster.load_model(path)
        return ("xgboost_booster", booster)
    if ext in [".pt", ".pth"]:
        torch = _lazy_import_torch()
        if not torch:
            raise RuntimeError("pytorch not installed. pip install torch")
        obj = torch.load(path, map_location="cpu")
        return ("torch", obj)
    # Fallback try joblib
    try:
        joblib = _lazy_import_joblib()
        if joblib:
            return ("sklearn_like", joblib.load(path))
    except Exception:
        pass
    raise ValueError(f"Unsupported model format: {path}")

def infer_feature_names(model_type, model_obj):
    names = None
    if model_type == "sklearn_like":
        names = getattr(model_obj, "feature_names_in_", None)
        if names is not None:
            names = list(names)
    elif model_type == "lightgbm_booster":
        try:
            names = model_obj.feature_name()
        except Exception:
            names = None
    elif model_type == "xgboost_booster":
        names = None
    elif model_type == "torch":
        names = getattr(model_obj, "feature_names_", None)
    return names

def run_predict(model_type, model_obj, Xdf):
    import numpy as np
    if model_type == "sklearn_like":
        if hasattr(model_obj, "predict_proba"):
            proba = model_obj.predict_proba(Xdf)[:, 1]
            label = (proba >= 0.5).astype(int)
            return float(proba[0]), int(label[0])
        if hasattr(model_obj, "decision_function"):
            scores = model_obj.decision_function(Xdf)
            proba = 1 / (1 + np.exp(-scores))
            label = (proba >= 0.5).astype(int)
            return float(proba[0]), int(label[0])
        label = model_obj.predict(Xdf)
        return None, int(np.asarray(label)[0])
    if model_type == "lightgbm_booster":
        proba = model_obj.predict(Xdf)
        if proba.ndim == 1:
            label = (proba >= 0.5).astype(int)
            return float(proba[0]), int(label[0])
        label = np.argmax(proba, axis=1)
        p = proba[:, 1] if proba.shape[1] > 1 else None
        return (float(p[0]) if p is not None else None), int(label[0])
    if model_type == "xgboost_booster":
        xgb = _lazy_import_xgboost()
        dmat = xgb.DMatrix(Xdf)
        proba = model_obj.predict(dmat)
        if proba.ndim == 1:
            label = (proba >= 0.5).astype(int)
            return float(proba[0]), int(label[0])
        label = np.argmax(proba, axis=1)
        p = proba[:, 1] if proba.shape[1] > 1 else None
        return (float(p[0]) if p is not None else None), int(label[0])
    if model_type == "torch":
        torch = _lazy_import_torch()
        model_obj.eval() if hasattr(model_obj, "eval") else None
        with torch.no_grad():
            t = torch.tensor(Xdf.values, dtype=torch.float32)
            out = model_obj(t)
            if out.ndim == 2 and out.shape[1] == 1:
                proba = torch.sigmoid(out).cpu().numpy().ravel()
                label = (proba >= 0.5).astype(int)
                return float(proba[0]), int(label[0])
            if out.ndim == 2:
                proba_full = torch.softmax(out, dim=1).cpu().numpy()
                label = np.argmax(proba_full, axis=1)
                p = proba_full[:, 1] if proba_full.shape[1] > 1 else None
                return (float(p[0]) if p is not None else None), int(label[0])
            label = out.cpu().numpy().ravel()
            return None, int(label[0])
    raise ValueError("Unsupported model type")

# --- Input collection for D days x 3 buckets ---
BUCKETS = [(0, "0-8"), (1, "9-16"), (2, "17-24")]

def collect_multi_day_inputs(days):
    # Returns dicts of shape: {bucket_index: [values per day]}
    eat = {b: [] for b,_ in BUCKETS}
    rest = {b: [] for b,_ in BUCKETS}
    alleys = {b: [] for b,_ in BUCKETS}

    print(f"\nEnter seconds per day for each bucket. Days={days}")
    print("Tip: press Enter to reuse the previous value for faster entry.\n")

    prev = {"eat": None, "rest": None, "alleys": None}
    for d in range(1, days+1):
        print(f"-- Day {d} --")
        for b, label in BUCKETS:
            def ask_int(prompt, prev_key):
                s = input(prompt).strip()
                if s == "" and prev[prev_key] is not None:
                    return prev[prev_key]
                val = int(s)
                prev[prev_key] = val
                return val
            e = ask_int(f"  EAT seconds ({label}): ", "eat")
            r = ask_int(f"  REST seconds ({label}): ", "rest")
            a = ask_int(f"  IN_ALLEYS seconds ({label}): ", "alleys")
            eat[b].append(e)
            rest[b].append(r)
            alleys[b].append(a)
    return eat, rest, alleys

def summarize_inputs(eat, rest, alleys):
    # Compute per-bucket mean/std and overall mean/std
    def stats(d):
        # d: dict bucket->list
        means = {k: float(np.mean(v)) if len(v)>0 else 0.0 for k,v in d.items()}
        stds  = {k: float(np.std(v, ddof=1)) if len(v)>1 else 0.0 for k,v in d.items()}
        all_vals = [x for arr in d.values() for x in arr]
        overall_mean = float(np.mean(all_vals)) if len(all_vals)>0 else 0.0
        overall_std  = float(np.std(all_vals, ddof=1)) if len(all_vals)>1 else 0.0
        return means, stds, overall_mean, overall_std

    e_mean, e_std, e_all_mean, e_all_std = stats(eat)
    r_mean, r_std, r_all_mean, r_all_std = stats(rest)
    a_mean, a_std, a_all_mean, a_all_std = stats(alleys)

    # Simple ratios using overall means
    denom = max(e_all_mean + r_all_mean, 1e-6)
    eat_ratio = e_all_mean / denom
    rest_ratio = r_all_mean / denom

    summary = {
        "eat_mean": e_mean, "eat_std": e_std, "eat_all_mean": e_all_mean, "eat_all_std": e_all_std,
        "rest_mean": r_mean, "rest_std": r_std, "rest_all_mean": r_all_mean, "rest_all_std": r_all_std,
        "alleys_mean": a_mean, "alleys_std": a_std, "alleys_all_mean": a_all_mean, "alleys_all_std": a_all_std,
        "eat_ratio": eat_ratio, "rest_ratio": rest_ratio
    }
    return summary

# --- Feature construction ---
def fill_features(feature_names, days, summary):
    row = {fn: 0.0 for fn in feature_names}

    # days/window
    for fn in feature_names:
        s = fn.lower()
        if re.search(r"(^|_)days?($|_)", s) or "window" in s or "ndays" in s:
            row[fn] = float(days)

    # helpers to set values by keywords
    def set_value_by_keywords(value, *keywords):
        for fn in feature_names:
            s = fn.lower().replace(" ", "")
            if all(k in s for k in keywords):
                row[fn] = float(value)

    # overall means
    set_value_by_keywords(summary["eat_all_mean"], "eat")
    set_value_by_keywords(summary["rest_all_mean"], "rest")
    set_value_by_keywords(summary["alleys_all_mean"], "alley")  # matches in_alleys too

    # bucket-specific features, e.g., EAT_0_8
    for fn in feature_names:
        fl = fn.lower()
        # eat
        if "eat" in fl:
            if "0_8" in fl or "_0to8" in fl:
                row[fn] = summary["eat_mean"].get(0, 0.0)
            elif "9_16" in fl or "_9to16" in fl:
                row[fn] = summary["eat_mean"].get(1, 0.0)
            elif "17_24" in fl or "_17to24" in fl:
                row[fn] = summary["eat_mean"].get(2, 0.0)
        # rest
        if "rest" in fl:
            if "0_8" in fl or "_0to8" in fl:
                row[fn] = summary["rest_mean"].get(0, 0.0)
            elif "9_16" in fl or "_9to16" in fl:
                row[fn] = summary["rest_mean"].get(1, 0.0)
            elif "17_24" in fl or "_17to24" in fl:
                row[fn] = summary["rest_mean"].get(2, 0.0)
        # alleys
        if "alley" in fl:
            if "0_8" in fl or "_0to8" in fl:
                row[fn] = summary["alleys_mean"].get(0, 0.0)
            elif "9_16" in fl or "_9to16" in fl:
                row[fn] = summary["alleys_mean"].get(1, 0.0)
            elif "17_24" in fl or "_17to24" in fl:
                row[fn] = summary["alleys_mean"].get(2, 0.0)

        # rollmean/rollstd handling
        if "rollmean" in fl:
            if "eat" in fl:
                row[fn] = summary["eat_all_mean"]
            elif "rest" in fl:
                row[fn] = summary["rest_all_mean"]
            elif "alley" in fl:
                row[fn] = summary["alleys_all_mean"]
        if "rollstd" in fl:
            if "eat" in fl:
                row[fn] = summary["eat_all_std"]
            elif "rest" in fl:
                row[fn] = summary["rest_all_std"]
            elif "alley" in fl:
                row[fn] = summary["alleys_all_std"]

        # ratios
        if "eat_ratio" in fl:
            row[fn] = summary["eat_ratio"]
        if "rest_ratio" in fl:
            row[fn] = summary["rest_ratio"]

    return pd.DataFrame([row])

def main():
    ap = argparse.ArgumentParser(description="Multi-day, 3-bucket interactive predictor")
    ap.add_argument("--table", required=True, help="Path to master comparison table CSV")
    ap.add_argument("--oversampling", default=None, help="Oversampling level to use (e.g., '25%'). If omitted, you will be prompted.")
    ap.add_argument("--days", type=int, default=None, help="Number of days (3-20). If omitted, you will be prompted.")
    ap.add_argument("--out", default=None, help="Optional CSV to save results")
    args = ap.parse_args()

    # Load table
    df = pd.read_csv(args.table)
    df.columns = [c.strip() for c in df.columns]
    rename = {
        'Oversampling %': 'oversampling_pct',
        'Model': 'model',
        'Disease': 'disease',
        'Precision (Class 1)': 'precision_pos',
        'Recall (Class 1)': 'recall_pos',
        'F1-score (Class 1)': 'f1_pos',
        'Accuracy': 'accuracy',
        'Model Path': 'model_path'
    }
    for k,v in rename.items():
        if k in df.columns:
            df.rename(columns={k:v}, inplace=True)

    avail_os = sorted(df['oversampling_pct'].unique().tolist())
    # Oversampling
    oversampling = args.oversampling
    if oversampling is None:
        print(f"Available oversampling levels: {', '.join(avail_os)}")
        oversampling = input("Select oversampling (e.g. 0%, 5%, 10%, 15%, 20%, 25%): ").strip()
    if oversampling not in avail_os:
        print(f"[ERROR] oversampling must be one of: {avail_os}")
        sys.exit(2)

    # Days
    if args.days is None:
        days = int(input("Enter number of days (3-20): ").strip())
    else:
        days = args.days
    if not (3 <= days <= 20):
        print("[ERROR] days must be in [3, 20]")
        sys.exit(2)

    # Collect inputs
    eat, rest, alleys = collect_multi_day_inputs(days)
    summary = summarize_inputs(eat, rest, alleys)

    # Choose best models per disease
    sub = df[df['oversampling_pct'] == oversampling].copy()
    if sub.empty:
        print(f"[ERROR] No rows for oversampling {oversampling}")
        sys.exit(2)
    sub = sub.sort_values(['disease','f1_pos'], ascending=[True, False]) \
             .groupby('disease', as_index=False).first()

    print("\nSelected best models at", oversampling)
    print(sub[['disease','model','f1_pos','accuracy','model_path']].to_string(index=False))

    # Predict
    results = []
    for _, r in sub.iterrows():
        disease = r['disease']
        mpath = r['model_path']
        mname = r['model']

        if not os.path.exists(mpath):
            print(f"[ERROR] Missing model file for {disease}: {mpath}")
            results.append({'disease': disease, 'status': 'missing_model', 'path': mpath})
            continue

        try:
            mtype, mobj = load_model(mpath, model_name_hint=mname)
        except Exception as e:
            print(f"[ERROR] Could not load model for {disease}: {e}")
            results.append({'disease': disease, 'status': 'load_failed', 'error': str(e)})
            continue

        feat_names = infer_feature_names(mtype, mobj)
        if not feat_names:
            print(f"[WARN] Feature names unavailable for {disease}. Please provide a model with feature names.")
            results.append({'disease': disease, 'status': 'no_feature_names'})
            continue

        X = fill_features(feat_names, days, summary)

        try:
            proba, pred = run_predict(mtype, mobj, X)
            results.append({
                'disease': disease,
                'prediction': int(pred),
                'probability_1': (float(proba) if proba is not None else None),
                'status': 'ok'
            })
            print(f"[OK] {disease}: pred={pred} prob={proba}")
        except Exception as e:
            print(f"[ERROR] Prediction failed for {disease}: {e}")
            results.append({'disease': disease, 'status': 'predict_failed', 'error': str(e)})

    if args.out:
        pd.DataFrame(results).to_csv(args.out, index=False)
        print(f"[SAVED] Results -> {args.out}")

if __name__ == "__main__":
    main()
