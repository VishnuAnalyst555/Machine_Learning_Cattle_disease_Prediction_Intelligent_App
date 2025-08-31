#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calibrate_and_threshold.py

Provides two subcommands:

1) FIT
   Learn per-(disease, oversampling) probability calibrators from a labeled CSV of predictions,
   then optionally pick data-driven thresholds (per disease & OS).

   Required input CSV columns:
     - disease (str)
     - oversampling (str, e.g., "0%", "5%")
     - y_true (0/1)
     - probability_1 (float, raw/un-calibrated)

  
     python calibrate_and_threshold.py fit \
       --in preds_labeled.csv \
       --outdir calibrators \
       --method platt \
       --pick-threshold youden

   Methods:
     - platt   : logistic regression calibration
    

   Threshold pickers:
     - none    : do not compute thresholds
     - youden  : maximize TPR - FPR
     - f1      : maximize F1 score on positives (class 1)

2) APPLY
   Calibrate and threshold raw predictions (without y_true), enforce one-hot exclusivity & no-disease guard.

   Required input CSV columns:
     - __source__ (optional) file tag
     - oversampling (str)
     - disease (str)
     - probability_1 (float)
     - (optional) prediction (will be ignored and recomputed)

   
     python calibrate_and_threshold.py apply \
       --in results_case_all_*.csv \
       --calibrators calibrators \
       --thresholds thresholds.json \
       --onehot \
       --out corrected_results.csv

Outputs:
  - Calibrators saved as joblib files under outdir, one per (disease, OS).
  - Thresholds (optional) saved to a JSON file (if --save-thresholds PATH given).
  - Corrected CSV in APPLY mode.
"""
import argparse, glob, json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd

def _ensure_joblib():
    try:
        import joblib
    except Exception:
        raise SystemExit("ERROR: joblib is required. Please `pip install joblib`.")
    return joblib

def _ensure_sklearn():
    try:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score, roc_curve
    except Exception:
        raise SystemExit("ERROR: scikit-learn is required. Please `pip install scikit-learn`.")
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_curve
    return IsotonicRegression, LogisticRegression, f1_score, roc_curve

# ------------- utilities -------------
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def clip01(x):
    return np.minimum(1.0, np.maximum(0.0, x))

def pick_threshold(y_true, p_cal, method="youden"):
    # returns threshold and a dict of stats
    from sklearn.metrics import f1_score, roc_curve, precision_recall_curve, auc
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p_cal).astype(float)

    # guard
    if len(np.unique(y_true))<2:
        # Cannot compute ROC; fallback to 0.5
        return 0.5, {"note":"single-class labels; defaulted to 0.5"}

    if method == "youden":
        fpr, tpr, thr = roc_curve(y_true, p)
        j = tpr - fpr
        k = int(np.argmax(j))
        return float(thr[k]), {"tpr":float(tpr[k]), "fpr":float(fpr[k])}
    elif method == "f1":
        # sweep thresholds on unique probs
        best_f1, best_t = -1.0, 0.5
        ts = np.unique(np.concatenate([p, [0.5]]))
        for t in ts:
            yhat = (p >= t).astype(int)
            f1 = f1_score(y_true, yhat, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return best_t, {"f1":float(best_f1)}
    else:
        return 0.5, {"note":"method=none; default 0.5"}

def load_thresholds(path_json):
    if not path_json or not os.path.exists(path_json):
        return {}
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    # normalize keys
    out = {}
    for disease, mapping in data.items():
        dd = disease.strip().lower()
        out[dd] = {}
        for os_level, thr in mapping.items():
            out[dd][str(os_level)] = float(thr)
    return out

def save_thresholds(path_json, thresholds_dict):
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, indent=2)

# ------------- FIT -------------
def cmd_fit(args):
    joblib = _ensure_joblib()
    IsotonicRegression, LogisticRegression, f1_score, roc_curve = _ensure_sklearn()

    # Load labeled predictions
    df = pd.read_csv(args.infile)
    needed = {"disease","oversampling","y_true","probability_1"}
    if not needed.issubset(df.columns):
        raise SystemExit(f"ERROR: Input CSV must include columns: {needed}")

    df["disease"] = df["disease"].str.lower().str.strip()
    df["oversampling"] = df["oversampling"].astype(str)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    thresholds_global = load_thresholds(args.thresholds) if args.thresholds else {}

    for (disease, os_level), grp in df.groupby(["disease","oversampling"], sort=False):
        y = grp["y_true"].astype(int).values
        p = grp["probability_1"].astype(float).values

        # Fit calibrator
        if args.method == "platt":
            # Logistic regression on raw probabilities
            # Avoid degenerate cases by adding small epsilon
            eps = 1e-6
            X = np.clip(p, eps, 1-eps).reshape(-1, 1)
            X = np.log(X/(1-X))  # logit transform aids linear separability
            lr = LogisticRegression(solver="lbfgs")
            lr.fit(X, y)
            model = ("platt", lr)
        elif args.method == "isotonic":
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(p, y)
            model = ("isotonic", ir)
        else:
            raise SystemExit("Unknown --method. Use: platt or isotonic.")

        # Save calibrator
        save_path = outdir / f"cal_{disease}__{os_level.replace('%','pct').replace('/','_')}.joblib"
        joblib.dump(model, save_path)

        # Compute calibrated probabilities on this data
        if model[0]=="platt":
            X = np.clip(p, 1e-6, 1-1e-6).reshape(-1,1)
            X = np.log(X/(1-X))
            p_cal = model[1].predict_proba(X)[:,1]
        else:
            p_cal = model[1].transform(p)
        p_cal = clip01(p_cal)

        # Threshold selection if requested
        if args.pick_threshold != "none":
            thr, stats = pick_threshold(y, p_cal, method=args.pick_threshold)
            thresholds_global.setdefault(disease, {})[os_level] = float(thr)

    # Save thresholds if requested
    if args.save_thresholds:
        save_thresholds(args.save_thresholds, thresholds_global)
        print(f"[SAVED] thresholds -> {args.save_thresholds}")

    print(f"[DONE] Calibrators saved under: {outdir}")

# ------------- APPLY -------------
def _load_calibrator(joblib, cal_dir, disease, os_level):
    # Try exact match; otherwise try fallback with stripped %
    candidates = [
        cal_dir / f"cal_{disease}__{os_level.replace('%','pct').replace('/','_')}.joblib",
        cal_dir / f"cal_{disease}__{os_level}.joblib",
    ]
    for c in candidates:
        if c.exists():
            return joblib.load(c)
    return None

def _apply_calibration(model, p_raw):
    kind, obj = model
    p_raw = np.asarray(p_raw, dtype=float)
    if kind == "platt":
        X = np.clip(p_raw, 1e-6, 1-1e-6).reshape(-1, 1)
        X = np.log(X/(1-X))
        p_cal = obj.predict_proba(X)[:,1]
    elif kind == "isotonic":
        p_cal = obj.transform(p_raw)
    else:
        p_cal = p_raw
    return clip01(p_cal)

def cmd_apply(args):
    joblib = _ensure_joblib()

    # Collect input files
    files = []
    for pat in args.infiles:
        files.extend(glob.glob(pat))
    if not files:
        raise SystemExit("ERROR: No input files matched.")

    # Load thresholds & calibrators
    thresholds = load_thresholds(args.thresholds) if args.thresholds else {}
    cal_dir = Path(args.calibrators) if args.calibrators else None

    rows = []
    for f in files:
        df = pd.read_csv(f)
        if "__source__" not in df.columns:
            df["__source__"] = os.path.basename(f)
        df["disease"] = df["disease"].str.lower().str.strip()
        df["oversampling"] = df["oversampling"].astype(str)

        # Calibrate each (disease, OS)
        pcal = []
        for (disease, os_level), grp in df.groupby(["disease","oversampling"], sort=False):
            raw = grp["probability_1"].values.astype(float)
            if cal_dir is not None:
                model = _load_calibrator(joblib, cal_dir, disease, os_level)
            else:
                model = None
            if model is None:
                p = clip01(raw)  # no calibration available
            else:
                p = _apply_calibration(model, raw)
            pcal.append(pd.DataFrame({
                "__source__": grp["__source__"].values,
                "oversampling": os_level,
                "disease": disease,
                "probability_raw": raw,
                "probability_cal": p,
            }))
        pcal = pd.concat(pcal, ignore_index=True) if pcal else pd.DataFrame()

        # Merge back
        df = df.drop(columns=[c for c in ["probability_cal","probability_raw"] if c in df.columns])
        df = df.merge(pcal, on=["__source__","oversampling","disease"], how="left")

        # Apply thresholds
        preds = []
        for (src, os_level), grp in df.groupby(["__source__","oversampling"], sort=False):
            g = grp.copy()
            g["threshold"] = [
                thresholds.get(d, {}).get(os_level, 0.5) for d in g["disease"].values
            ]
            g["prediction_cal"] = (g["probability_cal"] >= g["threshold"]).astype(int)

            if args.onehot:
                # Keep only the highest p among those >= threshold; else zeros
                idx_pos = np.where(g["prediction_cal"].values==1)[0]
                if len(idx_pos) >= 2:
                    # Multiple positives -> keep max prob only
                    k = int(np.argmax(g["probability_cal"].values[idx_pos]))
                    keep_idx = idx_pos[k]
                    mask = np.zeros(len(g), dtype=int)
                    mask[keep_idx] = 1
                    g["prediction_onehot"] = mask
                elif len(idx_pos) == 1:
                    mask = np.zeros(len(g), dtype=int)
                    mask[idx_pos[0]] = 1
                    g["prediction_onehot"] = mask
                else:
                    g["prediction_onehot"] = 0
            preds.append(g)
        out_df = pd.concat(preds, ignore_index=True)
        rows.append(out_df)

    final = pd.concat(rows, ignore_index=True)

    # Decide which prediction to output
    col_pred = "prediction_onehot" if args.onehot else "prediction_cal"
    final.rename(columns={col_pred:"prediction"}, inplace=True)

    # Order columns
    cols = ["__source__","oversampling","disease","probability_raw","probability_cal","threshold","prediction"]
    cols = [c for c in cols if c in final.columns] + [c for c in final.columns if c not in cols]
    final = final[cols]

    # Save
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    final.to_csv(args.out, index=False)
    print(f"[SAVED] corrected -> {args.out}")

def main():
    ap = argparse.ArgumentParser(description="Calibration + Thresholding for disease predictions")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # fit
    ap_fit = sub.add_parser("fit", help="Fit calibrators from labeled predictions")
    ap_fit.add_argument("--in", dest="infile", required=True, help="CSV with disease, oversampling, y_true, probability_1")
    ap_fit.add_argument("--outdir", required=True, help="Directory to save calibrators")
    ap_fit.add_argument("--method", choices=["platt","isotonic"], default="platt")
    ap_fit.add_argument("--pick-threshold", choices=["none","youden","f1"], default="youden")
    ap_fit.add_argument("--thresholds", default=None, help="Existing thresholds JSON to extend (optional)")
    ap_fit.add_argument("--save-thresholds", default=None, help="Path to write thresholds JSON (optional)")
    ap_fit.set_defaults(func=cmd_fit)

    # apply
    ap_apply = sub.add_parser("apply", help="Apply calibration and thresholds to raw predictions")
    ap_apply.add_argument("--in", dest="infiles", nargs="+", required=True, help="One or more CSVs (glob patterns ok)")
    ap_apply.add_argument("--calibrators", default=None, help="Directory containing .joblib calibrators")
    ap_apply.add_argument("--thresholds", default=None, help="Thresholds JSON")
    ap_apply.add_argument("--onehot", action="store_true", help="Enforce one-hot exclusivity (keep highest above threshold)")
    ap_apply.add_argument("--out", required=True, help="Output CSV path")
    ap_apply.set_defaults(func=cmd_apply)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
