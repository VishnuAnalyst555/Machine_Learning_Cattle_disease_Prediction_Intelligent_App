#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Find 3-day windows in your dataset that are:
  - NONE: no disease present across the window
  - OESTRUS_ONLY / CALVING_ONLY / LAMENESS_ONLY / MASTITIS_ONLY: exactly one disease present

For each selected window, save one CSV with minutes per day × 8-hour bucket:
  columns = date, bucket (0-8, 9-16, 17-24), EAT, REST, IN_ALLEYS  [minutes]

Usage (example):
  python C:/Users/vishn/find_and_extract_windows.py ^
    --dataset "C:/Users/vishn/full_data_unhealthy_imputed_reduced_enhanced.csv" ^
    --days 3 ^
    --per-class 3 ^
    --mode mean ^
    --outdir "C:/Users/vishn/cases"

Notes:
- Works with datasets having either 'hour_bin' (0/8/16) or 'hour' (0..23).
- Assumes EAT/REST/IN_ALLEYS are in SECONDS; this script converts to MINUTES.
- If your file is already reduced (one row per date × hour_bin with mean per hour),
  use --mode mean (recommended to match your reduced example). For hourly raw data, use --mode sum.
"""

import argparse, os
import pandas as pd
import numpy as np
from collections import defaultdict

DISEASES = ["oestrus","calving","lameness","mastitis"]
BUCKET_LABELS = ["0-8","9-16","17-24"]
BIN_MAP = {0:"0-8", 8:"9-16", 16:"17-24"}

def parse_date(df):
    for kwargs in (dict(dayfirst=True, errors="coerce"),
                   dict(errors="coerce")):
        try:
            df["date"] = pd.to_datetime(df["date"], **kwargs)
        except Exception:
            continue
    return df

def add_bucket(df):
    if "hour_bin" in df.columns:
        df["bucket"] = df["hour_bin"].astype(int).map(BIN_MAP)
    elif "hour" in df.columns:
        hb = (df["hour"].astype(int) // 8) * 8
        df["bucket"] = hb.map(BIN_MAP)
    else:
        raise ValueError("Dataset must have 'hour_bin' or 'hour'.")
    return df

def window_labels(daily_df, start_date, days):
    dates = pd.date_range(start_date, periods=days, freq="D")
    sub = daily_df.loc[daily_df.index.isin(dates)]
    if len(sub) < days:
        return None  # incomplete window
    w = sub[DISEASES].max(axis=0).astype(int)   # ANY day positive
    return w

def extract_minutes(df, cow, start_date, days, mode):
    # take rows for window
    end_date = start_date + pd.Timedelta(days=days-1)
    sub = df[(df["cow"]==cow) &
             (df["date"]>=start_date) &
             (df["date"]<=end_date)].copy()
    if sub.empty:
        return None

    agg_fn = mode  # 'sum' or 'mean'
    out = (sub.groupby(["date","bucket"], as_index=False)
             .agg({"EAT":agg_fn, "REST":agg_fn, "IN_ALLEYS":agg_fn}))
    # seconds -> minutes
    for col in ["EAT","REST","IN_ALLEYS"]:
        out[col] = (out[col].astype(float) / 60.0).clip(lower=0.0)

    # ensure all buckets per day
    idx = pd.MultiIndex.from_product([pd.date_range(start_date, end_date), BUCKET_LABELS],
                                     names=["date","bucket"])
    out = out.set_index(["date","bucket"]).reindex(idx).fillna(0.0).reset_index()
    # integers for convenience
    out[["EAT","REST","IN_ALLEYS"]] = out[["EAT","REST","IN_ALLEYS"]].round(0).astype(int)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--per-class", type=int, default=3, help="How many windows to save for each class")
    ap.add_argument("--mode", choices=["sum","mean"], default="mean",
                    help="Aggregate within bucket if hourly data. For reduced data, 'mean' matches your table.")
    ap.add_argument("--cow", type=int, default=None, help="Optional: restrict to a single cow")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.dataset)
    df.columns = [c.strip() for c in df.columns]
    needed = {"cow","date","EAT","REST","IN_ALLEYS"} | set(DISEASES)
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df = parse_date(df)
    df = add_bucket(df)

    # daily labels per cow (max over buckets each day)
    daily = (df.groupby(["cow","date"])[DISEASES].max().astype(int)
               .reset_index().set_index("date"))

    found = defaultdict(list)  # class_name -> list of (cow,start_date)
    cows = [args.cow] if args.cow is not None else sorted(df["cow"].unique())

    for cow in cows:
        daily_cow = daily[daily["cow"]==cow].drop(columns=["cow"]).sort_index()
        unique_dates = daily_cow.index.sort_values().unique()
        for i in range(0, len(unique_dates) - args.days + 1):
            s = unique_dates[i]
            wlab = window_labels(daily_cow, s, args.days)
            if wlab is None:
                continue
            vec = wlab.values
            if vec.sum() == 0:
                cls = "NONE"
            elif vec.sum() == 1:
                # exactly one disease present
                dname = DISEASES[int(np.argmax(vec))]
                cls = f"{dname.upper()}_ONLY"
            else:
                continue  # skip windows with multiple diseases

            if len(found[cls]) >= args.per_class:
                continue

            found[cls].append((cow, s))

            # extract minutes and save CSV
            minutes = extract_minutes(df, cow, s, args.days, args.mode)
            if minutes is None:
                continue
            start_str = pd.to_datetime(s).date().isoformat()
            outpath = os.path.join(
                args.outdir,
                f"{cls}_cow{cow}_{start_str}_{args.days}d.csv"
            )
            minutes.to_csv(outpath, index=False)
            print(f"[SAVED] {cls:14s} | cow={cow} start={start_str} days={args.days} -> {outpath}")

            # stop early if we already have enough of all classes
            if all(len(found[c]) >= args.per_class for c in ["NONE",
                                                             "OESTRUS_ONLY",
                                                             "CALVING_ONLY",
                                                             "LAMENESS_ONLY",
                                                             "MASTITIS_ONLY"] if c in found):
                pass  # continue scanning; remove this 'pass' to stop early

    # manifest
    rows=[]
    for cls, lst in found.items():
        for cow, s in lst:
            rows.append({"class":cls, "cow":cow, "start":pd.to_datetime(s).date().isoformat(),
                         "days":args.days, "csv":os.path.join(args.outdir,
                         f"{cls}_cow{cow}_{pd.to_datetime(s).date().isoformat()}_{args.days}d.csv")})
    if rows:
        man = pd.DataFrame(rows)
        manifest = os.path.join(args.outdir, f"manifest_{args.days}d.csv")
        man.to_csv(manifest, index=False)
        print(f"\n[SAVED] Manifest -> {manifest}")
    else:
        print("\n[INFO] No matching windows found. Try different cow, more days, or different mode.")

if __name__ == "__main__":
    main()
