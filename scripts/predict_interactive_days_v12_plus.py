#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_interactive_days_v12.py  (single-winner & multi-winner + hybrid gating)

- Hybrid ensemble: sklearn (.pkl/.joblib), XGBoost (.json), LightGBM (.txt),
  TabNet (.zip), PyTorch LSTM/TCN (.pth) w/ inline or sidecar arch metadata.
- Robust master table header normalization; excludes 0% oversampling.
- Uses known-case CSVs for Platt calibration + Youden threshold per disease@OS.
- Inputs are mean minutes PER HOUR by bucket (0–8, 9–16, 17–24) across N days.
- Optional hybrid gating: in a hybrid JSON you can add:
      "min_members": 2, "member_thr": 0.50
  to require at least 2 members each ≥ 0.50 before letting the ensemble fire.
- Default = single winner (one-hot). Use --multi to allow multiple winners.
- Writes Excel (chosen_os, predictions, model_debug, summary).
"""

import argparse, os, json, warnings, glob
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# ---------- lazy imports ----------
def _joblib():
    try: import joblib; return joblib
    except: return None
def _xgb():
    try: import xgboost as xgb; return xgb
    except: return None
def _lgb():
    try: import lightgbm as lgb; return lgb
    except: return None
def _tabnet():
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        return TabNetClassifier
    except: return None
def _torch():
    try:
        import torch, torch.nn as nn
        return torch, nn
    except: return None, None

# ---------- helpers ----------
def norm_disease(s):
    s = str(s).strip().lower()
    if s == "mastits": return "mastitis"
    return s

def combine_probs(probs, weights=None, combiner="weighted_mean"):
    """NaN-safe combiner."""
    arr = np.array(probs, dtype=float)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return np.nan
    arr = arr[mask]
    if combiner == "max":
        return float(np.max(arr))
    if combiner in ("mean","avg","average"):
        return float(np.mean(arr))
    # weighted mean (default)
    if weights is None:
        w = np.ones_like(arr) / len(arr)
    else:
        w_full = np.array(weights, dtype=float)
        w = w_full[mask] if len(w_full) == len(probs) else np.ones_like(arr)
        s = w.sum()
        w = (w / s) if s != 0 else (np.ones_like(arr) / len(arr))
    return float(np.dot(arr, w))

def is_hybrid_config(path):
    name = os.path.basename(str(path)).lower()
    return name.endswith((".hybrid.json",".hyb.json",".hybrid",".hyb")) or ("hybrid" in name and name.endswith(".json"))

def load_hybrid_config(path):
    with open(path,"r",encoding="utf-8-sig") as f:
        cfg=json.load(f)
    if not isinstance(cfg,dict) or "members" not in cfg:
        raise ValueError("Invalid hybrid config (no 'members').")
    return cfg.get("combiner","weighted_mean"), cfg["members"], cfg.get("threshold", None), cfg

def read_meta_json(meta_path):
    try:
        with open(meta_path,"r",encoding="utf-8-sig") as f: return json.load(f)
    except: return {}

def infer_feature_names(mtype, mobj):
    if mtype=="sklearn":
        return getattr(mobj,"feature_names_in_",None)
    if mtype=="lgbm":
        try: return mobj.feature_name()
        except: return None
    return None

def align_X_to_names(X, names):
    if names is None: return X
    row = {n: 0.0 for n in names}
    for col in X.columns:
        for n in names:
            if n.lower()==col.lower(): row[n]=float(X.iloc[0][col])
    return pd.DataFrame([row])

# ---------- model I/O ----------
def run_predict(mtype, mobj, X):
    if mtype=="sklearn":
        if hasattr(mobj,"predict_proba"): return float(mobj.predict_proba(X)[:,1][0])
        if hasattr(mobj,"decision_function"):
            s=mobj.decision_function(X); return float(1/(1+np.exp(-s[0])))
        return float(mobj.predict(X)[0])
    if mtype=="lgbm":
        p=mobj.predict(X); return float(p[0] if np.ndim(p)==1 else p[0,1])
    if mtype=="xgb":
        xgb=_xgb(); d=xgb.DMatrix(X); p=mobj.predict(d); return float(p[0] if np.ndim(p)==1 else p[0,1])
    if mtype=="tabnet":
        p=mobj.predict_proba(X.values.astype(np.float32))[:,1]; return float(p[0])
    if mtype in ("lstm","tcn"):
        torch, nn = _torch()
        mobj.eval()
        with torch.no_grad():
            arr = X.values.astype(np.float32)
            trials = []
            if mtype == "lstm":
                # (A) [B,T,F]
                trials.append(torch.from_numpy(arr[np.newaxis, 1:2, :]))
                # (B) [B,F]
                trials.append(torch.from_numpy(arr[np.newaxis, :]))
                # (C) [T,B,F]
                trials.append(torch.from_numpy(arr[np.newaxis, np.newaxis, :]).permute(1,0,2))
            else:
                # TCN common shapes
                trials.append(torch.from_numpy(arr[np.newaxis, np.newaxis, :]))
                trials.append(torch.from_numpy(arr[np.newaxis, :]))
            last_err = None
            for tens in trials:
                try:
                    out = mobj(tens)
                    val = float(out.squeeze().item())
                    return float(1.0 / (1.0 + np.exp(-val)))
                except Exception as e:
                    last_err = e
            raise last_err
    raise ValueError("Unsupported model type")

def try_load_model(path, member_type=None, arch=None):
    ext = os.path.splitext(path)[1].lower()
    mt  = (member_type or "").lower()
    jb  = _joblib()

    # sklearn default
    if ext in (".pkl",".joblib") and mt not in ("xgb","lgbm","tabnet","lstm","tcn"):
        if jb is None: raise RuntimeError("joblib not installed")
        return "sklearn", jb.load(path)

    # xgboost
    if ext==".json" or mt=="xgb":
        xgb=_xgb()
        if not xgb: raise RuntimeError("xgboost not installed")
        booster=xgb.Booster(); booster.load_model(path)
        return "xgb", booster

    # lightgbm
    if ext in (".txt",".lgb") or mt=="lgbm":
        lgb=_lgb()
        if not lgb: raise RuntimeError("lightgbm not installed")
        booster=lgb.Booster(model_file=path)
        return "lgbm", booster

    # tabnet
    if ext==".zip" or mt=="tabnet":
        TabNetClassifier=_tabnet()
        if TabNetClassifier is None: raise RuntimeError("pytorch-tabnet not installed")
        m = TabNetClassifier(); m.load_model(path)
        return "tabnet", m

    # pytorch LSTM/TCN
    if ext==".pth" or mt in ("lstm","tcn"):
        torch,nn=_torch()
        if torch is None: raise RuntimeError("torch not installed")
        meta = {}
        if isinstance(arch, dict): meta = arch
        else: meta = read_meta_json(path + ".meta.json")

        in_size   = int(meta.get("input_size", 8))
        hid       = int(meta.get("hidden_size", 64))
        n_layers  = int(meta.get("num_layers", 1))
        dropout   = float(meta.get("dropout", 0.0))

        if mt=="tcn" or meta.get("type","lstm")=="tcn":
            ksize = int(meta.get("kernel_size",3))
            dil   = int(meta.get("dilation",1))
            class Chomp1d(nn.Module):
                def __init__(self,c): super().__init__(); self.c=c
                def forward(self,x): return x[:,:,:-self.c].contiguous()
            class TemporalBlock(nn.Module):
                def __init__(self,n_in,n_out,k,s,d,p,drop):
                    super().__init__()
                    self.conv1 = nn.Conv1d(n_in,n_out,k,stride=s,padding=p,dilation=d)
                    self.chomp1= Chomp1d(p); self.relu1=nn.ReLU(); self.drop1=nn.Dropout(drop)
                    self.conv2 = nn.Conv1d(n_out,n_out,k,stride=s,padding=p,dilation=d)
                    self.chomp2= Chomp1d(p); self.relu2=nn.ReLU(); self.drop2=nn.Dropout(drop)
                    self.net   = nn.Sequential(self.conv1,self.chomp1,self.relu1,self.drop1,
                                               self.conv2,self.chomp2,self.relu2,self.drop2)
                    self.down  = nn.Conv1d(n_in,n_out,1) if n_in!=n_out else None
                    self.relu  = nn.ReLU()
                def forward(self,x):
                    out=self.net(x); res=x if self.down is None else self.down(x)
                    return self.relu(out+res)
            class TCNModel(nn.Module):
                def __init__(self,in_size,hid,n_layers,ksize,dil,drop):
                    super().__init__()
                    layers=[]; n_in=in_size
                    for i in range(n_layers):
                        d = dil**(i+1); p=(ksize-1)*d
                        layers.append(TemporalBlock(n_in,hid,ksize,1,d,p,drop))
                        n_in=hid
                    self.tcn=nn.Sequential(*layers); self.fc=nn.Linear(hid,1)
                def forward(self,x):  # [B,T,F] -> [B,F,T]
                    x=x.transpose(1,2)
                    out=self.tcn(x); last=out[:,:,-1]; return self.fc(last)
            model = TCNModel(in_size,hid,n_layers,ksize,dil,dropout)
        else:
            class LSTMModel(nn.Module):
                def __init__(self,in_size,hid,n_layers,drop):
                    super().__init__()
                    self.lstm=nn.LSTM(input_size=in_size,hidden_size=hid,
                                      num_layers=n_layers,batch_first=True,
                                      dropout=drop if n_layers>1 else 0.0)
                    self.fc=nn.Linear(hid,1)
                def forward(self,x):
                    out,_=self.lstm(x); return self.fc(out[:,-1,:])
            model = LSTMModel(in_size,hid,n_layers,dropout)

        state=torch.load(path, map_location="cpu")
        model.load_state_dict(state, strict=False)  # tolerant
        return (mt or meta.get("type","lstm")), model

    # last resort try joblib
    jb=_joblib()
    if jb is not None:
        try:
            obj=jb.load(path); return "sklearn", obj
        except Exception:
            pass

    raise RuntimeError("Unsupported model format or missing dependency")

# ---------- features ----------
EIGHT_FEATURES=['IN_ALLEYS','REST','EAT','ACTIVITY_LEVEL','hour_sin','hour_cos','eat_rest_ratio','activity_rest_ratio']
BUCKETS=["0-8","9-16","17-24"]; BUCKET_CENTER={"0-8":4,"9-16":12,"17-24":20}; BUCKET_SECONDS=3600

def bucket_hour_sin_cos(b):
    h=BUCKET_CENTER[b]
    return np.sin(2*np.pi*h/24.0), np.cos(2*np.pi*h/24.0)

def build_features(df_days_minutes):
    df=df_days_minutes.copy()
    # minutes/hr -> seconds/hr
    for col in ["EAT","REST","IN_ALLEYS"]:
        df[col]=(df[col].astype(float)*60).clip(lower=0)
    df["STANDING"]=(BUCKET_SECONDS-(df["REST"]+df["EAT"])).clip(lower=0)
    df["ACTIVITY_LEVEL"]=(-0.23*df["REST"]+0.16*df["STANDING"]+0.42*df["EAT"]).clip(lower=0)

    overall=df[["EAT","REST","IN_ALLEYS","ACTIVITY_LEVEL","hour_sin","hour_cos"]].mean()
    tot_e, tot_r = float(df["EAT"].sum()), float(df["REST"].sum())
    if (tot_e+tot_r)>0:
        eat_ratio = tot_e/(tot_e+tot_r); rest_ratio=1-eat_ratio
    else:
        eat_ratio = 0.5; rest_ratio=0.5
    activity_rest_ratio=float(df["ACTIVITY_LEVEL"].sum())/(tot_r if tot_r else 1)

    eight_vec=[
        float(overall.get("IN_ALLEYS",0.0)),
        float(overall.get("REST",0.0)),
        float(overall.get("EAT",0.0)),
        float(overall.get("ACTIVITY_LEVEL",0.0)),
        float(overall.get("hour_sin",0.0)),
        float(overall.get("hour_cos",0.0)),
        float(eat_ratio),
        float(activity_rest_ratio),
    ]
    eight_df=pd.DataFrame([dict(zip(EIGHT_FEATURES,eight_vec))])
    return eight_vec, eight_df

# ---------- prediction blocks ----------
def predict_member(path, mtype_hint, X, arch=None):
    try:
        mtype, mobj = try_load_model(path, member_type=mtype_hint, arch=arch)
        names = infer_feature_names(mtype, mobj)
        X_use = align_X_to_names(X, names)
        p = run_predict(mtype, mobj, X_use)
        return float(p), "ok", ""
    except Exception as e:
        return np.nan, "error", str(e)

def predict_hybrid(hjson_path, X):
    combiner, members, hthr, raw_cfg = load_hybrid_config(hjson_path)
    # Optional gate
    min_members = int(raw_cfg.get("min_members", 0))
    member_thr  = float(raw_cfg.get("member_thr", 0.5))

    probs_all, weights_all, logs = [], [], []
    ok_count = 0
    pos_count = 0
    for m in members:
        rel = str(m["path"]).replace("\\","/")
        pth = rel if os.path.isabs(rel) else os.path.abspath(os.path.join(os.path.dirname(hjson_path), rel))
        mtype = str(m.get("type","")).lower() or None
        arch  = m.get("arch", None)
        w     = float(m.get("weight", 1.0))
        p, st, err = predict_member(pth, mtype, X, arch=arch)
        logs.append({"member_path":pth,"type":mtype or "auto","weight":w,"status":st,"error":err,"prob":p})
        probs_all.append(p); weights_all.append(w)
        if np.isfinite(p):
            ok_count += 1
            if p >= member_thr:
                pos_count += 1

    # Combine
    p_final = combine_probs(probs_all, weights_all, combiner=combiner)

    # Gate
    gated = False
    if min_members > 0:
        if pos_count < min_members:
            gated = True
            # Soften (keeps calibration alive but won't win easily)
            p_final = float(min(p_final if np.isfinite(p_final) else 0.0, member_thr * 0.8))

    if ok_count == 0:
        status = "hybrid_all_failed"
    elif gated:
        status = f"gated_off({pos_count}/{min_members})"
    elif ok_count < len(members):
        status = "partial_ok"
    else:
        status = "ok"

    return float(p_final), logs, status

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser(description="v12 hybrid-ready predictor (single or multi winner)")
    ap.add_argument("--table",required=True)
    ap.add_argument("--known",required=True)   # glob for known case CSVs
    ap.add_argument("--out",required=True)
    ap.add_argument("--multi", action="store_true",
                    help="Allow multiple diseases if prob >= threshold (no one-hot).")
    args=ap.parse_args()

    # Master table
    mt = pd.read_csv(args.table)
    norm_cols = {c: c.strip().lower().replace(" ","_") for c in mt.columns}
    mt.rename(columns=norm_cols, inplace=True)
    def pick(df, *cands):
        for c in cands:
            if c in df.columns: return c
        raise KeyError(f"Missing among: {cands}")
    col_disease = pick(mt,"disease")
    col_os      = pick(mt,"oversampling_%","oversampling_pct","oversampling")
    col_model   = pick(mt,"model")
    col_path    = pick(mt,"model_path","path")
    mt.rename(columns={col_disease:"disease", col_os:"oversampling_pct", col_model:"model", col_path:"model_path"}, inplace=True)
    mt["disease"]=mt["disease"].map(norm_disease)
    mt["oversampling_pct"]=mt["oversampling_pct"].astype(str).str.strip()
    mt = mt[mt["oversampling_pct"]!="0%"].copy()

    # Known-case CSVs
    files = glob.glob(args.known)
    if not files: raise SystemExit("No known-case CSVs matched")
    known=[]
    for f in files:
        try: d=pd.read_csv(f); d["__file__"]=os.path.basename(f); known.append(d)
        except: pass
    known_df=pd.concat(known, ignore_index=True)
    known_df["disease"]=known_df["disease"].map(norm_disease)
    known_df["oversampling"]=known_df["oversampling"].astype(str)

    def infer_case(name):
        n=name.lower()
        if "nodisease" in n or "none" in n: return "none"
        for d in ["mastitis","lameness","calving","oestrus"]:
            if d in n: return d
        if "mastits" in n: return "mastitis"
        return "unknown"
    known_df["case"]=known_df["__file__"].map(infer_case)
    known_df["y_true"]=0
    idx=(known_df["case"]!="none") & (known_df["case"]==known_df["disease"])
    known_df.loc[idx,"y_true"]=1
    known_df.loc[known_df["case"]=="none","y_true"]=0

    # Choose best OS by F1 from known preds
    from sklearn.metrics import f1_score
    best_os={}
    for (d, os_level), grp in known_df.groupby(["disease","oversampling"]):
        if d not in ["mastitis","lameness","calving","oestrus"]: continue
        y=grp["y_true"].astype(int).values
        p=grp["prediction"].astype(int).values
        if y.size==0: continue
        f1=f1_score(y,p,zero_division=0)
        prev=best_os.get(d,(None,-1.0))
        if f1>prev[1]: best_os[d]=(os_level, f1)
    # fallback using master table f1 if missing
    if "f1-score_(class_1)" in mt.columns and "f1_pos" not in mt.columns:
        mt.rename(columns={"f1-score_(class_1)":"f1_pos"}, inplace=True)
    for d in ["mastitis","lameness","calving","oestrus"]:
        if d not in best_os:
            sub=mt[mt["disease"]==d]
            if "f1_pos" in sub.columns and not sub["f1_pos"].isna().all():
                j=sub.sort_values("f1_pos", ascending=False).head(1)
                if not j.empty: best_os[d]=(str(j["oversampling_pct"].iloc[0]), float(j["f1_pos"].iloc[0]))
            else:
                best_os[d]=("15%", 0.0)

    # Preflight model path summary
    print("\n[Preflight] Model paths for chosen OS (0% excluded):")
    for disease in ["mastitis","lameness","calving","oestrus"]:
        os_level=best_os[disease][0]
        sub=mt[(mt["disease"]==disease) & (mt["oversampling_pct"].astype(str)==str(os_level))]
        if sub.empty:
            print(f"  {disease:<8} -> {os_level} : MISSING in master table")
            continue
        chosen=sub.iloc[0]
        for _,r in sub.iterrows():
            p=str(r.get("model_path",""))
            if is_hybrid_config(p): chosen=r; break
        path=str(chosen.get("model_path","")).strip().strip('"')
        exists=os.path.exists(path)
        print(f"  {disease:<8} -> {os_level} : {path}  ({'OK' if exists else 'MISSING'})")

    # Collect interactive inputs
    def ask_int(prompt, low=None, high=None, default=None):
        while True:
            s=input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
            if s=="" and default is not None: return default
            try:
                v=int(s)
                if low is not None and v<low: print(f"Must be >= {low}"); continue
                if high is not None and v>high: print(f"Must be <= {high}"); continue
                return v
            except: print("Please enter an integer.")
    def ask_date(prompt, default=None):
        fmt="%Y-%m-%d"
        while True:
            s=input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
            if s=="" and default is not None: s=default
            try: return datetime.strptime(s, fmt).date()
            except: print(f"Use format {fmt}")

    days = int(input("Enter number of consecutive days (3-20) [3]: ") or "3")
    if days<3 or days>20: raise SystemExit("days must be in [3,20]")
    start = ask_date("Enter START date (YYYY-MM-DD)", default=str(datetime.today().date()))

    entries=[]; last={"EAT":0,"REST":0,"ALLEYS":0}
    for d in range(days):
        dt=start + timedelta(days=d)
        print(f"\n-- Day {d+1} ({dt}) --")
        for b in ["0-8","9-16","17-24"]:
            print(f"  Time bucket {b}")
            eat  = ask_int("    EAT mean minutes PER HOUR (0-60)", 0, 60, default=last["EAT"])
            rest = ask_int("    REST mean minutes PER HOUR (0-60)", 0, 60, default=last["REST"])
            al   = ask_int("    IN_ALLEYS mean minutes PER HOUR (0-60)", 0, 60, default=last["ALLEYS"])
            if eat+rest+al > 60:
                print("[WARN] Sum exceeds 60; capping IN_ALLEYS to fit.")
                al = max(0, 60 - (eat+rest))
            last={"EAT":eat,"REST":rest,"ALLEYS":al}
            hs,hc=bucket_hour_sin_cos(b)
            entries.append({"date":dt,"bucket":b,"EAT":eat,"REST":rest,"IN_ALLEYS":al,"hour_sin":hs,"hour_cos":hc})
    df_days_minutes=pd.DataFrame(entries)

    # Build features
    eight_vec, eight_df = build_features(df_days_minutes)
    X = pd.DataFrame([{
        'IN_ALLEYS': eight_vec[0], 'REST': eight_vec[1], 'EAT': eight_vec[2], 'ACTIVITY_LEVEL': eight_vec[3],
        'hour_sin': eight_vec[4], 'hour_cos': eight_vec[5], 'eat_rest_ratio': eight_vec[6], 'activity_rest_ratio': eight_vec[7]
    }])

    # Predict per disease
    results=[]; debug_rows=[]

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve

    for disease in ["mastitis","lameness","calving","oestrus"]:
        os_level = best_os[disease][0]
        sub = mt[(mt["disease"]==disease) & (mt["oversampling_pct"].astype(str)==str(os_level))]
        if sub.empty:
            debug_rows.append({"disease":disease,"chosen_os":os_level,"model":"<none>","model_path":"<none>","load_status":"no_row","error":""})
            results.append({"disease":disease,"oversampling":os_level,"prob_raw":np.nan,"prob_cal":np.nan,"threshold":np.nan,"prediction":0,"status":"no_row"})
            continue

        # prefer hybrid if present
        chosen = sub.iloc[0]
        for _,r in sub.iterrows():
            p=str(r.get("model_path",""))
            if is_hybrid_config(p): chosen=r; break
        model_name = str(chosen.get("model",""))
        model_path = str(chosen.get("model_path","")).strip().strip('"')

        if is_hybrid_config(model_path):
            try:
                p_raw, member_logs, load_status = predict_hybrid(model_path, X)
                error_msg = "" if load_status in ("ok","partial_ok") else "no valid member probabilities"
            except Exception as e:
                p_raw=np.nan; member_logs=[]; load_status="hybrid_error"; error_msg=str(e)
            debug_rows.append({"disease":disease,"chosen_os":os_level,"model":"hybrid","model_path":model_path,"load_status":load_status,"error":error_msg,"members":json.dumps(member_logs)})
        else:
            p_raw, load_status, error_msg = predict_member(model_path, model_name.lower(), X, arch=None)
            debug_rows.append({"disease":disease,"chosen_os":os_level,"model":model_name,"model_path":model_path,"load_status":load_status,"error":error_msg})

        # Calibrate from known_df for this disease@OS
        grp = known_df[(known_df["disease"]==disease) & (known_df["oversampling"]==str(os_level))]
        grp = grp.dropna(subset=["probability_1","y_true"])
        if grp["y_true"].nunique()>=2 and len(grp)>=5 and np.isfinite(p_raw):
            eps=1e-6
            lr=LogisticRegression(solver="lbfgs")
            pr = grp["probability_1"].astype(float).clip(eps,1-eps)
            Xcal = np.log(pr/(1-pr)).values.reshape(-1,1)
            ycal = grp["y_true"].astype(int).values
            lr.fit(Xcal, ycal)
            def _platt(z):
                z=np.clip(z,eps,1-eps)
                return lr.predict_proba(np.log(z/(1-z)).reshape(-1,1))[:,1]
            p_cal = float(_platt(np.array([p_raw]))[0])
            fpr,tpr,thr = roc_curve(ycal, _platt(pr.values))
            j=tpr-fpr; k=int(np.argmax(j)); thr_youden=float(thr[k])
        else:
            p_cal = float(np.clip(p_raw, 0, 1)) if np.isfinite(p_raw) else np.nan
            thr_youden = 0.5

        pred = int(np.isfinite(p_cal) and (p_cal >= thr_youden))
        results.append({"disease":disease,"oversampling":os_level,"prob_raw":p_raw,"prob_cal":p_cal,"threshold":thr_youden,"prediction":pred,"status":load_status})

    df_res=pd.DataFrame(results)

    # Single-winner vs multi-winner
    mask = df_res["prob_cal"].apply(np.isfinite)
    if args.multi:
        # Multi-label: mark all above their thresholds
        df_res.loc[:, "prediction"] = (df_res["prob_cal"] >= df_res["threshold"]).astype(int)
        if df_res["prediction"].sum() == 0:
            if mask.any():
                i = df_res.loc[mask, "prob_cal"].astype(float).idxmax()
                df_res.loc[i, "prediction"] = 1
                winner = df_res.loc[i, "disease"]
            else:
                winner = "none"
        else:
            winner = "+".join(df_res[df_res["prediction"]==1]["disease"].tolist())
    else:
        # One-hot (default)
        if mask.any():
            i = df_res.loc[mask, "prob_cal"].astype(float).idxmax()
            df_res.loc[:, "prediction"] = 0
            df_res.loc[i, "prediction"] = 1
            winner = df_res.loc[i,"disease"]
        else:
            winner = "none"

    # Apply disease interactivity rules and comments
    comments = []
    predicted_diseases = df_res[df_res["prediction"] == 1]["disease"].tolist()

    # Rule: Oestrus and calving do not overlap - choose higher prob if both
    if "oestrus" in predicted_diseases and "calving" in predicted_diseases:
        oestrus_prob = df_res[df_res["disease"] == "oestrus"]["prob_cal"].values[0]
        calving_prob = df_res[df_res["disease"] == "calving"]["prob_cal"].values[0]
        if oestrus_prob > calving_prob:
            df_res.loc[df_res["disease"] == "calving", "prediction"] = 0
            higher = "oestrus"
        else:
            df_res.loc[df_res["disease"] == "oestrus", "prediction"] = 0
            higher = "calving"
        comments.append(f"Oestrus and calving do not typically overlap. Prioritizing {higher} (higher probability).")
        predicted_diseases = df_res[df_res["prediction"] == 1]["disease"].tolist()
        winner = "+".join(predicted_diseases) if len(predicted_diseases) > 1 else predicted_diseases[0] if predicted_diseases else "none"

    # Rule: Mastitis alone - add rare event comment
    if predicted_diseases == ["mastitis"]:
        comments.append("Even though mastitis is present, the cow is probably healthy as mastitis is a rare event. Further, the cattle should be monitored for at least one week to 10 days; if symptoms persist, mastitis is likely present.")

    # Additional comments based on interactions
    if "lameness" in predicted_diseases and "oestrus" in predicted_diseases:
        comments.append("Lameness may suppress oestrus signs and delay return to cyclicity (odds ~3.5x higher for delay).")

    if "lameness" in predicted_diseases and "mastitis" in predicted_diseases:
        comments.append("Lameness and mastitis are linked: lameness can increase mastitis risk (OR ~1.4 in some studies), and mastitis can cause lameness.")

    if "mastitis" in predicted_diseases and "oestrus" in predicted_diseases:
        comments.append("Mastitis can impair reproduction, leading to lower conception rates and potential pregnancy loss (OR ~2.7-2.8 for loss in early gestation).")

    if "calving" in predicted_diseases:
        comments.append("Around calving and early lactation (first 100-120 DIM), baseline risks for mastitis and lameness are elevated.")

    if "lameness" in predicted_diseases:
        comments.append("If lameness is present, consider modestly raised vigilance for mastitis due to mechanistic links.")

    # CLI summary
    print("\n========================")
    print("Best oversampling chosen (0% excluded):")
    for d in ["mastitis","lameness","calving","oestrus"]:
        print(f"  {d:<8} -> {best_os[d][0]}")
    print("========================\n")
    print("Predictions (calibrated, " + ("multi" if args.multi else "one-hot") + "):")
    for d in ["mastitis","lameness","calving","oestrus"]:
        row=df_res[df_res["disease"]==d]
        if row.empty: print(f"  {d:<8} : [missing]"); continue
        p=row["prob_cal"].values[0]; t=row["threshold"].values[0]; pr=int(row["prediction"].values[0]); st=row["status"].values[0]
        print(f"  {d:<8} : prob={p if pd.notna(p) else 'nan'} (thr={t if pd.notna(t) else 'nan'}) -> {pr} [{st}]")
    print(f"\n>>> FINAL: {winner}\n")

    if comments:
        print("Additional Comments based on Disease Interactions:")
        for c in comments:
            print(f" - {c}")

    # Excel output
    out=args.out
    chosen_df = pd.DataFrame({"disease":[k for k in best_os.keys()],"chosen_os":[best_os[k][0] for k in best_os.keys()]})
    try:
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            chosen_df.to_excel(writer, sheet_name="chosen_os", index=False)
            df_res.to_excel(writer, sheet_name="predictions", index=False)
            pd.DataFrame(debug_rows).to_excel(writer, sheet_name="model_debug", index=False)
            summary_data = [{"final_disease":winner, "mode":"multi" if args.multi else "one-hot"}]
            if comments:
                summary_data[0]["comments"] = "; ".join(comments)
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="summary", index=False)
        print(f"Saved: {out}")
    except PermissionError:
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        out2=os.path.splitext(out)[0]+f"_{ts}.xlsx"
        with pd.ExcelWriter(out2, engine="openpyxl") as writer:
            chosen_df.to_excel(writer, sheet_name="chosen_os", index=False)
            df_res.to_excel(writer, sheet_name="predictions", index=False)
            pd.DataFrame(debug_rows).to_excel(writer, sheet_name="model_debug", index=False)
            summary_data = [{"final_disease":winner, "mode":"multi" if args.multi else "one-hot"}]
            if comments:
                summary_data[0]["comments"] = "; ".join(comments)
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="summary", index=False)
        print(f"[INFO] Output locked, saved as: {out2}")

if __name__=="__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        from sklearn.exceptions import InconsistentVersionWarning
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    except Exception:
        pass
    main()