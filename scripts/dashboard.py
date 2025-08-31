import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
import json
import os
import warnings
import shap
from streamlit_shap import st_shap
from io import BytesIO
from sklearn.metrics import f1_score, roc_curve
from sklearn.linear_model import LogisticRegression

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except:
    pass

# Custom CSS for styling
st.markdown("""
<style>
    body {
        background-color: #F5F7FA;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background-color: #F5F7FA;
    }
    .main-header {
        color: #1E88E5;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        color: #1E88E5;
        font-size: 1.8em;
        font-weight: bold;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 10px;
        margin-top: 20px;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.1em;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        color: white;
    }
    .stDataFrame {
        border: 1px solid #1E88E5;
        border-radius: 8px;
        background-color: white;
    }
    .prediction-badge {
        background-color: #4CAF50;
        color: white;
        padding: 8px 16px;
        border-radius: 12px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
    .warning-box {
        background-color: #FFF3E0;
        color: #FF9800;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #FF9800;
        margin-bottom: 15px;
    }
    .info-box {
        background-color: #E3F2FD;
        color: #1E88E5;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #1E88E5;
        margin-bottom: 15px;
    }
    .stExpander {
        background-color: white;
        border: 1px solid #1E88E5;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .stExpander > div > div > div {
        background-color: #E3F2FD;
        color: #1E88E5;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

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
    arr = np.array(probs, dtype=float)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return np.nan
    arr = arr[mask]
    if combiner == "max":
        return float(np.max(arr))
    if combiner in ("mean","avg","average"):
        return float(np.mean(arr))
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
    try:
        with open(path,"r",encoding="utf-8-sig") as f:
            cfg=json.load(f)
        if not isinstance(cfg,dict) or "members" not in cfg:
            raise ValueError("Invalid hybrid config (no 'members').")
        return cfg.get("combiner","weighted_mean"), cfg["members"], cfg.get("threshold", None), cfg
    except Exception as e:
        raise RuntimeError(f"Failed to load hybrid config {path}: {str(e)}")

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
    if mtype=="xgb":
        try: return mobj.feature_names
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
    try:
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
                    trials.append(torch.from_numpy(arr[np.newaxis, 1:2, :]))
                    trials.append(torch.from_numpy(arr[np.newaxis, :]))
                    trials.append(torch.from_numpy(arr[np.newaxis, np.newaxis, :]).permute(1,0,2))
                else:
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
    except Exception as e:
        raise RuntimeError(f"Prediction failed for {mtype}: {str(e)}")

def try_load_model(path, member_type=None, arch=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    mt  = (member_type or "").lower()
    jb  = _joblib()

    if ext in (".pkl",".joblib") and mt not in ("xgb","lgbm","tabnet","lstm","tcn"):
        if jb is None: raise RuntimeError("joblib not installed")
        return "sklearn", jb.load(path)

    if ext==".json" or mt=="xgb":
        xgb=_xgb()
        if not xgb: raise RuntimeError("xgboost not installed")
        booster=xgb.Booster(); booster.load_model(path)
        return "xgb", booster

    if ext in (".txt",".lgb") or mt=="lgbm":
        lgb=_lgb()
        if not lgb: raise RuntimeError("lightgbm not installed")
        booster=lgb.Booster(model_file=path)
        return "lgbm", booster

    if ext==".zip" or mt=="tabnet":
        TabNetClassifier=_tabnet()
        if TabNetClassifier is None: raise RuntimeError("pytorch-tabnet not installed")
        m = TabNetClassifier(); m.load_model(path)
        return "tabnet", m

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
                def forward(self,x):
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
        model.load_state_dict(state, strict=False)
        return (mt or meta.get("type","lstm")), model

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

# ---------- SHAP background dataset ----------
def generate_background_data(n_samples=100):
    np.random.seed(42)
    return pd.DataFrame({
        'IN_ALLEYS': np.random.uniform(0, 3600, n_samples),
        'REST': np.random.uniform(0, 3600, n_samples),
        'EAT': np.random.uniform(0, 3600, n_samples),
        'ACTIVITY_LEVEL': np.random.uniform(0, 3600, n_samples),
        'hour_sin': np.random.uniform(-1, 1, n_samples),
        'hour_cos': np.random.uniform(-1, 1, n_samples),
        'eat_rest_ratio': np.random.uniform(0, 1, n_samples),
        'activity_rest_ratio': np.random.uniform(0, 10, n_samples),
    })

# ---------- prediction blocks ----------
def predict_member(path, mtype_hint, X, arch=None):
    try:
        mtype, mobj = try_load_model(path, member_type=mtype_hint, arch=arch)
        names = infer_feature_names(mtype, mobj)
        X_use = align_X_to_names(X, names)
        p = run_predict(mtype, mobj, X_use)
        return float(p), "ok", "", mtype, mobj
    except Exception as e:
        return np.nan, "error", str(e), None, None

def predict_hybrid(hjson_path, X):
    try:
        combiner, members, hthr, raw_cfg = load_hybrid_config(hjson_path)
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
            p, st, err, _, _ = predict_member(pth, mtype, X, arch=arch)
            logs.append({"member_path":pth,"type":mtype or "auto","weight":w,"status":st,"error":err,"prob":p})
            probs_all.append(p); weights_all.append(w)
            if np.isfinite(p):
                ok_count += 1
                if p >= member_thr:
                    pos_count += 1

        p_final = combine_probs(probs_all, weights_all, combiner=combiner)
        gated = False
        if min_members > 0:
            if pos_count < min_members:
                gated = True
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
    except Exception as e:
        return np.nan, [], f"hybrid_error: {str(e)}"

# ---------- SHAP computation ----------
def compute_shap(mobj, mtype, X, feature_names):
    try:
        if mtype in ["sklearn", "xgb", "lgbm"]:
            background_data = generate_background_data()
            X_use = align_X_to_names(X, feature_names)
            background_data = align_X_to_names(background_data, feature_names)
            explainer = shap.Explainer(mobj, background_data)
            shap_values = explainer(X_use)
            return shap_values
        else:
            return None
    except Exception as e:
        st.warning(f"SHAP computation failed: {str(e)}")
        return None

# ---------- Prediction function ----------
def run_predictions(mt_path, known_glob, multi_mode, df_days_minutes):
    try:
        mt = pd.read_csv(mt_path)
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
        col_f1      = pick(mt,"f1-score_(class_1)","f1_score_(class_1)")
        mt.rename(columns={col_disease:"disease", col_os:"oversampling_pct", col_model:"model", col_path:"model_path", col_f1:"f1_pos"}, inplace=True)
        mt["disease"]=mt["disease"].map(norm_disease)
        mt["oversampling_pct"]=mt["oversampling_pct"].astype(str).str.strip()
        mt = mt[mt["oversampling_pct"]!="0%"].copy()
        mt["f1_pos"] = mt["f1_pos"].fillna(0.0).astype(float)

        files = glob.glob(known_glob)
        if not files: 
            st.warning("No known-case CSVs matched; using master table F1 scores.", extra={"class": "warning-box"})
            known_df = pd.DataFrame()
        else:
            known=[]
            for f in files:
                try: 
                    d=pd.read_csv(f); d["__file__"]=os.path.basename(f); known.append(d)
                except: pass
            known_df=pd.concat(known, ignore_index=True) if known else pd.DataFrame()
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

        # Select best OS based on known_df F1 score (as in v12)
        best_os={}
        for (d, os_level), grp in known_df.groupby(["disease","oversampling"]):
            if d not in ["mastitis","lameness","calving","oestrus"]: continue
            y=grp["y_true"].astype(int).values
            p=grp["prediction"].astype(int).values
            if y.size==0: continue
            f1=f1_score(y,p,zero_division=0)
            prev=best_os.get(d,(None,-1.0,"none",""))
            if f1>prev[1]: best_os[d]=(os_level, f1, "unknown", "")

        # Fallback to master table if no known_df F1 or if lower
        for d in ["mastitis","lameness","calving","oestrus"]:
            if d not in best_os:
                sub=mt[mt["disease"]==d]
                if "f1_pos" in sub.columns and not sub["f1_pos"].isna().all():
                    j=sub[sub["f1_pos"]==sub["f1_pos"].max()]
                    if not j.empty:
                        best_os[d]=(str(j["oversampling_pct"].iloc[0]), float(j["f1_pos"].iloc[0]), str(j["model"].iloc[0]), str(j["model_path"].iloc[0]))
                    else:
                        best_os[d]=("15%", 0.0, "none", "")
                else:
                    best_os[d]=("15%", 0.0, "none", "")
            else:
                # Check if master table has higher F1 for the same OS
                sub=mt[(mt["disease"]==d) & (mt["oversampling_pct"]==best_os[d][0])]
                if not sub.empty and "f1_pos" in sub.columns:
                    master_f1 = float(sub["f1_pos"].max())
                    if master_f1 > best_os[d][1]:
                        j=sub[sub["f1_pos"]==master_f1]
                        best_os[d]=(best_os[d][0], master_f1, str(j["model"].iloc[0]), str(j["model_path"].iloc[0]))

                # Fallback for model name/path if not set
                if best_os[d][2] == "unknown":
                    sub=mt[(mt["disease"]==d) & (mt["oversampling_pct"]==best_os[d][0])]
                    if not sub.empty:
                        j=sub[sub["f1_pos"]==sub["f1_pos"].max()]
                        if not j.empty:
                            best_os[d]=(best_os[d][0], best_os[d][1], str(j["model"].iloc[0]), str(j["model_path"].iloc[0]))
                        else:
                            best_os[d]=(best_os[d][0], best_os[d][1], "none", "")

        # Check for repeated inputs
        group = df_days_minutes.groupby('bucket')[['EAT', 'REST', 'IN_ALLEYS']].std()
        if group.eq(0).all().all():
            comments = ["🐄 Repeated inputs across days - possible sensor issue. Prediction made assuming consistent behavior."]
        else:
            comments = []

        eight_vec, eight_df = build_features(df_days_minutes)
        X = pd.DataFrame([{
            'IN_ALLEYS': eight_vec[0], 'REST': eight_vec[1], 'EAT': eight_vec[2], 'ACTIVITY_LEVEL': eight_vec[3],
            'hour_sin': eight_vec[4], 'hour_cos': eight_vec[5], 'eat_rest_ratio': eight_vec[6], 'activity_rest_ratio': eight_vec[7]
        }])

        results=[]; debug_rows=[]; model_objects={}
        for disease in ["mastitis","lameness","calving","oestrus"]:
            os_level, f1_val, model_name, model_path = best_os[disease]
            if not model_path or model_name == "none":
                debug_rows.append({"disease":disease,"chosen_os":os_level,"model":"none","model_path":"none","load_status":"no_row","error":"No valid model"})
                results.append({"disease":disease,"oversampling":os_level,"prob_raw":np.nan,"prob_cal":np.nan,"threshold":np.nan,"prediction":0,"status":"no_row","f1_score":f1_val})
                continue

            if is_hybrid_config(model_path):
                try:
                    p_raw, member_logs, load_status = predict_hybrid(model_path, X)
                    error_msg = "" if load_status in ("ok","partial_ok") else "No valid member probabilities"
                    debug_rows.append({"disease":disease,"chosen_os":os_level,"model":"hybrid","model_path":model_path,"load_status":load_status,"error":error_msg,"members":json.dumps(member_logs)})
                except Exception as e:
                    p_raw=np.nan; member_logs=[]; load_status="hybrid_error"; error_msg=str(e)
                    debug_rows.append({"disease":disease,"chosen_os":os_level,"model":"hybrid","model_path":model_path,"load_status":load_status,"error":error_msg,"members":json.dumps(member_logs)})
            else:
                try:
                    p_raw, load_status, error_msg, mtype, mobj = predict_member(model_path, model_name.lower(), X, arch=None)
                    debug_rows.append({"disease":disease,"chosen_os":os_level,"model":model_name,"model_path":model_path,"load_status":load_status,"error":error_msg})
                    model_objects[disease] = (mtype, mobj) if mtype and mobj else (None, None)
                except Exception as e:
                    p_raw=np.nan; load_status="error"; error_msg=str(e); mtype, mobj = None, None
                    debug_rows.append({"disease":disease,"chosen_os":os_level,"model":model_name,"model_path":model_path,"load_status":load_status,"error":error_msg})
                    model_objects[disease] = (None, None)

            grp = known_df[(known_df["disease"]==disease) & (known_df["oversampling"]==str(os_level))] if not known_df.empty else pd.DataFrame()
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
            results.append({"disease":disease,"oversampling":os_level,"prob_raw":p_raw,"prob_cal":p_cal,"threshold":thr_youden,"prediction":pred,"status":load_status,"f1_score":f1_val})

        df_res=pd.DataFrame(results)

        # Apply conditional formatting to highlight positive predictions
        def highlight_predictions(row):
            if row["prediction"] == 1:
                return ['background-color: #C8E6C9'] * len(row)
            return [''] * len(row)
        df_res_styled = df_res.style.apply(highlight_predictions, axis=1).format({"prob_raw": "{:.3f}", "prob_cal": "{:.3f}", "threshold": "{:.3f}", "f1_score": "{:.3f}"})

        mask = df_res["prob_cal"].apply(np.isfinite)
        if multi_mode:
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
            if mask.any():
                i = df_res.loc[mask, "prob_cal"].astype(float).idxmax()
                df_res.loc[:, "prediction"] = 0
                df_res.loc[i, "prediction"] = 1
                winner = df_res.loc[i,"disease"]
            else:
                winner = "none"

        # Rich comments
        predicted_diseases = df_res[df_res["prediction"] == 1]["disease"].tolist()

        if "oestrus" in predicted_diseases and "calving" in predicted_diseases:
            oestrus_prob = df_res[df_res["disease"] == "oestrus"]["prob_cal"].values[0]
            calving_prob = df_res[df_res["disease"] == "calving"]["prob_cal"].values[0]
            if oestrus_prob > calving_prob:
                df_res.loc[df_res["disease"] == "calving", "prediction"] = 0
                higher = "oestrus"
            else:
                df_res.loc[df_res["disease"] == "oestrus", "prediction"] = 0
                higher = "calving"
            comments.append(f"🐄 Oestrus and calving do not overlap. Around calving, cows enter post-partum anoestrus and typically don’t return to cycles/heat until ~20–40 days (often 45–70 days) after calving. Prioritizing {higher} based on higher probability. Sources: vetmed.auburn.edu, beefrepro.org, ScienceDirect.")
            predicted_diseases = df_res[df_res["prediction"] == 1]["disease"].tolist()
            winner = "+".join(predicted_diseases) if len(predicted_diseases) > 1 else predicted_diseases[0] if predicted_diseases else "none"

        if predicted_diseases == ["mastitis"]:
            comments.append("🐄 Even though mastitis is detected, note that mastitis is a rare event in healthy herds, so the cow is probably healthy overall. However, monitor the cattle closely for at least one week to 10 days for symptoms like reduced milk yield, swollen udders, or abnormal milk. If symptoms persist, mastitis is likely present and may require veterinary intervention to prevent spread or complications.")

        if "lameness" in predicted_diseases and "oestrus" in predicted_diseases:
            comments.append("🐄 Lameness can drive reproduction issues: Lame cows show fewer or poorer heat signs (e.g., less walking), with ~3.5× higher odds of delayed ovarian activity post-partum. This may suppress oestrus detection and delay return to cycles. Recommend checking foot health and considering pain management to improve reproductive performance. Sources: PubMed, Journal of Dairy Science.")

        if "lameness" in predicted_diseases and "mastitis" in predicted_diseases:
            comments.append("🐄 Lameness and mastitis are bidirectionally linked: Lameness increases lying time, raising intramammary infection risk (OR ≈1.4 in some studies), while acute mastitis (e.g., E. coli) can cause laminitis (hoof inflammation). This context-dependent coupling suggests monitoring both conditions, as one may exacerbate the other. Sources: MDPI, ScienceDirect, CABI Digital Library.")

        if "mastitis" in predicted_diseases and "oestrus" in predicted_diseases:
            comments.append("🐄 Mastitis impairs reproduction: Clinical mastitis around breeding lowers conception/Pregnancy per AI (e.g., 41.7% vs 58.2%), extends days open if occurring before/after AI, and increases pregnancy loss risk in early gestation (OR ≈2.7–2.8). Timing is key—vigilance for pregnancy loss if mastitis is in early gestation. Sources: PMC, NADIS, PubMed, ScienceDirect.")

        if "calving" in predicted_diseases:
            comments.append("🐄 Calving/early lactation (first 100–120 DIM) drives risks: Mastitis peaks ~2 weeks before/after calving, persisting in first 100 DIM; lameness has ~2.3× higher odds in first 120 DIM due to transition stress. Raise baseline vigilance for both in this window. Sources: PMC, BioMed Central.")

        if "lameness" in predicted_diseases:
            comments.append("🐄 Lameness present—modestly raise vigilance for mastitis due to mechanistic links (increased lying time raises infection risk). Also, never output oestrus on calving date (be cautious in first ~20–40 DIM). If near AI, expect fewer heat signs and delayed cyclicity (OR≈3.5). Sources: MDPI, ScienceDirect, PubMed, Journal of Dairy Science.")

        chosen_df = pd.DataFrame({
            "disease": [k for k in best_os.keys()],
            "chosen_os": [best_os[k][0] for k in best_os.keys()],
            "f1_score": [best_os[k][1] for k in best_os.keys()],
            "model": [best_os[k][2] for k in best_os.keys()]
        })
        model_debug_df = pd.DataFrame(debug_rows)
        summary_data = [{"final_disease": winner, "mode": "multi" if multi_mode else "one-hot"}]
        if comments:
            summary_data[0]["comments"] = "; ".join(comments)
        summary_df = pd.DataFrame(summary_data)

        return chosen_df, df_res, df_res_styled, model_debug_df, summary_df, best_os, comments, winner, X, model_objects
    except Exception as e:
        raise RuntimeError(f"Prediction error: {str(e)}")

# Streamlit app
st.markdown("<div class='main-header'>🐄 Cow Disease Prediction Dashboard</div>", unsafe_allow_html=True)

# Sidebar for configurations
with st.sidebar:
    st.markdown("<div class='section-header'>Configurations</div>", unsafe_allow_html=True)
    master_table = st.file_uploader("Upload Master Table CSV", type="csv")
    known_csvs = st.text_input("Known Cases CSV Glob Pattern", value=r"C:\Users\vishn\results_case_all_*.csv")
    multi_mode = st.checkbox("Multi-Winner Mode", value=False)

# Input section
st.markdown("<div class='section-header'>Input Data</div>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    days = st.number_input("Number of Consecutive Days (3-20)", min_value=3, max_value=20, value=3, step=1)
with col2:
    start_date = st.date_input("Start Date", value=datetime.today())

entries = []
last = {"EAT": 0, "REST": 0, "ALLEYS": 0}
for d in range(int(days)):
    dt = start_date + timedelta(days=d)
    with st.expander(f"Day {d+1} ({dt})"):
        for b in BUCKETS:
            st.markdown(f"**Time Bucket {b}**")
            col_eat, col_rest, col_alleys = st.columns(3)
            with col_eat:
                eat = st.number_input(f"EAT (min/hr)", min_value=0, max_value=60, value=last["EAT"], key=f"eat_{d}_{b}")
            with col_rest:
                rest = st.number_input(f"REST (min/hr)", min_value=0, max_value=60, value=last["REST"], key=f"rest_{d}_{b}")
            with col_alleys:
                alleys = st.number_input(f"IN_ALLEYS (min/hr)", min_value=0, max_value=60, value=last["ALLEYS"], key=f"alleys_{d}_{b}")
            if eat + rest + alleys > 60:
                st.markdown("<div class='warning-box'>Sum exceeds 60; capping IN_ALLEYS.</div>", unsafe_allow_html=True)
                alleys = max(0, 60 - (eat + rest))
            last = {"EAT": eat, "REST": rest, "ALLEYS": alleys}
            hs, hc = bucket_hour_sin_cos(b)
            entries.append({"date": dt, "bucket": b, "EAT": eat, "REST": rest, "IN_ALLEYS": alleys, "hour_sin": hs, "hour_cos": hc})

df_days_minutes = pd.DataFrame(entries)

if st.button("Run Prediction 🧠") and master_table:
    with st.spinner("Analyzing cow health data..."):
        mt_path = "temp_master.csv"
        with open(mt_path, "wb") as f:
            f.write(master_table.getvalue())

        try:
            chosen_df, df_res, df_res_styled, model_debug_df, summary_df, best_os, comments, winner, X, model_objects = run_predictions(
                mt_path, known_csvs, multi_mode, df_days_minutes
            )

            st.markdown("<div class='section-header'>Prediction Results</div>", unsafe_allow_html=True)
            st.markdown(f"<div>Final Prediction: <span class='prediction-badge'>{winner}</span></div>", unsafe_allow_html=True)
            st.dataframe(df_res_styled)

            st.markdown("<div class='section-header'>Disease Interaction Comments</div>", unsafe_allow_html=True)
            if comments:
                for c in comments:
                    st.markdown(f"<div class='info-box'>{c}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='info-box'>No specific disease interaction comments.</div>", unsafe_allow_html=True)

            st.markdown("<div class='info-box'>The model is oversampled using ADASYN and for higher F1 score and predictive efficiency the model is calibrated using Platt calibration and thresholded using Youden index.</div>", unsafe_allow_html=True)

            st.markdown("<div class='section-header'>F1 Scores for Selected Models</div>", unsafe_allow_html=True)
            st.dataframe(chosen_df.style.format({"f1_score": "{:.3f}"}))

            st.markdown("<div class='section-header'>SHAP Explanations</div>", unsafe_allow_html=True)
            selected_disease = st.selectbox("Select Disease for SHAP", df_res["disease"].tolist())
            if selected_disease:
                os_level, f1_val, model_name, model_path = best_os[selected_disease]
                if not model_path or model_name == "none":
                    st.markdown(f"<div class='warning-box'>No valid model for {selected_disease} at oversampling {os_level}</div>", unsafe_allow_html=True)
                else:
                    mtype, mobj = model_objects.get(selected_disease, (None, None))
                    if mtype and mobj and not is_hybrid_config(model_path):
                        feature_names = infer_feature_names(mtype, mobj) or EIGHT_FEATURES
                        X_use = align_X_to_names(X, feature_names)
                        shap_values = compute_shap(mobj, mtype, X_use, feature_names)
                        if shap_values is not None:
                            st.markdown("<div class='section-header'>SHAP Force Plot</div>", unsafe_allow_html=True)
                            st_shap(shap.force_plot(shap_values.base_values[0], shap_values.values[0], X_use, feature_names=feature_names))
                            st.markdown("<div class='section-header'>SHAP Summary Plot</div>", unsafe_allow_html=True)
                            st_shap(shap.summary_plot(shap_values, X_use, feature_names=feature_names))
                        else:
                            st.markdown(f"<div class='warning-box'>SHAP not supported for {selected_disease} model type: {mtype}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='warning-box'>SHAP not supported for hybrid models or failed to load model for {selected_disease}</div>", unsafe_allow_html=True)

            st.markdown("<div class='section-header'>Download Results</div>", unsafe_allow_html=True)
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                chosen_df.to_excel(writer, sheet_name="chosen_models", index=False)
                df_res.to_excel(writer, sheet_name="predictions", index=False)
                model_debug_df.to_excel(writer, sheet_name="model_debug", index=False)
                summary_df.to_excel(writer, sheet_name="summary", index=False)
            st.download_button("Download Excel 📊", output.getvalue(), file_name="results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.markdown(f"<div class='warning-box'>Error during prediction: {str(e)}</div>", unsafe_allow_html=True)
        finally:
            if os.path.exists(mt_path):
                os.remove(mt_path)