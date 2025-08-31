# 🐄 Dairy Cattle Disease Prediction (v12 CLI + Explainability)

End-to-end pipeline for predicting **lameness, mastitis, calving, oestrus** in dairy cattle using activity data (EAT / REST / IN_ALLEYS).  
Includes preprocessing, model training + calibration, bias checks, multi-day CLI prediction (v12 / v12+), and explainability (SHAP/LIME).

---

## 📂 Repository Structure

├─ Master_Table_Updated_25pct.csv # REQUIRED: master comparison table (for thresholds + model selection)
├─ all_cases_consolidated_fresh.xlsx # consolidated table (for thresholding / calibration)
│
├─ models/ # Trained models to be placed here
│ ├─ 10pct/
│ │ ├─ rf_lameness.pkl
│ │ └─ rf_calving.pkl
│ ├─ 25pct/
│ │ └─ rf_oestrus.pkl
│ └─ 15pct_ensemble/
│ └─ mastitis_15pct.hybrid.json
│
├─ predict_interactive_days_v12.py # Stable CLI predictor
├─ predict_interactive_days_v12_plus.py # v12 + richer domain comments
├─ calibrate_and_threshold.py # Calibration script
├─ bias_audit.py # Bias/fairness audit
├─ dashboard.py # Streamlit dashboard
├─ extract_minutes_from_dataset.py # Reduce dataset → daily minutes
└─ notebooks/ # Jupyter notebooks for EDA & enhancement


---

## ⚙️ Environment Setup

1. **Python**: 3.10–3.12 recommended  
2. **Create & activate venv**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # (Windows)

pip install -U pip wheel
pip install pandas numpy scikit-learn==1.5.1 imbalanced-learn ^
            shap lime xgboost lightgbm torch pytorch-tabnet ^
            matplotlib seaborn openpyxl xlsxwriter streamlit

python extract_minutes_from_dataset.py

python train_models.py --oversampling 10
python train_models.py --oversampling 15
python train_models.py --oversampling 25

models/10pct/rf_lameness.pkl
models/10pct/rf_calving.pkl
models/25pct/rf_oestrus.pkl
models/15pct_ensemble/mastitis_15pct.hybrid.json

python calibrate_and_threshold.py

python predict_interactive_days_v12.py ^
  --table "Master_Table_Updated_25pct.csv" ^
  --known "results_case_all_*.csv" ^
  --out "results_v12_corrected.xlsx" ^
  --multi

python predict_interactive_days_v12_plus.py ^
  --table "Master_Table_Updated_25pct.csv" ^
  --known "results_case_all_*.csv" ^
  --out "results_v12_plus.xlsx" ^
  --multi

**Parent directory**\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\streamlit.exe run dashboard_final.py
or`
python streamlit dashboard.py or dashboard_final.py

