# 🐄 Cattle Disease Prediction - Weekly Research Journal  
**Project Type:** Multi-output Classification  
**Diseases Predicted:** Mastitis, Lameness, Calving, Oestrus  
**Techniques Used:** Undersampling, Oversampling (ADASYN), Imputation, Calibration, Thresholding  
**Models Applied:** MLP, Random Forest, TabNet, LSTM, LightGBM, TCN, Hybrid Ensemble  

---

## 📅 Week 1 (Mar 10 – Mar 16, 2025)
### ✅ Summary
Initial efforts focused on dataset exploration and literature collection. Ethical form was drafted and first datasets were located. Supervisor guidance helped shape the initial research trajectory.

#### 📌 Commits by Date
- **10.03.2025**: Initial dataset found, literature review started, ethics form filled.
- **13.03.2025**: Continued dataset search and added to GitHub.

---

## 📅 Week 2 (Mar 17 – Mar 23, 2025)
### ✅ Summary
Work continued on acquiring suitable datasets and outlining the structure for artefact documentation. Supervisor emphasized early planning and visibility via version control.

#### 📌 Commits by Date
- **19.03.2025**: Added newly acquired datasets and notes from meeting.
- **21.03.2025**: Uploaded progress on literature summary and dataset description.

---

## 📅 Week 3 (Mar 24 – Mar 30, 2025)
### ✅ Summary
USDA and INRAE datasets selected. Ethics form was revised based on feedback. Literature review was expanded and organized into comparison matrices.

#### 📌 Commits by Date
- **24.03.2025**: Uploaded dataset feasibility check results.
- **26.03.2025**: Updated literature mapping and comparison matrix.

---

## 📅 Week 4 (Apr 01 – Apr 07, 2025)
### ✅ Summary
Final correction and submission of the ethical form. Continued literature refinement and alignment with GitHub artefact structure.

#### 📌 Commits by Date
- **02.04.2025**: Uploaded revised ethical form.
- **04.04.2025**: Literature review progress committed.

---

## 📅 Week 5 (Apr 08 – Apr 16, 2025)
### ✅ Summary
Started documentation of state-of-the-art models and cleaned datasets. Supervisor advised refining preprocessing by handling zero entries.

#### 📌 Commits by Date
- **10.04.2025**: Cleaned dataset committed.
- **13.04.2025**: Draft of literature review summary uploaded.

---

## 📅 Week 6 (Apr 17 – Apr 23, 2025)
### ✅ Summary
Began preparing interim report structure and continued data preprocessing. Identified challenges in class imbalance and data sparsity.

#### 📌 Commits by Date
- **20.04.2025**: Uploaded interim report format study and structure.
- **22.04.2025**: Preprocessing scripts committed.

---

## 📅 Week 7 (Apr 24 – Apr 30, 2025)
### ✅ Summary
Balanced exam schedule and project. Interim drafting began. Supervisor advised slowing project pace slightly during exams.

#### 📌 Commits by Date
- **28.04.2025**: Draft version of introduction and methodology.
- **30.04.2025**: Uploaded latest meeting minutes.

---

## 📅 Week 8 (May 01 – May 07, 2025)
### ✅ Summary
With exams over, focus shifted back to the project. Began collaboration on Overleaf and submitted initial slides for presentation.

#### 📌 Commits by Date
- **02.05.2025**: One-slide presentation submitted.
- **05.05.2025**: Overleaf link and interim literature committed.

---

## 📅 Week 9 (May 08 – May 15, 2025)
### ✅ Summary
Started exploring models for multi-output classification. Poster planning and brainstorming conducted.

#### 📌 Commits by Date
- **13.05.2025**: Preliminary poster layout and model comparisons added.
- **15.05.2025**: Overleaf synced and interim documentation started.

---

## 📅 Week 10 (May 16 – May 23, 2025)
### ✅ Summary
Supervisor approved the interim structure. Feedback received on how to improve clarity and flow of storytelling.

#### 📌 Commits by Date
- **18.05.2025**: Poster v1 committed.
- **22.05.2025**: Revised interim report literature section.

---

## 📅 Week 11 (May 24 – May 30, 2025)
### ✅ Summary
Final touches made to interim report. Presentation structure aligned with content. Final version sent for supervisor feedback.

#### 📌 Commits by Date
- **27.05.2025**: Final literature section and interim conclusion committed.
- **29.05.2025**: Uploaded minutes and final draft of interim.

---

## 📅 Week 12 (Jun 01 – Jun 09, 2025)
### ✅ Summary
Poster and report finalized after supervisor reviews. Last-minute formatting and proofreading completed.

#### 📌 Commits by Date
- **05.06.2025**: Submitted final interim report.
- **09.06.2025**: Poster and final dataset summary submitted.

---

## 📅 Week 13 (Jun 10 – Jun 20, 2025)
### ✅ Summary
Poster presentation conducted. Feedback received from assessors and incorporated into future planning.

#### 📌 Commits by Date
- **16.06.2025**: Presentation submitted.
- **19.06.2025**: Incorporated feedback from interim evaluation and added future plan for publication.

---

## 📅 Week 14 (Jun 21 – Jun 30, 2025)
### ✅ Summary
Focused on finalizing **ADASYN oversampled datasets (0–25%)** and started **model training** across all disease classes. Early ensemble experiments were initiated.

#### 📌 Commits by Date
- **24.06.2025**: Uploaded oversampled datasets to repo.
- **27.06.2025**: First training results for 10% OS with RF & LGBM committed.

---

## 📅 Week 15 (Jul 01 – Jul 10, 2025)
### ✅ Summary
Developed **hybrid ensemble (RF, LGBM, TabNet, LSTM, TCN, MLP)** pipeline. First **comparison metrics table** generated.

#### 📌 Commits by Date
- **04.07.2025**: Initial hybrid ensemble script committed.
- **08.07.2025**: Master metrics table v1 (precision, recall, F1, accuracy) uploaded.

---

## 📅 Week 16 (Jul 11 – Jul 20, 2025)
### ✅ Summary
Extended **TCN and LSTM experiments** for 15% and 20% OS. Integrated **weight normalization** and improved architecture.

#### 📌 Commits by Date
- **13.07.2025**: 15% OS TCN models saved.
- **18.07.2025**: LSTM training results for mastitis & lameness committed.

---

## 📅 Week 17 (Jul 21 – Jul 31, 2025)
### ✅ Summary
Finalized **master comparison table (0–25% OS, all models, all diseases)**. Began work on **predict_interactive_days_v8.py** for user inputs.

#### 📌 Commits by Date
- **24.07.2025**: Updated metrics table (all oversampling levels).
- **30.07.2025**: Interactive predictor v8 committed with calibration hooks.

---

## 📅 Week 18 (Aug 01 – Aug 10, 2025)
### ✅ Summary
Improved predictors to **v8b, v8c** with guards for no-signal cases. Built **bias/fairness audit framework** with thresholds & calibration JSON configs.

#### 📌 Commits by Date
- **03.08.2025**: Bias audit thresholds.json committed.
- **07.08.2025**: Predictor v8c (neutral ratios + guardrails) uploaded.

---

## 📅 Week 19 (Aug 11 – Aug 18, 2025)
### ✅ Summary
Assembled **Streamlit app framework** for multi-disease predictions. Completed **presentation prep** for poster session.

#### 📌 Commits by Date
- **14.08.2025**: Streamlit app scaffold pushed.
- **17.08.2025**: Poster slides (STAT module) finalized.

---

## 📅 Week 20 (Aug 19 – Aug 21, 2025)
### ✅ Summary
**Poster session delivered**. Received feedback on model calibration and oversampling drift. Refined plans for final write-up.

#### 📌 Commits by Date
- **20.08.2025**: Poster presentation files archived in `/docs`.

---

## 📅 Week 21 (Aug 22 – Aug 31, 2025)
### ✅ Summary
Drafted **final dissertation** (due Aug 31). Prepared **README.md, artefact structure, journal, and appendix prompts**. Extended prediction script to **v12** with corrected thresholds and calibration.

#### 📌 Commits by Date
- **25.08.2025**: README.md draft for GitHub committed.
- **28.08.2025**: Prediction script v12 uploaded.
- **31.08.2025**: Final dissertation draft submitted to supervisor.


## 🧾 Supervisor Meeting Minutes (by Date)

| Date         | Key Discussions & Actions |
|--------------|----------------------------|
| 10.03.2025   | Discussed the project concept and selected initial datasets. Supervisor advised drafting the ethics form to initiate approvals. Literature exploration was encouraged for early grounding. |
| 13.03.2025   | Reinforced version control importance. Supervisor emphasized documenting all artefacts on GitHub and ensuring datasets are stored with metadata. |
| 19.03.2025   | Supervisor recommended planning the artefact components and workflows. Emphasized the need to consider preprocessing and architecture early in the process. |
| 21.03.2025   | Suggested segmenting literature review into focused themes like preprocessing, model design, and multi-output classification. Encouraged tabular comparison. |
| 24.03.2025   | Reviewed initial dataset quality. Supervisor advised prioritizing datasets that were longitudinal, structured, and had relevant health metrics. |
| 02.04.2025   | Reviewed the revised ethics form and approved it for submission. Supervisor mentioned possible improvements in data handling protocols. |
| 10.04.2025   | Discussed preprocessing strategy. Supervisor emphasized addressing missing values, zero entries, and data normalization. Suggested plotting distributions. |
| 20.04.2025   | Interim structure was discussed. Supervisor advised outlining the report clearly and linking GitHub artefacts to Overleaf documentation. |
| 28.04.2025   | Supervisor approved report outline and methodology. Feedback was given on integrating model diagrams. Suggested optional pausing of some tasks due to exams. |
| 02.05.2025   | Supervisor reviewed the one-slide presentation and advised adding visual balance. Encouraged rehearsal for confidence. |
| 13.05.2025   | TabNet and LightGBM were discussed for the model phase. Poster planning was reviewed and design feedback was given to highlight comparison results. |
| 18.05.2025   | Reviewed interim report flow and cohesion. Supervisor suggested improving narrative transitions and adding more visual summaries. |
| 27.05.2025   | Supervisor approved the final interim submission. Encouraged final proofreading and clarity checks. |
| 09.06.2025   | Poster and report design reviewed. Supervisor suggested final layout changes, citation consistency, and accessible labeling. |
| 16.06.2025   | Supervisor discussed the poster presentation experience and commended clarity. Suggested using it as a template for future showcases. |
| 19.06.2025   | Final feedback meeting. Supervisor commended overall work and suggested publishing the results in a relevant conference or journal. Encouraged identifying venues aligned with agri-AI or predictive diagnostics. |

---

_Last updated: 23.07.2025_
