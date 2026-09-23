\# IS Project 2 - Respiratory Health Awareness and Prediction System



\## Machine Learning Component



This repository contains the machine learning component of IS Project 2.



The project uses the DDXPlus dataset to develop a symptom-based respiratory

health awareness and prediction system.



\## Current Scope



\- Dataset: DDXPlus

\- Domain: Respiratory conditions

\- Conditions: 21

\- Main model: Random Forest

\- Output: Top 3 possible conditions with confidence percentages

\- Languages: English and Swahili

\- Purpose: Health awareness support, not medical diagnosis



\## Respiratory Selection Rule



The respiratory subset contains conditions whose ICD-10 codes belong to

Chapter J together with:



\- Tuberculosis

\- Pulmonary neoplasm

\- Whooping cough



\## Repository Structure



\- `notebooks/` - Dataset exploration and model-development notebooks

\- `scripts/` - Data preparation and helper scripts

\- `data/raw/` - Original DDXPlus files (not committed to Git)

\- `data/processed/` - Generated model-ready data

\- `data/readable/` - Human-readable patient examples

\- `models/` - Saved machine-learning models

\- `results/figures/` - Generated charts and figures

\- `results/metrics/` - Model evaluation results

