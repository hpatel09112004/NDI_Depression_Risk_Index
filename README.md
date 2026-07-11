# NDI — Neutrosophic Depression Risk Index

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Platform: Google Colab](https://img.shields.io/badge/platform-Google%20Colab-orange.svg)](https://colab.research.google.com/)

> **Validating a multi-domain daily depression risk index from wearable physiology and sleep architecture: A methods development and simulation study**
>
> Authors: Harsh Patel, Dr. Ilanthenral Kandasamy

---

## What This Project Does

This project builds and rigorously validates the **Neutrosophic Depression Index (NDI)** — a daily depression risk score derived from wearable physiological and sleep-architecture signals. Each feature is decomposed into a neutrosophic triple **(T, I, F)** encoding baseline stability, indeterminacy (noise/missingness/ambiguity), and deviation toward elevated risk.

The pipeline is validated in **two independent arms**:

| Arm | Data | Purpose |
|-----|------|---------|
| **Arm 1** | DEPRESJON (n=54, actigraphy) + PMData (n=15, Fitbit multi-domain) | Real-world validation on public datasets |
| **Arm 2** | Controlled generative simulation with known ground truth | Five pre-registered experiments with fixed success criteria |

**Primary results:** within-participant Spearman ρ = 0.736 (DEPRESJON) and ρ = 0.849 (PMData) under leakage-free blocked forward-chaining cross-validation; leakage gap = 0.589 (Wilcoxon p < 0.0001); SHAP–permutation importance agreement ρ = 0.969; all five simulation experiments PASS.

---

## Repository Structure
```
NDI_Depression_Risk_Index/
│
├── NDI_Depression_Risk_Index.ipynb   ← Main Colab notebook (48 cells)
├── LICENSE                           ← MIT
├── README.md                         ← This file
│
└── ndi_project/                      ← Google Drive folder (see setup below)
├── ndi_pipeline.py               ← Shared pipeline module (Arm 1 + Arm 2)
├── depresjon-dataset/
│   └── data/                     ← DEPRESJON CSV files
├── pmdata/                       ← PMData Fitbit CSV + participant-overview.xlsx
└── results/                      ← Auto-created; all output CSVs and figures
```

> The datasets and results live on your **Google Drive**, not in this repository.
> After setup your Drive should look like:

```
MyDrive/ndi_project/
├── ndi_pipeline.py
├── depresjon-dataset/
│   └── data/
│       ├── scores.csv
│       ├── condition/
│       └── control/
├── pmdata/
│   ├── participant-overview.xlsx
│   ├── p01/
│   ├── p02/
│   └── ...
└── results/               # auto-created on first run
```

## Prerequisites

| Requirement | Details |
|---|---|
| Google Account | For Google Colab and Google Drive |
| Google Colab Pro (recommended) | Free tier works but Cell 18 (~6 hours) may timeout; Pro/Pro+ avoids this |
| Python 3.10 | Managed by Colab — no local install needed |
| ~2 GB free on Google Drive | Datasets + results + figures |

All Python dependencies are installed automatically by **Cell 2** in the notebook.
---

## Setup — Step by Step

### Step 1: Download the Public Datasets

**DEPRESJON**
1. Go to: https://datasets.simula.no/depresjon/
2. Download and extract the ZIP
3. You need the `data/` folder containing `condition/`, `control/` sub-folders and `scores.csv`

**PMData**
1. Go to: https://datasets.simula.no/pmdata/
2. Download and extract the ZIP
3. You need a `pmdata/` folder with one sub-folder per participant (`p01/`, `p02/`, …) and `participant-overview.xlsx` at the top level

---

### Step 2: Organise Your Google Drive
Refer the repository structure given above to create that structure in **My Drive**

---

### Step 3: Open the Notebook in Colab

**Option A — directly from GitHub:**
1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. File → Open notebook → GitHub tab
3. Paste this repo URL and select `NDI_Depression_Risk_Index.ipynb`

**Option B — upload manually:**
1. Download `NDI_Depression_Risk_Index.ipynb` from this repo
2. Go to [colab.research.google.com](https://colab.research.google.com/)
3. File → Upload notebook → select the file

---

### Step 4: Edit Cell 1 (the only cell you need to change)

**Cell 1** is the single configuration cell. All paths are set here:

```python
PIPELINE_MODULE = "/content/drive/MyDrive/ndi_project/ndi_pipeline.py"
DEPRESJON_DIR   = "/content/drive/MyDrive/ndi_project/depresjon-dataset/data"
PMDATA_DIR      = "/content/drive/MyDrive/ndi_project/pmdata"
PMDATA_OVERVIEW = "/content/drive/MyDrive/ndi_project/pmdata/participant-overview.xlsx"
RESULTS_DIR     = "/content/drive/MyDrive/ndi_project/results"
RANDOM_SEED     = 42
```

If your Drive folder is named differently, update these paths. **Everything else in the notebook reads from these variables — you never need to edit any other cell.**

---

### Step 5: Run

Run all cells at once or run them one at a time.

When Cell 1 executes, a Google Drive authorisation popup will appear — click **Connect to Google Drive** and allow access.

**Confirm setup succeeded — Cell 1 should print:**
```
Drive mounted.
DEPRESJON dir exists: True
PMData dir exists: True
Pipeline exists: True
```

If any line shows `False`, recheck your Drive folder structure against Step 2.

---

## Cell Execution Guide

| Cell | Name | Runtime | What it does |
|------|------|---------|--------------|
| 1 | Configuration | < 1 min | Mount Drive; set all paths; verify files exist |
| 2 | Install + Imports | 1–2 min | Install xgboost and shap; import all libraries |
| 3 | Load DEPRESJON | < 1 min | Read per-participant actigraphy CSVs + MADRS scores |
| 4 | DEPRESJON features | 1–2 min | Build 15-feature daily vectors; compute T/I/F components |
| 5 | Load PMData | < 1 min | Read Fitbit CSVs and participant-overview.xlsx |
| 6 | PMData feature engineering | 2–5 min | Merge wellness CSV; build 8-feature daily vectors; compute T/I/F |
| 7 | Verify neutrosophic decomposition | < 1 min | Sanity check: F ≠ 1−T confirmed throughout |
| **8** ⭐ | **Blocked TS-CV** | **15–30 min** | **Within-participant RF CV with g=6 embargo; produces ρ=0.736 and ρ=0.849** |
| 9 | Cross-participant analysis | 2–5 min | NDI vs MADRS correlation; AUROC |
| 10 | Baselines + lift | 5–10 min | B1/B2/B3 baselines; lift statistic L_B2 with bootstrap CI |
| 11 | Lambda grid search | 5–10 min | 231-point simplex sweep; λ* and stability radius |
| 12 | SHAP attribution | 5–10 min | TreeSHAP + permutation importance; ρ_SHAP-PI |
| 13 | Cross-dataset table | < 1 min | Summary comparison table |
| 14 | NDI time-series plots | 1–2 min | Per-participant NDI visualisations saved to Drive |
| 15 | Simulation config | < 1 min | SIM_CONFIG dict; all simulation hyperparameters |
| 16 | Simulation generator | < 1 min | AR latent state + Laplace noise + confounder model |
| 17 | Experiment 1: Weight recovery | 5–10 min | λ recovery error across (n_part, n_days) configurations |
| **18** ⭐ | **Experiment 2: Leakage** | **~6 hours** | **Naive vs blocked CV; leakage gap = 0.589** |
| 18(2) | Indeterminacy sensitivity sweep | included | Alpha-weight robustness across 22 simplex configs |
| 19 | Experiment 3: Baseline discrimination | 10–20 min | Lift CI in signal vs y-permuted null regime |
| 19(2) | Sensitivity figure | < 1 min | Sensitivity figure and narrative |
| 20 | Experiment 4: SHAP fidelity | 10–20 min | SHAP rank vs true SNR rank |
| 21 | Experiment 5: Power curve | 10–20 min | Minimum (n_part × n_days) for ≥80% detection power |
| 22 | Results summary | < 1 min | All primary results in one table |
| Addition 1 | SNR sweep | 20–30 min | SHAP reliability vs SNR; threshold ≈ 2.0 identified |
| Addition 2 | 2D power heat-map | 10–20 min | Power grid at realistic SNR = 1.0 |
| Addition 3 | Oracle ceiling | 10–15 min | Oracle ρ=0.950 vs blocked ρ=0.756 vs naive ρ=0.039 |
| Addition 4 | Missing-rate verification | 5 min | Confirm 10% missingness target (observed 0.906 ± 0.029) |
| Addition 5 | Cross-arm prediction check | 10–15 min | Simulator envelope vs real-data observed ρ |

> ⭐ = primary contribution cells. For a **quick reviewer run**, execute Cells 1–14 (Arm 1 only, ~1 hour) then jump to Cell 22 for the full summary.

---

## Expected Outputs

After a complete run, the following results should be confirmed:
**Arm 1: Real Data**
```
DEPRESJON  mean ρ = 0.736  SD = 0.311  median = 0.885  n = 54
PMData     mean ρ = 0.849  SD = 0.070  median = 0.863  n = 15
SHAP–permutation importance:  ρ_SHAP-PI = 0.969 for DEPRESJON and 0.948 for PMData
λ* DEPRESJON : (0.0, 1.0, 0.0)    stability radius: 6 simplex points
λ* PMData    : (0.0, 0.65, 0.35)  stability radius: 11 simplex points
L_B2 DEPRESJON : −0.264  95% CI [−0.405, −0.127]
L_B2 PMData    : +0.288  95% CI [+0.231, +0.343]
```
**Arm 2: Simulation**
```
Exp 1  Weight recovery         PASS  mean error 0.016 ± 0.010
Exp 2  Leakage demonstration   PASS  mean gap 0.589 (p < 0.0001)
Exp 3  Baseline discrimination PASS  signal 100%, null 4%
Exp 4  SHAP fidelity           PASS  mean ρ = 0.817
Exp 5  Power / sample size     PASS  minimum n = 3×30 days
Oracle ceiling  ρ = 0.950
Blocked CV      ρ = 0.756
SHAP SNR threshold ≈ 2.0
```
All figures and result CSVs are saved automatically to your `RESULTS_DIR` on Google Drive.

---

## Reproducibility

All results are fully reproducible from `RANDOM_SEED = 42` set in Cell 1. A fresh-kernel re-run (Runtime → Disconnect and delete runtime → reconnect → Run all) produces identical results to 4 decimal places.

Per-run simulation seeds are recorded in Cell 15's `SIM_CONFIG` and written to the results CSV for audit.

---

## Troubleshooting

**Cell 1 prints `False` for any path**
→ Your Drive folder name or structure does not match Cell 1's paths. Update the path variables in Cell 1 to match your actual Drive layout.

**`ModuleNotFoundError: No module named 'ndi_pipeline'`**
→ `ndi_pipeline.py` is missing from your Drive, or the `PIPELINE_MODULE` path in Cell 1 is wrong. Confirm the file exists at `MyDrive/ndi_project/ndi_pipeline.py`.

**Cell 8 shows PMData "skipped"**
→ `pmdata_participants` is empty. Check that `participant-overview.xlsx` is present in `pmdata/` and that each participant folder (`p01/`, `p02/`, …) contains `fitbit/` and `pmsys/` sub-folders.

**Session disconnects during Cell 18 (~6 hours)**
→ Use Colab Pro+ to avoid the free-tier runtime limit. On free Colab, run the DEPRESJON sweep first, save the CSV to Drive, then start a new session and run PMData separately.

**`DEPRESJON dir exists: False`**
→ The `data/` folder must contain `condition/` and `control/` sub-folders with numbered CSVs (e.g. `1.csv`, `2.csv`, …) and a `scores.csv` file at the same level.

**`openpyxl` not found**
→ Cell 2 does not install `openpyxl` explicitly. Run `!pip install openpyxl -q` in a new cell before Cell 5 if you see this error.

---

## Citation

If you use this pipeline or the datasets in your work, please cite:

**This project:**
Harsh Patel, Dr. Ilanthenral Kandasamy. Validating a multi-domain daily depression risk index from
wearable physiology and sleep architecture: A methods development and
simulation study. 2026.

**DEPRESJON dataset:**
Garcia-Ceja E, Riegler M, Jakobsen P, Torresen J, Nordgreen T, Oedegaard KJ, et al.
Depresjon: a motor activity database of depression episodes in unipolar and bipolar patients.
Proc 9th ACM MMSys, 2018. doi:10.1145/3204949.3208133

**PMData dataset:**
Thambawita V, Hicks SA, Borgli H, Stensland HK, Jha D, Svensen M, et al.
PMData: a sports logging dataset.
Proc 11th ACM MMSys, 2020. doi:10.1145/3339825.3394926

---

## License

MIT — see [LICENSE](LICENSE).
