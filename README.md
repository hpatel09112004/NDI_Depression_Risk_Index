# NDI: Neutrosophic Depression Index, leakage-free evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![Platform: Google Colab](https://img.shields.io/badge/platform-Google%20Colab-orange.svg)](https://colab.research.google.com/)

> **Manuscript:** *Evaluating a Neutrosophic Daily Depression Index From Wearable Data Against Self-Derived and Independent Targets: Leakage-Free Methodological Evaluation Study.*
> Harsh Patel, Dr. Ilanthenral Kandasamy. Submitted to *JMIR AI*, 2026.
>
> **This repository holds** the analysis code, the configuration files, the archived output of every run behind the manuscript, and the five Multimedia Appendix files. Every number in the manuscript can be found in a file here without running anything (see [Section 3](#3-where-every-manuscript-number-comes-from)). The release cited in the manuscript is **`v1.5-final`** (tagged 18 October 2026).

---

## Contents

1. [The study in one paragraph](#1-the-study-in-one-paragraph)
2. [Quick start for reviewers](#2-quick-start-for-reviewers)
3. [Where every manuscript number comes from](#3-where-every-manuscript-number-comes-from)
4. [Repository map](#4-repository-map)
5. [Version history, tags and corrections](#5-version-history-tags-and-corrections)
6. [Data](#6-data)
7. [Re-running the analyses](#7-re-running-the-analyses)
8. [Results folders, file by file](#8-results-folders-file-by-file)
9. [Methods facts used in the code](#9-methods-facts-used-in-the-code)
10. [Reproducibility safeguards](#10-reproducibility-safeguards)
11. [Troubleshooting](#11-troubleshooting)
12. [Citation and licence](#12-citation-and-licence)

---

## 1. The study in one paragraph

We built a three-channel neutrosophic daily index from wearable data:
- **T**: level normality;
- **I**: data quality;
- **F**: volatility against a population norm.

We evaluated it on all **16 PMData participants (2,398 participant-days)** with **blocked forward-chaining cross-validation**: a 6-day embargo, a purge of overlapping target windows, and fold-local statistics.

Four arms were compared:
- **V3**, a random forest on the three channels;
- **V-signed**, a directional-falsity variant;
- a **raw-feature random forest** (raw-feature RF);
- a **personalised z-score** model.

Each arm predicted targets that share progressively less structure with the index:
1. the index's own mean over the next 7 days (*self-derived*);
2. the same window over a composite of the raw features (*raw-feature*);
3. **next-day self-reported mood** (primary independent target);
4. **next-day self-reported stress**.

A no-model **persistence** forecast (today's value) is reported for every target. We also measured the inflation caused by five evaluation leaks, and ran a post hoc learnability diagnostic.

**Main finding:** the index outperformed both baselines only on the self-derived target (V3 vs raw-feature RF, Δρ = 0.213, 95% CI 0.125–0.301). There it did not exceed persistence (Δρ = −0.016, 95% CI −0.173 to 0.128), and it showed no advantage on any target it did not help construct.

---

## 2. Quick start for reviewers

| Goal | What to do | Time |
|---|---|---|
| **Check a number in the manuscript** | Look it up in [Section 3](#3-where-every-manuscript-number-comes-from) and open the file listed there. Each table is stored at full precision, and a `*_formatted.csv` copy shows it as printed. | minutes |
| **Read the supplementary material** | Open `NDI_Multimedia_Appendix_1.docx` to `NDI_Multimedia_Appendix_5.docx` (Section 4). Each appendix names the archive file behind every value it reports. | minutes |
| **See that the code ran and what it printed** | Open any notebook on GitHub. All notebooks are saved **with their outputs**, including the checks at the end of each run. | minutes |
| **Re-run an analysis** | Follow [Section 7](#7-re-running-the-analyses). The notebooks that build on earlier runs first reproduce the archived values, and stop if they cannot. | minutes to hours per notebook on Colab |

No account, token or payment is needed. The repository is public, and the data are public (Section 6).

---

## 3. Where every manuscript number comes from

Each path starts from the main page of this repository (for example, open the `results_v1.4` folder, then the file). The "Notebook" column names the notebook whose code wrote that file.

### Cohort and data preparation (Methods; Results, Cohort)

| Manuscript location | Number(s) | Source file | Notebook |
|---|---|---|---|
| Datasets; Cohort | 16 participants; 10 Fitbit features; series of 141–152 days; 2,398 participant-days; minimum training window t0 = 10 | `results_v1.4/data_facts_v1_4.txt` | `NDI_v1_4.ipynb` |
| Cohort | 2,030 scored test days on the 7-day targets (118–129 per participant) | `results_v1.4/rows_v1_3_vs_v1_4.csv` (column `scored_test_rows_v1_4`) | `NDI_v1_4.ipynb` |
| Cohort | 14 of 16 eligible for mood and for stress; 1,334 and 1,307 scored test days; reasons for exclusion | `results_v1.4_task3/task3_eligibility_log.csv` | `NDI_task3_v1_4.ipynb` |
| Datasets (DEPRESJON) | 55 recordings loaded; one with 14 days excluded by the 15-day rule; 54 used | printed output of the data-loading cells | `NDI_Pipeline.ipynb` |
| Data preparation, check (1): one row per date | 18 duplicated rows in 9 participants | `results_diagnostics/duplicate_days_summary.csv`, `duplicate_days_reports.csv`; `results_v1.4/rows_v1_3_vs_v1_4.csv` | `NDI_duplicate_days_check.ipynb`, `NDI_v1_4.ipynb` |
| Data preparation, check (2): participants loaded | p12 now loaded: 152 rows, 129 scored test folds | `results_v1.4/data_facts_v1_4.txt`; `results_v1.4/provenance.json` (16 participants loaded) | `NDI_v1_4_prechecks.ipynb`, `NDI_v1_4.ipynb` |
| Data preparation, check (3): dating | 1,834 sleep scores matched to their Fitbit sleep log, 0 date mismatches | `results_diagnostics/sleep_score_check_verdict.json`, `sleep_score_check_summary.csv`, `sleep_score_check_rows.csv` | `NDI_sleep_score_check.ipynb` |
| Data preparation, check (4): self-reports | 1,747 reports → 1,724 participant-days; 2 blank reports dropped; 20 days with several reports | `results_v1.4/data_facts_v1_4.txt`, `wellness_rules_log.csv`, `wellness_daily_W1_W4.csv` | `NDI_v1_4.ipynb` |
| Learnability diagnostic (Methods) | share of consecutive-day pairs where tomorrow's self-report equals today's: mood 69%, stress 62% | computed from `results_v1.4/wellness_daily_W1_W4.csv` | — |
| Corrections to the original implementation | V3 vs raw-feature RF: 0.193 (self-derived) and −0.076 (raw-feature) originally; 0.213 and 0.025 after all corrections | `results_v1.4/comparison_v1_v1_2_v1_3_v1_4_formatted.csv` | `NDI_v1_4.ipynb` |
| Figure 1 (first-fold day arithmetic) | days 10, 11–16 and 17; effective gap of 7 days | `results_v1.4/item2_day_arithmetic.txt` | `NDI_v1_4.ipynb` |

PMData participant ages (25–60 years, mean 34) and the collection period (November 2019 to March 2020) come from the PMData publication (Section 6), not from this repository.

### Main results (Abstract; Results; Tables 1–2; Figure 2)

| Manuscript location | Number(s) | Source file | Notebook |
|---|---|---|---|
| Self-derived target; Table 2 | V3 vs raw-feature RF 0.213 (0.125–0.301), 15/16, q = .002; V3 vs personalised z-score 0.283 (0.179–0.375), q = .001; V-signed 0.113 (q = .039) and 0.183 (q = .024); V3 vs V-signed 0.100 (0.030–0.176), P = .018 | `results_v1.4/task1_table2_formatted.csv` (same values in `results_v1.4_task3/task1_table2_formatted_v1_4_corrected.csv`) | `NDI_v1_4.ipynb` |
| Table 1, self-derived and raw-feature columns | arm means with 95% CI (e.g. V3 0.477, 0.371–0.590) | `results_v1.4/task1_absolute_rho_formatted.csv` | `NDI_v1_4.ipynb` |
| Table 1, mood and stress columns; Results | mood: every arm 0.064 or less; stress: V3 0.073 and V-signed 0.076, both P = .035 | `results_v1.4_task3/task3_absolute_rho_mood_formatted.csv`, `task3_absolute_rho_stress_formatted.csv` | `NDI_task3_v1_4.ipynb` |
| Table 2, mood and stress | mood 0.016, q > .99; stress 0.062 (0.013–0.107), P = .030, q = .12 | `results_v1.4_task3/task3_table2_mood_formatted.csv`, `task3_table2_stress_formatted.csv` | `NDI_task3_v1_4.ipynb` |
| Table 1, persistence row | 0.493 / 0.257 / 0.165 (n = 13) / 0.311 | `results_v1.4_task3/persistence_reference_summary_formatted.csv` | `NDI_task3_v1_4.ipynb` |
| V3 vs persistence (exploratory) | −0.016 (−0.173 to 0.128), 9/16, P = .90 | `results_v1.4_task3/v3_vs_persistence_self_derived_exploratory.csv` | `NDI_task3_v1_4.ipynb` |
| Raw-feature target | all q ≥ .55; equivalence 90% CI −0.063 to 0.115; upper 95% bounds 0.132 and 0.213 | `results_v1.4/task1_table2_bounded_null.csv` | `NDI_v1_4.ipynb` |
| Discussion, bounded nulls | advantages larger than 0.132 (raw-feature), 0.076 (mood) and 0.107 (stress) excluded | upper 95% bounds in `results_v1.4/task1_table2_formatted.csv`, `results_v1.4_task3/task3_table2_mood_formatted.csv`, `task3_table2_stress_formatted.csv` | as above |
| Target-swap shift (exploratory) | V3 vs raw-feature RF 0.188 (0.104–0.300), 13/16, q = .007; V3 vs personalised z-score 0.191 (0.058–0.327), q = .018; V-signed 0.085 and 0.089, q = .50 | `results_v1.4/exploratory_target_swap_shift_formatted.csv` | `NDI_v1_4.ipynb` |
| Same, without p12 | 0.173 (0.082–0.265) and 0.157 (0.049–0.280), both q = .025 | `results_v1.4_task3/exploratory_target_swap_shift_without_p12_formatted.csv` | `NDI_task3_v1_4.ipynb` |
| Figure 2 | per-participant values, three panels | `results_v1.4_task3/figure2_three_panels.pdf` / `.png` | `NDI_task3_v1_4.ipynb` |

### Persistence and learnability (post hoc)

| Manuscript location | Number(s) | Source file | Notebook |
|---|---|---|---|
| Learned vs no-model persistence | self-derived 0.480 vs 0.493 (Δρ −0.013, −0.138 to 0.098); raw-feature −0.015 vs 0.257; mood 0.041 vs 0.161; stress 0.078 vs 0.315 | `results_task4_stage0/stage0_learnability_summary_formatted.csv` | `NDI_stage0_learnability.ipynb` |
| Slope and intercept decomposition | slopes positive in 69%–71% of folds; slope term 0.165 / 0.075 / 0.195; intercept correlation −0.04 to −0.14; median SD 0.047 vs 0.031 (raw-feature) | `results_task4_stage0/stage0_diagnostic_summary.csv` | `NDI_stage0_learnability.ipynb` |
| Anomaly correlation for learned persistence; Limitations ("about half of that gap") | raw-feature 0.140 vs 0.281 | `results_task4_stage0/stage0b_anomaly_summary_formatted.csv` | `NDI_stage0_learnability.ipynb` |

The learnability values for mood and stress (0.161, 0.315) are scored on the days with today's value, so they differ slightly from the persistence row of Table 1 (0.165, 0.311), which uses all days with a report on both days.

### Evaluation leaks (Table 3; Figure 3)

| Manuscript location | Number(s) | Source file | Notebook |
|---|---|---|---|
| Table 3 (standalone inflation) | L0 +0.280; L4 +0.038; L1 +0.007; L3 0.000; L2 +0.016 (forward target) and +0.299 (trailing target) | `results_v1.4_task3/table4_standalone_inflation_L2_both_targets_formatted.csv` (built from `results_v1.4/task2_one_at_a_time.csv` and `results_v1.2/task2_one_at_a_time.csv`) | `NDI_task3_v1_4.ipynb` |
| Table 3 note | leak-free V3 0.477; with L0 alone 0.757 (0.716–0.804) | `results_v1.4/task2_one_at_a_time_formatted.csv` | `NDI_v1_4.ipynb` |
| Figure 3 (cumulative removal) | 0.778 → 0.750 → 0.750 → 0.757 → 0.477 | `results_v1.4_task3/figure4_from_L2_removed_bar_labels.csv`; archived plot `figure4_leak_waterfall_from_L2_removed.pdf` / `.png`; full chain from the all-leaks state in `results_v1.4/task2_waterfall_cumulative_formatted.csv` | `NDI_task3_v1_4.ipynb`, `NDI_v1_4.ipynb` |

### Sensitivity analyses

| Manuscript location | Number(s) | Source file | Notebook |
|---|---|---|---|
| Without p12 (self-derived) | V3 vs raw-feature RF 0.187 (0.104–0.282), q = .002; vs personalised z-score 0.234, q = .002; V-signed vs raw-feature RF q = .055; V3 vs V-signed 0.063, P = .14 | `results_v1.4/supplementary_without_p12_table2.csv` | `NDI_v1_4.ipynb` |
| Without p12 (mood, stress) | smallest q = .19 | `results_v1.4_task3/task3_table2_mood_without_p12_formatted.csv`, `task3_table2_stress_without_p12_formatted.csv` | `NDI_task3_v1_4.ipynb` |
| Anomaly correlation (post hoc) | V3 advantage 0.277, q = .001; raw-feature comparisons q = .90; persistence 0.281 vs at most 0.101 | `results_sensitivity_anomaly/anomaly_table2_formatted.csv`, `anomaly_absolute_rho_formatted.csv` | `NDI_sensitivity_anomaly.ipynb` |
| Trailing target (original alignment) | 0.097 (−0.017 to 0.203); raw-feature comparisons null | `results_v1.2/task1_table2_formatted.csv` (n = 15) | `NDI_v1.2_Corrections.ipynb` |

### Component independence and construct validity (original pipeline, tag `v1`)

| Manuscript location | Number(s) | Source file | Notebook |
|---|---|---|---|
| Component independence | PMData \|ρ\| ≤ 0.10; DEPRESJON \|ρ\| 0.25–0.58; dependence detected in all six dataset-pair combinations; effective rank 2.99 and 2.94 | `results_final_v1_frozen_backup/item7_independence_real_data.csv` | `NDI_Pipeline.ipynb` (cell B2) |
| Construct-validity simulation | T 0.85–0.93; F \|ρ\| 0.77–0.91; I 0.95–0.98 (missingness), ≈0.12 (flat-line instability) | `results_final_v1_frozen_backup/arm2_recovery_summary.csv` | `NDI_Pipeline.ipynb` (cell C2.4) |
| Independence on simulated data (Multimedia Appendix 3) | effective rank 2.998–3.000 | `results_final_v1_frozen_backup/arm2_independence_eff_rank.csv`, `arm2_independence_pairs.csv` | `NDI_Pipeline.ipynb` (cell C2.5) |

### Multimedia Appendices

| File | Manuscript title | Built from |
|---|---|---|
| `NDI_Multimedia_Appendix_1.docx` | Consolidated deviation note: original implementation to final | the five corrections; `results_v1.4/comparison_v1_v1_2_v1_3_v1_4.csv`; the configuration files; the data-integrity files above |
| `NDI_Multimedia_Appendix_2.docx` | Supplementary results: next-day stress figure; analyses without p12; trailing-target results; anomaly-correlation sensitivity (continuous targets); learnability diagnostic; cumulative leak table | `results_v1.4_task3/`, `results_v1.4/`, `results_v1.2/`, `results_sensitivity_anomaly/`, `results_task4_stage0/` |
| `NDI_Multimedia_Appendix_3.docx` | Component independence (per-pair statistics and figure) and construct-validity simulation (original pipeline version, repository tag v1) | `results_final_v1_frozen_backup/`; `NDI_Pipeline.ipynb` cells B2 and C2.0–C2.5 |
| `NDI_Multimedia_Appendix_4.docx` | Configuration files, tags, and run order | every `provenance.json` and configuration file |
| `NDI_Multimedia_Appendix_5.docx` | Completed TRIPOD+AI checklist | the manuscript (page and section for every item) |

Every table in Appendices 1–4 names the archive file each value comes from. No appendix required a new run.

---

## 4. Repository map

```
NDI_Depression_Risk_Index/
│
├── README.md                          ← this file
├── LICENSE                            ← MIT
├── .gitattributes
├── ndi_pipeline.py                    ← shared pipeline module (neutrosophic channels, folds, helpers)
│
├── ── Multimedia Appendices (submitted with the manuscript) ──
├── NDI_Multimedia_Appendix_1.docx     ← deviation note: original implementation to final
├── NDI_Multimedia_Appendix_2.docx     ← supplementary results
├── NDI_Multimedia_Appendix_3.docx     ← component independence and construct-validity simulation (tag v1)
├── NDI_Multimedia_Appendix_4.docx     ← configuration files, tags and run order
├── NDI_Multimedia_Appendix_5.docx     ← completed TRIPOD+AI checklist
│
├── ── Notebooks, in the order they were run ──
├── NDI_Pipeline.ipynb                 ← v1: original analysis (history; superseded)
├── NDI_v1.2_Corrections.ipynb         ← v1.2: fold-local statistics, purge, bounded-null statistics, leak quantification (trailing target)
├── NDI_v1_3_forward.ipynb             ← v1.3: target re-aligned to the following 7 days; target-swap shift
├── NDI_duplicate_days_check.ipynb     ← diagnostic: finds the 18 duplicated rows (read-only)
├── NDI_v1_4_prechecks.ipynb           ← diagnostic: p12 loader error; self-report zero counts (read-only)
├── NDI_v1_4.ipynb                     ← v1.4: one row per date, p12 included → final self-derived and raw-feature results
├── NDI_sleep_score_check.ipynb        ← diagnostic: sleep-score dating (read-only)
├── NDI_task3_v1_4.ipynb               ← mood and stress targets, persistence references, Figure 2, Table 3
├── NDI_stage0_learnability.ipynb      ← post hoc learnability diagnostic
├── NDI_sensitivity_anomaly.ipynb      ← post hoc anomaly-correlation sensitivity
│
├── ── Configuration files (each committed and tagged before its run) ──
├── config_v1_3_forward.json           ← tag v1.3-forward
├── config_v1_4.json                   ← tag v1.4
├── config_task3_v1_4.json             ← tag v1.4-task3
│
└── ── Archived outputs (exactly as written by the runs) ──
    ├── results_final_v1_frozen_backup/   ← v1 outputs, frozen before any correction
    ├── results_v1.2/                     ← trailing target, n = 15 (holds config_v1_2_correction.json)
    ├── results_v1.3/                     ← forward target, before de-duplication
    ├── results_v1.4/                     ← final self-derived and raw-feature results, leaks
    ├── results_v1.4_task3/               ← final mood and stress, persistence, Figure 2, Table 3
    ├── results_diagnostics/              ← duplicate-day and sleep-score checks
    ├── results_task4_stage0/             ← learnability diagnostic
    └── results_sensitivity_anomaly/      ← anomaly-correlation sensitivity
```

**About the names:**
- "Task 1", "Task 2" and "Task 3" in file names are the revision's work packages: corrected accuracy with bounded nulls (Task 1), leak quantification (Task 2), and the self-report targets with persistence references (Task 3).
- "Figure 4" and "Table 4" in some file names are the revision's working numbers. In the manuscript the leak waterfall is **Figure 3** and the leak table is **Table 3**.
- The folder `results_task4_stage0/` holds the learnability diagnostic and keeps the name the notebook writes to; renaming it would break the notebook's output path and its `provenance.json` record.
- Each run's results folder also holds a copy of the configuration file the run used.

**Not in the repository:** the raw datasets (Section 6), and scratch or working folders that the manuscript does not cite.

---

## 5. Version history, tags and corrections

| Version | Tag (commit) | What changed | Results |
|---|---|---|---|
| v1 | `v1` (`fca1d9c`) | Original analysis | `results_final_v1_frozen_backup/` |
| v1.2 | *(not tagged)* | Fold-local and causal statistics replace whole-series statistics; purge added; bounded-null statistics; leak quantification. Trailing target kept. | `results_v1.2/` |
| v1.3 | `v1.3-forward` (`54b5c73`) | Target window moved to the **following** 7 days; target-swap shift | `results_v1.3/` |
| v1.4 | `v1.4` (`4dabbfe`) | One row per calendar date (18 duplicated rows removed); p12 included (n = 16); self-report rules W1–W4; analysis without p12 | `results_v1.4/` |
| v1.4 with self-report targets | `v1.4-task3` (`1a3c297`) | Next-day mood and stress; persistence references; Figure 2; Table 3 | `results_v1.4_task3/` |
| Post hoc analyses | run from the `v1.4-task3` commit, not tagged separately | Learnability diagnostic; anomaly-correlation sensitivity | `results_task4_stage0/`, `results_sensitivity_anomaly/` |
| Final package | `v1.5-final` (tagged 18 October 2026) | Multimedia Appendices, this README and the repository state cited in the manuscript; no new analysis | all of the above |

Commit hashes are the first seven characters of those recorded in each run's `provenance.json` (for `v1`, from the repository's tag list). Run times and the full run order are in Multimedia Appendix 4.

**The five corrections to the original implementation** (manuscript: *Corrections to the original implementation*; details in Multimedia Appendix 1):
1. **Target alignment:** the endpoint was specified as forward-looking, but the code averaged the preceding 7 days. Fixed in v1.3; the trailing results are kept in `results_v1.2/`.
2. **Whole-series statistics:** the T channel, the imputation medians and the raw-feature target's normalisation used the full series. Fixed in v1.2 (fold-local statistics for features, causal statistics for targets).
3. **Purge:** the 6-day embargo left a one-day overlap between the last training row's target window and the test day. Fixed in v1.2.
4. **Duplicated days:** 18 rows in 9 participants, created by merging self-reports into the daily frame. Fixed in v1.4 (`results_diagnostics/`).
5. **Silently excluded participant (p12):** a loader error on a metadata field that is not a model feature. Fixed in v1.4 (`NDI_v1_4_prechecks.ipynb`).

`results_v1.4/comparison_v1_v1_2_v1_3_v1_4_formatted.csv` shows how each step moved every primary number.

**Status labels used in the manuscript:**
- **Original analysis:** the primary endpoint, arms, evaluation protocol and raw-feature control (tag `v1`).
- **Revision analyses:** specified after the original results were known, each fixed in a configuration file before its own run. From v1.3 onward each configuration was committed and tagged before the run; v1.2 was not tagged.
- **Post hoc:** quantities defined after their own results were seen: the learnability diagnostic and the anomaly correlation.
- **Exploratory:** quantities defined before their results but outside the primary family: the target-swap shift, and V3 vs persistence.

---

## 6. Data

Both datasets are public and are **not** redistributed here.

| Dataset | Role in the manuscript | Download | Citation |
|---|---|---|---|
| **PMData** | Primary dataset: 16 participants, Fitbit Versa 2 and PMSys self-reports | https://datasets.simula.no/pmdata/ | Thambawita V, et al. PMData: a sports logging dataset. *Proc 11th ACM MMSys*, 2020. doi:10.1145/3339825.3394926 |
| **DEPRESJON** | Component-independence check only (`NDI_Pipeline.ipynb`, Multimedia Appendix 3); not used for prediction | https://datasets.simula.no/depresjon/ | Garcia-Ceja E, et al. Depresjon: a motor activity database of depression episodes in unipolar and bipolar patients. *Proc 9th ACM MMSys*, 2018. doi:10.1145/3204949.3208125 |

PMData is released for research and teaching under CC BY-NC 4.0; DEPRESJON is released by its authors as an open dataset for research.

**Features (10 daily Fitbit features):**
- resting heart rate;
- steps;
- total, light, moderate and vigorous active minutes;
- sedentary minutes;
- sleep efficiency, minutes asleep, sleep score.

**Self-reports:** PMSys wellness, using mood and stress (1–5 scales). They are never used as features.

---

## 7. Re-running the analyses

### 7.1 One-time setup (Google Colab + Google Drive)

1. Download PMData (and DEPRESJON, if you want to run `NDI_Pipeline.ipynb`).
2. In **your** Google Drive, create this layout. Copy `ndi_pipeline.py` and the `results_*` folders from this repository; the notebooks read the archived folders from Drive.
   ```
   MyDrive/ndi_project/
   ├── ndi_pipeline.py                  ← from this repository
   ├── pmdata/
   │   ├── participant-overview.xlsx
   │   ├── p01/ (fitbit/, pmsys/, …)
   │   └── … p16/
   ├── depresjon-dataset/data/          ← only for NDI_Pipeline.ipynb
   ├── results_v1.4/                    ← from this repository
   ├── results_v1.4_task3/              ← from this repository
   └── (the other results_* folders)    ← from this repository
   ```
3. Open a notebook in Colab: **File → Open notebook → GitHub**, then paste this repository's URL.
4. Click **Runtime → Disconnect and delete runtime**, then **Runtime → Run all**, and allow Drive access when asked.

The **G-0** cell clones this repository and checks out the notebook's tag, so the code that runs is exactly the tagged code. No token is needed, because the repository is public.

### 7.2 Which notebook to run, and what you should see

| Notebook | Tag checked out | Run time of the archived run (Colab CPU) | Writes to | Should end with |
|---|---|---|---|---|
| `NDI_v1_4.ipynb` | `v1.4` | several hours | `results_v1.4/` | `handin_check.txt`: HAND-IN CHECK PASSED |
| `NDI_task3_v1_4.ipynb` | `v1.4-task3` | several hours | `results_v1.4_task3/` | first: v1.4 reproduced exactly; last: `handin_check_task3.txt` PASSED |
| `NDI_stage0_learnability.ipynb` | `v1.4-task3` | about 6 minutes | `results_task4_stage0/` | the learnability, decomposition and anomaly tables |
| `NDI_sensitivity_anomaly.ipynb` | `v1.4-task3` | about 1.7 hours | `results_sensitivity_anomaly/` | "REPRODUCTION CHECK PASSED: 64 participant-targets x 4 arms identical to the archive"; `handin_check_sensitivity.txt` PASSED |
| `NDI_duplicate_days_check.ipynb`, `NDI_sleep_score_check.ipynb`, `NDI_v1_4_prechecks.ipynb` | — | minutes | `results_diagnostics/` or printed only | the counts in Section 3 |

**Three notebooks are history; read them, don't Run all:**
- **`NDI_v1.2_Corrections.ipynb`** was not tagged and contains the whole v1 pipeline before its v1.2 cells. Re-running it would overwrite `results_v1.2/` and can differ in the last decimals under newer Colab package versions. Its saved outputs (run of 7 October 2026) include the v1.2 hand-in check: 22 of 22 result files present, and the v1 regeneration check passed.
- **`NDI_v1_3_forward.ipynb`** is superseded by v1.4. Its saved outputs and `results_v1.3/` document the forward-target step.
- **`NDI_Pipeline.ipynb`** is the original v1 analysis, kept for transparency and as the source of Multimedia Appendix 3.

**Tip:** to re-run without touching your copy of the archive, run on a separate Drive copy. The notebooks only read the folders of earlier stages, and their hand-in checks confirm that no earlier file was modified.

---

## 8. Results folders, file by file

Most tables come in two forms:
- `*_formatted.csv`: the table as printed, to three decimals;
- the plain file: full precision, every column.

Each run folder (`results_v1.2/` to `results_v1.4_task3/`, `results_task4_stage0/` and `results_sensitivity_anomaly/`) has a `provenance.json`: run start time, HEAD commit, tag commit, whether the configuration matches the tagged copy, and the participants loaded. All of them except `results_task4_stage0/` also end with a hand-in check (`handin_check*.txt`).

### `results_v1.4/` (final self-derived and raw-feature results, leaks)

| File(s) | Contents |
|---|---|
| `config_v1_4.json`, `provenance.json` | Configuration (tag `v1.4`); provenance |
| `data_facts_v1_4.txt`, `date_check_v1_4.csv`, `rows_v1_3_vs_v1_4.csv` | Data facts; date check; rows removed and scored test rows vs v1.3 |
| `wellness_daily_W1_W4.csv`, `wellness_rules_log.csv`, `wellness_zero_counts.csv` | Self-report table and the W1–W4 rule log |
| `table1_corrected_self_target_per_participant.csv`, `raw_target_corrected_per_participant.csv` | Per-participant ρ, four arms, both targets |
| `task1_absolute_rho_formatted.csv`, `task1_absolute_rho_both_targets.csv` | **Table 1** (self-derived and raw-feature columns) |
| `task1_table2_formatted.csv`, `task1_table2_bounded_null.csv` | **Table 2** (self-derived and raw-feature); equivalence and artifact bounds |
| `exploratory_target_swap_shift*.csv`, `exploratory_notes.txt` | Target-swap shift |
| `task2_one_at_a_time*.csv`, `task2_table4_inflation_column*.csv` | Standalone leak inflation (**Table 3** inputs) |
| `task2_waterfall_cumulative*.csv`, `figure4_leak_waterfall.pdf` / `.png`, `figure4_bar_labels.csv` | Cumulative leak removal starting from all five leaks (Multimedia Appendix 2) |
| `task2_v1_2_v1_3_v1_4.csv`, `comparison_v1_v1_2_v1_3_v1_4*.csv`, `v1_vs_corrected.csv` | Effect of each correction |
| `supplementary_without_p12_*.csv`, `supplementary_without_p12_notes.txt` | Analysis without p12 |
| `fold_log_self_target.csv`, `fold_log_raw_target.csv` | Every fold: training and test days, embargo, purge |
| `forward_target_dropped_test_days.csv`, `item2_day_arithmetic.txt` | Day arithmetic of the forward target window (Figure 1) |
| `independent_check_clean_endpoint.csv`, `independent_check_notes.txt` | Independent re-implementation of the fold loop |
| `task1_notes.txt`, `handin_check.txt` | Notes; hand-in check |

### `results_v1.4_task3/` (mood, stress, persistence, Figure 2, Table 3)

| File(s) | Contents |
|---|---|
| `config_task3_v1_4.json`, `provenance.json` | Configuration (tag `v1.4-task3`); provenance, including the sleep-score check verdict |
| `task3_eligibility_log.csv` | Eligible participants per target and reasons (14 each) |
| `task3_mood_per_participant.csv`, `task3_stress_per_participant.csv` | Per-participant ρ, four arms |
| `task3_absolute_rho_{mood,stress}*.csv` | **Table 1** (mood and stress columns) |
| `task3_table2_{mood,stress}*.csv` | **Table 2** (mood and stress) |
| `*_without_p12*` | Analyses without p12, including the target-swap shift |
| `persistence_reference_per_participant.csv`, `persistence_reference_summary*.csv` | Persistence row of **Table 1** |
| `v3_vs_persistence_self_derived_exploratory.csv` | V3 vs persistence |
| `table4_standalone_inflation_L2_both_targets*.csv` | **Table 3** |
| `figure2_three_panels.pdf` / `.png` | **Figure 2** |
| `figure_supplement_stress.pdf` / `.png` | Stress figure (Multimedia Appendix 2) |
| `figure4_leak_waterfall_from_L2_removed.*`, `figure4_from_L2_removed_bar_labels.csv` | **Figure 3** values and archived plot |
| `task1_table2_formatted_v1_4_corrected.csv`, `reporting_notes_v1_4.txt` | Reporting version of Table 2; reporting notes |
| `task3_fold_log_{mood,stress}.csv` | Fold logs |
| `task3_independent_check*.{csv,txt}` | Independent recomputation of the mood target and of V3 and raw-feature RF |
| `task3_notes.txt`, `task3_without_p12_notes.txt`, `handin_check_task3.txt` | Notes; hand-in check |

### `results_task4_stage0/` (learnability diagnostic, post hoc)

| File(s) | Contents |
|---|---|
| `stage0_config.json`, `provenance.json` | Settings; provenance |
| `stage0_learnability_per_participant.csv`, `stage0_learnability_summary*.csv` | Learned vs no-model persistence |
| `stage0_decision.json` | Outcome of the rule set before the run |
| `stage0_diagnostic_per_participant.csv`, `stage0_diagnostic_summary.csv` | Intercept and slope decomposition |
| `stage0b_anomaly_per_day.csv`, `stage0b_anomaly_per_participant.csv`, `stage0b_anomaly_summary*.csv`, `stage0b_reading.json` | Anomaly correlation for learned persistence |
| `stage0_notes.txt`, `stage0b_notes.txt` | Notes |

### `results_sensitivity_anomaly/` (anomaly-correlation sensitivity, post hoc)

| File(s) | Contents |
|---|---|
| `config_sensitivity_anomaly.json`, `provenance.json` | Metric definition; provenance |
| `reproduction_check.csv` | Archived vs re-run ρ: 64 participant-targets × 4 arms, all identical |
| `persistence_check.csv` | Ordinary persistence reproduces the archived values |
| `per_day_predictions.csv` | Every scored day: target, participant, day, fold mean a(s), outcome, today's value, the four arms' predictions |
| `anomaly_rho_per_participant.csv` | Anomaly and ordinary ρ per participant and arm |
| `anomaly_table2*.csv`, `anomaly_absolute_rho*.csv` | Comparisons and arm means under the anomaly correlation |
| `sensitivity_anomaly_notes.txt`, `handin_check_sensitivity.txt` | Notes; hand-in check |

### `results_diagnostics/` (data-integrity checks)

| File(s) | Contents |
|---|---|
| `duplicate_days_reports.csv`, `duplicate_days_summary.csv` | Duplicated dates, the reports behind them, and counts per participant (check 1) |
| `sleep_score_check_rows.csv`, `sleep_score_check_summary.csv`, `sleep_score_check_verdict.json` | Sleep scores matched to Fitbit sleep logs by log identifier (check 3) |

### `results_v1.2/` (trailing target, n = 15)

| File(s) | Contents |
|---|---|
| `config_v1_2_correction.json`, `provenance.json` | Configuration (fixed before the run, not tagged); provenance (no commit recorded) |
| `table1_corrected_self_target_per_participant.csv`, `raw_target_corrected_per_participant.csv`, `v1_vs_corrected.csv` | Per-participant ρ after corrections 2 and 3; v1 vs corrected |
| `task1_absolute_rho*.csv`, `task1_table2*.csv`, `task1_notes.txt` | Trailing-target results (Multimedia Appendix 2) |
| `task2_one_at_a_time.csv`, `task2_table4_inflation_column.csv`, `task2_waterfall_cumulative.csv`, `figure4_leak_waterfall.*` | Leaks under the trailing target, including L2 +0.299 (Table 3) |
| `sensitivity_per_fold_target_*.csv` | Self-derived target rebuilt per fold (sensitivity row) |
| `item2_day_arithmetic.txt`, `fold_log_*.csv` | First-fold day arithmetic that found corrections 1 and 3; fold logs |
| `independent_check_*`, `handin_check.txt` | Independent fold loop; hand-in check |

### `results_v1.3/` (forward target, before de-duplication)

The same structure as `results_v1.4/` without the self-report and p12 files, plus the three-version comparison `three_way_v1_v1_2_v1_3*.csv`, `task2_v1_2_vs_v1_3.csv` and `section4_confirmations.txt`. It is kept so that every step in `comparison_v1_v1_2_v1_3_v1_4*.csv` can be checked.

### `results_final_v1_frozen_backup/` (v1, frozen)

The original v1 outputs with their `manifest.json`, copied before any correction and never modified. They include `config_frozen.json`, `primary_pmdata_rho.csv`, `item7_independence_real_data.csv` (component independence), and `arm2_recovery_summary.csv`, `arm2_independence_eff_rank.csv` and `arm2_independence_pairs.csv` (construct-validity simulation). `manifest.json` lists every file with the notebook cell that wrote it.

---

## 9. Methods facts used in the code

| Item | Value | Where set |
|---|---|---|
| Participants | 16 (PMData); 14 eligible for mood and for stress | `results_v1.4/data_facts_v1_4.txt`; `task3_eligibility_log.csv` |
| Eligibility | ≥ 15 rows and ≥ 10 scored test folds; self-report targets also ≥ 3 distinct values | `config_v1_4.json`, `config_task3_v1_4.json` |
| Cross-validation | blocked forward-chaining, one test day per fold | `ndi_pipeline.py`; V12-1 and V12-2 cells |
| Minimum training window | t0 = min(10, max(5, n // 3)), so 10 days for every participant | `data_facts_v1_4.txt` |
| Embargo and purge | 6 days; training rows whose target window overlaps the test day or its target window are removed (1 row per fold for the 7-day targets, none for next-day targets) | `config_v1_4.json`; `item2_day_arithmetic.txt` |
| Targets | self-derived and raw-feature: mean over the following 7 days, complete window required; mood and stress: next-day value | `config_v1_4.json`, `config_task3_v1_4.json` |
| Index weights | fixed (λT, λI, λF) = (0.2, 0.3, 0.5), never tuned on the evaluation | `config_v1_4.json` |
| Random forests | 100 trees, max depth 4, fixed seed | `ndi_pipeline.py` |
| Personalised z-score | training-fold standardisation, then ridge regression with α = 1.0 | V12-2 cell |
| Statistics | within-participant Spearman ρ; BCa bootstrap 95% CI (10,000 resamples, seed 42); Hodges–Lehmann median; Wilcoxon signed-rank; Benjamini–Hochberg over the four neutrosophic-vs-baseline comparisons per target; TOST at ±0.10 on the 90% CI | `config_v1_4.json`; V12-6 cell |

---

## 10. Reproducibility safeguards

- **Configuration before results.** From v1.3 on, each configuration was committed and tagged before its run. Each tagged notebook checks that its configuration equals the tagged copy, and stops if not.
- **Tagged code runs.** G-0 checks out the tag, and `provenance.json` records HEAD, the tag commit and its time.
- **Reproduce before extending.** The v1.4-task3 notebook first reproduces the archived v1.4 results exactly; the anomaly-correlation notebook reproduces every archived per-participant ρ; the learnability notebook reproduces the archived scored days and persistence values. Each stops if it cannot.
- **Independent re-implementation.** A separate implementation of the fold loop reproduces the per-participant V3 and raw-feature RF results to machine precision (`independent_check_*`).
- **Archives are never modified.** Each hand-in check confirms that no file in an earlier results folder changed during the run.
- **No credentials or raw data** are stored in this repository.
- **Reporting checklist.** Reporting follows TRIPOD+AI (Collins et al., *BMJ* 2024;385:e078378). The completed checklist, with the manuscript page and section for every item, is `NDI_Multimedia_Appendix_5.docx`.

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| G-0: "could not check out" the tag | **Runtime → Disconnect and delete runtime**, then **Runtime → Run all**. |
| "config … differs from tag … STOP" | A configuration file in your Drive copy was changed. Copy it again from this repository. |
| `ModuleNotFoundError: ndi_pipeline` | `ndi_pipeline.py` must be at `MyDrive/ndi_project/ndi_pipeline.py`. |
| "archived files needed are missing" | Copy the `results_*` folders from this repository into `MyDrive/ndi_project/` (Section 7.1). |
| "Run … first. Missing: […]" | The notebook was not run from the top. Use **Runtime → Run all**. |
| `openpyxl` not found | Run `pip install openpyxl` in a new cell, then run all again. |
| Colab disconnects during a run | Run all again. Each notebook writes only to its own folder, so its outputs are simply rebuilt. |

---

## 12. Citation and licence

**This work:** Patel H, Kandasamy I. Evaluating a neutrosophic daily depression index from wearable data against self-derived and independent targets: leakage-free methodological evaluation study. Submitted to *JMIR AI*, 2026. (DOI to be added on publication.)

**This repository:** Patel H, Kandasamy I. NDI_Depression_Risk_Index: analysis code, configuration files, and results archive, release v1.5-final. GitHub. 2026. URL: https://github.com/hpatel09112004/NDI_Depression_Risk_Index [accessed 2026-10-18]

**Datasets:** please cite PMData and DEPRESJON as in Section 6.

**Licence:** MIT; see [LICENSE](LICENSE). The datasets keep their own licences.
