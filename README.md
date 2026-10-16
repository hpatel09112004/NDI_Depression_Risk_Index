# NDI: Neutrosophic Depression Index, leakage-free evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![Platform: Google Colab](https://img.shields.io/badge/platform-Google%20Colab-orange.svg)](https://colab.research.google.com/)

> **Manuscript:** *Apparent Performance of a Neutrosophic Daily Depression Index Reflects Target Affinity: Leakage-Free Methodological Evaluation Study.*
> Harsh Patel, Dr. Ilanthenral Kandasamy. Submitted to *JMIR AI*, 2026.
>
> **This repository holds** the analysis code, the frozen configuration files, and the archived output of every run behind the manuscript. Every number in the manuscript can be found in a file here, without running anything (see [Section 3](#3-where-every-manuscript-number-comes-from)).

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
- a **raw-feature random forest**;
- a **personalised z-score** model.

Each arm predicted targets that share progressively less structure with the index:
1. the index's own next-7-day mean (*self-derived*);
2. the same window over a raw-feature composite (*raw-feature*);
3. **next-day self-reported mood** (primary independent target);
4. **next-day self-reported stress**.

A no-model **persistence** forecast (today's value) is reported for every target. We also measured the inflation caused by five evaluation leaks, and ran a post hoc learnability diagnostic.

**Main finding:** the index outperformed both baselines only on the self-derived target (V3 vs raw-feature RF, Δρ = 0.213, 95% CI 0.125–0.301). Even there it did not beat persistence (Δρ = −0.016), and it showed no advantage on any target it did not help construct.

---

## 2. Quick start for reviewers

| Goal | What to do | Time |
|---|---|---|
| **Check a number in the manuscript** | Look it up in [Section 3](#3-where-every-manuscript-number-comes-from) and open the CSV listed there. Every value is stored at full precision, and a `*_formatted.csv` copy shows it exactly as printed. | minutes |
| **See that the code ran and what it printed** | Open any notebook on GitHub. All notebooks are saved **with their outputs**, including the hand-in checks at the end. | minutes |
| **Re-run an analysis** | Follow [Section 7](#7-re-running-the-analyses). Each notebook first reproduces the archived numbers exactly, and stops if it cannot. | 1–2 h per notebook on Colab |

No account, token or payment is needed. The repository is public, and the data are public (Section 6).

---

## 3. Where every manuscript number comes from

Each file path starts from the main page of this repository (e.g. open the results_v1.4 folder, then the file). The “Notebook” column names the notebook whose code produced that file.

### Abstract, Results and Tables 1–2

| Manuscript location | Number(s) | Source file | Notebook |
|---|---|---|---|
| Self-derived target (Abstract; Results; Table 2) | V3 vs raw-RF 0.213 [0.125, 0.301], 15/16, q = .002; V3 vs pers-Z 0.283 [0.179, 0.375], q = .001; V-signed vs raw-RF 0.113, q = .039; V-signed vs pers-Z 0.183, q = .024; V3 vs V-signed 0.100 [0.030, 0.176], P = .018 | `results_v1.4/task1_table2_formatted.csv` | `NDI_v1_4.ipynb` |
| Table 1, self-derived and raw-feature columns | Arm means with 95% CI (e.g. V3 0.477 [0.371, 0.590]) | `results_v1.4/task1_absolute_rho_formatted.csv` | `NDI_v1_4.ipynb` |
| Table 1, mood and stress columns; Results (mood "0.064 or less"; stress V3 0.073, V-signed 0.076, P = .035) | Arm means with 95% CI | `results_v1.4_task3/task3_absolute_rho_mood_formatted.csv`, `task3_absolute_rho_stress_formatted.csv` | `NDI_task3_v1_4.ipynb` |
| Table 2, mood and stress (Abstract: mood 0.016, q > .99; stress 0.062, q = .12) | Paired comparisons | `results_v1.4_task3/task3_table2_mood_formatted.csv`, `task3_table2_stress_formatted.csv` | `NDI_task3_v1_4.ipynb` |
| Table 1, persistence row | 0.493 / 0.257 / 0.165 (n = 13) / 0.311 | `results_v1.4_task3/persistence_reference_summary_formatted.csv` | `NDI_task3_v1_4.ipynb` |
| V3 vs persistence (Abstract; Results; Table 2 footnote) | −0.016 [−0.173, 0.128], 9/16, P = .90 | `results_v1.4_task3/v3_vs_persistence_self_derived_exploratory.csv` | `NDI_task3_v1_4.ipynb` |
| Raw-feature bounds | equivalence 90% CI −0.063 to 0.115; upper 95% bounds 0.132 and 0.213 | `results_v1.4/task1_table2_bounded_null.csv` | `NDI_v1_4.ipynb` |
| Target-swap shift (exploratory) | V3 vs raw-RF 0.188 [0.104, 0.300], 13/16, q = .007; V3 vs pers-Z 0.191 [0.058, 0.327], q = .018; V-signed 0.085 / 0.089 | `results_v1.4/exploratory_target_swap_shift_formatted.csv` | `NDI_v1_4.ipynb` |
| Same, without p12 | 0.173 [0.082, 0.265] and 0.157 [0.049, 0.280], q = .025 | `results_v1.4_task3/exploratory_target_swap_shift_without_p12_formatted.csv` | `NDI_task3_v1_4.ipynb` |

### Leaks, Table 3 and Figure 3

| Manuscript location | Number(s) | Source file | Notebook |
|---|---|---|---|
| Table 3 (standalone leak inflation) | L0 +0.280; L4 +0.038; L1 +0.007; L3 0.000; L2 +0.016 (forward target) / +0.299 (trailing target) | `results_v1.4_task3/table4_standalone_inflation_L2_both_targets_formatted.csv` (built from `results_v1.4/task2_table4_inflation_column.csv`) | `NDI_task3_v1_4.ipynb`, `NDI_v1_4.ipynb` |
| Figure 3 (cumulative leak removal) | 0.778 → 0.750 → 0.750 → 0.757 → 0.477; L0 alone 0.757 | `results_v1.4/task2_waterfall_cumulative_formatted.csv`, `figure4_bar_labels.csv`; image `results_v1.4/figure4_leak_waterfall.pdf` / `.png` | `NDI_v1_4.ipynb` |
| Figure 2 (per-participant values, three panels) | — | `results_v1.4_task3/figure2_three_panels.pdf` / `.png` | `NDI_task3_v1_4.ipynb` |

### Persistence and learnability (post hoc diagnostic)

| Manuscript location | Number(s) | Source file | Notebook |
|---|---|---|---|
| Learned vs no-model persistence | self-derived 0.480 vs 0.493; raw-feature −0.015 vs 0.257; mood 0.041 vs 0.161; stress 0.078 vs 0.315 | `results_task4_stage0/stage0_learnability_summary_formatted.csv` | `NDI_stage0_learnability.ipynb` |
| Decomposition into slope and intercept | slopes positive in 69%–71% of folds; slope term 0.165 / 0.075 / 0.195; intercept SD 0.047 vs slope SD 0.031 (raw-feature) | `results_task4_stage0/stage0_diagnostic_summary.csv` (per participant: `stage0_diagnostic_per_participant.csv`) | `NDI_stage0_learnability.ipynb` |
| Anomaly correlation for learned persistence | raw-feature 0.140 vs 0.281 | `results_task4_stage0/stage0b_anomaly_summary_formatted.csv` | `NDI_stage0_learnability.ipynb` |

### Sensitivity analyses

| Manuscript location | Number(s) | Source file | Notebook |
|---|---|---|---|
| Without p12 (self-derived) | V3 vs raw-RF 0.187 [0.104, 0.282], q = .002; vs pers-Z 0.234, q = .002; V-signed vs raw-RF q = .055; V3 vs V-signed 0.063, P = .14 | `results_v1.4/supplementary_without_p12_table2.csv` | `NDI_v1_4.ipynb` |
| Without p12 (mood, stress) | "smallest q = .19" | `results_v1.4_task3/task3_table2_mood_without_p12_formatted.csv`, `task3_table2_stress_without_p12_formatted.csv` | `NDI_task3_v1_4.ipynb` |
| Anomaly correlation (post hoc) | V3 advantage 0.277, q = .001; raw-feature comparisons q = .90; persistence 0.281 vs at most 0.101 | `results_sensitivity_anomaly/anomaly_table2_formatted.csv`, `anomaly_absolute_rho_formatted.csv` | `NDI_sensitivity_anomaly.ipynb` |
| Trailing target (original alignment) | 0.097 [−0.017, 0.203]; raw-feature null | `results_v1.2/task1_table2_formatted.csv` | `NDI_v1.2_Corrections.ipynb` |

### Corrections and data preparation

| Manuscript location | Number(s) | Source file | Notebook |
|---|---|---|---|
| Effect of the corrections | original V3 vs raw-RF 0.193 (self-derived) and −0.076 (raw-feature) → 0.213 and 0.025 | `results_v1.4/comparison_v1_v1_2_v1_3_v1_4_formatted.csv` | `NDI_v1_4.ipynb` |
| Check (1): one row per date | 18 duplicated rows in 9 participants | `results_diagnostics/duplicate_days_summary.csv`, `duplicate_days_reports.csv`; `results_v1.4/rows_v1_3_vs_v1_4.csv` | `NDI_duplicate_days_check.ipynb`, `NDI_v1_4.ipynb` |
| Check (2): p12 loaded | p12: 152 rows, 129 scored folds | `results_v1.4/data_facts_v1_4.txt` | `NDI_v1_4_prechecks.ipynb`, `NDI_v1_4.ipynb` |
| Check (3): sleep-score dating | 1,834 matched scores, 0 mismatches | `results_diagnostics/sleep_score_check_verdict.json`, `sleep_score_check_summary.csv`, `sleep_score_check_rows.csv` | `NDI_sleep_score_check.ipynb` |
| Check (4): wellness rules | 1,747 reports → 1,724 participant-days; 2 blank reports; 20 multi-report days | `results_v1.4/data_facts_v1_4.txt`, `wellness_rules_log.csv`, `wellness_daily_W1_W4.csv` | `NDI_v1_4.ipynb` |
| Datasets | 16 participants, 141–152 days, 2,398 participant-days, t0 = 10 | `results_v1.4/data_facts_v1_4.txt` | `NDI_v1_4.ipynb` |
| Component independence and construct-validity simulation (Multimedia Appendix 3) | earlier pipeline version | `results_final_v1_frozen_backup/` (e.g. `item7_independence_real_data.csv`, `components_pmdata.csv`) | `NDI_Pipeline.ipynb` |

---

## 4. Repository map

```
NDI_Depression_Risk_Index/
│
├── README.md                          ← this file
├── LICENSE                            ← MIT
├── TRIPOD_AI_checklist.docx           ← completed TRIPOD+AI reporting checklist (Multimedia Appendix 5)
├── ndi_pipeline.py                    ← shared pipeline module (neutrosophic decomposition, folds, helpers)
│
├── ── Notebooks, in the order they were run ──
├── NDI_Pipeline.ipynb                 ← v1: original analysis (history; superseded)
├── NDI_v1.2_Corrections.ipynb         ← v1.2: corrected Table 1, Task 1 statistics, leak waterfall (trailing target)
├── NDI_v1_3_forward.ipynb             ← v1.3: target re-aligned to the following 7 days
├── NDI_duplicate_days_check.ipynb     ← diagnostic: finds the 18 duplicated rows (read-only)
├── NDI_v1_4_prechecks.ipynb           ← diagnostic: p12 loader error; wellness zero counts (read-only)
├── NDI_v1_4.ipynb                     ← v1.4: one row per date, p12 included -> final self-derived / raw-feature results
├── NDI_sleep_score_check.ipynb        ← diagnostic: sleep-score dating (read-only)
├── NDI_task3_v1_4.ipynb               ← mood and stress targets, persistence, Figure 2, Table 3
├── NDI_stage0_learnability.ipynb      ← post hoc learnability diagnostic (learned persistence, slope/intercept, anomaly)
├── NDI_sensitivity_anomaly.ipynb      ← post hoc anomaly-correlation sensitivity for all four targets
│
├── ── Frozen configurations (each committed and tagged before its run) ──
├── config_v1_3_forward.json           ← tag v1.3-forward
├── config_v1_4.json                   ← tag v1.4
├── config_task3_v1_4.json             ← tag v1.4-task3
│
└── ── Archived outputs (exactly as written by the runs) ──
    ├── results_final_v1_frozen_backup/   ← v1 outputs, frozen before any correction
    ├── results_v1.2/
    ├── results_v1.3/
    ├── results_v1.4/                     ← final Tasks 1–2 (self-derived, raw-feature, leaks)
    ├── results_v1.4_task3/               ← final mood / stress, persistence, Figure 2, Table 3
    ├── results_diagnostics/              ← duplicate-day and sleep-score checks
    ├── results_task4_stage0/             ← learnability diagnostic
    └── results_sensitivity_anomaly/      ← anomaly-correlation sensitivity
```

**About the names:**
- The prefixes "Task 1", "Task 2" and "Task 3" in file names refer to the work packages of the revision: corrected accuracy, leak quantification, and external targets.
- The folder `results_task4_stage0/` keeps its original name because `NDI_stage0_learnability.ipynb` writes to that exact path.
- **Not in the repository:**
  - the raw datasets (Section 6);
  - scratch and working folders that are not cited in the manuscript.

---

## 5. Version history, tags and corrections

| Version | Tag | What changed | Results |
|---|---|---|---|
| v1 | `v1` | Original analysis | `results_final_v1_frozen_backup/` |
| v1.2 | *(not tagged)* | Whole-series statistics replaced by fold-local and causal statistics; purge added; leak quantification | `results_v1.2/` |
| v1.3 | `v1.3-forward` | Target window moved to the **following** 7 days | `results_v1.3/` |
| v1.4 | `v1.4` | One row per calendar date (18 duplicated rows removed); p12 included (n = 16); wellness table | `results_v1.4/` |
| v1.4 + external targets | `v1.4-task3` | Next-day mood and stress; persistence references; Figure 2; Table 3 | `results_v1.4_task3/` |
| Post hoc diagnostics | run at `v1.4-task3` | Learnability diagnostic; anomaly-correlation sensitivity | `results_task4_stage0/`, `results_sensitivity_anomaly/` |
| Final package | `v1.5-final` | This README and the final repository state cited in the manuscript | all |

**The five corrections to the original implementation** (manuscript: *Corrections to the original implementation*; Multimedia Appendix 1):
1. **Target alignment:** the endpoint was specified as forward-looking, but the code averaged the preceding 7 days. Fixed in v1.3; the trailing results are kept in `results_v1.2/`.
2. **Whole-series statistics:** the T channel, the imputation medians and the raw-target normalisation used the full series. Fixed in v1.2.
3. **Purge:** the embargo left the last training target window touching the test day. Fixed in v1.2.
4. **Duplicated days:** 18 rows. Fixed in v1.4 (`results_diagnostics/`).
5. **Silently excluded participant (p12):** a loader error. Fixed in v1.4 (`NDI_v1_4_prechecks.ipynb`).

`results_v1.4/comparison_v1_v1_2_v1_3_v1_4_formatted.csv` shows how each step moved every primary number.

**Status labels used in the manuscript:**
- **Original analysis:** fixed before results were inspected (tag `v1`).
- **Revision analyses:** specified after the original results were known, and each frozen in a tagged configuration before its own run.
- **Post hoc:** the learnability diagnostic and the anomaly correlation, defined after related results were seen.
- **Exploratory:** the target-swap shift, and V3 vs persistence.

---

## 6. Data

Both datasets are public and are **not** redistributed here.

| Dataset | Role | Download | Citation |
|---|---|---|---|
| **PMData** | Primary dataset: 16 participants, Fitbit Versa 2 and PMSys self-reports | https://datasets.simula.no/pmdata/ | Thambawita V, et al. PMData: a sports logging dataset. *Proc 11th ACM MMSys*, 2020. doi:10.1145/3339825.3394926 |
| **DEPRESJON** | Secondary between-participant analyses (v1 notebook) | https://datasets.simula.no/depresjon/ | Garcia-Ceja E, et al. Depresjon: a motor activity database of depression episodes in unipolar and bipolar patients. *Proc 9th ACM MMSys*, 2018. doi:10.1145/3204949.3208133 |

**Features (10 daily Fitbit features):**
- resting heart rate;
- steps;
- total, light, moderate and vigorous active minutes;
- sedentary minutes;
- sleep efficiency, minutes asleep, sleep score.

**Self-reports:** PMSys wellness, using mood and stress (1–5 scales).

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

The **G-0** cell clones this repository and checks out the notebook's tag, so the code that runs is exactly the tagged code. No token is needed: the repository is public.

### 7.2 Which notebook to run, and what you should see

| Notebook | Tag checked out | Approx. time (Colab CPU) | Writes to | Should end with |
|---|---|---|---|---|
| `NDI_v1_4.ipynb` | `v1.4` | 1–2 h | `results_v1.4/` | `handin_check.txt`: PASSED |
| `NDI_task3_v1_4.ipynb` | `v1.4-task3` | 1–2 h | `results_v1.4_task3/` | first: reproduction check vs `results_v1.4` passes; last: `handin_check_task3.txt` PASSED |
| `NDI_stage0_learnability.ipynb` | `v1.4-task3` | ~1 h | `results_task4_stage0/` | learnability table, diagnostic, Stage 0b table |
| `NDI_sensitivity_anomaly.ipynb` | `v1.4-task3` | 1–1.5 h | `results_sensitivity_anomaly/` | "REPRODUCTION CHECK PASSED: 64 participant-targets × 4 arms identical"; `handin_check_sensitivity.txt` PASSED |
| `NDI_duplicate_days_check.ipynb`, `NDI_sleep_score_check.ipynb`, `NDI_v1_4_prechecks.ipynb` | — | minutes | `results_diagnostics/` or printed only | the counts in Section 3 |

**Two notebooks are history; read them, don't Run all:**
- **`NDI_v1.2_Corrections.ipynb`** was not tagged and contains the whole v1 pipeline before its v1.2 cells. Re-running it would overwrite `results_v1.2/` and can differ in the last decimals under newer Colab package versions. Its saved outputs (run of 7 Oct 2026) include the v1.2 hand-in check: 22 of 22 files present, and the v1 regeneration check passed.
- **`NDI_Pipeline.ipynb`** is the original v1 analysis, kept for transparency.

**Tip:** to re-run without touching your copy of the archive, run on a separate Drive copy. The notebooks only read the folders of earlier stages, and their hand-in checks confirm that no earlier file was modified.

---

## 8. Results folders, file by file

Most tables come in two forms:
- `*_formatted.csv`: the table as printed in the manuscript;
- the plain file: full precision, every column.

Every folder from v1.2 onward has a `provenance.json` (run start time, HEAD commit, tag commit, whether the config matches the tagged copy, participants loaded) and a hand-in check (`handin_check*.txt`).

### `results_v1.4/` (final self-derived and raw-feature results, leaks)

| File(s) | Contents |
|---|---|
| `config_v1_4.json`, `provenance.json` | Frozen config (tag `v1.4`); provenance |
| `data_facts_v1_4.txt`, `date_check_v1_4.csv`, `rows_v1_3_vs_v1_4.csv` | Data facts; date check; rows removed vs v1.3 |
| `wellness_daily_W1_W4.csv`, `wellness_rules_log.csv`, `wellness_zero_counts.csv` | Self-report table and the W1–W4 rule log |
| `table1_corrected_self_target_per_participant.csv`, `raw_target_corrected_per_participant.csv` | Per-participant ρ, four arms, both targets |
| `task1_absolute_rho_formatted.csv`, `task1_absolute_rho_both_targets.csv` | **Table 1** (self-derived, raw-feature columns) |
| `task1_table2_formatted.csv`, `task1_table2_bounded_null.csv` | **Table 2** (self-derived, raw-feature); equivalence and artifact bounds |
| `exploratory_target_swap_shift*.csv`, `exploratory_notes.txt` | Target-swap shift |
| `task2_table4_inflation_column*.csv`, `task2_one_at_a_time*.csv` | Standalone leak inflation (**Table 3** inputs) |
| `task2_waterfall_cumulative*.csv`, `figure4_leak_waterfall.pdf/.png`, `figure4_bar_labels.csv` | **Figure 3** (cumulative leak removal) |
| `task2_v1_2_v1_3_v1_4.csv`, `comparison_v1_v1_2_v1_3_v1_4*.csv`, `v1_vs_corrected.csv` | Effect of each correction |
| `supplementary_without_p12_*.csv`, `supplementary_without_p12_notes.txt` | Sensitivity analysis without p12 |
| `fold_log_self_target.csv`, `fold_log_raw_target.csv` | Every fold: training/test days, embargo, purge |
| `forward_target_dropped_test_days.csv`, `item2_day_arithmetic.txt` | Day arithmetic of the forward target window |
| `independent_check_clean_endpoint.csv`, `independent_check_notes.txt` | Independent re-implementation of the fold loop (machine precision) |
| `task1_notes.txt`, `handin_check.txt` | Notes; hand-in check |

### `results_v1.4_task3/` (mood, stress, persistence, Figure 2, Table 3)

| File(s) | Contents |
|---|---|
| `config_task3_v1_4.json`, `provenance.json` | Frozen config (tag `v1.4-task3`); provenance |
| `task3_eligibility_log.csv` | Eligible participants per target, and why (14 each) |
| `task3_mood_per_participant.csv`, `task3_stress_per_participant.csv` | Per-participant ρ, four arms |
| `task3_absolute_rho_{mood,stress}*.csv` | **Table 1** (mood, stress columns) |
| `task3_table2_{mood,stress}*.csv` | **Table 2** (mood, stress) |
| `*_without_p12*` | Sensitivity analysis without p12 |
| `persistence_reference_per_participant.csv`, `persistence_reference_summary*.csv` | Persistence row of **Table 1** |
| `v3_vs_persistence_self_derived_exploratory.csv` | V3 vs persistence |
| `table4_standalone_inflation_L2_both_targets*.csv` | **Table 3** |
| `figure2_three_panels.pdf/.png` | **Figure 2** |
| `figure_supplement_stress.pdf/.png` | Stress figure (Multimedia Appendix 2) |
| `figure4_leak_waterfall_from_L2_removed.*`, `figure4_from_L2_removed_bar_labels.csv` | Leak waterfall variant |
| `task1_table2_formatted_v1_4_corrected.csv`, `reporting_notes_v1_4.txt` | Reporting items |
| `task3_fold_log_{mood,stress}.csv` | Fold logs |
| `task3_independent_check*.{csv,txt}` | Independent check |
| `task3_notes.txt`, `task3_without_p12_notes.txt`, `handin_check_task3.txt` | Notes; hand-in check |

### `results_task4_stage0/` (learnability diagnostic, post hoc)

| File(s) | Contents |
|---|---|
| `stage0_config.json`, `provenance.json` | Settings; provenance |
| `stage0_learnability_per_participant.csv`, `stage0_learnability_summary*.csv` | Learned vs no-model persistence |
| `stage0_decision.json` | Outcome of the pre-set learnability rule |
| `stage0_diagnostic_per_participant.csv`, `stage0_diagnostic_summary.csv` | Intercept vs slope decomposition |
| `stage0b_anomaly_per_day.csv`, `stage0b_anomaly_per_participant.csv`, `stage0b_anomaly_summary*.csv`, `stage0b_reading.json` | Anomaly correlation for learned persistence |
| `stage0_notes.txt`, `stage0b_notes.txt` | Notes |

### `results_sensitivity_anomaly/` (anomaly-correlation sensitivity, post hoc)

| File(s) | Contents |
|---|---|
| `config_sensitivity_anomaly.json`, `provenance.json` | Metric definition; provenance |
| `reproduction_check.csv` | Archived vs re-run ρ: 64 participant-targets × 4 arms, all identical |
| `persistence_check.csv` | Ordinary persistence reproduces the archived values |
| `per_day_predictions.csv` | Every scored day: target, participant, day, fold mean a(s), outcome, today's value, four arms' predictions |
| `anomaly_rho_per_participant.csv` | Anomaly and ordinary ρ per participant and arm |
| `anomaly_table2*.csv`, `anomaly_absolute_rho*.csv` | Comparisons and arm means under the anomaly correlation |
| `sensitivity_anomaly_notes.txt`, `handin_check_sensitivity.txt` | Notes; hand-in check |

### `results_diagnostics/` (data-integrity checks)

| File(s) | Contents |
|---|---|
| `duplicate_days_reports.csv`, `duplicate_days_summary.csv` | Duplicated dates, the reports behind them, and counts per participant (check 1) |
| `sleep_score_check_rows.csv`, `sleep_score_check_summary.csv`, `sleep_score_check_verdict.json` | Sleep scores matched to Fitbit sleep logs by log identifier (check 3) |

### `results_v1.2/` and `results_v1.3/` (earlier corrected versions)
These have the same structure as `results_v1.4/`:
- **v1.2** has the trailing target and n = 15. It's the source of the trailing-target sensitivity row.
- **v1.3** has the forward target, before de-duplication.

They're kept so that every step in `comparison_v1_v1_2_v1_3_v1_4*.csv` can be checked.

### `results_final_v1_frozen_backup/` (v1, frozen)
These are the original v1 outputs (with `manifest.json`), copied before any correction and never modified. Component independence and the construct-validity simulation come from this pipeline version.

---

## 9. Methods facts used in the code

| Item | Value | Where set |
|---|---|---|
| Participants | 16 (PMData); 14 eligible for mood and for stress | `results_v1.4/data_facts_v1_4.txt`; `task3_eligibility_log.csv` |
| Eligibility | ≥ 15 rows and ≥ 10 scored test folds (`MIN_FOLDS`) | `config_v1_4.json`, `data_facts_v1_4.txt` |
| Cross-validation | blocked forward-chaining, 1-day test folds | `ndi_pipeline.py`, V12-1 / V12-2 cells |
| Minimum training window | t0 = min(10, max(5, n // 3)), so 10 days for every participant | `data_facts_v1_4.txt` |
| Embargo / purge | 6 days; training rows whose target window overlaps the test day are removed | `config_v1_4.json` |
| Targets | self-derived and raw-feature: mean over the following 7 days; mood and stress: next day | `config_v1_4.json`, `config_task3_v1_4.json` |
| Index weights | fixed (0.2, 0.3, 0.5), never tuned on the evaluation | `config_v1_4.json` |
| Random forests | 100 trees, max depth 4, fixed seed | `ndi_pipeline.py` |
| Personalised z-score | training-fold standardisation + ridge, α = 1.0 | V12-2 cell |
| Statistics | Spearman ρ within participant; BCa bootstrap 95% CI (10,000 resamples, seed 42); Hodges–Lehmann; Wilcoxon signed-rank; Benjamini–Hochberg over the four neutrosophic-vs-baseline comparisons per target; TOST at ±0.10 on the 90% CI | `config_v1_4.json`; V12-6 cell |

---

## 10. Reproducibility safeguards

- **Config before results.** From v1.3 on, each configuration was committed and tagged before its run. Each notebook checks that its config equals the tagged copy, and stops if not.
- **Tagged code runs.** G-0 checks out the tag, and `provenance.json` records HEAD and the tag commit.
- **Reproduce before extending.** The Task 3 and sensitivity notebooks first reproduce the archived per-participant values exactly (tolerance 1e-9), and stop if they cannot.
- **Independent re-implementation.** A separate implementation of the fold loop reproduces the clean endpoint to machine precision (`independent_check_*`).
- **Archives are never modified.** Each hand-in check confirms that no file in an earlier results folder changed during the run.
- **No credentials or raw data** are stored in this repository.
- **Reporting checklist.** Reporting follows TRIPOD+AI (Collins et al., *BMJ* 2024;385:e078378). The completed checklist, with the manuscript page and section for every item, is in `TRIPOD_AI_checklist.docx` (Multimedia Appendix 5).

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

**This work:** Patel H, Kandasamy I. Apparent performance of a neutrosophic daily depression index reflects target affinity: leakage-free methodological evaluation study. Submitted to *JMIR AI*, 2026. (DOI to be added on publication.)

**Datasets:** please cite PMData and DEPRESJON as in Section 6.

**Licence:** MIT; see [LICENSE](LICENSE). The datasets keep their own licences.
