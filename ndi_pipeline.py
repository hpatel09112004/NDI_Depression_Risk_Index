"""
ndi_pipeline.py
===============
Shared NDI Pipeline Module — Neutrosophic Depression Index
Undergraduate Project Brief (Section 4):
  "Both the real-data arm and the simulation arm must call exactly the same functions."

Five technical components (Brief Section 2):
  1. neutrosophic_decompose()     — maps raw features to (T, I, F)
  2. blocked_tscv()               — leakage-free CV, embargo gap g=6
  3. run_baselines()              — B1/B2/B3 + lift with bootstrap CI
  4. lambda_grid_search()         — simplex search, step s=0.05, 231 points
  5. shap_attribution()           — SHAP + permutation importance

Reproducibility requirements (Brief Section 6):
  - Global seed RANDOM_SEED = 42, all per-run seeds derived from it
  - All parameters in CONFIG block (single source of truth)
  - Fold indices saved via save_fold_indices()
  - Every metric returned with 95% block-bootstrap CI
  - One module, zero duplication between Arm 1 and Arm 2

Author : VIT University NDI Project, May 2026
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score, roc_auc_score, roc_curve)
from sklearn.inspection import permutation_importance as sk_perm_imp
from scipy.stats import spearmanr, kendalltau
import warnings
warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURATION BLOCK  (Brief Section 6 — all params here)
# ================================================================
CONFIG = {
    # Reproducibility
    "RANDOM_SEED"    : 42,

    # Blocked TS-CV  (Brief Algorithm 1)
    "T0"             : 14,   # minimum training days
    "EMBARGO_G"      : 6,    # embargo gap = smoothing_k - 1
    "FOLD_H"         : 1,    # test fold size (days)
    "SMOOTH_K"       : 7,    # NDI smoothing window

    # Lambda simplex grid  (Brief Sec 4.1, step s=0.05 => 231 points)
    "LAMBDA_STEP"    : 0.05,
    "LAMBDA_ETA"     : 0.05, # stability-radius tolerance (Brief Eq 9)

    # Default NDI weights (overridden by grid-search result)
    "LAMBDA_T"       : 0.20,
    "LAMBDA_I"       : 0.30,
    "LAMBDA_F"       : 0.50,
    
    "F_SCALE"       : 3.0,

    # Bootstrap
    "N_BOOT"         : 2000,

    # Permutation importance repeats
    "PERM_REPEATS"   : 30,

    # Clinical thresholds
    "MADRS_THRESHOLD": 7,    # MADRS >= 7 = mild elevation
    "PHQ_THRESHOLD"  : 3,    # PHQ-9 >= 3

    # Model hyperparameters  (reported for reproducibility, Brief Sec 6)
    "RF_PARAMS": dict(n_estimators=100, max_depth=4,
              min_samples_leaf=2, random_state=42),
    "GB_PARAMS"      : dict(n_estimators=200, learning_rate=0.05,
                            max_depth=3, subsample=0.8, random_state=42),
    "RIDGE_ALPHA"    : 1.0,
}

np.random.seed(CONFIG["RANDOM_SEED"])
_MODULE_RNG = np.random.default_rng(CONFIG["RANDOM_SEED"])


# ================================================================
# CONFIG additions -- V3 (spec Sec 2, order-of-work #2)
# Provisional weights only (matches Cell 14.7's 0.5/0.5 go/no-go
# proxy). Set deliberately before running the real V3 frame -- do
# not tune after seeing clinical results (spec Sec 6).
# ================================================================
CONFIG["W_MISS"] = 0.5
CONFIG["W_INST"] = 0.5


# ================================================================
# FEATURE RISK-DIRECTION MAP
# Sign convention: -1 = DECREASING values indicate elevated risk,
#                  +1 = INCREASING values indicate elevated risk.
# Every feature passed through build_neutrosophic_features MUST
# appear here. Falsity (F) measures evidence of a trend moving in
# the RISK direction for that specific feature — a wrong or missing
# entry silently nulls out F for that feature (see get_risk_direction).
# ================================================================
FEATURE_RISK_DIRECTION = {
    # DEPRESJON actigraphy features (all of DEPRESJON_FEATURE_NAMES)
    # Depressed patients show reduced monotone motor activity.
    "sim_feature":       -1,
    "act_mean":          -1,
    "act_std":           -1,
    "act_median":        -1,
    "act_iqr":           -1,
    "act_max":           -1,
    "act_zero_rate":     +1,
    "act_total":         -1,
    "act_night_mean":    -1,
    "act_day_mean":      -1,
    "day_night_ratio":   -1,
    "L5":                -1,
    "M10":               -1,
    "RA":                -1,
    "act_trend":         -1,
    "act_entropy":       -1,
    # PMData Fitbit features
    "resting_hr":            +1,
    "hrv_rmssd":             -1,
    "steps":                 -1,
    "active_minutes":        -1,
    "active_minutes_total":  -1,
    "lightly_active_min":    -1,
    "moderately_active_min": -1,
    "very_active_min":       -1,
    "sedentary_minutes":     +1,
    "sedentary_min":         +1,
    "mean_activity":         -1,
    "std_activity":          -1,
    "sum_activity":          -1,
    "log1p_activity":        -1,
    "activity_range":        -1,
    "zero_fraction":         +1,
    # Sleep domain
    "sleep_efficiency":  -1,
    "sleep_minutes":     -1,
    "sleep_score":       -1,
    "sleep_quality":     -1,
    "rem_ratio":         -1,
    # PMData wellness
    "fatigue":           +1,
    "soreness":          +1,
    "stress":            +1,
    "readiness":         -1,
    "mood":              -1,
}


# ================================================================
# TWO-SIDED FEATURES
# Features pathological at BOTH extremes (e.g. sleep too short OR
# too long). For these, F should grade "how far from the healthy
# band" in either direction, not just drift in one risk_direction.
#
# This is a modelling decision, not an implementation detail. Per
# the advisor's closeout note (Item 2): sleep_minutes is confirmed
# as the one evidence-backed U-shaped case; other features (mood,
# stress, HR) are deliberately NOT included without similar evidence
# -- do not add more here without the same confirmation step.
# ================================================================
TWO_SIDED_FEATURES = {
    "sleep_minutes",
}


def get_two_sided(feature_name):
    """Whether this feature should use the U-shaped (two-sided) F.
    Defaults to False — only features explicitly added to
    TWO_SIDED_FEATURES (after advisor confirmation) opt in.
    """
    return feature_name in TWO_SIDED_FEATURES


def get_risk_direction(feature_name, default=None):
    """
    Look up a feature's risk direction. Fails loudly rather than
    silently defaulting — a wrong default nulls out F for that
    feature exactly the way the original T=F artifact did.

    Pass `default` explicitly (e.g. -1) only for exploratory/ad-hoc
    features not yet vetted; production feature lists should have
    every name present in FEATURE_RISK_DIRECTION.
    """
    if feature_name in FEATURE_RISK_DIRECTION:
        return FEATURE_RISK_DIRECTION[feature_name]
    if default is not None:
        return default
    raise KeyError(
        f"'{feature_name}' has no entry in FEATURE_RISK_DIRECTION. "
        f"Add it explicitly — do not guess a default, since the wrong "
        f"sign silently nulls this feature's Falsity contribution.")


def normalized_mi(T, F, bins=20):
    """
    MI(T,F) normalised by min(H(T),H(F)) — V2' spec Sec 5 / closeout
    note item 1. Distinguishes trivial-but-real dependence from
    material dependence, reported ALONGSIDE the shuffle-null MI test,
    not in place of it.

    IMPLEMENTATION NOTE (deliberate departure — see chat): the spec's
    formula reads naturally as continuous differential entropy, but
    that is NOT usable here — differential entropy can be negative
    for bounded/concentrated distributions (confirmed empirically: a
    strongly-dependent synthetic pair produced norm=-83 using
    scipy.stats.differential_entropy, which is meaningless as a 0-1
    ratio). Discrete/binned Shannon entropy is always >= 0 and
    bounded by log(bins), which is what a normalised-MI ratio needs
    to be interpretable. T and F are quantile-binned into `bins` bins
    each; MI and both marginal entropies are computed on the binned
    labels. This is a different estimator from the continuous
    (Kraskov) MI used elsewhere for the shuffle-null pass/fail gate —
    that gate is untouched by this function; this is purely an
    additional, separately-computed diagnostic number.

    Returns (norm_mi, mi_discrete_nats, H_T_nats, H_F_nats).
    """
    from sklearn.metrics import mutual_info_score

    T = np.asarray(T, dtype=float)
    F = np.asarray(F, dtype=float)
    valid = np.isfinite(T) & np.isfinite(F)
    T, F = T[valid], F[valid]

    def _bin(x, bins):
        edges = np.quantile(x, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)  # collapse duplicate edges (ties / saturation)
        if len(edges) < 3:
            return np.zeros(len(x), dtype=int)  # degenerate: all one bin
        return np.digitize(x, edges[1:-1])

    T_bin = _bin(T, bins)
    F_bin = _bin(F, bins)

    def _entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log(p)).sum())

    mi_discrete = float(mutual_info_score(T_bin, F_bin))
    H_T = _entropy(T_bin)
    H_F = _entropy(F_bin)
    denom = min(H_T, H_F)
    norm_mi = float(mi_discrete / denom) if denom > 1e-12 else float("nan")
    return norm_mi, mi_discrete, H_T, H_F


# ================================================================
# COMPONENT 1 — NEUTROSOPHIC DECOMPOSITION  (Brief Sec 2, item 1)
# N : R^d -> R^3d
# Personal-baseline statistics computed INSIDE each training fold.
# ================================================================

def neutrosophic_decompose(series, train_mask=None, std_ref=None,
                            trend_window=7, risk_direction=-1,
                            two_sided=False,
                            f_mode="personal",
                            pop_median=None, pop_mad=None,
                            pop_vol_median=None, pop_vol_mad=None,
                            i_mode="personal_volatility",
                            missing_rate=None, instability=None):
    """
    Map one raw feature time-series to neutrosophic components (T, I, F).

    Parameters
    ----------
    series         : array-like (T,)   raw daily values
    train_mask     : bool array (T,)   if given, baseline (median, MAD, std)
                      is computed on training days only — required for
                      fold-internal computation per Brief Algorithm 1 step 5-6.
                      Pass None for in-sample (expanding-window) mode.
    std_ref        : float, optional. Training-only normaliser for I.
                      Required (directly or via train_mask) — whole-series
                      max is look-ahead leakage.
    trend_window   : int, window length (days) for the causal rolling
                      trend used by Falsity.
    risk_direction : +1 if INCREASING values indicate elevated risk
                      (e.g. resting_hr, sedentary_minutes), or
                      -1 if DECREASING values indicate elevated risk
                      (e.g. steps, sleep_efficiency, hrv_rmssd).
                      Callers should look this up per feature via
                      get_risk_direction() / FEATURE_RISK_DIRECTION rather
                      than relying on this default — mixed-polarity
                      feature sets will silently null out F for whichever
                      features get the wrong sign.
    two_sided      : bool. If True, F grades distance-from-healthy-band
                      in EITHER direction (e.g. sleep_minutes — too
                      little or too much both indicate risk). Overrides
                      risk_direction for the F computation only; T and I
                      are unaffected. Look this up via get_two_sided() /
                      TWO_SIDED_FEATURES rather than hardcoding per call.
    f_mode         : 'personal' (default, V-signed) — F is the causal
                      trend normalised by the person's own MAD.
                      'population' (V2, spec Sec 2/4) — F is the
                      person's own causal median LEVEL compared against
                      a population norm. Requires pop_median/pop_mad.
    pop_median      : float, required if f_mode='population'. Per-
                      feature population median, computed in the outer
                      LOPO loop from donor participants only (never
                      from the participant this call's `series` belongs
                      to). No default — see raise below.
    pop_mad         : float, required if f_mode='population'. Per-
                      feature population MAD, paired with pop_median.
    f_mode also accepts 'population_vol' (V2', spec Sec 4) — F becomes
                      a BETWEEN-person volatility quantity: how this
                      person's day-to-day variability compares to the
                      population's, instead of a level comparison.
                      Fixes V2's T-F coupling (both were level-
                      anchored to the personal median) by putting F on
                      a different axis from T. Requires pop_vol_median/
                      pop_vol_mad instead of pop_median/pop_mad.
    pop_vol_median  : float, required if f_mode='population_vol'.
                      Median, across donor participants, of each
                      donor's OWN volatility (std) for this feature —
                      i.e. the population norm for "how variable is
                      variable". From the LOPO outer loop, donors only.
    pop_vol_mad     : float, required if f_mode='population_vol'.
                      MAD of donor volatilities, paired with
                      pop_vol_median.
    i_mode         : 'personal_volatility' (default, V-signed/V2/V2') —
                      I is normalised expanding std, exactly as before,
                      UNCHANGED. 'quality' (V3, spec Sec 2) — I is the
                      data-quality channel (missing_rate + instability)
                      via compute_I_v3, NOT a function of x's deviation.
                      One definition of I per mode, one path (note:
                      "I cleanup") — no silent fallback between them;
                      picking 'quality' without missing_rate/instability
                      raises rather than guessing.
    missing_rate    : float or array, required if i_mode='quality'.
                      From compute_quality_signal.
    instability     : float or array, required if i_mode='quality'.
                      From compute_quality_signal, paired with
                      missing_rate.

    Returns
    -------
    T, I, F : np.ndarray (T,), each in [0, 1]

    Notes
    -----
    Truth   = Gaussian closeness to personal baseline (magnitude of
              deviation, undirected).
    Indeter = mode-gated (see i_mode): normalised expanding std
              (default, V-signed/V2/V2') or data-quality channel
              (i_mode='quality', V3) — independent of T either way.
    Falsity = REFRAMED (previous revision used clip(robust_z/3), a
              monotonic function of the same z that defines T, forcing
              Spearman(T,F) ~= -1 by construction regardless of data —
              T and F were doing the same job). F now measures evidence
              of a trend moving in the RISK direction for this specific
              feature, sized against the TOTAL change over the trend
              window (not a single day's slope) relative to the
              feature's own personal MAD. A per-day slope threshold is
              unreachable for real physiological data (would require the
              entire personal MAD to shift in one day); comparing the
              window's total displacement to MAD gives F=1 at a
              realistic threshold — roughly one MAD of sustained drift
              across `trend_window` days. Only trend movement in the
              feature's own risk direction counts; movement the other
              way gives F=0.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)

    if train_mask is not None:
        # Fold-internal baseline — computed on training days only
        tr_vals     = x[np.asarray(train_mask, dtype=bool)]
        base_median = np.nanmedian(tr_vals)
        base_mad    = np.nanmedian(np.abs(tr_vals - base_median)) + 1e-8
        exp_median  = np.full(n, base_median)
        exp_mad     = np.full(n, base_mad)
        # exp_std stays genuinely per-day (expanding), NOT a flat
        # constant -- else Indeterminacy collapses to ~1.0 every day.
        # Only std_ref (below) is training-only; the statistic itself
        # stays causal-daily.
        exp_std     = pd.Series(x).expanding().std().fillna(0).values
    else:
        # In-sample expanding-window baseline
        s           = pd.Series(x)
        exp_median  = s.expanding().median().values
        exp_mad     = (s - pd.Series(exp_median)).abs()\
                       .expanding().median().values + 1e-8
        exp_std     = s.expanding().std().fillna(0).values

    # Truth: Gaussian closeness to personal median (magnitude only,
    # undirected — independent of trend direction)
    T = np.exp(-((x - exp_median) ** 2) / (2 * exp_mad ** 2 + 1e-8))

    # Indeterminacy — mode-gated (note "I cleanup": one definition,
    # one path, no silent fallback between them).
    if i_mode == "quality":
        if missing_rate is None or instability is None:
            raise ValueError(
                "i_mode='quality' requires missing_rate and instability "
                "from compute_quality_signal — refusing to fall back "
                "to personal-volatility I silently")
        I = compute_I_v3(missing_rate, instability)
    else:
        # personal_volatility (default) — UNCHANGED, exactly the
        # V-signed/V2/V2' formula, normalised expanding std.
        if std_ref is None:
            if train_mask is not None:
                std_ref = exp_std[np.asarray(train_mask, dtype=bool)].max() + 1e-8
            else:
                raise ValueError(
                    "std_ref or train_mask required — "
                    "whole-series max is look-ahead leakage")
        I = np.clip(exp_std / std_ref, 0, 1)

    # Falsity: causal rolling-window TOTAL change, signed by this
    # feature's own risk direction, normalised by MAD. A window ending
    # at day t uses only days <= t, so this is causal regardless of
    # train_mask -- same principle already applied to exp_std above.
    if f_mode == "population":
        if pop_median is None or pop_mad is None:
            raise ValueError(
                "f_mode='population' requires pop_median and pop_mad "
                "from the LOPO outer loop — refusing to fall back "
                "to personal baseline silently")
        # V2 (population-referenced F, spec Sec 2/4): compare the
        # person's own causal median LEVEL to the population norm,
        # instead of V-signed's TREND vs personal MAD. exp_median
        # above is already exactly this person-level quantity (same
        # one T is built from) — reused here, not recomputed. This is
        # the one deliberate conceptual change for this arm: F asks
        # "is this person's typical value abnormal?" (a level
        # question), not "is it drifting?" (a trend question). T and
        # I above are completely untouched by f_mode.
        pop_dev = (exp_median - pop_median) / (CONFIG["F_SCALE"] * pop_mad + 1e-8)
        if two_sided:
            F = np.clip(np.tanh(np.abs(pop_dev)), 0, 1)
        else:
            F = np.clip(np.tanh(np.clip(risk_direction * pop_dev, 0, None)), 0, 1)

        return T, I, F

    if f_mode == "population_vol":
        if pop_vol_median is None or pop_vol_mad is None:
            raise ValueError(
                "f_mode='population_vol' requires pop_vol_median and "
                "pop_vol_mad from the LOPO outer loop — refusing to "
                "fall back to personal baseline silently")
        # V2' (population-referenced VOLATILITY F, spec Sec 1/4): why
        # V2 failed — V2's F compared the person's own LEVEL to a
        # population level norm, and T also measures distance from
        # that same person's level. Both anchored to the personal
        # median, so they moved together (confirmed: DEPRESJON
        # Spearman(T,F) jumped to +0.43 under V2). V2' puts F on a
        # DIFFERENT axis — between-person VOLATILITY — so it reads
        # different information from T's within-person level measure.
        # T and I above are still completely untouched by f_mode.
        pers_vol = pd.Series(x).expanding(min_periods=3).std().fillna(0).values
        vol_dev = (pers_vol - pop_vol_median) / (CONFIG["F_SCALE"] * pop_vol_mad + 1e-8)
        if two_sided:
            # Unusually LOW or HIGH variability both count as risk.
            F = np.clip(np.tanh(np.abs(vol_dev)), 0, 1)
        else:
            # Direction note (spec Sec 4): risk is about variability,
            # not the raw feature, so the sign is flipped relative to
            # the level-based F — REDUCED variability is the risk
            # direction for most physiological/behavioural channels
            # (blunted affect/activity is a documented depression
            # marker). This is a per-domain assumption and must be
            # stated in the manuscript, same as FEATURE_RISK_DIRECTION.
            F = np.clip(np.tanh(np.clip(-vol_dev, 0, None)), 0, 1)

        return T, I, F

    # ---- f_mode == "personal" (V-signed) — unchanged below ----
    def _causal_slope(window_vals):
        if len(window_vals) < 3:
            return 0.0
        t_idx = np.arange(len(window_vals))
        valid = ~np.isnan(window_vals)
        if valid.sum() < 3:
            return 0.0
        slope = np.polyfit(t_idx[valid], window_vals[valid], 1)[0]
        return slope

    rolling_slope = pd.Series(x).rolling(
        trend_window, min_periods=3
    ).apply(_causal_slope, raw=True).fillna(0.0).values

    total_change        = rolling_slope * trend_window
    normalized_change   = total_change / (CONFIG["F_SCALE"] * exp_mad + 1e-8)

    if two_sided:
        # Either direction of sustained drift counts as risk —
        # risk_direction is ignored for two-sided features.
        # SIMPLIFICATION (closeout note, Item 2 — record for manuscript):
        # np.abs() scores drift toward MORE of the feature and drift
        # toward LESS of it as identical evidence, conflating two
        # clinically distinct states (e.g. oversleeping vs.
        # undersleeping both read as the same F). Acceptable as a
        # first cut per advisor sign-off; must be stated explicitly
        # as a simplification in the manuscript, not left implicit.
        F = np.clip(np.tanh(np.abs(normalized_change)), 0, 1)
    else:
        risk_aligned_change = risk_direction * normalized_change
        # clip(x,0,None) zeroes non-risk-direction drifts; tanh grades the rest
        F = np.clip(np.tanh(np.clip(risk_aligned_change, 0, None)), 0, 1)

    return T, I, F


def compute_ndi(T_dep, I_dep, F_dep, lambda_vec=None):
    """
    NDI formula  (Brief Section 2, Eq. 1).
    NDI_t = lF * F_dep + lI * I_dep - lT * T_dep,  clipped to [-1, 1].
    lambda_vec = (lT, lI, lF), must sum to 1.
    Defaults to CONFIG values if lambda_vec is None.
    """
    if lambda_vec is None:
        lT = CONFIG["LAMBDA_T"]
        lI = CONFIG["LAMBDA_I"]
        lF = CONFIG["LAMBDA_F"]
    else:
        lT, lI, lF = lambda_vec
    return np.clip(lF * F_dep + lI * I_dep - lT * T_dep, -1, 1)


def build_neutrosophic_features(feature_matrix, feature_names,
                                 train_mask=None,
                                 h_sleep=None,
                                 missing_rate=None,
                                 lambda_vec=None,
                                 risk_directions=None,
                                 two_sided_features=None,
                                 f_mode="personal",
                                 pop_median=None,
                                 pop_mad=None,
                                 pop_vol_median=None,
                                 pop_vol_mad=None):
    """
    Apply neutrosophic decomposition to a feature matrix and return a
    DataFrame with per-feature T/I/F columns, domain aggregates, NDI,
    and NDI_smooth.

    Parameters
    ----------
    feature_matrix  : np.ndarray (T, d)
    feature_names   : list[str]  length d
    train_mask      : bool array (T,) for fold-internal baselines
    h_sleep         : np.ndarray (T,) REM transition entropy (for improved I)
    missing_rate    : np.ndarray (T,) fraction of missing values per day
    lambda_vec      : (lT, lI, lF) — defaults to CONFIG values
    risk_directions : dict {feature_name: +1/-1}, optional. Overrides
                      FEATURE_RISK_DIRECTION for this call only. If a
                      feature name is missing from both this dict and
                      FEATURE_RISK_DIRECTION, get_risk_direction() raises
                      rather than silently guessing.
    two_sided_features : set[str], optional. Overrides TWO_SIDED_FEATURES
                      for this call only.
    f_mode          : 'personal' (default, V-signed) | 'population' (V2).
                      See neutrosophic_decompose.
    pop_median      : np.ndarray (d,), required if f_mode='population'.
                      Per-feature population median, one value per
                      entry in feature_names, SAME ORDER. Computed once
                      per held-out participant in the outer LOPO loop
                      (V2 spec Sec 4) from donor participants only —
                      never from the participant feature_matrix belongs
                      to.
    pop_mad         : np.ndarray (d,), required if f_mode='population'.
                      Per-feature population MAD, paired with pop_median.
    f_mode also accepts 'population_vol' (V2'). Requires pop_vol_median/
                      pop_vol_mad instead of pop_median/pop_mad.
    pop_vol_median  : np.ndarray (d,), required if f_mode='population_vol'.
                      Per-feature median of donor participants' OWN
                      volatility (std) for that feature — the
                      population norm for "how variable is variable".
                      SAME ORDER as feature_names. From the LOPO outer
                      loop, donors only.
    pop_vol_mad     : np.ndarray (d,), required if f_mode='population_vol'.
                      Per-feature MAD of donor volatilities, paired
                      with pop_vol_median.

    Returns
    -------
    pd.DataFrame  with all T/I/F columns, Phys/Sleep/Dream aggregates,
                  Depression_T/I/F, NDI, NDI_smooth
    """
    T_len = feature_matrix.shape[0]
    out   = {}

    if train_mask is None:
        raise ValueError(
            "train_mask required — whole-series max is look-ahead leakage. "
            "For ground-truth/generative use with no train/test split, "
            "pass train_mask=np.ones(T, dtype=bool) explicitly.")
    tm = np.asarray(train_mask, dtype=bool)

    if h_sleep is None:
        h_sleep = np.zeros(T_len)
    if missing_rate is None:
        missing_rate = np.zeros(T_len)

    risk_directions = risk_directions or {}
    two_sided_set   = two_sided_features if two_sided_features is not None \
                       else TWO_SIDED_FEATURES

    if f_mode == "population":
        if pop_median is None or pop_mad is None:
            raise ValueError(
                "f_mode='population' requires pop_median and pop_mad "
                "(per-feature arrays from the LOPO outer loop) — "
                "refusing to fall back to personal baseline silently")
        pop_median = np.asarray(pop_median, dtype=float)
        pop_mad    = np.asarray(pop_mad, dtype=float)
        if len(pop_median) != len(feature_names) or len(pop_mad) != len(feature_names):
            raise ValueError(
                f"pop_median/pop_mad length ({len(pop_median)},"
                f"{len(pop_mad)}) must match feature_names length "
                f"({len(feature_names)})")

    if f_mode == "population_vol":
        if pop_vol_median is None or pop_vol_mad is None:
            raise ValueError(
                "f_mode='population_vol' requires pop_vol_median and "
                "pop_vol_mad (per-feature arrays from the LOPO outer "
                "loop) — refusing to fall back to personal baseline "
                "silently")
        pop_vol_median = np.asarray(pop_vol_median, dtype=float)
        pop_vol_mad    = np.asarray(pop_vol_mad, dtype=float)
        if len(pop_vol_median) != len(feature_names) or len(pop_vol_mad) != len(feature_names):
            raise ValueError(
                f"pop_vol_median/pop_vol_mad length ({len(pop_vol_median)},"
                f"{len(pop_vol_mad)}) must match feature_names length "
                f"({len(feature_names)})")

    for j, name in enumerate(feature_names):
        # Look up this feature's risk direction explicitly (local
        # override first, then the module-level map). Raises if the
        # feature is unknown, rather than silently defaulting.
        risk_dir = risk_directions.get(
            name, get_risk_direction(name))
        is_two_sided = name in two_sided_set

        T_j, I_j, F_j = neutrosophic_decompose(
            feature_matrix[:, j], train_mask=tm, risk_direction=risk_dir,
            two_sided=is_two_sided,
            f_mode=f_mode,
            pop_median=(pop_median[j] if f_mode == "population" else None),
            pop_mad=(pop_mad[j] if f_mode == "population" else None),
            pop_vol_median=(pop_vol_median[j] if f_mode == "population_vol" else None),
            pop_vol_mad=(pop_vol_mad[j] if f_mode == "population_vol" else None))

        # Improved I formula (Brief teacher point 3.1):
        # I = 0.5*volatility + 0.3*H_sleep + 0.2*missing_rate
        vol     = I_j
        h_norm  = np.clip(h_sleep      / (h_sleep[tm].max()      + 1e-8), 0, 1)
        mr_norm = np.clip(missing_rate / (missing_rate[tm].max() + 1e-8), 0, 1)
        I_j = np.clip(0.5*vol + 0.3*h_norm + 0.2*mr_norm, 0, 1)

        out[f"{name}_T"] = T_j
        out[f"{name}_I"] = I_j
        out[f"{name}_F"] = F_j

    df = pd.DataFrame(out)

    # Domain aggregates (mean across features in each domain)
    T_cols = [f"{n}_T" for n in feature_names]
    I_cols = [f"{n}_I" for n in feature_names]
    F_cols = [f"{n}_F" for n in feature_names]

    df["Phys_T"] = df[T_cols].mean(axis=1)
    df["Phys_I"] = df[I_cols].mean(axis=1)
    df["Phys_F"] = df[F_cols].mean(axis=1)

    # Depression-level aggregates (same as Phys for two-domain case)
    df["Depression_T"] = df["Phys_T"]
    df["Depression_I"] = df["Phys_I"]
    df["Depression_F"] = df["Phys_F"]

    # NDI and smoothed NDI
    df["NDI"]        = compute_ndi(df["Depression_T"].values,
                                    df["Depression_I"].values,
                                    df["Depression_F"].values,
                                    lambda_vec)
    df["NDI_smooth"] = (df["NDI"]
                        .rolling(CONFIG["SMOOTH_K"], min_periods=1).mean())

    return df

# ================================================================
# V3 -- DATA-QUALITY SIGNAL (spec Sec 2/3, order-of-work #2)
# ================================================================
def _flatline_flags(series, run_len=3):
    s = pd.Series(series)
    same_as_prev = s.eq(s.shift(1)) & s.notna() & s.shift(1).notna()
    run = same_as_prev.groupby((~same_as_prev).cumsum()).cumsum()
    return (run >= (run_len - 1)).values

# _out_of_range_flags_causal REMOVED (was: |x - personal_med| > k*MAD).
# Diagnostic (Cell 35) confirmed this is a thresholded function of the
# SAME statistic T is built from, so it mechanically re-coupled I with
# T: Spearman(T,I) fell from -0.469/-0.764 to 0.061/-0.186 (PMData/
# DEPRESJON) once removed. Spec Sec 2's governing rule -- "I must not
# be a function of x's deviation, or it re-couples with T" -- overrides
# its own "e.g. out-of-range flags" illustration when the two conflict,
# which they do here. instability is now flat-line runs only.

def compute_quality_signal(df_raw, feat_names, train_mask):
    """
    missing_rate, instability : (n_days,) each, causal.
    instability = flat-line runs only (see removal note above).
    train_mask kept in the signature for interface stability even
    though it's now unused here -- callers pass it unchanged.
    """
    n = len(df_raw)
    present = np.zeros((n, len(feat_names)))
    instab  = np.zeros((n, len(feat_names)))
    for j, col in enumerate(feat_names):
        if col not in df_raw.columns:
            present[:, j] = np.nan
            instab[:, j]  = np.nan
            continue
        raw = pd.to_numeric(df_raw[col], errors="coerce")
        present[:, j] = raw.isna().values.astype(float)
        instab[:, j]  = _flatline_flags(raw).astype(float)
    missing_rate = np.nanmean(present, axis=1)
    instability  = np.nanmean(instab, axis=1)
    return missing_rate, instability

# ================================================================
# V3 -- THREE-CHANNEL DECOMPOSITION (spec Sec 2, order-of-work #3)
# T: cross-feature consensus level. I: data quality (NOT deviation).
# F: V2' volatility, unchanged, reused per-feature then averaged --
# same rollup convention this module already uses for Depression_F.
# ================================================================
def compute_T_v3(X, train_mask):
    """
    Cross-feature consensus Truth (spec Sec 2). ONE T per day, not
    per feature. Average the per-feature deviations z_j FIRST, THEN
    squash ONCE -- spec is explicit about this order; averaging
    already-squashed per-feature T values is a different (wrong)
    number, since squashing is nonlinear.
    """
    n, d = X.shape
    tr_mask = np.asarray(train_mask, dtype=bool)
    z = np.full((n, d), np.nan)
    for j in range(d):
        x_j = X[:, j]
        tr_vals = x_j[tr_mask]
        if np.sum(~np.isnan(tr_vals)) < 2:
            continue  # insufficient training data for this feature -- excluded, not zero-filled
        med_j = np.nanmedian(tr_vals)
        mad_j = np.nanmedian(np.abs(tr_vals - med_j)) + 1e-8
        z[:, j] = np.abs(x_j - med_j) / mad_j
    z_bar = np.nan_to_num(np.nanmean(z, axis=1), nan=0.0)
    T = np.exp(-(z_bar ** 2) / 2)
    return T

def compute_I_v3(missing_rate, instability, w_miss=None, w_inst=None):
    """
    Data-quality Indeterminacy (spec Sec 2, order-of-work #2). Loud
    raise if quality inputs are absent -- do not silently fall back
    to a deviation-based I.
    """
    if missing_rate is None or instability is None:
        raise ValueError(
            "V3 I requires missing_rate and instability from the "
            "I-availability report (Cell 14.7) -- refusing to fall "
            "back to a deviation-based I silently")
    w_miss = CONFIG["W_MISS"] if w_miss is None else w_miss
    w_inst = CONFIG["W_INST"] if w_inst is None else w_inst
    return np.clip(w_miss * missing_rate + w_inst * instability, 0, 1)

def compute_F_v3(X, feat_names, train_mask, pop_vol_median, pop_vol_mad,
                  two_sided_features=None):
    """F -- V2' volatility, unchanged (spec Sec 2), per-feature then
    averaged to one value per day (same convention as Depression_F)."""
    two_sided_features = two_sided_features if two_sided_features is not None else TWO_SIDED_FEATURES
    n, d = X.shape
    F_per_feature = np.full((n, d), np.nan)
    for j, feat in enumerate(feat_names):
        _, _, F_j = neutrosophic_decompose(
            X[:, j], train_mask=train_mask,
            risk_direction=get_risk_direction(feat),
            two_sided=(feat in two_sided_features),
            f_mode="population_vol",
            pop_vol_median=pop_vol_median[j], pop_vol_mad=pop_vol_mad[j])
        F_per_feature[:, j] = F_j
    return np.nanmean(F_per_feature, axis=1)


# ================================================================
# V3 -- SCORE, BOTH FORMS (spec Sec 4, order-of-work #5 -- defined
# now so downstream cells can call them; DO NOT run the CV
# calibration/comparison until order-of-work #4's independence
# results are in and reported, per spec Sec 7's own sequencing)
# ================================================================
def compute_ndi_v3_primary(T_v3, I_v3, F_v3, lambda_vec=None):
    """4a -- primary. Identical formula to compute_ndi; reused as-is."""
    return compute_ndi(T_v3, I_v3, F_v3, lambda_vec)

def compute_ndi_v3_confidence(T_v3, I_v3, F_v3, lambda_vec=None, baseline_risk=0.0):
    """4b -- conceptual, confidence-discount form. Note lam_I does
    NOT appear here -- spec's 4b formula uses I directly as the
    discount weight, not lam_I*I. At I=1 this collapses to
    baseline_risk; at I=0 it equals (lam_F*F - lam_T*T), which is
    where it coincides with 4a (since lam_I*0=0 there too)."""
    lT = CONFIG["LAMBDA_T"] if lambda_vec is None else lambda_vec[0]
    lF = CONFIG["LAMBDA_F"] if lambda_vec is None else lambda_vec[2]
    tf_part = lF * F_v3 - lT * T_v3
    ndi_b = (1 - I_v3) * tf_part + I_v3 * baseline_risk
    return np.clip(ndi_b, -1, 1)


# ================================================================
# COMPONENT 2 — LEAKAGE-FREE BLOCKED TS-CV  (Brief Algorithm 1)
# T0=14, embargo g=k-1=6, fold h=1
# Embargo prevents 7-day NDI_smooth window from leaking test day
# into training windows.
# ================================================================

def blocked_tscv(X, y, model_factory,
                 T0=None, g=None, h=None,
                 return_folds=True):
    """
    Leakage-free blocked forward-chaining cross-validation.

    Algorithm 1 (Brief Section 2.2):
      for tstart = T0+g+1 to T-h+1 step h:
        Itr = {1, ..., tstart-g-1}
        Ite = {tstart, ..., tstart+h-1}
        fit on Itr, predict on Ite

    Parameters
    ----------
    X             : np.ndarray (T, d)
    y             : np.ndarray (T,)
    model_factory : callable -> fresh sklearn model
    T0            : int  minimum training length  (default CONFIG["T0"])
    g             : int  embargo gap              (default CONFIG["EMBARGO_G"])
    h             : int  fold size                (default CONFIG["FOLD_H"])
    return_folds  : bool save fold indices for reproducibility

    Returns
    -------
    dict:
      rho, p, tau        Spearman correlation, p-value, Kendall tau
      rmse, mae, nrmse   error metrics
      n_eval             number of held-out evaluation points
      preds              predictions (NaN for non-evaluated points)
      ci_lo, ci_hi       95% block-bootstrap CI for Spearman
      folds              list of (train_idx, test_idx) if return_folds
    """
    T0 = T0 if T0 is not None else CONFIG["T0"]
    g  = g  if g  is not None else CONFIG["EMBARGO_G"]
    h  = h  if h  is not None else CONFIG["FOLD_H"]

    T_len = len(y)
    preds = np.full(T_len, np.nan)
    folds = []

    for ts in range(T0 + g, T_len - h + 1, h):
        tr_idx = list(range(0, ts - g))
        te_idx = list(range(ts, min(ts + h, T_len)))
        if len(tr_idx) < 5:
            continue
        sc = StandardScaler().fit(X[tr_idx])
        m  = model_factory()
        m.fit(sc.transform(X[tr_idx]), y[tr_idx])
        preds[te_idx] = m.predict(sc.transform(X[te_idx]))
        if return_folds:
            folds.append((tr_idx, te_idx))

    mask   = ~np.isnan(preds)
    n_eval = int(mask.sum())

    if n_eval < 3:
        res = dict(rho=np.nan, p=np.nan, tau=np.nan,
                   rmse=np.nan, mae=np.nan, nrmse=np.nan,
                   n_eval=n_eval, preds=preds,
                   ci_lo=np.nan, ci_hi=np.nan)
        if return_folds:
            res["folds"] = folds
        return res

    rho, p_val = spearmanr(y[mask], preds[mask])
    tau, _     = kendalltau(y[mask], preds[mask])
    rmse       = float(np.sqrt(mean_squared_error(y[mask], preds[mask])))
    mae        = float(mean_absolute_error(y[mask], preds[mask]))
    y_range    = float(y[mask].max() - y[mask].min())
    nrmse      = rmse / (y_range + 1e-8)
    ci_lo, ci_hi = block_bootstrap_ci(y[mask], preds[mask])

    res = dict(rho=rho, p=p_val, tau=tau,
               rmse=rmse, mae=mae, nrmse=nrmse,
               n_eval=n_eval, preds=preds,
               ci_lo=ci_lo, ci_hi=ci_hi)
    if return_folds:
        res["folds"] = folds
    return res


def block_bootstrap_ci(y_true, y_pred, n_boot=None, rng=None):
    """
    Block-bootstrap 95% CI for Spearman correlation.
    Block size b = ceil(T^(1/3))  per Brief Section 6.
    """
    n_boot = n_boot if n_boot is not None else CONFIG["N_BOOT"]
    n      = len(y_true)
    bs     = max(1, int(np.ceil(n ** (1.0/3.0))))
    if rng is None:
        rng = _MODULE_RNG   # uses module-level rng, seeded once at import
    rhos   = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - bs + 1,
                               size=int(np.ceil(n / bs)))
        idx    = np.concatenate(
                     [np.arange(s, min(s + bs, n)) for s in starts])[:n]
        r, _   = spearmanr(y_true[idx], y_pred[idx])
        rhos.append(r)
    return tuple(np.percentile(rhos, [2.5, 97.5]))


def save_fold_indices(fold_list, filepath):
    """
    Save CV fold indices to CSV for independent reproduction (Brief Sec 6).
    """
    rows = []
    for fold_num, (tr, te) in enumerate(fold_list):
        for idx in tr:
            rows.append({"fold": fold_num, "split": "train", "index": idx})
        for idx in te:
            rows.append({"fold": fold_num, "split": "test",  "index": idx})
    pd.DataFrame(rows).to_csv(filepath, index=False)
    print(f"  Fold indices saved: {filepath}")


# ================================================================
# MODEL FACTORIES  (seed fixed, hyperparams in CONFIG)
# ================================================================

def rf_factory():
    return RandomForestRegressor(**CONFIG["RF_PARAMS"])


def gb_factory():
    return GradientBoostingRegressor(**CONFIG["GB_PARAMS"])


class _RidgeModel:
    def __init__(self):
        self.alpha = CONFIG["RIDGE_ALPHA"]
    def fit(self, X, y):
        self.mu = X.mean(0); self.sg = X.std(0) + 1e-8
        self.m  = Ridge(alpha=self.alpha).fit(
                      (X - self.mu) / self.sg, y)
    def predict(self, X):
        return self.m.predict((X - self.mu) / self.sg)


# ================================================================
# COMPONENT 3 — BASELINES + LIFT  (Brief Sec 3, Table 1)
# B1: AR(p=1), AR(p=2)     Brief Eq. 3, ridge alpha=1
# B2: Raw-feature RF        Brief Eq. 4, no neutrosophication
# B3: Personalised z-score  Brief Eq. 5
# Lift L_b = rho_NDI - rho_baseline  Brief Eq. 6
# ================================================================

def run_baselines(X_raw, y, X_neutro):
    """
    Run B1/B2/B3 and compute neutrosophic lift over each baseline.

    Parameters
    ----------
    X_raw    : np.ndarray (T, d_raw)    raw physiological features only
    y        : np.ndarray (T,)          target (MADRS / PHQ / NDI)
    X_neutro : np.ndarray (T, d_neutro) full neutrosophic feature set

    Returns
    -------
    dict keyed by baseline name.  Each value is a blocked_tscv result
    augmented with "lift" (float) and "lift_ci" (lo, hi).
    Key "NDI_RF" contains the reference NDI result.
    """
    # Lagged arrays for AR baselines
    y_l1       = np.roll(y, 1); y_l1[0]  = y[0]
    y_l2       = np.roll(y, 2); y_l2[:2] = y[0]
    X_ar1      = y_l1.reshape(-1, 1)
    X_ar2      = np.column_stack([y_l1, y_l2])

    baselines  = {
        "B1_AR(p=1)"    : (X_ar1,   _RidgeModel),
        "B1_AR(p=2)"    : (X_ar2,   _RidgeModel),
        "B2_Raw_RF"     : (X_raw,   rf_factory),
        "B3_PersonalisedZ" : (X_raw, _RidgeModel),
    }

    results       = {}
    ndi_res       = blocked_tscv(X_neutro, y, rf_factory)
    results["NDI_RF"] = ndi_res
    ndi_rho       = ndi_res["rho"] if not np.isnan(ndi_res.get("rho", np.nan)) else 0.0

    rng = np.random.default_rng(CONFIG["RANDOM_SEED"])

    for bname, (X_b, mf) in baselines.items():
        res  = blocked_tscv(X_b, y, mf)
        results[bname] = res

        if np.isnan(res.get("rho", np.nan)):
            res["lift"] = np.nan
            res["lift_ci"] = (np.nan, np.nan)
            continue

        lift = ndi_rho - res["rho"]

        # Bootstrap CI of lift (brief Eq. 6 + Section 6)
        mask_b   = ~np.isnan(res["preds"])
        mask_ndi = ~np.isnan(ndi_res["preds"])
        lift_boot = []
        for _ in range(CONFIG["N_BOOT"]):
            nb = int(mask_b.sum())
            if nb < 3:
                break
            ib  = rng.integers(0, nb, size=nb)
            rho_b, _ = spearmanr(y[mask_b][ib],
                                  res["preds"][mask_b][ib])
            nn  = int(mask_ndi.sum())
            inn = rng.integers(0, nn, size=nn)
            rho_n, _ = spearmanr(y[mask_ndi][inn],
                                  ndi_res["preds"][mask_ndi][inn])
            lift_boot.append(rho_n - rho_b)

        lci = (tuple(np.percentile(lift_boot, [2.5, 97.5]))
               if lift_boot else (np.nan, np.nan))
        res["lift"]    = lift
        res["lift_ci"] = lci

    return results


def print_baseline_table(results):
    """Print baseline comparison table per Brief Table 1."""
    order = ["B1_AR(p=1)", "B1_AR(p=2)",
             "B2_Raw_RF", "B3_PersonalisedZ", "NDI_RF"]
    print(f"\n  {'Model':<25} {'Spearman(CV)':>14} {'p':>7}"
          f" {'RMSE':>8} {'MAE':>8} {'Lift L_b':>10}  95%CI(lift)")
    print("  " + "-" * 82)
    for bname in order:
        res = results.get(bname)
        if res is None or np.isnan(res.get("rho", np.nan)):
            print(f"  {bname:<25} {'N/A':>14}")
            continue
        if bname == "NDI_RF":
            lift_str = "reference"
            ci_str   = ""
        else:
            lift_str = f"{res['lift']:+.4f}"
            lci      = res.get("lift_ci", (np.nan, np.nan))
            ci_str   = f"[{lci[0]:+.3f},{lci[1]:+.3f}]"
        print(f"  {bname:<25} {res['rho']:>14.4f} {res['p']:>7.3f}"
              f" {res['rmse']:>8.4f} {res['mae']:>8.4f}"
              f" {lift_str:>10}  {ci_str}")


# ================================================================
# COMPONENT 4 — LAMBDA SIMPLEX GRID SEARCH  (Brief Sec 4.1)
# 231 points on Delta_2, step s=0.05.
# S(lambda) = blocked-CV Spearman  (Brief Eq. 7).
# Stability radius r_eta  (Brief Eq. 9, eta=0.05).
# ================================================================

def lambda_grid_search(T_dep, I_dep, F_dep, y,
                       X_extra=None,
                       step=None, eta=None):
    """
    Grid search over the neutrosophic weight simplex Delta_2.

    Parameters
    ----------
    T_dep, I_dep, F_dep : np.ndarray (T,)  domain-level T/I/F components
    y                   : np.ndarray (T,)  target
    X_extra             : np.ndarray (T, d) optional — augment NDI
                          candidate with additional features for CV
    step : float  grid step  (default CONFIG["LAMBDA_STEP"] = 0.05)
    eta  : float  stability-radius tolerance (Brief Eq. 9)

    Returns
    -------
    dict:
      best_lambda   (lT, lI, lF) that maximises S(lambda)
      best_rho      S(lambda*)
      original_rho  S(0.2, 0.3, 0.5)  the originally asserted weights
      lambdas       np.ndarray (N, 3)  all grid points
      grid_rhos     np.ndarray (N,)    all S(lambda) values
      n_stable      grid points within stability radius
      verdict       "Robust" or "Per-subject tuning recommended"
      top5          top-5 (rho, lambda) pairs
    """
    step = step if step is not None else CONFIG["LAMBDA_STEP"]
    eta  = eta  if eta  is not None else CONFIG["LAMBDA_ETA"]

    ndi_components = np.column_stack([F_dep, I_dep, T_dep])

    lambdas   = []
    grid_rhos = []

    for lt in np.arange(0, 1 + step, step):
        for li in np.arange(0, 1 - lt + step, step):
            lf = round(1.0 - lt - li, 8)
            if lf < -1e-9:
                continue
            lt_ = round(lt, 2)
            li_ = round(li, 2)
            lf_ = round(max(lf, 0.0), 2)
            lambdas.append((lt_, li_, lf_))

            ndi_cand = (lf_*ndi_components[:,0]
                        + li_*ndi_components[:,1]
                        - lt_*ndi_components[:,2])

            if X_extra is not None:
                X_cv = np.column_stack([ndi_cand.reshape(-1,1), X_extra])
            else:
                X_cv = ndi_cand.reshape(-1, 1)

            res = blocked_tscv(X_cv, y, rf_factory, return_folds=False)
            r   = res["rho"] if not np.isnan(res.get("rho", np.nan)) else -999.0
            grid_rhos.append(r)

    lambdas   = np.array(lambdas)
    grid_rhos = np.array(grid_rhos)

    best_idx    = int(np.argmax(grid_rhos))
    best_lambda = tuple(lambdas[best_idx])
    best_rho    = float(grid_rhos[best_idx])

    # S(0.2, 0.3, 0.5) — originally asserted weights  (Brief Sec 4.1)
    orig_mask   = ((np.abs(lambdas[:,0] - 0.2) < 0.01) &
                   (np.abs(lambdas[:,1] - 0.3) < 0.01))
    original_rho = float(grid_rhos[orig_mask][0]) if orig_mask.any() else np.nan

    # Stability radius r_eta  (Brief Eq. 9)
    opt_arr  = np.array(best_lambda)
    n_stable = int(sum(
        1 for (lam, r) in zip(lambdas, grid_rhos)
        if (np.linalg.norm(lam - opt_arr, 1) <= 0.20
            and r > -900
            and abs(best_rho - r) <= eta)
    ))
    verdict = ("Robust — weights stable across neighbourhood"
               if n_stable > 5
               else "Per-subject tuning recommended (small stability radius)")

    valid = [(float(r), tuple(l))
             for r, l in zip(grid_rhos, lambdas) if r > -900]
    top5  = sorted(valid, reverse=True)[:5]

    return dict(
        best_lambda  = best_lambda,
        best_rho     = best_rho,
        original_rho = original_rho,
        lambdas      = lambdas,
        grid_rhos    = grid_rhos,
        n_stable     = n_stable,
        verdict      = verdict,
        top5         = top5,
        n_grid       = len(lambdas),
    )


# ================================================================
# COMPONENT 5 — SHAP ATTRIBUTION  (Brief Sec 2, item 5)
# Feature-level and domain-level SHAP importance.
# Cross-checked against permutation importance.
# ================================================================

def shap_attribution(X, y, feature_names,
                     domain_map=None,
                     show_plots=True):
    """
    SHAP feature importance with permutation-importance cross-check.

    Parameters
    ----------
    X            : np.ndarray (T, d)
    y            : np.ndarray (T,)
    feature_names: list[str] length d
    domain_map   : dict {domain: [feature_names]} for aggregation
    show_plots   : bool (set False in simulation arm to suppress output)

    Returns
    -------
    dict:
      shap_values     np.ndarray (T, d)
      mean_abs_shap   np.ndarray (d,)  mean |SHAP| per feature
      perm_df         pd.DataFrame     permutation importance table
      domain_shap     dict {domain: total |SHAP|}
      shap_rank       list[str]        features sorted by |SHAP| desc
    """
    try:
        import shap as shap_lib
    except ImportError:
        raise ImportError("Run: pip install shap")

    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)

    model = rf_factory()
    model.fit(Xs, y)

    explainer = shap_lib.TreeExplainer(model)
    sv        = explainer.shap_values(Xs)
    mean_abs  = np.abs(sv).mean(axis=0)

    if show_plots:
        shap_lib.summary_plot(sv, Xs, feature_names=feature_names,
                               plot_type="bar", show=True)
        shap_lib.summary_plot(sv, Xs, feature_names=feature_names,
                               show=True)

    perm = sk_perm_imp(model, Xs, y,
                        n_repeats=CONFIG["PERM_REPEATS"],
                        random_state=CONFIG["RANDOM_SEED"])
    perm_df = pd.DataFrame({
        "feature"         : feature_names,
        "importance_mean" : perm.importances_mean,
        "importance_std"  : perm.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    domain_shap = {}
    if domain_map:
        for dom, cols in domain_map.items():
            idxs = [feature_names.index(c)
                    for c in cols if c in feature_names]
            domain_shap[dom] = float(mean_abs[idxs].sum()) if idxs else 0.0

    shap_rank = [feature_names[i] for i in np.argsort(-mean_abs)]

    return dict(
        shap_values   = sv,
        mean_abs_shap = mean_abs,
        perm_df       = perm_df,
        domain_shap   = domain_shap,
        shap_rank     = shap_rank,
    )


# ================================================================
# UTILITIES — AUROC + YOUDEN J  (Brief Sec 4.3)
# ================================================================

def auroc_youden(ndi_scores, y_binary, n_boot=None):
    """
    AUROC with bootstrap CI and Youden J optimal threshold.

    Returns
    -------
    dict: auroc, ci (lo,hi), tau_star, J_star, fpr, tpr, thresholds
    """
    n_boot = n_boot if n_boot is not None else CONFIG["N_BOOT"]
    T_len  = len(ndi_scores)

    if len(np.unique(y_binary)) < 2:
        return dict(auroc=np.nan, ci=(np.nan,np.nan),
                    tau_star=np.nan, J_star=np.nan,
                    fpr=None, tpr=None, thresholds=None)

    auroc                  = roc_auc_score(y_binary, ndi_scores)
    fpr_a, tpr_a, thr_a   = roc_curve(y_binary, ndi_scores)
    J_arr                  = tpr_a - fpr_a
    best_j                 = int(np.argmax(J_arr))
    tau_star               = float(thr_a[best_j])
    J_star                 = float(J_arr[best_j])

    rng = np.random.default_rng(CONFIG["RANDOM_SEED"])
    ab  = []
    for _ in range(n_boot):
        idx = rng.integers(0, T_len, size=T_len)
        if len(np.unique(y_binary[idx])) < 2:
            continue
        try:
            ab.append(roc_auc_score(y_binary[idx], ndi_scores[idx]))
        except Exception:
            pass
    ci = tuple(np.percentile(ab, [2.5, 97.5])) if ab else (np.nan, np.nan)

    return dict(auroc=auroc, ci=ci, tau_star=tau_star, J_star=J_star,
                fpr=fpr_a, tpr=tpr_a, thresholds=thr_a)


# ================================================================
# UTILITIES — PERMUTATION CALIBRATION TEST  (Brief Sec 4.4)
# ================================================================

def permutation_calibration_test(uncertainty, pred_errors, B=10000):
    """
    Permutation test for uncertainty calibration.

    FIX (this revision): the previous one-sided test,
    p = (1/B) * sum_b 1{rho^(b) >= rho_obs}, is only valid when
    rho_obs is known to be positive in advance. For a NEGATIVE
    observed rho, that comparison gives a p-value that contradicts
    the test's own logic -- most null-distribution draws (centred
    near 0 by construction, since permutation destroys any real
    relationship) exceed a negative threshold just by chance, so the
    one-sided test should return p near 1 (not significant) for a
    negative rho, not p near 0. Now two-sided: tests whether |rho| is
    larger than expected under the null, which is the correct framing
    for "does uncertainty track error" regardless of the direction of
    that relationship.

    Parameters
    ----------
    uncertainty : np.ndarray  Depression_I values
    pred_errors : np.ndarray  |y_pred - y_true|
    B           : int         permutations (Brief specifies 10,000)

    Returns
    -------
    dict: obs_rho, pval, significant (bool at alpha=0.05)
    """
    obs_rho, _ = spearmanr(uncertainty, pred_errors)
    rng        = np.random.default_rng(CONFIG["RANDOM_SEED"])
    null_rhos  = np.array([
        spearmanr(rng.permutation(uncertainty), pred_errors)[0]
        for _ in range(B)
    ])
    pval = float(np.mean(np.abs(null_rhos) >= np.abs(obs_rho)))
    return dict(obs_rho=float(obs_rho),
                pval=pval,
                significant=(pval < 0.05))


# ================================================================
# UTILITIES — PRETTY PRINTING
# ================================================================

def print_cv_result(label, target_name, res):
    """Print one blocked-CV result row with CI."""
    if np.isnan(res.get("rho", np.nan)):
        print(f"  {label:<20} {target_name:<12} N/A (n_eval={res.get('n_eval',0)})")
        return
    print(f"  {label:<20} {target_name:<12} "
          f"rho={res['rho']:+.4f}(p={res['p']:.3f}) "
          f"RMSE={res['rmse']:.4f} NRMSE={res['nrmse']:.4f} "
          f"tau={res['tau']:+.4f} n={res['n_eval']} "
          f"95%CI=[{res['ci_lo']:.3f},{res['ci_hi']:.3f}]")


def print_lambda_result(res):
    """Print lambda grid search result."""
    print(f"  Grid size       : {res['n_grid']} points on Delta_2")
    print(f"  Original (0.2,0.3,0.5) S = {res['original_rho']:.4f}")
    print(f"  Optimal lambda* {res['best_lambda']}  S* = {res['best_rho']:.4f}")
    print(f"  Stability radius: {res['n_stable']} points within eta=0.05")
    print(f"  Verdict: {res['verdict']}")
    print(f"  Top-5 configurations:")
    print(f"    {'lT':>6} {'lI':>6} {'lF':>6} {'S':>8}")
    for r, (lt,li,lf) in res["top5"]:
        tag = "  <- original" if (lt==0.2 and li==0.3 and lf==0.5) else ""
        print(f"    {lt:>6.2f} {li:>6.2f} {lf:>6.2f} {r:>8.4f}{tag}")


# ================================================================
# FEATURE ENGINEERING FOR DEPRESJON  (Arm 1)
# Extract interpretable daily features from 1-minute actigraphy.
# ================================================================

DEPRESJON_FEATURE_NAMES = [
    "act_mean", "act_std", "act_median", "act_iqr",
    "act_max", "act_zero_rate", "act_total",
    "act_night_mean", "act_day_mean", "day_night_ratio",
    "L5", "M10", "RA",
    "act_trend", "act_entropy",
]

DEPRESJON_DOMAIN_MAP = {
    "physiology" : ["act_mean","act_std","act_total","act_max",
                    "act_iqr","act_trend","act_entropy"],
    "sleep_proxy": ["act_night_mean","day_night_ratio","L5"],
    "circadian"  : ["M10","RA","act_zero_rate","act_day_mean","act_median"],
}


def extract_daily_features(minute_df):
    """
    Extract daily actigraphy features from DEPRESJON 1-minute data.

    Parameters
    ----------
    minute_df : pd.DataFrame  columns: timestamp, date, activity

    Returns
    -------
    pd.DataFrame  one row per date,  columns = DEPRESJON_FEATURE_NAMES
    """
    df = minute_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"]      = df["timestamp"].dt.hour
    df["date"]      = pd.to_datetime(df["date"])

    rows = []
    for date, grp in df.groupby("date"):
        act = grp["activity"].values.astype(float)
        if len(act) < 60:
            continue

        # Basic statistics
        act_mean   = float(act.mean())
        act_std    = float(act.std())
        act_median = float(np.median(act))
        act_iqr    = float(np.percentile(act,75) - np.percentile(act,25))
        act_max    = float(act.max())
        act_zero   = float((act == 0).mean())
        act_total  = float(act.sum())

        # Day/night split
        night_mask = (grp["hour"].values >= 22) | (grp["hour"].values < 6)
        day_mask   = ~night_mask
        act_night  = float(act[night_mask].mean()) if night_mask.any() else np.nan
        act_day    = float(act[day_mask].mean())   if day_mask.any()   else np.nan
        day_night  = (float(act_day / (act_night + 1e-8))
                      if (not np.isnan(act_night) and act_night > 0)
                      else np.nan)

        # Circadian markers: L5, M10, RA
        hourly = (grp.groupby("hour")["activity"]
                  .mean()
                  .reindex(range(24), fill_value=0)
                  .values.astype(float))
        if len(hourly) >= 10:
            l5_list  = [hourly[i:i+5].mean()  for i in range(20)]
            m10_list = [hourly[i:i+10].mean() for i in range(15)]
            L5  = float(min(l5_list))
            M10 = float(max(m10_list))
            RA  = float((M10 - L5) / (M10 + L5 + 1e-8))
        else:
            L5 = M10 = RA = np.nan

        # Linear trend
        t         = np.arange(len(act), dtype=float)
        act_trend = float(np.polyfit(t, act, 1)[0]) if len(act) > 1 else 0.0

        # Shannon entropy
        bins  = np.histogram(act, bins=10,
                              range=(0, max(act.max(), 1) + 1))[0]
        probs = bins / (bins.sum() + 1e-8)
        act_e = float(-np.sum(probs * np.log(probs + 1e-8)))

        rows.append(dict(
            date=date,
            act_mean=act_mean, act_std=act_std, act_median=act_median,
            act_iqr=act_iqr, act_max=act_max,
            act_zero_rate=act_zero, act_total=act_total,
            act_night_mean=act_night, act_day_mean=act_day,
            day_night_ratio=day_night,
            L5=L5, M10=M10, RA=RA,
            act_trend=act_trend, act_entropy=act_e,
        ))

    return pd.DataFrame(rows)


# ================================================================
# SELF-TEST
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ndi_pipeline.py — self-test")
    print("=" * 60)
    print(f"RANDOM_SEED = {CONFIG['RANDOM_SEED']}")
    print(f"T0={CONFIG['T0']}, g={CONFIG['EMBARGO_G']}, "
          f"lambda_step={CONFIG['LAMBDA_STEP']}")
    print()

    # Generate tiny synthetic dataset to smoke-test all components
    rng = np.random.default_rng(42)
    T   = 30
    d   = 5
    X_raw    = rng.standard_normal((T, d))
    X_neutro = rng.standard_normal((T, d * 3 + 2))
    y        = np.sin(np.linspace(0, 3, T)) + 0.1 * rng.standard_normal(T)
    names    = [f"feat_{i}" for i in range(d)]

    print("Component 1 — neutrosophic_decompose:")
    smoke_train_mask = np.ones(T, dtype=bool)  # ground-truth-style: use all days
    T_j, I_j, F_j = neutrosophic_decompose(X_raw[:, 0], train_mask=smoke_train_mask)
    assert T_j.shape == (T,) and I_j.shape == (T,) and F_j.shape == (T,)
    assert T_j.min() >= 0 and T_j.max() <= 1
    assert F_j.min() >= 0 and F_j.max() <= 1
    print(f"  T in [0,1]: OK  |  F in [0,1]: OK  |  "
          f"corr(T,F) = {np.corrcoef(T_j,F_j)[0,1]:.3f} (should not be -1)")

    print("Component 2 — blocked_tscv:")
    res = blocked_tscv(X_neutro, y, rf_factory, T0=10)
    print(f"  n_eval={res['n_eval']}  rho={res['rho']:.4f}  "
          f"CI=[{res['ci_lo']:.3f},{res['ci_hi']:.3f}]")

    print("Component 3 — run_baselines:")
    bres = run_baselines(X_raw, y, X_neutro)
    print(f"  Baselines computed: {list(bres.keys())}")

    print("Component 4 — lambda_grid_search:")
    T_dep = T_j; I_dep = I_j; F_dep = F_j
    gres = lambda_grid_search(T_dep, I_dep, F_dep, y)
    print(f"  Grid size={gres['n_grid']}  "
          f"best_lambda={gres['best_lambda']}  "
          f"best_rho={gres['best_rho']:.4f}")

    print("Component 5 — shap_attribution:")
    sres = shap_attribution(X_neutro, y,
                             [f"f{i}" for i in range(X_neutro.shape[1])],
                             show_plots=False)
    print(f"  SHAP computed. Top feature: {sres['shap_rank'][0]}")

    print()
    print("All 5 components: PASS")
    print("=" * 60)