#!/usr/bin/env python3
r"""
NDI Task 4 -- positive-control simulation
(Task 3 review and Task 4 execution memo, Sections 3-5; revision analyses memo, Task 4)

The evaluation protocol of the paper is not touched. CELL 14.4, V12-1, V12-2 and Task 3's next-day target
wrapper (T3-1) are executed from the Task 3 notebook committed under tag v1.4-task3; their source hashes are
recorded in config_task4.json and checked on every machine. One function is substituted: get_folds, by a
generator that returns exactly the folds of ndi.blocked_tscv without fitting a forest per fold (`check`
asserts that). Arms: V3 and raw-feature RF. Outcome: next-day value through Task 3's next_day wrapper (target
support {s + 1}); the purge removes nothing, asserted for every replicate. Generator: Arm 2 (CELL C2.0 / C2.1
of the v1 notebook, verbatim, SNR 1.0). Cohort: 16 participants with the actual v1.4 lengths.

COMMANDS (run inside the repository folder; Windows: `python`, Linux/macOS: `python3`)
  check      fold check and cache-isolation check, with timings (= re-benchmark)        before tagging
  calibrate  beta per signal level on one pilot cohort (no model fitting)               before tagging
  bench      re-benchmark: one cohort per worker, all workers at once
  run        replicates (tag v1.4-task4 only); one JSON per replicate; existing files are skipped
               --pilot                      5 replicates each of C0, C1b3, C2b3 -> <out>/pilot
               --cells all | C1b1,C0,...    --reps all | 0-49 | 0,5,9    --workers N    --hours H
  verify     re-run two replicates on this machine and compare with the files (determinism)
  merge      progress; when every cell is complete: per-cell results, MDE, Figure 5, hand-in check,
             deviation note -> <out>/analysis
Common options: --repo <repository folder> (default: this file's folder)
                --ndi <ndi_pipeline.py>   (default: <repo>/ndi_pipeline.py)   --out <results folder>

Laptop / HPC (inside the repository, checked out at the tag; same package versions as config_task4.json):
  git checkout v1.4-task4
  python task4_run.py run --ndi ndi_pipeline.py --out results_task4 --cells all --workers 8
  python task4_run.py merge --ndi ndi_pipeline.py --out results_task4
Several machines or sessions may run different shards (--cells / --reps) at the same time: copy every
<out>/cells/<cell>/rep_XXX.json into one results folder and run merge there.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")            # one thread per process; parallelism is across replicates only
import sys, io, re, ast, json, time, math, glob, hashlib, argparse, platform, datetime, warnings, contextlib
import subprocess, importlib.util
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, rankdata, norm
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")

TAG, TASK3_TAG, V14_TAG = "v1.4-task4", "v1.4-task3", "v1.4"
CONFIG_NAME, SCRIPT_NAME = "config_task4.json", "task4_run.py"
SIM_FEATS = ["sim_feature"]                 # the Arm 2 generator's single feature
SIM_TARGET = "sim_next_day"                 # routed by Task 3's next_day wrapper like "mood" / "stress"
T4_ARMS = ("v3", "rawrf")
SNR = 1.0
F_SCALE_FROZEN = 1.0
CHECK_REPS = (900000, 900001)               # fold / cache-isolation check cohorts: never a replicate
CHECK_SPEC = ("CHK", "C1", 1.0)             # check cohorts use C1 with beta = 1.0 (checks only, not a cell)
CALIBRATION_REP = 999999                    # beta calibration cohort: never a replicate
THRESHOLDS = (0.05, 0.0125)
MAIN_CELLS = ("C1b1", "C1b2", "C1b3", "C0", "C2b1", "C2b2", "C2b3")   # priority order (memo, Section 4 step 7)
EXTENSION_CELL = "C1b4"
LEVELS = {"C1b1": ("C1", 1, 0.15), "C1b2": ("C1", 2, 0.30), "C1b3": ("C1", 3, 0.45),
          "C2b1": ("C2", 1, 0.15), "C2b2": ("C2", 2, 0.30), "C2b3": ("C2", 3, 0.45),
          "C1b4": ("C1", 4, 0.60)}
PILOT_CELLS, PILOT_REPS = ("C0", "C1b3", "C2b3"), tuple(range(5))
REAL_UPPER_BOUNDS = {"raw-feature target, V3 vs raw-RF": 0.132, "next-day mood, V3 vs raw-RF": 0.076}

# ------------------------------------------------------------------------------------------------------------
# Arm 2 generator: CELL C2.0 and CELL C2.1 of the v1 notebook, verbatim (sha256 recorded in config_task4.json)
# ------------------------------------------------------------------------------------------------------------
GEN_C20 = r'''# ================================================================
# CELL C2.0  ARM 2 PRE-REGISTRATION -- locked before any generation
# or recovery code runs. Per memo: "do NOT re-tune SNR after seeing
# results -- the earlier tuned-SNR problem must not recur."
# ================================================================
ARM2_SEED = 20260817

ARM2_N_PARTICIPANTS = 15
ARM2_N_DAYS         = 140

ARM2_MEAN_DURATION_WELL    = 60
ARM2_MEAN_DURATION_EPISODE = 14
ARM2_P_STAY_WELL    = 1 - 1 / ARM2_MEAN_DURATION_WELL
ARM2_P_STAY_EPISODE = 1 - 1 / ARM2_MEAN_DURATION_EPISODE

ARM2_SIGMA_A = 1.0    # level-intercept SD across participants (T's target)
ARM2_SIGMA_B = 0.6    # log-volatility-intercept SD across participants (F's target)

ARM2_MU_WELL = 0.0
ARM2_BASE_SD = 1.0

ARM2_C_MIN, ARM2_C_MAX = 0.3, 1.0
ARM2_FLATLINE_RUN_LEN  = 3   # matches ndi_pipeline's own flatline detector

ARM2_SNR_LEVELS = [0.5, 1.0, 2.0]

print("ARM 2 PRE-REGISTRATION LOCKED")
print(f"  N_PARTICIPANTS={ARM2_N_PARTICIPANTS}  N_DAYS={ARM2_N_DAYS}  seed={ARM2_SEED}")
print(f"  Markov: P(stay well)={ARM2_P_STAY_WELL:.4f}  P(stay episode)={ARM2_P_STAY_EPISODE:.4f}")
print(f"  sigma_a={ARM2_SIGMA_A}  sigma_b={ARM2_SIGMA_B}  base_sd={ARM2_BASE_SD}")
print(f"  SNR levels (fixed, will not be re-tuned): {ARM2_SNR_LEVELS}")'''

GEN_C21 = r'''# ================================================================
# CELL C2.1  ARM 2 GENERATOR -- two-state Markov (episode persistence)
# + participant random intercepts + two emission densities +
# independent missingness/instability injection.
# ================================================================
def _simulate_markov_states(n_days, p_stay_well, p_stay_episode, rng):
    states = np.zeros(n_days, dtype=int)
    states[0] = 0 if rng.random() < 0.8 else 1
    for t in range(1, n_days):
        if states[t-1] == 0:
            states[t] = 0 if rng.random() < p_stay_well else 1
        else:
            states[t] = 1 if rng.random() < p_stay_episode else 0
    return states

def generate_arm2_participant(pid, n_days, snr, rng):
    a_i = rng.normal(0, ARM2_SIGMA_A)
    b_i = rng.normal(0, ARM2_SIGMA_B * snr)
    c_i = rng.uniform(ARM2_C_MIN, ARM2_C_MAX)

    states = _simulate_markov_states(n_days, ARM2_P_STAY_WELL, ARM2_P_STAY_EPISODE, rng)

    mean_gap = 2.0 * snr
    state_mean = np.where(states == 0, ARM2_MU_WELL, ARM2_MU_WELL + mean_gap)
    emission_sd = ARM2_BASE_SD * np.exp(b_i)

    x_true = state_mean + a_i + rng.normal(0, emission_sd, size=n_days)
    x_obs = x_true.copy()

    true_level_dev = np.abs(x_true - a_i) / emission_sd

    miss_rate = (1 - c_i) * 0.4 * snr
    missing_mask = rng.random(n_days) < miss_rate
    x_obs[missing_mask] = np.nan

    instab_true = np.zeros(n_days, dtype=bool)
    flatline_prob = (1 - c_i) * 0.02 * snr
    t = 0
    while t < n_days - ARM2_FLATLINE_RUN_LEN:
        if rng.random() < flatline_prob:
            frozen_val = x_obs[t] if not np.isnan(x_obs[t]) else 0.0
            run_len = ARM2_FLATLINE_RUN_LEN + rng.integers(0, 4)
            x_obs[t:t+run_len] = frozen_val
            instab_true[t:t+run_len] = True
            t += run_len
        else:
            t += 1

    df = pd.DataFrame({"sim_feature": x_obs})
    return dict(pid=pid, days_df=df, group="simulated",
                true_state=states, true_a=a_i, true_b=b_i, true_c=c_i,
                true_level_dev=true_level_dev, true_missing=missing_mask,
                true_instability=instab_true)

def build_arm2_cohort(snr, n_participants=None, n_days=None, seed=None):
    n_participants = n_participants or ARM2_N_PARTICIPANTS
    n_days = n_days or ARM2_N_DAYS
    seed = seed if seed is not None else ARM2_SEED
    rng = np.random.default_rng(seed)
    return [generate_arm2_participant(f"sim_{i:02d}", n_days, snr, rng)
            for i in range(n_participants)]'''

NDI = NS = GEN = None          # ndi_pipeline module; namespace with the Task 3 code; generator namespace
STATE = {}


def _norm_text(s):
    return s.replace("\r\n", "\n")


def sha256_text(s):
    return hashlib.sha256(_norm_text(s).encode("utf-8")).hexdigest()


def sha256_file(path):
    with open(path, "rb") as f:
        return sha256_text(f.read().decode("utf-8"))


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=True).encode()).hexdigest()


def script_path():
    return os.path.abspath(__file__)


def git(repo, *args):
    try:
        return subprocess.check_output(["git", "-C", repo, *args], stderr=subprocess.DEVNULL).decode("utf-8")
    except Exception:
        return None


def versions():
    import scipy, sklearn
    return dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, pandas=pd.__version__)


def machine_info():
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    try:
        import psutil
        phys = psutil.cpu_count(logical=False)
    except Exception:
        phys = None
    return dict(host=platform.node(), platform=platform.platform(), cpu=platform.processor(),
                python=platform.python_version(), logical_cores_available=cores, physical_cores=phys)


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------------------------------
# The Task 3 code (CELL 14.4, V12-1, V12-2, T3-1 wrapper) from the notebook under tag v1.4-task3
# ------------------------------------------------------------------------------------------------------------
def _cell_key(src):
    lines = src.splitlines()
    if not lines:
        return None
    first = lines[1] if lines[0].startswith("# ====") and len(lines) > 1 else lines[0]
    m = re.match(r"# ((?:V12|T3)-[0-9A-Z]+)\b", first) or re.match(r"# (CELL [\d.]+)\b", first)
    return m.group(1) if m else None


def read_task3_notebook(repo):
    """The Task 3 notebook as committed under tag v1.4-task3 (found by its T3-1 cell)."""
    names = git(repo, "ls-tree", "-r", "--name-only", TASK3_TAG)
    if names is None:
        raise RuntimeError(f"git cannot read tag {TASK3_TAG} in {repo}: run inside the repository clone")
    for name in [n for n in names.splitlines() if n.endswith(".ipynb")]:
        txt = git(repo, "show", f"{TASK3_TAG}:{name}")
        if txt and '"# T3-1 ' in txt and '"# V12-2 ' in txt:
            return txt, name
    raise RuntimeError(f"no notebook with cells V12-2 and T3-1 under tag {TASK3_TAG}")


def extract_task3_code(nb_text):
    nb = json.loads(nb_text)
    cells = {}
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        k = _cell_key(src)
        if k in ("CELL 14.4", "V12-1", "V12-2", "T3-1"):
            assert k not in cells, f"two cells named {k} in the Task 3 notebook"
            cells[k] = src
    missing = {"CELL 14.4", "V12-1", "V12-2", "T3-1"} - set(cells)
    assert not missing, f"cells missing from the Task 3 notebook: {missing}"
    t31 = cells["T3-1"]
    parts = []
    for st in ast.parse(t31).body:            # the wrapper only: TASK3_TARGETS, the wrap, the two functions
        seg = ast.get_source_segment(t31, st)
        if isinstance(st, ast.Assign) and any(getattr(t, "id", None) == "TASK3_TARGETS" for t in st.targets):
            parts.append(seg)
        elif isinstance(st, ast.If) and "_build_target_v14" in ast.get_source_segment(t31, st.test):
            parts.append(seg)
        elif isinstance(st, ast.FunctionDef) and st.name in ("target_support", "build_target"):
            parts.append(seg)
    assert len(parts) == 4, "T3-1 wrapper not found as expected (TASK3_TARGETS, wrap, target_support, build_target)"
    code = {"CELL 14.4": cells["CELL 14.4"], "V12-1": cells["V12-1"], "V12-2": cells["V12-2"],
            "T3-1 wrapper": "\n\n".join(parts)}
    return code, {k: sha256_text(v) for k, v in code.items()}


def _make_fast_get_folds(ns):
    def get_folds(pr, g):
        """Same folds as V12-2 get_folds (ndi.blocked_tscv, return_folds=True), without a forest per fold."""
        n, T0, h = pr["n"], pr["t0"], int(ns["ndi"].CONFIG["FOLD_H"])
        folds = []
        for ts in range(T0 + g, n - h + 1, h):
            tr_idx = list(range(0, ts - g))
            te_idx = list(range(ts, min(ts + h, n)))
            if len(tr_idx) < 5:
                continue
            folds.append((np.array(tr_idx, int), np.array(te_idx, int)))
        return folds
    return get_folds


def build_namespace(ndi_mod, v14_cfg, code):
    ns = dict(__name__="task3_code_v1_4_task3", np=np, pd=pd, warnings=warnings, time=time, spearmanr=spearmanr,
              Ridge=Ridge, ndi=ndi_mod, V1_2_CONFIG=v14_cfg, K=v14_cfg["SMOOTH_K"], G=v14_cfg["EMBARGO_G"],
              MF=v14_cfg["MIN_FOLDS"], ARMS=["vsigned", "v3", "rawrf", "persz"])
    for k in ("CELL 14.4", "V12-1", "V12-2", "T3-1 wrapper"):
        exec(compile(code[k], f"<{k}, tag {TASK3_TAG}>", "exec"), ns)
    ns["TASK3_Y"] = {}
    ns["TASK3_TARGETS"] = tuple(ns["TASK3_TARGETS"]) + (SIM_TARGET,)
    ns["_get_folds_v12"] = ns["get_folds"]                     # kept for the fold check
    ns["get_folds"] = _make_fast_get_folds(ns)
    return ns


def build_generator():
    g = dict(__name__="arm2_generator", np=np, pd=pd)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(GEN_C20, "<CELL C2.0>", "exec"), g)
        exec(compile(GEN_C21, "<CELL C2.1>", "exec"), g)
    return g


def load_ndi(path):
    spec = importlib.util.spec_from_file_location("ndi_pipeline", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.CONFIG["F_SCALE"] = F_SCALE_FROZEN           # as frozen by CELL 6.5 (v1.4: 1.0)
    return m


def setup(ndi_path=None, ndi_module=None, repo=None, task3_nb_text=None, v14_cfg=None, lengths=None, seed=None):
    """Load ndi_pipeline, the Task 3 code (from tag v1.4-task3) and the generator into this process."""
    global NDI, NS, GEN
    repo = os.path.abspath(repo or os.path.dirname(script_path()))
    NDI = ndi_module if ndi_module is not None else load_ndi(ndi_path)
    assert float(NDI.CONFIG["F_SCALE"]) == F_SCALE_FROZEN, "F_SCALE is not the frozen 1.0 -- STOP"
    assert NDI.CONFIG["RF_PARAMS"].get("n_jobs") in (None, 1), "forests must run with n_jobs = 1"
    nb_name = STATE.get("task3_notebook")
    if task3_nb_text is None:
        task3_nb_text, nb_name = read_task3_notebook(repo)
    if v14_cfg is None:
        t = git(repo, "show", f"{V14_TAG}:config_v1_4.json")
        assert t is not None, f"git cannot read {V14_TAG}:config_v1_4.json in {repo}"
        v14_cfg = json.loads(t)
    code, hashes = extract_task3_code(task3_nb_text)
    NS = build_namespace(NDI, v14_cfg, code)
    GEN = build_generator()
    STATE.update(repo=repo, ndi_path=os.path.abspath(getattr(NDI, "__file__", ndi_path or "")),
                 task3_nb_text=task3_nb_text, task3_notebook=nb_name, v14_cfg=v14_cfg, code_hashes=hashes,
                 lengths=None if lengths is None else [int(x) for x in lengths],
                 seed=int(GEN["ARM2_SEED"]) if seed is None else int(seed))
    STATE["ndi_sha256"] = sha256_file(STATE["ndi_path"])
    return dict(code_hashes=hashes, ndi_sha256=STATE["ndi_sha256"], generator_sha256=generator_hashes(),
                task3_notebook=nb_name, seed=STATE["seed"])


def generator_hashes():
    return {"CELL C2.0": sha256_text(GEN_C20), "CELL C2.1": sha256_text(GEN_C21)}


def worker_payload():
    return dict(ndi_path=STATE["ndi_path"], repo=STATE["repo"], task3_nb_text=STATE["task3_nb_text"],
                v14_cfg=STATE["v14_cfg"], lengths=STATE["lengths"], seed=STATE["seed"])


def _worker_init(payload):
    warnings.filterwarnings("ignore")
    setup(ndi_path=payload["ndi_path"], repo=payload["repo"], task3_nb_text=payload["task3_nb_text"],
          v14_cfg=payload["v14_cfg"], lengths=payload["lengths"], seed=payload["seed"])


# ------------------------------------------------------------------------------------------------------------
# Cohort, signals, outcome
# ------------------------------------------------------------------------------------------------------------
def make_cohort(cell, r):
    """16 participants with the v1.4 lengths. The data and the noise depend only on the replicate r (common random
    numbers): every cell of replicate r has the same cohort and the same epsilon; only the signal and beta change."""
    L, base = STATE["lengths"], STATE["seed"]
    assert L, "lengths not set"
    rng = np.random.default_rng([base, int(r)])
    cohort = [GEN["generate_arm2_participant"](f"{cell}_r{int(r):03d}_p{i + 1:02d}", int(n), SNR, rng)
              for i, n in enumerate(L)]
    erng = np.random.default_rng([base, int(r), 1])
    eps = [erng.standard_normal(int(n)) for n in L]
    return cohort, eps


def cohort_signal(cohort, condition):
    """Signal S_t per participant (0 where undefined), z-scored over the cohort's participant-days where defined.
    C1: v_t - v_pop, v_t = causal volatility as F reads it (expanding SD of the observed series up to day t,
        min_periods = 3), v_pop = the population volatility norm of CELL 14.4 (median of the other participants'
        SDs, the norm F uses).  C2: x_t, the observed level.  C0: no signal."""
    ns = [len(p["days_df"]) for p in cohort]
    if condition == "C0":
        return [np.zeros(n) for n in ns]
    xs = [pd.to_numeric(p["days_df"]["sim_feature"], errors="coerce").to_numpy(float) for p in cohort]
    if condition == "C2":
        raw = xs
    elif condition == "C1":
        raw = []
        for p, x in zip(cohort, xs):
            v = pd.Series(x).expanding(min_periods=3).std().to_numpy(float)
            vpop = float(NS["_compute_population_volatility_norm"](cohort, p["pid"], SIM_FEATS, False)[0][0])
            raw.append(v - vpop)
    else:
        raise ValueError(condition)
    allv = np.concatenate(raw)
    ok = np.isfinite(allv)
    mu, sd = float(allv[ok].mean()), float(allv[ok].std())
    return [np.where(np.isfinite(a), (a - mu) / sd, 0.0) for a in raw]


def next_day_outcome(s, e, beta):
    """Row s is scored against Y_{s+1} = beta * S_s + eps_{s+1}; the last row has no next day (NaN)."""
    y = np.full(len(s), np.nan)
    y[:-1] = beta * s[:-1] + e[1:]
    return y


def scored_days(n, g=None):
    g = NS["G"] if g is None else g
    t0 = min(10, max(5, n // 3))
    return np.arange(t0 + g, n - 1)


def _clear_caches():
    for c in ("_PREP_CACHE", "_CAUSAL_CACHE", "_RUN_CACHE"):
        NS[c].clear()
    NS["TASK3_Y"].clear()


def _assert_caches_empty():
    for c in ("_PREP_CACHE", "_CAUSAL_CACHE", "_RUN_CACHE"):
        assert not NS[c], f"{c} is not empty at the start of a replicate"
    assert not NS["TASK3_Y"], "TASK3_Y is not empty at the start of a replicate"


def _array_sha(arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.nan_to_num(np.asarray(a, float), nan=-1.234567e300).tobytes())
    return h.hexdigest()


def decision_stats(df, a="v3", b="rawrf", prefix=""):
    """Two-sided paired Wilcoxon, V3 vs raw-RF, across participants with both rho finite (the Task 1 call).
    a, b: the columns compared (primary rho, or the secondary anomaly rho); keys are prefixed with `prefix`."""
    d = df.dropna(subset=[a, b])
    diff = (d[a] - d[b]).to_numpy(float)
    try:
        p = float(wilcoxon(d[a], d[b]).pvalue)
    except ValueError:
        p = float("nan")
    nz = diff[diff != 0]
    rk = rankdata(np.abs(nz)) if len(nz) else np.array([])
    tp, tm = float(rk[nz > 0].sum()), float(rk[nz < 0].sum())
    direction = "v3" if tp > tm else ("rawrf" if tm > tp else "tie")
    out = dict(n_pairs=int(len(d)), mean_rho_v3=float(d[a].mean()), mean_rho_rawrf=float(d[b].mean()),
               mean_delta_rho=float(diff.mean()) if len(diff) else float("nan"),
               wins_v3=int((diff > 0).sum()), p=p, signed_rank_plus=tp, signed_rank_minus=tm, direction=direction)
    for thr in THRESHOLDS:
        k = f"{thr:g}".replace("0.", "p")
        sig = bool(np.isfinite(p) and p < thr)
        out[f"sig_{k}"] = sig
        out[f"v3_win_{k}"] = sig and direction == "v3"
        out[f"rawrf_win_{k}"] = sig and direction == "rawrf"
    return {prefix + k: v for k, v in out.items()}


def fold_training_means(pr, y, alignment, g=None):
    """a(s) for every scored test day s: the mean of the target over the training rows that fold uses -- run_arms'
    loop exactly (embargo check, purge, finite-target rows, >= 5 training rows); fold size 1. Stage 0 review memo."""
    g = NS["G"] if g is None else g
    n = pr["n"]
    days, a = [], []
    for tr_idx, te_idx in NS["get_folds"](pr, g):
        if int(te_idx.min()) - int(tr_idx.max()) < g:
            continue
        if NS["V1_2_CONFIG"]["PURGE_SUPPORT_OVERLAP"]:
            tr_idx, _ = NS["purge_training_rows"](tr_idx, te_idx, alignment, n)
        tr_v = tr_idx[np.isfinite(y[tr_idx])]
        te_v = te_idx[np.isfinite(y[te_idx])]
        if len(tr_v) < 5 or len(te_v) < 1:
            continue
        m = float(np.mean(y[tr_v]))
        days.extend(int(t) for t in te_v)
        a.extend([m] * len(te_v))
    return np.array(days, int), np.array(a, float)


def anomaly_rhos(pr, res, alignment, arms):
    """rho_anom = Spearman(f(s) - a(s), y(s) - a(s)) per arm over the participant's scored test days (secondary)."""
    days, a = fold_training_means(pr, np.asarray(res["y_fixed"], float), alignment)
    assert np.array_equal(days, np.asarray(res["te"], int)), f"{pr['pid']}: a(s) days differ from the scored days"
    yv = np.asarray(res["y"], float)
    out = {}
    for arm in arms:
        f = np.asarray(res["preds"][arm], float)
        ok = np.isfinite(f) & np.isfinite(yv)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[arm] = float(spearmanr(f[ok] - a[ok], yv[ok] - a[ok])[0]) if ok.sum() >= 3 else float("nan")
    return out


def run_replicate(cell, r, condition, beta):
    """One replicate: V12-2 run_arms (fast folds), arms V3 and raw-RF, next-day outcome via the T3-1 wrapper."""
    t_start = time.perf_counter()
    _assert_caches_empty()
    try:
        cohort, eps = make_cohort(cell, r)
        S = cohort_signal(cohort, condition)
        Y = {p["pid"]: next_day_outcome(s, e, beta) for p, s, e in zip(cohort, S, eps)}
        for pid, y in Y.items():
            NS["TASK3_Y"][(pid, SIM_TARGET)] = y
        label = f"task4 {cell} r{r}"
        with contextlib.redirect_stdout(io.StringIO()):
            out = NS["run_arms"](cohort, SIM_FEATS, False, target=SIM_TARGET, arms=list(T4_ARMS),
                                 alignment="next_day", label=label)
        flog = NS["RUN_LOGS"].pop(label)
        purged = int(flog["purged_rows_summed_over_folds"].sum()) if len(flog) else 0
        assert purged == 0, f"next-day purge removed {purged} training rows -- STOP"
        out = out.set_index("pid")
        rows = []
        for p, s in zip(cohort, S):
            pid, n = p["pid"], len(p["days_df"])
            hits = [v for k, v in NS["_RUN_CACHE"].items() if k[0] == pid]
            assert len(hits) == 1, f"{pid}: {len(hits)} cache entries"
            te = np.asarray(hits[0]["te"], int)
            assert np.array_equal(te, scored_days(n)), f"{pid}: scored days differ from the fold rule"
            y = Y[pid]
            orc = float(spearmanr(s[te], y[te])[0]) if beta != 0 else float("nan")
            have = pid in out.index
            an = anomaly_rhos(NS["prepare"](p, SIM_FEATS, False, cohort), hits[0], "next_day", T4_ARMS)
            rows.append(dict(pid=pid, n_rows=n, n_folds=int(out.loc[pid, "n_folds"]) if have else 0,
                             v3=float(out.loc[pid, "v3"]) if have else float("nan"),
                             rawrf=float(out.loc[pid, "rawrf"]) if have else float("nan"), oracle_rho=orc,
                             v3_anom=an["v3"], rawrf_anom=an["rawrf"]))
        x_sha = _array_sha([p["days_df"]["sim_feature"].to_numpy(float) for p in cohort])
        e_sha = _array_sha(eps)
    finally:
        _clear_caches()
    df = pd.DataFrame(rows)
    rec = dict(cell=cell, rep=int(r), condition=condition, beta=float(beta), participants=rows,
               oracle_rho_mean=float(np.nanmean(df["oracle_rho"])) if beta != 0 else float("nan"),
               purged_training_rows=purged, cohort_x_sha256=x_sha, eps_sha256=e_sha)
    rec.update(decision_stats(df))
    rec.update(decision_stats(df, "v3_anom", "rawrf_anom", prefix="anom_"))     # secondary metric (Stage 0 review memo)
    rec["seconds"] = round(time.perf_counter() - t_start, 2)
    return rec


def _same_record(a, b):
    for key in ("p", "anom_p"):
        x, y = a.get(key), b.get(key)
        if x != y and not (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y)):
            return False
    if a["cohort_x_sha256"] != b["cohort_x_sha256"] or a["eps_sha256"] != b["eps_sha256"]:
        return False
    for u, v in zip(a["participants"], b["participants"]):
        for k in ("pid", "n_folds", "v3", "rawrf", "oracle_rho", "v3_anom", "rawrf_anom"):
            x, y = u.get(k), v.get(k)
            if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
                continue
            if x != y:
                return False
    return len(a["participants"]) == len(b["participants"])


def _seq(jobs):
    """Run replicates one after another in this (fresh) process."""
    return [dict(run_replicate(*j), worker_pid=os.getpid()) for j in jobs]


# ------------------------------------------------------------------------------------------------------------
# Checks, re-benchmark, calibration, fidelity
# ------------------------------------------------------------------------------------------------------------
def fold_check():
    """The fast generator returns exactly the folds of ndi.blocked_tscv (V12-2 get_folds) on one full cohort."""
    cohort, _ = make_cohort(CHECK_SPEC[0], CHECK_REPS[0])
    rows, t = [], time.perf_counter()
    try:
        for p in cohort:
            pr = NS["prepare"](p, SIM_FEATS, False, cohort)
            ref = NS["_get_folds_v12"](pr, NS["G"])
            fast = NS["get_folds"](pr, NS["G"])
            same = len(ref) == len(fast) and all(np.array_equal(a, c) and np.array_equal(b, d)
                                                 for (a, b), (c, d) in zip(ref, fast))
            rows.append(dict(pid=p["pid"], n_rows=pr["n"], folds_blocked_tscv=len(ref), folds_fast=len(fast),
                             identical=bool(same)))
    finally:
        _clear_caches()
    ok = all(r["identical"] for r in rows)
    return dict(check="fold generator equals ndi.blocked_tscv", verdict="PASS" if ok else "FAIL",
                participants=len(rows), folds=int(sum(r["folds_fast"] for r in rows)),
                seconds=round(time.perf_counter() - t, 1), rows=rows)


def isolation_check():
    """Replicate A run alone in a fresh process vs run after replicate B in another fresh process: identical output.
    The two processes run at the same time, so the timings are per-cohort times with two processes busy."""
    a, b = CHECK_REPS[1], CHECK_REPS[0]
    cell, cond, beta = CHECK_SPEC
    ctx = mp.get_context("spawn")
    t = time.perf_counter()
    with ProcessPoolExecutor(1, mp_context=ctx, initializer=_worker_init, initargs=(worker_payload(),)) as ea, \
         ProcessPoolExecutor(1, mp_context=ctx, initializer=_worker_init, initargs=(worker_payload(),)) as eb:
        fa = ea.submit(_seq, [(cell, a, cond, beta)])
        fb = eb.submit(_seq, [(cell, b, cond, beta), (cell, a, cond, beta)])
        ra, rb = fa.result(), fb.result()
    wall = time.perf_counter() - t
    alone, after = ra[0], rb[1]
    same = _same_record(alone, after)
    distinct = alone["worker_pid"] != after["worker_pid"]
    return dict(check="cache isolation: replicate run alone == run after a different replicate",
                verdict="PASS" if (same and distinct) else "FAIL", identical=bool(same),
                separate_processes=bool(distinct), replicate_a=a, replicate_b=b, condition=cond, beta=beta,
                pids_example=[alone["participants"][0]["pid"], rb[0]["participants"][0]["pid"]],
                folds_per_cohort=int(sum(x["n_folds"] for x in alone["participants"])),
                seconds_alone_while_other_process_busy=alone["seconds"],
                seconds_b_then_a=[rb[0]["seconds"], rb[1]["seconds"]], wall_seconds=round(wall, 1),
                mean_delta_rho_a=alone["mean_delta_rho"], p_a=alone["p"])


def projection(sec_per_cohort, workers, replicates, n_cells=len(MAIN_CELLS)):
    cohorts = n_cells * replicates
    core_h = cohorts * sec_per_cohort / 3600
    return dict(seconds_per_cohort_per_process=round(sec_per_cohort, 1), workers=workers,
                replicates_per_cell=replicates, cells=n_cells, cohorts=cohorts, core_hours=round(core_h, 1),
                wall_hours=round(core_h / max(workers, 1), 1),
                extension_cell_core_hours=round(replicates * sec_per_cohort / 3600, 1))


def bench(workers):
    """All workers busy at once, one cohort each (check cohorts 900002...)."""
    ctx = mp.get_context("spawn")
    cell, cond, beta = CHECK_SPEC
    jobs = [[(cell, CHECK_REPS[1] + 1 + i, cond, beta)] for i in range(workers)]
    t = time.perf_counter()
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_worker_init, initargs=(worker_payload(),)) as ex:
        res = [r[0] for r in ex.map(_seq, jobs)]
    wall = time.perf_counter() - t
    per = [r["seconds"] for r in res]
    return dict(workers=workers, wall_seconds=round(wall, 1), seconds_per_cohort=per,
                median_seconds_per_cohort=round(float(np.median(per)), 1),
                folds_per_cohort=int(sum(x["n_folds"] for x in res[0]["participants"])),
                effective_seconds_per_cohort=round(wall / workers, 1))


def calibrate():
    """beta per level on one pilot cohort (replicate key 999999, used by no replicate): oracle rho = mean over the 16
    participants of Spearman(S_s, Y_{s+1}) on the scored test days. No model fitting."""
    cohort, eps = make_cohort("CAL", CALIBRATION_REP)
    days = [scored_days(len(p["days_df"])) for p in cohort]
    grid = np.unique(np.concatenate([[0.0], np.geomspace(1e-3, 1e3, 301)]))
    table, betas = [], {}
    for cond in ("C1", "C2"):
        S = cohort_signal(cohort, cond)

        def orc(b, S=S):
            return float(np.mean([spearmanr(s[d], b * s[d] + e[d + 1])[0] for s, e, d in zip(S, eps, days)]))
        vals = np.array([orc(b) for b in grid])
        table += [dict(condition=cond, beta=float(b), oracle_rho=float(v)) for b, v in zip(grid, vals)]
        for cell, (c, level, tgt) in LEVELS.items():
            if c != cond:
                continue
            j = int(np.argmax(vals >= tgt))
            assert vals[j] >= tgt and j > 0, f"{cell}: oracle rho {tgt} not reached on the beta grid"
            lo, hi = float(grid[j - 1]), float(grid[j])
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if orc(mid) < tgt else (lo, mid)
            b = float(f"{hi:.6g}")
            betas[cell] = dict(condition=cond, level=level, target_oracle_rho=tgt, beta=b,
                               oracle_rho_at_beta=round(orc(b), 6))
    cal = dict(replicate_key=CALIBRATION_REP, participants=len(cohort),
               lengths=[len(p["days_df"]) for p in cohort], scored_days=int(sum(len(d) for d in days)),
               oracle_definition="mean over participants of Spearman(S_s, Y_{s+1}) on the scored test days",
               betas=betas)
    return cal, pd.DataFrame(table)


def fidelity_check(participants, feat_names, y_by_pid, archived, target="mood"):
    """The Task 4 runner (V12-2 run_arms with the fast fold generator, arms V3 and raw-RF, next-day wrapper) on the
    real data vs the archived Task 3 per-participant values (results_v1.4_task3/task3_mood_per_participant.csv)."""
    _assert_caches_empty()
    t = time.perf_counter()
    try:
        for pid, y in y_by_pid.items():
            NS["TASK3_Y"][(pid, target)] = np.asarray(y, float)
        with contextlib.redirect_stdout(io.StringIO()):
            out = NS["run_arms"](participants, feat_names, False, target=target, arms=list(T4_ARMS),
                                 alignment="next_day", label="fidelity")
        flog = NS["RUN_LOGS"].pop("fidelity")
        purged = int(flog["purged_rows_summed_over_folds"].sum())
        anom = {}
        for p in participants:
            hits = [v for k, v in NS["_RUN_CACHE"].items() if k[0] == p["pid"] and k[2] == target]
            if len(hits) == 1 and len(hits[0]["y"]) >= 3:
                anom[p["pid"]] = anomaly_rhos(NS["prepare"](p, feat_names, False, participants), hits[0], "next_day", T4_ARMS)
    finally:
        _clear_caches()
    ref = archived.set_index("pid")
    new = out.set_index("pid")
    rows = []
    for pid in ref.index:
        have = pid in new.index
        rows.append(dict(pid=pid, eligible=bool(ref.loc[pid, "eligible"]) if "eligible" in ref.columns else True,
                         n_folds_task3=int(ref.loc[pid, "n_folds"]),
                         n_folds_task4=int(new.loc[pid, "n_folds"]) if have else -1,
                         v3_task3=float(ref.loc[pid, "v3"]), v3_task4=float(new.loc[pid, "v3"]) if have else np.nan,
                         rawrf_task3=float(ref.loc[pid, "rawrf"]),
                         rawrf_task4=float(new.loc[pid, "rawrf"]) if have else np.nan,
                         v3_anom_task4=anom.get(pid, {}).get("v3", np.nan), rawrf_anom_task4=anom.get(pid, {}).get("rawrf", np.nan)))
    df = pd.DataFrame(rows)
    df["abs_diff_v3"] = (df["v3_task4"] - df["v3_task3"]).abs()
    df["abs_diff_rawrf"] = (df["rawrf_task4"] - df["rawrf_task3"]).abs()
    el = df[df["eligible"]]
    worst = float(np.nanmax(np.r_[df["abs_diff_v3"].to_numpy(), df["abs_diff_rawrf"].to_numpy()]))
    ok = (bool((df["n_folds_task3"] == df["n_folds_task4"]).all()) and worst < 1e-9 and purged == 0
          and set(new.index) == set(ref.index) and df[["v3_task4", "rawrf_task4"]].notna().all().all())
    return df, dict(check="fidelity: Task 4 runner reproduces the Task 3 next-day-mood V3 and raw-RF rho",
                    verdict="PASS" if ok else "FAIL", participants=len(df), eligible=int(len(el)),
                    max_abs_diff=worst, max_abs_diff_eligible=float(np.nanmax(
                        np.r_[el["abs_diff_v3"].to_numpy(), el["abs_diff_rawrf"].to_numpy()])) if len(el) else None,
                    purged_training_rows=purged, seconds=round(time.perf_counter() - t, 1))


# ------------------------------------------------------------------------------------------------------------
# Config (built by the notebook before tagging; checked by every command that runs replicates)
# ------------------------------------------------------------------------------------------------------------
def build_config(calibration, replicates, machine, physical_cores, lengths_by_pid, v14_cfg, task3_cfg_sha):
    cells = {"C0": dict(condition="C0", level=0, target_oracle_rho=None, beta=0.0, oracle_rho_at_beta=None)}
    for cell in list(MAIN_CELLS) + [EXTENSION_CELL]:
        if cell != "C0":
            cells[cell] = dict(calibration["betas"][cell])
    return dict(
        VERSION="v1.4-task4", TAG=TAG, CODE_TAG=TASK3_TAG, SETTINGS_TAG=V14_TAG,
        MEMOS=["revision analyses memo, Task 4", "Task 3 review and Task 4 execution memo, Sections 3-5",
               "Stage 0 review, final experiments and scope freeze memo, Section 3 (secondary metric)"],
        STAGE0="learnability check passed before this config was written (results_task4_stage0/stage0_decision.json)",
        PURPOSE="show that the protocol detects a genuine advantage when one exists, does not manufacture one when "
                "none exists, and has an empirical minimum detectable delta rho at n = 16",
        PROTOCOL="unchanged from v1.4 / Task 3: T0 = min(10, max(5, n // 3)), embargo g = 6, fold size 1, "
                 "fold-local statistics truncated at the test day, purge rule, RF settings (ndi.CONFIG RF_PARAMS, "
                 "n_jobs = 1), F_SCALE 1.0, seed rules. Code: CELL 14.4, V12-1, V12-2 and the T3-1 wrapper executed "
                 "from the Task 3 notebook under tag v1.4-task3 (hashes below). The only substitution: get_folds "
                 "returns the folds of ndi.blocked_tscv without fitting a forest per fold (fold check).",
        ARMS=list(T4_ARMS), ARMS_REASON="the power curve is for V3 vs raw-RF (v1.3 memo); other arms add cost only",
        GENERATOR=dict(source="Arm 2, CELL C2.0 and CELL C2.1 of the v1 notebook, verbatim (two-state latent Markov "
                              "process, participant intercepts, injected missingness and flatlines)", snr=SNR,
                       feature=SIM_FEATS, sha256=generator_hashes()),
        COHORT=dict(participants=len(lengths_by_pid), lengths_v1_4=lengths_by_pid,
                    pid_format="<cell>_r<replicate, 3+ digits>_p<01-16>, e.g. C1b2_r037_p05 (unique per cell and "
                               "replicate, so no cache entry can be shared)"),
        OUTCOME=dict(
            alignment="next_day (Task 3's code): row s is scored against Y_{s+1}; target support {s + 1}; the purge "
                      "removes nothing (asserted for every replicate); the last row has no next day",
            model="Y_{s+1} = beta * S_s + eps_{s+1}, eps ~ N(0, 1) independent over participant-days",
            C0="no signal (beta = 0): Y independent of everything",
            C1="S = z(v_t - v_pop): v_t = causal volatility as the F channel reads it (expanding SD of the observed "
               "series over days <= t, min_periods = 3, NaN skipped); v_pop = population volatility norm of CELL "
               "14.4 (median of the other 15 participants' whole-series SDs), the norm F uses",
            C2="S = z(x_t): the observed raw feature level on day t",
            z="over all participant-days of the cohort where the quantity is defined (mean, SD with ddof 0); days "
              "where it is undefined (x_t missing; fewer than 3 observations for v_t) get S = 0",
            common_random_numbers="cohort data and eps depend only on the replicate r: numpy default_rng([ARM2_SEED, "
                                  "r]) for the cohort (participants generated in order p01..p16) and default_rng("
                                  "[ARM2_SEED, r, 1]) for eps. Every cell of replicate r has the same cohort and eps; "
                                  "only S and beta change. Never depends on the machine or shard."),
        SEED=dict(base="ARM2_SEED = 20260817 (CELL C2.0)", calibration_cohort=CALIBRATION_REP,
                  check_cohorts=list(CHECK_REPS)),
        CALIBRATION=dict(calibration, grid="beta = 0 and 301 log-spaced values 0.001..1000, then bisection; beta "
                                           "rounded to 6 significant digits",
                         mapping_file="results_task4/calibration/beta_to_oracle_rho.csv"),
        CELLS=cells, MAIN_CELLS_PRIORITY=list(MAIN_CELLS),
        EXTENSION=dict(cell=EXTENSION_CELL, target_oracle_rho=0.60,
                       rule="runs automatically with the same replicate count if the power of C1b3 (highest main C1 "
                            "level) is below 0.80 at P < .0125 -- i.e. at either threshold -- so both MDEs can be "
                            "estimated; nothing else may be added after results are seen", threshold=0.0125),
        REPLICATES_PER_CELL=int(replicates),
        REPLICATE_RULE="100 per cell; 200 only if the full-run machine has 16 or more physical cores (Task 3 review memo, "
                       "Section 3). The machine is confirmed by ma'am after this tag (Stage 0 review memo, Section 3), so the "
                       "count is fixed at the value above; any change would be a new config and tag before the full run",
        FULL_RUN_MACHINE=machine, FULL_RUN_PHYSICAL_CORES=physical_cores,
        PILOT=dict(cells=list(PILOT_CELLS), replicates=list(PILOT_REPS), folder="results_task4/pilot",
                   note="the full run recomputes these replicates in results_task4/cells; merge asserts they are "
                        "identical (determinism across sessions)"),
        SECONDARY_METRIC=dict(
            name="anomaly rho (pre-specified secondary metric; Stage 0 review memo, Section 3)",
            definition="for every scored test day s, a(s) = mean of the target over the training rows that day's fold uses "
                       "(run_arms' loop: embargo check, purge, finite-target rows, >= 5 rows); rho_anom = Spearman(f(s) - "
                       "a(s), y(s) - a(s)) per participant over the scored test days, for each arm, from the same predictions",
            recorded="per participant v3_anom and rawrf_anom; per replicate the same decision statistics as the primary "
                     "metric with prefix anom_ (two-sided paired Wilcoxon P, mean delta, direction, significance at "
                     "P < .05 and P < .0125)",
            reported="power per cell (Wilson 95% CI) and MDE (probit, bootstrap CI), as for the primary metric; the primary "
                     "metric stays primary; the extension rule and the reading of C0 / C2 use the primary metric",
            cost="none: the same predictions"),
        DECISION_RULE="per replicate: two-sided paired Wilcoxon (scipy.stats.wilcoxon(v3, rawrf), defaults, as Task 1) "
                      "on V3 vs raw-RF across the participants with both rho finite; record P, mean delta rho, both "
                      "arms' mean rho; significant at P < .05 (optimistic) and at P < .0125 (conservative: the "
                      "strictest threshold BH could apply across four comparisons). Direction = the larger signed-rank "
                      "sum.",
        POWER=dict(C1="proportion of replicates significant in V3's favour", C2="proportion significant in raw-RF's "
                   "favour; V3 win rate (significant in V3's favour) reported against 0.05",
                   C0="false-positive rate = proportion significant in either direction (and per direction)",
                   interval="Wilson 95% CI", mc_se="mean delta rho and arm means with Monte Carlo SE = SD / sqrt(R)"),
        MDE=dict(points="C0 and the C1 levels (with C1b4 if it ran): x = cell mean delta rho, y = proportion "
                        "significant in V3's favour",
                 fit="probit (binomial maximum likelihood), 80% crossing, at each threshold",
                 ci="bootstrap over replicates within each cell, 2,000 resamples, seed 42, percentile 95%",
                 fallback="if the observed power at the highest C1 level is below 0.80: 'MDE above delta rho = <that "
                          "level's mean delta rho>' (no extrapolation)",
                 compare_with=REAL_UPPER_BOUNDS),
        READING=["C0: false-positive rate should be at or below 0.05; its Wilson CI is reported either way",
                 "C1: the MDE is the delta rho at which the protocol detects V3's advantage 80% of the time with 16 "
                 "participants, set against the upper 95% bounds of the real comparisons (0.132 and 0.076)",
                 "C2: raw-RF should match or beat V3; a V3 win rate above 0.05 means the protocol favours the index",
                 "if C1 never reaches 80% even at the extension level, the report is 'MDE above delta rho = [value]'"],
        FIGURE5="power curves for C1 and C2 (both thresholds, Wilson 95% CI), C0 false-positive rate marked, probit "
                "fits, MDEs, the real upper bounds as reference lines",
        OUTPUT="results_task4/cells/<cell>/rep_XXX.json (one per replicate); results_task4/analysis",
        CODE_HASHES=STATE["code_hashes"], TASK3_NOTEBOOK_UNDER_TAG=STATE["task3_notebook"],
        TASK4_SCRIPT_SHA256=sha256_file(script_path()), NDI_PIPELINE_SHA256=STATE["ndi_sha256"],
        V1_4_CONFIG_SHA256=canonical_sha(v14_cfg), TASK3_CONFIG_SHA256=task3_cfg_sha,
        VERSIONS=versions(), REPORTING_DECIMALS=3)


def load_config(repo):
    path = os.path.join(repo, CONFIG_NAME)
    assert os.path.isfile(path), f"{CONFIG_NAME} not found in {repo}"
    return json.load(open(path)), path


def verify_tagged(repo, cfg):
    """T3-0 rule: refuse to run replicates unless HEAD is the commit of tag v1.4-task4, and the config and this
    script equal the tagged ones; package versions must equal the config's."""
    tag_commit = (git(repo, "rev-list", "-n", "1", TAG) or "").strip()
    head = (git(repo, "rev-parse", "HEAD") or "").strip()
    tagged_cfg = git(repo, "show", f"{TAG}:{CONFIG_NAME}")
    tagged_script = git(repo, "show", f"{TAG}:{SCRIPT_NAME}")
    tag_time = (git(repo, "log", "-1", "--format=%cI", TAG) or "").strip()
    problems = []
    if not tag_commit:
        problems.append(f"tag {TAG} not found")
    if head != tag_commit:
        problems.append(f"HEAD {head[:10]} is not the tag commit {tag_commit[:10]} (git checkout {TAG})")
    if tagged_cfg is None or json.loads(tagged_cfg) != cfg:
        problems.append(f"{CONFIG_NAME} differs from the tagged one")
    if tagged_script is None or _norm_text(tagged_script) != _norm_text(open(script_path(), encoding="utf-8").read()):
        problems.append(f"{SCRIPT_NAME} differs from the tagged one")
    if tag_time and datetime.datetime.fromisoformat(tag_time) >= datetime.datetime.now(datetime.timezone.utc):
        problems.append("tag time is not before now")
    v = versions()
    if v != cfg["VERSIONS"]:
        want = cfg["VERSIONS"]
        problems.append(f"package versions {v} differ from the config {want}: pip install numpy=={want['numpy']} "
                        f"scipy=={want['scipy']} scikit-learn=={want['sklearn']} pandas=={want['pandas']}")
    if problems:
        raise SystemExit("STOP (untagged or different code):\n  - " + "\n  - ".join(problems))
    return dict(tag=TAG, tag_commit=tag_commit, tag_time=tag_time, head_commit=head)


def check_code_against_config(cfg):
    problems = []
    if STATE["code_hashes"] != cfg["CODE_HASHES"]:
        problems.append("Task 3 code hashes differ from the config")
    if generator_hashes() != cfg["GENERATOR"]["sha256"]:
        problems.append("generator hashes differ from the config")
    if STATE["ndi_sha256"] != cfg["NDI_PIPELINE_SHA256"]:
        problems.append(f"ndi_pipeline.py ({STATE['ndi_path']}) differs from the config's")
    if canonical_sha(STATE["v14_cfg"]) != cfg["V1_4_CONFIG_SHA256"]:
        problems.append("config_v1_4.json under tag v1.4 differs from the config's")
    if problems:
        raise SystemExit("STOP:\n  - " + "\n  - ".join(problems))


def cell_spec(cfg, cell):
    c = cfg["CELLS"][cell]
    return c["condition"], float(c["beta"])


# ------------------------------------------------------------------------------------------------------------
# Running replicates
# ------------------------------------------------------------------------------------------------------------
def rep_path(out, cell, r, pilot=False):
    return os.path.join(out, "pilot" if pilot else "cells", cell, f"rep_{int(r):03d}.json")


def _valid(path):
    try:
        with open(path) as f:
            json.load(f)
        return True
    except Exception:
        return False


def _job(cell, r, condition, beta, path, meta):
    rec = run_replicate(cell, r, condition, beta)
    rec.update(meta)
    rec.update(machine_info(), versions=versions(), worker_pid=os.getpid(), finished=utcnow())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, path)
    return dict(cell=cell, rep=int(r), seconds=rec["seconds"], p=rec["p"], mean_delta_rho=rec["mean_delta_rho"],
                direction=rec["direction"])


def run_jobs(jobs, workers, meta, hours=None, log=print):
    """jobs: (cell, r, condition, beta, path). Keeps at most `workers` replicates running; with --hours, no new
    replicate starts unless it is expected to finish in time. A disconnect loses at most one replicate per worker."""
    todo = [j for j in jobs if not _valid(j[4])]
    log(f"{len(jobs) - len(todo)} of {len(jobs)} replicate files already present; {len(todo)} to run on "
        f"{workers} worker(s)" + (f", time budget {hours} h" if hours else ""))
    if not todo:
        return []
    deadline = time.time() + hours * 3600 if hours else None
    done_s, results, it = [], [], iter(todo)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_worker_init,
                             initargs=(worker_payload(),)) as ex:
        pending = set()
        while True:
            while len(pending) < workers:
                if deadline is not None and done_s and time.time() + 1.15 * float(np.median(done_s)) > deadline:
                    break
                j = next(it, None)
                if j is None:
                    break
                pending.add(ex.submit(_job, *j, meta))
            if not pending:
                break
            fin, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in fin:
                s = f.result()
                done_s.append(s["seconds"]); results.append(s)
                log(f"  {s['cell']:>5} r{s['rep']:03d}  {s['seconds']:6.0f} s  mean delta rho {s['mean_delta_rho']:+.3f}"
                    f"  P {s['p']:.4f}  ({s['direction']})   [{len(results)}/{len(todo)}, "
                    f"{(time.time() - t0) / 3600:.2f} h]")
    left = len(todo) - len(results)
    log(f"finished {len(results)} replicate(s)" + (f"; {left} left for the next session (time budget)" if left else ""))
    return results


def parse_reps(s, R):
    if s in (None, "", "all"):
        return list(range(R))
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(r for r in out if 0 <= r < R))


def load_cell(out, cell, pilot=False):
    recs = {}
    for f in sorted(glob.glob(os.path.join(out, "pilot" if pilot else "cells", cell, "rep_*.json"))):
        try:
            rec = json.load(open(f))
            recs[int(rec["rep"])] = rec
        except Exception:
            pass
    return recs


def extension_status(out, cfg):
    R, thr = cfg["REPLICATES_PER_CELL"], cfg["EXTENSION"]["threshold"]
    recs = load_cell(out, "C1b3")
    have = [recs[r] for r in range(R) if r in recs]
    if len(have) < R:
        return dict(status="pending", reason=f"C1b3 has {len(have)} of {R} replicates")
    k = sum(bool(x["p"] < thr and x["direction"] == "v3") for x in have)
    power = k / R
    return dict(status="triggered" if power < 0.80 else "not triggered", power_C1b3=power, threshold=thr,
                reason=f"C1b3 power at P < {thr:g} = {power:.3f} ({'<' if power < 0.80 else '>='} 0.80)")


# ------------------------------------------------------------------------------------------------------------
# Analysis (merge)
# ------------------------------------------------------------------------------------------------------------
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (0.0 if k == 0 else max(0.0, c - h)), (1.0 if k == n else min(1.0, c + h))


def probit_fit(x, k, n, x0=None):
    from scipy.optimize import minimize
    x, k, n = (np.asarray(v, float) for v in (x, k, n))

    def nll(th):
        pr = np.clip(norm.cdf(th[0] + th[1] * x), 1e-12, 1 - 1e-12)
        return -float(np.sum(k * np.log(pr) + (n - k) * np.log(1 - pr)))
    ph = np.clip((k + 0.5) / (n + 1), 1e-3, 1 - 1e-3)
    b0 = np.polyfit(x, norm.ppf(ph), 1) if np.ptp(x) > 0 else np.array([0.0, 0.0])
    best = None
    starts = (b0[::-1], np.array([norm.ppf(ph[0]), 1.0]), np.array([-2.0, 10.0])) if x0 is None else (np.asarray(x0, float),)
    for st in starts:
        r = minimize(nll, st, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-12, maxiter=40000, maxfev=40000))
        if best is None or r.fun < best.fun:
            best = r
    return float(best.x[0]), float(best.x[1])


def _x80(a, b):
    return (norm.ppf(0.80) - a) / b if b > 0 else float("nan")


def collect(out, cfg, cells):
    R = cfg["REPLICATES_PER_CELL"]
    data = {}
    for cell in cells:
        recs = load_cell(out, cell)
        data[cell] = [recs[r] for r in range(R) if r in recs]
    return data


def analyse(out, cfg, tag_info=None, log=print):
    R = cfg["REPLICATES_PER_CELL"]
    ext = extension_status(out, cfg)
    cells = list(MAIN_CELLS) + ([EXTENSION_CELL] if ext["status"] == "triggered" else [])
    data = collect(out, cfg, cells)
    prog = {c: len(v) for c, v in data.items()}
    complete = all(prog[c] == R for c in cells) and ext["status"] != "pending"
    log("PROGRESS: " + ", ".join(f"{c} {prog[c]}/{R}" for c in cells) + f" | extension: {ext['reason']}"
        + ("" if ext["status"] == "pending" else f" -> {ext['status']}"))
    if not complete:
        log("Not complete yet: run the same `run` command again (files that exist are skipped).")
        return None
    A = os.path.join(out, "analysis")
    os.makedirs(A, exist_ok=True)
    cfg_sha = canonical_sha(cfg)
    # ---- per replicate and per participant
    rep_rows, pp_rows, prov_bad = [], [], []
    for cell in cells:
        for x in data[cell]:
            _stat = ("n_pairs", "mean_rho_v3", "mean_rho_rawrf", "mean_delta_rho", "wins_v3", "p", "direction",
                     "sig_p05", "sig_p0125", "v3_win_p05", "v3_win_p0125", "rawrf_win_p05", "rawrf_win_p0125")
            rep_rows.append({k: x.get(k) for k in ("cell", "rep", "condition", "beta") + _stat + tuple("anom_" + k for k in _stat)
                             + ("oracle_rho_mean", "seconds", "host", "finished", "cohort_x_sha256", "eps_sha256")})
            for pp in x["participants"]:
                pp_rows.append(dict(cell=cell, rep=x["rep"], **pp))
            if (x.get("config_sha256") != cfg_sha or x.get("script_sha256") != cfg["TASK4_SCRIPT_SHA256"]
                    or (tag_info and x.get("tag_commit") != tag_info["tag_commit"])):
                prov_bad.append(f"{cell} r{x['rep']}")
    reps = pd.DataFrame(rep_rows)
    reps.to_csv(os.path.join(A, "task4_per_replicate.csv"), index=False)
    pd.DataFrame(pp_rows).to_csv(os.path.join(A, "task4_per_participant.csv"), index=False)
    # ---- common random numbers: same cohort and eps for replicate r in every cell
    crn = reps.groupby("rep")[["cohort_x_sha256", "eps_sha256"]].nunique()
    crn_ok = bool((crn == 1).all().all())
    # ---- per cell, MDE and Figure 5: primary metric, then the pre-specified secondary metric (anomaly rho)
    cellsum = cell_summary(reps, cells, cfg, "")
    cellsum.to_csv(os.path.join(A, "task4_per_cell.csv"), index=False)
    mde = mde_fit(reps, cellsum, cells, cfg, "")
    json.dump(mde, open(os.path.join(A, "task4_mde.json"), "w"), indent=2)
    figure5(cellsum, mde, A, "figure5_power_curves", reference_lines=True)
    cellsum2 = cell_summary(reps, cells, cfg, "anom_")
    cellsum2.to_csv(os.path.join(A, "task4_per_cell_secondary_anomaly.csv"), index=False)
    mde2 = mde_fit(reps, cellsum2, cells, cfg, "anom_")
    json.dump(mde2, open(os.path.join(A, "task4_mde_secondary_anomaly.json"), "w"), indent=2)
    figure5(cellsum2, mde2, A, "figure5_secondary_anomaly", reference_lines=False,
            xlabel="Cell mean delta anomaly rho (V3 - raw-feature RF)")
    # ---- determinism: pilot replicates vs the same replicates of the full run
    det_rows = []
    for cell in PILOT_CELLS:
        pil = load_cell(out, cell, pilot=True)
        full = {x["rep"]: x for x in data.get(cell, [])}
        for r, x in sorted(pil.items()):
            if r in full:
                det_rows.append(dict(source="pilot vs full run", cell=cell, rep=r, identical=_same_record(x, full[r]),
                                     host_a=x.get("host"), host_b=full[r].get("host"),
                                     finished_a=x.get("finished"), finished_b=full[r].get("finished")))
    for f in sorted(glob.glob(os.path.join(out, "determinism", "*", "rep_*.json"))):
        x = json.load(open(f))
        ref = {y["rep"]: y for y in data.get(x["cell"], [])}.get(x["rep"])
        if ref is not None:
            det_rows.append(dict(source="verify re-run", cell=x["cell"], rep=x["rep"], identical=_same_record(x, ref),
                                 host_a=x.get("host"), host_b=ref.get("host"), finished_a=x.get("finished"),
                                 finished_b=ref.get("finished")))
    det = pd.DataFrame(det_rows)
    det.to_csv(os.path.join(A, "task4_determinism_check.csv"), index=False)
    # ---- reading, notes, hand-in, deviation note
    reading = reading_text(cellsum, mde, ext, cfg) + [""] + reading_text(cellsum2, mde2, ext, cfg, secondary=True)
    open(os.path.join(A, "task4_reading.txt"), "w").write("\n".join(reading) + "\n")
    fmt = format_cells(cellsum)
    fmt.to_csv(os.path.join(A, "task4_per_cell_formatted.csv"), index=False)
    fmt2 = format_cells(cellsum2)
    fmt2.to_csv(os.path.join(A, "task4_per_cell_secondary_anomaly_formatted.csv"), index=False)
    open(os.path.join(A, "figure5_caption.txt"), "w").write(FIGURE5_CAPTION.format(n=cfg["COHORT"]["participants"]) + "\n")
    dev = deviation_note(cfg, ext, sorted(set(reps["host"].astype(str))))
    open(os.path.join(A, "task4_deviation_note.txt"), "w").write("\n".join(dev) + "\n")
    hand = handin(out, A, cfg, cells, data, prov_bad, crn_ok, det, ext, tag_info)
    log("PRIMARY METRIC (rho)\n" + fmt.to_string(index=False))
    log("\nSECONDARY METRIC (anomaly rho, pre-specified)\n" + fmt2.to_string(index=False))
    log("\n".join(reading))
    log("\n".join(hand))
    return dict(cellsum=cellsum, mde=mde, cellsum_secondary=cellsum2, mde_secondary=mde2, determinism=det, handin=hand)


FIGURE5_CAPTION = ("Positive-control simulation with {n} participants (Arm 2 generator, SNR 1.0, v1.4 recording lengths). "
                   "x: cell mean delta rho (V3 - raw-feature RF); y: proportion of replicates in which the paired Wilcoxon "
                   "test is significant -- (A) P < .05, (B) P < .0125. Blue: C1 index-favouring outcome, significant in "
                   "V3's favour, with the probit fit and the MDE (80% crossing). Orange: C2 raw-favouring outcome, "
                   "significant in raw-RF's favour. Black star: C0 null outcome, false-positive rate in either direction. "
                   "Bars: Wilson 95% CI. Dotted horizontal lines at 0.80 and 0.05. Grey vertical lines: upper 95% bounds of "
                   "the real comparisons (raw-feature target and next-day mood, V3 vs raw-RF). The secondary-metric "
                   "version (anomaly rho) has the same layout without the grey lines.")


def cell_summary(reps, cells, cfg, pre):
    rows = []
    for cell in cells:
        d = reps[reps["cell"] == cell]
        cond = cfg["CELLS"][cell]["condition"]
        n = len(d)
        row = dict(metric="primary rho" if not pre else "secondary anomaly rho", cell=cell, condition=cond,
                   level=cfg["CELLS"][cell]["level"], target_oracle_rho=cfg["CELLS"][cell]["target_oracle_rho"],
                   beta=cfg["CELLS"][cell]["beta"], replicates=n,
                   oracle_rho_mean=float(d["oracle_rho_mean"].mean()) if cond != "C0" else np.nan,
                   mean_rho_v3=d[pre + "mean_rho_v3"].mean(), mc_se_v3=d[pre + "mean_rho_v3"].std(ddof=1) / math.sqrt(n),
                   mean_rho_rawrf=d[pre + "mean_rho_rawrf"].mean(),
                   mc_se_rawrf=d[pre + "mean_rho_rawrf"].std(ddof=1) / math.sqrt(n),
                   mean_delta_rho=d[pre + "mean_delta_rho"].mean(),
                   mc_se_delta_rho=d[pre + "mean_delta_rho"].std(ddof=1) / math.sqrt(n),
                   min_pairs=int(d[pre + "n_pairs"].min()), median_seconds=float(d["seconds"].median()))
        for k in ("p05", "p0125"):
            for what in ("sig", "v3_win", "rawrf_win"):
                kk = int(d[f"{pre}{what}_{k}"].astype(bool).sum())
                lo, hi = wilson(kk, n)
                row[f"{what}_{k}"] = kk / n; row[f"{what}_{k}_lo"] = lo; row[f"{what}_{k}_hi"] = hi
            col = {"C1": "v3_win", "C2": "rawrf_win", "C0": "sig"}[cond]
            row[f"power_{k}"] = row[f"{col}_{k}"]; row[f"power_{k}_lo"] = row[f"{col}_{k}_lo"]
            row[f"power_{k}_hi"] = row[f"{col}_{k}_hi"]
        rows.append(row)
    return pd.DataFrame(rows)


def mde_fit(reps, cellsum, cells, cfg, pre):
    mde = {}
    c1 = ["C0"] + [c for c in cells if cfg["CELLS"][c]["condition"] == "C1"]
    c1 = sorted(c1, key=lambda c: float(cellsum.set_index("cell").loc[c, "mean_delta_rho"]))
    rng = np.random.default_rng(42)
    for k, thr in (("p05", 0.05), ("p0125", 0.0125)):
        wins = {c: reps[reps["cell"] == c][f"{pre}v3_win_{k}"].to_numpy(bool) for c in c1}
        dlt = {c: reps[reps["cell"] == c][f"{pre}mean_delta_rho"].to_numpy(float) for c in c1}
        xs = np.array([dlt[c].mean() for c in c1]); ks = np.array([wins[c].sum() for c in c1])
        ns = np.array([len(wins[c]) for c in c1])
        a, b = probit_fit(xs, ks, ns)
        top = max([c for c in c1 if c != "C0"], key=lambda c: cfg["CELLS"][c]["level"])
        top_power = float(wins[top].mean())
        boot = []
        for _ in range(2000):
            bx, bk = [], []
            for c in c1:
                idx = rng.integers(0, len(wins[c]), len(wins[c]))
                bx.append(dlt[c][idx].mean()); bk.append(wins[c][idx].sum())
            aa, bb = probit_fit(np.array(bx), np.array(bk), ns, x0=(a, b))   # bootstrap: start at the full-data fit
            boot.append(_x80(aa, bb))
        boot = np.array(boot)
        fin = boot[np.isfinite(boot)]
        if top_power >= 0.80:
            val, statement = _x80(a, b), None
        else:
            val = float("nan")
            statement = f"MDE above delta rho = {float(dlt[top].mean()):.3f} (power {top_power:.2f} at {top})"
        mde[k] = dict(metric="primary rho" if not pre else "secondary anomaly rho", threshold=thr,
                      points=[dict(cell=c, mean_delta_rho=float(x), significant_v3=int(kk), replicates=int(nn))
                              for c, x, kk, nn in zip(c1, xs, ks, ns)],
                      probit_a=a, probit_b=b, mde_delta_rho=val, statement=statement,
                      power_at_highest_C1_level=top_power, highest_C1_level=top,
                      highest_C1_mean_delta_rho=float(dlt[top].mean()),
                      bootstrap_ci95=[float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))] if len(fin) else None,
                      bootstrap_resamples=2000, bootstrap_undefined=int(len(boot) - len(fin)),
                      compare_with=REAL_UPPER_BOUNDS if not pre else "the anomaly-rho upper bounds of the post hoc "
                                                                      "sensitivity analysis (results_sensitivity_anomaly)")
    return mde


def _fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def format_cells(c):
    def pw(r, k):
        return f"{_fmt(r[f'power_{k}'], 2)} [{_fmt(r[f'power_{k}_lo'], 2)}, {_fmt(r[f'power_{k}_hi'], 2)}]"
    return pd.DataFrame({
        "Cell": c["cell"], "Condition": c["condition"], "beta": [f"{b:.4g}" for b in c["beta"]],
        "Oracle rho": [_fmt(v) for v in c["oracle_rho_mean"]], "R": c["replicates"],
        "V3 mean rho": [f"{_fmt(m)} ({_fmt(s)})" for m, s in zip(c["mean_rho_v3"], c["mc_se_v3"])],
        "Raw-RF mean rho": [f"{_fmt(m)} ({_fmt(s)})" for m, s in zip(c["mean_rho_rawrf"], c["mc_se_rawrf"])],
        "Mean delta rho (MC SE)": [f"{_fmt(m)} ({_fmt(s)})" for m, s in zip(c["mean_delta_rho"], c["mc_se_delta_rho"])],
        "Power P<.05 [Wilson 95% CI]": [pw(r, "p05") for _, r in c.iterrows()],
        "Power P<.0125 [Wilson 95% CI]": [pw(r, "p0125") for _, r in c.iterrows()],
        "V3 win rate P<.05": [_fmt(v, 2) for v in c["v3_win_p05"]],
        "Raw-RF win rate P<.05": [_fmt(v, 2) for v in c["rawrf_win_p05"]]})


def reading_text(c, mde, ext, cfg, secondary=False):
    s = c.set_index("cell")
    c0 = s.loc["C0"]
    lines = [("SECONDARY METRIC, anomaly rho (pre-specified before the tag; Stage 0 review memo, Section 3)" if secondary
              else "TASK 4 READING, primary metric rho (rules fixed in config_task4.json before any replicate)"),
             f"C0 false-positive rate (two-sided): P<.05 {c0['sig_p05']:.3f} [Wilson {c0['sig_p05_lo']:.3f}, "
             f"{c0['sig_p05_hi']:.3f}] -> {'at or below' if c0['sig_p05'] <= 0.05 else 'ABOVE'} 0.05; "
             f"P<.0125 {c0['sig_p0125']:.3f} [{c0['sig_p0125_lo']:.3f}, {c0['sig_p0125_hi']:.3f}]; "
             f"in V3's favour at P<.05 {c0['v3_win_p05']:.3f}, in raw-RF's favour {c0['rawrf_win_p05']:.3f}"]
    for k, lab in (("p05", "P < .05 (optimistic)"), ("p0125", "P < .0125 (conservative)")):
        m = mde[k]
        if m["statement"]:
            txt = m["statement"]
        else:
            ci = m["bootstrap_ci95"]
            txt = f"MDE delta rho = {m['mde_delta_rho']:.3f}" + (f" [bootstrap 95% CI {ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
        if m["statement"] is None:
            ref, what = m["mde_delta_rho"], "the MDE"
        else:
            ref, what = m["highest_C1_mean_delta_rho"], "the largest delta rho simulated"
        comp = "; ".join(f"{n}: upper 95% bound {v:.3f} is {'below' if v < ref else 'at or above'} {what} ({ref:.3f})"
                         for n, v in REAL_UPPER_BOUNDS.items())
        if secondary:
            comp = "To be set against the anomaly-rho upper bounds of the post hoc sensitivity analysis"
        lines.append(f"C1 MDE at {lab}: {txt}. {comp}.")
    for cell in [x for x in s.index if s.loc[x, "condition"] == "C2"]:
        r = s.loc[cell]
        lines.append(f"{cell}: mean delta rho {r['mean_delta_rho']:+.3f}; raw-RF significant {r['rawrf_win_p05']:.2f} "
                     f"(P<.05); V3 win rate {r['v3_win_p05']:.3f} at P<.05 -> "
                     + ("the protocol does NOT favour the index" if r["v3_win_p05"] <= 0.05 else
                        "ABOVE 0.05: the protocol favours the index"))
    if not secondary:
        lines.append(f"Extension ({EXTENSION_CELL}, oracle rho 0.60): {ext['status']} ({ext.get('reason', '')}); "
                     "decided on the primary metric")
    return lines


def deviation_note(cfg, ext, hosts=None):
    return ["TASK 4 DEVIATIONS FROM THE ORIGINAL SPECIFICATION (revision analyses memo, Task 4), all fixed before any "
            "replicate (Task 3 review and Task 4 execution memo, Section 3)",
            "1. Arms: V3 and raw-RF only (was 4): the power curve is for that comparison.",
            f"2. Participants: {cfg['COHORT']['participants']} per cohort with the actual v1.4 lengths "
            f"{cfg['COHORT']['lengths_v1_4']} (was 15 with resampled lengths).",
            "3. Signal levels: 3 per condition at oracle rho 0.15, 0.30, 0.45 (was 4), plus the pre-specified C1 "
            f"extension at 0.60 ({ext['status']}).",
            f"4. Replicates: {cfg['REPLICATES_PER_CELL']} per cell (target was 200, minimum 100); machine: "
            f"{cfg['FULL_RUN_MACHINE']}; replicate files written on: {', '.join(hosts or [])}.",
            "5. Outcome alignment: next-day, through Task 3's next_day code (target support {s+1}); the purge removed "
            "no training row in any replicate.",
            "6. Decision rule: one comparison, so BH is moot; power reported at P < .05 and P < .0125.",
            "7. Interpretations fixed in the config: v_t = the expanding volatility F reads (memo: 'causal rolling "
            "volatility'); z over the cohort's participant-days; undefined signal days = 0; eps ~ N(0, 1); oracle "
            "rho = mean per-participant Spearman on the scored days; direction = larger signed-rank sum; extension "
            "trigger evaluated at P < .0125.",
            "8. Fold generator: get_folds replaced by an equivalent generator (no forest fitted per fold); fold check "
            "and fidelity check passed.",
            "9. Secondary metric added before the tag (Stage 0 review memo, Section 3): anomaly rho = Spearman(f - a, y - a), "
            "a = the fold's training mean of the target; power and MDE reported for it from the same predictions. "
            "Specified after Stage 0, before any replicate.",
            "Carried from the Task 3 review: on 6 dates two sleep scores exist (naps); CELL 5 keeps the later score."]


def handin(out, A, cfg, cells, data, prov_bad, crn_ok, det, ext, tag_info):
    R = cfg["REPLICATES_PER_CELL"]
    lines, fails = [], []

    def check(name, ok):
        lines.append(f"   {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            fails.append(name)
    lines.append("TASK 4 HAND-IN CHECK")
    check(f"every cell complete ({', '.join(cells)}: {R} replicates each)", all(len(data[c]) == R for c in cells))
    check(f"extension decided from C1b3 ({ext['status']})", ext["status"] in ("triggered", "not triggered"))
    check("every replicate: config and script equal the tagged ones" + (", tag commit recorded" if tag_info else ""),
          not prov_bad)
    check("every replicate: purge removed no training row", all(x["purged_training_rows"] == 0 for c in cells for x in data[c]))
    check(f"every replicate: {cfg['COHORT']['participants']} complete pairs (primary and secondary metric)",
          all(x["n_pairs"] == cfg["COHORT"]["participants"] and x.get("anom_n_pairs") == cfg["COHORT"]["participants"]
              for c in cells for x in data[c]))
    check("common random numbers: same cohort and eps for each replicate across cells", crn_ok)
    vers = {json.dumps(x.get("versions"), sort_keys=True) for c in cells for x in data[c]}
    check(f"package versions equal the config's in every replicate ({len(vers)} distinct)",
          vers == {json.dumps(cfg["VERSIONS"], sort_keys=True)})
    for name in ("fold_check.json", "isolation_check.json", "fidelity_check.json"):
        p = os.path.join(out, "checks", name)
        ok = os.path.isfile(p) and json.load(open(p)).get("verdict") == "PASS"
        check(f"checks/{name} PASS", ok)
    pil = sum(len(load_cell(out, c, pilot=True)) for c in PILOT_CELLS)
    check(f"pilot ({pil} replicates) present", pil == len(PILOT_CELLS) * len(PILOT_REPS))
    check(f"determinism: {int(det['identical'].sum()) if len(det) else 0} of {len(det)} re-computed replicates identical"
          + (f" (re-runs on: {sorted(set(det['host_a']))})" if len(det) else ""), len(det) > 0 and bool(det["identical"].all()))
    files = ["task4_per_cell.csv", "task4_per_cell_formatted.csv", "task4_per_replicate.csv", "task4_per_participant.csv",
             "task4_mde.json", "figure5_power_curves.png", "figure5_power_curves.pdf", "task4_reading.txt",
             "task4_determinism_check.csv", "task4_deviation_note.txt", "task4_per_cell_secondary_anomaly.csv",
             "task4_per_cell_secondary_anomaly_formatted.csv", "task4_mde_secondary_anomaly.json",
             "figure5_secondary_anomaly.png", "figure5_secondary_anomaly.pdf", "figure5_caption.txt"]
    miss = [f for f in files if not os.path.isfile(os.path.join(A, f))]
    check(f"{len(files) - len(miss)} of {len(files)} analysis files present" + (f"; MISSING {miss}" if miss else ""), not miss)
    lines.append("TASK 4 HAND-IN CHECK " + ("PASSED" if not fails else f"FAILED: {fails}"))
    open(os.path.join(A, "handin_check_task4.txt"), "w").write("\n".join(lines) + "\n")
    return lines


def figure5(c, mde, A, fname, reference_lines=True, xlabel="Cell mean delta rho (V3 - raw-feature RF)"):
    """No figure number or title inside the image (Stage 0 review memo, Section 4); caption in figure5_caption.txt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    blue, orange, grey = "#1f4e79", "#b85c00", "#6b6b6b"
    lo_x = min(-0.05, float(c["mean_delta_rho"].min()) - 0.03)
    hi_x = max(0.25, float(c["mean_delta_rho"].max()) + 0.03,
               (max(REAL_UPPER_BOUNDS.values()) + 0.03) if reference_lines else 0.0)
    xs = np.linspace(lo_x, hi_x, 300)
    s = c.set_index("cell")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
    for ax, (k, lab) in zip(axes, (("p05", "A. P < .05 (optimistic)"), ("p0125", "B. P < .0125 (conservative)"))):
        def eb(d, col, fmt, ms=6, **kw):
            ax.errorbar(d["mean_delta_rho"], d[f"power_{k}"], yerr=np.clip([d[f"power_{k}"] - d[f"power_{k}_lo"],
                        d[f"power_{k}_hi"] - d[f"power_{k}"]], 0, None), fmt=fmt, color=col, ms=ms, capsize=3, lw=1, **kw)
        c1 = c[c["condition"] == "C1"].sort_values("mean_delta_rho")
        c2 = c[c["condition"] == "C2"].sort_values("mean_delta_rho")
        eb(c1, blue, "o")
        eb(c2, orange, "^-")
        c0 = s.loc["C0"]
        ax.errorbar([c0["mean_delta_rho"]], [c0[f"sig_{k}"]], yerr=np.clip([[c0[f"sig_{k}"] - c0[f"sig_{k}_lo"]],
                    [c0[f"sig_{k}_hi"] - c0[f"sig_{k}"]]], 0, None), fmt="*", ms=12, color="black", capsize=3, lw=1)
        m = mde[k]
        ax.plot(xs, norm.cdf(m["probit_a"] + m["probit_b"] * xs), color=blue, lw=1, alpha=0.6)
        mv = m["mde_delta_rho"]
        if mv is not None and np.isfinite(mv):
            ax.axvline(mv, color=blue, ls="--", lw=1)
            ax.text(mv, 0.02, f" MDE {mv:.3f}", color=blue, fontsize=8, ha="left", va="bottom")
        else:
            ax.text(0.02, 0.97, m["statement"] or "", transform=ax.transAxes, fontsize=7.5, color=blue, va="top")
        if reference_lines:
            for v, ls in zip(REAL_UPPER_BOUNDS.values(), ("-.", ":")):
                ax.axvline(v, color=grey, ls=ls, lw=1)
        ax.axhline(0.80, color="grey", ls=":", lw=0.7)
        ax.axhline(0.05, color="grey", ls=":", lw=0.7)
        ax.set_xlim(lo_x, hi_x); ax.set_ylim(-0.02, 1.02)
        ax.set_title(lab, fontsize=10, loc="left")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Proportion of replicates significant")
    names = list(REAL_UPPER_BOUNDS)
    handles = [Line2D([], [], color=blue, marker="o", ls="-", lw=1, label="C1 index-favouring: V3 significant (probit fit)"),
               Line2D([], [], color=orange, marker="^", ls="-", lw=1, label="C2 raw-favouring: raw-RF significant"),
               Line2D([], [], color="black", marker="*", ms=10, ls="", label="C0 null: false-positive rate (either direction)"),
               Line2D([], [], color=blue, ls="--", lw=1, label="MDE (80% crossing)")]
    if reference_lines:
        handles += [Line2D([], [], color=grey, ls="-.", lw=1, label=f"upper 95% bound, {names[0]}: {REAL_UPPER_BOUNDS[names[0]]:.3f}"),
                    Line2D([], [], color=grey, ls=":", lw=1, label=f"upper 95% bound, {names[1]}: {REAL_UPPER_BOUNDS[names[1]]:.3f}")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(os.path.join(A, f"{fname}.png"), dpi=600)
    fig.savefig(os.path.join(A, f"{fname}.pdf"))
    plt.close(fig)


def pilot_table(out, cfg=None):
    rows = []
    for cell in PILOT_CELLS:
        for r, x in sorted(load_cell(out, cell, pilot=True).items()):
            rows.append(dict(cell=cell, rep=r, beta=x["beta"], oracle_rho=x["oracle_rho_mean"],
                             mean_rho_v3=x["mean_rho_v3"], mean_rho_rawrf=x["mean_rho_rawrf"],
                             mean_delta_rho=x["mean_delta_rho"], wins_v3=f"{x['wins_v3']}/{x['n_pairs']}", p=x["p"],
                             direction=x["direction"], sig_p05=x["sig_p05"], sig_p0125=x["sig_p0125"],
                             anom_mean_delta_rho=x["anom_mean_delta_rho"], anom_p=x["anom_p"],
                             seconds=x["seconds"], host=x.get("host")))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------------------------------------------------
def _common_setup(a, need_config):
    repo = os.path.abspath(a.repo or os.path.dirname(script_path()))
    ndi_path = a.ndi or os.path.join(repo, "ndi_pipeline.py")
    assert os.path.isfile(ndi_path), f"ndi_pipeline.py not found: {ndi_path} (use --ndi)"
    cfg = None
    if need_config:
        cfg, _ = load_config(repo)
        lengths = list(cfg["COHORT"]["lengths_v1_4"].values())
    else:
        lengths = [int(x) for x in a.lengths.split(",")] if a.lengths else None
        assert lengths, "--lengths is required before the config exists (the 16 v1.4 lengths, p01..p16)"
    info = setup(ndi_path=ndi_path, repo=repo, lengths=lengths)
    if cfg is not None:
        check_code_against_config(cfg)
    return repo, cfg, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "calibrate", "bench", "run", "verify", "merge"])
    ap.add_argument("--repo", default=None)
    ap.add_argument("--ndi", default=None)
    ap.add_argument("--out", default="results_task4")
    ap.add_argument("--lengths", default=None, help="check/calibrate before the config exists: 16 comma-separated lengths")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--cells", default="all")
    ap.add_argument("--reps", default="all")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args(argv)
    mi = machine_info()
    workers = a.workers or (mi["physical_cores"] or mi["logical_cores_available"])
    print(json.dumps(dict(command=a.command, started=utcnow(), **mi, versions=versions(), workers=workers)), flush=True)
    need_cfg = a.command in ("run", "verify", "merge") or (a.command in ("check", "calibrate", "bench") and not a.lengths)
    repo, cfg, info = _common_setup(a, need_cfg)
    os.makedirs(a.out, exist_ok=True)
    if a.command == "check":
        fc = fold_check(); print(json.dumps({k: v for k, v in fc.items() if k != "rows"}), flush=True)
        ic = isolation_check(); print(json.dumps(ic), flush=True)
        R = cfg["REPLICATES_PER_CELL"] if cfg else 100
        print(json.dumps(projection(float(np.median([ic["seconds_alone_while_other_process_busy"]] + ic["seconds_b_then_a"][:1])),
                                    workers, R)), flush=True)
        assert fc["verdict"] == "PASS" and ic["verdict"] == "PASS", "CHECK FAILED -- STOP"
        return
    if a.command == "calibrate":
        cal, tab = calibrate(); print(json.dumps(cal, indent=1)); return
    if a.command == "bench":
        b = bench(workers)
        R = cfg["REPLICATES_PER_CELL"] if cfg else 100
        res = dict(b, **machine_info(), versions=versions(), measured=utcnow(),
                   projection=projection(b["effective_seconds_per_cohort"] * workers, workers, R))
        p = os.path.join(a.out, "checks", f"benchmark_{platform.node() or 'machine'}.json")
        os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(res, open(p, "w"), indent=2)
        print(json.dumps(res, indent=1)); return
    tag_info = verify_tagged(repo, cfg)
    meta = dict(config_sha256=canonical_sha(cfg), script_sha256=cfg["TASK4_SCRIPT_SHA256"], **tag_info,
                code_hashes=STATE["code_hashes"])
    R = cfg["REPLICATES_PER_CELL"]
    if a.command == "run":
        if a.pilot:
            jobs = [(c, r, *cell_spec(cfg, c), rep_path(a.out, c, r, pilot=True), ) for c in PILOT_CELLS for r in PILOT_REPS]
            run_jobs(jobs, workers, meta, a.hours)
            print(pilot_table(a.out).round(4).to_string(index=False))
            return
        reps = parse_reps(a.reps, R)
        cells = list(MAIN_CELLS) if a.cells == "all" else [c for c in a.cells.split(",") if c != EXTENSION_CELL]
        bad = [c for c in cells if c not in cfg["CELLS"]]
        assert not bad, f"unknown cells {bad}"
        cells = [c for c in MAIN_CELLS if c in cells]
        jobs = [(c, r, *cell_spec(cfg, c), rep_path(a.out, c, r)) for c in cells for r in reps]
        start = time.time()
        run_jobs(jobs, workers, meta, a.hours)
        if a.cells == "all" or EXTENSION_CELL in a.cells.split(","):
            ext = extension_status(a.out, cfg)
            print(f"EXTENSION {EXTENSION_CELL}: {ext['status']} -- {ext['reason']}", flush=True)
            if ext["status"] == "triggered":
                left = None if a.hours is None else a.hours - (time.time() - start) / 3600
                if left is None or left > 0:
                    jobs = [(EXTENSION_CELL, r, *cell_spec(cfg, EXTENSION_CELL), rep_path(a.out, EXTENSION_CELL, r)) for r in reps]
                    run_jobs(jobs, workers, meta, left)
        analyse(a.out, cfg, tag_info)
        return
    if a.command == "verify":
        cands = [("C1b3", 10), ("C2b3", 11)]
        jobs, refs = [], {}
        for c, r in cands:
            ref = load_cell(a.out, c).get(r)
            if ref is None:
                print(f"verify: {c} r{r} not in the results yet"); continue
            jobs.append((c, r, *cell_spec(cfg, c), os.path.join(a.out, "determinism", c, f"rep_{r:03d}.json")))
            refs[(c, r)] = ref
        for j in jobs:
            if os.path.isfile(j[4]):
                os.remove(j[4])
        run_jobs(jobs, max(1, min(workers, len(jobs))), meta)
        for (c, r), ref in refs.items():
            x = json.load(open(os.path.join(a.out, "determinism", c, f"rep_{r:03d}.json")))
            print(f"verify {c} r{r}: identical = {_same_record(x, ref)} (here: {x.get('host')}; file: {ref.get('host')})")
        return
    if a.command == "merge":
        analyse(a.out, cfg, tag_info)


if __name__ == "__main__":
    main()
