"""
Treatment-effect evaluation and policy-development pipeline for observational
data with a binary treatment.

The pipeline is dataset-agnostic: everything use-case specific (data file,
treatment/outcome columns, covariates, problem and data descriptions, model
hyperparameters) comes from a JSON configuration file.

  Phase 1  train every candidate model on D_tr, evaluate it on D_ev, select M*
  Phase 2  re-estimate all policy quantities for M* on D_po (no re-fitting)

and, optionally, the LLM interpretive layer for both phases.

Usage
-----
    python main.py --config examples/rhc/config.json
    python main.py --config my_study/config.json --skip-dragonnet
    python main.py --config my_study/config.json --llm anthropic

From Python:

    from main import load_config, run_pipeline
    cfg = load_config("my_study/config.json")
    out = run_pipeline(cfg, df=my_dataframe)   # df optional; else cfg["data"]

See README.md for the configuration reference.
"""

import argparse
import copy
import json
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")  # headless: figures are saved, not shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler

from models.cxgboost import CXGBoost
from models.cxgboost_treg import CXGBoostTreg
from models.s_learner import S_Learner
from models.t_learner import T_Learner
from utils.evaluator import CausalEvaluator
from utils.prompts import build_evaluation_prompt, build_policy_prompt


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Default hyperparameters of the candidate models. A config's "models" entry
# overrides them per model (keys are merged into these defaults).
DEFAULT_MODELS = {
    'S-LR': {},
    'T-LR': {},
    'R-Forest': {'n_estimators': 150, 'max_depth': 6, 'min_samples_split': 4,
                 'random_state': 42},
    'S-kNN': {'n_neighbors': 7, 'metric': 'l2'},
    'S-GBM': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05,
              'random_state': 42},
    'Dragonnet': {},
    'C-XGBoost (treg)': {'n_estimators': 150, 'max_depth': 5, 'min_child_weight': 1,
                         'max_leaves': 4, 'learning_rate': 0.05, 'reg_alpha': 3.0,
                         'tree_method': 'hist', 'epsilon': 0.001},
    'C-XGBoost': {'n_estimators': 200, 'max_depth': 4, 'min_child_weight': 2,
                  'max_leaves': 4, 'learning_rate': 0.1, 'tree_method': 'hist'},
}

CONFIG_DEFAULTS = {
    "name": "Treatment Effect",
    "covariates": None,          # None -> every column except treatment/outcome/exclude
    "exclude": [],
    "categorical": "auto",       # "auto" -> non-numeric covariates
    "actionable": True,          # structural judgement passed to the policy prompt
    # stratify: "auto" (outcome if binary, else none), "outcome", "treatment", null
    "split": {"eval_size": 0.2, "policy_size": 0.2, "seed": 1983,
              "stratify": "auto"},
    "n_bootstrap": 100,
    "evaluator": {},             # extra CausalEvaluator keyword arguments
    "models": None,              # None -> all models in DEFAULT_MODELS
}

# Placeholders that may appear in the description texts; they are filled
# from the data so the descriptions never go stale.
PLACEHOLDERS = ("{n}", "{n_covariates}", "{treatment_rate}", "{outcome_mean}")


def _read_text(value, base_dir):
    """A description is either inline text or a path to a text/markdown file."""
    if value is None:
        return ""
    path = value if os.path.isabs(value) else os.path.join(base_dir, value)
    if len(value) < 260 and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return value


def load_config(path):
    """Load a JSON config; relative paths inside it resolve against its folder."""
    with open(path, encoding="utf-8") as f:
        user = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(path))
    cfg = copy.deepcopy(CONFIG_DEFAULTS)
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    cfg["_base_dir"] = base_dir
    for key in ("treatment", "outcome"):
        if key not in cfg:
            raise ValueError(f"config is missing the required '{key}' section")
    cfg["problem_description"] = _read_text(cfg.get("problem_description"), base_dir)
    cfg["data_description"] = _read_text(cfg.get("data_description"), base_dir)
    data = cfg.get("data")
    if data and "://" not in data and not os.path.isabs(data):
        cfg["data"] = os.path.join(base_dir, data)
    return cfg


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def read_table(source):
    """Read a CSV / TSV / Parquet / Excel file (local path or URL)."""
    ext = os.path.splitext(source.split("?")[0])[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(source)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(source)
    if ext in (".tsv", ".tab"):
        return pd.read_csv(source, sep="\t")
    return pd.read_csv(source)


def _binary(series, spec, what):
    """Map a column to 0/1 using spec['treated_value' | 'positive_value']."""
    key = "treated_value" if what == "treatment" else "positive_value"
    if key in spec:
        return (series == spec[key]).astype(int).values
    values = set(pd.unique(series.dropna()))
    if values <= {0, 1, True, False}:
        return series.astype(int).values
    raise ValueError(f"{what} column '{spec['column']}' is not 0/1 "
                     f"(values: {sorted(map(str, values))[:10]}); "
                     f"set '{key}' in the config")


def prepare_data(df, cfg):
    """Build X, T, Y from a DataFrame according to the config."""
    t_spec, y_spec = cfg["treatment"], cfg["outcome"]
    t_col, y_col = t_spec["column"], y_spec["column"]
    for col in (t_col, y_col):
        if col not in df.columns:
            raise ValueError(f"column '{col}' not found in the data")

    covariates = cfg["covariates"]
    if covariates is None:
        drop = {t_col, y_col, *cfg["exclude"]}
        covariates = [c for c in df.columns if c not in drop]
    missing = [c for c in covariates if c not in df.columns]
    if missing:
        raise ValueError(f"covariates not found in the data: {missing}")

    X = df[covariates].copy()
    categorical = cfg["categorical"]
    if categorical == "auto":
        categorical = [c for c in covariates
                       if not pd.api.types.is_numeric_dtype(X[c])]
    for col in categorical:
        # Integer codes in sorted order of the observed values (as LabelEncoder).
        X[col] = pd.Categorical(X[col].astype(str)).codes

    na = X.columns[X.isna().any()].tolist()
    if na or df[[t_col, y_col]].isna().any().any():
        raise ValueError(f"missing values in {na or [t_col, y_col]}; impute or "
                         "drop them before running the pipeline")

    T = _binary(df[t_col], t_spec, "treatment")
    if y_spec.get("type", "binary") == "binary":
        Y = _binary(df[y_col], y_spec, "outcome")
    else:
        Y = pd.to_numeric(df[y_col]).astype(float).values
    return X.astype(float), T, Y, covariates


def split_data(X, T, Y, cfg):
    """Disjoint training / evaluation / policy partitions."""
    s = cfg["split"]
    held_out = s["eval_size"] + s["policy_size"]
    mode = s.get("stratify")
    if mode == "auto":
        mode = "outcome" if cfg["outcome"].get("type", "binary") == "binary" else None
    if mode not in ("outcome", "treatment", None):
        raise ValueError("split.stratify must be 'auto', 'outcome', 'treatment' or null")
    strat = {"outcome": Y, "treatment": T}.get(mode)
    Xtr, Xte, Ttr, Tte, Ytr, Yte = train_test_split(
        X, T, Y, test_size=held_out, random_state=s["seed"], stratify=strat)
    strat = {"outcome": Yte, "treatment": Tte}.get(mode)
    Xev, Xpo, Tev, Tpo, Yev, Ypo = train_test_split(
        Xte, Tte, Yte, test_size=s["policy_size"] / held_out,
        random_state=s["seed"], stratify=strat)
    return {"train": (Xtr, Ttr, Ytr), "eval": (Xev, Tev, Yev),
            "policy": (Xpo, Tpo, Ypo)}


def fill_placeholders(text, X, T, Y):
    values = {"{n}": f"{len(Y)}", "{n_covariates}": f"{X.shape[1]}",
              "{treatment_rate}": f"{100 * T.mean():.1f}%",
              "{outcome_mean}": (f"{100 * Y.mean():.1f}%"
                                 if set(np.unique(Y)) <= {0, 1}
                                 else f"{Y.mean():.3f}")}
    for k in PLACEHOLDERS:
        text = text.replace(k, values[k])
    return text


def policy_value(y, t, g, pi):
    """Self-normalised (Hajek) IPW value of policy pi."""
    e = np.where(pi == 1, g, 1 - g)
    w = (t == pi) / e
    return (w * y).sum() / w.sum()


# --------------------------------------------------------------------------
# Candidate models
# --------------------------------------------------------------------------

def build_model(name, params):
    p = dict(params)
    if name == 'S-LR':
        return S_Learner(base_learner=LinearRegression(**p))
    if name == 'T-LR':
        return T_Learner(base_learner=LinearRegression(**p))
    if name == 'R-Forest':
        return S_Learner(base_learner=RandomForestRegressor(n_jobs=-1, **p))
    if name == 'S-kNN':
        return S_Learner(base_learner=KNeighborsRegressor(**p))
    if name == 'S-GBM':
        return S_Learner(base_learner=GradientBoostingRegressor(**p))
    if name == 'Dragonnet':
        from models.dragonnet import Dragonnet  # needs TensorFlow
        return Dragonnet()
    if name == 'C-XGBoost (treg)':
        kw = {k: p.pop(k) for k in ("alpha", "beta", "epsilon") if k in p}
        return CXGBoostTreg(parameters={'n_jobs': -1, **p}, **kw)
    if name == 'C-XGBoost':
        task = p.pop("task", "multiclass")
        return CXGBoost(parameters={'n_jobs': -1, **p}, task=task)
    raise ValueError(f"unknown model '{name}'; choose from {list(DEFAULT_MODELS)}")


def resolve_models(cfg, skip_dragonnet=False):
    """{name: params} from the config's "models" (list of names or dict)."""
    spec = cfg["models"]
    if spec is None:
        spec = {k: {} for k in DEFAULT_MODELS}
    elif isinstance(spec, list):
        spec = {k: {} for k in spec}
    out = {}
    for name, override in spec.items():
        if name not in DEFAULT_MODELS:
            raise ValueError(f"unknown model '{name}'; choose from {list(DEFAULT_MODELS)}")
        if skip_dragonnet and name == 'Dragonnet':
            continue
        out[name] = {**DEFAULT_MODELS[name], **(override or {})}
    return out


def fit_model(model, name, params, X, T, Y):
    if isinstance(model, (CXGBoost, CXGBoostTreg)):
        model.fit(X=X, t=T, y=Y)
    elif name == 'Dragonnet':
        model.fit(X=X, T=T, Y=Y, parameters=params)
    else:
        model.fit(X=X, T=T, Y=Y)


# --------------------------------------------------------------------------
# LLM interpretive layer
# --------------------------------------------------------------------------

DEFAULT_LLM_MODEL = {"openai": "gpt-4.1", "anthropic": "claude-sonnet-5-5"}


def ask_llm(provider, model, prompt):
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # re-run on a fallback model if declined
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Request declined: {response.stop_details}")
        return "".join(b.text for b in response.content if b.type == "text")

    raise ValueError(f"Unknown LLM provider: {provider}")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def save_histogram(pred, name, out_dir):
    plt.figure(figsize=(4, 1.5))
    plt.hist(pred['y_0_hat'], label='y_0_hat')
    plt.hist(pred['y_1_hat'], alpha=0.7, label='y_1_hat')
    plt.title(name, fontsize=8)
    plt.legend(fontsize=6)
    plt.tight_layout()
    fname = name.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(os.path.join(out_dir, f"hist_{fname}.png"), dpi=120)
    plt.close()


def extract_row(results):
    return {
        'poor_factual_fit':     results.get('poor_factual_fit'),
        'degenerate_policy':    results.get('degenerate_policy'),
        'policy_value':         results.get('policy_value'),
        'policy_value_ci_lower': results.get('policy_value_ci_lower'),
        'policy_value_ci_upper': results.get('policy_value_ci_upper'),
        'rmse_t0':              results.get('rmse_t0'),
        'rmse_t1':              results.get('rmse_t1'),
        'rmse_ratio_t0':        results.get('rmse_ratio_t0'),
        'rmse_ratio_t1':        results.get('rmse_ratio_t1'),
        'rate':                 results.get('rate'),
        'rate_p_value':         results.get('rate_p_value'),
        'rate_significant':     results.get('rate_significant'),
        'max_smd_weighted':     results.get('balance', {}).get('max_smd_weighted'),
        'dr_policy_value':      results.get('dr_policy_value', {}).get('dr_policy_value'),
        'ipw_dr_disagreement_sd_units':
            results.get('dr_policy_value', {}).get('ipw_dr_disagreement_sd_units'),
        'evalue_point':         results.get('evalue', {}).get('evalue_point'),
        'evalue_ci_limit':      results.get('evalue', {}).get('evalue_ci_limit'),
    }


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _fmt(v, spec="+.3f"):
    return "n/a" if v is None else format(v, spec)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def run_pipeline(cfg, df=None, output_dir="results", skip_dragonnet=False,
                 llm="none", llm_model=None):
    """
    Run Phase 1 and Phase 2 (and optionally the LLM reports).

    Parameters
    ----------
    cfg : dict
        Configuration, as returned by :func:`load_config`.
    df : pandas.DataFrame, optional
        The data. When omitted it is read from ``cfg["data"]``.

    Returns
    -------
    dict with keys ``evaluation`` (per-model Phase 1 metrics), ``selection``,
    ``selected_model``, ``policy`` (Phase 2 metrics) and ``models`` (fitted).
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if df is None:
        if not cfg.get("data"):
            raise ValueError("no data: pass df= or set 'data' in the config")
        df = read_table(cfg["data"])

    y_spec = cfg["outcome"]
    evaluator = CausalEvaluator(problem_name=cfg["name"],
                                outcome_type=y_spec.get("type", "binary"),
                                outcome_direction=y_spec.get("direction", "adverse"),
                                n_bootstrap=cfg["n_bootstrap"],
                                **cfg["evaluator"])

    # ---- Data -------------------------------------------------------------
    print("\n### Data")
    X, T, Y, covariates = prepare_data(df, cfg)
    print(f"X shape: {X.shape}")
    print(f"Treatment rate: {T.mean():.3f}")
    print(f"Outcome mean: {Y.mean():.3f}")
    problem_description = fill_placeholders(cfg["problem_description"], X, T, Y)
    data_description = fill_placeholders(cfg["data_description"], X, T, Y)

    splits = split_data(X.values, T, Y, cfg)
    trainX, trainT, trainY = splits["train"]
    testX, testT, testY = splits["eval"]
    policyX, policyT, policyY = splits["policy"]
    # Scaled copies with column names: used for the balance diagnostics, so
    # the reports name the worst-balanced covariates.
    scaler = RobustScaler().fit(trainX)
    testX_named = pd.DataFrame(scaler.transform(testX), columns=covariates)
    policyX_named = pd.DataFrame(scaler.transform(policyX), columns=covariates)
    print(f"Split sizes (train/eval/policy): "
          f"{len(trainY)}/{len(testY)}/{len(policyY)}")

    # ---- Phase 1: train on D_tr, evaluate on D_ev --------------------------
    model_specs = resolve_models(cfg, skip_dragonnet=skip_dragonnet)
    fitted, d = {}, {}
    for name, params in model_specs.items():
        print(f"\n### {name}")
        start = time.time()
        model = build_model(name, params)
        fit_model(model, name, params, trainX, trainT, trainY)
        fitted[name] = model
        pred = model.predict(testX)
        d[name] = evaluator.evaluation_metrics(
            y_pred_0=pred['y_0_hat'],
            y_pred_1=pred['y_1_hat'],
            propensity=pred['propensity_score'],
            X=testX_named, Y=testY, T=testT,
            random_state=42,
        )
        save_histogram(pred, name, plots_dir)
        r = d[name]
        print(f"Fit + evaluation time: {time.time() - start:.1f}s")
        print("Poor factual fit: ", r['poor_factual_fit'])
        print("Degenerate policy:", r['degenerate_policy'])
        print("Policy value:", round(r['policy_value'], 3))
        print(f"RMSE_0: {round(r['rmse_t0'], 3)} | RMSE_1: {round(r['rmse_t1'], 3)}")

    # Constant reference rules, using the common Random-Forest propensity
    # (same configuration as the learners' default propensity model).
    g_ref = RandomForestClassifier(n_estimators=100, max_depth=3,
                                   class_weight='balanced', n_jobs=-1,
                                   random_state=42).fit(trainX, trainT)
    g_ref = g_ref.predict_proba(testX)[:, 1]
    references = {
        "treat_all":        policy_value(testY, testT, g_ref, np.ones_like(testT)),
        "treat_none":       policy_value(testY, testT, g_ref, np.zeros_like(testT)),
        "current_practice": policy_value(testY, testT, g_ref, testT),
    }
    print("\n### Reference policies (evaluation split)")
    for k, v in references.items():
        print(f"{k:17s}: {v:.3f}")

    print("\n### Evaluation/Analysis")
    table = pd.DataFrame({k: extract_row(v) for k, v in d.items()}).T
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(table)
    table.to_csv(os.path.join(output_dir, "phase1_evaluation.csv"))
    with open(os.path.join(output_dir, "phase1_evaluation.json"), "w") as f:
        json.dump({"models": d, "reference_policies": references}, f, indent=2)

    selection = CausalEvaluator.select_best_model(d)
    best = selection["selected_model"]
    print("\nSelection status:", selection["selection_status"])
    print("Selected model M*:", best, "| runner-up:", selection["runner_up"])
    print("RATE agrees with policy value:", selection["rate_agrees_with_policy_value"])
    if best is None:
        raise SystemExit("No eligible model; Phase 2 cannot run.")

    # ---- Phase 1 LLM report -------------------------------------------------
    prompt = build_evaluation_prompt(problem_description=problem_description,
                                     data_description=data_description,
                                     model_results=d)
    _write(os.path.join(output_dir, "phase1_prompt.txt"), prompt)
    if llm != "none":
        print(f"\n### Model evaluation analysis ({llm}: {llm_model})")
        report = ask_llm(llm, llm_model, prompt)
        _write(os.path.join(output_dir, "phase1_report.md"), report)
        print(report)

    # ---- Phase 2: policy development with M* on D_po ------------------------
    print(f"\n### Policy making analysis ({best})")
    pred = fitted[best].predict(policyX)
    results = evaluator.policy_metrics(
        y_pred_0=pred['y_0_hat'],
        y_pred_1=pred['y_1_hat'],
        propensity=pred['propensity_score'],
        X=policyX_named, Y=policyY, T=policyT,
        random_state=42,
    )
    with open(os.path.join(output_dir, "phase2_policy.json"), "w") as f:
        json.dump({"selected_model": best, **results}, f, indent=2)

    print(f"ATE: {_fmt(results['ate'])} "
          f"[{_fmt(results['ate_ci_lower'])}, {_fmt(results['ate_ci_upper'])}]")
    print(f"RATE: {_fmt(results['rate'])} (p = {_fmt(results['rate_p_value'], '.3f')})")
    print(f"CATE high: {_fmt(results['cate_high'].get('cate'))} | "
          f"CATE low: {_fmt(results['cate_low'].get('cate'))}")
    print(f"Delta r_treat: {_fmt(results['delta_r_treat'])} "
          f"({results['current_treat_rate']:.3f} -> "
          f"{results['model_policy_treat_rate']:.3f})")
    print(f"Delta V (IPW): {_fmt(results['delta_v'])} "
          f"[{_fmt(results['delta_v_ci_lower'])}, {_fmt(results['delta_v_ci_upper'])}]"
          f" | Delta V (DR): {_fmt(results['delta_v_dr'])}")
    print(f"max SMD_w: {_fmt(results['balance']['max_smd_weighted'], '.3f')} | "
          f"E-value: {results['evalue'].get('evalue_point')} / "
          f"{results['evalue'].get('evalue_ci_limit')}")
    print("Actionability gate:", results['actionability_gate'],
          results['actionability_gate_components'])

    # ---- Phase 2 LLM report -------------------------------------------------
    prompt = build_policy_prompt(problem_description=problem_description,
                                 model_results=results,
                                 alpha=int(bool(cfg["actionable"])))
    _write(os.path.join(output_dir, "phase2_prompt.txt"), prompt)
    if llm != "none":
        print(f"\n### Policy report ({llm}: {llm_model})")
        report = ask_llm(llm, llm_model, prompt)
        _write(os.path.join(output_dir, "phase2_report.md"), report)
        print(report)

    print(f"\nAll outputs written to: {os.path.abspath(output_dir)}")
    return {"evaluation": d, "selection": selection, "selected_model": best,
            "policy": results, "models": fitted}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="path to the JSON config")
    ap.add_argument("--data", default=None,
                    help="data file or URL; overrides 'data' in the config")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--n-bootstrap", type=int, default=None,
                    help="bootstrap replications; overrides the config")
    ap.add_argument("--skip-dragonnet", action="store_true",
                    help="skip Dragonnet (removes the TensorFlow dependency)")
    ap.add_argument("--llm", choices=["none", "openai", "anthropic"], default="none",
                    help="provider for the LLM interpretive layer (default: none; "
                         "prompts are always saved)")
    ap.add_argument("--llm-model", default=None,
                    help="LLM model id (default: gpt-4.1 / claude-sonnet-5-5)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data:
        cfg["data"] = args.data
    if args.n_bootstrap is not None:
        cfg["n_bootstrap"] = args.n_bootstrap

    if args.llm != "none":
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

    run_pipeline(cfg, output_dir=args.output_dir,
                 skip_dragonnet=args.skip_dragonnet, llm=args.llm,
                 llm_model=args.llm_model or DEFAULT_LLM_MODEL.get(args.llm))


if __name__ == "__main__":
    main()
