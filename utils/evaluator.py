"""
evaluation.py
=============

Two-phase evaluation of treatment effect estimation models on observational
data (no counterfactual ground truth).

    evaluation_metrics(...)   evaluation split, one call per candidate model
    select_best_model(...)    applies the selection hierarchy, returns M*
    policy_metrics(...)       policy split, one call for M* (never re-fitted)

The three partitions (training, evaluation, policy) are disjoint. Nothing is
carried over numerically between phases: every Phase 2 quantity, including the
diagnostics that qualify it, is re-estimated on the policy partition, because
the evaluation-partition estimates describe a different sample. Only M* itself
is carried forward, fixed.

Both exporters return exactly the fields the interpretive layer consumes.
Pass ``full=True`` for additional diagnostic detail.

Benefit-score convention
------------------------
Treatment effects live on the outcome scale, so the sign of tau_hat alone does
not say who benefits: for an adverse outcome (larger Y is worse) benefit means
tau_hat < 0. Every decision quantity is therefore expressed through the
treatment-benefit score

    b_hat(x) = s_dir * tau_hat(x),   s_dir = +1 (beneficial Y) / -1 (adverse Y)

so that b_hat > 0 always means "expected to benefit from treatment". The
model-induced policy is pi_M = 1{b_hat > 0}, the RATE priority score is b_hat
with the sign-standardised pseudo-outcome s_dir * psi (so a POSITIVE RATE always
means better benefit ranking), and the high/low-benefit subgroups are the upper
and lower quantiles of b_hat. Subgroup CATEs are still reported on the native
outcome scale, so their sign must be read against outcome_direction.

Selection hierarchy
-------------------
1. Necessary filter   factual RMSE, scale-free: excluded when RMSE exceeds the
                      within-arm outcome SD in BOTH arms.
2. Primary criterion  IPW policy value: decision utility at the deployed
                      threshold pi_M = 1{b_hat > 0}.
3. Second criterion   RATE (AUTOC by default, Qini optional): quality of the
                      ranking the model induces, independent of any threshold.
4. Post-hoc on M*     covariate balance, DR cross-check, E-value.

Estimators
----------
* Policy value is self-normalised (Hajek), sum(w y)/sum(w). The un-normalised
  Horvitz-Thompson form does not live on the outcome scale, so V_HT - mean(Y)
  is not a meaningful contrast. Hajek is consistent but carries a small
  finite-sample bias; HT is exported under full=True.
* ATE and subgroup CATEs are AIPW with influence-function standard errors. The
  plug-in mean(mu1_hat - mu0_hat) never contacts the observed outcomes and
  admits no valid resampling interval; exported under full=True as a comparator.
* RATE uses AIPW scores and half-sample bootstrap inference, the resampling
  scheme its central limit theorem justifies.
* delta_v and its interval are bootstrapped jointly, not obtained by subtracting
  mean(Y) from the policy-value interval endpoints.
* delta_v is ALWAYS the IPW contrast: IPW is the pre-specified primary
  estimator and DR is a pre-specified sensitivity check, so no estimator is
  chosen after seeing the data. When |V_IPW - V_DR| / sd(Y) exceeds
  dr_disagreement_threshold, or the two contrasts disagree in sign, the
  contrast is flagged delta_v_estimator_sensitive and does not support a
  directional policy conclusion. Both contrasts are always exported.
* actionability_gate implements the gate of the paper: ATE CI excluding zero,
  agreeing IPW/DR policy-value signs, |V_IPW - V_DR| / sd(Y) within
  dr_disagreement_threshold, weighted SMD within balance_poor_threshold, and no
  positivity flag. Its per-criterion verdicts are exported alongside it.
* The E-value uses the conversion matching the outcome scale: RR = 1 + ATE/E[Y(0)]
  for binary outcomes, RR = exp(0.91 d) with d = ATE/sd(Y) for continuous ones.
* Bootstrap resamples units only, holding fitted predictions and propensity
  scores fixed, so intervals do not propagate training-partition nuisance
  estimation uncertainty.

Configurable thresholds
------------------------
Every numeric threshold used anywhere in this module (RMSE filter cutoff,
DR-disagreement cutoff, balance quality bands, degenerate-policy bounds,
overlap/positivity bounds, E-value interpretation bands, reallocation-size
bands, bootstrap CI level, RATE significance level, default subgroup
quantile) is declared once as a ``DEFAULT_*`` class attribute on
``CausalEvaluator`` and exposed as a same-named constructor argument. Pass
``None`` (the default) to keep the framework default, or an explicit value to
override it for a given instance. Nothing is hardcoded inside the methods.
"""

from __future__ import annotations

from math import erf, sqrt
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["CausalEvaluator"]


# --- helpers ---------------------------------------------------------------

def _1d(a) -> np.ndarray:
    return np.asarray(a, dtype=float).ravel()


def _2d(X) -> Tuple[np.ndarray, Optional[List[str]]]:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float), [str(c) for c in X.columns]
    a = np.asarray(X, dtype=float)
    return (a.reshape(-1, 1) if a.ndim == 1 else a), None


def _native(o):
    if isinstance(o, dict):
        return {str(k): _native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_native(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, np.ndarray):
        return _native(o.tolist())
    return o


def _ci(draws: Sequence[float], alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI at level (1 - alpha). alpha defaults to 0.05
    (95% CI); call sites pass ``CausalEvaluator.bootstrap_ci_alpha``."""
    a = np.asarray([d for d in draws if d is not None and np.isfinite(d)], dtype=float)
    if a.size < 2:
        return None, None
    lo_pct, hi_pct = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)
    return float(np.percentile(a, lo_pct)), float(np.percentile(a, hi_pct))


def _mean_se(v: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(v.size))


def _aipw(mu0, mu1, g, Y, T) -> np.ndarray:
    """Per-unit doubly robust pseudo-outcome for the ATE (Robins et al., 1994)."""
    return (mu1 - mu0) + T / g * (Y - mu1) - (1 - T) / (1 - g) * (Y - mu0)


def _pv(policy, g, Y, T) -> Dict[str, float]:
    e = np.where(policy == 1, g, 1 - g)
    w = (T == policy).astype(float) / e
    s = float(w.sum())
    return {"hajek": float(np.sum(w * Y) / s) if s > 0 else float("nan"),
            "ht": float(np.mean(w * Y))}


def _pv_dr(policy, mu0, mu1, g, Y, T) -> float:
    mu = np.where(policy == 1, mu1, mu0)
    e = np.where(policy == 1, g, 1 - g)
    return float(np.mean(mu + (T == policy).astype(float) / e * (Y - mu)))


def _rate_point(psi: np.ndarray, score: np.ndarray, weight: str,
                rng: np.random.Generator) -> float:
    """
    theta_alpha = (1/n) sum_j alpha(j/n) TOC(j/n), with
    TOC(j/n) = (1/j) sum_{l<=j} psi_(l) - mean(psi).
    Units are ordered by decreasing priority score, ties broken at random.
    alpha(u) = 1 gives AUTOC; alpha(u) = u gives the Qini coefficient.
    """
    n = psi.size
    order = np.lexsort((rng.random(n), -score))
    j = np.arange(1, n + 1, dtype=float)
    toc = np.cumsum(psi[order]) / j - float(psi.mean())
    alpha = np.ones(n) if weight == "autoc" else j / n
    return float(np.mean(alpha * toc))


def _rate(psi, score, weight, n_boot, rng, ci_z: float = 1.96,
          alpha: float = 0.05) -> Dict[str, Any]:
    """
    RATE with half-sample bootstrap inference: each replicate re-estimates the
    metric on n/2 units drawn WITHOUT replacement, and the standard deviation of
    those estimates estimates the standard error of the full-sample statistic.
    This is the scheme the RATE central limit theorem justifies; the ordinary
    with-replacement bootstrap is not.

    ``ci_z`` sets the normal-approximation CI multiplier and ``alpha`` the
    significance level for the two-sided p-value test of H0: RATE = 0; call
    sites pass ``CausalEvaluator.ci_z`` / ``CausalEvaluator.rate_significance_alpha``.
    """
    n = psi.size
    theta = _rate_point(psi, score, weight, rng)
    half = max(n // 2, 2)
    draws = []
    for _ in range(n_boot):
        i = rng.choice(n, size=half, replace=False)
        draws.append(_rate_point(psi[i], score[i], weight, rng))
    sd = float(np.std(draws, ddof=1)) if len(draws) > 1 else float("nan")
    if not np.isfinite(sd) or sd <= 0:
        return {"rate": theta, "rate_weight": weight, "rate_ci_lower": None,
                "rate_ci_upper": None, "rate_p_value": None,
                "rate_significant": None, "_rate_se": None}
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf((abs(theta) / sd) / sqrt(2.0))))
    return {"rate": theta, "rate_weight": weight,
            "rate_ci_lower": theta - ci_z * sd, "rate_ci_upper": theta + ci_z * sd,
            "rate_p_value": float(p), "rate_significant": bool(p < alpha),
            "_rate_se": sd}


def _smd(X, T, w=None) -> np.ndarray:
    m1, m0 = T == 1, T == 0
    if w is None:
        mu1, mu0 = X[m1].mean(0), X[m0].mean(0)
        v1, v0 = X[m1].var(0), X[m0].var(0)
    else:
        mu1 = np.average(X[m1], axis=0, weights=w[m1])
        mu0 = np.average(X[m0], axis=0, weights=w[m0])
        v1 = np.average((X[m1] - mu1) ** 2, axis=0, weights=w[m1])
        v0 = np.average((X[m0] - mu0) ** 2, axis=0, weights=w[m0])
    p = np.sqrt((v1 + v0) / 2)
    return np.abs(mu1 - mu0) / np.where(p < 1e-12, np.nan, p)


def _evalue_rr(rr: float) -> float:
    if not np.isfinite(rr) or rr <= 0:
        return float("nan")
    if rr < 1:
        rr = 1 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))


# ---------------------------------------------------------------------------

class CausalEvaluator:
    """
    Parameters
    ----------
    problem_name : str
    outcome_type : {"binary", "continuous"}
        Declare ordinal outcomes as "continuous": this selects the standardised
        mean difference conversion for the E-value.
    outcome_direction : {"adverse", "beneficial"}
        Which direction of the policy value is better. Resolves the ranking
        order, delta_v_direction and the benefit-score sign s_dir (+1 for
        "beneficial", -1 for "adverse"); see the benefit-score convention in
        the module docstring.
    n_bootstrap : int
    propensity_clip : (float, float), optional
        Clipping bounds applied before any inverse weighting.
        Default: ``DEFAULT_PROPENSITY_CLIP``.
    dr_disagreement_threshold : float, optional
        Threshold on |V_IPW - V_DR| / sd(Y), i.e. in outcome standard deviation
        units, so one rule serves outcomes on different scales.
        Default: ``DEFAULT_DR_DISAGREEMENT_THRESHOLD``.
    rate_weight : {"autoc", "qini"}, optional
        Default: ``DEFAULT_RATE_WEIGHT``.
    factual_fit_threshold : float, optional
        A model fails the necessary filter when its RMSE exceeds this multiple
        of the within-arm outcome SD in BOTH arms (kappa in the paper).
        Default: ``DEFAULT_FACTUAL_FIT_THRESHOLD``.
    ci_z : float, optional
        Normal-approximation critical value used for ATE and RATE confidence
        intervals. Default: ``DEFAULT_CI_Z`` (1.96, ~95%).
    bootstrap_ci_alpha : float, optional
        Two-sided significance level for percentile bootstrap intervals
        (policy value, DR policy value, ATE-boot, Delta_V, CATE draws).
        Default: ``DEFAULT_BOOTSTRAP_CI_ALPHA`` (0.05, i.e. 95% CI).
    rate_significance_alpha : float, optional
        Significance level for the RATE two-sided test of H0: RATE = 0.
        Default: ``DEFAULT_RATE_SIGNIFICANCE_ALPHA``.
    balance_excellent_threshold, balance_poor_threshold : float, optional
        Weighted-SMD bands: "excellent" below the first, "poor" at or above
        the second, "acceptable" in between.
        Defaults: ``DEFAULT_BALANCE_EXCELLENT_THRESHOLD`` (0.10),
        ``DEFAULT_BALANCE_POOR_THRESHOLD`` (0.25).
    degenerate_treat_rate_low, degenerate_treat_rate_high : float, optional
        A model's induced policy is degenerate when its treat rate falls
        outside [low, high]. Defaults: ``DEFAULT_DEGENERATE_TREAT_RATE_LOW``
        (0.01), ``DEFAULT_DEGENERATE_TREAT_RATE_HIGH`` (0.99).
    degenerate_ite_std : float, optional
        A model's induced policy is also degenerate when the predicted ITE
        has essentially zero spread. Default: ``DEFAULT_DEGENERATE_ITE_STD``
        (1e-9).
    minority_arm_severe_threshold : float, optional
        Severe treatment imbalance is flagged when the smaller arm's share
        falls below this value. Default:
        ``DEFAULT_MINORITY_ARM_SEVERE_THRESHOLD`` (0.05).
    propensity_extreme_lo, propensity_extreme_hi : float, optional
        A unit's raw (unclipped) propensity outside [lo, hi] counts toward
        the overlap-violation share. Defaults:
        ``DEFAULT_PROPENSITY_EXTREME_LO`` (0.05),
        ``DEFAULT_PROPENSITY_EXTREME_HI`` (0.95).
    positivity_share_threshold : float, optional
        A positivity warning is raised when the overlap-violation share
        exceeds this value. Default: ``DEFAULT_POSITIVITY_SHARE_THRESHOLD``
        (0.10).
    evalue_moderate_threshold, evalue_robust_threshold : float, optional
        E-value interpretation bands: "fragile" at or below the first,
        "robust" above the second, "moderate" in between. Defaults:
        ``DEFAULT_EVALUE_MODERATE_THRESHOLD`` (1.5),
        ``DEFAULT_EVALUE_ROBUST_THRESHOLD`` (2.0).
    reallocation_near_threshold, reallocation_large_threshold : float, optional
        Implementation-only heuristic (not derived in the paper) classifying
        |Delta_r_treat|: "moderate" below the first, "near_threshold" from
        the first up to the second, "large" above the second. Defaults:
        ``DEFAULT_REALLOCATION_NEAR_THRESHOLD`` (0.25),
        ``DEFAULT_REALLOCATION_LARGE_THRESHOLD`` (0.30).
    tau_decimals : int, optional
        tau_hat is rounded to this many decimals, so floating-point noise can
        neither flip the induced policy nor reorder tied units in the RATE.
        Default: ``DEFAULT_TAU_DECIMALS`` (10).
    subgroup_quantile : float, optional
        Default tail fraction for high/low-benefit subgroups in
        :meth:`policy_metrics`, taken over the benefit score b_hat, used when
        the per-call argument is omitted.
        Default: ``DEFAULT_SUBGROUP_QUANTILE`` (0.25).

    Examples
    --------
    >>> ev = CausalEvaluator("RHC", outcome_type="binary")
    >>> # d[name] = ev.evaluation_metrics(pred['y_0_hat'], pred['y_1_hat'],
    >>> #                                 pred['propensity_score'],
    >>> #                                 X=testX, Y=testY, T=testT, random_state=42)
    >>> # sel = CausalEvaluator.select_best_model(d)
    >>> # res = ev.policy_metrics(..., X=policyX, Y=policyY, T=policyT, random_state=42)
    >>> # Override a threshold for this instance only:
    >>> ev2 = CausalEvaluator("RHC", outcome_type="binary",
    ...                       dr_disagreement_threshold=0.15,
    ...                       balance_poor_threshold=0.20)
    """

    # -- framework defaults; override per instance via matching kwargs -----
    DEFAULT_FACTUAL_FIT_THRESHOLD = 1.0            # kappa, Eq. (rmse_filter)
    DEFAULT_DR_DISAGREEMENT_THRESHOLD = 0.10       # delta / s_Y, Eq. (estimand_rule)
    DEFAULT_PROPENSITY_CLIP = (0.01, 0.99)
    DEFAULT_RATE_WEIGHT = "autoc"

    DEFAULT_CI_Z = 1.96                            # ~95% normal-approx CI
    DEFAULT_BOOTSTRAP_CI_ALPHA = 0.05              # 95% percentile bootstrap CI
    DEFAULT_RATE_SIGNIFICANCE_ALPHA = 0.05

    DEFAULT_BALANCE_EXCELLENT_THRESHOLD = 0.10     # weighted SMD, Sec. Post-Selection Diagnostics
    DEFAULT_BALANCE_POOR_THRESHOLD = 0.25

    DEFAULT_DEGENERATE_TREAT_RATE_LOW = 0.01
    DEFAULT_DEGENERATE_TREAT_RATE_HIGH = 0.99
    DEFAULT_DEGENERATE_ITE_STD = 1e-9

    DEFAULT_MINORITY_ARM_SEVERE_THRESHOLD = 0.05
    DEFAULT_PROPENSITY_EXTREME_LO = 0.05
    DEFAULT_PROPENSITY_EXTREME_HI = 0.95
    DEFAULT_POSITIVITY_SHARE_THRESHOLD = 0.10

    DEFAULT_EVALUE_MODERATE_THRESHOLD = 1.5
    DEFAULT_EVALUE_ROBUST_THRESHOLD = 2.0

    DEFAULT_REALLOCATION_NEAR_THRESHOLD = 0.25     # implementation-only heuristic
    DEFAULT_REALLOCATION_LARGE_THRESHOLD = 0.30    # implementation-only heuristic

    DEFAULT_SUBGROUP_QUANTILE = 0.25               # Eq. (subgroups)

    # tau_hat is rounded to this many decimals before any decision quantity is
    # computed. Floating-point summation order (e.g. n_jobs=-1 in tree
    # ensembles) perturbs tau_hat at ~1e-17: without rounding, a mathematically
    # zero effect gets a random sign, flipping the policy 1{b_hat > 0}, and
    # tied units change order in the RATE ranking, so results would not be
    # reproducible. Exact zeros are not treated.
    DEFAULT_TAU_DECIMALS = 10

    def __init__(self, problem_name="Treatment Effect", outcome_type="binary",
                 outcome_direction="adverse", n_bootstrap=1000,
                 propensity_clip=None, dr_disagreement_threshold=None,
                 rate_weight=None, factual_fit_threshold=None,
                 ci_z=None, bootstrap_ci_alpha=None, rate_significance_alpha=None,
                 balance_excellent_threshold=None, balance_poor_threshold=None,
                 degenerate_treat_rate_low=None, degenerate_treat_rate_high=None,
                 degenerate_ite_std=None,
                 minority_arm_severe_threshold=None,
                 propensity_extreme_lo=None, propensity_extreme_hi=None,
                 positivity_share_threshold=None,
                 evalue_moderate_threshold=None, evalue_robust_threshold=None,
                 reallocation_near_threshold=None, reallocation_large_threshold=None,
                 subgroup_quantile=None, tau_decimals=None):
        if outcome_type not in {"binary", "continuous"}:
            raise ValueError("outcome_type must be 'binary' or 'continuous'")
        if outcome_direction not in {"adverse", "beneficial"}:
            raise ValueError("outcome_direction must be 'adverse' or 'beneficial'")

        rate_weight = self.DEFAULT_RATE_WEIGHT if rate_weight is None else rate_weight
        if rate_weight not in {"autoc", "qini"}:
            raise ValueError("rate_weight must be 'autoc' or 'qini'")

        propensity_clip = (self.DEFAULT_PROPENSITY_CLIP if propensity_clip is None
                            else propensity_clip)
        if not (0 < propensity_clip[0] < propensity_clip[1] < 1):
            raise ValueError("propensity_clip must satisfy 0 < lo < hi < 1")

        factual_fit_threshold = (self.DEFAULT_FACTUAL_FIT_THRESHOLD
                                  if factual_fit_threshold is None else factual_fit_threshold)
        if factual_fit_threshold <= 0:
            raise ValueError("factual_fit_threshold must be positive")

        dr_disagreement_threshold = (self.DEFAULT_DR_DISAGREEMENT_THRESHOLD
                                      if dr_disagreement_threshold is None
                                      else dr_disagreement_threshold)
        if dr_disagreement_threshold <= 0:
            raise ValueError("dr_disagreement_threshold must be positive")

        ci_z = self.DEFAULT_CI_Z if ci_z is None else ci_z
        if ci_z <= 0:
            raise ValueError("ci_z must be positive")

        bootstrap_ci_alpha = (self.DEFAULT_BOOTSTRAP_CI_ALPHA if bootstrap_ci_alpha is None
                               else bootstrap_ci_alpha)
        if not (0 < bootstrap_ci_alpha < 1):
            raise ValueError("bootstrap_ci_alpha must lie in (0, 1)")

        rate_significance_alpha = (self.DEFAULT_RATE_SIGNIFICANCE_ALPHA
                                    if rate_significance_alpha is None
                                    else rate_significance_alpha)
        if not (0 < rate_significance_alpha < 1):
            raise ValueError("rate_significance_alpha must lie in (0, 1)")

        balance_excellent_threshold = (self.DEFAULT_BALANCE_EXCELLENT_THRESHOLD
                                        if balance_excellent_threshold is None
                                        else balance_excellent_threshold)
        balance_poor_threshold = (self.DEFAULT_BALANCE_POOR_THRESHOLD
                                   if balance_poor_threshold is None
                                   else balance_poor_threshold)
        if not (0 <= balance_excellent_threshold < balance_poor_threshold):
            raise ValueError("require 0 <= balance_excellent_threshold < balance_poor_threshold")

        degenerate_treat_rate_low = (self.DEFAULT_DEGENERATE_TREAT_RATE_LOW
                                      if degenerate_treat_rate_low is None
                                      else degenerate_treat_rate_low)
        degenerate_treat_rate_high = (self.DEFAULT_DEGENERATE_TREAT_RATE_HIGH
                                       if degenerate_treat_rate_high is None
                                       else degenerate_treat_rate_high)
        if not (0 < degenerate_treat_rate_low < degenerate_treat_rate_high < 1):
            raise ValueError("require 0 < degenerate_treat_rate_low < "
                              "degenerate_treat_rate_high < 1")

        degenerate_ite_std = (self.DEFAULT_DEGENERATE_ITE_STD if degenerate_ite_std is None
                               else degenerate_ite_std)
        if degenerate_ite_std <= 0:
            raise ValueError("degenerate_ite_std must be positive")

        minority_arm_severe_threshold = (self.DEFAULT_MINORITY_ARM_SEVERE_THRESHOLD
                                          if minority_arm_severe_threshold is None
                                          else minority_arm_severe_threshold)
        if not (0 < minority_arm_severe_threshold < 1):
            raise ValueError("minority_arm_severe_threshold must lie in (0, 1)")

        propensity_extreme_lo = (self.DEFAULT_PROPENSITY_EXTREME_LO
                                  if propensity_extreme_lo is None else propensity_extreme_lo)
        propensity_extreme_hi = (self.DEFAULT_PROPENSITY_EXTREME_HI
                                  if propensity_extreme_hi is None else propensity_extreme_hi)
        if not (0 <= propensity_extreme_lo < propensity_extreme_hi <= 1):
            raise ValueError("require 0 <= propensity_extreme_lo < propensity_extreme_hi <= 1")

        positivity_share_threshold = (self.DEFAULT_POSITIVITY_SHARE_THRESHOLD
                                       if positivity_share_threshold is None
                                       else positivity_share_threshold)
        if not (0 < positivity_share_threshold < 1):
            raise ValueError("positivity_share_threshold must lie in (0, 1)")

        evalue_moderate_threshold = (self.DEFAULT_EVALUE_MODERATE_THRESHOLD
                                      if evalue_moderate_threshold is None
                                      else evalue_moderate_threshold)
        evalue_robust_threshold = (self.DEFAULT_EVALUE_ROBUST_THRESHOLD
                                    if evalue_robust_threshold is None
                                    else evalue_robust_threshold)
        if not (1 <= evalue_moderate_threshold < evalue_robust_threshold):
            raise ValueError("require 1 <= evalue_moderate_threshold < evalue_robust_threshold")

        reallocation_near_threshold = (self.DEFAULT_REALLOCATION_NEAR_THRESHOLD
                                        if reallocation_near_threshold is None
                                        else reallocation_near_threshold)
        reallocation_large_threshold = (self.DEFAULT_REALLOCATION_LARGE_THRESHOLD
                                         if reallocation_large_threshold is None
                                         else reallocation_large_threshold)
        if not (0 < reallocation_near_threshold <= reallocation_large_threshold < 1):
            raise ValueError("require 0 < reallocation_near_threshold <= "
                              "reallocation_large_threshold < 1")

        subgroup_quantile = (self.DEFAULT_SUBGROUP_QUANTILE if subgroup_quantile is None
                              else subgroup_quantile)
        if not (0 < subgroup_quantile < 0.5):
            raise ValueError("subgroup_quantile must lie in (0, 0.5)")

        self.problem_name = problem_name
        self.outcome_type = outcome_type
        self.outcome_direction = outcome_direction
        # s_dir of the benefit score b = s_dir * tau (module docstring).
        self.benefit_sign = 1.0 if outcome_direction == "beneficial" else -1.0
        self.n_bootstrap = int(n_bootstrap)

        self.propensity_clip = (float(propensity_clip[0]), float(propensity_clip[1]))
        self.dr_disagreement_threshold = float(dr_disagreement_threshold)
        self.rate_weight = rate_weight
        self.factual_fit_threshold = float(factual_fit_threshold)

        self.ci_z = float(ci_z)
        self.bootstrap_ci_alpha = float(bootstrap_ci_alpha)
        self.rate_significance_alpha = float(rate_significance_alpha)

        self.balance_excellent_threshold = float(balance_excellent_threshold)
        self.balance_poor_threshold = float(balance_poor_threshold)

        self.degenerate_treat_rate_low = float(degenerate_treat_rate_low)
        self.degenerate_treat_rate_high = float(degenerate_treat_rate_high)
        self.degenerate_ite_std = float(degenerate_ite_std)

        self.minority_arm_severe_threshold = float(minority_arm_severe_threshold)
        self.propensity_extreme_lo = float(propensity_extreme_lo)
        self.propensity_extreme_hi = float(propensity_extreme_hi)
        self.positivity_share_threshold = float(positivity_share_threshold)

        self.evalue_moderate_threshold = float(evalue_moderate_threshold)
        self.evalue_robust_threshold = float(evalue_robust_threshold)

        self.reallocation_near_threshold = float(reallocation_near_threshold)
        self.reallocation_large_threshold = float(reallocation_large_threshold)

        self.subgroup_quantile = float(subgroup_quantile)

        tau_decimals = self.DEFAULT_TAU_DECIMALS if tau_decimals is None else tau_decimals
        self.tau_decimals = int(tau_decimals)

    # --- internals ---------------------------------------------------------

    def _prep(self, y_pred_0, y_pred_1, propensity, X, Y, T):
        mu0, mu1 = _1d(y_pred_0), _1d(y_pred_1)
        g0, Yv, Tv = _1d(propensity), _1d(Y), _1d(T)
        Xm, cols = _2d(X)
        n = Yv.size
        for nm, a in (("y_pred_0", mu0), ("y_pred_1", mu1),
                      ("propensity", g0), ("T", Tv)):
            if a.size != n:
                raise ValueError(f"{nm} has length {a.size}, expected {n}")
        if Xm.shape[0] != n:
            raise ValueError(f"X has {Xm.shape[0]} rows, expected {n}")
        if not np.all(np.isin(Tv, (0.0, 1.0))):
            raise ValueError("T must be binary (0/1)")
        if not (Tv == 1).any() or not (Tv == 0).any():
            raise ValueError("both treatment arms must be present")
        if np.any(~np.isfinite(g0)) or g0.min() < 0 or g0.max() > 1:
            raise ValueError("propensity must be finite and within [0, 1]")
        if self.outcome_type == "binary" and not np.all(np.isin(Yv, (0.0, 1.0))):
            raise ValueError("outcome_type='binary' but Y is not 0/1")

        lo, hi = self.propensity_clip
        n_clip = int(np.sum((g0 < lo) | (g0 > hi)))
        g = np.clip(g0, lo, hi)
        tau = mu1 - mu0
        tau = np.round(tau, self.tau_decimals) + 0.0  # + 0.0 turns -0.0 into 0.0
        benefit = self.benefit_sign * tau
        psi = _aipw(mu0, mu1, g, Yv, Tv)
        return {"mu0": mu0, "mu1": mu1, "g0": g0, "g": g, "Y": Yv, "T": Tv,
                "X": Xm, "cols": cols, "tau": tau, "benefit": benefit,
                "policy": (benefit > 0).astype(float),
                "psi": psi, "psi_benefit": self.benefit_sign * psi,
                "n": n, "n_clip": n_clip}

    def _boot(self, p, masks, random_state):
        rng = np.random.default_rng(random_state)
        n, mu0, mu1 = p["n"], p["mu0"], p["mu1"]
        g, Y, T, pol, psi = p["g"], p["Y"], p["T"], p["policy"], p["psi"]
        out = {k: [] for k in ("pv", "ht", "dr", "ate", "dv_ipw", "dv_dr")}
        for k in (masks or {}):
            out[f"cate_{k}"] = []
        for _ in range(self.n_bootstrap):
            i = rng.integers(0, n, size=n)
            Tb, Yb, gb, pb, psib = T[i], Y[i], g[i], pol[i], psi[i]
            if not (Tb == 1).any() or not (Tb == 0).any():
                continue
            v = _pv(pb, gb, Yb, Tb)
            dr = _pv_dr(pb, mu0[i], mu1[i], gb, Yb, Tb)
            yb = float(Yb.mean())
            out["pv"].append(v["hajek"]); out["ht"].append(v["ht"])
            out["dr"].append(dr); out["ate"].append(float(np.mean(psib)))
            out["dv_ipw"].append(v["hajek"] - yb); out["dv_dr"].append(dr - yb)
            for k, msk in (masks or {}).items():
                s = msk[i]
                if s.sum() >= 2:
                    out[f"cate_{k}"].append(float(np.mean(psib[s])))
        return out

    def _balance(self, p, full):
        w = np.where(p["T"] == 1, 1 / p["g"], 1 / (1 - p["g"]))
        su, sw = _smd(p["X"], p["T"], None), _smd(p["X"], p["T"], w)
        fu, fw = su[np.isfinite(su)], sw[np.isfinite(sw)]
        mx = float(fw.max()) if fw.size else None
        out = {"max_smd_weighted": mx,
               "balance_quality": (
                   None if mx is None
                   else "excellent" if mx < self.balance_excellent_threshold
                   else "acceptable" if mx < self.balance_poor_threshold
                   else "poor")}
        if out["balance_quality"] == "poor" and fw.size:
            out["worst_balanced_covariates"] = [
                {"covariate": p["cols"][j] if p["cols"] else f"x{j}",
                 "smd_weighted": float(sw[j])}
                for j in np.argsort(-np.where(np.isfinite(sw), sw, -np.inf))[:5]
                if np.isfinite(sw[j])]
        if full:
            out["max_smd_unweighted"] = float(fu.max()) if fu.size else None
        return out

    def _overlap(self, p):
        g0, T = p["g0"], p["T"]
        minority = float(min(T.mean(), 1 - T.mean()))
        share_ext = float(np.mean((g0 < self.propensity_extreme_lo)
                                   | (g0 > self.propensity_extreme_hi)))
        return {"minority_arm_share": minority,
                "severe_imbalance": bool(minority < self.minority_arm_severe_threshold),
                "share_outside_overlap": share_ext,
                "positivity_flag": bool(share_ext > self.positivity_share_threshold)}

    def _ate(self, p, boot):
        ate, se = _mean_se(p["psi"])
        y0 = float(p["Y"][p["T"] == 0].mean())
        lo_b, hi_b = _ci(boot["ate"], self.bootstrap_ci_alpha)
        z = self.ci_z
        return {"ate": ate, "ate_se": se, "ate_ci_lower": ate - z * se,
                "ate_ci_upper": ate + z * se,
                "ate_significant": bool((ate - z * se) * (ate + z * se) > 0),
                "ate_ci_lower_boot": lo_b, "ate_ci_upper_boot": hi_b,
                "ate_plugin": float(np.mean(p["tau"])),
                "baseline_outcome_control": y0,
                "relative_change_vs_control": (float(ate / y0)
                                               if abs(y0) > 1e-12 else None)}

    def _evalue(self, a, sd_Y):
        """
        E-value (VanderWeele & Ding, 2017), with the conversion matching the
        outcome scale, as in the `evalue` package (Linden et al., 2020).
        """
        lo, hi = a["ate_ci_lower"], a["ate_ci_upper"]
        spans_null = bool(lo * hi <= 0)
        if self.outcome_type == "binary":
            y0 = a["baseline_outcome_control"]
            if not np.isfinite(y0) or y0 <= 0:
                return {"applicable": False, "evalue_point": None,
                        "evalue_ci_limit": None, "interpretation": None,
                        "reason": "control-arm baseline risk is zero or undefined"}
            rr = 1 + a["ate"] / y0
            rr_lim = 1 + (lo if abs(lo) < abs(hi) else hi) / y0
            conv, approx = "risk difference: RR = 1 + ATE/E[Y(0)]", False
        else:
            if not np.isfinite(sd_Y) or sd_Y <= 0:
                return {"applicable": False, "evalue_point": None,
                        "evalue_ci_limit": None, "interpretation": None,
                        "reason": "outcome standard deviation is zero or undefined"}
            d, s_d = a["ate"] / sd_Y, a["ate_se"] / sd_Y
            rr = float(np.exp(0.91 * d))
            r_lo = float(np.exp(0.91 * d - 1.78 * s_d))
            r_hi = float(np.exp(0.91 * d + 1.78 * s_d))
            rr_lim = r_lo if abs(np.log(r_lo)) < abs(np.log(r_hi)) else r_hi
            conv, approx = "standardised mean difference: RR = exp(0.91 d)", True
        e_pt = _evalue_rr(rr)
        e_lim = 1.0 if spans_null else _evalue_rr(rr_lim)
        return {"applicable": True, "conversion": conv,
                "relies_on_additional_approximation": approx, "reason": None,
                "evalue_point": round(e_pt, 3), "evalue_ci_limit": round(e_lim, 3),
                "interpretation": ("robust" if e_lim > self.evalue_robust_threshold else
                                   "moderate" if e_lim > self.evalue_moderate_threshold else
                                   "fragile")}

    _BOOTSTRAP_SCOPE = ("units only; fitted predictions and propensity scores "
                        "held fixed, so intervals do not propagate "
                        "training-partition nuisance estimation uncertainty")

    _BENEFIT_SCORE_RULE = (
        "the treatment-benefit score is b_hat = benefit_sign * tau_hat, with "
        "benefit_sign = +1 for a beneficial outcome and -1 for an adverse one; "
        "b_hat > 0 always means the patient is predicted to benefit from "
        "treatment. The model-induced policy is 1{b_hat > 0}, the RATE ranks on "
        "b_hat with the sign-standardised pseudo-outcome (so a positive RATE "
        "always means better benefit ranking), and the high/low-benefit "
        "subgroups are the tails of b_hat. Subgroup CATEs remain on the native "
        "outcome scale and must be read against outcome_direction")

    # === PHASE 1 ===========================================================

    def evaluation_metrics(self, y_pred_0, y_pred_1, propensity, X, Y, T,
                           random_state=None, full=False) -> Dict[str, Any]:
        """
        Filter, ranking and vetting metrics for one candidate model. Exports no
        policy quantity: the ATE, Delta_V and reallocation belong to the policy
        partition (:meth:`policy_metrics`).

        Returns
        -------
        outcome_type, outcome_direction, policy_value_better, benefit_sign,
        benefit_score_rule, bootstrap_scope,
        policy_value, policy_value_ci_lower, policy_value_ci_upper,
        policy_treat_rate, rmse_t0, rmse_t1, rmse_ratio_t0, rmse_ratio_t1,
        factual_fit_threshold, degenerate_policy, poor_factual_fit,
        eligible_for_ranking, rate, rate_weight, rate_ci_lower, rate_ci_upper,
        rate_p_value, rate_significant, balance{...}, dr_policy_value{...},
        evalue{...}, minority_arm_share, severe_imbalance,
        share_outside_overlap, positivity_flag

        ``full=True`` adds n, the Horvitz-Thompson policy value with interval,
        the RATE standard error, within-arm outcome SDs, the diagnostic ATE,
        the predicted-CATE summary, unweighted balance, and the clipped share.
        """
        p = self._prep(y_pred_0, y_pred_1, propensity, X, Y, T)
        boot = self._boot(p, None, random_state)
        Yv, Tv, tau = p["Y"], p["T"], p["tau"]
        m1, m0 = Tv == 1, Tv == 0

        v = _pv(p["policy"], p["g"], Yv, Tv)
        v_lo, v_hi = _ci(boot["pv"], self.bootstrap_ci_alpha)
        treat_rate = float(p["policy"].mean())

        rmse1 = float(np.sqrt(np.mean((Yv[m1] - p["mu1"][m1]) ** 2)))
        rmse0 = float(np.sqrt(np.mean((Yv[m0] - p["mu0"][m0]) ** 2)))
        sd1, sd0 = float(Yv[m1].std(ddof=1)), float(Yv[m0].std(ddof=1))
        r0 = float(rmse0 / sd0) if sd0 > 1e-12 else None
        r1 = float(rmse1 / sd1) if sd1 > 1e-12 else None
        thr = self.factual_fit_threshold
        poor_fit = bool(r0 is not None and r1 is not None and r0 > thr and r1 > thr)
        degenerate = bool(treat_rate <= self.degenerate_treat_rate_low
                          or treat_rate >= self.degenerate_treat_rate_high
                          or float(tau.std(ddof=1)) < self.degenerate_ite_std)

        rate = _rate(p["psi_benefit"], p["benefit"], self.rate_weight,
                     self.n_bootstrap, np.random.default_rng(random_state),
                     ci_z=self.ci_z, alpha=self.rate_significance_alpha)
        rate_se = rate.pop("_rate_se")

        sd_Y = float(Yv.std(ddof=1))
        dr = _pv_dr(p["policy"], p["mu0"], p["mu1"], p["g"], Yv, Tv)
        dis = abs(dr - v["hajek"]) if np.isfinite(v["hajek"]) else None
        dis_rel = None if (dis is None or sd_Y <= 1e-12) else dis / sd_Y
        ate = self._ate(p, boot)

        out = {
            "outcome_type": self.outcome_type,
            "outcome_direction": self.outcome_direction,
            "policy_value_better": ("lower" if self.outcome_direction == "adverse"
                                    else "higher"),
            "benefit_sign": self.benefit_sign,
            "benefit_score_rule": self._BENEFIT_SCORE_RULE,
            "bootstrap_scope": self._BOOTSTRAP_SCOPE,
            "policy_value": v["hajek"],
            "policy_value_ci_lower": v_lo,
            "policy_value_ci_upper": v_hi,
            "policy_treat_rate": treat_rate,
            "rmse_t0": rmse0, "rmse_t1": rmse1,
            "rmse_ratio_t0": r0, "rmse_ratio_t1": r1,
            "factual_fit_threshold": thr,
            "degenerate_policy": degenerate,
            "poor_factual_fit": poor_fit,
            "eligible_for_ranking": bool(not degenerate and not poor_fit),
            **rate,
            "balance": self._balance(p, full),
            "dr_policy_value": {
                "dr_policy_value": dr,
                "ipw_dr_disagreement_sd_units": (None if dis_rel is None
                                                 else round(float(dis_rel), 6)),
                "propensity_misspec_flag": (
                    None if dis_rel is None
                    else bool(dis_rel > self.dr_disagreement_threshold))},
            "evalue": self._evalue(ate, sd_Y),
            **self._overlap(p),
        }

        if full:
            ht_lo, ht_hi = _ci(boot["ht"], self.bootstrap_ci_alpha)
            dr_lo, dr_hi = _ci(boot["dr"], self.bootstrap_ci_alpha)
            out.update({
                "n": int(p["n"]),
                "problem_name": self.problem_name,
                "policy_value_ht": v["ht"],
                "policy_value_ht_ci_lower": ht_lo,
                "policy_value_ht_ci_upper": ht_hi,
                "rate_se": rate_se,
                "outcome_sd_t0": sd0, "outcome_sd_t1": sd1,
                "share_clipped": float(p["n_clip"] / p["n"]),
                "ate_diagnostic": {k: ate[k] for k in
                                   ("ate", "ate_ci_lower", "ate_ci_upper",
                                    "ate_plugin")},
                "cate_summary": {"mean": float(tau.mean()),
                                 "std": float(tau.std(ddof=1)),
                                 "q25": float(np.percentile(tau, 25)),
                                 "median": float(np.percentile(tau, 50)),
                                 "q75": float(np.percentile(tau, 75)),
                                 "share_predicted_to_benefit":
                                     float(np.mean(p["benefit"] > 0))},
                "dr_policy_value_ci_lower": dr_lo,
                "dr_policy_value_ci_upper": dr_hi,
                "data_stats": {"n_treated": int(m1.sum()),
                               "n_control": int(m0.sum()),
                               "treatment_rate": float(Tv.mean()),
                               "outcome_mean": float(Yv.mean()),
                               "outcome_std": sd_Y}})
        return _native(out)

    # === MODEL SELECTION ===================================================

    @staticmethod
    def select_best_model(results: Dict[str, Dict[str, Any]],
                          outcome_direction: Optional[str] = None) -> Dict[str, Any]:
        """
        Apply the selection hierarchy to ``{model_name: evaluation_metrics(...)}``
        and return M*, so the caller knows which model to pass to
        :meth:`policy_metrics`.

        Models flagged degenerate or failing the factual-fit filter are excluded.
        Survivors are ranked EXCLUSIVELY by policy value, i.e. by the value of
        the model-induced policy 1{b_hat > 0}. The RATE is reported alongside as
        the secondary criterion: agreement between the two orderings is recorded
        as convergent evidence, and disagreement is recorded as a selection flag,
        with the policy value still governing the final selection.

        The same rules are encoded in the Phase 1 prompt, so the interpretive
        layer reaches this result from the per-model metrics alone; this method
        is the programmatic path to the same answer.
        """
        if not results:
            raise ValueError("results is empty")
        if outcome_direction is None:
            dirs = {r["outcome_direction"] for r in results.values()}
            if len(dirs) != 1:
                raise ValueError(f"inconsistent outcome_direction: {dirs}")
            outcome_direction = dirs.pop()

        rows = []
        for name, r in results.items():
            reasons = []
            if r["degenerate_policy"]:
                reasons.append(
                    f"degenerate policy (treat rate {r['policy_treat_rate']:.3f})")
            if r["poor_factual_fit"]:
                reasons.append(
                    f"factual fit no better than arm means (RMSE/SD "
                    f"{r['rmse_ratio_t0']:.2f} / {r['rmse_ratio_t1']:.2f})")
            rows.append({
                "model": name,
                "policy_value": r["policy_value"],
                "pv_ci_lower": r["policy_value_ci_lower"],
                "pv_ci_upper": r["policy_value_ci_upper"],
                "rate": r["rate"],
                "rate_significant": r["rate_significant"],
                "eligible": r["eligible_for_ranking"],
                "exclusion_reason": "; ".join(reasons) or None})

        df = pd.DataFrame(rows)
        elig = df[df["eligible"]].copy()
        if elig.empty:
            return {"outcome_direction": outcome_direction, "ranking": df,
                    "selected_model": None, "runner_up": None,
                    "ci_overlap_warning": None,
                    "rate_agrees_with_policy_value": None,
                    "selection_status": "no eligible model"}

        asc = outcome_direction == "adverse"
        elig = elig.sort_values("policy_value", ascending=asc, kind="mergesort")
        df["rank"] = pd.NA
        for i, idx in enumerate(elig.index, start=1):
            df.loc[idx, "rank"] = i

        best = elig.iloc[0]
        runner = elig.iloc[1] if len(elig) >= 2 else None
        overlap = agree = None
        if runner is not None:
            overlap = bool(best["pv_ci_lower"] < runner["pv_ci_upper"]
                           and runner["pv_ci_lower"] < best["pv_ci_upper"])
            agree = bool(elig.sort_values("rate", ascending=False)
                         .iloc[0]["model"] == best["model"])

        df = df.sort_values(["rank", "model"], na_position="last").reset_index(drop=True)
        return {
            "outcome_direction": outcome_direction,
            "ranking": df,
            "selected_model": str(best["model"]),
            "runner_up": None if runner is None else str(runner["model"]),
            "ci_overlap_warning": overlap,
            "rate_agrees_with_policy_value": agree,
            "selection_status": (
                "indicative (top-two policy-value CIs overlap)" if overlap
                else "conclusive at the 95% bootstrap level" if overlap is False
                else "single eligible model")}

    # === PHASE 2 ===========================================================

    def policy_metrics(self, y_pred_0, y_pred_1, propensity, X, Y, T,
                       random_state=None, subgroup_quantile=None,
                       covariate_subgroups=None, full=False) -> Dict[str, Any]:
        """
        Population effect, subgroup effects and policy impact for the selected
        model. The qualifying diagnostics (RATE, balance, DR cross-check,
        E-value) are re-estimated here as well: each of them qualifies a Phase 2
        quantity, and a value computed on another partition certifies nothing
        about this one.

        Parameters
        ----------
        subgroup_quantile : float, optional
            Tail fraction defining the high/low-benefit subgroups; 0.25 gives
            the top and bottom quartiles of the predicted treatment-benefit
            score b_hat. Defaults to ``self.subgroup_quantile`` (see the
            class-level ``DEFAULT_SUBGROUP_QUANTILE``) when omitted.
        covariate_subgroups : dict, optional
            ``{name: boolean mask}``; each yields a ``cate_<name>`` entry.

        Returns
        -------
        outcome_type, outcome_direction, policy_value_better, benefit_sign,
        benefit_score_rule, bootstrap_scope,
        ate, ate_ci_lower, ate_ci_upper, ate_significant,
        baseline_outcome_control, relative_change_vs_control,
        subgroup_quantile, subgroup_definition_rule, multiplicity_note,
        cate_high{...}, cate_low{...}, cate_<name>{...},
        rate, rate_weight, rate_ci_lower, rate_ci_upper, rate_p_value,
        rate_significant, current_treat_rate, model_policy_treat_rate,
        optimal_treat_rate (deprecated alias),
        delta_r_treat, reallocation_class, current_policy_value,
        delta_v_estimand, delta_v_estimand_rule, delta_v, delta_v_ci_lower,
        delta_v_ci_upper, delta_v_significant, delta_v_direction, delta_v_ipw,
        delta_v_dr, estimator_sign_disagreement,
        delta_v_estimator_sensitive, delta_v_supports_direction,
        ipw_dr_disagreement_sd_units, propensity_misspec_flag, balance{...},
        evalue{...}, minority_arm_share, severe_imbalance,
        share_outside_overlap, positivity_flag, actionability_gate,
        actionability_gate_components, actionability_gate_rule

        ``full=True`` adds n, the raw policy values with intervals, both
        Delta_V intervals, the plug-in ATE, benefit-score thresholds, factual
        RMSE, and unweighted balance.
        """
        if subgroup_quantile is None:
            subgroup_quantile = self.subgroup_quantile
        if not (0 < subgroup_quantile < 0.5):
            raise ValueError("subgroup_quantile must lie in (0, 0.5)")
        p = self._prep(y_pred_0, y_pred_1, propensity, X, Y, T)
        tau, psi, benefit = p["tau"], p["psi"], p["benefit"]

        # Tails of the benefit score, so "high" is always the group predicted
        # to gain most from treatment regardless of the outcome's direction.
        q_hi = float(np.quantile(benefit, 1 - subgroup_quantile))
        q_lo = float(np.quantile(benefit, subgroup_quantile))
        masks = {"high": benefit > q_hi, "low": benefit < q_lo}
        for name, m in (covariate_subgroups or {}).items():
            a = np.asarray(m)
            if a.dtype != bool:
                t = np.zeros(p["n"], dtype=bool); t[a.astype(int)] = True; a = t
            if a.size != p["n"]:
                raise ValueError(f"covariate_subgroups['{name}'] length mismatch")
            masks[name] = a

        boot = self._boot(p, masks, random_state)
        ate = self._ate(p, boot)
        rate = _rate(p["psi_benefit"], benefit, self.rate_weight,
                     self.n_bootstrap, np.random.default_rng(random_state),
                     ci_z=self.ci_z, alpha=self.rate_significance_alpha)
        rate_se = rate.pop("_rate_se")

        pct_hi = int(round((1 - subgroup_quantile) * 100))
        pct_lo = int(round(subgroup_quantile * 100))
        out = {
            "outcome_type": self.outcome_type,
            "outcome_direction": self.outcome_direction,
            "policy_value_better": ("lower" if self.outcome_direction == "adverse"
                                    else "higher"),
            "benefit_sign": self.benefit_sign,
            "benefit_score_rule": self._BENEFIT_SCORE_RULE,
            "bootstrap_scope": self._BOOTSTRAP_SCOPE,
            "ate": ate["ate"], "ate_ci_lower": ate["ate_ci_lower"],
            "ate_ci_upper": ate["ate_ci_upper"],
            "ate_significant": ate["ate_significant"],
            "baseline_outcome_control": ate["baseline_outcome_control"],
            "relative_change_vs_control": ate["relative_change_vs_control"],
            "subgroup_quantile": float(subgroup_quantile),
            "subgroup_definition_rule": (
                "high-benefit = predicted treatment-benefit score above the "
                f"{pct_hi}th percentile; low-benefit = below the {pct_lo}th "
                "percentile"),
            "multiplicity_note": ("no multiple-testing correction is applied "
                                  "across subgroups; estimates are "
                                  "hypothesis-generating"),
            **rate}

        for key, m in masks.items():
            n_s = int(m.sum())
            if n_s < 2:
                out[f"cate_{key}"] = {"cate": None, "n": n_s}
                continue
            mu, se = _mean_se(psi[m])
            z = self.ci_z
            entry = {"cate": mu, "cate_ci_lower": mu - z * se,
                     "cate_ci_upper": mu + z * se, "n": n_s,
                     "significant": bool((mu - z * se) * (mu + z * se) > 0)}
            if full:
                entry["cate_predicted_model_based"] = float(np.mean(tau[m]))
                entry["benefit_score_predicted"] = float(np.mean(benefit[m]))
            out[f"cate_{key}"] = entry

        v = _pv(p["policy"], p["g"], p["Y"], p["T"])
        dr = _pv_dr(p["policy"], p["mu0"], p["mu1"], p["g"], p["Y"], p["T"])
        y_bar, sd_Y = float(p["Y"].mean()), float(p["Y"].std(ddof=1))
        dis = abs(dr - v["hajek"]) if np.isfinite(v["hajek"]) else None
        dis_rel = None if (dis is None or sd_Y <= 1e-12) else dis / sd_Y
        misspec = (None if dis_rel is None
                   else bool(dis_rel > self.dr_disagreement_threshold))

        # IPW is the pre-specified primary estimator and DR the pre-specified
        # sensitivity check, so no estimator is chosen after seeing delta. A
        # large standardised discrepancy or a sign disagreement instead flags
        # the contrast as estimator-sensitive.
        dv_ipw, dv_dr = v["hajek"] - y_bar, dr - y_bar
        sign_disagreement = bool(dv_ipw * dv_dr < 0)
        estimator_sensitive = bool(misspec) or sign_disagreement

        delta_v = dv_ipw
        dv_lo, dv_hi = _ci(boot["dv_ipw"], self.bootstrap_ci_alpha)
        sig = None if (dv_lo is None or dv_hi is None) else bool(dv_lo * dv_hi > 0)
        if self.outcome_direction == "adverse":
            direction = ("beneficial" if delta_v < 0 else
                         "adverse" if delta_v > 0 else "none")
        else:
            direction = ("beneficial" if delta_v > 0 else
                         "adverse" if delta_v < 0 else "none")
        delta_r = float(p["policy"].mean() - p["T"].mean())

        balance = self._balance(p, full)
        overlap = self._overlap(p)
        max_smd = balance["max_smd_weighted"]

        gate_components = {
            "ate_ci_excludes_zero": bool(ate["ate_significant"]),
            "policy_value_signs_agree":  not sign_disagreement,
            "dr_discrepancy_within_threshold": bool(
                dis_rel is not None and dis_rel <= self.dr_disagreement_threshold),
            "balance_within_threshold": bool(
                max_smd is not None and max_smd <= self.balance_poor_threshold),
            "overlap_ok": not overlap["positivity_flag"],
        }

        out.update({
            "current_treat_rate": float(p["T"].mean()),
            "model_policy_treat_rate": float(p["policy"].mean()),
            # Deprecated alias: the policy is model-induced, not optimal.
            "optimal_treat_rate": float(p["policy"].mean()),
            "delta_r_treat": delta_r,
            "reallocation_class": ("large" if abs(delta_r) > self.reallocation_large_threshold
                                   else "near_threshold" if abs(delta_r) >= self.reallocation_near_threshold
                                   else "moderate"),
            "current_policy_value": y_bar,
            "delta_v_estimand": "ipw",
            "delta_v_estimand_rule": (
                "IPW is the pre-specified primary estimator and DR a "
                "pre-specified sensitivity check; no estimator is selected "
                "after observing the data. When |V_IPW - V_DR| / sd(Y) exceeds "
                f"{self.dr_disagreement_threshold}, or the two contrasts "
                "disagree in sign, the contrast is flagged estimator-sensitive "
                "and does not support a directional policy conclusion."),
            "delta_v": delta_v,
            "delta_v_ci_lower": dv_lo, "delta_v_ci_upper": dv_hi,
            "delta_v_significant": sig, "delta_v_direction": direction,
            "delta_v_ipw": dv_ipw, "delta_v_dr": dv_dr,
            "estimator_sign_disagreement": sign_disagreement,
            "delta_v_estimator_sensitive": estimator_sensitive,
            "delta_v_supports_direction": not estimator_sensitive,
            "ipw_dr_disagreement_sd_units": (None if dis_rel is None
                                             else round(float(dis_rel), 6)),
            "propensity_misspec_flag": misspec,
            "balance": balance,
            "evalue": self._evalue(ate, sd_Y),
            **overlap,
            "actionability_gate": int(all(gate_components.values())),
            "actionability_gate_components": gate_components,
            "actionability_gate_rule": (
                "alpha = 1 requires ALL of: the ATE confidence interval "
                "excluding zero; agreeing signs of the IPW and DR policy "
                "contrasts; |V_IPW - V_DR| / sd(Y) <= "
                f"{self.dr_disagreement_threshold}; max weighted SMD <= "
                f"{self.balance_poor_threshold}; and no positivity flag. It is "
                "a conservative evidence gate, not a test of causal truth: "
                "alpha = 1 permits cautious policy translation and does not "
                "establish prospective benefit.")})

        if full:
            v_lo, v_hi = _ci(boot["pv"], self.bootstrap_ci_alpha)
            dr_lo, dr_hi = _ci(boot["dr"], self.bootstrap_ci_alpha)
            di_lo, di_hi = _ci(boot["dv_ipw"], self.bootstrap_ci_alpha)
            dd_lo, dd_hi = _ci(boot["dv_dr"], self.bootstrap_ci_alpha)
            m1, m0 = p["T"] == 1, p["T"] == 0
            out.update({
                "n": int(p["n"]),
                "problem_name": self.problem_name,
                "ate_se": ate["ate_se"], "ate_plugin": ate["ate_plugin"],
                "ate_ci_lower_boot": ate["ate_ci_lower_boot"],
                "ate_ci_upper_boot": ate["ate_ci_upper_boot"],
                "rate_se": rate_se,
                "benefit_threshold_high": q_hi, "benefit_threshold_low": q_lo,
                "policy_value_ipw": v["hajek"],
                "policy_value_ipw_ci_lower": v_lo,
                "policy_value_ipw_ci_upper": v_hi,
                "policy_value_dr": dr,
                "policy_value_dr_ci_lower": dr_lo,
                "policy_value_dr_ci_upper": dr_hi,
                "delta_v_ipw_ci_lower": di_lo, "delta_v_ipw_ci_upper": di_hi,
                "delta_v_dr_ci_lower": dd_lo, "delta_v_dr_ci_upper": dd_hi,
                "share_clipped": float(p["n_clip"] / p["n"]),
                "rmse_t0": float(np.sqrt(np.mean((p["Y"][m0] - p["mu0"][m0]) ** 2))),
                "rmse_t1": float(np.sqrt(np.mean((p["Y"][m1] - p["mu1"][m1]) ** 2))),
                "data_stats": {"n_treated": int(m1.sum()),
                               "n_control": int(m0.sum()),
                               "treatment_rate": float(p["T"].mean()),
                               "outcome_mean": y_bar, "outcome_std": sd_Y}})
        return _native(out)