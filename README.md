# Treatment effect estimation on observational data

Code accompanying the paper *From Treatment Effect Estimation to Actionable
Policies: A Framework for Real-World Observational Data*.

The pipeline works with any tabular dataset that has a **binary treatment**
and a **binary or continuous outcome**. You describe your study in a JSON
configuration file and run `main.py`.

It runs in two phases on three disjoint partitions of the data (by default
60% training, 20% evaluation, 20% policy):

1. **Quantitative evaluation and model selection** (evaluation split). Each
   candidate model is screened for degenerate policies and for poor factual fit
   (RMSE / within-arm SD > 1 in both arms). The survivors are ranked by the IPW
   (Hájek) policy value of the induced rule `1{b_hat > 0}`, with RATE (AUTOC)
   as the secondary criterion. Covariate balance, a doubly robust policy value
   and the E-value are reported as post-selection diagnostics.
2. **Policy development** (policy split). The selected model `M*` is not
   re-fitted. On the policy split the code estimates the AIPW ATE, the
   high- and low-benefit subgroup CATEs, the policy contrast `ΔV`, the
   treatment-rate change `Δr_treat` and the actionability gate.

Each phase can optionally be passed to an LLM interpretive layer. The LLM
receives only pre-computed fields, your problem and data descriptions, and a
rule-governed prompt. It performs no arithmetic.

## Repository structure

```
main.py               Pipeline (CLI and Python API)
examples/rhc/         Example configuration: right heart catheterization (public data)
models/
  s_learner.py        S-Learner (any scikit-learn regressor)
  t_learner.py        T-Learner (any scikit-learn regressor)
  dragonnet.py        Dragonnet with targeted regularization (TensorFlow)
  losses.py           Dragonnet losses
  cxgboost.py         C-XGBoost
  cxgboost_treg.py    C-XGBoost with propensity head and targeted regularization
utils/
  evaluator.py        CausalEvaluator: Phase 1 metrics, model selection, Phase 2 metrics
  prompts.py          System prompts for the two LLM reports
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.8 (TensorFlow 2.10, xgboost 1.7.3) and Python 3.10
(xgboost 2.1). TensorFlow is needed only for Dragonnet. `openai`, `anthropic`
and `python-dotenv` are needed only for the LLM reports.

## Quick start

Run the example, which downloads the public SUPPORT/RHC dataset:

```bash
python main.py --config examples/rhc/config.json
python main.py --config examples/rhc/config.json --skip-dragonnet   # no TensorFlow
```

## Running on your own data

### 1. Prepare the data

Use one row per unit, saved as CSV, TSV, Parquet or Excel. The table needs:

- a **treatment** column with two values, either 0/1 or any two labels;
- an **outcome** column, either binary (0/1 or two labels) or numeric;
- the **covariates**, which must be measured before treatment. Numeric and
  text columns are both accepted. Text and categorical columns are converted
  to integer codes automatically.

The data must not contain missing values. The pipeline stops and lists the
affected columns, so impute or drop missing values before you run it.

### 2. Write the two descriptions

These texts are inserted into the LLM prompts. Write them as plain-text or
Markdown files, using [examples/rhc/](examples/rhc/) as a template.

- **Problem description**: the research question, why the causal effect is
  uncertain, the mechanisms involved and the main confounding threats.
- **Data description**: what T = 1 and T = 0 mean, what the covariates are,
  what the outcome measures and which direction is worse. Add any framing
  that matters for policy recommendations, for example who makes the
  treatment decision.

The following placeholders are filled from the data, so the figures you
quote always match it:

| Placeholder | Replaced with |
|---|---|
| `{n}` | number of rows |
| `{n_covariates}` | number of covariates |
| `{treatment_rate}` | share of treated units, e.g. `38.1%` |
| `{outcome_mean}` | outcome rate for binary outcomes (e.g. `33.4%`), mean otherwise |

### 3. Write the configuration

Minimal example:

```json
{
  "name": "Effect of weekend admission on length of stay",
  "data": "data/admissions.csv",
  "treatment": {"column": "admission_day", "treated_value": "weekend"},
  "outcome":   {"column": "los", "type": "continuous", "direction": "adverse"},
  "exclude": ["patient_id"],
  "problem_description": "problem_description.md",
  "data_description": "data_description.md"
}
```

Relative paths are resolved against the folder that contains the config.

| Key | Required | Default | Description |
|---|---|---|---|
| `name` | | `"Treatment Effect"` | Short study title, used in the reports |
| `data` | yes* | | Path or URL of the data file. *Can be replaced by `--data` or by passing `df=` in Python |
| `treatment.column` | yes | | Treatment column |
| `treatment.treated_value` | if not 0/1 | | Value that means "treated". Every other value means control |
| `outcome.column` | yes | | Outcome column |
| `outcome.type` | | `"binary"` | `"binary"` or `"continuous"`. Declare ordinal outcomes as `"continuous"` |
| `outcome.positive_value` | if binary and not 0/1 | | Value coded as 1 |
| `outcome.direction` | | `"adverse"` | `"adverse"` if a larger outcome is worse, `"beneficial"` if it is better |
| `covariates` | | all other columns | Explicit list of covariates |
| `exclude` | | `[]` | Columns to drop when `covariates` is not given, such as IDs or post-treatment variables |
| `categorical` | | `"auto"` | List of columns to encode as categories, or `"auto"` for every non-numeric covariate |
| `problem_description` | | `""` | File path or inline text |
| `data_description` | | `""` | File path or inline text |
| `actionable` | | `true` | Whether the treatment is a decision someone can change. If it is not (for example, admission day), the policy report gives no treatment recommendations |
| `split` | | `{"eval_size": 0.2, "policy_size": 0.2, "seed": 1983, "stratify": "auto"}` | Partition sizes, seed, and stratification: `"auto"` stratifies on the outcome when it is binary, `"outcome"` and `"treatment"` force one variable, and `null` turns stratification off. For an ordinal outcome with few levels, `"outcome"` also works |
| `n_bootstrap` | | `100` | Bootstrap replications |
| `models` | | all eight | List of model names, or an object `{name: hyperparameters}` that overrides the defaults below |
| `evaluator` | | `{}` | Extra `CausalEvaluator` arguments, e.g. `{"balance_poor_threshold": 0.2}`. See the docstring in [utils/evaluator.py](utils/evaluator.py) |

### 4. Run it

```bash
python main.py --config my_study/config.json
```

| Option | Default | Description |
|---|---|---|
| `--config` | (required) | JSON configuration |
| `--data` | config `data` | Override the data file or URL |
| `--output-dir` | `results` | Where all outputs are written |
| `--n-bootstrap` | config value | Override the number of bootstrap replications |
| `--skip-dragonnet` | off | Leave Dragonnet out of the candidate pool |
| `--llm` | `none` | `openai` or `anthropic` to generate the two LLM reports |
| `--llm-model` | `gpt-4.1` / `claude-sonnet-5-5` | Model id for the chosen provider |

You can also call the pipeline from Python, for example after loading or
cleaning the data yourself:

```python
import pandas as pd
from main import load_config, run_pipeline

df = pd.read_sql("SELECT ...", connection)       # any DataFrame
cfg = load_config("my_study/config.json")
out = run_pipeline(cfg, df=df, output_dir="results/my_study")

out["selected_model"]     # name of M*
out["evaluation"]         # Phase 1 metrics per model
out["policy"]             # Phase 2 metrics for M*
out["models"]             # fitted models
```

## Candidate models

| Name | Model | Default hyperparameters |
|---|---|---|
| `S-LR` | S-Learner, linear regression | none |
| `T-LR` | T-Learner, linear regression | none |
| `R-Forest` | S-Learner, random forest | `n_estimators=150, max_depth=6, min_samples_split=4` |
| `S-kNN` | S-Learner, k-nearest neighbours | `n_neighbors=7, metric="l2"` |
| `S-GBM` | S-Learner, gradient boosting | `n_estimators=200, max_depth=5, learning_rate=0.05` |
| `Dragonnet` | Dragonnet with targeted regularization | see [models/dragonnet.py](models/dragonnet.py) |
| `C-XGBoost (treg)` | C-XGBoost with propensity head and targeted regularization | `n_estimators=150, max_depth=5, max_leaves=4, learning_rate=0.05, reg_alpha=3.0, epsilon=0.001` |
| `C-XGBoost` | C-XGBoost | `n_estimators=200, max_depth=4, min_child_weight=2, max_leaves=4, learning_rate=0.1` |

The defaults are the values tuned for the RHC example. Tune them for your own
data and set them in `models`. The keys are passed to the underlying
scikit-learn or XGBoost estimator. Three keys are handled separately:
`alpha`, `beta` and `epsilon` for `C-XGBoost (treg)`, `task` for `C-XGBoost`,
and the Dragonnet keys (`reg_l2`, `batch_size`, `validation_split`,
`loss_ratio`), which go to `Dragonnet.fit`.

Every model except Dragonnet and C-XGBoost (treg) uses the same Random Forest
propensity model. Those two estimate the propensity score themselves.

## LLM reports (optional)

Both prompts are always saved to the output folder, so you can inspect the
results and the exact LLM input without an API key. To generate the reports as
well, set the key in the environment or in a `.env` file and choose a
provider:

```bash
# .env
OPENAI_API_KEY=...        # for --llm openai
ANTHROPIC_API_KEY=...     # for --llm anthropic

python main.py --config my_study/config.json --llm anthropic
```

The interpretive layer does not depend on the provider: it reproduces
pre-computed fields verbatim, so any instruction-following model can be used.

## Outputs

| File | Content |
|---|---|
| `phase1_evaluation.csv` | One row per model: filters, policy value with CI, RMSE ratios, RATE, diagnostics |
| `phase1_evaluation.json` | Full per-model metrics and the treat-all, treat-none and current-practice reference values |
| `phase1_prompt.txt` / `phase1_report.md` | Phase 1 LLM input and report |
| `phase2_policy.json` | Phase 2 metrics for `M*`, including the actionability gate and its components |
| `phase2_prompt.txt` / `phase2_report.md` | Phase 2 LLM input and report |
| `plots/hist_<model>.png` | Distributions of `y_0_hat` and `y_1_hat` on the evaluation split |

`M*` is chosen with `CausalEvaluator.select_best_model`, using the selection
hierarchy described in the paper.

## Reproducibility notes

- The data split, the propensity model and the tree-based learners are seeded,
  and `tau_hat` is rounded to 10 decimals before any decision quantity is
  computed. Repeated runs on the same machine therefore give the same results.
  Dragonnet is the exception: TensorFlow training is not bit-for-bit
  deterministic.
- Confidence intervals come from a unit-level bootstrap that keeps fitted
  predictions and propensity scores fixed. They do not include
  nuisance-estimation uncertainty.

## Example data

The RHC example uses the public SUPPORT dataset,
<https://hbiostat.org/data/repo/rhc.csv>. The example configuration
reproduces Use case 2 of the paper.

## License

MIT. See [LICENSE](LICENSE).
