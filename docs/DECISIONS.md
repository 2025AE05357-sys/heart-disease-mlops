# Design decisions

Each decision records the options I considered, what I chose and why. IDs follow the project
plan (D0–D15); each one is added at the stage that needs it.

## Summary

| ID | Stage | Decision | Choice | Date |
|---|---|---|---|---|
| D0 | 0 | Originality theme | Clinical-grade traceability | 30 Sep 2026 |
| D1 | 1 | Data scope | Train and tune on Cleveland; external validation on Hungary, Switzerland and VA Long Beach | 30 Sep 2026 |
| D2 | 1 | Physiologically impossible zeros (`chol = 0`, `trestbps = 0`) | Treat as missing; impute inside the model pipeline | 30 Sep 2026 |
| D3 | 2 | Engineered features | `hr_pct_pred` searched as a hyperparameter; no missing-value indicators | 30 Sep 2026 (provisional) |
| D4 | 2 | Model families | Logistic regression, random forest, histogram gradient boosting | 30 Sep 2026 (provisional) |
| D5 | 2 | Champion selection rule | Best mean CV ROC-AUC; within 0.01, higher mean CV recall; then the simpler model | 30 Sep 2026 (provisional) |
| D6 | 2/6 | Decision threshold and risk bands | Class threshold 0.5; risk bands low < 0.3 ≤ moderate < 0.6 ≤ high; recall-0.90 threshold reported only | 30 Sep 2026 (provisional) |
| D7 | 4 | Model serialisation | MLflow sklearn flavour stored with skops, loaded with a trust list pinned in code | 30 Sep 2026 (provisional) |
| D8 | 4 | Commit `models/champion/`? | Yes: the exact registered artifact plus `lineage.json` | 30 Sep 2026 (provisional) |
| D9 | 5 | CI quality gate | Fail the train job if the champion's hold-out ROC-AUC is below 0.80 | 30 Sep 2026 (provisional) |
| D10 | 5/6 | Push the image to GHCR from CI? | Not yet: CI builds and smoke-tests the image; the cluster uses the locally built image | 30 Sep 2026 (provisional) |

"Provisional" marks a decision taken with the plan's recommendation while I was away, to be
confirmed or changed. Each lives in `configs/train.yaml`, so changing one is a config edit and a
re-run of `make train`.

---

## D0: Originality theme

| Option | What it means |
|---|---|
| **Clinical-grade traceability** | Every prediction can be traced to the model version, the training run, the code commit and the exact dataset bytes |
| Robustness and drift | Lead with external validation and drift monitoring |
| No explicit theme | Personalise each stage independently |

**Choice:** clinical-grade traceability.

**Reason:** I work on GxP-regulated clinical data systems, where a result without an audit trail
is not acceptable. Applying the same standard here gives every graded layer a concrete,
testable requirement instead of a cosmetic difference:

| Layer | Traceability feature |
|---|---|
| Data | `data/raw/MANIFEST.json` with a SHA-256 per source file, verified on every run; a validation gate that rejects bad rows with a reason instead of dropping them silently |
| Training | MLflow tags for git commit, dirty flag, data SHA-256 and library versions |
| Model | Registry alias `champion`, `models/champion/lineage.json`, a model card |
| Serving | `/model-info`, `X-Model-Version` and `X-Request-ID` headers, a JSON audit line per prediction |
| Deployment | Image tagged with the git short SHA; OCI labels and Kubernetes annotations carrying the model version |
| Monitoring | Model-version info metric; prediction-distribution metrics for the drift demonstration |

Drift monitoring is not dropped: it becomes one part of the monitoring stage rather than the theme.

---

## D1: Data scope

The UCI Heart Disease dataset is four hospital databases. Counts from the processed files
(downloaded and checked on 30 Sep 2026):

| Site | Rows | Rows with a missing value | `ca` missing | `thal` missing | `chol = 0` | Disease (`num > 0`) |
|---|---|---|---|---|---|---|
| Cleveland | 303 | 6 | 4 | 2 | 0 | 45.9% |
| Hungary | 294 | 293 | 291 | 266 | 0 | 36.1% |
| Switzerland | 123 | 123 | 118 | 52 | 123 | 93.5% |
| VA Long Beach | 200 | 199 | 198 | 166 | 49 | 74.5% |

| Option | Training data | For | Against |
|---|---|---|---|
| Pool all four sites | 920 rows | Three times the sample | The site predicts the label (Switzerland is 93.5% positive) and missingness reveals the site, so a model can learn *which hospital* instead of physiology; the drift demonstration stops being a real shift |
| **Train on Cleveland, validate externally on the other three** | 303 rows | Clean training data; an honest measure of how the model transfers to other hospitals; the external sites drive the drift demonstration | Small training set, so cross-validation variance is wider |
| Train on Cleveland + Hungary, validate on Switzerland + VA | 597 rows | More training data | Hungary is missing `ca` in 291 of 294 rows, so the missingness shortcut returns |

**Choice:** train and tune on Cleveland; use Hungary, Switzerland and VA Long Beach as external
validation sets.

**Reason:** I first favoured pooling for the larger sample, but pooling mixes site and label:
a model could score well by recognising the hospital from its missing-value pattern. Training
on Cleveland keeps that shortcut out of the training data, and turns the other three sites into
the most useful evidence in the project: how the model behaves on patients from a different
hospital. The same sites then feed the monitoring stage's drift demonstration.

**Consequences:**

| Area | Effect |
|---|---|
| Split | Stratified 80/20 on Cleveland: 242 training rows, 61 locked hold-out rows |
| Reporting | Cross-validation results as mean ± standard deviation; external metrics reported per site, expected to be lower, and discussed as site shift rather than hidden |
| Missing values | Still handled for real: the pipeline imputes `ca`, `thal` and `slope` for external sites, and the API accepts them as null |
| EDA | Covers all four sites, including missingness and distribution shift per site |

---

## D2: Physiologically impossible zeros

`chol = 0` appears in all 123 Swiss rows and in 49 VA rows; `trestbps = 0` appears in one VA
row; Cleveland and Hungary have none. Hungary records missing cholesterol as `?` (23 rows)
instead, so the same fact is coded two different ways across sites.

| Option | Effect |
|---|---|
| **Treat as missing (NaN); impute inside the model pipeline** | The zeros become honest gaps, and the data card reports true missingness per site |
| Keep the literal 0 | 152 of the 172 rows with `chol = 0` (88%) have disease, so the zero would act as a "Switzerland or VA, therefore sick" signal: leakage |
| Drop the affected rows | Removes every Swiss row, a third of the external validation data |

**Choice:** treat zeros in `chol` and `trestbps` as missing in the cleaning step.

**Reason:** a cholesterol or resting blood pressure of zero is not a measurement; it is a site's
code for "not measured". Imputation stays out of `clean.py` and inside the model pipeline, fitted
on training folds only, so no information leaks from validation data into training.

---

## D3: Engineered features (provisional)

| Option | What it means |
|---|---|
| **`hr_pct_pred` as a searched option; no missing indicators** | Add the share of age-predicted maximum heart rate reached, `thalach / (220 − age)`, and let cross-validation decide whether it helps |
| Add missing-value indicators as well | One extra 0/1 column per feature with gaps |
| No engineered features | Use the 13 source features only |

**Choice:** `hr_pct_pred` is a hyperparameter of every grid (`engineer__enabled: [false, true]`),
so each family's search reports whether it helps. No missing-value indicators.

**Reason:** a low share of the age-predicted maximum heart rate (chronotropic incompetence) is a
recognised marker of coronary disease, and searching it costs nothing. Missing-value indicators
were rejected on the data: the Cleveland training set has only 6 rows with a gap (`ca` 4, `thal`
2), too few to learn what "missing" means, while at the external sites nearly every row is missing
`ca`. An indicator learned from 6 rows would inject noise exactly where the model is weakest.

**Result (30 Sep 2026):** cross-validation selected `hr_pct_pred` only for gradient boosting;
logistic regression (the champion) and random forest scored better without it. The feature
stays in the search, so the question is re-asked on every retraining.

The formula:

$$\text{hr\_pct\_pred} = \frac{\text{thalach}}{220 - \text{age}}$$

**Notation:** $\text{thalach}$ = maximum heart rate reached in the exercise test (beats/min);
$\text{age}$ = age (years); $220 - \text{age}$ = conventional age-predicted maximum heart rate
(beats/min); the result is unitless (0.8 means 80% of the predicted maximum was reached).

---

## D4: Model families (provisional)

| Option | What it means |
|---|---|
| **Logistic regression + random forest + histogram gradient boosting** | A linear baseline, a bagged tree ensemble and a boosted tree ensemble |
| Logistic regression + random forest only | The brief's minimum |

**Choice:** all three, each tuned by grid search under stratified 5-fold cross-validation.

**Reason:** the three families fail in different ways (linear vs. interactions, variance vs.
bias), so the comparison is informative; gradient boosting also tests whether more capacity helps
on 242 training rows. scikit-learn 1.9 deprecates `LogisticRegression(penalty=...)`, so the plan's
"L1 or L2" choice is expressed as `l1_ratio ∈ {0, 1}` with the `saga` solver (same models).

---

## D5: Champion selection rule (provisional)

| Option | What it means |
|---|---|
| **Best mean CV ROC-AUC; within 0.01, higher mean CV recall; then the simpler model** | Discrimination first, then the clinically costly error, then parsimony |
| Best mean CV ROC-AUC only | Ignores recall and simplicity |
| Best hold-out score | Would turn the locked hold-out set into a tuning set |

**Choice:** the first. Order of simplicity: logistic regression, gradient boosting, random forest.

**Reason:** ROC-AUC does not depend on the threshold or on prevalence (which ranges from 36% to 94%
across the sites), so it is the fair headline. Differences under 0.01 are inside the fold-to-fold
spread, so they should not decide; recall then decides, because a missed diagnosis costs more than
a false alarm. The hold-out set is never used for selection. Result on 30 Sep 2026: logistic
regression (CV ROC-AUC 0.909) tied with random forest (0.904); logistic regression had the higher
recall (0.819 vs 0.774) and became champion; random forest is the registered challenger.

---

## D6: Decision threshold and risk bands (provisional)

| Option | What it means |
|---|---|
| **Threshold 0.5 for the class; risk bands as an extra; recall-0.90 threshold reported only** | Standard class output, a graded view for the API, and the recall trade-off documented |
| Lower the threshold to reach 90% recall | Fewer missed cases, more false alarms, chosen on 61 hold-out rows |

**Choice:** class = 1 when the probability is at least 0.5; the API also returns a band (low < 0.3
≤ moderate < 0.6 ≤ high); the threshold that reaches 90% hold-out recall is reported in
`reports/threshold_analysis.csv` but not used.

**Reason:** a threshold tuned on 61 hold-out patients would overfit them, and the external sites
show that calibration does not transfer across hospitals anyway. On the hold-out set the 0.5
threshold already gives recall 0.89; reaching 0.90 needs 0.48, a negligible change.

---

## D7: Model serialisation (provisional)

| Option | What it means |
|---|---|
| **skops, via MLflow's sklearn flavour** | A declarative format: loading reconstructs only types on an explicit trust list |
| pickle / cloudpickle (MLflow's older default) | Loading a pickle can execute arbitrary code embedded in the file |
| ONNX | Portable, but needs a converter for the custom feature step and a second runtime |

**Choice:** skops. The five non-default types a heart-risk pipeline contains are pinned in
`heartrisk.model_io.SKOPS_TRUSTED_TYPES`; the loader refuses any other type, and refuses a model
folder whose own declared trust list goes beyond the pinned one.

**Reason:** a model file is an input the API trusts at start-up; with pickle, whoever can replace
the file can run code in the container. MLflow 3.16 supports skops natively, so the choice costs
nothing. `tests/test_model_io.py` shows a file carrying an extra type being refused.

---

## D8: Commit `models/champion/` (provisional)

| Option | What it means |
|---|---|
| **Commit it** | The exact registered artifact (283 KB) with `lineage.json` lives in git |
| Download from the MLflow registry at build time | The build depends on a registry the CI runner cannot reach |

**Choice:** commit it.

**Reason:** the Docker build and CI then work from a clean clone without retraining or network
access to a registry, and git history shows every model change as a reviewable diff of
`lineage.json`. `make package` copies the registered artifact byte for byte and checks that it
reproduces the hold-out ROC-AUC logged at training time before writing anything.

---

## D9: CI quality gate (provisional)

| Option | What it means |
|---|---|
| **Hold-out ROC-AUC ≥ 0.80** | Fails the pipeline on a clearly broken model, with room for normal variation |
| Match the current score (≥ 0.95) | Any retraining noise on 61 hold-out patients would fail CI |
| No gate | A broken model could be packaged silently |

**Choice:** `python -m heartrisk.gate --min-roc-auc 0.80` after `make package`; it also writes the
metrics table into the GitHub job summary.

**Reason:** the gate exists to catch a broken pipeline (wrong labels, a leak removed, a column
shifted), not to police small fluctuations. The current champion scores 0.950, so 0.80 is a safety
net, not a target. External sites are reported but not gated: their shift is a known property of
the data (D1), not a regression.

---

## D10: Push the image to GHCR from CI? (provisional)

| Option | What it means |
|---|---|
| **Not yet** | CI builds and smoke-tests the image on every push; the local cluster runs the image built locally |
| Push on `main` to a private GHCR package | A registry copy of every main-branch image; the cluster would need an image-pull secret |

**Choice:** not yet.

**Reason:** the deployment target is the local Docker Desktop cluster, which uses the local image
store, so a registry adds a secret to manage without changing what is deployed. It is a small,
self-contained addition (`packages: write`, login, push) if a remote cluster is ever used.
