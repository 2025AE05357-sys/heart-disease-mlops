# Design decisions

Each decision records the options I considered, what I chose and why. IDs follow the project
plan (D0–D15); each one is added at the stage that needs it.

## Summary

| ID | Stage | Decision | Choice | Date |
|---|---|---|---|---|
| D0 | 0 | Originality theme | Clinical-grade traceability | 30 Sep 2026 |
| D1 | 1 | Data scope | Train and tune on Cleveland; external validation on Hungary, Switzerland and VA Long Beach | 30 Sep 2026 |
| D2 | 1 | Physiologically impossible zeros (`chol = 0`, `trestbps = 0`) | Treat as missing; impute inside the model pipeline | 30 Sep 2026 |

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
