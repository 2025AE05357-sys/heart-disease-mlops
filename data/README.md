# Data

## Source

| Field | Value |
|---|---|
| Dataset | UCI Heart Disease (dataset 45): four hospital databases, 14 attributes each |
| Files used | `processed.cleveland.data`, `processed.hungarian.data`, `processed.switzerland.data`, `processed.va.data` from `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/` |
| Fallback | `https://archive.ics.uci.edu/static/public/45/heart+disease.zip` (same files, zipped) |
| DOI | https://doi.org/10.24432/C52P4X |
| Licence | CC BY 4.0: reuse and redistribution allowed with attribution, which is why the files are committed here |
| Citation | Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X |
| Intro paper | Detrano, R. et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304–310 |

## Layout

| Path | Content | Produced by |
|---|---|---|
| `raw/<site>.data` | The source files, byte for byte, renamed by site | `make data` |
| `raw/MANIFEST.json` | URL, size, SHA-256 and retrieval time of every raw file, plus licence and citation | `make data` |
| `processed/heart_clean.csv` | 920 validated rows: `record_id`, `site`, 13 features, `num_raw`, `target` | `make prepare` |
| `processed/rejected.csv` | Rows the validation gate refused, each with its reason (none for the current data) | `make prepare` |
| `processed/validation_report.json` | Per-site counts, recoded zeros, duplicates, missingness, input and output SHA-256 | `make prepare` |
| `processed/DATA_CARD.md` | Human-readable summary of the above | `make prepare` |

`record_id` is `<site>-<line>`, the 1-based line number in the raw file, so every clean row can be
traced back to the exact source line.

## Re-download and verify

```bash
make data          # downloads only files that are missing or damaged; refuses changed upstream bytes
make verify-data   # re-hashes data/raw against MANIFEST.json; exits 1 on any mismatch
make prepare       # validation gate + cleaning; exits 1 if the gate fails
```

The manifest pins the SHA-256 of each source file. If UCI ever changed a file, `make data` would
stop with an error instead of silently training on different data; accepting new bytes is a
deliberate act (delete the file and its manifest entry, re-run, review the diff, commit).

## Cleaning rules (`configs/data.yaml`)

| Rule | Detail |
|---|---|
| Missing values | `?` in the source becomes a missing value |
| "Not measured" zeros (decision D2) | `chol = 0` (all 123 Swiss rows, 49 VA rows) and `trestbps = 0` (1 VA row) become missing |
| Plausibility ranges | age 18–100 years; resting BP 60–260 mm Hg; cholesterol 80–700 mg/dl; max heart rate 50–230 beats/min; ST depression −3 to 7 mm |
| Allowed codes | `sex` {0,1}, `cp` {1–4}, `fbs` {0,1}, `restecg` {0,1,2}, `exang` {0,1}, `slope` {1,2,3}, `ca` {0–3}, `thal` {3,6,7}, `num` {0–4} |
| Target | `target = 1` if `num > 0` (any severity), else `0`; the original value is kept as `num_raw` |
| Gate | A row breaking any rule goes to `rejected.csv` with a reason. A site fails the run if its row count differs from the config or more than 5% of its rows are rejected |
| Not done here | Imputation and scaling: both belong in the model pipeline, fitted on training folds only |

> Educational project: not a medical device and not medical advice.
