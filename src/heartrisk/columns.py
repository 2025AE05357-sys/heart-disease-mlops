"""Column names and clinical wording shared by data, model, API and report.

Plain constants with no dependencies, so that the serving image can import the model's
feature code without pulling in the data-validation stack.
"""

RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]  # fmt: skip
FEATURES = RAW_COLUMNS[:-1]
FLOAT_COLUMNS = ["oldpeak"]
INTEGER_FEATURES = [c for c in FEATURES if c not in FLOAT_COLUMNS]
CLEAN_COLUMNS = ["record_id", "site", *FEATURES, "num_raw", "target"]

SITE_LABELS = {
    "cleveland": "Cleveland",
    "hungary": "Hungary",
    "switzerland": "Switzerland",
    "va": "VA Long Beach",
}

# Human-readable meaning of each column and code (UCI documentation), shared by the EDA,
# the report and the API so that every surface uses the same clinical wording.
DESCRIPTIONS = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest-pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG",
    "thalach": "Maximum heart rate (beats/min)",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression, exercise vs rest (mm)",
    "slope": "Slope of peak-exercise ST segment",
    "ca": "Major vessels coloured by fluoroscopy",
    "thal": "Thallium stress test",
}
CODE_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"},
    "fbs": {0: "No", 1: "Yes"},
    "restecg": {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}
