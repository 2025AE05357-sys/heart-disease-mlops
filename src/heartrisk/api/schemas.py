"""Request and response contract of the prediction API.

The numeric bounds and code sets mirror configs/data.yaml exactly (a test enforces it), so the
API accepts precisely what the data validation gate accepts.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Row 3 of the Cleveland file (diagnosis num = 1).
EXAMPLE = {
    "age": 67,
    "sex": 1,
    "cp": 4,
    "trestbps": 120,
    "chol": 229,
    "fbs": 0,
    "restecg": 2,
    "thalach": 129,
    "exang": 1,
    "oldpeak": 2.6,
    "slope": 2,
    "ca": 2,
    "thal": 7,
}


class Patient(BaseModel):
    """One patient's measurements.

    Only age, sex and chest-pain type are required (they are recorded at every site). Any other
    value may be null: the model pipeline then imputes it, and the response lists it under
    `imputed_fields` so that the caller knows the prediction rests on an estimate.
    """

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, json_schema_extra={"examples": [EXAMPLE]}
    )

    age: int = Field(ge=18, le=100, description="Age (years)")
    sex: Literal[0, 1] = Field(description="Sex: 0 = female, 1 = male")
    cp: Literal[1, 2, 3, 4] = Field(
        description="Chest-pain type: 1 typical angina, 2 atypical angina, 3 non-anginal pain, "
        "4 asymptomatic"
    )
    trestbps: float | None = Field(
        None, ge=60, le=260, description="Resting blood pressure (mm Hg)"
    )
    chol: float | None = Field(None, ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] | None = Field(
        None, description="Fasting blood sugar > 120 mg/dl: 0 = no, 1 = yes"
    )
    restecg: Literal[0, 1, 2] | None = Field(
        None,
        description="Resting ECG: 0 normal, 1 ST-T abnormality, 2 left-ventricular hypertrophy",
    )
    thalach: float | None = Field(
        None, ge=50, le=230, description="Maximum heart rate achieved (beats/min)"
    )
    exang: Literal[0, 1] | None = Field(
        None, description="Exercise-induced angina: 0 = no, 1 = yes"
    )
    oldpeak: float | None = Field(
        None, ge=-3.0, le=7.0, description="ST depression, exercise relative to rest (mm)"
    )
    slope: Literal[1, 2, 3] | None = Field(
        None, description="Slope of the peak-exercise ST segment: 1 up, 2 flat, 3 down"
    )
    ca: Literal[0, 1, 2, 3] | None = Field(
        None, description="Major vessels coloured by fluoroscopy (0-3)"
    )
    thal: Literal[3, 6, 7] | None = Field(
        None, description="Thallium stress test: 3 normal, 6 fixed defect, 7 reversible defect"
    )


class ModelRef(BaseModel):
    """Which model answered: registry name, version, MLflow run and model family."""

    name: str
    version: int
    run_id: str
    family: str


class Scored(BaseModel):
    """The model's answer for one patient."""

    prediction: Literal[0, 1] = Field(description="1 = heart disease predicted")
    label: Literal["no_disease", "disease"]
    probability: float = Field(description="Estimated probability of heart disease")
    confidence: float = Field(description="Probability of the predicted class: max(p, 1 - p)")
    risk_band: Literal["low", "moderate", "high"] = Field(
        description="low < 0.3 <= moderate < 0.6 <= high (a reading aid, not a clinical score)"
    )
    imputed_fields: list[str] = Field(description="Inputs sent as null and imputed by the model")


class PredictionResponse(Scored):
    """Response of POST /predict."""

    request_id: str
    model: ModelRef


class BatchRequest(BaseModel):
    """Up to 100 patients scored in one call."""

    model_config = ConfigDict(extra="forbid")
    patients: list[Patient] = Field(min_length=1, max_length=100)


class BatchResponse(BaseModel):
    """Response of POST /predict/batch; predictions keep the order of the request."""

    request_id: str
    model: ModelRef
    predictions: list[Scored]
