"""The scoring service: the exported pipeline plus its lineage, loaded once at start-up."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from heartrisk import model_io
from heartrisk.api.schemas import Patient
from heartrisk.columns import FEATURES


class ModelService:
    """Loads models/champion (skops pipeline + lineage.json) and scores patients."""

    def __init__(self, model_dir: Path):
        self.pipeline = model_io.load_model(model_dir)
        self.lineage: dict[str, Any] = json.loads(
            (model_dir / "lineage.json").read_text(encoding="utf-8")
        )
        model = self.lineage["model"]
        self.ref = {k: model[k] for k in ("name", "version", "run_id", "family")}
        self.threshold: float = self.lineage["decision"]["threshold"]
        self.bands: dict[str, float] = self.lineage["decision"]["risk_bands"]

    def score(self, patients: list[Patient]) -> list[dict[str, Any]]:
        """Class, probability, confidence, risk band and imputed fields for each patient."""
        rows = [p.model_dump() for p in patients]
        frame = pd.DataFrame(rows, columns=FEATURES).astype("float64")  # None -> NaN
        probabilities = self.pipeline.predict_proba(frame)[:, 1]
        results = []
        for row, probability in zip(rows, probabilities, strict=True):
            p = float(probability)
            prediction = int(p >= self.threshold)
            results.append(
                {
                    "prediction": prediction,
                    "label": "disease" if prediction else "no_disease",
                    "probability": p,
                    "confidence": max(p, 1 - p),
                    "risk_band": model_io.risk_band(p, **self.bands),
                    "imputed_fields": [f for f in FEATURES if row[f] is None],
                }
            )
        return results
