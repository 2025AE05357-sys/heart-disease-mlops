"""Serving-safe model utilities: load an exported model and map probabilities to risk bands.

Deliberately free of MLflow, matplotlib and training code, so that the API image stays small.

Models are stored with skops (decision D7), not pickle: loading a pickle can execute arbitrary
code, whereas skops only reconstructs types it has been told to trust. The trusted list is pinned
here, in code; the list a model folder declares about itself is checked against it, never
believed.
"""

from pathlib import Path

import skops.io as sio
import yaml

# The only non-default types a heart-risk pipeline contains (checked with
# skops.io.get_untrusted_types on a fitted pipeline of every model family).
SKOPS_TRUSTED_TYPES = [
    "heartrisk.features.HeartRateReserve",
    "numpy.dtype",
    "sklearn.compose._column_transformer.make_column_selector",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.tree._tree.Tree",
]


def load_model(model_dir: str | Path):
    """Load the scikit-learn pipeline from an exported MLflow model folder (skops format)."""
    model_dir = Path(model_dir)
    flavor = yaml.safe_load((model_dir / "MLmodel").read_text(encoding="utf-8"))["flavors"][
        "sklearn"
    ]
    if flavor.get("serialization_format") != "skops":
        raise ValueError(
            f"{model_dir}: serialization format {flavor.get('serialization_format')!r} refused; "
            "only skops models are loaded"
        )
    declared = set(flavor.get("skops_trusted_types") or [])
    extra = sorted(declared - set(SKOPS_TRUSTED_TYPES))
    if extra:
        raise ValueError(f"{model_dir}: model declares types outside the trusted list: {extra}")
    return sio.load(model_dir / flavor["pickled_model"], trusted=SKOPS_TRUSTED_TYPES)


def risk_band(probability: float, low: float, high: float) -> str:
    """'low' below `low`, 'high' at or above `high`, otherwise 'moderate' (decision D6)."""
    if probability < low:
        return "low"
    return "high" if probability >= high else "moderate"
