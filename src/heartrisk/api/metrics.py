"""Prometheus series exposed on /metrics (one uvicorn worker per pod, so no multiprocess mode).

Besides request rate and latency, the model-facing series make drift visible: the mix of
predicted classes, the probability distribution, selected input distributions and how often
inputs had to be imputed.
"""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "heartrisk_http_requests_total",
    "HTTP requests by method, route template and status code",
    ["method", "route", "status"],
    registry=REGISTRY,
)
LATENCY = Histogram(
    "heartrisk_http_request_duration_seconds",
    "Request latency by route template",
    ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=REGISTRY,
)
PREDICTIONS = Counter(
    "heartrisk_predictions_total", "Predictions by predicted label", ["label"], registry=REGISTRY
)
PROBABILITY = Histogram(
    "heartrisk_prediction_probability",
    "Predicted probability of heart disease",
    buckets=tuple(round(0.1 * i, 1) for i in range(1, 11)),
    registry=REGISTRY,
)
INPUT_VALUE = Histogram(
    "heartrisk_input_value",
    "Distribution of selected inputs (drift signals)",
    ["feature"],
    buckets=(30, 40, 50, 60, 70, 80, 100, 120, 140, 160, 180, 200, 250, 300, 400, 600),
    registry=REGISTRY,
)
IMPUTED = Counter(
    "heartrisk_imputed_fields_total",
    "Inputs sent as null and imputed by the pipeline",
    ["field"],
    registry=REGISTRY,
)
VALIDATION_ERRORS = Counter(
    "heartrisk_validation_errors_total", "Requests rejected as invalid (422)", registry=REGISTRY
)
MODEL_INFO = Gauge(
    "heartrisk_model_info",
    "The loaded model (value is always 1)",
    ["name", "version", "run_id", "family"],
    registry=REGISTRY,
)
