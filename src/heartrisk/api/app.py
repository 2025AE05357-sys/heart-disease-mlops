"""FastAPI prediction service.

Every response carries `X-Request-ID` (propagated from the caller or generated) and
`X-Model-Version`; every prediction writes one JSON audit line; /metrics exposes Prometheus
series. The model loads once at start-up; if it cannot, the process stays up and /ready returns
503, so Kubernetes keeps the pod out of the Service instead of restarting it in a loop.

    uvicorn heartrisk.api.app:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from heartrisk import __version__
from heartrisk.api import audit, metrics
from heartrisk.api.model import ModelService
from heartrisk.api.schemas import BatchRequest, BatchResponse, Patient, PredictionResponse

log = logging.getLogger("heartrisk.api")

SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
DRIFT_FEATURES = ("age", "chol", "thalach")
QUIET_PATHS = {"/metrics", "/health", "/ready"}  # probes and scrapes: counted, not access-logged
DISCLAIMER = "Educational project: not a medical device and not medical advice."


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "false").strip().lower() in {"1", "true", "yes"}


def create_app(model_dir: str | Path | None = None, audit_inputs: bool | None = None) -> FastAPI:
    """Build the app. MODEL_DIR, AUDIT_LOG_INPUTS and LOG_LEVEL come from the environment."""
    model_dir = Path(model_dir or os.environ.get("MODEL_DIR", "models/champion"))
    audit_inputs = _env_flag("AUDIT_LOG_INPUTS") if audit_inputs is None else audit_inputs
    audit.configure_logging(os.environ.get("LOG_LEVEL", "INFO"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.service = ModelService(model_dir)
            ref = app.state.service.ref
            metrics.MODEL_INFO.labels(**{k: str(v) for k, v in ref.items()}).set(1)
            log.info("model loaded", extra={"payload": {"event": "model_loaded", **ref}})
        except Exception as exc:  # stay up; /ready reports 503 until a valid model is mounted
            app.state.service = None
            log.error(
                "model not loaded",
                extra={
                    "payload": {
                        "event": "model_load_failed",
                        "model_dir": str(model_dir),
                        "error": repr(exc),
                    }
                },
            )
        yield

    app = FastAPI(
        title="Heart-disease risk API",
        version=__version__,
        description=f"Predicts heart-disease risk from 13 clinical measurements. {DISCLAIMER}",
        lifespan=lifespan,
    )
    app.state.service = None

    @app.middleware("http")
    async def trace_and_measure(request: Request, call_next):
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if SAFE_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        route = getattr(request.scope.get("route"), "path", "unmatched")
        response.headers["X-Request-ID"] = request_id
        if request.app.state.service is not None:
            response.headers["X-Model-Version"] = str(request.app.state.service.ref["version"])
        metrics.REQUESTS.labels(request.method, route, str(response.status_code)).inc()
        metrics.LATENCY.labels(route).observe(elapsed)
        if route not in QUIET_PATHS:
            audit.ACCESS.info(
                "request",
                extra={
                    "payload": {
                        "event": "access",
                        "request_id": request_id,
                        "method": request.method,
                        "path": route,
                        "status": response.status_code,
                        "latency_ms": round(elapsed * 1000, 2),
                    }
                },
            )
        return response

    @app.exception_handler(RequestValidationError)
    async def on_invalid_input(request: Request, exc: RequestValidationError):
        metrics.VALIDATION_ERRORS.inc()
        return await request_validation_exception_handler(request, exc)

    def service(request: Request) -> ModelService:
        if request.app.state.service is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return request.app.state.service

    def score(request: Request, patients: list[Patient]) -> list[dict[str, Any]]:
        svc = service(request)
        start = time.perf_counter()
        results = svc.score(patients)
        per_patient_ms = (time.perf_counter() - start) * 1000 / len(patients)
        for index, (patient, result) in enumerate(zip(patients, results, strict=True)):
            values = patient.model_dump()
            metrics.PREDICTIONS.labels(result["label"]).inc()
            metrics.PROBABILITY.observe(result["probability"])
            for feature in DRIFT_FEATURES:
                if values[feature] is not None:
                    metrics.INPUT_VALUE.labels(feature).observe(values[feature])
            for field in result["imputed_fields"]:
                metrics.IMPUTED.labels(field).inc()
            record = {
                "event": "prediction",
                "request_id": request.state.request_id,
                "item": index,
                "model_version": svc.ref["version"],
                "input_sha256": audit.input_sha256(values),
                "prediction": result["prediction"],
                "probability": round(result["probability"], 6),
                "risk_band": result["risk_band"],
                "imputed_fields": result["imputed_fields"],
                "latency_ms": round(per_patient_ms, 3),
            }
            if audit_inputs:
                record["input"] = values
            audit.AUDIT.info("prediction", extra={"payload": record})
        return results

    @app.get("/", summary="Service overview")
    def root(request: Request) -> dict[str, Any]:
        svc = request.app.state.service
        return {
            "name": app.title,
            "version": __version__,
            "model": svc.ref if svc else None,
            "endpoints": [
                "/health",
                "/ready",
                "/predict",
                "/predict/batch",
                "/model-info",
                "/metrics",
                "/docs",
            ],
            "disclaimer": DISCLAIMER,
        }

    @app.get("/health", summary="Liveness: the process is up")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", summary="Readiness: a model is loaded")
    def ready(request: Request):
        svc = request.app.state.service
        if svc is None:
            return JSONResponse({"status": "not ready", "reason": "model not loaded"}, 503)
        return {"status": "ready", "model_version": svc.ref["version"]}

    @app.post("/predict", response_model=PredictionResponse, summary="Score one patient")
    def predict(patient: Patient, request: Request) -> dict[str, Any]:
        result = score(request, [patient])[0]
        return {"request_id": request.state.request_id, "model": service(request).ref, **result}

    @app.post("/predict/batch", response_model=BatchResponse, summary="Score up to 100 patients")
    def predict_batch(batch: BatchRequest, request: Request) -> dict[str, Any]:
        results = score(request, batch.patients)
        return {
            "request_id": request.state.request_id,
            "model": service(request).ref,
            "predictions": results,
        }

    @app.get("/model-info", summary="Lineage of the loaded model")
    def model_info(request: Request) -> dict[str, Any]:
        return service(request).lineage

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.REGISTRY), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
