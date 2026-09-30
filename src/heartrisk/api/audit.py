"""JSON-lines logging: one access record per request, one audit record per prediction.

Audit records identify an input by the SHA-256 of its canonical JSON, not by its values, so the
log can prove which input produced which answer without storing clinical data. Raw inputs are
added only when AUDIT_LOG_INPUTS=true.
"""

import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

ACCESS = logging.getLogger("heartrisk.access")
AUDIT = logging.getLogger("heartrisk.audit")


class JsonFormatter(logging.Formatter):
    """One JSON object per line: timestamp, level, logger, then the record's payload."""

    def format(self, record: logging.LogRecord) -> str:
        created = datetime.fromtimestamp(record.created, UTC)
        body: dict[str, Any] = {
            "ts": created.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname.lower(),
            "logger": record.name,
        }
        payload = getattr(record, "payload", None)
        body.update(payload if isinstance(payload, dict) else {"message": record.getMessage()})
        return json.dumps(body, separators=(",", ":"), default=str)


def configure_logging(level: str = "INFO") -> None:
    """Send the `heartrisk` loggers to stdout as JSON lines (idempotent)."""
    logger = logging.getLogger("heartrisk")
    if not any(getattr(h, "heartrisk_json", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        handler.heartrisk_json = True
        logger.addHandler(handler)
    logger.setLevel(level)


def input_sha256(values: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON of an input (sorted keys, no whitespace)."""
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
