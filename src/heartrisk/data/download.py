"""Checksum-verified download of the UCI Heart Disease files.

`data/raw/MANIFEST.json` pins the SHA-256 of every source file on first download. Later runs
reuse local files that still match it, re-fetch damaged ones, and refuse upstream bytes that
differ from the pinned digest, so the training data cannot change without a visible diff.

    python -m heartrisk.data.download            # download (idempotent)
    python -m heartrisk.data.download --verify   # re-hash local files, exit 1 on any mismatch
"""

import argparse
import hashlib
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from heartrisk.config import load_config

log = logging.getLogger(__name__)

MANIFEST_NAME = "MANIFEST.json"


class DataIntegrityError(RuntimeError):
    """Downloaded bytes differ from the digest pinned in the manifest."""


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, timeout: float = 30.0, retries: int = 3, backoff: float = 2.0) -> bytes:
    """GET `url`, retrying transient network errors with linear backoff."""
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise
            log.warning("fetch %s failed (%s); retry %d/%d", url, exc, attempt, retries - 1)
            time.sleep(backoff * attempt)
    raise AssertionError("unreachable")  # pragma: no cover


def load_manifest(raw_dir: Path) -> dict[str, Any]:
    """The manifest in `raw_dir`, or an empty dict if there is none yet."""
    path = raw_dir / MANIFEST_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def download_all(
    config: dict[str, Any], raw_dir: Path, fetcher: Callable[[str], bytes] = fetch
) -> dict[str, Any]:
    """Download every configured site into `raw_dir` and write the manifest."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    source = config["source"]
    pinned = {entry["file"]: entry for entry in load_manifest(raw_dir).get("files", [])}
    entries = []
    for site, spec in config["sites"].items():
        name = f"{site}.data"
        path = raw_dir / name
        url = source["base_url"] + spec["file"]
        known = pinned.get(name)
        if known and path.exists() and sha256_bytes(path.read_bytes()) == known["sha256"]:
            log.info("%-17s present   sha256 %s…  (checksum OK)", name, known["sha256"][:16])
            entries.append(known)
            continue
        if known and path.exists():
            log.warning("%s does not match the manifest; re-downloading", name)
        data = fetcher(url)
        digest = sha256_bytes(data)
        if known and digest != known["sha256"]:
            raise DataIntegrityError(
                f"{name}: upstream bytes changed (manifest {known['sha256'][:16]}…, "
                f"downloaded {digest[:16]}…); refusing to replace pinned data"
            )
        path.write_bytes(data)
        log.info("%-17s download  sha256 %s…  %6d bytes", name, digest[:16], len(data))
        entries.append(
            {
                "site": site,
                "file": name,
                "url": url,
                "bytes": len(data),
                "sha256": digest,
                "retrieved_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
    manifest = {
        "dataset": source["name"],
        "doi": source["doi"],
        "licence": source["licence"],
        "citation": source["citation"],
        "files": entries,
    }
    text = json.dumps(manifest, indent=2) + "\n"
    (raw_dir / MANIFEST_NAME).write_text(text, encoding="utf-8")
    return manifest


def verify(config: dict[str, Any], raw_dir: Path) -> list[str]:
    """Re-hash local files against the manifest; returns one message per problem."""
    manifest = load_manifest(raw_dir)
    if not manifest:
        return [f"{MANIFEST_NAME} not found in {raw_dir}"]
    listed = {entry["site"] for entry in manifest["files"]}
    problems = [
        f"{site}: not listed in {MANIFEST_NAME}" for site in config["sites"] if site not in listed
    ]
    for entry in manifest["files"]:
        path = raw_dir / entry["file"]
        if not path.exists():
            problems.append(f"{entry['file']}: missing")
            continue
        digest = sha256_bytes(path.read_bytes())
        if digest != entry["sha256"]:
            problems.append(
                f"{entry['file']}: sha256 {digest[:16]}… does not match manifest "
                f"{entry['sha256'][:16]}…"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml", help="data config (YAML)")
    parser.add_argument("--raw-dir", help="override paths.raw_dir from the config")
    parser.add_argument("--verify", action="store_true", help="only re-hash local files")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    config = load_config(args.config)
    raw_dir = Path(args.raw_dir or config["paths"]["raw_dir"])
    if not args.verify:
        download_all(config, raw_dir)
        log.info("manifest written: %s", raw_dir / MANIFEST_NAME)
    problems = verify(config, raw_dir)
    for problem in problems:
        log.error("verify: %s", problem)
    if problems:
        return 1
    log.info("verify: %d files match %s", len(config["sites"]), MANIFEST_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
