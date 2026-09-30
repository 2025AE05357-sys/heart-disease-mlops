"""Checksum-verified download: manifest contents, idempotency, tamper detection, retries.

No network: the fetcher is injected, and urlopen is replaced only in the retry tests.
"""

import hashlib
import json
import urllib.error

import pytest
import yaml

from heartrisk.data import download as dl

# NIST test vector: SHA-256 of b"abc".
ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

CONFIG = {
    "source": {
        "name": "Test dataset",
        "base_url": "https://example.test/hd/",
        "doi": "10.0000/test",
        "licence": "CC BY 4.0",
        "citation": "Test citation (1989).",
    },
    "sites": {
        "cleveland": {"file": "processed.cleveland.data", "expected_rows": 1},
        "va": {"file": "processed.va.data", "expected_rows": 1},
    },
}
CLEVELAND_URL = "https://example.test/hd/processed.cleveland.data"
VA_URL = "https://example.test/hd/processed.va.data"


class RecordingFetcher:
    """Returns fixed bytes per URL and records every call."""

    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        return self.payloads[url]


@pytest.fixture
def fetcher():
    return RecordingFetcher({CLEVELAND_URL: b"abc", VA_URL: b"xyz\n"})


def _manifest(raw_dir):
    return json.loads((raw_dir / "MANIFEST.json").read_text())


def test_download_writes_site_files_and_records_url_size_and_sha256(tmp_path, fetcher):
    dl.download_all(CONFIG, tmp_path, fetcher=fetcher)

    assert (tmp_path / "cleveland.data").read_bytes() == b"abc"
    assert (tmp_path / "va.data").read_bytes() == b"xyz\n"
    entries = {e["file"]: e for e in _manifest(tmp_path)["files"]}
    assert entries["cleveland.data"]["sha256"] == ABC_SHA256
    assert entries["cleveland.data"]["bytes"] == 3
    assert entries["cleveland.data"]["url"] == CLEVELAND_URL
    assert entries["va.data"]["sha256"] == hashlib.sha256(b"xyz\n").hexdigest()
    assert entries["va.data"]["site"] == "va"


def test_manifest_carries_licence_and_citation(tmp_path, fetcher):
    dl.download_all(CONFIG, tmp_path, fetcher=fetcher)

    manifest = _manifest(tmp_path)
    assert manifest["licence"] == "CC BY 4.0"
    assert manifest["citation"] == "Test citation (1989)."
    assert manifest["doi"] == "10.0000/test"


def test_second_run_reuses_verified_files_without_refetching(tmp_path, fetcher):
    dl.download_all(CONFIG, tmp_path, fetcher=fetcher)
    first = _manifest(tmp_path)

    dl.download_all(CONFIG, tmp_path, fetcher=fetcher)

    assert fetcher.calls == [CLEVELAND_URL, VA_URL]  # no new calls on the second run
    assert _manifest(tmp_path) == first  # retrieval timestamps preserved


def test_locally_corrupted_file_is_refetched_when_source_is_unchanged(tmp_path, fetcher):
    dl.download_all(CONFIG, tmp_path, fetcher=fetcher)
    (tmp_path / "cleveland.data").write_bytes(b"abd")

    dl.download_all(CONFIG, tmp_path, fetcher=fetcher)

    assert (tmp_path / "cleveland.data").read_bytes() == b"abc"
    assert fetcher.calls.count(CLEVELAND_URL) == 2


def test_changed_source_bytes_are_refused_and_not_written(tmp_path, fetcher):
    dl.download_all(CONFIG, tmp_path, fetcher=fetcher)
    (tmp_path / "cleveland.data").unlink()
    fetcher.payloads[CLEVELAND_URL] = b"abd"  # upstream file silently changed

    with pytest.raises(dl.DataIntegrityError, match="cleveland.data"):
        dl.download_all(CONFIG, tmp_path, fetcher=fetcher)
    assert not (tmp_path / "cleveland.data").exists()


def test_verify_is_clean_right_after_download(tmp_path, fetcher):
    dl.download_all(CONFIG, tmp_path, fetcher=fetcher)

    assert dl.verify(CONFIG, tmp_path) == []


def test_verify_reports_a_tampered_file_by_name(tmp_path, fetcher):
    dl.download_all(CONFIG, tmp_path, fetcher=fetcher)
    (tmp_path / "cleveland.data").write_bytes(b"abd")  # same size, different bytes

    problems = dl.verify(CONFIG, tmp_path)

    assert len(problems) == 1
    assert "cleveland.data" in problems[0]


def test_verify_reports_missing_file_and_unlisted_site(tmp_path, fetcher):
    dl.download_all(CONFIG, tmp_path, fetcher=fetcher)
    (tmp_path / "va.data").unlink()
    config = {**CONFIG, "sites": {**CONFIG["sites"], "hungary": {"file": "h.data"}}}

    problems = dl.verify(config, tmp_path)

    assert any("va.data" in p and "missing" in p for p in problems)
    assert any("hungary" in p for p in problems)


def test_verify_without_manifest_is_a_problem(tmp_path):
    assert dl.verify(CONFIG, tmp_path) == [f"MANIFEST.json not found in {tmp_path}"]


class FakeResponse:
    """Stands in for http.client.HTTPResponse: a context manager with read()."""

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


def test_fetch_retries_transient_errors_then_returns_the_body(monkeypatch):
    attempts = []

    def flaky_urlopen(url, timeout):
        attempts.append(url)
        if len(attempts) < 3:
            raise urllib.error.URLError("temporary failure")
        return FakeResponse(b"payload")

    monkeypatch.setattr(dl.urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)

    assert dl.fetch("https://example.test/x", retries=3) == b"payload"
    assert len(attempts) == 3


def test_fetch_gives_up_after_the_last_retry(monkeypatch):
    attempts = []

    def failing_urlopen(url, timeout):
        attempts.append(url)
        raise urllib.error.URLError("down")

    monkeypatch.setattr(dl.urllib.request, "urlopen", failing_urlopen)
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)

    with pytest.raises(urllib.error.URLError):
        dl.fetch("https://example.test/x", retries=3)
    assert len(attempts) == 3


def test_cli_verify_exits_nonzero_on_tampered_file(tmp_path, fetcher):
    config_path = tmp_path / "data.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    raw_dir = tmp_path / "raw"
    dl.download_all(CONFIG, raw_dir, fetcher=fetcher)
    args = ["--config", str(config_path), "--raw-dir", str(raw_dir), "--verify"]

    assert dl.main(args) == 0
    (raw_dir / "va.data").write_bytes(b"tampered")
    assert dl.main(args) == 1
