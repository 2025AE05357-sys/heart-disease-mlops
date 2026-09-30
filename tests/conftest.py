"""Shared fixtures: the real data config sized to the hand-made raw files in tests/fixtures/raw."""

from pathlib import Path

import pytest

from heartrisk.config import load_config
from heartrisk.data.download import download_all

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE_ROWS = {"cleveland": 3, "hungary": 2, "switzerland": 2, "va": 2}


@pytest.fixture
def data_config():
    """configs/data.yaml with expected row counts matching the fixture files."""
    config = load_config(REPO / "configs" / "data.yaml")
    for site, rows in FIXTURE_ROWS.items():
        config["sites"][site]["expected_rows"] = rows
    return config


@pytest.fixture
def make_raw_dir(tmp_path, data_config):
    """Build a raw dir from the fixtures plus extra lines, pinned by the real downloader."""

    def _make(extra_lines=None, name="raw"):
        extra_lines = extra_lines or {}
        base = data_config["source"]["base_url"]
        payloads = {}
        for site, spec in data_config["sites"].items():
            data = (FIXTURES / "raw" / f"{site}.data").read_bytes()
            data += "".join(f"{line}\n" for line in extra_lines.get(site, [])).encode()
            payloads[base + spec["file"]] = data
        target = tmp_path / name
        download_all(data_config, target, fetcher=payloads.__getitem__)
        return target

    return _make


@pytest.fixture
def raw_dir(make_raw_dir):
    """The unmodified fixture files, with a matching MANIFEST.json."""
    return make_raw_dir()
