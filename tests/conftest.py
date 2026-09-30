"""Shared fixtures: the real data config sized to the hand-made raw files in tests/fixtures/raw."""

from pathlib import Path

import pytest

from heartrisk import train
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


@pytest.fixture(scope="session")
def fast_config(tmp_path_factory):
    """configs/train.yaml shrunk to run in seconds on tests/fixtures/heart_sample.csv.

    The sample is every 4th Cleveland row plus 15 rows from each external site.
    """
    tmp = tmp_path_factory.mktemp("train")
    config = train.load_train_config(REPO / "configs" / "train.yaml")
    config["n_jobs"] = 1  # process start-up costs more than it saves on 2-candidate grids
    config["data"]["clean_file"] = str(FIXTURES / "heart_sample.csv")
    config["data"]["manifest"] = str(tmp / "no-manifest.json")
    config["cv"]["folds"] = 3
    config["outputs"]["reports_dir"] = str(tmp / "reports")
    config["models"] = {
        "logistic_regression": {"grid": {"model__C": [1.0]}},
        "random_forest": {"grid": {"model__n_estimators": [25], "model__max_depth": [3]}},
        "hist_gradient_boosting": {"grid": {"model__max_leaf_nodes": [7]}},
    }
    config["features"]["hr_reserve"] = [False, True]  # 2 candidates per family
    return config


@pytest.fixture(scope="session")
def trained(fast_config, tmp_path_factory):
    """One fast training run for the whole session: (result, config, tracking URI)."""
    uri = f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'mlflow.db'}"
    return train.run(fast_config, tracking_uri=uri), fast_config, uri
