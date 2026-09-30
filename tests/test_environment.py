"""Environment gate: fails fast if the runtime drifts from the pinned setup."""

import importlib
import sys

import pytest

import heartrisk


def test_python_is_312():
    assert sys.version_info[:2] == (3, 12), f"expected Python 3.12, got {sys.version}"


@pytest.mark.parametrize("module", ["numpy", "pandas", "pandera", "sklearn", "mlflow", "yaml"])
def test_core_dependencies_import(module):
    importlib.import_module(module)


def test_package_version_is_set():
    assert heartrisk.__version__
