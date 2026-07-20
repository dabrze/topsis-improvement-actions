import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DASHBOARD_DATA_DIR = REPO_ROOT.parent / "topsis-postfactum-dashboard" / "data"
BASELINE_REVISION = "e43b384e0c90684063e76ddcff7cdad9db628e23"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import wmsd as current_module  # noqa: E402


DATASET_CASES = {
    "students": {
        "csv": "students.csv",
        "settings": "students_settings.json",
        "sep": ",",
    },
    "bus": {
        "csv": "bus.csv",
        "settings": "bus_settings.json",
        "sep": ";",
    },
}


def load_dataset(case_name):
    case = DATASET_CASES[case_name]
    df = pd.read_csv(DASHBOARD_DATA_DIR / case["csv"], index_col=0, sep=case["sep"])
    with open(DASHBOARD_DATA_DIR / case["settings"], "r", encoding="utf-8") as handle:
        params = json.load(handle)

    criteria_columns = [
        name for name, cfg in params.items() if cfg.get("id_column") != "true"
    ]
    return {
        "decision_df": df[criteria_columns],
        "criteria_columns": criteria_columns,
        "weights": [params[name]["weight"] for name in criteria_columns],
        "objectives": [params[name]["objective"] for name in criteria_columns],
        "expert_ranges": [
            [params[name]["expert_min"], params[name]["expert_max"]]
            for name in criteria_columns
        ],
    }


def build_transformer(module, agg_name, dataset):
    agg_class = getattr(module, agg_name)
    transformer = module.WMSDTransformer(agg_class)
    transformed = transformer.fit_transform(
        dataset["decision_df"].copy(),
        dataset["weights"],
        dataset["objectives"],
        dataset["expert_ranges"],
    )
    return transformer, transformed


@pytest.fixture(scope="session")
def baseline_module():
    source = subprocess.run(
        ["git", "show", f"{BASELINE_REVISION}:src/WMSDTransformer.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    temp_dir = tempfile.TemporaryDirectory()
    module_path = Path(temp_dir.name) / "WMSDTransformer_baseline.py"
    module_path.write_text(source, encoding="utf-8")

    spec = importlib.util.spec_from_file_location(
        "WMSDTransformer_baseline",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    yield module

    temp_dir.cleanup()


@pytest.fixture(scope="session")
def current_wmsd_module():
    return current_module


@pytest.fixture
def dataset(request):
    return load_dataset(request.param)
