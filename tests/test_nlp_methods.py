import numpy as np
import pandas as pd
import pytest

from conftest import build_transformer, load_dataset


METHOD_CASES = {
    "RTOPSIS": {"dataset": "students", "source_rank": 4, "target_rank": 2},
    "SAW": {"dataset": "students", "source_rank": 3, "target_rank": 2},
    "ARAS": {"dataset": "students", "source_rank": 4, "target_rank": 2},
    "COPRAS": {"dataset": "students", "source_rank": 3, "target_rank": 2},
    "WASPAS": {"dataset": "students", "source_rank": 3, "target_rank": 2},
}


def _apply_modification(transformer, transformed, source_rank, result_df):
    source_id = transformer._ranked_alternatives[source_rank - 1]
    source_row = transformer.X.loc[source_id].copy()
    modified_row = (source_row + result_df.iloc[0]).to_frame().T
    modified_row.index = [source_id]
    return transformed.loc[source_id], transformer.transform(modified_row).iloc[0]


@pytest.mark.parametrize("method_name", list(METHOD_CASES))
def test_supported_methods_find_exact_nlp_improvements(method_name, current_wmsd_module):
    case = METHOD_CASES[method_name]
    dataset = load_dataset(case["dataset"])
    transformer, transformed = build_transformer(current_wmsd_module, method_name, dataset)

    result = transformer.improvement(
        "improvement_non_linear_programming",
        case["source_rank"],
        case["target_rank"],
        1e-4,
        features_to_change=dataset["criteria_columns"],
    )

    assert result is not None
    assert not result.empty

    source_row = transformed.loc[transformer._ranked_alternatives[case["source_rank"] - 1]]
    target_row = transformed.loc[transformer._ranked_alternatives[case["target_rank"] - 1]]
    _, updated_row = _apply_modification(
        transformer, transformed, case["source_rank"], result
    )

    assert updated_row[str(transformer.agg_fn.letter)] >= target_row[str(transformer.agg_fn.letter)] - 1e-6
    assert updated_row[str(transformer.agg_fn.letter)] > source_row[str(transformer.agg_fn.letter)]


def test_saw_nlp_respects_feature_subset_and_boundary_values(current_wmsd_module):
    dataset = load_dataset("students")
    transformer, _ = build_transformer(current_wmsd_module, "SAW", dataset)

    result = transformer.improvement(
        "improvement_non_linear_programming",
        4,
        2,
        1e-4,
        features_to_change=["Math", "Bio"],
        boundary_values=[100.0, 6.0],
    )

    assert result is not None
    assert not result.empty
    non_zero_columns = [
        column for column, value in result.iloc[0].items() if not np.isclose(value, 0.0)
    ]
    assert set(non_zero_columns).issubset({"Math", "Bio"})
