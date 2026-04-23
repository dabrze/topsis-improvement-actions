import numpy as np
import pandas as pd
import pymcdm.methods as pymcdm_methods
import pytest

from conftest import build_transformer, load_dataset


METHODS = ["ARAS", "COPRAS", "WASPAS"]
SINGLE_FEATURE_CASES = {
    ("ARAS", "students"): {"source_rank": 4, "target_rank": 2, "feature": "Math"},
    ("ARAS", "bus"): {"source_rank": 5, "target_rank": 4, "feature": "MaxSpeed"},
    ("COPRAS", "students"): {"source_rank": 4, "target_rank": 2, "feature": "Math"},
    ("COPRAS", "bus"): {"source_rank": 7, "target_rank": 6, "feature": "MaxSpeed"},
    ("WASPAS", "students"): {"source_rank": 3, "target_rank": 2, "feature": "Math"},
    ("WASPAS", "bus"): {"source_rank": 5, "target_rank": 4, "feature": "MaxSpeed"},
}
MULTI_CASES = {
    ("ARAS", "students"): (2, 1),
    ("ARAS", "bus"): (2, 1),
    ("COPRAS", "students"): (2, 1),
    ("COPRAS", "bus"): (2, 1),
    ("WASPAS", "students"): (2, 1),
    ("WASPAS", "bus"): (2, 1),
}
GENETIC_CASES = {
    ("ARAS", "students"): (3, 2),
    ("ARAS", "bus"): (2, 1),
    ("COPRAS", "students"): (3, 2),
    ("COPRAS", "bus"): (2, 1),
    ("WASPAS", "students"): (3, 2),
    ("WASPAS", "bus"): (2, 1),
}


def identity_normalization(x, cost=False):
    return np.asarray(x, dtype=float)


@pytest.mark.parametrize("dataset_name", ["students", "bus"])
def test_aras_matches_pymcdm_on_utility_matrix(dataset_name, current_wmsd_module):
    dataset = load_dataset(dataset_name)
    transformer, transformed = build_transformer(current_wmsd_module, "ARAS", dataset)
    utility_matrix = transformed.loc[:, dataset["criteria_columns"]].to_numpy()
    weights = np.asarray(transformer.weights, dtype=float)
    weights = weights / np.sum(weights)

    expected = pymcdm_methods.ARAS(
        normalization_function=identity_normalization,
        esp=np.ones(utility_matrix.shape[1]),
    )(utility_matrix, weights, np.ones(utility_matrix.shape[1], dtype=int))
    np.testing.assert_allclose(transformed["K"].to_numpy(), expected, atol=1e-12)


@pytest.mark.parametrize("dataset_name", ["students", "bus"])
def test_waspas_matches_pymcdm_on_utility_matrix(dataset_name, current_wmsd_module):
    dataset = load_dataset(dataset_name)
    transformer, transformed = build_transformer(current_wmsd_module, "WASPAS", dataset)
    utility_matrix = transformed.loc[:, dataset["criteria_columns"]].to_numpy()
    weights = np.asarray(transformer.weights, dtype=float)
    weights = weights / np.sum(weights)

    expected = pymcdm_methods.WASPAS(
        normalization_function=identity_normalization,
        l=0.5,
    )(utility_matrix, weights, np.ones(utility_matrix.shape[1], dtype=int))
    np.testing.assert_allclose(transformed["W"].to_numpy(), expected, atol=1e-12)


@pytest.mark.parametrize("dataset_name", ["students", "bus"])
def test_copras_matches_manual_formula(dataset_name, current_wmsd_module):
    dataset = load_dataset(dataset_name)
    transformer, transformed = build_transformer(current_wmsd_module, "COPRAS", dataset)
    utility_matrix = transformed.loc[:, dataset["criteria_columns"]].to_numpy()
    weights = np.asarray(transformer.weights, dtype=float)
    objectives = np.asarray(transformer.objectives)

    gain_mask = objectives == "max"
    cost_mask = objectives == "min"
    sp = np.sum(utility_matrix[:, gain_mask] * weights[gain_mask], axis=1)
    if np.any(cost_mask):
        sm = np.sum((1 - utility_matrix[:, cost_mask]) * weights[cost_mask], axis=1)
        expected = sp / np.maximum(sm, 1e-12)
    else:
        expected = sp

    np.testing.assert_allclose(transformed["C"].to_numpy(), expected, atol=1e-12)


@pytest.mark.parametrize("method_name", METHODS)
@pytest.mark.parametrize("dataset_name", ["students", "bus"])
def test_phase_c_single_feature_support(method_name, dataset_name, current_wmsd_module):
    dataset = load_dataset(dataset_name)
    case = SINGLE_FEATURE_CASES[(method_name, dataset_name)]
    transformer, _ = build_transformer(current_wmsd_module, method_name, dataset)
    result = transformer.improvement(
        "improvement_single_feature",
        case["source_rank"],
        case["target_rank"],
        1e-4,
        feature_to_change=case["feature"],
    )
    assert result is not None
    assert not result.empty


@pytest.mark.parametrize("method_name", METHODS)
@pytest.mark.parametrize("dataset_name", ["students", "bus"])
def test_phase_c_reuses_generic_feature_search(method_name, dataset_name, current_wmsd_module):
    dataset = load_dataset(dataset_name)
    source_rank, target_rank = MULTI_CASES[(method_name, dataset_name)]
    transformer, _ = build_transformer(current_wmsd_module, method_name, dataset)

    result = transformer.improvement(
        "improvement_features",
        source_rank,
        target_rank,
        1e-4,
        features_to_change=dataset["criteria_columns"],
    )
    assert result is not None
    assert not result.empty


@pytest.mark.parametrize("method_name", METHODS)
@pytest.mark.parametrize("dataset_name", ["students", "bus"])
def test_phase_c_reuses_genetic_search(method_name, dataset_name, current_wmsd_module):
    dataset = load_dataset(dataset_name)
    source_rank, target_rank = GENETIC_CASES[(method_name, dataset_name)]
    transformer, _ = build_transformer(current_wmsd_module, method_name, dataset)

    result = transformer.improvement(
        "improvement_genetic",
        source_rank,
        target_rank,
        1e-4,
        features_to_change=dataset["criteria_columns"],
        popsize=20,
        n_generations=2,
    )
    assert result is not None
    result_df, metadata = result
    assert metadata is None
    assert result_df is not None
    assert not result_df.empty
