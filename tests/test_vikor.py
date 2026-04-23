import numpy as np
import pymcdm.methods as pymcdm_methods
import pytest

from conftest import build_transformer, load_dataset


def identity_normalization(x, cost=False):
    return np.asarray(x, dtype=float)


@pytest.mark.parametrize("dataset_name", ["students", "bus"])
def test_vikor_matches_pymcdm_q_score(dataset_name, current_wmsd_module):
    dataset = load_dataset(dataset_name)
    transformer, transformed = build_transformer(current_wmsd_module, "VIKOR", dataset)
    utility_matrix = transformed.loc[:, dataset["criteria_columns"]].to_numpy()
    weights = np.asarray(transformer.weights, dtype=float)
    weights = weights / np.sum(weights)

    q_values = pymcdm_methods.VIKOR(
        normalization_function=identity_normalization,
        v=0.5,
    )(utility_matrix, weights, np.ones(utility_matrix.shape[1], dtype=int))
    np.testing.assert_allclose(transformed["V"].to_numpy(), 1 - q_values, atol=1e-12)


@pytest.mark.parametrize(
    ("dataset_name", "source_rank", "target_rank"),
    [
        ("students", 3, 2),
        ("bus", 3, 2),
    ],
)
def test_vikor_reuses_generic_feature_and_genetic_methods(
    dataset_name,
    source_rank,
    target_rank,
    current_wmsd_module,
):
    dataset = load_dataset(dataset_name)
    transformer, _ = build_transformer(current_wmsd_module, "VIKOR", dataset)

    feature_result = transformer.improvement(
        "improvement_features",
        source_rank,
        target_rank,
        1e-4,
        features_to_change=dataset["criteria_columns"],
    )
    assert feature_result is not None and not feature_result.empty

    genetic_result = transformer.improvement(
        "improvement_genetic",
        source_rank,
        target_rank,
        1e-4,
        features_to_change=dataset["criteria_columns"],
        popsize=20,
        n_generations=2,
    )
    assert genetic_result is not None
    result_df, metadata = genetic_result
    assert metadata is None
    assert result_df is not None and not result_df.empty
