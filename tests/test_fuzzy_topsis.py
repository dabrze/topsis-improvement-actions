import numpy as np
import pandas as pd
import pytest


def build_tuple_fuzzy_dataset():
    return pd.DataFrame(
        {
            "Quality": [(3, 4, 5), (4, 5, 6), (2, 3, 4)],
            "Cost": [(5, 6, 7), (4, 5, 6), (6, 7, 8)],
        },
        index=["A1", "A2", "A3"],
    )


def build_flat_fuzzy_dataset():
    return pd.DataFrame(
        {
            "Quality_l": [3, 4, 2],
            "Quality_m": [4, 5, 3],
            "Quality_u": [5, 6, 4],
            "Cost_l": [5, 4, 6],
            "Cost_m": [6, 5, 7],
            "Cost_u": [7, 6, 8],
        },
        index=["A1", "A2", "A3"],
    )


def manual_fuzzy_topsis_scores():
    matrix = np.array(
        [
            [[3, 4, 5], [5, 6, 7]],
            [[4, 5, 6], [4, 5, 6]],
            [[2, 3, 4], [6, 7, 8]],
        ],
        dtype=float,
    )
    weights = np.array([0.6, 0.4], dtype=float)

    normalized = np.zeros_like(matrix)
    normalized[:, 0, :] = matrix[:, 0, :] / np.max(matrix[:, 0, 2])
    min_cost = np.min(matrix[:, 1, 0])
    normalized[:, 1, 0] = min_cost / matrix[:, 1, 2]
    normalized[:, 1, 1] = min_cost / matrix[:, 1, 1]
    normalized[:, 1, 2] = min_cost / matrix[:, 1, 0]

    weighted = normalized * weights.reshape(1, -1, 1)
    fpis = np.max(weighted, axis=0)
    fnis = np.min(weighted, axis=0)

    def vertex_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2, axis=-1) / 3.0)

    d_pos = np.sum(vertex_distance(weighted, fpis), axis=1)
    d_neg = np.sum(vertex_distance(weighted, fnis), axis=1)
    return d_neg / (d_pos + d_neg)


def apply_tuple_modification(base_row, modification_row, index_label):
    updated = {}
    for criterion in base_row.index:
        updated[criterion] = [
            tuple(
                np.asarray(base_row[criterion], dtype=float)
                + np.asarray(modification_row[criterion], dtype=float)
            )
        ]
    return pd.DataFrame(updated, index=[index_label])


def test_fuzzy_topsis_matches_manual_vertex_formula(current_wmsd_module):
    dataset = build_tuple_fuzzy_dataset()
    transformer = current_wmsd_module.FuzzyTOPSIS()
    transformed = transformer.fit_transform(
        dataset,
        weights=[0.6, 0.4],
        objectives=["max", "min"],
    )

    expected_scores = manual_fuzzy_topsis_scores()
    np.testing.assert_allclose(transformed["F"].to_numpy(), expected_scores, atol=1e-12)


def test_fuzzy_topsis_supports_flat_column_contract(current_wmsd_module):
    tuple_dataset = build_tuple_fuzzy_dataset()
    flat_dataset = build_flat_fuzzy_dataset()

    tuple_transformer = current_wmsd_module.FuzzyTOPSIS()
    flat_transformer = current_wmsd_module.FuzzyTOPSIS()

    tuple_scores = tuple_transformer.fit_transform(
        tuple_dataset,
        weights={"Quality": 0.6, "Cost": 0.4},
        objectives={"Quality": "max", "Cost": "min"},
    )["F"].to_numpy()
    flat_scores = flat_transformer.fit_transform(
        flat_dataset,
        weights={"Quality": 0.6, "Cost": 0.4},
        objectives={"Quality": "max", "Cost": "min"},
    )["F"].to_numpy()

    np.testing.assert_allclose(tuple_scores, flat_scores, atol=1e-12)


def test_fuzzy_topsis_return_ranking_orders_by_score(current_wmsd_module):
    dataset = build_tuple_fuzzy_dataset()
    transformer = current_wmsd_module.FuzzyTOPSIS()
    transformer.fit(dataset, weights=[0.6, 0.4], objectives=["max", "min"])

    ranking = transformer.return_ranking(normalized=False)
    assert ranking.index.tolist() == ["A2", "A1", "A3"]
    assert ranking["Rank"].tolist() == [1, 2, 3]


def test_fuzzy_topsis_requires_triangular_cells(current_wmsd_module):
    dataset = pd.DataFrame({"Quality": [(1, 2), (2, 3)]})
    transformer = current_wmsd_module.FuzzyTOPSIS()

    with pytest.raises(ValueError, match="triangular fuzzy number"):
        transformer.fit(dataset, weights=[1], objectives=["max"])


def test_fuzzy_topsis_genetic_postfactum_reaches_target_score(current_wmsd_module):
    dataset = build_tuple_fuzzy_dataset()
    transformer = current_wmsd_module.FuzzyTOPSIS()
    transformer.fit(dataset, weights=[0.6, 0.4], objectives=["max", "min"])

    result, checkpoints = transformer.improvement(
        "improvement_genetic",
        "A3",
        "A1",
        1e-4,
        features_to_change=["Quality", "Cost"],
        popsize=120,
        n_generations=120,
    )

    assert checkpoints is None
    assert result is not None
    assert not result.empty

    modified_row = apply_tuple_modification(dataset.loc["A3"], result.iloc[0], "A3")
    updated = transformer.transform(modified_row).iloc[0]

    assert updated["F"] > transformer.X_new.loc["A3", "F"]
    assert updated["F"] >= transformer.X_new.loc["A1", "F"] + 1e-4 - 1e-6


def test_fuzzy_topsis_genetic_postfactum_preserves_flat_output_contract(current_wmsd_module):
    dataset = build_flat_fuzzy_dataset()
    transformer = current_wmsd_module.FuzzyTOPSIS()
    transformer.fit(
        dataset,
        weights={"Quality": 0.6, "Cost": 0.4},
        objectives={"Quality": "max", "Cost": "min"},
    )

    result, _ = transformer.improvement(
        "improvement_genetic",
        "A3",
        "A1",
        1e-4,
        features_to_change=["Quality", "Cost"],
        popsize=80,
        n_generations=80,
    )

    assert result is not None
    assert list(result.columns) == list(dataset.columns)


def test_fuzzy_topsis_exact_nlp_reaches_target_score(current_wmsd_module):
    dataset = build_tuple_fuzzy_dataset()
    transformer = current_wmsd_module.FuzzyTOPSIS()
    transformer.fit(dataset, weights=[0.6, 0.4], objectives=["max", "min"])

    result = transformer.improvement(
        "improvement_non_linear_programming",
        "A3",
        "A1",
        1e-4,
        features_to_change=["Quality", "Cost"],
    )

    assert result is not None
    assert not result.empty

    modified_row = apply_tuple_modification(dataset.loc["A3"], result.iloc[0], "A3")
    updated = transformer.transform(modified_row).iloc[0]

    assert updated["F"] > transformer.X_new.loc["A3", "F"]
    assert updated["F"] >= transformer.X_new.loc["A1", "F"] + 1e-4 - 1e-6


def test_fuzzy_topsis_exact_nlp_preserves_flat_output_contract(current_wmsd_module):
    dataset = build_flat_fuzzy_dataset()
    transformer = current_wmsd_module.FuzzyTOPSIS()
    transformer.fit(
        dataset,
        weights={"Quality": 0.6, "Cost": 0.4},
        objectives={"Quality": "max", "Cost": "min"},
    )

    result = transformer.improvement(
        "improvement_non_linear_programming",
        "A3",
        "A1",
        1e-4,
        features_to_change=["Quality", "Cost"],
    )

    assert result is not None
    assert list(result.columns) == list(dataset.columns)
