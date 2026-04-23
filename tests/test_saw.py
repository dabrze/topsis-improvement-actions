import numpy as np
import pandas as pd
import pytest

from conftest import build_transformer


SINGLE_FEATURE_CASES = {
    "students": {"source_rank": 3, "target_rank": 2, "feature": "Bio"},
    "bus": {"source_rank": 2, "target_rank": 1, "feature": "SummerCons"},
}


@pytest.mark.parametrize("dataset", ["students", "bus"], indirect=True)
def test_saw_scores_match_manual_weighted_sum(dataset, current_wmsd_module):
    transformer, transformed = build_transformer(current_wmsd_module, "SAW", dataset)
    expected_scores = transformed.loc[:, dataset["criteria_columns"]].to_numpy() @ np.asarray(
        transformer.weights,
        dtype=float,
    )
    np.testing.assert_allclose(transformed["U"].to_numpy(), expected_scores, atol=1e-12)


@pytest.mark.parametrize("dataset", ["students", "bus"], indirect=True)
def test_saw_single_feature_matches_closed_form(dataset, current_wmsd_module, request):
    dataset_name = request.node.callspec.params["dataset"]
    case = SINGLE_FEATURE_CASES[dataset_name]

    transformer, transformed = build_transformer(current_wmsd_module, "SAW", dataset)
    result = transformer.improvement(
        "improvement_single_feature",
        case["source_rank"],
        case["target_rank"],
        1e-6,
        feature_to_change=case["feature"],
    )
    assert result is not None

    source_id = transformer._ranked_alternatives[case["source_rank"] - 1]
    target_id = transformer._ranked_alternatives[case["target_rank"] - 1]
    source_row = transformed.loc[source_id]
    target_row = transformed.loc[target_id]

    feature_idx = dataset["criteria_columns"].index(case["feature"])
    criterion_weight = transformer.weights[feature_idx]
    target_score = target_row["U"] + 5e-7
    required_delta = (target_score - source_row["U"]) / criterion_weight
    expected_change = required_delta * transformer._value_range[feature_idx]
    if transformer.objectives[feature_idx] == "min":
        expected_change *= -1

    expected = pd.DataFrame(
        [np.zeros(len(dataset["criteria_columns"]))],
        columns=dataset["criteria_columns"],
    )
    expected.loc[0, case["feature"]] = expected_change

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected,
        check_dtype=False,
        check_exact=False,
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("dataset_name", "source_rank", "target_rank"),
    [
        ("students", 3, 2),
        ("bus", 2, 1),
    ],
)
def test_saw_reuses_generic_feature_and_genetic_methods(
    dataset_name,
    source_rank,
    target_rank,
    current_wmsd_module,
):
    from conftest import load_dataset

    dataset = load_dataset(dataset_name)
    transformer, _ = build_transformer(current_wmsd_module, "SAW", dataset)

    feature_result = transformer.improvement(
        "improvement_features",
        source_rank,
        target_rank,
        1e-6,
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
