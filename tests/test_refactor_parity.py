import pandas as pd
import pytest

from conftest import build_transformer


AGGREGATIONS = ["RTOPSIS", "ATOPSIS", "ITOPSIS"]
FEATURE_CASES = {
    "students": (2, 1),
    "bus": (2, 1),
}
GENETIC_CASES = {
    "students": (3, 2),
    "bus": (2, 1),
}


def assert_frames_match(current, baseline):
    if current is None or baseline is None:
        assert current is baseline
        return

    pd.testing.assert_frame_equal(
        current.reset_index(drop=True),
        baseline.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("dataset", ["students", "bus"], indirect=True)
@pytest.mark.parametrize("agg_name", AGGREGATIONS)
def test_fit_transform_matches_baseline(dataset, agg_name, current_wmsd_module, baseline_module):
    current_transformer, current_transformed = build_transformer(
        current_wmsd_module,
        agg_name,
        dataset,
    )
    baseline_transformer, baseline_transformed = build_transformer(
        baseline_module,
        agg_name,
        dataset,
    )

    score_column = str(current_transformer.agg_fn.letter)
    columns_to_compare = dataset["criteria_columns"] + ["Mean", "Std", score_column]

    pd.testing.assert_frame_equal(
        current_transformed.loc[:, columns_to_compare],
        baseline_transformed.loc[:, columns_to_compare],
        check_dtype=False,
        check_exact=False,
        atol=1e-12,
        rtol=0,
    )
    assert current_transformer._ranked_alternatives == baseline_transformer._ranked_alternatives


@pytest.mark.parametrize("dataset", ["students", "bus"], indirect=True)
@pytest.mark.parametrize("agg_name", AGGREGATIONS)
def test_improvement_features_matches_baseline(
    dataset,
    agg_name,
    current_wmsd_module,
    baseline_module,
    request,
):
    dataset_name = request.node.callspec.params["dataset"]
    source_rank, target_rank = FEATURE_CASES[dataset_name]

    current_transformer, _ = build_transformer(current_wmsd_module, agg_name, dataset)
    baseline_transformer, _ = build_transformer(baseline_module, agg_name, dataset)

    kwargs = {
        "features_to_change": dataset["criteria_columns"],
    }
    current_result = current_transformer.improvement(
        "improvement_features",
        source_rank,
        target_rank,
        1e-4,
        **kwargs,
    )
    baseline_result = baseline_transformer.improvement(
        "improvement_features",
        source_rank,
        target_rank,
        1e-4,
        **kwargs,
    )

    assert_frames_match(current_result, baseline_result)


@pytest.mark.parametrize("dataset", ["students", "bus"], indirect=True)
@pytest.mark.parametrize("agg_name", AGGREGATIONS)
def test_improvement_genetic_matches_baseline(
    dataset,
    agg_name,
    current_wmsd_module,
    baseline_module,
    request,
):
    dataset_name = request.node.callspec.params["dataset"]
    source_rank, target_rank = GENETIC_CASES[dataset_name]

    current_transformer, _ = build_transformer(current_wmsd_module, agg_name, dataset)
    baseline_transformer, _ = build_transformer(baseline_module, agg_name, dataset)

    kwargs = {
        "features_to_change": dataset["criteria_columns"],
        "popsize": 20,
        "n_generations": 2,
    }
    current_result = current_transformer.improvement(
        "improvement_genetic",
        source_rank,
        target_rank,
        1e-4,
        **kwargs,
    )
    baseline_result = baseline_transformer.improvement(
        "improvement_genetic",
        source_rank,
        target_rank,
        1e-4,
        **kwargs,
    )

    if current_result is None or baseline_result is None:
        assert current_result is baseline_result
        return

    current_df, current_metadata = current_result
    baseline_df, baseline_metadata = baseline_result

    assert current_metadata is baseline_metadata is None
    assert_frames_match(current_df, baseline_df)
