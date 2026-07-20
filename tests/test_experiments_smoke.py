import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.perspektywy.common import (
    CRITERIA,
    build_transformer,
    load_year,
    margin_epsilon,
    make_plan,
    rank_after_delta,
    rank_of,
    result_deltas,
    score_table,
)
from experiments.perspektywy.config import E4_CRITERIA_SUBSETS, E4_DIRECT_CRITERION, FOCAL_UNIVERSITY


def test_perspektywy_data_loads_without_missing_criteria():
    for year in range(2022, 2027):
        data, ranking = load_year(year)
        assert data.shape == (23, 7)
        assert not data[CRITERIA].isna().any().any()
        assert len(ranking) == 23


def test_e1_nlp_plan_is_feasible_and_reaches_strict_target():
    data, _ = load_year(2025)
    transformer, _ = build_transformer("RTOPSIS", data, 2025)
    source_rank = rank_of(transformer, FOCAL_UNIVERSITY)
    delta = make_plan(
        transformer, "improvement_non_linear_programming", FOCAL_UNIVERSITY, source_rank - 1, features=CRITERIA
    )
    assert delta is not None
    achieved, _ = rank_after_delta(data, 2025, FOCAL_UNIVERSITY, delta)
    assert achieved <= source_rank - 1


def test_zero_delta_is_not_counted_as_survival_of_requested_gain():
    data, _ = load_year(2025)
    transformer, _ = build_transformer("RTOPSIS", data, 2025)
    rank = rank_of(transformer, FOCAL_UNIVERSITY)
    zero = np.zeros(len(CRITERIA))
    achieved, _ = rank_after_delta(data, 2025, FOCAL_UNIVERSITY, zero)
    survived = bool(np.linalg.norm(zero / 100) > 1e-12 and achieved <= rank - 1)
    assert not survived


def test_e1_nlp_is_deterministic():
    data, _ = load_year(2025)
    transformer, _ = build_transformer("RTOPSIS", data, 2025)
    target = rank_of(transformer, FOCAL_UNIVERSITY) - 1
    first = make_plan(transformer, "improvement_non_linear_programming", FOCAL_UNIVERSITY, target, features=CRITERIA)
    second = make_plan(transformer, "improvement_non_linear_programming", FOCAL_UNIVERSITY, target, features=CRITERIA)
    assert first is not None and second is not None
    assert np.allclose(first.to_numpy(), second.to_numpy(), atol=1e-9)


def test_latex_tables_are_nonempty_and_balanced(tmp_path):
    from experiments.perspektywy.common import to_latex_table
    import pandas as pd

    path = Path(tmp_path) / "smoke.tex"
    to_latex_table(pd.DataFrame({"Prestiz": [1.0]}), "Smoke", "tab:smoke", path)
    content = path.read_text(encoding="utf-8")
    assert content.count(r"\begin{table}") == content.count(r"\end{table}") == 1
    assert r"\toprule" in content and path.stat().st_size > 0


def test_generic_and_topsis_genetic_apis_accept_seed():
    import wmsd

    assert "seed" in inspect.signature(wmsd.SAW.improvement_genetic).parameters
    assert "seed" in inspect.signature(wmsd.RTOPSIS.improvement_genetic).parameters


def test_margin_epsilon_uses_requested_rank_gap():
    data, _ = load_year(2025)
    transformer, _ = build_transformer("RTOPSIS", data, 2025)
    scores = score_table(transformer)["score"].to_numpy()
    assert margin_epsilon(transformer, 4, 0.0) == pytest.approx(1e-6)
    assert margin_epsilon(transformer, 4, 0.5) == pytest.approx(abs(scores[2] - scores[3]) / 2)
    assert margin_epsilon(transformer, 1, 0.5) == pytest.approx(abs(scores[0] - scores[1]) / 2)


@pytest.mark.parametrize("method", ["ARAS", "WASPAS"])
def test_extended_methods_accept_experiment_improvement_calls(method):
    data, _ = load_year(2025)
    transformer, _ = build_transformer(method, data, 2025)
    target = rank_of(transformer, FOCAL_UNIVERSITY) - 2
    assert target >= 1
    epsilon = margin_epsilon(transformer, target, 0.0)
    transformer.improvement(
        "improvement_non_linear_programming",
        FOCAL_UNIVERSITY,
        target,
        epsilon,
        features_to_change=E4_CRITERIA_SUBSETS[2],
    )
    transformer.improvement(
        "improvement_features",
        FOCAL_UNIVERSITY,
        target,
        epsilon,
        features_to_change=E4_CRITERIA_SUBSETS[2],
    )
    transformer.improvement(
        "improvement_genetic",
        FOCAL_UNIVERSITY,
        target,
        epsilon,
        features_to_change=E4_CRITERIA_SUBSETS[2],
        popsize=20,
        n_generations=1,
        seed=23,
    )


def test_e4_direct_is_feasible_with_finite_score_excess():
    from experiments.perspektywy.e4_benchmark import _request, _score_excess

    data, _ = load_year(2025)
    transformer, _ = build_transformer("RTOPSIS", data, 2025)
    target = rank_of(transformer, FOCAL_UNIVERSITY) - 2
    target_score = float(
        transformer.X_new.loc[transformer._ranked_alternatives[target - 1], str(transformer.agg_fn.letter)]
    )
    for _ in range(5):
        delta = result_deltas(
            _request(transformer, "direct", FOCAL_UNIVERSITY, target, [E4_DIRECT_CRITERION], quick=True)
        )
        assert delta is not None
        assert np.isfinite(_score_excess(transformer, data, FOCAL_UNIVERSITY, delta, target_score))


def test_e2_alpha_zero_regression_margin_monotonicity_and_ablation_control(tmp_path):
    from experiments.perspektywy.e2_plan_vs_reality import run

    plans = run(
        FOCAL_UNIVERSITY,
        [(2025, 2026)],
        tmp_path,
        alphas=[0.0, 0.5],
        ablation=True,
    )
    baseline_path = (
        Path(__file__).parents[1] / "experiments" / "perspektywy" / "results" / "e2_plans_all_pairs.csv"
    )
    baseline = pd.read_csv(baseline_path)
    if "alpha" in baseline:
        baseline = baseline.loc[baseline["alpha"] == 0.0]
    baseline = baseline.loc[(baseline["year"] == 2025) & (baseline["next_year"] == 2026)]
    current = plans.loc[plans["alpha"] == 0.0]
    keys = ["year", "next_year", "university", "k"]
    deltas = [*CRITERIA, "planned_norm"]
    merged = current.merge(baseline, on=keys, suffixes=("_current", "_baseline"), validate="one_to_one")
    assert len(merged) == len(current) == len(baseline)
    for name in deltas:
        assert np.allclose(
            merged[f"{name}_current"],
            merged[f"{name}_baseline"],
            atol=1e-9,
            equal_nan=True,
        )
    focal = plans.loc[(plans["university"] == FOCAL_UNIVERSITY) & plans["feasible"]]
    for _, group in focal.groupby("k"):
        ordered = group.sort_values("alpha")["planned_norm"].to_numpy()
        assert np.all(np.diff(ordered) >= -1e-12)
    ablation = pd.read_csv(tmp_path / "results" / "e2_ablation.csv")
    assert (ablation.loc[ablation["scenario"] == "weights_only", "survived"]).all()


def test_generated_tables_do_not_contain_tuple_indexes():
    tables = Path(__file__).parents[1] / "experiments" / "perspektywy" / "tables"
    assert all("('" not in path.read_text(encoding="utf-8") for path in tables.glob("*.tex"))
