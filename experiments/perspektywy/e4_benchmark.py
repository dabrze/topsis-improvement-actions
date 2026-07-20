"""E4: timing benchmark of post-factum algorithms on the Perspektywy field."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.perspektywy.common import (  # noqa: E402
    CRITERIA,
    PALETTE,
    build_transformer,
    load_year,
    margin_epsilon,
    output_dirs,
    rank_of,
    result_deltas,
    runtime_metadata,
    save_figure,
    to_latex_table,
    validate_focal,
    write_results,
)
from experiments.perspektywy.config import (  # noqa: E402
    E4_CRITERIA_SUBSETS,
    E4_DIRECT_CRITERION,
    E4_REPETITIONS,
    E4_REPETITIONS_EVOLUTIONARY,
    FOCAL_UNIVERSITY,
    MAIN_YEAR,
    METHODS_EXTENDED,
    SEED,
)


def _request(
    transformer: Any,
    algorithm: str,
    focal: str,
    target: int,
    features: list[str],
    quick: bool,
) -> Any:
    epsilon = margin_epsilon(transformer, target, 0.0)
    if algorithm == "direct":
        return transformer.improvement(
            "improvement_single_feature", focal, target, epsilon, feature_to_change=E4_DIRECT_CRITERION
        )
    if algorithm == "lexicographic":
        return transformer.improvement(
            "improvement_features", focal, target, epsilon, features_to_change=features
        )
    if algorithm == "nlp":
        return transformer.improvement(
            "improvement_non_linear_programming", focal, target, epsilon, features_to_change=features
        )
    if algorithm == "evolutionary":
        return transformer.improvement(
            "improvement_genetic",
            focal,
            target,
            epsilon,
            features_to_change=features,
            popsize=200 if quick else 1000,
            n_generations=5 if quick else 200,
            seed=SEED,
        )
    if algorithm == "wm":
        return transformer.improvement("improvement_mean", focal, target, epsilon, solutions_number=None)
    raise ValueError(algorithm)


def _score_excess(transformer: Any, data: pd.DataFrame, focal: str, delta: pd.Series, target_score: float) -> float:
    candidate = pd.DataFrame([data.loc[focal, CRITERIA] + delta], columns=CRITERIA)
    score = float(transformer.transform(candidate).iloc[0][str(transformer.agg_fn.letter)])
    return score - target_score


def _wm_score_excess(transformer: Any, focal: str, result: pd.DataFrame, target_score: float) -> float:
    original = transformer.X_new.loc[focal]
    score = transformer.agg_fn.score_from_wmsd(
        float(np.mean(transformer.weights)),
        float(original["Mean"] + result.iloc[0]["Mean"]),
        float(original["Std"] + result.iloc[0].get("Std", 0.0)),
    )
    return float(score - target_score)


def _cells() -> list[tuple[str, list[str], int, list[str]]]:
    return [
        ("direct", ["RTOPSIS"], 1, [E4_DIRECT_CRITERION]),
        *[("lexicographic", METHODS_EXTENDED, count, features) for count, features in E4_CRITERIA_SUBSETS.items()],
        *[("nlp", METHODS_EXTENDED, count, features) for count, features in E4_CRITERIA_SUBSETS.items()],
        ("evolutionary", METHODS_EXTENDED, 2, E4_CRITERIA_SUBSETS[2]),
        ("wm", ["RTOPSIS"], len(CRITERIA), CRITERIA),
    ]


def run(
    focal: str = FOCAL_UNIVERSITY,
    year: int = MAIN_YEAR,
    outdir: str | Path | None = None,
    *,
    quick: bool = False,
) -> pd.DataFrame:
    dirs = output_dirs(outdir)
    data, _ = load_year(year)
    validate_focal(data, focal)
    timing_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    timing_outliers: dict[str, list[int]] = {}
    unsupported: list[str] = []

    for algorithm, applicable, n_criteria, features in _cells():
        repetitions = (
            (5 if quick else E4_REPETITIONS_EVOLUTIONARY)
            if algorithm == "evolutionary"
            else (20 if quick else E4_REPETITIONS)
        )
        for method in applicable:
            transformer, _ = build_transformer(method, data, year)
            focal_rank = rank_of(transformer, focal)
            target = focal_rank - 2
            cell = {"algorithm": algorithm, "method": method, "n_criteria": n_criteria}
            if target < 1:
                aggregate_rows.append(
                    {**cell, "status": "n/a", "n_solutions": 0, "repetitions": repetitions,
                     "mean_ms": np.nan, "std_ms": np.nan, "median_ms": np.nan, "iqr_ms": np.nan,
                     "mean_score_excess": np.nan}
                )
                continue
            target_score = float(
                transformer.X_new.loc[
                    transformer._ranked_alternatives[target - 1], str(transformer.agg_fn.letter)
                ]
            )
            cell_rows: list[dict[str, Any]] = []
            unsupported_error: Exception | None = None
            for rep in range(repetitions):
                started = time.perf_counter()
                try:
                    result = _request(transformer, algorithm, focal, target, features, quick)
                    delta = result_deltas(result)
                    feasible = delta is not None or (algorithm == "wm" and result is not None)
                    if delta is not None:
                        excess = _score_excess(transformer, data, focal, delta, target_score)
                    elif algorithm == "wm" and result is not None:
                        excess = _wm_score_excess(transformer, focal, result, target_score)
                    else:
                        excess = np.nan
                except (AttributeError, NotImplementedError, TypeError) as error:
                    unsupported_error = error
                    break
                except (ArithmeticError, RuntimeError, ValueError):
                    feasible = False
                    excess = np.nan
                cell_rows.append(
                    {
                        **cell,
                        "rep": rep,
                        "ms": (time.perf_counter() - started) * 1000,
                        "feasible": feasible,
                        "score_excess": excess,
                    }
                )
            if unsupported_error is not None:
                unsupported.append(f"{algorithm}/{method}/{n_criteria}: {unsupported_error}")
                aggregate_rows.append(
                    {**cell, "status": "n/a", "n_solutions": 0, "repetitions": repetitions,
                     "mean_ms": np.nan, "std_ms": np.nan, "median_ms": np.nan, "iqr_ms": np.nan,
                     "mean_score_excess": np.nan}
                )
                continue
            timing_rows.extend(cell_rows)
            timings = np.asarray([row["ms"] for row in cell_rows], dtype=float)
            feasible_rows = [row for row in cell_rows if row["feasible"]]
            median = float(np.median(timings))
            iqr = float(np.percentile(timings, 75) - np.percentile(timings, 25))
            outlier_reps = [row["rep"] for row in cell_rows if row["ms"] > 10 * median]
            if outlier_reps:
                key = f"{algorithm}/{method}/{n_criteria}"
                timing_outliers[key] = outlier_reps
                print(f"Timing outliers for {key}: {[(row['rep'], row['ms']) for row in cell_rows if row['rep'] in outlier_reps]}")
            aggregate_rows.append(
                {
                    **cell,
                    "status": "ok",
                    "n_solutions": len(feasible_rows),
                    "repetitions": repetitions,
                    "mean_ms": float(np.mean(timings)),
                    "std_ms": float(np.std(timings, ddof=1)) if len(timings) > 1 else 0.0,
                    "median_ms": median,
                    "iqr_ms": iqr,
                    "mean_score_excess": (
                        float(np.mean([row["score_excess"] for row in feasible_rows]))
                        if feasible_rows
                        else np.nan
                    ),
                }
            )
    benchmark = pd.DataFrame(aggregate_rows)
    timings = pd.DataFrame(
        timing_rows,
        columns=["algorithm", "method", "n_criteria", "rep", "ms", "feasible", "score_excess"],
    )
    config = {
        "focal": focal,
        "year": year,
        "quick": quick,
        "repetitions": E4_REPETITIONS,
        "e4_repetitions_evolutionary": 5 if quick else E4_REPETITIONS_EVOLUTIONARY,
        "evolutionary_popsize": 200 if quick else 1000,
        "evolutionary_generations": 5 if quick else 200,
        "methods": METHODS_EXTENDED,
        "evolutionary_methods": METHODS_EXTENDED,
    }
    metadata = {**runtime_metadata(), "timing_outliers": timing_outliers, "unsupported_combinations": unsupported}
    write_results(
        timings,
        dirs["results"] / "e4_timings.csv",
        script=__file__,
        config=config,
        extra_metadata=metadata,
    )
    write_results(
        benchmark,
        dirs["results"] / "e4_benchmark.csv",
        script=__file__,
        config=config,
        extra_metadata=metadata,
    )

    display = benchmark.copy()
    display["time_ms"] = display.apply(
        lambda row: "n/a" if row["status"] == "n/a" else f"{row['median_ms']:.2f} ({row['iqr_ms']:.2f})",
        axis=1,
    )
    to_latex_table(
        display[["algorithm", "method", "n_criteria", "status", "n_solutions", "time_ms", "mean_score_excess"]],
        "Post-factum algorithm timing benchmark; runtime is median (IQR) ms",
        "tab:e4-benchmark",
        dirs["tables"] / "e4_benchmark.tex",
        index=False,
    )
    ok = benchmark.loc[benchmark["status"] == "ok"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    values = ok["median_ms"].to_numpy()
    errors = ok["iqr_ms"].to_numpy()
    labels = [f"{row.algorithm}\n{row.method}\n{row.n_criteria} criteria" for row in ok.itertuples()]
    ax.bar(
        range(len(ok)),
        np.maximum(values, 1e-3),
        yerr=errors,
        capsize=2,
        color=[PALETTE[index % len(PALETTE)] for index in range(len(ok))],
    )
    ax.set_yscale("log")
    ax.set_ylabel("Median runtime (ms, log scale; IQR whiskers)", fontsize=9)
    ax.set_xticks(range(len(ok)), labels, rotation=42, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=8)
    save_figure(fig, dirs["figures"] / "e4_times")
    return benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focal", default=FOCAL_UNIVERSITY)
    parser.add_argument("--year", type=int, default=MAIN_YEAR)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--quick", action="store_true", help="Use 20 regular and 5 evolutionary repetitions.")
    args = parser.parse_args()
    run(args.focal, args.year, args.outdir, quick=args.quick)


if __name__ == "__main__":
    main()
