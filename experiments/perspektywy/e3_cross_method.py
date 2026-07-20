"""E3: compare improvement plans produced by four MCDA aggregation methods."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.perspektywy.common import (  # noqa: E402
    CRITERIA,
    DISPLAY_NAMES,
    PALETTE,
    build_transformer,
    load_year,
    margin_epsilon,
    make_plan,
    output_dirs,
    rank_after_delta,
    rank_of,
    save_figure,
    validate_focal,
    write_results,
    to_latex_table,
)
from experiments.perspektywy.config import (  # noqa: E402
    FOCAL_UNIVERSITY,
    MAIN_PAIR,
    MAIN_YEAR,
    METHODS_EXTENDED,
    SLOW_CRITERIA,
)


def run(
    focal: str = FOCAL_UNIVERSITY,
    year: int = MAIN_YEAR,
    outdir: str | Path | None = None,
    *,
    survival: bool = False,
) -> pd.DataFrame:
    dirs = output_dirs(outdir)
    data, _ = load_year(year)
    validate_focal(data, focal)
    records: list[dict] = []
    plans: dict[tuple[str, int, str], pd.Series | None] = {}
    for method in METHODS_EXTENDED:
        transformer, _ = build_transformer(method, data, year)
        focal_rank = rank_of(transformer, focal)
        for scope, features in (
            ("all", CRITERIA),
            ("fast_moving", [criterion for criterion in CRITERIA if criterion not in SLOW_CRITERIA]),
        ):
            for k in (1, 2, 3):
                target = focal_rank - k
                delta = (
                    None
                    if target < 1
                    else make_plan(
                        transformer,
                        "improvement_non_linear_programming",
                        focal,
                        target,
                        features=features,
                        epsilon=margin_epsilon(transformer, target, 0.0),
                    )
                )
                plans[(method, k, scope)] = delta
                records.append(
                    {
                        "method": method,
                        "scope": scope,
                        "focal_rank": focal_rank,
                        "k": k,
                        "target_rank": target if target >= 1 else np.nan,
                        "feasible": delta is not None,
                        **({criterion: float(delta[criterion]) for criterion in CRITERIA} if delta is not None else {}),
                    }
                )
    output = pd.DataFrame(records)
    config = {"focal": focal, "year": year, "survival": survival}
    write_results(output, dirs["results"] / "e3_plans.csv", script=__file__, config=config)

    all_plans = output.loc[output["scope"] == "all"]
    delta_table = pd.DataFrame(index=CRITERIA)
    for method in METHODS_EXTENDED:
        for k in (1, 2, 3):
            row = all_plans.loc[(all_plans["method"] == method) & (all_plans["k"] == k)]
            delta_table[f"{method} gain {k}"] = row.iloc[0][CRITERIA] if not row.empty else np.nan
    to_latex_table(delta_table, "Cross-method NLP improvement deltas", "tab:e3-deltas", dirs["tables"] / "e3_deltas.tex")

    cross_records: list[dict] = []
    for source_method in METHODS_EXTENDED:
        delta = plans[(source_method, 2, "all")]
        for target_method in METHODS_EXTENDED:
            achieved = np.nan if delta is None else rank_after_delta(data, year, focal, delta, method=target_method)[0]
            cross_records.append(
                {"plan_method": source_method, "evaluation_method": target_method, "achieved_rank": achieved}
            )
    cross = pd.DataFrame(cross_records)
    cross_table = cross.pivot(index="evaluation_method", columns="plan_method", values="achieved_rank")
    cross_table.index.name = "Evaluating method"
    cross_table.columns.name = "Plan-origin method"
    to_latex_table(
        cross_table,
        "Ranks after cross-method evaluation of gain-two plans (rows: evaluating method; columns: plan-origin method)",
        "tab:e3-cross-eval",
        dirs["tables"] / "e3_cross_eval.tex",
    )

    fig, ax = plt.subplots(figsize=(8, 4.1))
    width = 0.12
    x = np.arange(len(CRITERIA))
    for offset, method in enumerate(METHODS_EXTENDED):
        row = all_plans.loc[(all_plans["method"] == method) & (all_plans["k"] == 2)]
        values = (
            np.zeros(len(CRITERIA))
            if row.empty
            else pd.to_numeric(row.iloc[0][CRITERIA], errors="coerce").fillna(0).to_numpy(float)
        )
        ax.bar(
            x + (offset - (len(METHODS_EXTENDED) - 1) / 2) * width,
            values,
            width,
            label=method,
            color=PALETTE[offset % len(PALETTE)],
        )
    ax.set_xticks(x, [DISPLAY_NAMES[criterion] for criterion in CRITERIA], rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("Required Δ (relative-score points)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=8)
    save_figure(fig, dirs["figures"] / "e3_deltas_bars")

    if survival:
        following, _ = load_year(MAIN_PAIR[1])
        survival_records: list[dict] = []
        for method in METHODS_EXTENDED:
            for k in (1, 2, 3):
                delta = plans[(method, k, "all")]
                source_rank = int(output.loc[(output["method"] == method) & (output["scope"] == "all"), "focal_rank"].iloc[0])
                target = source_rank - k
                rank = np.nan if delta is None else rank_after_delta(
                    following, MAIN_PAIR[1], focal, delta, method=method, baseline=data.loc[focal]
                )[0]
                survival_records.append(
                    {
                        "method": method,
                        "k": k,
                        "target_rank": target,
                        "feasible": delta is not None,
                        "achieved_rank": rank,
                        "survived": bool(delta is not None and rank <= target),
                    }
                )
        write_results(
            pd.DataFrame(survival_records),
            dirs["results"] / "e3_survival.csv",
            script=__file__,
            config=config,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focal", default=FOCAL_UNIVERSITY)
    parser.add_argument("--year", type=int, default=MAIN_YEAR)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--survival", action="store_true", help="Evaluate focal plans in the 2026 field.")
    args = parser.parse_args()
    run(args.focal, args.year, args.outdir, survival=args.survival)


if __name__ == "__main__":
    main()
