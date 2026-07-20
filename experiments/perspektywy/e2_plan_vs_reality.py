"""E2: test next-cycle survival, safety margins, and regime-change ablations."""

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
    PALETTE,
    build_transformer,
    load_year,
    make_plan,
    margin_epsilon,
    normalized_norm,
    output_dirs,
    rank_after_delta,
    rank_of,
    save_figure,
    score_table,
    to_latex_table,
    validate_focal,
    write_results,
)
from experiments.perspektywy.config import ALL_PAIRS, FOCAL_UNIVERSITY, MAIN_PAIR, MARGIN_ALPHAS  # noqa: E402


def _resolving_effort(
    next_field: pd.DataFrame, focal: str, next_year: int, target_rank: int
) -> float:
    transformer, _ = build_transformer("RTOPSIS", next_field, next_year)
    if rank_of(transformer, focal) <= target_rank:
        return 0.0
    delta = make_plan(transformer, "improvement_non_linear_programming", focal, target_rank, features=CRITERIA)
    return np.nan if delta is None else normalized_norm(delta)


def _evaluate_plan(
    field: pd.DataFrame,
    evaluation_year: int,
    weights_year: int,
    university: str,
    delta: pd.Series,
    baseline: pd.Series,
    target_rank: int,
) -> int:
    return rank_after_delta(
        field,
        evaluation_year,
        university,
        delta,
        baseline=baseline,
        weights_year=weights_year,
    )[0]


def _write_margin_figure(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.4), sharex="col")
    for column, k in enumerate((1, 2, 3)):
        subset = summary.loc[summary["k"] == k]
        for index, (pair, frame) in enumerate(subset.groupby("pair", sort=False)):
            color = PALETTE[index % len(PALETTE)]
            axes[0, column].plot(frame["alpha"], frame["survival_rate"], marker="o", color=color, label=pair)
            axes[1, column].plot(
                frame["alpha"], frame["median_planned_norm_ratio"], marker="o", color=color, label=pair
            )
        axes[0, column].set_title(f"Gain {k}", fontsize=9)
        axes[0, column].set_ylim(-0.05, 1.05)
        axes[0, column].set_ylabel("Survival rate", fontsize=8)
        axes[1, column].set_ylabel("Median effort / alpha 0", fontsize=8)
        axes[1, column].set_xlabel("Margin alpha", fontsize=8)
        for axis in axes[:, column]:
            axis.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    save_figure(fig, path)


def _ablation_records(plans: pd.DataFrame, pairs: list[tuple[int, int]]) -> pd.DataFrame:
    records: list[dict] = []
    alpha_zero = plans.loc[(plans["alpha"] == 0.0) & plans["feasible"]].copy()
    for year, next_year in pairs:
        current, _ = load_year(year)
        following, _ = load_year(next_year)
        pair_plans = alpha_zero.loc[(alpha_zero["year"] == year) & (alpha_zero["next_year"] == next_year)]
        for row in pair_plans.itertuples(index=False):
            delta = pd.Series({criterion: getattr(row, criterion) for criterion in CRITERIA})
            for scenario, field, evaluation_year, weights_year in (
                ("full", following, next_year, next_year),
                ("values_only", following, next_year, year),
                ("weights_only", current, year, next_year),
            ):
                achieved = _evaluate_plan(
                    field,
                    evaluation_year,
                    weights_year,
                    row.university,
                    delta,
                    current.loc[row.university],
                    row.target_rank,
                )
                records.append(
                    {
                        "year": year,
                        "next_year": next_year,
                        "pair": row.pair,
                        "university": row.university,
                        "k": row.k,
                        "target_rank": row.target_rank,
                        "scenario": scenario,
                        "achieved_rank": achieved,
                        "survived": bool(achieved <= row.target_rank),
                    }
                )
    return pd.DataFrame(records)


def run(
    focal: str = FOCAL_UNIVERSITY,
    pairs: list[tuple[int, int]] | None = None,
    outdir: str | Path | None = None,
    *,
    alphas: list[float] | None = None,
    ablation: bool = False,
) -> pd.DataFrame:
    dirs = output_dirs(outdir)
    pairs = pairs or ALL_PAIRS
    alphas = list(MARGIN_ALPHAS if alphas is None else alphas)
    if not alphas or any(alpha < 0 for alpha in alphas):
        raise ValueError("Alphas must be a non-empty sequence of non-negative values.")

    records: list[dict] = []
    for year, next_year in pairs:
        current, _ = load_year(year)
        following, _ = load_year(next_year)
        validate_focal(current, focal)
        current_transformer, _ = build_transformer("RTOPSIS", current, year)
        following_transformer, _ = build_transformer("RTOPSIS", following, next_year)
        current_scores = score_table(current_transformer).set_index("id")
        following_scores = score_table(following_transformer).set_index("id")
        for university in current_scores.index:
            initial_rank = int(current_scores.loc[university, "rank"])
            if initial_rank == 1:
                continue
            for k in (1, 2, 3):
                target_rank = initial_rank - k
                if target_rank < 1:
                    continue
                target_id = current_transformer._ranked_alternatives[target_rank - 1]
                for alpha in alphas:
                    epsilon = margin_epsilon(current_transformer, target_rank, alpha)
                    delta = make_plan(
                        current_transformer,
                        "improvement_non_linear_programming",
                        university,
                        target_rank,
                        features=CRITERIA,
                        epsilon=epsilon,
                    )
                    record: dict = {
                        "year": year,
                        "next_year": next_year,
                        "pair": f"{year}→{next_year}",
                        "main": (year, next_year) == MAIN_PAIR,
                        "university": university,
                        "initial_rank": initial_rank,
                        "k": k,
                        "alpha": alpha,
                        "epsilon": epsilon,
                        "target_rank": target_rank,
                        "target_university": target_id,
                        "feasible": delta is not None,
                        "planned_norm": np.nan,
                        "achieved_rank": np.nan,
                        "realized_next_rank": int(following_scores.loc[university, "rank"]),
                        "survived": False,
                        "resolved_norm": np.nan,
                        "effort_ratio": np.nan,
                        "competitors_moved": bool(
                            following_scores.loc[target_id, "score"] > current_scores.loc[target_id, "score"]
                        ),
                    }
                    if delta is not None:
                        plan_norm = normalized_norm(delta)
                        achieved_rank = _evaluate_plan(
                            following,
                            next_year,
                            next_year,
                            university,
                            delta,
                            current.loc[university],
                            target_rank,
                        )
                        resolved_norm = _resolving_effort(following, university, next_year, target_rank)
                        record.update(
                            planned_norm=plan_norm,
                            achieved_rank=achieved_rank,
                            survived=bool(plan_norm > 1e-12 and achieved_rank <= target_rank),
                            resolved_norm=resolved_norm,
                            effort_ratio=(
                                np.nan
                                if not np.isfinite(resolved_norm) or plan_norm <= 1e-12
                                else resolved_norm / plan_norm
                            ),
                            **{name: float(delta[name]) for name in CRITERIA},
                        )
                    records.append(record)
    plans = pd.DataFrame(records)
    config = {"focal": focal, "pairs": pairs, "alphas": alphas, "ablation": ablation}
    write_results(plans, dirs["results"] / "e2_plans_all_pairs.csv", script=__file__, config=config)

    alpha_zero = plans.loc[(plans["alpha"] == 0.0) & plans["feasible"]].copy()
    summary = (
        alpha_zero.groupby(["pair", "k"], as_index=False, sort=False)
        .agg(survival_rate=("survived", "mean"), median_effort_ratio=("effort_ratio", "median"), n=("survived", "size"))
    )
    to_latex_table(
        summary,
        "Plan survival by evaluation cycle and target gain (alpha = 0)",
        "tab:e2-survival",
        dirs["tables"] / "e2_survival_summary.tex",
        index=False,
    )
    focal_table = plans.loc[
        (plans["university"] == focal) & plans["alpha"].isin([0.0, 0.5]),
        ["pair", "k", "alpha", "target_rank", "feasible", "achieved_rank", "survived", "effort_ratio", *CRITERIA],
    ]
    to_latex_table(
        focal_table,
        "Focal university plan outcomes",
        "tab:e2-focal",
        dirs["tables"] / "e2_focal.tex",
        index=False,
    )

    baseline_norms = alpha_zero.set_index(["pair", "university", "k"])["planned_norm"]
    margins = plans.loc[plans["feasible"]].copy()
    margins["baseline_norm"] = [
        baseline_norms.get((row.pair, row.university, row.k), np.nan) for row in margins.itertuples(index=False)
    ]
    margins["planned_norm_ratio"] = margins["planned_norm"] / margins["baseline_norm"]
    margin_summary = (
        margins.groupby(["pair", "k", "alpha"], as_index=False, sort=False)
        .agg(
            survival_rate=("survived", "mean"),
            median_planned_norm_ratio=("planned_norm_ratio", "median"),
            n=("survived", "size"),
        )
    )
    to_latex_table(
        margin_summary,
        "Safety-margin survival and relative planned effort",
        "tab:e2-margin",
        dirs["tables"] / "e2_margin_summary.tex",
        index=False,
    )
    _write_margin_figure(margin_summary, dirs["figures"] / "e2_margin_curves")

    heat = summary.pivot(index="pair", columns="k", values="survival_rate").reindex(columns=[1, 2, 3])
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    image = ax.imshow(heat.fillna(0).to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3), ["Gain 1", "Gain 2", "Gain 3"], fontsize=8)
    ax.set_yticks(range(len(heat.index)), heat.index, fontsize=8)
    for row in range(len(heat.index)):
        for column in range(3):
            value = heat.iloc[row, column]
            ax.text(column, row, "—" if pd.isna(value) else f"{value:.0%}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Survival rate")
    save_figure(fig, dirs["figures"] / "e2_survival_heatmap")

    ratios = alpha_zero.loc[
        (alpha_zero["pair"] == f"{MAIN_PAIR[0]}→{MAIN_PAIR[1]}") & alpha_zero["effort_ratio"].notna()
    ]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    nonempty = [(k, ratios.loc[ratios["k"] == k, "effort_ratio"].to_numpy()) for k in (1, 2, 3)]
    nonempty = [(k, value) for k, value in nonempty if len(value)]
    if nonempty:
        ax.boxplot([value for _, value in nonempty], labels=[f"Gain {k}" for k, _ in nonempty])
        for position, (_, value) in enumerate(nonempty, start=1):
            ax.scatter(np.full(len(value), position), value, color=PALETTE[0], alpha=0.45, s=10)
    ax.set_ylabel("Re-solved / planned effort ratio", fontsize=9)
    ax.tick_params(labelsize=8)
    save_figure(fig, dirs["figures"] / "e2_effort_ratio")

    if ablation:
        ablation_frame = _ablation_records(plans, pairs)
        write_results(
            ablation_frame,
            dirs["results"] / "e2_ablation.csv",
            script=__file__,
            config=config,
        )
        ablation_summary = (
            ablation_frame.groupby(["pair", "k", "scenario"], as_index=False, sort=False)
            .agg(survival_rate=("survived", "mean"), n=("survived", "size"))
        )
        to_latex_table(
            ablation_summary,
            "Plan survival under values/weights ablations (alpha = 0)",
            "tab:e2-ablation",
            dirs["tables"] / "e2_ablation.tex",
            index=False,
        )
    return plans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focal", default=FOCAL_UNIVERSITY)
    parser.add_argument("--pair", nargs=2, type=int, metavar=("YEAR", "NEXT_YEAR"))
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--alphas", nargs="+", type=float, default=MARGIN_ALPHAS)
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()
    run(
        args.focal,
        [tuple(args.pair)] if args.pair else None,
        args.outdir,
        alphas=args.alphas,
        ablation=args.ablation,
    )


if __name__ == "__main__":
    main()
