"""E1: post-factum improvement-action walkthrough for one Perspektywy edition."""

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
    make_plan,
    normalized_norm,
    output_dirs,
    rank_after_delta,
    rank_of,
    result_deltas,
    save_figure,
    score_table,
    to_latex_table,
    validate_focal,
    write_results,
)
from experiments.perspektywy.config import (  # noqa: E402
    EPSILON,
    FOCAL_UNIVERSITY,
    MAIN_YEAR,
    SEED,
    SLOW_CRITERIA,
)


def _target_ranks(rank: int) -> list[int]:
    return list(range(rank - 1, max(1, rank - 3) - 1, -1))


def _records_to_table(records: list[dict], targets: list[int]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {key: record[key] for key in ("criterion",) if key in record}
        if "ordering" in record:
            row["criterion"] = record["ordering"]
        row[f"Rank {record['target_rank']}"] = (
            "—" if not record["feasible"] else record.get("summary", record["norm"])
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby("criterion", sort=False).first().reindex(
        columns=[f"Rank {rank}" for rank in targets]
    )


def run(
    focal: str = FOCAL_UNIVERSITY,
    year: int = MAIN_YEAR,
    outdir: str | Path | None = None,
    *,
    quick: bool = False,
) -> dict[str, Path]:
    dirs = output_dirs(outdir)
    data, _ = load_year(year)
    validate_focal(data, focal)
    transformer, _ = build_transformer("RTOPSIS", data, year)
    rank = rank_of(transformer, focal)
    targets = _target_ranks(rank)
    config = {"focal": focal, "year": year, "epsilon": EPSILON, "quick": quick}

    ranking = data.copy()
    ranked_scores = score_table(transformer).set_index("id")
    ranking["R"] = ranked_scores.loc[ranking.index, "score"]
    ranking["rank"] = ranked_scores.loc[ranking.index, "rank"].astype(int)
    ranking.insert(0, "canonical", ranking.index)
    ranking = ranking.sort_values("rank")
    write_results(ranking, dirs["results"] / "e1_ranking.csv", script=__file__, config=config)
    to_latex_table(
        ranking.set_index("canonical"),
        f"Technical-university ranking in {year} (strict TOPSIS ranks)",
        "tab:e1-ranking",
        dirs["tables"] / "e1_ranking.tex",
    )

    direct_records: list[dict] = []
    for criterion in CRITERIA:
        for target in targets:
            delta = make_plan(
                transformer, "improvement_single_feature", focal, target, features=[criterion]
            )
            feasible = delta is not None and rank_after_delta(data, year, focal, delta)[0] <= target
            direct_records.append(
                {
                    "criterion": criterion,
                    "target_rank": target,
                    "feasible": feasible,
                    "norm": np.nan if not feasible else normalized_norm(delta),
                    **({name: delta[name] for name in CRITERIA} if feasible else {}),
                }
            )
    direct = pd.DataFrame(direct_records)
    write_results(direct, dirs["results"] / "e1_direct.csv", script=__file__, config=config)
    direct_table = direct.pivot(index="criterion", columns="target_rank", values="norm").reindex(CRITERIA)
    direct_table.columns = [f"Rank {column}" for column in direct_table.columns]
    to_latex_table(
        direct_table,
        "Direct single-criterion effort (Euclidean norm in /100 normalized score space)",
        "tab:e1-direct",
        dirs["tables"] / "e1_direct.tex",
    )

    orderings = [
        ["Umiedzynarodowienie", "Innowacyjnosc", "EfektywnoscNaukowa"],
        ["EfektywnoscNaukowa", "Innowacyjnosc", "Umiedzynarodowienie"],
    ]
    lex_records: list[dict] = []
    for ordering in orderings:
        for target in targets:
            delta = make_plan(transformer, "improvement_features", focal, target, features=ordering)
            feasible = delta is not None and rank_after_delta(data, year, focal, delta)[0] <= target
            lex_records.append(
                {
                    "ordering": " → ".join(ordering),
                    "target_rank": target,
                    "feasible": feasible,
                    "norm": np.nan if not feasible else float(np.linalg.norm(delta / 100)),
                    **({name: delta[name] for name in CRITERIA} if feasible else {}),
                }
            )
    lex = pd.DataFrame(lex_records)
    write_results(lex, dirs["results"] / "e1_lexicographic.csv", script=__file__, config=config)
    lex_table = lex.pivot(index="ordering", columns="target_rank", values="norm")
    lex_table.columns = [f"Rank {column}" for column in lex_table.columns]
    to_latex_table(
        lex_table,
        "Lexicographic effort (Euclidean norm in /100 normalized score space)",
        "tab:e1-lexicographic",
        dirs["tables"] / "e1_lexicographic.tex",
    )

    nlp_records: list[dict] = []
    for scope, features in (("All criteria", CRITERIA), ("Fast-moving criteria", [c for c in CRITERIA if c not in SLOW_CRITERIA])):
        for target in targets:
            delta = make_plan(transformer, "improvement_non_linear_programming", focal, target, features=features)
            feasible = delta is not None and rank_after_delta(data, year, focal, delta)[0] <= target
            nlp_records.append(
                {
                    "scope": scope,
                    "target_rank": target,
                    "feasible": feasible,
                    "norm": np.nan if not feasible else float(np.linalg.norm(delta / 100)),
                    **({name: delta[name] for name in CRITERIA} if feasible else {}),
                }
            )
    nlp = pd.DataFrame(nlp_records)
    write_results(nlp, dirs["results"] / "e1_nlp.csv", script=__file__, config=config)
    nlp_table = nlp.pivot(index="scope", columns="target_rank", values="norm")
    nlp_table.columns = [f"Rank {column}" for column in nlp_table.columns]
    to_latex_table(
        nlp_table,
        "NLP effort (Euclidean norm in /100 normalized score space)",
        "tab:e1-nlp",
        dirs["tables"] / "e1_nlp.tex",
    )

    wm_records: list[dict] = []
    for target in targets:
        mean_result = transformer.improvement(
            "improvement_mean", focal, target, EPSILON, solutions_number=None
        )
        mean_delta = (
            np.nan
            if mean_result is None or "Mean" not in mean_result.columns
            else float(mean_result.iloc[0]["Mean"])
        )
        wm_records.append(
            {
                "target_rank": target,
                "feasible": mean_result is not None,
                "mean_delta": mean_delta,
            }
        )
    wm = pd.DataFrame(wm_records)
    write_results(wm, dirs["results"] / "e1_wm.csv", script=__file__, config=config)
    to_latex_table(wm.set_index("target_rank")[["feasible", "mean_delta"]], "Retaining-WM improvement effort", "tab:e1-wm", dirs["tables"] / "e1_wm.tex")

    genetic_records: list[dict] = []
    subset = ["EfektywnoscNaukowa", "Umiedzynarodowienie"]
    for target in targets[:2]:
        raw = transformer.improvement(
            "improvement_genetic",
            focal,
            target,
            EPSILON,
            features_to_change=subset,
            popsize=200 if quick else 1000,
            n_generations=5 if quick else 200,
            seed=SEED,
        )
        population = raw[0] if isinstance(raw, tuple) else None
        if population is not None:
            for _, delta in population.iterrows():
                feasible = rank_after_delta(data, year, focal, delta)[0] <= target
                genetic_records.append(
                    {
                        "target_rank": target,
                        "feasible": feasible,
                        **({name: float(delta[name]) for name in CRITERIA} if feasible else {}),
                    }
                )
        else:
            genetic_records.append({"target_rank": target, "feasible": False})
    genetic = pd.DataFrame(genetic_records)
    write_results(genetic, dirs["results"] / "e1_evolutionary.csv", script=__file__, config=config)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    feasible = genetic.loc[genetic["feasible"] == True]  # noqa: E712
    if not feasible.empty:
        ax.scatter(feasible[subset[0]], feasible[subset[1]], s=18, alpha=0.65, color=PALETTE[0], label="Evolutionary")
    for label, frame, color in (("NLP", nlp, PALETTE[1]), ("Lexicographic", lex, PALETTE[2])):
        subset_frame = frame.loc[frame["feasible"] == True]  # noqa: E712
        if not subset_frame.empty:
            ax.scatter(subset_frame[subset[0]], subset_frame[subset[1]], marker="X", s=65, color=color, label=label)
    ax.set_xlabel(f"Δ {DISPLAY_NAMES[subset[0]]}", fontsize=9)
    ax.set_ylabel(f"Δ {DISPLAY_NAMES[subset[1]]}", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    save_figure(fig, dirs["figures"] / "e1_pareto_2d")
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focal", default=FOCAL_UNIVERSITY)
    parser.add_argument("--year", type=int, default=MAIN_YEAR)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(args.focal, args.year, args.outdir, quick=args.quick)


if __name__ == "__main__":
    main()
