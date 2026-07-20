"""Shared I/O, ranking, and rendering utilities for the case study."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    CRITERIA,
    DATA_PATH,
    DEFAULT_OUTDIR,
    DISPLAY_NAMES,
    EXPERT_RANGE,
    GROUP_WEIGHTS,
    OBJECTIVES,
    REPO_ROOT,
)

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
import wmsd  # noqa: E402


METHOD_CLASSES = {name: getattr(wmsd, name) for name in ("RTOPSIS", "SAW", "VIKOR", "COPRAS", "ARAS", "WASPAS")}
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9")


def output_dirs(outdir: str | Path | None = None) -> dict[str, Path]:
    root = Path(outdir) if outdir else DEFAULT_OUTDIR
    dirs = {name: root / name for name in ("results", "tables", "figures")}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def load_year(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the criterion-score matrix and strict TOPSIS score table for one edition."""
    data = pd.read_csv(DATA_PATH)
    rows = data.loc[data["year"] == year, ["canonical", *[f"{c}_rel" for c in CRITERIA]]].copy()
    if len(rows) != 23:
        raise ValueError(f"Expected 23 technical universities for {year}, found {len(rows)}.")
    rows = rows.set_index("canonical")
    rows.columns = CRITERIA
    rows = rows.astype(float)
    if rows.isna().any().any():
        raise ValueError(f"Missing criterion score in {year}.")
    transformer, _ = build_transformer("RTOPSIS", rows, year)
    return rows, score_table(transformer)


def build_transformer(
    method: str,
    year_df: pd.DataFrame,
    year: int,
    *,
    weights: Iterable[float] | None = None,
) -> tuple[wmsd.WMSDTransformer, pd.DataFrame]:
    """Fit a method on a criterion matrix using edition-specific group weights."""
    if method not in METHOD_CLASSES:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(METHOD_CLASSES)}.")
    transformer = wmsd.WMSDTransformer(METHOD_CLASSES[method])
    transformed = transformer.fit_transform(
        year_df.loc[:, CRITERIA].copy(),
        list(weights) if weights is not None else GROUP_WEIGHTS[year],
        OBJECTIVES,
        EXPERT_RANGE,
    )
    return transformer, transformed


def score_table(transformer: wmsd.WMSDTransformer) -> pd.DataFrame:
    """Return scores and strict ranks; score ties receive the same rank."""
    score_column = str(transformer.agg_fn.letter)
    scores = transformer.X_new[score_column].astype(float)
    result = pd.DataFrame({"id": scores.index, "score": scores.to_numpy()})
    result["rank"] = [int((scores > score).sum() + 1) for score in scores]
    return result.sort_values(["rank", "id"], kind="stable").reset_index(drop=True)


def rank_of(transformer: wmsd.WMSDTransformer, canonical: str) -> int:
    table = score_table(transformer).set_index("id")
    if canonical not in table.index:
        valid = ", ".join(map(str, table.index.tolist()))
        raise ValueError(f"Unknown focal university {canonical!r}. Valid canonical ids: {valid}")
    return int(table.loc[canonical, "rank"])


margin_epsilon = wmsd.margin_epsilon


def validate_focal(year_df: pd.DataFrame, focal: str) -> None:
    if focal not in year_df.index:
        raise ValueError(
            f"Unknown focal university {focal!r}. Valid canonical ids: {', '.join(year_df.index)}"
        )


def result_deltas(result: Any) -> pd.Series | None:
    """Normalize library return differences to one row of score-scale deltas."""
    if result is None:
        return None
    frame = result[0] if isinstance(result, tuple) else result
    if frame is None or frame.empty or not set(CRITERIA).issubset(frame.columns):
        return None
    return frame.loc[frame.index[0], CRITERIA].astype(float)


def make_plan(
    transformer: wmsd.WMSDTransformer,
    algorithm: str,
    focal: str,
    target_rank: int,
    *,
    features: Iterable[str] | None = None,
    epsilon: float = 1e-6,
    seed: int | None = None,
    popsize: int | None = None,
    n_generations: int = 200,
) -> pd.Series | None:
    """Request one plan, returning ``None`` for an infeasible library result."""
    features = list(features or CRITERIA)
    kwargs: dict[str, Any] = {}
    if algorithm == "improvement_single_feature":
        if len(features) != 1:
            raise ValueError("Single-feature plans require exactly one criterion.")
        kwargs["feature_to_change"] = features[0]
    elif algorithm in {"improvement_features", "improvement_non_linear_programming"}:
        kwargs["features_to_change"] = features
    elif algorithm == "improvement_genetic":
        kwargs.update(
            features_to_change=features,
            popsize=popsize,
            n_generations=n_generations,
        )
        if seed is not None:
            kwargs["seed"] = seed
    elif algorithm != "improvement_mean":
        raise ValueError(f"Unsupported plan algorithm {algorithm!r}.")
    try:
        return result_deltas(transformer.improvement(algorithm, focal, target_rank, epsilon, **kwargs))
    except (ArithmeticError, RuntimeError, ValueError):
        return None


def rank_after_delta(
    field: pd.DataFrame,
    year: int,
    focal: str,
    delta: pd.Series | np.ndarray,
    method: str = "RTOPSIS",
    baseline: pd.Series | None = None,
    weights_year: int | None = None,
) -> tuple[int, pd.DataFrame]:
    """Apply score-scale deltas with clipping and re-rank using strict scores."""
    modified = field.loc[:, CRITERIA].copy()
    source = baseline.loc[CRITERIA] if baseline is not None else modified.loc[focal]
    modified.loc[focal, CRITERIA] = np.clip(
        source.to_numpy(dtype=float) + np.asarray(delta, dtype=float), 0.0, 100.0
    )
    transformer, _ = build_transformer(method, modified, year, weights=GROUP_WEIGHTS[weights_year or year])
    return rank_of(transformer, focal), score_table(transformer)


def normalized_norm(delta: pd.Series | np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(delta, dtype=float) / 100.0))


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, check=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def write_results(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    script: str,
    config: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Write an output CSV and its reproducibility sidecar."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    metadata: dict[str, Any] = {
        "git_commit": git_commit(),
        "config": config,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "script": script,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    destination.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def to_latex_table(
    frame: pd.DataFrame, caption: str, label: str, path: str | Path, *, index: bool = True
) -> None:
    """Write a standalone booktabs table fragment with corrected English labels."""
    display = frame.copy()
    display = display.rename(columns=DISPLAY_NAMES)
    display.index = [DISPLAY_NAMES.get(index, index) for index in display.index]
    latex = display.to_latex(
        index=index,
        escape=True,
        float_format=lambda value: f"{value:.2f}",
        na_rep="—",
        bold_rows=False,
    )
    caption = caption.replace("_", r"\_")
    label = label.replace("_", r"\_")
    rendered = (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"{latex}"
        "\\end{table}\n"
    )
    Path(path).write_text(rendered, encoding="utf-8")


def save_figure(fig: plt.Figure, stem: str | Path) -> None:
    """Save paper figures in both required high-resolution formats."""
    path = Path(stem)
    fig.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def runtime_metadata() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "processor": platform.processor() or "unavailable",
    }
