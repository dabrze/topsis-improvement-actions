"""Score-space safety margins for strict ranking targets."""

from typing import Any

import pandas as pd


def margin_gap(transformer: Any, target_rank: int) -> float:
    """Return the score gap used to define a relative margin for ``target_rank``.

    Scores are ordered by strict rank and alternative identifier, matching the
    experiment ranking semantics. For rank one, the gap is measured to the
    second-ranked alternative.
    """
    score_column = str(transformer.agg_fn.letter)
    scores = transformer.X_new[score_column].astype(float)
    table = pd.DataFrame({"id": scores.index, "score": scores.to_numpy()})
    table["rank"] = [int((scores > score).sum() + 1) for score in scores]
    scores = table.sort_values(["rank", "id"], kind="stable")["score"].to_numpy(dtype=float)
    if not 1 <= target_rank <= len(scores):
        raise ValueError(f"Target rank {target_rank} is outside 1..{len(scores)}.")
    lower_index = 1 if target_rank == 1 else target_rank - 1
    upper_index = 0 if target_rank == 1 else target_rank - 2
    return abs(scores[upper_index] - scores[lower_index])


def margin_epsilon(transformer: Any, target_rank: int, alpha: float) -> float:
    """Convert a non-negative relative safety margin into score-space epsilon.

    The returned epsilon is ``max(1e-6, alpha * gap)``, where ``gap`` is the
    score distance to the alternative directly above the requested strict rank.
    Rank one uses the score distance between the first and second alternatives.
    """
    if alpha < 0:
        raise ValueError("Margin alpha must be non-negative.")
    return max(1e-6, alpha * margin_gap(transformer, target_rank))
