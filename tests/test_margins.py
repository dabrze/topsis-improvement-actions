import numpy as np
import pytest

import wmsd
from experiments.perspektywy.common import build_transformer, load_year, score_table


@pytest.fixture
def transformer():
    data, _ = load_year(2025)
    transformer, _ = build_transformer("RTOPSIS", data, 2025)
    return transformer


def test_margin_gap_uses_rank_one_special_case(transformer):
    scores = score_table(transformer)["score"].to_numpy()

    assert wmsd.margin_gap(transformer, 1) == pytest.approx(abs(scores[0] - scores[1]))


def test_margin_epsilon_uses_interior_rank_gap(transformer):
    scores = score_table(transformer)["score"].to_numpy()

    assert wmsd.margin_epsilon(transformer, 4, 0.5) == pytest.approx(
        abs(scores[2] - scores[3]) / 2
    )


def test_margin_epsilon_applies_floor_for_zero_alpha(transformer):
    assert wmsd.margin_epsilon(transformer, 4, 0.0) == pytest.approx(1e-6)


def test_margin_epsilon_rejects_negative_alpha(transformer):
    with pytest.raises(ValueError, match="Margin alpha must be non-negative"):
        wmsd.margin_epsilon(transformer, 4, -np.finfo(float).eps)
