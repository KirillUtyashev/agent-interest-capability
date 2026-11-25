import pandas as pd
import pytest

from task_collection.performance_matching import fixed_sequence_paired_noninferiority


def _scores(width_differences: dict[int, list[float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = pd.DataFrame(
        {
            "condition": [1] * 5,
            "seed": [1, 2, 3, 4, 5],
            "score": [1.0] * 5,
        }
    )
    candidate_rows = []
    for width, differences in width_differences.items():
        for seed, difference in enumerate(differences, start=1):
            candidate_rows.append(
                {
                    "condition": 1,
                    "model_width": width,
                    "seed": seed,
                    "score": 1.0 + difference,
                }
            )
    return pd.DataFrame(candidate_rows), reference


def test_fixed_sequence_stops_at_first_failure() -> None:
    candidate, reference = _scores(
        {
            8: [0.10] * 5,
            4: [0.00] * 5,
            2: [-0.10] * 5,
            1: [0.20] * 5,
        }
    )

    tests, summary = fixed_sequence_paired_noninferiority(
        candidate,
        reference,
        condition_cols="condition",
        width_order=[8, 4, 2, 1],
        candidate_score_col="score",
        reference_score_col="score",
        epsilon=0.05,
    )

    assert tests["decision"].tolist() == [
        "non_inferior",
        "non_inferior",
        "failed",
        "not_tested_after_failure",
    ]
    assert summary.loc[0, "smallest_noninferior_width"] == 4
    assert summary.loc[0, "first_failure_width"] == 2


def test_roles_can_be_reversed_by_swapping_inputs() -> None:
    candidate, reference = _scores({8: [0.10] * 5})
    tests, _ = fixed_sequence_paired_noninferiority(
        reference.assign(model_width=8),
        candidate.drop(columns="model_width"),
        condition_cols="condition",
        width_order=[8],
        candidate_score_col="score",
        reference_score_col="score",
        epsilon=0.05,
    )
    assert tests.loc[0, "decision"] == "failed"


def test_unmatched_seeds_are_rejected() -> None:
    candidate, reference = _scores({8: [0.10] * 5})
    with pytest.raises(ValueError, match="Unmatched seeds"):
        fixed_sequence_paired_noninferiority(
            candidate[candidate["seed"] != 5],
            reference,
            condition_cols="condition",
            width_order=[8],
            candidate_score_col="score",
            reference_score_col="score",
            epsilon=0.05,
        )
