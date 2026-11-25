"""Paired fixed-sequence non-inferiority analysis for model-width sweeps."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def fixed_sequence_paired_noninferiority(
    candidate_scores: pd.DataFrame,
    reference_scores: pd.DataFrame,
    *,
    condition_cols: str | Sequence[str],
    width_order: Sequence[int],
    width_col: str = "model_width",
    seed_col: str = "seed",
    candidate_score_col: str = "seed_performance",
    reference_score_col: str = "seed_performance",
    epsilon: float = 0.05,
    alpha: float = 0.05,
    expected_seeds: int | None = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test candidate widths against a fixed reference using matched seeds.

    The caller selects the fixed reference before calling this function. To
    reverse the algorithm roles, pass the varying centralized scores as
    ``candidate_scores`` and the fixed decentralized scores as
    ``reference_scores``.

    Widths are tested in exactly ``width_order``. Testing stops within each
    condition at the first failure, and later widths are reported but not
    tested. A width passes when the one-sided lower confidence bound for
    ``candidate - reference`` is strictly greater than ``-epsilon``.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if not width_order:
        raise ValueError("width_order must contain at least one width")
    if len(set(width_order)) != len(width_order):
        raise ValueError(f"width_order contains duplicates: {width_order}")

    conditions = [condition_cols] if isinstance(condition_cols, str) else list(condition_cols)
    if not conditions:
        raise ValueError("condition_cols must contain at least one column")

    candidate_required = set(conditions + [width_col, seed_col, candidate_score_col])
    reference_required = set(conditions + [seed_col, reference_score_col])
    missing_candidate = candidate_required.difference(candidate_scores.columns)
    missing_reference = reference_required.difference(reference_scores.columns)
    if missing_candidate:
        raise ValueError(f"candidate_scores is missing columns: {sorted(missing_candidate)}")
    if missing_reference:
        raise ValueError(f"reference_scores is missing columns: {sorted(missing_reference)}")

    candidate_key = conditions + [width_col, seed_col]
    reference_key = conditions + [seed_col]
    if candidate_scores.duplicated(candidate_key).any():
        raise ValueError(f"candidate_scores has duplicate rows for {candidate_key}")
    if reference_scores.duplicated(reference_key).any():
        raise ValueError(f"reference_scores has duplicate rows for {reference_key}")

    candidate_conditions = candidate_scores[conditions].drop_duplicates()
    reference_conditions = reference_scores[conditions].drop_duplicates()
    condition_check = candidate_conditions.merge(
        reference_conditions,
        on=conditions,
        how="outer",
        indicator=True,
    )
    if not condition_check["_merge"].eq("both").all():
        raise ValueError(
            "Candidate and reference condition sets differ: "
            f"{condition_check.to_dict(orient='records')}"
        )

    test_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for condition_values in candidate_conditions.itertuples(index=False, name=None):
        condition = dict(zip(conditions, condition_values))
        candidate_condition = candidate_scores
        reference_condition = reference_scores
        for name, value in condition.items():
            candidate_condition = candidate_condition[candidate_condition[name] == value]
            reference_condition = reference_condition[reference_condition[name] == value]

        sequence_open = True
        smallest_established_width: int | None = None
        first_failure_width: int | None = None

        for sequence_index, width in enumerate(width_order):
            candidate_width = candidate_condition[candidate_condition[width_col] == width]
            if candidate_width.empty:
                raise ValueError(f"Missing candidate width {width} for condition {condition}")

            paired = candidate_width[
                conditions + [seed_col, candidate_score_col]
            ].rename(columns={candidate_score_col: "candidate_performance"}).merge(
                reference_condition[
                    conditions + [seed_col, reference_score_col]
                ].rename(columns={reference_score_col: "reference_performance"}),
                on=conditions + [seed_col],
                how="outer",
                validate="one_to_one",
                indicator=True,
            )
            if not paired["_merge"].eq("both").all():
                raise ValueError(
                    f"Unmatched seeds for width {width}, condition {condition}: "
                    f"{paired[[seed_col, '_merge']].to_dict(orient='records')}"
                )
            paired = paired.drop(columns="_merge").sort_values(seed_col)
            n_seeds = len(paired)
            if expected_seeds is not None and n_seeds != expected_seeds:
                raise ValueError(
                    f"Expected {expected_seeds} matched seeds for width {width}, "
                    f"condition {condition}; found {n_seeds}"
                )
            if n_seeds < 2:
                raise ValueError("At least two matched seeds are required")

            differences = (
                paired["candidate_performance"] - paired["reference_performance"]
            ).to_numpy(dtype=float)
            mean_difference = float(np.mean(differences))
            difference_std = float(np.std(differences, ddof=1))
            standard_error = difference_std / np.sqrt(n_seeds)
            degrees_freedom = n_seeds - 1
            critical_value = float(student_t.ppf(1.0 - alpha, degrees_freedom))
            lower_bound = mean_difference - critical_value * standard_error

            if standard_error == 0.0:
                boundary_distance = mean_difference + epsilon
                test_statistic = (
                    np.inf if boundary_distance > 0.0
                    else -np.inf if boundary_distance < 0.0
                    else 0.0
                )
                p_value = 0.0 if boundary_distance > 0.0 else 1.0 if boundary_distance < 0.0 else 0.5
            else:
                test_statistic = (mean_difference + epsilon) / standard_error
                p_value = float(student_t.sf(test_statistic, degrees_freedom))

            confidence_passes = bool(lower_bound > -epsilon)
            tested = sequence_open
            established = tested and confidence_passes
            if established:
                smallest_established_width = int(width)
                decision = "non_inferior"
            elif tested:
                first_failure_width = int(width)
                sequence_open = False
                decision = "failed"
            else:
                decision = "not_tested_after_failure"

            row: dict[str, object] = {
                **condition,
                "sequence_index": sequence_index,
                "model_width": int(width),
                "n_seeds": n_seeds,
                "seeds": tuple(int(seed) for seed in paired[seed_col]),
                "candidate_mean": float(paired["candidate_performance"].mean()),
                "reference_mean": float(paired["reference_performance"].mean()),
                "mean_difference": mean_difference,
                "difference_std": difference_std,
                "standard_error": standard_error,
                "degrees_freedom": degrees_freedom,
                "critical_value": critical_value,
                "lower_confidence_bound": lower_bound,
                "noninferiority_boundary": -epsilon,
                "test_statistic": float(test_statistic),
                "one_sided_p_value": p_value,
                "confidence_passes": confidence_passes,
                "tested": tested,
                "established_noninferior": established,
                "decision": decision,
            }
            test_rows.append(row)

        summary_rows.append(
            {
                **condition,
                "smallest_noninferior_width": smallest_established_width,
                "first_failure_width": first_failure_width,
                "all_widths_noninferior": first_failure_width is None,
                "epsilon": epsilon,
                "alpha": alpha,
                "expected_seeds": expected_seeds,
                "width_order": tuple(int(width) for width in width_order),
            }
        )

    tests = pd.DataFrame(test_rows).sort_values(conditions + ["sequence_index"])
    summary = pd.DataFrame(summary_rows).sort_values(conditions)
    return tests.reset_index(drop=True), summary.reset_index(drop=True)
