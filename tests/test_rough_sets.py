"""
Unit tests for the Rough Set core logic (src/rough_sets.py).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rough_sets import backward_prune, dependency_degree, quick_reduct, significance

TARGET = "decision"


def test_perfect_classification():
    """A feature that uniquely separates decision classes gives dependency 1.0."""
    df = pd.DataFrame({
        "A": [0, 1, 2, 3],
        TARGET: [False, True, False, True],
    })
    result = dependency_degree(df, ["A"], TARGET)
    assert result.gamma == pytest.approx(1.0)
    assert result.positive_region_count == 4
    assert result.num_inconsistent_classes == 0


def test_inconsistent_equivalence_class():
    """
    Objects 0 and 1 share A=0 but disagree in decision -> inconsistent class,
    contributes nothing to the positive region. Object 2 (A=1) is alone and
    consistent, so contributes 1 object out of 3.
    """
    df = pd.DataFrame({
        "A": [0, 0, 1],
        TARGET: [False, True, True],
    })
    result = dependency_degree(df, ["A"], TARGET)
    assert result.gamma == pytest.approx(1 / 3)
    assert result.positive_region_count == 1
    assert result.num_equivalence_classes == 2
    assert result.num_inconsistent_classes == 1


def test_redundant_attribute_not_required():
    """
    A alone gives the same dependency as A+B (B is redundant given A).
    QuickReduct + pruning should not require B in the final reduct.
    """
    df = pd.DataFrame({
        "A": [0, 0, 1, 1],
        "B": [0, 1, 0, 1],
        TARGET: [False, False, True, True],
    })
    gamma_a = dependency_degree(df, ["A"], TARGET).gamma
    gamma_ab = dependency_degree(df, ["A", "B"], TARGET).gamma
    assert gamma_a == pytest.approx(gamma_ab)
    assert gamma_a == pytest.approx(1.0)

    reduct_result = quick_reduct(df, ["A", "B"], TARGET)
    pruned = backward_prune(df, reduct_result.selected_features, TARGET, reduct_result.full_dependency)
    assert pruned == ["A"]


def test_deterministic_tie_breaking():
    """
    A and B each independently perfectly separate the decision, so they tie
    on dependency gain in the first QuickReduct iteration. The tie must be
    broken by original column order: A (listed first) is chosen.
    """
    df = pd.DataFrame({
        "A": [0, 0, 1, 1],
        "B": [0, 0, 1, 1],
        TARGET: [False, False, True, True],
    })
    reduct_result = quick_reduct(df, ["A", "B"], TARGET)
    assert reduct_result.iterations[0].chosen_feature == "A"

    # Reversing the candidate order should still pick whichever is listed
    # first in that call, confirming the tie-break follows argument order.
    reduct_result_reversed = quick_reduct(df, ["B", "A"], TARGET)
    assert reduct_result_reversed.iterations[0].chosen_feature == "B"


def test_empty_feature_subset_single_label():
    """With P = {} and a single decision label overall, dependency is 1.0."""
    df = pd.DataFrame({
        "A": [0, 1, 2],
        TARGET: [True, True, True],
    })
    result = dependency_degree(df, [], TARGET)
    assert result.gamma == pytest.approx(1.0)
    assert result.num_equivalence_classes == 1
    assert result.num_inconsistent_classes == 0


def test_empty_feature_subset_multiple_labels():
    """With P = {} and multiple decision labels overall, dependency is 0.0."""
    df = pd.DataFrame({
        "A": [0, 1, 2],
        TARGET: [True, False, True],
    })
    result = dependency_degree(df, [], TARGET)
    assert result.gamma == pytest.approx(0.0)
    assert result.num_equivalence_classes == 1
    assert result.num_inconsistent_classes == 1


def test_significance_matches_dependency_difference():
    df = pd.DataFrame({
        "A": [0, 0, 1, 1],
        "B": [0, 1, 0, 1],
        TARGET: [False, True, True, True],
    })
    gamma_empty = dependency_degree(df, [], TARGET).gamma
    gamma_a = dependency_degree(df, ["A"], TARGET).gamma
    sig = significance(df, [], "A", TARGET)
    assert sig == pytest.approx(gamma_a - gamma_empty)


def test_quick_reduct_reaches_full_dependency():
    df = pd.DataFrame({
        "A": [0, 0, 1, 1, 0, 1],
        "B": [0, 1, 0, 1, 1, 0],
        "C": [1, 1, 1, 1, 1, 1],
        TARGET: [False, False, True, True, False, True],
    })
    full_gamma = dependency_degree(df, ["A", "B", "C"], TARGET).gamma
    reduct_result = quick_reduct(df, ["A", "B", "C"], TARGET)
    selected_gamma = dependency_degree(df, reduct_result.selected_features, TARGET).gamma
    assert selected_gamma == pytest.approx(full_gamma)
    assert reduct_result.full_dependency == pytest.approx(full_gamma)
