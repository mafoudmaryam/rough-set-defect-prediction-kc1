"""
Unit tests for quantile discretization (src/discretization.py).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from discretization import (
    apply_quantile_bin_edges,
    discretize_dataframe,
    discretize_feature,
    fit_apply_train_test,
    fit_quantile_bin_edges,
)


def test_numerical_data_becomes_integer_categories():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    codes, actual_bins = discretize_feature(series, bins=4)
    assert codes.dtype.kind in ("i", "u")
    assert set(codes.unique()).issubset({0, 1, 2, 3})
    assert actual_bins <= 4


def test_target_is_preserved():
    df = pd.DataFrame({
        "loc": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "defects": [True, False, True, False, True, False, True, False],
    })
    result = discretize_dataframe(df, ["loc"], "defects", bins=4)
    assert list(result.discretized["defects"]) == list(df["defects"])


def test_repeated_values_do_not_crash():
    """Many repeated values (e.g. all 1.0) should collapse gracefully, not raise."""
    series = pd.Series([1.0] * 10)
    codes, actual_bins = discretize_feature(series, bins=4)
    assert actual_bins == 1
    assert (codes == 0).all()


def test_mostly_repeated_values_report_fewer_bins():
    series = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    codes, actual_bins = discretize_feature(series, bins=4)
    assert actual_bins < 4
    assert codes.notna().all()


def test_actual_bin_counts_reported_per_feature():
    df = pd.DataFrame({
        "constant": [5.0] * 6,
        "varied": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "defects": [True, False, True, False, True, False],
    })
    result = discretize_dataframe(df, ["constant", "varied"], "defects", bins=4)
    assert result.actual_bins_per_feature["constant"] == 1
    assert result.actual_bins_per_feature["varied"] <= 4
    assert result.requested_bins == 4


def test_train_test_edges_fit_on_training_data_only():
    """
    Bin edges must come from the training series alone. Changing the test
    series (even drastically, e.g. all-outliers) must not change the edges
    or the codes assigned to the (unchanged) training series.
    """
    train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    test_a = pd.Series([2.5, 6.5])
    test_b = pd.Series([-1000.0, 1000.0, 0.0])  # wildly different test fold

    edges_a = fit_quantile_bin_edges(train, bins=4)
    train_codes_a, test_codes_a, bins_a = fit_apply_train_test(train, test_a, bins=4)
    train_codes_b, test_codes_b, bins_b = fit_apply_train_test(train, test_b, bins=4)

    assert list(train_codes_a) == list(train_codes_b)
    assert bins_a == bins_b
    # The edges themselves are a pure function of the training series.
    assert list(edges_a) == list(fit_quantile_bin_edges(train, bins=4))


def test_out_of_range_test_values_are_clipped_not_dropped():
    """
    A test-fold value more extreme than anything in training must still
    receive a valid category (clipped to the outer bin), not NaN.
    """
    train = pd.Series([10.0, 20.0, 30.0, 40.0])
    edges = fit_quantile_bin_edges(train, bins=4)
    test = pd.Series([-500.0, 500.0])
    codes = apply_quantile_bin_edges(test, edges)
    assert codes.notna().all()
    assert codes.iloc[0] == 0  # below range -> first bin
    assert codes.iloc[1] == codes.max()  # above range -> last bin


def test_constant_training_feature_does_not_crash_apply():
    train = pd.Series([7.0] * 5)
    test = pd.Series([7.0, 3.0, 99.0])
    train_codes, test_codes, actual_bins = fit_apply_train_test(train, test, bins=8)
    assert actual_bins == 1
    assert (train_codes == 0).all()
    assert (test_codes == 0).all()
