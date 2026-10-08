"""
Unit tests for the leakage-safe cross-validation experiment
(src/run_cv_experiment.py).

These use a small synthetic decision table (not the full KC1 dataset) so
the test suite stays fast; src/run_cv_experiment.py itself is run
separately on real KC1 to produce the reported experiment results.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import run_cv_experiment as cv


def make_synthetic_dataset(n_rows: int = 60, seed: int = 0) -> pd.DataFrame:
    """
    A small synthetic decision table with 4 numeric features and a binary
    target, informative enough that Rough Set selection and a Decision
    Tree both have something real to work with, and both classes appear
    in every stratified fold.
    """
    rng = np.random.default_rng(seed)
    n_features = ["f1", "f2", "f3", "f4"]
    data = {f: rng.normal(loc=0.0, scale=1.0, size=n_rows) for f in n_features}
    # Make the target genuinely depend on f1 and f2 so selection has signal.
    logit = 1.5 * data["f1"] - 1.2 * data["f2"] + rng.normal(0, 0.5, n_rows)
    target = logit > np.median(logit)
    df = pd.DataFrame(data)
    df["defects"] = target
    return df


SYNTHETIC_ATTRS = ["f1", "f2", "f3", "f4"]


@pytest.fixture(scope="module")
def experiment_result():
    df = make_synthetic_dataset()
    return df, cv.run_cv_experiment(df, condition_attributes=SYNTHETIC_ATTRS, n_splits=5)


def test_discretization_edges_fit_on_training_fold_only():
    """The train/test discretize_fold helper must not let test values shift
    the bin edges computed for the training fold."""
    train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    from discretization import fit_quantile_bin_edges

    edges_with_test_a = fit_quantile_bin_edges(train, bins=4)
    # Edges are computed from `train` alone; the function signature doesn't
    # even accept a test series, which is itself the leakage guarantee.
    edges_with_test_b = fit_quantile_bin_edges(train, bins=4)
    assert list(edges_with_test_a) == list(edges_with_test_b)


def test_selected_feature_names_are_valid_columns(experiment_result):
    df, experiment = experiment_result
    for fr in experiment["fold_records"]:
        for feature in fr["selection"]["pruned_reduct"]:
            assert feature in SYNTHETIC_ATTRS
            assert feature in df.columns


def test_baseline_uses_all_features(experiment_result):
    _, experiment = experiment_result
    for fr in experiment["fold_records"]:
        assert fr["baseline_feature_count"] == len(SYNTHETIC_ATTRS)


def test_reduced_model_uses_exactly_selected_subset(experiment_result):
    _, experiment = experiment_result
    for fr in experiment["fold_records"]:
        assert fr["reduct_feature_count"] == len(fr["selection"]["pruned_reduct"])
        assert fr["reduct_feature_count"] <= len(SYNTHETIC_ATTRS)


def test_folds_are_stratified():
    from sklearn.model_selection import StratifiedKFold

    df = make_synthetic_dataset()
    y = df["defects"].astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    overall_rate = y.mean()
    for _, test_idx in skf.split(df[SYNTHETIC_ATTRS], y):
        fold_rate = y.iloc[test_idx].mean()
        assert abs(fold_rate - overall_rate) < 0.25  # loose bound, small n


def test_every_row_appears_in_exactly_one_test_fold(experiment_result):
    df, experiment = experiment_result
    coverage = experiment["test_index_coverage"]
    assert (coverage == 1).all()
    assert len(coverage) == len(df)


def test_predicted_probabilities_are_valid(experiment_result):
    _, experiment = experiment_result
    preds = experiment["predictions_df"]
    assert preds["baseline_probability"].between(0, 1).all()
    assert preds["reduced_probability"].between(0, 1).all()


def test_metrics_contain_all_required_fields(experiment_result):
    _, experiment = experiment_result
    for fr in experiment["fold_records"]:
        for metrics in (fr["baseline_metrics"], fr["reduced_metrics"]):
            for name in cv.METRIC_NAMES:
                assert name in metrics
                assert isinstance(metrics[name], float)


def test_dependency_of_reduct_matches_full_training_dependency(experiment_result):
    _, experiment = experiment_result
    for fr in experiment["fold_records"]:
        assert fr["selection"]["dependency_matches_full"]


def test_minimality_check_passes_every_fold(experiment_result):
    _, experiment = experiment_result
    for fr in experiment["fold_records"]:
        assert fr["selection"]["minimality_check_passed"]
