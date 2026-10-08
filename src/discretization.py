"""
Quantile-based discretization for Rough Set analysis.

Rough Set Theory requires categorical (discrete) condition attributes in
order to compute indiscernibility relations. KC1's 21 software metrics are
continuous and heavily skewed with large outliers, so this module converts
each numerical feature into a small number of integer category codes using
quantile bins (equal-frequency bins) rather than equal-width bins.

IMPORTANT METHODOLOGY NOTE:
This module discretizes using quantiles computed over whichever DataFrame is
passed in. When called on the full dataset (as in run_rough_reduction.py),
this is valid ONLY for validating the Rough Set engine itself, not for a
predictive evaluation. A future cross-validation protocol must fit quantile
edges on the training fold only and apply them to the test fold, to avoid
leakage. That protocol is out of scope for this task.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_BINS = 4


@dataclass
class DiscretizationResult:
    """Holds the discretized table plus bookkeeping about actual bin counts."""

    discretized: pd.DataFrame
    actual_bins_per_feature: dict[str, int]
    requested_bins: int


def discretize_feature(series: pd.Series, bins: int = DEFAULT_BINS) -> tuple[pd.Series, int]:
    """
    Discretize a single numerical feature into integer category codes
    (0, 1, 2, ...) using quantile-based binning (pandas.qcut).

    Quantile binning places roughly equal numbers of objects in each bin,
    which suits skewed software-metric distributions better than equal-width
    binning.

    If the feature has many repeated values (common in KC1, e.g. many rows
    with loc == 1), the requested quantile edges may collapse into fewer
    than `bins` unique edges. duplicates="drop" is used to handle this
    gracefully instead of raising an error; the actual number of bins
    produced is returned alongside the discretized series.

    A constant feature (a single unique value) cannot be split into more
    than one bin; in that case every object receives category code 0.

    Returns:
        (discretized_series, actual_bin_count)
    """
    if series.nunique(dropna=False) <= 1:
        return pd.Series(0, index=series.index, name=series.name), 1

    codes = pd.qcut(series, q=bins, labels=False, duplicates="drop")
    actual_bins = int(codes.nunique(dropna=True))
    codes = codes.astype("int64")
    codes.name = series.name
    return codes, actual_bins


def discretize_dataframe(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    bins: int = DEFAULT_BINS,
) -> DiscretizationResult:
    """
    Discretize every column in `feature_columns` using quantile binning and
    leave `target_column` (and any other non-feature columns) unchanged.

    Discretization is deterministic: pandas.qcut on a fixed input series
    always produces the same bin edges and codes.

    Returns a DiscretizationResult with the new DataFrame (feature columns
    replaced by integer codes, target column untouched) and a record of the
    actual number of bins produced per feature.
    """
    result_df = df.copy()
    actual_bins_per_feature: dict[str, int] = {}

    for column in feature_columns:
        codes, actual_bins = discretize_feature(df[column], bins=bins)
        result_df[column] = codes
        actual_bins_per_feature[column] = actual_bins

    return DiscretizationResult(
        discretized=result_df,
        actual_bins_per_feature=actual_bins_per_feature,
        requested_bins=bins,
    )


# ---------------------------------------------------------------------------
# Leakage-safe train/test discretization.
#
# discretize_feature/discretize_dataframe above call pandas.qcut directly on
# whatever DataFrame they are given. That is fine for validating the Rough
# Set engine on the full dataset, but it must NOT be used inside a
# train/test split: calling qcut separately on the test fold would let test
# values influence their own bin edges, leaking information the model
# should not have.
#
# The functions below split "fit" (compute bin edges from training data
# only) from "apply" (assign category codes to any data, train or test,
# using those already-fitted edges), matching the fit/transform contract a
# cross-validation loop needs.
# ---------------------------------------------------------------------------


def fit_quantile_bin_edges(train_series: pd.Series, bins: int = DEFAULT_BINS) -> np.ndarray:
    """
    Compute quantile bin edges from a TRAINING series only.

    Returns a sorted array of unique edges. Duplicate quantile edges
    (caused by repeated values in the training data) are dropped, so the
    number of resulting bins is `len(edges) - 1`, which may be less than
    `bins`. A constant training feature yields a single edge (0 usable
    bins boundaries), signalling that every value must map to category 0.
    """
    if train_series.nunique(dropna=False) <= 1:
        value = train_series.iloc[0]
        return np.array([value, value])

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = train_series.quantile(quantiles).to_numpy()
    edges = np.unique(edges)
    if len(edges) < 2:
        # All quantiles collapsed to one value despite nunique() > 1
        # (e.g. one extreme outlier among otherwise-identical values).
        value = train_series.iloc[0]
        edges = np.array([value, value])
    return edges


def apply_quantile_bin_edges(series: pd.Series, edges: np.ndarray) -> pd.Series:
    """
    Assign integer category codes to `series` using FIXED bin edges that
    were already fitted elsewhere (typically on a training fold).

    This is the "apply to unseen data" half of the fit/apply split: it does
    not look at `series` to decide where the boundaries go, so it is safe
    to call on a held-out validation/test fold without leaking test values
    into the discretization.

    Values outside the training edge range are clipped to the nearest
    training boundary (i.e. the outer bins are treated as open-ended) so a
    test-fold value more extreme than anything seen in training still gets
    a valid category instead of becoming NaN or raising an error.
    """
    if len(edges) < 2 or edges[0] == edges[-1]:
        return pd.Series(0, index=series.index, name=series.name)

    clipped = series.clip(lower=edges[0], upper=edges[-1])
    codes = pd.cut(
        clipped, bins=edges, labels=False, include_lowest=True, duplicates="drop"
    )
    codes = codes.astype("int64")
    codes.name = series.name
    return codes


def fit_apply_train_test(
    train_series: pd.Series, test_series: pd.Series, bins: int = DEFAULT_BINS
) -> tuple[pd.Series, pd.Series, int]:
    """
    Fit quantile bin edges on `train_series` only, then apply those edges
    to both `train_series` and `test_series`.

    Returns (train_codes, test_codes, actual_bin_count). This is the
    building block for leakage-safe per-fold discretization in the
    cross-validation experiment: the test fold's own values never
    contribute to where the bin boundaries are placed.
    """
    edges = fit_quantile_bin_edges(train_series, bins=bins)
    train_codes = apply_quantile_bin_edges(train_series, edges)
    test_codes = apply_quantile_bin_edges(test_series, edges)
    actual_bins = max(len(edges) - 1, 1)
    return train_codes, test_codes, actual_bins


# ---------------------------------------------------------------------------
# Unified-pipeline discretizers (fit on TRAINING data only, then transform).
#
# PRIMARY_BINS = 3 (Low / Medium / High) is the official primary setup.
# QuantileDiscretizer fits interior-quantile cut points on training data only.
# MDLPDiscretizer is a supervised ALTERNATIVE used for sensitivity analysis
# only; its fit() receives training labels and nothing else.
# ---------------------------------------------------------------------------

PRIMARY_BINS = 3


class QuantileDiscretizer:
    """
    Equal-frequency discretizer: fit(X_train) -> transform(any X).

    Cut points are the INTERIOR quantiles (1/b, ..., (b-1)/b) of the training
    column; duplicate cut points (heavy ties) are merged. A value equal to a
    cut point falls in the lower bin. Unlike the legacy edge-based helpers
    above, a tied lowest quantile (e.g. many zeros) still leaves "== min" as
    its own bin instead of being merged into the next one. Actual bin counts
    are the number of NON-EMPTY training bins (recorded in `actual_bins_`).
    """

    def __init__(self, bins: int = PRIMARY_BINS):
        self.bins = bins

    def fit(self, X: pd.DataFrame) -> "QuantileDiscretizer":
        qs = np.linspace(0.0, 1.0, self.bins + 1)[1:-1]
        self.edges_ = {c: np.unique(np.quantile(X[c].to_numpy(dtype=float), qs)) for c in X.columns}
        self.actual_bins_ = {
            c: int(np.unique(np.searchsorted(self.edges_[c], X[c].to_numpy(dtype=float), side="left")).size)
            for c in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {c: np.searchsorted(self.edges_[c], X[c].to_numpy(dtype=float), side="left") for c in X.columns},
            index=X.index,
        )

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def _entropy_vec(counts: np.ndarray) -> np.ndarray:
    """Shannon entropy (bits) of rows of a (n, 2) count matrix."""
    total = counts.sum(axis=1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = counts / total[:, None]
        terms = np.where(counts > 0, p * np.log2(p), 0.0)
    return -terms.sum(axis=1)


class MDLPDiscretizer:
    """
    Supervised entropy-based discretization with the Fayyad & Irani (1993)
    MDL stopping rule, for a binary label. A feature that never passes the
    MDL test gets a single bin. Cut points are midpoints between adjacent
    distinct values. Adapted from the teammate's implementation (same
    criterion), vectorised over candidate cuts for speed.
    """

    def fit(self, X: pd.DataFrame, y) -> "MDLPDiscretizer":
        y = np.asarray(y).astype(int)
        self.edges_ = {
            c: np.array(sorted(self._cuts(X[c].to_numpy(dtype=float), y))) for c in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {c: np.searchsorted(self.edges_[c], X[c].to_numpy(dtype=float), side="left") for c in X.columns},
            index=X.index,
        )

    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @property
    def actual_bins_(self) -> dict[str, int]:
        return {c: len(e) + 1 for c, e in self.edges_.items()}

    def _cuts(self, x: np.ndarray, y: np.ndarray) -> list[float]:
        order = np.argsort(x, kind="mergesort")
        out: list[float] = []
        self._split(x[order], y[order], out)
        return out

    def _split(self, x: np.ndarray, y: np.ndarray, out: list[float]) -> None:
        n = len(y)
        if n < 2:
            return
        total = np.bincount(y, minlength=2)
        ent = float(_entropy_vec(total[None, :])[0])
        cand = np.flatnonzero(x[1:] != x[:-1])
        if len(cand) == 0:
            return
        cum = np.cumsum(np.eye(2, dtype=np.int64)[y], axis=0)
        left = cum[cand]
        right = total[None, :] - left
        n_left = cand + 1
        e_left, e_right = _entropy_vec(left), _entropy_vec(right)
        weighted = (n_left * e_left + (n - n_left) * e_right) / n
        b = int(np.argmin(weighted))  # first minimum -> deterministic
        gain = ent - weighted[b]
        k = int((total > 0).sum())
        k1, k2 = int((left[b] > 0).sum()), int((right[b] > 0).sum())
        delta = np.log2(3**k - 2) - (k * ent - k1 * e_left[b] - k2 * e_right[b])
        if gain <= (np.log2(n - 1) + delta) / n:
            return
        i = int(cand[b])
        out.append(float((x[i] + x[i + 1]) / 2))
        self._split(x[: i + 1], y[: i + 1], out)
        self._split(x[i + 1:], y[i + 1:], out)
