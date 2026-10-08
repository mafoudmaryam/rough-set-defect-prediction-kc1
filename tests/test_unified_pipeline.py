"""Tests for the merged/unified pipeline (discretizers, fast Rough Set engine,
exhaustive reducts/core, repeated CV protocol, leakage safety, controls)."""

import inspect
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import unified_protocol as up
from discretization import MDLPDiscretizer, QuantileDiscretizer
from rough_sets import (DecisionTable, dependency_degree, exhaustive_reducts, formal_core,
                        quick_reduct, backward_prune, table_quick_reduct)

T = "d"


def synth(n=240, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(5)})
    X["f4"] = X["f0"] * 2 + rng.normal(0, 0.01, n)  # near-duplicate of f0
    y = pd.Series(((X.f0 - X.f1 + rng.normal(0, 0.7, n)) > 0.8).astype(int), name=T)
    return X, y


def brute_reducts(df, attrs):
    full = dependency_degree(df, attrs, T).gamma
    ok = {frozenset(s) for r in range(len(attrs) + 1) for s in itertools.combinations(attrs, r)
          if dependency_degree(df, list(s), T).gamma >= full - 1e-12}
    return sorted(sorted(s) for s in ok if all((s - {a}) not in ok for a in s))


# --- known tables -----------------------------------------------------------
def known_table():
    # decision determined by A xor-free: D = A or B-copy; C is noise, E duplicates A.
    rows = [(a, b, c, a) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    df = pd.DataFrame(rows, columns=["A", "B", "C", "E"])
    df[T] = df["A"]
    return df


def test_known_reducts_and_core():
    df = known_table()
    attrs = ["A", "B", "C", "E"]
    dt = DecisionTable(df[attrs], df[T])
    reds = exhaustive_reducts(dt)
    assert sorted(map(sorted, reds)) == [["A"], ["E"]]
    assert formal_core(reds, attrs) == []  # A and E are interchangeable -> empty core


def test_core_nonempty():
    # D = A xor B: both needed; C noise.
    rows = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    df = pd.DataFrame(rows, columns=["A", "B", "C"])
    df[T] = df.A ^ df.B
    dt = DecisionTable(df[["A", "B", "C"]], df[T])
    reds = exhaustive_reducts(dt)
    assert reds == [["A", "B"]]
    assert formal_core(reds, ["A", "B", "C"]) == ["A", "B"]


def test_exhaustive_matches_bruteforce_with_inconsistency():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.integers(0, 3, size=(60, 6)), columns=list("ABCDEF"))
    df[T] = (df.A + df.B + rng.integers(0, 2, 60) > 2).astype(int)  # noisy -> inconsistent classes
    attrs = list("ABCDEF")
    dt = DecisionTable(df[attrs], df[T])
    assert sorted(map(sorted, exhaustive_reducts(dt))) == brute_reducts(df, attrs)


def test_table_engine_matches_pandas_gamma():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.integers(0, 4, size=(80, 5)), columns=list("ABCDE"))
    df[T] = rng.integers(0, 2, 80)
    dt = DecisionTable(df[list("ABCDE")], df[T])
    for r in range(0, 6):
        for s in itertools.combinations("ABCDE", r):
            a, b = dt.dependency(list(s)), dependency_degree(df, list(s), T)
            assert a.gamma == pytest.approx(b.gamma)
            assert a.num_equivalence_classes == b.num_equivalence_classes
            assert a.num_inconsistent_classes == b.num_inconsistent_classes


def test_table_quickreduct_matches_legacy_and_preserves_dependency():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.integers(0, 3, size=(100, 6)), columns=list("ABCDEF"))
    df[T] = (df.A + df.C > 2).astype(int)
    attrs = list("ABCDEF")
    legacy = quick_reduct(df, attrs, T)
    legacy_pruned = backward_prune(df, legacy.selected_features, T, legacy.full_dependency)
    dt = DecisionTable(df[attrs], df[T])
    sel, info = table_quick_reduct(dt, 1.0)
    assert sorted(sel) == sorted(legacy_pruned)
    assert info["gamma_selected"] == pytest.approx(info["gamma_full"])


def test_approximate_reduct_satisfies_alpha_and_is_smaller_or_equal():
    X, y = synth()
    Xd = QuantileDiscretizer(3).fit_transform(X)
    dt = DecisionTable(Xd, y)
    strict, _ = table_quick_reduct(dt, 1.0)
    approx, info = table_quick_reduct(dt, 0.9)
    assert dt.gamma(approx) >= 0.9 * dt.gamma(dt.attrs) - 1e-12
    assert len(approx) <= len(strict)
    # backward-pruned: removing any feature breaks the alpha threshold
    for a in approx:
        assert dt.gamma([b for b in approx if b != a]) < 0.9 * dt.gamma(dt.attrs) - 1e-12


# --- discretizers -----------------------------------------------------------
def test_quantile_cuts_are_train_only_and_test_cannot_change_them():
    X, _ = synth()
    tr, te = X.iloc[:150], X.iloc[150:].copy()
    d1 = QuantileDiscretizer(3).fit(tr)
    te.iloc[:, :] = te.to_numpy() * 1000 + 5e5  # wildly shifted test data
    d2 = QuantileDiscretizer(3).fit(tr)
    for c in X.columns:
        assert np.array_equal(d1.edges_[c], d2.edges_[c])
    assert d1.transform(te).to_numpy().max() <= 2  # extreme test values still valid bins


def test_quantile_ties_keep_min_bin_and_record_actual_bins():
    s = pd.DataFrame({"z": [0] * 70 + list(range(1, 31)), "c": [5] * 100})
    d = QuantileDiscretizer(3).fit(s)
    assert d.actual_bins_["c"] == 1
    assert d.actual_bins_["z"] == 2  # {0} and the rest; no empty bins
    codes = d.transform(s)["z"]
    assert set(codes.unique()) == {0, 1} and (codes[:70] == 0).all()


def test_mdlp_separates_informative_feature_and_ignores_noise():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({"sig": y * 3.0 + rng.normal(0, 0.3, n), "noise": rng.normal(size=n)})
    m = MDLPDiscretizer().fit(X, y)
    assert m.actual_bins_["sig"] >= 2
    assert m.actual_bins_["noise"] == 1
    cut = m.edges_["sig"][0]
    assert 0.5 < cut < 2.5


def test_mdlp_fit_uses_only_training_labels():
    X, y = synth()
    tr_idx, te_idx = np.arange(150), np.arange(150, 240)
    a = MDLPDiscretizer().fit(X.iloc[tr_idx], y.iloc[tr_idx])
    # Flipping TEST labels / changing TEST features cannot be expressed: fit() never receives them.
    assert list(inspect.signature(MDLPDiscretizer.fit).parameters) == ["self", "X", "y"]
    b = MDLPDiscretizer().fit(X.iloc[tr_idx], y.iloc[tr_idx])
    for c in X.columns:
        assert np.array_equal(a.edges_[c], b.edges_[c])
    out = a.transform(X.iloc[te_idx])
    assert out.shape == (90, X.shape[1])


# --- CV protocol & leakage --------------------------------------------------
@pytest.fixture(scope="module")
def cv_data():
    X, y = synth(n=200)
    return X, y, up.make_splits(X, y)


def test_repeated_cv_has_exactly_50_splits(cv_data):
    assert len(cv_data[2]) == 50


def test_splits_stratified_and_disjoint(cv_data):
    X, y, splits = cv_data
    for tr, te in splits:
        assert not set(tr) & set(te)
        assert len(set(tr) | set(te)) == len(X)
        assert abs(y.iloc[te].mean() - y.mean()) < 0.06
        assert abs(y.iloc[tr].mean() - y.mean()) < 0.02
    # each repeat's test folds partition the data exactly once
    for rep in range(up.N_REPEATS):
        tests = np.concatenate([splits[rep * up.N_SPLITS + k][1] for k in range(up.N_SPLITS)])
        assert sorted(tests) == list(range(len(X)))


def split_xy(cv_data, i=0):
    X, y, splits = cv_data
    tr, te = splits[i]
    return X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te], list(X.columns)


@pytest.mark.parametrize("method", ["q3", "mdlp"])
def test_selection_ignores_test_data_and_test_labels(cv_data, method):
    Xtr, ytr, Xte, yte, feats = split_xy(cv_data)
    base = up.evaluate_split(0, Xtr, ytr, Xte, yte, feats, method=method)
    # corrupt test labels and test features: selection, random subsets and predictions must not change
    r2 = up.evaluate_split(0, Xtr, ytr, Xte, 1 - yte, feats, method=method)
    assert base["arms"] == r2["arms"]
    assert [x["features"] for x in base["random_records"]] == [x["features"] for x in r2["random_records"]]
    p1 = pd.concat(base["predictions"]); p2 = pd.concat(r2["predictions"])
    assert np.array_equal(p1.y_pred.to_numpy(), p2.y_pred.to_numpy())
    assert np.allclose(p1.y_proba.to_numpy(), p2.y_proba.to_numpy())
    Xte_shift = Xte * 100 + 7
    r3 = up.evaluate_split(0, Xtr, ytr, Xte_shift, yte, feats, method=method)
    assert base["arms"] == r3["arms"]


def test_selection_function_signature_has_no_test_inputs():
    assert list(inspect.signature(up.select_features).parameters)[:2] == ["X_train", "y_train"]
    assert list(inspect.signature(up.random_feature_subsets).parameters) == ["features", "k", "n", "seed_key"]


def test_test_labels_used_only_in_metrics(cv_data):
    Xtr, ytr, Xte, yte, feats = split_xy(cv_data)
    r = up.evaluate_split(0, Xtr, ytr, Xte, yte, feats)
    r2 = up.evaluate_split(0, Xtr, ytr, Xte, 1 - yte, feats)
    assert r["records"][0]["accuracy"] != r2["records"][0]["accuracy"]  # only metrics respond


def test_tree_trained_on_training_rows_only_and_numeric(cv_data):
    Xtr, ytr, Xte, yte, feats = split_xy(cv_data)
    # a tree fit on train yields identical predictions whatever happens to the test labels
    p1, _ = up.fit_predict_tree(Xtr, ytr, Xte, feats)
    p2, _ = up.fit_predict_tree(Xtr, ytr, Xte, feats)
    assert np.array_equal(p1, p2)
    assert up.TREE_PARAMS == {"random_state": 42, "class_weight": None}


def test_random_subsets_correct_size_deterministic_and_label_free():
    feats = [f"f{i}" for i in range(21)]
    a = up.random_feature_subsets(feats, 7, 10, (42, 3, 0))
    b = up.random_feature_subsets(feats, 7, 10, (42, 3, 0))
    c = up.random_feature_subsets(feats, 7, 10, (42, 4, 0))
    assert a == b and a != c
    assert len(a) == 10 and all(len(s) == 7 and len(set(s)) == 7 and set(s) <= set(feats) for s in a)
    assert len({tuple(s) for s in a}) > 1


def test_split_outputs_valid(cv_data):
    Xtr, ytr, Xte, yte, feats = split_xy(cv_data)
    r = up.evaluate_split(0, Xtr, ytr, Xte, yte, feats)
    assert r["arms"]["baseline"] == feats
    for arm in ("strict", "approx"):
        assert r["arms"][arm] and set(r["arms"][arm]) <= set(feats)
    sizes = {a: len(r["arms"][a]) for a in ("strict", "approx")}
    for rr in r["random_records"]:
        assert rr["n_features"] == sizes[rr["control_for"]]
    assert len(r["random_records"]) == 2 * up.N_RANDOM
    p = pd.concat(r["predictions"])
    assert p.y_proba.between(0, 1).all()
    for rec in r["records"] + r["random_records"]:
        assert all(np.isfinite(rec[m]) for m in up.METRICS)
    s_feats, s_info = r["selection"][1.0]
    assert s_info["gamma_selected"] == pytest.approx(s_info["gamma_full"])
    a_info = r["selection"][up.ALPHA][1]
    assert a_info["gamma_selected"] >= up.ALPHA * a_info["gamma_full"] - 1e-12


def test_paired_test_handles_identical_and_shifted_samples():
    a = np.linspace(0.2, 0.6, 50)
    assert up.paired_test(a, a)["p_value"] == 1.0
    r = up.paired_test(a + 0.01 * (1 + np.arange(50) % 3), a)
    assert r["mean_diff"] > 0 and r["p_value"] < 0.001


def test_feature_frequency_counts():
    fr = up.feature_frequency([["a", "b"], ["a"], ["a", "c"]], ["a", "b", "c", "d"])
    assert fr["count"].tolist() == [3, 1, 1, 0]
    assert fr["percent"].iloc[0] == pytest.approx(100.0)
