"""
Unified evaluation protocol: 10x5 repeated stratified CV, train-only
discretization + Rough Set selection, same-size random controls, metrics,
paired Wilcoxon tests and selection-frequency summaries.

Leakage rules enforced by construction: select_features() receives training
rows/labels only; random_feature_subsets() receives no data at all; test
labels are touched only inside compute_metrics().
"""

from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from discretization import MDLPDiscretizer, QuantileDiscretizer, PRIMARY_BINS
from rough_sets import DecisionTable, table_quick_reduct

TARGET = "defects"
N_SPLITS, N_REPEATS, CV_SEED = 10, 5, 42
ALPHA = 0.90
N_RANDOM = 10
TREE_PARAMS = {"random_state": 42, "class_weight": None}
METRICS = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc"]
TEST_METRICS = ["f1", "mcc", "roc_auc", "pr_auc"]


def load_kc1(path):
    df = pd.read_csv(path)
    features = [c for c in df.columns if c != TARGET]
    return df, features


def make_splits(X, y):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=CV_SEED)
    return list(cv.split(X, y))


def make_discretizer(method: str):
    """method: 'q3', 'q5', 'q8' (quantile, any int) or 'mdlp'."""
    if method == "mdlp":
        return MDLPDiscretizer()
    if method.startswith("q"):
        return QuantileDiscretizer(int(method[1:]))
    raise ValueError(method)


def select_features(X_train: pd.DataFrame, y_train, method: str = f"q{PRIMARY_BINS}",
                    alphas=(1.0, ALPHA)) -> dict:
    """
    TRAINING DATA ONLY. Fit discretizer on X_train (and y_train for MDLP),
    discretize the training copy, run QuickReduct + backward pruning for each
    alpha. Returns {alpha: (features, info)} plus bookkeeping.
    """
    disc = make_discretizer(method)
    disc = disc.fit(X_train, y_train) if method == "mdlp" else disc.fit(X_train)
    dt = DecisionTable(disc.transform(X_train), y_train)
    out = {"actual_bins": dict(disc.actual_bins_), "gamma_full": dt.gamma(dt.attrs)}
    for a in alphas:
        out[a] = table_quick_reduct(dt, alpha=a)
    return out


def random_feature_subsets(features: list[str], k: int, n: int, seed_key: tuple) -> list[list[str]]:
    """n random size-k subsets drawn with a deterministic seed. Uses no data/labels."""
    rng = np.random.default_rng(list(seed_key))
    out = []
    for _ in range(n):
        idx = np.sort(rng.choice(len(features), size=k, replace=False))
        out.append([features[i] for i in idx])
    return out


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def fit_predict_tree(X_train, y_train, X_test, cols):
    """Decision Tree on ORIGINAL numeric columns `cols`; train rows only."""
    clf = DecisionTreeClassifier(**TREE_PARAMS)
    clf.fit(X_train[cols], y_train)
    proba = clf.predict_proba(X_test[cols])[:, list(clf.classes_).index(1)]
    return clf.predict(X_test[cols]), proba


def evaluate_split(split_id, X_train, y_train, X_test, y_test, features, method=f"q{PRIMARY_BINS}",
                   keep_predictions=True) -> dict:
    """One split of the primary protocol. Returns records for all arms."""
    sel = select_features(X_train, y_train, method)
    arms = {"baseline": list(features), "strict": sel[1.0][0], "approx": sel[ALPHA][0]}
    records, rand_records, preds = [], [], []
    for arm, cols in arms.items():
        pred, proba = fit_predict_tree(X_train, y_train, X_test, cols)
        records.append({"split": split_id, "arm": arm, "n_features": len(cols), **compute_metrics(y_test, pred, proba)})
        if keep_predictions:
            preds.append(pd.DataFrame({"split": split_id, "model": arm, "rand_rep": -1,
                                       "row_index": X_test.index, "y_true": np.asarray(y_test),
                                       "y_pred": pred, "y_proba": proba}))
    for ai, arm in enumerate(["strict", "approx"]):
        k = len(arms[arm])
        subsets = random_feature_subsets(features, k, N_RANDOM, (CV_SEED, split_id, ai))
        for r, cols in enumerate(subsets):
            pred, proba = fit_predict_tree(X_train, y_train, X_test, cols)
            rand_records.append({"split": split_id, "control_for": arm, "rand_rep": r, "n_features": len(cols),
                                 "features": "|".join(cols), **compute_metrics(y_test, pred, proba)})
            if keep_predictions:
                preds.append(pd.DataFrame({"split": split_id, "model": f"random_{arm}", "rand_rep": r,
                                           "row_index": X_test.index, "y_true": np.asarray(y_test),
                                           "y_pred": pred, "y_proba": proba}))
    return {"selection": sel, "arms": arms, "records": records, "random_records": rand_records,
            "predictions": preds}


def paired_test(a, b) -> dict:
    """Paired comparison a - b over the same splits (Wilcoxon signed-rank)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    nz = int((d != 0).sum())
    if nz == 0:
        stat, p = 0.0, 1.0
    else:
        res = wilcoxon(a, b)
        stat, p = float(res.statistic), float(res.pvalue)
    return {"n_pairs": len(d), "n_nonzero_diff": nz, "mean_a": a.mean(), "mean_b": b.mean(),
            "mean_diff": d.mean(), "median_diff": float(np.median(d)),
            "wilcoxon_statistic": stat, "p_value": p}


def feature_frequency(selected_lists: list[list[str]], features: list[str]) -> pd.DataFrame:
    n = len(selected_lists)
    cnt = Counter(f for s in selected_lists for f in s)
    return pd.DataFrame({"feature": features, "count": [cnt[f] for f in features],
                         "n_splits": n, "percent": [100 * cnt[f] / n for f in features]})
