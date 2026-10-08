"""
5-fold cross-validated comparison: full-feature Decision Tree baseline vs.
a Rough-Set-reduced Decision Tree, on the PROMISE KC1 dataset.

LEAKAGE-SAFETY DESIGN (read this before changing anything):

For each fold:
  1. Split rows into train/test with StratifiedKFold (labels preserved only
     to define the split; never used inside discretization or selection).
  2. Fit 8-bin quantile discretization edges on the TRAINING fold's raw
     features only (`discretization.fit_quantile_bin_edges`), then apply
     those fixed edges to the training fold (`apply_quantile_bin_edges`).
     The test fold's feature values are never used to decide bin edges.
  3. Run Rough Set QuickReduct + backward pruning on the discretized
     TRAINING fold plus the TRAINING labels only. The test fold's features
     AND labels are not touched by this step.
  4. Train two Decision Trees on the TRAINING fold's *original numeric*
     values (not the discretized codes): one on all 21 features (baseline),
     one on only the Rough-Set-selected features (reduced).
  5. Evaluate both trees on the TEST fold's original numeric values, which
     up to this point have not influenced discretization, selection, or
     training in any way.

Rough Set reduction therefore happens independently inside every fold, so
the reported reduced-model performance is an honest estimate of how the
method would behave on genuinely unseen data.

This script does not perform any resampling (no SMOTE/class weighting) and
does not tune the Decision Tree; both are explicit, deliberate choices for
this phase of the project (see task instructions).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from discretization import fit_apply_train_test
from rough_sets import backward_prune, dependency_degree, quick_reduct

DATA_PATH = Path("data/raw/kc1.csv")
TARGET_COLUMN = "defects"
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"

CONDITION_ATTRIBUTES = [
    "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t",
    "lOCode", "lOComment", "lOBlank", "locCodeAndComment",
    "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount",
]

N_SPLITS = 5
DISCRETIZATION_BINS = 8
PRUNE_TOLERANCE = 1e-12
CV_RANDOM_STATE = 42
TREE_RANDOM_STATE = 42

METRIC_NAMES = [
    "accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc",
]

# Categorical palette (validated, CVD-safe order) -- see dataviz skill.
COLOR_BASELINE = "#2a78d6"  # blue: slot 1
COLOR_REDUCED = "#eb6834"   # orange: slot 2
COLOR_SEQUENTIAL = "#2a78d6"


def discretize_fold(
    X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame, condition_attributes: list[str], bins: int
):
    """
    Fit quantile bin edges per feature on the training fold only, apply
    them to both folds. Returns the discretized training DataFrame (used
    for Rough Set selection) and a per-feature actual-bin-count record.
    Discretized test values are computed too (for the leakage audit) but
    are NOT used anywhere downstream -- the Decision Tree always sees raw
    numeric features.
    """
    train_disc = pd.DataFrame(index=X_train_raw.index)
    test_disc = pd.DataFrame(index=X_test_raw.index)
    actual_bins = {}
    clipped_test_value_counts = {}

    for column in condition_attributes:
        train_codes, test_codes, n_bins = fit_apply_train_test(
            X_train_raw[column], X_test_raw[column], bins=bins
        )
        train_disc[column] = train_codes
        test_disc[column] = test_codes
        actual_bins[column] = n_bins

        train_min, train_max = X_train_raw[column].min(), X_train_raw[column].max()
        clipped_test_value_counts[column] = int(
            ((X_test_raw[column] < train_min) | (X_test_raw[column] > train_max)).sum()
        )

    return train_disc, test_disc, actual_bins, clipped_test_value_counts


def select_features_for_fold(train_disc: pd.DataFrame, y_train: pd.Series, condition_attributes: list[str]):
    """
    Run QuickReduct + backward pruning on the discretized TRAINING fold
    only. Returns the greedy sequence, pruned reduct, and dependency
    bookkeeping needed for the per-fold audit trail.
    """
    table = train_disc.copy()
    table[TARGET_COLUMN] = y_train.to_numpy()

    full_dep = dependency_degree(table, condition_attributes, TARGET_COLUMN)
    reduct_result = quick_reduct(table, condition_attributes, TARGET_COLUMN)
    pruned = backward_prune(
        table,
        reduct_result.selected_features,
        TARGET_COLUMN,
        target_dependency=reduct_result.full_dependency,
        tolerance=PRUNE_TOLERANCE,
    )
    reduct_dep = dependency_degree(table, pruned, TARGET_COLUMN)

    minimality_passed = True
    for feature in pruned:
        trial = [f for f in pruned if f != feature]
        trial_gamma = dependency_degree(table, trial, TARGET_COLUMN).gamma
        if trial_gamma >= reduct_result.full_dependency - PRUNE_TOLERANCE:
            minimality_passed = False

    return {
        "full_dependency": full_dep.gamma,
        "full_positive_region_count": full_dep.positive_region_count,
        "greedy_selected_features": reduct_result.selected_features,
        "pruned_reduct": pruned,
        "reduct_dependency": reduct_dep.gamma,
        "minimality_check_passed": minimality_passed,
        "dependency_matches_full": abs(reduct_dep.gamma - full_dep.gamma) <= 1e-9,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Standard binary-classification metrics with `True`/1 (defective) as the
    positive class. zero_division=0 avoids warnings/crashes on folds where
    a model predicts no positives at all.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def positive_class_proba(clf: DecisionTreeClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    Probability of the positive class (1 = defective). Assumes both classes
    are present in the training fold (guaranteed here by StratifiedKFold
    given KC1's ~326 positive examples split five ways).
    """
    class_index = list(clf.classes_).index(1)
    return clf.predict_proba(X)[:, class_index]


def run_cv_experiment(
    df: pd.DataFrame,
    condition_attributes: list[str] = None,
    n_splits: int = N_SPLITS,
    bins: int = DISCRETIZATION_BINS,
    cv_random_state: int = CV_RANDOM_STATE,
    tree_random_state: int = TREE_RANDOM_STATE,
) -> dict:
    """
    Run the full leakage-safe cross-validation experiment and return a
    dictionary of all raw per-fold results plus per-row predictions. This
    is the reusable core, callable with a smaller synthetic DataFrame from
    unit tests as well as with the full KC1 dataset from main().
    """
    if condition_attributes is None:
        condition_attributes = CONDITION_ATTRIBUTES

    X_raw = df[condition_attributes].reset_index(drop=True)
    y = df[TARGET_COLUMN].astype(int).reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_random_state)

    fold_records = []
    prediction_rows = []
    test_index_coverage = np.zeros(len(df), dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_raw, y), start=1):
        test_index_coverage[test_idx] += 1

        X_train_raw = X_raw.iloc[train_idx]
        X_test_raw = X_raw.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        train_disc, _test_disc, actual_bins, clipped_counts = discretize_fold(
            X_train_raw, X_test_raw, condition_attributes, bins=bins
        )
        selection = select_features_for_fold(train_disc, y_train, condition_attributes)
        selected = selection["pruned_reduct"]

        baseline_clf = DecisionTreeClassifier(random_state=tree_random_state)
        baseline_clf.fit(X_train_raw[condition_attributes], y_train)
        baseline_pred = baseline_clf.predict(X_test_raw[condition_attributes])
        baseline_proba = positive_class_proba(baseline_clf, X_test_raw[condition_attributes])

        reduced_clf = DecisionTreeClassifier(random_state=tree_random_state)
        reduced_clf.fit(X_train_raw[selected], y_train)
        reduced_pred = reduced_clf.predict(X_test_raw[selected])
        reduced_proba = positive_class_proba(reduced_clf, X_test_raw[selected])

        baseline_metrics = compute_metrics(y_test.to_numpy(), baseline_pred, baseline_proba)
        reduced_metrics = compute_metrics(y_test.to_numpy(), reduced_pred, reduced_proba)

        fold_records.append({
            "fold": fold_idx,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "baseline_feature_count": len(condition_attributes),
            "reduct_feature_count": len(selected),
            "reduction_percentage": round(
                100.0 * (1 - len(selected) / len(condition_attributes)), 2
            ),
            "actual_bins_per_feature": actual_bins,
            "clipped_test_value_counts": clipped_counts,
            "selection": selection,
            "baseline_metrics": baseline_metrics,
            "reduced_metrics": reduced_metrics,
        })

        original_row_ids = df.index.to_numpy()[test_idx]
        for i, row_id in enumerate(original_row_ids):
            prediction_rows.append({
                "row_id": int(row_id),
                "fold": fold_idx,
                "true_label": int(y_test.iloc[i]),
                "baseline_prediction": int(baseline_pred[i]),
                "baseline_probability": float(baseline_proba[i]),
                "reduced_prediction": int(reduced_pred[i]),
                "reduced_probability": float(reduced_proba[i]),
            })

    predictions_df = pd.DataFrame(prediction_rows).sort_values("row_id").reset_index(drop=True)

    return {
        "fold_records": fold_records,
        "predictions_df": predictions_df,
        "test_index_coverage": test_index_coverage,
        "condition_attributes": condition_attributes,
    }


def aggregate_metrics(fold_records: list[dict], key: str) -> dict:
    """Mean and (population) std of each metric across folds, for `key` in
    {'baseline_metrics', 'reduced_metrics'}."""
    aggregated = {}
    for metric in METRIC_NAMES:
        values = [fr[key][metric] for fr in fold_records]
        aggregated[metric] = {
            "fold_values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return aggregated


def build_confusion_matrix(predictions_df: pd.DataFrame, pred_column: str) -> dict:
    cm = confusion_matrix(
        predictions_df["true_label"], predictions_df[pred_column], labels=[0, 1]
    )
    tn, fp, fn, tp = cm.ravel()
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def feature_selection_frequency(fold_records: list[dict], condition_attributes: list[str]) -> dict:
    frequency = {f: 0 for f in condition_attributes}
    for fr in fold_records:
        for feature in fr["selection"]["pruned_reduct"]:
            frequency[feature] += 1
    return frequency


def make_plots(df, fold_records, predictions_df, baseline_agg, reduced_agg,
               frequency, cm_baseline, cm_reduced):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c9c8c0",
        "axes.labelcolor": "#0b0b0b",
        "text.color": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })

    # 1. Class distribution.
    counts = df[TARGET_COLUMN].value_counts().reindex([False, True])
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["False (non-defective)", "True (defective)"], counts.values,
                   color=[COLOR_BASELINE, COLOR_REDUCED], width=0.55)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f"{count}\n({count / len(df):.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of modules")
    ax.set_title("KC1 class distribution (defects)")
    ax.set_ylim(0, counts.max() * 1.2)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Feature selection frequency.
    freq_sorted = dict(sorted(frequency.items(), key=lambda kv: -kv[1]))
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(freq_sorted))
    ax.barh(y_pos, list(freq_sorted.values()), color=COLOR_SEQUENTIAL)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(freq_sorted.keys()), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Folds selected in (out of 5)")
    ax.set_xlim(0, 5)
    ax.set_title("Rough Set feature selection frequency across CV folds")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_selection_frequency.png", dpi=150)
    plt.close(fig)

    # 3. Metric comparison: baseline vs reduced (mean +/- std).
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(METRIC_NAMES))
    width = 0.35
    baseline_means = [baseline_agg[m]["mean"] for m in METRIC_NAMES]
    baseline_stds = [baseline_agg[m]["std"] for m in METRIC_NAMES]
    reduced_means = [reduced_agg[m]["mean"] for m in METRIC_NAMES]
    reduced_stds = [reduced_agg[m]["std"] for m in METRIC_NAMES]
    ax.bar(x - width / 2, baseline_means, width, yerr=baseline_stds, capsize=3,
           label="Baseline (21 features)", color=COLOR_BASELINE)
    ax.bar(x + width / 2, reduced_means, width, yerr=reduced_stds, capsize=3,
           label="Rough-Set-reduced", color=COLOR_REDUCED)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in METRIC_NAMES])
    ax.set_ylabel("Score")
    ax.set_ylim(min(0, min(baseline_means + reduced_means) - 0.1), 1.05)
    ax.set_title("Baseline vs. Rough-Set-reduced Decision Tree (mean +/- std over 5 folds)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "metric_comparison.png", dpi=150)
    plt.close(fig)

    # 4. Feature count comparison.
    fig, ax = plt.subplots(figsize=(6, 4))
    folds = [fr["fold"] for fr in fold_records]
    baseline_counts = [fr["baseline_feature_count"] for fr in fold_records]
    reduct_counts = [fr["reduct_feature_count"] for fr in fold_records]
    x = np.arange(len(folds))
    width = 0.35
    ax.bar(x - width / 2, baseline_counts, width, label="Baseline (all features)",
           color=COLOR_BASELINE)
    ax.bar(x + width / 2, reduct_counts, width, label="Rough-Set-reduced",
           color=COLOR_REDUCED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylabel("Number of features")
    ax.set_title("Feature count per fold: baseline vs. reduced")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_count_comparison.png", dpi=150)
    plt.close(fig)

    # Optional: aggregated confusion matrices.
    for name, cm in [("baseline", cm_baseline), ("reduced", cm_reduced)]:
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: False", "Pred: True"])
        ax.set_yticklabels(["Actual: False", "Actual: True"])
        for i in range(2):
            for j in range(2):
                value = matrix[i, j]
                color = "white" if value > matrix.max() / 2 else "black"
                ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=14)
        ax.set_title(f"Aggregated confusion matrix -- {name}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"confusion_matrix_{name}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH}: {df.shape[0]} rows, {df.shape[1]} columns")

    experiment = run_cv_experiment(df)
    fold_records = experiment["fold_records"]
    predictions_df = experiment["predictions_df"]
    test_index_coverage = experiment["test_index_coverage"]

    # --- Validation (Section 26) ---
    assert (test_index_coverage == 1).all(), "Every row must appear in exactly one test fold."
    assert len(predictions_df) == len(df)
    for fr in fold_records:
        assert fr["baseline_feature_count"] == len(CONDITION_ATTRIBUTES)
        assert fr["reduct_feature_count"] <= len(CONDITION_ATTRIBUTES)
        for feature in fr["selection"]["pruned_reduct"]:
            assert feature in df.columns
        assert fr["selection"]["dependency_matches_full"]
    assert predictions_df["baseline_probability"].between(0, 1).all()
    assert predictions_df["reduced_probability"].between(0, 1).all()
    for fr in fold_records:
        for metrics in (fr["baseline_metrics"], fr["reduced_metrics"]):
            for name in METRIC_NAMES:
                value = metrics[name]
                if name == "mcc":
                    assert -1.0 - 1e-9 <= value <= 1.0 + 1e-9, f"MCC out of range: {value}"
                else:
                    assert -1e-9 <= value <= 1.0 + 1e-9, f"{name} out of range: {value}"
    print("All validation checks passed.")

    baseline_agg = aggregate_metrics(fold_records, "baseline_metrics")
    reduced_agg = aggregate_metrics(fold_records, "reduced_metrics")

    metric_differences = {}
    for metric in ["f1", "mcc", "roc_auc", "pr_auc"]:
        diffs = [
            fr["reduced_metrics"][metric] - fr["baseline_metrics"][metric]
            for fr in fold_records
        ]
        metric_differences[metric] = {
            "fold_differences": diffs,
            "mean_difference": float(np.mean(diffs)),
        }

    # Optional, exploratory-only paired comparison (scipy ships as a
    # scikit-learn dependency). Not used as the headline result.
    exploratory_ttests = {}
    try:
        from scipy import stats
        for metric in ["f1", "mcc", "roc_auc", "pr_auc"]:
            baseline_vals = [fr["baseline_metrics"][metric] for fr in fold_records]
            reduced_vals = [fr["reduced_metrics"][metric] for fr in fold_records]
            t_stat, p_value = stats.ttest_rel(reduced_vals, baseline_vals)
            exploratory_ttests[metric] = {"t_statistic": float(t_stat), "p_value": float(p_value)}
    except ImportError:
        exploratory_ttests = None

    selected_features_by_fold = {
        f"fold_{fr['fold']}": fr["selection"]["pruned_reduct"] for fr in fold_records
    }
    frequency = feature_selection_frequency(fold_records, CONDITION_ATTRIBUTES)
    stable_all_folds = sorted([f for f, c in frequency.items() if c == N_SPLITS])
    stable_at_least_4 = sorted([f for f, c in frequency.items() if c >= 4])

    reduct_sizes = [fr["reduct_feature_count"] for fr in fold_records]
    reduction_percentages = [fr["reduction_percentage"] for fr in fold_records]

    cm_baseline = build_confusion_matrix(predictions_df, "baseline_prediction")
    cm_reduced = build_confusion_matrix(predictions_df, "reduced_prediction")

    RESULTS_DIR.mkdir(exist_ok=True)

    # --- results/cv_fold_results.csv ---
    fold_rows = []
    for fr in fold_records:
        row = {
            "fold": fr["fold"],
            "train_size": fr["train_size"],
            "test_size": fr["test_size"],
            "baseline_feature_count": fr["baseline_feature_count"],
            "reduct_feature_count": fr["reduct_feature_count"],
            "reduction_percentage": fr["reduction_percentage"],
            "full_training_dependency": fr["selection"]["full_dependency"],
            "reduct_training_dependency": fr["selection"]["reduct_dependency"],
            "minimality_check_passed": fr["selection"]["minimality_check_passed"],
            "dependency_matches_full": fr["selection"]["dependency_matches_full"],
        }
        for metric in METRIC_NAMES:
            row[f"baseline_{metric}"] = fr["baseline_metrics"][metric]
            row[f"reduced_{metric}"] = fr["reduced_metrics"][metric]
        fold_rows.append(row)
    pd.DataFrame(fold_rows).to_csv(RESULTS_DIR / "cv_fold_results.csv", index=False)

    # --- results/feature_selection_frequency.csv ---
    freq_rows = [
        {"feature": f, "times_selected": c, "frequency_fraction": c / N_SPLITS}
        for f, c in sorted(frequency.items(), key=lambda kv: -kv[1])
    ]
    pd.DataFrame(freq_rows).to_csv(RESULTS_DIR / "feature_selection_frequency.csv", index=False)

    # --- results/cv_predictions.csv ---
    predictions_df.to_csv(RESULTS_DIR / "cv_predictions.csv", index=False)

    # --- results/cv_experiment_summary.json ---
    summary = {
        "dataset": "PROMISE KC1",
        "rows": int(df.shape[0]),
        "original_feature_count": len(CONDITION_ATTRIBUTES),
        "cv": {
            "type": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": CV_RANDOM_STATE,
        },
        "discretization": {
            "method": "training-fold quantile (fit on train, applied to train/test via fixed edges)",
            "requested_bins": DISCRETIZATION_BINS,
        },
        "feature_selection": {
            "method": "Rough Set dependency-based QuickReduct + backward pruning, run inside training fold only",
        },
        "classifier": {
            "type": "DecisionTreeClassifier",
            "random_state": TREE_RANDOM_STATE,
        },
        "resampling": "none (natural class distribution preserved)",
        "baseline_metrics": baseline_agg,
        "reduced_metrics": reduced_agg,
        "metric_differences_reduced_minus_baseline": metric_differences,
        "exploratory_paired_ttest_reduced_vs_baseline": exploratory_ttests,
        "selected_features_by_fold": selected_features_by_fold,
        "feature_selection_frequency": frequency,
        "cross_fold_stable_selected_features": stable_all_folds,
        "features_selected_in_at_least_4_of_5_folds": stable_at_least_4,
        "mean_selected_feature_count": float(np.mean(reduct_sizes)),
        "min_selected_feature_count": int(np.min(reduct_sizes)),
        "max_selected_feature_count": int(np.max(reduct_sizes)),
        "mean_reduction_percentage": float(np.mean(reduction_percentages)),
        "aggregated_confusion_matrix_baseline": cm_baseline,
        "aggregated_confusion_matrix_reduced": cm_reduced,
        "per_fold_discretization_audit": [
            {
                "fold": fr["fold"],
                "actual_bins_per_feature": fr["actual_bins_per_feature"],
                "features_collapsed_to_1_bin": [
                    f for f, b in fr["actual_bins_per_feature"].items() if b == 1
                ],
                "clipped_test_value_counts": fr["clipped_test_value_counts"],
                "full_training_dependency": fr["selection"]["full_dependency"],
                "full_training_positive_region_count": fr["selection"]["full_positive_region_count"],
                "reduct_training_dependency": fr["selection"]["reduct_dependency"],
                "greedy_selected_features": fr["selection"]["greedy_selected_features"],
                "pruned_reduct": fr["selection"]["pruned_reduct"],
                "minimality_check_passed": fr["selection"]["minimality_check_passed"],
            }
            for fr in fold_records
        ],
        "leakage_safety_statement": (
            "In every fold: (1) quantile bin edges were fit using only the "
            "training fold's feature values and then applied unchanged to "
            "the test fold; (2) Rough Set dependency, QuickReduct, and "
            "backward pruning ran only on the discretized training fold and "
            "training labels; (3) both Decision Trees were fit only on "
            "training-fold rows; the test fold contributed nothing until "
            "the final .predict()/.predict_proba() call."
        ),
        "disclaimers": [
            "cross_fold_stable_selected_features lists attributes selected "
            "in every one of the 5 CV folds; this is an empirical "
            "cross-validation observation, not the formal Rough Set core.",
            "The optional paired t-test is exploratory only (n=5 folds) and "
            "is not the basis for any claim of statistical significance.",
        ],
    }
    with open(RESULTS_DIR / "cv_experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- results/cv_experiment_summary.txt ---
    write_text_summary(summary, fold_records)

    # --- plots ---
    make_plots(df, fold_records, predictions_df, baseline_agg, reduced_agg,
               frequency, cm_baseline, cm_reduced)

    print(f"\nSaved: {RESULTS_DIR / 'cv_experiment_summary.json'}")
    print(f"Saved: {RESULTS_DIR / 'cv_experiment_summary.txt'}")
    print(f"Saved: {RESULTS_DIR / 'cv_fold_results.csv'}")
    print(f"Saved: {RESULTS_DIR / 'feature_selection_frequency.csv'}")
    print(f"Saved: {RESULTS_DIR / 'cv_predictions.csv'}")
    print(f"Saved plots to: {PLOTS_DIR}/")


def write_text_summary(summary: dict, fold_records: list[dict]) -> None:
    path = RESULTS_DIR / "cv_experiment_summary.txt"
    with open(path, "w") as f:
        f.write("=" * 74 + "\n")
        f.write("5-FOLD CV: BASELINE (21 FEATURES) vs ROUGH-SET-REDUCED DECISION TREE\n")
        f.write("KC1 -- PROMISE Dataset\n")
        f.write("=" * 74 + "\n\n")

        f.write("EXPERIMENT SETUP\n")
        f.write(f"  Rows: {summary['rows']}\n")
        f.write(f"  Original features: {summary['original_feature_count']}\n")
        f.write(f"  CV: StratifiedKFold(n_splits={summary['cv']['n_splits']}, "
                f"shuffle=True, random_state={summary['cv']['random_state']})\n")
        f.write(f"  Discretization: {summary['discretization']['method']}, "
                f"requested_bins={summary['discretization']['requested_bins']}\n")
        f.write(f"  Feature selection: {summary['feature_selection']['method']}\n")
        f.write(f"  Classifier: {summary['classifier']['type']}"
                f"(random_state={summary['classifier']['random_state']})\n")
        f.write(f"  Resampling: {summary['resampling']}\n\n")

        f.write("LEAKAGE-SAFETY STATEMENT\n")
        f.write(f"  {summary['leakage_safety_statement']}\n\n")

        f.write("FOLD-BY-FOLD ROUGH SET AUDIT\n")
        for fr in fold_records:
            sel = fr["selection"]
            f.write(f"  Fold {fr['fold']}:\n")
            f.write(f"    Full training dependency: {sel['full_dependency']:.6f} "
                    f"(POS region {sel['full_positive_region_count']}/{fr['train_size']})\n")
            f.write(f"    Reduct training dependency: {sel['reduct_dependency']:.6f}\n")
            f.write(f"    Dependency matches full (within tolerance): "
                    f"{sel['dependency_matches_full']}\n")
            f.write(f"    Minimality check passed: {sel['minimality_check_passed']}\n")
            f.write(f"    Greedy selected: {sel['greedy_selected_features']}\n")
            f.write(f"    Pruned reduct ({fr['reduct_feature_count']} features, "
                    f"{fr['reduction_percentage']}% reduction): {sel['pruned_reduct']}\n")
            collapsed = [f2 for f2, b in fr["actual_bins_per_feature"].items() if b == 1]
            f.write(f"    Features collapsed to 1 bin in this fold's training data: {collapsed}\n\n")

        f.write("BASELINE METRICS BY FOLD (all 21 features)\n")
        for i, fr in enumerate(fold_records):
            f.write(f"  Fold {fr['fold']}: " +
                    ", ".join(f"{m}={fr['baseline_metrics'][m]:.4f}" for m in METRIC_NAMES) + "\n")
        f.write("\nREDUCED METRICS BY FOLD (Rough-Set-selected features)\n")
        for fr in fold_records:
            f.write(f"  Fold {fr['fold']}: " +
                    ", ".join(f"{m}={fr['reduced_metrics'][m]:.4f}" for m in METRIC_NAMES) + "\n")

        f.write("\nMEAN +/- STD ACROSS 5 FOLDS\n")
        f.write(f"  {'metric':10s} {'baseline':>18s} {'reduced':>18s}\n")
        for m in METRIC_NAMES:
            b = summary["baseline_metrics"][m]
            r = summary["reduced_metrics"][m]
            f.write(f"  {m:10s} {b['mean']:.4f} +/- {b['std']:.4f}    "
                    f"{r['mean']:.4f} +/- {r['std']:.4f}\n")

        f.write("\nMETRIC DIFFERENCES (reduced - baseline), mean over folds\n")
        for m, d in summary["metric_differences_reduced_minus_baseline"].items():
            f.write(f"  {m:10s} mean diff = {d['mean_difference']:+.4f} "
                    f"(fold diffs: {[round(x, 4) for x in d['fold_differences']]})\n")

        if summary["exploratory_paired_ttest_reduced_vs_baseline"]:
            f.write("\nEXPLORATORY paired t-test (reduced vs baseline, n=5 folds -- "
                    "NOT a basis for significance claims)\n")
            for m, t in summary["exploratory_paired_ttest_reduced_vs_baseline"].items():
                f.write(f"  {m:10s} t={t['t_statistic']:+.4f}, p={t['p_value']:.4f}\n")

        f.write(f"\nFEATURE COUNT SUMMARY\n")
        f.write(f"  Mean selected feature count: {summary['mean_selected_feature_count']:.2f}\n")
        f.write(f"  Min: {summary['min_selected_feature_count']}, "
                f"Max: {summary['max_selected_feature_count']}\n")
        f.write(f"  Mean reduction percentage: {summary['mean_reduction_percentage']:.2f}%\n\n")

        f.write("FEATURE SELECTION FREQUENCY (out of 5 folds)\n")
        for feature, count in sorted(summary["feature_selection_frequency"].items(),
                                      key=lambda kv: -kv[1]):
            f.write(f"  {feature:20s}: {count}/5\n")
        f.write(f"\nFeatures selected in ALL 5 CV folds (cross-fold stable, NOT a formal core):\n")
        f.write(f"  {summary['cross_fold_stable_selected_features']}\n")
        f.write(f"\nFeatures selected in at least 4/5 folds:\n")
        f.write(f"  {summary['features_selected_in_at_least_4_of_5_folds']}\n\n")

        f.write("AGGREGATED CONFUSION MATRICES (out-of-fold, defective=True positive)\n")
        for name, cm in [("Baseline", summary["aggregated_confusion_matrix_baseline"]),
                          ("Reduced", summary["aggregated_confusion_matrix_reduced"])]:
            f.write(f"  {name}: TP={cm['tp']}, FP={cm['fp']}, TN={cm['tn']}, FN={cm['fn']}\n")

        f.write("\nDISCLAIMERS\n")
        for d in summary["disclaimers"]:
            f.write(f"  - {d}\n")


if __name__ == "__main__":
    main()
