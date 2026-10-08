"""
Run the Rough Set feature-reduction methodology on the full KC1 dataset.

This script validates the Rough Set engine (discretization, dependency
degree, QuickReduct, backward pruning) end-to-end on the complete,
unmodified KC1 dataset. It is a methodology-validation run, not the final
predictive evaluation: discretization here uses quantile edges fit on the
whole dataset, which would leak information in a train/test evaluation.
A later phase will refit discretization per cross-validation fold.

Usage (from project root):
    source .venv/bin/activate
    python src/run_rough_reduction.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from discretization import DEFAULT_BINS, discretize_dataframe
from rough_sets import backward_prune, dependency_degree, quick_reduct

DATA_PATH = Path("data/raw/kc1.csv")
TARGET_COLUMN = "defects"
RESULTS_DIR = Path("results")

CONDITION_ATTRIBUTES = [
    "loc",
    "v(g)",
    "ev(g)",
    "iv(g)",
    "n",
    "v",
    "l",
    "d",
    "i",
    "e",
    "b",
    "t",
    "lOCode",
    "lOComment",
    "lOBlank",
    "locCodeAndComment",
    "uniq_Op",
    "uniq_Opnd",
    "total_Op",
    "total_Opnd",
    "branchCount",
]

PRUNE_TOLERANCE = 1e-12


def load_and_validate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing_columns = [c for c in CONDITION_ATTRIBUTES + [TARGET_COLUMN] if c not in df.columns]
    if missing_columns:
        raise ValueError(f"KC1 is missing expected columns: {missing_columns}")

    return df


def main() -> None:
    df = load_and_validate(DATA_PATH)
    print(f"Loaded {DATA_PATH}: {df.shape[0]} rows, {df.shape[1]} columns")

    disc_result = discretize_dataframe(
        df, CONDITION_ATTRIBUTES, TARGET_COLUMN, bins=DEFAULT_BINS
    )
    discretized_df = disc_result.discretized
    print(f"Discretized {len(CONDITION_ATTRIBUTES)} condition attributes "
          f"(requested {DEFAULT_BINS} quantile bins).")

    full_dep = dependency_degree(discretized_df, CONDITION_ATTRIBUTES, TARGET_COLUMN)
    print(f"Full dependency gamma_C(D) = {full_dep.gamma:.6f} "
          f"(POS region = {full_dep.positive_region_count}/{full_dep.universe_size})")

    reduct_result = quick_reduct(
        discretized_df, CONDITION_ATTRIBUTES, TARGET_COLUMN
    )
    print(f"QuickReduct greedy selection: {reduct_result.selected_features}")

    pruned = backward_prune(
        discretized_df,
        reduct_result.selected_features,
        TARGET_COLUMN,
        target_dependency=reduct_result.full_dependency,
        tolerance=PRUNE_TOLERANCE,
    )
    reduct_dep = dependency_degree(discretized_df, pruned, TARGET_COLUMN)
    print(f"Pruned reduct candidate: {pruned}")
    print(f"Reduct dependency gamma_reduct(D) = {reduct_dep.gamma:.6f}")

    # --- Validation / sanity checks (Section 14) ---
    assert 0.0 <= full_dep.gamma <= 1.0
    assert 0.0 <= reduct_dep.gamma <= 1.0
    assert abs(reduct_dep.gamma - full_dep.gamma) <= 1e-9, (
        "Reduct dependency does not match full dependency within tolerance."
    )
    for feature in pruned:
        trial = [f for f in pruned if f != feature]
        trial_gamma = dependency_degree(discretized_df, trial, TARGET_COLUMN).gamma
        assert trial_gamma < reduct_result.full_dependency - PRUNE_TOLERANCE, (
            f"Pruning incomplete: removing '{feature}' did not lower dependency "
            f"below the target ({trial_gamma} vs {reduct_result.full_dependency})."
        )
    assert len(pruned) <= len(CONDITION_ATTRIBUTES)
    assert all(f in df.columns for f in pruned)
    print("All sanity checks passed.")

    # Attributes whose individual removal from the FULL feature set lowers
    # full dependency. These are "indispensable relative to the full
    # attribute set", not a formal CORE(C) (which requires enumerating all
    # reducts, out of scope here).
    indispensable_wrt_full_set = []
    for feature in CONDITION_ATTRIBUTES:
        trial = [f for f in CONDITION_ATTRIBUTES if f != feature]
        trial_gamma = dependency_degree(discretized_df, trial, TARGET_COLUMN).gamma
        if trial_gamma < full_dep.gamma - PRUNE_TOLERANCE:
            indispensable_wrt_full_set.append(feature)

    reduction_percentage = round(
        100.0 * (1 - len(pruned) / len(CONDITION_ATTRIBUTES)), 2
    )

    summary = {
        "dataset": "PROMISE KC1",
        "rows": int(df.shape[0]),
        "original_feature_count": len(CONDITION_ATTRIBUTES),
        "discretization": {
            "method": "quantile",
            "requested_bins": disc_result.requested_bins,
            "actual_bins_per_feature": disc_result.actual_bins_per_feature,
            "note": (
                "Fit on the full dataset for methodology validation only; "
                "NOT the final predictive evaluation protocol (leakage risk "
                "if used for train/test comparisons)."
            ),
        },
        "full_dependency": full_dep.gamma,
        "full_positive_region_count": full_dep.positive_region_count,
        "full_num_equivalence_classes": full_dep.num_equivalence_classes,
        "full_num_inconsistent_classes": full_dep.num_inconsistent_classes,
        "greedy_selected_features": reduct_result.selected_features,
        "pruned_reduct_candidate": pruned,
        "reduct_feature_count": len(pruned),
        "reduction_percentage": reduction_percentage,
        "reduct_dependency": reduct_dep.gamma,
        "iterations": [
            {
                "iteration": it.iteration,
                "chosen_feature": it.chosen_feature,
                "dependency_before": it.dependency_before,
                "dependency_after": it.dependency_after,
                "significance_gain": it.significance_gain,
                "num_selected": it.num_selected,
            }
            for it in reduct_result.iterations
        ],
        "indispensable_attributes_relative_to_full_set": indispensable_wrt_full_set,
        "disclaimers": [
            "pruned_reduct_candidate is an inclusion-minimal reduct CANDIDATE "
            "derived from one greedy QuickReduct run; it is not claimed to be "
            "the unique reduct or the globally optimal reduct.",
            "indispensable_attributes_relative_to_full_set is NOT the formal "
            "Rough Set CORE(C), since CORE(C) requires enumerating or "
            "correctly deriving all reducts, which this task does not do.",
        ],
    }

    RESULTS_DIR.mkdir(exist_ok=True)

    json_path = RESULTS_DIR / "rough_reduction_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    txt_path = RESULTS_DIR / "rough_reduction_summary.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("ROUGH SET FEATURE REDUCTION - KC1 METHODOLOGY VALIDATION RUN\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset: PROMISE KC1 ({df.shape[0]} rows)\n")
        f.write(f"Original condition attributes: {len(CONDITION_ATTRIBUTES)}\n\n")

        f.write("Discretization: quantile-based, requested bins = "
                f"{disc_result.requested_bins}\n")
        f.write("Actual bins per feature:\n")
        for feat, n_bins in disc_result.actual_bins_per_feature.items():
            f.write(f"  {feat:20s}: {n_bins} bins\n")
        f.write("\nNOTE: discretization fit on the full dataset. This is valid\n"
                "for validating the Rough Set engine only, NOT for a leakage-free\n"
                "predictive evaluation.\n\n")

        f.write(f"Full dependency gamma_C(D): {full_dep.gamma:.6f}\n")
        f.write(f"Full positive region count: {full_dep.positive_region_count} / "
                f"{full_dep.universe_size}\n")
        f.write(f"Number of equivalence classes (full attribute set): "
                f"{full_dep.num_equivalence_classes}\n")
        f.write(f"Number of inconsistent equivalence classes: "
                f"{full_dep.num_inconsistent_classes}\n\n")

        f.write("QuickReduct greedy iterations:\n")
        for it in reduct_result.iterations:
            f.write(
                f"  [{it.iteration}] +{it.chosen_feature:20s} "
                f"dep {it.dependency_before:.6f} -> {it.dependency_after:.6f} "
                f"(gain {it.significance_gain:.6f}), selected={it.num_selected}\n"
            )
        f.write(f"\nGreedy selected features ({len(reduct_result.selected_features)}): "
                f"{reduct_result.selected_features}\n")

        f.write(f"\nPost-pruning reduct candidate ({len(pruned)} features): {pruned}\n")
        f.write(f"Reduct dependency: {reduct_dep.gamma:.6f}\n")
        f.write(f"Reduction: {len(CONDITION_ATTRIBUTES)} -> {len(pruned)} features "
                f"({reduction_percentage}% reduction)\n\n")

        f.write("Attributes indispensable relative to the FULL attribute set\n"
                "(removing them individually from all 21 features lowers full\n"
                "dependency). This is NOT the formal Rough Set CORE(C).\n")
        f.write(f"  {indispensable_wrt_full_set}\n\n")

        f.write("Disclaimers:\n")
        for d in summary["disclaimers"]:
            f.write(f"  - {d}\n")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
