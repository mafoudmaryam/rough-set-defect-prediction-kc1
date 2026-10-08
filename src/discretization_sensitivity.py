"""
Discretization sensitivity analysis for the KC1 Rough Set engine.

This script does NOT change the Rough Set methodology. It re-runs the
existing quantile discretization + QuickReduct + backward-pruning pipeline
(from discretization.py / rough_sets.py) at several requested bin counts,
plus a raw/exact-value diagnostic reference, so the effect of the
discretization granularity choice can be inspected before it is frozen.

It intentionally does NOT declare a "best" bin count. Higher dependency at
finer granularity is expected (more distinct equivalence classes = fewer
accidental collisions between differently-labeled objects) and is not by
itself evidence that a given bin count is the right methodological choice.

Usage (from project root):
    source .venv/bin/activate
    python src/discretization_sensitivity.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from discretization import discretize_dataframe
from rough_sets import backward_prune, dependency_degree, quick_reduct

DATA_PATH = Path("data/raw/kc1.csv")
TARGET_COLUMN = "defects"
RESULTS_DIR = Path("results")
TOLERANCE = 1e-12

CONDITION_ATTRIBUTES = [
    "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t",
    "lOCode", "lOComment", "lOBlank", "locCodeAndComment",
    "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount",
]

BIN_COUNTS_TO_TEST = [3, 4, 5, 6, 8, 10]


def raw_exact_reference(df: pd.DataFrame) -> dict:
    """
    Diagnostic-only: Rough Set dependency computed directly on the raw
    continuous feature values (exact-value equality, no discretization).
    This is NOT a candidate methodology -- continuous exact-match
    equivalence classes are almost always singletons except where rows are
    literal duplicates. It exists purely to quantify the theoretical upper
    bound on dependency given KC1's existing duplicate/conflicting rows.
    """
    result = dependency_degree(df, CONDITION_ATTRIBUTES, TARGET_COLUMN)
    return {
        "full_dependency": result.gamma,
        "positive_region_count": result.positive_region_count,
        "universe_size": result.universe_size,
        "num_equivalence_classes": result.num_equivalence_classes,
        "num_inconsistent_classes": result.num_inconsistent_classes,
    }


def run_for_bin_count(df: pd.DataFrame, bins: int) -> dict:
    disc_result = discretize_dataframe(df, CONDITION_ATTRIBUTES, TARGET_COLUMN, bins=bins)
    discretized_df = disc_result.discretized

    full_dep = dependency_degree(discretized_df, CONDITION_ATTRIBUTES, TARGET_COLUMN)

    reduct_result = quick_reduct(discretized_df, CONDITION_ATTRIBUTES, TARGET_COLUMN)
    pruned = backward_prune(
        discretized_df,
        reduct_result.selected_features,
        TARGET_COLUMN,
        target_dependency=reduct_result.full_dependency,
        tolerance=TOLERANCE,
    )
    reduct_dep = dependency_degree(discretized_df, pruned, TARGET_COLUMN)

    # Sanity checks (Section D)
    assert 0.0 <= full_dep.gamma <= 1.0
    assert 0.0 <= reduct_dep.gamma <= 1.0
    assert abs(reduct_dep.gamma - full_dep.gamma) <= 1e-9

    minimality_passed = True
    for feature in pruned:
        trial = [f for f in pruned if f != feature]
        trial_gamma = dependency_degree(discretized_df, trial, TARGET_COLUMN).gamma
        if trial_gamma >= reduct_result.full_dependency - TOLERANCE:
            minimality_passed = False

    actual_bins = disc_result.actual_bins_per_feature
    num_collapsed_to_1 = sum(1 for b in actual_bins.values() if b == 1)
    num_2plus = sum(1 for b in actual_bins.values() if b >= 2)

    reduction_percentage = round(
        100.0 * (1 - len(pruned) / len(CONDITION_ATTRIBUTES)), 2
    )

    return {
        "requested_bins": bins,
        "actual_bins_per_feature": actual_bins,
        "num_features_collapsed_to_1_bin": num_collapsed_to_1,
        "num_features_with_2plus_bins": num_2plus,
        "full_dependency": full_dep.gamma,
        "positive_region_count": full_dep.positive_region_count,
        "num_equivalence_classes": full_dep.num_equivalence_classes,
        "num_inconsistent_classes": full_dep.num_inconsistent_classes,
        "greedy_selected_features": reduct_result.selected_features,
        "pruned_reduct_candidate": pruned,
        "reduct_size": len(pruned),
        "reduction_percentage": reduction_percentage,
        "reduct_dependency": reduct_dep.gamma,
        "minimality_check_passed": minimality_passed,
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH}: {df.shape[0]} rows, {df.shape[1]} columns")

    raw_ref = raw_exact_reference(df)
    print(f"Raw exact-value reference: gamma = {raw_ref['full_dependency']:.6f} "
          f"(POS = {raw_ref['positive_region_count']}/{raw_ref['universe_size']})")

    runs = []
    for bins in BIN_COUNTS_TO_TEST:
        print(f"Running quantile discretization with requested bins = {bins} ...")
        run = run_for_bin_count(df, bins)
        runs.append(run)
        print(f"  -> full gamma = {run['full_dependency']:.6f}, "
              f"reduct size = {run['reduct_size']}, "
              f"1-bin features = {run['num_features_collapsed_to_1_bin']}, "
              f"minimality_passed = {run['minimality_check_passed']}")

    summary_table = [
        {
            "bins": r["requested_bins"],
            "dependency": round(r["full_dependency"], 6),
            "positive_region": r["positive_region_count"],
            "one_bin_features": r["num_features_collapsed_to_1_bin"],
            "reduct_size": r["reduct_size"],
            "reduction_pct": r["reduction_percentage"],
        }
        for r in runs
    ]

    output = {
        "dataset": "PROMISE KC1",
        "rows": int(df.shape[0]),
        "original_feature_count": len(CONDITION_ATTRIBUTES),
        "note": (
            "This is a sensitivity/diagnostic analysis. No bin count is "
            "declared 'best' here -- higher dependency at finer granularity "
            "is expected and does not by itself justify a methodology choice."
        ),
        "raw_exact_reference": raw_ref,
        "runs": runs,
        "summary_table": summary_table,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / "discretization_sensitivity.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    txt_path = RESULTS_DIR / "discretization_sensitivity.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("DISCRETIZATION SENSITIVITY ANALYSIS - KC1\n")
        f.write("=" * 70 + "\n\n")
        f.write("No bin count is declared 'best'. This is diagnostic only.\n\n")

        f.write("RAW EXACT-VALUE REFERENCE (diagnostic only, not a candidate methodology)\n")
        f.write(f"  Full dependency (exact match, all 21 raw features): "
                f"{raw_ref['full_dependency']:.6f}\n")
        f.write(f"  Positive region: {raw_ref['positive_region_count']} / "
                f"{raw_ref['universe_size']}\n")
        f.write(f"  Equivalence classes: {raw_ref['num_equivalence_classes']}\n")
        f.write(f"  Inconsistent classes: {raw_ref['num_inconsistent_classes']}\n\n")

        f.write("SUMMARY TABLE\n")
        header = f"{'bins':>4} | {'dependency':>10} | {'pos region':>10} | {'1-bin feats':>11} | {'reduct size':>11} | {'reduction %':>11}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")
        for row in summary_table:
            f.write(
                f"{row['bins']:>4} | {row['dependency']:>10.6f} | "
                f"{row['positive_region']:>10} | {row['one_bin_features']:>11} | "
                f"{row['reduct_size']:>11} | {row['reduction_pct']:>11}\n"
            )
        f.write("\n")

        for r in runs:
            f.write("-" * 70 + "\n")
            f.write(f"Requested bins = {r['requested_bins']}\n")
            f.write(f"  Actual bins per feature: {r['actual_bins_per_feature']}\n")
            f.write(f"  Features collapsed to 1 bin: {r['num_features_collapsed_to_1_bin']}\n")
            f.write(f"  Features with 2+ bins: {r['num_features_with_2plus_bins']}\n")
            f.write(f"  Full dependency: {r['full_dependency']:.6f}\n")
            f.write(f"  Positive region: {r['positive_region_count']}\n")
            f.write(f"  Equivalence classes: {r['num_equivalence_classes']}\n")
            f.write(f"  Inconsistent classes: {r['num_inconsistent_classes']}\n")
            f.write(f"  Greedy selected: {r['greedy_selected_features']}\n")
            f.write(f"  Pruned reduct: {r['pruned_reduct_candidate']}\n")
            f.write(f"  Reduct size: {r['reduct_size']} "
                    f"({r['reduction_percentage']}% reduction)\n")
            f.write(f"  Reduct dependency: {r['reduct_dependency']:.6f}\n")
            f.write(f"  Minimality check passed: {r['minimality_check_passed']}\n")
        f.write("-" * 70 + "\n")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
