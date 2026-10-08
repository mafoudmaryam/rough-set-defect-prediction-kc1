"""
QuickReduct feature-order robustness analysis for the KC1 Rough Set engine.

QuickReduct's tie-breaking rule falls back to input column order whenever
candidate attributes tie (within tolerance) on dependency gain. At the
4-bin quantile discretization, every individual feature has dependency
0.0, so the very first greedy choice is entirely order-determined. This
script checks how much the resulting reduct candidate actually changes
across different input feature orderings, using the SAME 4-bin discretized
dataset for every run (only the order of `condition_attributes` passed to
quick_reduct changes).

This does NOT change the Rough Set methodology and does NOT claim any
kind of formal core. The intersection of attributes across the sampled
orderings is reported as "attributes common to all sampled greedy reduct
candidates" -- a purely empirical observation from a small sample of
orderings, not a mathematical core (which would require enumerating all
possible orderings/subsets).

Usage (from project root):
    source .venv/bin/activate
    python src/quickreduct_order_sensitivity.py
"""

import json
import random
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
BINS_FOR_THIS_ANALYSIS = 4

CONDITION_ATTRIBUTES = [
    "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t",
    "lOCode", "lOComment", "lOBlank", "locCodeAndComment",
    "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount",
]

SHUFFLE_SEEDS = [0, 1, 2, 3, 4]


def build_orderings() -> list[tuple[str, list[str]]]:
    orderings = [
        ("original", list(CONDITION_ATTRIBUTES)),
        ("reversed", list(reversed(CONDITION_ATTRIBUTES))),
    ]
    for seed in SHUFFLE_SEEDS:
        rng = random.Random(seed)
        shuffled = list(CONDITION_ATTRIBUTES)
        rng.shuffle(shuffled)
        orderings.append((f"shuffled_seed_{seed}", shuffled))
    return orderings


def run_for_ordering(discretized_df: pd.DataFrame, name: str, order: list[str], full_dependency: float) -> dict:
    reduct_result = quick_reduct(discretized_df, order, TARGET_COLUMN)
    pruned = backward_prune(
        discretized_df,
        reduct_result.selected_features,
        TARGET_COLUMN,
        target_dependency=full_dependency,
        tolerance=TOLERANCE,
    )
    reduct_dep = dependency_degree(discretized_df, pruned, TARGET_COLUMN)

    assert abs(reduct_dep.gamma - full_dependency) <= 1e-9

    minimality_passed = True
    for feature in pruned:
        trial = [f for f in pruned if f != feature]
        trial_gamma = dependency_degree(discretized_df, trial, TARGET_COLUMN).gamma
        if trial_gamma >= full_dependency - TOLERANCE:
            minimality_passed = False

    return {
        "name": name,
        "input_order": order,
        "greedy_sequence": reduct_result.selected_features,
        "pruned_reduct": sorted(pruned),
        "reduct_size": len(pruned),
        "dependency": reduct_dep.gamma,
        "minimality_check_passed": minimality_passed,
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    disc_result = discretize_dataframe(
        df, CONDITION_ATTRIBUTES, TARGET_COLUMN, bins=BINS_FOR_THIS_ANALYSIS
    )
    discretized_df = disc_result.discretized

    full_dependency = dependency_degree(
        discretized_df, CONDITION_ATTRIBUTES, TARGET_COLUMN
    ).gamma
    print(f"Using {BINS_FOR_THIS_ANALYSIS}-bin discretized KC1, full dependency = "
          f"{full_dependency:.6f}")

    orderings = build_orderings()
    run_results = []
    for name, order in orderings:
        print(f"Running QuickReduct for ordering: {name} ...")
        result = run_for_ordering(discretized_df, name, order, full_dependency)
        run_results.append(result)
        print(f"  -> reduct size {result['reduct_size']}, "
              f"minimality_passed = {result['minimality_check_passed']}")

    # Empirical aggregation across the sampled orderings.
    reduct_sets = [frozenset(r["pruned_reduct"]) for r in run_results]
    common_to_all = sorted(set.intersection(*[set(s) for s in reduct_sets]))

    frequency: dict[str, int] = {f: 0 for f in CONDITION_ATTRIBUTES}
    for s in reduct_sets:
        for feature in s:
            frequency[feature] += 1
    frequency = dict(sorted(frequency.items(), key=lambda kv: (-kv[1], CONDITION_ATTRIBUTES.index(kv[0]))))

    num_runs = len(run_results)
    most_common_count = max(frequency.values())
    attributes_in_most = [f for f, c in frequency.items() if c == most_common_count]

    reduct_sizes = [r["reduct_size"] for r in run_results]
    distinct_reducts = sorted({s for s in reduct_sets}, key=lambda s: (len(s), sorted(s)))
    distinct_reducts_list = [sorted(s) for s in distinct_reducts]

    output = {
        "dataset": "PROMISE KC1",
        "requested_bins": BINS_FOR_THIS_ANALYSIS,
        "full_dependency": full_dependency,
        "note": (
            "Empirical robustness check across a small sample of feature "
            "orderings, not an exhaustive search. The intersection reported "
            "below is 'attributes common to all sampled greedy reduct "
            "candidates', NOT a formal Rough Set core."
        ),
        "orderings": run_results,
        "attributes_common_to_all_sampled_reducts": common_to_all,
        "attribute_frequency_across_reducts": frequency,
        "attributes_appearing_in_most_reducts": attributes_in_most,
        "most_common_appearance_count": most_common_count,
        "num_orderings_tested": num_runs,
        "min_reduct_size": min(reduct_sizes),
        "max_reduct_size": max(reduct_sizes),
        "num_distinct_reduct_candidates": len(distinct_reducts_list),
        "distinct_reduct_candidates": distinct_reducts_list,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / "quickreduct_order_sensitivity.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    txt_path = RESULTS_DIR / "quickreduct_order_sensitivity.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("QUICKREDUCT FEATURE-ORDER ROBUSTNESS ANALYSIS - KC1 (4-bin)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Full dependency (4-bin, all 21 features): {full_dependency:.6f}\n")
        f.write(f"Orderings tested: {num_runs}\n\n")

        for r in run_results:
            f.write("-" * 70 + "\n")
            f.write(f"Ordering: {r['name']}\n")
            f.write(f"  Greedy sequence: {r['greedy_sequence']}\n")
            f.write(f"  Pruned reduct ({r['reduct_size']} features): {r['pruned_reduct']}\n")
            f.write(f"  Dependency: {r['dependency']:.6f}\n")
            f.write(f"  Minimality check passed: {r['minimality_check_passed']}\n")
        f.write("-" * 70 + "\n\n")

        f.write("Attributes common to ALL sampled greedy reduct candidates "
                "(empirical, NOT a formal core):\n")
        f.write(f"  {common_to_all}\n\n")

        f.write("Attribute frequency across the sampled reduct candidates:\n")
        for feature, count in frequency.items():
            f.write(f"  {feature:20s}: {count}/{num_runs}\n")
        f.write(f"\nAttributes appearing in the most reducts "
                f"({most_common_count}/{num_runs}): {attributes_in_most}\n\n")

        f.write(f"Min reduct size: {min(reduct_sizes)}\n")
        f.write(f"Max reduct size: {max(reduct_sizes)}\n")
        f.write(f"Distinct reduct candidates found: {len(distinct_reducts_list)}\n")
        for i, cand in enumerate(distinct_reducts_list, start=1):
            f.write(f"  Candidate {i} ({len(cand)} features): {cand}\n")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
