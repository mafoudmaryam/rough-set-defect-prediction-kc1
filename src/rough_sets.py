"""
Classical decision-table Rough Set primitives.

Terminology used throughout this module:

- Universe (U): the set of all objects (rows) in the decision table.
- Condition attributes (C): the features describing each object
  (here, the discretized software metrics).
- Decision attribute (D): the target column (here, `defects`).
- Equivalence / indiscernibility class: for a subset of condition
  attributes P, two objects belong to the same class if and only if
  they have identical values for every attribute in P.
- Positive region POS_P(D): the union of all equivalence classes
  (induced by P) in which every object shares the same decision value.
  An equivalence class contributes to the positive region only when it
  is decision-consistent; if a class contains objects with different
  decision values, none of its objects belong to the positive region.
- Dependency degree gamma_P(D) = |POS_P(D)| / |U|, a number in [0, 1]
  describing how well P alone determines the decision.

Everything here is implemented explicitly (no external Rough Set
library) so the logic stays inspectable and teachable.
"""

from dataclasses import dataclass, field

import pandas as pd

TIE_TOLERANCE = 1e-12


@dataclass
class DependencyResult:
    """Result of computing gamma_P(D) for one attribute subset P."""

    gamma: float
    positive_region_count: int
    universe_size: int
    num_equivalence_classes: int
    num_inconsistent_classes: int


@dataclass
class QuickReductIteration:
    """One greedy-selection step, recorded for transparency/reporting."""

    iteration: int
    chosen_feature: str
    dependency_before: float
    dependency_after: float
    significance_gain: float
    num_selected: int


@dataclass
class QuickReductResult:
    full_dependency: float
    selected_features: list[str]
    iterations: list[QuickReductIteration] = field(default_factory=list)


def dependency_degree(
    df: pd.DataFrame, columns: list[str], target_column: str
) -> DependencyResult:
    """
    Compute gamma_P(D) for condition-attribute subset P = `columns`.

    Objects are grouped by their values in `columns` to form equivalence
    classes. A class belongs to the positive region only if every object
    inside it has the same decision value (`target_column`).

    The empty subset (columns == []) is a special case: with no condition
    attributes, all objects are mutually indiscernible, so there is exactly
    one equivalence class containing the whole universe. That class is
    decision-consistent only if the whole table has a single decision value.
    """
    universe_size = len(df)

    if universe_size == 0:
        return DependencyResult(0.0, 0, 0, 0, 0)

    if len(columns) == 0:
        num_labels = df[target_column].nunique(dropna=False)
        is_consistent = num_labels <= 1
        positive_region_count = universe_size if is_consistent else 0
        return DependencyResult(
            gamma=positive_region_count / universe_size,
            positive_region_count=positive_region_count,
            universe_size=universe_size,
            num_equivalence_classes=1,
            num_inconsistent_classes=0 if is_consistent else 1,
        )

    grouped = df.groupby(list(columns), dropna=False, sort=False)[target_column]
    class_sizes = grouped.size()
    class_label_counts = grouped.nunique(dropna=False)

    consistent_mask = class_label_counts <= 1
    positive_region_count = int(class_sizes[consistent_mask].sum())
    num_equivalence_classes = len(class_sizes)
    num_inconsistent_classes = int((~consistent_mask).sum())

    return DependencyResult(
        gamma=positive_region_count / universe_size,
        positive_region_count=positive_region_count,
        universe_size=universe_size,
        num_equivalence_classes=num_equivalence_classes,
        num_inconsistent_classes=num_inconsistent_classes,
    )


def significance(
    df: pd.DataFrame,
    selected: list[str],
    candidate: str,
    target_column: str,
) -> float:
    """
    Attribute significance of `candidate` with respect to the current
    selected subset P = `selected`:

        significance(a | P) = gamma_(P union {a})(D) - gamma_P(D)

    A larger value means adding `candidate` to P resolves more decision
    inconsistencies (grows the positive region more).
    """
    gamma_before = dependency_degree(df, selected, target_column).gamma
    gamma_after = dependency_degree(df, selected + [candidate], target_column).gamma
    return gamma_after - gamma_before


def quick_reduct(
    df: pd.DataFrame,
    condition_attributes: list[str],
    target_column: str,
    tolerance: float = TIE_TOLERANCE,
) -> QuickReductResult:
    """
    Dependency-based QuickReduct greedy algorithm.

    Starting from an empty selected set, repeatedly add the unselected
    attribute that most increases the dependency degree, until the
    selected set's dependency reaches the full-attribute-set dependency
    (within `tolerance`).

    Deterministic tie-breaking: when multiple candidate attributes produce
    the same (or effectively the same, within `tolerance`) dependency gain,
    the one appearing first in `condition_attributes` (original column
    order) is chosen. This guarantees repeated runs give the same reduct
    candidate.
    """
    full_dependency = dependency_degree(df, condition_attributes, target_column).gamma

    selected: list[str] = []
    remaining = list(condition_attributes)
    iterations: list[QuickReductIteration] = []

    current_gamma = dependency_degree(df, selected, target_column).gamma
    iteration_number = 0

    while remaining and (full_dependency - current_gamma) > tolerance:
        best_feature = None
        best_gamma = -1.0

        for feature in remaining:
            candidate_gamma = dependency_degree(
                df, selected + [feature], target_column
            ).gamma
            if candidate_gamma > best_gamma + tolerance:
                best_gamma = candidate_gamma
                best_feature = feature
            # else: within tolerance of the current best (or worse) -> keep
            # the earlier-seen feature, preserving original column order.

        iteration_number += 1
        gain = best_gamma - current_gamma
        selected.append(best_feature)
        remaining.remove(best_feature)

        iterations.append(
            QuickReductIteration(
                iteration=iteration_number,
                chosen_feature=best_feature,
                dependency_before=current_gamma,
                dependency_after=best_gamma,
                significance_gain=gain,
                num_selected=len(selected),
            )
        )
        current_gamma = best_gamma

    return QuickReductResult(
        full_dependency=full_dependency,
        selected_features=selected,
        iterations=iterations,
    )


def backward_prune(
    df: pd.DataFrame,
    selected: list[str],
    target_column: str,
    target_dependency: float,
    tolerance: float = TIE_TOLERANCE,
) -> list[str]:
    """
    Backward minimality check ("pruning") on a greedy QuickReduct result.

    QuickReduct is a greedy heuristic: it is not guaranteed to return an
    inclusion-minimal subset. This pass tries removing each selected
    attribute (in original order) and keeps it removed if the remaining
    attributes still achieve `target_dependency` (within `tolerance`). It
    repeats until no further attribute can be dropped.

    This produces an inclusion-minimal reduct candidate with respect to the
    attributes QuickReduct happened to select. It does NOT search other
    attribute combinations, so it makes no claim about global minimality or
    uniqueness among all possible reducts.
    """
    pruned = list(selected)
    changed = True

    while changed:
        changed = False
        for feature in list(pruned):
            trial = [f for f in pruned if f != feature]
            trial_gamma = dependency_degree(df, trial, target_column).gamma
            if trial_gamma >= target_dependency - tolerance:
                pruned = trial
                changed = True
                break  # restart scan in original order after a removal

    return pruned


# ---------------------------------------------------------------------------
# Fast integer-coded decision table (used by the unified pipeline).
#
# Same definitions as dependency_degree() above, but on numpy arrays so that
# the 2^21 exhaustive search and the 50-split CV stay fast. The idea of a
# partition-refinement DecisionTable and of the exhaustive reduct search comes
# from the teammate's implementation; it was re-validated against the pandas
# implementation and brute force (see tests/test_unified_pipeline.py) and
# fixed so that mixed-radix keys cannot overflow int64.
# ---------------------------------------------------------------------------

import numpy as np


def _refine(ids: np.ndarray, column: np.ndarray, radix: int) -> np.ndarray:
    """Refine a partition (class ids) by one more attribute column."""
    return np.unique(ids * radix + column, return_inverse=True)[1].ravel()


class DecisionTable:
    """Integer-coded decision table with binary decision y in {0, 1}."""

    def __init__(self, X_disc: pd.DataFrame, y):
        self.attrs = list(X_disc.columns)
        self.codes = X_disc.to_numpy().astype(np.int64)
        self.y = np.asarray(y).astype(np.int64)
        self.n = len(self.y)
        self.radix = self.codes.max(axis=0) + 1 if self.n else np.ones(len(self.attrs), dtype=np.int64)
        self._col = {a: j for j, a in enumerate(self.attrs)}

    def granules(self, subset) -> np.ndarray:
        """Equivalence-class id of every object under IND(subset)."""
        ids = np.zeros(self.n, dtype=np.int64)
        for a in subset:
            j = self._col[a]
            ids = _refine(ids, self.codes[:, j], int(self.radix[j]))
        return ids

    def _pos_from_ids(self, ids: np.ndarray) -> tuple[int, int, int]:
        size = np.bincount(ids)
        ones = np.bincount(ids, weights=self.y)
        pure = (ones == 0) | (ones == size)
        return int(size[pure].sum()), len(size), int((~pure).sum())

    def dependency(self, subset) -> DependencyResult:
        pos, n_cls, n_inc = self._pos_from_ids(self.granules(subset))
        return DependencyResult(pos / self.n, pos, self.n, n_cls, n_inc)

    def gamma(self, subset) -> float:
        return self.dependency(subset).gamma

    def approximations(self, subset, target_value: int = 1) -> dict:
        """Lower/upper approximation and boundary of the set {y == target_value}."""
        ids = self.granules(subset)
        size = np.bincount(ids)
        hit = np.bincount(ids, weights=(self.y == target_value))
        lower = (hit == size)[ids]
        upper = (hit > 0)[ids]
        return {
            "target_size": int((self.y == target_value).sum()),
            "lower": int(lower.sum()),
            "upper": int(upper.sum()),
            "boundary": int((upper & ~lower).sum()),
        }


def table_quick_reduct(dt: DecisionTable, alpha: float = 1.0, tolerance: float = TIE_TOLERANCE):
    """
    QuickReduct (forward greedy on dependency) followed by backward pruning,
    on a DecisionTable. Target: gamma_R(D) >= alpha * gamma_C(D); alpha=1
    gives the classical (strict) reduct, alpha<1 an APPROXIMATE reduct.
    Tie-breaking: first attribute in original order. Returns
    (selected_in_original_order, info).
    """
    gamma_full = dt.gamma(dt.attrs)
    target = alpha * gamma_full
    selected: list[str] = []
    steps = []
    current = dt.gamma(selected)
    while current < target - tolerance and len(selected) < len(dt.attrs):
        best, best_gamma = None, -1.0
        for a in dt.attrs:
            if a in selected:
                continue
            g = dt.gamma(selected + [a])
            if g > best_gamma + tolerance:
                best, best_gamma = a, g
        steps.append({"iteration": len(steps) + 1, "chosen_feature": best,
                      "dependency_before": current, "dependency_after": best_gamma,
                      "significance_gain": best_gamma - current})
        selected.append(best)
        current = best_gamma
    forward = list(selected)
    removed = []
    changed = True
    while changed:
        changed = False
        for a in list(selected):
            trial = [b for b in selected if b != a]
            if dt.gamma(trial) >= target - tolerance:
                selected = trial
                removed.append(a)
                changed = True
                break
    selected = [a for a in dt.attrs if a in selected]
    return selected, {"alpha": alpha, "gamma_full": gamma_full, "target_gamma": target,
                      "gamma_selected": dt.gamma(selected), "forward_selected": forward,
                      "forward_steps": steps, "backward_removed": removed}


def exhaustive_reducts(dt: DecisionTable, tolerance: float = TIE_TOLERANCE) -> list[list[str]]:
    """
    Exhaustive strict-reduct enumeration over all 2^m attribute subsets.
    ok[S] = (gamma_S(D) == gamma_C(D)). A strict reduct is an ok subset such
    that removing any single attribute makes it not ok (gamma is monotone, so
    this is inclusion-minimality). Depth-first partition refinement keeps each
    subset's cost to one refinement step.
    """
    m = len(dt.attrs)
    target = dt.gamma(dt.attrs)
    ok = np.zeros(1 << m, dtype=bool)
    ok[0] = dt.gamma([]) >= target - tolerance

    def visit(start: int, ids: np.ndarray, mask: int) -> None:
        for j in range(start, m):
            new_ids = _refine(ids, dt.codes[:, j], int(dt.radix[j]))
            new_mask = mask | (1 << j)
            ok[new_mask] = dt._pos_from_ids(new_ids)[0] / dt.n >= target - tolerance
            visit(j + 1, new_ids, new_mask)

    visit(0, np.zeros(dt.n, dtype=np.int64), 0)

    masks = np.flatnonzero(ok)
    minimal = np.ones(len(masks), dtype=bool)
    for j in range(m):
        has_bit = (masks >> j) & 1 == 1
        minimal &= ~(has_bit & ok[masks & ~(1 << j)])
    return [[dt.attrs[j] for j in range(m) if mask >> j & 1] for mask in masks[minimal]]


def formal_core(reducts: list[list[str]], attrs: list[str]) -> list[str]:
    """Intersection of the given reducts, in original attribute order."""
    common = set(reducts[0]).intersection(*map(set, reducts[1:])) if reducts else set()
    return [a for a in attrs if a in common]
