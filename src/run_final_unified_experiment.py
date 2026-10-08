"""
Final unified experiment: Rough Set feature reduction for KC1 defect prediction.

    python src/run_final_unified_experiment.py [--skip-exhaustive]

Outputs go to results/final/ (previous results/ files are never touched).
See unified_protocol.py for the leakage-safe per-split protocol.
"""

import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import unified_protocol as up
from discretization import PRIMARY_BINS
from rough_sets import (DecisionTable, exhaustive_reducts, formal_core, table_quick_reduct)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "kc1.csv"
OUT = ROOT / "results" / "final"
PLOTS = OUT / "plots"
SENS_METHODS = ["q3", "q5", "q8", "mdlp"]
COLORS = {"baseline": "#4C566A", "strict": "#1B7F79", "random_strict": "#C0A062",
          "approx": "#3B6FB6", "random_approx": "#C0A062"}


def audit(df, features):
    X = df[features]
    dup = int(df.duplicated().sum())
    g = df.groupby(features)[up.TARGET].nunique()
    conflict_rows = int(df.groupby(features)[up.TARGET].transform("nunique").gt(1).sum())
    return {"rows": len(df), "n_condition_attributes": len(features),
            "class_counts": {str(k): int(v) for k, v in df[up.TARGET].value_counts().items()},
            "missing_values": int(df.isna().sum().sum()), "repeated_rows_beyond_first": dup,
            "conflicting_label_rows": conflict_rows,
            "conflicting_feature_vectors": int((g > 1).sum())}


def full_dataset_analysis(X, y, features, run_exhaustive):
    disc = up.make_discretizer(f"q{PRIMARY_BINS}")
    Xd = disc.fit_transform(X)  # DESCRIPTIVE: whole dataset, never used for test folds
    dt = DecisionTable(Xd, y)
    dep = dt.dependency(features)
    qr, info = table_quick_reduct(dt, 1.0)
    forward = info["forward_selected"]
    res = {"discretization": f"{PRIMARY_BINS}-bin equal-frequency (descriptive, whole dataset)",
           "actual_bins": disc.actual_bins_,
           "full_dependency": dep.gamma, "positive_region_size": dep.positive_region_count,
           "universe_size": dep.universe_size, "equivalence_class_count": dep.num_equivalence_classes,
           "inconsistent_equivalence_class_count": dep.num_inconsistent_classes,
           "defects_true_approximation": dt.approximations(features, 1),
           "quickreduct_forward": forward, "quickreduct_forward_steps": info["forward_steps"],
           "quickreduct_backward_removed": info["backward_removed"],
           "quickreduct_pruned": qr, "quickreduct_pruned_size": len(qr),
           "quickreduct_pruned_gamma": dt.gamma(qr)}
    if run_exhaustive:
        t0 = time.time()
        reducts = exhaustive_reducts(dt)
        secs = time.time() - t0
        sizes = [len(r) for r in reducts]
        mn = min(sizes)
        minimum = [r for r in reducts if len(r) == mn]
        res["exhaustive"] = {
            "subsets_searched": 2 ** len(features), "seconds": round(secs, 1),
            "n_strict_reducts": len(reducts),
            "strict_reduct_size_distribution": {str(k): sizes.count(k) for k in sorted(set(sizes))},
            "min_reduct_size": mn, "n_minimum_reducts": len(minimum), "minimum_reducts": minimum,
            "formal_core_intersection_of_minimum_reducts": formal_core(minimum, features),
            "intersection_of_all_strict_reducts": formal_core(reducts, features),
            "all_strict_reducts": reducts,
        }
        res["exhaustive"]["strict_reducts_verified_preserve_gamma"] = all(
            abs(dt.gamma(r) - dep.gamma) < 1e-12 for r in reducts)
        res["quickreduct_is_strict_reduct"] = sorted(qr) in [sorted(r) for r in reducts]
        res["quickreduct_matches_a_minimum_reduct"] = sorted(qr) in [sorted(r) for r in minimum]
        res["quickreduct_size_minus_minimum_size"] = len(qr) - mn
    return res


def full_dataset_sensitivity(X, y, features):
    rows = []
    for m in SENS_METHODS:
        disc = up.make_discretizer(m)
        Xd = disc.fit_transform(X, y) if m == "mdlp" else disc.fit_transform(X)
        dt = DecisionTable(Xd, y)
        dep = dt.dependency(features)
        qr, _ = table_quick_reduct(dt, 1.0)
        ar, _ = table_quick_reduct(dt, up.ALPHA)
        bins = disc.actual_bins_
        rows.append({"method": m, "full_dependency": dep.gamma, "positive_region": dep.positive_region_count,
                     "equivalence_classes": dep.num_equivalence_classes,
                     "collapsed_1bin_features": sum(1 for v in bins.values() if v == 1),
                     "collapsed_feature_names": "|".join(c for c, v in bins.items() if v == 1),
                     "strict_reduct_size": len(qr), "strict_reduction_pct": 100 * (1 - len(qr) / len(features)),
                     "approx_reduct_size": len(ar), "approx_reduction_pct": 100 * (1 - len(ar) / len(features)),
                     "strict_reduct": "|".join(qr)})
    return pd.DataFrame(rows)


def run_primary_cv(X, y, features, splits):
    cv_rows, rand_rows, preds = [], [], []
    sel_strict, sel_approx, checks = [], [], []
    for sid, (tr, te) in enumerate(splits):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        assert not set(tr) & set(te), "train/test overlap"
        r = up.evaluate_split(sid, Xtr, ytr, Xte, yte, features)
        sel = r["selection"]
        s_feats, s_info = sel[1.0]
        a_feats, a_info = sel[up.ALPHA]
        for fs in (s_feats, a_feats):
            assert fs and all(f in features for f in fs) and len(set(fs)) == len(fs), "invalid selection"
        assert abs(s_info["gamma_selected"] - s_info["gamma_full"]) < 1e-12, "strict reduct lost dependency"
        assert a_info["gamma_selected"] >= up.ALPHA * a_info["gamma_full"] - 1e-12, "approx below alpha"
        assert len(r["arms"]["baseline"]) == len(features)
        for rr in r["random_records"]:
            exp = len(s_feats) if rr["control_for"] == "strict" else len(a_feats)
            assert rr["n_features"] == exp, "random control size mismatch"
        cv_rows += [{**rec, "features": "|".join(r["arms"][rec["arm"]])} for rec in r["records"]]
        rand_rows += r["random_records"]
        preds += r["predictions"]
        sel_strict.append(s_feats)
        sel_approx.append(a_feats)
        checks.append({"split": sid, "n_train": len(tr), "n_test": len(te),
                       "train_pos_rate": float(ytr.mean()), "test_pos_rate": float(yte.mean()),
                       "gamma_full_train": s_info["gamma_full"], "gamma_strict_train": s_info["gamma_selected"],
                       "gamma_approx_train": a_info["gamma_selected"],
                       "approx_ratio": a_info["gamma_selected"] / a_info["gamma_full"] if a_info["gamma_full"] else 1.0,
                       "actual_bins_train": json.dumps(sel["actual_bins"])})
    return (pd.DataFrame(cv_rows), pd.DataFrame(rand_rows), pd.concat(preds, ignore_index=True),
            sel_strict, sel_approx, pd.DataFrame(checks))


def cv_sensitivity(X, y, features, splits):
    """Strict-reduct predictive check for each discretization method (same 50 splits)."""
    rows = []
    for m in SENS_METHODS:
        for sid, (tr, te) in enumerate(splits):
            Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
            sel = up.select_features(Xtr, ytr, m, alphas=(1.0,))
            cols = sel[1.0][0]
            pred, proba = up.fit_predict_tree(Xtr, ytr, Xte, cols)
            rows.append({"method": m, "split": sid, "n_features": len(cols),
                         "n_collapsed_bins": sum(v == 1 for v in sel["actual_bins"].values()),
                         **up.compute_metrics(yte, pred, proba)})
    df = pd.DataFrame(rows)
    return df.groupby("method", sort=False)[["n_features", "n_collapsed_bins"] + up.METRICS].agg(["mean", "std"])


def summarize(frame, group_col, cols):
    return {g: {m: {"mean": float(d[m].mean()), "std": float(d[m].std(ddof=1))} for m in cols}
            for g, d in frame.groupby(group_col, sort=False)}


def per_split_random_mean(rand_df, control_for):
    d = rand_df[rand_df.control_for == control_for]
    return d.groupby("split")[["n_features"] + up.METRICS].mean().sort_index()


def statistical_tests(cv_df, rand_df):
    piv = {a: cv_df[cv_df.arm == a].set_index("split").sort_index() for a in ["baseline", "strict", "approx"]}
    rs, ra = per_split_random_mean(rand_df, "strict"), per_split_random_mean(rand_df, "approx")
    comps = [("strict_vs_baseline", piv["strict"], piv["baseline"]),
             ("strict_vs_random_strict", piv["strict"], rs),
             ("approx_vs_baseline", piv["approx"], piv["baseline"]),
             ("approx_vs_random_approx", piv["approx"], ra)]
    rows = []
    for name, a, b in comps:
        for m in up.TEST_METRICS:
            rows.append({"comparison": name, "metric": m, **up.paired_test(a[m].to_numpy(), b[m].to_numpy())})
    return pd.DataFrame(rows), rs, ra


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def make_plots(df, cv_df, rand_df, rs, ra, freq_s, freq_a):
    PLOTS.mkdir(parents=True, exist_ok=True)
    # class distribution
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    counts = df[up.TARGET].value_counts().reindex([False, True])
    ax.bar(["No defect", "Defect"], counts.values, color=[COLORS["baseline"], COLORS["strict"]])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({100 * v / len(df):.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Modules"); ax.set_title("KC1 class distribution"); ax.set_ylim(0, counts.max() * 1.18); style(ax)
    fig.tight_layout(); fig.savefig(PLOTS / "class_distribution.png", dpi=150); plt.close(fig)
    # frequencies
    for name, fr, col in [("strict", freq_s, COLORS["strict"]), ("approx", freq_a, COLORS["approx"])]:
        f = fr.sort_values("count")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.barh(f.feature, f["count"], color=col)
        ax.set_xlim(0, fr.n_splits.iloc[0]); ax.set_xlabel(f"Splits selected (of {fr.n_splits.iloc[0]})")
        ax.set_title(f"Cross-fold selection frequency: {name} reduct")
        ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
        fig.savefig(PLOTS / f"{name}_feature_frequency.png", dpi=150); plt.close(fig)
    # average feature counts
    labels = ["Baseline", "Strict RS", "Random (strict size)", "Approx RS", "Random (approx size)"]
    vals = [cv_df[cv_df.arm == "baseline"].n_features.mean(), cv_df[cv_df.arm == "strict"].n_features.mean(),
            rs.n_features.mean(), cv_df[cv_df.arm == "approx"].n_features.mean(), ra.n_features.mean()]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(labels, vals, color=[COLORS[k] for k in ["baseline", "strict", "random_strict", "approx", "random_approx"]])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean features used"); ax.set_title("Average feature counts (50 splits)"); style(ax)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right"); fig.tight_layout()
    fig.savefig(PLOTS / "average_feature_counts.png", dpi=150); plt.close(fig)
    # metric comparisons
    for tag, arm, rnd, rkey in [("strict", "strict", rs, "random_strict"), ("approx", "approx", ra, "random_approx")]:
        groups = [("Baseline (21)", cv_df[cv_df.arm == "baseline"], "baseline"),
                  (f"{arm.capitalize()} RS", cv_df[cv_df.arm == arm], arm),
                  ("Random same size", rnd, rkey)]
        fig, ax = plt.subplots(figsize=(10, 4.2))
        w = 0.26
        for gi, (lab, d, ck) in enumerate(groups):
            m = [d[k].mean() for k in up.METRICS]; s = [d[k].std(ddof=1) for k in up.METRICS]
            ax.bar(np.arange(len(up.METRICS)) + (gi - 1) * w, m, w, yerr=s, capsize=2, label=lab, color=COLORS[ck])
        ax.set_xticks(np.arange(len(up.METRICS))); ax.set_xticklabels(up.METRICS)
        ax.set_ylabel("Mean ± SD over 50 splits"); ax.set_ylim(0, 1.05)
        ax.set_title(f"Baseline vs {tag} Rough Set vs same-size random control"); ax.legend(frameon=False, ncol=3, loc="upper right"); style(ax)
        fig.tight_layout(); fig.savefig(PLOTS / f"metrics_baseline_vs_{tag}_vs_random.png", dpi=150); plt.close(fig)


def fmt_ms(d, m):
    return f"{d[m].mean():.4f} ± {d[m].std(ddof=1):.4f}"


def main():
    run_exh = "--skip-exhaustive" not in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    df, features = up.load_kc1(DATA_PATH)
    X, y = df[features], df[up.TARGET].astype(int)
    aud = audit(df, features)
    assert aud["rows"] == 2109 and len(features) == 21 and aud["missing_values"] == 0
    print("audit", aud, flush=True)

    print("full-dataset analysis (exhaustive 2^21)...", flush=True)
    full = full_dataset_analysis(X, y, features, run_exh)
    (OUT / "full_dataset_exact_reducts.json").write_text(json.dumps(full, indent=2, default=float))
    print("  gamma", full["full_dependency"], "QR", full["quickreduct_pruned_size"],
          full.get("exhaustive", {}).get("n_minimum_reducts"), flush=True)

    sens = full_dataset_sensitivity(X, y, features)
    sens.to_csv(OUT / "discretization_sensitivity_full_dataset.csv", index=False)

    splits = up.make_splits(X, y)
    assert len(splits) == 50
    print("primary CV...", flush=True)
    cv_df, rand_df, preds, sel_s, sel_a, checks = run_primary_cv(X, y, features, splits)
    cv_df.to_csv(OUT / "cv_split_results.csv", index=False)
    rand_df.to_csv(OUT / "random_control_results.csv", index=False)
    preds.to_csv(OUT / "predictions.csv", index=False)
    checks.to_csv(OUT / "split_checks.csv", index=False)

    freq_s, freq_a = up.feature_frequency(sel_s, features), up.feature_frequency(sel_a, features)
    freq_s.to_csv(OUT / "strict_feature_frequency.csv", index=False)
    freq_a.to_csv(OUT / "approx_feature_frequency.csv", index=False)

    tests, rs, ra = statistical_tests(cv_df, rand_df)
    tests.to_csv(OUT / "statistical_tests.csv", index=False)

    print("CV sensitivity...", flush=True)
    cvs = cv_sensitivity(X, y, features, splits)
    cvs.to_csv(OUT / "discretization_sensitivity_cv.csv")

    make_plots(df, cv_df, rand_df, rs, ra, freq_s, freq_a)

    # ---- validation
    metric_cols = up.METRICS
    probs_ok = bool(preds.y_proba.between(0, 1).all())
    finite_ok = bool(np.isfinite(cv_df[metric_cols].to_numpy()).all() and np.isfinite(rand_df[metric_cols].to_numpy()).all())
    stratified_ok = bool(((checks.test_pos_rate - y.mean()).abs() < 0.02).all())
    validation = {
        "n_splits_completed": len(splits),
        "no_train_test_overlap": True,  # asserted per split
        "strict_reduct_preserves_train_dependency_all_splits": bool(
            (checks.gamma_strict_train - checks.gamma_full_train).abs().max() < 1e-12),
        "approx_reaches_alpha_all_splits": bool(checks.approx_ratio.min() >= up.ALPHA - 1e-12),
        "min_approx_ratio": float(checks.approx_ratio.min()),
        "baseline_always_21_features": bool((cv_df[cv_df.arm == "baseline"].n_features == 21).all()),
        "random_matches_rs_count": True,  # asserted per split
        "all_selected_names_valid": True,  # asserted per split
        "probabilities_in_0_1": probs_ok, "all_metrics_finite": finite_ok,
        "test_prevalence_within_2pp_of_global": stratified_ok,
        "exhaustive_2^21_completed": run_exh,
    }

    def arm_summary(d): return {m: {"mean": float(d[m].mean()), "std": float(d[m].std(ddof=1))} for m in metric_cols}
    base = cv_df[cv_df.arm == "baseline"]; st = cv_df[cv_df.arm == "strict"]; ap = cv_df[cv_df.arm == "approx"]
    sizes = lambda l: {"mean": float(np.mean([len(s) for s in l])), "min": min(map(len, l)), "max": max(map(len, l))}
    ss, sa = sizes(sel_s), sizes(sel_a)
    stable = lambda fr, t: [f for f, c in zip(fr.feature, fr["count"]) if c >= t]
    summary = {
        "title": "Rough Set-Based Feature Reduction for Software Defect Prediction",
        "metadata": {
            "python": sys.version, "platform": platform.platform(),
            "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                         "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
            "dataset_path": str(DATA_PATH.relative_to(ROOT)), "dataset_rows": len(df),
            "dataset_columns": list(df.columns), "primary_discretization": f"{PRIMARY_BINS}-bin equal-frequency (train-fold only)",
            "requested_bin_count": PRIMARY_BINS,
            "actual_bin_counts_full_dataset": full["actual_bins"],
            "actual_bin_counts_per_split": "see split_checks.csv (actual_bins_train)",
            "cv": {"class": "RepeatedStratifiedKFold", "n_splits": up.N_SPLITS, "n_repeats": up.N_REPEATS,
                   "random_state": up.CV_SEED, "total_splits": len(splits)},
            "decision_tree": up.TREE_PARAMS, "alpha": up.ALPHA, "random_control_count_per_split": up.N_RANDOM,
            "random_seeds": f"np.random.default_rng([{up.CV_SEED}, split_id, 0 (strict) | 1 (approx)]); {up.N_RANDOM} subsets drawn sequentially",
        },
        "audit": aud, "validation": validation,
        "full_dataset_3bin": {k: v for k, v in full.items() if k != "exhaustive"} | (
            {"exhaustive": {k: v for k, v in full["exhaustive"].items() if k != "all_strict_reducts"}} if run_exh else {}),
        "primary_cv": {
            "baseline": arm_summary(base), "strict": arm_summary(st), "random_strict": arm_summary(rs),
            "approx": arm_summary(ap), "random_approx": arm_summary(ra)},
        "feature_reduction": {
            "strict": ss | {"mean_reduction_pct": 100 * (1 - ss["mean"] / 21)},
            "approx": sa | {"mean_reduction_pct": 100 * (1 - sa["mean"] / 21)}},
        "cross_fold_stable_features": {
            "strict": {"50/50": stable(freq_s, 50), ">=45/50": stable(freq_s, 45), ">=40/50": stable(freq_s, 40)},
            "approx": {"50/50": stable(freq_a, 50), ">=45/50": stable(freq_a, 45), ">=40/50": stable(freq_a, 40)}},
        "statistical_tests": tests.to_dict(orient="records"),
        "sensitivity_full_dataset": sens.to_dict(orient="records"),
        "sensitivity_cv_strict": {m: {c: float(v) for c, v in zip(["/".join(k) for k in cvs.columns], cvs.loc[m].values)} for m in cvs.index},
    }
    (OUT / "final_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_text(summary, tests, sens, cvs, full, freq_s, freq_a, base, st, ap, rs, ra, ss, sa)
    print((OUT / "final_summary.txt").read_text())


def write_text(S, tests, sens, cvs, full, freq_s, freq_a, base, st, ap, rs, ra, ss, sa):
    L = []
    P = L.append
    P("=" * 78); P(S["title"]); P("Official unified pipeline (PROMISE KC1)"); P("=" * 78)
    md = S["metadata"]
    P(f"Python {md['python'].split()[0]} | " + ", ".join(f"{k} {v}" for k, v in md["packages"].items()))
    P(f"Dataset: {md['dataset_path']} ({md['dataset_rows']} rows, 21 conditions + defects)")
    a = S["audit"]
    P(f"Audit: classes {a['class_counts']}, missing {a['missing_values']}, repeated rows {a['repeated_rows_beyond_first']}, "
      f"conflicting-label rows {a['conflicting_label_rows']} ({a['conflicting_feature_vectors']} vectors). Nothing removed.")
    P(f"Primary discretization: {md['primary_discretization']} | CV: {up.N_SPLITS}-fold x {up.N_REPEATS} "
      f"stratified, random_state={up.CV_SEED} (50 splits) | DecisionTree {up.TREE_PARAMS} | alpha={up.ALPHA} | "
      f"{up.N_RANDOM} random controls/split")
    P("")
    P("VALIDATION"); P("-" * 78)
    for k, v in S["validation"].items(): P(f"  {k}: {v}")
    P("")
    P("FULL-DATASET 3-BIN ROUGH SET ANALYSIS (descriptive only; never used on test folds)"); P("-" * 78)
    P(f"  gamma_C(D) = {full['full_dependency']:.4f}; positive region = {full['positive_region_size']}/{full['universe_size']}; "
      f"equivalence classes = {full['equivalence_class_count']}; inconsistent classes = {full['inconsistent_equivalence_class_count']}")
    P(f"  actual bins: {full['actual_bins']}")
    P(f"  defects=True approximation: {full['defects_true_approximation']}")
    P(f"  QuickReduct forward ({len(full['quickreduct_forward'])}): {full['quickreduct_forward']}")
    P(f"  backward removed: {full['quickreduct_backward_removed']}")
    P(f"  QuickReduct+pruned ({full['quickreduct_pruned_size']}): {full['quickreduct_pruned']}")
    if "exhaustive" in full:
        e = full["exhaustive"]
        P(f"  Exhaustive search of {e['subsets_searched']:,} subsets in {e['seconds']}s: {e['n_strict_reducts']} strict reducts, "
          f"size distribution {e['strict_reduct_size_distribution']}")
        P(f"  Minimum reduct size = {e['min_reduct_size']}; {e['n_minimum_reducts']} minimum reducts:")
        for r in e["minimum_reducts"]: P(f"    {r}")
        P(f"  Formal core (intersection of minimum reducts, {len(e['formal_core_intersection_of_minimum_reducts'])}): {e['formal_core_intersection_of_minimum_reducts']}")
        P(f"  Intersection of ALL strict reducts ({len(e['intersection_of_all_strict_reducts'])}): {e['intersection_of_all_strict_reducts']}")
        P(f"  QuickReduct is a strict reduct: {full['quickreduct_is_strict_reduct']}; matches a minimum reduct: "
          f"{full['quickreduct_matches_a_minimum_reduct']} (size diff {full['quickreduct_size_minus_minimum_size']})")
    P("")
    P("PRIMARY PREDICTIVE RESULTS (mean ± SD over 50 splits; positive class = defects True)"); P("-" * 78)
    hdr = f"  {'arm':<24}" + "".join(f"{m:>17}" for m in up.METRICS)
    P(hdr)
    for lab, d in [("Baseline (21)", base), (f"Strict RS (~{ss['mean']:.1f})", st), ("Random, strict size", rs),
                   (f"Approx RS (~{sa['mean']:.1f})", ap), ("Random, approx size", ra)]:
        P(f"  {lab:<24}" + "".join(f"{fmt_ms(d, m):>17}" for m in up.METRICS))
    P("")
    P("FEATURE REDUCTION"); P("-" * 78)
    for k, s in S["feature_reduction"].items():
        P(f"  {k}: mean size {s['mean']:.2f}, min {s['min']}, max {s['max']}, mean reduction {s['mean_reduction_pct']:.1f}%")
    for name, fr in [("strict", freq_s), ("approx", freq_a)]:
        P(f"  cross-fold selection frequency ({name}, out of 50):")
        P("    " + ", ".join(f"{r.feature}={int(r['count'])}" for _, r in fr.sort_values('count', ascending=False).iterrows()))
        st_ = S["cross_fold_stable_features"][name]
        P(f"    50/50: {st_['50/50']}\n    >=45/50: {st_['>=45/50']}\n    >=40/50: {st_['>=40/50']}")
    P("  (these are cross-fold stable selected features, NOT the formal core)")
    P("")
    P("PAIRED WILCOXON SIGNED-RANK TESTS (same 50 splits; differences = first - second)"); P("-" * 78)
    P(f"  {'comparison':<26}{'metric':<9}{'mean diff':>11}{'median diff':>13}{'W':>9}{'p':>10}{'nonzero':>9}")
    for r in tests.to_dict(orient="records"):
        P(f"  {r['comparison']:<26}{r['metric']:<9}{r['mean_diff']:>11.4f}{r['median_diff']:>13.4f}"
          f"{r['wilcoxon_statistic']:>9.1f}{r['p_value']:>10.4f}{r['n_nonzero_diff']:>9d}")
    P("  Note: repeated-CV splits overlap, so p-values are indicative, not exact; no multiplicity correction applied.")
    P("")
    P("DISCRETIZATION SENSITIVITY, FULL DATASET (descriptive)"); P("-" * 78)
    P(sens.drop(columns=["collapsed_feature_names", "strict_reduct"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    P("")
    P("DISCRETIZATION SENSITIVITY, 50-SPLIT CV (strict reduct, Decision Tree; mean/std)"); P("-" * 78)
    P(cvs.round(4).to_string())
    P("  3-bin quantile is the PRIMARY method; the others are sensitivity checks (MDLP is supervised, fit on training folds only).")
    (OUT / "final_summary.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
