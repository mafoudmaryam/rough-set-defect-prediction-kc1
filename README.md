# Granular Computing Project — Official Unified Team Package

## Project

**Rough Set-Based Feature Reduction for Software Defect Prediction**

Team:
- Achraf Ait Tayeb
- Maryam Mafoud
- Ouassim Lahraoui

Course: Granular Computing, Sichuan University

---

## Important

This folder contains the **official unified version of the project**.

Earlier experiments carried out separately by team members used different settings and should be treated as developmental work.

If an older result conflicts with the files in this package, use the results in this package.

Do not mix old experimental numbers with the official unified results.

---

## Dataset

Dataset:

`data/raw/kc1.csv`

PROMISE / NASA KC1, 2,109-instance version.

- 2,109 software modules
- 21 condition attributes
- target: `defects`
- 1,783 non-defective
- 326 defective
- no missing values

The exact dataset contains duplicate observations and some identical feature vectors with conflicting defect labels. These observations are intentionally retained.

---

## Official Primary Method

The primary Rough Set setup uses:

- 3-bin equal-frequency discretization
- Low / Medium / High information granules
- dependency-based QuickReduct
- backward pruning
- strict reduct preserving full dependency

Secondary analysis:

- approximate reduct with alpha = 0.90
- random same-size controls
- 5-bin, 8-bin and MDLP discretization sensitivity

---

## Predictive Evaluation

Official evaluation:

- Repeated Stratified 10-Fold Cross-Validation
- 5 repeats
- 50 total test evaluations
- random state = 42

Within every split:

1. Split training and test data.
2. Fit discretization using training data only.
3. Perform Rough Set feature selection using training data only.
4. Apply the selected feature names to the original numerical data.
5. Train a Decision Tree.
6. Evaluate on the untouched test fold.

This design prevents test-data leakage.

Classifier:

`DecisionTreeClassifier(random_state=42, class_weight=None)`

The main experiment does not use:

- SMOTE
- oversampling
- undersampling
- class weighting
- hyperparameter tuning

---

## Official Results

### Baseline

21 features

Mean performance over 50 splits:

- Accuracy: 0.823
- F1: 0.393
- MCC: 0.294
- ROC-AUC: 0.617
- PR-AUC: 0.266

### Strict Rough Set Reduct

Average features: **16.94**

Average reduction: **19.3%**

- Accuracy: 0.822
- F1: 0.396
- MCC: 0.296
- ROC-AUC: 0.619
- PR-AUC: 0.268

Interpretation:

The strict reduct removes about 19% of the software metrics while predictive performance remains essentially unchanged.

### Approximate Rough Set Reduct

alpha = 0.90

Average features: **12.32**

Average reduction: **41.3%**

- Accuracy: 0.825
- F1: 0.416
- MCC: 0.316
- ROC-AUC: 0.632
- PR-AUC: 0.278

Interpretation:

The approximate reduct provides substantially stronger compression while remaining competitive in the preliminary predictive evaluation.

Do not claim that Rough Sets universally improve prediction.

---

## Exact Full-Dataset Rough Set Analysis

Using the primary 3-bin representation:

- dependency gamma = 0.3224
- positive region = 680 / 2109
- 408 equivalence classes / granules

All:

`2^21 = 2,097,152`

feature subsets were examined.

Results:

- exactly 2 strict minimum reducts
- both contain 17 features
- they differ only by `e` versus `t`
- their intersection contains 16 features
- this 16-feature intersection is the formal core for this decision table
- QuickReduct found one of the exact minimum reducts

The exhaustive full-dataset analysis is descriptive and is not used to leak information into predictive test folds.

---

## Random Controls

For each cross-validation split, 10 random subsets containing the same number of features as the Rough Set subset are evaluated.

The strict Rough Set subset and same-size random subsets perform similarly.

Therefore we do **not** claim that strict Rough Set selection is predictively superior to random feature removal.

---

## Automated Validation

The unified implementation passed:

**47 automated tests**

Tests include:

- Rough Set calculations
- discretization
- cross-validation
- strict dependency preservation
- approximate reduct threshold
- exhaustive-search correctness
- train/test separation
- leakage checks
- random-control feature counts
- output validity

---

## Important Files

### Main experiment

`src/run_final_unified_experiment.py`

### Unified methodology

`src/unified_protocol.py`

### Rough Set functions

`src/rough_sets.py`

### Discretization

`src/discretization.py`

### Dataset audit

`src/audit_kc1.py`

### Tests

`tests/`

### Official results

`results/`

The most useful summary is:

`results/final_summary.txt`

---

## How to Run

Create and activate a virtual environment if needed:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the tests:

```bash
python -m pytest
```

Run the full experiment (about 5 minutes; results are written to `results/final/`):

```bash
python src/run_final_unified_experiment.py
```

---

## Credits

Team: Achraf Ait Tayeb, Ouassim Lahraoui, Maryam Mafoud (Sichuan University, Granular Computing course).

Dataset: PROMISE / NASA KC1 (public). These are early project-proposal results, not final results.
