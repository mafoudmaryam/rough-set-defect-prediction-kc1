from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/kc1.csv")
TARGET_COLUMN = "defects"

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("KC1 DATASET AUDIT")
print("=" * 70)

print("\n1. DATASET SHAPE")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n2. COLUMN NAMES")
for i, column in enumerate(df.columns, start=1):
    print(f"{i:2d}. {column}")

print("\n3. DATA TYPES")
print(df.dtypes)

print("\n4. MISSING VALUES")
missing = df.isna().sum()
print(missing)

print("\nTotal missing values:", int(missing.sum()))

print("\n5. TARGET DISTRIBUTION")
print(df[TARGET_COLUMN].value_counts(dropna=False))

print("\nTarget percentages:")
print(
    df[TARGET_COLUMN]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
)

print("\n6. EXACT DUPLICATE ROWS")
exact_duplicate_mask = df.duplicated(keep=False)
exact_duplicate_count = int(exact_duplicate_mask.sum())

print(f"Rows involved in exact duplicates: {exact_duplicate_count}")
print(f"Number of repeated rows beyond first occurrence: {df.duplicated().sum()}")

print("\n7. DUPLICATE FEATURE VECTORS")
feature_columns = [c for c in df.columns if c != TARGET_COLUMN]

duplicate_feature_mask = df.duplicated(
    subset=feature_columns,
    keep=False
)

print(
    "Rows sharing identical feature values:",
    int(duplicate_feature_mask.sum())
)

print("\n8. CONFLICTING LABELS")
grouped = (
    df.groupby(feature_columns, dropna=False)[TARGET_COLUMN]
    .nunique()
)

conflicting_feature_vectors = grouped[grouped > 1]

print(
    "Unique feature vectors with conflicting labels:",
    len(conflicting_feature_vectors)
)

if len(conflicting_feature_vectors) > 0:
    conflicting_keys = conflicting_feature_vectors.index

    conflicts = (
        df.set_index(feature_columns)
        .loc[conflicting_keys]
        .reset_index()
        .sort_values(feature_columns)
    )

    print(
        "Rows involved in conflicting-label groups:",
        len(conflicts)
    )

    print("\nSample conflicting rows:")
    print(conflicts.head(20).to_string(index=False))

print("\n9. NUMERICAL SUMMARY")
print(df[feature_columns].describe().T)

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)