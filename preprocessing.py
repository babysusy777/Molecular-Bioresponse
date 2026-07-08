from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Configuration

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "Dataset" / "raw" / "train.csv"

TARGET_COL = "Activity"

QUASI_CONSTANT_THRESHOLD = 0.99
HIGH_CORRELATION_THRESHOLD = 0.95

OUTPUT_DIR = BASE_DIR / "Dataset"
REPORT_DIR = BASE_DIR / "reports" / "correlation_analysis"
PLOTS_DIR = BASE_DIR / "plots" / "correlation_analysis"

OUTPUT_FILTERED_DATASET_PATH = OUTPUT_DIR / "train_filtered_no_activity.csv"
OUTPUT_TARGET_PATH = OUTPUT_DIR / "train_activity_target.csv"

QUASI_CONSTANT_REPORT_PATH = REPORT_DIR / "quasi_constant_report.csv"
CORRELATION_MATRIX_PATH = REPORT_DIR / "feature_correlation_matrix.csv"


# Utils

def dominant_value_ratio(series: pd.Series) -> float:
    """
    Computes the frequency ratio of the most common value.
    """
    return series.value_counts(normalize=True, dropna=False).iloc[0]


def remove_quasi_constant_features(X: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Removes features whose dominant value appears with frequency >= threshold.
    """
    report_rows = []

    for col in X.columns:
        ratio = dominant_value_ratio(X[col])

        report_rows.append({
            "feature": col,
            "dominant_value_ratio": ratio,
            "dominant_value_pct": ratio * 100,
            "n_unique": X[col].nunique(dropna=False),
            "drop": ratio >= threshold
        })

    report = pd.DataFrame(report_rows)

    features_to_drop = report.loc[report["drop"], "feature"].tolist()

    X_filtered = X.drop(columns=features_to_drop)

    return X_filtered, report


def compute_highly_correlated_pairs(corr_matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Extracts feature pairs whose absolute Pearson correlation
    is greater than or equal to the selected threshold.
    """
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    pairs = upper_triangle.stack().reset_index()
    pairs.columns = ["feature_1", "feature_2", "correlation"]

    pairs["abs_correlation"] = pairs["correlation"].abs()

    pairs = pairs[pairs["abs_correlation"] >= threshold]

    pairs = pairs.sort_values(by="abs_correlation", ascending=False)

    return pairs



# Main


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")


    # 1. Separate target

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    print(f"Target column removed from feature matrix: {TARGET_COL}")
    print(f"Initial number of features: {X.shape[1]}")

    y.to_csv(OUTPUT_TARGET_PATH, index=False)


    # 2. Remove quasi-constant features

    X_filtered, quasi_constant_report = remove_quasi_constant_features(X,threshold=QUASI_CONSTANT_THRESHOLD)
    X_filtered.to_csv(OUTPUT_FILTERED_DATASET_PATH,index=False)

    removed_quasi_constant = int(quasi_constant_report["drop"].sum())

    print()
    print("Quasi-constant feature filtering")
    print("-" * 60)
    print(f"Threshold: {QUASI_CONSTANT_THRESHOLD * 100:.2f}%")
    print(f"Removed quasi-constant features: {removed_quasi_constant}")
    print(f"Remaining features: {X_filtered.shape[1]}")


    # 3. Compute correlation matrix

    print()
    print("Computing Pearson feature-feature correlation matrix...")

    corr_matrix = X_filtered.corr(method="pearson")
    corr_matrix.to_csv(CORRELATION_MATRIX_PATH)

    print(f"Correlation matrix saved to: {CORRELATION_MATRIX_PATH}")


    # 4. Check highly correlated feature pairs

    highly_corr_pairs = compute_highly_correlated_pairs(corr_matrix, threshold=HIGH_CORRELATION_THRESHOLD) 
    print(f"High correlation threshold: {HIGH_CORRELATION_THRESHOLD}")

    if len(highly_corr_pairs) == 0:
        print()
        print("No redundant features were identified by correlation analysis.")

    print()
    print("Analysis completed.")



if __name__ == "__main__":
    main()