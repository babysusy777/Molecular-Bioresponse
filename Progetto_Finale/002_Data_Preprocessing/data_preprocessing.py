from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

INPUT_PATH = PROJECT_DIR / "Dataset" / "raw" / "train.csv"

OUTPUT_DIR = BASE_DIR / "Dataset"
REPORT_DIR = BASE_DIR / "preprocessing" / "reports" / "correlation_analysis"

TARGET_COL = "Activity"

QUASI_CONSTANT_THRESHOLD = 0.99
CORRELATION_THRESHOLD = 0.95

OUTPUT_FILTERED_DATASET_PATH = (
    OUTPUT_DIR / "train_filtered_no_activity.csv"
)
OUTPUT_TARGET_PATH = (
    OUTPUT_DIR / "train_activity_target.csv"
)


# ============================================================
# Quasi-constant feature analysis
# ============================================================

def dominant_value_ratio(series: pd.Series) -> float:
    """
    Returns the relative frequency of the most common value.
    """
    frequencies = series.value_counts(
        normalize=True,
        dropna=False
    )

    if frequencies.empty:
        return np.nan

    return frequencies.iloc[0]


def remove_quasi_constant_features(
    X: pd.DataFrame,
    threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Removes features whose most common value appears with
    relative frequency greater than or equal to threshold.

    Returns
    -------
    X_filtered:
        Dataset without quasi-constant features.

    report:
        Report containing the statistics and removal decision
        for each feature.
    """
    report_rows = []

    for column in X.columns:
        ratio = dominant_value_ratio(X[column])

        report_rows.append({
            "feature": column,
            "dominant_value_ratio": ratio,
            "dominant_value_pct": ratio * 100,
            "n_unique": X[column].nunique(dropna=False),
            "drop": ratio >= threshold,
        })

    report = pd.DataFrame(report_rows)

    features_to_drop = report.loc[
        report["drop"], "feature"
    ].tolist()

    X_filtered = X.drop(columns=features_to_drop)

    return X_filtered, report


# ============================================================
# Feature type detection
# ============================================================

def identify_feature_types(
    X: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """
    Identifies binary and non-binary numerical features.

    A feature is considered binary if all its non-missing
    values belong to {0, 1}.
    """
    binary_columns = []

    for column in X.columns:
        unique_values = set(X[column].dropna().unique())

        if unique_values and unique_values.issubset({0, 1}):
            binary_columns.append(column)

    numeric_columns = [
        column
        for column in X.columns
        if column not in binary_columns
    ]

    return binary_columns, numeric_columns


# ============================================================
# Correlation analysis
# ============================================================

def extract_highly_correlated_pairs(
    correlation_matrix: pd.DataFrame,
    threshold: float
) -> pd.DataFrame:
    """
    Extracts feature pairs whose absolute Pearson correlation
    is greater than or equal to threshold.

    Because binary variables are encoded as 0/1:

    - binary-binary Pearson correlation = Phi coefficient;
    - binary-numeric Pearson correlation = point-biserial;
    - numeric-numeric correlation = ordinary Pearson.
    """
    upper_triangle_mask = np.triu(
        np.ones(correlation_matrix.shape, dtype=bool),
        k=1
    )

    upper_triangle = correlation_matrix.where(
        upper_triangle_mask
    )

    pairs = (
        upper_triangle
        .stack()
        .reset_index()
    )

    pairs.columns = [
        "feature_1",
        "feature_2",
        "correlation"
    ]

    pairs["abs_correlation"] = pairs["correlation"].abs()

    pairs = pairs[
        pairs["abs_correlation"] >= threshold
    ]

    return pairs.sort_values(
        by="abs_correlation",
        ascending=False
    ).reset_index(drop=True)


def classify_correlation_type(
    feature_1: str,
    feature_2: str,
    binary_columns: set[str]
) -> str:
    """
    Assigns the correct name to the correlation according
    to the types of the two features.
    """
    feature_1_binary = feature_1 in binary_columns
    feature_2_binary = feature_2 in binary_columns

    if feature_1_binary and feature_2_binary:
        return "phi"

    if not feature_1_binary and not feature_2_binary:
        return "pearson"

    return "point_biserial"


# ============================================================
# Chi-square analysis
# ============================================================

def compute_binary_chi_square(
    X: pd.DataFrame,
    binary_columns: list[str],
    correlation_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes the chi-square statistic for every pair of binary
    features.

    For a 2x2 contingency table without Yates correction:

        chi-square = n * phi^2
    """
    results = []

    for feature_1, feature_2 in combinations(binary_columns, 2):
        pair_data = X[[feature_1, feature_2]].dropna()
        pair_n = len(pair_data)

        if pair_n == 0:
            continue

        phi = correlation_matrix.loc[
            feature_1,
            feature_2
        ]

        if pd.isna(phi):
            continue

        chi2_statistic = pair_n * phi**2
        p_value = chi2.sf(chi2_statistic, df=1)

        results.append({
            "feature_1": feature_1,
            "feature_2": feature_2,
            "n_observations": pair_n,
            "chi2": chi2_statistic,
            "p_value": p_value,
            "phi": phi,
            "abs_phi": abs(phi),
        })

    return pd.DataFrame(results).sort_values(
        by="abs_phi",
        ascending=False
    ).reset_index(drop=True)


# ============================================================
# Correlated feature removal
# ============================================================

def select_correlated_features_to_drop(
    correlated_pairs: pd.DataFrame
) -> tuple[list[str], pd.DataFrame]:
    """
    Selects one feature to remove from each highly correlated
    pair.

    The procedure is greedy:

    - feature_1 is retained;
    - feature_2 is removed;
    - pairs involving a feature already removed are skipped.

    Since the pairs are ordered by decreasing absolute
    correlation, the strongest redundancies are handled first.
    """
    features_to_drop = set()
    decisions = []

    for row in correlated_pairs.itertuples(index=False):
        feature_1 = row.feature_1
        feature_2 = row.feature_2

        if feature_1 in features_to_drop:
            decisions.append({
                "feature_1": feature_1,
                "feature_2": feature_2,
                "correlation": row.correlation,
                "abs_correlation": row.abs_correlation,
                "kept_feature": None,
                "dropped_feature": None,
                "decision": "skipped_feature_1_already_dropped",
            })
            continue

        if feature_2 in features_to_drop:
            decisions.append({
                "feature_1": feature_1,
                "feature_2": feature_2,
                "correlation": row.correlation,
                "abs_correlation": row.abs_correlation,
                "kept_feature": feature_1,
                "dropped_feature": feature_2,
                "decision": "already_dropped",
            })
            continue

        features_to_drop.add(feature_2)

        decisions.append({
            "feature_1": feature_1,
            "feature_2": feature_2,
            "correlation": row.correlation,
            "abs_correlation": row.abs_correlation,
            "kept_feature": feature_1,
            "dropped_feature": feature_2,
            "decision": "dropped",
        })

    decision_report = pd.DataFrame(decisions)

    return sorted(features_to_drop), decision_report


# ============================================================
# Complete preprocessing pipeline
# ============================================================

def preprocess_dataset(
    input_path: Path,
    target_column: str,
    output_dataset_path: Path,
    output_target_path: Path,
    report_dir: Path,
    quasi_constant_threshold: float = 0.99,
    correlation_threshold: float = 0.95,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Executes the complete preprocessing pipeline:

    1. loads the dataset;
    2. separates the target;
    3. removes quasi-constant features;
    4. identifies binary and numerical features;
    5. performs correlation analysis;
    6. performs chi-square tests on binary pairs;
    7. removes highly correlated features;
    8. saves datasets and reports.
    """
    output_dataset_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    output_target_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(input_path)

    print(
        f"Loaded dataset: "
        f"{df.shape[0]} rows x {df.shape[1]} columns"
    )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    non_numeric_columns = [
        column
        for column in df.columns
        if column != target_column
        and not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric_columns:
        raise TypeError(
            "Correlation analysis requires numerical features. "
            f"Non-numerical columns found: {non_numeric_columns}"
        )

    # --------------------------------------------------------
    # 2. Separate target
    # --------------------------------------------------------

    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()

    initial_features = X.shape[1]

    # --------------------------------------------------------
    # 3. Remove quasi-constant features
    # --------------------------------------------------------

    X, quasi_constant_report = (
        remove_quasi_constant_features(
            X,
            threshold=quasi_constant_threshold
        )
    )

    quasi_constant_report.to_csv(
        report_dir / "quasi_constant_report.csv",
        index=False
    )

    removed_quasi_constant = int(
        quasi_constant_report["drop"].sum()
    )

    # --------------------------------------------------------
    # 4. Identify feature types
    # --------------------------------------------------------

    binary_columns, numeric_columns = (
        identify_feature_types(X)
    )

    feature_type_report = pd.DataFrame({
        "feature": X.columns,
        "type": [
            "binary" if column in binary_columns
            else "numeric"
            for column in X.columns
        ],
    })

    feature_type_report.to_csv(
        report_dir / "feature_types.csv",
        index=False
    )

    # --------------------------------------------------------
    # 5. Compute correlations
    # --------------------------------------------------------

    correlation_matrix = X.corr(method="pearson")

    correlation_matrix.to_csv(
        report_dir / "complete_correlation_matrix.csv"
    )

    correlation_matrix.loc[
        numeric_columns,
        numeric_columns
    ].to_csv(
        report_dir / "pearson_numeric_numeric.csv"
    )

    correlation_matrix.loc[
        binary_columns,
        binary_columns
    ].to_csv(
        report_dir / "phi_binary_binary.csv"
    )

    correlation_matrix.loc[
        numeric_columns,
        binary_columns
    ].to_csv(
        report_dir / "point_biserial_numeric_binary.csv"
    )

    # --------------------------------------------------------
    # 6. Extract highly correlated pairs
    # --------------------------------------------------------

    correlated_pairs = extract_highly_correlated_pairs(
        correlation_matrix,
        threshold=correlation_threshold
    )

    binary_column_set = set(binary_columns)

    if not correlated_pairs.empty:
        correlated_pairs["correlation_type"] = [
            classify_correlation_type(
                feature_1,
                feature_2,
                binary_column_set
            )
            for feature_1, feature_2 in zip(
                correlated_pairs["feature_1"],
                correlated_pairs["feature_2"]
            )
        ]

    correlated_pairs.to_csv(
        report_dir / "highly_correlated_pairs.csv",
        index=False
    )

    # --------------------------------------------------------
    # 7. Chi-square tests for binary pairs
    # --------------------------------------------------------

    chi_square_results = compute_binary_chi_square(
        X,
        binary_columns,
        correlation_matrix
    )

    chi_square_results.to_csv(
        report_dir / "chi2_binary_binary.csv",
        index=False
    )

    # --------------------------------------------------------
    # 8. Remove redundant correlated features
    # --------------------------------------------------------

    correlated_features_to_drop, removal_report = (
        select_correlated_features_to_drop(
            correlated_pairs
        )
    )

    removal_report.to_csv(
        report_dir / "correlated_feature_removal.csv",
        index=False
    )

    X_final = X.drop(
        columns=correlated_features_to_drop
    )

    # --------------------------------------------------------
    # 9. Save final datasets
    # --------------------------------------------------------

    X_final.to_csv(
        output_dataset_path,
        index=False
    )

    y.to_csv(
        output_target_path,
        index=False,
        header=True
    )

    # --------------------------------------------------------
    # 10. Print summary
    # --------------------------------------------------------

    print()
    print("Preprocessing summary")
    print("-" * 60)
    print(f"Initial features: {initial_features}")
    print(
        "Removed quasi-constant features: "
        f"{removed_quasi_constant}"
    )
    print(
        "Features before correlation filtering: "
        f"{X.shape[1]}"
    )
    print(f"Binary features: {len(binary_columns)}")
    print(f"Numeric features: {len(numeric_columns)}")
    print(
        f"Correlation threshold: "
        f"{correlation_threshold}"
    )
    print(
        "Highly correlated pairs: "
        f"{len(correlated_pairs)}"
    )
    print(
        "Removed correlated features: "
        f"{len(correlated_features_to_drop)}"
    )
    print(
        "Dropped correlated features: "
        f"{correlated_features_to_drop}"
    )
    print(f"Final features: {X_final.shape[1]}")
    print(f"Dataset saved in: {output_dataset_path}")
    print(f"Target saved in: {output_target_path}")
    print(f"Reports saved in: {report_dir}")

    return X_final, y


# ============================================================
# Main
# ============================================================

def main() -> None:
    preprocess_dataset(
        input_path=INPUT_PATH,
        target_column=TARGET_COL,
        output_dataset_path=OUTPUT_FILTERED_DATASET_PATH,
        output_target_path=OUTPUT_TARGET_PATH,
        report_dir=REPORT_DIR,
        quasi_constant_threshold=QUASI_CONSTANT_THRESHOLD,
        correlation_threshold=CORRELATION_THRESHOLD,
    )


if __name__ == "__main__":
    main()