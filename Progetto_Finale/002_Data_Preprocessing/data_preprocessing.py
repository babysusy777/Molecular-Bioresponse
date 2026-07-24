from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

INPUT_PATH = PROJECT_DIR / "000_Dataset" / "raw" / "train.csv"
OUTPUT_DIR = PROJECT_DIR / "000_Dataset"
REPORT_DIR = BASE_DIR / "preprocessing" / "reports" / "correlation_analysis"

TARGET_COL = "Activity"
QUASI_CONSTANT_THRESHOLD = 0.99
CORRELATION_THRESHOLD = 0.95

OUTPUT_DATASET_PATH = OUTPUT_DIR / "train_filtered_no_activity.csv"
OUTPUT_TARGET_PATH = OUTPUT_DIR / "train_activity_target.csv"


def remove_quasi_constant_features(
    X: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove columns whose dominant value has frequency >= threshold."""
    rows = []
    for col in X.columns:
        frequencies = X[col].value_counts(normalize=True, dropna=False)
        ratio = frequencies.iloc[0] if not frequencies.empty else np.nan
        rows.append({
            "feature": col,
            "dominant_value_ratio": ratio,
            "dominant_value_pct": ratio * 100,
            "n_unique": X[col].nunique(dropna=False),
            "drop": ratio >= threshold,
        })

    report = pd.DataFrame(rows)
    to_drop = report.loc[report["drop"], "feature"].tolist()
    return X.drop(columns=to_drop), report


def identify_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return binary and non-binary numeric column names."""
    binary = [
        col for col in X.columns
        if (values := set(X[col].dropna().unique()))
        and values.issubset({0, 1})
    ]
    return binary, [col for col in X.columns if col not in binary]


def extract_correlated_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return pairs with absolute Pearson correlation >= threshold."""
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().reset_index()
    pairs.columns = ["feature_1", "feature_2", "correlation"]
    pairs["abs_correlation"] = pairs["correlation"].abs()
    return (
        pairs[pairs["abs_correlation"] >= threshold]
        .sort_values("abs_correlation", ascending=False)
        .reset_index(drop=True)
    )


def correlation_type(feature_1: str, feature_2: str, binary: set[str]) -> str:
    """Name the correlation according to the two feature types."""
    types = feature_1 in binary, feature_2 in binary
    if all(types):
        return "phi"
    if not any(types):
        return "pearson"
    return "point_biserial"


def select_correlated_features_to_drop(
    pairs: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame]:
    """Greedily retain feature_1 and remove feature_2 from correlated pairs."""
    to_drop, decisions = set(), []

    for row in pairs.itertuples(index=False):
        feature_1, feature_2 = row.feature_1, row.feature_2
        common = {
            "feature_1": feature_1,
            "feature_2": feature_2,
            "correlation": row.correlation,
            "abs_correlation": row.abs_correlation,
        }

        if feature_1 in to_drop:
            decision = {
                "kept_feature": None,
                "dropped_feature": None,
                "decision": "skipped_feature_1_already_dropped",
            }
        elif feature_2 in to_drop:
            decision = {
                "kept_feature": feature_1,
                "dropped_feature": feature_2,
                "decision": "already_dropped",
            }
        else:
            to_drop.add(feature_2)
            decision = {
                "kept_feature": feature_1,
                "dropped_feature": feature_2,
                "decision": "dropped",
            }
        decisions.append(common | decision)

    return sorted(to_drop), pd.DataFrame(decisions)


def preprocess_dataset(
    input_path: Path,
    target_column: str,
    output_dataset_path: Path,
    output_target_path: Path,
    report_dir: Path,
    quasi_constant_threshold: float = 0.99,
    correlation_threshold: float = 0.95,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run the complete preprocessing and correlation-analysis pipeline."""
    for path in {output_dataset_path.parent, output_target_path.parent, report_dir}:
        path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    non_numeric = [
        col for col in df.columns
        if col != target_column and not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise TypeError(
            "Correlation analysis requires numerical features. "
            f"Non-numerical columns found: {non_numeric}"
        )

    y = df[target_column].copy()
    X = df.drop(columns=target_column).copy()
    initial_features = X.shape[1]

    X, quasi_report = remove_quasi_constant_features(
        X, quasi_constant_threshold
    )
    quasi_report.to_csv(report_dir / "quasi_constant_report.csv", index=False)

    binary, numeric = identify_feature_types(X)
    pd.DataFrame({
        "feature": X.columns,
        "type": ["binary" if col in binary else "numeric" for col in X.columns],
    }).to_csv(report_dir / "feature_types.csv", index=False)

    corr = X.corr(method="pearson")
    matrices = {
        "complete_correlation_matrix.csv": corr,
        "pearson_numeric_numeric.csv": corr.loc[numeric, numeric],
        "phi_binary_binary.csv": corr.loc[binary, binary],
        "point_biserial_numeric_binary.csv": corr.loc[numeric, binary],
    }
    for filename, matrix in matrices.items():
        matrix.to_csv(report_dir / filename)

    pairs = extract_correlated_pairs(corr, correlation_threshold)
    binary_set = set(binary)
    if not pairs.empty:
        pairs["correlation_type"] = [
            correlation_type(a, b, binary_set)
            for a, b in zip(pairs["feature_1"], pairs["feature_2"])
        ]
    pairs.to_csv(report_dir / "highly_correlated_pairs.csv", index=False)

    correlated_to_drop, removal_report = select_correlated_features_to_drop(pairs)
    removal_report.to_csv(
        report_dir / "correlated_feature_removal.csv", index=False
    )

    X_final = X.drop(columns=correlated_to_drop)
    X_final.to_csv(output_dataset_path, index=False)
    y.to_csv(output_target_path, index=False, header=True)

    print(
        "\nPreprocessing summary\n"
        f"{'-' * 60}\n"
        f"Initial features: {initial_features}\n"
        f"Removed quasi-constant features: {int(quasi_report['drop'].sum())}\n"
        f"Features before correlation filtering: {X.shape[1]}\n"
        f"Binary features: {len(binary)}\n"
        f"Numeric features: {len(numeric)}\n"
        f"Correlation threshold: {correlation_threshold}\n"
        f"Highly correlated pairs: {len(pairs)}\n"
        f"Removed correlated features: {len(correlated_to_drop)}\n"
        f"Dropped correlated features: {correlated_to_drop}\n"
        f"Final features: {X_final.shape[1]}\n"
        f"Dataset saved in: {output_dataset_path}\n"
        f"Target saved in: {output_target_path}\n"
        f"Reports saved in: {report_dir}"
    )
    return X_final, y


def main() -> None:
    preprocess_dataset(
        INPUT_PATH,
        TARGET_COL,
        OUTPUT_DATASET_PATH,
        OUTPUT_TARGET_PATH,
        REPORT_DIR,
        QUASI_CONSTANT_THRESHOLD,
        CORRELATION_THRESHOLD,
    )


if __name__ == "__main__":
    main()