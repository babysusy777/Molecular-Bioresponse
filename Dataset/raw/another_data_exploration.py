import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MISSING_TOKENS = ["", " ", "NA", "N/A", "na", "n/a", "?", "unknown", "Unknown", "None", "none", "null", "NULL"]


def is_binary_01(series: pd.Series) -> bool:
    values = series.dropna().unique()
    if len(values) == 0:
        return False
    return set(values).issubset({0, 1, 0.0, 1.0})


def compute_feature_report(X: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in X.columns:
        s = X[col]
        missing_count = s.isna().sum()
        missing_pct = missing_count / len(s) * 100

        n_unique = s.nunique(dropna=True)
        unique_ratio = n_unique / len(s)

        value_counts = s.value_counts(dropna=True)
        dominant_value_pct = np.nan
        if len(value_counts) > 0:
            dominant_value_pct = value_counts.iloc[0] / len(s) * 100

        is_numeric = pd.api.types.is_numeric_dtype(s)
        is_binary = is_binary_01(s) if is_numeric else False

        row = {
            "feature": col,
            "dtype": str(s.dtype),
            "is_numeric": is_numeric,
            "is_binary_01": is_binary,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "n_unique": n_unique,
            "unique_ratio": unique_ratio,
            "dominant_value_pct": dominant_value_pct,
            "constant": n_unique <= 1,
            "quasi_constant_99": dominant_value_pct >= 99 if not np.isnan(dominant_value_pct) else False,
            "quasi_constant_995": dominant_value_pct >= 99.5 if not np.isnan(dominant_value_pct) else False,
        }

        if is_numeric:
            s_num = pd.to_numeric(s, errors="coerce")

            row.update({
                "mean": s_num.mean(),
                "std": s_num.std(),
                "variance": s_num.var(),
                "min": s_num.min(),
                "q1": s_num.quantile(0.25),
                "median": s_num.median(),
                "q3": s_num.quantile(0.75),
                "max": s_num.max(),
                "skewness": s_num.skew(),
                "zero_pct": (s_num == 0).mean() * 100,
                "one_pct": (s_num == 1).mean() * 100,
            })

            # IQR outlier count only makes real sense for non-binary numerical features
            if not is_binary:
                q1 = s_num.quantile(0.25)
                q3 = s_num.quantile(0.75)
                iqr = q3 - q1

                if pd.notna(iqr) and iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_iqr_count = ((s_num < lower) | (s_num > upper)).sum()
                    outlier_iqr_pct = outlier_iqr_count / len(s_num) * 100
                else:
                    outlier_iqr_count = 0
                    outlier_iqr_pct = 0.0

                row.update({
                    "outlier_iqr_count": outlier_iqr_count,
                    "outlier_iqr_pct": outlier_iqr_pct,
                })
            else:
                row.update({
                    "outlier_iqr_count": np.nan,
                    "outlier_iqr_pct": np.nan,
                })

        rows.append(row)

    return pd.DataFrame(rows)


def save_basic_plots(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, report: pd.DataFrame, plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Target distribution
    plt.figure(figsize=(6, 4))
    y.value_counts(dropna=False).sort_index().plot(kind="bar")
    plt.title("Target distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plots_dir / "target_distribution.png", dpi=150)
    plt.close()

    # Missingness top 30
    missing_top = report.sort_values("missing_pct", ascending=False).head(30)
    plt.figure(figsize=(10, 6))
    plt.barh(missing_top["feature"], missing_top["missing_pct"])
    plt.gca().invert_yaxis()
    plt.title("Top 30 features by missing percentage")
    plt.xlabel("Missing values (%)")
    plt.tight_layout()
    plt.savefig(plots_dir / "missingness_top30.png", dpi=150)
    plt.close()

    # Feature type counts
    type_counts = pd.Series({
        "binary_0_1": int(report["is_binary_01"].sum()),
        "numeric_non_binary": int((report["is_numeric"] & ~report["is_binary_01"]).sum()),
        "non_numeric": int((~report["is_numeric"]).sum()),
    })

    plt.figure(figsize=(6, 4))
    type_counts.plot(kind="bar")
    plt.title("Feature type overview")
    plt.ylabel("Number of features")
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_type_overview.png", dpi=150)
    plt.close()

    # Variance distribution for numeric features
    numeric_variances = report.loc[report["is_numeric"], "variance"].dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(numeric_variances, bins=50)
    plt.title("Variance distribution of numeric features")
    plt.xlabel("Variance")
    plt.ylabel("Number of features")
    plt.tight_layout()
    plt.savefig(plots_dir / "variance_distribution.png", dpi=150)
    plt.close()

    # Sparsity / zero percentage
    zero_pct = report.loc[report["is_numeric"], "zero_pct"].dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(zero_pct, bins=50)
    plt.title("Zero percentage distribution")
    plt.xlabel("Zero values (%)")
    plt.ylabel("Number of features")
    plt.tight_layout()
    plt.savefig(plots_dir / "zero_percentage_distribution.png", dpi=150)
    plt.close()


def write_text_summary(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    report: pd.DataFrame,
    target_col: str,
    output_path: Path,
):
    n_rows, n_cols = df.shape
    n_features = X.shape[1]

    duplicate_full_rows = df.duplicated().sum()
    duplicate_feature_rows = X.duplicated().sum()

    target_missing = y.isna().sum()
    target_counts = y.value_counts(dropna=False)

    numeric_features = report["is_numeric"].sum()
    binary_features = report["is_binary_01"].sum()
    non_numeric_features = (~report["is_numeric"]).sum()

    constant_features = report["constant"].sum()
    quasi_constant_99 = report["quasi_constant_99"].sum()
    features_with_missing = (report["missing_count"] > 0).sum()
    features_missing_over_20 = (report["missing_pct"] > 20).sum()
    features_missing_over_50 = (report["missing_pct"] > 50).sum()

    high_zero_features = (report["zero_pct"] > 95).sum() if "zero_pct" in report.columns else 0

    lines = []
    lines.append("DATA QUALITY AUDIT - BIORESPONSE DATASET")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Dataset shape: {n_rows} rows x {n_cols} columns")
    lines.append(f"Target column: {target_col}")
    lines.append(f"Number of features: {n_features}")
    lines.append("")
    lines.append("TARGET")
    lines.append("-" * 60)
    lines.append(f"Missing target values: {target_missing}")
    lines.append("Target distribution:")
    lines.append(str(target_counts))
    lines.append("")
    lines.append("DUPLICATES")
    lines.append("-" * 60)
    lines.append(f"Duplicated full rows: {duplicate_full_rows}")
    lines.append(f"Duplicated feature vectors, ignoring target: {duplicate_feature_rows}")
    lines.append("")
    lines.append("FEATURE TYPES")
    lines.append("-" * 60)
    lines.append(f"Numeric features: {numeric_features}")
    lines.append(f"Binary 0/1 features: {binary_features}")
    lines.append(f"Numeric non-binary features: {numeric_features - binary_features}")
    lines.append(f"Non-numeric features: {non_numeric_features}")
    lines.append("")
    lines.append("MISSING VALUES")
    lines.append("-" * 60)
    lines.append(f"Features with at least one missing value: {features_with_missing}")
    lines.append(f"Features with >20% missing values: {features_missing_over_20}")
    lines.append(f"Features with >50% missing values: {features_missing_over_50}")
    lines.append("")
    lines.append("LOW-INFORMATION FEATURES")
    lines.append("-" * 60)
    lines.append(f"Constant features: {constant_features}")
    lines.append(f"Quasi-constant features, dominant value >= 99%: {quasi_constant_99}")
    lines.append(f"Features with more than 95% zeros: {high_zero_features}")
    lines.append("")
    lines.append("TOP 20 FEATURES BY MISSING PERCENTAGE")
    lines.append("-" * 60)
    lines.append(str(report.sort_values("missing_pct", ascending=False)[["feature", "missing_pct", "dtype"]].head(20)))
    lines.append("")
    lines.append("TOP 20 QUASI-CONSTANT FEATURES")
    lines.append("-" * 60)
    lines.append(str(report.sort_values("dominant_value_pct", ascending=False)[["feature", "dominant_value_pct", "n_unique", "is_binary_01"]].head(20)))
    lines.append("")
    lines.append("TOP 20 FEATURES BY IQR OUTLIER PERCENTAGE")
    lines.append("-" * 60)
    if "outlier_iqr_pct" in report.columns:
        lines.append(str(report.sort_values("outlier_iqr_pct", ascending=False)[["feature", "outlier_iqr_pct", "is_binary_01"]].head(20)))
    else:
        lines.append("No numerical outlier information available.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the CSV dataset.")
    parser.add_argument("--target", default="Activity", help="Name of the target column. Example: Activity.")
    parser.add_argument("--reports_dir", default="reports", help="Directory where reports will be saved.")
    parser.add_argument("--plots_dir", default="plots", help="Directory where plots will be saved.")
    args = parser.parse_args()

    input_path = Path(args.input)
    reports_dir = Path(args.reports_dir)
    plots_dir = Path(args.plots_dir)

    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, na_values=MISSING_TOKENS, keep_default_na=True)

    # Replace infinities with NaN, because they are not usable as ordinary numerical values.
    df = df.replace([np.inf, -np.inf], np.nan)

    target_col = args.target  

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is not in the dataset columns.")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    report = compute_feature_report(X)

    report_path = reports_dir / "feature_quality_report.csv"
    summary_path = reports_dir / "quality_summary.txt"
    target_path = reports_dir / "target_distribution.csv"

    report.to_csv(report_path, index=False)
    y.value_counts(dropna=False).rename_axis(target_col).reset_index(name="count").to_csv(target_path, index=False)

    write_text_summary(df, X, y, report, target_col, summary_path)
    save_basic_plots(df, X, y, report, plots_dir)

    print("Quality audit completed.")
    print(f"Feature report saved to: {report_path}")
    print(f"Text summary saved to: {summary_path}")
    print(f"Target distribution saved to: {target_path}")
    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()