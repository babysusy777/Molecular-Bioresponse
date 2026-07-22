"""
Complete and reproducible spectral-biclustering analysis pipeline.

The script performs six distinct phases:

1. Data validation and feature profiling.
2. Screening of several (k_rows, k_columns) configurations.
3. Transparent candidate selection based on descriptive quality and parsimony.
4. Stability analysis across multiple random seeds using Adjusted Rand Index.
5. Detailed interpretation of the selected model:
   - row clusters and Activity association;
   - column-cluster feature composition;
   - checkerboard block statistics;
   - representative features and diagnostic plots.
6. Column-wise permutation test preserving each feature's marginal distribution.

Important methodological point
------------------------------
Activity is never used to fit SpectralBiclustering or to choose k. It is used
only after clustering as an external interpretation variable.

Expected project layout
-----------------------
Molecular-Bioresponse/
├── Dataset/processed/train_filtered_no_activity.csv
├── Dataset/processed/train_activity_target.csv
├── models/spectral_biclustering_pipeline.py
└── reports/

Dependencies
------------
numpy, pandas, matplotlib, scipy, scikit-learn
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass, replace
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import chi2_contingency
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = SCRIPT_DIR.parent


@dataclass(frozen=True)
class Config:
    """All experiment settings are centralized here."""

    input_path: Path = (
        PROJECT_DIR / "Dataset" / "train_filtered_no_activity.csv"
    )
    target_path: Path = (
        PROJECT_DIR / "Dataset" / "train_activity_target.csv"
    )
    report_dir: Path = PROJECT_DIR / "reports" / "spectral_biclustering_pipeline"
    delta_report_dir: Path = PROJECT_DIR / "reports" / "delta_biclustering"

    # Initial grid. k_rows and k_columns are deliberately allowed to differ.
    methods: tuple[str, ...] = ("bistochastic",)
    row_cluster_values: tuple[int, ...] = (2, 3, 4, 5, 6)
    column_cluster_values: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
    screening_seeds: tuple[int, ...] = (42,)

    # Parameters of sklearn.cluster.SpectralBiclustering.
    n_components: int = 10
    n_best: int = 5
    n_init: int = 20
    svd_method: str = "randomized"

    # Feature profiling.
    sparse_feature_threshold: float = 0.95
    representative_features_per_cluster: int = 15

    # Automatic candidate selection.
    # Retain configurations whose block R² is at most this amount below the
    # best R² for the same normalization method; among them prefer fewer blocks.
    r2_tolerance: float = 0.01
    minimum_row_entropy: float = 0.80
    minimum_column_entropy: float = 0.50
    number_of_candidates: int = 3

    # Optional manual override. Example:
    # manual_candidates = (("bistochastic", 3, 6), ("bistochastic", 4, 6))
    manual_candidates: tuple[tuple[str, int, int], ...] = ()

    # Stability analysis.
    stability_seeds: tuple[int, ...] = (42, 7, 99, 123, 2026)
    minimum_row_ari: float = 0.80
    minimum_column_ari: float = 0.80

    # Null experiment. Column-wise permutation preserves every feature's
    # marginal distribution, zero fraction and (for binary features) one rate.
    # With 20 permutations the smallest attainable Monte-Carlo p-value is 1/21.
    null_permutations: int = 20
    null_seed: int = 3107

    # Output controls.
    save_screening_block_statistics: bool = True
    save_screening_assignments: bool = False
    top_interesting_blocks: int = 15
    plot_dpi: int = 180


@dataclass(frozen=True)
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series | None
    original_row_ids: np.ndarray
    feature_profiles: pd.DataFrame

    @property
    def values(self) -> np.ndarray:
        return self.X.to_numpy(dtype=np.float64, copy=False)


@dataclass(frozen=True)
class PartitionResult:
    method: str
    n_row_clusters: int
    n_column_clusters: int
    seed: int
    row_labels: np.ndarray
    column_labels: np.ndarray
    metrics: dict[str, Any]
    block_statistics: pd.DataFrame
    column_cluster_statistics: pd.DataFrame

    @property
    def model_id(self) -> str:
        return (
            f"{self.method}_r{self.n_row_clusters}_"
            f"c{self.n_column_clusters}_seed{self.seed}"
        )


# =============================================================================
# General utilities
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete spectral-biclustering analysis pipeline."
    )
    parser.add_argument(
        "--skip-null",
        action="store_true",
        help="Skip the column-wise permutation test.",
    )
    parser.add_argument(
        "--null-permutations",
        type=int,
        default=None,
        help="Override the number of null permutations.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use a smaller grid, three stability seeds and no null test. "
            "Useful only for checking that the pipeline runs."
        ),
    )
    return parser.parse_args()


def configure_from_arguments(config: Config, args: argparse.Namespace) -> Config:
    if args.quick:
        config = replace(
            config,
            row_cluster_values=(3, 4),
            column_cluster_values=(5, 6),
            stability_seeds=(42, 7, 99),
            number_of_candidates=2,
            null_permutations=0,
        )

    if args.skip_null:
        config = replace(config, null_permutations=0)

    if args.null_permutations is not None:
        if args.null_permutations < 0:
            raise ValueError("--null-permutations must be non-negative.")
        config = replace(config, null_permutations=args.null_permutations)

    return config


def setup_output(config: Config) -> dict[str, Path]:
    paths = {
        "root": config.report_dir,
        "screening": config.report_dir / "01_screening",
        "candidates": config.report_dir / "02_candidates_and_stability",
        "final": config.report_dir / "03_final_model",
        "null": config.report_dir / "04_null_test",
        "comparison": config.report_dir / "05_method_comparison",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
        force=True,
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_ready(data), file, indent=2, ensure_ascii=False)


def model_metadata(
    method: str,
    n_row_clusters: int,
    n_column_clusters: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "method": method,
        "n_row_clusters": int(n_row_clusters),
        "n_column_clusters": int(n_column_clusters),
        "n_blocks": int(n_row_clusters * n_column_clusters),
        "seed": int(seed),
        "model_id": (
            f"{method}_r{n_row_clusters}_c{n_column_clusters}_seed{seed}"
        ),
    }


# =============================================================================
# Data loading and profiling
# =============================================================================


def load_numeric_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    frame = pd.read_csv(path)
    unnamed_columns = [
        column for column in frame.columns if str(column).startswith("Unnamed:")
    ]
    if unnamed_columns:
        frame = frame.drop(columns=unnamed_columns)

    return frame.apply(pd.to_numeric, errors="raise")


def build_feature_profiles(X: pd.DataFrame, config: Config) -> pd.DataFrame:
    values = X.to_numpy(dtype=np.float64, copy=False)
    zero_mask = np.isclose(values, 0.0)
    one_mask = np.isclose(values, 1.0)
    binary_mask = np.all(zero_mask | one_mask, axis=0)

    profiles = pd.DataFrame(
        {
            "feature": X.columns.astype(str),
            "feature_index": np.arange(X.shape[1], dtype=int),
            "is_binary": binary_mask,
            "feature_type": np.where(binary_mask, "binary", "numeric"),
            "mean": values.mean(axis=0),
            "std": values.std(axis=0),
            "variance": values.var(axis=0),
            "zero_fraction": zero_mask.mean(axis=0),
            "nonzero_fraction": (~zero_mask).mean(axis=0),
            "one_fraction": one_mask.mean(axis=0),
            "n_unique": [X[column].nunique(dropna=False) for column in X.columns],
        }
    )
    profiles["is_sparse"] = (
        profiles["zero_fraction"] >= config.sparse_feature_threshold
    )
    return profiles


def load_dataset(config: Config, output_paths: dict[str, Path]) -> DatasetBundle:
    X = load_numeric_csv(config.input_path)
    if X.empty:
        raise ValueError("The feature matrix is empty.")
    if X.isna().any().any():
        raise ValueError("The feature matrix contains missing values.")

    values = X.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("The feature matrix contains NaN or infinity.")
    if values.min() < 0:
        raise ValueError(
            "SpectralBiclustering with scale/bistochastic normalization "
            "requires a non-negative matrix."
        )

    original_row_ids = np.arange(len(X), dtype=int)

    # Zero-sum rows and columns cannot be scaled safely by the algorithm.
    valid_columns = ~np.isclose(values.sum(axis=0), 0.0)
    if not np.all(valid_columns):
        removed = pd.DataFrame(
            {"removed_zero_sum_feature": X.columns[~valid_columns]}
        )
        removed.to_csv(
            output_paths["root"] / "removed_zero_sum_features.csv", index=False
        )
        X = X.loc[:, valid_columns].copy()
        values = X.to_numpy(dtype=np.float64)

    valid_rows = ~np.isclose(values.sum(axis=1), 0.0)
    if not np.all(valid_rows):
        pd.DataFrame(
            {"removed_zero_sum_original_row": original_row_ids[~valid_rows]}
        ).to_csv(
            output_paths["root"] / "removed_zero_sum_rows.csv", index=False
        )

    X = X.loc[valid_rows].reset_index(drop=True)
    original_row_ids = original_row_ids[valid_rows]

    y: pd.Series | None = None
    if config.target_path.exists():
        target_frame = load_numeric_csv(config.target_path)
        if target_frame.shape[1] != 1:
            raise ValueError("The target CSV must contain exactly one column.")
        if len(target_frame) != len(valid_rows):
            raise ValueError(
                "The target and the original feature matrix have different lengths."
            )
        y = target_frame.iloc[:, 0].loc[valid_rows].reset_index(drop=True)
        y.name = target_frame.columns[0]
    else:
        logging.warning(
            "Target file not found. Activity-based external interpretation "
            "will be skipped: %s",
            config.target_path,
        )

    feature_profiles = build_feature_profiles(X, config)
    feature_profiles.to_csv(
        output_paths["root"] / "global_feature_profiles.csv", index=False
    )

    return DatasetBundle(
        X=X,
        y=y,
        original_row_ids=original_row_ids,
        feature_profiles=feature_profiles,
    )


# =============================================================================
# Metrics and partition interpretation
# =============================================================================


def normalized_entropy(cluster_sizes: np.ndarray) -> float:
    sizes = np.asarray(cluster_sizes, dtype=float)
    sizes = sizes[sizes > 0]
    if len(sizes) <= 1:
        return 0.0

    probabilities = sizes / sizes.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(len(probabilities)))


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    grand_mean = float(values.mean())
    total_ss = float(np.sum((values - grand_mean) ** 2))
    if total_ss <= 0:
        return 0.0

    between_ss = 0.0
    for label in np.unique(labels):
        group = values[labels == label]
        between_ss += len(group) * float((group.mean() - grand_mean) ** 2)
    return float(between_ss / total_ss)


def safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def summarize_values(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "n_values": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "zero_fraction": float("nan"),
            "one_fraction": float("nan"),
        }

    return {
        "n_values": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "zero_fraction": float(np.mean(np.isclose(values, 0.0))),
        "one_fraction": float(np.mean(np.isclose(values, 1.0))),
    }


def compute_block_statistics(
    values: np.ndarray,
    row_labels: np.ndarray,
    column_labels: np.ndarray,
    feature_profiles: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Describe each checkerboard block on the original input matrix."""

    binary_features = feature_profiles["is_binary"].to_numpy(dtype=bool)
    sparse_features = feature_profiles["is_sparse"].to_numpy(dtype=bool)

    records: list[dict[str, Any]] = []
    total_block_sse = 0.0
    h_score_weighted_sum = 0.0
    total_volume = 0

    for row_cluster in sorted(np.unique(row_labels)):
        row_mask = row_labels == row_cluster

        for column_cluster in sorted(np.unique(column_labels)):
            column_mask = column_labels == column_cluster
            block = values[np.ix_(row_mask, column_mask)]

            block_mean = float(block.mean())
            row_means = block.mean(axis=1, keepdims=True)
            column_means = block.mean(axis=0, keepdims=True)
            residues = block - row_means - column_means + block_mean
            h_score = float(np.mean(residues**2))

            block_sse = float(np.sum((block - block_mean) ** 2))
            total_block_sse += block_sse
            h_score_weighted_sum += h_score * block.size
            total_volume += block.size

            binary_column_mask = column_mask & binary_features
            numeric_column_mask = column_mask & ~binary_features
            binary_values = values[np.ix_(row_mask, binary_column_mask)]
            numeric_values = values[np.ix_(row_mask, numeric_column_mask)]

            binary_summary = summarize_values(binary_values)
            numeric_summary = summarize_values(numeric_values)

            n_features = int(column_mask.sum())
            n_binary = int(binary_column_mask.sum())
            n_numeric = int(numeric_column_mask.sum())
            n_sparse = int((column_mask & sparse_features).sum())

            column_cluster_values = values[:, column_mask]
            column_cluster_global_mean = float(column_cluster_values.mean())
            column_cluster_global_std = float(column_cluster_values.std())
            mean_contrast = block_mean - column_cluster_global_mean
            standardized_mean_contrast = safe_fraction(
                mean_contrast, column_cluster_global_std
            )

            block_zero_fraction = float(np.mean(np.isclose(block, 0.0)))
            column_cluster_zero_fraction = float(
                np.mean(np.isclose(column_cluster_values, 0.0))
            )

            records.append(
                {
                    **metadata,
                    "row_cluster": int(row_cluster),
                    "column_cluster": int(column_cluster),
                    "n_rows": int(row_mask.sum()),
                    "n_features": n_features,
                    "volume": int(block.size),
                    "n_binary_features": n_binary,
                    "n_numeric_features": n_numeric,
                    "binary_feature_fraction": safe_fraction(n_binary, n_features),
                    "n_sparse_features": n_sparse,
                    "sparse_feature_fraction": safe_fraction(n_sparse, n_features),
                    "block_mean": block_mean,
                    "block_std": float(block.std()),
                    "block_zero_fraction": block_zero_fraction,
                    "block_nonzero_fraction": 1.0 - block_zero_fraction,
                    "block_h_score": h_score,
                    "column_cluster_global_mean": column_cluster_global_mean,
                    "mean_contrast_to_column_cluster": mean_contrast,
                    "absolute_mean_contrast": abs(mean_contrast),
                    "standardized_mean_contrast": standardized_mean_contrast,
                    "column_cluster_global_zero_fraction": (
                        column_cluster_zero_fraction
                    ),
                    "zero_fraction_contrast_to_column_cluster": (
                        block_zero_fraction - column_cluster_zero_fraction
                    ),
                    "binary_n_values": binary_summary["n_values"],
                    "binary_mean": binary_summary["mean"],
                    "binary_std": binary_summary["std"],
                    "binary_zero_fraction": binary_summary["zero_fraction"],
                    "binary_one_fraction": binary_summary["one_fraction"],
                    "numeric_n_values": numeric_summary["n_values"],
                    "numeric_mean": numeric_summary["mean"],
                    "numeric_std": numeric_summary["std"],
                    "numeric_zero_fraction": numeric_summary["zero_fraction"],
                }
            )

    block_frame = pd.DataFrame(records)
    total_sst = float(np.sum((values - values.mean()) ** 2))
    block_r2 = (
        1.0 - total_block_sse / total_sst if total_sst > 0 else float("nan")
    )
    weighted_h_score = safe_fraction(h_score_weighted_sum, total_volume)

    aggregate = {
        "block_r2": float(block_r2),
        "weighted_h_score": float(weighted_h_score),
        "within_block_sse": float(total_block_sse),
        "total_sst": float(total_sst),
        "block_mean_range": float(
            block_frame["block_mean"].max() - block_frame["block_mean"].min()
        ),
        "block_zero_fraction_range": float(
            block_frame["block_zero_fraction"].max()
            - block_frame["block_zero_fraction"].min()
        ),
    }
    return block_frame, aggregate


def compute_column_cluster_statistics(
    column_labels: np.ndarray,
    feature_profiles: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    assignments = feature_profiles.copy()
    assignments["column_cluster"] = column_labels

    records: list[dict[str, Any]] = []
    for column_cluster, group in assignments.groupby("column_cluster", sort=True):
        n_features = len(group)
        n_binary = int(group["is_binary"].sum())
        n_sparse = int(group["is_sparse"].sum())

        records.append(
            {
                **metadata,
                "column_cluster": int(column_cluster),
                "n_features": int(n_features),
                "feature_cluster_fraction": safe_fraction(
                    n_features, len(assignments)
                ),
                "n_binary_features": n_binary,
                "n_numeric_features": int(n_features - n_binary),
                "binary_feature_fraction": safe_fraction(n_binary, n_features),
                "numeric_feature_fraction": safe_fraction(
                    n_features - n_binary, n_features
                ),
                "n_sparse_features": n_sparse,
                "sparse_feature_fraction": safe_fraction(n_sparse, n_features),
                "mean_feature_zero_fraction": float(group["zero_fraction"].mean()),
                "median_feature_zero_fraction": float(
                    group["zero_fraction"].median()
                ),
                "minimum_feature_zero_fraction": float(
                    group["zero_fraction"].min()
                ),
                "maximum_feature_zero_fraction": float(
                    group["zero_fraction"].max()
                ),
                "mean_feature_value": float(group["mean"].mean()),
                "mean_feature_std": float(group["std"].mean()),
                "median_unique_values": float(group["n_unique"].median()),
            }
        )

    return pd.DataFrame(records)


def compute_activity_association(
    row_labels: np.ndarray,
    y: pd.Series | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if y is None:
        return pd.DataFrame(), pd.DataFrame(), {}

    contingency = pd.crosstab(
        pd.Series(row_labels, name="row_cluster"),
        pd.Series(y.to_numpy(), name=str(y.name or "Activity")),
        dropna=False,
    )

    chi2_result = chi2_contingency(contingency.to_numpy(), correction=False)
    chi2 = float(chi2_result.statistic)
    p_value = float(chi2_result.pvalue)
    degrees_of_freedom = int(chi2_result.dof)
    expected = np.asarray(chi2_result.expected_freq, dtype=float)

    n = float(contingency.to_numpy().sum())
    min_dimension = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = (
        float(np.sqrt(chi2 / (n * min_dimension)))
        if n > 0 and min_dimension > 0
        else float("nan")
    )
    target_nmi = float(
        normalized_mutual_info_score(y.to_numpy(), row_labels)
    )

    expected_frame = pd.DataFrame(
        expected,
        index=contingency.index,
        columns=contingency.columns,
    )
    standardized_residuals = (contingency - expected_frame) / np.sqrt(
        expected_frame
    )

    long_records: list[dict[str, Any]] = []
    for row_cluster in contingency.index:
        row_total = int(contingency.loc[row_cluster].sum())
        for target_value in contingency.columns:
            observed = int(contingency.loc[row_cluster, target_value])
            long_records.append(
                {
                    "row_cluster": int(row_cluster),
                    "target_value": target_value,
                    "observed_count": observed,
                    "expected_count": float(expected_frame.loc[row_cluster, target_value]),
                    "standardized_residual": float(
                        standardized_residuals.loc[row_cluster, target_value]
                    ),
                    "within_cluster_fraction": safe_fraction(observed, row_total),
                }
            )

    association_metrics = {
        "activity_chi2": chi2,
        "activity_chi2_p_value": p_value,
        "activity_chi2_dof": degrees_of_freedom,
        "activity_cramers_v": cramers_v,
        "activity_nmi": target_nmi,
        "minimum_expected_frequency": float(expected.min()),
    }
    return contingency, pd.DataFrame(long_records), association_metrics


def compute_partition_metrics(
    values: np.ndarray,
    row_labels: np.ndarray,
    column_labels: np.ndarray,
    feature_profiles: pd.DataFrame,
    y: pd.Series | None,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    row_sizes = np.bincount(
        row_labels, minlength=int(metadata["n_row_clusters"])
    )
    column_sizes = np.bincount(
        column_labels, minlength=int(metadata["n_column_clusters"])
    )

    blocks, block_metrics = compute_block_statistics(
        values,
        row_labels,
        column_labels,
        feature_profiles,
        metadata,
    )
    columns = compute_column_cluster_statistics(
        column_labels, feature_profiles, metadata
    )

    binary_flag = feature_profiles["is_binary"].astype(int).to_numpy()
    sparse_flag = feature_profiles["is_sparse"].astype(int).to_numpy()
    zero_fraction = feature_profiles["zero_fraction"].to_numpy(dtype=float)
    _, _, activity_metrics = compute_activity_association(row_labels, y)

    metrics: dict[str, Any] = {
        **metadata,
        **block_metrics,
        "row_cluster_entropy": normalized_entropy(row_sizes),
        "smallest_row_cluster": int(row_sizes.min()),
        "largest_row_cluster": int(row_sizes.max()),
        "smallest_row_cluster_fraction": safe_fraction(
            row_sizes.min(), row_sizes.sum()
        ),
        "largest_row_cluster_fraction": safe_fraction(
            row_sizes.max(), row_sizes.sum()
        ),
        "column_cluster_entropy": normalized_entropy(column_sizes),
        "smallest_column_cluster": int(column_sizes.min()),
        "largest_column_cluster": int(column_sizes.max()),
        "smallest_column_cluster_fraction": safe_fraction(
            column_sizes.min(), column_sizes.sum()
        ),
        "largest_column_cluster_fraction": safe_fraction(
            column_sizes.max(), column_sizes.sum()
        ),
        "feature_type_nmi": float(
            normalized_mutual_info_score(binary_flag, column_labels)
        ),
        "sparse_flag_nmi": float(
            normalized_mutual_info_score(sparse_flag, column_labels)
        ),
        "feature_zero_fraction_eta2": eta_squared(
            zero_fraction, column_labels
        ),
        **activity_metrics,
    }
    return metrics, blocks, columns


# =============================================================================
# Model fitting
# =============================================================================


def fit_partition(
    bundle: DatasetBundle,
    config: Config,
    method: str,
    n_row_clusters: int,
    n_column_clusters: int,
    seed: int,
) -> PartitionResult:
    metadata = model_metadata(
        method, n_row_clusters, n_column_clusters, seed
    )

    model = SpectralBiclustering(
        n_clusters=(n_row_clusters, n_column_clusters),
        method=method,
        n_components=config.n_components,
        n_best=config.n_best,
        svd_method=config.svd_method,
        n_init=config.n_init,
        random_state=seed,
    )
    model.fit(bundle.values)

    row_labels = model.row_labels_.astype(int, copy=True)
    column_labels = model.column_labels_.astype(int, copy=True)
    metrics, blocks, columns = compute_partition_metrics(
        bundle.values,
        row_labels,
        column_labels,
        bundle.feature_profiles,
        bundle.y,
        metadata,
    )

    return PartitionResult(
        method=method,
        n_row_clusters=n_row_clusters,
        n_column_clusters=n_column_clusters,
        seed=seed,
        row_labels=row_labels,
        column_labels=column_labels,
        metrics=metrics,
        block_statistics=blocks,
        column_cluster_statistics=columns,
    )


# =============================================================================
# Phase 1: screening and candidate selection
# =============================================================================


def run_screening(
    bundle: DatasetBundle,
    config: Config,
    output_paths: dict[str, Path],
) -> pd.DataFrame:
    configurations = list(
        product(
            config.methods,
            config.row_cluster_values,
            config.column_cluster_values,
            config.screening_seeds,
        )
    )

    run_records: list[dict[str, Any]] = []
    all_blocks: list[pd.DataFrame] = []
    all_column_stats: list[pd.DataFrame] = []
    all_row_assignments: list[pd.DataFrame] = []
    all_column_assignments: list[pd.DataFrame] = []

    for run_index, (method, n_rows, n_columns, seed) in enumerate(
        configurations, start=1
    ):
        model_id = model_metadata(method, n_rows, n_columns, seed)["model_id"]
        logging.info(
            "Screening [%d/%d] %s",
            run_index,
            len(configurations),
            model_id,
        )

        try:
            result = fit_partition(
                bundle, config, method, n_rows, n_columns, seed
            )
            run_records.append({**result.metrics, "status": "ok", "error": ""})
            all_column_stats.append(result.column_cluster_statistics)

            if config.save_screening_block_statistics:
                all_blocks.append(result.block_statistics)

            if config.save_screening_assignments:
                row_assignment = pd.DataFrame(
                    {
                        **model_metadata(method, n_rows, n_columns, seed),
                        "original_row_id": bundle.original_row_ids,
                        "row_cluster": result.row_labels,
                    }
                )
                if bundle.y is not None:
                    row_assignment[str(bundle.y.name)] = bundle.y.to_numpy()
                all_row_assignments.append(row_assignment)

                column_assignment = bundle.feature_profiles.copy()
                for key, value in model_metadata(
                    method, n_rows, n_columns, seed
                ).items():
                    column_assignment[key] = value
                column_assignment["column_cluster"] = result.column_labels
                all_column_assignments.append(column_assignment)

        except Exception as error:  # Continue the grid after one failed fit.
            logging.exception("Failed configuration: %s", model_id)
            run_records.append(
                {
                    **model_metadata(method, n_rows, n_columns, seed),
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    runs = pd.DataFrame(run_records)
    runs.to_csv(output_paths["screening"] / "screening_runs.csv", index=False)

    if all_blocks:
        pd.concat(all_blocks, ignore_index=True).to_csv(
            output_paths["screening"] / "screening_block_statistics.csv",
            index=False,
        )
    if all_column_stats:
        pd.concat(all_column_stats, ignore_index=True).to_csv(
            output_paths["screening"]
            / "screening_column_cluster_statistics.csv",
            index=False,
        )
    if all_row_assignments:
        pd.concat(all_row_assignments, ignore_index=True).to_csv(
            output_paths["screening"] / "screening_row_assignments.csv",
            index=False,
        )
    if all_column_assignments:
        pd.concat(all_column_assignments, ignore_index=True).to_csv(
            output_paths["screening"] / "screening_column_assignments.csv",
            index=False,
        )

    successful = runs[runs["status"] == "ok"].copy()
    if successful.empty:
        raise RuntimeError(
            "All screening configurations failed. Inspect screening_runs.csv."
        )

    metric_columns = [
        "block_r2",
        "weighted_h_score",
        "row_cluster_entropy",
        "column_cluster_entropy",
        "smallest_row_cluster_fraction",
        "largest_row_cluster_fraction",
        "smallest_column_cluster_fraction",
        "largest_column_cluster_fraction",
        "feature_type_nmi",
        "sparse_flag_nmi",
        "feature_zero_fraction_eta2",
        "block_mean_range",
        "block_zero_fraction_range",
    ]
    activity_columns = [
        "activity_chi2_p_value",
        "activity_cramers_v",
        "activity_nmi",
    ]
    metric_columns.extend(
        column for column in activity_columns if column in successful.columns
    )

    comparison = (
        successful.groupby(
            ["method", "n_row_clusters", "n_column_clusters", "n_blocks"],
            as_index=False,
        )[metric_columns]
        .agg(["mean", "std"])
    )
    comparison.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in comparison.columns
    ]
    comparison = comparison.rename(
        columns={
            "method_": "method",
            "n_row_clusters_": "n_row_clusters",
            "n_column_clusters_": "n_column_clusters",
            "n_blocks_": "n_blocks",
        }
    )
    comparison = comparison.sort_values(
        ["method", "n_row_clusters", "n_column_clusters"]
    ).reset_index(drop=True)
    comparison.to_csv(
        output_paths["screening"] / "screening_comparison.csv", index=False
    )

    save_screening_plots(comparison, output_paths["screening"], config)
    return comparison


def select_candidates(
    comparison: pd.DataFrame,
    config: Config,
    output_paths: dict[str, Path],
) -> pd.DataFrame:
    if config.manual_candidates:
        records: list[pd.DataFrame] = []
        for rank, (method, n_rows, n_columns) in enumerate(
            config.manual_candidates, start=1
        ):
            match = comparison[
                (comparison["method"] == method)
                & (comparison["n_row_clusters"] == n_rows)
                & (comparison["n_column_clusters"] == n_columns)
            ].copy()
            if match.empty:
                raise ValueError(
                    "Manual candidate not present in the screening grid: "
                    f"{(method, n_rows, n_columns)}"
                )
            match["candidate_rank"] = rank
            match["selection_reason"] = "manual"
            records.append(match)
        candidates = pd.concat(records, ignore_index=True)
    else:
        eligible_parts: list[pd.DataFrame] = []

        for method, method_data in comparison.groupby("method", sort=False):
            best_r2 = float(method_data["block_r2_mean"].max())
            eligible = method_data[
                (method_data["block_r2_mean"] >= best_r2 - config.r2_tolerance)
                & (
                    method_data["row_cluster_entropy_mean"]
                    >= config.minimum_row_entropy
                )
                & (
                    method_data["column_cluster_entropy_mean"]
                    >= config.minimum_column_entropy
                )
            ].copy()

            if eligible.empty:
                eligible = method_data.nlargest(
                    config.number_of_candidates, "block_r2_mean"
                ).copy()

            eligible["best_r2_for_method"] = best_r2
            eligible["r2_gap_from_best"] = (
                best_r2 - eligible["block_r2_mean"]
            )
            eligible_parts.append(eligible)

        pool = pd.concat(eligible_parts, ignore_index=True)
        pool = pool.sort_values(
            [
                "n_blocks",
                "block_r2_mean",
                "weighted_h_score_mean",
                "row_cluster_entropy_mean",
            ],
            ascending=[True, False, True, False],
        )
        candidates = pool.head(config.number_of_candidates).copy()
        candidates["candidate_rank"] = np.arange(1, len(candidates) + 1)
        candidates["selection_reason"] = (
            f"R² within {config.r2_tolerance:.3f} of method maximum; "
            "then minimum number of blocks"
        )

    candidates.to_csv(
        output_paths["candidates"] / "selected_candidates.csv", index=False
    )
    return candidates


# =============================================================================
# Phase 2: stability and final model selection
# =============================================================================


def pairwise_ari_records(
    label_sets: dict[int, np.ndarray], label_kind: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for first_seed, second_seed in combinations(sorted(label_sets), 2):
        records.append(
            {
                "label_kind": label_kind,
                "first_seed": first_seed,
                "second_seed": second_seed,
                "ari": float(
                    adjusted_rand_score(
                        label_sets[first_seed], label_sets[second_seed]
                    )
                ),
            }
        )
    return records


def seed_centrality(
    row_labels: dict[int, np.ndarray],
    column_labels: dict[int, np.ndarray],
) -> pd.DataFrame:
    """Select a representative seed: the partition medoid under ARI."""
    seeds = sorted(row_labels)
    records: list[dict[str, Any]] = []

    for seed in seeds:
        other_seeds = [other for other in seeds if other != seed]
        if not other_seeds:
            row_mean = float("nan")
            column_mean = float("nan")
            combined = float("nan")
        else:
            row_scores = [
                adjusted_rand_score(row_labels[seed], row_labels[other])
                for other in other_seeds
            ]
            column_scores = [
                adjusted_rand_score(
                    column_labels[seed], column_labels[other]
                )
                for other in other_seeds
            ]
            row_mean = float(np.mean(row_scores))
            column_mean = float(np.mean(column_scores))
            combined = float(np.mean([row_mean, column_mean]))

        records.append(
            {
                "seed": seed,
                "mean_row_ari_to_other_seeds": row_mean,
                "mean_column_ari_to_other_seeds": column_mean,
                "mean_combined_ari_to_other_seeds": combined,
            }
        )

    return pd.DataFrame(records).sort_values(
        "mean_combined_ari_to_other_seeds", ascending=False
    )


def run_stability_analysis(
    candidates: pd.DataFrame,
    bundle: DatasetBundle,
    config: Config,
    output_paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int, int], int]]:
    run_records: list[dict[str, Any]] = []
    pairwise_records: list[dict[str, Any]] = []
    centrality_records: list[pd.DataFrame] = []
    representative_seeds: dict[tuple[str, int, int], int] = {}

    for candidate in candidates.itertuples(index=False):
        method = str(candidate.method)
        n_rows = int(candidate.n_row_clusters)
        n_columns = int(candidate.n_column_clusters)
        key = (method, n_rows, n_columns)

        row_label_sets: dict[int, np.ndarray] = {}
        column_label_sets: dict[int, np.ndarray] = {}

        for seed in config.stability_seeds:
            logging.info(
                "Stability fit: %s r=%d c=%d seed=%d",
                method,
                n_rows,
                n_columns,
                seed,
            )
            try:
                result = fit_partition(
                    bundle, config, method, n_rows, n_columns, seed
                )
                row_label_sets[seed] = result.row_labels
                column_label_sets[seed] = result.column_labels
                run_records.append(
                    {
                        **result.metrics,
                        "candidate_rank": int(candidate.candidate_rank),
                        "status": "ok",
                        "error": "",
                    }
                )
            except Exception as error:
                logging.exception(
                    "Stability fit failed: %s r=%d c=%d seed=%d",
                    method,
                    n_rows,
                    n_columns,
                    seed,
                )
                run_records.append(
                    {
                        **model_metadata(method, n_rows, n_columns, seed),
                        "candidate_rank": int(candidate.candidate_rank),
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

        if not row_label_sets:
            continue

        row_pairs = pairwise_ari_records(row_label_sets, "row")
        column_pairs = pairwise_ari_records(column_label_sets, "column")
        for record in row_pairs + column_pairs:
            pairwise_records.append(
                {
                    "method": method,
                    "n_row_clusters": n_rows,
                    "n_column_clusters": n_columns,
                    **record,
                }
            )

        centrality = seed_centrality(row_label_sets, column_label_sets)
        centrality.insert(0, "method", method)
        centrality.insert(1, "n_row_clusters", n_rows)
        centrality.insert(2, "n_column_clusters", n_columns)
        centrality_records.append(centrality)
        representative_seeds[key] = int(centrality.iloc[0]["seed"])

    stability_runs = pd.DataFrame(run_records)
    stability_runs.to_csv(
        output_paths["candidates"] / "stability_runs.csv", index=False
    )

    pairwise = pd.DataFrame(pairwise_records)
    pairwise.to_csv(
        output_paths["candidates"] / "pairwise_ari.csv", index=False
    )

    if centrality_records:
        pd.concat(centrality_records, ignore_index=True).to_csv(
            output_paths["candidates"] / "representative_seed_scores.csv",
            index=False,
        )

    successful = stability_runs[stability_runs["status"] == "ok"].copy()
    if successful.empty:
        raise RuntimeError("All candidate stability fits failed.")

    metric_columns = [
        "block_r2",
        "weighted_h_score",
        "row_cluster_entropy",
        "column_cluster_entropy",
        "largest_row_cluster_fraction",
        "largest_column_cluster_fraction",
        "feature_type_nmi",
        "feature_zero_fraction_eta2",
    ]
    stability_summary = (
        successful.groupby(
            ["method", "n_row_clusters", "n_column_clusters"], as_index=False
        )[metric_columns]
        .agg(["mean", "std"])
    )
    stability_summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in stability_summary.columns
    ]
    stability_summary = stability_summary.rename(
        columns={
            "method_": "method",
            "n_row_clusters_": "n_row_clusters",
            "n_column_clusters_": "n_column_clusters",
        }
    )

    if not pairwise.empty:
        ari_summary = (
            pairwise.groupby(
                [
                    "method",
                    "n_row_clusters",
                    "n_column_clusters",
                    "label_kind",
                ],
                as_index=False,
            )["ari"]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        row_ari = ari_summary[ari_summary["label_kind"] == "row"].drop(
            columns="label_kind"
        )
        column_ari = ari_summary[
            ari_summary["label_kind"] == "column"
        ].drop(columns="label_kind")
        row_ari = row_ari.rename(
            columns={
                "mean": "row_ari_mean",
                "std": "row_ari_std",
                "min": "row_ari_min",
                "max": "row_ari_max",
            }
        )
        column_ari = column_ari.rename(
            columns={
                "mean": "column_ari_mean",
                "std": "column_ari_std",
                "min": "column_ari_min",
                "max": "column_ari_max",
            }
        )
        stability_summary = stability_summary.merge(
            row_ari,
            on=["method", "n_row_clusters", "n_column_clusters"],
            how="left",
        ).merge(
            column_ari,
            on=["method", "n_row_clusters", "n_column_clusters"],
            how="left",
        )
    else:
        stability_summary["row_ari_mean"] = np.nan
        stability_summary["column_ari_mean"] = np.nan

    stability_summary["minimum_mean_ari"] = stability_summary[
        ["row_ari_mean", "column_ari_mean"]
    ].min(axis=1)
    stability_summary["meets_stability_threshold"] = (
        (stability_summary["row_ari_mean"] >= config.minimum_row_ari)
        & (
            stability_summary["column_ari_mean"]
            >= config.minimum_column_ari
        )
    )

    stability_summary = candidates[
        ["method", "n_row_clusters", "n_column_clusters", "candidate_rank"]
    ].merge(
        stability_summary,
        on=["method", "n_row_clusters", "n_column_clusters"],
        how="left",
    )
    stability_summary.to_csv(
        output_paths["candidates"] / "stability_summary.csv", index=False
    )
    return stability_summary, stability_runs, representative_seeds


def choose_final_configuration(
    stability_summary: pd.DataFrame,
    representative_seeds: dict[tuple[str, int, int], int],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    stable = stability_summary[
        stability_summary["meets_stability_threshold"] == True  # noqa: E712
    ].copy()

    if not stable.empty:
        chosen_row = stable.sort_values(
            ["candidate_rank", "minimum_mean_ari"],
            ascending=[True, False],
        ).iloc[0]
        selection_reason = (
            "First parsimonious candidate satisfying both ARI thresholds"
        )
    else:
        chosen_row = stability_summary.sort_values(
            ["minimum_mean_ari", "candidate_rank"],
            ascending=[False, True],
        ).iloc[0]
        selection_reason = (
            "No candidate satisfied both ARI thresholds; selected the most "
            "stable candidate"
        )

    key = (
        str(chosen_row["method"]),
        int(chosen_row["n_row_clusters"]),
        int(chosen_row["n_column_clusters"]),
    )
    chosen = {
        "method": key[0],
        "n_row_clusters": key[1],
        "n_column_clusters": key[2],
        "representative_seed": int(representative_seeds[key]),
        "selection_reason": selection_reason,
        "candidate_rank": int(chosen_row["candidate_rank"]),
        "row_ari_mean": float(chosen_row["row_ari_mean"]),
        "column_ari_mean": float(chosen_row["column_ari_mean"]),
    }
    save_json(chosen, output_paths["candidates"] / "final_selection.json")
    return chosen


# =============================================================================
# Phase 3: detailed final-model interpretation
# =============================================================================


def final_row_cluster_summary(
    row_labels: np.ndarray,
    bundle: DatasetBundle,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "original_row_id": bundle.original_row_ids,
            "row_cluster": row_labels,
        }
    )
    if bundle.y is not None:
        frame[str(bundle.y.name)] = bundle.y.to_numpy()

    records: list[dict[str, Any]] = []
    for row_cluster, group in frame.groupby("row_cluster", sort=True):
        record: dict[str, Any] = {
            "row_cluster": int(row_cluster),
            "n_rows": int(len(group)),
            "row_fraction": safe_fraction(len(group), len(frame)),
        }
        if bundle.y is not None:
            target_name = str(bundle.y.name)
            counts = group[target_name].value_counts(dropna=False)
            for target_value, count in counts.items():
                record[f"target_{target_value}_count"] = int(count)
                record[f"target_{target_value}_fraction"] = safe_fraction(
                    count, len(group)
                )
        records.append(record)
    return pd.DataFrame(records)


def representative_features(
    values: np.ndarray,
    column_labels: np.ndarray,
    feature_profiles: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """
    Rank features by RMS distance from the mean feature profile of their
    column cluster. Smaller distance means more representative of the cluster.
    """
    records: list[dict[str, Any]] = []

    for column_cluster in sorted(np.unique(column_labels)):
        indices = np.flatnonzero(column_labels == column_cluster)
        cluster_values = values[:, indices]
        centroid = cluster_values.mean(axis=1, keepdims=True)
        distances = np.sqrt(np.mean((cluster_values - centroid) ** 2, axis=0))
        ranking = np.argsort(distances)[: min(top_n, len(indices))]

        for rank, local_index in enumerate(ranking, start=1):
            feature_index = int(indices[local_index])
            profile = feature_profiles.iloc[feature_index]
            records.append(
                {
                    "column_cluster": int(column_cluster),
                    "representative_rank": rank,
                    "feature": profile["feature"],
                    "feature_index": feature_index,
                    "distance_to_cluster_centroid": float(
                        distances[local_index]
                    ),
                    "feature_type": profile["feature_type"],
                    "zero_fraction": float(profile["zero_fraction"]),
                    "mean": float(profile["mean"]),
                    "std": float(profile["std"]),
                }
            )

    return pd.DataFrame(records)


def save_final_outputs(
    final_result: PartitionResult,
    bundle: DatasetBundle,
    config: Config,
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    final_dir = output_paths["final"]

    row_assignments = pd.DataFrame(
        {
            "original_row_id": bundle.original_row_ids,
            "row_cluster": final_result.row_labels,
        }
    )
    if bundle.y is not None:
        row_assignments[str(bundle.y.name)] = bundle.y.to_numpy()
    row_assignments.to_csv(final_dir / "row_assignments.csv", index=False)

    column_assignments = bundle.feature_profiles.copy()
    column_assignments["column_cluster"] = final_result.column_labels
    column_assignments.to_csv(
        final_dir / "column_assignments.csv", index=False
    )

    row_summary = final_row_cluster_summary(final_result.row_labels, bundle)
    row_summary.to_csv(final_dir / "row_cluster_summary.csv", index=False)

    final_result.column_cluster_statistics.to_csv(
        final_dir / "column_cluster_summary.csv", index=False
    )
    final_result.block_statistics.to_csv(
        final_dir / "block_statistics.csv", index=False
    )

    interesting_blocks = final_result.block_statistics.sort_values(
        ["absolute_mean_contrast", "block_h_score"],
        ascending=[False, True],
    ).head(config.top_interesting_blocks)
    interesting_blocks.to_csv(
        final_dir / "most_distinctive_blocks.csv", index=False
    )

    representatives = representative_features(
        bundle.values,
        final_result.column_labels,
        bundle.feature_profiles,
        config.representative_features_per_cluster,
    )
    representatives.to_csv(
        final_dir / "representative_features_by_column_cluster.csv",
        index=False,
    )

    contingency, activity_long, activity_metrics = compute_activity_association(
        final_result.row_labels, bundle.y
    )
    if not contingency.empty:
        contingency.to_csv(final_dir / "activity_contingency_table.csv")
        activity_long.to_csv(
            final_dir / "activity_cluster_residuals.csv", index=False
        )
        pd.DataFrame([activity_metrics]).to_csv(
            final_dir / "activity_association_summary.csv", index=False
        )

    save_final_plots(final_result, bundle, row_summary, output_paths, config)

    final_summary = {
        **final_result.metrics,
        **activity_metrics,
        "interpretation_warning": (
            "Feature clusters must not be interpreted biologically from "
            "sparsity or binary/numeric composition alone."
        ),
    }
    save_json(final_summary, final_dir / "final_model_metrics.json")
    return final_summary


# =============================================================================
# Phase 4: column-wise permutation null test
# =============================================================================


def permute_within_columns(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    permuted = np.empty_like(values)
    for column_index in range(values.shape[1]):
        permuted[:, column_index] = rng.permutation(values[:, column_index])
    return permuted


def run_null_test(
    final_selection: dict[str, Any],
    observed_result: PartitionResult,
    bundle: DatasetBundle,
    config: Config,
    output_paths: dict[str, Path],
) -> dict[str, float]:
    if config.null_permutations <= 0:
        logging.info("Null test skipped.")
        return {}

    rng = np.random.default_rng(config.null_seed)
    records: list[dict[str, Any]] = []

    for permutation_index in range(1, config.null_permutations + 1):
        logging.info(
            "Null permutation [%d/%d]",
            permutation_index,
            config.null_permutations,
        )
        permuted_values = permute_within_columns(bundle.values, rng)

        model = SpectralBiclustering(
            n_clusters=(
                int(final_selection["n_row_clusters"]),
                int(final_selection["n_column_clusters"]),
            ),
            method=str(final_selection["method"]),
            n_components=config.n_components,
            n_best=config.n_best,
            svd_method=config.svd_method,
            n_init=config.n_init,
            random_state=int(final_selection["representative_seed"]),
        )
        model.fit(permuted_values)

        metadata = model_metadata(
            str(final_selection["method"]),
            int(final_selection["n_row_clusters"]),
            int(final_selection["n_column_clusters"]),
            int(final_selection["representative_seed"]),
        )
        _, aggregate = compute_block_statistics(
            permuted_values,
            model.row_labels_.astype(int),
            model.column_labels_.astype(int),
            bundle.feature_profiles,
            metadata,
        )
        records.append(
            {
                "permutation": permutation_index,
                "block_r2": aggregate["block_r2"],
                "weighted_h_score": aggregate["weighted_h_score"],
            }
        )

    null_frame = pd.DataFrame(records)
    null_frame.to_csv(
        output_paths["null"] / "null_distribution.csv", index=False
    )

    observed_r2 = float(observed_result.metrics["block_r2"])
    null_r2 = null_frame["block_r2"].to_numpy(dtype=float)
    p_value = float((1 + np.sum(null_r2 >= observed_r2)) / (len(null_r2) + 1))
    null_std = float(null_r2.std(ddof=1)) if len(null_r2) > 1 else float("nan")
    z_score = (
        float((observed_r2 - null_r2.mean()) / null_std)
        if np.isfinite(null_std) and null_std > 0
        else float("nan")
    )

    summary = {
        "observed_block_r2": observed_r2,
        "null_block_r2_mean": float(null_r2.mean()),
        "null_block_r2_std": null_std,
        "null_block_r2_min": float(null_r2.min()),
        "null_block_r2_max": float(null_r2.max()),
        "monte_carlo_p_value": p_value,
        "observed_r2_z_score": z_score,
        "n_permutations": int(len(null_r2)),
    }
    save_json(summary, output_paths["null"] / "null_test_summary.json")
    save_null_plot(null_frame, observed_r2, output_paths["null"], config)
    return summary


# =============================================================================
# Optional comparison with an existing delta-biclustering report
# =============================================================================


def find_delta_summary(delta_report_dir: Path) -> tuple[Path, pd.Series] | None:
    if not delta_report_dir.exists():
        return None

    required = {"h_score", "n_rows", "n_features"}
    for csv_path in sorted(delta_report_dir.rglob("*.csv")):
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            continue
        if not frame.empty and required.issubset(frame.columns):
            return csv_path, frame.iloc[0]
    return None


def save_method_comparison(
    final_result: PartitionResult,
    config: Config,
    output_paths: dict[str, Path],
) -> None:
    records: list[dict[str, Any]] = [
        {
            "method": "spectral_biclustering",
            "scope": "global checkerboard partition",
            "n_rows": len(final_result.row_labels),
            "n_features": len(final_result.column_labels),
            "n_row_clusters": final_result.n_row_clusters,
            "n_column_clusters": final_result.n_column_clusters,
            "n_blocks": final_result.n_row_clusters
            * final_result.n_column_clusters,
            "block_r2": final_result.metrics["block_r2"],
            "weighted_h_score": final_result.metrics["weighted_h_score"],
        }
    ]

    delta = find_delta_summary(config.delta_report_dir)
    if delta is not None:
        path, row = delta
        records.append(
            {
                "method": "delta_biclustering",
                "scope": "local coherent submatrix",
                "n_rows": row.get("n_rows", np.nan),
                "n_features": row.get("n_features", np.nan),
                "n_row_clusters": np.nan,
                "n_column_clusters": np.nan,
                "n_blocks": 1,
                "block_r2": np.nan,
                "weighted_h_score": row.get("h_score", np.nan),
                "source_file": str(path),
            }
        )

    pd.DataFrame(records).to_csv(
        output_paths["comparison"] / "spectral_vs_delta_summary.csv",
        index=False,
    )

    note = (
        "The metrics of the two methods are not optimization-equivalent.\n"
        "Spectral biclustering partitions all rows and columns into a global "
        "checkerboard.\n"
        "Delta biclustering extracts one or more local submatrices satisfying "
        "a direct coherence constraint.\n"
        "Therefore block R² and delta H-score must not be used as if they were "
        "the same objective.\n"
    )
    (output_paths["comparison"] / "interpretation_note.txt").write_text(
        note, encoding="utf-8"
    )


# =============================================================================
# Plotting
# =============================================================================


def save_metric_grid(
    comparison: pd.DataFrame,
    method: str,
    metric: str,
    title: str,
    output_path: Path,
    config: Config,
    value_format: str = ".3f",
) -> None:
    method_data = comparison[comparison["method"] == method]
    matrix = method_data.pivot(
        index="n_row_clusters",
        columns="n_column_clusters",
        values=metric,
    ).sort_index().sort_index(axis=1)
    if matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest")
    fig.colorbar(image, ax=ax, label=metric)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix.iloc[row_index, column_index]
            label = "NA" if pd.isna(value) else format(value, value_format)
            ax.text(column_index, row_index, label, ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel("Number of column clusters")
    ax.set_ylabel("Number of row clusters")
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.plot_dpi, bbox_inches="tight")
    plt.close(fig)


def save_screening_plots(
    comparison: pd.DataFrame,
    output_dir: Path,
    config: Config,
) -> None:
    metrics = [
        ("block_r2_mean", "Block R²", ".3f"),
        ("weighted_h_score_mean", "Weighted H-score", ".5f"),
        ("row_cluster_entropy_mean", "Row-cluster entropy", ".3f"),
        ("column_cluster_entropy_mean", "Column-cluster entropy", ".3f"),
        (
            "feature_zero_fraction_eta2_mean",
            "Feature sparsity explained (eta²)",
            ".3f",
        ),
        (
            "largest_column_cluster_fraction_mean",
            "Largest column-cluster fraction",
            ".3f",
        ),
    ]

    for method in comparison["method"].unique():
        for metric, title, value_format in metrics:
            if metric not in comparison.columns:
                continue
            save_metric_grid(
                comparison,
                method,
                metric,
                f"{title} — {method}",
                output_dir / f"grid_{metric}_{method}.png",
                config,
                value_format,
            )


def save_reordered_matrix_plot(
    values: np.ndarray,
    row_labels: np.ndarray,
    column_labels: np.ndarray,
    output_path: Path,
    config: Config,
) -> None:
    row_order = np.argsort(row_labels, kind="stable")
    column_order = np.argsort(column_labels, kind="stable")
    reordered = values[row_order][:, column_order]
    sorted_rows = row_labels[row_order]
    sorted_columns = column_labels[column_order]

    fig, ax = plt.subplots(figsize=(16, 10))
    image = ax.imshow(reordered, aspect="auto", interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Feature value")
    for boundary in np.flatnonzero(np.diff(sorted_rows)) + 0.5:
        ax.axhline(boundary, linewidth=0.8)
    for boundary in np.flatnonzero(np.diff(sorted_columns)) + 0.5:
        ax.axvline(boundary, linewidth=0.8)
    ax.set_title("Matrix reordered by spectral biclustering")
    ax.set_xlabel("Features ordered by column cluster")
    ax.set_ylabel("Molecules ordered by row cluster")
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.plot_dpi, bbox_inches="tight")
    plt.close(fig)


def save_block_heatmap(
    block_statistics: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    config: Config,
    value_format: str = ".3f",
) -> None:
    matrix = block_statistics.pivot(
        index="row_cluster", columns="column_cluster", values=metric
    ).sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest")
    fig.colorbar(image, ax=ax, label=metric)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix.iloc[row_index, column_index]
            label = "NA" if pd.isna(value) else format(value, value_format)
            ax.text(column_index, row_index, label, ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel("Column cluster")
    ax.set_ylabel("Row cluster")
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.plot_dpi, bbox_inches="tight")
    plt.close(fig)


def save_activity_plot(
    row_summary: pd.DataFrame,
    output_path: Path,
    config: Config,
) -> None:
    fraction_columns = [
        column
        for column in row_summary.columns
        if column.startswith("target_") and column.endswith("_fraction")
    ]
    if not fraction_columns:
        return

    x = np.arange(len(row_summary))
    bottom = np.zeros(len(row_summary), dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    for column in fraction_columns:
        values = row_summary[column].fillna(0.0).to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=column)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(row_summary["row_cluster"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Row cluster")
    ax.set_ylabel("Within-cluster target fraction")
    ax.set_title("Target distribution across row clusters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.plot_dpi, bbox_inches="tight")
    plt.close(fig)


def save_feature_composition_plot(
    column_summary: pd.DataFrame,
    output_path: Path,
    config: Config,
) -> None:
    x = np.arange(len(column_summary))
    binary = column_summary["n_binary_features"].to_numpy(dtype=float)
    numeric = column_summary["n_numeric_features"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, binary, label="Binary")
    ax.bar(x, numeric, bottom=binary, label="Numeric")
    ax.set_xticks(x)
    ax.set_xticklabels(column_summary["column_cluster"])
    ax.set_xlabel("Column cluster")
    ax.set_ylabel("Number of features")
    ax.set_title("Feature-type composition by column cluster")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.plot_dpi, bbox_inches="tight")
    plt.close(fig)


def save_sparsity_boxplot(
    column_labels: np.ndarray,
    feature_profiles: pd.DataFrame,
    output_path: Path,
    config: Config,
) -> None:
    groups = [
        feature_profiles.loc[column_labels == cluster, "zero_fraction"].to_numpy()
        for cluster in sorted(np.unique(column_labels))
    ]
    labels = [str(cluster) for cluster in sorted(np.unique(column_labels))]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(groups, tick_labels=labels)
    ax.set_xlabel("Column cluster")
    ax.set_ylabel("Feature zero fraction")
    ax.set_title("Feature sparsity by column cluster")
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.plot_dpi, bbox_inches="tight")
    plt.close(fig)


def save_final_plots(
    final_result: PartitionResult,
    bundle: DatasetBundle,
    row_summary: pd.DataFrame,
    output_paths: dict[str, Path],
    config: Config,
) -> None:
    final_dir = output_paths["final"]
    save_reordered_matrix_plot(
        bundle.values,
        final_result.row_labels,
        final_result.column_labels,
        final_dir / "reordered_matrix.png",
        config,
    )
    save_block_heatmap(
        final_result.block_statistics,
        "block_mean",
        "Mean value by checkerboard block",
        final_dir / "block_mean_heatmap.png",
        config,
    )
    save_block_heatmap(
        final_result.block_statistics,
        "block_zero_fraction",
        "Zero fraction by checkerboard block",
        final_dir / "block_zero_fraction_heatmap.png",
        config,
    )
    save_block_heatmap(
        final_result.block_statistics,
        "block_h_score",
        "H-score by checkerboard block",
        final_dir / "block_h_score_heatmap.png",
        config,
        value_format=".5f",
    )
    save_block_heatmap(
        final_result.block_statistics,
        "mean_contrast_to_column_cluster",
        "Block mean contrast relative to its feature cluster",
        final_dir / "block_mean_contrast_heatmap.png",
        config,
    )
    save_activity_plot(
        row_summary, final_dir / "activity_distribution.png", config
    )
    save_feature_composition_plot(
        final_result.column_cluster_statistics,
        final_dir / "feature_type_composition.png",
        config,
    )
    save_sparsity_boxplot(
        final_result.column_labels,
        bundle.feature_profiles,
        final_dir / "feature_sparsity_by_column_cluster.png",
        config,
    )


def save_null_plot(
    null_frame: pd.DataFrame,
    observed_r2: float,
    output_dir: Path,
    config: Config,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_frame["block_r2"], bins="auto")
    ax.axvline(observed_r2, linewidth=2, label="Observed block R²")
    ax.set_xlabel("Block R²")
    ax.set_ylabel("Frequency")
    ax.set_title("Column-wise permutation null distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "null_block_r2_distribution.png",
        dpi=config.plot_dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# Human-readable summary
# =============================================================================


def write_interpretation_summary(
    final_selection: dict[str, Any],
    final_summary: dict[str, Any],
    null_summary: dict[str, float],
    output_path: Path,
) -> None:
    lines = [
        "SPECTRAL BICLUSTERING — AUTOMATIC ANALYSIS SUMMARY",
        "",
        "Selected configuration",
        f"- method: {final_selection['method']}",
        f"- row clusters: {final_selection['n_row_clusters']}",
        f"- column clusters: {final_selection['n_column_clusters']}",
        f"- representative seed: {final_selection['representative_seed']}",
        f"- selection rule: {final_selection['selection_reason']}",
        "",
        "Stability",
        f"- mean row ARI: {final_selection['row_ari_mean']:.4f}",
        f"- mean column ARI: {final_selection['column_ari_mean']:.4f}",
        "",
        "Global checkerboard description",
        f"- block R²: {final_summary['block_r2']:.4f}",
        f"- weighted H-score: {final_summary['weighted_h_score']:.6f}",
        f"- row-cluster entropy: {final_summary['row_cluster_entropy']:.4f}",
        f"- column-cluster entropy: {final_summary['column_cluster_entropy']:.4f}",
        f"- sparsity eta²: {final_summary['feature_zero_fraction_eta2']:.4f}",
        f"- binary/numeric NMI: {final_summary['feature_type_nmi']:.4f}",
    ]

    if "activity_cramers_v" in final_summary:
        lines.extend(
            [
                "",
                "External association with Activity",
                f"- chi-square p-value: {final_summary['activity_chi2_p_value']:.6g}",
                f"- Cramer's V: {final_summary['activity_cramers_v']:.4f}",
                f"- NMI: {final_summary['activity_nmi']:.4f}",
                "Activity was not used to fit or select the model.",
            ]
        )

    if null_summary:
        lines.extend(
            [
                "",
                "Column-wise permutation test",
                f"- null mean block R²: {null_summary['null_block_r2_mean']:.4f}",
                f"- Monte-Carlo p-value: {null_summary['monte_carlo_p_value']:.6g}",
                f"- observed R² z-score: {null_summary['observed_r2_z_score']:.4f}",
            ]
        )

    lines.extend(
        [
            "",
            "Interpretation order",
            "1. Verify ARI stability.",
            "2. Inspect row-cluster target distributions only as external validation.",
            "3. Inspect column-cluster sparsity and feature-type composition.",
            "4. Interpret blocks by contrasts within the same column cluster.",
            "5. Use the permutation test to determine whether the checkerboard is non-trivial.",
            "6. Do not equate spectral block R² with the delta-biclustering objective.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_arguments()
    config = configure_from_arguments(Config(), args)
    output_paths = setup_output(config)
    setup_logging(output_paths["root"] / "pipeline.log")

    save_json(
        {
            "configuration": asdict(config),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": plt.matplotlib.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "platform": platform.platform(),
            },
        },
        output_paths["root"] / "run_metadata.json",
    )

    bundle = load_dataset(config, output_paths)
    logging.info(
        "Dataset loaded: %d rows x %d features; value range [%.6f, %.6f]",
        bundle.X.shape[0],
        bundle.X.shape[1],
        bundle.values.min(),
        bundle.values.max(),
    )
    logging.info(
        "Feature types: %d binary, %d numeric",
        int(bundle.feature_profiles["is_binary"].sum()),
        int((~bundle.feature_profiles["is_binary"]).sum()),
    )

    screening = run_screening(bundle, config, output_paths)
    candidates = select_candidates(screening, config, output_paths)
    logging.info(
        "Selected candidates:\n%s",
        candidates[
            [
                "candidate_rank",
                "method",
                "n_row_clusters",
                "n_column_clusters",
                "block_r2_mean",
                "n_blocks",
            ]
        ].to_string(index=False),
    )

    stability_summary, _, representative_seeds = run_stability_analysis(
        candidates, bundle, config, output_paths
    )
    final_selection = choose_final_configuration(
        stability_summary, representative_seeds, output_paths
    )
    logging.info("Final selection: %s", final_selection)

    final_result = fit_partition(
        bundle=bundle,
        config=config,
        method=str(final_selection["method"]),
        n_row_clusters=int(final_selection["n_row_clusters"]),
        n_column_clusters=int(final_selection["n_column_clusters"]),
        seed=int(final_selection["representative_seed"]),
    )
    final_summary = save_final_outputs(
        final_result, bundle, config, output_paths
    )

    null_summary = run_null_test(
        final_selection,
        final_result,
        bundle,
        config,
        output_paths,
    )
    save_method_comparison(final_result, config, output_paths)
    write_interpretation_summary(
        final_selection,
        final_summary,
        null_summary,
        output_paths["root"] / "interpretation_summary.txt",
    )

    logging.info("Pipeline completed successfully.")
    logging.info("Reports saved in: %s", config.report_dir)
    logging.info(
        "Final model: %s r=%d c=%d seed=%d | block R²=%.4f",
        final_selection["method"],
        final_selection["n_row_clusters"],
        final_selection["n_column_clusters"],
        final_selection["representative_seed"],
        final_summary["block_r2"],
    )


if __name__ == "__main__":
    main()