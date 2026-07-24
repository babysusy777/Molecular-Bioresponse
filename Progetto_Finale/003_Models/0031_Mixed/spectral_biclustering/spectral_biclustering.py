from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralBiclustering


# Script location:
# Molecular-Bioresponse/models/spectral_biclustering.py
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

INPUT_PATH = (
    PROJECT_DIR / "Dataset" / "train_filtered_no_activity.csv"
)
TARGET_PATH = (
    PROJECT_DIR / "Dataset" / "train_activity_target.csv"
)
REPORT_DIR = PROJECT_DIR / "reports" / "spectral_biclustering"

N_ROW_CLUSTERS = 4
N_COLUMN_CLUSTERS = 4
RANDOM_STATE = 42


def load_numeric_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    frame = pd.read_csv(path)
    unnamed = [c for c in frame.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        frame = frame.drop(columns=unnamed)

    return frame.apply(pd.to_numeric, errors="raise")


def load_data() -> tuple[pd.DataFrame, pd.Series | None]:
    X = load_numeric_csv(INPUT_PATH)

    if X.empty:
        raise ValueError("The feature matrix is empty.")
    if X.isna().any().any():
        raise ValueError("The feature matrix contains missing values.")

    values = X.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("The feature matrix contains NaN or infinity.")
    if values.min() < 0:
        raise ValueError(
            "SpectralBiclustering with bistochastic normalization "
            "requires non-negative values."
        )

    # Zero-sum rows/columns make bistochastic normalization undefined.
    valid_columns = ~np.isclose(values.sum(axis=0), 0.0)
    removed_columns = X.columns[~valid_columns].tolist()
    if removed_columns:
        pd.Series(
            removed_columns, name="removed_zero_sum_feature"
        ).to_csv(
            REPORT_DIR / "removed_zero_sum_features.csv", index=False
        )
        X = X.loc[:, valid_columns]
        values = X.to_numpy(dtype=np.float64)

    valid_rows = ~np.isclose(values.sum(axis=1), 0.0)
    removed_rows = np.flatnonzero(~valid_rows)
    if len(removed_rows):
        pd.Series(
            removed_rows, name="removed_zero_sum_original_row"
        ).to_csv(
            REPORT_DIR / "removed_zero_sum_rows.csv", index=False
        )
        X = X.loc[valid_rows].copy()

    y = None
    if TARGET_PATH.exists():
        target = load_numeric_csv(TARGET_PATH)
        if target.shape[1] != 1:
            raise ValueError("The target file must have one column.")
        if len(target) != len(valid_rows):
            raise ValueError(
                "Target and original feature matrix have different lengths."
            )
        y = target.iloc[:, 0].loc[valid_rows].reset_index(drop=True)

    return X.reset_index(drop=True), y


def calculate_block_statistics(
    values: np.ndarray,
    row_labels: np.ndarray,
    column_labels: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    records = []
    total_sse = 0.0

    for row_cluster in np.unique(row_labels):
        row_mask = row_labels == row_cluster

        for column_cluster in np.unique(column_labels):
            column_mask = column_labels == column_cluster
            block = values[np.ix_(row_mask, column_mask)]

            block_mean = float(block.mean())
            row_means = block.mean(axis=1, keepdims=True)
            column_means = block.mean(axis=0, keepdims=True)

            residue = block - row_means - column_means + block_mean
            h_score = float(np.mean(residue**2))
            block_sse = float(np.sum((block - block_mean) ** 2))
            total_sse += block_sse

            records.append(
                {
                    "row_cluster": int(row_cluster),
                    "column_cluster": int(column_cluster),
                    "n_rows": int(block.shape[0]),
                    "n_features": int(block.shape[1]),
                    "volume": int(block.size),
                    "mean": block_mean,
                    "std": float(block.std()),
                    "zero_fraction": float(np.mean(block == 0)),
                    "h_score": h_score,
                }
            )

    total_sst = float(np.sum((values - values.mean()) ** 2))
    block_r2 = (
        1.0 - total_sse / total_sst
        if total_sst > 0
        else float("nan")
    )

    return pd.DataFrame(records), block_r2


def save_reordered_heatmap(
    values: np.ndarray,
    row_labels: np.ndarray,
    column_labels: np.ndarray,
) -> None:
    row_order = np.argsort(row_labels, kind="stable")
    column_order = np.argsort(column_labels, kind="stable")
    reordered = values[row_order][:, column_order]

    sorted_rows = row_labels[row_order]
    sorted_columns = column_labels[column_order]

    fig, ax = plt.subplots(figsize=(16, 10))
    image = ax.imshow(
        reordered, aspect="auto", interpolation="nearest"
    )
    fig.colorbar(image, ax=ax, label="Feature value")

    for boundary in np.flatnonzero(np.diff(sorted_rows)) + 0.5:
        ax.axhline(boundary, linewidth=0.8)

    for boundary in np.flatnonzero(np.diff(sorted_columns)) + 0.5:
        ax.axvline(boundary, linewidth=0.8)

    ax.set_title("Spectral biclustering: reordered matrix")
    ax.set_xlabel("Features ordered by column cluster")
    ax.set_ylabel("Molecules ordered by row cluster")
    fig.tight_layout()
    fig.savefig(
        REPORT_DIR / "reordered_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_block_mean_heatmap(block_stats: pd.DataFrame) -> None:
    matrix = block_stats.pivot(
        index="row_cluster",
        columns="column_cluster",
        values="mean",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        matrix.to_numpy(), aspect="auto", interpolation="nearest"
    )
    fig.colorbar(image, ax=ax, label="Block mean")

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f"{matrix.iloc[i, j]:.3f}",
                ha="center", va="center"
            )

    ax.set_title("Mean of each checkerboard block")
    ax.set_xlabel("Column cluster")
    ax.set_ylabel("Row cluster")
    fig.tight_layout()
    fig.savefig(
        REPORT_DIR / "block_mean_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    values = X.to_numpy(dtype=np.float64)

    print(f"Dataset: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"Value range: [{values.min():.6f}, {values.max():.6f}]")

    model = SpectralBiclustering(
        n_clusters=(N_ROW_CLUSTERS, N_COLUMN_CLUSTERS),
        method="scale",
        n_components=10,
        n_best=5,
        svd_method="randomized",
        n_init=20,
        random_state=RANDOM_STATE,
    )
    model.fit(X)

    row_labels = model.row_labels_
    column_labels = model.column_labels_

    row_assignments = pd.DataFrame(
        {
            "row_id": np.arange(len(X)),
            "row_cluster": row_labels,
        }
    )
    if y is not None:
        row_assignments["Activity"] = y.to_numpy()

    row_assignments.to_csv(
        REPORT_DIR / "row_assignments.csv", index=False
    )

    column_assignments = pd.DataFrame(
        {
            "feature": X.columns,
            "column_cluster": column_labels,
            "is_binary": [
                set(pd.unique(X[column])).issubset({0, 1})
                for column in X.columns
            ],
        }
    )
    column_assignments.to_csv(
        REPORT_DIR / "column_assignments.csv", index=False
    )

    feature_summary = (
        column_assignments.groupby("column_cluster", as_index=False)
        .agg(
            n_features=("feature", "size"),
            n_binary=("is_binary", "sum"),
        )
    )
    feature_summary["n_numeric"] = (
        feature_summary["n_features"] - feature_summary["n_binary"]
    )
    feature_summary["binary_fraction"] = (
        feature_summary["n_binary"] / feature_summary["n_features"]
    )
    feature_summary.to_csv(
        REPORT_DIR / "feature_types_by_column_cluster.csv",
        index=False,
    )

    if y is not None:
        pd.crosstab(
            row_assignments["row_cluster"],
            row_assignments["Activity"],
        ).to_csv(REPORT_DIR / "activity_counts_by_row_cluster.csv")

        pd.crosstab(
            row_assignments["row_cluster"],
            row_assignments["Activity"],
            normalize="index",
        ).to_csv(
            REPORT_DIR / "activity_fractions_by_row_cluster.csv"
        )

    block_stats, block_r2 = calculate_block_statistics(
        values, row_labels, column_labels
    )
    block_stats.to_csv(
        REPORT_DIR / "block_statistics.csv", index=False
    )

    save_reordered_heatmap(values, row_labels, column_labels)
    save_block_mean_heatmap(block_stats)

    row_sizes = np.bincount(row_labels)
    column_sizes = np.bincount(column_labels)

    summary = pd.DataFrame(
        [
            {
                "n_rows": X.shape[0],
                "n_features": X.shape[1],
                "n_row_clusters": N_ROW_CLUSTERS,
                "n_column_clusters": N_COLUMN_CLUSTERS,
                "n_blocks": N_ROW_CLUSTERS * N_COLUMN_CLUSTERS,
                "method": "bistochastic",
                "block_r2": block_r2,
                "smallest_row_cluster": int(row_sizes.min()),
                "largest_row_cluster": int(row_sizes.max()),
                "smallest_column_cluster": int(column_sizes.min()),
                "largest_column_cluster": int(column_sizes.max()),
            }
        ]
    )
    summary.to_csv(REPORT_DIR / "summary.csv", index=False)

    print("\nSpectral biclustering completed")
    print("Row cluster sizes:", row_sizes.tolist())
    print("Column cluster sizes:", column_sizes.tolist())
    print(f"Checkerboard block R^2: {block_r2:.6f}")
    print(f"Reports saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
