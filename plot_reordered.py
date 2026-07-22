#!/usr/bin/env python3
"""
Create readable reordered-matrix plots for the final spectral biclustering model.

The script DOES NOT refit the model. It reads:
- the preprocessed feature matrix;
- row_assignments.csv;
- column_assignments.csv;
- row_cluster_summary.csv.

It produces:
1. Full reordered matrix, all features;
2. Reordered binary-feature matrix;
3. Reordered numeric-feature matrix with a linear scale;
4. Reordered numeric-feature matrix with enhanced visibility for small values;
5. Block-smoothed reordered matrix;
6. CSV files recording the exact displayed row and feature order.

Place this script either:
- in the project root; or
- in a subdirectory such as models/.

The project root is detected automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    PowerNorm,
)
import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class Config:
    input_relative_path: Path = Path(
        "Dataset/train_filtered_no_activity.csv"
    )
    final_report_relative_path: Path = Path(
        "reports/spectral_biclustering_pipeline/03_final_model"
    )
    output_subdirectory: str = "reordered_matrix_plots"

    dpi: int = 300
    figure_width: float = 18.0
    figure_height: float = 10.0

    # Sorting inside each spectral cluster.
    # Rows retain their original dataset order.
    # Features are ordered by type, sparsity, and original feature index.
    binary_before_numeric: bool = True

    # Gamma < 1 increases visibility of low non-zero numerical values.
    numeric_enhanced_gamma: float = 0.40


# =============================================================================
# Project paths
# =============================================================================

def find_project_dir(start: Path) -> Path:
    """Find the project root without assuming where this script is stored."""
    start = start.resolve()
    candidates = [start, *start.parents]

    for candidate in candidates:
        dataset_dir = candidate / "Dataset" 
        reports_dir = candidate / "reports"
        if dataset_dir.is_dir() and reports_dir.is_dir():
            return candidate

    raise FileNotFoundError(
        "Project root not found. Expected a parent directory containing both "
        "'Dataset/processed' and 'reports'."
    )


def resolve_paths(config: Config) -> dict[str, Path]:
    project_dir = find_project_dir(Path(__file__).resolve().parent)
    final_dir = project_dir / config.final_report_relative_path
    output_dir = final_dir / config.output_subdirectory

    paths = {
        "project_dir": project_dir,
        "input": project_dir / config.input_relative_path,
        "final_dir": final_dir,
        "output_dir": output_dir,
        "row_assignments": final_dir / "row_assignments.csv",
        "column_assignments": final_dir / "column_assignments.csv",
        "row_summary": final_dir / "row_cluster_summary.csv",
    }

    required = [
        "input",
        "row_assignments",
        "column_assignments",
        "row_summary",
    ]
    missing = [str(paths[name]) for name in required if not paths[name].exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Required files not found:\n{formatted}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return paths


# =============================================================================
# Loading and validation
# =============================================================================

def load_numeric_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    unnamed = [c for c in frame.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        frame = frame.drop(columns=unnamed)
    return frame.apply(pd.to_numeric, errors="raise")


def load_inputs(paths: dict[str, Path]) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    X = load_numeric_csv(paths["input"])
    rows = pd.read_csv(paths["row_assignments"])
    columns = pd.read_csv(paths["column_assignments"])
    row_summary = pd.read_csv(paths["row_summary"])

    required_row_columns = {"original_row_id", "row_cluster"}
    required_feature_columns = {
        "feature",
        "feature_index",
        "is_binary",
        "feature_type",
        "zero_fraction",
        "column_cluster",
    }

    if not required_row_columns.issubset(rows.columns):
        missing = required_row_columns - set(rows.columns)
        raise ValueError(f"Missing row-assignment columns: {sorted(missing)}")

    if not required_feature_columns.issubset(columns.columns):
        missing = required_feature_columns - set(columns.columns)
        raise ValueError(f"Missing column-assignment columns: {sorted(missing)}")

    if rows["original_row_id"].duplicated().any():
        raise ValueError("row_assignments.csv contains duplicate original_row_id values.")

    if columns["feature"].duplicated().any():
        raise ValueError("column_assignments.csv contains duplicate feature names.")

    missing_features = sorted(set(columns["feature"]) - set(X.columns))
    if missing_features:
        preview = missing_features[:10]
        raise ValueError(
            "Some assigned features are absent from the input matrix. "
            f"First missing names: {preview}"
        )

    invalid_rows = rows.loc[
        ~rows["original_row_id"].between(0, len(X) - 1),
        "original_row_id",
    ]
    if not invalid_rows.empty:
        raise ValueError("row_assignments.csv contains invalid original row indices.")

    return X, rows, columns, row_summary


# =============================================================================
# Ordering
# =============================================================================

def prepare_row_order(rows: pd.DataFrame) -> pd.DataFrame:
    ordered = rows.copy()
    ordered["original_row_id"] = ordered["original_row_id"].astype(int)
    ordered["row_cluster"] = ordered["row_cluster"].astype(int)

    ordered = ordered.sort_values(
        ["row_cluster", "original_row_id"],
        kind="stable",
    ).reset_index(drop=True)

    ordered.insert(0, "plot_position", np.arange(len(ordered), dtype=int))
    return ordered


def prepare_feature_order(
    columns: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    ordered = columns.copy()
    ordered["feature"] = ordered["feature"].astype(str)
    ordered["column_cluster"] = ordered["column_cluster"].astype(int)
    ordered["feature_index"] = ordered["feature_index"].astype(int)
    ordered["is_binary"] = ordered["is_binary"].astype(bool)

    if config.binary_before_numeric:
        ordered["_type_order"] = np.where(ordered["is_binary"], 0, 1)
    else:
        ordered["_type_order"] = np.where(ordered["is_binary"], 1, 0)

    ordered = ordered.sort_values(
        [
            "column_cluster",
            "_type_order",
            "zero_fraction",
            "feature_index",
        ],
        ascending=[True, True, True, True],
        kind="stable",
    ).drop(columns="_type_order")

    ordered = ordered.reset_index(drop=True)
    ordered.insert(0, "plot_position", np.arange(len(ordered), dtype=int))
    return ordered


# =============================================================================
# Plot helpers
# =============================================================================

def cluster_boundaries_and_centres(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cluster ids, boundary positions, and label centres."""
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [len(labels)]))
    cluster_ids = labels[boundaries[:-1]]
    centres = (boundaries[:-1] + boundaries[1:] - 1) / 2.0
    return cluster_ids, boundaries.astype(float), centres


def activity_fraction_column(row_summary: pd.DataFrame) -> str | None:
    candidates = [
        "target_1_fraction",
        "Activity_1_fraction",
        "activity_1_fraction",
    ]
    for name in candidates:
        if name in row_summary.columns:
            return name
    return None


def make_row_tick_labels(
    row_cluster_ids: np.ndarray,
    row_summary: pd.DataFrame,
) -> list[str]:
    summary = row_summary.set_index("row_cluster")
    activity_column = activity_fraction_column(row_summary)
    labels: list[str] = []

    for cluster in row_cluster_ids:
        if cluster not in summary.index:
            labels.append(f"R{cluster}")
            continue

        record = summary.loc[cluster]
        n_rows = int(record.get("n_rows", record.get("size", 0)))
        text = f"R{cluster}\nn={n_rows}"

        if activity_column is not None:
            activity = 100.0 * float(record[activity_column])
            text += f"\nActivity 1={activity:.1f}%"

        labels.append(text)

    return labels


def make_column_tick_labels(
    cluster_ids: np.ndarray,
    ordered_features: pd.DataFrame,
    feature_kind: str,
) -> list[str]:
    labels: list[str] = []

    for cluster in cluster_ids:
        subset = ordered_features.loc[
            ordered_features["column_cluster"] == cluster
        ]
        if feature_kind == "binary":
            count = int(subset["is_binary"].sum())
            labels.append(f"C{cluster}\nbin={count}")
        elif feature_kind == "numeric":
            count = int((~subset["is_binary"]).sum())
            labels.append(f"C{cluster}\nnum={count}")
        else:
            binary_count = int(subset["is_binary"].sum())
            numeric_count = int((~subset["is_binary"]).sum())
            labels.append(
                f"C{cluster}\nn={len(subset)}\n"
                f"bin={binary_count}, num={numeric_count}"
            )

    return labels


def add_cluster_structure(
    ax: plt.Axes,
    row_labels: np.ndarray,
    column_labels: np.ndarray,
    row_summary: pd.DataFrame,
    ordered_features: pd.DataFrame,
    feature_kind: str,
) -> None:
    row_ids, row_boundaries, row_centres = cluster_boundaries_and_centres(
        row_labels
    )
    col_ids, col_boundaries, col_centres = cluster_boundaries_and_centres(
        column_labels
    )

    # Internal boundaries only.
    for boundary in row_boundaries[1:-1]:
        ax.axhline(boundary - 0.5, color="black", linewidth=1.1)

    for boundary in col_boundaries[1:-1]:
        ax.axvline(boundary - 0.5, color="black", linewidth=1.1)

    ax.set_yticks(row_centres)
    ax.set_yticklabels(
        make_row_tick_labels(row_ids, row_summary),
        fontsize=9,
    )

    ax.set_xticks(col_centres)
    ax.set_xticklabels(
        make_column_tick_labels(col_ids, ordered_features, feature_kind),
        fontsize=8,
        rotation=0,
    )

    ax.set_xlabel("Cluster di feature")
    ax.set_ylabel("Cluster di molecole")


def save_matrix_plot(
    matrix: np.ndarray,
    row_labels: np.ndarray,
    column_labels: np.ndarray,
    row_summary: pd.DataFrame,
    ordered_features: pd.DataFrame,
    feature_kind: str,
    title: str,
    output_path: Path,
    config: Config,
    *,
    cmap,
    norm=None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str,
    colorbar_ticks: list[float] | None = None,
) -> None:
    fig, ax = plt.subplots(
        figsize=(config.figure_width, config.figure_height)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )

    add_cluster_structure(
        ax=ax,
        row_labels=row_labels,
        column_labels=column_labels,
        row_summary=row_summary,
        ordered_features=ordered_features,
        feature_kind=feature_kind,
    )

    ax.set_title(title, fontsize=16, pad=14)

    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label(colorbar_label)
    if colorbar_ticks is not None:
        colorbar.set_ticks(colorbar_ticks)

    fig.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Reordered matrices
# =============================================================================

def reorder_matrix(
    X: pd.DataFrame,
    ordered_rows: pd.DataFrame,
    ordered_features: pd.DataFrame,
) -> np.ndarray:
    row_ids = ordered_rows["original_row_id"].to_numpy(dtype=int)
    feature_names = ordered_features["feature"].tolist()
    return X.iloc[row_ids].loc[:, feature_names].to_numpy(dtype=np.float32)


def block_smoothed_matrix(
    matrix: np.ndarray,
    row_labels: np.ndarray,
    column_labels: np.ndarray,
) -> np.ndarray:
    """Replace every bicluster block with its observed mean."""
    smoothed = np.empty_like(matrix, dtype=np.float32)

    for row_cluster in np.unique(row_labels):
        row_mask = row_labels == row_cluster
        for column_cluster in np.unique(column_labels):
            col_mask = column_labels == column_cluster
            block = matrix[np.ix_(row_mask, col_mask)]
            smoothed[np.ix_(row_mask, col_mask)] = float(block.mean())

    return smoothed


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    config = Config()
    paths = resolve_paths(config)

    X, rows, columns, row_summary = load_inputs(paths)
    ordered_rows = prepare_row_order(rows)
    ordered_features = prepare_feature_order(columns, config)

    # Save the exact order shown in the plots.
    ordered_rows.to_csv(
        paths["output_dir"] / "reordered_row_order.csv",
        index=False,
    )
    ordered_features.to_csv(
        paths["output_dir"] / "reordered_feature_order.csv",
        index=False,
    )

    full_matrix = reorder_matrix(X, ordered_rows, ordered_features)
    row_labels = ordered_rows["row_cluster"].to_numpy(dtype=int)
    column_labels = ordered_features["column_cluster"].to_numpy(dtype=int)

    # Explicit, intuitive sequential colour map:
    # white = zero; progressively darker blue = larger value.
    full_cmap = LinearSegmentedColormap.from_list(
        "white_to_blue",
        ["#ffffff", "#9bd3e3", "#2b8cbe", "#08306b"],
    )

    save_matrix_plot(
        matrix=full_matrix,
        row_labels=row_labels,
        column_labels=column_labels,
        row_summary=row_summary,
        ordered_features=ordered_features,
        feature_kind="all",
        title="Matrice riordinata — tutte le feature",
        output_path=paths["output_dir"] / "01_reordered_full_matrix.png",
        config=config,
        cmap=full_cmap,
        vmin=0.0,
        vmax=1.0,
        colorbar_label="Valore originale normalizzato: 0 = bianco, 1 = blu scuro",
    )

    # Binary features: exact two-colour interpretation.
    binary_features = ordered_features.loc[
        ordered_features["is_binary"]
    ].reset_index(drop=True)
    if not binary_features.empty:
        binary_matrix = reorder_matrix(X, ordered_rows, binary_features)
        binary_column_labels = binary_features["column_cluster"].to_numpy(dtype=int)

        binary_cmap = ListedColormap(["#ffffff", "#08306b"])
        binary_norm = BoundaryNorm([-0.5, 0.5, 1.5], binary_cmap.N)

        save_matrix_plot(
            matrix=binary_matrix,
            row_labels=row_labels,
            column_labels=binary_column_labels,
            row_summary=row_summary,
            ordered_features=binary_features,
            feature_kind="binary",
            title="Matrice riordinata — sole feature binarie",
            output_path=paths["output_dir"] / "02_reordered_binary_matrix.png",
            config=config,
            cmap=binary_cmap,
            norm=binary_norm,
            colorbar_label="Valore binario",
            colorbar_ticks=[0.0, 1.0],
        )

    # Numeric features, linear scale.
    numeric_features = ordered_features.loc[
        ~ordered_features["is_binary"]
    ].reset_index(drop=True)
    if not numeric_features.empty:
        numeric_matrix = reorder_matrix(X, ordered_rows, numeric_features)
        numeric_column_labels = numeric_features[
            "column_cluster"
        ].to_numpy(dtype=int)

        numeric_cmap = LinearSegmentedColormap.from_list(
            "white_to_red",
            ["#ffffff", "#fdd49e", "#fc8d59", "#b30000"],
        )

        save_matrix_plot(
            matrix=numeric_matrix,
            row_labels=row_labels,
            column_labels=numeric_column_labels,
            row_summary=row_summary,
            ordered_features=numeric_features,
            feature_kind="numeric",
            title="Matrice riordinata — sole feature numeriche",
            output_path=paths["output_dir"] / "03_reordered_numeric_matrix_linear.png",
            config=config,
            cmap=numeric_cmap,
            vmin=0.0,
            vmax=1.0,
            colorbar_label="Valore numerico normalizzato",
        )

        # A second version makes small non-zero values visible.
        save_matrix_plot(
            matrix=numeric_matrix,
            row_labels=row_labels,
            column_labels=numeric_column_labels,
            row_summary=row_summary,
            ordered_features=numeric_features,
            feature_kind="numeric",
            title=(
                "Matrice riordinata — feature numeriche "
                "(contrasto visivo aumentato)"
            ),
            output_path=paths["output_dir"] / "04_reordered_numeric_matrix_enhanced.png",
            config=config,
            cmap=numeric_cmap,
            norm=PowerNorm(
                gamma=config.numeric_enhanced_gamma,
                vmin=0.0,
                vmax=1.0,
            ),
            colorbar_label=(
                "Valore numerico normalizzato "
                "(scala cromatica non lineare)"
            ),
        )

    # Smoothed version: every block is represented by its block mean.
    smoothed = block_smoothed_matrix(
        full_matrix,
        row_labels,
        column_labels,
    )

    save_matrix_plot(
        matrix=smoothed,
        row_labels=row_labels,
        column_labels=column_labels,
        row_summary=row_summary,
        ordered_features=ordered_features,
        feature_kind="all",
        title="Matrice riordinata — medie dei blocchi",
        output_path=paths["output_dir"] / "05_reordered_block_mean_matrix.png",
        config=config,
        cmap=full_cmap,
        vmin=0.0,
        vmax=1.0,
        colorbar_label="Media osservata nel blocco",
    )

    print("Reordered-matrix plots completed.")
    print(f"Input matrix: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"Displayed rows: {len(ordered_rows)}")
    print(f"Displayed features: {len(ordered_features)}")
    print(f"Output directory: {paths['output_dir']}")
    print()
    print("Interpretation:")
    print("- Full matrix: white = 0; darker blue = larger value.")
    print("- Binary matrix: white = 0; dark blue = 1.")
    print("- Numeric matrix: white = low value; dark red = high value.")
    print("- Black lines delimit spectral row and column clusters.")
    print("- CSV files preserve the exact displayed row and feature order.")


if __name__ == "__main__":
    main()
