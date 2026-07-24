#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    PowerNorm,
    TwoSlopeNorm,
)
import numpy as np
import pandas as pd




@dataclass(frozen=True)
class Config:
    """Plot settings; no model parameter is defined or changed here."""

    focus_column_cluster: int = 1
    dpi: int = 300
    matrix_figure_width: float = 18.0
    matrix_figure_height: float = 10.0
    numeric_enhanced_gamma: float = 0.40
    contrast_annotation_threshold: float = 0.03
    binary_before_numeric: bool = True


@dataclass(frozen=True)
class Paths:
    project_dir: Path
    input_matrix: Path
    final_report_dir: Path
    reordered_output_dir: Path
    type_output_dir: Path

    @property
    def row_assignments(self) -> Path:
        return self.final_report_dir / "row_assignments.csv"

    @property
    def column_assignments(self) -> Path:
        return self.final_report_dir / "column_assignments.csv"

    @property
    def row_summary(self) -> Path:
        return self.final_report_dir / "row_cluster_summary.csv"

    @property
    def column_summary(self) -> Path:
        return self.final_report_dir / "column_cluster_summary.csv"

    @property
    def block_statistics(self) -> Path:
        return self.final_report_dir / "block_statistics.csv"


@dataclass(frozen=True)
class Reports:
    X: pd.DataFrame
    row_assignments: pd.DataFrame
    column_assignments: pd.DataFrame
    row_summary: pd.DataFrame
    column_summary: pd.DataFrame
    block_statistics: pd.DataFrame


def find_project_dir(start: Path) -> tuple[Path, str]:
    """Find the project root and identify the current or legacy layout."""
    for candidate in (start.resolve(), *start.resolve().parents):
        current_input = (
            candidate / "000_Dataset" / "train_filtered_no_activity.csv"
        )
        if current_input.is_file() and (candidate / "003_Models").is_dir():
            return candidate, "current"

        legacy_input = (
            candidate / "Dataset" / "train_filtered_no_activity.csv"
        )
        if legacy_input.is_file():
            return candidate, "legacy"

    raise FileNotFoundError(
        "Project root not found. Expected either "
        "'000_Dataset/train_filtered_no_activity.csv' or "
        "'Dataset/train_filtered_no_activity.csv' in this directory or one "
        "of its parents."
    )


def resolve_paths(script_dir: Path) -> Paths:
    """Resolve paths using the same final-model directory as the main pipeline."""
    project_dir, layout = find_project_dir(script_dir)

    if layout == "current":
        input_matrix = (
            project_dir / "000_Dataset" / "train_filtered_no_activity.csv"
        )
        final_report_dir = (
            project_dir
            / "003_Models"
            / "0031_Mixed"
            / "spectral_biclustering"
            / "reports"
            / "spectral_biclustering_pipeline"
            / "03_final_model"
        )
    else:
        input_matrix = (
            project_dir / "Dataset" / "train_filtered_no_activity.csv"
        )
        final_report_dir = (
            project_dir
            / "reports"
            / "spectral_biclustering_pipeline"
            / "03_final_model"
        )

    paths = Paths(
        project_dir=project_dir,
        input_matrix=input_matrix,
        final_report_dir=final_report_dir,
        reordered_output_dir=final_report_dir / "reordered_matrix_plots",
        type_output_dir=final_report_dir / "type_specific_plots",
    )

    required = (
        paths.input_matrix,
        paths.row_assignments,
        paths.column_assignments,
        paths.row_summary,
        paths.column_summary,
        paths.block_statistics,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Required final-model files were not found:\n" + details
        )

    paths.reordered_output_dir.mkdir(parents=True, exist_ok=True)
    paths.type_output_dir.mkdir(parents=True, exist_ok=True)
    return paths


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def read_csv_checked(
    path: Path,
    required_columns: set[str] | None = None,
    *,
    numeric_only: bool = False,
) -> pd.DataFrame:
    """Read a CSV, remove index artefacts and validate its schema."""
    frame = pd.read_csv(path)
    unnamed = [
        column
        for column in frame.columns
        if str(column).startswith("Unnamed:")
    ]
    if unnamed:
        frame = frame.drop(columns=unnamed)

    if required_columns:
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {sorted(missing)}"
            )

    if numeric_only:
        frame = frame.apply(pd.to_numeric, errors="raise")
    return frame


def coerce_boolean(series: pd.Series, name: str) -> pd.Series:
    """Convert a bool/0-1/True-False column without treating strings as truthy."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        invalid = ~series.isin([0, 1])
        if invalid.any():
            raise ValueError(f"{name} contains values other than 0 and 1.")
        return series.astype(bool)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid_values = sorted(set(normalized) - set(mapping))
    if invalid_values:
        raise ValueError(
            f"{name} contains invalid Boolean values: {invalid_values[:10]}"
        )
    return normalized.map(mapping).astype(bool)


def validate_complete_block_grid(blocks: pd.DataFrame) -> None:
    """Ensure that every row-cluster/column-cluster block is represented."""
    pairs = blocks[["row_cluster", "column_cluster"]]
    if pairs.duplicated().any():
        raise ValueError(
            "block_statistics.csv contains duplicate block identifiers."
        )

    expected = (
        blocks["row_cluster"].nunique()
        * blocks["column_cluster"].nunique()
    )
    if len(pairs) != expected:
        raise ValueError(
            "block_statistics.csv does not contain a complete block grid: "
            f"expected {expected} blocks, found {len(pairs)}."
        )


def load_reports(paths: Paths) -> Reports:
    """Load exactly the files generated for the selected final model."""
    X = read_csv_checked(paths.input_matrix, numeric_only=True)
    rows = read_csv_checked(
        paths.row_assignments,
        {"original_row_id", "row_cluster"},
    )
    columns = read_csv_checked(
        paths.column_assignments,
        {
            "feature",
            "feature_index",
            "is_binary",
            "feature_type",
            "zero_fraction",
            "column_cluster",
        },
    )
    row_summary = read_csv_checked(
        paths.row_summary,
        {"row_cluster", "n_rows"},
    )
    column_summary = read_csv_checked(
        paths.column_summary,
        {
            "column_cluster",
            "n_features",
            "n_binary_features",
            "n_numeric_features",
        },
    )
    blocks = read_csv_checked(
        paths.block_statistics,
        {
            "row_cluster",
            "column_cluster",
            "mean_contrast_to_column_cluster",
            "n_binary_features",
            "n_numeric_features",
            "binary_n_values",
            "binary_mean",
            "numeric_n_values",
            "numeric_mean",
        },
    )

    rows["original_row_id"] = pd.to_numeric(
        rows["original_row_id"], errors="raise"
    ).astype(int)
    rows["row_cluster"] = pd.to_numeric(
        rows["row_cluster"], errors="raise"
    ).astype(int)
    columns["column_cluster"] = pd.to_numeric(
        columns["column_cluster"], errors="raise"
    ).astype(int)
    columns["feature_index"] = pd.to_numeric(
        columns["feature_index"], errors="raise"
    ).astype(int)
    columns["is_binary"] = coerce_boolean(
        columns["is_binary"], "column_assignments.is_binary"
    )

    if rows["original_row_id"].duplicated().any():
        raise ValueError(
            "row_assignments.csv contains duplicate original_row_id values."
        )
    if columns["feature"].duplicated().any():
        raise ValueError(
            "column_assignments.csv contains duplicate feature names."
        )
    if not rows["original_row_id"].between(0, len(X) - 1).all():
        raise ValueError(
            "row_assignments.csv contains invalid original row indices."
        )

    missing_features = sorted(set(columns["feature"]) - set(X.columns))
    if missing_features:
        raise ValueError(
            "Assigned features are absent from the input matrix. "
            f"First missing names: {missing_features[:10]}"
        )

    if set(rows["row_cluster"]) != set(row_summary["row_cluster"]):
        raise ValueError(
            "row_assignments.csv and row_cluster_summary.csv refer to "
            "different row clusters."
        )
    if set(columns["column_cluster"]) != set(
        column_summary["column_cluster"]
    ):
        raise ValueError(
            "column_assignments.csv and column_cluster_summary.csv refer to "
            "different column clusters."
        )

    validate_complete_block_grid(blocks)
    return Reports(
        X=X,
        row_assignments=rows,
        column_assignments=columns,
        row_summary=row_summary,
        column_summary=column_summary,
        block_statistics=blocks,
    )


# ---------------------------------------------------------------------------
# Shared ordering and labels
# ---------------------------------------------------------------------------


def activity_fraction_column(row_summary: pd.DataFrame) -> str | None:
    """Return the Activity=1 fraction column used by the main pipeline."""
    candidates = (
        "target_1_fraction",
        "Activity_1_fraction",
        "activity_1_fraction",
    )
    return next(
        (name for name in candidates if name in row_summary.columns),
        None,
    )


def prepare_row_order(rows: pd.DataFrame) -> pd.DataFrame:
    ordered = rows.sort_values(
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
    ordered["_type_order"] = np.where(
        ordered["is_binary"] == config.binary_before_numeric,
        0,
        1,
    )
    ordered = (
        ordered.sort_values(
            [
                "column_cluster",
                "_type_order",
                "zero_fraction",
                "feature_index",
            ],
            kind="stable",
        )
        .drop(columns="_type_order")
        .reset_index(drop=True)
    )
    ordered.insert(0, "plot_position", np.arange(len(ordered), dtype=int))
    return ordered


def cluster_boundaries_and_centres(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        empty = np.array([], dtype=float)
        return np.array([], dtype=int), empty, empty

    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [len(labels)]))
    cluster_ids = labels[boundaries[:-1]]
    centres = (boundaries[:-1] + boundaries[1:] - 1) / 2.0
    return cluster_ids, boundaries.astype(float), centres


def row_label_map(rows: pd.DataFrame) -> dict[int, str]:
    activity_column = activity_fraction_column(rows)
    labels: dict[int, str] = {}

    for row in rows.sort_values("row_cluster").itertuples(index=False):
        cluster = int(row.row_cluster)
        text = f"R{cluster}\nn={int(row.n_rows)}"
        if activity_column is not None:
            activity = float(getattr(row, activity_column))
            text += f"\nActivity 1={100 * activity:.1f}%"
        labels[cluster] = text

    return labels


def column_label_map(
    columns: pd.DataFrame,
    feature_type: str,
) -> dict[int, str]:
    if feature_type not in {"binary", "numeric", "all"}:
        raise ValueError(f"Unsupported feature type: {feature_type}")

    labels: dict[int, str] = {}
    for row in columns.sort_values("column_cluster").itertuples(index=False):
        cluster = int(row.column_cluster)
        if feature_type == "binary":
            labels[cluster] = (
                f"C{cluster}\nbin={int(row.n_binary_features)}"
            )
        elif feature_type == "numeric":
            labels[cluster] = (
                f"C{cluster}\nnum={int(row.n_numeric_features)}"
            )
        else:
            labels[cluster] = (
                f"C{cluster}\nn={int(row.n_features)}\n"
                f"bin={int(row.n_binary_features)}, "
                f"num={int(row.n_numeric_features)}"
            )
    return labels


# ---------------------------------------------------------------------------
# Reordered-matrix plots
# ---------------------------------------------------------------------------


def reorder_matrix(
    X: pd.DataFrame,
    ordered_rows: pd.DataFrame,
    ordered_features: pd.DataFrame,
) -> np.ndarray:
    row_ids = ordered_rows["original_row_id"].to_numpy(dtype=int)
    feature_names = ordered_features["feature"].astype(str).tolist()
    return (
        X.iloc[row_ids]
        .loc[:, feature_names]
        .to_numpy(dtype=np.float32)
    )


def add_cluster_structure(
    ax: plt.Axes,
    row_labels: np.ndarray,
    column_labels: np.ndarray,
    row_summary: pd.DataFrame,
    column_summary: pd.DataFrame,
    feature_type: str,
) -> None:
    row_ids, row_boundaries, row_centres = (
        cluster_boundaries_and_centres(row_labels)
    )
    column_ids, column_boundaries, column_centres = (
        cluster_boundaries_and_centres(column_labels)
    )

    for boundary in row_boundaries[1:-1]:
        ax.axhline(boundary - 0.5, color="black", linewidth=1.1)
    for boundary in column_boundaries[1:-1]:
        ax.axvline(boundary - 0.5, color="black", linewidth=1.1)

    row_labels_by_cluster = row_label_map(row_summary)
    column_labels_by_cluster = column_label_map(
        column_summary,
        feature_type,
    )
    ax.set_yticks(row_centres)
    ax.set_yticklabels(
        [row_labels_by_cluster[int(cluster)] for cluster in row_ids],
        fontsize=9,
    )
    ax.set_xticks(column_centres)
    ax.set_xticklabels(
        [
            column_labels_by_cluster[int(cluster)]
            for cluster in column_ids
        ],
        fontsize=8,
    )
    ax.set_xlabel("Cluster di feature")
    ax.set_ylabel("Cluster di molecole")


def save_matrix_plot(
    matrix: np.ndarray,
    row_labels: np.ndarray,
    column_labels: np.ndarray,
    row_summary: pd.DataFrame,
    column_summary: pd.DataFrame,
    feature_type: str,
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
        figsize=(
            config.matrix_figure_width,
            config.matrix_figure_height,
        )
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
        ax,
        row_labels,
        column_labels,
        row_summary,
        column_summary,
        feature_type,
    )
    ax.set_title(title, fontsize=16, pad=14)

    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label(colorbar_label)
    if colorbar_ticks is not None:
        colorbar.set_ticks(colorbar_ticks)

    fig.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


def block_smoothed_matrix(
    matrix: np.ndarray,
    row_labels: np.ndarray,
    column_labels: np.ndarray,
) -> np.ndarray:
    """Replace every checkerboard block with its observed mean."""
    smoothed = np.empty_like(matrix, dtype=np.float32)
    for row_cluster in np.unique(row_labels):
        row_mask = row_labels == row_cluster
        for column_cluster in np.unique(column_labels):
            column_mask = column_labels == column_cluster
            block = matrix[np.ix_(row_mask, column_mask)]
            smoothed[np.ix_(row_mask, column_mask)] = float(block.mean())
    return smoothed


def save_reordered_plots(
    reports: Reports,
    paths: Paths,
    config: Config,
) -> None:
    ordered_rows = prepare_row_order(reports.row_assignments)
    ordered_features = prepare_feature_order(
        reports.column_assignments,
        config,
    )
    ordered_rows.to_csv(
        paths.reordered_output_dir / "reordered_row_order.csv",
        index=False,
    )
    ordered_features.to_csv(
        paths.reordered_output_dir / "reordered_feature_order.csv",
        index=False,
    )

    full_matrix = reorder_matrix(
        reports.X,
        ordered_rows,
        ordered_features,
    )
    row_clusters = ordered_rows["row_cluster"].to_numpy(dtype=int)
    column_clusters = ordered_features[
        "column_cluster"
    ].to_numpy(dtype=int)
    full_cmap = LinearSegmentedColormap.from_list(
        "white_to_blue",
        ["#ffffff", "#9bd3e3", "#2b8cbe", "#08306b"],
    )

    save_matrix_plot(
        full_matrix,
        row_clusters,
        column_clusters,
        reports.row_summary,
        reports.column_summary,
        "all",
        "Matrice riordinata — tutte le feature",
        paths.reordered_output_dir / "01_reordered_full_matrix.png",
        config,
        cmap=full_cmap,
        vmin=0.0,
        vmax=1.0,
        colorbar_label=(
            "Valore originale normalizzato: "
            "0 = bianco, 1 = blu scuro"
        ),
    )

    binary_features = ordered_features.loc[
        ordered_features["is_binary"]
    ].reset_index(drop=True)
    if not binary_features.empty:
        binary_cmap = ListedColormap(["#ffffff", "#08306b"])
        binary_norm = BoundaryNorm([-0.5, 0.5, 1.5], binary_cmap.N)
        save_matrix_plot(
            reorder_matrix(reports.X, ordered_rows, binary_features),
            row_clusters,
            binary_features["column_cluster"].to_numpy(dtype=int),
            reports.row_summary,
            reports.column_summary,
            "binary",
            "Matrice riordinata — sole feature binarie",
            (
                paths.reordered_output_dir
                / "02_reordered_binary_matrix.png"
            ),
            config,
            cmap=binary_cmap,
            norm=binary_norm,
            colorbar_label="Valore binario",
            colorbar_ticks=[0.0, 1.0],
        )

    numeric_features = ordered_features.loc[
        ~ordered_features["is_binary"]
    ].reset_index(drop=True)
    if not numeric_features.empty:
        numeric_matrix = reorder_matrix(
            reports.X,
            ordered_rows,
            numeric_features,
        )
        numeric_clusters = numeric_features[
            "column_cluster"
        ].to_numpy(dtype=int)
        numeric_cmap = LinearSegmentedColormap.from_list(
            "white_to_red",
            ["#ffffff", "#fdd49e", "#fc8d59", "#b30000"],
        )

        save_matrix_plot(
            numeric_matrix,
            row_clusters,
            numeric_clusters,
            reports.row_summary,
            reports.column_summary,
            "numeric",
            "Matrice riordinata — sole feature numeriche",
            (
                paths.reordered_output_dir
                / "03_reordered_numeric_matrix_linear.png"
            ),
            config,
            cmap=numeric_cmap,
            vmin=0.0,
            vmax=1.0,
            colorbar_label="Valore numerico normalizzato",
        )
        save_matrix_plot(
            numeric_matrix,
            row_clusters,
            numeric_clusters,
            reports.row_summary,
            reports.column_summary,
            "numeric",
            (
                "Matrice riordinata — feature numeriche "
                "(contrasto visivo aumentato)"
            ),
            (
                paths.reordered_output_dir
                / "04_reordered_numeric_matrix_enhanced.png"
            ),
            config,
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

    save_matrix_plot(
        block_smoothed_matrix(
            full_matrix,
            row_clusters,
            column_clusters,
        ),
        row_clusters,
        column_clusters,
        reports.row_summary,
        reports.column_summary,
        "all",
        "Matrice riordinata — medie dei blocchi",
        (
            paths.reordered_output_dir
            / "05_reordered_block_mean_matrix.png"
        ),
        config,
        cmap=full_cmap,
        vmin=0.0,
        vmax=1.0,
        colorbar_label="Media osservata nel blocco",
    )


# ---------------------------------------------------------------------------
# Complete block-contrast heatmap
# ---------------------------------------------------------------------------


def save_global_contrast_heatmap(
    reports: Reports,
    output_path: Path,
    config: Config,
) -> None:
    matrix = (
        reports.block_statistics.pivot(
            index="row_cluster",
            columns="column_cluster",
            values="mean_contrast_to_column_cluster",
        )
        .sort_index()
        .sort_index(axis=1)
    )
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("All block-contrast values are non-finite.")
    limit = max(float(np.max(np.abs(finite))), 1e-9)

    row_labels = row_label_map(reports.row_summary)
    column_labels = column_label_map(reports.column_summary, "all")
    fig_width = max(10.0, 1.5 * matrix.shape[1] + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 7.0))
    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        [column_labels[int(cluster)] for cluster in matrix.columns],
        fontsize=9,
    )
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(
        [row_labels[int(cluster)] for cluster in matrix.index],
        fontsize=9,
    )

    for row_index, column_index in np.ndindex(values.shape):
        value = values[row_index, column_index]
        if not np.isfinite(value) or abs(value) < 0.05:
            continue
        ax.text(
            column_index,
            row_index,
            f"{value:+.2f}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold" if abs(value) >= 0.15 else "normal",
        )

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Contrasto medio rispetto al cluster di feature")
    ax.set_title("Profili distintivi dei cluster di molecole", fontsize=16)
    ax.set_xlabel("Cluster di feature")
    ax.set_ylabel("Cluster di molecole")
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Type-specific statistics and plots
# ---------------------------------------------------------------------------


def weighted_means_by_column_cluster(
    blocks: pd.DataFrame,
    value_column: str,
    weight_column: str,
) -> pd.Series:
    """Compute cell-count-weighted means without mixing feature types."""
    valid = blocks[value_column].notna() & (blocks[weight_column] > 0)
    valid_blocks = blocks.loc[
        valid,
        ["column_cluster", value_column, weight_column],
    ].copy()
    weighted_values = (
        valid_blocks[value_column] * valid_blocks[weight_column]
    )
    numerator = weighted_values.groupby(
        valid_blocks["column_cluster"]
    ).sum()
    denominator = valid_blocks.groupby("column_cluster")[
        weight_column
    ].sum()
    return numerator / denominator


def add_type_specific_contrasts(blocks: pd.DataFrame) -> pd.DataFrame:
    enriched = blocks.copy()
    binary_global = weighted_means_by_column_cluster(
        enriched,
        "binary_mean",
        "binary_n_values",
    ).rename("binary_column_cluster_global_mean")
    numeric_global = weighted_means_by_column_cluster(
        enriched,
        "numeric_mean",
        "numeric_n_values",
    ).rename("numeric_column_cluster_global_mean")

    enriched = enriched.merge(
        binary_global,
        left_on="column_cluster",
        right_index=True,
        how="left",
    )
    enriched = enriched.merge(
        numeric_global,
        left_on="column_cluster",
        right_index=True,
        how="left",
    )
    enriched["binary_contrast"] = (
        enriched["binary_mean"]
        - enriched["binary_column_cluster_global_mean"]
    )
    enriched["numeric_contrast"] = (
        enriched["numeric_mean"]
        - enriched["numeric_column_cluster_global_mean"]
    )
    return enriched


def pivot_metric(
    blocks: pd.DataFrame,
    metric: str,
    valid_clusters: list[int],
) -> pd.DataFrame:
    return (
        blocks.pivot(
            index="row_cluster",
            columns="column_cluster",
            values=metric,
        )
        .sort_index()
        .reindex(columns=valid_clusters)
    )


def plot_mean_heatmap(
    matrix: pd.DataFrame,
    row_labels: dict[int, str],
    column_labels: dict[int, str],
    title: str,
    colorbar_label: str,
    output_path: Path,
    config: Config,
) -> None:
    values = matrix.to_numpy(dtype=float)
    fig_width = max(9.0, 1.45 * matrix.shape[1] + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 7.0))
    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        [column_labels[int(cluster)] for cluster in matrix.columns],
        fontsize=9,
    )
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(
        [row_labels[int(cluster)] for cluster in matrix.index],
        fontsize=9,
    )

    for row_index, column_index in np.ndindex(values.shape):
        value = values[row_index, column_index]
        if np.isfinite(value):
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(colorbar_label)
    ax.set_title(title, fontsize=15, pad=14)
    ax.set_xlabel("Cluster di feature")
    ax.set_ylabel("Cluster di molecole")
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_contrast_heatmap(
    matrix: pd.DataFrame,
    row_labels: dict[int, str],
    column_labels: dict[int, str],
    title: str,
    colorbar_label: str,
    output_path: Path,
    config: Config,
) -> None:
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    limit = max(float(np.max(np.abs(finite))), 1e-9)

    fig_width = max(9.0, 1.45 * matrix.shape[1] + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 7.0))
    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        [column_labels[int(cluster)] for cluster in matrix.columns],
        fontsize=9,
    )
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(
        [row_labels[int(cluster)] for cluster in matrix.index],
        fontsize=9,
    )

    for row_index, column_index in np.ndindex(values.shape):
        value = values[row_index, column_index]
        if (
            not np.isfinite(value)
            or abs(value) < config.contrast_annotation_threshold
        ):
            continue
        ax.text(
            column_index,
            row_index,
            f"{value:+.2f}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold" if abs(value) >= 0.15 else "normal",
        )

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(colorbar_label)
    ax.set_title(title, fontsize=15, pad=14)
    ax.set_xlabel("Cluster di feature")
    ax.set_ylabel("Cluster di molecole")
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_focus_cluster(
    blocks: pd.DataFrame,
    rows: pd.DataFrame,
    column_cluster: int,
    mean_column: str,
    global_mean_column: str,
    feature_type_label: str,
    ylabel: str,
    output_path: Path,
    config: Config,
) -> None:
    subset = (
        blocks.loc[blocks["column_cluster"] == column_cluster]
        .sort_values("row_cluster")
        .copy()
    )
    if subset.empty or subset[mean_column].notna().sum() == 0:
        return

    row_info = rows.set_index("row_cluster")
    activity_column = activity_fraction_column(rows)
    x_labels: list[str] = []
    for cluster in subset["row_cluster"].astype(int):
        label = f"R{cluster}\nn={int(row_info.loc[cluster, 'n_rows'])}"
        if activity_column is not None:
            activity = float(row_info.loc[cluster, activity_column])
            label += f"\nAct.1={100 * activity:.1f}%"
        x_labels.append(label)

    values = subset[mean_column].to_numpy(dtype=float)
    global_values = subset[global_mean_column].dropna()
    if global_values.empty:
        return
    global_mean = float(global_values.iloc[0])

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    positions = np.arange(len(subset))
    bars = ax.bar(positions, values)
    ax.axhline(
        global_mean,
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Media globale C{column_cluster} = {global_mean:.3f}"
        ),
    )
    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels)
    finite_values = values[np.isfinite(values)]
    upper_limit = (
        max(1.0, float(finite_values.max()) + 0.12)
        if finite_values.size
        else 1.0
    )
    ax.set_ylim(0.0, min(1.05, upper_limit))
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Cluster di molecole")
    ax.set_title(
        f"C{column_cluster}: profilo delle feature {feature_type_label}",
        fontsize=14,
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


def save_type_specific_plots(
    reports: Reports,
    paths: Paths,
    config: Config,
) -> None:
    blocks = add_type_specific_contrasts(reports.block_statistics)
    row_labels = row_label_map(reports.row_summary)
    binary_labels = column_label_map(
        reports.column_summary,
        "binary",
    )
    numeric_labels = column_label_map(
        reports.column_summary,
        "numeric",
    )
    binary_clusters = (
        reports.column_summary.loc[
            reports.column_summary["n_binary_features"] > 0,
            "column_cluster",
        ]
        .astype(int)
        .sort_values()
        .tolist()
    )
    numeric_clusters = (
        reports.column_summary.loc[
            reports.column_summary["n_numeric_features"] > 0,
            "column_cluster",
        ]
        .astype(int)
        .sort_values()
        .tolist()
    )

    if binary_clusters:
        plot_mean_heatmap(
            pivot_metric(blocks, "binary_mean", binary_clusters),
            row_labels,
            binary_labels,
            "Feature binarie: frazione media di valori 1",
            "Frazione di valori 1",
            paths.type_output_dir / "01_binary_block_means.png",
            config,
        )
        plot_contrast_heatmap(
            pivot_metric(blocks, "binary_contrast", binary_clusters),
            row_labels,
            binary_labels,
            "Feature binarie: contrasto della frazione di valori 1",
            "Differenza dalla media binaria del cluster",
            paths.type_output_dir / "02_binary_block_contrasts.png",
            config,
        )

    if numeric_clusters:
        plot_mean_heatmap(
            pivot_metric(blocks, "numeric_mean", numeric_clusters),
            row_labels,
            numeric_labels,
            "Feature numeriche: valore normalizzato medio",
            "Valore normalizzato medio",
            paths.type_output_dir / "03_numeric_block_means.png",
            config,
        )
        plot_contrast_heatmap(
            pivot_metric(blocks, "numeric_contrast", numeric_clusters),
            row_labels,
            numeric_labels,
            "Feature numeriche: contrasto del valore medio",
            "Differenza dalla media numerica del cluster",
            paths.type_output_dir / "04_numeric_block_contrasts.png",
            config,
        )

    focus = config.focus_column_cluster
    available_clusters = set(
        reports.column_summary["column_cluster"].astype(int)
    )
    if focus not in available_clusters:
        raise ValueError(
            f"Focus cluster C{focus} does not exist. "
            f"Available clusters: {sorted(available_clusters)}"
        )

    plot_focus_cluster(
        blocks,
        reports.row_summary,
        focus,
        "binary_mean",
        "binary_column_cluster_global_mean",
        "binarie",
        "Frazione media di valori 1",
        paths.type_output_dir / f"05_C{focus}_binary_profile.png",
        config,
    )
    plot_focus_cluster(
        blocks,
        reports.row_summary,
        focus,
        "numeric_mean",
        "numeric_column_cluster_global_mean",
        "numeriche",
        "Valore normalizzato medio",
        paths.type_output_dir / f"06_C{focus}_numeric_profile.png",
        config,
    )

    focus_features = (
        reports.column_assignments.loc[
            reports.column_assignments["column_cluster"] == focus
        ]
        .sort_values(["feature_type", "zero_fraction", "feature"])
        .reset_index(drop=True)
    )
    focus_features.to_csv(
        paths.type_output_dir / f"C{focus}_feature_list.csv",
        index=False,
    )

    output_columns = [
        "row_cluster",
        "column_cluster",
        "n_binary_features",
        "n_numeric_features",
        "binary_mean",
        "binary_column_cluster_global_mean",
        "binary_contrast",
        "numeric_mean",
        "numeric_column_cluster_global_mean",
        "numeric_contrast",
    ]
    blocks[output_columns].sort_values(
        ["column_cluster", "row_cluster"]
    ).to_csv(
        paths.type_output_dir / "type_specific_block_statistics.csv",
        index=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    config = Config()
    paths = resolve_paths(Path(__file__).resolve().parent)
    reports = load_reports(paths)

    save_reordered_plots(reports, paths, config)
    save_global_contrast_heatmap(
        reports,
        paths.final_report_dir / "block_contrast_heatmap_readable.png",
        config,
    )
    save_type_specific_plots(reports, paths, config)

    print("Spectral-biclustering post-processing completed.")
    print(f"Final model reports: {paths.final_report_dir}")
    print(f"Reordered matrices:  {paths.reordered_output_dir}")
    print(f"Type-specific plots: {paths.type_output_dir}")
    print(f"Focus cluster:       C{config.focus_column_cluster}")


if __name__ == "__main__":
    main()