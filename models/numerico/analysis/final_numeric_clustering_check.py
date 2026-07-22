#!/usr/bin/env python3
"""
Final check of clustering structure in the numeric Molecular Bioresponse data.

The script deliberately avoids column-wise z-score standardization.

Pipelines
---------
1. Presence/Jaccard:
   B_ij = 1 if X_ij > 0, otherwise 0.
   Samples are compared through Jaccard distance.

2. Row-normalized numeric profile/Cosine:
   each molecular profile is normalized to unit L2 norm.
   Samples are compared through cosine distance.

For both representations:
- average-linkage hierarchical clustering;
- evaluation for k = 2,...,10;
- global and per-cluster silhouette;
- cluster-size balance;
- post-hoc comparison with Activity;
- stability under random feature subsampling.

A clustering is considered "admissible" only if:
- every cluster contains at least MIN_CLUSTER_FRACTION of the samples;
- every cluster has positive mean silhouette;
- the requested number of clusters is actually produced.

3. Optional controlled DBSCAN:
- row L2 normalization;
- TruncatedSVD without column standardization;
- row normalization of the SVD representation;
- eps candidates derived from k-distance quantiles;
- explicit reporting of noise fraction, cluster sizes and silhouette.

Activity is never used to construct clusters. It is used only afterwards
to interpret them.
"""

from __future__ import annotations

import gc
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2_contingency
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    "/Users/susannabaldo/Desktop/Machine_Learning_Project/"
    "Molecular-Bioresponse"
)

DATA_PATH = (
    BASE_DIR
    / "Dataset"
    / "train_numeric_only.csv"
)

TARGET_PATH = (
    BASE_DIR
    / "Dataset"
    / "train_activity_target.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "numerico"
    / "analysis"
    / "final_clustering_check"
)

RANDOM_STATE = 42
DPI = 180

# Hierarchical clustering
K_MIN = 2
K_MAX = 10
LINKAGE_METHOD = "average"

# Minimum requirements for accepting a partition
MIN_CLUSTER_FRACTION = 0.05
REQUIRE_POSITIVE_MEAN_SILHOUETTE_PER_CLUSTER = True

# Stability under random subsets of the original features
RUN_FEATURE_SUBSAMPLING = True
N_STABILITY_RUNS = 5
FEATURE_SUBSAMPLE_FRACTION = 0.70

# DBSCAN
RUN_DBSCAN = True
SVD_COMPONENTS_FOR_DBSCAN = 50
DBSCAN_MIN_SAMPLES_VALUES = [5, 10, 20, 30]
DBSCAN_EPS_QUANTILES = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]

# Requirements for treating a DBSCAN result as interpretable
DBSCAN_MIN_CLUSTER_SIZE = 50
DBSCAN_MAX_NOISE_FRACTION = 0.50
DBSCAN_MAX_NUMBER_OF_CLUSTERS = 10


# ============================================================
# GENERAL UTILITIES
# ============================================================

def create_directories() -> dict[str, Path]:
    directories = {
        "root": OUTPUT_DIR,
        "presence": OUTPUT_DIR / "presence_jaccard",
        "cosine": OUTPUT_DIR / "row_normalized_cosine",
        "comparison": OUTPUT_DIR / "comparison",
        "stability": OUTPUT_DIR / "feature_subsampling_stability",
        "dbscan": OUTPUT_DIR / "dbscan_cosine_svd",
    }

    for directory in directories.values():
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    return directories


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close()


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Numeric dataset not found: {DATA_PATH}"
        )

    if not TARGET_PATH.exists():
        raise FileNotFoundError(
            f"Activity target not found: {TARGET_PATH}"
        )

    X = pd.read_csv(DATA_PATH)

    if "Activity" in X.columns:
        warnings.warn(
            "Activity was found among the features and has been removed."
        )
        X = X.drop(columns=["Activity"])

    non_numeric = [
        column
        for column in X.columns
        if not pd.api.types.is_numeric_dtype(X[column])
    ]

    if non_numeric:
        raise ValueError(
            f"Non-numeric columns found: {non_numeric[:10]}"
        )

    if X.isna().any().any():
        raise ValueError(
            "The numeric dataset contains missing values."
        )

    values = X.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "The numeric dataset contains infinite values."
        )

    constant_columns = [
        column
        for column in X.columns
        if X[column].nunique(dropna=False) <= 1
    ]

    if constant_columns:
        warnings.warn(
            f"Removing {len(constant_columns)} constant features."
        )
        X = X.drop(columns=constant_columns)

    target_df = pd.read_csv(TARGET_PATH)

    if "Activity" in target_df.columns:
        y = target_df["Activity"]
    else:
        y = target_df.iloc[:, 0]

    y = pd.Series(
        y.to_numpy(),
        name="Activity",
    ).reset_index(drop=True)

    X = X.reset_index(drop=True)

    if len(X) != len(y):
        raise ValueError(
            "Feature matrix and Activity target have different lengths."
        )

    if not set(pd.unique(y)).issubset({0, 1}):
        raise ValueError(
            "Activity must be binary and encoded as 0/1."
        )

    print(f"Numeric dataset: {X.shape}")
    print(
        "Activity distribution: "
        f"{y.value_counts().sort_index().to_dict()}"
    )

    return X, y.astype(int)


def cramers_v_from_labels(
    labels: np.ndarray,
    y: pd.Series,
) -> tuple[float, float]:
    table = pd.crosstab(
        pd.Series(labels, name="cluster"),
        y,
    )

    if (
        table.shape[0] < 2
        or table.shape[1] < 2
    ):
        return np.nan, np.nan

    chi2, p_value, _, _ = chi2_contingency(table)

    n = table.to_numpy().sum()

    denominator = min(
        table.shape[0] - 1,
        table.shape[1] - 1,
    )

    if denominator <= 0:
        return float(p_value), np.nan

    cramers_v = np.sqrt(
        chi2 / (n * denominator)
    )

    return float(p_value), float(cramers_v)


def summarize_partition(
    labels: np.ndarray,
    sample_silhouettes: np.ndarray,
    y: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    unique_labels = np.sort(
        np.unique(labels)
    )

    records = []

    for cluster in unique_labels:
        mask = labels == cluster
        cluster_activity = y.to_numpy()[mask]

        records.append(
            {
                "cluster": int(cluster),
                "n_samples": int(mask.sum()),
                "sample_fraction": float(mask.mean()),
                "mean_silhouette": float(
                    sample_silhouettes[mask].mean()
                ),
                "median_silhouette": float(
                    np.median(
                        sample_silhouettes[mask]
                    )
                ),
                "minimum_silhouette": float(
                    sample_silhouettes[mask].min()
                ),
                "activity_1_fraction": float(
                    cluster_activity.mean()
                ),
                "activity_1_enrichment": float(
                    cluster_activity.mean()
                    - y.mean()
                ),
            }
        )

    cluster_summary = pd.DataFrame(records)

    minimum_fraction = float(
        cluster_summary["sample_fraction"].min()
    )

    maximum_fraction = float(
        cluster_summary["sample_fraction"].max()
    )

    minimum_mean_silhouette = float(
        cluster_summary["mean_silhouette"].min()
    )

    size_ratio = float(
        cluster_summary["n_samples"].max()
        / cluster_summary["n_samples"].min()
    )

    metadata = {
        "n_clusters": int(len(unique_labels)),
        "minimum_cluster_fraction": minimum_fraction,
        "maximum_cluster_fraction": maximum_fraction,
        "minimum_cluster_mean_silhouette": (
            minimum_mean_silhouette
        ),
        "cluster_size_ratio": size_ratio,
    }

    return cluster_summary, metadata


def partition_is_admissible(
    actual_n_clusters: int,
    requested_k: int,
    minimum_cluster_fraction: float,
    minimum_cluster_mean_silhouette: float,
) -> tuple[bool, str]:
    reasons = []

    if actual_n_clusters != requested_k:
        reasons.append(
            "actual number of clusters differs from requested k"
        )

    if minimum_cluster_fraction < MIN_CLUSTER_FRACTION:
        reasons.append(
            "at least one cluster is smaller than "
            f"{MIN_CLUSTER_FRACTION:.1%}"
        )

    if (
        REQUIRE_POSITIVE_MEAN_SILHOUETTE_PER_CLUSTER
        and minimum_cluster_mean_silhouette <= 0
    ):
        reasons.append(
            "at least one cluster has non-positive mean silhouette"
        )

    admissible = len(reasons) == 0

    return (
        admissible,
        "admissible" if admissible else "; ".join(reasons),
    )


# ============================================================
# DISTANCE REPRESENTATIONS
# ============================================================

def presence_jaccard_distance(
    X_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    binary = X_values > 0

    all_zero_rows = np.flatnonzero(
        binary.sum(axis=1) == 0
    )

    if all_zero_rows.size:
        warnings.warn(
            f"{all_zero_rows.size} samples have no positive features."
        )

    print("Computing condensed Jaccard distances...")

    condensed = pdist(
        binary,
        metric="jaccard",
    )

    if not np.isfinite(condensed).all():
        raise ValueError(
            "Non-finite Jaccard distances were produced."
        )

    square = squareform(
        condensed
    ).astype(
        np.float32,
        copy=False,
    )

    return condensed, square


def row_normalized_cosine_distance(
    X_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_norms = np.linalg.norm(
        X_values,
        axis=1,
    )

    zero_rows = np.flatnonzero(
        row_norms == 0
    )

    if zero_rows.size:
        warnings.warn(
            f"{zero_rows.size} samples have an all-zero numeric profile."
        )

    row_normalized = normalize(
        X_values,
        norm="l2",
        axis=1,
        copy=True,
    )

    print("Computing cosine distance matrix...")

    square = cosine_distances(
        row_normalized
    ).astype(
        np.float32,
        copy=False,
    )

    square = (
        square + square.T
    ) / 2.0

    np.fill_diagonal(
        square,
        0.0,
    )

    square = np.clip(
        square,
        0.0,
        2.0,
    )

    condensed = squareform(
        square,
        checks=False,
    ).astype(
        np.float64,
        copy=False,
    )

    return condensed, square, row_normalized


# ============================================================
# HIERARCHICAL CLUSTERING
# ============================================================

def evaluate_hierarchical_representation(
    representation_name: str,
    condensed_distance: np.ndarray,
    square_distance: np.ndarray,
    y: pd.Series,
    output_dir: Path,
) -> dict:
    print(
        f"Building {LINKAGE_METHOD}-linkage hierarchy "
        f"for {representation_name}..."
    )

    hierarchy = linkage(
        condensed_distance,
        method=LINKAGE_METHOD,
        optimal_ordering=False,
    )

    model_records = []
    labels_by_k: dict[int, np.ndarray] = {}
    cluster_summaries: dict[int, pd.DataFrame] = {}

    for requested_k in range(
        K_MIN,
        K_MAX + 1,
    ):
        labels = (
            fcluster(
                hierarchy,
                t=requested_k,
                criterion="maxclust",
            )
            - 1
        ).astype(int)

        actual_n_clusters = int(
            np.unique(labels).size
        )

        if actual_n_clusters < 2:
            continue

        sample_silhouettes = silhouette_samples(
            square_distance,
            labels,
            metric="precomputed",
        )

        overall_silhouette = float(
            sample_silhouettes.mean()
        )

        cluster_summary, metadata = summarize_partition(
            labels,
            sample_silhouettes,
            y,
        )

        admissible, reason = partition_is_admissible(
            actual_n_clusters=actual_n_clusters,
            requested_k=requested_k,
            minimum_cluster_fraction=metadata[
                "minimum_cluster_fraction"
            ],
            minimum_cluster_mean_silhouette=metadata[
                "minimum_cluster_mean_silhouette"
            ],
        )

        chi_square_p, cramers_v = cramers_v_from_labels(
            labels,
            y,
        )

        model_records.append(
            {
                "requested_k": requested_k,
                "actual_n_clusters": actual_n_clusters,
                "overall_silhouette": overall_silhouette,
                "minimum_cluster_fraction": metadata[
                    "minimum_cluster_fraction"
                ],
                "minimum_cluster_mean_silhouette": metadata[
                    "minimum_cluster_mean_silhouette"
                ],
                "cluster_size_ratio": metadata[
                    "cluster_size_ratio"
                ],
                "admissible": admissible,
                "admissibility_reason": reason,
                "activity_chi_square_p_value": (
                    chi_square_p
                ),
                "activity_cramers_v": cramers_v,
                "activity_nmi": float(
                    normalized_mutual_info_score(
                        y,
                        labels,
                    )
                ),
                "activity_ari": float(
                    adjusted_rand_score(
                        y,
                        labels,
                    )
                ),
            }
        )

        labels_by_k[requested_k] = labels
        cluster_summaries[
            requested_k
        ] = cluster_summary

        print(
            f"  k={requested_k}: "
            f"silhouette={overall_silhouette:.4f}, "
            f"min_fraction="
            f"{metadata['minimum_cluster_fraction']:.4f}, "
            f"min_cluster_silhouette="
            f"{metadata['minimum_cluster_mean_silhouette']:.4f}, "
            f"admissible={admissible}"
        )

    model_selection = pd.DataFrame(
        model_records
    )

    if model_selection.empty:
        raise RuntimeError(
            f"No valid partition was produced for {representation_name}."
        )

    model_selection.to_csv(
        output_dir / "model_selection.csv",
        index=False,
    )

    admissible_rows = model_selection[
        model_selection["admissible"]
    ]

    if not admissible_rows.empty:
        selected_row = admissible_rows.loc[
            admissible_rows[
                "overall_silhouette"
            ].idxmax()
        ]

        selected_is_admissible = True
    else:
        selected_row = model_selection.loc[
            model_selection[
                "overall_silhouette"
            ].idxmax()
        ]

        selected_is_admissible = False

    selected_k = int(
        selected_row["requested_k"]
    )

    selected_labels = labels_by_k[
        selected_k
    ]

    selected_summary = cluster_summaries[
        selected_k
    ]

    selected_summary.to_csv(
        output_dir / "selected_cluster_summary.csv",
        index=False,
    )

    assignments = pd.DataFrame(
        {
            "row_index": np.arange(
                len(selected_labels)
            ),
            "cluster": selected_labels,
            "Activity": y.to_numpy(),
        }
    )

    assignments.to_csv(
        output_dir / "selected_cluster_assignments.csv",
        index=False,
    )

    with open(
        output_dir / "selected_solution.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "representation": representation_name,
                "selected_k": selected_k,
                "selected_is_admissible": (
                    selected_is_admissible
                ),
                "overall_silhouette": float(
                    selected_row[
                        "overall_silhouette"
                    ]
                ),
                "minimum_cluster_fraction": float(
                    selected_row[
                        "minimum_cluster_fraction"
                    ]
                ),
                "minimum_cluster_mean_silhouette": float(
                    selected_row[
                        "minimum_cluster_mean_silhouette"
                    ]
                ),
                "admissibility_reason": str(
                    selected_row[
                        "admissibility_reason"
                    ]
                ),
            },
            file,
            indent=2,
        )

    # Silhouette versus k
    plt.figure(figsize=(9, 6))
    plt.plot(
        model_selection["requested_k"],
        model_selection["overall_silhouette"],
        marker="o",
        label="Overall silhouette",
    )
    plt.plot(
        model_selection["requested_k"],
        model_selection[
            "minimum_cluster_mean_silhouette"
        ],
        marker="o",
        label="Minimum cluster mean silhouette",
    )
    plt.axhline(
        0.0,
        linewidth=1.0,
    )
    plt.xlabel("Number of clusters k")
    plt.ylabel("Silhouette")
    plt.title(
        f"{representation_name}: silhouette diagnostics"
    )
    plt.legend()
    save_figure(
        output_dir / "silhouette_by_k.png"
    )

    # Minimum cluster fraction
    plt.figure(figsize=(9, 6))
    plt.plot(
        model_selection["requested_k"],
        model_selection[
            "minimum_cluster_fraction"
        ],
        marker="o",
    )
    plt.axhline(
        MIN_CLUSTER_FRACTION,
        linestyle="--",
        linewidth=1.2,
        label=(
            f"Acceptance threshold "
            f"({MIN_CLUSTER_FRACTION:.0%})"
        ),
    )
    plt.xlabel("Number of clusters k")
    plt.ylabel("Smallest cluster fraction")
    plt.title(
        f"{representation_name}: partition balance"
    )
    plt.legend()
    save_figure(
        output_dir / "minimum_cluster_fraction_by_k.png"
    )

    # Dendrogram without sample labels
    plt.figure(figsize=(15, 7))
    dendrogram(
        hierarchy,
        no_labels=True,
        color_threshold=None,
    )
    plt.xlabel("Samples")
    plt.ylabel("Linkage distance")
    plt.title(
        f"{representation_name}: average-linkage dendrogram"
    )
    save_figure(
        output_dir / "dendrogram.png"
    )

    return {
        "representation": representation_name,
        "hierarchy": hierarchy,
        "model_selection": model_selection,
        "selected_k": selected_k,
        "selected_labels": selected_labels,
        "selected_summary": selected_summary,
        "selected_is_admissible": (
            selected_is_admissible
        ),
        "selected_row": selected_row,
    }


# ============================================================
# FEATURE-SUBSAMPLING STABILITY
# ============================================================

def cluster_from_feature_subset(
    X_subset: np.ndarray,
    representation: str,
    k: int,
) -> np.ndarray:
    if representation == "presence_jaccard":
        binary = X_subset > 0

        condensed = pdist(
            binary,
            metric="jaccard",
        )

    elif representation == "row_normalized_cosine":
        row_normalized = normalize(
            X_subset,
            norm="l2",
            axis=1,
            copy=True,
        )

        square = cosine_distances(
            row_normalized
        )

        square = (
            square + square.T
        ) / 2.0

        np.fill_diagonal(
            square,
            0.0,
        )

        condensed = squareform(
            square,
            checks=False,
        )

    else:
        raise ValueError(
            f"Unknown representation: {representation}"
        )

    hierarchy = linkage(
        condensed,
        method=LINKAGE_METHOD,
        optimal_ordering=False,
    )

    labels = (
        fcluster(
            hierarchy,
            t=k,
            criterion="maxclust",
        )
        - 1
    ).astype(int)

    return labels


def evaluate_feature_subsampling_stability(
    X_values: np.ndarray,
    representation: str,
    baseline_labels: np.ndarray,
    selected_k: int,
    output_dir: Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        RANDOM_STATE
    )

    n_features = X_values.shape[1]

    subset_size = max(
        2,
        int(
            round(
                FEATURE_SUBSAMPLE_FRACTION
                * n_features
            )
        ),
    )

    records = []

    for run in range(
        N_STABILITY_RUNS
    ):
        print(
            f"  {representation}: stability run "
            f"{run + 1}/{N_STABILITY_RUNS}"
        )

        selected_columns = rng.choice(
            n_features,
            size=subset_size,
            replace=False,
        )

        labels = cluster_from_feature_subset(
            X_values[
                :,
                selected_columns
            ],
            representation=representation,
            k=selected_k,
        )

        records.append(
            {
                "run": run + 1,
                "representation": representation,
                "selected_feature_count": subset_size,
                "actual_n_clusters": int(
                    np.unique(labels).size
                ),
                "ari_against_full_feature_partition": float(
                    adjusted_rand_score(
                        baseline_labels,
                        labels,
                    )
                ),
                "nmi_against_full_feature_partition": float(
                    normalized_mutual_info_score(
                        baseline_labels,
                        labels,
                    )
                ),
            }
        )

        del labels
        gc.collect()

    report = pd.DataFrame(records)

    report.to_csv(
        output_dir
        / f"{representation}_stability.csv",
        index=False,
    )

    return report


# ============================================================
# DBSCAN ON ROW-NORMALIZED SVD REPRESENTATION
# ============================================================

def prepare_dbscan_representation(
    X_values: np.ndarray,
) -> tuple[np.ndarray, TruncatedSVD]:
    row_normalized = normalize(
        X_values,
        norm="l2",
        axis=1,
        copy=True,
    )

    n_components = min(
        SVD_COMPONENTS_FOR_DBSCAN,
        X_values.shape[1] - 1,
        X_values.shape[0] - 1,
    )

    svd = TruncatedSVD(
        n_components=n_components,
        algorithm="randomized",
        random_state=RANDOM_STATE,
    )

    reduced = svd.fit_transform(
        row_normalized
    )

    reduced = normalize(
        reduced,
        norm="l2",
        axis=1,
        copy=False,
    )

    return reduced, svd


def evaluate_dbscan(
    X_values: np.ndarray,
    y: pd.Series,
    output_dir: Path,
) -> dict:
    print(
        "Preparing row-normalized TruncatedSVD representation "
        "for DBSCAN..."
    )

    reduced, svd = prepare_dbscan_representation(
        X_values
    )

    pd.DataFrame(
        {
            "component": np.arange(
                1,
                len(
                    svd.explained_variance_ratio_
                )
                + 1
            ),
            "explained_variance_ratio": (
                svd.explained_variance_ratio_
            ),
            "cumulative_explained_variance": (
                np.cumsum(
                    svd.explained_variance_ratio_
                )
            ),
        }
    ).to_csv(
        output_dir / "svd_explained_variance.csv",
        index=False,
    )

    model_records = []
    labels_by_configuration: dict[
        tuple[int, float],
        np.ndarray,
    ] = {}

    for min_samples in (
        DBSCAN_MIN_SAMPLES_VALUES
    ):
        neighbour_model = NearestNeighbors(
            n_neighbors=min_samples,
            metric="euclidean",
            n_jobs=-1,
        )

        neighbour_model.fit(
            reduced
        )

        distances, _ = (
            neighbour_model.kneighbors(
                reduced
            )
        )

        kth_distances = np.sort(
            distances[:, -1]
        )

        plt.figure(figsize=(9, 6))
        plt.plot(
            np.arange(
                1,
                len(kth_distances) + 1
            ),
            kth_distances,
        )
        plt.xlabel("Samples ordered by k-distance")
        plt.ylabel(
            f"Distance to neighbour {min_samples}"
        )
        plt.title(
            f"DBSCAN k-distance curve, min_samples={min_samples}"
        )
        save_figure(
            output_dir
            / f"k_distance_min_samples_{min_samples}.png"
        )

        eps_candidates = np.unique(
            np.quantile(
                kth_distances,
                DBSCAN_EPS_QUANTILES,
            )
        )

        for eps in eps_candidates:
            model = DBSCAN(
                eps=float(eps),
                min_samples=min_samples,
                metric="euclidean",
                n_jobs=-1,
            )

            labels = model.fit_predict(
                reduced
            )

            non_noise_mask = (
                labels != -1
            )

            cluster_labels = np.unique(
                labels[
                    non_noise_mask
                ]
            )

            n_clusters = int(
                len(cluster_labels)
            )

            noise_fraction = float(
                np.mean(
                    ~non_noise_mask
                )
            )

            non_noise_count = int(
                non_noise_mask.sum()
            )

            if n_clusters > 0:
                cluster_sizes = (
                    pd.Series(
                        labels[
                            non_noise_mask
                        ]
                    )
                    .value_counts()
                )

                minimum_cluster_size = int(
                    cluster_sizes.min()
                )

                maximum_cluster_size = int(
                    cluster_sizes.max()
                )
            else:
                minimum_cluster_size = 0
                maximum_cluster_size = 0

            overall_silhouette = np.nan
            minimum_cluster_mean_silhouette = (
                np.nan
            )
            activity_nmi = np.nan
            activity_ari = np.nan
            activity_cramers_v = np.nan
            activity_chi_square_p = np.nan

            if (
                n_clusters >= 2
                and non_noise_count > n_clusters
            ):
                non_noise_labels = labels[
                    non_noise_mask
                ]

                sample_silhouettes = silhouette_samples(
                    reduced[
                        non_noise_mask
                    ],
                    non_noise_labels,
                    metric="euclidean",
                )

                overall_silhouette = float(
                    sample_silhouettes.mean()
                )

                cluster_mean_silhouettes = []

                for cluster_label in cluster_labels:
                    cluster_mean_silhouettes.append(
                        float(
                            sample_silhouettes[
                                non_noise_labels
                                == cluster_label
                            ].mean()
                        )
                    )

                minimum_cluster_mean_silhouette = float(
                    min(
                        cluster_mean_silhouettes
                    )
                )

                activity_subset = y[
                    non_noise_mask
                ].reset_index(drop=True)

                activity_nmi = float(
                    normalized_mutual_info_score(
                        activity_subset,
                        non_noise_labels,
                    )
                )

                activity_ari = float(
                    adjusted_rand_score(
                        activity_subset,
                        non_noise_labels,
                    )
                )

                (
                    activity_chi_square_p,
                    activity_cramers_v,
                ) = cramers_v_from_labels(
                    non_noise_labels,
                    activity_subset,
                )

            admissible = (
                2 <= n_clusters
                <= DBSCAN_MAX_NUMBER_OF_CLUSTERS
                and noise_fraction
                <= DBSCAN_MAX_NOISE_FRACTION
                and minimum_cluster_size
                >= DBSCAN_MIN_CLUSTER_SIZE
                and np.isfinite(
                    overall_silhouette
                )
                and minimum_cluster_mean_silhouette
                > 0
            )

            model_records.append(
                {
                    "min_samples": min_samples,
                    "eps": float(eps),
                    "n_clusters": n_clusters,
                    "noise_fraction": noise_fraction,
                    "non_noise_count": non_noise_count,
                    "minimum_cluster_size": (
                        minimum_cluster_size
                    ),
                    "maximum_cluster_size": (
                        maximum_cluster_size
                    ),
                    "overall_silhouette_non_noise": (
                        overall_silhouette
                    ),
                    "minimum_cluster_mean_silhouette_non_noise": (
                        minimum_cluster_mean_silhouette
                    ),
                    "admissible": admissible,
                    "activity_nmi_non_noise": (
                        activity_nmi
                    ),
                    "activity_ari_non_noise": (
                        activity_ari
                    ),
                    "activity_chi_square_p_value_non_noise": (
                        activity_chi_square_p
                    ),
                    "activity_cramers_v_non_noise": (
                        activity_cramers_v
                    ),
                }
            )

            labels_by_configuration[
                (
                    min_samples,
                    float(eps),
                )
            ] = labels

            print(
                f"  DBSCAN min_samples={min_samples}, "
                f"eps={eps:.5f}: "
                f"clusters={n_clusters}, "
                f"noise={noise_fraction:.3f}, "
                f"silhouette={overall_silhouette}, "
                f"admissible={admissible}"
            )

    report = pd.DataFrame(
        model_records
    )

    report.to_csv(
        output_dir / "dbscan_parameter_sweep.csv",
        index=False,
    )

    admissible_rows = report[
        report["admissible"]
    ]

    selected_configuration = None

    if not admissible_rows.empty:
        selected_row = admissible_rows.loc[
            admissible_rows[
                "overall_silhouette_non_noise"
            ].idxmax()
        ]

        selected_key = (
            int(
                selected_row[
                    "min_samples"
                ]
            ),
            float(
                selected_row["eps"]
            ),
        )

        selected_labels = (
            labels_by_configuration[
                selected_key
            ]
        )

        selected_configuration = {
            key: (
                bool(value)
                if isinstance(
                    value,
                    (np.bool_, bool),
                )
                else int(value)
                if isinstance(
                    value,
                    (np.integer,),
                )
                else float(value)
                if isinstance(
                    value,
                    (np.floating,),
                )
                else value
            )
            for key, value in selected_row.to_dict().items()
        }

        pd.DataFrame(
            {
                "row_index": np.arange(
                    len(selected_labels)
                ),
                "dbscan_cluster": (
                    selected_labels
                ),
                "is_noise": (
                    selected_labels == -1
                ),
                "Activity": y.to_numpy(),
            }
        ).to_csv(
            output_dir
            / "selected_dbscan_assignments.csv",
            index=False,
        )

        cluster_records = []

        for cluster in np.sort(
            np.unique(
                selected_labels[
                    selected_labels != -1
                ]
            )
        ):
            mask = selected_labels == cluster

            cluster_records.append(
                {
                    "cluster": int(cluster),
                    "n_samples": int(
                        mask.sum()
                    ),
                    "sample_fraction_total": float(
                        mask.mean()
                    ),
                    "activity_1_fraction": float(
                        y.to_numpy()[
                            mask
                        ].mean()
                    ),
                }
            )

        pd.DataFrame(
            cluster_records
        ).to_csv(
            output_dir
            / "selected_dbscan_cluster_summary.csv",
            index=False,
        )

    with open(
        output_dir / "selected_dbscan_solution.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "selected_configuration": (
                    selected_configuration
                ),
                "svd_components": int(
                    reduced.shape[1]
                ),
                "svd_retained_variance": float(
                    svd.explained_variance_ratio_.sum()
                ),
            },
            file,
            indent=2,
        )

    return {
        "report": report,
        "selected_configuration": (
            selected_configuration
        ),
    }


# ============================================================
# FINAL COMPARISON AND REPORT
# ============================================================

def compare_selected_partitions(
    presence_result: dict,
    cosine_result: dict,
    output_dir: Path,
) -> pd.DataFrame:
    presence_labels = presence_result[
        "selected_labels"
    ]

    cosine_labels = cosine_result[
        "selected_labels"
    ]

    comparison = pd.DataFrame(
        [
            {
                "partition_1": (
                    "presence_jaccard"
                ),
                "partition_2": (
                    "row_normalized_cosine"
                ),
                "ari": float(
                    adjusted_rand_score(
                        presence_labels,
                        cosine_labels,
                    )
                ),
                "nmi": float(
                    normalized_mutual_info_score(
                        presence_labels,
                        cosine_labels,
                    )
                ),
            }
        ]
    )

    comparison.to_csv(
        output_dir
        / "selected_partition_comparison.csv",
        index=False,
    )

    return comparison


def write_final_summary(
    X: pd.DataFrame,
    presence_result: dict,
    cosine_result: dict,
    comparison: pd.DataFrame,
    stability_reports: list[pd.DataFrame],
    dbscan_result: dict | None,
    output_path: Path,
) -> None:
    lines = [
        "FINAL NUMERIC CLUSTERING CHECK",
        "=" * 44,
        "",
        f"Samples: {X.shape[0]}",
        f"Numeric features: {X.shape[1]}",
        "",
    ]

    for title, result in [
        (
            "PRESENCE / JACCARD",
            presence_result,
        ),
        (
            "ROW-NORMALIZED PROFILE / COSINE",
            cosine_result,
        ),
    ]:
        row = result[
            "selected_row"
        ]

        lines.extend(
            [
                title,
                "-" * len(title),
                (
                    "Selected k: "
                    f"{result['selected_k']}"
                ),
                (
                    "Admissible under the predefined criteria: "
                    f"{result['selected_is_admissible']}"
                ),
                (
                    "Overall silhouette: "
                    f"{float(row['overall_silhouette']):.6f}"
                ),
                (
                    "Minimum cluster fraction: "
                    f"{float(row['minimum_cluster_fraction']):.6f}"
                ),
                (
                    "Minimum cluster mean silhouette: "
                    f"{float(row['minimum_cluster_mean_silhouette']):.6f}"
                ),
                (
                    "Cluster-size ratio: "
                    f"{float(row['cluster_size_ratio']):.6f}"
                ),
                (
                    "Activity Cramer's V: "
                    f"{float(row['activity_cramers_v']):.6f}"
                ),
                (
                    "Activity NMI: "
                    f"{float(row['activity_nmi']):.6f}"
                ),
                (
                    "Activity ARI: "
                    f"{float(row['activity_ari']):.6f}"
                ),
                (
                    "Reason/status: "
                    f"{row['admissibility_reason']}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "PARTITION AGREEMENT",
            "-" * 20,
            (
                "ARI between selected Jaccard and cosine partitions: "
                f"{comparison.iloc[0]['ari']:.6f}"
            ),
            (
                "NMI between selected Jaccard and cosine partitions: "
                f"{comparison.iloc[0]['nmi']:.6f}"
            ),
            "",
        ]
    )

    if stability_reports:
        lines.extend(
            [
                "FEATURE-SUBSAMPLING STABILITY",
                "-" * 30,
            ]
        )

        combined = pd.concat(
            stability_reports,
            ignore_index=True,
        )

        for representation, group in combined.groupby(
            "representation"
        ):
            lines.append(
                f"{representation}: "
                f"mean ARI={group['ari_against_full_feature_partition'].mean():.6f}, "
                f"minimum ARI={group['ari_against_full_feature_partition'].min():.6f}, "
                f"mean NMI={group['nmi_against_full_feature_partition'].mean():.6f}"
            )

        lines.append("")

    lines.extend(
        [
            "DBSCAN",
            "-" * 20,
        ]
    )

    if (
        dbscan_result is None
        or dbscan_result[
            "selected_configuration"
        ]
        is None
    ):
        lines.append(
            "No DBSCAN configuration satisfied all predefined "
            "requirements."
        )
    else:
        selected = dbscan_result[
            "selected_configuration"
        ]

        lines.extend(
            [
                "An admissible DBSCAN configuration was found.",
                (
                    "min_samples: "
                    f"{selected['min_samples']}"
                ),
                (
                    "eps: "
                    f"{selected['eps']:.6f}"
                ),
                (
                    "clusters: "
                    f"{selected['n_clusters']}"
                ),
                (
                    "noise fraction: "
                    f"{selected['noise_fraction']:.6f}"
                ),
                (
                    "minimum cluster size: "
                    f"{selected['minimum_cluster_size']}"
                ),
                (
                    "silhouette on non-noise samples: "
                    f"{selected['overall_silhouette_non_noise']:.6f}"
                ),
                (
                    "minimum cluster mean silhouette: "
                    f"{selected['minimum_cluster_mean_silhouette_non_noise']:.6f}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "DECISION RULE",
            "-" * 20,
            (
                "A sample clustering should be retained only if it is "
                "admissible, stable under feature subsampling and not "
                "specific to a single representation."
            ),
            (
                "If neither Jaccard nor cosine produces an admissible "
                "and stable result, and DBSCAN has no admissible "
                "configuration, the defensible conclusion is that the "
                "numeric descriptors do not exhibit robust global "
                "sample clusters."
            ),
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    directories = create_directories()

    X, y = load_data()
    X_values = X.to_numpy(dtype=float)

    # --------------------------------------------------------
    # 1. Presence/Jaccard
    # --------------------------------------------------------

    print("\n1/5 Presence/Jaccard clustering")

    (
        jaccard_condensed,
        jaccard_square,
    ) = presence_jaccard_distance(
        X_values
    )

    presence_result = (
        evaluate_hierarchical_representation(
            representation_name=(
                "presence_jaccard"
            ),
            condensed_distance=(
                jaccard_condensed
            ),
            square_distance=jaccard_square,
            y=y,
            output_dir=directories[
                "presence"
            ],
        )
    )

    del jaccard_condensed
    del jaccard_square
    gc.collect()

    # --------------------------------------------------------
    # 2. Row-normalized profile/Cosine
    # --------------------------------------------------------

    print(
        "\n2/5 Row-normalized profile/Cosine clustering"
    )

    (
        cosine_condensed,
        cosine_square,
        row_normalized,
    ) = row_normalized_cosine_distance(
        X_values
    )

    cosine_result = (
        evaluate_hierarchical_representation(
            representation_name=(
                "row_normalized_cosine"
            ),
            condensed_distance=(
                cosine_condensed
            ),
            square_distance=cosine_square,
            y=y,
            output_dir=directories[
                "cosine"
            ],
        )
    )

    del cosine_condensed
    del cosine_square
    del row_normalized
    gc.collect()

    # --------------------------------------------------------
    # 3. Compare selected partitions
    # --------------------------------------------------------

    print("\n3/5 Comparing selected partitions")

    comparison = compare_selected_partitions(
        presence_result,
        cosine_result,
        directories["comparison"],
    )

    # --------------------------------------------------------
    # 4. Stability under feature subsampling
    # --------------------------------------------------------

    stability_reports = []

    if RUN_FEATURE_SUBSAMPLING:
        print(
            "\n4/5 Feature-subsampling stability"
        )

        stability_reports.append(
            evaluate_feature_subsampling_stability(
                X_values=X_values,
                representation=(
                    "presence_jaccard"
                ),
                baseline_labels=presence_result[
                    "selected_labels"
                ],
                selected_k=presence_result[
                    "selected_k"
                ],
                output_dir=directories[
                    "stability"
                ],
            )
        )

        stability_reports.append(
            evaluate_feature_subsampling_stability(
                X_values=X_values,
                representation=(
                    "row_normalized_cosine"
                ),
                baseline_labels=cosine_result[
                    "selected_labels"
                ],
                selected_k=cosine_result[
                    "selected_k"
                ],
                output_dir=directories[
                    "stability"
                ],
            )
        )
    else:
        print(
            "\n4/5 Feature-subsampling stability skipped"
        )

    # --------------------------------------------------------
    # 5. Controlled DBSCAN
    # --------------------------------------------------------

    dbscan_result = None

    if RUN_DBSCAN:
        print("\n5/5 Controlled DBSCAN sweep")

        dbscan_result = evaluate_dbscan(
            X_values=X_values,
            y=y,
            output_dir=directories[
                "dbscan"
            ],
        )
    else:
        print("\n5/5 DBSCAN skipped")

    write_final_summary(
        X=X,
        presence_result=presence_result,
        cosine_result=cosine_result,
        comparison=comparison,
        stability_reports=stability_reports,
        dbscan_result=dbscan_result,
        output_path=(
            directories["root"]
            / "final_clustering_summary.txt"
        ),
    )

    print("\nFinal clustering check completed.")
    print(
        f"Results saved in: {OUTPUT_DIR}"
    )
    print(
        "Start from final_clustering_summary.txt."
    )


if __name__ == "__main__":
    main()
