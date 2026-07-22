#!/usr/bin/env python3
"""
Verifica se il piccolo cluster K-means rappresenta:

1. un sottotipo molecolare stabile e coeso;
2. un insieme di outlier multivariati;
3. un artefatto della standardizzazione di feature numeriche sparse.

Lo script usa come riferimento le assegnazioni K-means già prodotte
dall'analisi precedente: cluster 0 = grande gruppo, cluster 1 = 73 campioni.

Analisi
-------
A. Persistenza del piccolo cluster sotto rappresentazioni alternative:
   - StandardScaler + PCA 90%  [controllo]
   - dati raw + PCA 90%
   - RobustScaler + PCA 90%
   - log1p + StandardScaler + PCA 90%
   - presenza/assenza (X > 0) + StandardScaler + PCA 90%

B. Capacità di singole quantità globali di predire il cluster:
   - PC1
   - norme L1/L2
   - somma e media raw
   - numero/frazione di valori non nulli
   Se una sola quantità ottiene AUC quasi 1, il clustering è
   sostanzialmente una separazione per magnitudine/sparsità.

C. Outlier detection nello spazio PCA standardizzato:
   - Isolation Forest
   - Local Outlier Factor
   - distanza di Mahalanobis regolarizzata
   Tutti i metodi selezionano lo stesso numero di campioni del piccolo cluster.

D. Coesione del piccolo cluster:
   - purezza dei k-nearest neighbours;
   - distanza media interna;
   - distanza verso il grande cluster;
   - silhouette media per cluster.

E. Stabilità rispetto a sottoinsiemi casuali di feature:
   si ripete PCA + K-means usando sottoinsiemi casuali delle feature e si
   misura quanto spesso ciascuna molecola viene nuovamente assegnata al
   piccolo cluster.

La diagnosi automatica finale è euristica e viene esplicitamente indicata
come tale. I dati quantitativi rimangono il risultato principale.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    adjusted_rand_score,
    roc_auc_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler


# ============================================================
# CONFIGURAZIONE
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

REPORT_ROOT_CANDIDATES = [
    (
        BASE_DIR
        / "models"
        / "numerico"
        / "numeric_feature_exploration"
        / "reports"
    ),
    (
        BASE_DIR
        / "models"
        / "numerico"
        / "analysis"
        / "reports"
    ),
]

OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "numerico"
    / "analysis"
    / "subtype_vs_outliers"
)

RANDOM_STATE = 42
KMEANS_N_INIT = 30
PCA_VARIANCE_THRESHOLD = 0.90

# Coesione locale
N_NEIGHBORS = 10

# Stabilità tramite sottoinsiemi casuali di feature
N_FEATURE_SUBSAMPLING_RUNS = 20
FEATURE_SUBSAMPLE_FRACTION = 0.70
SUBSAMPLE_PCA_COMPONENTS = 100

# Grafici
DPI = 180


# ============================================================
# UTILITÀ
# ============================================================

def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close()


def find_report_root() -> Path:
    for candidate in REPORT_ROOT_CANDIDATES:
        assignment_path = (
            candidate
            / "pca_clustering"
            / "kmeans_cluster_assignments.csv"
        )

        score_path = (
            candidate
            / "pca_clustering"
            / "pca_scores.csv"
        )

        if assignment_path.exists() and score_path.exists():
            return candidate

    searched = "\n".join(
        f"- {path}"
        for path in REPORT_ROOT_CANDIDATES
    )

    raise FileNotFoundError(
        "Non trovo i risultati dell'analisi precedente.\n"
        f"Percorsi controllati:\n{searched}"
    )


def normalize_two_cluster_labels(
    labels: np.ndarray,
) -> np.ndarray:
    """
    Rietichetta la soluzione affinché:
    cluster 1 = cluster meno numeroso;
    cluster 0 = cluster più numeroso.
    """
    labels = np.asarray(labels)
    unique_labels, counts = np.unique(
        labels,
        return_counts=True,
    )

    if len(unique_labels) != 2:
        raise ValueError(
            "Sono richiesti esattamente due cluster."
        )

    small_original_label = unique_labels[
        np.argmin(counts)
    ]

    return (
        labels == small_original_label
    ).astype(int)


def jaccard_binary(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)

    union = np.logical_or(
        first,
        second,
    ).sum()

    if union == 0:
        return 1.0

    intersection = np.logical_and(
        first,
        second,
    ).sum()

    return float(intersection / union)


def safe_auc(
    labels: np.ndarray,
    values: np.ndarray,
) -> float:
    values = np.asarray(values, dtype=float)

    if np.all(values == values[0]):
        return 0.5

    auc = float(
        roc_auc_score(
            labels,
            values,
        )
    )

    # Interessa la capacità discriminante indipendentemente dal verso.
    return max(auc, 1.0 - auc)


def fit_pca_kmeans(
    representation: np.ndarray,
    representation_name: str,
) -> dict:
    pca = PCA(
        n_components=PCA_VARIANCE_THRESHOLD,
        svd_solver="full",
        random_state=RANDOM_STATE,
    )

    transformed = pca.fit_transform(
        representation
    )

    model = KMeans(
        n_clusters=2,
        n_init=KMEANS_N_INIT,
        random_state=RANDOM_STATE,
    )

    labels = normalize_two_cluster_labels(
        model.fit_predict(transformed)
    )

    silhouette = float(
        silhouette_score(
            transformed,
            labels,
        )
    )

    return {
        "name": representation_name,
        "scores": transformed,
        "labels": labels,
        "n_components": int(
            transformed.shape[1]
        ),
        "retained_variance": float(
            pca.explained_variance_ratio_.sum()
        ),
        "silhouette": silhouette,
        "small_cluster_size": int(
            labels.sum()
        ),
    }


# ============================================================
# CARICAMENTO
# ============================================================

def load_inputs() -> dict:
    report_root = find_report_root()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    subdirectories = {
        "root": OUTPUT_DIR,
        "preprocessing": OUTPUT_DIR / "preprocessing_stability",
        "scalar": OUTPUT_DIR / "scalar_predictors",
        "outliers": OUTPUT_DIR / "outlier_detection",
        "cohesion": OUTPUT_DIR / "cluster_cohesion",
        "subsampling": OUTPUT_DIR / "feature_subsampling",
    }

    for directory in subdirectories.values():
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    X = pd.read_csv(DATA_PATH)

    if "Activity" in X.columns:
        X = X.drop(
            columns=["Activity"]
        )

    if X.isna().any().any():
        raise ValueError(
            "Il dataset numerico contiene valori mancanti."
        )

    target_df = pd.read_csv(
        TARGET_PATH
    )

    if "Activity" in target_df.columns:
        y = target_df["Activity"]
    else:
        y = target_df.iloc[:, 0]

    assignments = pd.read_csv(
        report_root
        / "pca_clustering"
        / "kmeans_cluster_assignments.csv"
    ).sort_values("row_index")

    baseline_labels = normalize_two_cluster_labels(
        assignments[
            "kmeans_cluster"
        ].to_numpy()
    )

    pca_scores_df = pd.read_csv(
        report_root
        / "pca_clustering"
        / "pca_scores.csv"
    ).sort_values("row_index")

    pc_columns = [
        column
        for column in pca_scores_df.columns
        if column.startswith("PC")
    ]

    baseline_pca_scores = pca_scores_df[
        pc_columns
    ].to_numpy(dtype=float)

    if not (
        len(X)
        == len(y)
        == len(baseline_labels)
        == len(baseline_pca_scores)
    ):
        raise ValueError(
            "Numero di righe incoerente tra gli input."
        )

    print(f"Numeric dataset: {X.shape}")
    print(
        "Baseline cluster sizes: "
        f"{pd.Series(baseline_labels).value_counts().sort_index().to_dict()}"
    )
    print(
        f"Baseline small-cluster fraction: "
        f"{baseline_labels.mean():.4%}"
    )

    return {
        "X": X,
        "y": pd.Series(
            y.to_numpy(),
            name="Activity",
        ).astype(int),
        "baseline_labels": baseline_labels,
        "baseline_pca_scores": baseline_pca_scores,
        "pc_columns": pc_columns,
        "dirs": subdirectories,
    }


# ============================================================
# A. STABILITÀ RISPETTO AL PREPROCESSING
# ============================================================

def preprocessing_stability(
    X: pd.DataFrame,
    baseline_labels: np.ndarray,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    raw = X.to_numpy(dtype=float)

    representations = {
        "standard_scaler": StandardScaler().fit_transform(
            raw
        ),
        "raw_unscaled": raw,
        "robust_scaler": RobustScaler(
            quantile_range=(25.0, 75.0),
        ).fit_transform(raw),
        "log1p_standard_scaler": StandardScaler().fit_transform(
            np.log1p(raw)
        ),
        "presence_standard_scaler": StandardScaler().fit_transform(
            (raw > 0).astype(float)
        ),
    }

    records = []
    labels_by_method: dict[str, np.ndarray] = {}

    for name, representation in representations.items():
        print(
            f"  preprocessing: {name}"
        )

        result = fit_pca_kmeans(
            representation,
            name,
        )

        labels = result["labels"]
        labels_by_method[name] = labels

        small_mask = labels == 1
        baseline_small_mask = (
            baseline_labels == 1
        )

        overlap_count = int(
            np.logical_and(
                small_mask,
                baseline_small_mask,
            ).sum()
        )

        precision = (
            overlap_count
            / max(int(small_mask.sum()), 1)
        )

        recall = (
            overlap_count
            / max(
                int(
                    baseline_small_mask.sum()
                ),
                1,
            )
        )

        records.append(
            {
                "representation": name,
                "n_components": result[
                    "n_components"
                ],
                "retained_variance": result[
                    "retained_variance"
                ],
                "silhouette": result[
                    "silhouette"
                ],
                "small_cluster_size": result[
                    "small_cluster_size"
                ],
                "overlap_with_baseline_count": (
                    overlap_count
                ),
                "jaccard_with_baseline_small_cluster": (
                    jaccard_binary(
                        baseline_small_mask,
                        small_mask,
                    )
                ),
                "precision_against_baseline_small_cluster": (
                    precision
                ),
                "recall_of_baseline_small_cluster": (
                    recall
                ),
                "ari_against_baseline_partition": float(
                    adjusted_rand_score(
                        baseline_labels,
                        labels,
                    )
                ),
            }
        )

    report = pd.DataFrame(
        records
    ).sort_values(
        "jaccard_with_baseline_small_cluster",
        ascending=False,
    )

    report.to_csv(
        output_dir
        / "preprocessing_stability.csv",
        index=False,
    )

    plot_data = report.sort_values(
        "jaccard_with_baseline_small_cluster"
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        plot_data["representation"],
        plot_data[
            "jaccard_with_baseline_small_cluster"
        ],
    )
    plt.xlabel(
        "Jaccard overlap with baseline 73-sample cluster"
    )
    plt.ylabel("Representation")
    plt.title(
        "Persistence of the small cluster across preprocessing choices"
    )
    save_figure(
        output_dir
        / "preprocessing_jaccard_overlap.png"
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        plot_data["representation"],
        plot_data["silhouette"],
    )
    plt.xlabel("Silhouette")
    plt.ylabel("Representation")
    plt.title(
        "K-means silhouette across preprocessing choices"
    )
    save_figure(
        output_dir
        / "preprocessing_silhouette.png"
    )

    membership_table = pd.DataFrame(
        {
            "row_index": np.arange(
                len(X)
            ),
            "baseline_small_cluster": (
                baseline_labels
            ),
        }
    )

    for method, labels in labels_by_method.items():
        membership_table[
            f"small_cluster_{method}"
        ] = labels

    membership_table.to_csv(
        output_dir
        / "preprocessing_memberships.csv",
        index=False,
    )

    return report, labels_by_method


# ============================================================
# B. PREDITTORI SCALARI
# ============================================================

def scalar_predictors(
    X: pd.DataFrame,
    baseline_labels: np.ndarray,
    baseline_pca_scores: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    raw = X.to_numpy(dtype=float)
    standardized = StandardScaler().fit_transform(
        raw
    )

    nonzero_count = (
        raw != 0
    ).sum(axis=1)

    metrics = {
        "PC1": baseline_pca_scores[:, 0],
        "PC2": baseline_pca_scores[:, 1],
        "raw_sum": raw.sum(axis=1),
        "raw_mean": raw.mean(axis=1),
        "raw_l1_norm": np.abs(raw).sum(axis=1),
        "raw_l2_norm": np.linalg.norm(
            raw,
            axis=1,
        ),
        "standardized_l1_norm": np.abs(
            standardized
        ).sum(axis=1),
        "standardized_l2_norm": np.linalg.norm(
            standardized,
            axis=1,
        ),
        "nonzero_count": nonzero_count,
        "nonzero_fraction": (
            nonzero_count
            / raw.shape[1]
        ),
        "zero_fraction": (
            raw == 0
        ).mean(axis=1),
        "raw_maximum": raw.max(axis=1),
    }

    records = []

    for metric_name, values in metrics.items():
        values = np.asarray(
            values,
            dtype=float,
        )

        records.append(
            {
                "metric": metric_name,
                "auc_absolute_direction": safe_auc(
                    baseline_labels,
                    values,
                ),
                "cluster_0_mean": float(
                    values[
                        baseline_labels == 0
                    ].mean()
                ),
                "cluster_1_mean": float(
                    values[
                        baseline_labels == 1
                    ].mean()
                ),
                "mean_difference_1_minus_0": float(
                    values[
                        baseline_labels == 1
                    ].mean()
                    - values[
                        baseline_labels == 0
                    ].mean()
                ),
            }
        )

    report = pd.DataFrame(
        records
    ).sort_values(
        "auc_absolute_direction",
        ascending=False,
    )

    report.to_csv(
        output_dir
        / "scalar_predictor_auc.csv",
        index=False,
    )

    plot_data = report.sort_values(
        "auc_absolute_direction"
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        plot_data["metric"],
        plot_data[
            "auc_absolute_direction"
        ],
    )
    plt.axvline(
        0.5,
        linestyle="--",
        linewidth=1.0,
    )
    plt.xlabel(
        "AUC for predicting membership in the 73-sample cluster"
    )
    plt.ylabel("Single scalar property")
    plt.title(
        "Can one global quantity reproduce the K-means split?"
    )
    save_figure(
        output_dir
        / "scalar_predictor_auc.png"
    )

    return report


# ============================================================
# C. OUTLIER DETECTION
# ============================================================

def outlier_detection(
    baseline_pca_scores: np.ndarray,
    baseline_labels: np.ndarray,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_samples = len(
        baseline_labels
    )

    n_outliers = int(
        baseline_labels.sum()
    )

    contamination = (
        n_outliers / n_samples
    )

    # Si limita la dimensionalità per stabilità dei detector.
    detector_dimension = min(
        30,
        baseline_pca_scores.shape[1],
    )

    scores = baseline_pca_scores[
        :,
        :detector_dimension,
    ]

    isolation_forest = IsolationForest(
        n_estimators=500,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    isolation_predictions = (
        isolation_forest.fit_predict(
            scores
        )
        == -1
    )

    isolation_score = (
        -isolation_forest.score_samples(
            scores
        )
    )

    lof = LocalOutlierFactor(
        n_neighbors=35,
        contamination=contamination,
    )

    lof_predictions = (
        lof.fit_predict(scores)
        == -1
    )

    lof_score = (
        -lof.negative_outlier_factor_
    )

    covariance = LedoitWolf().fit(
        scores
    )

    mahalanobis_score = covariance.mahalanobis(
        scores
    )

    mahalanobis_threshold = np.partition(
        mahalanobis_score,
        -n_outliers,
    )[-n_outliers]

    mahalanobis_predictions = (
        mahalanobis_score
        >= mahalanobis_threshold
    )

    detector_table = pd.DataFrame(
        {
            "row_index": np.arange(
                n_samples
            ),
            "baseline_small_cluster": (
                baseline_labels
            ),
            "isolation_forest_flag": (
                isolation_predictions.astype(int)
            ),
            "isolation_forest_score": (
                isolation_score
            ),
            "lof_flag": (
                lof_predictions.astype(int)
            ),
            "lof_score": lof_score,
            "mahalanobis_flag": (
                mahalanobis_predictions.astype(int)
            ),
            "mahalanobis_score": (
                mahalanobis_score
            ),
        }
    )

    detector_table[
        "outlier_consensus_count"
    ] = (
        detector_table[
            [
                "isolation_forest_flag",
                "lof_flag",
                "mahalanobis_flag",
            ]
        ]
        .sum(axis=1)
    )

    detector_table.to_csv(
        output_dir
        / "sample_outlier_flags.csv",
        index=False,
    )

    detector_records = []

    for detector_name, flag_column in [
        (
            "IsolationForest",
            "isolation_forest_flag",
        ),
        (
            "LocalOutlierFactor",
            "lof_flag",
        ),
        (
            "Mahalanobis_LedoitWolf",
            "mahalanobis_flag",
        ),
    ]:
        detector_flags = (
            detector_table[
                flag_column
            ].to_numpy()
            == 1
        )

        baseline_flags = (
            baseline_labels
            == 1
        )

        overlap = int(
            np.logical_and(
                detector_flags,
                baseline_flags,
            ).sum()
        )

        detector_records.append(
            {
                "detector": detector_name,
                "flagged_samples": int(
                    detector_flags.sum()
                ),
                "overlap_with_baseline_73": (
                    overlap
                ),
                "recall_of_baseline_73": (
                    overlap
                    / max(
                        int(
                            baseline_flags.sum()
                        ),
                        1,
                    )
                ),
                "precision_against_baseline_73": (
                    overlap
                    / max(
                        int(
                            detector_flags.sum()
                        ),
                        1,
                    )
                ),
                "jaccard_with_baseline_73": (
                    jaccard_binary(
                        baseline_flags,
                        detector_flags,
                    )
                ),
            }
        )

    summary = pd.DataFrame(
        detector_records
    ).sort_values(
        "jaccard_with_baseline_73",
        ascending=False,
    )

    summary.to_csv(
        output_dir
        / "outlier_detector_overlap.csv",
        index=False,
    )

    consensus_by_cluster = (
        detector_table
        .groupby(
            "baseline_small_cluster"
        )[
            "outlier_consensus_count"
        ]
        .value_counts(
            normalize=True
        )
        .rename(
            "fraction"
        )
        .reset_index()
    )

    consensus_by_cluster.to_csv(
        output_dir
        / "outlier_consensus_by_cluster.csv",
        index=False,
    )

    plt.figure(figsize=(9, 6))

    for cluster in (0, 1):
        values = detector_table.loc[
            detector_table[
                "baseline_small_cluster"
            ]
            == cluster,
            "outlier_consensus_count",
        ]

        plt.hist(
            values,
            bins=np.arange(-0.5, 4.5, 1),
            alpha=0.55,
            label=f"Baseline cluster {cluster}",
        )

    plt.xticks([0, 1, 2, 3])
    plt.xlabel(
        "Number of outlier detectors flagging the sample"
    )
    plt.ylabel("Number of samples")
    plt.title(
        "Outlier-detector consensus by baseline K-means cluster"
    )
    plt.legend()
    save_figure(
        output_dir
        / "outlier_consensus_histogram.png"
    )

    return summary, detector_table


# ============================================================
# D. COESIONE DEL PICCOLO CLUSTER
# ============================================================

def cluster_cohesion(
    baseline_pca_scores: np.ndarray,
    baseline_labels: np.ndarray,
    output_dir: Path,
) -> dict:
    scores = baseline_pca_scores
    small_mask = (
        baseline_labels == 1
    )
    large_mask = (
        baseline_labels == 0
    )

    # Purezza dei vicini.
    neighbour_count = min(
        N_NEIGHBORS + 1,
        len(scores),
    )

    nearest_neighbours = NearestNeighbors(
        n_neighbors=neighbour_count,
        metric="euclidean",
    ).fit(scores)

    neighbour_indices = (
        nearest_neighbours.kneighbors(
            scores,
            return_distance=False,
        )[:, 1:]
    )

    neighbour_purity = (
        baseline_labels[
            neighbour_indices
        ]
        == baseline_labels[:, None]
    ).mean(axis=1)

    small_cluster_neighbour_purity = (
        baseline_labels[
            neighbour_indices[
                small_mask
            ]
        ]
        == 1
    ).mean(axis=1)

    # Distanze interne ed esterne.
    small_scores = scores[
        small_mask
    ]

    large_scores = scores[
        large_mask
    ]

    within_small_distances = pdist(
        small_scores,
        metric="euclidean",
    )

    # Tutte le distanze 73 x 3678 sono gestibili.
    between_distances = cdist(
        small_scores,
        large_scores,
        metric="euclidean",
    )

    small_to_large_nearest_distance = (
        between_distances.min(axis=1)
    )

    silhouette_values = silhouette_samples(
        scores,
        baseline_labels,
    )

    result = {
        "overall_silhouette": float(
            silhouette_score(
                scores,
                baseline_labels,
            )
        ),
        "small_cluster_mean_silhouette": float(
            silhouette_values[
                small_mask
            ].mean()
        ),
        "large_cluster_mean_silhouette": float(
            silhouette_values[
                large_mask
            ].mean()
        ),
        "small_cluster_mean_knn_purity": float(
            small_cluster_neighbour_purity.mean()
        ),
        "small_cluster_median_knn_purity": float(
            np.median(
                small_cluster_neighbour_purity
            )
        ),
        "all_samples_mean_same_cluster_knn_purity": float(
            neighbour_purity.mean()
        ),
        "small_cluster_mean_internal_pairwise_distance": float(
            within_small_distances.mean()
        ),
        "small_cluster_median_internal_pairwise_distance": float(
            np.median(
                within_small_distances
            )
        ),
        "small_cluster_mean_nearest_large_cluster_distance": float(
            small_to_large_nearest_distance.mean()
        ),
        "small_cluster_median_nearest_large_cluster_distance": float(
            np.median(
                small_to_large_nearest_distance
            )
        ),
        "internal_to_nearest_external_distance_ratio": float(
            within_small_distances.mean()
            / max(
                small_to_large_nearest_distance.mean(),
                1e-12,
            )
        ),
    }

    with open(
        output_dir
        / "cluster_cohesion.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    member_table = pd.DataFrame(
        {
            "row_index": np.arange(
                len(scores)
            ),
            "baseline_small_cluster": (
                baseline_labels
            ),
            "same_cluster_knn_purity": (
                neighbour_purity
            ),
            "silhouette": (
                silhouette_values
            ),
        }
    )

    member_table.to_csv(
        output_dir
        / "sample_cohesion_metrics.csv",
        index=False,
    )

    plt.figure(figsize=(9, 6))
    plt.hist(
        small_cluster_neighbour_purity,
        bins=np.linspace(
            0,
            1,
            11,
        ),
        edgecolor="black",
        linewidth=0.4,
    )
    plt.xlabel(
        f"Fraction of {N_NEIGHBORS} nearest neighbours also in the 73-sample cluster"
    )
    plt.ylabel(
        "Number of small-cluster samples"
    )
    plt.title(
        "Local cohesion of the small K-means cluster"
    )
    save_figure(
        output_dir
        / "small_cluster_knn_purity.png"
    )

    return result


# ============================================================
# E. STABILITÀ SU SOTTOINSIEMI DI FEATURE
# ============================================================

def feature_subsampling_stability(
    X: pd.DataFrame,
    baseline_labels: np.ndarray,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = X.to_numpy(dtype=float)

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    n_features = raw.shape[1]

    subset_size = max(
        2,
        int(
            round(
                FEATURE_SUBSAMPLE_FRACTION
                * n_features
            )
        ),
    )

    membership_counts = np.zeros(
        len(X),
        dtype=int,
    )

    run_records = []

    for run_index in range(
        N_FEATURE_SUBSAMPLING_RUNS
    ):
        print(
            f"  feature-subsampling run "
            f"{run_index + 1}/"
            f"{N_FEATURE_SUBSAMPLING_RUNS}"
        )

        selected_features = rng.choice(
            n_features,
            size=subset_size,
            replace=False,
        )

        standardized = StandardScaler().fit_transform(
            raw[
                :,
                selected_features
            ]
        )

        n_components = min(
            SUBSAMPLE_PCA_COMPONENTS,
            standardized.shape[1],
            standardized.shape[0] - 1,
        )

        pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=(
                RANDOM_STATE
                + run_index
            ),
        )

        scores = pca.fit_transform(
            standardized
        )

        model = KMeans(
            n_clusters=2,
            n_init=KMEANS_N_INIT,
            random_state=(
                RANDOM_STATE
                + run_index
            ),
        )

        labels = normalize_two_cluster_labels(
            model.fit_predict(scores)
        )

        membership_counts += labels

        overlap = int(
            np.logical_and(
                labels == 1,
                baseline_labels == 1,
            ).sum()
        )

        run_records.append(
            {
                "run": run_index + 1,
                "selected_feature_count": (
                    subset_size
                ),
                "retained_variance_in_fixed_components": float(
                    pca.explained_variance_ratio_.sum()
                ),
                "small_cluster_size": int(
                    labels.sum()
                ),
                "overlap_with_baseline_73": (
                    overlap
                ),
                "jaccard_with_baseline_73": (
                    jaccard_binary(
                        baseline_labels == 1,
                        labels == 1,
                    )
                ),
                "recall_of_baseline_73": (
                    overlap
                    / max(
                        int(
                            baseline_labels.sum()
                        ),
                        1,
                    )
                ),
                "ari_against_baseline_partition": float(
                    adjusted_rand_score(
                        baseline_labels,
                        labels,
                    )
                ),
            }
        )

    run_report = pd.DataFrame(
        run_records
    )

    run_report.to_csv(
        output_dir
        / "feature_subsampling_runs.csv",
        index=False,
    )

    frequency = (
        membership_counts
        / N_FEATURE_SUBSAMPLING_RUNS
    )

    sample_report = pd.DataFrame(
        {
            "row_index": np.arange(
                len(X)
            ),
            "baseline_small_cluster": (
                baseline_labels
            ),
            "small_cluster_selection_frequency": (
                frequency
            ),
        }
    )

    sample_report.to_csv(
        output_dir
        / "feature_subsampling_sample_frequency.csv",
        index=False,
    )

    plt.figure(figsize=(9, 6))

    for cluster in (0, 1):
        values = sample_report.loc[
            sample_report[
                "baseline_small_cluster"
            ]
            == cluster,
            "small_cluster_selection_frequency",
        ]

        plt.hist(
            values,
            bins=np.linspace(
                0,
                1,
                21,
            ),
            alpha=0.55,
            label=f"Baseline cluster {cluster}",
        )

    plt.xlabel(
        "Frequency of selection into the smaller cluster"
    )
    plt.ylabel("Number of samples")
    plt.title(
        "Stability of small-cluster membership under feature subsampling"
    )
    plt.legend()
    save_figure(
        output_dir
        / "feature_subsampling_membership_frequency.png"
    )

    return run_report, sample_report


# ============================================================
# DIAGNOSI FINALE
# ============================================================

def write_summary(
    preprocessing_report: pd.DataFrame,
    scalar_report: pd.DataFrame,
    outlier_report: pd.DataFrame,
    outlier_table: pd.DataFrame,
    cohesion: dict,
    subsampling_runs: pd.DataFrame,
    subsampling_samples: pd.DataFrame,
    baseline_labels: np.ndarray,
    y: pd.Series,
    output_path: Path,
) -> None:
    baseline_small_mask = (
        baseline_labels == 1
    )

    nonbaseline_methods = preprocessing_report[
        preprocessing_report[
            "representation"
        ]
        != "standard_scaler"
    ]

    median_alternative_jaccard = float(
        nonbaseline_methods[
            "jaccard_with_baseline_small_cluster"
        ].median()
    )

    raw_jaccard_series = preprocessing_report.loc[
        preprocessing_report[
            "representation"
        ]
        == "raw_unscaled",
        "jaccard_with_baseline_small_cluster",
    ]

    robust_jaccard_series = preprocessing_report.loc[
        preprocessing_report[
            "representation"
        ]
        == "robust_scaler",
        "jaccard_with_baseline_small_cluster",
    ]

    presence_jaccard_series = preprocessing_report.loc[
        preprocessing_report[
            "representation"
        ]
        == "presence_standard_scaler",
        "jaccard_with_baseline_small_cluster",
    ]

    raw_jaccard = float(
        raw_jaccard_series.iloc[0]
    )

    robust_jaccard = float(
        robust_jaccard_series.iloc[0]
    )

    presence_jaccard = float(
        presence_jaccard_series.iloc[0]
    )

    strongest_scalar = scalar_report.iloc[
        0
    ]

    detector_mean_recall = float(
        outlier_report[
            "recall_of_baseline_73"
        ].mean()
    )

    small_consensus = outlier_table.loc[
        baseline_small_mask,
        "outlier_consensus_count",
    ]

    fraction_small_flagged_by_at_least_two = float(
        np.mean(
            small_consensus >= 2
        )
    )

    small_subsampling_frequency = (
        subsampling_samples.loc[
            baseline_small_mask,
            "small_cluster_selection_frequency",
        ]
    )

    median_small_subsampling_frequency = float(
        small_subsampling_frequency.median()
    )

    mean_run_jaccard = float(
        subsampling_runs[
            "jaccard_with_baseline_73"
        ].mean()
    )

    activity_small = float(
        y[
            baseline_small_mask
        ].mean()
    )

    activity_large = float(
        y[
            ~baseline_small_mask
        ].mean()
    )

    # Heuristic scores.
    subtype_score = 0
    outlier_score = 0
    standardization_artifact_score = 0

    if robust_jaccard >= 0.70:
        subtype_score += 1
    elif robust_jaccard < 0.40:
        standardization_artifact_score += 1

    if raw_jaccard >= 0.70:
        subtype_score += 1
    elif raw_jaccard < 0.40:
        standardization_artifact_score += 1

    if presence_jaccard >= 0.70:
        subtype_score += 1

    if cohesion[
        "small_cluster_mean_knn_purity"
    ] >= 0.70:
        subtype_score += 1
    elif cohesion[
        "small_cluster_mean_knn_purity"
    ] < 0.40:
        outlier_score += 1

    if median_small_subsampling_frequency >= 0.70:
        subtype_score += 1
    elif median_small_subsampling_frequency < 0.40:
        standardization_artifact_score += 1

    if fraction_small_flagged_by_at_least_two >= 0.70:
        outlier_score += 2
    elif fraction_small_flagged_by_at_least_two >= 0.40:
        outlier_score += 1

    if strongest_scalar[
        "auc_absolute_direction"
    ] >= 0.98:
        standardization_artifact_score += 1
        outlier_score += 1

    scores = {
        "stable_subtype_evidence": subtype_score,
        "outlier_evidence": outlier_score,
        "standardization_or_magnitude_artifact_evidence": (
            standardization_artifact_score
        ),
    }

    highest_score = max(
        scores.values()
    )

    leading_hypotheses = [
        hypothesis
        for hypothesis, score in scores.items()
        if score == highest_score
    ]

    lines = [
        "SUBTYPE VS OUTLIERS DIAGNOSIS",
        "=" * 42,
        "",
        "BASELINE",
        "-" * 20,
        (
            "Small cluster size: "
            f"{int(baseline_small_mask.sum())}"
        ),
        (
            "Small cluster fraction: "
            f"{baseline_small_mask.mean():.6f}"
        ),
        (
            "Activity=1 fraction in large cluster: "
            f"{activity_large:.6f}"
        ),
        (
            "Activity=1 fraction in small cluster: "
            f"{activity_small:.6f}"
        ),
        "",
        "PREPROCESSING STABILITY",
        "-" * 28,
        (
            "Jaccard under raw unscaled PCA: "
            f"{raw_jaccard:.6f}"
        ),
        (
            "Jaccard under RobustScaler PCA: "
            f"{robust_jaccard:.6f}"
        ),
        (
            "Jaccard under presence-only PCA: "
            f"{presence_jaccard:.6f}"
        ),
        (
            "Median Jaccard across alternative representations: "
            f"{median_alternative_jaccard:.6f}"
        ),
        "",
        "SINGLE-SCALAR EXPLANATION",
        "-" * 28,
        (
            "Best scalar predictor: "
            f"{strongest_scalar['metric']}"
        ),
        (
            "Best scalar AUC: "
            f"{strongest_scalar['auc_absolute_direction']:.6f}"
        ),
        "",
        "OUTLIER DETECTORS",
        "-" * 20,
        (
            "Mean recall of the baseline 73 across detectors: "
            f"{detector_mean_recall:.6f}"
        ),
        (
            "Fraction of the 73 flagged by at least two detectors: "
            f"{fraction_small_flagged_by_at_least_two:.6f}"
        ),
        "",
        "LOCAL COHESION",
        "-" * 20,
        (
            "Mean kNN purity inside the small cluster: "
            f"{cohesion['small_cluster_mean_knn_purity']:.6f}"
        ),
        (
            "Median kNN purity inside the small cluster: "
            f"{cohesion['small_cluster_median_knn_purity']:.6f}"
        ),
        (
            "Mean silhouette of the small cluster: "
            f"{cohesion['small_cluster_mean_silhouette']:.6f}"
        ),
        (
            "Mean internal / nearest-external distance ratio: "
            f"{cohesion['internal_to_nearest_external_distance_ratio']:.6f}"
        ),
        "",
        "FEATURE-SUBSAMPLING STABILITY",
        "-" * 32,
        (
            "Mean run-level Jaccard with the baseline 73: "
            f"{mean_run_jaccard:.6f}"
        ),
        (
            "Median selection frequency among the baseline 73: "
            f"{median_small_subsampling_frequency:.6f}"
        ),
        "",
        "HEURISTIC EVIDENCE SCORES",
        "-" * 28,
        (
            "Stable subtype evidence: "
            f"{subtype_score}"
        ),
        (
            "Outlier evidence: "
            f"{outlier_score}"
        ),
        (
            "Standardization/magnitude artifact evidence: "
            f"{standardization_artifact_score}"
        ),
        (
            "Leading hypothesis/hypotheses: "
            f"{', '.join(leading_hypotheses)}"
        ),
        "",
        "IMPORTANT",
        "-" * 20,
        (
            "The final classification above is heuristic. "
            "Interpret the quantitative stability, cohesion and "
            "outlier-overlap measures directly."
        ),
    ]

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    with open(
        output_path.with_suffix(".json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "scores": scores,
                "leading_hypotheses": (
                    leading_hypotheses
                ),
                "raw_jaccard": raw_jaccard,
                "robust_jaccard": robust_jaccard,
                "presence_jaccard": presence_jaccard,
                "median_alternative_jaccard": (
                    median_alternative_jaccard
                ),
                "best_scalar_predictor": str(
                    strongest_scalar["metric"]
                ),
                "best_scalar_auc": float(
                    strongest_scalar[
                        "auc_absolute_direction"
                    ]
                ),
                "fraction_small_flagged_by_at_least_two": (
                    fraction_small_flagged_by_at_least_two
                ),
                "small_cluster_mean_knn_purity": (
                    cohesion[
                        "small_cluster_mean_knn_purity"
                    ]
                ),
                "median_small_subsampling_frequency": (
                    median_small_subsampling_frequency
                ),
            },
            file,
            indent=2,
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    inputs = load_inputs()

    X = inputs["X"]
    y = inputs["y"]
    baseline_labels = inputs[
        "baseline_labels"
    ]
    baseline_pca_scores = inputs[
        "baseline_pca_scores"
    ]
    dirs = inputs["dirs"]

    print(
        "\nA/5 Stability across preprocessing choices"
    )
    (
        preprocessing_report,
        labels_by_method,
    ) = preprocessing_stability(
        X,
        baseline_labels,
        dirs["preprocessing"],
    )

    print(
        "\nB/5 Single-scalar explanations"
    )
    scalar_report = scalar_predictors(
        X,
        baseline_labels,
        baseline_pca_scores,
        dirs["scalar"],
    )

    print(
        "\nC/5 Outlier detection"
    )
    (
        outlier_report,
        outlier_table,
    ) = outlier_detection(
        baseline_pca_scores,
        baseline_labels,
        dirs["outliers"],
    )

    print(
        "\nD/5 Cluster cohesion"
    )
    cohesion = cluster_cohesion(
        baseline_pca_scores,
        baseline_labels,
        dirs["cohesion"],
    )

    print(
        "\nE/5 Feature-subsampling stability"
    )
    (
        subsampling_runs,
        subsampling_samples,
    ) = feature_subsampling_stability(
        X,
        baseline_labels,
        dirs["subsampling"],
    )

    write_summary(
        preprocessing_report=(
            preprocessing_report
        ),
        scalar_report=scalar_report,
        outlier_report=outlier_report,
        outlier_table=outlier_table,
        cohesion=cohesion,
        subsampling_runs=(
            subsampling_runs
        ),
        subsampling_samples=(
            subsampling_samples
        ),
        baseline_labels=baseline_labels,
        y=y,
        output_path=(
            dirs["root"]
            / "subtype_vs_outliers_summary.txt"
        ),
    )

    print("\nAnalysis completed.")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(
        "Start from subtype_vs_outliers_summary.txt."
    )


if __name__ == "__main__":
    main()
