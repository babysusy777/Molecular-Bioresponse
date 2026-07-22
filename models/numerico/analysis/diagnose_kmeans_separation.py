#!/usr/bin/env python3
"""
Diagnosi della separazione prodotta da K-means sulle feature numeriche.

Lo script NON ricalcola K-means: usa le assegnazioni già salvate da
numeric_feature_exploration.py, così da analizzare esattamente la soluzione
che ha prodotto silhouette elevata.

Verifiche principali
--------------------
1. Bilanciamento e silhouette dei cluster.
2. Proprietà globali delle molecole:
   - numero/frazione di feature non nulle;
   - somma, media, massimo;
   - norme L1 e L2;
   - distanza dal centro globale;
   - PC1, PC2 e altre componenti.
3. Feature che separano maggiormente i cluster:
   - differenza delle medie standardizzate;
   - Cohen's d;
   - Mann-Whitney U;
   - rank-biserial correlation;
   - differenza nella frazione di zeri.
4. Componenti principali più associate all'assegnazione K-means.
5. Gruppi Spearman di feature maggiormente responsabili della separazione.
6. Confronto post-hoc con Activity.

Output principale:
    kmeans_separation_summary.txt
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
from scipy.stats import mannwhitneyu
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


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

# Lo script prova automaticamente più cartelle comuni.
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
    (
        BASE_DIR
        / "models"
        / "numerico"
        / "analysis"
        / "numeric_feature_exploration"
        / "reports"
    ),
]

OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "numerico"
    / "analysis"
    / "kmeans_separation_diagnosis"
)

TOP_N = 30
DPI = 180
RANDOM_STATE = 42


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
    required_relative_paths = [
        Path("pca_clustering") / "kmeans_cluster_assignments.csv",
        Path("pca_clustering") / "pca_scores.csv",
    ]

    for candidate in REPORT_ROOT_CANDIDATES:
        if all(
            (candidate / relative_path).exists()
            for relative_path in required_relative_paths
        ):
            return candidate

    searched = "\n".join(
        f"- {candidate}"
        for candidate in REPORT_ROOT_CANDIDATES
    )

    raise FileNotFoundError(
        "Non trovo i risultati prodotti dallo script precedente.\n"
        "Cartelle controllate:\n"
        f"{searched}\n\n"
        "Modifica REPORT_ROOT_CANDIDATES inserendo la cartella "
        "che contiene pca_clustering/."
    )


def cohen_d(
    values_1: np.ndarray,
    values_0: np.ndarray,
) -> float:
    """
    Cohen's d = media cluster 1 - media cluster 0,
    divisa per la deviazione standard pooled.
    """
    n1 = len(values_1)
    n0 = len(values_0)

    if n1 < 2 or n0 < 2:
        return np.nan

    variance_1 = np.var(
        values_1,
        ddof=1,
    )
    variance_0 = np.var(
        values_0,
        ddof=1,
    )

    pooled_variance = (
        (n1 - 1) * variance_1
        + (n0 - 1) * variance_0
    ) / (n1 + n0 - 2)

    if pooled_variance <= 0:
        return 0.0

    return float(
        (
            np.mean(values_1)
            - np.mean(values_0)
        )
        / np.sqrt(pooled_variance)
    )


def rank_biserial_from_u(
    u_statistic: float,
    n1: int,
    n0: int,
) -> float:
    """
    Positivo: valori tendenzialmente maggiori nel cluster 1.
    """
    return float(
        2.0 * u_statistic / (n1 * n0) - 1.0
    )


def compare_binary_groups(
    values: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Confronta cluster 1 contro cluster 0.
    Richiede esattamente due cluster codificati 0 e 1.
    """
    values_0 = values[labels == 0]
    values_1 = values[labels == 1]

    try:
        result = mannwhitneyu(
            values_1,
            values_0,
            alternative="two-sided",
            method="auto",
        )

        u_statistic = float(
            result.statistic
        )

        p_value = float(
            result.pvalue
        )

        rank_biserial = rank_biserial_from_u(
            u_statistic,
            len(values_1),
            len(values_0),
        )
    except ValueError:
        u_statistic = np.nan
        p_value = 1.0
        rank_biserial = np.nan

    return {
        "cluster_0_mean": float(
            np.mean(values_0)
        ),
        "cluster_1_mean": float(
            np.mean(values_1)
        ),
        "cluster_0_median": float(
            np.median(values_0)
        ),
        "cluster_1_median": float(
            np.median(values_1)
        ),
        "mean_difference_1_minus_0": float(
            np.mean(values_1)
            - np.mean(values_0)
        ),
        "median_difference_1_minus_0": float(
            np.median(values_1)
            - np.median(values_0)
        ),
        "cohen_d_1_minus_0": cohen_d(
            values_1,
            values_0,
        ),
        "mann_whitney_u": u_statistic,
        "mann_whitney_p_value": p_value,
        "rank_biserial_1_minus_0": rank_biserial,
    }


# ============================================================
# CARICAMENTO
# ============================================================

def load_inputs() -> dict:
    report_root = find_report_root()

    output_directories = {
        "root": OUTPUT_DIR,
        "sample_metrics": OUTPUT_DIR / "sample_metrics",
        "features": OUTPUT_DIR / "feature_separation",
        "pca": OUTPUT_DIR / "pca_separation",
        "feature_clusters": OUTPUT_DIR / "feature_cluster_separation",
    }

    for directory in output_directories.values():
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    X = pd.read_csv(DATA_PATH)

    if "Activity" in X.columns:
        X = X.drop(columns=["Activity"])

    if X.isna().any().any():
        raise ValueError(
            "Il dataset numerico contiene valori mancanti."
        )

    target_df = pd.read_csv(TARGET_PATH)

    if "Activity" in target_df.columns:
        y = target_df["Activity"]
    else:
        y = target_df.iloc[:, 0]

    y = pd.Series(
        y.to_numpy(),
        name="Activity",
    ).reset_index(drop=True)

    assignments_path = (
        report_root
        / "pca_clustering"
        / "kmeans_cluster_assignments.csv"
    )

    assignments = pd.read_csv(
        assignments_path
    ).sort_values("row_index")

    labels = assignments[
        "kmeans_cluster"
    ].to_numpy(dtype=int)

    unique_labels = np.sort(
        np.unique(labels)
    )

    if len(unique_labels) != 2:
        raise ValueError(
            "Questo script è progettato per la soluzione K-means "
            f"con due cluster. Cluster trovati: {unique_labels.tolist()}"
        )

    # Rietichetta eventualmente i due cluster come 0 e 1.
    label_map = {
        int(unique_labels[0]): 0,
        int(unique_labels[1]): 1,
    }

    labels = np.array(
        [
            label_map[int(label)]
            for label in labels
        ],
        dtype=int,
    )

    pca_scores_path = (
        report_root
        / "pca_clustering"
        / "pca_scores.csv"
    )

    pca_scores = pd.read_csv(
        pca_scores_path
    ).sort_values("row_index")

    pc_columns = [
        column
        for column in pca_scores.columns
        if column.startswith("PC")
    ]

    if not pc_columns:
        raise ValueError(
            "Nessuna componente principale trovata in pca_scores.csv."
        )

    X_pca = pca_scores[
        pc_columns
    ].to_numpy(dtype=float)

    feature_assignment_path = (
        report_root
        / "feature_clustering"
        / "feature_cluster_assignments.csv"
    )

    feature_assignments = None

    if feature_assignment_path.exists():
        feature_assignments = pd.read_csv(
            feature_assignment_path
        )

    pca_loadings_path = (
        report_root
        / "pca_clustering"
        / "pca_loadings.csv"
    )

    pca_loadings = None

    if pca_loadings_path.exists():
        pca_loadings = pd.read_csv(
            pca_loadings_path,
            index_col=0,
        )

    n_samples = len(X)

    if not (
        len(y)
        == len(labels)
        == len(X_pca)
        == n_samples
    ):
        raise ValueError(
            "Numero di righe incoerente tra dataset, target, "
            "assegnazioni K-means e PCA scores."
        )

    print(f"Numeric dataset: {X.shape}")
    print(
        "K-means cluster sizes: "
        f"{pd.Series(labels).value_counts().sort_index().to_dict()}"
    )
    print(f"Previous report root: {report_root}")
    print(f"Diagnosis output: {OUTPUT_DIR}")

    return {
        "X": X,
        "y": y.astype(int),
        "labels": labels,
        "X_pca": X_pca,
        "pc_columns": pc_columns,
        "feature_assignments": feature_assignments,
        "pca_loadings": pca_loadings,
        "report_root": report_root,
        "dirs": output_directories,
    }


# ============================================================
# 1. METRICHE PER MOLECOLA
# ============================================================

def analyze_sample_level_properties(
    X: pd.DataFrame,
    y: pd.Series,
    labels: np.ndarray,
    X_pca: np.ndarray,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_values = X.to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(raw_values)

    nonzero_mask = raw_values != 0
    positive_mask = raw_values > 0

    nonzero_count = nonzero_mask.sum(axis=1)
    positive_count = positive_mask.sum(axis=1)

    positive_sum = np.where(
        positive_mask,
        raw_values,
        0.0,
    ).sum(axis=1)

    positive_mean = np.divide(
        positive_sum,
        positive_count,
        out=np.zeros(
            len(X),
            dtype=float,
        ),
        where=positive_count > 0,
    )

    global_centroid_pca = X_pca.mean(
        axis=0,
        keepdims=True,
    )

    distance_to_global_centroid_pca = np.linalg.norm(
        X_pca - global_centroid_pca,
        axis=1,
    )

    cluster_centroids_pca = {
        cluster: X_pca[
            labels == cluster
        ].mean(
            axis=0,
            keepdims=True,
        )
        for cluster in (0, 1)
    }

    distance_to_own_centroid_pca = np.empty(
        len(X),
        dtype=float,
    )

    for cluster in (0, 1):
        mask = labels == cluster

        distance_to_own_centroid_pca[mask] = np.linalg.norm(
            X_pca[mask]
            - cluster_centroids_pca[cluster],
            axis=1,
        )

    sample_metrics = pd.DataFrame(
        {
            "row_index": np.arange(len(X)),
            "kmeans_cluster": labels,
            "Activity": y.to_numpy(),
            "zero_count": (raw_values == 0).sum(axis=1),
            "zero_fraction": (raw_values == 0).mean(axis=1),
            "nonzero_count": nonzero_count,
            "nonzero_fraction": nonzero_mask.mean(axis=1),
            "positive_count": positive_count,
            "positive_fraction": positive_mask.mean(axis=1),
            "raw_sum": raw_values.sum(axis=1),
            "raw_mean": raw_values.mean(axis=1),
            "raw_max": raw_values.max(axis=1),
            "positive_mean": positive_mean,
            "l1_norm_raw": np.abs(raw_values).sum(axis=1),
            "l2_norm_raw": np.linalg.norm(
                raw_values,
                axis=1,
            ),
            "l1_norm_standardized": np.abs(
                X_scaled
            ).sum(axis=1),
            "l2_norm_standardized": np.linalg.norm(
                X_scaled,
                axis=1,
            ),
            "distance_to_global_centroid_pca": (
                distance_to_global_centroid_pca
            ),
            "distance_to_own_centroid_pca": (
                distance_to_own_centroid_pca
            ),
            "PC1": X_pca[:, 0],
            "PC2": (
                X_pca[:, 1]
                if X_pca.shape[1] >= 2
                else np.nan
            ),
        }
    )

    sample_metrics.to_csv(
        output_dir / "sample_level_metrics.csv",
        index=False,
    )

    metric_columns = [
        column
        for column in sample_metrics.columns
        if column not in {
            "row_index",
            "kmeans_cluster",
            "Activity",
        }
    ]

    comparison_records = []

    for metric in metric_columns:
        comparison = compare_binary_groups(
            sample_metrics[
                metric
            ].to_numpy(dtype=float),
            labels,
        )

        comparison_records.append(
            {
                "metric": metric,
                **comparison,
            }
        )

    metric_comparison = pd.DataFrame(
        comparison_records
    )

    rejected, adjusted_p_values, _, _ = multipletests(
        metric_comparison[
            "mann_whitney_p_value"
        ],
        alpha=0.05,
        method="fdr_bh",
    )

    metric_comparison[
        "mann_whitney_p_fdr_bh"
    ] = adjusted_p_values

    metric_comparison[
        "significant_fdr_0.05"
    ] = rejected

    metric_comparison[
        "absolute_cohen_d"
    ] = metric_comparison[
        "cohen_d_1_minus_0"
    ].abs()

    metric_comparison = metric_comparison.sort_values(
        "absolute_cohen_d",
        ascending=False,
    )

    metric_comparison.to_csv(
        output_dir / "sample_metric_comparison.csv",
        index=False,
    )

    # Un grafico separato per ogni metrica più importante.
    top_metrics = metric_comparison.head(8)[
        "metric"
    ].tolist()

    for metric in top_metrics:
        values_0 = sample_metrics.loc[
            sample_metrics["kmeans_cluster"] == 0,
            metric,
        ].to_numpy()

        values_1 = sample_metrics.loc[
            sample_metrics["kmeans_cluster"] == 1,
            metric,
        ].to_numpy()

        plt.figure(figsize=(8, 6))
        plt.boxplot(
            [values_0, values_1],
            tick_labels=[
                "Cluster 0",
                "Cluster 1",
            ],
            showfliers=False,
        )
        plt.ylabel(metric)
        plt.title(
            f"{metric} by K-means cluster"
        )
        save_figure(
            output_dir
            / f"boxplot_{metric}.png"
        )

    # PC1 contro norma standardizzata.
    plt.figure(figsize=(9, 7))

    for cluster in (0, 1):
        mask = labels == cluster

        plt.scatter(
            sample_metrics.loc[
                mask,
                "PC1",
            ],
            sample_metrics.loc[
                mask,
                "l2_norm_standardized",
            ],
            s=18,
            alpha=0.60,
            label=f"Cluster {cluster}",
        )

    plt.xlabel("PC1 score")
    plt.ylabel("L2 norm in standardized feature space")
    plt.title(
        "PC1 versus standardized magnitude"
    )
    plt.legend()
    save_figure(
        output_dir
        / "pc1_vs_standardized_l2_norm.png"
    )

    return sample_metrics, metric_comparison


# ============================================================
# 2. SILHOUETTE E BILANCIAMENTO
# ============================================================

def analyze_cluster_geometry(
    X_pca: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
) -> dict:
    sample_silhouette = silhouette_samples(
        X_pca,
        labels,
    )

    overall_silhouette = float(
        silhouette_score(
            X_pca,
            labels,
        )
    )

    geometry_records = []

    for cluster in (0, 1):
        mask = labels == cluster

        geometry_records.append(
            {
                "cluster": cluster,
                "n_samples": int(
                    mask.sum()
                ),
                "fraction": float(
                    mask.mean()
                ),
                "silhouette_mean": float(
                    sample_silhouette[mask].mean()
                ),
                "silhouette_median": float(
                    np.median(
                        sample_silhouette[mask]
                    )
                ),
                "silhouette_minimum": float(
                    sample_silhouette[mask].min()
                ),
                "silhouette_maximum": float(
                    sample_silhouette[mask].max()
                ),
            }
        )

    geometry = pd.DataFrame(
        geometry_records
    )

    geometry.to_csv(
        output_dir / "cluster_geometry.csv",
        index=False,
    )

    silhouette_table = pd.DataFrame(
        {
            "row_index": np.arange(
                len(labels)
            ),
            "kmeans_cluster": labels,
            "silhouette": sample_silhouette,
        }
    )

    silhouette_table.to_csv(
        output_dir / "sample_silhouette.csv",
        index=False,
    )

    plt.figure(figsize=(9, 6))

    for cluster in (0, 1):
        mask = labels == cluster

        plt.hist(
            sample_silhouette[mask],
            bins=30,
            alpha=0.55,
            label=f"Cluster {cluster}",
        )

    plt.xlabel("Sample silhouette")
    plt.ylabel("Number of samples")
    plt.title(
        f"Silhouette distribution; overall={overall_silhouette:.4f}"
    )
    plt.legend()
    save_figure(
        output_dir / "silhouette_distribution.png"
    )

    size_ratio = float(
        geometry["n_samples"].max()
        / geometry["n_samples"].min()
    )

    result = {
        "overall_silhouette": overall_silhouette,
        "cluster_size_ratio_large_to_small": size_ratio,
        "geometry": geometry,
    }

    with open(
        output_dir / "cluster_geometry.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "overall_silhouette": overall_silhouette,
                "cluster_size_ratio_large_to_small": size_ratio,
            },
            file,
            indent=2,
        )

    return result


# ============================================================
# 3. FEATURE CHE SEPARANO I CLUSTER
# ============================================================

def analyze_feature_separation(
    X: pd.DataFrame,
    labels: np.ndarray,
    feature_assignments: pd.DataFrame | None,
    pca_loadings: pd.DataFrame | None,
    output_dir: Path,
) -> pd.DataFrame:
    raw_values = X.to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(
        raw_values
    )

    feature_cluster_map = {}

    if feature_assignments is not None:
        feature_cluster_map = (
            feature_assignments
            .set_index("feature")[
                "feature_cluster"
            ]
            .to_dict()
        )

    records = []

    for feature_index, feature in enumerate(
        X.columns
    ):
        raw_feature = raw_values[
            :,
            feature_index,
        ]

        standardized_feature = X_scaled[
            :,
            feature_index,
        ]

        comparison = compare_binary_groups(
            raw_feature,
            labels,
        )

        standardized_mean_0 = float(
            standardized_feature[
                labels == 0
            ].mean()
        )

        standardized_mean_1 = float(
            standardized_feature[
                labels == 1
            ].mean()
        )

        zero_fraction_0 = float(
            np.mean(
                raw_feature[
                    labels == 0
                ]
                == 0
            )
        )

        zero_fraction_1 = float(
            np.mean(
                raw_feature[
                    labels == 1
                ]
                == 0
            )
        )

        record = {
            "feature": feature,
            **comparison,
            "standardized_mean_cluster_0": (
                standardized_mean_0
            ),
            "standardized_mean_cluster_1": (
                standardized_mean_1
            ),
            "standardized_mean_difference_1_minus_0": (
                standardized_mean_1
                - standardized_mean_0
            ),
            "zero_fraction_cluster_0": (
                zero_fraction_0
            ),
            "zero_fraction_cluster_1": (
                zero_fraction_1
            ),
            "zero_fraction_difference_1_minus_0": (
                zero_fraction_1
                - zero_fraction_0
            ),
            "feature_cluster": feature_cluster_map.get(
                feature,
                np.nan,
            ),
        }

        if (
            pca_loadings is not None
            and feature in pca_loadings.index
        ):
            if "PC1" in pca_loadings.columns:
                record["PC1_loading"] = float(
                    pca_loadings.loc[
                        feature,
                        "PC1",
                    ]
                )

            if "PC2" in pca_loadings.columns:
                record["PC2_loading"] = float(
                    pca_loadings.loc[
                        feature,
                        "PC2",
                    ]
                )

        records.append(record)

    report = pd.DataFrame(records)

    rejected, adjusted_p_values, _, _ = multipletests(
        report["mann_whitney_p_value"],
        alpha=0.05,
        method="fdr_bh",
    )

    report[
        "mann_whitney_p_fdr_bh"
    ] = adjusted_p_values

    report[
        "significant_fdr_0.05"
    ] = rejected

    report[
        "absolute_standardized_mean_difference"
    ] = report[
        "standardized_mean_difference_1_minus_0"
    ].abs()

    report[
        "absolute_rank_biserial"
    ] = report[
        "rank_biserial_1_minus_0"
    ].abs()

    report = report.sort_values(
        by=[
            "absolute_standardized_mean_difference",
            "absolute_rank_biserial",
        ],
        ascending=[
            False,
            False,
        ],
    )

    report.to_csv(
        output_dir / "feature_separation.csv",
        index=False,
    )

    top_features = report.head(
        TOP_N
    ).sort_values(
        "standardized_mean_difference_1_minus_0"
    )

    plt.figure(figsize=(11, 10))
    plt.barh(
        top_features["feature"],
        top_features[
            "standardized_mean_difference_1_minus_0"
        ],
    )
    plt.axvline(
        0.0,
        linewidth=1.0,
    )
    plt.xlabel(
        "Standardized mean difference: cluster 1 minus cluster 0"
    )
    plt.ylabel("Feature")
    plt.title(
        f"Top {TOP_N} features separating the K-means clusters"
    )
    save_figure(
        output_dir
        / "top_feature_standardized_differences.png"
    )

    top_zero_differences = report.reindex(
        report[
            "zero_fraction_difference_1_minus_0"
        ]
        .abs()
        .sort_values(
            ascending=False
        )
        .index
    ).head(TOP_N).sort_values(
        "zero_fraction_difference_1_minus_0"
    )

    plt.figure(figsize=(11, 10))
    plt.barh(
        top_zero_differences["feature"],
        top_zero_differences[
            "zero_fraction_difference_1_minus_0"
        ],
    )
    plt.axvline(
        0.0,
        linewidth=1.0,
    )
    plt.xlabel(
        "Zero-fraction difference: cluster 1 minus cluster 0"
    )
    plt.ylabel("Feature")
    plt.title(
        f"Top {TOP_N} differences in feature sparsity"
    )
    save_figure(
        output_dir
        / "top_feature_zero_fraction_differences.png"
    )

    return report


# ============================================================
# 4. COMPONENTI PRINCIPALI CHE SEPARANO I CLUSTER
# ============================================================

def analyze_pca_separation(
    X_pca: np.ndarray,
    pc_columns: list[str],
    labels: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    records = []

    for component_index, component in enumerate(
        pc_columns
    ):
        comparison = compare_binary_groups(
            X_pca[
                :,
                component_index,
            ],
            labels,
        )

        records.append(
            {
                "component": component,
                "component_number": component_index + 1,
                **comparison,
            }
        )

    report = pd.DataFrame(records)

    rejected, adjusted_p_values, _, _ = multipletests(
        report[
            "mann_whitney_p_value"
        ],
        alpha=0.05,
        method="fdr_bh",
    )

    report[
        "mann_whitney_p_fdr_bh"
    ] = adjusted_p_values

    report[
        "significant_fdr_0.05"
    ] = rejected

    report[
        "absolute_cohen_d"
    ] = report[
        "cohen_d_1_minus_0"
    ].abs()

    report = report.sort_values(
        "absolute_cohen_d",
        ascending=False,
    )

    report.to_csv(
        output_dir / "pca_component_separation.csv",
        index=False,
    )

    top_components = report.head(
        min(TOP_N, len(report))
    ).sort_values(
        "cohen_d_1_minus_0"
    )

    plt.figure(figsize=(10, 9))
    plt.barh(
        top_components["component"],
        top_components[
            "cohen_d_1_minus_0"
        ],
    )
    plt.axvline(
        0.0,
        linewidth=1.0,
    )
    plt.xlabel(
        "Cohen's d: cluster 1 minus cluster 0"
    )
    plt.ylabel("Principal component")
    plt.title(
        "Principal components most responsible for K-means separation"
    )
    save_figure(
        output_dir
        / "top_pca_component_effects.png"
    )

    return report


# ============================================================
# 5. CLUSTER SPEARMAN DI FEATURE
# ============================================================

def analyze_feature_cluster_separation(
    feature_report: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame | None:
    if (
        "feature_cluster"
        not in feature_report.columns
        or feature_report[
            "feature_cluster"
        ].isna().all()
    ):
        warnings.warn(
            "Assegnazioni dei cluster Spearman non disponibili: "
            "analisi aggregata saltata."
        )
        return None

    valid = feature_report.dropna(
        subset=["feature_cluster"]
    ).copy()

    valid["feature_cluster"] = valid[
        "feature_cluster"
    ].astype(int)

    summary = (
        valid
        .groupby("feature_cluster")
        .agg(
            n_features=(
                "feature",
                "size",
            ),
            mean_standardized_difference=(
                "standardized_mean_difference_1_minus_0",
                "mean",
            ),
            median_standardized_difference=(
                "standardized_mean_difference_1_minus_0",
                "median",
            ),
            mean_absolute_standardized_difference=(
                "absolute_standardized_mean_difference",
                "mean",
            ),
            maximum_absolute_standardized_difference=(
                "absolute_standardized_mean_difference",
                "max",
            ),
            mean_zero_fraction_difference=(
                "zero_fraction_difference_1_minus_0",
                "mean",
            ),
        )
        .reset_index()
    )

    top_feature_per_cluster = (
        valid.sort_values(
            "absolute_standardized_mean_difference",
            ascending=False,
        )
        .groupby(
            "feature_cluster",
            as_index=False,
        )
        .first()[
            [
                "feature_cluster",
                "feature",
                "standardized_mean_difference_1_minus_0",
            ]
        ]
        .rename(
            columns={
                "feature": "top_separating_feature",
                "standardized_mean_difference_1_minus_0": (
                    "top_feature_standardized_difference"
                ),
            }
        )
    )

    summary = summary.merge(
        top_feature_per_cluster,
        on="feature_cluster",
        how="left",
    )

    summary[
        "absolute_mean_standardized_difference"
    ] = summary[
        "mean_standardized_difference"
    ].abs()

    summary = summary.sort_values(
        "absolute_mean_standardized_difference",
        ascending=False,
    )

    summary.to_csv(
        output_dir
        / "feature_cluster_separation.csv",
        index=False,
    )

    plot_data = summary.sort_values(
        "mean_standardized_difference"
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        plot_data[
            "feature_cluster"
        ].astype(str),
        plot_data[
            "mean_standardized_difference"
        ],
    )
    plt.axvline(
        0.0,
        linewidth=1.0,
    )
    plt.xlabel(
        "Mean standardized difference: cluster 1 minus cluster 0"
    )
    plt.ylabel("Spearman feature cluster")
    plt.title(
        "Feature groups driving K-means separation"
    )
    save_figure(
        output_dir
        / "feature_cluster_mean_differences.png"
    )

    return summary


# ============================================================
# 6. CONFRONTO CON ACTIVITY
# ============================================================

def analyze_activity_composition(
    y: pd.Series,
    labels: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    summary = (
        pd.DataFrame(
            {
                "kmeans_cluster": labels,
                "Activity": y.to_numpy(),
            }
        )
        .groupby("kmeans_cluster")
        .agg(
            n_samples=("Activity", "size"),
            activity_0_count=(
                "Activity",
                lambda values: int(
                    np.sum(values == 0)
                ),
            ),
            activity_1_count=(
                "Activity",
                lambda values: int(
                    np.sum(values == 1)
                ),
            ),
            activity_1_fraction=(
                "Activity",
                "mean",
            ),
        )
        .reset_index()
    )

    summary[
        "global_activity_1_fraction"
    ] = float(y.mean())

    summary[
        "activity_1_enrichment"
    ] = (
        summary["activity_1_fraction"]
        - float(y.mean())
    )

    summary.to_csv(
        output_dir
        / "activity_composition_by_cluster.csv",
        index=False,
    )

    return summary


# ============================================================
# 7. REPORT FINALE
# ============================================================

def write_summary(
    cluster_geometry: dict,
    sample_metric_comparison: pd.DataFrame,
    feature_report: pd.DataFrame,
    pca_report: pd.DataFrame,
    feature_cluster_report: pd.DataFrame | None,
    activity_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    geometry = cluster_geometry[
        "geometry"
    ]

    cluster_sizes = {
        int(row["cluster"]): int(
            row["n_samples"]
        )
        for _, row in geometry.iterrows()
    }

    top_sample_metrics = (
        sample_metric_comparison
        .head(8)
    )

    top_features = feature_report.head(
        15
    )

    top_components = pca_report.head(
        10
    )

    lines = [
        "K-MEANS SEPARATION DIAGNOSIS",
        "=" * 42,
        "",
        "CLUSTER GEOMETRY",
        "-" * 20,
        (
            "Cluster sizes: "
            f"{cluster_sizes}"
        ),
        (
            "Large/small cluster size ratio: "
            f"{cluster_geometry['cluster_size_ratio_large_to_small']:.4f}"
        ),
        (
            "Overall silhouette: "
            f"{cluster_geometry['overall_silhouette']:.6f}"
        ),
        "",
        "SAMPLE-LEVEL PROPERTIES WITH LARGEST EFFECT",
        "-" * 48,
    ]

    for _, row in top_sample_metrics.iterrows():
        lines.append(
            f"{row['metric']}: "
            f"mean0={row['cluster_0_mean']:.6f}, "
            f"mean1={row['cluster_1_mean']:.6f}, "
            f"d={row['cohen_d_1_minus_0']:.6f}, "
            f"r_rb={row['rank_biserial_1_minus_0']:.6f}, "
            f"FDR p={row['mann_whitney_p_fdr_bh']:.6g}"
        )

    lines.extend(
        [
            "",
            "FEATURES WITH LARGEST STANDARDIZED DIFFERENCE",
            "-" * 52,
        ]
    )

    for _, row in top_features.iterrows():
        lines.append(
            f"{row['feature']}: "
            f"std_diff={row['standardized_mean_difference_1_minus_0']:.6f}, "
            f"d={row['cohen_d_1_minus_0']:.6f}, "
            f"zero_diff={row['zero_fraction_difference_1_minus_0']:.6f}, "
            f"feature_cluster={row['feature_cluster']}"
        )

    lines.extend(
        [
            "",
            "PRINCIPAL COMPONENTS WITH LARGEST EFFECT",
            "-" * 47,
        ]
    )

    for _, row in top_components.iterrows():
        lines.append(
            f"{row['component']}: "
            f"mean_diff={row['mean_difference_1_minus_0']:.6f}, "
            f"d={row['cohen_d_1_minus_0']:.6f}, "
            f"r_rb={row['rank_biserial_1_minus_0']:.6f}"
        )

    if feature_cluster_report is not None:
        lines.extend(
            [
                "",
                "SPEARMAN FEATURE CLUSTERS WITH LARGEST EFFECT",
                "-" * 50,
            ]
        )

        for _, row in feature_cluster_report.head(
            8
        ).iterrows():
            lines.append(
                f"FeatureCluster {int(row['feature_cluster'])}: "
                f"n={int(row['n_features'])}, "
                f"mean_std_diff={row['mean_standardized_difference']:.6f}, "
                f"top_feature={row['top_separating_feature']}, "
                f"top_diff={row['top_feature_standardized_difference']:.6f}"
            )

    lines.extend(
        [
            "",
            "ACTIVITY COMPOSITION",
            "-" * 20,
        ]
    )

    for _, row in activity_summary.iterrows():
        lines.append(
            f"Cluster {int(row['kmeans_cluster'])}: "
            f"n={int(row['n_samples'])}, "
            f"Activity=1 fraction={row['activity_1_fraction']:.6f}, "
            f"enrichment={row['activity_1_enrichment']:.6f}"
        )

    # Diagnosi automatica prudente.
    strongest_metric = sample_metric_comparison.iloc[
        0
    ]

    strongest_pc = pca_report.iloc[
        0
    ]

    lines.extend(
        [
            "",
            "AUTOMATIC DIAGNOSTIC CUES",
            "-" * 28,
            (
                "Largest sample-level standardized effect: "
                f"{strongest_metric['metric']} "
                f"(Cohen's d={strongest_metric['cohen_d_1_minus_0']:.4f})."
            ),
            (
                "Principal component most associated with the split: "
                f"{strongest_pc['component']} "
                f"(Cohen's d={strongest_pc['cohen_d_1_minus_0']:.4f})."
            ),
        ]
    )

    if (
        abs(
            strongest_metric[
                "cohen_d_1_minus_0"
            ]
        )
        >= 1.0
    ):
        lines.append(
            "The separation is strongly associated with this "
            "sample-level property."
        )

    if (
        cluster_geometry[
            "cluster_size_ratio_large_to_small"
        ]
        >= 5.0
    ):
        lines.append(
            "The solution is strongly imbalanced and may represent "
            "a large central population plus a small extreme group."
        )

    if (
        strongest_pc["component"]
        == "PC1"
        and abs(
            strongest_pc[
                "cohen_d_1_minus_0"
            ]
        )
        >= 1.0
    ):
        lines.append(
            "PC1 is a dominant direction of the K-means separation."
        )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    inputs = load_inputs()

    X = inputs["X"]
    y = inputs["y"]
    labels = inputs["labels"]
    X_pca = inputs["X_pca"]
    pc_columns = inputs["pc_columns"]
    feature_assignments = inputs[
        "feature_assignments"
    ]
    pca_loadings = inputs[
        "pca_loadings"
    ]
    dirs = inputs["dirs"]

    print("\n1/6 Cluster geometry and silhouette")
    cluster_geometry = analyze_cluster_geometry(
        X_pca,
        labels,
        dirs["root"],
    )

    print("\n2/6 Sample-level properties")
    (
        sample_metrics,
        sample_metric_comparison,
    ) = analyze_sample_level_properties(
        X,
        y,
        labels,
        X_pca,
        dirs["sample_metrics"],
    )

    print("\n3/6 Feature-level separation")
    feature_report = analyze_feature_separation(
        X,
        labels,
        feature_assignments,
        pca_loadings,
        dirs["features"],
    )

    print("\n4/6 PCA-component separation")
    pca_report = analyze_pca_separation(
        X_pca,
        pc_columns,
        labels,
        dirs["pca"],
    )

    print("\n5/6 Spearman feature-cluster separation")
    feature_cluster_report = (
        analyze_feature_cluster_separation(
            feature_report,
            dirs["feature_clusters"],
        )
    )

    print("\n6/6 Activity composition")
    activity_summary = analyze_activity_composition(
        y,
        labels,
        dirs["root"],
    )

    write_summary(
        cluster_geometry=cluster_geometry,
        sample_metric_comparison=(
            sample_metric_comparison
        ),
        feature_report=feature_report,
        pca_report=pca_report,
        feature_cluster_report=(
            feature_cluster_report
        ),
        activity_summary=activity_summary,
        output_path=(
            dirs["root"]
            / "kmeans_separation_summary.txt"
        ),
    )

    print("\nDiagnosis completed.")
    print(
        f"Results saved in: {OUTPUT_DIR}"
    )
    print(
        "Start from kmeans_separation_summary.txt, then inspect:\n"
        "- sample_metrics/sample_metric_comparison.csv\n"
        "- feature_separation/feature_separation.csv\n"
        "- pca_separation/pca_component_separation.csv\n"
        "- feature_cluster_separation/feature_cluster_separation.csv"
    )


if __name__ == "__main__":
    main()
