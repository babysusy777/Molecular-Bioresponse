#!/usr/bin/env python3
"""
Analisi esplorativa delle feature numeriche del Molecular Bioresponse dataset.

Pipeline
--------
1. Statistiche descrittive e analisi delle distribuzioni.
2. Clustering gerarchico delle feature mediante Spearman.
3. PCA sulle feature standardizzate e K-means nello spazio PCA.
4. Matrice riordinata per cluster delle righe e dendrogramma delle colonne.
5. Mean matrix e contrast matrix tra cluster di righe e cluster di feature.
6. Confronto post-hoc con Activity.

Activity non viene usata per costruire PCA o cluster.
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
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency, kurtosis, mannwhitneyu, pointbiserialr, skew
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(
    "/Users/susannabaldo/Desktop/Machine_Learning_Project/"
    "Molecular-Bioresponse"
)

DATA_PATH = BASE_DIR / "Dataset" / "train_numeric_only.csv"
TARGET_PATH = BASE_DIR / "Dataset" / "train_activity_target.csv"
OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "numerico"
    / "numeric_feature_exploration"
    / "reports"
)

RANDOM_STATE = 42

# Clustering delle feature
USE_ABSOLUTE_SPEARMAN = True
FEATURE_LINKAGE_METHOD = "average"
N_FEATURE_CLUSTERS = 12

# PCA e clustering delle molecole
PCA_VARIANCE_THRESHOLD = 0.90
K_MIN = 2
K_MAX = 10
KMEANS_N_INIT = 20
SILHOUETTE_SAMPLE_SIZE = 2000

# Grafici
TOP_N_FEATURES = 25
MATRIX_CLIP_Z = 3.0
DPI = 180


# ============================================================
# UTILITÀ
# ============================================================

def make_output_dirs() -> dict[str, Path]:
    dirs = {
        "root": OUTPUT_DIR,
        "distributions": OUTPUT_DIR / "distributions",
        "feature_clustering": OUTPUT_DIR / "feature_clustering",
        "pca": OUTPUT_DIR / "pca_clustering",
        "matrix": OUTPUT_DIR / "reordered_matrix",
        "activity": OUTPUT_DIR / "activity_comparison",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset numerico non trovato: {DATA_PATH}")
    if not TARGET_PATH.exists():
        raise FileNotFoundError(f"Target non trovato: {TARGET_PATH}")

    X = pd.read_csv(DATA_PATH)
    if "Activity" in X.columns:
        warnings.warn("Activity rimossa dal file delle feature.")
        X = X.drop(columns=["Activity"])

    non_numeric = [
        column for column in X.columns
        if not pd.api.types.is_numeric_dtype(X[column])
    ]
    if non_numeric:
        raise ValueError(f"Colonne non numeriche: {non_numeric[:10]}")
    if X.isna().any().any():
        raise ValueError("Il dataset contiene valori mancanti.")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Il dataset contiene valori infiniti.")

    target_df = pd.read_csv(TARGET_PATH)
    y = (
        target_df["Activity"]
        if "Activity" in target_df.columns
        else target_df.iloc[:, 0]
    ).reset_index(drop=True)

    X = X.reset_index(drop=True)
    if len(X) != len(y):
        raise ValueError("X e Activity hanno un numero diverso di righe.")
    if not set(pd.unique(y)).issubset({0, 1}):
        raise ValueError("Activity deve essere binaria e codificata come 0/1.")

    constant_columns = [
        column for column in X.columns
        if X[column].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        warnings.warn(f"Rimosse {len(constant_columns)} feature costanti.")
        X = X.drop(columns=constant_columns)

    print(f"Loaded numeric dataset: {X.shape}")
    print(f"Activity distribution: {y.value_counts().sort_index().to_dict()}")
    return X, y.astype(int)


# ============================================================
# 1. DISTRIBUZIONI NUMERICHE
# ============================================================

def analyze_distributions(X: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []

    for feature in X.columns:
        values = X[feature].to_numpy(dtype=float)
        positive = values[values > 0]
        frequencies = pd.Series(values).value_counts(normalize=True)

        rows.append({
            "feature": feature,
            "minimum": float(values.min()),
            "q01": float(np.quantile(values, 0.01)),
            "q05": float(np.quantile(values, 0.05)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "q75": float(np.quantile(values, 0.75)),
            "q95": float(np.quantile(values, 0.95)),
            "q99": float(np.quantile(values, 0.99)),
            "maximum": float(values.max()),
            "std": float(values.std(ddof=1)),
            "variance": float(values.var(ddof=1)),
            "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            "skewness": float(skew(values, bias=False)),
            "kurtosis_fisher": float(kurtosis(values, fisher=True, bias=False)),
            "zero_fraction": float(np.mean(values == 0)),
            "positive_fraction": float(np.mean(values > 0)),
            "unique_values": int(np.unique(values).size),
            "dominant_value_fraction": float(frequencies.iloc[0]),
            "positive_mean": float(positive.mean()) if positive.size else np.nan,
            "positive_median": float(np.median(positive)) if positive.size else np.nan,
        })

    report = pd.DataFrame(rows)
    report.to_csv(output_dir / "numeric_distribution_summary.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.hist(report["zero_fraction"], bins=30, edgecolor="black", linewidth=0.4)
    plt.xlabel("Zero fraction")
    plt.ylabel("Number of features")
    plt.title("Distribution of zero fractions")
    save_plot(output_dir / "zero_fraction_distribution.png")

    finite_skew = report["skewness"].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite_skew.empty:
        clipped = finite_skew.clip(
            lower=finite_skew.quantile(0.01),
            upper=finite_skew.quantile(0.99),
        )
        plt.figure(figsize=(9, 5))
        plt.hist(clipped, bins=30, edgecolor="black", linewidth=0.4)
        plt.xlabel("Skewness, clipped at 1st–99th percentiles")
        plt.ylabel("Number of features")
        plt.title("Distribution of feature skewness")
        save_plot(output_dir / "skewness_distribution.png")

    top_variance = report.nlargest(TOP_N_FEATURES, "variance").sort_values("variance")
    plt.figure(figsize=(10, 8))
    plt.barh(top_variance["feature"], top_variance["variance"])
    plt.xlabel("Variance")
    plt.ylabel("Feature")
    plt.title(f"Top {TOP_N_FEATURES} features by variance")
    save_plot(output_dir / "top_features_by_variance.png")

    top_zero = (
        report[report["zero_fraction"] < 1.0]
        .nlargest(TOP_N_FEATURES, "zero_fraction")
        .sort_values("zero_fraction")
    )
    plt.figure(figsize=(10, 8))
    plt.barh(top_zero["feature"], top_zero["zero_fraction"])
    plt.xlabel("Zero fraction")
    plt.ylabel("Feature")
    plt.title(f"Top {TOP_N_FEATURES} features by zero fraction")
    save_plot(output_dir / "top_features_by_zero_fraction.png")

    plt.figure(figsize=(8, 6))
    plt.scatter(report["zero_fraction"], report["std"], s=18, alpha=0.65)
    plt.xlabel("Zero fraction")
    plt.ylabel("Standard deviation")
    plt.title("Feature sparsity versus variability")
    save_plot(output_dir / "zero_fraction_vs_std.png")

    print(f"Distribution analysis completed: {len(report)} features.")
    return report


# ============================================================
# 2. CLUSTERING GERARCHICO DELLE FEATURE
# ============================================================

def cluster_features_spearman(X: pd.DataFrame, output_dir: Path) -> dict:
    print("Computing Spearman feature-correlation matrix...")
    correlation = X.corr(method="spearman")

    correlation = correlation.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)

    # Crea una copia NumPy esplicitamente scrivibile
    correlation_values = correlation.to_numpy(
        dtype=float,
        copy=True,
    )

    np.fill_diagonal(
        correlation_values,
        1.0,
    )

    # Ricostruisce il DataFrame mantenendo nomi e ordine delle feature
    correlation = pd.DataFrame(
        correlation_values,
        index=correlation.index,
        columns=correlation.columns,
    )

    if USE_ABSOLUTE_SPEARMAN:
        distance = 1.0 - correlation.abs()
        distance_name = "1 - |Spearman rho|"
    else:
        distance = 1.0 - correlation
        distance_name = "1 - Spearman rho"

    distance_values = distance.to_numpy(dtype=float)
    distance_values = (distance_values + distance_values.T) / 2.0
    distance_values = np.clip(distance_values, 0.0, 2.0)
    np.fill_diagonal(distance_values, 0.0)

    condensed = squareform(distance_values, checks=False)
    Z = linkage(
        condensed,
        method=FEATURE_LINKAGE_METHOD,
        optimal_ordering=True,
    )

    order_indices = leaves_list(Z)
    ordered_features = X.columns[order_indices].tolist()
    cluster_labels = fcluster(Z, t=N_FEATURE_CLUSTERS, criterion="maxclust")

    assignments = pd.DataFrame({
        "feature": X.columns,
        "feature_cluster": cluster_labels,
    })
    order_map = {feature: index for index, feature in enumerate(ordered_features)}
    assignments["dendrogram_order"] = assignments["feature"].map(order_map)
    assignments = assignments.sort_values("dendrogram_order")

    assignments.to_csv(output_dir / "feature_cluster_assignments.csv", index=False)
    correlation.to_csv(output_dir / "spearman_correlation_matrix.csv")
    pd.DataFrame({"feature": ordered_features}).to_csv(
        output_dir / "feature_dendrogram_order.csv", index=False
    )

    plt.figure(figsize=(16, 7))
    dendrogram(Z, no_labels=True, color_threshold=None)
    plt.xlabel("Numeric features")
    plt.ylabel("Linkage distance")
    plt.title(
        "Hierarchical clustering of numeric features\n"
        f"distance={distance_name}, linkage={FEATURE_LINKAGE_METHOD}"
    )
    save_plot(output_dir / "feature_dendrogram.png")

    reordered_correlation = correlation.loc[ordered_features, ordered_features]
    plt.figure(figsize=(11, 10))
    image = plt.imshow(
        reordered_correlation.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    plt.xlabel("Features in dendrogram order")
    plt.ylabel("Features in dendrogram order")
    plt.title("Reordered Spearman correlation matrix")
    plt.colorbar(image, label="Spearman correlation")
    save_plot(output_dir / "reordered_spearman_matrix.png")

    print(f"Feature clustering completed: {assignments['feature_cluster'].nunique()} clusters.")
    return {
        "correlation": correlation,
        "linkage": Z,
        "ordered_features": ordered_features,
        "assignments": assignments,
        "distance_name": distance_name,
    }


# ============================================================
# 3. PCA E K-MEANS
# ============================================================

def run_pca_kmeans(X: pd.DataFrame, y: pd.Series, output_dir: Path) -> dict:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, svd_solver="full")
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    pd.DataFrame({
        "component": np.arange(1, len(explained) + 1),
        "explained_variance_ratio": explained,
        "cumulative_explained_variance": cumulative,
    }).to_csv(output_dir / "pca_explained_variance.csv", index=False)

    scores = pd.DataFrame(
        X_pca,
        columns=[f"PC{i}" for i in range(1, X_pca.shape[1] + 1)],
    )
    scores.insert(0, "row_index", np.arange(len(scores)))
    scores["Activity"] = y.to_numpy()
    scores.to_csv(output_dir / "pca_scores.csv", index=False)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=[f"PC{i}" for i in range(1, X_pca.shape[1] + 1)],
    )
    loadings.to_csv(output_dir / "pca_loadings.csv")

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", markersize=3)
    plt.axhline(PCA_VARIANCE_THRESHOLD, linestyle="--", linewidth=1.2)
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title(
        f"PCA cumulative variance: {len(explained)} components retain "
        f"{cumulative[-1]:.4f}"
    )
    save_plot(output_dir / "pca_cumulative_variance.png")

    metric_rows: list[dict] = []
    candidate_models: dict[int, KMeans] = {}
    candidate_labels: dict[int, np.ndarray] = {}

    for k in range(K_MIN, K_MAX + 1):
        model = KMeans(
            n_clusters=k,
            n_init=KMEANS_N_INIT,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(X_pca)
        silhouette = silhouette_score(
            X_pca,
            labels,
            sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(X_pca)),
            random_state=RANDOM_STATE,
        )
        metric_rows.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette": float(silhouette),
            "calinski_harabasz": float(calinski_harabasz_score(X_pca, labels)),
            "davies_bouldin": float(davies_bouldin_score(X_pca, labels)),
        })
        candidate_models[k] = model
        candidate_labels[k] = labels
        print(f"k={k}: silhouette={silhouette:.4f}, inertia={model.inertia_:.4f}")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "kmeans_model_selection.csv", index=False)
    best_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
    best_model = candidate_models[best_k]
    labels = candidate_labels[best_k]

    plt.figure(figsize=(8, 5))
    plt.plot(metrics["k"], metrics["silhouette"], marker="o")
    plt.xlabel("Number of clusters k")
    plt.ylabel("Silhouette score")
    plt.title("K-means selection by silhouette")
    save_plot(output_dir / "kmeans_silhouette.png")

    plt.figure(figsize=(8, 5))
    plt.plot(metrics["k"], metrics["inertia"], marker="o")
    plt.xlabel("Number of clusters k")
    plt.ylabel("Inertia")
    plt.title("K-means elbow curve")
    save_plot(output_dir / "kmeans_elbow.png")

    # ============================================================
    # PCA 3D: cluster K-means
    # ============================================================

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    cluster_plot = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2],
        c=labels,
        s=18,
        alpha=0.70,
        cmap="tab10",
    )

    ax.set_xlabel(f"PC1 ({explained[0]:.2%})")
    ax.set_ylabel(f"PC2 ({explained[1]:.2%})")
    ax.set_zlabel(f"PC3 ({explained[2]:.2%})")

    ax.set_title(
        f"PCA numeric features: K-means clusters, k={best_k}"
    )

    # Angolo iniziale di visualizzazione
    ax.view_init(elev=22, azim=45)

    fig.colorbar(
        cluster_plot,
        ax=ax,
        label="K-means cluster",
        shrink=0.70,
        pad=0.10,
    )

    save_plot(
        output_dir / "pca_3d_scatter_kmeans.png"
    )


    # ============================================================
    # PCA 3D: Activity
    # ============================================================

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    activity_markers = {
        0: "o",
        1: "^",
    }

    for activity in (0, 1):
        mask = y.to_numpy() == activity

        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            X_pca[mask, 2],
            s=18,
            alpha=0.55,
            marker=activity_markers[activity],
            label=f"Activity={activity}",
        )

    ax.set_xlabel(f"PC1 ({explained[0]:.2%})")
    ax.set_ylabel(f"PC2 ({explained[1]:.2%})")
    ax.set_zlabel(f"PC3 ({explained[2]:.2%})")

    ax.set_title("PCA numeric features by Activity")

    ax.view_init(elev=22, azim=45)
    ax.legend()

    save_plot(
        output_dir / "pca_3d_scatter_activity.png"
    )

    pd.DataFrame({
        "row_index": np.arange(len(X)),
        "kmeans_cluster": labels,
        "Activity": y.to_numpy(),
    }).to_csv(output_dir / "kmeans_cluster_assignments.csv", index=False)

    print(f"PCA retained {X_pca.shape[1]} components; best k={best_k}.")
    return {
        "scaler": scaler,
        "X_scaled": X_scaled,
        "pca": pca,
        "X_pca": X_pca,
        "explained": explained,
        "metrics": metrics,
        "best_k": best_k,
        "model": best_model,
        "labels": labels,
    }


# ============================================================
# 4. REORDERED MATRIX, MEAN E CONTRAST
# ============================================================

def make_reordered_matrix(
    X: pd.DataFrame,
    feature_results: dict,
    pca_results: dict,
    output_dir: Path,
) -> dict:
    X_scaled = pca_results["X_scaled"]
    X_pca = pca_results["X_pca"]
    row_labels = pca_results["labels"]
    ordered_features = feature_results["ordered_features"]

    assignment_map = (
        feature_results["assignments"]
        .set_index("feature")["feature_cluster"]
    )
    ordered_feature_clusters = assignment_map.loc[ordered_features].to_numpy(dtype=int)
    column_indices = np.array([X.columns.get_loc(f) for f in ordered_features], dtype=int)

    # Ordina le righe prima per cluster K-means e poi per PC1.
    row_indices = np.lexsort((X_pca[:, 0], row_labels))
    ordered_row_labels = row_labels[row_indices]

    raw_matrix = X.to_numpy(dtype=float)[np.ix_(row_indices, column_indices)]
    z_matrix = X_scaled[np.ix_(row_indices, column_indices)]

    pd.DataFrame({
        "reordered_position": np.arange(len(row_indices)),
        "original_row_index": row_indices,
        "kmeans_cluster": ordered_row_labels,
    }).to_csv(output_dir / "row_order.csv", index=False)

    pd.DataFrame({
        "reordered_position": np.arange(len(ordered_features)),
        "feature": ordered_features,
        "feature_cluster": ordered_feature_clusters,
    }).to_csv(output_dir / "column_order.csv", index=False)

    row_boundaries = np.flatnonzero(np.diff(ordered_row_labels) != 0) + 0.5
    column_boundaries = np.flatnonzero(np.diff(ordered_feature_clusters) != 0) + 0.5

    plt.figure(figsize=(18, 10))
    raw_image = plt.imshow(raw_matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    for boundary in row_boundaries:
        plt.axhline(boundary, linewidth=0.7, color="white")
    for boundary in column_boundaries:
        plt.axvline(boundary, linewidth=0.5, color="white")
    plt.xlabel("Features in Spearman dendrogram order")
    plt.ylabel("Samples ordered by K-means cluster and PC1")
    plt.title("Reordered raw numeric matrix")
    plt.colorbar(raw_image, label="Raw value")
    save_plot(output_dir / "reordered_raw_matrix.png")

    clipped_z = np.clip(z_matrix, -MATRIX_CLIP_Z, MATRIX_CLIP_Z)
    plt.figure(figsize=(18, 10))
    z_image = plt.imshow(
        clipped_z,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-MATRIX_CLIP_Z,
        vmax=MATRIX_CLIP_Z,
    )
    for boundary in row_boundaries:
        plt.axhline(boundary, linewidth=0.7, color="black")
    for boundary in column_boundaries:
        plt.axvline(boundary, linewidth=0.5, color="black")
    plt.xlabel("Features in Spearman dendrogram order")
    plt.ylabel("Samples ordered by K-means cluster and PC1")
    plt.title(f"Reordered standardized matrix, clipped to ±{MATRIX_CLIP_Z:g}")
    plt.colorbar(z_image, label="Standardized value")
    save_plot(output_dir / "reordered_standardized_matrix.png")

    feature_cluster_by_original_column = np.array(
        [assignment_map.loc[feature] for feature in X.columns], dtype=int
    )
    feature_cluster_ids = list(dict.fromkeys(ordered_feature_clusters.tolist()))
    sample_cluster_ids = sorted(np.unique(row_labels).tolist())

    mean_matrix = np.full((len(sample_cluster_ids), len(feature_cluster_ids)), np.nan)
    contrast_matrix = np.full_like(mean_matrix, np.nan)

    for i, sample_cluster in enumerate(sample_cluster_ids):
        inside = row_labels == sample_cluster
        outside = ~inside
        for j, feature_cluster in enumerate(feature_cluster_ids):
            selected_features = feature_cluster_by_original_column == feature_cluster
            inside_mean = float(X_scaled[np.ix_(inside, selected_features)].mean())
            outside_mean = float(X_scaled[np.ix_(outside, selected_features)].mean())
            mean_matrix[i, j] = inside_mean
            contrast_matrix[i, j] = inside_mean - outside_mean

    row_names = [f"SampleCluster_{c}" for c in sample_cluster_ids]
    column_names = [f"FeatureCluster_{c}" for c in feature_cluster_ids]
    mean_df = pd.DataFrame(mean_matrix, index=row_names, columns=column_names)
    contrast_df = pd.DataFrame(contrast_matrix, index=row_names, columns=column_names)
    mean_df.to_csv(output_dir / "block_mean_matrix.csv")
    contrast_df.to_csv(output_dir / "block_contrast_matrix.csv")

    mean_limit = max(float(np.abs(mean_matrix).max()), 1e-9)
    plt.figure(figsize=(13, 7))
    mean_image = plt.imshow(
        mean_matrix, aspect="auto", cmap="coolwarm", vmin=-mean_limit, vmax=mean_limit
    )
    plt.xticks(np.arange(len(column_names)), column_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(row_names)), row_names)
    plt.xlabel("Spearman feature cluster")
    plt.ylabel("K-means sample cluster")
    plt.title("Mean matrix: mean standardized value in each block")
    plt.colorbar(mean_image, label="Mean standardized value")
    save_plot(output_dir / "block_mean_matrix.png")

    contrast_limit = max(float(np.abs(contrast_matrix).max()), 1e-9)
    plt.figure(figsize=(13, 7))
    contrast_image = plt.imshow(
        contrast_matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-contrast_limit,
        vmax=contrast_limit,
    )
    plt.xticks(np.arange(len(column_names)), column_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(row_names)), row_names)
    plt.xlabel("Spearman feature cluster")
    plt.ylabel("K-means sample cluster")
    plt.title("Contrast matrix: cluster mean minus outside-cluster mean")
    plt.colorbar(contrast_image, label="Standardized mean contrast")
    save_plot(output_dir / "block_contrast_matrix.png")

    print("Reordered matrix, mean matrix and contrast matrix completed.")
    return {
        "feature_cluster_by_original_column": feature_cluster_by_original_column,
        "feature_cluster_ids": feature_cluster_ids,
        "mean_matrix": mean_df,
        "contrast_matrix": contrast_df,
    }


# ============================================================
# 5. CONFRONTO CON ACTIVITY
# ============================================================

def compare_with_activity(
    X: pd.DataFrame,
    y: pd.Series,
    feature_results: dict,
    pca_results: dict,
    matrix_results: dict,
    output_dir: Path,
) -> dict:
    labels = pca_results["labels"]
    X_scaled = pca_results["X_scaled"]

    # Associazione tra cluster K-means e Activity.
    contingency = pd.crosstab(
        pd.Series(labels, name="kmeans_cluster"),
        pd.Series(y.to_numpy(), name="Activity"),
    )
    contingency.to_csv(output_dir / "kmeans_activity_contingency.csv")

    chi2, chi2_p, degrees_freedom, _ = chi2_contingency(contingency)
    denominator = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = (
        float(np.sqrt(chi2 / (contingency.to_numpy().sum() * denominator)))
        if denominator > 0 else np.nan
    )
    association = {
        "chi_square": float(chi2),
        "chi_square_p_value": float(chi2_p),
        "degrees_of_freedom": int(degrees_freedom),
        "cramers_v": cramers_v,
        "normalized_mutual_information": float(normalized_mutual_info_score(y, labels)),
        "adjusted_rand_index": float(adjusted_rand_score(y, labels)),
        "global_activity_1_fraction": float(y.mean()),
    }
    (output_dir / "kmeans_activity_association.json").write_text(
        json.dumps(association, indent=2), encoding="utf-8"
    )

    cluster_summary = (
        pd.DataFrame({"kmeans_cluster": labels, "Activity": y.to_numpy()})
        .groupby("kmeans_cluster")
        .agg(
            n_samples=("Activity", "size"),
            activity_0_count=("Activity", lambda v: int(np.sum(v == 0))),
            activity_1_count=("Activity", lambda v: int(np.sum(v == 1))),
            activity_1_fraction=("Activity", "mean"),
        )
        .reset_index()
    )
    cluster_summary["global_activity_1_fraction"] = float(y.mean())
    cluster_summary["activity_1_enrichment"] = (
        cluster_summary["activity_1_fraction"] - float(y.mean())
    )
    cluster_summary.to_csv(output_dir / "kmeans_activity_summary.csv", index=False)

    plt.figure(figsize=(9, 6))
    plt.bar(cluster_summary["kmeans_cluster"].astype(str), cluster_summary["activity_1_fraction"])
    plt.axhline(float(y.mean()), linestyle="--", linewidth=1.4, label="Global fraction")
    plt.xlabel("K-means cluster")
    plt.ylabel("Activity=1 fraction")
    plt.title("Activity composition of PCA/K-means clusters")
    plt.legend()
    save_plot(output_dir / "activity_fraction_by_kmeans_cluster.png")

    # Associazione tra singole feature e Activity.
    y_values = y.to_numpy()
    mask_0 = y_values == 0
    mask_1 = y_values == 1
    n0 = int(mask_0.sum())
    n1 = int(mask_1.sum())
    mutual_information = mutual_info_classif(
        X.to_numpy(dtype=float), y_values,
        discrete_features=False,
        random_state=RANDOM_STATE,
    )

    feature_cluster_map = (
        feature_results["assignments"]
        .set_index("feature")["feature_cluster"]
    )
    feature_rows: list[dict] = []

    for index, feature in enumerate(X.columns):
        values = X[feature].to_numpy(dtype=float)
        values_0 = values[mask_0]
        values_1 = values[mask_1]

        mw = mannwhitneyu(values_1, values_0, alternative="two-sided", method="auto")
        u_statistic = float(mw.statistic)
        rank_biserial = float(2.0 * u_statistic / (n1 * n0) - 1.0)
        pb_correlation, pb_p = pointbiserialr(y_values, values)

        feature_rows.append({
            "feature": feature,
            "feature_cluster": int(feature_cluster_map.loc[feature]),
            "activity_0_mean": float(values_0.mean()),
            "activity_1_mean": float(values_1.mean()),
            "mean_difference_1_minus_0": float(values_1.mean() - values_0.mean()),
            "activity_0_median": float(np.median(values_0)),
            "activity_1_median": float(np.median(values_1)),
            "mann_whitney_u": u_statistic,
            "mann_whitney_p_value": float(mw.pvalue),
            "rank_biserial_correlation": rank_biserial,
            "point_biserial_correlation": float(pb_correlation),
            "point_biserial_p_value": float(pb_p),
            "mutual_information": float(mutual_information[index]),
        })

    feature_activity = pd.DataFrame(feature_rows)
    rejected, adjusted_p, _, _ = multipletests(
        feature_activity["mann_whitney_p_value"].to_numpy(),
        alpha=0.05,
        method="fdr_bh",
    )
    feature_activity["mann_whitney_p_fdr_bh"] = adjusted_p
    feature_activity["significant_fdr_0.05"] = rejected
    feature_activity["absolute_rank_biserial"] = (
        feature_activity["rank_biserial_correlation"].abs()
    )
    feature_activity = feature_activity.sort_values(
        ["absolute_rank_biserial", "mutual_information"], ascending=[False, False]
    )
    feature_activity.to_csv(output_dir / "feature_activity_comparison.csv", index=False)

    top_effects = feature_activity.head(TOP_N_FEATURES).sort_values("rank_biserial_correlation")
    plt.figure(figsize=(11, 9))
    plt.barh(top_effects["feature"], top_effects["rank_biserial_correlation"])
    plt.axvline(0.0, linewidth=1.0)
    plt.xlabel("Rank-biserial correlation; positive means larger for Activity=1")
    plt.ylabel("Feature")
    plt.title(f"Top {TOP_N_FEATURES} numeric features associated with Activity")
    save_plot(output_dir / "top_activity_effects.png")

    # Contrasto Activity aggregato per cluster di feature.
    feature_cluster_by_column = matrix_results["feature_cluster_by_original_column"]
    activity_cluster_rows: list[dict] = []
    for feature_cluster in matrix_results["feature_cluster_ids"]:
        selected = feature_cluster_by_column == feature_cluster
        mean_0 = float(X_scaled[np.ix_(mask_0, selected)].mean())
        mean_1 = float(X_scaled[np.ix_(mask_1, selected)].mean())
        activity_cluster_rows.append({
            "feature_cluster": feature_cluster,
            "n_features": int(selected.sum()),
            "activity_0_standardized_mean": mean_0,
            "activity_1_standardized_mean": mean_1,
            "activity_contrast_1_minus_0": mean_1 - mean_0,
        })

    activity_feature_clusters = pd.DataFrame(activity_cluster_rows).sort_values(
        "activity_contrast_1_minus_0"
    )
    activity_feature_clusters.to_csv(
        output_dir / "activity_contrast_by_feature_cluster.csv", index=False
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        activity_feature_clusters["feature_cluster"].astype(str),
        activity_feature_clusters["activity_contrast_1_minus_0"],
    )
    plt.axvline(0.0, linewidth=1.0)
    plt.xlabel("Mean standardized contrast: Activity=1 minus Activity=0")
    plt.ylabel("Spearman feature cluster")
    plt.title("Activity contrast by feature cluster")
    save_plot(output_dir / "activity_contrast_by_feature_cluster.png")

    print("Activity comparison completed.")
    return {
        "association": association,
        "cluster_summary": cluster_summary,
        "feature_activity": feature_activity,
    }


# ============================================================
# REPORT RIASSUNTIVO
# ============================================================

def write_summary(
    X: pd.DataFrame,
    y: pd.Series,
    distributions: pd.DataFrame,
    feature_results: dict,
    pca_results: dict,
    activity_results: dict,
    output_path: Path,
) -> None:
    lines = [
        "NUMERIC FEATURE EXPLORATION SUMMARY",
        "=" * 42,
        f"Samples: {X.shape[0]}",
        f"Numeric features: {X.shape[1]}",
        f"Global Activity=1 fraction: {y.mean():.6f}",
        "",
        "DISTRIBUTIONS",
        f"Median zero fraction: {distributions['zero_fraction'].median():.6f}",
        f"Features with zero fraction >= 0.95: {(distributions['zero_fraction'] >= 0.95).sum()}",
        f"Features with dominant value fraction >= 0.95: {(distributions['dominant_value_fraction'] >= 0.95).sum()}",
        "",
        "FEATURE CLUSTERING",
        f"Distance: {feature_results['distance_name']}",
        f"Linkage: {FEATURE_LINKAGE_METHOD}",
        f"Feature clusters: {feature_results['assignments']['feature_cluster'].nunique()}",
        "",
        "PCA AND K-MEANS",
        f"Retained PCs: {pca_results['X_pca'].shape[1]}",
        f"Retained variance: {pca_results['explained'].sum():.6f}",
        f"Best k by silhouette: {pca_results['best_k']}",
        f"Best silhouette: {pca_results['metrics']['silhouette'].max():.6f}",
        "",
        "ACTIVITY ASSOCIATION",
        f"Chi-square p-value: {activity_results['association']['chi_square_p_value']:.6g}",
        f"Cramer's V: {activity_results['association']['cramers_v']:.6f}",
        f"NMI: {activity_results['association']['normalized_mutual_information']:.6f}",
        f"ARI: {activity_results['association']['adjusted_rand_index']:.6f}",
        "",
        "Top features by absolute rank-biserial effect:",
    ]

    for row in activity_results["feature_activity"].head(10).itertuples():
        lines.append(
            f"{row.feature}: r_rb={row.rank_biserial_correlation:.6f}, "
            f"FDR p={row.mann_whitney_p_fdr_bh:.6g}, MI={row.mutual_information:.6f}"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    dirs = make_output_dirs()
    X, y = load_data()

    print("\n1/5 Distribution analysis")
    distributions = analyze_distributions(X, dirs["distributions"])

    print("\n2/5 Spearman feature clustering")
    feature_results = cluster_features_spearman(X, dirs["feature_clustering"])

    print("\n3/5 PCA and K-means")
    pca_results = run_pca_kmeans(X, y, dirs["pca"])

    print("\n4/5 Reordered matrix, mean and contrast")
    matrix_results = make_reordered_matrix(
        X, feature_results, pca_results, dirs["matrix"]
    )

    print("\n5/5 Comparison with Activity")
    activity_results = compare_with_activity(
        X, y, feature_results, pca_results, matrix_results, dirs["activity"]
    )

    write_summary(
        X,
        y,
        distributions,
        feature_results,
        pca_results,
        activity_results,
        dirs["root"] / "summary.txt",
    )

    print("\nAnalysis completed.")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("Activity was used only for post-hoc interpretation.")


if __name__ == "__main__":
    main()
