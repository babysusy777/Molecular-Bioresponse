from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_PATH = BASE_DIR / "Dataset" / "raw" / "train.csv"

OUTPUT_DIR = BASE_DIR / "reports" / "numeric_minmax_pca_kmeans"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
PCA_VARIANCE_THRESHOLD = 0.90
K_VALUES = [2, 3]


# ============================================================
# LETTURA DATI
# ============================================================

train = pd.read_csv(TRAIN_PATH)

if "Activity" in train.columns:
    y = train["Activity"].copy()
    X = train.drop(columns="Activity")
else:
    y = None
    X = train.copy()

X = X.apply(pd.to_numeric, errors="raise")

if X.isna().any().any():
    raise ValueError("Il dataset contiene valori mancanti.")

if np.isinf(X.to_numpy()).any():
    raise ValueError("Il dataset contiene valori infiniti.")


# ============================================================
# SELEZIONE DELLE SOLE FEATURE NUMERICHE NON BINARIE
# ============================================================

binary_features = [
    column
    for column in X.columns
    if X[column].dropna().isin([0, 1]).all()
]

numeric_non_binary_features = [
    column
    for column in X.select_dtypes(include="number").columns
    if column not in binary_features
]

X_numeric = X[numeric_non_binary_features].copy()

print("\nSELEZIONE DELLE FEATURE")
print("-" * 60)
print(f"Feature totali originali: {X.shape[1]}")
print(f"Feature binarie escluse: {len(binary_features)}")
print(
    "Feature numeriche non binarie utilizzate: "
    f"{X_numeric.shape[1]}"
)
print(f"Campioni: {X_numeric.shape[0]}")

if X_numeric.shape[1] == 0:
    raise ValueError("Non sono state trovate feature numeriche non binarie.")


# ============================================================
# NORMALIZZAZIONE MIN-MAX
# ============================================================

scaler = MinMaxScaler(feature_range=(0, 1))

X_numeric_scaled_array = scaler.fit_transform(X_numeric)

X_numeric_scaled = pd.DataFrame(
    X_numeric_scaled_array,
    columns=X_numeric.columns,
    index=X_numeric.index,
)

print("\nNORMALIZZAZIONE MIN-MAX")
print("-" * 60)
print(
    f"Minimo globale: "
    f"{X_numeric_scaled.min().min():.6f}"
)
print(
    f"Massimo globale: "
    f"{X_numeric_scaled.max().max():.6f}"
)

feature_ranges = pd.DataFrame({
    "original_min": X_numeric.min(),
    "original_max": X_numeric.max(),
    "scaled_min": X_numeric_scaled.min(),
    "scaled_max": X_numeric_scaled.max(),
})

feature_ranges["original_range"] = (
    feature_ranges["original_max"]
    - feature_ranges["original_min"]
)

feature_ranges["scaled_range"] = (
    feature_ranges["scaled_max"]
    - feature_ranges["scaled_min"]
)

feature_ranges.to_csv(
    OUTPUT_DIR / "numeric_feature_ranges.csv",
    index_label="feature",
)


# ============================================================
# PCA AL 90% DELLA VARIANZA
# ============================================================

pca = PCA(
    n_components=PCA_VARIANCE_THRESHOLD,
    svd_solver="full",
)

X_pca = pca.fit_transform(X_numeric_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\nPCA SULLE FEATURE NUMERICHE")
print("-" * 60)
print(f"Feature iniziali: {X_numeric_scaled.shape[1]}")
print(f"Componenti mantenute: {X_pca.shape[1]}")
print(
    "Varianza spiegata cumulativa: "
    f"{cumulative_variance[-1]:.4%}"
)
print(f"Varianza PC1: {explained_variance[0]:.4%}")
print(f"Varianza PC2: {explained_variance[1]:.4%}")

if X_pca.shape[1] >= 3:
    print(f"Varianza PC3: {explained_variance[2]:.4%}")


# ============================================================
# SCREE PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    np.arange(1, len(cumulative_variance) + 1),
    cumulative_variance,
)

plt.axhline(
    y=PCA_VARIANCE_THRESHOLD,
    linestyle="--",
    label="90% della varianza",
)

plt.xlabel("Numero di componenti principali")
plt.ylabel("Varianza spiegata cumulativa")
plt.title("Min-Max + PCA sulle sole feature numeriche")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "numeric_pca_cumulative_variance.png",
    dpi=300,
)

plt.close()


# ============================================================
# K-MEANS PER K = 3 E K = 4
# ============================================================

all_metrics = []

for k in K_VALUES:

    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=50,
        max_iter=500,
        random_state=RANDOM_STATE,
    )

    labels = kmeans.fit_predict(X_pca)

    silhouette = silhouette_score(X_pca, labels)
    davies_bouldin = davies_bouldin_score(X_pca, labels)
    calinski_harabasz = calinski_harabasz_score(
        X_pca,
        labels,
    )

    all_metrics.append({
        "k": k,
        "n_original_features": X_numeric.shape[1],
        "n_pca_components": X_pca.shape[1],
        "explained_variance": cumulative_variance[-1],
        "inertia": kmeans.inertia_,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
    })

    print(f"\nK-MEANS, K={k}")
    print("-" * 60)
    print(f"Inertia: {kmeans.inertia_:.6f}")
    print(f"Silhouette: {silhouette:.6f}")
    print(f"Davies-Bouldin: {davies_bouldin:.6f}")
    print(f"Calinski-Harabasz: {calinski_harabasz:.6f}")

    # --------------------------------------------------------
    # ASSEGNAZIONI E COMPOSIZIONE DEI CLUSTER
    # --------------------------------------------------------

    assignments = pd.DataFrame({
        "sample_index": X_numeric.index,
        "cluster": labels,
    })

    if y is not None:
        assignments["Activity"] = y.to_numpy()

        cluster_summary = (
            assignments
            .groupby("cluster", as_index=False)
            .agg(
                size=("cluster", "size"),
                activity_0=(
                    "Activity",
                    lambda values: (values == 0).sum(),
                ),
                activity_1=(
                    "Activity",
                    lambda values: (values == 1).sum(),
                ),
                activity_1_fraction=("Activity", "mean"),
            )
        )
    else:
        cluster_summary = (
            assignments
            .groupby("cluster", as_index=False)
            .size()
        )

    cluster_summary["cluster_fraction"] = (
        cluster_summary["size"] / len(assignments)
    )

    print("\nComposizione dei cluster:")
    print(cluster_summary.to_string(index=False))

    assignments.to_csv(
        OUTPUT_DIR / f"numeric_kmeans_k{k}_assignments.csv",
        index=False,
    )

    cluster_summary.to_csv(
        OUTPUT_DIR / f"numeric_kmeans_k{k}_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # GRAFICO 2D
    # --------------------------------------------------------

    centroids = kmeans.cluster_centers_

    plt.figure(figsize=(9, 7))

    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        s=18,
        alpha=0.70,
        cmap="tab10",
    )

    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=220,
        edgecolors="black",
        linewidths=1.5,
        label="Centroidi",
    )

    plt.xlabel(f"PC1 ({explained_variance[0]:.2%})")
    plt.ylabel(f"PC2 ({explained_variance[1]:.2%})")

    plt.title(
        "Feature numeriche: Min-Max + PCA + K-means\n"
        f"K={k}, silhouette={silhouette:.4f}"
    )

    plt.colorbar(scatter, label="Cluster")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"numeric_pca_kmeans_k{k}_2d.png",
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------------
    # GRAFICO 3D
    # --------------------------------------------------------

    if X_pca.shape[1] >= 3:

        fig = plt.figure(figsize=(10, 8))
        axis = fig.add_subplot(111, projection="3d")

        scatter_3d = axis.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            X_pca[:, 2],
            c=labels,
            s=18,
            alpha=0.65,
            cmap="tab10",
        )

        axis.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            marker="X",
            s=250,
            edgecolors="black",
            linewidths=1.5,
            label="Centroidi",
        )

        axis.set_xlabel(
            f"PC1 ({explained_variance[0]:.2%})"
        )
        axis.set_ylabel(
            f"PC2 ({explained_variance[1]:.2%})"
        )
        axis.set_zlabel(
            f"PC3 ({explained_variance[2]:.2%})"
        )

        axis.set_title(
            "Feature numeriche: Min-Max + PCA + K-means\n"
            f"K={k}"
        )

        fig.colorbar(
            scatter_3d,
            ax=axis,
            label="Cluster",
            shrink=0.7,
        )

        axis.legend()
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"numeric_pca_kmeans_k{k}_3d.png",
            dpi=300,
        )

        plt.close()


# ============================================================
# CONFRONTO FINALE
# ============================================================

metrics_df = pd.DataFrame(all_metrics)

metrics_df.to_csv(
    OUTPUT_DIR / "numeric_kmeans_metrics.csv",
    index=False,
)

print("\nCONFRONTO K=3 E K=4")
print("-" * 110)
print(metrics_df.to_string(index=False))

best_row = metrics_df.loc[
    metrics_df["silhouette"].idxmax()
]

print(
    "\nConfigurazione con silhouette maggiore: "
    f"K={int(best_row['k'])}, "
    f"silhouette={best_row['silhouette']:.6f}"
)

print(f"\nRisultati salvati in:\n{OUTPUT_DIR}")