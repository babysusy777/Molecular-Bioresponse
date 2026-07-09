from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D



BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

INPUT_PATH = PROJECT_DIR / "Dataset" / "train_filtered_no_activity.csv"

PCA_OUTPUT_PATH = PROJECT_DIR / "Dataset" / "train_pca_90_no_activity.csv"

PCA_REPORT_PATH = PROJECT_DIR / "reports" / "pca_explained_variance.csv"
PCA_PLOT_PATH = PROJECT_DIR / "reports" / "pca_scree_plot_90.png"

ELBOW_REPORT_PATH = PROJECT_DIR / "reports" / "kmeans_elbow_values.csv"
ELBOW_PLOT_PATH = PROJECT_DIR / "reports" / "kmeans_elbow_plot.png"

CLUSTER_PLOTS_DIR = PROJECT_DIR / "reports" / "kmeans_cluster_plots"
TOP_N_CLUSTER_PLOTS = 3

VARIANCE_TO_KEEP = 0.90
K_RANGE = range(2, 16)


def main():
    X = pd.read_csv(INPUT_PATH)

    print(f"Loaded dataset: {X.shape[0]} rows x {X.shape[1]} features")

    # PCA 90%
    pca = PCA(n_components=VARIANCE_TO_KEEP)
    X_pca = pca.fit_transform(X)

    pc_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca_df = pd.DataFrame(X_pca, columns=pc_columns)

    PCA_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PCA_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    X_pca_df.to_csv(PCA_OUTPUT_PATH, index=False)

    explained = pca.explained_variance_ratio_
    cumulative = explained.cumsum()

    pca_report = pd.DataFrame({"component": pc_columns, "explained_variance_ratio": explained, "cumulative_explained_variance": cumulative})
    pca_report.to_csv(PCA_REPORT_PATH, index=False)

    # PCA plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(explained) + 1), explained, label="Explained variance ratio")
    plt.plot(range(1, len(cumulative) + 1), cumulative, marker="o", label="Cumulative explained variance")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance")
    plt.title("PCA scree plot - 90% variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PCA_PLOT_PATH, dpi=150)
    plt.close()

    print()
    print("PCA completed")
    print(f"Original features: {X.shape[1]}")
    print(f"PCA components: {X_pca.shape[1]}")
    print(f"Explained variance kept: {explained.sum():.4f}")

    # K-Means elbow
    inertias = []
    silhouettes = []

    for k in K_RANGE:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = kmeans.fit_predict(X_pca)

        inertia = kmeans.inertia_
        silhouette = silhouette_score(X_pca, labels)

        inertias.append(inertia)
        silhouettes.append(silhouette)


        print(
            f"k={k} | "
            f"inertia={inertia:.2f} | "
            f"silhouette={silhouette:.4f} "
        )

    elbow_report = pd.DataFrame({
        "k": list(K_RANGE),
        "inertia": inertias,
        "silhouette": silhouettes
    })

    elbow_report.to_csv(ELBOW_REPORT_PATH, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(list(K_RANGE), inertias, marker="o")
    plt.xlabel("Number of clusters k")
    plt.ylabel("Inertia")
    plt.title("K-Means elbow method on PCA space")
    plt.xticks(list(K_RANGE))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ELBOW_PLOT_PATH, dpi=150)
    plt.close()

        # Plot clusters for the 3 best k values by silhouette score
    # Plot clusters for the 3 best k values by silhouette score in 3D
    CLUSTER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if X_pca.shape[1] < 3:
        print("Cannot plot 3D clusters: PCA produced less than 3 components.")
    else:
        top_k = elbow_report.sort_values("silhouette", ascending=False).head(TOP_N_CLUSTER_PLOTS)

        print()
        print("Best k values by silhouette:")
        print(top_k)

        for _, row in top_k.iterrows():
            k = int(row["k"])
            silhouette = row["silhouette"]

            kmeans = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=20
            )

            labels = kmeans.fit_predict(X_pca)
            centroids = kmeans.cluster_centers_

            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(111, projection="3d")

            ax.scatter(
                X_pca[:, 0],
                X_pca[:, 1],
                X_pca[:, 2],
                c=labels,
                s=10
            )

            ax.scatter(
                centroids[:, 0],
                centroids[:, 1],
                centroids[:, 2],
                marker="X",
                s=160,
                edgecolors="black"
            )

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")

            ax.set_title(f"K-Means clustering | k={k} | silhouette={silhouette:.4f}")

            plt.tight_layout()

            plot_path = CLUSTER_PLOTS_DIR / f"kmeans_clusters_3d_k{k}_silhouette_{silhouette:.4f}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()

            print(f"3D cluster plot saved: {plot_path}")

    print()
    print("Saved files:")
    print(f"PCA dataset: {PCA_OUTPUT_PATH}")
    print(f"PCA report: {PCA_REPORT_PATH}")
    print(f"PCA plot: {PCA_PLOT_PATH}")
    print(f"Elbow report: {ELBOW_REPORT_PATH}")
    print(f"Elbow plot: {ELBOW_PLOT_PATH}")

if __name__ == "__main__":
    main()

"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

INPUT_PATH = PROJECT_DIR / "Dataset" / "train_filtered_no_activity.csv"

REPORT_PATH = PROJECT_DIR / "reports" / "kmeans_pca_dimensions_test.csv"
SILHOUETTE_PLOT_PATH = PROJECT_DIR / "reports" / "kmeans_silhouette_by_pca_dim.png"

PCA_DIMS = [2, 3, 5, 10, 20, 30, 50]
K_RANGE = range(2, 16)


def main():
    X = pd.read_csv(INPUT_PATH)

    print(f"Loaded dataset: {X.shape[0]} rows x {X.shape[1]} features")

    results = []

    for n_components in PCA_DIMS:
        print()
        print(f"Testing PCA with {n_components} components")

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        explained_variance = pca.explained_variance_ratio_.sum()

        for k in K_RANGE:
            kmeans = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=50,
                random_state=42
            )

            labels = kmeans.fit_predict(X_pca)

            inertia = kmeans.inertia_
            silhouette = silhouette_score(X_pca, labels)

            results.append({
                "pca_components": n_components,
                "k": k,
                "explained_variance": explained_variance,
                "inertia": inertia,
                "silhouette": silhouette
            })

            print(
                f"PC={n_components:>2} | "
                f"k={k:>2} | "
                f"explained={explained_variance:.4f} | "
                f"inertia={inertia:.2f} | "
                f"silhouette={silhouette:.4f}"
            )

    results_df = pd.DataFrame(results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORT_PATH, index=False)

    best = results_df.sort_values("silhouette", ascending=False).iloc[0]

    print()
    print("Best configuration by silhouette:")
    print(f"PCA components: {int(best['pca_components'])}")
    print(f"k: {int(best['k'])}")
    print(f"Explained variance: {best['explained_variance']:.4f}")
    print(f"Silhouette: {best['silhouette']:.4f}")

    plt.figure(figsize=(9, 5))

    for n_components in PCA_DIMS:
        subset = results_df[results_df["pca_components"] == n_components]
        plt.plot(
            subset["k"],
            subset["silhouette"],
            marker="o",
            label=f"{n_components} PCs"
        )

    plt.xlabel("Number of clusters k")
    plt.ylabel("Silhouette score")
    plt.title("K-Means silhouette score for different PCA dimensions")
    plt.xticks(list(K_RANGE))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(SILHOUETTE_PLOT_PATH, dpi=150)
    plt.close()

    print()
    print("Saved files:")
    print(f"Report: {REPORT_PATH}")
    print(f"Silhouette plot: {SILHOUETTE_PLOT_PATH}")
"""



