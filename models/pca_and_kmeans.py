from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

INPUT_PATH = PROJECT_DIR / "Dataset" / "train_filtered_no_activity.csv"

PCA_OUTPUT_PATH = PROJECT_DIR / "Dataset" / "train_pca_90_no_activity.csv"

PCA_REPORT_PATH = PROJECT_DIR / "reports" / "pca_explained_variance.csv"
PCA_PLOT_PATH = PROJECT_DIR / "reports" / "pca_scree_plot_90.png"

ELBOW_REPORT_PATH = PROJECT_DIR / "reports" / "kmeans_elbow_values.csv"
ELBOW_PLOT_PATH = PROJECT_DIR / "reports" / "kmeans_elbow_plot.png"

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
        "inertia": inertias
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

    print()
    print("Saved files:")
    print(f"PCA dataset: {PCA_OUTPUT_PATH}")
    print(f"PCA report: {PCA_REPORT_PATH}")
    print(f"PCA plot: {PCA_PLOT_PATH}")
    print(f"Elbow report: {ELBOW_REPORT_PATH}")
    print(f"Elbow plot: {ELBOW_PLOT_PATH}")


if __name__ == "__main__":
    main()