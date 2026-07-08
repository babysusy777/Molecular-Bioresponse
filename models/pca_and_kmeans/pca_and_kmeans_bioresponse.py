from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

INPUT_PATH = PROJECT_DIR / "Dataset" / "train_filtered_no_activity.csv"
TARGET_PATH = PROJECT_DIR / "Dataset" / "train_activity_target.csv"

REPORT_DIR = PROJECT_DIR / "reports" / "cluster_target_distribution"

VARIANCE_TO_KEEP = 0.90
K_VALUES = [2, 4]


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(INPUT_PATH)
    y = pd.read_csv(TARGET_PATH).iloc[:, 0]

    pca = PCA(n_components=VARIANCE_TO_KEEP)
    X_pca = pca.fit_transform(X)

    print(f"PCA components: {X_pca.shape[1]}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print()

    for k in K_VALUES:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = kmeans.fit_predict(X_pca)

        result = pd.DataFrame({
            "cluster": labels,
            "Activity": y
        })

        counts = pd.crosstab(
            result["cluster"],
            result["Activity"]
        )

        percentages = pd.crosstab(
            result["cluster"],
            result["Activity"],
            normalize="index"
        ) * 100

        counts.to_csv(REPORT_DIR / f"kmeans_k{k}_activity_counts.csv")
        percentages.to_csv(REPORT_DIR / f"kmeans_k{k}_activity_percentages.csv")

        print(f"K-Means with k={k}")
        print("-" * 60)
        print("Counts:")
        print(counts)
        print()
        print("Percentages by cluster:")
        print(percentages.round(2))
        print()

        percentages.plot(
            kind="bar",
            stacked=True,
            figsize=(7, 5)
        )

        plt.title(f"Activity distribution inside K-Means clusters - k={k}")
        plt.xlabel("Cluster")
        plt.ylabel("Percentage")
        plt.legend(title="Activity")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / f"kmeans_k{k}_activity_distribution.png", dpi=150)
        plt.close()

    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()