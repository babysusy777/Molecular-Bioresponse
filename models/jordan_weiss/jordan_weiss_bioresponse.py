from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import spectral_embedding
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

INPUT_PATH = PROJECT_DIR / "Dataset" / "train_filtered_no_activity.csv"
TARGET_PATH = PROJECT_DIR / "Dataset" / "train_activity_target.csv"

REPORT_DIR = PROJECT_DIR / "reports" / "jordan_weiss_activity_distribution"

GAMMA = 0.001
K_VALUES = [2, 3]


def row_normalize(U):
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return U / norms


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(INPUT_PATH)
    y = pd.read_csv(TARGET_PATH).iloc[:, 0]

    print(f"Loaded dataset: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"Loaded target: {y.shape[0]} labels")
    print(f"Using gamma={GAMMA}")
    print()

    # Gaussian similarity matrix
    W = rbf_kernel(X, gamma=GAMMA)

    # Compute eigenvectors once
    U_full = spectral_embedding(
        W,
        n_components=max(K_VALUES),
        random_state=42,
        drop_first=False
    )

    for k in K_VALUES:
        U = U_full[:, :k]
        Y = row_normalize(U)

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = kmeans.fit_predict(Y)

        sil_original = silhouette_score(X, labels)
        sil_spectral = silhouette_score(Y, labels)

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

        counts.to_csv(REPORT_DIR / f"jordan_weiss_k{k}_activity_counts.csv")
        percentages.to_csv(REPORT_DIR / f"jordan_weiss_k{k}_activity_percentages.csv")

        print(f"Jordan-Weiss with k={k}")
        print("-" * 60)
        print(f"Silhouette original space: {sil_original:.4f}")
        print(f"Silhouette spectral space: {sil_spectral:.4f}")
        print()
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

        plt.title(f"Activity distribution inside Jordan-Weiss clusters - k={k}")
        plt.xlabel("Cluster")
        plt.ylabel("Percentage")
        plt.legend(title="Activity")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / f"jordan_weiss_k{k}_activity_distribution.png", dpi=150)
        plt.close()

    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()