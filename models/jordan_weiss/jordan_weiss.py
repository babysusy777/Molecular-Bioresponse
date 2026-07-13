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

REPORT_PATH = PROJECT_DIR / "reports" / "jordan_weiss_rbf_results.csv"
BEST_VALID_REPORT_PATH = PROJECT_DIR / "reports" / "jordan_weiss_rbf_best_valid.csv"
LABELS_OUTPUT_PATH = PROJECT_DIR / "Dataset" / "jordan_weiss_rbf_best_labels.csv"
EIGENVECTOR_PLOT_PATH = PROJECT_DIR / "reports" / "jordan_weiss_rbf_eigenvectors_best.png"

K_RANGE = range(2, 8)
GAMMA_LIST = [0.0001, 0.0005, 0.001, 0.002, 0.005]

MIN_CLUSTER_FRACTION = 0.05
SILHOUETTE_SAMPLE_SIZE = None


def row_normalize(U):
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return U / norms


def is_valid_cluster_size(cluster_sizes, min_cluster_size):
    return min(cluster_sizes.values()) >= min_cluster_size


def main():
    X = pd.read_csv(INPUT_PATH)

    n_samples = X.shape[0]
    min_cluster_size = int(MIN_CLUSTER_FRACTION * n_samples)

    print(f"Loaded dataset: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"Minimum allowed cluster size: {min_cluster_size}")

    results = []

    max_k = max(K_RANGE)

    for gamma in GAMMA_LIST:
        print()
        print(f"Testing gamma={gamma}")

        # 1. Gaussian affinity matrix W
        W = rbf_kernel(X, gamma=gamma)

        # 2. Spectral embedding computed once for max_k
        U_full = spectral_embedding(
            W,
            n_components=max_k,
            random_state=42,
            drop_first=False
        )

        for k in K_RANGE:
            # 3. Use first k eigenvectors
            U = U_full[:, :k]

            # 4. Ng-Jordan-Weiss row normalization
            Y = row_normalize(U)

            # 5. K-Means in spectral space
            kmeans = KMeans(
                n_clusters=k,
                n_init=20,
                random_state=42
            )

            labels = kmeans.fit_predict(Y)

            # 6. Silhouette in original space
            silhouette_original = silhouette_score(
                X,
                labels,
                sample_size=SILHOUETTE_SAMPLE_SIZE,
                random_state=42
            )

            # 7. Silhouette in spectral space
            silhouette_spectral = silhouette_score(
                Y,
                labels,
                sample_size=SILHOUETTE_SAMPLE_SIZE,
                random_state=42
            )

            cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()
            valid = is_valid_cluster_size(cluster_sizes, min_cluster_size)

            results.append({
                "k": k,
                "gamma": gamma,
                "silhouette_original": silhouette_original,
                "silhouette_spectral": silhouette_spectral,
                "valid": valid,
                "cluster_sizes": cluster_sizes
            })

            print(
                f"k={k:>2} | "
                f"gamma={gamma:<7} | "
                f"sil_original={silhouette_original:.4f} | "
                f"sil_spectral={silhouette_spectral:.4f} | "
                f"valid={valid} | "
                f"sizes={cluster_sizes}"
            )

    results_df = pd.DataFrame(results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(REPORT_PATH, index=False)

    valid_results = results_df[results_df["valid"] == True]

    if valid_results.empty:
        print()
        print("No valid configurations found. Try reducing MIN_CLUSTER_FRACTION.")
        return

    best_valid = valid_results.sort_values(
        "silhouette_original",
        ascending=False
    ).head(10)

    best_valid.to_csv(BEST_VALID_REPORT_PATH, index=False)

    print()
    print("Best valid configurations:")
    print(best_valid)

    # Best valid configuration
    best = best_valid.iloc[0]
    best_k = int(best["k"])
    best_gamma = float(best["gamma"])

    print()
    print("Selected configuration:")
    print(f"k={best_k}")
    print(f"gamma={best_gamma}")
    print(f"silhouette_original={best['silhouette_original']:.4f}")
    print(f"silhouette_spectral={best['silhouette_spectral']:.4f}")
    print(f"cluster_sizes={best['cluster_sizes']}")

    # Recompute labels for best configuration
    W = rbf_kernel(X, gamma=best_gamma)

    U_full = spectral_embedding(
        W,
        n_components=max(max_k, 3),
        random_state=42,
        drop_first=False
    )

    U_best = U_full[:, :best_k]
    Y_best = row_normalize(U_best)

    kmeans = KMeans(
        n_clusters=best_k,
        n_init=20,
        random_state=42
    )

    best_labels = kmeans.fit_predict(Y_best)

    pd.DataFrame({
        "cluster": best_labels
    }).to_csv(LABELS_OUTPUT_PATH, index=False)

    # Plot on eigenvectors, not PCA
    Y_plot = row_normalize(U_full[:, :3])

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        Y_plot[:, 0],
        Y_plot[:, 1],
        Y_plot[:, 2],
        c=best_labels,
        s=10
    )

    ax.set_xlabel("Eigenvector 1")
    ax.set_ylabel("Eigenvector 2")
    ax.set_zlabel("Eigenvector 3")

    ax.set_title(
        f"Jordan-Weiss RBF | k={best_k} | gamma={best_gamma}"
    )

    plt.tight_layout()
    plt.savefig(EIGENVECTOR_PLOT_PATH, dpi=150)
    plt.close()

    print()
    print("Saved files:")
    print(f"Full report: {REPORT_PATH}")
    print(f"Best valid report: {BEST_VALID_REPORT_PATH}")
    print(f"Best labels: {LABELS_OUTPUT_PATH}")
    print(f"Eigenvector plot: {EIGENVECTOR_PLOT_PATH}")


if __name__ == "__main__":
    main()