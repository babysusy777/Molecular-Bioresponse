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

OUTPUT_DIR = PROJECT_DIR / "reports" / "jordan_weiss_k2_k3"
LABELS_DIR = PROJECT_DIR / "Dataset" / "jordan_weiss_labels"

GAMMA = 0.001
TARGET_K = [2, 3]
TOP_FEATURES_HEATMAP = 30
TOP_FEATURES_PER_CLUSTER = 15


def row_normalize(U):
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return U / norms


def main():
    X = pd.read_csv(INPUT_PATH)

    print(f"Loaded dataset: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"Using gamma={GAMMA}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # Gaussian affinity matrix
    W = rbf_kernel(X, gamma=GAMMA)

    # Compute first 3 eigenvectors.
    # For K=2 we use first 2, for K=3 we use first 3.
    U_full = spectral_embedding(
        W,
        n_components=3,
        random_state=42,
        drop_first=False
    )

    summary = []

    for k in TARGET_K:
        print()
        print(f"Running Jordan-Weiss with K={k}")

        U = U_full[:, :k]
        Y = row_normalize(U)

        kmeans = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=42
        )

        labels = kmeans.fit_predict(Y)

        sil_original = silhouette_score(X, labels)
        sil_spectral = silhouette_score(Y, labels)

        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        cluster_fractions = cluster_sizes / len(labels)

        print(f"K={k}")
        print(f"Silhouette original space: {sil_original:.4f}")
        print(f"Silhouette spectral space: {sil_spectral:.4f}")

        print()
        print("Cluster sizes:")
        print(cluster_sizes)

        print()
        print("Cluster fractions:")
        print(cluster_fractions)

        summary.append({
            "k": k,
            "gamma": GAMMA,
            "silhouette_original": sil_original,
            "silhouette_spectral": sil_spectral,
            "cluster_sizes": cluster_sizes.to_dict(),
            "cluster_fractions": cluster_fractions.to_dict()
        })

        # Save labels
        labels_path = LABELS_DIR / f"jordan_weiss_k{k}_gamma_{GAMMA}.csv"
        pd.DataFrame({"cluster": labels}).to_csv(labels_path, index=False)

        # -------------------------------------------------
        # 1. Spectral space plot
        # -------------------------------------------------

        if k == 2:
            plt.figure(figsize=(7, 6))
            plt.scatter(Y[:, 0], Y[:, 1], c=labels, s=10)
            plt.xlabel("Eigenvector 1")
            plt.ylabel("Eigenvector 2")
            plt.title(f"Jordan-Weiss spectral space | K={k} | gamma={GAMMA}")
            plt.tight_layout()

            spectral_plot_path = OUTPUT_DIR / f"spectral_space_k{k}.png"
            plt.savefig(spectral_plot_path, dpi=150)
            plt.close()

        else:
            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(111, projection="3d")

            ax.scatter(
                Y[:, 0],
                Y[:, 1],
                Y[:, 2],
                c=labels,
                s=10
            )

            ax.set_xlabel("Eigenvector 1")
            ax.set_ylabel("Eigenvector 2")
            ax.set_zlabel("Eigenvector 3")
            ax.set_title(f"Jordan-Weiss spectral space | K={k} | gamma={GAMMA}")

            plt.tight_layout()

            spectral_plot_path = OUTPUT_DIR / f"spectral_space_k{k}.png"
            plt.savefig(spectral_plot_path, dpi=150)
            plt.close()

        # -------------------------------------------------
        # 2. Original space analysis
        # -------------------------------------------------

        X_labeled = X.copy()
        X_labeled["cluster"] = labels

        cluster_means = X_labeled.groupby("cluster").mean()
        global_mean = X.mean()

        feature_scores = cluster_means.var(axis=0).sort_values(ascending=False)
        top_3_features = feature_scores.head(3).index.tolist()

        print()
        print("Top 3 original features for visualization:")
        print(top_3_features)

        # -------------------------------------------------
        # 2a. Original space plot using top 3 original features
        # -------------------------------------------------

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            X[top_3_features[0]],
            X[top_3_features[1]],
            X[top_3_features[2]],
            c=labels,
            s=10
        )

        ax.set_xlabel(top_3_features[0])
        ax.set_ylabel(top_3_features[1])
        ax.set_zlabel(top_3_features[2])
        ax.set_title(f"Original feature space | K={k}")

        plt.tight_layout()

        original_plot_path = OUTPUT_DIR / f"original_top_features_k{k}.png"
        plt.savefig(original_plot_path, dpi=150)
        plt.close()

        # -------------------------------------------------
        # 2b. Cluster activation summary
        # -------------------------------------------------

        cluster_activation = pd.DataFrame({
            "cluster": labels,
            "row_sum": X.sum(axis=1),
            "nonzero_features": (X > 0).sum(axis=1),
            "mean_feature_value": X.mean(axis=1)
        })

        activation_mean = cluster_activation.groupby("cluster").mean()
        activation_median = cluster_activation.groupby("cluster").median()

        activation_report = activation_mean.add_suffix("_mean").join(
            activation_median.add_suffix("_median")
        )

        activation_report["cluster_size"] = cluster_sizes
        activation_report["cluster_fraction"] = cluster_fractions

        activation_report_path = OUTPUT_DIR / f"cluster_activation_summary_k{k}.csv"
        activation_report.to_csv(activation_report_path)

        print()
        print("Cluster activation summary:")
        print(activation_report)

        # -------------------------------------------------
        # 2c. Top positive / negative features per cluster
        # -------------------------------------------------

        feature_diff_rows = []

        for cluster_id in cluster_means.index:
            diff = cluster_means.loc[cluster_id] - global_mean

            top_positive = diff.sort_values(ascending=False).head(TOP_FEATURES_PER_CLUSTER)
            top_negative = diff.sort_values(ascending=True).head(TOP_FEATURES_PER_CLUSTER)

            print()
            print(f"Cluster {cluster_id} - top positive features:")
            print(top_positive)

            print()
            print(f"Cluster {cluster_id} - top negative features:")
            print(top_negative)

            for rank, (feature, value) in enumerate(top_positive.items(), start=1):
                feature_diff_rows.append({
                    "cluster": cluster_id,
                    "direction": "positive",
                    "rank": rank,
                    "feature": feature,
                    "difference_from_global_mean": value,
                    "cluster_mean": cluster_means.loc[cluster_id, feature],
                    "global_mean": global_mean[feature]
                })

            for rank, (feature, value) in enumerate(top_negative.items(), start=1):
                feature_diff_rows.append({
                    "cluster": cluster_id,
                    "direction": "negative",
                    "rank": rank,
                    "feature": feature,
                    "difference_from_global_mean": value,
                    "cluster_mean": cluster_means.loc[cluster_id, feature],
                    "global_mean": global_mean[feature]
                })

        feature_diff_report = pd.DataFrame(feature_diff_rows)
        feature_diff_report_path = OUTPUT_DIR / f"cluster_top_features_k{k}.csv"
        feature_diff_report.to_csv(feature_diff_report_path, index=False)

        # -------------------------------------------------
        # 3. Heatmap of cluster means
        # -------------------------------------------------

        top_features = feature_scores.head(TOP_FEATURES_HEATMAP).index.tolist()
        heatmap_data = cluster_means[top_features].to_numpy()

        plt.figure(figsize=(14, 5))
        plt.imshow(heatmap_data, aspect="auto")
        plt.colorbar(label="Mean feature value")

        plt.yticks(
            ticks=range(cluster_means.shape[0]),
            labels=[f"Cluster {i}" for i in cluster_means.index]
        )

        plt.xticks(
            ticks=range(len(top_features)),
            labels=top_features,
            rotation=90
        )

        plt.xlabel("Original features")
        plt.ylabel("Cluster")
        plt.title(f"Cluster mean profiles | K={k}")

        plt.tight_layout()

        heatmap_path = OUTPUT_DIR / f"cluster_means_heatmap_k{k}.png"
        plt.savefig(heatmap_path, dpi=150)
        plt.close()

        print()
        print(f"Saved labels: {labels_path}")
        print(f"Saved spectral plot: {spectral_plot_path}")
        print(f"Saved original feature plot: {original_plot_path}")
        print(f"Saved heatmap: {heatmap_path}")
        print(f"Saved activation summary: {activation_report_path}")
        print(f"Saved top feature report: {feature_diff_report_path}")

    summary_path = OUTPUT_DIR / "jordan_weiss_k2_k3_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)

    print()
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()