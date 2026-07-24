#!/usr/bin/env python3
"""
Analisi robusta e interamente non supervisionata delle feature numeriche.

Pipeline
--------
1. Statistiche descrittive di distribuzione, sparsità e variabilità.
2. Selezione di k con 30 sottocampionamenti casuali 80/20:
   PCA e K-means sono ristimati in ogni ripetizione sui valori originali.
3. Valutazione mediante silhouette sui campioni esclusi e stabilità ARI.
4. Rifit sull'intero dataset con più seed.
5. Matrici riordinate per cluster delle righe e varianza delle colonne.
6. Sensibilità alla soglia PCA e allo scaling.

La pipeline principale non divide per la deviazione standard: PCA effettua
soltanto la centratura automatica. StandardScaler e RobustScaler compaiono
esclusivamente nell'analisi di sensibilità.

Non sono inclusi clustering delle feature, Activity o modello nullo.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler


PROJECT_DIR = Path(
    "/Users/susannabaldo/Desktop/Machine_Learning_Project/"
    "Molecular-Bioresponse/Progetto_Finale"
)


@dataclass(frozen=True)
class Config:
    input_path: Path = PROJECT_DIR / "000_Dataset" / "train_numeric_only.csv"
    report_dir: Path = (
        PROJECT_DIR
        / "003_Models"
        / "0033_Mixed"
        / "numeric_feature_exploration"
        / "reports"
        / "raw_pca_kmeans_pipeline"
    )

    pca_variance: float = 0.90
    k_values: tuple[int, ...] = tuple(range(2, 11))
    kmeans_n_init: int = 20

    n_stability_repeats: int = 30
    subsample_fraction: float = 0.80
    stability_seed: int = 42
    final_refit_seeds: tuple[int, ...] = (
        11,
        29,
        42,
        71,
        101,
        137,
        173,
        211,
        251,
        307,
    )

    pca_sensitivity_values: tuple[float, ...] = (0.80, 0.90, 0.95)
    top_n_features: int = 25
    centered_matrix_clip_quantile: float = 0.995
    dpi: int = 180


@dataclass
class Projection:
    pca: PCA
    scores: np.ndarray


def make_output_dirs(config: Config) -> dict[str, Path]:
    root = config.report_dir
    dirs = {
        "root": root,
        "distributions": root / "01_distributions",
        "stability": root / "02_stability_selection",
        "pca": root / "03_final_pca_kmeans",
        "matrix": root / "04_reordered_matrix",
        "diagnostics": root / "05_sensitivity_diagnostics",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def save_plot(path: Path, dpi: int) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    n_labels = np.unique(labels).size
    if n_labels < 2 or n_labels >= len(labels):
        return np.nan
    return float(silhouette_score(X, labels))


def load_data(config: Config) -> pd.DataFrame:
    if not config.input_path.exists():
        raise FileNotFoundError(
            f"Dataset numerico non trovato: {config.input_path}"
        )

    X = pd.read_csv(config.input_path)
    if "Activity" in X.columns:
        warnings.warn(
            "La colonna Activity è stata rimossa: questa analisi è "
            "interamente non supervisionata."
        )
        X = X.drop(columns="Activity")

    non_numeric = [
        column
        for column in X.columns
        if not pd.api.types.is_numeric_dtype(X[column])
    ]
    if non_numeric:
        raise ValueError(f"Colonne non numeriche: {non_numeric[:10]}")

    values = X.to_numpy(dtype=float)
    if X.isna().any().any() or not np.isfinite(values).all():
        raise ValueError("Il dataset contiene NaN o valori infiniti.")

    constant = X.columns[X.nunique(dropna=False) <= 1].tolist()
    if constant:
        warnings.warn(f"Rimosse {len(constant)} feature costanti.")
        X = X.drop(columns=constant)

    X = X.reset_index(drop=True)
    if len(X) < 10 or X.shape[1] < 3:
        raise ValueError("Servono almeno 10 campioni e 3 feature non costanti.")

    print(f"Loaded numeric dataset: {X.shape}")
    return X


def analyze_distributions(
    X: pd.DataFrame, output_dir: Path, config: Config
) -> pd.DataFrame:
    rows = []
    for feature in X.columns:
        values = X[feature].to_numpy(dtype=float)
        positive = values[values > 0]
        dominant_fraction = pd.Series(values).value_counts(
            normalize=True
        ).iloc[0]
        rows.append(
            {
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
                "iqr": float(
                    np.quantile(values, 0.75)
                    - np.quantile(values, 0.25)
                ),
                "skewness": float(skew(values, bias=False)),
                "kurtosis_fisher": float(
                    kurtosis(values, fisher=True, bias=False)
                ),
                "zero_fraction": float(np.mean(values == 0)),
                "positive_fraction": float(np.mean(values > 0)),
                "unique_values": int(np.unique(values).size),
                "dominant_value_fraction": float(dominant_fraction),
                "positive_mean": (
                    float(positive.mean()) if positive.size else np.nan
                ),
                "positive_median": (
                    float(np.median(positive)) if positive.size else np.nan
                ),
            }
        )

    report = pd.DataFrame(rows)
    report.to_csv(
        output_dir / "numeric_distribution_summary.csv", index=False
    )

    plt.figure(figsize=(9, 5))
    plt.hist(
        report["zero_fraction"],
        bins=30,
        edgecolor="black",
        linewidth=0.4,
    )
    plt.xlabel("Zero fraction")
    plt.ylabel("Number of features")
    plt.title("Distribution of zero fractions")
    save_plot(output_dir / "zero_fraction_distribution.png", config.dpi)

    finite_skew = report["skewness"].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if not finite_skew.empty:
        clipped = finite_skew.clip(
            finite_skew.quantile(0.01), finite_skew.quantile(0.99)
        )
        plt.figure(figsize=(9, 5))
        plt.hist(clipped, bins=30, edgecolor="black", linewidth=0.4)
        plt.xlabel("Skewness, clipped at 1st–99th percentiles")
        plt.ylabel("Number of features")
        plt.title("Distribution of feature skewness")
        save_plot(output_dir / "skewness_distribution.png", config.dpi)

    plots = (
        (
            "variance",
            "features by variance",
            "top_features_by_variance.png",
        ),
        (
            "zero_fraction",
            "features by zero fraction",
            "top_features_by_zero_fraction.png",
        ),
    )
    for column, title, filename in plots:
        source = (
            report
            if column == "variance"
            else report[report["zero_fraction"] < 1.0]
        )
        top = source.nlargest(
            config.top_n_features, column
        ).sort_values(column)
        plt.figure(figsize=(10, 8))
        plt.barh(top["feature"], top[column])
        plt.xlabel(column.replace("_", " ").title())
        plt.ylabel("Feature")
        plt.title(f"Top {config.top_n_features} {title}")
        save_plot(output_dir / filename, config.dpi)

    plt.figure(figsize=(8, 6))
    plt.scatter(report["zero_fraction"], report["std"], s=18, alpha=0.65)
    plt.xlabel("Zero fraction")
    plt.ylabel("Standard deviation")
    plt.title("Feature sparsity versus variability")
    save_plot(output_dir / "zero_fraction_vs_std.png", config.dpi)

    print(f"Distribution analysis completed: {len(report)} features.")
    return report


def fit_projection(
    X_fit: np.ndarray,
    X_transform: np.ndarray,
    variance_threshold: float,
) -> Projection:
    """Stima la PCA sui dati originali e trasforma X_transform.

    PCA centra internamente ogni feature usando la media di X_fit, ma non
    divide per la deviazione standard.
    """
    pca = PCA(n_components=variance_threshold, svd_solver="full")
    pca.fit(X_fit)
    return Projection(pca=pca, scores=pca.transform(X_transform))


def pairwise_ari(labels_list: list[np.ndarray]) -> np.ndarray:
    return np.asarray(
        [
            adjusted_rand_score(first, second)
            for first, second in combinations(labels_list, 2)
        ],
        dtype=float,
    )


def plot_interval(
    summary: pd.DataFrame,
    mean_column: str,
    low_column: str,
    high_column: str,
    ylabel: str,
    title: str,
    path: Path,
    dpi: int,
    selected_k: int,
) -> None:
    x = summary["k"].to_numpy(dtype=int)
    center = summary[mean_column].to_numpy(dtype=float)
    low = summary[low_column].to_numpy(dtype=float)
    high = summary[high_column].to_numpy(dtype=float)
    plt.figure(figsize=(9, 5))
    plt.plot(x, center, marker="o")
    plt.fill_between(x, low, high, alpha=0.2)
    plt.axvline(selected_k, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Number of clusters k")
    plt.ylabel(ylabel)
    plt.title(title)
    save_plot(path, dpi)


def summarize_stability(
    repetitions: pd.DataFrame,
    labels_by_k: dict[int, list[np.ndarray]],
    config: Config,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    rows = []
    ari_by_k = {}

    for k in config.k_values:
        current = repetitions[repetitions["k"] == k]
        labels = labels_by_k[k]
        if current.empty or len(labels) < 2:
            continue

        silhouettes = current["validation_silhouette"].dropna()
        if silhouettes.empty:
            continue

        aris = pairwise_ari(labels)
        ari_by_k[k] = aris
        rows.append(
            {
                "k": k,
                "n_successful_repeats": len(current),
                "silhouette_mean": float(silhouettes.mean()),
                "silhouette_std": float(silhouettes.std(ddof=1)),
                "silhouette_se": float(
                    silhouettes.std(ddof=1) / np.sqrt(len(silhouettes))
                ),
                "silhouette_median": float(silhouettes.median()),
                "silhouette_q10": float(silhouettes.quantile(0.10)),
                "silhouette_q90": float(silhouettes.quantile(0.90)),
                "calinski_harabasz_median": float(
                    current["validation_calinski_harabasz"].median()
                ),
                "davies_bouldin_median": float(
                    current["validation_davies_bouldin"].median()
                ),
                "mean_pairwise_ari": float(aris.mean()),
                "median_pairwise_ari": float(np.median(aris)),
                "q10_pairwise_ari": float(np.quantile(aris, 0.10)),
                "q90_pairwise_ari": float(np.quantile(aris, 0.90)),
                "minimum_cluster_fraction_median": float(
                    current["minimum_cluster_fraction"].median()
                ),
                "minimum_cluster_fraction_q10": float(
                    current["minimum_cluster_fraction"].quantile(0.10)
                ),
                "n_components_median": float(
                    current["n_components"].median()
                ),
            }
        )

    summary = pd.DataFrame(rows).sort_values("k")
    if summary.empty:
        raise RuntimeError("La selezione di k non ha prodotto metriche valide.")
    return summary, ari_by_k


def run_stability_selection(
    X: pd.DataFrame, output_dir: Path, config: Config
) -> dict:
    """
    Seleziona k mediante repeated holdout 80/20 e stabilità ARI.

    In ogni ripetizione PCA e K-means sono stimati sull'80% dei valori
    originali; il silhouette è calcolato sul 20% escluso. L'ARI confronta
    invece le assegnazioni predette su tutti i campioni.
    """
    values = X.to_numpy(dtype=float)
    n_samples = len(values)
    train_size = int(round(config.subsample_fraction * n_samples))
    if not 2 <= train_size <= n_samples - 2:
        raise ValueError("subsample_fraction non lascia una validation valida.")

    sequences = np.random.SeedSequence(config.stability_seed).spawn(
        config.n_stability_repeats
    )
    rows = []
    labels_by_k = {k: [] for k in config.k_values}

    for repeat, sequence in enumerate(sequences):
        rng = np.random.default_rng(sequence)
        train_indices = np.sort(
            rng.choice(n_samples, size=train_size, replace=False)
        )
        validation_mask = np.ones(n_samples, dtype=bool)
        validation_mask[train_indices] = False
        validation_indices = np.flatnonzero(validation_mask)

        projection = fit_projection(
            values[train_indices], values, config.pca_variance
        )
        scores = projection.scores
        train_scores = scores[train_indices]
        validation_scores = scores[validation_indices]

        for k in config.k_values:
            if k >= len(train_indices) or k >= len(validation_indices):
                continue

            model_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            model = KMeans(
                n_clusters=k,
                n_init=config.kmeans_n_init,
                random_state=model_seed,
            ).fit(train_scores)
            labels_all = model.predict(scores)
            labels_validation = labels_all[validation_indices]
            labels_by_k[k].append(labels_all)

            unique_validation = np.unique(labels_validation)
            valid_partition = (
                2 <= len(unique_validation) < len(validation_indices)
            )
            silhouette = safe_silhouette(
                validation_scores, labels_validation
            )
            calinski = (
                float(
                    calinski_harabasz_score(
                        validation_scores, labels_validation
                    )
                )
                if valid_partition
                else np.nan
            )
            davies = (
                float(
                    davies_bouldin_score(
                        validation_scores, labels_validation
                    )
                )
                if valid_partition
                else np.nan
            )
            cluster_sizes = np.bincount(labels_all, minlength=k)

            rows.append(
                {
                    "repeat": repeat,
                    "split_seed": int(sequence.generate_state(1)[0]),
                    "kmeans_seed": model_seed,
                    "k": k,
                    "n_train": len(train_indices),
                    "n_validation": len(validation_indices),
                    "n_components": scores.shape[1],
                    "retained_variance": float(
                        projection.pca.explained_variance_ratio_.sum()
                    ),
                    "validation_silhouette": silhouette,
                    "validation_calinski_harabasz": calinski,
                    "validation_davies_bouldin": davies,
                    "train_inertia": float(model.inertia_),
                    "minimum_cluster_fraction": float(
                        cluster_sizes.min() / n_samples
                    ),
                    "maximum_cluster_fraction": float(
                        cluster_sizes.max() / n_samples
                    ),
                    "empty_clusters_in_validation": int(
                        k - len(unique_validation)
                    ),
                }
            )

        print(
            f"Stability repeat {repeat + 1}/"
            f"{config.n_stability_repeats} completed."
        )

    repetitions = pd.DataFrame(rows)
    repetitions.to_csv(
        output_dir / "kmeans_stability_repetitions.csv", index=False
    )
    summary, ari_by_k = summarize_stability(
        repetitions, labels_by_k, config
    )

    best = summary.loc[summary["silhouette_mean"].idxmax()]
    one_se_cutoff = float(
        best["silhouette_mean"] - best["silhouette_se"]
    )
    eligible = summary[summary["silhouette_mean"] >= one_se_cutoff]
    selected = eligible.sort_values(
        ["mean_pairwise_ari", "k"], ascending=[False, True]
    ).iloc[0]
    selected_k = int(selected["k"])

    summary["within_one_se"] = summary["silhouette_mean"] >= one_se_cutoff
    summary["selected"] = summary["k"] == selected_k
    summary.to_csv(
        output_dir / "kmeans_stability_summary.csv", index=False
    )

    pd.concat(
        [
            pd.DataFrame({"k": k, "pairwise_ari": values})
            for k, values in ari_by_k.items()
        ],
        ignore_index=True,
    ).to_csv(output_dir / "kmeans_pairwise_ari.csv", index=False)

    selection = {
        "selected_k": selected_k,
        "best_mean_silhouette_k": int(best["k"]),
        "best_mean_silhouette": float(best["silhouette_mean"]),
        "one_standard_error_cutoff": one_se_cutoff,
        "selected_mean_silhouette": float(selected["silhouette_mean"]),
        "selected_mean_pairwise_ari": float(
            selected["mean_pairwise_ari"]
        ),
        "rule": (
            "Among k values within one standard error of the best mean "
            "out-of-sample silhouette, select the highest mean pairwise ARI; "
            "break exact ties in favor of smaller k."
        ),
    }
    (output_dir / "k_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )

    plot_interval(
        summary,
        "silhouette_mean",
        "silhouette_q10",
        "silhouette_q90",
        "Out-of-sample silhouette",
        "Repeated subsampling: validation silhouette",
        output_dir / "validation_silhouette_by_k.png",
        config.dpi,
        selected_k,
    )
    plot_interval(
        summary,
        "mean_pairwise_ari",
        "q10_pairwise_ari",
        "q90_pairwise_ari",
        "Pairwise ARI",
        "Stability of full-data predictions across subsamples",
        output_dir / "partition_stability_by_k.png",
        config.dpi,
        selected_k,
    )

    plt.figure(figsize=(9, 5))
    plt.plot(
        summary["k"],
        summary["minimum_cluster_fraction_median"],
        marker="o",
        label="Median",
    )
    plt.plot(
        summary["k"],
        summary["minimum_cluster_fraction_q10"],
        marker="o",
        label="10th percentile",
    )
    plt.axvline(selected_k, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Number of clusters k")
    plt.ylabel("Smallest cluster fraction")
    plt.title("Minimum cluster size across stability repetitions")
    plt.legend()
    save_plot(
        output_dir / "minimum_cluster_fraction_by_k.png", config.dpi
    )

    print(f"Selected k={selected_k}.")
    return {
        "repetitions": repetitions,
        "summary": summary,
        "labels_by_k": labels_by_k,
        "selected_k": selected_k,
        "selection": selection,
    }


def relabel_by_pc1(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, dict[int, int]]:
    ordered = sorted(
        np.unique(labels),
        key=lambda label: float(scores[labels == label, 0].mean()),
    )
    mapping = {int(old): new for new, old in enumerate(ordered)}
    relabeled = np.asarray(
        [mapping[int(label)] for label in labels], dtype=int
    )
    return relabeled, mapping


def save_pca_outputs(
    X: pd.DataFrame,
    projection: Projection,
    labels: np.ndarray,
    output_dir: Path,
    config: Config,
) -> None:
    explained = projection.pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    component_names = [
        f"PC{i}" for i in range(1, projection.scores.shape[1] + 1)
    ]

    pd.DataFrame(
        {
            "component": np.arange(1, len(explained) + 1),
            "explained_variance_ratio": explained,
            "cumulative_explained_variance": cumulative,
        }
    ).to_csv(output_dir / "pca_explained_variance.csv", index=False)

    score_df = pd.DataFrame(projection.scores, columns=component_names)
    score_df.insert(0, "row_index", np.arange(len(score_df)))
    score_df["kmeans_cluster"] = labels
    score_df.to_csv(output_dir / "pca_scores.csv", index=False)

    pd.DataFrame(
        projection.pca.components_.T,
        index=X.columns,
        columns=component_names,
    ).to_csv(output_dir / "pca_loadings.csv")

    pd.DataFrame(
        {
            "row_index": np.arange(len(X)),
            "kmeans_cluster": labels,
        }
    ).to_csv(output_dir / "kmeans_cluster_assignments.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(
        np.arange(1, len(cumulative) + 1),
        cumulative,
        marker="o",
        markersize=3,
    )
    plt.axhline(config.pca_variance, linestyle="--", linewidth=1.2)
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title(
        f"PCA cumulative variance: {len(explained)} components retain "
        f"{cumulative[-1]:.4f}"
    )
    save_plot(output_dir / "pca_cumulative_variance.png", config.dpi)

    if projection.scores.shape[1] >= 2:
        plt.figure(figsize=(9, 7))
        scatter = plt.scatter(
            projection.scores[:, 0],
            projection.scores[:, 1],
            c=labels,
            s=18,
            alpha=0.70,
            cmap="tab10",
        )
        plt.xlabel(f"PC1 ({explained[0]:.2%})")
        plt.ylabel(f"PC2 ({explained[1]:.2%})")
        plt.title("Final PCA/K-means partition")
        plt.colorbar(scatter, label="K-means cluster")
        save_plot(output_dir / "pca_2d_scatter_kmeans.png", config.dpi)

    if projection.scores.shape[1] >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            projection.scores[:, 0],
            projection.scores[:, 1],
            projection.scores[:, 2],
            c=labels,
            s=18,
            alpha=0.70,
            cmap="tab10",
        )
        ax.set_xlabel(f"PC1 ({explained[0]:.2%})")
        ax.set_ylabel(f"PC2 ({explained[1]:.2%})")
        ax.set_zlabel(f"PC3 ({explained[2]:.2%})")
        ax.set_title("Final PCA/K-means partition")
        ax.view_init(elev=22, azim=45)
        fig.colorbar(
            scatter,
            ax=ax,
            label="K-means cluster",
            shrink=0.7,
            pad=0.1,
        )
        save_plot(output_dir / "pca_3d_scatter_kmeans.png", config.dpi)


def fit_final_model(
    X: pd.DataFrame,
    selected_k: int,
    output_dir: Path,
    config: Config,
) -> dict:
    values = X.to_numpy(dtype=float)
    projection = fit_projection(values, values, config.pca_variance)
    labels_by_seed = []
    models = []
    rows = []

    for seed in config.final_refit_seeds:
        model = KMeans(
            n_clusters=selected_k,
            n_init=config.kmeans_n_init,
            random_state=seed,
        )
        labels = model.fit_predict(projection.scores)
        labels_by_seed.append(labels)
        models.append(model)
        rows.append(
            {
                "seed": seed,
                "inertia": float(model.inertia_),
                "silhouette": safe_silhouette(projection.scores, labels),
                "minimum_cluster_fraction": float(
                    np.bincount(labels, minlength=selected_k).min()
                    / len(labels)
                ),
            }
        )

    n_seeds = len(labels_by_seed)
    ari_matrix = np.eye(n_seeds)
    for i, j in combinations(range(n_seeds), 2):
        ari = adjusted_rand_score(labels_by_seed[i], labels_by_seed[j])
        ari_matrix[i, j] = ari
        ari_matrix[j, i] = ari
    mean_ari = (
        (ari_matrix.sum(axis=1) - 1) / (n_seeds - 1)
        if n_seeds > 1
        else np.ones(1)
    )

    seed_report = pd.DataFrame(rows)
    seed_report["mean_ari_to_other_seeds"] = mean_ari
    representative_index = int(
        seed_report.sort_values(
            ["mean_ari_to_other_seeds", "inertia"],
            ascending=[False, True],
        ).index[0]
    )
    seed_report["representative"] = (
        seed_report.index == representative_index
    )
    seed_report.to_csv(
        output_dir / "final_refit_seed_stability.csv", index=False
    )
    pd.DataFrame(
        ari_matrix,
        index=[f"seed_{seed}" for seed in config.final_refit_seeds],
        columns=[f"seed_{seed}" for seed in config.final_refit_seeds],
    ).to_csv(output_dir / "final_refit_seed_ari_matrix.csv")

    raw_labels = labels_by_seed[representative_index]
    labels, mapping = relabel_by_pc1(raw_labels, projection.scores)
    model = models[representative_index]
    metrics = {
        "selected_k": selected_k,
        "representative_seed": int(
            config.final_refit_seeds[representative_index]
        ),
        "label_mapping_old_to_pc1_order": mapping,
        "n_components": int(projection.scores.shape[1]),
        "retained_variance": float(
            projection.pca.explained_variance_ratio_.sum()
        ),
        "silhouette_full": safe_silhouette(projection.scores, labels),
        "calinski_harabasz_full": float(
            calinski_harabasz_score(projection.scores, labels)
        ),
        "davies_bouldin_full": float(
            davies_bouldin_score(projection.scores, labels)
        ),
        "inertia": float(model.inertia_),
        "mean_seed_ari_of_representative": float(
            mean_ari[representative_index]
        ),
        "cluster_sizes": {
            str(cluster): int(size)
            for cluster, size in enumerate(
                np.bincount(labels, minlength=selected_k)
            )
        },
    }
    (output_dir / "final_model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    save_pca_outputs(X, projection, labels, output_dir, config)

    print(
        f"Final refit completed with k={selected_k}, "
        f"representative seed={metrics['representative_seed']}."
    )
    return {
        "projection": projection,
        "model": model,
        "labels": labels,
        "raw_labels": raw_labels,
        "metrics": metrics,
        "seed_report": seed_report,
    }


def make_reordered_matrix(
    X: pd.DataFrame,
    distributions: pd.DataFrame,
    final_results: dict,
    output_dir: Path,
    config: Config,
) -> None:
    """
    Ordina le righe per cluster e PC1, le colonne per varianza decrescente.

    L'ordinamento delle colonne è descrittivo e non introduce cluster di
    feature.
    """
    projection = final_results["projection"]
    labels = final_results["labels"]
    scores = projection.scores
    values = X.to_numpy(dtype=float)
    centered_values = values - projection.pca.mean_

    row_indices = np.lexsort((scores[:, 0], labels))
    column_report = (
        distributions.set_index("feature")
        .reindex(X.columns)
        .rename_axis("feature")
    )
    column_indices = np.argsort(
        -column_report["variance"].to_numpy(dtype=float), kind="stable"
    )
    ordered_labels = labels[row_indices]
    ordered_features = X.columns[column_indices]

    raw_matrix = values[np.ix_(row_indices, column_indices)]
    centered_matrix = centered_values[
        np.ix_(row_indices, column_indices)
    ]

    pd.DataFrame(
        {
            "reordered_position": np.arange(len(row_indices)),
            "original_row_index": row_indices,
            "kmeans_cluster": ordered_labels,
        }
    ).to_csv(output_dir / "row_order.csv", index=False)

    column_report.iloc[column_indices].reset_index()[
        ["feature", "variance", "zero_fraction"]
    ].assign(
        reordered_position=np.arange(len(column_indices))
    )[
        ["reordered_position", "feature", "variance", "zero_fraction"]
    ].to_csv(output_dir / "column_order.csv", index=False)

    row_boundaries = np.flatnonzero(np.diff(ordered_labels) != 0) + 0.5
    centered_limit = float(
        np.quantile(
            np.abs(centered_matrix),
            config.centered_matrix_clip_quantile,
        )
    )
    matrix_specs = (
        (
            raw_matrix,
            "viridis",
            None,
            None,
            "Raw value",
            "Reordered raw numeric matrix",
            "reordered_raw_matrix.png",
            "white",
        ),
        (
            np.clip(centered_matrix, -centered_limit, centered_limit),
            "coolwarm",
            -centered_limit,
            centered_limit,
            "Mean-centered value",
            (
                "Reordered mean-centered matrix, clipped at the "
                f"{config.centered_matrix_clip_quantile:.1%} "
                "absolute-value quantile"
            ),
            "reordered_centered_matrix.png",
            "black",
        ),
    )

    for matrix, cmap, vmin, vmax, colorbar, title, filename, line_color in (
        matrix_specs
    ):
        plt.figure(figsize=(18, 10))
        image = plt.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        for boundary in row_boundaries:
            plt.axhline(boundary, linewidth=0.7, color=line_color)
        plt.xlabel("Features ordered by decreasing variance")
        plt.ylabel("Samples ordered by K-means cluster and PC1")
        plt.title(title)
        plt.colorbar(image, label=colorbar)
        save_plot(output_dir / filename, config.dpi)

    print("Reordered matrices completed.")


def run_pca_sensitivity(
    X: pd.DataFrame,
    final_results: dict,
    selected_k: int,
    output_dir: Path,
    config: Config,
) -> pd.DataFrame:
    values = X.to_numpy(dtype=float)
    reference_labels = final_results["labels"]
    seed = int(final_results["metrics"]["representative_seed"])
    rows = []

    for threshold in config.pca_sensitivity_values:
        if np.isclose(threshold, config.pca_variance):
            projection = final_results["projection"]
            labels = reference_labels
            inertia = float(final_results["metrics"]["inertia"])
        else:
            projection = fit_projection(values, values, threshold)
            model = KMeans(
                n_clusters=selected_k,
                n_init=config.kmeans_n_init,
                random_state=seed,
            )
            labels = model.fit_predict(projection.scores)
            inertia = float(model.inertia_)

        rows.append(
            {
                "pca_variance_threshold": threshold,
                "n_components": projection.scores.shape[1],
                "retained_variance": float(
                    projection.pca.explained_variance_ratio_.sum()
                ),
                "silhouette": safe_silhouette(
                    projection.scores, labels
                ),
                "ari_vs_primary_partition": float(
                    adjusted_rand_score(reference_labels, labels)
                ),
                "inertia": inertia,
            }
        )

    report = pd.DataFrame(rows)
    report.to_csv(
        output_dir / "pca_threshold_sensitivity.csv", index=False
    )
    plt.figure(figsize=(8, 5))
    plt.plot(
        report["pca_variance_threshold"],
        report["ari_vs_primary_partition"],
        marker="o",
    )
    plt.axhline(0.8, color="black", linestyle="--", linewidth=1)
    plt.xlabel("PCA retained-variance threshold")
    plt.ylabel("ARI versus primary partition")
    plt.title("Sensitivity to the PCA variance threshold")
    save_plot(output_dir / "pca_threshold_sensitivity.png", config.dpi)
    return report


def run_representation_sensitivity(
    X: pd.DataFrame,
    final_results: dict,
    selected_k: int,
    output_dir: Path,
    config: Config,
) -> pd.DataFrame:
    """
    Confronta la partizione primaria grezza con StandardScaler e RobustScaler.

    Non è una seconda selezione: k e seed restano quelli della pipeline
    primaria.
    """
    values = X.to_numpy(dtype=float)
    reference_labels = final_results["labels"]
    seed = int(final_results["metrics"]["representative_seed"])
    rows = []

    for representation in ("raw", "standard", "robust"):
        if representation == "raw":
            scores = final_results["projection"].scores
            labels = reference_labels
            inertia = float(final_results["metrics"]["inertia"])
            retained_variance = float(
                final_results["metrics"]["retained_variance"]
            )
        else:
            scaler = (
                StandardScaler()
                if representation == "standard"
                else RobustScaler()
            )
            transformed = scaler.fit_transform(values)
            pca = PCA(
                n_components=config.pca_variance,
                svd_solver="full",
            )
            scores = pca.fit_transform(transformed)
            model = KMeans(
                n_clusters=selected_k,
                n_init=config.kmeans_n_init,
                random_state=seed,
            )
            labels = model.fit_predict(scores)
            inertia = float(model.inertia_)
            retained_variance = float(
                pca.explained_variance_ratio_.sum()
            )

        rows.append(
            {
                "representation": representation,
                "n_components": scores.shape[1],
                "retained_variance": retained_variance,
                "silhouette": safe_silhouette(scores, labels),
                "ari_vs_raw_partition": float(
                    adjusted_rand_score(reference_labels, labels)
                ),
                "minimum_cluster_fraction": float(
                    np.bincount(labels, minlength=selected_k).min()
                    / len(labels)
                ),
                "inertia": inertia,
            }
        )

    report = pd.DataFrame(rows)
    report.to_csv(
        output_dir / "representation_sensitivity.csv", index=False
    )
    plt.figure(figsize=(8, 5))
    plt.bar(
        report["representation"],
        report["ari_vs_raw_partition"],
    )
    plt.axhline(0.8, color="black", linestyle="--", linewidth=1)
    lower = min(-0.05, float(report["ari_vs_raw_partition"].min()) - 0.05)
    plt.ylim(lower, 1.05)
    plt.xlabel("Numeric representation")
    plt.ylabel("ARI versus raw-data partition")
    plt.title("Sensitivity to feature scaling")
    save_plot(output_dir / "representation_sensitivity.png", config.dpi)
    return report


def write_summary(
    X: pd.DataFrame,
    distributions: pd.DataFrame,
    stability_results: dict,
    final_results: dict,
    pca_sensitivity: pd.DataFrame,
    representation_sensitivity: pd.DataFrame,
    output_path: Path,
) -> None:
    selected_k = stability_results["selected_k"]
    selected = stability_results["summary"].set_index("k").loc[selected_k]
    final = final_results["metrics"]
    cluster_sizes = ", ".join(
        f"{cluster}: {size}"
        for cluster, size in final["cluster_sizes"].items()
    )

    lines = [
        "RAW-DATA PCA/K-MEANS FEATURE EXPLORATION",
        "=" * 42,
        f"Samples: {X.shape[0]}",
        f"Numeric features: {X.shape[1]}",
        "",
        "DISTRIBUTIONS",
        (
            "Median zero fraction: "
            f"{distributions['zero_fraction'].median():.6f}"
        ),
        (
            "Features with zero fraction >= 0.95: "
            f"{(distributions['zero_fraction'] >= 0.95).sum()}"
        ),
        (
            "Features with dominant value fraction >= 0.95: "
            f"{(distributions['dominant_value_fraction'] >= 0.95).sum()}"
        ),
        "",
        "REPEATED OUT-OF-SAMPLE SELECTION",
        f"Selected k: {selected_k}",
        f"Mean validation silhouette: {selected['silhouette_mean']:.6f}",
        (
            "10th-90th percentile validation silhouette: "
            f"[{selected['silhouette_q10']:.6f}, "
            f"{selected['silhouette_q90']:.6f}]"
        ),
        (
            "Mean pairwise ARI across subsamples: "
            f"{selected['mean_pairwise_ari']:.6f}"
        ),
        (
            "10th percentile minimum cluster fraction: "
            f"{selected['minimum_cluster_fraction_q10']:.6f}"
        ),
        "",
        "FINAL REFIT",
        f"Representative seed: {final['representative_seed']}",
        f"Retained PCs: {final['n_components']}",
        f"Retained variance: {final['retained_variance']:.6f}",
        f"Full-data silhouette: {final['silhouette_full']:.6f}",
        (
            "Representative-seed mean ARI: "
            f"{final['mean_seed_ari_of_representative']:.6f}"
        ),
        f"Final cluster sizes: {cluster_sizes}",
        "",
        "SENSITIVITY",
        (
            "Minimum ARI across PCA thresholds: "
            f"{pca_sensitivity['ari_vs_primary_partition'].min():.6f}"
        ),
        (
            "Minimum ARI across numeric representations: "
            f"{representation_sensitivity['ari_vs_raw_partition'].min():.6f}"
        ),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_methodology(output_path: Path, config: Config) -> None:
    text = f"""METHODOLOGICAL NOTES
====================

Data
----
- The input contains numerical descriptors only.
- Activity is neither loaded nor analyzed.
- Constant features are removed before the analysis.

Selection of k
--------------
- For each of {config.n_stability_repeats} repetitions,
  {config.subsample_fraction:.0%} of rows are sampled without replacement.
- PCA is re-estimated on the original values inside every repetition.
- PCA centers the features using the training-subset means, but it does not
  divide them by their standard deviations.
- K-means is fitted on sampled rows and evaluated by silhouette on held-out
  rows.
- Stability is measured by pairwise ARI between full-data predictions from
  repeated fits.
- Among k values within one standard error of the best mean held-out
  silhouette, the most stable k is selected; ties favor the smaller k.

Final estimate
--------------
- The selected configuration is refitted on all rows using multiple external
  seeds and n_init={config.kmeans_n_init}.
- The representative seed has the highest mean ARI to the other final fits;
  ties favor lower inertia.

Sensitivity
-----------
- PCA thresholds {config.pca_sensitivity_values} are compared by ARI with the
  primary partition.
- The primary raw-data partition is compared with StandardScaler and
  RobustScaler representations.
- A low ARI indicates dependence on feature weighting.

Visualization
-------------
- Matrix rows are ordered by K-means cluster and PC1.
- Columns are ordered by decreasing variance, without clustering the features.
- The second matrix displays mean-centered rather than standardized values.
"""
    output_path.write_text(text, encoding="utf-8")


def main(config: Config | None = None) -> None:
    config = config or Config()
    dirs = make_output_dirs(config)
    X = load_data(config)

    print("\n1/5 Distribution analysis")
    distributions = analyze_distributions(
        X, dirs["distributions"], config
    )

    print("\n2/5 Repeated stability selection")
    stability_results = run_stability_selection(
        X, dirs["stability"], config
    )

    print("\n3/5 Final full-data refit")
    final_results = fit_final_model(
        X,
        stability_results["selected_k"],
        dirs["pca"],
        config,
    )

    print("\n4/5 Reordered matrices")
    make_reordered_matrix(
        X,
        distributions,
        final_results,
        dirs["matrix"],
        config,
    )

    print("\n5/5 Sensitivity analysis")
    pca_sensitivity = run_pca_sensitivity(
        X,
        final_results,
        stability_results["selected_k"],
        dirs["diagnostics"],
        config,
    )
    representation_sensitivity = run_representation_sensitivity(
        X,
        final_results,
        stability_results["selected_k"],
        dirs["diagnostics"],
        config,
    )

    write_summary(
        X,
        distributions,
        stability_results,
        final_results,
        pca_sensitivity,
        representation_sensitivity,
        dirs["root"] / "summary.txt",
    )
    write_methodology(
        dirs["root"] / "methodological_notes.txt", config
    )

    print("\nAnalysis completed.")
    print(f"Results saved in: {config.report_dir}")


if __name__ == "__main__":
    main()