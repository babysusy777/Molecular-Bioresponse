from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42
TARGET_COL = "Activity"

# A feature is removed when one value occurs in at least 99% of observations.
QUASI_CONSTANT_THRESHOLD = 0.99

# Sensitivity analysis requested for NJW.
K_VALUES = list(range(2, 11))
SIGMA_MULTIPLIERS = [0.5, 1.0, 2.0]

# None = classical Gower: equal weight for each contributing feature.
# A numeric value = total weight assigned to the numerical block;
# the binary block receives the complementary weight.
GOWER_CONFIGURATIONS: dict[str, float | None] = {
    "classic": None,
    "block_weight_num_0.4": 0.4,
    "block_weight_num_0.5": 0.5,
    "block_weight_num_0.6": 0.6,
}

KMEANS_N_INIT = 50
MIN_CLUSTER_FRACTION = 0.05
DISTANCE_DTYPE = np.float32
EIGEN_TOLERANCE = 1e-6
EIGEN_MAX_ITERATIONS = 5000
FIGURE_DPI = 200


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

INPUT_PATH = PROJECT_DIR / "Dataset" / "raw" / "train.csv"

REPORTS_DIR = PROJECT_DIR / "reports" / "jordan_weiss_gower_full_analysis"
RESULTS_PATH = REPORTS_DIR / "jordan_weiss_gower_all_results.csv"
BEST_RESULTS_PATH = REPORTS_DIR / "jordan_weiss_gower_best_configurations.csv"
PREPROCESSING_PATH = REPORTS_DIR / "jordan_weiss_gower_preprocessing.csv"
FINAL_CONFIG_PATH = REPORTS_DIR / "jordan_weiss_gower_final_configuration.csv"
FINAL_LABELS_PATH = REPORTS_DIR / "jordan_weiss_gower_final_labels.csv"
FINAL_EMBEDDING_PATH = REPORTS_DIR / "jordan_weiss_gower_final_embedding.csv"
FINAL_PLOT_PATH = REPORTS_DIR / "jordan_weiss_gower_final_embedding.png"
HEATMAPS_DIR = REPORTS_DIR / "sensitivity_heatmaps"
EIGENVECTOR_PLOTS_DIR = REPORTS_DIR / "eigenvector_diagnostics"

# Representative configurations selected for direct inspection of the first
# two eigenvectors. They cover: the automatically selected solution, the
# alternative classical-Gower bandwidth with a larger eigengap, the most
# informative block-balanced four-cluster solution, and the degenerate local
# graph that isolates only a few observations.
REPRESENTATIVE_EIGENVECTOR_PLOTS = [
    ("classic", 2.0, 2, "selected_classic_sigma2_k2"),
    ("classic", 1.0, 2, "alternative_classic_sigma1_k2"),
    ("block_weight_num_0.4", 0.5, 4, "balanced_num04_sigma05_k4"),
    ("classic", 0.5, 2, "degenerate_classic_sigma05_k2"),
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class Preprocessor:
    retained: list[str]
    binary: list[str]
    numeric: list[str]
    minima: pd.Series
    ranges: pd.Series


@dataclass
class DistanceComponents:
    d_num: np.ndarray
    d_bin: np.ndarray
    binary_union: np.ndarray
    p_num: int


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataset = pd.read_csv(path)

    unnamed_columns = [
        column for column in dataset.columns
        if str(column).startswith("Unnamed:")
    ]
    if unnamed_columns:
        dataset = dataset.drop(columns=unnamed_columns)

    if TARGET_COL not in dataset.columns:
        raise ValueError(f"Target column '{TARGET_COL}' was not found.")

    y = dataset[TARGET_COL].copy()
    X = dataset.drop(columns=TARGET_COL).copy()

    non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        raise TypeError(f"Non-numeric feature columns found: {non_numeric_columns}")

    if X.isna().any().any():
        missing_columns = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Missing values found in columns: {missing_columns}")

    return X, y


def dominant_ratio(series: pd.Series) -> float:
    frequencies = series.value_counts(normalize=True, dropna=False)
    return float(frequencies.iloc[0])


def identify_binary_columns(X: pd.DataFrame) -> list[str]:
    allowed_values = {0, 1, 0.0, 1.0, False, True}
    binary = []

    for column in X.columns:
        values = set(X[column].dropna().unique().tolist())
        if values and values.issubset(allowed_values):
            binary.append(column)

    return binary


def fit_preprocessor(X: pd.DataFrame) -> tuple[Preprocessor, dict[str, object]]:
    dominant_ratios = X.apply(dominant_ratio, axis=0)

    quasi_constant_features = dominant_ratios[
        dominant_ratios >= QUASI_CONSTANT_THRESHOLD
    ].index.tolist()

    retained_features = [
        column for column in X.columns
        if column not in quasi_constant_features
    ]

    if not retained_features:
        raise ValueError("All features were removed as quasi-constant.")

    X_filtered = X.loc[:, retained_features]

    binary_features = identify_binary_columns(X_filtered)
    binary_set = set(binary_features)
    numeric_features = [
        column for column in retained_features
        if column not in binary_set
    ]

    if not numeric_features:
        raise ValueError("No numerical features remain after preprocessing.")
    if not binary_features:
        raise ValueError("No binary features remain after preprocessing.")

    minima = X_filtered[numeric_features].min(axis=0)
    ranges = X_filtered[numeric_features].max(axis=0) - minima

    zero_range_features = ranges[ranges <= 0].index.tolist()
    if zero_range_features:
        retained_features = [
            column for column in retained_features
            if column not in zero_range_features
        ]
        numeric_features = [
            column for column in numeric_features
            if column not in zero_range_features
        ]
        minima = X_filtered[numeric_features].min(axis=0)
        ranges = X_filtered[numeric_features].max(axis=0) - minima

    preprocessor = Preprocessor(
        retained=retained_features,
        binary=binary_features,
        numeric=numeric_features,
        minima=minima,
        ranges=ranges,
    )

    diagnostics = {
        "original_features": X.shape[1],
        "quasi_constant_removed": len(quasi_constant_features),
        "zero_range_numeric_removed": len(zero_range_features),
        "remaining_features": len(retained_features),
        "remaining_numeric_features": len(numeric_features),
        "remaining_binary_features": len(binary_features),
        "quasi_constant_feature_names": ";".join(quasi_constant_features),
        "zero_range_numeric_feature_names": ";".join(zero_range_features),
    }

    return preprocessor, diagnostics


def transform(X: pd.DataFrame, preprocessor: Preprocessor) -> tuple[np.ndarray, np.ndarray]:
    missing_features = [
        feature for feature in preprocessor.retained
        if feature not in X.columns
    ]
    if missing_features:
        raise ValueError(f"Missing retained features: {missing_features}")

    X_numeric = X.loc[:, preprocessor.numeric]
    X_binary = X.loc[:, preprocessor.binary]

    X_numeric_scaled = (
        (X_numeric - preprocessor.minima) / preprocessor.ranges
    ).clip(lower=0.0, upper=1.0)

    return (
        X_numeric_scaled.to_numpy(dtype=DISTANCE_DTYPE, copy=True),
        X_binary.to_numpy(dtype=DISTANCE_DTYPE, copy=True),
    )


# =============================================================================
# GOWER DISTANCE
# =============================================================================

def numerical_component(X: np.ndarray) -> np.ndarray:
    distances = pairwise_distances(
        X,
        metric="manhattan",
        n_jobs=-1,
    )
    distances = distances.astype(DISTANCE_DTYPE, copy=False)
    distances /= float(X.shape[1])
    return distances


def binary_components(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    intersection = X @ X.T
    number_of_ones = X.sum(axis=1, dtype=np.float64)

    union = (
        number_of_ones[:, None]
        + number_of_ones[None, :]
        - intersection
    ).astype(DISTANCE_DTYPE, copy=False)

    mismatches = union - intersection

    jaccard_distance = np.divide(
        mismatches,
        union,
        out=np.zeros_like(union, dtype=DISTANCE_DTYPE),
        where=union > 0,
    )

    return jaccard_distance, union


def compute_distance_components(
    X_numeric: np.ndarray,
    X_binary: np.ndarray,
) -> DistanceComponents:
    d_num = numerical_component(X_numeric)
    d_bin, binary_union = binary_components(X_binary)

    return DistanceComponents(
        d_num=d_num,
        d_bin=d_bin,
        binary_union=binary_union,
        p_num=X_numeric.shape[1],
    )


def build_gower_distance(
    components: DistanceComponents,
    numerical_block_weight: float | None,
) -> np.ndarray:
    if numerical_block_weight is None:
        # Classical Gower: every contributing feature has equal weight.
        # d_bin * binary_union is exactly the number of binary mismatches.
        numerator = (
            components.d_num * components.p_num
            + components.d_bin * components.binary_union
        )
        denominator = components.p_num + components.binary_union
    else:
        alpha = float(numerical_block_weight)
        beta = 1.0 - alpha

        binary_available = (
            components.binary_union > 0
        ).astype(DISTANCE_DTYPE)

        numerator = (
            alpha * components.d_num
            + beta * binary_available * components.d_bin
        )
        denominator = alpha + beta * binary_available

    distance = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(components.d_num, dtype=DISTANCE_DTYPE),
        where=denominator > 0,
    )

    distance = np.clip(distance, 0.0, 1.0).astype(
        DISTANCE_DTYPE,
        copy=False,
    )
    np.fill_diagonal(distance, 0.0)
    return distance


# =============================================================================
# NG–JORDAN–WEISS
# =============================================================================

def row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def median_positive_distance(distance_matrix: np.ndarray) -> float:
    values = distance_matrix[
        np.triu_indices(distance_matrix.shape[0], k=1)
    ]
    positive_values = values[values > 0]

    if len(positive_values) == 0:
        raise ValueError("All off-diagonal Gower distances are zero.")

    return float(np.median(positive_values))


def gaussian_affinity(distance_matrix: np.ndarray, sigma: float) -> np.ndarray:
    affinity_matrix = np.exp(
        -(distance_matrix.astype(np.float64, copy=False) ** 2)
        / (2.0 * sigma**2)
    )
    np.fill_diagonal(affinity_matrix, 0.0)
    return affinity_matrix


def njw_eigendecomposition(
    affinity_matrix: np.ndarray,
    n_eigenvectors: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    W = np.asarray(affinity_matrix, dtype=np.float64)
    degrees = W.sum(axis=1)

    if np.any(degrees <= 0):
        raise ValueError("At least one object has zero graph degree. Increase sigma.")

    inverse_sqrt_degrees = 1.0 / np.sqrt(degrees)
    n_samples = W.shape[0]

    def matrix_vector_product(vector: np.ndarray) -> np.ndarray:
        return inverse_sqrt_degrees * (
            W @ (inverse_sqrt_degrees * vector)
        )

    def matrix_matrix_product(matrix: np.ndarray) -> np.ndarray:
        return inverse_sqrt_degrees[:, None] * (
            W @ (inverse_sqrt_degrees[:, None] * matrix)
        )

    normalized_operator = LinearOperator(
        shape=(n_samples, n_samples),
        matvec=matrix_vector_product,
        matmat=matrix_matrix_product,
        dtype=np.float64,
    )

    random_generator = np.random.default_rng(random_state)

    eigenvalues, eigenvectors = eigsh(
        normalized_operator,
        k=n_eigenvectors,
        which="LA",
        v0=random_generator.normal(size=n_samples),
        tol=EIGEN_TOLERANCE,
        maxiter=EIGEN_MAX_ITERATIONS,
    )

    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


# =============================================================================
# METRICS
# =============================================================================

def precomputed_silhouette(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
) -> float:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return np.nan

    matrix = np.asarray(distance_matrix, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)

    return float(
        silhouette_score(
            matrix,
            labels,
            metric="precomputed",
        )
    )


def euclidean_silhouette(
    embedding: np.ndarray,
    labels: np.ndarray,
) -> float:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return np.nan

    return float(
        silhouette_score(
            embedding,
            labels,
            metric="euclidean",
        )
    )


def mean_within_cluster_distance(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
) -> float:
    total_distance = 0.0
    total_pairs = 0

    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        if len(indices) < 2:
            continue

        block = distance_matrix[np.ix_(indices, indices)]
        upper_triangle = block[np.triu_indices(len(indices), k=1)]

        total_distance += float(upper_triangle.sum())
        total_pairs += len(upper_triangle)

    if total_pairs == 0:
        return np.nan

    return total_distance / total_pairs


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def save_first_two_eigenvector_plots(
    eigenvectors: np.ndarray,
    embedding: np.ndarray,
    labels: np.ndarray,
    gower_name: str,
    sigma_multiplier: float,
    k: int,
    file_stem: str,
) -> None:
    """Save raw and row-normalized projections on the first two eigenvectors.

    The raw plot shows the first two eigenvectors returned by the normalized
    affinity eigendecomposition. The normalized plot shows the first two
    coordinates after the Ng–Jordan–Weiss row normalization, i.e. the space
    actually supplied to K-means. For k > 2, the normalized plot is only a
    two-dimensional projection of the full k-dimensional spectral embedding.
    """

    EIGENVECTOR_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_coordinates = eigenvectors[:, :2]
    normalized_coordinates = embedding[:, :2]

    common_title = (
        f"{gower_name} | sigma={sigma_multiplier} x sigma_med | k={k}"
    )

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(
        raw_coordinates[:, 0],
        raw_coordinates[:, 1],
        c=labels,
        s=12,
        alpha=0.65,
    )
    axis.set_xlabel("Eigenvector 1")
    axis.set_ylabel("Eigenvector 2")
    axis.set_title("First two raw spectral eigenvectors\n" + common_title)
    figure.tight_layout()
    figure.savefig(
        EIGENVECTOR_PLOTS_DIR / f"{file_stem}_raw_eigenvectors.png",
        dpi=FIGURE_DPI,
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(
        normalized_coordinates[:, 0],
        normalized_coordinates[:, 1],
        c=labels,
        s=12,
        alpha=0.65,
    )
    axis.set_xlabel("Normalized spectral coordinate 1")
    axis.set_ylabel("Normalized spectral coordinate 2")
    axis.set_title("First two NJW row-normalized coordinates\n" + common_title)
    figure.tight_layout()
    figure.savefig(
        EIGENVECTOR_PLOTS_DIR / f"{file_stem}_normalized_coordinates.png",
        dpi=FIGURE_DPI,
    )
    plt.close(figure)

    pd.DataFrame(
        {
            "raw_eigenvector_1": raw_coordinates[:, 0],
            "raw_eigenvector_2": raw_coordinates[:, 1],
            "normalized_coordinate_1": normalized_coordinates[:, 0],
            "normalized_coordinate_2": normalized_coordinates[:, 1],
            "cluster": labels,
        }
    ).to_csv(
        EIGENVECTOR_PLOTS_DIR / f"{file_stem}_coordinates.csv",
        index=False,
    )


def run_sensitivity_analysis(
    components: DistanceComponents,
    n_samples: int,
) -> pd.DataFrame:
    results: list[dict[str, object]] = []
    minimum_cluster_size = max(
        1,
        int(np.ceil(MIN_CLUSTER_FRACTION * n_samples)),
    )
    n_eigenvectors = max(K_VALUES) + 1

    for gower_index, (gower_name, numerical_weight) in enumerate(
        GOWER_CONFIGURATIONS.items()
    ):
        print("\n" + "=" * 88)
        print(f"GOWER CONFIGURATION: {gower_name}")

        distance_matrix = build_gower_distance(
            components,
            numerical_weight,
        )

        sigma_reference = median_positive_distance(distance_matrix)
        print(f"Median positive Gower distance: {sigma_reference:.6f}")

        for sigma_multiplier in SIGMA_MULTIPLIERS:
            sigma = sigma_multiplier * sigma_reference
            print(
                f"\nSigma multiplier={sigma_multiplier:.1f} | "
                f"sigma={sigma:.6f}"
            )

            affinity_matrix = gaussian_affinity(distance_matrix, sigma)

            seed = (
                RANDOM_STATE
                + gower_index * 1000
                + int(sigma_multiplier * 100)
            )

            eigenvalues, eigenvectors = njw_eigendecomposition(
                affinity_matrix,
                n_eigenvectors,
                seed,
            )

            for k in K_VALUES:
                embedding = row_normalize(eigenvectors[:, :k])

                kmeans = KMeans(
                    n_clusters=k,
                    init="k-means++",
                    n_init=KMEANS_N_INIT,
                    random_state=(
                        RANDOM_STATE
                        + gower_index * 10000
                        + int(sigma_multiplier * 1000)
                        + k
                    ),
                )
                labels = kmeans.fit_predict(embedding)

                for (
                    plot_gower,
                    plot_sigma_multiplier,
                    plot_k,
                    plot_file_stem,
                ) in REPRESENTATIVE_EIGENVECTOR_PLOTS:
                    if (
                        gower_name == plot_gower
                        and np.isclose(sigma_multiplier, plot_sigma_multiplier)
                        and k == plot_k
                    ):
                        save_first_two_eigenvector_plots(
                            eigenvectors=eigenvectors,
                            embedding=embedding,
                            labels=labels,
                            gower_name=gower_name,
                            sigma_multiplier=sigma_multiplier,
                            k=k,
                            file_stem=plot_file_stem,
                        )

                cluster_sizes = np.bincount(labels, minlength=k)
                valid_cluster_sizes = bool(
                    np.all(cluster_sizes >= minimum_cluster_size)
                )

                silhouette_gower = precomputed_silhouette(
                    distance_matrix,
                    labels,
                )
                silhouette_spectral = euclidean_silhouette(
                    embedding,
                    labels,
                )
                within_gower = mean_within_cluster_distance(
                    distance_matrix,
                    labels,
                )
                eigengap = float(eigenvalues[k - 1] - eigenvalues[k])

                results.append(
                    {
                        "gower_configuration": gower_name,
                        "numerical_block_weight": numerical_weight,
                        "binary_block_weight": (
                            None
                            if numerical_weight is None
                            else 1.0 - numerical_weight
                        ),
                        "sigma_reference_median": sigma_reference,
                        "sigma_multiplier": sigma_multiplier,
                        "sigma": sigma,
                        "k": k,
                        "gower_silhouette": silhouette_gower,
                        "spectral_silhouette": silhouette_spectral,
                        "mean_within_gower": within_gower,
                        "eigengap": eigengap,
                        "kmeans_inertia_spectral": float(kmeans.inertia_),
                        "valid_cluster_sizes": valid_cluster_sizes,
                        "minimum_required_cluster_size": minimum_cluster_size,
                        "minimum_cluster_size": int(cluster_sizes.min()),
                        "maximum_cluster_size": int(cluster_sizes.max()),
                        "cluster_sizes": json.dumps(cluster_sizes.tolist()),
                    }
                )

                print(
                    f"k={k:>2} | "
                    f"sil_gower={silhouette_gower:>7.4f} | "
                    f"sil_spectral={silhouette_spectral:>7.4f} | "
                    f"eigengap={eigengap:>10.7f} | "
                    f"valid={valid_cluster_sizes} | "
                    f"sizes={cluster_sizes.tolist()}"
                )

            del affinity_matrix
            del eigenvalues
            del eigenvectors
            gc.collect()

        del distance_matrix
        gc.collect()

    return pd.DataFrame(results)


def select_best_configuration(results: pd.DataFrame) -> pd.Series:
    valid_results = results[results["valid_cluster_sizes"]].copy()

    if valid_results.empty:
        print(
            "\nNo configuration satisfies the minimum cluster-size constraint. "
            "The constraint will therefore be used as a preference rather than "
            "as a hard exclusion."
        )
        candidates = results.copy()
    else:
        candidates = valid_results

    # Primary criterion: Gower silhouette.
    # Supporting criteria: eigengap, spectral silhouette, then smaller k.
    ordered = candidates.sort_values(
        by=[
            "gower_silhouette",
            "eigengap",
            "spectral_silhouette",
            "k",
        ],
        ascending=[False, False, False, True],
    )

    return ordered.iloc[0]


# =============================================================================
# PLOTS
# =============================================================================

def plot_heatmaps(results: pd.DataFrame) -> None:
    HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)

    for gower_name in GOWER_CONFIGURATIONS:
        subset = results[
            results["gower_configuration"] == gower_name
        ]

        pivot = subset.pivot(
            index="k",
            columns="sigma_multiplier",
            values="gower_silhouette",
        ).reindex(index=K_VALUES, columns=SIGMA_MULTIPLIERS)

        values = pivot.to_numpy(dtype=float)

        figure, axis = plt.subplots(figsize=(7, 8))
        image = axis.imshow(values, aspect="auto")

        axis.set_xticks(
            range(len(SIGMA_MULTIPLIERS)),
            SIGMA_MULTIPLIERS,
        )
        axis.set_yticks(
            range(len(K_VALUES)),
            K_VALUES,
        )
        axis.set_xlabel("Sigma multiplier relative to median Gower distance")
        axis.set_ylabel("Number of clusters k")
        axis.set_title(f"Gower silhouette sensitivity — {gower_name}")

        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                if np.isfinite(value):
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                    )

        figure.colorbar(image, ax=axis, label="Gower silhouette")
        figure.tight_layout()
        figure.savefig(
            HEATMAPS_DIR / f"silhouette_heatmap_{gower_name}.png",
            dpi=FIGURE_DPI,
        )
        plt.close(figure)


# =============================================================================
# FINAL MODEL
# =============================================================================

def fit_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    components: DistanceComponents,
    preprocessor: Preprocessor,
    best_configuration: pd.Series,
) -> None:
    gower_name = str(best_configuration["gower_configuration"])
    numerical_weight = GOWER_CONFIGURATIONS[gower_name]
    sigma_multiplier = float(best_configuration["sigma_multiplier"])
    k = int(best_configuration["k"])

    distance_matrix = build_gower_distance(
        components,
        numerical_weight,
    )
    sigma_reference = median_positive_distance(distance_matrix)
    sigma = sigma_multiplier * sigma_reference

    affinity_matrix = gaussian_affinity(distance_matrix, sigma)
    n_plot_eigenvectors = max(k, 3)

    eigenvalues, eigenvectors = njw_eigendecomposition(
        affinity_matrix,
        n_plot_eigenvectors,
        RANDOM_STATE,
    )

    clustering_embedding = row_normalize(eigenvectors[:, :k])

    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=KMEANS_N_INIT,
        random_state=RANDOM_STATE,
    )
    labels = kmeans.fit_predict(clustering_embedding)

    cluster_sizes = np.bincount(labels, minlength=k)
    final_gower_silhouette = precomputed_silhouette(
        distance_matrix,
        labels,
    )
    final_spectral_silhouette = euclidean_silhouette(
        clustering_embedding,
        labels,
    )

    final_configuration = pd.DataFrame(
        [
            {
                "gower_configuration": gower_name,
                "numerical_block_weight": numerical_weight,
                "binary_block_weight": (
                    None
                    if numerical_weight is None
                    else 1.0 - numerical_weight
                ),
                "sigma_reference_median": sigma_reference,
                "sigma_multiplier": sigma_multiplier,
                "sigma": sigma,
                "k": k,
                "gower_silhouette": final_gower_silhouette,
                "spectral_silhouette": final_spectral_silhouette,
                "cluster_sizes": json.dumps(cluster_sizes.tolist()),
                "retained_features": len(preprocessor.retained),
                "numeric_features": len(preprocessor.numeric),
                "binary_features": len(preprocessor.binary),
            }
        ]
    )
    final_configuration.to_csv(FINAL_CONFIG_PATH, index=False)

    labels_output = pd.DataFrame(
        {
            "row_index": X.index,
            "cluster": labels,
            # Activity is added only for post-hoc interpretation.
            TARGET_COL: y.to_numpy(),
        }
    )
    labels_output.to_csv(FINAL_LABELS_PATH, index=False)

    plotting_embedding = row_normalize(eigenvectors[:, :3])
    embedding_output = pd.DataFrame(
        {
            "spectral_coordinate_1": plotting_embedding[:, 0],
            "spectral_coordinate_2": plotting_embedding[:, 1],
            "spectral_coordinate_3": plotting_embedding[:, 2],
            "cluster": labels,
        }
    )
    embedding_output.to_csv(FINAL_EMBEDDING_PATH, index=False)

    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        plotting_embedding[:, 0],
        plotting_embedding[:, 1],
        plotting_embedding[:, 2],
        c=labels,
        s=12,
        alpha=0.65,
    )
    axis.set_xlabel("Normalized spectral coordinate 1")
    axis.set_ylabel("Normalized spectral coordinate 2")
    axis.set_zlabel("Normalized spectral coordinate 3")
    axis.set_title(
        "Ng–Jordan–Weiss with Gower distance\n"
        f"{gower_name}, k={k}, sigma={sigma_multiplier} × sigma_med"
    )
    figure.tight_layout()
    figure.savefig(FINAL_PLOT_PATH, dpi=FIGURE_DPI)
    plt.close(figure)

    print("\n" + "=" * 88)
    print("FINAL MODEL")
    print(f"Gower configuration: {gower_name}")
    print(f"k: {k}")
    print(f"Sigma multiplier: {sigma_multiplier}")
    print(f"Sigma: {sigma:.6f}")
    print(f"Gower silhouette: {final_gower_silhouette:.6f}")
    print(f"Spectral silhouette: {final_spectral_silhouette:.6f}")
    print(f"Cluster sizes: {cluster_sizes.tolist()}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data(INPUT_PATH)

    print("=" * 88)
    print("GOWER + NG–JORDAN–WEISS + K-MEANS")
    print("FULL-DATA INTERNAL VALIDATION AND SENSITIVITY ANALYSIS")
    print(f"Dataset: {X.shape[0]} objects × {X.shape[1]} input features")
    print(f"'{TARGET_COL}' is excluded from every unsupervised step.")
    print(f"Gower configurations: {list(GOWER_CONFIGURATIONS)}")
    print(f"Sigma multipliers: {SIGMA_MULTIPLIERS}")
    print(f"k values: {K_VALUES}")

    preprocessor, preprocessing_diagnostics = fit_preprocessor(X)
    pd.DataFrame([preprocessing_diagnostics]).to_csv(
        PREPROCESSING_PATH,
        index=False,
    )

    X_numeric, X_binary = transform(X, preprocessor)

    print("\nPREPROCESSING")
    print(f"Retained features: {len(preprocessor.retained)}")
    print(f"Numerical features: {len(preprocessor.numeric)}")
    print(f"Binary features: {len(preprocessor.binary)}")

    components = compute_distance_components(X_numeric, X_binary)

    results = run_sensitivity_analysis(
        components,
        n_samples=len(X),
    )

    results = results.sort_values(
        by=[
            "valid_cluster_sizes",
            "gower_silhouette",
            "eigengap",
            "spectral_silhouette",
            "k",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    results.to_csv(RESULTS_PATH, index=False)
    results.head(20).to_csv(BEST_RESULTS_PATH, index=False)

    best_configuration = select_best_configuration(results)

    plot_heatmaps(results)

    columns_to_show = [
        "gower_configuration",
        "sigma_multiplier",
        "k",
        "gower_silhouette",
        "spectral_silhouette",
        "eigengap",
        "valid_cluster_sizes",
        "cluster_sizes",
    ]

    print("\n" + "=" * 88)
    print("TOP CONFIGURATIONS")
    print(results[columns_to_show].head(20).to_string(index=False))

    print("\nSELECTED CONFIGURATION")
    print(best_configuration[columns_to_show].to_string())

    fit_final_model(
        X,
        y,
        components,
        preprocessor,
        best_configuration,
    )

    print("\nSAVED FILES")
    for path in [
        PREPROCESSING_PATH,
        RESULTS_PATH,
        BEST_RESULTS_PATH,
        HEATMAPS_DIR,
        EIGENVECTOR_PLOTS_DIR,
        FINAL_CONFIG_PATH,
        FINAL_LABELS_PATH,
        FINAL_EMBEDDING_PATH,
        FINAL_PLOT_PATH,
    ]:
        print(path)


if __name__ == "__main__":
    main()
