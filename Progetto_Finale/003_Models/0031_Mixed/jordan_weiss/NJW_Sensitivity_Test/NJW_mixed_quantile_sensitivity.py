from __future__ import annotations

"""Mixed-data quantile sensitivity analysis with Gower + Ng-Jordan-Weiss.

The numerical descriptors are transformed feature-wise through an empirical
quantile map to Uniform[0, 1]. Binary descriptors remain unchanged and are
handled as asymmetric attributes through Jaccard dissimilarity. This script is
an explicit sensitivity analysis: it does not replace the standard mixed-data
Gower baseline.

The target ``Activity`` is excluded from transformation, distance construction,
parameter selection and clustering. When available, it is appended only to the
exported label table for post-hoc interpretation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import gc
import json
import math
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score
from sklearn.preprocessing import QuantileTransformer


# =============================================================================
# Configuration and project paths
# =============================================================================

RANDOM_STATE = 42
TARGET_COL = "Activity"

K_VALUES = tuple(range(2, 11))
SIGMA_MULTIPLIERS = (0.5, 1.0, 2.0)
SENSITIVITY_ALPHAS = (0.4, 0.5, 0.6)
BALANCED_ALPHA = 0.5

KMEANS_N_INIT = 50
MIN_CLUSTER_FRACTION = 0.05
GRID_SILHOUETTE_SAMPLE_SIZE: int | None = None
QUANTILE_MAX_LANDMARKS = 1000
DISTANCE_DTYPE = np.float32
EIGEN_TOLERANCE = 1e-6
EIGEN_MAX_ITERATIONS = 5000
FIGURE_DPI = 220

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = SCRIPT_DIR.parent.parent

DATASET_DIR = PROJECT_DIR / "000_Dataset"
MIXED_MODEL_DIR = PROJECT_DIR / "003_Models" / "0031_Mixed" / "jordan_weiss"
ANALYSIS_DIR = MIXED_MODEL_DIR / "NJW_Sensitivity_Test"

FEATURES_PATH = DATASET_DIR / "train_filtered_no_activity.csv"
ACTIVITY_PATH = DATASET_DIR / "train_activity_target.csv"
RAW_TRAIN_PATH = DATASET_DIR / "raw" / "train.csv"

REPORT_DIR = ANALYSIS_DIR / "report_NJW_mixed_quantile_sensitivity"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"
COORDINATE_DIR = REPORT_DIR / "coordinates"

PREPROCESSING_CSV = TABLE_DIR / "01_preprocessing_summary.csv"
GRID_RESULTS_CSV = TABLE_DIR / "02_model_selection_grid.csv"
SELECTED_SOLUTIONS_CSV = TABLE_DIR / "03_selected_solutions.csv"
SENSITIVITY_CSV = TABLE_DIR / "04_weight_sensitivity.csv"
SENSITIVITY_ARI_CSV = TABLE_DIR / "05_weight_sensitivity_ari.csv"
NESTING_COUNTS_CSV = TABLE_DIR / "06_balanced_k2_vs_k4_counts.csv"
NESTING_PERCENTAGES_CSV = TABLE_DIR / "07_balanced_k2_vs_k4_row_percentages.csv"
LABELS_CSV = TABLE_DIR / "08_mixed_quantile_labels.csv"
REPORT_VALUES_JSON = TABLE_DIR / "09_report_values.json"
CROSS_REPRESENTATION_CSV = TABLE_DIR / "10_cross_representation_ari.csv"
RESULTS_SUMMARY_MD = REPORT_DIR / "results_summary.md"

ORIGINAL_MIXED_LABELS = (
    MIXED_MODEL_DIR / "report_jordan_weiss" / "tables" / "08_mixed_baseline_labels.csv"
)

BINARY_REFERENCE_CANDIDATES = (
    PROJECT_DIR / "003_Models" / "0032_Binary" / "report_NJW_binary" / "tables" / "10_binary_labels.csv",
    PROJECT_DIR / "003_Models" / "0032_Binary" / "report_jordan_weiss" / "tables" / "10_binary_labels.csv",
    PROJECT_DIR / "003_Models" / "0032_Binary" / "jordan_weiss" / "report_jordan_weiss" / "tables" / "10_binary_labels.csv",
)
NUMERIC_REFERENCE_CANDIDATES = (
    PROJECT_DIR / "003_Models" / "0033_Numeric" / "report_NJW_numeric" / "tables" / "10_numeric_labels.csv",
    PROJECT_DIR / "003_Models" / "0033_Numeric" / "report_jordan_weiss" / "tables" / "10_numeric_labels.csv",
    PROJECT_DIR / "003_Models" / "0033_Numeric" / "jordan_weiss" / "report_jordan_weiss" / "tables" / "10_numeric_labels.csv",
)
NUMERIC_QUANTILE_REFERENCE_CANDIDATES = (
    PROJECT_DIR / "003_Models" / "0033_Numeric" / "NJW_Sensitivity_Test" / "report_NJW_numeric_quantile" / "tables" / "10_numeric_quantile_labels.csv",
    PROJECT_DIR / "003_Models" / "0033_Numeric" / "report_NJW_numeric_quantile" / "tables" / "10_numeric_quantile_labels.csv",
    PROJECT_DIR / "003_Models" / "0033_Numeric" / "jordan_weiss" / "report_jordan_weiss_quantile" / "tables" / "10_numeric_quantile_labels.csv",
)


# =============================================================================
# Data structures
# =============================================================================


@dataclass(frozen=True)
class ProjectData:
    X: pd.DataFrame
    activity: pd.Series | None
    row_alignment_verified: bool


@dataclass(frozen=True)
class FeatureSchema:
    numeric: list[str]
    binary: list[str]
    removed_zero_range_numeric: list[str]


@dataclass
class DistanceComponents:
    numerical: np.ndarray
    binary: np.ndarray
    binary_union: np.ndarray
    n_numeric: int


@dataclass
class SpectralBasis:
    configuration: str
    alpha: float | None
    sigma_reference: float
    sigma_multiplier: float
    sigma: float
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


@dataclass
class Solution:
    configuration: str
    alpha: float | None
    sigma_reference: float
    sigma_multiplier: float
    sigma: float
    k: int
    eigenvalues: np.ndarray
    raw_eigenvectors: np.ndarray
    embedding: np.ndarray
    labels: np.ndarray
    cluster_sizes: np.ndarray
    gower_silhouette: float
    spectral_silhouette: float
    eigengap: float
    valid_cluster_sizes: bool


@dataclass(frozen=True)
class ReferenceLabels:
    name: str
    frame: pd.DataFrame
    source_path: Path


# =============================================================================
# Data loading and transformation
# =============================================================================


def read_csv_clean(path: Path, *, required: bool = True) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return None
    frame = pd.read_csv(path)
    unnamed = [column for column in frame.columns if str(column).startswith("Unnamed:")]
    return frame.drop(columns=unnamed) if unnamed else frame


def load_activity(X: pd.DataFrame) -> tuple[pd.Series | None, bool]:
    activity_file: pd.Series | None = None
    if ACTIVITY_PATH.exists():
        frame = read_csv_clean(ACTIVITY_PATH)
        assert frame is not None
        if TARGET_COL not in frame.columns:
            raise ValueError(f"'{TARGET_COL}' not found in {ACTIVITY_PATH}")
        activity_file = frame[TARGET_COL].reset_index(drop=True)
        if len(activity_file) != len(X):
            raise ValueError("Activity and feature matrix have different row counts.")

    raw_activity: pd.Series | None = None
    alignment_verified = False
    if RAW_TRAIN_PATH.exists():
        raw = read_csv_clean(RAW_TRAIN_PATH)
        assert raw is not None
        raw = raw.reset_index(drop=True)
        if len(raw) != len(X):
            raise ValueError("raw/train.csv and the filtered matrix have different row counts.")
        if TARGET_COL in raw.columns:
            raw_activity = raw[TARGET_COL].reset_index(drop=True)

        missing = [column for column in X.columns if column not in raw.columns]
        if missing:
            warnings.warn(
                "Row alignment could not be verified against raw/train.csv because "
                f"{len(missing)} retained columns are absent from the raw file.",
                stacklevel=2,
            )
        else:
            raw_values = raw.loc[:, X.columns].to_numpy(dtype=np.float64, copy=False)
            filtered_values = X.to_numpy(dtype=np.float64, copy=False)
            if not np.allclose(raw_values, filtered_values, rtol=1e-10, atol=1e-12):
                raise ValueError(
                    "The filtered feature matrix is not row-aligned with raw/train.csv."
                )
            alignment_verified = True

    if activity_file is not None and raw_activity is not None:
        if not activity_file.equals(raw_activity):
            raise ValueError(
                "train_activity_target.csv does not match Activity in raw/train.csv."
            )

    return activity_file if activity_file is not None else raw_activity, alignment_verified


def load_project_data() -> ProjectData:
    frame = read_csv_clean(FEATURES_PATH)
    assert frame is not None
    X = frame.reset_index(drop=True)

    if TARGET_COL in X.columns:
        raise ValueError(
            f"{FEATURES_PATH.name} must not contain the target '{TARGET_COL}'."
        )
    if X.empty:
        raise ValueError("The feature matrix is empty.")

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise TypeError(f"Non-numeric descriptors found: {non_numeric[:10]}")
    if X.isna().any().any():
        missing = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Missing values found in: {missing[:10]}")

    duplicates = int(X.duplicated().sum())
    if duplicates:
        warnings.warn(
            f"{duplicates} duplicate feature rows were retained to preserve alignment.",
            stacklevel=2,
        )

    activity, alignment_verified = load_activity(X)
    return ProjectData(X=X, activity=activity, row_alignment_verified=alignment_verified)


def identify_binary_columns(X: pd.DataFrame) -> list[str]:
    allowed = {0, 1, 0.0, 1.0, False, True}
    return [
        column
        for column in X.columns
        if set(X[column].dropna().unique().tolist()).issubset(allowed)
    ]


def fit_schema(X: pd.DataFrame) -> FeatureSchema:
    binary = identify_binary_columns(X)
    binary_set = set(binary)
    numeric = [column for column in X.columns if column not in binary_set]

    if not binary:
        raise ValueError("No binary descriptors were identified.")
    if not numeric:
        raise ValueError("No non-binary numerical descriptors were identified.")

    ranges = X[numeric].max(axis=0) - X[numeric].min(axis=0)
    removed = ranges[ranges <= 0].index.tolist()
    if removed:
        removed_set = set(removed)
        numeric = [column for column in numeric if column not in removed_set]
        warnings.warn(
            f"Removed {len(removed)} zero-range numerical descriptors.",
            stacklevel=2,
        )

    if not numeric:
        raise ValueError("No numerical descriptors remain after zero-range removal.")
    return FeatureSchema(
        numeric=numeric,
        binary=binary,
        removed_zero_range_numeric=removed,
    )


def transform_features(
    X: pd.DataFrame,
    schema: FeatureSchema,
) -> tuple[np.ndarray, np.ndarray]:
    numerical = X[schema.numeric].to_numpy(dtype=np.float64, copy=True)
    binary = X[schema.binary].to_numpy(dtype=DISTANCE_DTYPE, copy=True)

    if not np.isfinite(numerical).all() or not np.isfinite(binary).all():
        raise ValueError("Non-finite values found before transformation.")

    n_objects = len(X)
    transformer = QuantileTransformer(
        n_quantiles=min(QUANTILE_MAX_LANDMARKS, n_objects),
        output_distribution="uniform",
        subsample=n_objects,
        random_state=RANDOM_STATE,
        copy=True,
    )
    numerical = transformer.fit_transform(numerical)
    numerical = np.clip(numerical, 0.0, 1.0).astype(DISTANCE_DTYPE, copy=False)

    if not np.isfinite(numerical).all():
        raise ValueError("Quantile transformation produced non-finite values.")
    return numerical, binary


def preprocessing_summary(data: ProjectData, schema: FeatureSchema) -> dict[str, object]:
    binary_values = data.X[schema.binary].to_numpy(dtype=np.float64, copy=False)
    retained = len(schema.numeric) + len(schema.binary)
    one_percentage = float(binary_values.mean() * 100.0)

    return {
        "project_dir": str(PROJECT_DIR),
        "feature_input": str(FEATURES_PATH),
        "activity_source": (
            str(ACTIVITY_PATH)
            if ACTIVITY_PATH.exists()
            else str(RAW_TRAIN_PATH) if RAW_TRAIN_PATH.exists() else None
        ),
        "n_objects": len(data.X),
        "features_in_filtered_file": data.X.shape[1],
        "zero_range_numeric_removed": len(schema.removed_zero_range_numeric),
        "retained_features": retained,
        "numeric_features": len(schema.numeric),
        "binary_features": len(schema.binary),
        "numeric_percentage": 100.0 * len(schema.numeric) / retained,
        "binary_percentage": 100.0 * len(schema.binary) / retained,
        "binary_one_percentage": one_percentage,
        "binary_zero_percentage": 100.0 - one_percentage,
        "numerical_transform": "feature-wise empirical quantile to Uniform[0,1]",
        "quantile_landmarks": min(QUANTILE_MAX_LANDMARKS, len(data.X)),
        "quantile_subsampling": "disabled: all observations used",
        "binary_transform": "unchanged asymmetric Jaccard block",
        "analysis_role": "sensitivity analysis",
        "activity_available_post_hoc_only": data.activity is not None,
        "activity_row_alignment_verified_against_raw": data.row_alignment_verified,
    }


# =============================================================================
# Mixed dissimilarities
# =============================================================================


def compute_distance_components(
    X_numeric: np.ndarray,
    X_binary: np.ndarray,
) -> DistanceComponents:
    print("Computing quantile-numerical Gower component...")
    d_numeric = pairwise_distances(X_numeric, metric="manhattan", n_jobs=-1)
    d_numeric = d_numeric.astype(DISTANCE_DTYPE, copy=False)
    d_numeric /= float(X_numeric.shape[1])
    np.fill_diagonal(d_numeric, 0.0)

    print("Computing asymmetric binary Gower component...")
    intersection = X_binary @ X_binary.T
    ones = X_binary.sum(axis=1, dtype=np.float64)
    union = (ones[:, None] + ones[None, :] - intersection).astype(
        DISTANCE_DTYPE, copy=False
    )
    mismatches = union - intersection
    d_binary = np.divide(
        mismatches,
        union,
        out=np.zeros_like(union, dtype=DISTANCE_DTYPE),
        where=union > 0,
    )
    np.fill_diagonal(d_binary, 0.0)

    return DistanceComponents(
        numerical=d_numeric,
        binary=d_binary,
        binary_union=union,
        n_numeric=X_numeric.shape[1],
    )


def configuration_name(alpha: float | None) -> str:
    return "quantile_classical" if alpha is None else f"quantile_weighted_alpha_{alpha:.1f}"


def build_gower_distance(
    components: DistanceComponents,
    alpha: float | None,
) -> np.ndarray:
    if alpha is None:
        numerator = (
            components.numerical * components.n_numeric
            + components.binary * components.binary_union
        )
        denominator = components.n_numeric + components.binary_union
    else:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1.")
        beta = 1.0 - alpha
        binary_available = (components.binary_union > 0).astype(DISTANCE_DTYPE)
        numerator = (
            alpha * components.numerical
            + beta * binary_available * components.binary
        )
        denominator = alpha + beta * binary_available

    distance = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(components.numerical, dtype=DISTANCE_DTYPE),
        where=denominator > 0,
    )
    np.clip(distance, 0.0, 1.0, out=distance)
    np.fill_diagonal(distance, 0.0)
    return distance


# =============================================================================
# Ng-Jordan-Weiss clustering
# =============================================================================


def median_positive_distance(distance: np.ndarray) -> float:
    upper = distance[np.triu_indices(distance.shape[0], k=1)]
    positive = upper[upper > 0]
    if positive.size == 0:
        raise ValueError("All off-diagonal dissimilarities are zero.")
    return float(np.median(positive))


def gaussian_affinity(distance: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    affinity = np.exp(-(distance.astype(np.float64) ** 2) / (2.0 * sigma**2))
    np.fill_diagonal(affinity, 0.0)
    return affinity


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def eigendecompose_affinity(
    affinity: np.ndarray,
    n_eigenvectors: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    affinity = np.asarray(affinity, dtype=np.float64)
    degrees = affinity.sum(axis=1)
    if np.any(degrees <= 0):
        raise ValueError("Zero graph degree encountered. Increase sigma.")

    inv_sqrt_degree = 1.0 / np.sqrt(degrees)
    n_objects = affinity.shape[0]

    def matvec(vector: np.ndarray) -> np.ndarray:
        return inv_sqrt_degree * (affinity @ (inv_sqrt_degree * vector))

    def matmat(matrix: np.ndarray) -> np.ndarray:
        return inv_sqrt_degree[:, None] * (
            affinity @ (inv_sqrt_degree[:, None] * matrix)
        )

    operator = LinearOperator(
        shape=(n_objects, n_objects),
        matvec=matvec,
        matmat=matmat,
        dtype=np.float64,
    )

    rng = np.random.default_rng(seed)
    try:
        eigenvalues, eigenvectors = eigsh(
            operator,
            k=n_eigenvectors,
            which="LA",
            v0=rng.normal(size=n_objects),
            tol=EIGEN_TOLERANCE,
            maxiter=EIGEN_MAX_ITERATIONS,
        )
    except ArpackNoConvergence as error:
        if error.eigenvalues is None or len(error.eigenvalues) < n_eigenvectors:
            raise RuntimeError("ARPACK did not converge to enough eigenpairs.") from error
        warnings.warn(
            "ARPACK reached the iteration limit; converged eigenpairs are used.",
            stacklevel=2,
        )
        eigenvalues = error.eigenvalues
        eigenvectors = error.eigenvectors

    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def build_basis(
    distance: np.ndarray,
    alpha: float | None,
    sigma_multiplier: float,
    seed: int,
) -> SpectralBasis:
    sigma_reference = median_positive_distance(distance)
    sigma = sigma_multiplier * sigma_reference
    affinity = gaussian_affinity(distance, sigma)
    eigenvalues, eigenvectors = eigendecompose_affinity(
        affinity,
        n_eigenvectors=max(K_VALUES) + 1,
        seed=seed,
    )
    del affinity
    gc.collect()

    return SpectralBasis(
        configuration=configuration_name(alpha),
        alpha=alpha,
        sigma_reference=sigma_reference,
        sigma_multiplier=sigma_multiplier,
        sigma=sigma,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
    )


def compute_silhouette(
    data: np.ndarray,
    labels: np.ndarray,
    metric: str,
    sample_size: int | None,
    seed: int,
) -> float:
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(labels):
        return float("nan")

    matrix = data
    if metric == "precomputed":
        matrix = np.asarray(data, dtype=np.float64).copy()
        np.fill_diagonal(matrix, 0.0)

    kwargs: dict[str, object] = {"metric": metric}
    if sample_size is not None and sample_size < len(labels):
        kwargs.update(sample_size=sample_size, random_state=seed)
    return float(silhouette_score(matrix, labels, **kwargs))


def fit_solution(
    basis: SpectralBasis,
    distance: np.ndarray,
    k: int,
    seed: int,
    exact_silhouettes: bool,
) -> Solution:
    raw = basis.eigenvectors[:, :k]
    embedding = row_normalize(raw)
    labels = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=KMEANS_N_INIT,
        random_state=seed,
    ).fit_predict(embedding)

    sizes = np.bincount(labels, minlength=k)
    minimum_size = math.ceil(MIN_CLUSTER_FRACTION * len(labels))
    sample_size = None if exact_silhouettes else GRID_SILHOUETTE_SAMPLE_SIZE

    return Solution(
        configuration=basis.configuration,
        alpha=basis.alpha,
        sigma_reference=basis.sigma_reference,
        sigma_multiplier=basis.sigma_multiplier,
        sigma=basis.sigma,
        k=k,
        eigenvalues=basis.eigenvalues,
        raw_eigenvectors=raw,
        embedding=embedding,
        labels=labels,
        cluster_sizes=sizes,
        gower_silhouette=compute_silhouette(
            distance, labels, "precomputed", sample_size, seed
        ),
        spectral_silhouette=compute_silhouette(
            embedding, labels, "euclidean", sample_size, seed
        ),
        eigengap=float(basis.eigenvalues[k - 1] - basis.eigenvalues[k]),
        valid_cluster_sizes=bool(np.all(sizes >= minimum_size)),
    )


def solution_row(role: str, solution: Solution, exact: bool) -> dict[str, object]:
    return {
        "role": role,
        "configuration": solution.configuration,
        "numerical_block_weight": solution.alpha,
        "binary_block_weight": None if solution.alpha is None else 1.0 - solution.alpha,
        "k": solution.k,
        "sigma_reference_median": solution.sigma_reference,
        "sigma_multiplier": solution.sigma_multiplier,
        "sigma": solution.sigma,
        "gower_silhouette": solution.gower_silhouette,
        "spectral_silhouette": solution.spectral_silhouette,
        "silhouettes_exact": exact,
        "eigengap": solution.eigengap,
        "valid_cluster_sizes": solution.valid_cluster_sizes,
        "minimum_cluster_size": int(solution.cluster_sizes.min()),
        "maximum_cluster_size": int(solution.cluster_sizes.max()),
        "cluster_sizes": json.dumps(solution.cluster_sizes.tolist()),
    }


def model_selection_seed(alpha: float | None, sigma_multiplier: float) -> int:
    configuration_index = 0 if alpha is None else 1
    sigma_index = SIGMA_MULTIPLIERS.index(float(sigma_multiplier))
    return RANDOM_STATE + 1000 * configuration_index + 100 * sigma_index


def run_model_selection(
    components: DistanceComponents,
) -> tuple[pd.DataFrame, dict[float | None, np.ndarray]]:
    rows: list[dict[str, object]] = []
    distances: dict[float | None, np.ndarray] = {}

    for alpha in (None, BALANCED_ALPHA):
        name = configuration_name(alpha)
        print(f"\n{'=' * 72}\n{name.upper()} MODEL SELECTION\n{'=' * 72}")
        distance = build_gower_distance(components, alpha)
        distances[alpha] = distance
        print(f"Median positive dissimilarity: {median_positive_distance(distance):.6f}")

        grid_exact = GRID_SILHOUETTE_SAMPLE_SIZE is None
        for multiplier in SIGMA_MULTIPLIERS:
            seed = model_selection_seed(alpha, multiplier)
            basis = build_basis(distance, alpha, multiplier, seed)

            for k in K_VALUES:
                solution = fit_solution(
                    basis,
                    distance,
                    k=k,
                    seed=seed + k,
                    exact_silhouettes=grid_exact,
                )
                rows.append(solution_row("model_selection", solution, exact=grid_exact))
                print(
                    f"sigma={multiplier:>3} | k={k:>2} | "
                    f"sil_G={solution.gower_silhouette: .4f} | "
                    f"gap={solution.eigengap: .6f} | "
                    f"sizes={solution.cluster_sizes.tolist()}"
                )

            del basis
            gc.collect()

    return pd.DataFrame(rows), distances


def rank_candidates(group: pd.DataFrame) -> pd.DataFrame:
    return group.sort_values(
        by=[
            "valid_cluster_sizes",
            "gower_silhouette",
            "eigengap",
            "spectral_silhouette",
            "k",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def refit_from_row(
    distance: np.ndarray,
    alpha: float | None,
    row: pd.Series,
    seed: int,
) -> Solution:
    basis = build_basis(
        distance,
        alpha=alpha,
        sigma_multiplier=float(row["sigma_multiplier"]),
        seed=seed,
    )
    solution = fit_solution(
        basis,
        distance,
        k=int(row["k"]),
        seed=seed + int(row["k"]),
        exact_silhouettes=True,
    )
    del basis
    gc.collect()
    return solution


def fit_shared_resolutions(
    distance: np.ndarray,
    alpha: float,
    sigma_multiplier: float,
    k_values: Iterable[int],
    seed: int,
) -> dict[int, Solution]:
    basis = build_basis(distance, alpha, sigma_multiplier, seed)
    solutions = {
        k: fit_solution(
            basis,
            distance,
            k=k,
            seed=seed + k,
            exact_silhouettes=True,
        )
        for k in sorted(set(k_values))
    }
    del basis
    gc.collect()
    return solutions


# =============================================================================
# Comparisons and exports
# =============================================================================


def build_ari_matrix(solutions: dict[str, Solution]) -> pd.DataFrame:
    names = list(solutions)
    values = np.zeros((len(names), len(names)), dtype=float)
    for i, first in enumerate(names):
        for j, second in enumerate(names):
            values[i, j] = adjusted_rand_score(
                solutions[first].labels,
                solutions[second].labels,
            )
    return pd.DataFrame(values, index=names, columns=names)


def reorder_fine_clusters(coarse: np.ndarray, fine: np.ndarray) -> np.ndarray:
    table = pd.crosstab(coarse, fine)
    ordered: list[int] = []
    for coarse_cluster in table.index:
        for column in table.loc[coarse_cluster].sort_values(ascending=False).index:
            value = int(column)
            if value not in ordered:
                ordered.append(value)
    mapping = {old: new for new, old in enumerate(ordered)}
    return np.array([mapping[int(label)] for label in fine], dtype=int)


def build_nesting_tables(
    k2: Solution,
    k4: Solution,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    reordered = reorder_fine_clusters(k2.labels, k4.labels)
    counts = pd.crosstab(
        k2.labels,
        reordered,
        rownames=["k2_macrocluster"],
        colnames=["k4_subcluster"],
    )
    counts.index = [f"k2_cluster_{value}" for value in counts.index]
    counts.columns = [f"k4_cluster_{value}" for value in counts.columns]
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100.0
    return counts, percentages, reordered


def save_coordinates(solution: Solution, filename: str) -> None:
    columns: dict[str, np.ndarray] = {
        f"raw_eigenvector_{index + 1}": solution.raw_eigenvectors[:, index]
        for index in range(solution.k)
    }
    columns.update(
        {
            f"normalized_coordinate_{index + 1}": solution.embedding[:, index]
            for index in range(solution.k)
        }
    )
    columns["cluster"] = solution.labels
    pd.DataFrame(columns).to_csv(COORDINATE_DIR / filename, index=False)


def first_existing(paths: Sequence[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def load_reference_labels(
    name: str,
    candidates: Sequence[Path],
    expected_rows: int,
) -> ReferenceLabels | None:
    path = first_existing(candidates)
    if path is None:
        warnings.warn(
            f"Reference labels not found for {name}; related comparisons are skipped.",
            stacklevel=2,
        )
        return None

    frame = read_csv_clean(path)
    assert frame is not None
    frame = frame.reset_index(drop=True)
    if len(frame) != expected_rows:
        raise ValueError(
            f"{name} labels have {len(frame)} rows instead of {expected_rows}: {path}"
        )
    if "row_index" in frame.columns:
        expected = np.arange(expected_rows)
        if not np.array_equal(frame["row_index"].to_numpy(), expected):
            raise ValueError(f"row_index is not aligned in {path}")
    return ReferenceLabels(name=name, frame=frame, source_path=path)


def add_ari_comparison(
    rows: list[dict[str, object]],
    candidate_name: str,
    candidate_labels: np.ndarray,
    reference_name: str,
    reference_labels: np.ndarray,
) -> None:
    candidate_k = int(np.unique(candidate_labels).size)
    reference_k = int(np.unique(reference_labels).size)
    rows.append(
        {
            "candidate_partition": candidate_name,
            "reference_partition": reference_name,
            "candidate_k": candidate_k,
            "reference_k": reference_k,
            "same_resolution": candidate_k == reference_k,
            "adjusted_rand_index": float(
                adjusted_rand_score(candidate_labels, reference_labels)
            ),
        }
    )


def build_cross_representation_table(
    quantile_classical: Solution,
    quantile_balanced: Solution,
    resolutions: dict[int, Solution],
    references: Sequence[ReferenceLabels | None],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    add_ari_comparison(
        rows,
        "mixed_quantile_classical_selected",
        quantile_classical.labels,
        "mixed_quantile_balanced_selected",
        quantile_balanced.labels,
    )

    selected_columns = {
        "original_mixed": ("mixed_classical_selected", "mixed_balanced_selected"),
        "binary_only": ("binary_selected",),
        "numeric_baseline": ("numeric_selected",),
        "numeric_quantile": ("numeric_quantile_selected",),
    }
    matched_templates = {
        "original_mixed": "mixed_balanced_k{k}",
        "binary_only": "binary_k{k}",
        "numeric_baseline": "numeric_k{k}",
        "numeric_quantile": "numeric_quantile_k{k}",
    }

    for reference in references:
        if reference is None:
            continue
        for column in selected_columns.get(reference.name, ()):
            if column not in reference.frame.columns:
                warnings.warn(f"Column '{column}' absent in {reference.source_path}")
                continue
            labels = reference.frame[column].to_numpy(dtype=int)
            add_ari_comparison(
                rows,
                "mixed_quantile_classical_selected",
                quantile_classical.labels,
                column,
                labels,
            )
            add_ari_comparison(
                rows,
                "mixed_quantile_balanced_selected",
                quantile_balanced.labels,
                column,
                labels,
            )

        template = matched_templates.get(reference.name)
        if template is None:
            continue
        for k in (2, 3, 4):
            column = template.format(k=k)
            if column not in reference.frame.columns:
                warnings.warn(f"Column '{column}' absent in {reference.source_path}")
                continue
            add_ari_comparison(
                rows,
                f"mixed_quantile_balanced_k{k}",
                resolutions[k].labels,
                column,
                reference.frame[column].to_numpy(dtype=int),
            )

    return pd.DataFrame(rows)


def align_candidate_labels(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
) -> np.ndarray:
    reference_values = np.sort(np.unique(reference_labels))
    candidate_values = np.sort(np.unique(candidate_labels))
    contingency = pd.crosstab(reference_labels, candidate_labels).reindex(
        index=reference_values,
        columns=candidate_values,
        fill_value=0,
    )
    rows, columns = linear_sum_assignment(-contingency.to_numpy())
    mapping = {
        int(candidate_values[column]): int(reference_values[row])
        for row, column in zip(rows, columns)
    }
    next_label = int(reference_values.max()) + 1 if reference_values.size else 0
    for candidate in candidate_values:
        value = int(candidate)
        if value not in mapping:
            mapping[value] = next_label
            next_label += 1
    return np.array([mapping[int(label)] for label in candidate_labels], dtype=int)


def build_contingency(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = align_candidate_labels(reference_labels, candidate_labels)
    counts = pd.crosstab(
        reference_labels,
        aligned,
        rownames=["reference_cluster"],
        colnames=["mixed_quantile_cluster"],
    )
    counts.index = [f"reference_cluster_{value}" for value in counts.index]
    counts.columns = [f"mixed_quantile_cluster_{value}" for value in counts.columns]
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100.0
    return counts, percentages


# =============================================================================
# Figures
# =============================================================================


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def save_model_selection_figure(
    results: pd.DataFrame,
    configuration: str,
    filename: str,
) -> None:
    subset = results[results["configuration"] == configuration]
    metrics = (
        ("gower_silhouette", "Gower silhouette"),
        ("eigengap", "Eigengap"),
        ("spectral_silhouette", "Spectral silhouette"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2))

    for ax, (metric, label) in zip(axes, metrics):
        for multiplier in SIGMA_MULTIPLIERS:
            curve = subset[subset["sigma_multiplier"] == multiplier].sort_values("k")
            ax.plot(
                curve["k"],
                curve[metric],
                marker="o",
                linewidth=1.5,
                label=f"sigma = {multiplier} x median",
            )
        ax.set_xticks(K_VALUES)
        ax.set_xlabel("Number of clusters k")
        ax.set_ylabel(label)
        ax.set_title(label)

    axes[0].legend()
    fig.suptitle(f"Model-selection diagnostics: {configuration}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURE_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_embedding_figure(solution: Solution, filename: str) -> None:
    if solution.k < 2:
        raise ValueError("At least two coordinates are required for this figure.")

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))
    axes[0].scatter(
        solution.raw_eigenvectors[:, 0],
        solution.raw_eigenvectors[:, 1],
        c=solution.labels,
        s=10,
        alpha=0.65,
    )
    axes[0].set_title("Before NJW row normalization")
    axes[0].set_xlabel("Eigenvector 1")
    axes[0].set_ylabel("Eigenvector 2")

    axes[1].scatter(
        solution.embedding[:, 0],
        solution.embedding[:, 1],
        c=solution.labels,
        s=10,
        alpha=0.65,
    )
    axes[1].set_title("Final row-normalized NJW embedding")
    axes[1].set_xlabel("Normalized coordinate 1")
    axes[1].set_ylabel("Normalized coordinate 2")

    fig.suptitle(
        f"{solution.configuration} | k={solution.k} | "
        f"sigma={solution.sigma_multiplier} x median\n"
        f"Gower silhouette={solution.gower_silhouette:.4f}; "
        f"spectral silhouette={solution.spectral_silhouette:.4f}; "
        f"eigengap={solution.eigengap:.5f}; "
        f"sizes={solution.cluster_sizes.tolist()}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(FIGURE_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_k3_embedding(solution: Solution, filename: str) -> None:
    if solution.k != 3:
        raise ValueError("save_k3_embedding requires a k=3 solution.")
    fig = plt.figure(figsize=(9.0, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        solution.embedding[:, 0],
        solution.embedding[:, 1],
        solution.embedding[:, 2],
        c=solution.labels,
        s=10,
        alpha=0.65,
    )
    ax.set_xlabel("Coordinate 1")
    ax.set_ylabel("Coordinate 2")
    ax.set_zlabel("Coordinate 3")
    ax.set_title(f"Three-dimensional NJW embedding | {solution.configuration} | k=3")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_k4_pairwise_projections(solution: Solution, filename: str) -> None:
    if solution.k != 4:
        raise ValueError("save_k4_pairwise_projections requires a k=4 solution.")
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0))

    for ax, (first, second) in zip(axes.ravel(), pairs):
        ax.scatter(
            solution.embedding[:, first],
            solution.embedding[:, second],
            c=solution.labels,
            s=8,
            alpha=0.58,
        )
        ax.set_xlabel(f"Coordinate {first + 1}")
        ax.set_ylabel(f"Coordinate {second + 1}")
        ax.set_title(f"Coordinates {first + 1} and {second + 1}")

    fig.suptitle(
        "All pairwise views of the four-dimensional NJW embedding\n"
        f"{solution.configuration} | "
        f"sigma={solution.sigma_multiplier} x median | k=4",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_ari_heatmap(ari: pd.DataFrame, filename: str) -> None:
    values = ari.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(values, vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(len(ari.columns)), ari.columns)
    ax.set_yticks(range(len(ari.index)), ari.index)
    ax.set_title("Stability across block weights")

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center")

    fig.colorbar(image, ax=ax, label="Adjusted Rand Index")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_nesting_heatmap(
    counts: pd.DataFrame,
    percentages: pd.DataFrame,
    filename: str,
) -> None:
    values = percentages.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    image = ax.imshow(values, vmin=0.0, vmax=100.0, aspect="auto")
    ax.set_xticks(range(len(percentages.columns)), percentages.columns)
    ax.set_yticks(range(len(percentages.index)), percentages.index)
    ax.set_xlabel("Balanced k=4 subcluster")
    ax.set_ylabel("Balanced k=2 macrocluster")
    ax.set_title("k=4 refinement within the k=2 partition")

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{int(counts.iloc[row, column])}\n({values[row, column]:.1f}%)",
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax, label="Percentage within k=2 cluster")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_eigenvalue_spectra(
    classical: Solution,
    balanced: Solution,
    filename: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    for ax, solution, title in (
        (axes[0], classical, "Quantile-classical Gower"),
        (axes[1], balanced, "Quantile-balanced Gower"),
    ):
        values = solution.eigenvalues[: max(K_VALUES) + 1]
        indices = np.arange(1, len(values) + 1)
        ax.plot(indices, values, marker="o")
        ax.axvline(solution.k + 0.5, linestyle="--", linewidth=1.2)
        ax.set_xticks(indices)
        ax.set_xlabel("Ordered eigenvalue index")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(f"{title}\nselected k={solution.k}, gap={solution.eigengap:.5f}")

    fig.suptitle("Normalized-affinity eigenvalue spectra", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURE_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_cross_ari_figure(comparisons: pd.DataFrame, filename: str) -> None:
    if comparisons.empty:
        return
    ordered = comparisons.sort_values("adjusted_rand_index", ascending=True)
    height = max(6.0, 0.42 * len(ordered) + 2.0)
    fig, ax = plt.subplots(figsize=(12.0, height))
    labels = (
        ordered["candidate_partition"] + " vs " + ordered["reference_partition"]
    )
    ax.barh(labels, ordered["adjusted_rand_index"])
    lower = min(-0.05, float(ordered["adjusted_rand_index"].min()) - 0.03)
    ax.set_xlim(lower, 1.0)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_xlabel("Adjusted Rand Index")
    ax.set_title("Agreement with existing project partitions")
    for index, value in enumerate(ordered["adjusted_rand_index"]):
        x = float(value) + (0.012 if value >= 0 else -0.012)
        ax.text(x, index, f"{value:.3f}", va="center", ha="left" if value >= 0 else "right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_contingency_heatmap(
    counts: pd.DataFrame,
    percentages: pd.DataFrame,
    ari: float,
    k: int,
    reference_name: str,
    filename: str,
) -> None:
    values = percentages.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    image = ax.imshow(values, vmin=0.0, vmax=100.0, aspect="equal")
    ax.set_xticks(range(len(counts.columns)), counts.columns)
    ax.set_yticks(range(len(counts.index)), counts.index)
    ax.set_xlabel("Mixed-quantile cluster after label alignment")
    ax.set_ylabel(f"{reference_name} cluster")
    ax.set_title(f"Mixed quantile vs {reference_name} | k={k} | ARI={ari:.4f}")

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{int(counts.iloc[row, column])}\n({values[row, column]:.1f}%)",
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax, label=f"Percentage within {reference_name} cluster")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def export_matched_contingencies(
    resolutions: dict[int, Solution],
    references: Sequence[ReferenceLabels | None],
) -> list[Path]:
    templates = {
        "original_mixed": "mixed_balanced_k{k}",
        "binary_only": "binary_k{k}",
        "numeric_baseline": "numeric_k{k}",
        "numeric_quantile": "numeric_quantile_k{k}",
    }
    generated: list[Path] = []

    for reference in references:
        if reference is None or reference.name not in templates:
            continue
        for k in (2, 3, 4):
            column = templates[reference.name].format(k=k)
            if column not in reference.frame.columns:
                continue
            reference_labels = reference.frame[column].to_numpy(dtype=int)
            candidate_labels = resolutions[k].labels
            counts, percentages = build_contingency(reference_labels, candidate_labels)
            stem = f"contingency_{reference.name}_k{k}"
            counts_path = TABLE_DIR / f"{stem}_counts.csv"
            percentages_path = TABLE_DIR / f"{stem}_row_percentages.csv"
            figure_path = FIGURE_DIR / f"{stem}.png"
            counts.to_csv(counts_path)
            percentages.to_csv(percentages_path)
            save_contingency_heatmap(
                counts,
                percentages,
                ari=float(adjusted_rand_score(reference_labels, candidate_labels)),
                k=k,
                reference_name=reference.name.replace("_", " "),
                filename=figure_path.name,
            )
            generated.extend((counts_path, percentages_path, figure_path))

    return generated


# =============================================================================
# Result summary and validation
# =============================================================================


def json_value(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_results_summary(
    summary: dict[str, object],
    classical: Solution,
    balanced: Solution,
    sensitivity_ari: pd.DataFrame,
    k2: Solution,
    k3: Solution,
    k4: Solution,
    comparisons: pd.DataFrame,
    references: Sequence[ReferenceLabels | None],
) -> None:
    off_diagonal = sensitivity_ari.to_numpy()[
        np.triu_indices(len(sensitivity_ari), k=1)
    ]
    lines = [
        "# Mixed-data quantile sensitivity analysis",
        "",
        "## Input and representation",
        "",
        f"- Observations: {summary['n_objects']}",
        f"- Numerical descriptors: {summary['numeric_features']}",
        f"- Binary descriptors: {summary['binary_features']}",
        "- Numerical transformation: feature-wise empirical quantile map to Uniform[0,1]",
        "- Binary treatment: asymmetric Jaccard; joint zeros ignored",
        "- Activity: excluded from all unsupervised operations and appended only to final labels",
        "",
        "## Selected solutions",
        "",
        f"- Quantile-classical: k={classical.k}, sigma multiplier={classical.sigma_multiplier}, "
        f"sizes={classical.cluster_sizes.tolist()}, Gower silhouette={classical.gower_silhouette:.4f}, "
        f"spectral silhouette={classical.spectral_silhouette:.4f}, eigengap={classical.eigengap:.6f}",
        f"- Quantile-balanced: k={balanced.k}, sigma multiplier={balanced.sigma_multiplier}, "
        f"sizes={balanced.cluster_sizes.tolist()}, Gower silhouette={balanced.gower_silhouette:.4f}, "
        f"spectral silhouette={balanced.spectral_silhouette:.4f}, eigengap={balanced.eigengap:.6f}",
        f"- Weight sensitivity: minimum ARI={float(off_diagonal.min()):.4f}, "
        f"mean ARI={float(off_diagonal.mean()):.4f}",
        "",
        "## Balanced resolution checks",
        "",
        f"- k=2: sizes={k2.cluster_sizes.tolist()}, silhouette={k2.gower_silhouette:.4f}",
        f"- k=3: sizes={k3.cluster_sizes.tolist()}, silhouette={k3.gower_silhouette:.4f}",
        f"- k=4: sizes={k4.cluster_sizes.tolist()}, silhouette={k4.gower_silhouette:.4f}",
        "",
        "## Reference files",
        "",
    ]
    used = [reference for reference in references if reference is not None]
    lines.extend(
        [f"- {reference.name}: `{reference.source_path}`" for reference in used]
        or ["- No optional reference label files were available."]
    )
    lines.extend(["", "## Cross-representation ARI", ""])
    if comparisons.empty:
        lines.append("No cross-representation comparisons were available.")
    else:
        for row in comparisons.sort_values("adjusted_rand_index", ascending=False).itertuples():
            lines.append(
                f"- {row.candidate_partition} vs {row.reference_partition}: "
                f"ARI={row.adjusted_rand_index:.4f}"
            )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "Silhouette values are compared only within the same transformed geometry. "
            "Cross-representation conclusions use matched-resolution ARI and contingency tables. "
            "This run remains a sensitivity analysis because quantile transformation changes "
            "the numerical distance geometry.",
        ]
    )
    RESULTS_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def verify_outputs(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError("Expected outputs were not created:\n" + "\n".join(missing))


# =============================================================================
# Main pipeline
# =============================================================================


def main() -> None:
    for directory in (REPORT_DIR, TABLE_DIR, FIGURE_DIR, COORDINATE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    configure_plots()

    print(f"Project dir: {PROJECT_DIR}")
    print(f"Input features: {FEATURES_PATH}")
    print(f"Output report: {REPORT_DIR}")

    data = load_project_data()
    schema = fit_schema(data.X)
    summary = preprocessing_summary(data, schema)
    pd.DataFrame([summary]).to_csv(PREPROCESSING_CSV, index=False)

    print(
        f"Objects: {summary['n_objects']} | retained features: "
        f"{summary['numeric_features']} numerical + {summary['binary_features']} binary"
    )
    print(f"'{TARGET_COL}' is excluded from all unsupervised operations.")

    X_numeric, X_binary = transform_features(data.X, schema)
    components = compute_distance_components(X_numeric, X_binary)

    # Select parameters only within the two geometries used in the report.
    grid, distances = run_model_selection(components)
    grid.to_csv(GRID_RESULTS_CSV, index=False)

    classical_name = configuration_name(None)
    balanced_name = configuration_name(BALANCED_ALPHA)
    classical_row = rank_candidates(grid[grid["configuration"] == classical_name]).iloc[0]
    balanced_row = rank_candidates(grid[grid["configuration"] == balanced_name]).iloc[0]

    classical = refit_from_row(
        distances[None],
        alpha=None,
        row=classical_row,
        seed=model_selection_seed(None, float(classical_row["sigma_multiplier"])),
    )

    balanced_resolutions = fit_shared_resolutions(
        distances[BALANCED_ALPHA],
        alpha=BALANCED_ALPHA,
        sigma_multiplier=float(balanced_row["sigma_multiplier"]),
        k_values={int(balanced_row["k"]), 2, 3, 4},
        seed=model_selection_seed(
            BALANCED_ALPHA, float(balanced_row["sigma_multiplier"])
        ),
    )
    balanced = balanced_resolutions[int(balanced_row["k"])]
    k2, k3, k4 = (
        balanced_resolutions[2],
        balanced_resolutions[3],
        balanced_resolutions[4],
    )

    pd.DataFrame(
        [
            solution_row("quantile_classical_selected", classical, exact=True),
            solution_row("quantile_balanced_selected", balanced, exact=True),
            solution_row("quantile_balanced_k2", k2, exact=True),
            solution_row("quantile_balanced_k3", k3, exact=True),
            solution_row("quantile_balanced_k4", k4, exact=True),
        ]
    ).to_csv(SELECTED_SOLUTIONS_CSV, index=False)

    # Controlled sensitivity: only alpha changes; k, normalized bandwidth and seeds stay fixed.
    sensitivity: dict[str, Solution] = {}
    sensitivity_seed = RANDOM_STATE + 40_000
    for alpha in SENSITIVITY_ALPHAS:
        distance = (
            distances[BALANCED_ALPHA]
            if alpha == BALANCED_ALPHA
            else build_gower_distance(components, alpha)
        )
        basis = build_basis(
            distance,
            alpha=alpha,
            sigma_multiplier=balanced.sigma_multiplier,
            seed=sensitivity_seed,
        )
        sensitivity[f"alpha={alpha:.1f}"] = fit_solution(
            basis,
            distance,
            k=balanced.k,
            seed=sensitivity_seed + balanced.k,
            exact_silhouettes=True,
        )
        del basis
        if alpha != BALANCED_ALPHA:
            del distance
        gc.collect()

    pd.DataFrame(
        [solution_row("weight_sensitivity", solution, exact=True) for solution in sensitivity.values()]
    ).to_csv(SENSITIVITY_CSV, index=False)
    sensitivity_ari = build_ari_matrix(sensitivity)
    sensitivity_ari.to_csv(SENSITIVITY_ARI_CSV)

    nesting_counts, nesting_percentages, reordered_k4 = build_nesting_tables(k2, k4)
    nesting_counts.to_csv(NESTING_COUNTS_CSV)
    nesting_percentages.to_csv(NESTING_PERCENTAGES_CSV)

    labels = pd.DataFrame(
        {
            "row_index": np.arange(len(data.X)),
            "mixed_quantile_classical_selected": classical.labels,
            "mixed_quantile_balanced_selected": balanced.labels,
            "mixed_quantile_balanced_k2": k2.labels,
            "mixed_quantile_balanced_k3": k3.labels,
            "mixed_quantile_balanced_k4": reordered_k4,
        }
    )
    if data.activity is not None:
        labels[TARGET_COL] = data.activity.to_numpy()
    labels.to_csv(LABELS_CSV, index=False)

    original_mixed = load_reference_labels(
        "original_mixed", (ORIGINAL_MIXED_LABELS,), len(data.X)
    )
    binary = load_reference_labels(
        "binary_only", BINARY_REFERENCE_CANDIDATES, len(data.X)
    )
    numeric = load_reference_labels(
        "numeric_baseline", NUMERIC_REFERENCE_CANDIDATES, len(data.X)
    )
    numeric_quantile = load_reference_labels(
        "numeric_quantile", NUMERIC_QUANTILE_REFERENCE_CANDIDATES, len(data.X)
    )
    references = (original_mixed, binary, numeric, numeric_quantile)

    comparisons = build_cross_representation_table(
        classical,
        balanced,
        {2: k2, 3: k3, 4: k4},
        references,
    )
    comparisons.to_csv(CROSS_REPRESENTATION_CSV, index=False)
    optional_outputs = export_matched_contingencies(
        {2: k2, 3: k3, 4: k4}, references
    )

    save_coordinates(classical, "mixed_quantile_classical_selected_coordinates.csv")
    save_coordinates(balanced, "mixed_quantile_balanced_selected_coordinates.csv")
    save_coordinates(k2, "mixed_quantile_balanced_k2_coordinates.csv")
    save_coordinates(k3, "mixed_quantile_balanced_k3_coordinates.csv")
    save_coordinates(k4, "mixed_quantile_balanced_k4_coordinates.csv")

    save_model_selection_figure(
        grid, classical_name, "01_quantile_classical_model_selection.png"
    )
    save_embedding_figure(
        classical, "02_quantile_classical_selected_embedding.png"
    )
    save_model_selection_figure(
        grid, balanced_name, "03_quantile_balanced_model_selection.png"
    )
    save_ari_heatmap(
        sensitivity_ari, "04_quantile_weight_sensitivity_ari.png"
    )
    save_embedding_figure(
        balanced, "05_quantile_balanced_selected_embedding.png"
    )
    save_embedding_figure(k2, "06_quantile_balanced_k2_embedding.png")
    save_embedding_figure(k3, "07_quantile_balanced_k3_embedding_2d.png")
    save_k3_embedding(k3, "08_quantile_balanced_k3_embedding_3d.png")
    save_embedding_figure(k4, "09_quantile_balanced_k4_embedding_2d.png")
    save_k4_pairwise_projections(k4, "10_quantile_balanced_k4_all_pairs.png")
    save_eigenvalue_spectra(
        classical, balanced, "11_quantile_eigenvalue_spectra.png"
    )
    save_nesting_heatmap(
        nesting_counts,
        nesting_percentages,
        "12_quantile_balanced_k2_vs_k4_nesting.png",
    )
    save_cross_ari_figure(
        comparisons, "13_mixed_quantile_cross_representation_ari.png"
    )

    off_diagonal = sensitivity_ari.to_numpy()[
        np.triu_indices(len(sensitivity_ari), k=1)
    ]
    report_values = {
        "preprocessing": summary,
        "quantile_classical_selected": solution_row(
            "quantile_classical_selected", classical, exact=True
        ),
        "quantile_balanced_selected": solution_row(
            "quantile_balanced_selected", balanced, exact=True
        ),
        "quantile_balanced_k2": solution_row("quantile_balanced_k2", k2, exact=True),
        "quantile_balanced_k3": solution_row("quantile_balanced_k3", k3, exact=True),
        "quantile_balanced_k4": solution_row("quantile_balanced_k4", k4, exact=True),
        "minimum_ari_across_alphas": float(off_diagonal.min()),
        "mean_ari_across_alphas": float(off_diagonal.mean()),
        "ari_quantile_classical_vs_quantile_balanced": float(
            adjusted_rand_score(classical.labels, balanced.labels)
        ),
        "reference_files": {
            reference.name: str(reference.source_path)
            for reference in references
            if reference is not None
        },
        "cross_representation_comparisons": comparisons.to_dict(orient="records"),
        "interpretation_rule": (
            "Silhouettes are compared only within the same transformed geometry; "
            "cross-representation comparisons use matched-resolution ARI and "
            "contingency tables. Quantile transformation remains a sensitivity analysis."
        ),
    }
    with REPORT_VALUES_JSON.open("w", encoding="utf-8") as handle:
        json.dump(report_values, handle, indent=2, default=json_value)

    write_results_summary(
        summary,
        classical,
        balanced,
        sensitivity_ari,
        k2,
        k3,
        k4,
        comparisons,
        references,
    )

    core_figures = tuple(
        FIGURE_DIR / name
        for name in (
            "01_quantile_classical_model_selection.png",
            "02_quantile_classical_selected_embedding.png",
            "03_quantile_balanced_model_selection.png",
            "04_quantile_weight_sensitivity_ari.png",
            "05_quantile_balanced_selected_embedding.png",
            "06_quantile_balanced_k2_embedding.png",
            "07_quantile_balanced_k3_embedding_2d.png",
            "08_quantile_balanced_k3_embedding_3d.png",
            "09_quantile_balanced_k4_embedding_2d.png",
            "10_quantile_balanced_k4_all_pairs.png",
            "11_quantile_eigenvalue_spectra.png",
            "12_quantile_balanced_k2_vs_k4_nesting.png",
            "13_mixed_quantile_cross_representation_ari.png",
        )
    )
    mandatory_outputs = (
        PREPROCESSING_CSV,
        GRID_RESULTS_CSV,
        SELECTED_SOLUTIONS_CSV,
        SENSITIVITY_CSV,
        SENSITIVITY_ARI_CSV,
        NESTING_COUNTS_CSV,
        NESTING_PERCENTAGES_CSV,
        LABELS_CSV,
        REPORT_VALUES_JSON,
        CROSS_REPRESENTATION_CSV,
        RESULTS_SUMMARY_MD,
        *core_figures,
    )
    verify_outputs((*mandatory_outputs, *optional_outputs))

    print("\nMixed quantile sensitivity analysis completed.")
    print(f"Report directory: {REPORT_DIR}")
    print(f"Labels: {LABELS_CSV}")
    print(f"Cross-representation ARI: {CROSS_REPRESENTATION_CSV}")


if __name__ == "__main__":
    main()
