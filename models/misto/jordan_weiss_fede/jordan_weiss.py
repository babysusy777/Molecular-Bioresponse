from __future__ import annotations

"""Mixed-data baseline: Gower dissimilarity + Ng-Jordan-Weiss clustering.

Scientific narrative implemented by this script
================================================
A. Establish a mixed-data baseline with the standard, feature-wise Gower
   dissimilarity. Binary descriptors are treated as asymmetric, so joint zeros
   do not contribute to similarity.
B. Explore k = 2,...,10 and sigma in {0.5, 1, 2} times the median positive
   dissimilarity. Select the representative solution *within* the classical
   Gower geometry using Gower silhouette, eigengap, spectral silhouette, and a
   minimum cluster-size constraint.
C. Ask whether the result depends on the implicit relative contribution of the
   numerical and binary blocks in classical Gower. Perform a controlled
   sensitivity analysis with block weights alpha in {0.4, 0.5, 0.6}, where

       d_weighted = alpha*d_numeric + (1-alpha)*d_binary.

   The asymmetric definition of the binary component is unchanged.
D. Use alpha = 0.5 as the neutral representative of the weighted family, not
   because it must "beat" classical Gower, but because it gives the two blocks
   equal explicit contribution. Compare alpha values through ARI at the same k
   and sigma multiplier.
E. Investigate whether k = 4 is a nested refinement of k = 2 under the balanced
   formulation, rather than treating every k as a disconnected result.
F. Export mixed-data labels and metrics as a baseline for the later numerical-
   only and binary-only analyses.

Input policy
============
The script first looks for ``train_filtered_no_activity.csv`` and uses it
without repeating quasi-constant filtering. This keeps the analysis aligned
with the dataset already used for PCA and the other project methods. Only when
that file is unavailable does it fall back to ``train.csv`` and reproduce the
99% dominant-value filtering internally.

The Activity target is never used to construct distances, choose parameters,
or obtain clusters. When available, it is appended only to the final exported
labels for post-hoc interpretation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import gc
import json
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

RANDOM_STATE = 42
TARGET_COL = "Activity"
QUASI_CONSTANT_THRESHOLD = 0.99

K_VALUES = tuple(range(2, 11))
SIGMA_MULTIPLIERS = (0.5, 1.0, 2.0)
WEIGHTED_ALPHAS = (0.4, 0.5, 0.6)
BALANCED_ALPHA = 0.5

KMEANS_N_INIT = 50
MIN_CLUSTER_FRACTION = 0.05
DISTANCE_DTYPE = np.float32
EIGEN_TOLERANCE = 1e-6
EIGEN_MAX_ITERATIONS = 5000
FIGURE_DPI = 220

# Keep None for exact silhouettes. A positive integer can be used only if the
# complete grid is too slow; final representative silhouettes remain exact.
GRID_SILHOUETTE_SAMPLE_SIZE: int | None = None

# Set this to a concrete path only when automatic discovery is not suitable.
FORCE_INPUT_PATH: Path | None = None

# The script is designed to be placed anywhere inside the project tree. It
# searches upward for a directory containing "Dataset".
SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# 2. OUTPUT STRUCTURE
# =============================================================================


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "Dataset").exists():
            return candidate
    return start


PROJECT_DIR = find_project_root(SCRIPT_DIR)
REPORT_DIR = PROJECT_DIR / "reports" / "njw_mixed_baseline"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"
COORDINATE_DIR = REPORT_DIR / "coordinates"

PREPROCESSING_CSV = TABLE_DIR / "01_preprocessing_summary.csv"
ALL_RESULTS_CSV = TABLE_DIR / "02_all_grid_results.csv"
CLASSICAL_RANKING_CSV = TABLE_DIR / "03_classical_candidate_ranking.csv"
WEIGHTED_RANKING_CSV = TABLE_DIR / "04_balanced_candidate_ranking.csv"
NARRATIVE_SUMMARY_CSV = TABLE_DIR / "05_narrative_summary.csv"
WEIGHT_SENSITIVITY_CSV = TABLE_DIR / "06_weight_sensitivity_metrics.csv"
WEIGHT_ARI_CSV = TABLE_DIR / "07_weight_sensitivity_ari.csv"
NESTING_COUNTS_CSV = TABLE_DIR / "08_balanced_k2_vs_k4_counts.csv"
NESTING_ROWS_CSV = TABLE_DIR / "09_balanced_k2_vs_k4_row_percentages.csv"
BASELINE_LABELS_CSV = TABLE_DIR / "10_mixed_baseline_labels.csv"
REPORT_VALUES_JSON = TABLE_DIR / "11_report_values.json"
AUTO_INTERPRETATION_MD = REPORT_DIR / "automatic_interpretation.md"


# =============================================================================
# 3. DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class LoadedData:
    X: pd.DataFrame
    activity: pd.Series | None
    source_path: Path
    source_kind: str  # "filtered" or "raw"


@dataclass(frozen=True)
class FeatureSchema:
    retained: list[str]
    numeric: list[str]
    binary: list[str]
    minima: pd.Series
    ranges: pd.Series
    preprocessing_summary: dict[str, object]


@dataclass
class DistanceComponents:
    numerical: np.ndarray
    binary: np.ndarray
    binary_union: np.ndarray
    n_numeric: int


@dataclass
class SpectralBasis:
    gower_name: str
    alpha: float | None
    sigma_reference: float
    sigma_multiplier: float
    sigma: float
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


@dataclass
class FittedSolution:
    gower_name: str
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


# =============================================================================
# 4. INPUT DISCOVERY AND PREPROCESSING
# =============================================================================


def candidate_filtered_paths(root: Path) -> list[Path]:
    return [
        root / "Dataset" / "processed" / "train_filtered_no_activity.csv",
        root / "Dataset" / "preprocessed" / "train_filtered_no_activity.csv",
        root / "Dataset" / "raw" / "train_filtered_no_activity.csv",
        root / "Dataset" / "train_filtered_no_activity.csv",
        root / "train_filtered_no_activity.csv",
        SCRIPT_DIR / "train_filtered_no_activity.csv",
    ]


def candidate_raw_paths(root: Path) -> list[Path]:
    return [
        root / "Dataset" / "raw" / "train.csv",
        root / "Dataset" / "train.csv",
        root / "train.csv",
        SCRIPT_DIR / "train.csv",
    ]


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def read_csv_clean(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    unnamed = [column for column in frame.columns if str(column).startswith("Unnamed:")]
    if unnamed:
        frame = frame.drop(columns=unnamed)
    return frame


def load_activity_if_available(root: Path, expected_rows: int) -> pd.Series | None:
    raw_path = first_existing(candidate_raw_paths(root))
    if raw_path is None:
        return None
    raw = read_csv_clean(raw_path)
    if TARGET_COL not in raw.columns or len(raw) != expected_rows:
        return None
    return raw[TARGET_COL].reset_index(drop=True)


def validate_feature_frame(X: pd.DataFrame) -> None:
    if X.empty:
        raise ValueError("The feature matrix is empty.")

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise TypeError(f"Non-numeric descriptors found: {non_numeric[:10]}")

    if X.isna().any().any():
        missing = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Missing values found in descriptors: {missing[:10]}")

    duplicate_rows = int(X.duplicated().sum())
    if duplicate_rows:
        warnings.warn(
            f"{duplicate_rows} duplicate feature rows were found. They are retained "
            "to preserve alignment with the original dataset.",
            stacklevel=2,
        )


def load_project_data() -> LoadedData:
    if FORCE_INPUT_PATH is not None:
        input_path = Path(FORCE_INPUT_PATH)
        if not input_path.exists():
            raise FileNotFoundError(f"FORCE_INPUT_PATH does not exist: {input_path}")
        data = read_csv_clean(input_path)
        source_kind = "filtered" if "filtered_no_activity" in input_path.stem else "raw"
    else:
        filtered_path = first_existing(candidate_filtered_paths(PROJECT_DIR))
        if filtered_path is not None:
            input_path = filtered_path
            source_kind = "filtered"
            data = read_csv_clean(input_path)
        else:
            raw_path = first_existing(candidate_raw_paths(PROJECT_DIR))
            if raw_path is None:
                searched = candidate_filtered_paths(PROJECT_DIR) + candidate_raw_paths(PROJECT_DIR)
                raise FileNotFoundError(
                    "No input dataset was found. Searched:\n" +
                    "\n".join(f"  - {path}" for path in searched)
                )
            input_path = raw_path
            source_kind = "raw"
            data = read_csv_clean(input_path)

    activity: pd.Series | None
    if TARGET_COL in data.columns:
        activity = data[TARGET_COL].reset_index(drop=True)
        X = data.drop(columns=TARGET_COL).copy()
    else:
        X = data.copy()
        activity = load_activity_if_available(PROJECT_DIR, len(X))

    X = X.reset_index(drop=True)
    validate_feature_frame(X)
    return LoadedData(X=X, activity=activity, source_path=input_path, source_kind=source_kind)


def dominant_ratio(series: pd.Series) -> float:
    return float(series.value_counts(normalize=True, dropna=False).iloc[0])


def identify_binary_columns(X: pd.DataFrame) -> list[str]:
    allowed = {0, 1, 0.0, 1.0, False, True}
    return [
        column
        for column in X.columns
        if set(X[column].dropna().unique().tolist()).issubset(allowed)
    ]


def fit_feature_schema(data: LoadedData) -> FeatureSchema:
    X = data.X
    original_features = X.shape[1]

    if data.source_kind == "filtered":
        quasi_constant_removed: list[str] = []
        retained = list(X.columns)
    else:
        dominant = X.apply(dominant_ratio, axis=0)
        quasi_constant_removed = dominant[
            dominant >= QUASI_CONSTANT_THRESHOLD
        ].index.tolist()
        retained = [column for column in X.columns if column not in quasi_constant_removed]

    X_retained = X.loc[:, retained]
    binary = identify_binary_columns(X_retained)
    binary_set = set(binary)
    numeric = [column for column in retained if column not in binary_set]

    if not binary:
        raise ValueError("No binary descriptors were identified after preprocessing.")
    if not numeric:
        raise ValueError("No non-binary numerical descriptors were identified after preprocessing.")

    minima = X_retained[numeric].min(axis=0)
    ranges = X_retained[numeric].max(axis=0) - minima
    zero_range_numeric = ranges[ranges <= 0].index.tolist()

    if zero_range_numeric:
        zero_range_set = set(zero_range_numeric)
        numeric = [column for column in numeric if column not in zero_range_set]
        retained = [column for column in retained if column not in zero_range_set]
        minima = X_retained[numeric].min(axis=0)
        ranges = X_retained[numeric].max(axis=0) - minima

    remaining = len(retained)
    n_numeric = len(numeric)
    n_binary = len(binary)

    summary = {
        "input_path": str(data.source_path),
        "input_kind": data.source_kind,
        "n_objects": len(X),
        "features_in_input_file": original_features,
        "quasi_constant_threshold": QUASI_CONSTANT_THRESHOLD,
        "quasi_constant_removed_in_this_script": len(quasi_constant_removed),
        "zero_range_numeric_removed_in_this_script": len(zero_range_numeric),
        "remaining_features": remaining,
        "remaining_numeric_features": n_numeric,
        "remaining_binary_features": n_binary,
        "numeric_percentage": 100.0 * n_numeric / remaining,
        "binary_percentage": 100.0 * n_binary / remaining,
        "activity_available_for_post_hoc_only": data.activity is not None,
    }

    return FeatureSchema(
        retained=retained,
        numeric=numeric,
        binary=binary,
        minima=minima,
        ranges=ranges,
        preprocessing_summary=summary,
    )


def transform_features(
    X: pd.DataFrame,
    schema: FeatureSchema,
) -> tuple[np.ndarray, np.ndarray]:
    X_numeric = (
        (X[schema.numeric] - schema.minima) / schema.ranges
    ).clip(0.0, 1.0)
    X_binary = X[schema.binary]

    numeric_array = X_numeric.to_numpy(dtype=DISTANCE_DTYPE, copy=True)
    binary_array = X_binary.to_numpy(dtype=DISTANCE_DTYPE, copy=True)

    if not np.isfinite(numeric_array).all() or not np.isfinite(binary_array).all():
        raise ValueError("Non-finite values were produced during feature transformation.")

    return numeric_array, binary_array


# =============================================================================
# 5. GOWER COMPONENTS
# =============================================================================


def compute_numerical_dissimilarity(X_numeric: np.ndarray) -> np.ndarray:
    """Average range-normalized Manhattan distance over numerical features."""
    distances = pairwise_distances(X_numeric, metric="manhattan", n_jobs=-1)
    distances = distances.astype(DISTANCE_DTYPE, copy=False)
    distances /= float(X_numeric.shape[1])
    np.fill_diagonal(distances, 0.0)
    return distances


def compute_asymmetric_binary_dissimilarity(
    X_binary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Jaccard dissimilarity and pairwise union counts.

    For a pair of observations, 1-1 contributes a match, 1-0 and 0-1 are
    mismatches, and 0-0 is excluded. When both rows contain no active binary
    descriptor, the binary block is unavailable for that pair.
    """
    intersection = X_binary @ X_binary.T
    ones = X_binary.sum(axis=1, dtype=np.float64)
    union = (
        ones[:, None] + ones[None, :] - intersection
    ).astype(DISTANCE_DTYPE, copy=False)
    mismatches = union - intersection

    dissimilarity = np.divide(
        mismatches,
        union,
        out=np.zeros_like(union, dtype=DISTANCE_DTYPE),
        where=union > 0,
    )
    np.fill_diagonal(dissimilarity, 0.0)
    return dissimilarity, union


def compute_distance_components(
    X_numeric: np.ndarray,
    X_binary: np.ndarray,
) -> DistanceComponents:
    print("Computing numerical Gower component...")
    d_numeric = compute_numerical_dissimilarity(X_numeric)
    print("Computing asymmetric-binary Gower component...")
    d_binary, binary_union = compute_asymmetric_binary_dissimilarity(X_binary)
    return DistanceComponents(
        numerical=d_numeric,
        binary=d_binary,
        binary_union=binary_union,
        n_numeric=X_numeric.shape[1],
    )


def build_gower_distance(
    components: DistanceComponents,
    alpha: float | None,
) -> np.ndarray:
    """Build classical or block-weighted mixed dissimilarity.

    Classical feature-wise Gower:
        every numerical descriptor contributes one valid comparison;
        each asymmetric binary descriptor contributes only when at least one
        member of the pair has value 1.

    Block-weighted Gower:
        alpha controls the numerical block and (1-alpha) the binary block.
        If the binary block is unavailable for a pair, the numerical component
        receives all available weight through denominator renormalization.
    """
    if alpha is None:
        numerator = (
            components.numerical * components.n_numeric
            + components.binary * components.binary_union
        )
        denominator = components.n_numeric + components.binary_union
    else:
        alpha = float(alpha)
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
    distance = np.clip(distance, 0.0, 1.0).astype(DISTANCE_DTYPE, copy=False)
    np.fill_diagonal(distance, 0.0)
    return distance


def gower_name(alpha: float | None) -> str:
    if alpha is None:
        return "classical"
    return f"weighted_alpha_{alpha:.1f}"


# =============================================================================
# 6. NG-JORDAN-WEISS SPECTRAL CLUSTERING
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
    affinity = np.exp(
        -(distance.astype(np.float64) ** 2) / (2.0 * sigma**2)
    )
    np.fill_diagonal(affinity, 0.0)
    return affinity


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def njw_eigendecomposition(
    affinity: np.ndarray,
    n_eigenvectors: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    affinity = np.asarray(affinity, dtype=np.float64)
    degrees = affinity.sum(axis=1)
    if np.any(degrees <= 0):
        raise ValueError("A zero graph degree was encountered. Increase sigma.")

    inv_sqrt_degree = 1.0 / np.sqrt(degrees)
    n_objects = affinity.shape[0]

    def matvec(vector: np.ndarray) -> np.ndarray:
        return inv_sqrt_degree * (
            affinity @ (inv_sqrt_degree * vector)
        )

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
            raise RuntimeError(
                "ARPACK did not converge to the requested number of eigenpairs. "
                "Try increasing EIGEN_MAX_ITERATIONS or relaxing EIGEN_TOLERANCE."
            ) from error
        warnings.warn(
            "ARPACK reached the iteration limit; converged eigenpairs are used.",
            stacklevel=2,
        )
        eigenvalues = error.eigenvalues
        eigenvectors = error.eigenvectors

    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def compute_silhouette(
    data: np.ndarray,
    labels: np.ndarray,
    metric: str,
    sample_size: int | None,
    seed: int,
) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan")

    matrix = data
    if metric == "precomputed":
        matrix = np.asarray(data, dtype=np.float64).copy()
        np.fill_diagonal(matrix, 0.0)

    kwargs: dict[str, object] = {"metric": metric}
    if sample_size is not None and sample_size < len(labels):
        kwargs["sample_size"] = sample_size
        kwargs["random_state"] = seed

    return float(silhouette_score(matrix, labels, **kwargs))


def build_spectral_basis(
    distance: np.ndarray,
    alpha: float | None,
    sigma_multiplier: float,
    seed: int,
) -> SpectralBasis:
    sigma_reference = median_positive_distance(distance)
    sigma = sigma_multiplier * sigma_reference
    affinity = gaussian_affinity(distance, sigma)
    eigenvalues, eigenvectors = njw_eigendecomposition(
        affinity,
        n_eigenvectors=max(K_VALUES) + 1,
        seed=seed,
    )
    del affinity
    gc.collect()

    return SpectralBasis(
        gower_name=gower_name(alpha),
        alpha=alpha,
        sigma_reference=sigma_reference,
        sigma_multiplier=sigma_multiplier,
        sigma=sigma,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
    )


def fit_k_from_basis(
    basis: SpectralBasis,
    distance: np.ndarray,
    k: int,
    seed: int,
    exact_silhouettes: bool,
) -> FittedSolution:
    raw = basis.eigenvectors[:, :k]
    embedding = row_normalize(raw)
    labels = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=KMEANS_N_INIT,
        random_state=seed,
    ).fit_predict(embedding)

    sizes = np.bincount(labels, minlength=k)
    minimum_required = int(math.ceil(MIN_CLUSTER_FRACTION * len(labels)))
    sample_size = None if exact_silhouettes else GRID_SILHOUETTE_SAMPLE_SIZE

    return FittedSolution(
        gower_name=basis.gower_name,
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
            distance,
            labels,
            metric="precomputed",
            sample_size=sample_size,
            seed=seed,
        ),
        spectral_silhouette=compute_silhouette(
            embedding,
            labels,
            metric="euclidean",
            sample_size=sample_size,
            seed=seed,
        ),
        eigengap=float(basis.eigenvalues[k - 1] - basis.eigenvalues[k]),
        valid_cluster_sizes=bool(np.all(sizes >= minimum_required)),
    )


# =============================================================================
# 7. COMPLETE GRID AND WITHIN-GEOMETRY SELECTION
# =============================================================================


def solution_to_row(solution: FittedSolution) -> dict[str, object]:
    minimum_required = int(
        math.ceil(MIN_CLUSTER_FRACTION * solution.labels.size)
    )
    return {
        "gower_configuration": solution.gower_name,
        "numerical_block_weight": solution.alpha,
        "binary_block_weight": (
            None if solution.alpha is None else 1.0 - solution.alpha
        ),
        "sigma_reference_median": solution.sigma_reference,
        "sigma_multiplier": solution.sigma_multiplier,
        "sigma": solution.sigma,
        "k": solution.k,
        "gower_silhouette": solution.gower_silhouette,
        "spectral_silhouette": solution.spectral_silhouette,
        "eigengap": solution.eigengap,
        "valid_cluster_sizes": solution.valid_cluster_sizes,
        "minimum_required_cluster_size": minimum_required,
        "minimum_cluster_size": int(solution.cluster_sizes.min()),
        "maximum_cluster_size": int(solution.cluster_sizes.max()),
        "cluster_sizes": json.dumps(solution.cluster_sizes.tolist()),
    }


def run_complete_grid(components: DistanceComponents, n_objects: int) -> pd.DataFrame:
    del n_objects  # retained in signature to emphasize the object-level analysis
    rows: list[dict[str, object]] = []
    configurations: tuple[float | None, ...] = (None, *WEIGHTED_ALPHAS)

    for configuration_index, alpha in enumerate(configurations):
        name = gower_name(alpha)
        print(f"\n{'=' * 78}\n{name.upper()}\n{'=' * 78}")
        distance = build_gower_distance(components, alpha)
        sigma_reference = median_positive_distance(distance)
        print(f"Median positive Gower dissimilarity: {sigma_reference:.6f}")

        for sigma_index, multiplier in enumerate(SIGMA_MULTIPLIERS):
            basis_seed = (
                RANDOM_STATE
                + 1000 * configuration_index
                + 100 * sigma_index
            )
            basis = build_spectral_basis(
                distance,
                alpha=alpha,
                sigma_multiplier=multiplier,
                seed=basis_seed,
            )

            for k in K_VALUES:
                solution = fit_k_from_basis(
                    basis,
                    distance,
                    k=k,
                    seed=basis_seed + k,
                    exact_silhouettes=False,
                )
                rows.append(solution_to_row(solution))
                print(
                    f"sigma={multiplier:>3} | k={k:>2} | "
                    f"sil_G={solution.gower_silhouette: .4f} | "
                    f"sil_spectral={solution.spectral_silhouette: .4f} | "
                    f"gap={solution.eigengap: .6f} | "
                    f"sizes={solution.cluster_sizes.tolist()}"
                )

            del basis
            gc.collect()

        del distance
        gc.collect()

    return pd.DataFrame(rows)


def rank_candidates(group: pd.DataFrame) -> pd.DataFrame:
    """Transparent within-geometry ranking.

    Primary criterion: Gower silhouette, because it evaluates the partition in
    the same original dissimilarity geometry used to build the graph.
    Tie-breaks: eigengap, spectral silhouette, then lower k. Invalid cluster-size
    configurations are retained in the table but ranked after valid ones.
    """
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


def select_representative_row(group: pd.DataFrame) -> pd.Series:
    ranked = rank_candidates(group)
    return ranked.iloc[0]


# =============================================================================
# 8. REFITTING SELECTED AND CONTROLLED COMPARISON SOLUTIONS
# =============================================================================


def refit_solution(
    components: DistanceComponents,
    alpha: float | None,
    sigma_multiplier: float,
    k: int,
    seed_offset: int,
) -> FittedSolution:
    distance = build_gower_distance(components, alpha)
    basis = build_spectral_basis(
        distance,
        alpha=alpha,
        sigma_multiplier=float(sigma_multiplier),
        seed=RANDOM_STATE + seed_offset,
    )
    solution = fit_k_from_basis(
        basis,
        distance,
        k=int(k),
        seed=RANDOM_STATE + seed_offset + int(k),
        exact_silhouettes=True,
    )
    del distance, basis
    gc.collect()
    return solution


def refit_shared_basis_solutions(
    components: DistanceComponents,
    alpha: float | None,
    sigma_multiplier: float,
    k_values: Iterable[int],
    seed_offset: int,
) -> dict[int, FittedSolution]:
    distance = build_gower_distance(components, alpha)
    basis = build_spectral_basis(
        distance,
        alpha=alpha,
        sigma_multiplier=float(sigma_multiplier),
        seed=RANDOM_STATE + seed_offset,
    )
    solutions: dict[int, FittedSolution] = {}
    for k in sorted(set(int(value) for value in k_values)):
        solutions[k] = fit_k_from_basis(
            basis,
            distance,
            k=k,
            seed=RANDOM_STATE + seed_offset + k,
            exact_silhouettes=True,
        )
    del distance, basis
    gc.collect()
    return solutions


# =============================================================================
# 9. PLOTTING HELPERS
# =============================================================================


def configure_plot_defaults() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.22,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })


def save_model_selection_figure(
    results: pd.DataFrame,
    configuration_name: str,
    output_name: str,
) -> None:
    subset = results[
        results["gower_configuration"] == configuration_name
    ].copy()

    metrics = [
        ("gower_silhouette", "Gower silhouette"),
        ("eigengap", "Eigengap"),
        ("spectral_silhouette", "Spectral silhouette"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    for ax, (metric, label) in zip(axes, metrics):
        for multiplier in SIGMA_MULTIPLIERS:
            curve = subset[
                subset["sigma_multiplier"] == multiplier
            ].sort_values("k")
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
    fig.suptitle(f"Model-selection diagnostics: {configuration_name}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_raw_vs_normalized(solution: FittedSolution, output_name: str) -> None:
    if solution.k < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
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
        f"{solution.gower_name} | k={solution.k} | "
        f"sigma={solution.sigma_multiplier} x median\n"
        f"Gower silhouette={solution.gower_silhouette:.4f}; "
        f"spectral silhouette={solution.spectral_silhouette:.4f}; "
        f"eigengap={solution.eigengap:.5f}; "
        f"sizes={solution.cluster_sizes.tolist()}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_k3_embedding(solution: FittedSolution, output_name: str) -> None:
    if solution.k != 3:
        return
    fig = plt.figure(figsize=(9, 7.5))
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
    ax.set_title(
        f"Three-dimensional NJW embedding | {solution.gower_name} | k=3"
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_k4_pairwise_projections(
    solution: FittedSolution,
    output_name: str,
) -> None:
    if solution.k != 4:
        return

    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
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
        f"All pairwise views of the four-dimensional NJW embedding\n"
        f"{solution.gower_name} | sigma={solution.sigma_multiplier} x median | k=4",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_eigenvalue_spectrum(
    classical: FittedSolution,
    balanced: FittedSolution,
    output_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, solution, title in (
        (axes[0], classical, "Classical Gower"),
        (axes[1], balanced, "Balanced block-weighted Gower"),
    ):
        values = solution.eigenvalues[: max(K_VALUES) + 1]
        positions = np.arange(1, len(values) + 1)
        ax.plot(positions, values, marker="o")
        ax.axvline(solution.k + 0.5, linestyle="--", linewidth=1.2)
        ax.set_xticks(positions)
        ax.set_xlabel("Ordered eigenvalue index")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(
            f"{title}\nselected k={solution.k}, gap={solution.eigengap:.5f}"
        )

    fig.suptitle("Normalized-affinity eigenvalue spectra", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_ari_heatmap(ari: pd.DataFrame, output_name: str) -> None:
    values = ari.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    image = ax.imshow(values, vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(len(ari.columns)), ari.columns)
    ax.set_yticks(range(len(ari.index)), ari.index)
    ax.set_title("Partition stability across block weights (ARI)")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{values[row, column]:.3f}",
                ha="center",
                va="center",
            )
    fig.colorbar(image, ax=ax, label="Adjusted Rand Index")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_nesting_heatmap(
    counts: pd.DataFrame,
    row_percentages: pd.DataFrame,
    output_name: str,
) -> None:
    values = row_percentages.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(values, vmin=0.0, vmax=100.0, aspect="auto")
    ax.set_xticks(range(len(row_percentages.columns)), row_percentages.columns)
    ax.set_yticks(range(len(row_percentages.index)), row_percentages.index)
    ax.set_xlabel("Balanced k=4 cluster")
    ax.set_ylabel("Balanced k=2 macrocluster")
    ax.set_title("Nesting of the four-cluster refinement within k=2")

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{int(counts.iloc[row, column])}\n({values[row, column]:.1f}%)",
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax, label="Percentage within each k=2 cluster")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 10. EXPORT HELPERS
# =============================================================================


def save_coordinates(solution: FittedSolution, stem: str) -> None:
    data: dict[str, np.ndarray] = {
        f"raw_eigenvector_{index + 1}": solution.raw_eigenvectors[:, index]
        for index in range(solution.k)
    }
    data.update({
        f"normalized_coordinate_{index + 1}": solution.embedding[:, index]
        for index in range(solution.k)
    })
    data["cluster"] = solution.labels
    pd.DataFrame(data).to_csv(COORDINATE_DIR / f"{stem}.csv", index=False)


def build_ari_matrix(solutions: dict[str, FittedSolution]) -> pd.DataFrame:
    names = list(solutions)
    values = np.zeros((len(names), len(names)), dtype=float)
    for i, first in enumerate(names):
        for j, second in enumerate(names):
            values[i, j] = adjusted_rand_score(
                solutions[first].labels,
                solutions[second].labels,
            )
    return pd.DataFrame(values, index=names, columns=names)


def reorder_fine_clusters_by_coarse(
    coarse_labels: np.ndarray,
    fine_labels: np.ndarray,
) -> np.ndarray:
    """Relabel fine clusters only to make the contingency table readable."""
    table = pd.crosstab(coarse_labels, fine_labels)
    ordered_columns: list[int] = []
    for coarse_cluster in table.index:
        columns = table.loc[coarse_cluster].sort_values(ascending=False).index.tolist()
        for column in columns:
            if int(column) not in ordered_columns:
                ordered_columns.append(int(column))
    mapping = {old: new for new, old in enumerate(ordered_columns)}
    return np.array([mapping[int(label)] for label in fine_labels], dtype=int)


def build_nesting_tables(
    k2_solution: FittedSolution,
    k4_solution: FittedSolution,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    reordered_k4 = reorder_fine_clusters_by_coarse(
        k2_solution.labels,
        k4_solution.labels,
    )
    counts = pd.crosstab(
        k2_solution.labels,
        reordered_k4,
        rownames=["k2_macrocluster"],
        colnames=["k4_subcluster"],
    )
    counts.columns = [f"k4_cluster_{column}" for column in counts.columns]
    counts.index = [f"k2_cluster_{index}" for index in counts.index]
    row_percentages = counts.div(counts.sum(axis=1), axis=0) * 100.0
    return counts, row_percentages, reordered_k4


def solution_summary_row(role: str, solution: FittedSolution) -> dict[str, object]:
    return {
        "role_in_narrative": role,
        "gower_configuration": solution.gower_name,
        "numerical_block_weight": solution.alpha,
        "binary_block_weight": (
            None if solution.alpha is None else 1.0 - solution.alpha
        ),
        "k": solution.k,
        "sigma_multiplier": solution.sigma_multiplier,
        "sigma_reference_median": solution.sigma_reference,
        "sigma": solution.sigma,
        "gower_silhouette_exact": solution.gower_silhouette,
        "spectral_silhouette_exact": solution.spectral_silhouette,
        "eigengap": solution.eigengap,
        "cluster_sizes": json.dumps(solution.cluster_sizes.tolist()),
        "valid_cluster_sizes": solution.valid_cluster_sizes,
    }


def json_compatible(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_automatic_interpretation(
    preprocessing: dict[str, object],
    classical: FittedSolution,
    balanced: FittedSolution,
    weight_ari: pd.DataFrame,
    k2: FittedSolution,
    k4: FittedSolution,
    counts: pd.DataFrame,
) -> None:
    off_diagonal = weight_ari.to_numpy()[
        np.triu_indices(len(weight_ari), k=1)
    ]
    minimum_weight_ari = float(off_diagonal.min())
    classic_vs_balanced = adjusted_rand_score(classical.labels, balanced.labels)

    lines = [
        "# Mixed-data Gower + NJW baseline: automatic interpretation",
        "",
        "## 1. Dataset used",
        (
            f"The analysis used **{preprocessing['n_objects']} observations** and "
            f"**{preprocessing['remaining_features']} retained descriptors**: "
            f"**{preprocessing['remaining_numeric_features']} numerical "
            f"({preprocessing['numeric_percentage']:.1f}%) and "
            f"**{preprocessing['remaining_binary_features']} binary "
            f"({preprocessing['binary_percentage']:.1f}%)**."
        ),
        (
            f"Input source: `{preprocessing['input_path']}`. "
            "The Activity variable was excluded from all unsupervised steps."
        ),
        "",
        "## 2. Classical Gower baseline",
        (
            f"The representative classical solution used k={classical.k} and "
            f"sigma={classical.sigma_multiplier} times the median positive "
            f"Gower dissimilarity. It produced cluster sizes "
            f"{classical.cluster_sizes.tolist()}, Gower silhouette "
            f"{classical.gower_silhouette:.4f}, spectral silhouette "
            f"{classical.spectral_silhouette:.4f}, and eigengap "
            f"{classical.eigengap:.6f}."
        ),
        "",
        "## 3. Why block weighting was explored",
        (
            "Classical Gower is retained as the standard reference, but its "
            "feature-wise aggregation does not explicitly control the total "
            "contribution of the numerical and binary descriptor blocks. The "
            "weighted analysis therefore asks whether the discovered structure "
            "is robust to moderate, controlled changes in block contribution; "
            "it is not an attempt to declare a different distance universally "
            "superior."
        ),
        "",
        "## 4. Weighted sensitivity and balanced representative",
        (
            f"At the balanced representative resolution (k={balanced.k}, "
            f"sigma multiplier={balanced.sigma_multiplier}), the minimum "
            f"pairwise ARI across alpha=0.4, 0.5, and 0.6 was "
            f"{minimum_weight_ari:.4f}. The alpha=0.5 partition had cluster "
            f"sizes {balanced.cluster_sizes.tolist()}, Gower silhouette "
            f"{balanced.gower_silhouette:.4f}, spectral silhouette "
            f"{balanced.spectral_silhouette:.4f}, and eigengap "
            f"{balanced.eigengap:.6f}."
        ),
        (
            f"The ARI between the selected classical and balanced partitions "
            f"was {classic_vs_balanced:.4f}. This comparison is based on label "
            "agreement; silhouettes from the two Gower definitions are not "
            "ranked against one another because the underlying geometries differ."
        ),
        "",
        "## 5. Resolution analysis",
        (
            f"Under alpha=0.5 and the selected weighted bandwidth, k=2 gave "
            f"sizes {k2.cluster_sizes.tolist()}, whereas k=4 gave "
            f"{k4.cluster_sizes.tolist()}. The contingency table in "
            f"`{NESTING_COUNTS_CSV.name}` shows whether the four clusters are "
            "nested refinements of the two macroclusters."
        ),
        "",
        "## 6. Baseline for the next analyses",
        (
            "The exported mixed-data labels provide the reference against which "
            "the same NJW procedure can be run on the numerical-only and binary-"
            "only blocks. ARI and contingency comparisons will then quantify "
            "which block reproduces the mixed organization and which clusters "
            "depend on the interaction between descriptor types."
        ),
        "",
        "## Balanced k=2 versus k=4 counts",
        "",
        "```",
        counts.to_string(),
        "```",
    ]
    AUTO_INTERPRETATION_MD.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 11. MAIN NARRATIVE PIPELINE
# =============================================================================


def main() -> None:
    for directory in (REPORT_DIR, TABLE_DIR, FIGURE_DIR, COORDINATE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    configure_plot_defaults()

    # -------------------------------------------------------------------------
    # A. Load exactly the project dataset that should act as the mixed baseline.
    # -------------------------------------------------------------------------
    loaded = load_project_data()
    schema = fit_feature_schema(loaded)
    preprocessing = schema.preprocessing_summary
    pd.DataFrame([preprocessing]).to_csv(PREPROCESSING_CSV, index=False)

    print(f"Input file: {loaded.source_path}")
    print(f"Input kind: {loaded.source_kind}")
    print(f"Objects: {preprocessing['n_objects']}")
    print(
        "Retained descriptors: "
        f"{preprocessing['remaining_features']} = "
        f"{preprocessing['remaining_numeric_features']} numerical "
        f"({preprocessing['numeric_percentage']:.2f}%) + "
        f"{preprocessing['remaining_binary_features']} binary "
        f"({preprocessing['binary_percentage']:.2f}%)"
    )
    print(f"'{TARGET_COL}' is excluded from all unsupervised steps.")

    X_numeric, X_binary = transform_features(loaded.X, schema)
    components = compute_distance_components(X_numeric, X_binary)

    # -------------------------------------------------------------------------
    # B. Complete grid. This produces evidence, not the final story by itself.
    # -------------------------------------------------------------------------
    results = run_complete_grid(components, len(loaded.X))
    results.to_csv(ALL_RESULTS_CSV, index=False)

    # -------------------------------------------------------------------------
    # C. Classical baseline: choose only within classical Gower geometry.
    # -------------------------------------------------------------------------
    classical_group = results[
        results["gower_configuration"] == "classical"
    ]
    classical_ranking = rank_candidates(classical_group)
    classical_ranking.to_csv(CLASSICAL_RANKING_CSV, index=False)
    classical_row = select_representative_row(classical_group)

    classical_solution = refit_solution(
        components,
        alpha=None,
        sigma_multiplier=float(classical_row["sigma_multiplier"]),
        k=int(classical_row["k"]),
        seed_offset=20_000,
    )

    # -------------------------------------------------------------------------
    # D. Balanced representative: alpha=0.5 selected within its own geometry.
    #    It is the neutral member of the weighted sensitivity family.
    # -------------------------------------------------------------------------
    balanced_name = gower_name(BALANCED_ALPHA)
    balanced_group = results[
        results["gower_configuration"] == balanced_name
    ]
    balanced_ranking = rank_candidates(balanced_group)
    balanced_ranking.to_csv(WEIGHTED_RANKING_CSV, index=False)
    balanced_row = select_representative_row(balanced_group)

    balanced_solution = refit_solution(
        components,
        alpha=BALANCED_ALPHA,
        sigma_multiplier=float(balanced_row["sigma_multiplier"]),
        k=int(balanced_row["k"]),
        seed_offset=30_000,
    )

    # -------------------------------------------------------------------------
    # E. Controlled sensitivity: same k and sigma multiplier for all alphas.
    #    This isolates the effect of the block weight itself.
    # -------------------------------------------------------------------------
    sensitivity_solutions: dict[str, FittedSolution] = {}
    sensitivity_rows: list[dict[str, object]] = []

    for index, alpha in enumerate(WEIGHTED_ALPHAS):
        solution = refit_solution(
            components,
            alpha=alpha,
            sigma_multiplier=balanced_solution.sigma_multiplier,
            k=balanced_solution.k,
            seed_offset=40_000 + 1000 * index,
        )
        key = f"alpha={alpha:.1f}"
        sensitivity_solutions[key] = solution
        sensitivity_rows.append(solution_summary_row("weight_sensitivity", solution))

    sensitivity_table = pd.DataFrame(sensitivity_rows)
    sensitivity_table.to_csv(WEIGHT_SENSITIVITY_CSV, index=False)
    weight_ari = build_ari_matrix(sensitivity_solutions)
    weight_ari.to_csv(WEIGHT_ARI_CSV)

    # -------------------------------------------------------------------------
    # F. Resolution analysis at the balanced bandwidth. k=3 remains in the grid
    #    but k=2 and k=4 are compared directly to test macrocluster refinement.
    # -------------------------------------------------------------------------
    balanced_resolutions = refit_shared_basis_solutions(
        components,
        alpha=BALANCED_ALPHA,
        sigma_multiplier=balanced_solution.sigma_multiplier,
        k_values=(2, 3, 4),
        seed_offset=50_000,
    )
    balanced_k2 = balanced_resolutions[2]
    balanced_k3 = balanced_resolutions[3]
    balanced_k4 = balanced_resolutions[4]

    nesting_counts, nesting_rows, reordered_k4 = build_nesting_tables(
        balanced_k2,
        balanced_k4,
    )
    nesting_counts.to_csv(NESTING_COUNTS_CSV)
    nesting_rows.to_csv(NESTING_ROWS_CSV)

    # -------------------------------------------------------------------------
    # G. Export the concise table that follows the written narrative.
    # -------------------------------------------------------------------------
    narrative_summary = pd.DataFrame([
        solution_summary_row("classical_mixed_baseline", classical_solution),
        solution_summary_row("balanced_weighted_representative", balanced_solution),
        solution_summary_row("balanced_k2_macrostructure", balanced_k2),
        solution_summary_row("balanced_k3_intermediate_check", balanced_k3),
        solution_summary_row("balanced_k4_refinement", balanced_k4),
    ])
    narrative_summary.to_csv(NARRATIVE_SUMMARY_CSV, index=False)

    # -------------------------------------------------------------------------
    # H. Figures, produced in the same order as the report narrative.
    # -------------------------------------------------------------------------
    save_model_selection_figure(
        results,
        configuration_name="classical",
        output_name="01_classical_model_selection.png",
    )
    save_raw_vs_normalized(
        classical_solution,
        output_name="02_classical_selected_embedding.png",
    )
    save_model_selection_figure(
        results,
        configuration_name=balanced_name,
        output_name="03_balanced_model_selection.png",
    )
    save_ari_heatmap(
        weight_ari,
        output_name="04_weight_sensitivity_ari.png",
    )
    save_raw_vs_normalized(
        balanced_solution,
        output_name="05_balanced_selected_embedding.png",
    )

    if balanced_solution.k == 3:
        save_k3_embedding(
            balanced_solution,
            output_name="06_balanced_selected_embedding_3d.png",
        )
    if balanced_solution.k == 4:
        save_k4_pairwise_projections(
            balanced_solution,
            output_name="06_balanced_selected_embedding_all_pairs.png",
        )

    # The k=4 projection is useful for the resolution discussion even when the
    # automatically selected balanced k is 2.
    save_k4_pairwise_projections(
        balanced_k4,
        output_name="07_balanced_k4_refinement_all_pairs.png",
    )
    save_eigenvalue_spectrum(
        classical_solution,
        balanced_solution,
        output_name="08_eigenvalue_spectra.png",
    )
    save_nesting_heatmap(
        nesting_counts,
        nesting_rows,
        output_name="09_balanced_k2_vs_k4_nesting.png",
    )

    # -------------------------------------------------------------------------
    # I. Coordinates and labels for reproducibility and later block comparisons.
    # -------------------------------------------------------------------------
    save_coordinates(classical_solution, "classical_selected_coordinates")
    save_coordinates(balanced_solution, "balanced_selected_coordinates")
    save_coordinates(balanced_k2, "balanced_k2_coordinates")
    save_coordinates(balanced_k4, "balanced_k4_coordinates")

    baseline_labels = pd.DataFrame({
        "row_index": np.arange(len(loaded.X)),
        "mixed_classical_selected": classical_solution.labels,
        "mixed_balanced_selected": balanced_solution.labels,
        "mixed_balanced_k2": balanced_k2.labels,
        "mixed_balanced_k4": reordered_k4,
    })
    if loaded.activity is not None:
        baseline_labels[TARGET_COL] = loaded.activity.to_numpy()
    baseline_labels.to_csv(BASELINE_LABELS_CSV, index=False)

    # -------------------------------------------------------------------------
    # J. Machine-readable report values and concise automatic interpretation.
    # -------------------------------------------------------------------------
    off_diagonal_weight_ari = weight_ari.to_numpy()[
        np.triu_indices(len(weight_ari), k=1)
    ]
    report_values = {
        "preprocessing": preprocessing,
        "classical_selected": solution_summary_row(
            "classical_mixed_baseline", classical_solution
        ),
        "balanced_selected": solution_summary_row(
            "balanced_weighted_representative", balanced_solution
        ),
        "balanced_k2": solution_summary_row(
            "balanced_k2_macrostructure", balanced_k2
        ),
        "balanced_k4": solution_summary_row(
            "balanced_k4_refinement", balanced_k4
        ),
        "minimum_ari_across_weighted_alphas": float(
            off_diagonal_weight_ari.min()
        ),
        "mean_ari_across_weighted_alphas": float(
            off_diagonal_weight_ari.mean()
        ),
        "ari_classical_vs_balanced_selected": float(
            adjusted_rand_score(
                classical_solution.labels,
                balanced_solution.labels,
            )
        ),
        "important_interpretation_rule": (
            "Do not compare silhouette magnitudes directly across different "
            "Gower definitions; use ARI, eigengap, sizes, stability, and "
            "interpretability for cross-formulation comparison."
        ),
    }
    with REPORT_VALUES_JSON.open("w", encoding="utf-8") as handle:
        json.dump(
            report_values,
            handle,
            indent=2,
            default=json_compatible,
        )

    write_automatic_interpretation(
        preprocessing,
        classical_solution,
        balanced_solution,
        weight_ari,
        balanced_k2,
        balanced_k4,
        nesting_counts,
    )

    print("\n" + "=" * 78)
    print("ANALYSIS COMPLETED")
    print("=" * 78)
    print(f"Reports directory: {REPORT_DIR}")
    print("Narrative order:")
    print("  1. preprocessing summary")
    print("  2. classical Gower model selection and selected embedding")
    print("  3. motivation for explicit block weighting")
    print("  4. alpha=0.4/0.5/0.6 stability at a controlled resolution")
    print("  5. balanced representative and k=2 -> k=4 refinement")
    print("  6. exported mixed labels for numerical-only/binary-only comparison")
    print("\nMain files:")
    for path in (
        PREPROCESSING_CSV,
        NARRATIVE_SUMMARY_CSV,
        WEIGHT_ARI_CSV,
        NESTING_COUNTS_CSV,
        BASELINE_LABELS_CSV,
        REPORT_VALUES_JSON,
        AUTO_INTERPRETATION_MD,
        FIGURE_DIR,
    ):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
