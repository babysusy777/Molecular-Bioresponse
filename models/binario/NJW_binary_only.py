from __future__ import annotations

"""Binary-only clustering: asymmetric Jaccard + Ng-Jordan-Weiss.

It keeps the same clustering protocol used for the mixed-data baseline while
changing only the representation:

    retained asymmetric binary descriptors
        -> Jaccard dissimilarity
        -> Gaussian affinity
        -> Ng-Jordan-Weiss spectral embedding
        -> K-means

The script explores k=2,...,10 and sigma in {0.5, 1, 2} times the median
positive Jaccard dissimilarity. It always refits and exports k=2, k=3, and k=4
at the bandwidth selected within the binary geometry.

When the mixed-baseline labels are available, it also compares binary-only and
mixed partitions using Adjusted Rand Index and contingency tables. Silhouette
values are never used for cross-geometry ranking.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
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
from sklearn.metrics import adjusted_rand_score, silhouette_score


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

SCRIPT_VERSION = "binary-only-v1-2026-07-22"
RANDOM_STATE = 42
TARGET_COL = "Activity"
QUASI_CONSTANT_THRESHOLD = 0.99

K_VALUES = tuple(range(2, 11))
SIGMA_MULTIPLIERS = (0.5, 1.0, 2.0)
KMEANS_N_INIT = 50
MIN_CLUSTER_FRACTION = 0.05
DISTANCE_DTYPE = np.float32
EIGEN_TOLERANCE = 1e-6
EIGEN_MAX_ITERATIONS = 5000
FIGURE_DPI = 220

# None means exact silhouettes for the full grid. Set an integer only if the
# complete grid is too slow. Representative solutions are always recomputed
# with exact silhouettes.
GRID_SILHOUETTE_SAMPLE_SIZE: int | None = None

# Optional manual overrides. Normally leave both as None.
FORCE_INPUT_PATH: Path | None = None
FORCE_MIXED_LABELS_PATH: Path | None = None

SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# 2. PROJECT AND OUTPUT PATHS
# =============================================================================


def find_project_root(start: Path) -> Path:
    """Find the nearest ancestor containing the project Dataset directory."""
    for candidate in (start, *start.parents):
        if (candidate / "Dataset").exists():
            return candidate
    return start


PROJECT_DIR = find_project_root(SCRIPT_DIR)

# Since the file is intended for models/binario, results are kept beside it.
REPORT_DIR = SCRIPT_DIR / "reports" / "njw_binary_only"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"
COORDINATE_DIR = REPORT_DIR / "coordinates"

PREPROCESSING_CSV = TABLE_DIR / "01_binary_preprocessing_summary.csv"
GRID_RESULTS_CSV = TABLE_DIR / "02_binary_grid_results.csv"
RANKING_CSV = TABLE_DIR / "03_binary_candidate_ranking.csv"
RESOLUTION_SUMMARY_CSV = TABLE_DIR / "04_binary_resolution_summary.csv"
ARI_COMPARISON_CSV = TABLE_DIR / "05_binary_vs_mixed_ari.csv"
BINARY_K2_K3_COUNTS_CSV = TABLE_DIR / "06_binary_k2_vs_k3_counts.csv"
BINARY_K2_K3_ROWS_CSV = TABLE_DIR / "07_binary_k2_vs_k3_row_percentages.csv"
BINARY_K2_K4_COUNTS_CSV = TABLE_DIR / "08_binary_k2_vs_k4_counts.csv"
BINARY_K2_K4_ROWS_CSV = TABLE_DIR / "09_binary_k2_vs_k4_row_percentages.csv"
BINARY_LABELS_CSV = TABLE_DIR / "10_binary_labels.csv"
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
    source_kind: str


@dataclass(frozen=True)
class BinarySchema:
    retained_features: list[str]
    binary_features: list[str]
    preprocessing_summary: dict[str, object]


@dataclass
class SpectralBasis:
    sigma_reference: float
    sigma_multiplier: float
    sigma: float
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


@dataclass
class FittedSolution:
    sigma_reference: float
    sigma_multiplier: float
    sigma: float
    k: int
    eigenvalues: np.ndarray
    raw_eigenvectors: np.ndarray
    embedding: np.ndarray
    labels: np.ndarray
    cluster_sizes: np.ndarray
    jaccard_silhouette: float
    spectral_silhouette: float
    eigengap: float
    valid_cluster_sizes: bool


@dataclass(frozen=True)
class MixedLabels:
    frame: pd.DataFrame
    source_path: Path


# =============================================================================
# 4. INPUT DISCOVERY AND FEATURE SELECTION
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

    duplicates = int(X.duplicated().sum())
    if duplicates:
        warnings.warn(
            f"{duplicates} duplicate rows were found and retained to preserve alignment.",
            stacklevel=2,
        )


def load_project_data() -> LoadedData:
    if FORCE_INPUT_PATH is not None:
        input_path = Path(FORCE_INPUT_PATH).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"FORCE_INPUT_PATH does not exist: {input_path}")
        data = read_csv_clean(input_path)
        source_kind = "filtered" if "filtered_no_activity" in input_path.stem else "raw"
    else:
        filtered = first_existing(candidate_filtered_paths(PROJECT_DIR))
        if filtered is not None:
            input_path = filtered
            source_kind = "filtered"
            data = read_csv_clean(input_path)
        else:
            raw = first_existing(candidate_raw_paths(PROJECT_DIR))
            if raw is None:
                searched = candidate_filtered_paths(PROJECT_DIR) + candidate_raw_paths(PROJECT_DIR)
                raise FileNotFoundError(
                    "No input dataset found. Searched:\n" +
                    "\n".join(f"  - {path}" for path in searched)
                )
            input_path = raw
            source_kind = "raw"
            data = read_csv_clean(input_path)

    if TARGET_COL in data.columns:
        activity = data[TARGET_COL].reset_index(drop=True)
        X = data.drop(columns=TARGET_COL).copy()
    else:
        X = data.copy()
        activity = load_activity_if_available(PROJECT_DIR, len(X))

    X = X.reset_index(drop=True)
    validate_feature_frame(X)
    return LoadedData(
        X=X,
        activity=activity,
        source_path=input_path,
        source_kind=source_kind,
    )


def dominant_ratio(series: pd.Series) -> float:
    return float(series.value_counts(normalize=True, dropna=False).iloc[0])


def identify_binary_columns(X: pd.DataFrame) -> list[str]:
    allowed = {0, 1, 0.0, 1.0, False, True}
    return [
        column
        for column in X.columns
        if set(X[column].dropna().unique().tolist()).issubset(allowed)
    ]


def fit_binary_schema(data: LoadedData) -> BinarySchema:
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
    if not binary:
        raise ValueError("No binary descriptors were identified after preprocessing.")

    binary_values = X_retained[binary].to_numpy(dtype=np.float64, copy=False)
    ones_per_object = binary_values.sum(axis=1)
    prevalence = binary_values.mean(axis=0)

    summary = {
        "script_version": SCRIPT_VERSION,
        "input_path": str(data.source_path),
        "input_kind": data.source_kind,
        "n_objects": len(X),
        "features_in_input_file": original_features,
        "quasi_constant_threshold": QUASI_CONSTANT_THRESHOLD,
        "quasi_constant_removed_in_this_script": len(quasi_constant_removed),
        "retained_features_before_binary_selection": len(retained),
        "binary_features_used": len(binary),
        "mean_binary_prevalence": float(prevalence.mean()),
        "median_binary_prevalence": float(np.median(prevalence)),
        "mean_ones_per_object": float(ones_per_object.mean()),
        "median_ones_per_object": float(np.median(ones_per_object)),
        "minimum_ones_per_object": int(ones_per_object.min()),
        "maximum_ones_per_object": int(ones_per_object.max()),
        "objects_with_no_active_binary_descriptor": int(np.sum(ones_per_object == 0)),
        "activity_available_for_post_hoc_only": data.activity is not None,
    }

    return BinarySchema(
        retained_features=retained,
        binary_features=binary,
        preprocessing_summary=summary,
    )


def transform_binary_features(X: pd.DataFrame, schema: BinarySchema) -> np.ndarray:
    binary = X[schema.binary_features].to_numpy(dtype=DISTANCE_DTYPE, copy=True)
    if not np.isfinite(binary).all():
        raise ValueError("Non-finite values found in the binary matrix.")
    unique = np.unique(binary)
    if not set(unique.tolist()).issubset({0.0, 1.0}):
        raise ValueError(f"Unexpected values in binary matrix: {unique[:10]}")
    return binary


# =============================================================================
# 5. ASYMMETRIC JACCARD DISSIMILARITY
# =============================================================================


def compute_jaccard_dissimilarity(X_binary: np.ndarray) -> np.ndarray:
    """Compute pairwise asymmetric-binary Jaccard dissimilarity.

    For each pair:
      - 1/1 is an informative match;
      - 1/0 and 0/1 are mismatches;
      - 0/0 is ignored.

    If both observations contain no active binary descriptor, their binary
    dissimilarity is set to zero. The dataset summary reports whether this case
    actually occurs.
    """
    intersection = X_binary @ X_binary.T
    ones = X_binary.sum(axis=1, dtype=np.float64)
    union = (
        ones[:, None] + ones[None, :] - intersection
    ).astype(DISTANCE_DTYPE, copy=False)
    mismatches = union - intersection

    distance = np.divide(
        mismatches,
        union,
        out=np.zeros_like(union, dtype=DISTANCE_DTYPE),
        where=union > 0,
    )
    distance = np.clip(distance, 0.0, 1.0).astype(DISTANCE_DTYPE, copy=False)
    np.fill_diagonal(distance, 0.0)
    return distance


def median_positive_distance(distance: np.ndarray) -> float:
    upper = distance[np.triu_indices(distance.shape[0], k=1)]
    positive = upper[upper > 0]
    if positive.size == 0:
        raise ValueError("All off-diagonal Jaccard dissimilarities are zero.")
    return float(np.median(positive))


# =============================================================================
# 6. NG-JORDAN-WEISS SPECTRAL CLUSTERING
# =============================================================================


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
            raise RuntimeError(
                "ARPACK did not converge to the requested eigenpairs. "
                "Increase EIGEN_MAX_ITERATIONS or relax EIGEN_TOLERANCE."
            ) from error
        warnings.warn(
            "ARPACK reached its iteration limit; converged eigenpairs are used.",
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
    sigma_multiplier: float,
    seed: int,
) -> SpectralBasis:
    sigma_reference = median_positive_distance(distance)
    sigma = float(sigma_multiplier) * sigma_reference
    affinity = gaussian_affinity(distance, sigma)
    eigenvalues, eigenvectors = njw_eigendecomposition(
        affinity,
        n_eigenvectors=max(K_VALUES) + 1,
        seed=seed,
    )
    del affinity
    gc.collect()

    return SpectralBasis(
        sigma_reference=sigma_reference,
        sigma_multiplier=float(sigma_multiplier),
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
        sigma_reference=basis.sigma_reference,
        sigma_multiplier=basis.sigma_multiplier,
        sigma=basis.sigma,
        k=k,
        eigenvalues=basis.eigenvalues,
        raw_eigenvectors=raw,
        embedding=embedding,
        labels=labels,
        cluster_sizes=sizes,
        jaccard_silhouette=compute_silhouette(
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
# 7. GRID SEARCH AND REPRESENTATIVE-SOLUTION SELECTION
# =============================================================================


def solution_to_row(solution: FittedSolution) -> dict[str, object]:
    minimum_required = int(
        math.ceil(MIN_CLUSTER_FRACTION * solution.labels.size)
    )
    return {
        "distance": "asymmetric_jaccard",
        "sigma_reference_median": solution.sigma_reference,
        "sigma_multiplier": solution.sigma_multiplier,
        "sigma": solution.sigma,
        "k": solution.k,
        "jaccard_silhouette": solution.jaccard_silhouette,
        "spectral_silhouette": solution.spectral_silhouette,
        "eigengap": solution.eigengap,
        "valid_cluster_sizes": solution.valid_cluster_sizes,
        "minimum_required_cluster_size": minimum_required,
        "minimum_cluster_size": int(solution.cluster_sizes.min()),
        "maximum_cluster_size": int(solution.cluster_sizes.max()),
        "cluster_sizes": json.dumps(solution.cluster_sizes.tolist()),
    }


def run_complete_grid(distance: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sigma_reference = median_positive_distance(distance)
    print(f"Median positive Jaccard dissimilarity: {sigma_reference:.6f}")

    for sigma_index, multiplier in enumerate(SIGMA_MULTIPLIERS):
        basis_seed = RANDOM_STATE + 1000 * sigma_index
        basis = build_spectral_basis(
            distance,
            sigma_multiplier=multiplier,
            seed=basis_seed,
        )

        print("\n" + "=" * 78)
        print(f"BINARY-ONLY | sigma={multiplier} x median")
        print("=" * 78)

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
                f"k={k:>2} | "
                f"sil_J={solution.jaccard_silhouette: .4f} | "
                f"sil_spectral={solution.spectral_silhouette: .4f} | "
                f"gap={solution.eigengap: .6f} | "
                f"sizes={solution.cluster_sizes.tolist()}"
            )

        del basis
        gc.collect()

    return pd.DataFrame(rows)


def rank_candidates(results: pd.DataFrame) -> pd.DataFrame:
    """Rank configurations only within the binary Jaccard geometry."""
    return results.sort_values(
        by=[
            "valid_cluster_sizes",
            "jaccard_silhouette",
            "eigengap",
            "spectral_silhouette",
            "k",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def refit_solution(
    distance: np.ndarray,
    sigma_multiplier: float,
    k: int,
    seed_offset: int,
) -> FittedSolution:
    basis = build_spectral_basis(
        distance,
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
    del basis
    gc.collect()
    return solution


def refit_shared_basis_solutions(
    distance: np.ndarray,
    sigma_multiplier: float,
    k_values: Iterable[int],
    seed_offset: int,
) -> dict[int, FittedSolution]:
    basis = build_spectral_basis(
        distance,
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
    del basis
    gc.collect()
    return solutions


# =============================================================================
# 8. MIXED-BASELINE LABEL DISCOVERY AND COMPARISON
# =============================================================================


def discover_mixed_labels_path(root: Path) -> Path | None:
    if FORCE_MIXED_LABELS_PATH is not None:
        forced = Path(FORCE_MIXED_LABELS_PATH).expanduser().resolve()
        if not forced.exists():
            raise FileNotFoundError(
                f"FORCE_MIXED_LABELS_PATH does not exist: {forced}"
            )
        return forced

    preferred = [
        root / "reports" / "njw_mixed_baseline_k234_v2" / "tables" /
        "10_mixed_baseline_labels.csv",
        root / "reports" / "njw_mixed_baseline" / "tables" /
        "10_mixed_baseline_labels.csv",
    ]
    direct = first_existing(preferred)
    if direct is not None:
        return direct

    matches = list(root.rglob("10_mixed_baseline_labels.csv"))
    if not matches:
        return None

    def priority(path: Path) -> tuple[int, float]:
        text = str(path).lower()
        preferred_score = 0 if "k234_v2" in text else 1
        return preferred_score, -path.stat().st_mtime

    return sorted(matches, key=priority)[0]


def load_mixed_labels(root: Path, expected_rows: int) -> MixedLabels | None:
    path = discover_mixed_labels_path(root)
    if path is None:
        return None

    frame = read_csv_clean(path)
    if len(frame) != expected_rows:
        raise ValueError(
            "Mixed-label file has a different number of rows: "
            f"{len(frame)} instead of {expected_rows}. File: {path}"
        )

    if "row_index" in frame.columns:
        expected = np.arange(expected_rows)
        actual = frame["row_index"].to_numpy()
        if not np.array_equal(actual, expected):
            raise ValueError(
                "The row_index column in the mixed-label file is not aligned "
                "with the binary dataset."
            )

    return MixedLabels(frame=frame.reset_index(drop=True), source_path=path)


def build_ari_comparison_table(
    binary_selected: FittedSolution,
    binary_resolutions: dict[int, FittedSolution],
    mixed: MixedLabels | None,
) -> pd.DataFrame:
    columns = [
        "comparison",
        "binary_partition",
        "mixed_partition",
        "binary_k",
        "mixed_k",
        "same_resolution",
        "adjusted_rand_index",
    ]
    if mixed is None:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []

    selected_mixed_columns = [
        "mixed_classical_selected",
        "mixed_balanced_selected",
    ]
    for column in selected_mixed_columns:
        if column not in mixed.frame.columns:
            continue
        labels = mixed.frame[column].to_numpy(dtype=int)
        mixed_k = int(np.unique(labels).size)
        rows.append({
            "comparison": f"binary_selected_vs_{column}",
            "binary_partition": "binary_selected",
            "mixed_partition": column,
            "binary_k": binary_selected.k,
            "mixed_k": mixed_k,
            "same_resolution": binary_selected.k == mixed_k,
            "adjusted_rand_index": adjusted_rand_score(
                binary_selected.labels,
                labels,
            ),
        })

    for k in (2, 3, 4):
        column = f"mixed_balanced_k{k}"
        if column not in mixed.frame.columns:
            continue
        mixed_labels = mixed.frame[column].to_numpy(dtype=int)
        rows.append({
            "comparison": f"binary_k{k}_vs_mixed_balanced_k{k}",
            "binary_partition": f"binary_k{k}",
            "mixed_partition": column,
            "binary_k": k,
            "mixed_k": int(np.unique(mixed_labels).size),
            "same_resolution": True,
            "adjusted_rand_index": adjusted_rand_score(
                binary_resolutions[k].labels,
                mixed_labels,
            ),
        })

    return pd.DataFrame(rows, columns=columns)


def align_candidate_labels_to_reference(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
) -> np.ndarray:
    """Relabel candidate clusters to maximize diagonal overlap.

    This changes only cluster names, never membership or ARI.
    """
    reference_values = np.sort(np.unique(reference_labels))
    candidate_values = np.sort(np.unique(candidate_labels))
    contingency = pd.crosstab(reference_labels, candidate_labels).reindex(
        index=reference_values,
        columns=candidate_values,
        fill_value=0,
    )

    row_ind, col_ind = linear_sum_assignment(-contingency.to_numpy())
    mapping = {
        int(candidate_values[column]): int(reference_values[row])
        for row, column in zip(row_ind, col_ind)
    }

    used_targets = set(mapping.values())
    remaining_targets = [
        int(value) for value in reference_values if int(value) not in used_targets
    ]
    for candidate in candidate_values:
        candidate_int = int(candidate)
        if candidate_int not in mapping:
            mapping[candidate_int] = remaining_targets.pop(0)

    return np.array([mapping[int(label)] for label in candidate_labels], dtype=int)


def build_mixed_binary_contingency(
    mixed_labels: np.ndarray,
    binary_labels: np.ndarray,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    aligned_binary = align_candidate_labels_to_reference(mixed_labels, binary_labels)
    counts = pd.crosstab(
        mixed_labels,
        aligned_binary,
        rownames=["mixed_cluster"],
        colnames=["binary_cluster"],
    ).reindex(index=range(k), columns=range(k), fill_value=0)
    counts.index = [f"mixed_cluster_{value}" for value in counts.index]
    counts.columns = [f"binary_cluster_{value}" for value in counts.columns]
    row_percentages = counts.div(counts.sum(axis=1), axis=0) * 100.0
    return counts, row_percentages, aligned_binary


# =============================================================================
# 9. RESOLUTION NESTING WITHIN THE BINARY GEOMETRY
# =============================================================================


def reorder_fine_clusters_by_coarse(
    coarse_labels: np.ndarray,
    fine_labels: np.ndarray,
) -> np.ndarray:
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
    coarse_solution: FittedSolution,
    fine_solution: FittedSolution,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    reordered = reorder_fine_clusters_by_coarse(
        coarse_solution.labels,
        fine_solution.labels,
    )
    counts = pd.crosstab(
        coarse_solution.labels,
        reordered,
        rownames=[f"k{coarse_solution.k}_macrocluster"],
        colnames=[f"k{fine_solution.k}_subcluster"],
    )
    counts.index = [
        f"k{coarse_solution.k}_cluster_{value}" for value in counts.index
    ]
    counts.columns = [
        f"k{fine_solution.k}_cluster_{value}" for value in counts.columns
    ]
    row_percentages = counts.div(counts.sum(axis=1), axis=0) * 100.0
    return counts, row_percentages, reordered


# =============================================================================
# 10. PLOTTING
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


def save_model_selection_figure(results: pd.DataFrame, output_name: str) -> None:
    metrics = [
        ("jaccard_silhouette", "Jaccard silhouette"),
        ("eigengap", "Eigengap"),
        ("spectral_silhouette", "Spectral silhouette"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    for ax, (metric, label) in zip(axes, metrics):
        for multiplier in SIGMA_MULTIPLIERS:
            curve = results[
                results["sigma_multiplier"] == multiplier
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
    fig.suptitle("Binary-only model-selection diagnostics", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_raw_vs_normalized(solution: FittedSolution, output_name: str) -> None:
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
        f"Binary-only Jaccard + NJW | k={solution.k} | "
        f"sigma={solution.sigma_multiplier} x median\n"
        f"Jaccard silhouette={solution.jaccard_silhouette:.4f}; "
        f"spectral silhouette={solution.spectral_silhouette:.4f}; "
        f"eigengap={solution.eigengap:.5f}; "
        f"sizes={solution.cluster_sizes.tolist()}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_k3_embedding_3d(solution: FittedSolution, output_name: str) -> None:
    if solution.k != 3:
        raise ValueError("save_k3_embedding_3d requires a k=3 solution.")
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
    ax.set_title("Binary-only three-dimensional NJW embedding | k=3")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_k4_pairwise_projections(
    solution: FittedSolution,
    output_name: str,
) -> None:
    if solution.k != 4:
        raise ValueError("save_k4_pairwise_projections requires a k=4 solution.")

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
        "All pairwise views of the binary-only four-dimensional NJW embedding\n"
        f"sigma={solution.sigma_multiplier} x median | k=4",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_eigenvalue_spectrum(
    solution: FittedSolution,
    output_name: str,
) -> None:
    values = solution.eigenvalues[: max(K_VALUES) + 1]
    positions = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.plot(positions, values, marker="o")
    for k in (2, 3, 4):
        gap = values[k - 1] - values[k]
        ax.axvline(k + 0.5, linestyle="--", linewidth=1.0)
        ax.text(k + 0.55, values[k], f"gap k={k}: {gap:.4f}")
    ax.set_xticks(positions)
    ax.set_xlabel("Ordered eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(
        "Binary-only normalized-affinity eigenvalue spectrum\n"
        f"sigma={solution.sigma_multiplier} x median"
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_nesting_heatmap(
    counts: pd.DataFrame,
    row_percentages: pd.DataFrame,
    title: str,
    output_name: str,
) -> None:
    values = row_percentages.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(values, vmin=0.0, vmax=100.0, aspect="auto")
    ax.set_xticks(range(len(row_percentages.columns)), row_percentages.columns)
    ax.set_yticks(range(len(row_percentages.index)), row_percentages.index)
    ax.set_title(title)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{int(counts.iloc[row, column])}\n({values[row, column]:.1f}%)",
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax, label="Percentage within each coarse cluster")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_ari_comparison_figure(
    comparisons: pd.DataFrame,
    output_name: str,
) -> None:
    if comparisons.empty:
        return
    ordered = comparisons.sort_values("adjusted_rand_index", ascending=True)
    fig_height = max(4.5, 0.65 * len(ordered) + 1.8)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(ordered["comparison"], ordered["adjusted_rand_index"])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Adjusted Rand Index")
    ax.set_title("Agreement between binary-only and mixed-data partitions")
    for index, value in enumerate(ordered["adjusted_rand_index"]):
        ax.text(min(float(value) + 0.015, 0.97), index, f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_contingency_heatmap(
    counts: pd.DataFrame,
    row_percentages: pd.DataFrame,
    ari: float,
    k: int,
    output_name: str,
) -> None:
    values = row_percentages.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    image = ax.imshow(values, vmin=0.0, vmax=100.0, aspect="equal")
    ax.set_xticks(range(len(counts.columns)), counts.columns)
    ax.set_yticks(range(len(counts.index)), counts.index)
    ax.set_xlabel("Binary-only cluster after label alignment")
    ax.set_ylabel("Mixed balanced cluster")
    ax.set_title(f"Binary-only versus mixed balanced | k={k} | ARI={ari:.4f}")

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{int(counts.iloc[row, column])}\n({values[row, column]:.1f}%)",
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax, label="Percentage within each mixed cluster")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 11. EXPORT AND REPORT HELPERS
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


def solution_summary_row(role: str, solution: FittedSolution) -> dict[str, object]:
    return {
        "role_in_narrative": role,
        "distance": "asymmetric_jaccard",
        "k": solution.k,
        "sigma_multiplier": solution.sigma_multiplier,
        "sigma_reference_median": solution.sigma_reference,
        "sigma": solution.sigma,
        "jaccard_silhouette_exact": solution.jaccard_silhouette,
        "spectral_silhouette_exact": solution.spectral_silhouette,
        "eigengap": solution.eigengap,
        "cluster_sizes": json.dumps(solution.cluster_sizes.tolist()),
        "valid_cluster_sizes": solution.valid_cluster_sizes,
    }


def json_compatible(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_automatic_interpretation(
    preprocessing: dict[str, object],
    selected: FittedSolution,
    k2: FittedSolution,
    k3: FittedSolution,
    k4: FittedSolution,
    comparisons: pd.DataFrame,
    mixed_source: Path | None,
) -> None:
    lines = [
        "# Binary-only asymmetric-Jaccard + NJW: automatic interpretation",
        "",
        "## 1. Data representation",
        (
            f"The analysis used **{preprocessing['n_objects']} observations** and "
            f"**{preprocessing['binary_features_used']} retained binary descriptors**. "
            "Joint zeros were ignored, so the pairwise dissimilarity is the "
            "asymmetric Jaccard dissimilarity."
        ),
        (
            f"The mean number of active descriptors per observation was "
            f"{preprocessing['mean_ones_per_object']:.2f}, with median "
            f"{preprocessing['median_ones_per_object']:.2f}. Activity was excluded "
            "from distance construction, parameter selection, and clustering."
        ),
        "",
        "## 2. Representative binary-only solution",
        (
            f"The selected binary-only configuration used k={selected.k} and "
            f"sigma={selected.sigma_multiplier} times the median positive Jaccard "
            f"dissimilarity. Cluster sizes were {selected.cluster_sizes.tolist()}, "
            f"Jaccard silhouette was {selected.jaccard_silhouette:.4f}, spectral "
            f"silhouette was {selected.spectral_silhouette:.4f}, and eigengap was "
            f"{selected.eigengap:.6f}."
        ),
        "",
        "## 3. Resolution analysis",
        (
            f"At the selected binary bandwidth, k=2 produced "
            f"{k2.cluster_sizes.tolist()}, k=3 produced {k3.cluster_sizes.tolist()}, "
            f"and k=4 produced {k4.cluster_sizes.tolist()}. The nesting tables show "
            "whether the higher-resolution solutions subdivide the same binary "
            "macrostructure."
        ),
        "",
        "## 4. Comparison with the mixed-data baseline",
    ]

    if comparisons.empty:
        lines.extend([
            (
                "No mixed-label file was found, so the binary analysis completed "
                "without cross-representation ARI comparisons."
            ),
            (
                "Expected mixed labels are normally stored in "
                "`reports/njw_mixed_baseline_k234_v2/tables/"
                "10_mixed_baseline_labels.csv`."
            ),
        ])
    else:
        lines.append(f"Mixed labels source: `{mixed_source}`.")
        lines.append("")
        for row in comparisons.itertuples(index=False):
            lines.append(
                f"- {row.comparison}: ARI={row.adjusted_rand_index:.4f} "
                f"(binary k={row.binary_k}, mixed k={row.mixed_k})."
            )
        lines.extend([
            "",
            (
                "ARI measures agreement between memberships and is the appropriate "
                "cross-representation comparison. Jaccard and mixed-Gower silhouette "
                "magnitudes must not be ranked directly because they refer to "
                "different dissimilarity geometries."
            ),
        ])

    AUTO_INTERPRETATION_MD.write_text("\n".join(lines), encoding="utf-8")


def verify_required_figures() -> None:
    required = [
        "01_binary_model_selection.png",
        "02_binary_selected_embedding.png",
        "03_binary_k2_embedding.png",
        "04_binary_k3_embedding_2d.png",
        "05_binary_k3_embedding_3d.png",
        "06_binary_k4_embedding_2d.png",
        "07_binary_k4_all_pairs.png",
        "08_binary_eigenvalue_spectrum.png",
        "09_binary_k2_vs_k3_nesting.png",
        "10_binary_k2_vs_k4_nesting.png",
    ]
    missing = [name for name in required if not (FIGURE_DIR / name).exists()]
    if missing:
        raise RuntimeError(
            "The analysis finished but required figures are missing:\n" +
            "\n".join(f"  - {name}" for name in missing)
        )


# =============================================================================
# 12. MAIN PIPELINE
# =============================================================================


def main() -> None:
    print(f"Running script version: {SCRIPT_VERSION}")
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Project root detected: {PROJECT_DIR}")
    print(f"Binary reports will be saved in: {REPORT_DIR}")

    for directory in (REPORT_DIR, TABLE_DIR, FIGURE_DIR, COORDINATE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    configure_plot_defaults()

    # A. Load the same preprocessed dataset used by the mixed baseline.
    loaded = load_project_data()
    schema = fit_binary_schema(loaded)
    preprocessing = schema.preprocessing_summary
    pd.DataFrame([preprocessing]).to_csv(PREPROCESSING_CSV, index=False)

    print(f"Input file: {loaded.source_path}")
    print(f"Input kind: {loaded.source_kind}")
    print(f"Objects: {preprocessing['n_objects']}")
    print(f"Binary descriptors used: {preprocessing['binary_features_used']}")
    print(f"'{TARGET_COL}' is excluded from all unsupervised steps.")

    X_binary = transform_binary_features(loaded.X, schema)
    print("Computing asymmetric Jaccard dissimilarity...")
    distance = compute_jaccard_dissimilarity(X_binary)

    # B. Complete binary-only grid.
    results = run_complete_grid(distance)
    results.to_csv(GRID_RESULTS_CSV, index=False)

    ranking = rank_candidates(results)
    ranking.to_csv(RANKING_CSV, index=False)
    selected_row = ranking.iloc[0]

    # C. Exact representative solution.
    selected_solution = refit_solution(
        distance,
        sigma_multiplier=float(selected_row["sigma_multiplier"]),
        k=int(selected_row["k"]),
        seed_offset=20_000,
    )

    # D. Explicit k=2, k=3, k=4 at the selected binary bandwidth.
    resolutions = refit_shared_basis_solutions(
        distance,
        sigma_multiplier=selected_solution.sigma_multiplier,
        k_values=(2, 3, 4),
        seed_offset=30_000,
    )
    binary_k2 = resolutions[2]
    binary_k3 = resolutions[3]
    binary_k4 = resolutions[4]

    resolution_summary = pd.DataFrame([
        solution_summary_row("binary_selected", selected_solution),
        solution_summary_row("binary_k2_macrostructure", binary_k2),
        solution_summary_row("binary_k3_intermediate_check", binary_k3),
        solution_summary_row("binary_k4_refinement", binary_k4),
    ])
    resolution_summary.to_csv(RESOLUTION_SUMMARY_CSV, index=False)

    # E. Internal nesting across binary resolutions.
    k2_k3_counts, k2_k3_rows, reordered_k3 = build_nesting_tables(
        binary_k2,
        binary_k3,
    )
    k2_k4_counts, k2_k4_rows, reordered_k4 = build_nesting_tables(
        binary_k2,
        binary_k4,
    )
    k2_k3_counts.to_csv(BINARY_K2_K3_COUNTS_CSV)
    k2_k3_rows.to_csv(BINARY_K2_K3_ROWS_CSV)
    k2_k4_counts.to_csv(BINARY_K2_K4_COUNTS_CSV)
    k2_k4_rows.to_csv(BINARY_K2_K4_ROWS_CSV)

    # F. Load and compare mixed-data labels when available.
    mixed = load_mixed_labels(PROJECT_DIR, expected_rows=len(loaded.X))
    if mixed is None:
        print(
            "WARNING: mixed baseline labels were not found. Binary-only results "
            "will still be produced, but cross-representation ARI will be skipped."
        )
    else:
        print(f"Mixed baseline labels: {mixed.source_path}")

    comparisons = build_ari_comparison_table(
        selected_solution,
        resolutions,
        mixed,
    )
    comparisons.to_csv(ARI_COMPARISON_CSV, index=False)

    aligned_binary_for_mixed: dict[int, np.ndarray] = {}
    if mixed is not None:
        for k in (2, 3, 4):
            mixed_column = f"mixed_balanced_k{k}"
            if mixed_column not in mixed.frame.columns:
                warnings.warn(
                    f"Column {mixed_column} is absent from mixed labels; "
                    f"same-k comparison for k={k} is skipped.",
                    stacklevel=2,
                )
                continue

            mixed_labels = mixed.frame[mixed_column].to_numpy(dtype=int)
            binary_solution = resolutions[k]
            counts, rows, aligned = build_mixed_binary_contingency(
                mixed_labels,
                binary_solution.labels,
                k=k,
            )
            aligned_binary_for_mixed[k] = aligned
            counts_path = TABLE_DIR / f"mixed_balanced_k{k}_vs_binary_k{k}_counts.csv"
            rows_path = TABLE_DIR / f"mixed_balanced_k{k}_vs_binary_k{k}_row_percentages.csv"
            counts.to_csv(counts_path)
            rows.to_csv(rows_path)

            ari = adjusted_rand_score(mixed_labels, binary_solution.labels)
            save_contingency_heatmap(
                counts,
                rows,
                ari=ari,
                k=k,
                output_name=f"12_mixed_vs_binary_k{k}_contingency.png",
            )

    # G. Figures in report order.
    save_model_selection_figure(
        results,
        output_name="01_binary_model_selection.png",
    )
    save_raw_vs_normalized(
        selected_solution,
        output_name="02_binary_selected_embedding.png",
    )
    save_raw_vs_normalized(
        binary_k2,
        output_name="03_binary_k2_embedding.png",
    )
    save_raw_vs_normalized(
        binary_k3,
        output_name="04_binary_k3_embedding_2d.png",
    )
    save_k3_embedding_3d(
        binary_k3,
        output_name="05_binary_k3_embedding_3d.png",
    )
    save_raw_vs_normalized(
        binary_k4,
        output_name="06_binary_k4_embedding_2d.png",
    )
    save_k4_pairwise_projections(
        binary_k4,
        output_name="07_binary_k4_all_pairs.png",
    )
    save_eigenvalue_spectrum(
        binary_k4,
        output_name="08_binary_eigenvalue_spectrum.png",
    )
    save_nesting_heatmap(
        k2_k3_counts,
        k2_k3_rows,
        title="Binary-only nesting: k=2 versus k=3",
        output_name="09_binary_k2_vs_k3_nesting.png",
    )
    save_nesting_heatmap(
        k2_k4_counts,
        k2_k4_rows,
        title="Binary-only nesting: k=2 versus k=4",
        output_name="10_binary_k2_vs_k4_nesting.png",
    )
    save_ari_comparison_figure(
        comparisons,
        output_name="11_binary_vs_mixed_ari.png",
    )

    # H. Coordinates and labels.
    save_coordinates(selected_solution, "binary_selected_coordinates")
    save_coordinates(binary_k2, "binary_k2_coordinates")
    save_coordinates(binary_k3, "binary_k3_coordinates")
    save_coordinates(binary_k4, "binary_k4_coordinates")

    labels_frame = pd.DataFrame({
        "row_index": np.arange(len(loaded.X)),
        "binary_selected": selected_solution.labels,
        "binary_k2": binary_k2.labels,
        "binary_k3": reordered_k3,
        "binary_k4": reordered_k4,
    })

    for k, aligned in aligned_binary_for_mixed.items():
        labels_frame[f"binary_k{k}_aligned_to_mixed"] = aligned

    if mixed is not None:
        for column in mixed.frame.columns:
            if column.startswith("mixed_"):
                labels_frame[column] = mixed.frame[column].to_numpy()

    if loaded.activity is not None:
        labels_frame[TARGET_COL] = loaded.activity.to_numpy()

    labels_frame.to_csv(BINARY_LABELS_CSV, index=False)

    # I. Machine-readable report values and interpretation.
    report_values = {
        "script_version": SCRIPT_VERSION,
        "preprocessing": preprocessing,
        "binary_selected": solution_summary_row(
            "binary_selected", selected_solution
        ),
        "binary_k2": solution_summary_row(
            "binary_k2_macrostructure", binary_k2
        ),
        "binary_k3": solution_summary_row(
            "binary_k3_intermediate_check", binary_k3
        ),
        "binary_k4": solution_summary_row(
            "binary_k4_refinement", binary_k4
        ),
        "mixed_labels_source": None if mixed is None else str(mixed.source_path),
        "binary_vs_mixed_comparisons": comparisons.to_dict(orient="records"),
        "important_interpretation_rule": (
            "Use ARI and contingency tables for binary-versus-mixed comparison. "
            "Do not rank Jaccard and mixed-Gower silhouette magnitudes directly."
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
        selected_solution,
        binary_k2,
        binary_k3,
        binary_k4,
        comparisons,
        None if mixed is None else mixed.source_path,
    )

    verify_required_figures()

    print("\n" + "=" * 78)
    print("BINARY-ONLY ANALYSIS COMPLETED")
    print("=" * 78)
    print(f"Reports directory: {REPORT_DIR}")
    print(f"Figures directory: {FIGURE_DIR}")
    print(f"Tables directory: {TABLE_DIR}")
    print(f"Coordinates directory: {COORDINATE_DIR}")
    print("\nVerified core figures:")
    for path in sorted(FIGURE_DIR.glob("*.png")):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
