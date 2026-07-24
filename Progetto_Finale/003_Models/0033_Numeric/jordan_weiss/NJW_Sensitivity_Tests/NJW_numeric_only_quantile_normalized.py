from __future__ import annotations

"""Numeric-only quantile sensitivity analysis with Ng-Jordan-Weiss.

Expected location::

    Dmml_project/Molecular-Bioresponse/PROGETTO_FINALE/
    003_Models/0033_Numeric/jordan_weiss/NJW_Sensitivity_Tests/
    NJW_numeric_only_quantile_normalized.py

The script reads ``000_Dataset/train_numeric_only.csv``, transforms every
retained numerical descriptor independently through its empirical quantile
function to Uniform[0,1], computes the mean L1 dissimilarity in quantile space,
and runs the Ng-Jordan-Weiss spectral-clustering procedure.

This is a nonlinear sensitivity analysis and does not replace the min-max
numeric baseline. Activity is excluded from transformation, distance
construction, parameter selection and clustering; when available, it is added
only to the exported labels for post-hoc analysis.
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
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import QuantileTransformer


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

SCRIPT_VERSION = "numeric-only-quantile-final-2026-07-24"
RANDOM_STATE = 42
TARGET_COL = "Activity"

K_VALUES = tuple(range(2, 11))
SIGMA_MULTIPLIERS = (0.5, 1.0, 2.0)
KMEANS_N_INIT = 50
MIN_CLUSTER_FRACTION = 0.05
DISTANCE_DTYPE = np.float32
DISTANCE_BLOCK_ROWS = 256
EIGEN_TOLERANCE = 1e-6
EIGEN_MAX_ITERATIONS = 5000
FIGURE_DPI = 220
GRID_SILHOUETTE_SAMPLE_SIZE: int | None = None

SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# 2. PROJECT AND OUTPUT PATHS
# =============================================================================


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "000_Dataset").is_dir() and (
            candidate / "003_Models"
        ).is_dir():
            return candidate
    raise FileNotFoundError(
        "Project root not found. Expected an ancestor containing "
        "'000_Dataset' and '003_Models'."
    )


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
DATASET_DIR = PROJECT_ROOT / "000_Dataset"
FEATURES_PATH = DATASET_DIR / "train_numeric_only.csv"
FILTERED_FEATURES_PATH = DATASET_DIR / "train_filtered_no_activity.csv"
ACTIVITY_PATH = DATASET_DIR / "train_activity_target.csv"
RAW_TRAIN_PATH = DATASET_DIR / "raw" / "train.csv"

REPORT_DIR = SCRIPT_DIR / "Report_njw_numeric_only_quantile_normalized"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"
COORDINATE_DIR = REPORT_DIR / "coordinates"

PREPROCESSING_CSV = TABLE_DIR / "01_numeric_quantile_preprocessing_summary.csv"
GRID_RESULTS_CSV = TABLE_DIR / "02_numeric_quantile_grid_results.csv"
SELECTED_SOLUTIONS_CSV = TABLE_DIR / "03_numeric_quantile_selected_solutions.csv"
ARI_COMPARISON_CSV = TABLE_DIR / "04_numeric_quantile_cross_representation_ari.csv"
NUMERIC_K2_K3_COUNTS_CSV = TABLE_DIR / "05_numeric_quantile_k2_vs_k3_counts.csv"
NUMERIC_K2_K3_ROWS_CSV = TABLE_DIR / "06_numeric_quantile_k2_vs_k3_row_percentages.csv"
NUMERIC_K2_K4_COUNTS_CSV = TABLE_DIR / "07_numeric_quantile_k2_vs_k4_counts.csv"
NUMERIC_K2_K4_ROWS_CSV = TABLE_DIR / "08_numeric_quantile_k2_vs_k4_row_percentages.csv"
NUMERIC_LABELS_CSV = TABLE_DIR / "10_numeric_quantile_labels.csv"
REPORT_VALUES_JSON = TABLE_DIR / "11_numeric_quantile_report_values.json"
RESULTS_SUMMARY_MD = REPORT_DIR / "results_summary.md"

MIXED_LABELS_PATH = (
    PROJECT_ROOT / "003_Models" / "0031_Mixed" / "jordan_weiss"
    / "report_jordan_weiss" / "tables" / "08_mixed_baseline_labels.csv"
)
BINARY_LABELS_PATH = (
    PROJECT_ROOT / "003_Models" / "0032_Binary" / "jordan_weiss"
    / "Report_NJW_binary_only" / "tables" / "10_binary_labels.csv"
)
BASELINE_NUMERIC_LABELS_PATH = (
    PROJECT_ROOT / "003_Models" / "0033_Numeric" / "jordan_weiss"
    / "Report_njw_numeric_only" / "tables" / "10_numeric_labels.csv"
)


# =============================================================================
# 3. DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class LoadedData:
    X: pd.DataFrame
    activity: pd.Series | None


@dataclass(frozen=True)
class NumericSchema:
    numeric_features: list[str]
    zero_range_numeric_removed: list[str]
    feature_minima: np.ndarray
    feature_ranges: np.ndarray
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
    quantile_l1_silhouette: float
    spectral_silhouette: float
    eigengap: float
    valid_cluster_sizes: bool


@dataclass(frozen=True)
class ReferenceLabels:
    name: str
    frame: pd.DataFrame
    source_path: Path


# =============================================================================
# 4. INPUT VALIDATION AND NUMERICAL PREPROCESSING
# =============================================================================


def read_csv_clean(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    unnamed = [column for column in frame.columns if str(column).startswith("Unnamed:")]
    return frame.drop(columns=unnamed) if unnamed else frame


def validate_feature_frame(X: pd.DataFrame) -> None:
    if X.empty:
        raise ValueError("The feature matrix is empty.")
    if TARGET_COL in X.columns:
        raise ValueError(f"{FEATURES_PATH.name} must not contain '{TARGET_COL}'.")

    non_numeric = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    if non_numeric:
        raise TypeError(f"Non-numeric descriptors found: {non_numeric[:10]}")
    if X.isna().any().any():
        missing = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Missing values found in descriptors: {missing[:10]}")
    if not np.isfinite(X.to_numpy(dtype=np.float64, copy=False)).all():
        raise ValueError("The feature matrix contains non-finite values.")

    duplicates = int(X.duplicated().sum())
    if duplicates:
        warnings.warn(
            f"{duplicates} duplicate feature rows were found and retained to "
            "preserve row alignment.",
            stacklevel=2,
        )


def assert_feature_alignment(
    numeric: pd.DataFrame,
    reference: pd.DataFrame,
    reference_name: str,
) -> None:
    if TARGET_COL in reference.columns:
        reference = reference.drop(columns=TARGET_COL)
    if len(numeric) != len(reference):
        raise ValueError(
            f"{FEATURES_PATH.name} and {reference_name} have different row counts: "
            f"{len(numeric)} versus {len(reference)}."
        )

    missing = [column for column in numeric.columns if column not in reference.columns]
    if missing:
        raise ValueError(
            f"Some numerical descriptors are absent from {reference_name}: {missing[:10]}"
        )

    block_size = 128
    for start in range(0, numeric.shape[1], block_size):
        columns = numeric.columns[start : start + block_size]
        left = numeric.loc[:, columns].to_numpy(dtype=np.float64, copy=False)
        right = reference.loc[:, columns].to_numpy(dtype=np.float64, copy=False)
        if not np.allclose(left, right, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Row or value alignment failed between {FEATURES_PATH.name} "
                f"and {reference_name}."
            )


def load_activity(expected_rows: int) -> pd.Series | None:
    activity: pd.Series | None = None
    if ACTIVITY_PATH.exists():
        frame = read_csv_clean(ACTIVITY_PATH)
        if TARGET_COL in frame.columns:
            activity = frame[TARGET_COL].reset_index(drop=True)
        elif frame.shape[1] == 1:
            activity = frame.iloc[:, 0].reset_index(drop=True)
            activity.name = TARGET_COL
        else:
            raise ValueError(f"Could not identify Activity in {ACTIVITY_PATH}.")
        if len(activity) != expected_rows:
            raise ValueError(
                f"{ACTIVITY_PATH.name} has {len(activity)} rows instead of {expected_rows}."
            )

    if RAW_TRAIN_PATH.exists():
        raw = read_csv_clean(RAW_TRAIN_PATH)
        if len(raw) != expected_rows:
            raise ValueError(
                f"{RAW_TRAIN_PATH.name} has {len(raw)} rows instead of {expected_rows}."
            )
        if TARGET_COL in raw.columns:
            raw_activity = raw[TARGET_COL].reset_index(drop=True)
            if activity is None:
                activity = raw_activity
            elif not np.array_equal(activity.to_numpy(), raw_activity.to_numpy()):
                raise ValueError(
                    "Activity mismatch between train_activity_target.csv and raw/train.csv."
                )
    return activity


def load_project_data() -> LoadedData:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Numerical feature file not found: {FEATURES_PATH}")

    X = read_csv_clean(FEATURES_PATH).reset_index(drop=True)
    validate_feature_frame(X)

    if FILTERED_FEATURES_PATH.exists():
        assert_feature_alignment(
            X,
            read_csv_clean(FILTERED_FEATURES_PATH).reset_index(drop=True),
            FILTERED_FEATURES_PATH.name,
        )
    elif RAW_TRAIN_PATH.exists():
        assert_feature_alignment(
            X,
            read_csv_clean(RAW_TRAIN_PATH).reset_index(drop=True),
            "raw/train.csv",
        )

    return LoadedData(X=X, activity=load_activity(len(X)))


def fit_numeric_schema(data: LoadedData) -> NumericSchema:
    X = data.X
    ranges = X.max(axis=0) - X.min(axis=0)
    zero_range = ranges[ranges <= 0].index.tolist()
    numeric = [column for column in X.columns if column not in set(zero_range)]
    if not numeric:
        raise ValueError("All numerical descriptors have zero range.")

    values = X[numeric].to_numpy(dtype=np.float64, copy=False)
    used_ranges = ranges[numeric].to_numpy(dtype=np.float64)
    standard_deviations = values.std(axis=0)

    summary = {
        "script_version": SCRIPT_VERSION,
        "analysis_role": "numeric-only quantile sensitivity analysis",
        "input_path": str(FEATURES_PATH),
        "n_objects": len(X),
        "features_in_input_file": X.shape[1],
        "numeric_features_used": len(numeric),
        "zero_range_numeric_removed": len(zero_range),
        "normalization_method": "feature-wise empirical quantile to Uniform[0,1]",
        "quantile_n_quantiles": min(1000, len(X)),
        "quantile_subsampling": "disabled: all objects used",
        "mean_numeric_standard_deviation_before_transform": float(standard_deviations.mean()),
        "median_numeric_standard_deviation_before_transform": float(np.median(standard_deviations)),
        "minimum_numeric_range": float(used_ranges.min()),
        "median_numeric_range": float(np.median(used_ranges)),
        "maximum_numeric_range": float(used_ranges.max()),
        "fraction_values_le_0_01_before_transform": float(np.mean(values <= 0.01)),
        "fraction_values_le_0_05_before_transform": float(np.mean(values <= 0.05)),
        "fraction_values_le_0_10_before_transform": float(np.mean(values <= 0.10)),
        "median_of_feature_medians_before_transform": float(np.median(np.median(values, axis=0))),
        "activity_available_for_post_hoc_only": data.activity is not None,
        "silhouettes_exact": GRID_SILHOUETTE_SAMPLE_SIZE is None,
    }

    return NumericSchema(
        numeric_features=numeric,
        zero_range_numeric_removed=zero_range,
        feature_minima=X[numeric].min(axis=0).to_numpy(dtype=np.float64),
        feature_ranges=used_ranges,
        preprocessing_summary=summary,
    )


def transform_numeric_features(X: pd.DataFrame, schema: NumericSchema) -> np.ndarray:
    values = X[schema.numeric_features].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite values found before quantile transformation.")

    transformer = QuantileTransformer(
        n_quantiles=min(1000, len(values)),
        output_distribution="uniform",
        subsample=len(values),
        random_state=RANDOM_STATE,
        copy=True,
    )
    transformed = transformer.fit_transform(values)
    transformed = np.clip(transformed, 0.0, 1.0)
    if not np.isfinite(transformed).all():
        raise ValueError("Quantile transformation produced non-finite values.")
    return transformed.astype(DISTANCE_DTYPE, copy=False)


# =============================================================================
# 5. QUANTILE-SPACE L1 DISSIMILARITY
# =============================================================================


def compute_quantile_l1_dissimilarity(
    X_scaled: np.ndarray,
    block_rows: int = DISTANCE_BLOCK_ROWS,
) -> np.ndarray:
    n_objects, n_features = X_scaled.shape
    if n_features == 0:
        raise ValueError("The quantile-transformed numerical matrix has zero columns.")

    distance = np.empty((n_objects, n_objects), dtype=DISTANCE_DTYPE)
    reference = np.asarray(X_scaled, dtype=np.float64)
    for start in range(0, n_objects, block_rows):
        stop = min(start + block_rows, n_objects)
        block = cdist(reference[start:stop], reference, metric="cityblock")
        block /= float(n_features)
        distance[start:stop] = block.astype(DISTANCE_DTYPE, copy=False)
        print(f"  quantile-space L1 rows {start + 1}-{stop} / {n_objects}")

    distance = ((distance + distance.T) * 0.5).astype(DISTANCE_DTYPE, copy=False)
    distance = np.clip(distance, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)

    if distance.shape[0] != distance.shape[1]:
        raise ValueError("The quantile-space L1 matrix is not square.")
    if not np.isfinite(distance).all():
        raise ValueError("The quantile-space L1 matrix contains non-finite values.")
    if not np.allclose(distance, distance.T, atol=1e-6):
        raise ValueError("The quantile-space L1 matrix is not symmetric.")
    if not np.allclose(np.diag(distance), 0.0, atol=1e-7):
        raise ValueError("The quantile-space L1 diagonal is not zero.")
    if distance.min() < -1e-7 or distance.max() > 1.0 + 1e-7:
        raise ValueError("The quantile-space L1 matrix is outside [0, 1].")

    return distance


def median_positive_distance(distance: np.ndarray) -> float:
    upper = distance[np.triu_indices(distance.shape[0], k=1)]
    positive = upper[upper > 0]
    if positive.size == 0:
        raise ValueError("All off-diagonal quantile-space L1 dissimilarities are zero.")
    return float(np.median(positive))


# =============================================================================
# 6. NG-JORDAN-WEISS SPECTRAL CLUSTERING
# =============================================================================


def gaussian_affinity(distance: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    affinity = np.exp(-(distance.astype(np.float64) ** 2) / (2.0 * sigma ** 2))
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
        return inv_sqrt_degree[:, None] * (affinity @ (inv_sqrt_degree[:, None] * matrix))

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
                "ARPACK did not converge to the requested eigenpairs."
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


def build_spectral_basis(distance: np.ndarray, sigma_multiplier: float, seed: int) -> SpectralBasis:
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
        quantile_l1_silhouette=compute_silhouette(
            distance, labels, metric="precomputed", sample_size=sample_size, seed=seed
        ),
        spectral_silhouette=compute_silhouette(
            embedding, labels, metric="euclidean", sample_size=sample_size, seed=seed
        ),
        eigengap=float(basis.eigenvalues[k - 1] - basis.eigenvalues[k]),
        valid_cluster_sizes=bool(np.all(sizes >= minimum_required)),
    )


# =============================================================================
# 7. GRID SEARCH AND SELECTION
# =============================================================================


def solution_to_row(solution: FittedSolution) -> dict[str, object]:
    minimum_required = int(math.ceil(MIN_CLUSTER_FRACTION * solution.labels.size))
    return {
        "distance": "quantile_uniform_mean_l1",
        "sigma_reference_median": solution.sigma_reference,
        "sigma_multiplier": solution.sigma_multiplier,
        "sigma": solution.sigma,
        "k": solution.k,
        "quantile_l1_silhouette": solution.quantile_l1_silhouette,
        "spectral_silhouette": solution.spectral_silhouette,
        "eigengap": solution.eigengap,
        "valid_cluster_sizes": solution.valid_cluster_sizes,
        "minimum_required_cluster_size": minimum_required,
        "minimum_cluster_size": int(solution.cluster_sizes.min()),
        "maximum_cluster_size": int(solution.cluster_sizes.max()),
        "cluster_sizes": json.dumps(solution.cluster_sizes.tolist()),
        "silhouettes_exact": GRID_SILHOUETTE_SAMPLE_SIZE is None,
    }


def run_complete_grid(distance: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    print(f"Median positive quantile-space L1 dissimilarity: {median_positive_distance(distance):.6f}")

    for sigma_index, multiplier in enumerate(SIGMA_MULTIPLIERS):
        basis_seed = RANDOM_STATE + 1000 * sigma_index
        basis = build_spectral_basis(distance, sigma_multiplier=multiplier, seed=basis_seed)

        print("\n" + "=" * 78)
        print(f"NUMERIC-ONLY QUANTILE | sigma={multiplier} x median")
        print("=" * 78)

        for k in K_VALUES:
            solution = fit_k_from_basis(
                basis, distance, k=k, seed=basis_seed + k, exact_silhouettes=False
            )
            rows.append(solution_to_row(solution))
            print(
                f"k={k:>2} | sil_quantileL1={solution.quantile_l1_silhouette: .4f} | "
                f"sil_spectral={solution.spectral_silhouette: .4f} | "
                f"gap={solution.eigengap: .6f} | sizes={solution.cluster_sizes.tolist()}"
            )

        del basis
        gc.collect()

    return pd.DataFrame(rows)


def rank_candidates(results: pd.DataFrame) -> pd.DataFrame:
    return results.sort_values(
        by=[
            "valid_cluster_sizes",
            "quantile_l1_silhouette",
            "eigengap",
            "spectral_silhouette",
            "k",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def refit_solution(distance: np.ndarray, sigma_multiplier: float, k: int) -> FittedSolution:
    sigma_index = SIGMA_MULTIPLIERS.index(float(sigma_multiplier))
    basis_seed = RANDOM_STATE + 1000 * sigma_index
    basis = build_spectral_basis(distance, sigma_multiplier=float(sigma_multiplier), seed=basis_seed)
    solution = fit_k_from_basis(
        basis,
        distance,
        k=int(k),
        seed=basis_seed + int(k),
        exact_silhouettes=True,
    )
    del basis
    gc.collect()
    return solution


def refit_shared_basis_solutions(distance: np.ndarray, sigma_multiplier: float, k_values: Iterable[int]) -> dict[int, FittedSolution]:
    sigma_index = SIGMA_MULTIPLIERS.index(float(sigma_multiplier))
    basis_seed = RANDOM_STATE + 1000 * sigma_index
    basis = build_spectral_basis(distance, sigma_multiplier=float(sigma_multiplier), seed=basis_seed)
    solutions: dict[int, FittedSolution] = {}
    for k in sorted(set(int(value) for value in k_values)):
        solutions[k] = fit_k_from_basis(
            basis,
            distance,
            k=k,
            seed=basis_seed + k,
            exact_silhouettes=True,
        )
    del basis
    gc.collect()
    return solutions


# =============================================================================
# 8. REFERENCE LABELS AND COMPARISONS
# =============================================================================


def load_reference_labels(name: str, path: Path, expected_rows: int) -> ReferenceLabels | None:
    if not path.exists():
        return None
    frame = read_csv_clean(path).reset_index(drop=True)
    if len(frame) != expected_rows:
        raise ValueError(f"{name} labels have {len(frame)} rows instead of {expected_rows}: {path}")
    if "row_index" in frame.columns:
        expected = np.arange(expected_rows)
        actual = frame["row_index"].to_numpy()
        if not np.array_equal(actual, expected):
            raise ValueError(f"The row_index column is not aligned in {path}")
    return ReferenceLabels(name=name, frame=frame, source_path=path)


def add_ari_row(rows: list[dict[str, object]], comparison: str, numeric_name: str, numeric_labels: np.ndarray, reference_name: str, reference_labels: np.ndarray) -> None:
    rows.append({
        "comparison": comparison,
        "numeric_partition": numeric_name,
        "reference_partition": reference_name,
        "numeric_k": int(np.unique(numeric_labels).size),
        "reference_k": int(np.unique(reference_labels).size),
        "same_resolution": int(np.unique(numeric_labels).size) == int(np.unique(reference_labels).size),
        "adjusted_rand_index": float(adjusted_rand_score(numeric_labels, reference_labels)),
    })


def build_ari_comparison_table(
    numeric_selected: FittedSolution,
    numeric_resolutions: dict[int, FittedSolution],
    mixed: ReferenceLabels | None,
    binary: ReferenceLabels | None,
    baseline_numeric: ReferenceLabels | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if mixed is not None:
        for column in ("mixed_classical_selected", "mixed_balanced_selected"):
            if column in mixed.frame.columns:
                add_ari_row(
                    rows,
                    f"numeric_quantile_selected_vs_{column}",
                    "numeric_quantile_selected",
                    numeric_selected.labels,
                    column,
                    mixed.frame[column].to_numpy(dtype=int),
                )
        for k in (2, 3, 4):
            column = f"mixed_balanced_k{k}"
            if column in mixed.frame.columns:
                add_ari_row(
                    rows,
                    f"numeric_quantile_k{k}_vs_mixed_balanced_k{k}",
                    f"numeric_quantile_k{k}",
                    numeric_resolutions[k].labels,
                    column,
                    mixed.frame[column].to_numpy(dtype=int),
                )

    if binary is not None:
        if "binary_selected" in binary.frame.columns:
            add_ari_row(
                rows,
                "numeric_quantile_selected_vs_binary_selected",
                "numeric_quantile_selected",
                numeric_selected.labels,
                "binary_selected",
                binary.frame["binary_selected"].to_numpy(dtype=int),
            )
        for k in (2, 3, 4):
            column = f"binary_k{k}"
            if column in binary.frame.columns:
                add_ari_row(
                    rows,
                    f"numeric_quantile_k{k}_vs_binary_k{k}",
                    f"numeric_quantile_k{k}",
                    numeric_resolutions[k].labels,
                    column,
                    binary.frame[column].to_numpy(dtype=int),
                )

    if baseline_numeric is not None:
        if "numeric_selected" in baseline_numeric.frame.columns:
            add_ari_row(
                rows,
                "numeric_quantile_selected_vs_numeric_selected",
                "numeric_quantile_selected",
                numeric_selected.labels,
                "numeric_selected",
                baseline_numeric.frame["numeric_selected"].to_numpy(dtype=int),
            )
        for k in (2, 3, 4):
            column = f"numeric_k{k}"
            if column in baseline_numeric.frame.columns:
                add_ari_row(
                    rows,
                    f"numeric_quantile_k{k}_vs_numeric_k{k}",
                    f"numeric_quantile_k{k}",
                    numeric_resolutions[k].labels,
                    column,
                    baseline_numeric.frame[column].to_numpy(dtype=int),
                )

    return pd.DataFrame(rows, columns=[
        "comparison", "numeric_partition", "reference_partition",
        "numeric_k", "reference_k", "same_resolution", "adjusted_rand_index",
    ])


def align_candidate_labels_to_reference(reference_labels: np.ndarray, candidate_labels: np.ndarray) -> np.ndarray:
    reference_values = np.sort(np.unique(reference_labels))
    candidate_values = np.sort(np.unique(candidate_labels))
    contingency = pd.crosstab(reference_labels, candidate_labels).reindex(
        index=reference_values,
        columns=candidate_values,
        fill_value=0,
    )
    row_ind, col_ind = linear_sum_assignment(-contingency.to_numpy())
    mapping = {int(candidate_values[column]): int(reference_values[row]) for row, column in zip(row_ind, col_ind)}
    next_new_target = int(reference_values.max()) + 1 if reference_values.size else 0
    for candidate in candidate_values:
        candidate_int = int(candidate)
        if candidate_int not in mapping:
            mapping[candidate_int] = next_new_target
            next_new_target += 1
    return np.array([mapping[int(label)] for label in candidate_labels], dtype=int)


def build_reference_numeric_contingency(reference_labels: np.ndarray, numeric_labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    aligned_numeric = align_candidate_labels_to_reference(reference_labels, numeric_labels)
    counts = pd.crosstab(
        reference_labels,
        aligned_numeric,
        rownames=["reference_cluster"],
        colnames=["numeric_cluster"],
    )
    counts.index = [f"reference_cluster_{value}" for value in counts.index]
    counts.columns = [f"numeric_cluster_{value}" for value in counts.columns]
    row_percentages = counts.div(counts.sum(axis=1), axis=0) * 100.0
    return counts, row_percentages, aligned_numeric


# =============================================================================
# 9. INTERNAL RESOLUTION NESTING
# =============================================================================


def reorder_fine_clusters_by_coarse(coarse_labels: np.ndarray, fine_labels: np.ndarray) -> np.ndarray:
    table = pd.crosstab(coarse_labels, fine_labels)
    ordered_columns: list[int] = []
    for coarse_cluster in table.index:
        columns = table.loc[coarse_cluster].sort_values(ascending=False).index.tolist()
        for column in columns:
            if int(column) not in ordered_columns:
                ordered_columns.append(int(column))
    mapping = {old: new for new, old in enumerate(ordered_columns)}
    return np.array([mapping[int(label)] for label in fine_labels], dtype=int)


def build_nesting_tables(coarse_solution: FittedSolution, fine_solution: FittedSolution) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    reordered = reorder_fine_clusters_by_coarse(coarse_solution.labels, fine_solution.labels)
    counts = pd.crosstab(
        coarse_solution.labels,
        reordered,
        rownames=[f"k{coarse_solution.k}_macrocluster"],
        colnames=[f"k{fine_solution.k}_subcluster"],
    )
    counts.index = [f"k{coarse_solution.k}_cluster_{value}" for value in counts.index]
    counts.columns = [f"k{fine_solution.k}_cluster_{value}" for value in counts.columns]
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
        ("quantile_l1_silhouette", "Quantile-L1 silhouette"),
        ("eigengap", "Eigengap"),
        ("spectral_silhouette", "Spectral silhouette"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, (metric, label) in zip(axes, metrics):
        for multiplier in SIGMA_MULTIPLIERS:
            curve = results[results["sigma_multiplier"] == multiplier].sort_values("k")
            ax.plot(curve["k"], curve[metric], marker="o", linewidth=1.5, label=f"sigma = {multiplier} x median")
        ax.set_xticks(K_VALUES)
        ax.set_xlabel("Number of clusters k")
        ax.set_ylabel(label)
        ax.set_title(label)
    axes[0].legend()
    fig.suptitle("Numeric-only quantile model-selection diagnostics", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_raw_vs_normalized(solution: FittedSolution, output_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(solution.raw_eigenvectors[:, 0], solution.raw_eigenvectors[:, 1], c=solution.labels, s=10, alpha=0.65)
    axes[0].set_title("Before NJW row normalization")
    axes[0].set_xlabel("Eigenvector 1")
    axes[0].set_ylabel("Eigenvector 2")
    axes[1].scatter(solution.embedding[:, 0], solution.embedding[:, 1], c=solution.labels, s=10, alpha=0.65)
    axes[1].set_title("Final row-normalized NJW embedding")
    axes[1].set_xlabel("Normalized coordinate 1")
    axes[1].set_ylabel("Normalized coordinate 2")
    fig.suptitle(
        f"Numeric-only quantile + NJW | k={solution.k} | sigma={solution.sigma_multiplier} x median\n"
        f"quantile-L1 silhouette={solution.quantile_l1_silhouette:.4f}; "
        f"spectral silhouette={solution.spectral_silhouette:.4f}; eigengap={solution.eigengap:.5f}; "
        f"sizes={solution.cluster_sizes.tolist()}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_k3_embedding_3d(solution: FittedSolution, output_name: str) -> None:
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(solution.embedding[:, 0], solution.embedding[:, 1], solution.embedding[:, 2], c=solution.labels, s=10, alpha=0.65)
    ax.set_xlabel("Coordinate 1")
    ax.set_ylabel("Coordinate 2")
    ax.set_zlabel("Coordinate 3")
    ax.set_title("Numeric-only quantile three-dimensional NJW embedding | k=3")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_k4_pairwise_projections(solution: FittedSolution, output_name: str) -> None:
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (first, second) in zip(axes.ravel(), pairs):
        ax.scatter(solution.embedding[:, first], solution.embedding[:, second], c=solution.labels, s=8, alpha=0.58)
        ax.set_xlabel(f"Coordinate {first + 1}")
        ax.set_ylabel(f"Coordinate {second + 1}")
        ax.set_title(f"Coordinates {first + 1} and {second + 1}")
    fig.suptitle(
        "All pairwise views of the numeric-only quantile four-dimensional NJW embedding\n"
        f"sigma={solution.sigma_multiplier} x median | k=4",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_eigenvalue_spectrum(solution: FittedSolution, output_name: str) -> None:
    values = solution.eigenvalues[:max(K_VALUES) + 1]
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
    ax.set_title("Numeric-only quantile normalized-affinity eigenvalue spectrum\n" f"sigma={solution.sigma_multiplier} x median")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_nesting_heatmap(counts: pd.DataFrame, row_percentages: pd.DataFrame, title: str, output_name: str) -> None:
    values = row_percentages.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(values, vmin=0.0, vmax=100.0, aspect="auto")
    ax.set_xticks(range(len(row_percentages.columns)), row_percentages.columns)
    ax.set_yticks(range(len(row_percentages.index)), row_percentages.index)
    ax.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{int(counts.iloc[row, column])}\n({values[row, column]:.1f}%)", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Percentage within each coarse cluster")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_ari_comparison_figure(comparisons: pd.DataFrame, output_name: str) -> None:
    if comparisons.empty:
        return
    ordered = comparisons.sort_values("adjusted_rand_index", ascending=True)
    fig_height = max(5.0, 0.55 * len(ordered) + 2.0)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.barh(ordered["comparison"], ordered["adjusted_rand_index"])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Adjusted Rand Index")
    ax.set_title("Agreement of numeric-quantile partitions with mixed and binary partitions")
    for index, value in enumerate(ordered["adjusted_rand_index"]):
        ax.text(min(float(value) + 0.015, 0.97), index, f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_contingency_heatmap(counts: pd.DataFrame, row_percentages: pd.DataFrame, ari: float, k: int, reference_display_name: str, output_name: str) -> None:
    values = row_percentages.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    image = ax.imshow(values, vmin=0.0, vmax=100.0, aspect="equal")
    ax.set_xticks(range(len(counts.columns)), counts.columns)
    ax.set_yticks(range(len(counts.index)), counts.index)
    ax.set_xlabel("Numeric-only cluster after label alignment")
    ax.set_ylabel(f"{reference_display_name} cluster")
    ax.set_title(f"Numeric-quantile versus {reference_display_name} | k={k} | ARI={ari:.4f}")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{int(counts.iloc[row, column])}\n({values[row, column]:.1f}%)", ha="center", va="center")
    fig.colorbar(image, ax=ax, label=f"Percentage within each {reference_display_name} cluster")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / output_name, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 11. EXPORT AND SUMMARY HELPERS
# =============================================================================


def save_coordinates(solution: FittedSolution, stem: str) -> None:
    data: dict[str, np.ndarray] = {f"raw_eigenvector_{index + 1}": solution.raw_eigenvectors[:, index] for index in range(solution.k)}
    data.update({f"normalized_coordinate_{index + 1}": solution.embedding[:, index] for index in range(solution.k)})
    data["cluster"] = solution.labels
    pd.DataFrame(data).to_csv(COORDINATE_DIR / f"{stem}.csv", index=False)


def solution_summary_row(role: str, solution: FittedSolution) -> dict[str, object]:
    return {
        "role_in_narrative": role,
        "distance": "quantile_uniform_mean_l1",
        "k": solution.k,
        "sigma_multiplier": solution.sigma_multiplier,
        "sigma_reference_median": solution.sigma_reference,
        "sigma": solution.sigma,
        "quantile_l1_silhouette_exact": solution.quantile_l1_silhouette,
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


def write_results_summary(
    preprocessing: dict[str, object],
    selected: FittedSolution,
    k2: FittedSolution,
    k3: FittedSolution,
    k4: FittedSolution,
    comparisons: pd.DataFrame,
    mixed_source: Path | None,
    binary_source: Path | None,
    baseline_numeric_source: Path | None,
) -> None:
    lines = [
        "# Numeric-only quantile NJW results summary",
        "",
        "## Data representation",
        f"- Objects: {preprocessing['n_objects']}",
        f"- Numerical descriptors used: {preprocessing['numeric_features_used']}",
        f"- Input file: `{preprocessing['input_path']}`",
        "- Numerical preprocessing: feature-wise empirical quantile transformation to Uniform[0,1]",
        "- Distance: mean L1 dissimilarity in quantile-transformed space",
        "- Analysis role: sensitivity analysis; it does not replace the min-max numeric baseline",
        "- Activity: excluded from all unsupervised steps and used only post-hoc in the exported labels",
        "",
        "## Selected numeric-quantile solution",
        f"- k = {selected.k}",
        f"- sigma multiplier = {selected.sigma_multiplier}",
        f"- cluster sizes = {selected.cluster_sizes.tolist()}",
        f"- quantile-L1 silhouette = {selected.quantile_l1_silhouette:.4f}",
        f"- spectral silhouette = {selected.spectral_silhouette:.4f}",
        f"- eigengap = {selected.eigengap:.6f}",
        "",
        "## Explicit resolutions at the selected bandwidth",
        f"- k=2 sizes = {k2.cluster_sizes.tolist()}",
        f"- k=3 sizes = {k3.cluster_sizes.tolist()}",
        f"- k=4 sizes = {k4.cluster_sizes.tolist()}",
        "",
        "## External references used",
        f"- Mixed labels: `{mixed_source}`" if mixed_source is not None else "- Mixed labels: not available",
        f"- Binary labels: `{binary_source}`" if binary_source is not None else "- Binary labels: not available",
        f"- Min-max numeric labels: `{baseline_numeric_source}`" if baseline_numeric_source is not None else "- Min-max numeric labels: not available",
        "",
        "## Cross-representation ARI",
    ]
    if comparisons.empty:
        lines.append("- No external comparison was available.")
    else:
        for row in comparisons.sort_values("adjusted_rand_index", ascending=False).itertuples(index=False):
            lines.append(f"- {row.comparison}: ARI = {row.adjusted_rand_index:.4f}")
    lines.extend([
        "",
        "## Interpretation rule",
        "- Use ARI and contingency tables for cross-representation comparisons.",
        "- Do not rank silhouettes directly across quantile-L1, min-max numerical Gower, asymmetric Jaccard and mixed Gower geometries.",
    ])
    RESULTS_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def verify_required_figures(include_external_comparison: bool) -> None:
    required = [
        "01_numeric_quantile_model_selection.png",
        "02_numeric_quantile_selected_embedding.png",
        "03_numeric_quantile_k2_embedding.png",
        "04_numeric_quantile_k3_embedding_2d.png",
        "05_numeric_quantile_k3_embedding_3d.png",
        "06_numeric_quantile_k4_embedding_2d.png",
        "07_numeric_quantile_k4_all_pairs.png",
        "08_numeric_quantile_eigenvalue_spectrum.png",
        "09_numeric_quantile_k2_vs_k3_nesting.png",
        "10_numeric_quantile_k2_vs_k4_nesting.png",
    ]
    if include_external_comparison:
        required.append("11_numeric_quantile_cross_representation_ari.png")
    missing = [name for name in required if not (FIGURE_DIR / name).exists()]
    if missing:
        raise RuntimeError("Required figures are missing:\n" + "\n".join(f"  - {name}" for name in missing))


# =============================================================================
# 12. MAIN PIPELINE
# =============================================================================


def main() -> None:
    print(f"Running script version: {SCRIPT_VERSION}")
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Project root detected: {PROJECT_ROOT}")
    print(f"Numeric-quantile reports will be saved in: {REPORT_DIR}")

    for directory in (REPORT_DIR, TABLE_DIR, FIGURE_DIR, COORDINATE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    configure_plot_defaults()

    loaded = load_project_data()
    schema = fit_numeric_schema(loaded)
    preprocessing = schema.preprocessing_summary
    pd.DataFrame([preprocessing]).to_csv(PREPROCESSING_CSV, index=False)

    print(f"Input file: {FEATURES_PATH}")
    print(f"Objects: {preprocessing['n_objects']}")
    print(f"Numerical descriptors used: {preprocessing['numeric_features_used']}")
    print(f"'{TARGET_COL}' is excluded from all unsupervised steps.")

    X_numeric = transform_numeric_features(loaded.X, schema)
    print("Computing quantile-space L1 dissimilarity...")
    distance = compute_quantile_l1_dissimilarity(X_numeric)

    results = run_complete_grid(distance)
    results.to_csv(GRID_RESULTS_CSV, index=False)

    ranking = rank_candidates(results)
    selected_row = ranking.iloc[0]
    selected_solution = refit_solution(
        distance,
        sigma_multiplier=float(selected_row["sigma_multiplier"]),
        k=int(selected_row["k"]),
    )

    resolutions = refit_shared_basis_solutions(
        distance,
        sigma_multiplier=selected_solution.sigma_multiplier,
        k_values=(2, 3, 4),
    )
    numeric_k2 = resolutions[2]
    numeric_k3 = resolutions[3]
    numeric_k4 = resolutions[4]

    selected_solutions = pd.DataFrame([
        solution_summary_row("numeric_quantile_selected", selected_solution),
        solution_summary_row("numeric_quantile_k2_macrostructure", numeric_k2),
        solution_summary_row("numeric_quantile_k3_intermediate_check", numeric_k3),
        solution_summary_row("numeric_quantile_k4_refinement", numeric_k4),
    ])
    selected_solutions.to_csv(SELECTED_SOLUTIONS_CSV, index=False)

    k2_k3_counts, k2_k3_rows, reordered_k3 = build_nesting_tables(numeric_k2, numeric_k3)
    k2_k4_counts, k2_k4_rows, reordered_k4 = build_nesting_tables(numeric_k2, numeric_k4)
    k2_k3_counts.to_csv(NUMERIC_K2_K3_COUNTS_CSV)
    k2_k3_rows.to_csv(NUMERIC_K2_K3_ROWS_CSV)
    k2_k4_counts.to_csv(NUMERIC_K2_K4_COUNTS_CSV)
    k2_k4_rows.to_csv(NUMERIC_K2_K4_ROWS_CSV)

    mixed = load_reference_labels("mixed", MIXED_LABELS_PATH, expected_rows=len(loaded.X))
    binary = load_reference_labels("binary", BINARY_LABELS_PATH, expected_rows=len(loaded.X))
    baseline_numeric = load_reference_labels(
        "baseline numeric", BASELINE_NUMERIC_LABELS_PATH, expected_rows=len(loaded.X)
    )

    if mixed is None:
        print(f"WARNING: mixed labels not found at: {MIXED_LABELS_PATH}")
    else:
        print(f"Mixed baseline labels: {mixed.source_path}")
    if binary is None:
        print(f"WARNING: binary labels not found at: {BINARY_LABELS_PATH}")
    else:
        print(f"Binary-only labels: {binary.source_path}")
    if baseline_numeric is None:
        print(f"WARNING: baseline numeric labels not found at: {BASELINE_NUMERIC_LABELS_PATH}")
    else:
        print(f"Baseline numeric labels: {baseline_numeric.source_path}")

    comparisons = build_ari_comparison_table(
        selected_solution, resolutions, mixed, binary, baseline_numeric
    )
    comparisons.to_csv(ARI_COMPARISON_CSV, index=False)

    aligned_numeric_to_mixed: dict[int, np.ndarray] = {}
    aligned_numeric_to_binary: dict[int, np.ndarray] = {}

    if mixed is not None:
        for k in (2, 3, 4):
            column = f"mixed_balanced_k{k}"
            if column not in mixed.frame.columns:
                warnings.warn(f"Column {column} is absent; k={k} mixed comparison skipped.")
                continue
            reference_labels = mixed.frame[column].to_numpy(dtype=int)
            counts, rows, aligned = build_reference_numeric_contingency(reference_labels, resolutions[k].labels)
            aligned_numeric_to_mixed[k] = aligned
            counts.to_csv(TABLE_DIR / f"mixed_balanced_k{k}_vs_numeric_quantile_k{k}_counts.csv")
            rows.to_csv(TABLE_DIR / f"mixed_balanced_k{k}_vs_numeric_quantile_k{k}_row_percentages.csv")
            save_contingency_heatmap(
                counts, rows,
                ari=float(adjusted_rand_score(reference_labels, resolutions[k].labels)),
                k=k,
                reference_display_name="mixed balanced",
                output_name=f"12_mixed_vs_numeric_quantile_k{k}_contingency.png",
            )

    if binary is not None:
        for k in (2, 3, 4):
            column = f"binary_k{k}"
            if column not in binary.frame.columns:
                warnings.warn(f"Column {column} is absent; k={k} binary comparison skipped.")
                continue
            reference_labels = binary.frame[column].to_numpy(dtype=int)
            counts, rows, aligned = build_reference_numeric_contingency(reference_labels, resolutions[k].labels)
            aligned_numeric_to_binary[k] = aligned
            counts.to_csv(TABLE_DIR / f"binary_k{k}_vs_numeric_quantile_k{k}_counts.csv")
            rows.to_csv(TABLE_DIR / f"binary_k{k}_vs_numeric_quantile_k{k}_row_percentages.csv")
            save_contingency_heatmap(
                counts, rows,
                ari=float(adjusted_rand_score(reference_labels, resolutions[k].labels)),
                k=k,
                reference_display_name="binary-only",
                output_name=f"13_binary_vs_numeric_quantile_k{k}_contingency.png",
            )

    if baseline_numeric is not None:
        for k in (2, 3, 4):
            column = f"numeric_k{k}"
            if column not in baseline_numeric.frame.columns:
                warnings.warn(
                    f"Column {column} is absent; k={k} baseline numeric comparison skipped."
                )
                continue
            reference_labels = baseline_numeric.frame[column].to_numpy(dtype=int)
            counts, rows, _ = build_reference_numeric_contingency(
                reference_labels, resolutions[k].labels
            )
            counts.to_csv(
                TABLE_DIR / f"numeric_k{k}_vs_numeric_quantile_k{k}_counts.csv"
            )
            rows.to_csv(
                TABLE_DIR / f"numeric_k{k}_vs_numeric_quantile_k{k}_row_percentages.csv"
            )
            save_contingency_heatmap(
                counts,
                rows,
                ari=float(adjusted_rand_score(reference_labels, resolutions[k].labels)),
                k=k,
                reference_display_name="min-max numeric baseline",
                output_name=f"14_numeric_baseline_vs_quantile_k{k}_contingency.png",
            )

    save_model_selection_figure(results, "01_numeric_quantile_model_selection.png")
    save_raw_vs_normalized(selected_solution, "02_numeric_quantile_selected_embedding.png")
    save_raw_vs_normalized(numeric_k2, "03_numeric_quantile_k2_embedding.png")
    save_raw_vs_normalized(numeric_k3, "04_numeric_quantile_k3_embedding_2d.png")
    save_k3_embedding_3d(numeric_k3, "05_numeric_quantile_k3_embedding_3d.png")
    save_raw_vs_normalized(numeric_k4, "06_numeric_quantile_k4_embedding_2d.png")
    save_k4_pairwise_projections(numeric_k4, "07_numeric_quantile_k4_all_pairs.png")
    save_eigenvalue_spectrum(numeric_k4, "08_numeric_quantile_eigenvalue_spectrum.png")
    save_nesting_heatmap(k2_k3_counts, k2_k3_rows, "Numeric-only quantile nesting: k=2 versus k=3", "09_numeric_quantile_k2_vs_k3_nesting.png")
    save_nesting_heatmap(k2_k4_counts, k2_k4_rows, "Numeric-only quantile nesting: k=2 versus k=4", "10_numeric_quantile_k2_vs_k4_nesting.png")
    save_ari_comparison_figure(comparisons, "11_numeric_quantile_cross_representation_ari.png")

    save_coordinates(selected_solution, "numeric_quantile_selected_coordinates")
    save_coordinates(numeric_k2, "numeric_quantile_k2_coordinates")
    save_coordinates(numeric_k3, "numeric_quantile_k3_coordinates")
    save_coordinates(numeric_k4, "numeric_quantile_k4_coordinates")

    labels_frame = pd.DataFrame({
        "row_index": np.arange(len(loaded.X)),
        "numeric_quantile_selected": selected_solution.labels,
        "numeric_quantile_k2": numeric_k2.labels,
        "numeric_quantile_k3": reordered_k3,
        "numeric_quantile_k4": reordered_k4,
    })
    for k, aligned in aligned_numeric_to_mixed.items():
        labels_frame[f"numeric_quantile_k{k}_aligned_to_mixed"] = aligned
    for k, aligned in aligned_numeric_to_binary.items():
        labels_frame[f"numeric_quantile_k{k}_aligned_to_binary"] = aligned
    if loaded.activity is not None:
        labels_frame[TARGET_COL] = loaded.activity.to_numpy()
    labels_frame.to_csv(NUMERIC_LABELS_CSV, index=False)

    report_values = {
        "script_version": SCRIPT_VERSION,
        "preprocessing": preprocessing,
        "numeric_quantile_selected": solution_summary_row("numeric_quantile_selected", selected_solution),
        "numeric_quantile_k2": solution_summary_row("numeric_quantile_k2_macrostructure", numeric_k2),
        "numeric_quantile_k3": solution_summary_row("numeric_quantile_k3_intermediate_check", numeric_k3),
        "numeric_quantile_k4": solution_summary_row("numeric_quantile_k4_refinement", numeric_k4),
        "mixed_labels_source": None if mixed is None else str(mixed.source_path),
        "binary_labels_source": None if binary is None else str(binary.source_path),
        "baseline_numeric_labels_source": (
            None if baseline_numeric is None else str(baseline_numeric.source_path)
        ),
        "cross_representation_comparisons": comparisons.to_dict(orient="records"),
        "interpretation_rule": (
            "Treat quantile normalization as a nonlinear sensitivity analysis. "
            "Use ARI and contingency tables across representations; do not rank "
            "silhouettes directly across different geometries."
        ),
    }
    with REPORT_VALUES_JSON.open("w", encoding="utf-8") as handle:
        json.dump(report_values, handle, indent=2, default=json_compatible)

    write_results_summary(
        preprocessing,
        selected_solution,
        numeric_k2,
        numeric_k3,
        numeric_k4,
        comparisons,
        None if mixed is None else mixed.source_path,
        None if binary is None else binary.source_path,
        None if baseline_numeric is None else baseline_numeric.source_path,
    )

    verify_required_figures(include_external_comparison=not comparisons.empty)

    print("\n" + "=" * 78)
    print("NUMERIC-ONLY QUANTILE ANALYSIS COMPLETED")
    print("=" * 78)
    print(f"Reports directory: {REPORT_DIR}")
    print(f"Figures directory: {FIGURE_DIR}")
    print(f"Tables directory: {TABLE_DIR}")
    print(f"Coordinates directory: {COORDINATE_DIR}")
    print("\nGenerated figures:")
    for path in sorted(FIGURE_DIR.glob("*.png")):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
