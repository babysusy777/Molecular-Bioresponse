""""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D



BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

INPUT_PATH = PROJECT_DIR / "Dataset" / "train_filtered_no_activity.csv"

PCA_OUTPUT_PATH = PROJECT_DIR / "Dataset" / "train_pca_90_no_activity.csv"

PCA_REPORT_PATH = PROJECT_DIR / "reports" / "pca_explained_variance.csv"
PCA_PLOT_PATH = PROJECT_DIR / "reports" / "pca_scree_plot_90.png"

ELBOW_REPORT_PATH = PROJECT_DIR / "reports" / "kmeans_elbow_values.csv"
ELBOW_PLOT_PATH = PROJECT_DIR / "reports" / "kmeans_elbow_plot.png"

CLUSTER_PLOTS_DIR = PROJECT_DIR / "reports" / "kmeans_cluster_plots"
TOP_N_CLUSTER_PLOTS = 3

VARIANCE_TO_KEEP = 0.90
K_RANGE = range(2, 16)


def main():
    X = pd.read_csv(INPUT_PATH)

    print(f"Loaded dataset: {X.shape[0]} rows x {X.shape[1]} features")

    # PCA 90%
    pca = PCA(n_components=VARIANCE_TO_KEEP)
    X_pca = pca.fit_transform(X)

    pc_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca_df = pd.DataFrame(X_pca, columns=pc_columns)

    PCA_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PCA_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    X_pca_df.to_csv(PCA_OUTPUT_PATH, index=False)

    explained = pca.explained_variance_ratio_
    cumulative = explained.cumsum()

    pca_report = pd.DataFrame({"component": pc_columns, "explained_variance_ratio": explained, "cumulative_explained_variance": cumulative})
    pca_report.to_csv(PCA_REPORT_PATH, index=False)

    # PCA plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(explained) + 1), explained, label="Explained variance ratio")
    plt.plot(range(1, len(cumulative) + 1), cumulative, marker="o", label="Cumulative explained variance")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance")
    plt.title("PCA scree plot - 90% variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PCA_PLOT_PATH, dpi=150)
    plt.close()

    print()
    print("PCA completed")
    print(f"Original features: {X.shape[1]}")
    print(f"PCA components: {X_pca.shape[1]}")
    print(f"Explained variance kept: {explained.sum():.4f}")

    # K-Means elbow
    inertias = []
    silhouettes = []

    for k in K_RANGE:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = kmeans.fit_predict(X_pca)

        inertia = kmeans.inertia_
        silhouette = silhouette_score(X_pca, labels)

        inertias.append(inertia)
        silhouettes.append(silhouette)


        print(
            f"k={k} | "
            f"inertia={inertia:.2f} | "
            f"silhouette={silhouette:.4f} "
        )

    elbow_report = pd.DataFrame({
        "k": list(K_RANGE),
        "inertia": inertias,
        "silhouette": silhouettes
    })

    elbow_report.to_csv(ELBOW_REPORT_PATH, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(list(K_RANGE), inertias, marker="o")
    plt.xlabel("Number of clusters k")
    plt.ylabel("Inertia")
    plt.title("K-Means elbow method on PCA space")
    plt.xticks(list(K_RANGE))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ELBOW_PLOT_PATH, dpi=150)
    plt.close()

        # Plot clusters for the 3 best k values by silhouette score
    # Plot clusters for the 3 best k values by silhouette score in 3D
    CLUSTER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if X_pca.shape[1] < 3:
        print("Cannot plot 3D clusters: PCA produced less than 3 components.")
    else:
        top_k = elbow_report.sort_values("silhouette", ascending=False).head(TOP_N_CLUSTER_PLOTS)

        print()
        print("Best k values by silhouette:")
        print(top_k)

        for _, row in top_k.iterrows():
            k = int(row["k"])
            silhouette = row["silhouette"]

            kmeans = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=20
            )

            labels = kmeans.fit_predict(X_pca)
            centroids = kmeans.cluster_centers_

            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(111, projection="3d")

            ax.scatter(
                X_pca[:, 0],
                X_pca[:, 1],
                X_pca[:, 2],
                c=labels,
                s=10
            )

            ax.scatter(
                centroids[:, 0],
                centroids[:, 1],
                centroids[:, 2],
                marker="X",
                s=160,
                edgecolors="black"
            )

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")

            ax.set_title(f"K-Means clustering | k={k} | silhouette={silhouette:.4f}")

            plt.tight_layout()

            plot_path = CLUSTER_PLOTS_DIR / f"kmeans_clusters_3d_k{k}_silhouette_{silhouette:.4f}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()

            print(f"3D cluster plot saved: {plot_path}")

    print()
    print("Saved files:")
    print(f"PCA dataset: {PCA_OUTPUT_PATH}")
    print(f"PCA report: {PCA_REPORT_PATH}")
    print(f"PCA plot: {PCA_PLOT_PATH}")
    print(f"Elbow report: {ELBOW_REPORT_PATH}")
    print(f"Elbow plot: {ELBOW_PLOT_PATH}")

if __name__ == "__main__":
    main()

"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, t, ttest_1samp

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.model_selection import KFold


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TARGET_COL = "Activity"

# Preprocessing
QUASI_CONSTANT_THRESHOLD = 0.99

PEARSON_THRESHOLD = 0.95
PHI_THRESHOLD = 0.95
POINT_BISERIAL_THRESHOLD = 0.95
ALPHA = 0.05

# PCA
VARIANCE_TO_KEEP = 0.90

# K-Means
K_VALUES = list(range(2, 11))
KMEANS_N_INIT = 50

# Cross-validation convenzionale
N_SPLITS = 10
CONFIDENCE_LEVEL = 0.95

# Modelli finali da visualizzare
FINAL_CLUSTER_VALUES = [4, 5]

FIGURE_DPI = 200


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

INPUT_PATH = (
    PROJECT_DIR
    / "Dataset"
    / "raw"
    / "train.csv"
)

REPORTS_DIR = (
    PROJECT_DIR
    / "reports"
    / "kmeans_pca_cv_standard"
)

SUMMARY_REPORT_PATH = (
    REPORTS_DIR
    / "kmeans_pca_cv_summary.csv"
)

FOLD_REPORT_PATH = (
    REPORTS_DIR
    / "kmeans_pca_cv_fold_results.csv"
)

DIFFERENCE_TEST_PATH = (
    REPORTS_DIR
    / "kmeans_adjacent_k_difference_tests.csv"
)

PREPROCESSING_REPORT_PATH = (
    REPORTS_DIR
    / "preprocessing_fold_diagnostics.csv"
)

ELBOW_PLOT_PATH = (
    REPORTS_DIR
    / "kmeans_elbow_plot_cv.png"
)

SILHOUETTE_PLOT_PATH = (
    REPORTS_DIR
    / "kmeans_silhouette_plot_cv.png"
)

CLUSTER_PLOTS_DIR = (
    REPORTS_DIR
    / "final_cluster_plots"
)


# ============================================================
# DATA LOADING
# ============================================================

def load_raw_dataset(
    path: Path,
) -> tuple[pd.DataFrame, pd.Series]:

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset non trovato: {path}"
        )

    dataset = pd.read_csv(path)

    unnamed_columns = [
        column
        for column in dataset.columns
        if str(column).startswith("Unnamed:")
    ]

    if unnamed_columns:
        dataset = dataset.drop(columns=unnamed_columns)

    if TARGET_COL not in dataset.columns:
        raise ValueError(
            f"Target '{TARGET_COL}' non trovato."
        )

    y = dataset[TARGET_COL].copy()
    X = dataset.drop(columns=[TARGET_COL]).copy()

    non_numeric_columns = X.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    if non_numeric_columns:
        raise TypeError(
            "Sono presenti feature non numeriche: "
            f"{non_numeric_columns}"
        )

    if X.isna().any().any():
        missing_columns = X.columns[
            X.isna().any()
        ].tolist()

        raise ValueError(
            "Sono presenti valori mancanti nelle feature: "
            f"{missing_columns}"
        )

    return X, y


# ============================================================
# FEATURE UTILITIES
# ============================================================

def dominant_value_ratio(
    series: pd.Series,
) -> float:

    frequencies = series.value_counts(
        normalize=True,
        dropna=False,
    )

    return float(frequencies.iloc[0])


def identify_binary_columns(
    X: pd.DataFrame,
) -> list[str]:

    binary_columns = []

    for column in X.columns:
        values = X[column].dropna().unique()

        if (
            len(values) > 0
            and set(values).issubset(
                {0, 1, 0.0, 1.0, False, True}
            )
        ):
            binary_columns.append(column)

    return binary_columns


# ============================================================
# QUASI-CONSTANT FILTERING
# ============================================================

def find_quasi_constant_features(
    X_train: pd.DataFrame,
    threshold: float,
) -> tuple[list[str], pd.DataFrame]:

    report_rows = []

    for column in X_train.columns:
        ratio = dominant_value_ratio(
            X_train[column]
        )

        report_rows.append(
            {
                "feature": column,
                "dominant_value_ratio": ratio,
                "dominant_value_percentage": ratio * 100,
                "n_unique": X_train[column].nunique(
                    dropna=False
                ),
                "drop": ratio >= threshold,
            }
        )

    report = pd.DataFrame(report_rows)

    features_to_drop = report.loc[
        report["drop"],
        "feature",
    ].tolist()

    return features_to_drop, report


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def empty_correlation_pairs() -> pd.DataFrame:

    return pd.DataFrame(
        columns=[
            "feature_1",
            "feature_2",
            "coefficient",
            "abs_coefficient",
            "pair_type",
            "chi2",
            "p_value",
        ]
    )


def find_highly_correlated_pairs(
    X_train: pd.DataFrame,
) -> pd.DataFrame:

    if X_train.shape[1] < 2:
        return empty_correlation_pairs()

    binary_columns = identify_binary_columns(
        X_train
    )

    binary_set = set(binary_columns)

    column_names = np.asarray(
        X_train.columns,
        dtype=object,
    )

    is_binary = np.asarray(
        [
            column in binary_set
            for column in column_names
        ],
        dtype=bool,
    )

    correlation_matrix = (
        X_train
        .corr(method="pearson")
        .to_numpy(dtype=np.float64)
    )

    row_indices, column_indices = np.triu_indices(
        X_train.shape[1],
        k=1,
    )

    coefficients = correlation_matrix[
        row_indices,
        column_indices,
    ]

    absolute_coefficients = np.abs(
        coefficients
    )

    left_binary = is_binary[row_indices]
    right_binary = is_binary[column_indices]

    binary_binary_mask = (
        left_binary & right_binary
    )

    numeric_numeric_mask = (
        ~left_binary & ~right_binary
    )

    mixed_mask = (
        left_binary ^ right_binary
    )

    finite_mask = np.isfinite(
        coefficients
    )

    chi2_statistics = np.full(
        len(coefficients),
        np.nan,
    )

    p_values = np.full(
        len(coefficients),
        np.nan,
    )

    binary_positions = np.where(
        binary_binary_mask & finite_mask
    )[0]

    # Per tabelle 2x2:
    # chi2 = n * phi^2
    chi2_statistics[binary_positions] = (
        len(X_train)
        * coefficients[binary_positions] ** 2
    )

    p_values[binary_positions] = chi2.sf(
        chi2_statistics[binary_positions],
        df=1,
    )

    selected_numeric = (
        numeric_numeric_mask
        & finite_mask
        & (
            absolute_coefficients
            >= PEARSON_THRESHOLD
        )
    )

    selected_binary = (
        binary_binary_mask
        & finite_mask
        & (
            absolute_coefficients
            >= PHI_THRESHOLD
        )
        & (p_values < ALPHA)
    )

    selected_mixed = (
        mixed_mask
        & finite_mask
        & (
            absolute_coefficients
            >= POINT_BISERIAL_THRESHOLD
        )
    )

    selected_mask = (
        selected_numeric
        | selected_binary
        | selected_mixed
    )

    selected_positions = np.where(
        selected_mask
    )[0]

    if len(selected_positions) == 0:
        return empty_correlation_pairs()

    pair_types = np.select(
        [
            numeric_numeric_mask[selected_positions],
            binary_binary_mask[selected_positions],
            mixed_mask[selected_positions],
        ],
        [
            "numeric_numeric",
            "binary_binary",
            "numeric_binary",
        ],
        default="unknown",
    )

    pairs = pd.DataFrame(
        {
            "feature_1": column_names[
                row_indices[selected_positions]
            ],
            "feature_2": column_names[
                column_indices[selected_positions]
            ],
            "coefficient": coefficients[
                selected_positions
            ],
            "abs_coefficient": (
                absolute_coefficients[
                    selected_positions
                ]
            ),
            "pair_type": pair_types,
            "chi2": chi2_statistics[
                selected_positions
            ],
            "p_value": p_values[
                selected_positions
            ],
        }
    )

    return pairs.sort_values(
        "abs_coefficient",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# CORRELATED FEATURE REMOVAL
# ============================================================

def choose_feature_to_drop(
    feature_1: str,
    feature_2: str,
    variances: pd.Series,
    dominant_ratios: pd.Series,
) -> str:

    variance_1 = float(
        variances[feature_1]
    )

    variance_2 = float(
        variances[feature_2]
    )

    # Prima regola: elimina la feature con minore varianza.
    if not np.isclose(
        variance_1,
        variance_2,
        rtol=1e-10,
        atol=1e-12,
    ):
        return (
            feature_1
            if variance_1 < variance_2
            else feature_2
        )

    dominant_1 = float(
        dominant_ratios[feature_1]
    )

    dominant_2 = float(
        dominant_ratios[feature_2]
    )

    # Seconda regola: elimina quella più sbilanciata.
    if not np.isclose(
        dominant_1,
        dominant_2,
        rtol=1e-10,
        atol=1e-12,
    ):
        return (
            feature_1
            if dominant_1 > dominant_2
            else feature_2
        )

    # Spareggio deterministico.
    return max(feature_1, feature_2)


def select_correlated_features_to_drop(
    X_train: pd.DataFrame,
    correlated_pairs: pd.DataFrame,
) -> list[str]:

    if correlated_pairs.empty:
        return []

    variances = X_train.var(
        axis=0,
        ddof=0,
    )

    dominant_ratios = X_train.apply(
        dominant_value_ratio,
        axis=0,
    )

    dropped_features: set[str] = set()

    for pair in correlated_pairs.itertuples(
        index=False
    ):
        feature_1 = pair.feature_1
        feature_2 = pair.feature_2

        if (
            feature_1 in dropped_features
            or feature_2 in dropped_features
        ):
            continue

        feature_to_drop = choose_feature_to_drop(
            feature_1,
            feature_2,
            variances,
            dominant_ratios,
        )

        dropped_features.add(
            feature_to_drop
        )

    return sorted(dropped_features)


# ============================================================
# COMPLETE PREPROCESSING
# ============================================================

def fit_preprocessing(
    X_train_raw: pd.DataFrame,
) -> tuple[list[str], dict[str, object]]:

    quasi_constant_features, _ = (
        find_quasi_constant_features(
            X_train_raw,
            QUASI_CONSTANT_THRESHOLD,
        )
    )

    X_after_quasi_constant = (
        X_train_raw.drop(
            columns=quasi_constant_features
        )
    )

    if X_after_quasi_constant.shape[1] == 0:
        raise ValueError(
            "Tutte le feature sono state eliminate "
            "come quasi-costanti."
        )

    correlated_pairs = (
        find_highly_correlated_pairs(
            X_after_quasi_constant
        )
    )

    correlated_features = (
        select_correlated_features_to_drop(
            X_after_quasi_constant,
            correlated_pairs,
        )
    )

    retained_features = [
        column
        for column in X_after_quasi_constant.columns
        if column not in correlated_features
    ]

    if not retained_features:
        raise ValueError(
            "Tutte le feature sono state eliminate."
        )

    pair_counts = (
        correlated_pairs["pair_type"]
        .value_counts()
        if not correlated_pairs.empty
        else pd.Series(dtype=int)
    )

    diagnostics = {
        "original_features": (
            X_train_raw.shape[1]
        ),
        "quasi_constant_removed": (
            len(quasi_constant_features)
        ),
        "features_after_quasi_constant": (
            X_after_quasi_constant.shape[1]
        ),
        "numeric_numeric_pairs": int(
            pair_counts.get(
                "numeric_numeric",
                0,
            )
        ),
        "binary_binary_pairs": int(
            pair_counts.get(
                "binary_binary",
                0,
            )
        ),
        "numeric_binary_pairs": int(
            pair_counts.get(
                "numeric_binary",
                0,
            )
        ),
        "correlated_features_removed": (
            len(correlated_features)
        ),
        "remaining_features": (
            len(retained_features)
        ),
        "quasi_constant_feature_names": ";".join(
            quasi_constant_features
        ),
        "correlated_feature_names": ";".join(
            correlated_features
        ),
    }

    return retained_features, diagnostics


def transform_preprocessing(
    X: pd.DataFrame,
    retained_features: list[str],
) -> pd.DataFrame:

    missing_features = [
        feature
        for feature in retained_features
        if feature not in X.columns
    ]

    if missing_features:
        raise ValueError(
            "Feature mancanti: "
            f"{missing_features}"
        )

    return X.loc[
        :,
        retained_features,
    ].copy()


# ============================================================
# STATISTICAL UTILITIES
# ============================================================

def t_confidence_interval(
    values: np.ndarray | pd.Series,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> tuple[float, float, float, float]:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    n = len(values)

    if n < 2:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    mean = values.mean()
    standard_deviation = values.std(
        ddof=1
    )

    standard_error = (
        standard_deviation
        / np.sqrt(n)
    )

    alpha = 1 - confidence_level

    critical_value = t.ppf(
        1 - alpha / 2,
        df=n - 1,
    )

    margin = (
        critical_value
        * standard_error
    )

    return (
        float(mean),
        float(standard_deviation),
        float(mean - margin),
        float(mean + margin),
    )


def holm_adjust(
    p_values: np.ndarray | pd.Series,
) -> np.ndarray:

    p_values = np.asarray(
        p_values,
        dtype=np.float64,
    )

    number_of_tests = len(p_values)

    order = np.argsort(
        p_values
    )

    adjusted = np.empty(
        number_of_tests,
        dtype=np.float64,
    )

    running_maximum = 0.0

    for rank, original_index in enumerate(
        order
    ):
        multiplier = (
            number_of_tests - rank
        )

        corrected_value = min(
            1.0,
            multiplier
            * p_values[original_index],
        )

        running_maximum = max(
            running_maximum,
            corrected_value,
        )

        adjusted[original_index] = (
            running_maximum
        )

    return adjusted


# ============================================================
# CROSS-VALIDATION
# ============================================================

def run_cross_validation(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    cross_validator = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_records = []
    preprocessing_records = []

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        cross_validator.split(X),
        start=1,
    ):
        X_train_raw = X.iloc[
            train_indices
        ].copy()

        X_validation_raw = X.iloc[
            validation_indices
        ].copy()

        # ----------------------------------------------------
        # Preprocessing fit solo sul training fold
        # ----------------------------------------------------

        (
            retained_features,
            preprocessing_diagnostics,
        ) = fit_preprocessing(
            X_train_raw
        )

        X_train_preprocessed = (
            transform_preprocessing(
                X_train_raw,
                retained_features,
            )
        )

        X_validation_preprocessed = (
            transform_preprocessing(
                X_validation_raw,
                retained_features,
            )
        )

        # ----------------------------------------------------
        # PCA fit solo sul training fold
        # ----------------------------------------------------

        pca = PCA(
            n_components=VARIANCE_TO_KEEP,
            svd_solver="full",
        )

        X_train_pca = pca.fit_transform(
            X_train_preprocessed
        )

        X_validation_pca = pca.transform(
            X_validation_preprocessed
        )

        explained_variance = float(
            pca.explained_variance_ratio_.sum()
        )

        number_of_components = (
            X_train_pca.shape[1]
        )

        preprocessing_records.append(
            {
                "fold": fold_number,
                "train_size": len(train_indices),
                "validation_size": (
                    len(validation_indices)
                ),
                **preprocessing_diagnostics,
                "pca_components": (
                    number_of_components
                ),
                "explained_variance": (
                    explained_variance
                ),
            }
        )

        # Una sola matrice delle distanze per fold.
        validation_distance_matrix = (
            pairwise_distances(
                X_validation_pca,
                metric="euclidean",
                n_jobs=-1,
            )
        )

        print(
            f"\nFold {fold_number}/{N_SPLITS} | "
            f"feature="
            f"{X_train_preprocessed.shape[1]} | "
            f"PCA components="
            f"{number_of_components} | "
            f"variance="
            f"{explained_variance:.6f}"
        )

        for k in K_VALUES:
            kmeans = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=KMEANS_N_INIT,
                random_state=(
                    RANDOM_STATE
                    + fold_number * 100
                    + k
                ),
            )

            kmeans.fit(
                X_train_pca
            )

            validation_labels = (
                kmeans.predict(
                    X_validation_pca
                )
            )

            assigned_centroids = (
                kmeans.cluster_centers_[
                    validation_labels
                ]
            )

            squared_distances = np.sum(
                (
                    X_validation_pca
                    - assigned_centroids
                ) ** 2,
                axis=1,
            )

            mean_within_distance = float(
                squared_distances.mean()
            )

            unique_labels = np.unique(
                validation_labels
            )

            if (
                len(unique_labels) >= 2
                and len(unique_labels)
                < len(validation_labels)
            ):
                mean_silhouette = float(
                    silhouette_score(
                        validation_distance_matrix,
                        validation_labels,
                        metric="precomputed",
                    )
                )
            else:
                mean_silhouette = np.nan

            cluster_sizes = np.bincount(
                validation_labels,
                minlength=k,
            )

            fold_records.append(
                {
                    "fold": fold_number,
                    "k": k,
                    "mean_validation_within_distance": (
                        mean_within_distance
                    ),
                    "mean_validation_silhouette": (
                        mean_silhouette
                    ),
                    "non_empty_clusters": int(
                        np.count_nonzero(
                            cluster_sizes
                        )
                    ),
                    "minimum_cluster_size": int(
                        cluster_sizes.min()
                    ),
                    "maximum_cluster_size": int(
                        cluster_sizes.max()
                    ),
                    "remaining_features": (
                        X_train_preprocessed.shape[1]
                    ),
                    "pca_components": (
                        number_of_components
                    ),
                    "explained_variance": (
                        explained_variance
                    ),
                }
            )

            print(
                f"  k={k:>2} | "
                f"within="
                f"{mean_within_distance:10.4f} | "
                f"silhouette="
                f"{mean_silhouette:7.4f}"
            )

    return (
        pd.DataFrame(fold_records),
        pd.DataFrame(preprocessing_records),
    )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:

    summary_rows = []

    for k in K_VALUES:
        subset = fold_results[
            fold_results["k"] == k
        ]

        (
            within_mean,
            within_std,
            within_lower,
            within_upper,
        ) = t_confidence_interval(
            subset[
                "mean_validation_within_distance"
            ]
        )

        (
            silhouette_mean,
            silhouette_std,
            silhouette_lower,
            silhouette_upper,
        ) = t_confidence_interval(
            subset[
                "mean_validation_silhouette"
            ]
        )

        summary_rows.append(
            {
                "k": k,
                "n_folds": len(subset),
                "mean_validation_within_distance": (
                    within_mean
                ),
                "std_validation_within_distance": (
                    within_std
                ),
                "within_ci_lower_95": (
                    within_lower
                ),
                "within_ci_upper_95": (
                    within_upper
                ),
                "mean_validation_silhouette": (
                    silhouette_mean
                ),
                "std_validation_silhouette": (
                    silhouette_std
                ),
                "silhouette_ci_lower_95": (
                    silhouette_lower
                ),
                "silhouette_ci_upper_95": (
                    silhouette_upper
                ),
            }
        )

    return pd.DataFrame(
        summary_rows
    )


# ============================================================
# PAIRED DIFFERENCE TESTS
# ============================================================

def build_difference_tests(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:

    within_pivot = fold_results.pivot(
        index="fold",
        columns="k",
        values="mean_validation_within_distance",
    )

    silhouette_pivot = fold_results.pivot(
        index="fold",
        columns="k",
        values="mean_validation_silhouette",
    )

    test_rows = []

    for current_k, next_k in zip(
        K_VALUES[:-1],
        K_VALUES[1:],
    ):
        # Positive = riduzione della within distance.
        within_difference = (
            within_pivot[current_k]
            - within_pivot[next_k]
        )

        (
            within_mean,
            within_std,
            within_lower,
            within_upper,
        ) = t_confidence_interval(
            within_difference
        )

        within_test = ttest_1samp(
            within_difference,
            popmean=0,
            nan_policy="omit",
        )

        relative_reduction = (
            (
                within_pivot[current_k]
                - within_pivot[next_k]
            )
            / within_pivot[current_k]
            * 100
        ).mean()

        test_rows.append(
            {
                "metric": "within_distance",
                "comparison": (
                    f"k={current_k} vs k={next_k}"
                ),
                "k_from": current_k,
                "k_to": next_k,
                "mean_difference": within_mean,
                "std_difference": within_std,
                "difference_ci_lower_95": (
                    within_lower
                ),
                "difference_ci_upper_95": (
                    within_upper
                ),
                "mean_relative_change_percentage": (
                    relative_reduction
                ),
                "t_statistic": float(
                    within_test.statistic
                ),
                "p_value": float(
                    within_test.pvalue
                ),
            }
        )

        # Positive = aumento della silhouette.
        silhouette_difference = (
            silhouette_pivot[next_k]
            - silhouette_pivot[current_k]
        )

        (
            silhouette_mean,
            silhouette_std,
            silhouette_lower,
            silhouette_upper,
        ) = t_confidence_interval(
            silhouette_difference
        )

        silhouette_test = ttest_1samp(
            silhouette_difference,
            popmean=0,
            nan_policy="omit",
        )

        test_rows.append(
            {
                "metric": "silhouette",
                "comparison": (
                    f"k={current_k} vs k={next_k}"
                ),
                "k_from": current_k,
                "k_to": next_k,
                "mean_difference": (
                    silhouette_mean
                ),
                "std_difference": (
                    silhouette_std
                ),
                "difference_ci_lower_95": (
                    silhouette_lower
                ),
                "difference_ci_upper_95": (
                    silhouette_upper
                ),
                "mean_relative_change_percentage": (
                    np.nan
                ),
                "t_statistic": float(
                    silhouette_test.statistic
                ),
                "p_value": float(
                    silhouette_test.pvalue
                ),
            }
        )

    tests = pd.DataFrame(
        test_rows
    )

    tests["p_value_holm"] = np.nan

    for metric in [
        "within_distance",
        "silhouette",
    ]:
        mask = (
            tests["metric"] == metric
        )

        tests.loc[
            mask,
            "p_value_holm",
        ] = holm_adjust(
            tests.loc[
                mask,
                "p_value",
            ].to_numpy()
        )

    tests["significant_holm_0_05"] = (
        tests["p_value_holm"] < 0.05
    )

    return tests


# ============================================================
# PLOTS
# ============================================================

def plot_elbow(
    summary: pd.DataFrame,
) -> None:

    means = summary[
        "mean_validation_within_distance"
    ].to_numpy()

    lower_errors = (
        means
        - summary[
            "within_ci_lower_95"
        ].to_numpy()
    )

    upper_errors = (
        summary[
            "within_ci_upper_95"
        ].to_numpy()
        - means
    )

    plt.figure(figsize=(9, 6))

    plt.errorbar(
        summary["k"],
        means,
        yerr=[
            lower_errors,
            upper_errors,
        ],
        marker="o",
        capsize=5,
    )

    plt.xlabel("Numero di cluster k")

    plt.ylabel(
        "Distanza quadratica media intra-cluster "
        "sui validation fold"
    )

    plt.title(
        "K-Means elbow plot con 10-fold cross-validation"
    )

    plt.xticks(K_VALUES)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        ELBOW_PLOT_PATH,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close()


def plot_silhouette(
    summary: pd.DataFrame,
) -> None:

    means = summary[
        "mean_validation_silhouette"
    ].to_numpy()

    lower_errors = (
        means
        - summary[
            "silhouette_ci_lower_95"
        ].to_numpy()
    )

    upper_errors = (
        summary[
            "silhouette_ci_upper_95"
        ].to_numpy()
        - means
    )

    plt.figure(figsize=(9, 6))

    plt.errorbar(
        summary["k"],
        means,
        yerr=[
            lower_errors,
            upper_errors,
        ],
        marker="o",
        capsize=5,
    )

    plt.xlabel("Numero di cluster k")
    plt.ylabel(
        "Silhouette media sui validation fold"
    )

    plt.title(
        "Silhouette media con 10-fold cross-validation"
    )

    plt.xticks(K_VALUES)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        SILHOUETTE_PLOT_PATH,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# FINAL MODELS AND CLUSTER PLOTS
# ============================================================

def fit_final_models_and_plot(
    X: pd.DataFrame,
) -> None:

    CLUSTER_PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Preprocessing riadattato sull'intero dataset.
    retained_features, diagnostics = (
        fit_preprocessing(X)
    )

    X_preprocessed = transform_preprocessing(
        X,
        retained_features,
    )

    pca = PCA(
        n_components=VARIANCE_TO_KEEP,
        svd_solver="full",
    )

    X_pca = pca.fit_transform(
        X_preprocessed
    )

    explained = (
        pca.explained_variance_ratio_
    )

    print("\nModello finale")
    print("-" * 60)
    print(
        f"Feature finali: "
        f"{X_preprocessed.shape[1]}"
    )
    print(
        f"Componenti PCA: "
        f"{X_pca.shape[1]}"
    )
    print(
        f"Varianza conservata: "
        f"{explained.sum():.6f}"
    )

    for k in FINAL_CLUSTER_VALUES:
        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=KMEANS_N_INIT,
            random_state=RANDOM_STATE + k,
        )

        labels = kmeans.fit_predict(
            X_pca
        )

        centroids = (
            kmeans.cluster_centers_
        )

        cluster_sizes = np.bincount(
            labels,
            minlength=k,
        )

        print(
            f"k={k} | cluster sizes: "
            f"{cluster_sizes.tolist()}"
        )

        plt.figure(figsize=(9, 7))

        plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=labels,
            s=12,
            alpha=0.55,
        )

        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="X",
            s=200,
            edgecolors="black",
        )

        plt.xlabel(
            "PC1 — varianza spiegata: "
            f"{explained[0] * 100:.2f}%"
        )

        plt.ylabel(
            "PC2 — varianza spiegata: "
            f"{explained[1] * 100:.2f}%"
        )

        plt.title(
            f"K-Means con k={k} — "
            "visualizzazione su PC1 e PC2"
        )

        plt.tight_layout()

        output_path = (
            CLUSTER_PLOTS_DIR
            / f"kmeans_clusters_pc1_pc2_k{k}.png"
        )

        plt.savefig(
            output_path,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"Cluster plot salvato: "
            f"{output_path}"
        )


# ============================================================
# PRINTING
# ============================================================

def print_summary(
    summary: pd.DataFrame,
) -> None:

    print("\n" + "=" * 105)
    print(
        "RISULTATI 10-FOLD CROSS-VALIDATION "
        "CON INTERVALLI t AL 95%"
    )
    print("=" * 105)

    for row in summary.itertuples(
        index=False
    ):
        print(
            f"k={row.k:>2} | "
            f"within="
            f"{row.mean_validation_within_distance:10.4f} "
            f"[{row.within_ci_lower_95:.4f}, "
            f"{row.within_ci_upper_95:.4f}] | "
            f"silhouette="
            f"{row.mean_validation_silhouette:7.4f} "
            f"[{row.silhouette_ci_lower_95:.4f}, "
            f"{row.silhouette_ci_upper_95:.4f}]"
        )


def print_difference_tests(
    tests: pd.DataFrame,
) -> None:

    print("\n" + "=" * 105)
    print("TEST t APPAIATI TRA VALORI CONSECUTIVI DI k")
    print("=" * 105)

    for row in tests.itertuples(
        index=False
    ):
        print(
            f"{row.metric:<16} | "
            f"{row.comparison:<14} | "
            f"diff={row.mean_difference:9.5f} | "
            f"IC=[{row.difference_ci_lower_95:.5f}, "
            f"{row.difference_ci_upper_95:.5f}] | "
            f"p-Holm={row.p_value_holm:.6g} | "
            f"significativo="
            f"{row.significant_holm_0_05}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X, y = load_raw_dataset(
        INPUT_PATH
    )

    print("=" * 75)
    print(
        "PREPROCESSING + PCA + K-MEANS "
        "10-FOLD CROSS-VALIDATION"
    )
    print("=" * 75)

    print(
        f"Dataset: {X.shape[0]} molecole "
        f"x {X.shape[1]} feature"
    )

    print(
        f"Target separato: {TARGET_COL}"
    )

    print(
        "Il target non viene utilizzato "
        "nel clustering."
    )

    print(
        f"Distribuzione target: "
        f"{y.value_counts().to_dict()}"
    )

    fold_results, preprocessing_results = (
        run_cross_validation(X)
    )

    summary = build_summary(
        fold_results
    )

    difference_tests = build_difference_tests(
        fold_results
    )

    print_summary(
        summary
    )

    print_difference_tests(
        difference_tests
    )

    fold_results.to_csv(
        FOLD_REPORT_PATH,
        index=False,
    )

    preprocessing_results.to_csv(
        PREPROCESSING_REPORT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_REPORT_PATH,
        index=False,
    )

    difference_tests.to_csv(
        DIFFERENCE_TEST_PATH,
        index=False,
    )

    plot_elbow(
        summary
    )

    plot_silhouette(
        summary
    )

    fit_final_models_and_plot(
        X
    )

    print("\nFile salvati:")
    print(
        f"Fold results: "
        f"{FOLD_REPORT_PATH}"
    )
    print(
        f"Summary: "
        f"{SUMMARY_REPORT_PATH}"
    )
    print(
        f"Difference tests: "
        f"{DIFFERENCE_TEST_PATH}"
    )
    print(
        f"Preprocessing diagnostics: "
        f"{PREPROCESSING_REPORT_PATH}"
    )
    print(
        f"Elbow plot: "
        f"{ELBOW_PLOT_PATH}"
    )
    print(
        f"Silhouette plot: "
        f"{SILHOUETTE_PLOT_PATH}"
    )
    print(
        f"Cluster plots: "
        f"{CLUSTER_PLOTS_DIR}"
    )


if __name__ == "__main__":
    main()