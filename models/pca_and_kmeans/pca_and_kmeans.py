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
from scipy.stats import chi2

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_samples
from sklearn.model_selection import RepeatedKFold


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

# Mantenuto per coerenza con la precedente analisi.
# Con |phi| >= 0.95 e migliaia di campioni, la condizione
# p < 0.05 sarà normalmente sempre soddisfatta.
ALPHA = 0.05

# PCA
VARIANCE_TO_KEEP = 0.90

# K-Means
K_VALUES = list(range(2, 11))
KMEANS_N_INIT = 50

# Repeated cross-validation:
# 5 fold × 6 ripetizioni = 30 valutazioni per ciascun k.
N_SPLITS = 5
N_REPEATS = 6

# Intervalli di confidenza
CONFIDENCE_LEVEL = 0.95
N_BOOTSTRAP = 2_000
BOOTSTRAP_CHUNK_SIZE = 200

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

REPORTS_DIR = PROJECT_DIR / "reports" / "kmeans_pca_cv"

SUMMARY_REPORT_PATH = (
    REPORTS_DIR
    / "kmeans_pca_cv_summary.csv"
)

FOLD_REPORT_PATH = (
    REPORTS_DIR
    / "kmeans_pca_cv_fold_results.csv"
)

PREPROCESSING_REPORT_PATH = (
    REPORTS_DIR
    / "preprocessing_fold_diagnostics.csv"
)

ELBOW_PLOT_PATH = (
    REPORTS_DIR
    / "kmeans_pca_cv_elbow_plot.png"
)

SILHOUETTE_PLOT_PATH = (
    REPORTS_DIR
    / "kmeans_pca_cv_silhouette_plot.png"
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
    """
    Frequenza relativa del valore più comune.
    """
    frequencies = series.value_counts(
        normalize=True,
        dropna=False,
    )

    return float(frequencies.iloc[0])


def identify_binary_columns(
    X: pd.DataFrame,
) -> list[str]:
    """
    Una feature è binaria se tutti i valori osservati
    nel training fold appartengono a {0, 1}.
    """
    binary_columns = []

    for column in X.columns:
        unique_values = X[column].dropna().unique()

        if (
            len(unique_values) > 0
            and set(unique_values).issubset(
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

def find_highly_correlated_pairs(
    X_train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcola sul solo training fold:

    - Pearson: numerica-numerica;
    - Phi: binaria-binaria;
    - punto-biseriale: numerica-binaria.

    Le tre misure corrispondono alla correlazione di Pearson
    quando le variabili binarie sono codificate come 0/1.
    """
    if X_train.shape[1] < 2:
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

    binary_columns = identify_binary_columns(
        X_train
    )

    binary_set = set(binary_columns)

    columns = np.asarray(
        X_train.columns,
        dtype=object,
    )

    is_binary = np.asarray(
        [
            column in binary_set
            for column in columns
        ],
        dtype=bool,
    )

    correlation_matrix = (
        X_train
        .corr(method="pearson")
        .to_numpy(dtype=np.float64)
    )

    row_indices, column_indices = np.triu_indices(
        len(columns),
        k=1,
    )

    coefficients = correlation_matrix[
        row_indices,
        column_indices,
    ]

    absolute_coefficients = np.abs(
        coefficients
    )

    left_is_binary = is_binary[row_indices]
    right_is_binary = is_binary[column_indices]

    binary_binary_mask = (
        left_is_binary
        & right_is_binary
    )

    numeric_numeric_mask = (
        ~left_is_binary
        & ~right_is_binary
    )

    mixed_mask = (
        left_is_binary
        ^ right_is_binary
    )

    finite_mask = np.isfinite(coefficients)

    # Chi-quadrato per le sole coppie binaria-binaria:
    # chi2 = n * phi^2
    chi2_statistics = np.full(
        len(coefficients),
        np.nan,
        dtype=np.float64,
    )

    p_values = np.full(
        len(coefficients),
        np.nan,
        dtype=np.float64,
    )

    binary_positions = np.where(
        binary_binary_mask
        & finite_mask
    )[0]

    chi2_statistics[binary_positions] = (
        len(X_train)
        * coefficients[binary_positions] ** 2
    )

    p_values[binary_positions] = chi2.sf(
        chi2_statistics[binary_positions],
        df=1,
    )

    selected_numeric_numeric = (
        numeric_numeric_mask
        & finite_mask
        & (
            absolute_coefficients
            >= PEARSON_THRESHOLD
        )
    )

    selected_binary_binary = (
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
        selected_numeric_numeric
        | selected_binary_binary
        | selected_mixed
    )

    selected_positions = np.where(
        selected_mask
    )[0]

    if len(selected_positions) == 0:
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

    pair_types = np.empty(
        len(selected_positions),
        dtype=object,
    )

    for output_position, original_position in enumerate(
        selected_positions
    ):
        if numeric_numeric_mask[original_position]:
            pair_types[output_position] = (
                "numeric_numeric"
            )
        elif binary_binary_mask[original_position]:
            pair_types[output_position] = (
                "binary_binary"
            )
        else:
            pair_types[output_position] = (
                "numeric_binary"
            )

    pairs = pd.DataFrame(
        {
            "feature_1": columns[
                row_indices[selected_positions]
            ],
            "feature_2": columns[
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
        by="abs_coefficient",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# CORRELATED-FEATURE SELECTION
# ============================================================

def choose_feature_to_drop(
    feature_1: str,
    feature_2: str,
    variances: pd.Series,
    dominant_ratios: pd.Series,
) -> str:
    """
    Regola deterministica e non supervisionata.

    1. Elimina la feature con varianza minore.
    2. A parità, elimina quella con dominant ratio maggiore.
    3. A ulteriore parità, elimina quella lessicograficamente
       successiva.
    """
    variance_1 = float(variances[feature_1])
    variance_2 = float(variances[feature_2])

    if not np.isclose(
        variance_1,
        variance_2,
        rtol=1e-10,
        atol=1e-12,
    ):
        if variance_1 < variance_2:
            return feature_1

        return feature_2

    dominant_1 = float(
        dominant_ratios[feature_1]
    )

    dominant_2 = float(
        dominant_ratios[feature_2]
    )

    if not np.isclose(
        dominant_1,
        dominant_2,
        rtol=1e-10,
        atol=1e-12,
    ):
        if dominant_1 > dominant_2:
            return feature_1

        return feature_2

    return max(feature_1, feature_2)


def select_redundant_features_to_drop(
    X_train: pd.DataFrame,
    correlated_pairs: pd.DataFrame,
) -> list[str]:
    """
    Analizza le coppie dalla correlazione più alta alla più bassa.

    Se una delle due feature è già stata eliminata, la coppia
    è già risolta e non viene eliminata un'altra feature.
    """
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
            feature_1=feature_1,
            feature_2=feature_2,
            variances=variances,
            dominant_ratios=dominant_ratios,
        )

        dropped_features.add(
            feature_to_drop
        )

    return sorted(dropped_features)


# ============================================================
# COMPLETE FOLD PREPROCESSING
# ============================================================

def fit_fold_preprocessing(
    X_train_raw: pd.DataFrame,
) -> tuple[
    list[str],
    dict[str, object],
]:
    """
    Adatta tutte le decisioni di preprocessing esclusivamente
    sul training fold.
    """

    # --------------------------------------------------------
    # 1. Quasi-constant filtering
    # --------------------------------------------------------

    (
        quasi_constant_features,
        quasi_constant_report,
    ) = find_quasi_constant_features(
        X_train=X_train_raw,
        threshold=QUASI_CONSTANT_THRESHOLD,
    )

    X_after_quasi_constant = X_train_raw.drop(
        columns=quasi_constant_features,
    )

    if X_after_quasi_constant.empty:
        raise ValueError(
            "Tutte le feature sono state eliminate "
            "dal filtro quasi-costante."
        )

    # --------------------------------------------------------
    # 2. Correlation analysis
    # --------------------------------------------------------

    correlated_pairs = find_highly_correlated_pairs(
        X_after_quasi_constant
    )

    correlated_features_to_drop = (
        select_redundant_features_to_drop(
            X_train=X_after_quasi_constant,
            correlated_pairs=correlated_pairs,
        )
    )

    retained_features = [
        column
        for column in X_after_quasi_constant.columns
        if column
        not in correlated_features_to_drop
    ]

    if not retained_features:
        raise ValueError(
            "Tutte le feature sono state eliminate "
            "dalla correlation analysis."
        )

    binary_columns = identify_binary_columns(
        X_after_quasi_constant
    )

    pair_type_counts = (
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
        "binary_features_after_quasi_constant": (
            len(binary_columns)
        ),
        "numeric_features_after_quasi_constant": (
            X_after_quasi_constant.shape[1]
            - len(binary_columns)
        ),
        "numeric_numeric_pairs_above_threshold": int(
            pair_type_counts.get(
                "numeric_numeric",
                0,
            )
        ),
        "binary_binary_pairs_above_threshold": int(
            pair_type_counts.get(
                "binary_binary",
                0,
            )
        ),
        "numeric_binary_pairs_above_threshold": int(
            pair_type_counts.get(
                "numeric_binary",
                0,
            )
        ),
        "correlated_features_removed": (
            len(correlated_features_to_drop)
        ),
        "remaining_features": (
            len(retained_features)
        ),
        "quasi_constant_feature_names": ";".join(
            quasi_constant_features
        ),
        "correlated_feature_names": ";".join(
            correlated_features_to_drop
        ),
    }

    return retained_features, diagnostics


def transform_fold_preprocessing(
    X: pd.DataFrame,
    retained_features: list[str],
) -> pd.DataFrame:
    """
    Applica a training o validation l'elenco di feature
    determinato esclusivamente sul training fold.
    """
    missing_features = [
        feature
        for feature in retained_features
        if feature not in X.columns
    ]

    if missing_features:
        raise ValueError(
            "Feature mancanti durante la trasformazione: "
            f"{missing_features}"
        )

    return X.loc[
        :,
        retained_features,
    ].copy()


# ============================================================
# VALIDATION METRICS
# ============================================================

def validation_squared_distances(
    validation_data: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Distanza euclidea quadratica dal centroide assegnato
    per ogni molecola del validation fold.
    """
    assigned_centroids = centroids[labels]

    residuals = (
        validation_data
        - assigned_centroids
    )

    return np.einsum(
        "ij,ij->i",
        residuals,
        residuals,
    )


def silhouette_is_defined(
    labels: np.ndarray,
) -> bool:
    number_of_clusters = len(
        np.unique(labels)
    )

    return (
        number_of_clusters >= 2
        and number_of_clusters < len(labels)
    )


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def bootstrap_mean_confidence_interval(
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """
    Intervallo percentile bootstrap per la media.

    Restituisce:
    - media;
    - deviazione standard tra molecole;
    - limite inferiore;
    - limite superiore.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    number_of_values = len(values)

    bootstrap_means = np.empty(
        N_BOOTSTRAP,
        dtype=np.float64,
    )

    for start in range(
        0,
        N_BOOTSTRAP,
        BOOTSTRAP_CHUNK_SIZE,
    ):
        end = min(
            start + BOOTSTRAP_CHUNK_SIZE,
            N_BOOTSTRAP,
        )

        chunk_size = end - start

        bootstrap_indices = rng.integers(
            low=0,
            high=number_of_values,
            size=(
                chunk_size,
                number_of_values,
            ),
        )

        bootstrap_means[start:end] = (
            values[bootstrap_indices]
            .mean(axis=1)
        )

    alpha = 1 - CONFIDENCE_LEVEL

    lower = np.quantile(
        bootstrap_means,
        alpha / 2,
    )

    upper = np.quantile(
        bootstrap_means,
        1 - alpha / 2,
    )

    return (
        float(values.mean()),
        float(values.std(ddof=1)),
        float(lower),
        float(upper),
    )


# ============================================================
# REPEATED CROSS-VALIDATION
# ============================================================

def run_repeated_cross_validation(
    X: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    number_of_samples = len(X)

    cross_validator = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    within_sums = {
        k: np.zeros(
            number_of_samples,
            dtype=np.float64,
        )
        for k in K_VALUES
    }

    within_counts = {
        k: np.zeros(
            number_of_samples,
            dtype=np.int32,
        )
        for k in K_VALUES
    }

    silhouette_sums = {
        k: np.zeros(
            number_of_samples,
            dtype=np.float64,
        )
        for k in K_VALUES
    }

    silhouette_counts = {
        k: np.zeros(
            number_of_samples,
            dtype=np.int32,
        )
        for k in K_VALUES
    }

    fold_records = []
    preprocessing_records = []

    total_splits = (
        N_SPLITS
        * N_REPEATS
    )

    for split_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        cross_validator.split(X),
        start=1,
    ):
        repeat_number = (
            (split_number - 1)
            // N_SPLITS
            + 1
        )

        fold_number = (
            (split_number - 1)
            % N_SPLITS
            + 1
        )

        X_train_raw = X.iloc[
            train_indices
        ].copy()

        X_validation_raw = X.iloc[
            validation_indices
        ].copy()

        # ====================================================
        # PREPROCESSING FIT SUL SOLO TRAINING FOLD
        # ====================================================

        (
            retained_features,
            preprocessing_diagnostics,
        ) = fit_fold_preprocessing(
            X_train_raw
        )

        X_train_preprocessed = (
            transform_fold_preprocessing(
                X=X_train_raw,
                retained_features=retained_features,
            )
        )

        X_validation_preprocessed = (
            transform_fold_preprocessing(
                X=X_validation_raw,
                retained_features=retained_features,
            )
        )

        X_train_array = (
            X_train_preprocessed.to_numpy(
                dtype=np.float64
            )
        )

        X_validation_array = (
            X_validation_preprocessed.to_numpy(
                dtype=np.float64
            )
        )

        # ====================================================
        # PCA FIT SUL SOLO TRAINING FOLD
        # ====================================================

        pca = PCA(
            n_components=VARIANCE_TO_KEEP,
            svd_solver="full",
        )

        X_train_pca = pca.fit_transform(
            X_train_array
        )

        X_validation_pca = pca.transform(
            X_validation_array
        )

        number_of_components = (
            X_train_pca.shape[1]
        )

        explained_variance = float(
            pca.explained_variance_ratio_.sum()
        )

        preprocessing_records.append(
            {
                "split": split_number,
                "repeat": repeat_number,
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

        # Calcolata una volta per fold e riutilizzata
        # per tutti i valori di k.
        validation_distance_matrix = (
            pairwise_distances(
                X_validation_pca,
                metric="euclidean",
                n_jobs=-1,
            )
        )

        print(
            f"\nSplit {split_number:>2}/{total_splits} | "
            f"repeat={repeat_number} | "
            f"fold={fold_number}"
        )

        print(
            "  Feature: "
            f"{X_train_raw.shape[1]} -> "
            f"{X_train_preprocessed.shape[1]} | "
            "quasi-constant removed="
            f"{preprocessing_diagnostics['quasi_constant_removed']} | "
            "correlated removed="
            f"{preprocessing_diagnostics['correlated_features_removed']}"
        )

        print(
            f"  PCA components={number_of_components} | "
            f"variance={explained_variance:.6f}"
        )

        # ====================================================
        # K-MEANS PER k = 2, ..., 10
        # ====================================================

        for k in K_VALUES:
            kmeans = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=KMEANS_N_INIT,
                random_state=(
                    RANDOM_STATE
                    + split_number * 100
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

            squared_distances = (
                validation_squared_distances(
                    validation_data=(
                        X_validation_pca
                    ),
                    labels=validation_labels,
                    centroids=(
                        kmeans.cluster_centers_
                    ),
                )
            )

            within_sums[k][
                validation_indices
            ] += squared_distances

            within_counts[k][
                validation_indices
            ] += 1

            mean_within = float(
                squared_distances.mean()
            )

            if silhouette_is_defined(
                validation_labels
            ):
                silhouette_values = (
                    silhouette_samples(
                        validation_distance_matrix,
                        validation_labels,
                        metric="precomputed",
                    )
                )

                silhouette_sums[k][
                    validation_indices
                ] += silhouette_values

                silhouette_counts[k][
                    validation_indices
                ] += 1

                mean_silhouette = float(
                    silhouette_values.mean()
                )
            else:
                mean_silhouette = np.nan

            cluster_sizes = np.bincount(
                validation_labels,
                minlength=k,
            )

            fold_records.append(
                {
                    "split": split_number,
                    "repeat": repeat_number,
                    "fold": fold_number,
                    "k": k,
                    "train_size": len(
                        train_indices
                    ),
                    "validation_size": len(
                        validation_indices
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
                    "mean_validation_within_variance": (
                        mean_within
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
                }
            )

            print(
                f"    k={k:>2} | "
                f"within={mean_within:10.4f} | "
                f"silhouette={mean_silhouette:7.4f}"
            )

    fold_results = pd.DataFrame(
        fold_records
    )

    preprocessing_results = pd.DataFrame(
        preprocessing_records
    )

    per_sample_within = {}
    per_sample_silhouette = {}

    for k in K_VALUES:
        if np.any(
            within_counts[k] == 0
        ):
            raise RuntimeError(
                f"Alcune molecole non sono mai "
                f"state valutate per k={k}."
            )

        per_sample_within[k] = (
            within_sums[k]
            / within_counts[k]
        )

        silhouette_values = np.full(
            number_of_samples,
            np.nan,
            dtype=np.float64,
        )

        valid_mask = (
            silhouette_counts[k] > 0
        )

        silhouette_values[valid_mask] = (
            silhouette_sums[k][valid_mask]
            / silhouette_counts[k][valid_mask]
        )

        per_sample_silhouette[k] = (
            silhouette_values
        )

    return (
        fold_results,
        preprocessing_results,
        per_sample_within,
        per_sample_silhouette,
    )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    per_sample_within: dict[int, np.ndarray],
    per_sample_silhouette: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []

    for k in K_VALUES:
        (
            within_mean,
            within_std,
            within_lower,
            within_upper,
        ) = bootstrap_mean_confidence_interval(
            per_sample_within[k],
            rng,
        )

        (
            silhouette_mean,
            silhouette_std,
            silhouette_lower,
            silhouette_upper,
        ) = bootstrap_mean_confidence_interval(
            per_sample_silhouette[k],
            rng,
        )

        rows.append(
            {
                "k": k,
                "mean_validation_within_variance": (
                    within_mean
                ),
                "std_validation_within_variance": (
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

    return pd.DataFrame(rows)


def print_summary(
    summary: pd.DataFrame,
) -> None:
    print("\n" + "=" * 105)
    print(
        "RISULTATI MEDI OUT-OF-FOLD "
        f"CON IC {CONFIDENCE_LEVEL:.0%}"
    )
    print("=" * 105)

    for row in summary.itertuples(
        index=False
    ):
        print(
            f"k={row.k:>2} | "
            f"within="
            f"{row.mean_validation_within_variance:10.4f} "
            f"[{row.within_ci_lower_95:.4f}, "
            f"{row.within_ci_upper_95:.4f}] | "
            f"silhouette="
            f"{row.mean_validation_silhouette:7.4f} "
            f"[{row.silhouette_ci_lower_95:.4f}, "
            f"{row.silhouette_ci_upper_95:.4f}]"
        )

    best_row = summary.loc[
        summary[
            "mean_validation_silhouette"
        ].idxmax()
    ]

    print(
        "\nMigliore configurazione "
        "per silhouette media:"
    )

    print(
        f"k = {int(best_row['k'])}"
    )

    print(
        "Silhouette media = "
        f"{best_row['mean_validation_silhouette']:.4f}"
    )

    print(
        f"IC {CONFIDENCE_LEVEL:.0%} = "
        f"[{best_row['silhouette_ci_lower_95']:.4f}, "
        f"{best_row['silhouette_ci_upper_95']:.4f}]"
    )


# ============================================================
# PLOTS
# ============================================================

def plot_elbow(
    summary: pd.DataFrame,
) -> None:
    means = summary[
        "mean_validation_within_variance"
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
        capsize=4,
    )

    plt.xlabel("Number of clusters k")

    plt.ylabel(
        "Mean squared within-cluster distance "
        "on validation folds"
    )

    plt.title(
        "K-Means elbow plot "
        "with repeated cross-validation"
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
        capsize=4,
    )

    plt.xlabel("Number of clusters k")
    plt.ylabel(
        "Mean silhouette on validation folds"
    )

    plt.title(
        "K-Means silhouette "
        "with repeated cross-validation"
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
# MAIN
# ============================================================

def main() -> None:
    rng = np.random.default_rng(
        RANDOM_STATE
    )

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
        "REPEATED CROSS-VALIDATION"
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
        "nel preprocessing o nel clustering."
    )

    print(
        f"Target distribution: "
        f"{y.value_counts().to_dict()}"
    )

    print(
        f"Quasi-constant threshold: "
        f"{QUASI_CONSTANT_THRESHOLD:.0%}"
    )

    print(
        f"Correlation thresholds: "
        f"Pearson={PEARSON_THRESHOLD}, "
        f"Phi={PHI_THRESHOLD}, "
        f"Point-biserial="
        f"{POINT_BISERIAL_THRESHOLD}"
    )

    print(
        f"PCA variance threshold: "
        f"{VARIANCE_TO_KEEP:.0%}"
    )

    print(
        f"k range: "
        f"{min(K_VALUES)}-{max(K_VALUES)}"
    )

    print(
        f"Cross-validation: "
        f"{N_SPLITS} folds × "
        f"{N_REPEATS} repeats = "
        f"{N_SPLITS * N_REPEATS} "
        "valutazioni per k"
    )

    (
        fold_results,
        preprocessing_results,
        per_sample_within,
        per_sample_silhouette,
    ) = run_repeated_cross_validation(
        X
    )

    summary = build_summary(
        per_sample_within,
        per_sample_silhouette,
        rng,
    )

    print_summary(
        summary
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

    plot_elbow(
        summary
    )

    plot_silhouette(
        summary
    )

    print("\nDiagnostica del preprocessing")
    print("-" * 60)

    print(
        "Feature quasi-costanti rimosse, media: "
        f"{preprocessing_results['quasi_constant_removed'].mean():.2f}"
    )

    print(
        "Feature correlate rimosse, media: "
        f"{preprocessing_results['correlated_features_removed'].mean():.2f}"
    )

    print(
        "Feature finali, media: "
        f"{preprocessing_results['remaining_features'].mean():.2f}"
    )

    print(
        "Componenti PCA, media: "
        f"{preprocessing_results['pca_components'].mean():.2f}"
    )

    print(
        "Componenti PCA, intervallo: "
        f"[{preprocessing_results['pca_components'].min()}, "
        f"{preprocessing_results['pca_components'].max()}]"
    )

    print(
        "Varianza media conservata: "
        f"{preprocessing_results['explained_variance'].mean():.6f}"
    )

    print("\nFile salvati:")
    print(
        f"Summary: {SUMMARY_REPORT_PATH}"
    )

    print(
        f"Fold results: {FOLD_REPORT_PATH}"
    )

    print(
        "Preprocessing diagnostics: "
        f"{PREPROCESSING_REPORT_PATH}"
    )

    print(
        f"Elbow plot: {ELBOW_PLOT_PATH}"
    )

    print(
        "Silhouette plot: "
        f"{SILHOUETTE_PLOT_PATH}"
    )


if __name__ == "__main__":
    main()