from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ============================================================
# CONFIGURATION
# ============================================================

# Soglia utilizzata per produrre il dataset PCA finale.
SELECTED_VARIANCE_THRESHOLD = 0.90

# Soglie confrontate nell'analisi.
VARIANCE_THRESHOLDS = [0.80, 0.85, 0.90, 0.95]

# Numero di componenti mostrate nello scree plot ingrandito.
N_COMPONENTS_DETAIL_PLOT = 50

# Salvataggio del dataset trasformato.
SAVE_TRANSFORMED_DATA = True

FIGURE_DPI = 200


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_directory(start: Path) -> Path:
    """
    Cerca la directory principale del progetto risalendo
    l'albero delle cartelle fino a trovare Dataset/.
    """
    for candidate in [start, *start.parents]:
        if (candidate / "Dataset").exists():
            return candidate

    raise FileNotFoundError(
        "Non è stata trovata la directory principale del progetto "
        "contenente la cartella 'Dataset'."
    )


PROJECT_DIR = find_project_directory(SCRIPT_DIR)

INPUT_CANDIDATES = [
    PROJECT_DIR
    / "Dataset"
    / "train_filtered_no_activity.csv",

    PROJECT_DIR
    / "Dataset"
    / "processed"
    / "train_filtered_no_activity.csv",

    PROJECT_DIR
    / "Dataset"
    / "preprocessed"
    / "train_filtered_no_activity.csv",

    PROJECT_DIR
    / "train_filtered_no_activity.csv",
]

OUTPUT_DIR = PROJECT_DIR / "reports" / "pca_analysis"


def find_input_file() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path

    attempted_paths = "\n".join(
        str(path) for path in INPUT_CANDIDATES
    )

    raise FileNotFoundError(
        "File train_filtered_no_activity.csv non trovato.\n"
        f"Percorsi controllati:\n{attempted_paths}"
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(path)

    # Rimuove eventuali colonne indice salvate nel CSV.
    unnamed_columns = [
        column
        for column in dataset.columns
        if str(column).startswith("Unnamed:")
    ]

    if unnamed_columns:
        dataset = dataset.drop(columns=unnamed_columns)

    # Controlla che tutte le colonne siano numeriche.
    non_numeric_columns = dataset.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    if non_numeric_columns:
        raise TypeError(
            "Sono presenti colonne non numeriche: "
            f"{non_numeric_columns}"
        )

    # Controlla la presenza di valori mancanti.
    if dataset.isna().any().any():
        missing_columns = dataset.columns[
            dataset.isna().any()
        ].tolist()

        raise ValueError(
            "Il dataset contiene valori mancanti nelle colonne: "
            f"{missing_columns}"
        )

    return dataset


# ============================================================
# PCA UTILITIES
# ============================================================

def components_for_threshold(
    cumulative_variance: np.ndarray,
    threshold: float,
) -> int:
    """
    Restituisce il numero minimo di componenti necessario
    per raggiungere la soglia di varianza cumulativa.
    """
    return int(
        np.searchsorted(
            cumulative_variance,
            threshold,
        )
        + 1
    )


def save_figure(filename: str) -> None:
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# PLOTS
# ============================================================

def plot_full_scree(
    explained_variance_ratio: np.ndarray,
) -> None:
    component_numbers = np.arange(
        1,
        len(explained_variance_ratio) + 1,
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        component_numbers,
        explained_variance_ratio,
        linewidth=1,
    )

    plt.xlabel("Componente principale")
    plt.ylabel("Varianza spiegata individuale")
    plt.title("PCA scree plot completo")

    save_figure("01_full_scree_plot.png")


def plot_detailed_scree(
    explained_variance_ratio: np.ndarray,
) -> None:
    number_of_components = min(
        N_COMPONENTS_DETAIL_PLOT,
        len(explained_variance_ratio),
    )

    selected_variance = explained_variance_ratio[
        :number_of_components
    ]

    component_numbers = np.arange(
        1,
        number_of_components + 1,
    )

    plt.figure(figsize=(11, 6))

    plt.bar(
        component_numbers,
        selected_variance,
        edgecolor="black",
    )

    plt.xlabel("Componente principale")
    plt.ylabel("Varianza spiegata individuale")
    plt.title(
        f"PCA scree plot — prime "
        f"{number_of_components} componenti"
    )

    save_figure("02_detailed_scree_plot.png")


def plot_cumulative_variance(
    cumulative_variance: np.ndarray,
    threshold_components: dict[float, int],
) -> None:
    component_numbers = np.arange(
        1,
        len(cumulative_variance) + 1,
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        component_numbers,
        cumulative_variance,
        linewidth=2,
    )

    for threshold, n_components in threshold_components.items():
        plt.axhline(
            threshold,
            linestyle="--",
            label=(
                f"{threshold:.0%}: "
                f"{n_components} componenti"
            ),
        )

        plt.axvline(
            n_components,
            linestyle=":",
            alpha=0.7,
        )

    plt.xlabel("Numero di componenti principali")
    plt.ylabel("Varianza spiegata cumulativa")
    plt.title("PCA — varianza spiegata cumulativa")
    plt.legend()

    save_figure("03_cumulative_explained_variance.png")


def plot_first_two_components(
    transformed_data: np.ndarray,
    explained_variance_ratio: np.ndarray,
) -> None:
    """
    Visualizzazione non supervisionata delle prime due componenti.
    Non vengono utilizzate etichette di classe.
    """
    if transformed_data.shape[1] < 2:
        return

    pc1_variance = explained_variance_ratio[0] * 100
    pc2_variance = explained_variance_ratio[1] * 100

    plt.figure(figsize=(9, 7))

    plt.scatter(
        transformed_data[:, 0],
        transformed_data[:, 1],
        s=10,
        alpha=0.5,
    )

    plt.xlabel(
        f"PC1 — varianza spiegata: {pc1_variance:.2f}%"
    )

    plt.ylabel(
        f"PC2 — varianza spiegata: {pc2_variance:.2f}%"
    )

    plt.title("Proiezione dei dati sulle prime due componenti")

    save_figure("04_pca_first_two_components.png")


# ============================================================
# PCA ANALYSIS
# ============================================================

def run_full_pca(
    data: pd.DataFrame,
) -> tuple[PCA, np.ndarray, np.ndarray]:
    """
    Esegue la PCA completa per ottenere l'intero spettro
    della varianza spiegata.

    PCA centra automaticamente ogni feature.

    Non viene applicato StandardScaler perché il dataset
    preprocessato è già nell'intervallo [0, 1].
    """
    pca_full = PCA(
        n_components=None,
        svd_solver="full",
    )

    transformed_full = pca_full.fit_transform(data)

    explained_variance_ratio = (
        pca_full.explained_variance_ratio_
    )

    cumulative_variance = np.cumsum(
        explained_variance_ratio
    )

    return (
        pca_full,
        transformed_full,
        cumulative_variance,
    )


def create_transformed_dataframe(
    transformed_data: np.ndarray,
    number_of_components: int,
    original_index: pd.Index,
) -> pd.DataFrame:
    columns = [
        f"PC{i}"
        for i in range(1, number_of_components + 1)
    ]

    return pd.DataFrame(
        transformed_data[:, :number_of_components],
        columns=columns,
        index=original_index,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_path = find_input_file()
    dataset = load_dataset(input_path)

    number_of_samples = dataset.shape[0]
    number_of_original_features = dataset.shape[1]

    print("=" * 70)
    print("PCA ANALYSIS")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Campioni: {number_of_samples}")
    print(
        f"Feature originali: "
        f"{number_of_original_features}"
    )

    minimum_value = dataset.min().min()
    maximum_value = dataset.max().max()

    print(
        f"Intervallo globale dei dati: "
        f"[{minimum_value:.6f}, {maximum_value:.6f}]"
    )

    if minimum_value < 0 or maximum_value > 1:
        print(
            "\nATTENZIONE: i dati non risultano interamente "
            "compresi nell'intervallo [0, 1]."
        )

    # --------------------------------------------------------
    # PCA completa
    # --------------------------------------------------------

    (
        pca_full,
        transformed_full,
        cumulative_variance,
    ) = run_full_pca(dataset)

    explained_variance_ratio = (
        pca_full.explained_variance_ratio_
    )

    total_components = len(explained_variance_ratio)

    print(
        f"Componenti PCA complessivamente calcolate: "
        f"{total_components}"
    )

    # --------------------------------------------------------
    # Componenti richieste dalle diverse soglie
    # --------------------------------------------------------

    threshold_components = {}

    print("\nNumero di componenti per soglia")
    print("-" * 50)

    for threshold in VARIANCE_THRESHOLDS:
        n_components = components_for_threshold(
            cumulative_variance,
            threshold,
        )

        threshold_components[threshold] = n_components

        variance_kept = cumulative_variance[
            n_components - 1
        ]

        reduction_percentage = (
            1
            - n_components / number_of_original_features
        ) * 100

        print(
            f"{threshold:.0%} della varianza: "
            f"{n_components:>4} componenti | "
            f"varianza effettiva = {variance_kept:.6f} | "
            f"riduzione = {reduction_percentage:.2f}%"
        )

    # --------------------------------------------------------
    # Varianza delle prime componenti
    # --------------------------------------------------------

    print("\nPrime componenti")
    print("-" * 50)

    first_components_to_print = min(
        10,
        total_components,
    )

    for component_index in range(
        first_components_to_print
    ):
        individual_variance = (
            explained_variance_ratio[component_index]
        )

        cumulative = cumulative_variance[
            component_index
        ]

        print(
            f"PC{component_index + 1:<3}: "
            f"individuale = "
            f"{individual_variance * 100:7.3f}% | "
            f"cumulativa = "
            f"{cumulative * 100:7.3f}%"
        )

    if total_components >= 2:
        first_two_variance = (
            cumulative_variance[1]
        )

        print(
            "\nVarianza spiegata dalle prime due componenti: "
            f"{first_two_variance * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Grafici
    # --------------------------------------------------------

    plot_full_scree(
        explained_variance_ratio
    )

    plot_detailed_scree(
        explained_variance_ratio
    )

    plot_cumulative_variance(
        cumulative_variance,
        threshold_components,
    )

    plot_first_two_components(
        transformed_full,
        explained_variance_ratio,
    )

    # --------------------------------------------------------
    # Dataset PCA alla soglia selezionata
    # --------------------------------------------------------

    selected_components = components_for_threshold(
        cumulative_variance,
        SELECTED_VARIANCE_THRESHOLD,
    )

    selected_variance = cumulative_variance[
        selected_components - 1
    ]

    transformed_selected = create_transformed_dataframe(
        transformed_data=transformed_full,
        number_of_components=selected_components,
        original_index=dataset.index,
    )

    print("\nPCA selezionata")
    print("-" * 50)
    print(
        f"Soglia richiesta: "
        f"{SELECTED_VARIANCE_THRESHOLD:.0%}"
    )
    print(
        f"Componenti mantenute: "
        f"{selected_components}"
    )
    print(
        f"Varianza effettivamente mantenuta: "
        f"{selected_variance:.6f}"
    )

    if SAVE_TRANSFORMED_DATA:
        output_dataset_path = (
            OUTPUT_DIR
            / (
                "train_pca_"
                f"{int(SELECTED_VARIANCE_THRESHOLD * 100)}.csv"
            )
        )

        transformed_selected.to_csv(
            output_dataset_path,
            index=False,
        )

        print(
            f"Dataset PCA salvato in: "
            f"{output_dataset_path}"
        )

    print("\n" + "=" * 70)
    print("PCA COMPLETATA")
    print("=" * 70)
    print(f"Grafici salvati in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()