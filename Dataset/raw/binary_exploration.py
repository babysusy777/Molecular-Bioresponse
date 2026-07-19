from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Numero minimo convenzionale per stimare la variabilità
# tra ripetizioni mediante un intervallo t.
MIN_REPETITIONS = 30
N_REPETITIONS = 30

CONFIDENCE_LEVEL = 0.95

# Numero di coppie casuali valutate in ogni ripetizione.
N_PAIRS_PER_REPETITION = 20_000

# Confronto dei vicini.
N_ANCHORS = 100
N_CANDIDATES = 1_000
TOP_K = 10

# Bootstrap per statistiche a livello di molecola.
N_BOOTSTRAP = 2_000

FIGURE_DPI = 200


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_directory(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "Dataset").exists():
            return candidate

    raise FileNotFoundError(
        "Non è stata trovata la directory principale del progetto "
        "contenente la cartella Dataset."
    )


PROJECT_DIR = find_project_directory(SCRIPT_DIR)

INPUT_CANDIDATES = [
    PROJECT_DIR / "Dataset" / "train_filtered_no_activity.csv",
    PROJECT_DIR / "Dataset" / "processed" / "train_filtered_no_activity.csv",
    PROJECT_DIR / "Dataset" / "preprocessed" / "train_filtered_no_activity.csv",
    PROJECT_DIR / "train_filtered_no_activity.csv",
]

OUTPUT_DIR = PROJECT_DIR / "reports" / "binary_feature_analysis"


def find_input_file() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path

    attempted = "\n".join(str(path) for path in INPUT_CANDIDATES)

    raise FileNotFoundError(
        "File train_filtered_no_activity.csv non trovato.\n"
        f"Percorsi controllati:\n{attempted}"
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(path)

    unnamed_columns = [
        column
        for column in dataset.columns
        if str(column).startswith("Unnamed:")
    ]

    if unnamed_columns:
        dataset = dataset.drop(columns=unnamed_columns)

    return dataset


def identify_binary_features(dataset: pd.DataFrame) -> list[str]:
    binary_columns = []

    for column in dataset.columns:
        values = dataset[column].dropna().unique()

        if len(values) == 0:
            continue

        if set(values).issubset({0, 1, 0.0, 1.0, False, True}):
            binary_columns.append(column)

    return binary_columns


# ============================================================
# FEATURE CATEGORIES
# ============================================================

def assign_binary_category(one_percentage: float) -> str:
    """
    Categorie mutuamente esclusive.
    """
    if one_percentage < 5:
        return "rare_ones"

    if one_percentage < 40:
        return "zero_dominant"

    if one_percentage <= 60:
        return "balanced"

    if one_percentage <= 95:
        return "one_dominant"

    return "rare_zeros"


# ============================================================
# CONFIDENCE INTERVALS
# ============================================================

def t_confidence_interval(
    values: np.ndarray | pd.Series,
    confidence_level: float = 0.95,
) -> tuple[float, float, float, float]:
    """
    Intervallo di confidenza t per la media delle ripetizioni.

    Restituisce:
        media, deviazione standard, limite inferiore, limite superiore
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)

    if n < 2:
        return np.nan, np.nan, np.nan, np.nan

    mean = values.mean()
    standard_deviation = values.std(ddof=1)
    standard_error = standard_deviation / np.sqrt(n)

    alpha = 1 - confidence_level
    critical_value = t.ppf(
        1 - alpha / 2,
        df=n - 1,
    )

    margin = critical_value * standard_error

    return (
        mean,
        standard_deviation,
        mean - margin,
        mean + margin,
    )


def bootstrap_mean_confidence_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = 2_000,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """
    Bootstrap percentile interval per la media.

    Il ricampionamento viene effettuato sulle molecole.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("values deve essere un vettore monodimensionale.")

    n = len(values)

    bootstrap_indices = rng.integers(
        low=0,
        high=n,
        size=(n_bootstrap, n),
    )

    bootstrap_means = values[bootstrap_indices].mean(axis=1)

    alpha = 1 - confidence_level

    lower = np.quantile(
        bootstrap_means,
        alpha / 2,
    )

    upper = np.quantile(
        bootstrap_means,
        1 - alpha / 2,
    )

    return values.mean(), lower, upper


# ============================================================
# PLOTTING
# ============================================================

def save_figure(filename: str) -> None:
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close()


def plot_one_percentage_distribution(
    one_percentages: pd.Series,
) -> None:
    plt.figure(figsize=(9, 6))

    plt.hist(
        one_percentages,
        bins=np.arange(0, 102, 2),
        edgecolor="black",
    )

    for threshold in [5, 40, 60, 95]:
        plt.axvline(
            threshold,
            linestyle="--",
            label=f"{threshold}%",
        )

    plt.xlabel("Percentuale di valori 1 nella feature")
    plt.ylabel("Numero di feature")
    plt.title("Distribuzione della percentuale di 1")
    plt.legend()

    save_figure("01_one_percentage_per_feature.png")


def plot_ones_per_molecule(
    ones_per_molecule: np.ndarray,
) -> None:
    plt.figure(figsize=(9, 6))

    plt.hist(
        ones_per_molecule,
        bins=50,
        edgecolor="black",
    )

    plt.axvline(
        np.mean(ones_per_molecule),
        linestyle="--",
        label=f"Media = {np.mean(ones_per_molecule):.2f}",
    )

    plt.axvline(
        np.median(ones_per_molecule),
        linestyle=":",
        label=f"Mediana = {np.median(ones_per_molecule):.2f}",
    )

    plt.xlabel("Numero di valori 1 per molecola")
    plt.ylabel("Numero di molecole")
    plt.title("Distribuzione del numero di 1 per molecola")
    plt.legend()

    save_figure("02_ones_per_molecule.png")


def plot_global_zero_one_distribution(
    zero_percentage: float,
    one_percentage: float,
) -> None:
    plt.figure(figsize=(7, 6))

    labels = ["0", "1"]
    percentages = [zero_percentage, one_percentage]

    bars = plt.bar(
        labels,
        percentages,
        edgecolor="black",
    )

    for bar, value in zip(bars, percentages):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    plt.xlabel("Valore")
    plt.ylabel("Percentuale")
    plt.title("Distribuzione complessiva di 0 e 1")
    plt.ylim(0, 100)

    save_figure("03_global_zero_one_distribution.png")


def plot_category_distribution(
    category_counts: pd.Series,
) -> None:
    categories = [
        "rare_ones",
        "zero_dominant",
        "balanced",
        "one_dominant",
        "rare_zeros",
    ]

    values = [
        category_counts.get(category, 0)
        for category in categories
    ]

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        categories,
        values,
        edgecolor="black",
    )

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            str(value),
            ha="center",
            va="bottom",
        )

    plt.xlabel("Categoria")
    plt.ylabel("Numero di feature")
    plt.title("Categorie delle feature binarie")
    plt.xticks(rotation=20)

    save_figure("04_binary_feature_categories.png")


# ============================================================
# PAIRWISE SIMILARITIES
# ============================================================

def sample_distinct_pairs(
    number_of_samples: int,
    number_of_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estrae coppie ordinate uniformemente con rimpiazzo,
    imponendo che i due indici siano diversi.

    La presenza di coppie ripetute è ammessa: si tratta di una
    stima Monte Carlo della distribuzione delle similarità.
    """
    left_indices = rng.integers(
        0,
        number_of_samples,
        size=number_of_pairs,
    )

    right_indices = rng.integers(
        0,
        number_of_samples,
        size=number_of_pairs,
    )

    equal_mask = left_indices == right_indices

    while np.any(equal_mask):
        right_indices[equal_mask] = rng.integers(
            0,
            number_of_samples,
            size=equal_mask.sum(),
        )

        equal_mask = left_indices == right_indices

    return left_indices, right_indices


def calculate_pair_similarities(
    binary_matrix: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcola Jaccard e Simple Matching Similarity
    sulle coppie specificate.
    """
    left = binary_matrix[left_indices]
    right = binary_matrix[right_indices]

    intersections = np.logical_and(
        left,
        right,
    ).sum(axis=1)

    mismatches = np.not_equal(
        left,
        right,
    ).sum(axis=1)

    unions = intersections + mismatches

    jaccard = np.divide(
        intersections,
        unions,
        out=np.full(
            intersections.shape,
            np.nan,
            dtype=float,
        ),
        where=unions != 0,
    )

    number_of_features = binary_matrix.shape[1]

    simple_matching = (
        1 - mismatches / number_of_features
    )

    return jaccard, simple_matching


def calculate_similarity_metrics(
    jaccard: np.ndarray,
    simple_matching: np.ndarray,
) -> dict[str, float]:
    valid_mask = (
        np.isfinite(jaccard)
        & np.isfinite(simple_matching)
    )

    jaccard = jaccard[valid_mask]
    simple_matching = simple_matching[valid_mask]

    if len(jaccard) < 2:
        raise ValueError(
            "Numero insufficiente di coppie valide."
        )

    pearson = np.corrcoef(
        jaccard,
        simple_matching,
    )[0, 1]

    spearman_result = spearmanr(
        jaccard,
        simple_matching,
    )

    mean_absolute_difference = np.mean(
        np.abs(jaccard - simple_matching)
    )

    return {
        "mean_jaccard": np.mean(jaccard),
        "mean_smc": np.mean(simple_matching),
        "median_jaccard": np.median(jaccard),
        "median_smc": np.median(simple_matching),
        "pearson": pearson,
        "spearman": spearman_result.statistic,
        "mean_absolute_difference": mean_absolute_difference,
    }


# ============================================================
# TOP-K NEIGHBOR OVERLAP
# ============================================================

def calculate_anchor_candidate_similarities(
    binary_matrix: np.ndarray,
    anchor_indices: np.ndarray,
    candidate_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    anchors = binary_matrix[anchor_indices].astype(
        np.int32,
        copy=False,
    )

    candidates = binary_matrix[candidate_indices].astype(
        np.int32,
        copy=False,
    )

    intersections = anchors @ candidates.T

    anchor_ones = anchors.sum(axis=1)
    candidate_ones = candidates.sum(axis=1)

    unions = (
        anchor_ones[:, None]
        + candidate_ones[None, :]
        - intersections
    )

    jaccard = np.divide(
        intersections,
        unions,
        out=np.full(
            intersections.shape,
            np.nan,
            dtype=float,
        ),
        where=unions != 0,
    )

    mismatches = (
        anchor_ones[:, None]
        + candidate_ones[None, :]
        - 2 * intersections
    )

    number_of_features = binary_matrix.shape[1]

    simple_matching = (
        1 - mismatches / number_of_features
    )

    return jaccard, simple_matching


def deterministic_top_k(
    similarities: np.ndarray,
    candidate_indices: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Se esistono pareggi, usa l'indice originale della molecola
    come criterio deterministico di spareggio.
    """
    number_of_rows = similarities.shape[0]

    top_k_indices = np.empty(
        (number_of_rows, k),
        dtype=int,
    )

    for row in range(number_of_rows):
        ranking = np.lexsort(
            (
                candidate_indices,
                -similarities[row],
            )
        )

        top_k_indices[row] = candidate_indices[
            ranking[:k]
        ]

    return top_k_indices


def calculate_top_k_overlap(
    binary_matrix: np.ndarray,
    rng: np.random.Generator,
) -> float:
    number_of_samples = binary_matrix.shape[0]

    number_of_anchors = min(
        N_ANCHORS,
        number_of_samples,
    )

    number_of_candidates = min(
        N_CANDIDATES,
        number_of_samples,
    )

    if TOP_K >= number_of_candidates:
        raise ValueError(
            "TOP_K deve essere inferiore al numero di candidati."
        )

    anchor_indices = rng.choice(
        number_of_samples,
        size=number_of_anchors,
        replace=False,
    )

    candidate_indices = rng.choice(
        number_of_samples,
        size=number_of_candidates,
        replace=False,
    )

    jaccard, simple_matching = (
        calculate_anchor_candidate_similarities(
            binary_matrix,
            anchor_indices,
            candidate_indices,
        )
    )

    # Elimina l'eventuale confronto di una molecola con sé stessa.
    for row, anchor_index in enumerate(anchor_indices):
        self_mask = candidate_indices == anchor_index

        jaccard[row, self_mask] = -np.inf
        simple_matching[row, self_mask] = -np.inf

    jaccard = np.nan_to_num(
        jaccard,
        nan=-np.inf,
    )

    jaccard_neighbors = deterministic_top_k(
        jaccard,
        candidate_indices,
        TOP_K,
    )

    smc_neighbors = deterministic_top_k(
        simple_matching,
        candidate_indices,
        TOP_K,
    )

    overlaps = []

    for row in range(number_of_anchors):
        jaccard_set = set(jaccard_neighbors[row])
        smc_set = set(smc_neighbors[row])

        overlap = len(
            jaccard_set.intersection(smc_set)
        ) / TOP_K

        overlaps.append(overlap)

    return float(np.mean(overlaps))


# ============================================================
# REPEATED MONTE CARLO ANALYSIS
# ============================================================

def repeated_similarity_analysis(
    binary_matrix: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if N_REPETITIONS < MIN_REPETITIONS:
        raise ValueError(
            f"N_REPETITIONS deve essere almeno {MIN_REPETITIONS}."
        )

    results = []

    number_of_samples = binary_matrix.shape[0]

    for repetition in range(1, N_REPETITIONS + 1):
        left_indices, right_indices = sample_distinct_pairs(
            number_of_samples=number_of_samples,
            number_of_pairs=N_PAIRS_PER_REPETITION,
            rng=rng,
        )

        jaccard, simple_matching = (
            calculate_pair_similarities(
                binary_matrix,
                left_indices,
                right_indices,
            )
        )

        metrics = calculate_similarity_metrics(
            jaccard,
            simple_matching,
        )

        metrics["top_k_overlap"] = calculate_top_k_overlap(
            binary_matrix,
            rng,
        )

        metrics["repetition"] = repetition

        results.append(metrics)

        print(
            f"Ripetizione {repetition:>2}/{N_REPETITIONS}: "
            f"J={metrics['mean_jaccard']:.4f}, "
            f"SMC={metrics['mean_smc']:.4f}, "
            f"overlap={metrics['top_k_overlap'] * 100:.2f}%"
        )

    return pd.DataFrame(results)


def print_repeated_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    metric_labels = {
        "mean_jaccard": "Jaccard media",
        "mean_smc": "SMC media",
        "median_jaccard": "Jaccard mediana",
        "median_smc": "SMC mediana",
        "pearson": "Correlazione Pearson",
        "spearman": "Correlazione Spearman",
        "mean_absolute_difference": "Differenza assoluta media",
        "top_k_overlap": "Overlap top-k",
    }

    summary_rows = []

    print("\n" + "=" * 86)
    print(
        f"RISULTATI SU {N_REPETITIONS} RIPETIZIONI "
        f"— INTERVALLO DI CONFIDENZA {CONFIDENCE_LEVEL:.0%}"
    )
    print("=" * 86)

    for metric, label in metric_labels.items():
        mean, std, lower, upper = t_confidence_interval(
            results[metric],
            confidence_level=CONFIDENCE_LEVEL,
        )

        summary_rows.append(
            {
                "metric": metric,
                "label": label,
                "mean": mean,
                "std": std,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )

        if metric == "top_k_overlap":
            print(
                f"{label:<28}: "
                f"{mean * 100:7.2f}% "
                f"± {std * 100:6.2f}% | "
                f"IC: [{lower * 100:.2f}%, "
                f"{upper * 100:.2f}%]"
            )
        else:
            print(
                f"{label:<28}: "
                f"{mean:8.6f} "
                f"± {std:8.6f} | "
                f"IC: [{lower:.6f}, {upper:.6f}]"
            )

    return pd.DataFrame(summary_rows)


# ============================================================
# RESULTS PLOTS
# ============================================================

def plot_repetition_distributions(
    results: pd.DataFrame,
) -> None:
    plt.figure(figsize=(9, 6))

    plt.boxplot(
        [
            results["mean_jaccard"],
            results["mean_smc"],
        ],
        tick_labels=[
            "Jaccard",
            "Simple Matching",
        ],
    )

    plt.ylabel("Similarità media")
    plt.title(
        "Distribuzione delle similarità medie "
        "tra le ripetizioni"
    )

    save_figure("05_repeated_similarity_boxplot.png")

    plt.figure(figsize=(9, 6))

    plt.hist(
        results["top_k_overlap"] * 100,
        bins=10,
        edgecolor="black",
    )

    mean_overlap = results["top_k_overlap"].mean() * 100

    plt.axvline(
        mean_overlap,
        linestyle="--",
        label=f"Media = {mean_overlap:.2f}%",
    )

    plt.xlabel("Overlap dei vicini top-k (%)")
    plt.ylabel("Numero di ripetizioni")
    plt.title(
        f"Distribuzione dell'overlap top-{TOP_K}"
    )
    plt.legend()

    save_figure("06_top_k_overlap_repetitions.png")


def plot_confidence_intervals(
    summary: pd.DataFrame,
) -> None:
    metrics = [
        "mean_jaccard",
        "mean_smc",
        "pearson",
        "spearman",
        "top_k_overlap",
    ]

    selected = summary[
        summary["metric"].isin(metrics)
    ].copy()

    selected = selected.set_index(
        "metric"
    ).loc[metrics].reset_index()

    labels = [
        "Jaccard",
        "SMC",
        "Pearson",
        "Spearman",
        "Overlap",
    ]

    means = selected["mean"].to_numpy()

    lower_errors = (
        means - selected["ci_lower"].to_numpy()
    )

    upper_errors = (
        selected["ci_upper"].to_numpy() - means
    )

    plt.figure(figsize=(10, 6))

    plt.errorbar(
        labels,
        means,
        yerr=[lower_errors, upper_errors],
        fmt="o",
        capsize=5,
    )

    plt.ylabel("Valore medio")
    plt.title(
        f"Medie e intervalli di confidenza "
        f"al {CONFIDENCE_LEVEL:.0%}"
    )

    save_figure("07_confidence_intervals.png")


# ============================================================
# SUPPORT ANALYSIS
# ============================================================

def print_support_analysis(
    one_fractions: pd.Series,
) -> None:
    support_table = pd.DataFrame(
        {
            "feature": one_fractions.index,
            "support": one_fractions.values,
            "support_percentage": one_fractions.values * 100,
        }
    )

    support_table = support_table.sort_values(
        "support",
        ascending=False,
    )

    formatters = {
        "support": lambda value: f"{value:.6f}",
        "support_percentage": lambda value: f"{value:.2f}%",
    }

    print("\nFeature con supporto maggiore")
    print("-" * 60)
    print(
        support_table.head(20).to_string(
            index=False,
            formatters=formatters,
        )
    )

    positive_support = support_table[
        support_table["support"] > 0
    ]

    print("\nFeature non costanti con supporto minore")
    print("-" * 60)
    print(
        positive_support.tail(20).to_string(
            index=False,
            formatters=formatters,
        )
    )

    print(
        "\nFeature con supporto inferiore al 5%: "
        f"{(one_fractions < 0.05).sum()}"
    )

    print(
        "Feature con supporto nullo: "
        f"{(one_fractions == 0).sum()}"
    )

    print(
        "Feature con supporto unitario: "
        f"{(one_fractions == 1).sum()}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_path = find_input_file()
    dataset = load_dataset(input_path)

    binary_columns = identify_binary_features(dataset)

    if not binary_columns:
        raise ValueError(
            "Non è stata trovata alcuna feature binaria."
        )

    binary_data = dataset[binary_columns].copy()

    if binary_data.isna().any().any():
        raise ValueError(
            "Le feature binarie contengono valori mancanti."
        )

    binary_data = binary_data.astype(np.uint8)
    binary_matrix = binary_data.to_numpy(dtype=np.uint8)

    number_of_molecules = binary_data.shape[0]
    number_of_binary_features = binary_data.shape[1]

    print("=" * 70)
    print("BINARY FEATURE ANALYSIS")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Molecole: {number_of_molecules}")
    print(f"Feature totali: {dataset.shape[1]}")
    print(
        f"Feature binarie identificate: "
        f"{number_of_binary_features}"
    )

    # --------------------------------------------------------
    # Percentuale di 1 per feature
    # --------------------------------------------------------

    one_fractions = binary_data.mean(axis=0)
    one_percentages = one_fractions * 100

    feature_categories = one_percentages.apply(
        assign_binary_category
    )

    category_order = [
        "rare_ones",
        "zero_dominant",
        "balanced",
        "one_dominant",
        "rare_zeros",
    ]

    category_counts = (
        feature_categories
        .value_counts()
        .reindex(category_order, fill_value=0)
    )

    print("\nCategorie delle feature binarie")
    print("-" * 60)

    for category in category_order:
        count = category_counts[category]
        percentage = (
            count / number_of_binary_features * 100
        )

        print(
            f"{category:<15}: "
            f"{count:>4} ({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Distribuzione degli 1 per molecola
    # --------------------------------------------------------

    ones_per_molecule = binary_matrix.sum(axis=1)
    one_fraction_per_molecule = (
        ones_per_molecule / number_of_binary_features
    )

    mean_ones, lower_ones, upper_ones = (
        bootstrap_mean_confidence_interval(
            ones_per_molecule,
            rng=rng,
            n_bootstrap=N_BOOTSTRAP,
            confidence_level=CONFIDENCE_LEVEL,
        )
    )

    mean_one_fraction, lower_fraction, upper_fraction = (
        bootstrap_mean_confidence_interval(
            one_fraction_per_molecule,
            rng=rng,
            n_bootstrap=N_BOOTSTRAP,
            confidence_level=CONFIDENCE_LEVEL,
        )
    )

    print("\nNumero di 1 per molecola")
    print("-" * 60)
    print(
        f"Media:   {mean_ones:.2f}"
    )
    print(
        f"IC {CONFIDENCE_LEVEL:.0%}: "
        f"[{lower_ones:.2f}, {upper_ones:.2f}]"
    )
    print(
        f"Mediana: {np.median(ones_per_molecule):.2f}"
    )
    print(
        f"Minimo:  {np.min(ones_per_molecule)}"
    )
    print(
        f"Massimo: {np.max(ones_per_molecule)}"
    )
    print(
        f"Std:     {np.std(ones_per_molecule, ddof=1):.2f}"
    )

    # --------------------------------------------------------
    # Distribuzione globale 0/1
    # --------------------------------------------------------

    total_elements = binary_matrix.size
    total_ones = int(binary_matrix.sum())
    total_zeros = total_elements - total_ones

    one_percentage_global = (
        total_ones / total_elements * 100
    )

    zero_percentage_global = (
        total_zeros / total_elements * 100
    )

    print("\nDistribuzione complessiva")
    print("-" * 60)
    print(f"Elementi binari totali: {total_elements:,}")
    print(f"Valori 0: {total_zeros:,}")
    print(f"Valori 1: {total_ones:,}")
    print(
        f"Percentuale 0: {zero_percentage_global:.2f}%"
    )
    print(
        f"Percentuale 1: {one_percentage_global:.2f}%"
    )
    print(
        f"IC bootstrap {CONFIDENCE_LEVEL:.0%} "
        f"della percentuale media di 1: "
        f"[{lower_fraction * 100:.2f}%, "
        f"{upper_fraction * 100:.2f}%]"
    )

    # --------------------------------------------------------
    # Grafici descrittivi
    # --------------------------------------------------------

    plot_one_percentage_distribution(
        one_percentages
    )

    plot_ones_per_molecule(
        ones_per_molecule
    )

    plot_global_zero_one_distribution(
        zero_percentage_global,
        one_percentage_global,
    )

    plot_category_distribution(
        category_counts
    )

    # --------------------------------------------------------
    # Supporto delle singole feature
    # --------------------------------------------------------

    print_support_analysis(
        one_fractions
    )

    # --------------------------------------------------------
    # Confronto ripetuto Jaccard-SMC
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONFRONTO RIPETUTO JACCARD-SMC")
    print("=" * 70)
    print(f"Ripetizioni: {N_REPETITIONS}")
    print(
        "Coppie per ripetizione: "
        f"{N_PAIRS_PER_REPETITION:,}"
    )
    print(f"Anchor per ripetizione: {N_ANCHORS}")
    print(
        "Candidati per anchor: "
        f"{N_CANDIDATES}"
    )
    print(f"Top-k: {TOP_K}")

    repeated_results = repeated_similarity_analysis(
        binary_matrix,
        rng,
    )

    summary = print_repeated_results(
        repeated_results
    )

    plot_repetition_distributions(
        repeated_results
    )

    plot_confidence_intervals(
        summary
    )

    print("\n" + "=" * 70)
    print("ANALISI COMPLETATA")
    print("=" * 70)
    print(f"Grafici salvati in: {OUTPUT_DIR}")
    print("Non è stato generato alcun file CSV.")
    print("FP-Growth non è stato eseguito.")


if __name__ == "__main__":
    main()