from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"

PEARSON_THRESHOLD = 0.95
PHI_THRESHOLD = 0.95
POINT_BISERIAL_THRESHOLD = 0.95
ALPHA = 0.05


def filter_symmetric_matrix(input_path, output_path, threshold):
    """Estrae dalla matrice solo le coppie nell'upper triangle."""
    matrix = pd.read_csv(input_path, index_col=0)

    rows, cols = np.triu_indices_from(matrix, k=1)

    results = pd.DataFrame({
        "feature_1": matrix.index[rows],
        "feature_2": matrix.columns[cols],
        "coefficient": matrix.to_numpy()[rows, cols],
    })

    results["abs_coefficient"] = results["coefficient"].abs()

    results = results[
        results["abs_coefficient"] >= threshold
    ].sort_values("abs_coefficient", ascending=False)

    results.to_csv(output_path, index=False)

    return len(results)


# Numerica–numerica: Pearson
n_pearson = filter_symmetric_matrix(
    REPORT_DIR / "pearson_numeric_numeric.csv",
    REPORT_DIR / "pearson_above_threshold.csv",
    PEARSON_THRESHOLD,
)

# Binaria–binaria: Phi
n_phi = filter_symmetric_matrix(
    REPORT_DIR / "phi_binary_binary.csv",
    REPORT_DIR / "phi_above_threshold.csv",
    PHI_THRESHOLD,
)

# Numerica–binaria: punto-biseriale
point_biserial = pd.read_csv(
    REPORT_DIR / "point_biserial_numeric_binary.csv",
    index_col=0,
)

point_biserial = (
    point_biserial
    .stack()
    .reset_index()
)

point_biserial.columns = [
    "numeric_feature",
    "binary_feature",
    "coefficient",
]

point_biserial["abs_coefficient"] = (
    point_biserial["coefficient"].abs()
)

point_biserial = point_biserial[
    point_biserial["abs_coefficient"]
    >= POINT_BISERIAL_THRESHOLD
].sort_values("abs_coefficient", ascending=False)

point_biserial.to_csv(
    REPORT_DIR / "point_biserial_above_threshold.csv",
    index=False,
)

# Chi-quadrato: significatività e forza dell'associazione
chi2_results = pd.read_csv(
    REPORT_DIR / "chi2_binary_binary.csv"
)

chi2_selected = chi2_results[
    (chi2_results["p_value"] < ALPHA)
    & (chi2_results["abs_phi"] >= PHI_THRESHOLD)
].sort_values("abs_phi", ascending=False)

chi2_selected.to_csv(
    REPORT_DIR / "chi2_above_threshold.csv",
    index=False,
)

pearson_pairs = pd.read_csv(
    REPORT_DIR / "pearson_above_threshold.csv"
)

if pearson_pairs.empty:
    print("Nessuna coppia numerica sopra la soglia.")
else:
    print("\nCoppie numeriche sopra la soglia:")
    print(
        pearson_pairs[
            ["feature_1", "feature_2", "coefficient", "abs_coefficient"]
        ].to_string(index=False)
    )

print(f"Pearson sopra soglia: {n_pearson}")
pearson_pairs = pd.read_csv(
    REPORT_DIR / "pearson_above_threshold.csv"
)

if pearson_pairs.empty:
    print("Nessuna coppia numerica sopra la soglia.")
else:
    print("\nCoppie numeriche sopra la soglia:")
    print(
        pearson_pairs[
            ["feature_1", "feature_2", "coefficient", "abs_coefficient"]
        ].to_string(index=False)
    )

print(f"Phi sopra soglia: {n_phi}")
print(f"Punto-biseriale sopra soglia: {len(point_biserial)}")
print(f"Chi-quadrato significativo e Phi sopra soglia: {len(chi2_selected)}")