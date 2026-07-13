from pathlib import Path
from itertools import combinations

import pandas as pd
from scipy.stats import chi2


BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / ".." / "Dataset" / "raw" / "train.csv"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)


X = pd.read_csv(INPUT_PATH)

binary_cols = [
    col for col in X.columns
    if set(X[col].dropna().unique()).issubset({0, 1})
]
numeric_cols = [col for col in X.columns if col not in binary_cols]

# Pearson, Phi e punto-biseriale sono tutti ottenibili dalla correlazione
# di Pearson quando le variabili binarie sono codificate come 0/1.
corr = X.corr(method="pearson")

corr.loc[numeric_cols, numeric_cols].to_csv(
    REPORT_DIR / "pearson_numeric_numeric.csv"
)

corr.loc[binary_cols, binary_cols].to_csv(
    REPORT_DIR / "phi_binary_binary.csv"
)

corr.loc[numeric_cols, binary_cols].to_csv(
    REPORT_DIR / "point_biserial_numeric_binary.csv"
)


# Test chi-quadrato binaria-binaria.
# Per una tabella 2x2, senza correzione di Yates: chi2 = n * phi^2.
results = []
n = len(X)

for feature_1, feature_2 in combinations(binary_cols, 2):
    phi = corr.loc[feature_1, feature_2]

    if pd.isna(phi):
        continue

    chi2_statistic = n * phi**2
    p_value = chi2.sf(chi2_statistic, df=1)

    results.append({
        "feature_1": feature_1,
        "feature_2": feature_2,
        "chi2": chi2_statistic,
        "p_value": p_value,
        "phi": phi,
        "abs_phi": abs(phi),
    })

pd.DataFrame(results).to_csv(
    REPORT_DIR / "chi2_binary_binary.csv",
    index=False
)

print(f"Feature numeriche: {len(numeric_cols)}")
print(f"Feature binarie: {len(binary_cols)}")
print(f"Coppie binarie analizzate: {len(results)}")
print(f"Risultati salvati in: {REPORT_DIR}")