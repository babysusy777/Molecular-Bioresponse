import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(
    "/Users/susannabaldo/Desktop/Machine_Learning_Project/"
    "Molecular-Bioresponse/Dataset/train_filtered_no_activity.csv"
)

OUTPUT_PATH = DATA_PATH.parent / "train_numeric_only.csv"

X = pd.read_csv(DATA_PATH)


def is_binary(series: pd.Series) -> bool:
    values = np.unique(series.dropna().to_numpy(dtype=float))
    return len(values) <= 2 and np.all(np.isin(values, [0.0, 1.0]))


constant_cols = [
    column for column in X.columns
    if X[column].nunique(dropna=False) <= 1
]

binary_cols = [
    column for column in X.columns
    if column not in constant_cols and is_binary(X[column])
]

numeric_cols = [
    column for column in X.columns
    if column not in binary_cols
    and column not in constant_cols
]

X_numeric = X[numeric_cols].copy()

print(f"Dataset completo: {X.shape}")
print(f"Feature binarie escluse: {len(binary_cols)}")
print(f"Feature costanti escluse: {len(constant_cols)}")
print(f"Dataset numerico per CLIQUE: {X_numeric.shape}")
print(
    f"Intervallo globale: "
    f"[{X_numeric.min().min():.6f}, {X_numeric.max().max():.6f}]"
)

assert not X_numeric.isna().any().any(), "Sono presenti valori mancanti."
assert len(X_numeric) == len(X), "Il numero di molecole è cambiato."

X_numeric.to_csv(OUTPUT_PATH, index=False)
print(f"Salvato in: {OUTPUT_PATH}")