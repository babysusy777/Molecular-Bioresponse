from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

INPUT_PATH = PROJECT_DIR / "train_filtered_no_activity.csv"

df = pd.read_csv(INPUT_PATH)
X = df.drop(columns=["Activity"], errors="ignore")

# Feature che assumono esclusivamente valori 0 e 1
binary_cols = [
    col for col in X.columns
    if X[col].dropna().isin([0, 1]).all()
]

zero_ratio = (X[binary_cols] == 0).mean()

# Feature binarie con più del xx% dei valori uguali a 0
majority_zero = zero_ratio[zero_ratio > 0.98].sort_values(ascending=False)

print(f"Feature totali: {X.shape[1]}")
print(f"Feature binarie: {len(binary_cols)}")
print(f"Feature binarie con maggioranza di 0: {len(majority_zero)}")

print("\nPercentuale di zeri per feature:")
print((majority_zero * 100).round(2).to_string())

X_binary = df[binary_cols]

n_zeros = (X_binary == 0).sum().sum()
n_elements = X_binary.size
zero_percentage = n_zeros / n_elements * 100

print(f"Numero di feature binarie: {len(binary_cols)}")
print(f"Numero totale di elementi binari: {n_elements}")
print(f"Numero totale di valori 0: {n_zeros}")
print(f"Percentuale totale di valori 0: {zero_percentage:.2f}%")