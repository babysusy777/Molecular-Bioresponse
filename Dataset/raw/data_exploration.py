from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "train.csv"
TEST_PATH = BASE_DIR / "test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Activity può non essere presente nel test
X_train = train.drop(columns="Activity", errors="ignore")
X_test = test.drop(columns="Activity", errors="ignore")

# Feature binarie e numeriche non binarie

binary_features = [
    col for col in X_train.columns
    if X_train[col].dropna().isin([0, 1]).all()
]

numeric_non_binary_features = [
    col for col in X_train.select_dtypes(include="number").columns
    if col not in binary_features
]

print("\nTipologia delle feature:")
print(f"Feature totali: {X_train.shape[1]}")
print(f"Feature binarie: {len(binary_features)}")
print(f"Feature numeriche non binarie: {len(numeric_non_binary_features)}")

# Controllo che le feature coincidano
if set(X_train.columns) != set(X_test.columns):
    raise ValueError("Train e test non contengono le stesse feature.")

X_test = X_test[X_train.columns]

X_all = pd.concat([X_train, X_test], ignore_index=True)

mins = X_all.min()
maxs = X_all.max()

outside_range = (mins < 0) | (maxs > 1)

# Distribuzione delle feature binarie

binary_distribution = []
quit
for col in binary_features:
    pct_zeros = (X_train[col] == 0).mean() * 100
    pct_ones = (X_train[col] == 1).mean() * 100

    if 40 <= pct_ones <= 60:
        distribution = "approximately symmetric"
    elif pct_ones < 40:
        distribution = "asymmetric: zeros dominant"
    else:
        distribution = "asymmetric: ones dominant"

    binary_distribution.append({
        "feature": col,
        "zero_pct": pct_zeros,
        "one_pct": pct_ones,
        "distribution": distribution
    })

binary_distribution = pd.DataFrame(binary_distribution)

print("\nBINARY FEATURE DISTRIBUTION")
print("-" * 60)
print(binary_distribution["distribution"].value_counts())

print(f"Numero totale di feature: {X_all.shape[1]}")
print(f"Feature interamente comprese in [0, 1]: {(~outside_range).sum()}")
print(f"Feature con valori fuori da [0, 1]: {outside_range.sum()}")
print("Valori mancanti:", X_all.isna().sum().sum())
print("Valori infiniti:", np.isinf(X_all.to_numpy()).sum())
print(train["D1"].dtype)
print(train["D1"].head())

if outside_range.any():
    print("\nFeature fuori intervallo:")
    print(
        pd.DataFrame({
            "min": mins[outside_range],
            "max": maxs[outside_range]
        })
    )
else:
    print("\nTutte le feature sono comprese tra 0 e 1.")