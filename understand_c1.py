import pandas as pd

X = pd.read_csv(
    "Dataset/train_filtered_no_activity.csv"
)

assignments = pd.read_csv(
    "reports/spectral_biclustering_pipeline/"
    "03_final_model/column_assignments.csv"
)

numeric_c1 = assignments.loc[
    (assignments["column_cluster"] == 1)
    & (~assignments["is_binary"]),
    "feature",
]

X_c1_numeric = X[numeric_c1]

exact_zero_fraction = (X_c1_numeric == 0).to_numpy().mean()
near_zero_fraction = (X_c1_numeric < 0.01).to_numpy().mean()

print(f"Feature numeriche in C1: {len(numeric_c1)}")
print(f"Frazione esattamente zero: {exact_zero_fraction:.4f}")
print(f"Frazione inferiore a 0.01: {near_zero_fraction:.4f}")
print(f"Media: {X_c1_numeric.to_numpy().mean():.6f}")