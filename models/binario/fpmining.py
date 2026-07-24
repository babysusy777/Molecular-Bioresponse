from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralCoclustering


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_PATH = BASE_DIR / "Dataset" / "raw" / "train.csv"

OUTPUT_DIR = BASE_DIR / "reports" / "binary_spectral_coclustering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Filtro delle feature troppo rare o quasi sempre attive
MIN_PREVALENCE = 0.02
MAX_PREVALENCE = 0.98

# Numero di cluster di righe e colonne
ROW_CLUSTERS = 4
COLUMN_CLUSTERS = 4


# ============================================================
# LETTURA DATI
# ============================================================

train = pd.read_csv(TRAIN_PATH)

if "Activity" in train.columns:
    activity = train["Activity"].copy()
    X = train.drop(columns="Activity")
else:
    activity = None
    X = train.copy()

X = X.apply(pd.to_numeric, errors="raise")

if X.isna().any().any():
    raise ValueError("Il dataset contiene valori mancanti.")

if np.isinf(X.to_numpy()).any():
    raise ValueError("Il dataset contiene valori infiniti.")


# ============================================================
# SELEZIONE DELLE FEATURE BINARIE
# ============================================================

binary_features = [
    column
    for column in X.columns
    if X[column].isin([0, 1]).all()
]

X_binary = X[binary_features].copy()

print("\nFEATURE BINARIE")
print("-" * 60)
print(f"Feature binarie iniziali: {X_binary.shape[1]}")


# ============================================================
# FILTRO PER PREVALENZA
# ============================================================

prevalence = X_binary.mean(axis=0)

selected_features = prevalence[
    (prevalence >= MIN_PREVALENCE)
    & (prevalence <= MAX_PREVALENCE)
].index.tolist()

X_filtered = X_binary[selected_features].copy()

print(f"Prevalenza minima: {MIN_PREVALENCE:.2%}")
print(f"Prevalenza massima: {MAX_PREVALENCE:.2%}")
print(f"Feature mantenute: {X_filtered.shape[1]}")
print(f"Feature eliminate: {X_binary.shape[1] - X_filtered.shape[1]}")

if X_filtered.shape[1] == 0:
    raise ValueError(
        "Nessuna feature supera il filtro di prevalenza."
    )


# ============================================================
# CONVERSIONE IN MATRICE SPARSA
# ============================================================

X_sparse = csr_matrix(
    X_filtered.to_numpy(dtype=float)
)


# ============================================================
# SPECTRAL CO-CLUSTERING
# ============================================================

if ROW_CLUSTERS != COLUMN_CLUSTERS:
    raise ValueError(
        "SpectralCoclustering richiede lo stesso numero "
        "di cluster per righe e colonne."
    )

model = SpectralCoclustering(
    n_clusters=4,
    random_state=42,
)

model.fit(X_sparse)

model.fit(X_sparse)

model.fit(X_sparse)

row_labels = model.row_labels_
column_labels = model.column_labels_

print("\nSPECTRAL CO-CLUSTERING")
print("-" * 60)
print(f"Cluster di righe: {ROW_CLUSTERS}")
print(f"Cluster di colonne: {COLUMN_CLUSTERS}")


# ============================================================
# ANALISI DEI BLOCCHI
# ============================================================

block_records = []

for row_cluster in range(ROW_CLUSTERS):

    row_mask = row_labels == row_cluster
    row_indices = np.flatnonzero(row_mask)

    for column_cluster in range(COLUMN_CLUSTERS):

        column_mask = column_labels == column_cluster
        column_indices = np.flatnonzero(column_mask)

        block = X_filtered.iloc[
            row_indices,
            column_indices,
        ]

        n_rows = block.shape[0]
        n_columns = block.shape[1]

        if n_rows == 0 or n_columns == 0:
            density = np.nan
            volume = 0
            one_count = 0
        else:
            one_count = int(block.to_numpy().sum())
            volume = n_rows * n_columns
            density = one_count / volume

        record = {
            "row_cluster": row_cluster,
            "column_cluster": column_cluster,
            "n_rows": n_rows,
            "n_features": n_columns,
            "volume": volume,
            "one_count": one_count,
            "one_density": density,
        }

        if activity is not None and n_rows > 0:
            activity_subset = activity.iloc[row_indices]

            record.update({
                "activity_0_count": int(
                    (activity_subset == 0).sum()
                ),
                "activity_1_count": int(
                    (activity_subset == 1).sum()
                ),
                "activity_1_fraction": float(
                    activity_subset.mean()
                ),
            })

        block_records.append(record)


block_summary = pd.DataFrame(block_records)

block_summary = block_summary.sort_values(
    by="one_density",
    ascending=False,
)

print("\nDENSITÀ DEI BLOCCHI")
print("-" * 100)
print(block_summary.to_string(index=False))

row_order = np.argsort(row_labels)
column_order = np.argsort(column_labels)

X_reordered = X_filtered.iloc[
    row_order,
    column_order
].to_numpy()

sorted_row_labels = row_labels[row_order]
sorted_column_labels = column_labels[column_order]

# Posizioni in cui cambia il cluster
row_boundaries = (
    np.flatnonzero(
        sorted_row_labels[1:] != sorted_row_labels[:-1]
    ) + 0.5
)

column_boundaries = (
    np.flatnonzero(
        sorted_column_labels[1:] != sorted_column_labels[:-1]
    ) + 0.5
)

plt.figure(figsize=(14, 9))

plt.imshow(
    X_reordered,
    aspect="auto",
    interpolation="nearest",
    cmap="binary",
)

for boundary in row_boundaries:
    plt.axhline(
        boundary,
        linewidth=1.5,
        linestyle="--",
    )

for boundary in column_boundaries:
    plt.axvline(
        boundary,
        linewidth=1.5,
        linestyle="--",
    )

plt.xlabel("Feature binarie riordinate")
plt.ylabel("Molecole riordinate")
plt.title("Spectral co-clustering sulle feature binarie")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "reordered_binary_matrix_with_boundaries.png",
    dpi=300,
)
plt.close()

print("\nDIMENSIONI DEI CLUSTER DI RIGHE")
print(
    pd.Series(row_labels)
    .value_counts()
    .sort_index()
    .rename("n_rows")
)

print("\nDIMENSIONI DEI CLUSTER DI COLONNE")
print(
    pd.Series(column_labels)
    .value_counts()
    .sort_index()
    .rename("n_features")
)


# ============================================================
# REPORT DEI CLUSTER DI RIGHE
# ============================================================

row_records = []

for row_index, row_cluster in enumerate(row_labels):

    record = {
        "row_index": row_index,
        "row_cluster": row_cluster,
    }

    if activity is not None:
        record["Activity"] = activity.iloc[row_index]

    row_records.append(record)

row_assignments = pd.DataFrame(row_records)


# ============================================================
# REPORT DEI CLUSTER DI COLONNE
# ============================================================

column_assignments = pd.DataFrame({
    "feature": X_filtered.columns,
    "column_cluster": column_labels,
    "prevalence": prevalence[X_filtered.columns].to_numpy(),
})


# ============================================================
# MATRICE RIORDINATA
# ============================================================

row_order = np.argsort(row_labels)
column_order = np.argsort(column_labels)

X_reordered = X_filtered.iloc[
    row_order,
    column_order,
].to_numpy()


plt.figure(figsize=(14, 9))

plt.imshow(
    X_reordered,
    aspect="auto",
    interpolation="nearest",
    cmap="binary",
)

plt.xlabel("Feature binarie riordinate")
plt.ylabel("Molecole riordinate")
plt.title(
    "Spectral co-clustering sulle feature binarie"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "reordered_binary_matrix.png",
    dpi=300,
)

plt.close()


# ============================================================
# HEATMAP DELLA DENSITÀ DEI BLOCCHI
# ============================================================

density_matrix = block_summary.pivot(
    index="row_cluster",
    columns="column_cluster",
    values="one_density",
)

plt.figure(figsize=(8, 6))

image = plt.imshow(
    density_matrix,
    aspect="auto",
    interpolation="nearest",
)

plt.colorbar(
    image,
    label="Densità di valori 1",
)

plt.xticks(
    range(COLUMN_CLUSTERS),
    density_matrix.columns,
)

plt.yticks(
    range(ROW_CLUSTERS),
    density_matrix.index,
)

plt.xlabel("Cluster di feature")
plt.ylabel("Cluster di molecole")
plt.title("Densità dei blocchi righe-colonne")

for row_position in range(ROW_CLUSTERS):
    for column_position in range(COLUMN_CLUSTERS):

        value = density_matrix.iloc[
            row_position,
            column_position,
        ]

        plt.text(
            column_position,
            row_position,
            f"{value:.3f}",
            ha="center",
            va="center",
        )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "block_density_heatmap.png",
    dpi=300,
)

plt.close()


# ============================================================
# SALVATAGGIO
# ============================================================

block_summary.to_csv(
    OUTPUT_DIR / "block_summary.csv",
    index=False,
)

row_assignments.to_csv(
    OUTPUT_DIR / "row_clusters.csv",
    index=False,
)

column_assignments.to_csv(
    OUTPUT_DIR / "feature_clusters.csv",
    index=False,
)

pd.DataFrame({
    "feature": X_binary.columns,
    "prevalence": prevalence,
    "selected": X_binary.columns.isin(selected_features),
}).to_csv(
    OUTPUT_DIR / "binary_feature_prevalence.csv",
    index=False,
)

print(f"\nRisultati salvati in:\n{OUTPUT_DIR}")