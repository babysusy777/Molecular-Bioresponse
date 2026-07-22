from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


BASE = Path(
    "reports/spectral_biclustering_pipeline/03_final_model"
)

blocks = pd.read_csv(BASE / "most_distinctive_blocks.csv")
rows = pd.read_csv(BASE / "row_cluster_summary.csv")
columns = pd.read_csv(BASE / "column_cluster_summary.csv")

matrix = (
    blocks.pivot(
        index="row_cluster",
        columns="column_cluster",
        values="mean_contrast_to_column_cluster",
    )
    .sort_index()
    .sort_index(axis=1)
)

row_labels = {
    int(row.row_cluster): (
        f"R{int(row.row_cluster)}\n"
        f"n={int(row.n_rows)}\n"
        f"Activity 1={100 * row.target_1_fraction:.1f}%"
    )
    for _, row in rows.iterrows()
}

column_labels = {
    int(row.column_cluster): (
        f"C{int(row.column_cluster)}\n"
        f"n={int(row.n_features)}\n"
        f"bin={100 * row.binary_feature_fraction:.0f}%"
    )
    for _, row in columns.iterrows()
}

values = matrix.to_numpy(dtype=float)
limit = np.nanmax(np.abs(values))

fig, ax = plt.subplots(figsize=(14, 7))

image = ax.imshow(
    values,
    aspect="auto",
    interpolation="nearest",
    cmap="RdBu_r",
    norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
)

ax.set_xticks(np.arange(matrix.shape[1]))
ax.set_xticklabels(
    [column_labels[int(c)] for c in matrix.columns],
    fontsize=9,
)

ax.set_yticks(np.arange(matrix.shape[0]))
ax.set_yticklabels(
    [row_labels[int(r)] for r in matrix.index],
    fontsize=9,
)

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        value = values[i, j]

        # Non scrive i valori poco informativi.
        if abs(value) < 0.05:
            continue

        ax.text(
            j,
            i,
            f"{value:+.2f}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold" if abs(value) >= 0.15 else "normal",
        )

colorbar = fig.colorbar(image, ax=ax, pad=0.02)
colorbar.set_label("Contrasto medio rispetto al cluster di feature")

ax.set_title(
    "Profili distintivi dei cluster di molecole",
    fontsize=16,
)
ax.set_xlabel("Cluster di feature")
ax.set_ylabel("Cluster di molecole")

fig.tight_layout()
fig.savefig(
    BASE / "block_contrast_heatmap_readable.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)