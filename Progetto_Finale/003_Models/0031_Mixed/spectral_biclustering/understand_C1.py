#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    focus_column_cluster: int = 1
    near_zero_threshold: float = 0.01


def find_project_paths(start: Path) -> tuple[Path, Path]:
    """Return the input matrix and final-model assignment paths."""
    for candidate in (start.resolve(), *start.resolve().parents):
        current_input = (
            candidate / "000_Dataset" / "train_filtered_no_activity.csv"
        )
        if current_input.is_file() and (candidate / "003_Models").is_dir():
            assignments = (
                candidate
                / "003_Models"
                / "0031_Mixed"
                / "spectral_biclustering"
                / "reports"
                / "spectral_biclustering_pipeline"
                / "03_final_model"
                / "column_assignments.csv"
            )
            return current_input, assignments

        legacy_input = (
            candidate / "Dataset" / "train_filtered_no_activity.csv"
        )
        if legacy_input.is_file():
            assignments = (
                candidate
                / "reports"
                / "spectral_biclustering_pipeline"
                / "03_final_model"
                / "column_assignments.csv"
            )
            return legacy_input, assignments

    raise FileNotFoundError(
        "Project root not found. Expected the preprocessed feature matrix in "
        "'000_Dataset' or 'Dataset'."
    )


def coerce_boolean(series: pd.Series) -> pd.Series:
    """Safely convert bool/0-1/True-False values to Boolean."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        if not series.isin([0, 1]).all():
            raise ValueError("is_binary contains values other than 0 and 1.")
        return series.astype(bool)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = sorted(set(normalized) - set(mapping))
    if invalid:
        raise ValueError(
            f"is_binary contains invalid values: {invalid[:10]}"
        )
    return normalized.map(mapping).astype(bool)


def analyze_numeric_cluster(
    X: pd.DataFrame,
    assignments: pd.DataFrame,
    config: Config,
) -> dict[str, float | int]:
    """Compute the same C1 diagnostics as the original script."""
    required = {"feature", "column_cluster", "is_binary"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(
            f"column_assignments.csv is missing columns: {sorted(missing)}"
        )

    assignments = assignments.copy()
    assignments["is_binary"] = coerce_boolean(assignments["is_binary"])
    numeric_features = assignments.loc[
        (
            assignments["column_cluster"].astype(int)
            == config.focus_column_cluster
        )
        & ~assignments["is_binary"],
        "feature",
    ].astype(str)

    missing_features = sorted(set(numeric_features) - set(X.columns))
    if missing_features:
        raise ValueError(
            "Assigned features are absent from the input matrix. "
            f"First missing names: {missing_features[:10]}"
        )
    if numeric_features.empty:
        raise ValueError(
            f"C{config.focus_column_cluster} contains no numeric features."
        )

    values = (
        X.loc[:, numeric_features.tolist()]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=float)
    )
    if not np.isfinite(values).all():
        raise ValueError("The selected numeric values contain NaN or infinity.")

    return {
        "n_numeric_features": int(len(numeric_features)),
        "exact_zero_fraction": float(np.mean(values == 0.0)),
        "near_zero_fraction": float(
            np.mean(values < config.near_zero_threshold)
        ),
        "mean": float(values.mean()),
    }


def main() -> None:
    config = Config()
    input_path, assignments_path = find_project_paths(
        Path(__file__).resolve().parent
    )
    if not assignments_path.is_file():
        raise FileNotFoundError(
            f"Final-model assignments not found: {assignments_path}"
        )

    X = pd.read_csv(input_path)
    X = X.loc[
        :,
        ~X.columns.astype(str).str.startswith("Unnamed:"),
    ]
    assignments = pd.read_csv(assignments_path)
    result = analyze_numeric_cluster(X, assignments, config)

    cluster = config.focus_column_cluster
    threshold = config.near_zero_threshold
    print(
        f"Feature numeriche in C{cluster}: "
        f"{result['n_numeric_features']}"
    )
    print(
        "Frazione esattamente zero: "
        f"{result['exact_zero_fraction']:.4f}"
    )
    print(
        f"Frazione inferiore a {threshold:g}: "
        f"{result['near_zero_fraction']:.4f}"
    )
    print(f"Media: {result['mean']:.6f}")


if __name__ == "__main__":
    main()