from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# This file is inside:
# Molecular-Bioresponse/models/hopkins.py
# Therefore:
# BASE_DIR = Molecular-Bioresponse/models
# PROJECT_DIR = Molecular-Bioresponse
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

INPUT_PATH = PROJECT_DIR / "Dataset" / "train_pca_90_no_activity.csv"
REPORT_PATH = PROJECT_DIR / "reports" / "hopkins_report.csv"
SUMMARY_PATH = PROJECT_DIR / "reports" / "hopkins_summary.csv"

N_REPETITIONS = 30
SAMPLE_FRACTION = 0.10
MIN_SAMPLES = 50
MAX_SAMPLES = 500


def interpret_hopkins(h):
    """Simple qualitative interpretation of the Hopkins statistic."""
    if h < 0.60:
        return "No clear clustering tendency / approximately random"
    if h < 0.75:
        return "Moderate clustering tendency"
    return "Strong clustering tendency"


def compute_hopkins(X, n_samples, seed):
    """
    Hopkins statistic compares:
    - distances among real points
    - distances between random uniform points and real data

    If random points are much farther than real points, H tends to 1,
    meaning that the dataset has a clustering tendency.
    """

    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)

    n_rows, n_features = X.shape

    # Fit nearest-neighbor model once on the real dataset
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X)

    # 1) Sample real points from the dataset
    real_indices = rng.choice(n_rows, size=n_samples, replace=False)
    X_real = X[real_indices]

    # For real points, the nearest neighbor is the point itself.
    # So we take the second nearest neighbor.
    real_distances = nn.kneighbors(X_real, return_distance=True)[0][:, 1]

    # 2) Generate random points uniformly in the same feature ranges
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    X_random = rng.uniform(mins, maxs, size=(n_samples, n_features))

    # For random points, take the nearest real data point
    random_distances = nn.kneighbors(
        X_random,
        n_neighbors=1,
        return_distance=True
    )[0][:, 0]

    # 3) Hopkins formula
    H = random_distances.sum() / (real_distances.sum() + random_distances.sum())

    return H


def main():
    print("Starting Hopkins script...", flush=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    # Load PCA-reduced molecular descriptors, without Activity
    X = pd.read_csv(INPUT_PATH)

    # Keep only numeric columns, just to be safe
    X = X.drop(columns=["Unnamed: 0"], errors="ignore")
    X = X.select_dtypes(include=["number"])

    if X.empty:
        raise ValueError("No numeric columns found in the input dataset.")

    n_rows = X.shape[0]

    # Choose how many points to sample for each repetition
    n_samples = int(SAMPLE_FRACTION * n_rows)
    n_samples = max(n_samples, MIN_SAMPLES)
    n_samples = min(n_samples, MAX_SAMPLES, n_rows - 1)

    print(f"Input file: {INPUT_PATH}", flush=True)
    print(f"Dataset shape: {X.shape}", flush=True)
    print(f"Samples per repetition: {n_samples}", flush=True)
    print(f"Repetitions: {N_REPETITIONS}", flush=True)

    hopkins_values = []

    for seed in range(N_REPETITIONS):
        h = compute_hopkins(X, n_samples=n_samples, seed=seed)
        hopkins_values.append(h)
        print(f"Repetition {seed + 1}/{N_REPETITIONS}: H = {h:.4f}", flush=True)

    hopkins_values = np.array(hopkins_values)

    mean_h = hopkins_values.mean()
    std_h = hopkins_values.std()
    interpretation = interpret_hopkins(mean_h)

    # Save all Hopkins values
    report = pd.DataFrame({
        "repetition": range(1, N_REPETITIONS + 1),
        "hopkins": hopkins_values
    })

    # Save summary
    summary = pd.DataFrame({
        "mean_hopkins": [mean_h],
        "std_hopkins": [std_h],
        "n_repetitions": [N_REPETITIONS],
        "n_samples": [n_samples],
        "interpretation": [interpretation]
    })

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report.to_csv(REPORT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nHopkins statistic completed")
    print(f"Mean Hopkins: {mean_h:.4f}")
    print(f"Std Hopkins: {std_h:.4f}")
    print(f"Interpretation: {interpretation}")
    print(f"Saved full report to: {REPORT_PATH}")
    print(f"Saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()