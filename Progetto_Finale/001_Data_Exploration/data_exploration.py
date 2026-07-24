#!/usr/bin/env python3
"""Essential exploratory analysis for the Molecular Bioresponse dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t


@dataclass(frozen=True)
class Config:
    repetitions: int = 30
    pairs: int = 20_000
    anchors: int = 100
    candidates: int = 1_000
    top_k: int = 10
    confidence: float = 0.95
    seed: int = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a dataset and compare Jaccard with SMC on binary features."
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--binary-input", type=Path)
    parser.add_argument("--target", default="Activity")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/data_exploration"))
    parser.add_argument("--skip-similarity", action="store_true")
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--pairs", type=int, default=20_000)
    parser.add_argument("--anchors", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=1_000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_features(path: Path, target: str) -> tuple[pd.DataFrame, pd.Series | None]:
    df = pd.read_csv(path).drop(
        columns=lambda c: str(c).startswith("Unnamed:"), errors="ignore"
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    y = df[target].copy() if target in df else None
    return df.drop(columns=[target], errors="ignore"), y


def is_binary(series: pd.Series) -> bool:
    values = set(series.dropna().unique())
    return bool(values) and values.issubset({0, 1, False, True})


def feature_report(X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, s in X.items():
        numeric = pd.api.types.is_numeric_dtype(s)
        binary = is_binary(s)
        valid = s.dropna()
        dominant = valid.value_counts(normalize=True).iloc[0] if len(valid) else np.nan
        row = {
            "feature": name,
            "dtype": str(s.dtype),
            "is_numeric": numeric,
            "is_binary": binary,
            "missing_pct": s.isna().mean() * 100,
            "n_unique": s.nunique(dropna=True),
            "dominant_pct": dominant * 100,
            "constant": s.nunique(dropna=True) <= 1,
            "quasi_constant_99": dominant >= 0.99 if pd.notna(dominant) else False,
        }
        if numeric:
            row |= {
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "median": s.median(),
                "max": s.max(),
                "zero_pct": s.eq(0).mean() * 100,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def run_audit(X: pd.DataFrame, y: pd.Series | None, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    report = feature_report(X)
    report.to_csv(output / "feature_report.csv", index=False)

    if y is not None:
        y.value_counts(dropna=False).sort_index().rename_axis("class").reset_index(
            name="count"
        ).to_csv(output / "target_distribution.csv", index=False)

        y.value_counts(dropna=False).sort_index().plot(kind="bar", figsize=(6, 4))
        plt.title("Target distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        save_plot(output / "target_distribution.png")

    summary = pd.Series(
        {
            "rows": len(X),
            "features": X.shape[1],
            "binary_features": int(report["is_binary"].sum()),
            "numeric_non_binary_features": int(
                (report["is_numeric"] & ~report["is_binary"]).sum()
            ),
            "non_numeric_features": int((~report["is_numeric"]).sum()),
            "missing_values": int(X.isna().sum().sum()),
            "duplicate_rows": int(X.duplicated().sum()),
            "constant_features": int(report["constant"].sum()),
            "quasi_constant_99_features": int(report["quasi_constant_99"].sum()),
            "features_over_95_pct_zeros": int(report["zero_pct"].gt(95).sum()),
        },
        name="value",
    )
    summary.rename_axis("metric").to_csv(output / "audit_summary.csv")
    print("\nGENERAL AUDIT\n", summary.to_string())


def binary_report(B: pd.DataFrame) -> pd.DataFrame:
    ones = B.sum()
    percentages = ones / len(B) * 100
    categories = pd.cut(
        percentages,
        bins=[-np.inf, 5, 40, 60, 95, np.inf],
        labels=["rare_ones", "zero_dominant", "balanced", "one_dominant", "rare_zeros"],
        right=False,
    )
    return pd.DataFrame(
        {
            "feature": B.columns,
            "zero_count": len(B) - ones.to_numpy(),
            "one_count": ones.to_numpy(),
            "zero_pct": 100 - percentages.to_numpy(),
            "one_pct": percentages.to_numpy(),
            "support": percentages.to_numpy() / 100,
            "category": categories.astype(str).to_numpy(),
        }
    ).sort_values("support")


def prepare_binary_data(
    path: Path, target: str, expected_rows: int, output: Path
) -> np.ndarray:
    X, _ = load_features(path, target)
    if len(X) != expected_rows:
        raise ValueError("Binary input and training data have different row counts.")

    columns = [c for c in X if is_binary(X[c])]
    if not columns:
        raise ValueError("No binary 0/1 features found.")

    B = X[columns]
    if B.isna().any().any():
        raise ValueError("Binary features contain missing values.")

    output.mkdir(parents=True, exist_ok=True)
    report = binary_report(B)
    report.to_csv(output / "binary_feature_report.csv", index=False)

    ones_per_row = B.sum(axis=1).to_numpy()
    summary = pd.Series(
        {
            "molecules": len(B),
            "binary_features": B.shape[1],
            "global_zero_pct": B.eq(0).to_numpy().mean() * 100,
            "global_one_pct": B.eq(1).to_numpy().mean() * 100,
            "mean_ones_per_molecule": ones_per_row.mean(),
            "median_ones_per_molecule": np.median(ones_per_row),
            "min_ones_per_molecule": ones_per_row.min(),
            "max_ones_per_molecule": ones_per_row.max(),
        },
        name="value",
    )
    summary.rename_axis("metric").to_csv(output / "binary_summary.csv")

    report["category"].value_counts().plot(kind="bar", figsize=(8, 4))
    plt.title("Binary feature categories")
    plt.ylabel("Features")
    save_plot(output / "binary_categories.png")
    print("\nBINARY ANALYSIS\n", summary.to_string())
    return B.to_numpy(dtype=np.uint8)


def pair_similarities(
    B: np.ndarray, left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    A, C = B[left], B[right]
    intersections = np.logical_and(A, C).sum(axis=1)
    mismatches = np.not_equal(A, C).sum(axis=1)
    unions = intersections + mismatches
    jaccard = np.divide(
        intersections,
        unions,
        out=np.full(len(unions), np.nan),
        where=unions != 0,
    )
    return jaccard, 1 - mismatches / B.shape[1]


def neighbor_overlap(B: np.ndarray, rng: np.random.Generator, cfg: Config) -> float:
    n = len(B)
    n_anchors, n_candidates = min(cfg.anchors, n), min(cfg.candidates, n)
    if cfg.top_k >= n_candidates:
        raise ValueError("top_k must be smaller than the candidate pool.")

    anchors = rng.choice(n, n_anchors, replace=False)
    candidates = rng.choice(n, n_candidates, replace=False)
    A, C = B[anchors].astype(np.int32), B[candidates].astype(np.int32)
    intersections = A @ C.T
    unions = A.sum(1)[:, None] + C.sum(1)[None, :] - intersections
    jaccard = np.divide(
        intersections,
        unions,
        out=np.full(unions.shape, -np.inf, dtype=float),
        where=unions != 0,
    )
    smc = 1 - (
        A.sum(1)[:, None] + C.sum(1)[None, :] - 2 * intersections
    ) / B.shape[1]

    for row, anchor in enumerate(anchors):
        self_match = candidates == anchor
        jaccard[row, self_match] = smc[row, self_match] = -np.inf

    overlaps = []
    for j_row, s_row in zip(jaccard, smc):
        j_top = candidates[np.lexsort((candidates, -j_row))[: cfg.top_k]]
        s_top = candidates[np.lexsort((candidates, -s_row))[: cfg.top_k]]
        overlaps.append(len(set(j_top) & set(s_top)) / cfg.top_k)
    return float(np.mean(overlaps))


def confidence_interval(values: pd.Series, confidence: float) -> tuple[float, float]:
    mean = values.mean()
    margin = t.ppf((1 + confidence) / 2, len(values) - 1) * values.sem()
    return mean - margin, mean + margin


def run_similarity(B: np.ndarray, output: Path, cfg: Config) -> None:
    if cfg.repetitions < 2 or not 0 < cfg.confidence < 1:
        raise ValueError("Use at least 2 repetitions and confidence in (0, 1).")

    rng = np.random.default_rng(cfg.seed)
    rows = []
    for repetition in range(1, cfg.repetitions + 1):
        left = rng.integers(0, len(B), cfg.pairs)
        right = rng.integers(0, len(B), cfg.pairs)
        equal = left == right
        while equal.any():
            right[equal] = rng.integers(0, len(B), equal.sum())
            equal = left == right

        jaccard, smc = pair_similarities(B, left, right)
        valid = np.isfinite(jaccard)
        jaccard, smc = jaccard[valid], smc[valid]
        rows.append(
            {
                "repetition": repetition,
                "mean_jaccard": jaccard.mean(),
                "mean_smc": smc.mean(),
                "pearson": np.corrcoef(jaccard, smc)[0, 1],
                "spearman": spearmanr(jaccard, smc).statistic,
                "mean_absolute_difference": np.abs(jaccard - smc).mean(),
                "top_k_overlap": neighbor_overlap(B, rng, cfg),
            }
        )

    results = pd.DataFrame(rows)
    metrics = results.columns.drop("repetition")
    summary_rows = []
    for metric in metrics:
        lower, upper = confidence_interval(results[metric], cfg.confidence)
        summary_rows.append(
            {
                "metric": metric,
                "mean": results[metric].mean(),
                "std": results[metric].std(ddof=1),
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
    summary = pd.DataFrame(summary_rows)

    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "similarity_repetitions.csv", index=False)
    summary.to_csv(output / "similarity_summary.csv", index=False)

    results[["mean_jaccard", "mean_smc"]].plot.box(figsize=(7, 5))
    plt.title("Jaccard vs Simple Matching")
    plt.ylabel("Mean similarity")
    save_plot(output / "jaccard_vs_smc.png")
    print("\nSIMILARITY ANALYSIS\n", summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    cfg = Config(
        args.repetitions,
        args.pairs,
        args.anchors,
        args.candidates,
        args.top_k,
        args.confidence,
        args.seed,
    )
    X, y = load_features(args.train, args.target)
    run_audit(X, y, args.output_dir / "general")

    binary_path = args.binary_input or args.train
    B = prepare_binary_data(
        binary_path, args.target, len(X), args.output_dir / "binary"
    )
    if not args.skip_similarity:
        run_similarity(B, args.output_dir / "similarity", cfg)

    print(f"\nResults saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()