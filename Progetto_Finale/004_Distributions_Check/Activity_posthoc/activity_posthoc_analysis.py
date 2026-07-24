from __future__ import annotations

"""Post-hoc analysis of Activity distributions across finalized NJW partitions.

Expected location
-----------------
PROGETTO_FINALE/
└── 004_Distributions_Check/
    └── activity_posthoc/
        └── activity_posthoc_analysis.py

The script reads already finalized clustering labels from the mixed, binary-only
and numeric-only NJW analyses. Activity is joined only after clustering and is
never used for preprocessing, distance construction, parameter selection or
cluster formation.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# =============================================================================
# 1. CONFIGURATION AND PROJECT PATHS
# =============================================================================

SCRIPT_VERSION = "activity-posthoc-pathfix-v2-2026-07-24"
TARGET_COL = "Activity"
ROW_COL = "row_index"
MIN_CLUSTER_FRACTION = 0.05
FIGURE_DPI = 220

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path | None:
    """Find PROGETTO_FINALE by its required internal directories."""
    for candidate in (start, *start.parents):
        if (candidate / "000_Dataset").is_dir() and (
            candidate / "003_Models"
        ).is_dir():
            return candidate
    return None


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
_DEFAULT_ROOT = (
    PROJECT_ROOT if PROJECT_ROOT is not None else Path("__PROJECT_ROOT_NOT_FOUND__")
)


def _normalized_name(value: str) -> str:
    """Normalize a path component without broad recursive searching."""
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _direct_child(parent: Path, *accepted_names: str) -> Path:
    """Resolve one direct child using only the explicitly accepted names.

    Matching ignores capitalization, spaces, hyphens and underscores. This is
    useful for the project's actual folder spelling while keeping the search
    restricted to the expected branch.
    """
    expected = {_normalized_name(name) for name in accepted_names}
    if parent.is_dir():
        matches = [
            child
            for child in parent.iterdir()
            if _normalized_name(child.name) in expected
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous path below {parent}: {[str(path) for path in matches]}"
            )
    return parent / accepted_names[0]


def _project_file(
    model_folder: str,
    report_chain: tuple[tuple[str, ...], ...],
    filenames: tuple[str, ...],
) -> Path:
    """Build one model-output path from the known project hierarchy."""
    current = _direct_child(_DEFAULT_ROOT, "003_Models")
    current = _direct_child(current, model_folder)
    current = _direct_child(current, "jordan_weiss")
    for accepted_names in report_chain:
        current = _direct_child(current, *accepted_names)
    return _direct_child(current, *filenames)


# Actual paths in PROGETTO_FINALE. The alternate 'lables' spelling is accepted
# only for the final filename because some local outputs may contain that typo.
DEFAULT_MIXED_PATH = _project_file(
    "0031_Mixed",
    (("report_jordan_weiss",), ("tables",)),
    ("10_mixed_baseline_labels.csv", "10_mixed_baseline_lables.csv"),
)
DEFAULT_BINARY_PATH = _project_file(
    "0032_Binary",
    (("reports",), ("njw_binary_only",), ("tables",)),
    ("10_binary_labels.csv", "10_binary_lables.csv"),
)
DEFAULT_NUMERIC_PATH = _project_file(
    "0033_Numeric",
    (("Report_njw_numeric_only",), ("tables",)),
    ("10_numeric_labels.csv", "10_numeric_lables.csv"),
)
DEFAULT_ACTIVITY_PATH = _direct_child(
    _direct_child(_DEFAULT_ROOT, "000_Dataset"),
    "train_activity_target.csv",
)
DEFAULT_REPORT_DIR = SCRIPT_DIR / "Report_activity_posthoc"


# =============================================================================
# 2. DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class SourceSpec:
    model: str
    path: Path
    partitions: tuple[str, ...]


@dataclass(frozen=True)
class PartitionResult:
    model: str
    partition: str
    labels: np.ndarray
    contingency: pd.DataFrame
    cluster_profiles: pd.DataFrame
    summary: dict[str, object]


# =============================================================================
# 3. INPUT VALIDATION
# =============================================================================


def read_csv_clean(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    frame = pd.read_csv(path)
    unnamed = [column for column in frame.columns if str(column).startswith("Unnamed:")]
    if unnamed:
        frame = frame.drop(columns=unnamed)
    return frame


def validate_row_index(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    if ROW_COL not in frame.columns:
        raise ValueError(f"Missing '{ROW_COL}' in {path}")
    if frame[ROW_COL].isna().any() or frame[ROW_COL].duplicated().any():
        raise ValueError(f"Invalid or duplicated row_index values in {path}")

    result = frame.sort_values(ROW_COL).reset_index(drop=True)
    expected = np.arange(len(result))
    if not np.array_equal(result[ROW_COL].to_numpy(), expected):
        raise ValueError(
            f"{path} must contain row_index values 0,...,{len(result) - 1}."
        )
    return result


def read_activity_file(path: Path, expected_rows: int) -> pd.Series | None:
    if not path.exists():
        return None
    frame = read_csv_clean(path)
    if TARGET_COL in frame.columns:
        activity = frame[TARGET_COL]
    elif frame.shape[1] == 1:
        activity = frame.iloc[:, 0]
    else:
        raise ValueError(f"Cannot identify Activity in {path}")

    activity = activity.reset_index(drop=True)
    if len(activity) != expected_rows:
        raise ValueError(
            f"{path} has {len(activity)} rows instead of {expected_rows}."
        )
    return validate_activity(activity, str(path))


def validate_activity(activity: pd.Series, source: str) -> pd.Series:
    if activity.isna().any():
        raise ValueError(f"Missing Activity values in {source}")
    values = set(activity.unique().tolist())
    if not values.issubset({0, 1, 0.0, 1.0, False, True}):
        raise ValueError(f"Activity must be binary 0/1 in {source}; found {sorted(values)}")
    return activity.astype(int).reset_index(drop=True)


def select_canonical_activity(
    frames: dict[str, pd.DataFrame],
    activity_path: Path,
    expected_rows: int,
) -> tuple[pd.Series, str]:
    external = read_activity_file(activity_path, expected_rows)
    candidates: list[tuple[str, pd.Series]] = []
    if external is not None:
        candidates.append((str(activity_path), external))

    for model, frame in frames.items():
        if TARGET_COL in frame.columns:
            candidates.append(
                (f"{model} labels", validate_activity(frame[TARGET_COL], model))
            )

    if not candidates:
        raise ValueError(
            "Activity was not found in train_activity_target.csv or in any label file."
        )

    canonical_source, canonical = candidates[0]
    for source, candidate in candidates[1:]:
        if not canonical.equals(candidate):
            raise ValueError(
                f"Activity mismatch between '{canonical_source}' and '{source}'."
            )
    return canonical, canonical_source


def load_inputs(
    mixed_path: Path,
    binary_path: Path,
    numeric_path: Path,
    activity_path: Path,
) -> tuple[dict[str, pd.DataFrame], pd.Series, str, list[dict[str, object]]]:
    paths = {
        "mixed": mixed_path,
        "binary": binary_path,
        "numeric": numeric_path,
    }
    frames = {
        model: validate_row_index(read_csv_clean(path), path)
        for model, path in paths.items()
    }

    row_counts = {model: len(frame) for model, frame in frames.items()}
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"Label files have different row counts: {row_counts}")

    expected_rows = next(iter(row_counts.values()))
    reference_index = frames["mixed"][ROW_COL]
    for model, frame in frames.items():
        if not reference_index.equals(frame[ROW_COL]):
            raise ValueError(f"row_index is not aligned between mixed and {model} labels.")

    activity, activity_source = select_canonical_activity(
        frames,
        activity_path,
        expected_rows,
    )

    validations: list[dict[str, object]] = []
    for model, frame in frames.items():
        validations.append(
            {
                "model": model,
                "path": str(paths[model]),
                "n_rows": len(frame),
                "n_columns": frame.shape[1],
                "row_index_aligned": True,
                "contains_activity": TARGET_COL in frame.columns,
                "activity_matches_canonical": (
                    True
                    if TARGET_COL not in frame.columns
                    else validate_activity(frame[TARGET_COL], model).equals(activity)
                ),
            }
        )

    return frames, activity, activity_source, validations


# =============================================================================
# 4. PARTITION ANALYSIS
# =============================================================================


def entropy_bits(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    return float(-(positive * np.log2(positive)).sum())


def corrected_cramers_v(contingency: np.ndarray) -> tuple[float, float, float]:
    chi2, p_value, _, _ = chi2_contingency(contingency, correction=False)
    n = contingency.sum()
    rows, columns = contingency.shape
    phi2 = chi2 / n
    phi2_corrected = max(
        0.0,
        phi2 - ((columns - 1) * (rows - 1)) / max(n - 1, 1),
    )
    rows_corrected = rows - ((rows - 1) ** 2) / max(n - 1, 1)
    columns_corrected = columns - ((columns - 1) ** 2) / max(n - 1, 1)
    denominator = min(rows_corrected - 1, columns_corrected - 1)
    value = math.sqrt(phi2_corrected / denominator) if denominator > 0 else float("nan")
    return float(value), float(chi2), float(p_value)


def analyze_partition(
    model: str,
    partition: str,
    labels: pd.Series,
    activity: pd.Series,
) -> PartitionResult:
    if labels.isna().any():
        raise ValueError(f"Missing cluster labels in {model}/{partition}")

    labels_array = labels.astype(int).to_numpy()
    activity_array = activity.to_numpy(dtype=int)
    unique_labels = np.sort(np.unique(labels_array))
    if len(unique_labels) < 2:
        raise ValueError(f"{model}/{partition} contains fewer than two clusters.")

    contingency = pd.crosstab(
        pd.Series(labels_array, name="cluster"),
        pd.Series(activity_array, name=TARGET_COL),
    ).reindex(index=unique_labels, columns=[0, 1], fill_value=0)

    cluster_sizes = contingency.sum(axis=1)
    global_counts = contingency.sum(axis=0)
    global_rates = global_counts / global_counts.sum()
    within_cluster = contingency.div(cluster_sizes, axis=0)
    class_across_clusters = contingency.div(global_counts, axis=1)

    profile_rows: list[dict[str, object]] = []
    for cluster in contingency.index:
        counts = contingency.loc[cluster]
        size = int(cluster_sizes.loc[cluster])
        probabilities = within_cluster.loc[cluster].to_numpy(dtype=float)
        majority = int(counts.idxmax())
        profile_rows.append(
            {
                "model": model,
                "partition": partition,
                "cluster": int(cluster),
                "cluster_size": size,
                "cluster_fraction": size / len(labels_array),
                "activity_0_count": int(counts.loc[0]),
                "activity_1_count": int(counts.loc[1]),
                "activity_0_within_cluster": float(within_cluster.loc[cluster, 0]),
                "activity_1_within_cluster": float(within_cluster.loc[cluster, 1]),
                "fraction_of_all_activity_0": float(class_across_clusters.loc[cluster, 0]),
                "fraction_of_all_activity_1": float(class_across_clusters.loc[cluster, 1]),
                "activity_0_enrichment": float(
                    within_cluster.loc[cluster, 0] / global_rates.loc[0]
                ),
                "activity_1_enrichment": float(
                    within_cluster.loc[cluster, 1] / global_rates.loc[1]
                ),
                "activity_1_difference_from_global": float(
                    within_cluster.loc[cluster, 1] - global_rates.loc[1]
                ),
                "majority_activity": majority,
                "cluster_purity": float(probabilities.max()),
                "cluster_entropy_bits": entropy_bits(probabilities),
                "below_minimum_cluster_fraction": bool(
                    size < math.ceil(MIN_CLUSTER_FRACTION * len(labels_array))
                ),
            }
        )

    profiles = pd.DataFrame(profile_rows)
    weighted_purity = float(
        (profiles["cluster_purity"] * profiles["cluster_size"]).sum()
        / len(labels_array)
    )
    weighted_entropy = float(
        (profiles["cluster_entropy_bits"] * profiles["cluster_size"]).sum()
        / len(labels_array)
    )
    cramers_v, chi2, p_value = corrected_cramers_v(contingency.to_numpy())

    summary = {
        "model": model,
        "partition": partition,
        "n_objects": len(labels_array),
        "n_clusters": len(unique_labels),
        "cluster_sizes": json.dumps(cluster_sizes.astype(int).tolist()),
        "minimum_cluster_size": int(cluster_sizes.min()),
        "minimum_cluster_fraction": float(cluster_sizes.min() / len(labels_array)),
        "valid_minimum_cluster_size": bool(
            cluster_sizes.min() >= math.ceil(MIN_CLUSTER_FRACTION * len(labels_array))
        ),
        "global_activity_0_count": int(global_counts.loc[0]),
        "global_activity_1_count": int(global_counts.loc[1]),
        "global_activity_1_rate": float(global_rates.loc[1]),
        "adjusted_rand_index_with_activity": float(
            adjusted_rand_score(activity_array, labels_array)
        ),
        "normalized_mutual_information_with_activity": float(
            normalized_mutual_info_score(activity_array, labels_array)
        ),
        "cramers_v_corrected": cramers_v,
        "weighted_purity": weighted_purity,
        "weighted_entropy_bits": weighted_entropy,
        "chi_square": chi2,
        "chi_square_p_value": p_value,
    }

    return PartitionResult(
        model=model,
        partition=partition,
        labels=labels_array,
        contingency=contingency,
        cluster_profiles=profiles,
        summary=summary,
    )


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    n = len(values)
    adjusted_ranked = ranked * n / np.arange(1, n + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return pd.Series(adjusted, index=p_values.index)


# =============================================================================
# 5. PLOTS
# =============================================================================


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def save_composition_plot(result: PartitionResult, figure_dir: Path) -> None:
    profiles = result.cluster_profiles.sort_values("cluster")
    clusters = profiles["cluster"].astype(str)
    activity_0 = profiles["activity_0_within_cluster"] * 100.0
    activity_1 = profiles["activity_1_within_cluster"] * 100.0

    fig, ax = plt.subplots(figsize=(max(7.0, 1.1 * len(clusters) + 4.0), 5.8))
    ax.bar(clusters, activity_0, label="Activity = 0")
    ax.bar(clusters, activity_1, bottom=activity_0, label="Activity = 1")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Percentage within cluster")
    ax.set_title(f"Activity composition | {result.model} | {result.partition}")
    ax.legend()

    for index, row in profiles.reset_index(drop=True).iterrows():
        ax.text(index, 102, f"n={int(row['cluster_size'])}", ha="center", va="bottom")
        ax.text(index, row["activity_0_within_cluster"] * 50.0, f"{row['activity_0_within_cluster'] * 100:.1f}%", ha="center", va="center")
        ax.text(index, row["activity_0_within_cluster"] * 100.0 + row["activity_1_within_cluster"] * 50.0, f"{row['activity_1_within_cluster'] * 100:.1f}%", ha="center", va="center")

    fig.tight_layout()
    filename = f"composition_{slugify(result.model)}_{slugify(result.partition)}.png"
    fig.savefig(figure_dir / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_enrichment_plot(result: PartitionResult, figure_dir: Path) -> None:
    profiles = result.cluster_profiles.sort_values("cluster")
    fig, ax = plt.subplots(figsize=(max(7.0, 1.1 * len(profiles) + 4.0), 5.2))
    ax.bar(profiles["cluster"].astype(str), profiles["activity_1_enrichment"])
    ax.axhline(1.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Activity = 1 enrichment ratio")
    ax.set_title(f"Activity = 1 enrichment | {result.model} | {result.partition}")
    for index, value in enumerate(profiles["activity_1_enrichment"]):
        ax.text(index, float(value), f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    filename = f"enrichment_{slugify(result.model)}_{slugify(result.partition)}.png"
    fig.savefig(figure_dir / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_summary_metric(summary: pd.DataFrame, metric: str, label: str, output: Path) -> None:
    ordered = summary.sort_values(metric, ascending=True)
    names = ordered["model"] + " | " + ordered["partition"]
    fig_height = max(6.0, 0.45 * len(ordered) + 2.0)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(names, ordered[metric])
    ax.set_xlabel(label)
    ax.set_title(f"Post-hoc association with Activity: {label}")
    for index, value in enumerate(ordered[metric]):
        ax.text(float(value), index, f" {value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 6. OUTPUT SUMMARY
# =============================================================================


def write_results_summary(
    output_path: Path,
    summary: pd.DataFrame,
    activity_source: str,
    n_objects: int,
) -> None:
    global_rate = float(summary["global_activity_1_rate"].iloc[0])
    strongest = summary.sort_values("cramers_v_corrected", ascending=False).iloc[0]
    invalid = summary.loc[~summary["valid_minimum_cluster_size"]]

    lines = [
        "# Activity post-hoc analysis",
        "",
        "## Scope",
        f"- Objects analyzed: {n_objects}",
        f"- Canonical Activity source: `{activity_source}`",
        f"- Global Activity = 1 prevalence: {global_rate:.4f}",
        "- Activity was joined only after the clustering partitions had been finalized.",
        "- No partition or model was selected using Activity.",
        "",
        "## Models included",
        "- Mixed NJW baseline",
        "- Binary-only NJW",
        "- Numeric-only NJW baseline",
        "",
        "## Main outputs",
        "- `tables/01_partition_activity_summary.csv`: one row per partition.",
        "- `tables/02_cluster_activity_profiles.csv`: counts, percentages and enrichment by cluster.",
        "- `tables/03_input_validation.csv`: input and alignment checks.",
        "- `figures/composition_*.png`: within-cluster Activity composition.",
        "- `figures/enrichment_*.png`: Activity = 1 enrichment relative to the full dataset.",
        "",
        "## Strongest observed association",
        (
            f"- By corrected Cramér's V: `{strongest['model']} / "
            f"{strongest['partition']}` with V = "
            f"{strongest['cramers_v_corrected']:.4f}."
        ),
        "",
        "## Methodological notes",
        "- ARI, NMI and corrected Cramér's V quantify association with Activity without requiring cluster-label alignment.",
        "- Purity should not be compared alone across different k because it tends to increase as the number of clusters increases.",
        "- Chi-square p-values are reported together with Benjamini-Hochberg adjusted p-values; effect sizes and cluster compositions remain the primary interpretation.",
    ]

    if invalid.empty:
        lines.extend(["", "## Cluster-size check", "- All analyzed partitions satisfy the 5% minimum-cluster-size reference."])
    else:
        lines.extend(["", "## Cluster-size warnings"])
        for row in invalid.itertuples(index=False):
            lines.append(
                f"- `{row.model} / {row.partition}` has minimum cluster size "
                f"{row.minimum_cluster_size} ({row.minimum_cluster_fraction:.4%})."
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed", type=Path, default=None)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--numeric", type=Path, default=None)
    parser.add_argument("--activity", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_runtime_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve defaults at runtime from the actual project tree.

    This intentionally avoids storing a guessed binary report path in the CLI
    defaults. The binary branch is resolved as:
    0032_Binary/jordan_weiss/reports/njw_binary_only/tables/10_binary_labels.csv
    with case/underscore-insensitive matching of each direct path component.
    """
    defaults = {
        "mixed": _project_file(
            "0031_Mixed",
            (("report_jordan_weiss",), ("tables",)),
            ("10_mixed_baseline_labels.csv", "10_mixed_baseline_lables.csv"),
        ),
        "binary": _project_file(
            "0032_Binary",
            (("reports",), ("njw_binary_only",), ("tables",)),
            ("10_binary_labels.csv", "10_binary_lables.csv"),
        ),
        "numeric": _project_file(
            "0033_Numeric",
            (("Report_njw_numeric_only",), ("tables",)),
            ("10_numeric_labels.csv", "10_numeric_lables.csv"),
        ),
        "activity": _direct_child(
            _direct_child(_DEFAULT_ROOT, "000_Dataset"),
            "train_activity_target.csv",
        ),
    }
    resolved = {
        "mixed": (args.mixed or defaults["mixed"]).resolve(),
        "binary": (args.binary or defaults["binary"]).resolve(),
        "numeric": (args.numeric or defaults["numeric"]).resolve(),
        "activity": (args.activity or defaults["activity"]).resolve(),
        "output": (args.output or DEFAULT_REPORT_DIR).resolve(),
    }
    return resolved


def main() -> None:
    args = parse_args()
    selected_paths = resolve_runtime_paths(args)
    report_dir = selected_paths["output"]
    table_dir = report_dir / "tables"
    figure_dir = report_dir / "figures"
    for directory in (report_dir, table_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    configure_plots()

    print(f"Running: {Path(__file__).resolve()}")
    print(f"Script version: {SCRIPT_VERSION}")
    print("Resolved input paths:")
    for name, path in selected_paths.items():
        print(f"  {name}: {path}")

    frames, activity, activity_source, validations = load_inputs(
        selected_paths["mixed"],
        selected_paths["binary"],
        selected_paths["numeric"],
        selected_paths["activity"],
    )

    specs = (
        SourceSpec(
            model="mixed",
            path=selected_paths["mixed"],
            partitions=(
                "mixed_classical_selected",
                "mixed_balanced_selected",
                "mixed_balanced_k2",
                "mixed_balanced_k3",
                "mixed_balanced_k4",
            ),
        ),
        SourceSpec(
            model="binary",
            path=selected_paths["binary"],
            partitions=(
                "binary_selected",
                "binary_k2",
                "binary_k3",
                "binary_k4",
            ),
        ),
        SourceSpec(
            model="numeric",
            path=selected_paths["numeric"],
            partitions=(
                "numeric_selected",
                "numeric_k2",
                "numeric_k3",
                "numeric_k4",
            ),
        ),
    )

    results: list[PartitionResult] = []
    for spec in specs:
        frame = frames[spec.model]
        missing = [column for column in spec.partitions if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns in {spec.path}: {missing}")
        for partition in spec.partitions:
            result = analyze_partition(
                spec.model,
                partition,
                frame[partition],
                activity,
            )
            results.append(result)
            save_composition_plot(result, figure_dir)
            save_enrichment_plot(result, figure_dir)

    summary = pd.DataFrame([result.summary for result in results])
    summary["chi_square_p_value_bh"] = benjamini_hochberg(
        summary["chi_square_p_value"]
    )
    profiles = pd.concat(
        [result.cluster_profiles for result in results],
        ignore_index=True,
    )

    summary.to_csv(table_dir / "01_partition_activity_summary.csv", index=False)
    profiles.to_csv(table_dir / "02_cluster_activity_profiles.csv", index=False)
    pd.DataFrame(validations).to_csv(table_dir / "03_input_validation.csv", index=False)
    pd.DataFrame(
        [
            {
                "n_objects": len(activity),
                "activity_0_count": int((activity == 0).sum()),
                "activity_1_count": int((activity == 1).sum()),
                "activity_0_rate": float((activity == 0).mean()),
                "activity_1_rate": float((activity == 1).mean()),
                "activity_source": activity_source,
            }
        ]
    ).to_csv(table_dir / "04_global_activity_distribution.csv", index=False)

    save_summary_metric(
        summary,
        "cramers_v_corrected",
        "Corrected Cramér's V",
        figure_dir / "summary_cramers_v.png",
    )
    save_summary_metric(
        summary,
        "adjusted_rand_index_with_activity",
        "Adjusted Rand Index",
        figure_dir / "summary_ari.png",
    )
    save_summary_metric(
        summary,
        "normalized_mutual_information_with_activity",
        "Normalized Mutual Information",
        figure_dir / "summary_nmi.png",
    )

    write_results_summary(
        report_dir / "results_summary.md",
        summary,
        activity_source,
        len(activity),
    )

    print("Activity post-hoc analysis completed.")
    print(f"Report directory: {report_dir}")
    print(f"Partitions analyzed: {len(results)}")


if __name__ == "__main__":
    main()
