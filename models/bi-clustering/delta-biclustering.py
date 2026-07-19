from pathlib import Path

import numpy as np
import pandas as pd


# Molecular-Bioresponse/models/delta_biclustering.py
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

INPUT_PATH = PROJECT_DIR / "Dataset" / "train_filtered_no_activity.csv"
TARGET_PATH = PROJECT_DIR / "Dataset" / "train_activity_target.csv"
REPORT_DIR = PROJECT_DIR / "reports" / "delta_biclustering"

DELTA = 0.01          # valore iniziale esplorativo: da tarare sui risultati
ALPHA = 0.9           # multiple-node deletion
MIN_ROWS = 20
MIN_COLS = 5


def msr_scores(matrix):
    row_mean = matrix.mean(axis=1, keepdims=True)
    col_mean = matrix.mean(axis=0, keepdims=True)
    residue = matrix - row_mean - col_mean + matrix.mean()
    residue **= 2

    return (
        float(residue.mean()),
        residue.mean(axis=1),
        residue.mean(axis=0),
    )


def limit_removals(scores, positions, current_size, minimum_size):
    max_removals = current_size - minimum_size
    if max_removals <= 0:
        return np.array([], dtype=int)
    if len(positions) <= max_removals:
        return positions
    return positions[np.argsort(scores[positions])[-max_removals:]]


def deletion_phase(data):
    rows = np.arange(data.shape[0])
    cols = np.arange(data.shape[1])
    iteration = 0

    while True:
        h_score, row_scores, col_scores = msr_scores(data[np.ix_(rows, cols)])

        if iteration % 10 == 0:
            print(
                f"Deletion {iteration}: H={h_score:.6f}, "
                f"rows={len(rows)}, features={len(cols)}"
            )

        if h_score <= DELTA:
            return rows, cols, h_score

        bad_rows = np.where(row_scores > ALPHA * h_score)[0]
        bad_cols = np.where(col_scores > ALPHA * h_score)[0]

        bad_rows = limit_removals(
            row_scores, bad_rows, len(rows), MIN_ROWS
        )
        bad_cols = limit_removals(
            col_scores, bad_cols, len(cols), MIN_COLS
        )

        if len(bad_rows) or len(bad_cols):
            rows = np.delete(rows, bad_rows)
            cols = np.delete(cols, bad_cols)
        else:
            can_remove_row = len(rows) > MIN_ROWS
            can_remove_col = len(cols) > MIN_COLS

            if not can_remove_row and not can_remove_col:
                raise RuntimeError(
                    f"No bicluster found with H <= {DELTA}. "
                    "Increase DELTA or reduce the minimum dimensions."
                )

            max_row = row_scores.max() if can_remove_row else -np.inf
            max_col = col_scores.max() if can_remove_col else -np.inf

            if max_row >= max_col:
                rows = np.delete(rows, np.argmax(row_scores))
            else:
                cols = np.delete(cols, np.argmax(col_scores))

        iteration += 1


def candidate_row_h(data, rows, cols, candidates):
    current = data[np.ix_(rows, cols)]
    candidate_values = data[np.ix_(candidates, cols)]
    n_rows, n_cols = current.shape

    new_sum_squares = (
        np.sum(current ** 2)
        + np.sum(candidate_values ** 2, axis=1)
    )
    new_row_term = (
        np.sum(current.mean(axis=1) ** 2)
        + candidate_values.mean(axis=1) ** 2
    )
    new_col_sums = current.sum(axis=0) + candidate_values
    new_total = current.sum() + candidate_values.sum(axis=1)

    rss = (
        new_sum_squares
        - n_cols * new_row_term
        - np.sum(new_col_sums ** 2, axis=1) / (n_rows + 1)
        + new_total ** 2 / ((n_rows + 1) * n_cols)
    )

    return np.maximum(rss, 0) / ((n_rows + 1) * n_cols)


def candidate_col_h(data, rows, cols, candidates):
    current = data[np.ix_(rows, cols)]
    candidate_values = data[np.ix_(rows, candidates)]
    n_rows, n_cols = current.shape

    new_sum_squares = (
        np.sum(current ** 2)
        + np.sum(candidate_values ** 2, axis=0)
    )
    new_row_sums = current.sum(axis=1)[:, None] + candidate_values
    candidate_col_sums = candidate_values.sum(axis=0)
    new_total = current.sum() + candidate_col_sums

    rss = (
        new_sum_squares
        - np.sum(new_row_sums ** 2, axis=0) / (n_cols + 1)
        - (
            np.sum(current.sum(axis=0) ** 2)
            + candidate_col_sums ** 2
        ) / n_rows
        + new_total ** 2 / (n_rows * (n_cols + 1))
    )

    return np.maximum(rss, 0) / (n_rows * (n_cols + 1))


def addition_phase(data, rows, cols):
    selected_rows = np.zeros(data.shape[0], dtype=bool)
    selected_cols = np.zeros(data.shape[1], dtype=bool)
    selected_rows[rows] = True
    selected_cols[cols] = True

    while True:
        rows = np.flatnonzero(selected_rows)
        cols = np.flatnonzero(selected_cols)
        outside_rows = np.flatnonzero(~selected_rows)
        outside_cols = np.flatnonzero(~selected_cols)

        best_type = None
        best_index = None
        best_h = np.inf

        if len(outside_rows):
            row_h = candidate_row_h(data, rows, cols, outside_rows)
            position = np.argmin(row_h)
            best_type = "row"
            best_index = outside_rows[position]
            best_h = float(row_h[position])

        if len(outside_cols):
            col_h = candidate_col_h(data, rows, cols, outside_cols)
            position = np.argmin(col_h)

            if col_h[position] < best_h:
                best_type = "column"
                best_index = outside_cols[position]
                best_h = float(col_h[position])

        if best_type is None or best_h > DELTA:
            final_h, _, _ = msr_scores(data[np.ix_(rows, cols)])
            return rows, cols, final_h

        if best_type == "row":
            selected_rows[best_index] = True
        else:
            selected_cols[best_index] = True


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    feature_df = pd.read_csv(INPUT_PATH)
    data = feature_df.to_numpy(dtype=np.float64)

    if not np.isfinite(data).all():
        raise ValueError("The input matrix contains missing or infinite values.")

    target = None
    if TARGET_PATH.exists():
        target_df = pd.read_csv(TARGET_PATH)
        target = target_df["Activity"] if "Activity" in target_df else target_df.iloc[:, 0]

        if len(target) != len(feature_df):
            raise ValueError("Feature matrix and target have different row counts.")

    full_h, _, _ = msr_scores(data)
    print(f"Dataset: {data.shape[0]} rows x {data.shape[1]} features")
    print(f"Full-matrix H-score: {full_h:.6f}")
    print(f"Delta: {DELTA}")

    rows, cols, _ = deletion_phase(data)
    rows, cols, final_h = addition_phase(data, rows, cols)

    block = data[np.ix_(rows, cols)]
    binary_features = np.all(
        (data[:, cols] == 0) | (data[:, cols] == 1),
        axis=0,
    )

    summary = {
        "delta": DELTA,
        "h_score": final_h,
        "n_rows": len(rows),
        "n_features": len(cols),
        "volume": len(rows) * len(cols),
        "zero_fraction": float(np.mean(block == 0)),
        "binary_feature_fraction": float(binary_features.mean()),
    }

    if target is not None:
        summary["activity_1_fraction"] = float(target.iloc[rows].mean())

    pd.DataFrame([summary]).to_csv(
        REPORT_DIR / "bicluster_summary.csv", index=False
    )

    row_report = pd.DataFrame({"row_index": rows})
    if target is not None:
        row_report["Activity"] = target.iloc[rows].to_numpy()
    row_report.to_csv(REPORT_DIR / "bicluster_rows.csv", index=False)

    pd.DataFrame({
        "feature": feature_df.columns[cols],
        "is_binary": binary_features,
        "zero_fraction": np.mean(data[:, cols] == 0, axis=0),
    }).to_csv(REPORT_DIR / "bicluster_features.csv", index=False)

    print("\nBicluster found")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Reports saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()