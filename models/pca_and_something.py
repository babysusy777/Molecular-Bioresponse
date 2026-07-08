from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

INPUT_PATH = PROJECT_DIR / "Dataset" / "train_filtered_no_activity.csv"
OUTPUT_PATH = PROJECT_DIR / "Dataset" / "train_pca_90_no_activity.csv"
REPORT_PATH = PROJECT_DIR / "reports" / "pca_explained_variance.csv"

VARIANCE_TO_KEEP = 0.90


def main():
    X = pd.read_csv(INPUT_PATH)

    pca = PCA(n_components=VARIANCE_TO_KEEP)
    X_pca = pca.fit_transform(X)

    pc_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca_df = pd.DataFrame(X_pca, columns=pc_columns)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    X_pca_df.to_csv(OUTPUT_PATH, index=False)

    report = pd.DataFrame({"component": pc_columns, "explained_variance_ratio": pca.explained_variance_ratio_, "cumulative_explained_variance": pca.explained_variance_ratio_.cumsum()})

    report.to_csv(REPORT_PATH, index=False)

    print("PCA completed")
    print(f"Original features: {X.shape[1]}")
    print(f"PCA components: {X_pca.shape[1]}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"Saved PCA dataset to: {OUTPUT_PATH}")
    print(f"Saved PCA report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()