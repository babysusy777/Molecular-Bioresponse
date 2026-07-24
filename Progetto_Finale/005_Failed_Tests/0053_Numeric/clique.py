from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path
from typing import TypeAlias
from heapq import nlargest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, triu


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(
    "/Users/susannabaldo/Desktop/Machine_Learning_Project/"
    "Molecular-Bioresponse"
)

DATA_PATH = BASE_DIR / "Dataset" / "train_numeric_only.csv"
TARGET_PATH = BASE_DIR / "Dataset" / "train_activity_target.csv"
OUTPUT_DIR = BASE_DIR / "models" / "numerico" / "clique" / "reports" 

MAX_DOMINANT_BIN_FRACTION = 0.75
XI = 10
TAU = 0.10
MAX_DIM = 3

# Protezione contro esplosione combinatoria per dimensioni >= 3
MAX_CANDIDATES = 2_000_000


# Unità CLIQUE:
# ((dimensione_1, intervallo_1), ..., (dimensione_k, intervallo_k))
Unit: TypeAlias = tuple[tuple[int, int], ...]


def filter_features_for_clique(
    X: pd.DataFrame,
    xi: int = 10,
    max_dominant_bin_fraction: float = 0.90,
    min_unique_values: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    retained_features = []
    records = []

    for feature in X.columns:
        values = X[feature].to_numpy(dtype=float)

        counts, edges = np.histogram(
            values,
            bins=xi,
            range=(values.min(), values.max()),
        )

        dominant_bin = int(np.argmax(counts))
        dominant_bin_fraction = float(
            counts[dominant_bin] / len(values)
        )

        unique_values = int(
            np.unique(values).size
        )

        retained = (
            dominant_bin_fraction <= max_dominant_bin_fraction
            and unique_values >= min_unique_values
        )

        records.append(
            {
                "feature": feature,
                "unique_values": unique_values,
                "dominant_bin": dominant_bin,
                "dominant_bin_lower": edges[dominant_bin],
                "dominant_bin_upper": edges[dominant_bin + 1],
                "dominant_bin_fraction": dominant_bin_fraction,
                "retained": retained,
            }
        )

        if retained:
            retained_features.append(feature)

    report = pd.DataFrame(records)

    return X[retained_features].copy(), report

# ============================================================
# IMPLEMENTAZIONE CLIQUE
# ============================================================

class CLIQUE:
    def __init__(
        self,
        xi: int = 5,
        tau: float = 0.05,
        max_dim: int = 2,
        max_candidates: int = 2_000_000,
    ) -> None:

        if xi < 2:
            raise ValueError("xi deve essere almeno 2.")

        if not 0 < tau <= 1:
            raise ValueError("tau deve essere compreso tra 0 e 1.")

        if max_dim < 1:
            raise ValueError("max_dim deve essere almeno 1.")

        self.xi = xi
        self.tau = tau
        self.max_dim = max_dim
        self.max_candidates = max_candidates

        self.columns_: list[str] = []
        self.bin_ids_: np.ndarray | None = None
        self.bin_edges_: list[np.ndarray] = []

        self.n_samples_: int = 0
        self.n_features_: int = 0
        self.min_count_: int = 0

        self.dense_units_: dict[int, list[Unit]] = {}
        self.unit_bitsets_: dict[tuple[int, int], int] = {}
        self.clusters_: list[dict] = []

        self.dropped_constant_columns_: list[str] = []

    def fit(self, X: pd.DataFrame) -> "CLIQUE":
        X = self._validate_data(X)

        self.columns_ = list(X.columns)
        self.n_samples_, self.n_features_ = X.shape
        self.min_count_ = int(np.ceil(self.tau * self.n_samples_))

        print(f"Samples: {self.n_samples_}")
        print(f"Numeric features: {self.n_features_}")
        print(f"xi: {self.xi}")
        print(f"tau: {self.tau}")
        print(f"Minimum support: {self.min_count_}")
        print(f"Maximum subspace dimensionality: {self.max_dim}")

        self.bin_ids_, self.bin_edges_ = self._discretize(X)

        # Livello 1
        dense_1d = self._find_dense_1d_units()
        self.dense_units_[1] = dense_1d

        print(f"\nDense 1D units: {len(dense_1d)}")

        self._build_1d_bitsets(dense_1d)

        # Livello 2, calcolato efficientemente tramite matrice sparsa
        if self.max_dim >= 2 and dense_1d:
            dense_2d = self._find_dense_2d_units(dense_1d)
            self.dense_units_[2] = dense_2d

            print(f"Dense 2D units: {len(dense_2d)}")

        # Livelli >= 3 con candidate generation Apriori
        for dimensionality in range(3, self.max_dim + 1):
            previous = self.dense_units_.get(dimensionality - 1, [])

            if not previous:
                print(
                    f"Nessuna unità densa di dimensione "
                    f"{dimensionality - 1}: arresto."
                )
                break

            candidates = self._generate_candidates(
                previous_units=previous,
                dimensionality=dimensionality,
            )

            print(
                f"Candidate {dimensionality}D units: "
                f"{len(candidates)}"
            )

            dense_units = []

            for candidate in candidates:
                if self._unit_support(candidate) >= self.min_count_:
                    dense_units.append(candidate)

            dense_units = sorted(set(dense_units))
            self.dense_units_[dimensionality] = dense_units

            print(
                f"Dense {dimensionality}D units: "
                f"{len(dense_units)}"
            )

            if not dense_units:
                break

        self.clusters_ = self._construct_clusters()

        print(f"\nTotal CLIQUE clusters: {len(self.clusters_)}")

        return self

    def _validate_data(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X deve essere un pandas DataFrame.")

        X = X.copy()

        non_numeric = [
            column
            for column in X.columns
            if not pd.api.types.is_numeric_dtype(X[column])
        ]

        if non_numeric:
            raise ValueError(
                f"Colonne non numeriche presenti: {non_numeric[:10]}"
            )

        if X.isna().any().any():
            raise ValueError("Il dataset contiene valori mancanti.")

        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("Il dataset contiene valori infiniti.")

        constant_mask = X.nunique(dropna=False) <= 1

        self.dropped_constant_columns_ = list(
            X.columns[constant_mask]
        )

        if self.dropped_constant_columns_:
            print(
                f"Constant features removed: "
                f"{len(self.dropped_constant_columns_)}"
            )

            X = X.loc[:, ~constant_mask]

        if X.shape[1] == 0:
            raise ValueError("Non rimangono feature utilizzabili.")

        return X

    def _discretize(
        self,
        X: pd.DataFrame,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Divide separatamente ciascuna dimensione in xi intervalli
        di uguale ampiezza.
        """
        values = X.to_numpy(dtype=float)

        n_samples, n_features = values.shape

        bin_ids = np.empty(
            shape=(n_samples, n_features),
            dtype=np.int16,
        )

        bin_edges = []

        for dimension in range(n_features):
            column = values[:, dimension]

            minimum = column.min()
            maximum = column.max()

            edges = np.linspace(
                minimum,
                maximum,
                self.xi + 1,
            )

            # searchsorted restituisce valori tra 0 e xi
            bins = np.searchsorted(
                edges,
                column,
                side="right",
            ) - 1

            # Il valore massimo finirebbe nel bin xi:
            # viene riportato nel bin xi - 1
            bins = np.clip(
                bins,
                0,
                self.xi - 1,
            )

            bin_ids[:, dimension] = bins
            bin_edges.append(edges)

        return bin_ids, bin_edges

    def _find_dense_1d_units(self) -> list[Unit]:
        assert self.bin_ids_ is not None

        dense_units: list[Unit] = []

        for dimension in range(self.n_features_):
            counts = np.bincount(
                self.bin_ids_[:, dimension],
                minlength=self.xi,
            )

            dense_bins = np.flatnonzero(
                counts >= self.min_count_
            )

            for bin_index in dense_bins:
                unit: Unit = (
                    (dimension, int(bin_index)),
                )

                dense_units.append(unit)

        return sorted(dense_units)

    def _build_1d_bitsets(
        self,
        dense_1d_units: list[Unit],
    ) -> None:
        """
        Memorizza le righe appartenenti a ogni unità 1D come bitset.

        Questo permette di calcolare il supporto di un'unità
        multidimensionale mediante intersezioni bitwise.
        """
        assert self.bin_ids_ is not None

        self.unit_bitsets_.clear()

        for unit in dense_1d_units:
            dimension, bin_index = unit[0]

            mask = (
                self.bin_ids_[:, dimension] == bin_index
            )

            packed = np.packbits(
                mask.astype(np.uint8),
                bitorder="little",
            )

            bitset = int.from_bytes(
                packed.tobytes(),
                byteorder="little",
            )

            self.unit_bitsets_[(dimension, bin_index)] = bitset

    def _find_dense_2d_units(
        self,
        dense_1d_units: list[Unit],
    ) -> list[Unit]:
        """
        Costruisce una matrice binaria:

            righe = molecole
            colonne = unità dense 1D

        Z.T @ Z restituisce il numero di molecole condivise
        da ogni coppia di unità 1D.
        """
        assert self.bin_ids_ is not None

        one_dimensional_items = [
            unit[0]
            for unit in dense_1d_units
        ]

        row_indices = []
        column_indices = []

        for unit_index, (dimension, bin_index) in enumerate(
            one_dimensional_items
        ):
            rows = np.flatnonzero(
                self.bin_ids_[:, dimension] == bin_index
            )

            row_indices.append(rows)

            column_indices.append(
                np.full(
                    rows.size,
                    unit_index,
                    dtype=np.int32,
                )
            )

        if not row_indices:
            return []

        rows = np.concatenate(row_indices)
        columns = np.concatenate(column_indices)

        data = np.ones(
            rows.size,
            dtype=np.int32,
        )

        incidence_matrix = csr_matrix(
            (data, (rows, columns)),
            shape=(
                self.n_samples_,
                len(one_dimensional_items),
            ),
            dtype=np.int32,
        )

        # Solo triangolo superiore: evita coppie duplicate
        cooccurrence = triu(
            incidence_matrix.T @ incidence_matrix,
            k=1,
        ).tocoo()

        dense_2d_units: list[Unit] = []

        for left, right, support in zip(
            cooccurrence.row,
            cooccurrence.col,
            cooccurrence.data,
        ):
            if support < self.min_count_:
                continue

            item_left = one_dimensional_items[left]
            item_right = one_dimensional_items[right]

            dimension_left = item_left[0]
            dimension_right = item_right[0]

            # Una unità non può contenere due intervalli
            # della stessa dimensione.
            if dimension_left == dimension_right:
                continue

            unit: Unit = tuple(
                sorted((item_left, item_right))
            )

            dense_2d_units.append(unit)

        return sorted(set(dense_2d_units))

    def _generate_candidates(
        self,
        previous_units: list[Unit],
        dimensionality: int,
    ) -> list[Unit]:
        """
        Candidate generation in stile Apriori.

        Un'unità k-dimensionale è candidata solo se tutte le sue
        proiezioni (k-1)-dimensionali sono dense.
        """
        previous_set = set(previous_units)

        prefix_length = dimensionality - 2

        grouped_units: dict[Unit, list[Unit]] = defaultdict(list)

        for unit in previous_units:
            prefix = unit[:prefix_length]
            grouped_units[prefix].append(unit)

        candidates: set[Unit] = set()

        for group in grouped_units.values():
            group = sorted(group)

            for unit_a, unit_b in combinations(group, 2):
                candidate_items = tuple(
                    sorted(set(unit_a).union(unit_b))
                )

                if len(candidate_items) != dimensionality:
                    continue

                dimensions = [
                    dimension
                    for dimension, _ in candidate_items
                ]

                # Non possono esserci due bin della stessa feature.
                if len(set(dimensions)) != dimensionality:
                    continue

                # Downward-closure:
                # tutte le proiezioni devono essere dense.
                all_projections_dense = True

                for removed_position in range(dimensionality):
                    projection = (
                        candidate_items[:removed_position]
                        + candidate_items[removed_position + 1:]
                    )

                    if projection not in previous_set:
                        all_projections_dense = False
                        break

                if all_projections_dense:
                    candidates.add(candidate_items)

                if len(candidates) > self.max_candidates:
                    raise RuntimeError(
                        "Numero di candidati eccessivo. "
                        "Ridurre MAX_DIM, diminuire XI oppure "
                        "aumentare TAU."
                    )

        return sorted(candidates)

    def _unit_bitset(self, unit: Unit) -> int:
        result: int | None = None

        for item in unit:
            item_bitset = self.unit_bitsets_.get(item)

            if item_bitset is None:
                return 0

            if result is None:
                result = item_bitset
            else:
                result &= item_bitset

            if result == 0:
                return 0

        return result if result is not None else 0

    def _unit_support(self, unit: Unit) -> int:
        return self._unit_bitset(unit).bit_count()

    def _construct_clusters(self) -> list[dict]:
        """
        Raggruppa, per ogni sottospazio, le unità dense connesse.

        Due unità sono adiacenti se differiscono di un solo
        intervallo lungo una sola dimensione.
        """
        clusters = []
        cluster_id = 0

        for dimensionality, dense_units in self.dense_units_.items():
            units_by_subspace: dict[
                tuple[int, ...],
                list[Unit],
            ] = defaultdict(list)

            for unit in dense_units:
                subspace = tuple(
                    dimension
                    for dimension, _ in unit
                )

                units_by_subspace[subspace].append(unit)

            for subspace, subspace_units in units_by_subspace.items():
                coordinate_to_unit = {
                    tuple(
                        bin_index
                        for _, bin_index in unit
                    ): unit
                    for unit in subspace_units
                }

                unvisited = set(coordinate_to_unit)

                while unvisited:
                    start = unvisited.pop()

                    queue = deque([start])
                    component_coordinates = [start]

                    while queue:
                        coordinate = queue.popleft()

                        for axis in range(dimensionality):
                            for direction in (-1, 1):
                                neighbour = list(coordinate)
                                neighbour[axis] += direction
                                neighbour = tuple(neighbour)

                                if neighbour in unvisited:
                                    unvisited.remove(neighbour)
                                    queue.append(neighbour)
                                    component_coordinates.append(
                                        neighbour
                                    )

                    component_units = [
                        coordinate_to_unit[coordinate]
                        for coordinate in component_coordinates
                    ]

                    cluster_bitset = 0

                    for unit in component_units:
                        cluster_bitset |= self._unit_bitset(unit)

                    row_indices = self._bitset_to_indices(
                        cluster_bitset
                    )

                    clusters.append(
                        {
                            "cluster_id": cluster_id,
                            "dimensionality": dimensionality,
                            "subspace": subspace,
                            "units": component_units,
                            "row_indices": row_indices,
                            "n_points": len(row_indices),
                        }
                    )

                    cluster_id += 1

        return clusters

    def _bitset_to_indices(
        self,
        bitset: int,
    ) -> np.ndarray:
        byte_length = (self.n_samples_ + 7) // 8

        packed = bitset.to_bytes(
            byte_length,
            byteorder="little",
        )

        bits = np.unpackbits(
            np.frombuffer(packed, dtype=np.uint8),
            bitorder="little",
        )

        return np.flatnonzero(
            bits[:self.n_samples_]
        )

    def unit_description(self, unit: Unit) -> str:
        descriptions = []

        for dimension, bin_index in unit:
            column = self.columns_[dimension]
            edges = self.bin_edges_[dimension]

            lower = edges[bin_index]
            upper = edges[bin_index + 1]

            if bin_index == self.xi - 1:
                condition = (
                    f"{lower:.6g} <= {column} <= {upper:.6g}"
                )
            else:
                condition = (
                    f"{lower:.6g} <= {column} < {upper:.6g}"
                )

            descriptions.append(condition)

        return " AND ".join(descriptions)


# ============================================================
# ESPORTAZIONE RISULTATI
# ============================================================

def export_results(
    model: CLIQUE,
    output_dir: Path,
    target: pd.Series | None = None,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Unità dense
    # --------------------------------------------------------

    dense_unit_records = []

    for dimensionality, units in model.dense_units_.items():
        for unit in units:
            subspace_columns = [
                model.columns_[dimension]
                for dimension, _ in unit
            ]

            bins = [
                bin_index
                for _, bin_index in unit
            ]

            dense_unit_records.append(
                {
                    "dimensionality": dimensionality,
                    "subspace": ",".join(subspace_columns),
                    "bins": ",".join(map(str, bins)),
                    "support": model._unit_support(unit),
                    "support_fraction": (
                        model._unit_support(unit)
                        / model.n_samples_
                    ),
                    "region": model.unit_description(unit),
                }
            )

    dense_units_df = pd.DataFrame(
        dense_unit_records
    )

    dense_units_df.to_csv(
        output_dir / "dense_units.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Riepilogo cluster
    # --------------------------------------------------------

    cluster_records = []
    

    for cluster in model.clusters_:
        subspace_names = [
            model.columns_[dimension]
            for dimension in cluster["subspace"]
        ]

        record = {
            "cluster_id": cluster["cluster_id"],
            "dimensionality": cluster["dimensionality"],
            "subspace": ",".join(subspace_names),
            "n_dense_units": len(cluster["units"]),
            "n_points": cluster["n_points"],
            "point_fraction": (
                cluster["n_points"]
                / model.n_samples_
            ),
        }

        if target is not None:
            cluster_target = target.iloc[
                cluster["row_indices"]
            ]

            record["activity_1_count"] = int(
                cluster_target.sum()
            )

            record["activity_1_fraction"] = float(
                cluster_target.mean()
            )

        cluster_records.append(record)


    clusters_df = pd.DataFrame(cluster_records)

    if not clusters_df.empty:
        clusters_df = clusters_df.sort_values(
            by=[
                "dimensionality",
                "n_points",
            ],
            ascending=[
                True,
                False,
            ],
        )

    clusters_df.to_csv(
        output_dir / "clusters_summary.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Feature maggiormente presenti nei sottospazi
    # --------------------------------------------------------

    feature_records = []

    for cluster in model.clusters_:
        for dimension in cluster["subspace"]:
            feature_records.append(
                {
                    "feature": model.columns_[dimension],
                    "cluster_id": cluster["cluster_id"],
                    "dimensionality": cluster["dimensionality"],
                }
            )

    feature_df = pd.DataFrame(feature_records)

    if not feature_df.empty:
        feature_summary = (
            feature_df
            .groupby("feature")
            .agg(
                n_clusters=("cluster_id", "nunique"),
                max_dimensionality=(
                    "dimensionality",
                    "max",
                ),
            )
            .sort_values(
                "n_clusters",
                ascending=False,
            )
            .reset_index()
        )

        feature_summary.to_csv(
            output_dir / "feature_participation.csv",
            index=False,
        )

    print(f"\nResults saved in: {output_dir}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    # ========================================================
    # PARAMETRI DEL FILTRO E DELL'ESPORTAZIONE
    # ========================================================

    MAX_DOMINANT_BIN_FRACTION = 0.90

    # Numero massimo di risultati da esportare
    MAX_EXPORTED_DENSE_UNITS = 10_000
    MAX_EXPORTED_CLUSTERS = 1_000

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. CARICAMENTO DEL DATASET NUMERICO
    # ========================================================

    X_numeric = pd.read_csv(DATA_PATH)

    # Protezione nel caso Activity sia rimasta nel file
    if "Activity" in X_numeric.columns:
        X_numeric = X_numeric.drop(columns=["Activity"])

    print(f"Loaded numeric dataset: {X_numeric.shape}")

    if X_numeric.empty:
        raise ValueError("Il dataset numerico è vuoto.")

    if X_numeric.isna().any().any():
        raise ValueError(
            "Il dataset numerico contiene valori mancanti."
        )

    if not np.isfinite(
        X_numeric.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Il dataset contiene valori infiniti."
        )

    # ========================================================
    # 2. FILTRAGGIO DELLE FEATURE QUASI DEGENERI
    # ========================================================

    retained_features = []
    filtering_records = []

    for feature in X_numeric.columns:
        values = X_numeric[feature].to_numpy(
            dtype=float
        )

        minimum = float(values.min())
        maximum = float(values.max())
        variance = float(values.var())
        unique_values = int(np.unique(values).size)
        zero_fraction = float(
            np.mean(values == 0)
        )

        if minimum == maximum:
            dominant_bin = 0
            dominant_bin_lower = minimum
            dominant_bin_upper = maximum
            dominant_bin_count = len(values)
            dominant_bin_fraction = 1.0
            top_3_bins_fraction = 1.0

        else:
            counts, edges = np.histogram(
                values,
                bins=XI,
                range=(minimum, maximum),
            )

            dominant_bin = int(np.argmax(counts))

            dominant_bin_count = int(
                counts[dominant_bin]
            )

            dominant_bin_fraction = float(
                dominant_bin_count / len(values)
            )

            dominant_bin_lower = float(
                edges[dominant_bin]
            )

            dominant_bin_upper = float(
                edges[dominant_bin + 1]
            )

            # Frazione di osservazioni contenute nei 3 bin più popolati
            sorted_counts = np.sort(counts)[::-1]

            top_3_bins_fraction = float(
                sorted_counts[:3].sum() / len(values)
            )

        retained = (
            unique_values > 1
            and dominant_bin_fraction <= 0.75
            and top_3_bins_fraction <= 0.90
        )

        filtering_records.append(
            {
                "feature": feature,
                "minimum": minimum,
                "maximum": maximum,
                "variance": variance,
                "unique_values": unique_values,
                "zero_fraction": zero_fraction,
                "dominant_bin": dominant_bin,
                "dominant_bin_lower": (
                    dominant_bin_lower
                ),
                "dominant_bin_upper": (
                    dominant_bin_upper
                ),
                "dominant_bin_count": (
                    dominant_bin_count
                ),
                "dominant_bin_fraction": (
                    dominant_bin_fraction
                ),
                "retained": retained,
                "top_3_bins_fraction": top_3_bins_fraction
            }
        )

        if retained:
            retained_features.append(feature)

    filtering_report = pd.DataFrame(
        filtering_records
    )

    filtering_report = filtering_report.sort_values(
        by="dominant_bin_fraction",
        ascending=False,
    )

    filtering_report.to_csv(
        OUTPUT_DIR / "feature_filtering_report.csv",
        index=False,
    )

    X_clique = X_numeric[
        retained_features
    ].copy()

    print(
        f"Original numeric features: "
        f"{X_numeric.shape[1]}"
    )

    print(
        f"Features retained for CLIQUE: "
        f"{X_clique.shape[1]}"
    )

    print(
        f"Features removed: "
        f"{X_numeric.shape[1] - X_clique.shape[1]}"
    )

    print(
        "Filtering criteria: "
        "dominant bin fraction <= 0.75; "
        "top-3 bins fraction <= 0.90"
    )

    if X_clique.shape[1] < 2:
        raise RuntimeError(
            "Dopo il filtraggio rimangono meno di due "
            "feature. Aumentare "
            "MAX_DOMINANT_BIN_FRACTION."
        )

    X_clique.to_csv(
        OUTPUT_DIR / "numeric_features_used_by_clique.csv",
        index=False,
    )

    # ========================================================
    # 3. CARICAMENTO DEL TARGET
    # ========================================================

    target = None

    if TARGET_PATH.exists():
        target_df = pd.read_csv(TARGET_PATH)

        if "Activity" in target_df.columns:
            target = target_df[
                "Activity"
            ].reset_index(drop=True)
        else:
            target = target_df.iloc[
                :, 0
            ].reset_index(drop=True)

        if len(target) != len(X_clique):
            raise ValueError(
                "Il target e il dataset numerico hanno "
                "un numero differente di righe."
            )

        print("Activity target loaded.")

    else:
        print(
            "Activity target not found: "
            "target analysis skipped."
        )

    # ========================================================
    # 4. ESECUZIONE DI CLIQUE
    # ========================================================

    model = CLIQUE(
        xi=XI,
        tau=TAU,
        max_dim=MAX_DIM,
        max_candidates=MAX_CANDIDATES,
    )

    model.fit(X_clique)

    number_dense_units = sum(
        len(units)
        for units in model.dense_units_.values()
    )

    print(
        f"\nTotal dense units: "
        f"{number_dense_units}"
    )

    print(
        f"Total CLIQUE clusters: "
        f"{len(model.clusters_)}"
    )

    # ========================================================
    # 5. ESPORTAZIONE LIMITATA DELLE UNITÀ DENSE
    # ========================================================

    unit_candidates = []

    for dimensionality, units in (
        model.dense_units_.items()
    ):
        for unit in units:
            support = model._unit_support(unit)

            unit_candidates.append(
                (
                    support,
                    dimensionality,
                    unit,
                )
            )

    selected_dense_units = nlargest(
        MAX_EXPORTED_DENSE_UNITS,
        unit_candidates,
        key=lambda element: element[0],
    )

    dense_unit_records = []

    for (
        support,
        dimensionality,
        unit,
    ) in selected_dense_units:

        feature_names = [
            model.columns_[dimension]
            for dimension, _ in unit
        ]

        bin_indices = [
            bin_index
            for _, bin_index in unit
        ]

        dense_unit_records.append(
            {
                "dimensionality": dimensionality,
                "subspace": ",".join(
                    feature_names
                ),
                "bins": ",".join(
                    map(str, bin_indices)
                ),
                "support": support,
                "support_fraction": (
                    support / model.n_samples_
                ),
                "region": (
                    model.unit_description(unit)
                ),
            }
        )

    dense_units_df = pd.DataFrame(
        dense_unit_records
    )

    dense_units_df.to_csv(
        OUTPUT_DIR / "dense_units_top.csv",
        index=False,
    )

    # ========================================================
    # 6. ESPORTAZIONE LIMITATA DEI CLUSTER
    # ========================================================

    selected_clusters = nlargest(
        MAX_EXPORTED_CLUSTERS,
        model.clusters_,
        key=lambda cluster: cluster["n_points"],
    )

    cluster_records = []

    for cluster in selected_clusters:
        feature_names = [
            model.columns_[dimension]
            for dimension in cluster["subspace"]
        ]

        record = {
            "cluster_id": cluster["cluster_id"],
            "dimensionality": (
                cluster["dimensionality"]
            ),
            "subspace": ",".join(
                feature_names
            ),
            "n_dense_units": len(
                cluster["units"]
            ),
            "n_points": cluster["n_points"],
            "point_fraction": (
                cluster["n_points"]
                / model.n_samples_
            ),
        }

        if target is not None:
            row_indices = cluster[
                "row_indices"
            ]

            cluster_target = target.iloc[
                row_indices
            ]

            record["activity_0_count"] = int(
                (cluster_target == 0).sum()
            )

            record["activity_1_count"] = int(
                (cluster_target == 1).sum()
            )

            record[
                "activity_1_fraction"
            ] = float(
                cluster_target.mean()
            )

        cluster_records.append(record)

    clusters_df = pd.DataFrame(
        cluster_records
    )

    if not clusters_df.empty:
        clusters_df = clusters_df.sort_values(
            by=[
                "dimensionality",
                "n_points",
            ],
            ascending=[
                True,
                False,
            ],
        )

    clusters_df.to_csv(
        OUTPUT_DIR / "clusters_summary_top.csv",
        index=False,
    )

    # ========================================================
    # 7. PARTECIPAZIONE DELLE FEATURE
    # ========================================================

    feature_participation_records = []

    for cluster in selected_clusters:
        for dimension in cluster["subspace"]:
            feature_participation_records.append(
                {
                    "feature": (
                        model.columns_[dimension]
                    ),
                    "cluster_id": (
                        cluster["cluster_id"]
                    ),
                    "dimensionality": (
                        cluster[
                            "dimensionality"
                        ]
                    ),
                    "n_points": (
                        cluster["n_points"]
                    ),
                }
            )

    feature_participation_df = pd.DataFrame(
        feature_participation_records
    )

    if not feature_participation_df.empty:
        feature_summary = (
            feature_participation_df
            .groupby("feature")
            .agg(
                n_clusters=(
                    "cluster_id",
                    "nunique",
                ),
                max_dimensionality=(
                    "dimensionality",
                    "max",
                ),
                maximum_cluster_size=(
                    "n_points",
                    "max",
                ),
            )
            .sort_values(
                by="n_clusters",
                ascending=False,
            )
            .reset_index()
        )

        feature_summary.to_csv(
            OUTPUT_DIR
            / "feature_participation.csv",
            index=False,
        )

    # ========================================================
    # 8. STAMPA DEI CLUSTER 2D PIÙ GRANDI
    # ========================================================

    clusters_2d = [
        cluster
        for cluster in model.clusters_
        if cluster["dimensionality"] == 2
    ]

    largest_2d_clusters = nlargest(
        10,
        clusters_2d,
        key=lambda cluster: cluster["n_points"],
    )

    print("\nLargest 2D clusters:")

    if not largest_2d_clusters:
        print("No 2D clusters found.")

    for cluster in largest_2d_clusters:
        feature_names = [
            model.columns_[dimension]
            for dimension in cluster["subspace"]
        ]

        point_fraction = (
            cluster["n_points"]
            / model.n_samples_
        )

        print(
            f"Cluster {cluster['cluster_id']} | "
            f"features={feature_names} | "
            f"units={len(cluster['units'])} | "
            f"points={cluster['n_points']} | "
            f"fraction={point_fraction:.4f}"
        )

    # ========================================================
    # 9. RIEPILOGO FINALE
    # ========================================================

    print(f"\nResults saved in: {OUTPUT_DIR}")

    print(
        "Exported dense units: "
        f"{min(number_dense_units, MAX_EXPORTED_DENSE_UNITS)}"
    )

    print(
        "Exported clusters: "
        f"{min(len(model.clusters_), MAX_EXPORTED_CLUSTERS)}"
    )

    print(
        "Cluster memberships were not exported "
        "to avoid excessive memory consumption."
    )



if __name__ == "__main__":
    main()