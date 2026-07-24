# Activity post-hoc analysis

## Scope
- Objects analyzed: 3751
- Canonical Activity source: `/Users/federico.colangelo/Desktop/dmml_project/Molecular-Bioresponse/Progetto_Finale/000_Dataset/train_activity_target.csv`
- Global Activity = 1 prevalence: 0.5423
- Activity was joined only after the clustering partitions had been finalized.
- No partition or model was selected using Activity.

## Models included
- Mixed NJW baseline
- Binary-only NJW
- Numeric-only NJW baseline

## Main outputs
- `tables/01_partition_activity_summary.csv`: one row per partition.
- `tables/02_cluster_activity_profiles.csv`: counts, percentages and enrichment by cluster.
- `tables/03_input_validation.csv`: input and alignment checks.
- `figures/composition_*.png`: within-cluster Activity composition.
- `figures/enrichment_*.png`: Activity = 1 enrichment relative to the full dataset.

## Strongest observed association
- By corrected Cramér's V: `numeric / numeric_selected` with V = 0.2326.

## Methodological notes
- ARI, NMI and corrected Cramér's V quantify association with Activity without requiring cluster-label alignment.
- Purity should not be compared alone across different k because it tends to increase as the number of clusters increases.
- Chi-square p-values are reported together with Benjamini-Hochberg adjusted p-values; effect sizes and cluster compositions remain the primary interpretation.

## Cluster-size warnings
- `numeric / numeric_k2` has minimum cluster size 3 (0.0800%).