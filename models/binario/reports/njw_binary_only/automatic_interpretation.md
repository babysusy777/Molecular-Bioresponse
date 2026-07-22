# Binary-only asymmetric-Jaccard + NJW: automatic interpretation

## 1. Data representation
The analysis used **3751 observations** and **828 retained binary descriptors**. Joint zeros were ignored, so the pairwise dissimilarity is the asymmetric Jaccard dissimilarity.
The mean number of active descriptors per observation was 109.99, with median 93.00. Activity was excluded from distance construction, parameter selection, and clustering.

## 2. Representative binary-only solution
The selected binary-only configuration used k=4 and sigma=0.5 times the median positive Jaccard dissimilarity. Cluster sizes were [492, 1267, 1056, 936], Jaccard silhouette was 0.1848, spectral silhouette was 0.8485, and eigengap was 0.084003.

## 3. Resolution analysis
At the selected binary bandwidth, k=2 produced [1428, 2323], k=3 produced [1056, 1428, 1267], and k=4 produced [1056, 936, 1267, 492]. The nesting tables show whether the higher-resolution solutions subdivide the same binary macrostructure.

## 4. Comparison with the mixed-data baseline
Mixed labels source: `/Users/susannabaldo/Desktop/Machine_Learning_Project/Molecular-Bioresponse/reports/njw_mixed_baseline_k234_v2/tables/10_mixed_baseline_labels.csv`.

- binary_selected_vs_mixed_classical_selected: ARI=0.4054 (binary k=4, mixed k=2).
- binary_selected_vs_mixed_balanced_selected: ARI=0.9924 (binary k=4, mixed k=4).
- binary_k2_vs_mixed_balanced_k2: ARI=1.0000 (binary k=2, mixed k=2).
- binary_k3_vs_mixed_balanced_k3: ARI=1.0000 (binary k=3, mixed k=3).
- binary_k4_vs_mixed_balanced_k4: ARI=0.9924 (binary k=4, mixed k=4).

ARI measures agreement between memberships and is the appropriate cross-representation comparison. Jaccard and mixed-Gower silhouette magnitudes must not be ranked directly because they refer to different dissimilarity geometries.