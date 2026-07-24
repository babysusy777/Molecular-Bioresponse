# Numeric-only quantile-L1 + NJW: automatic interpretation

## 1. Data representation
The analysis used **3751 observations** and **695 retained numerical descriptors**. Each numerical descriptor was scaled by its observed range, and pairwise dissimilarity was the mean normalized absolute difference, which is exactly the numerical component used by Gower.
The binary block containing 828 descriptors was completely excluded. Activity was excluded from distance construction, parameter selection, and clustering.

## 2. Representative numeric-only solution
The selected configuration used k=2 and sigma=2.0 times the median positive numerical Gower dissimilarity. Cluster sizes were [2278, 1473], quantile-L1 silhouette was 0.1937, spectral silhouette was 0.6061, and eigengap was 0.015953.

## 3. Resolution analysis
At the selected numeric bandwidth, k=2 produced [2276, 1475], k=3 produced [1546, 1337, 868], and k=4 produced [796, 1339, 962, 654].

## 4. Cross-representation comparison
Mixed labels source: `/Users/federico.colangelo/Desktop/dmml_project/Molecular-Bioresponse/reports/njw_mixed_baseline_k234_v2/tables/10_mixed_baseline_labels.csv`.
Binary labels source: `/Users/federico.colangelo/Desktop/dmml_project/Molecular-Bioresponse/models/binario/reports/njw_binary_only/tables/10_binary_labels.csv`.

- numeric_selected_vs_mixed_classical_selected: ARI=0.2545 (numeric k=2, reference k=2).
- numeric_selected_vs_mixed_balanced_selected: ARI=0.1293 (numeric k=2, reference k=4).
- numeric_k2_vs_mixed_balanced_k2: ARI=0.1896 (numeric k=2, reference k=2).
- numeric_k3_vs_mixed_balanced_k3: ARI=0.2050 (numeric k=3, reference k=3).
- numeric_k4_vs_mixed_balanced_k4: ARI=0.1128 (numeric k=4, reference k=4).
- numeric_selected_vs_binary_selected: ARI=0.1283 (numeric k=2, reference k=4).
- numeric_k2_vs_binary_k2: ARI=0.1896 (numeric k=2, reference k=2).
- numeric_k3_vs_binary_k3: ARI=0.2050 (numeric k=3, reference k=3).
- numeric_k4_vs_binary_k4: ARI=0.1125 (numeric k=4, reference k=4).
- numeric_quantile_selected_vs_numeric_baseline_selected: ARI=0.2312 (numeric k=2, reference k=4).
- numeric_quantile_k2_vs_numeric_baseline_k2: ARI=0.0009 (numeric k=2, reference k=2).
- numeric_quantile_k3_vs_numeric_baseline_k3: ARI=0.3223 (numeric k=3, reference k=3).
- numeric_quantile_k4_vs_numeric_baseline_k4: ARI=0.3975 (numeric k=4, reference k=4).

ARI and contingency tables are the valid cross-representation comparisons. Quantile-L1, asymmetric-Jaccard, and mixed-Gower silhouette values belong to different geometries and must not be ranked directly against one another.