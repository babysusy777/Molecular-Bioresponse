# Numeric-only numerical-Gower + NJW: automatic interpretation

## 1. Data representation
The analysis used **3751 observations** and **695 retained numerical descriptors**. Each numerical descriptor was scaled by its observed range, and pairwise dissimilarity was the mean normalized absolute difference, which is exactly the numerical component used by Gower.
The binary block containing 828 descriptors was completely excluded. Activity was excluded from distance construction, parameter selection, and clustering.

## 2. Representative numeric-only solution
The selected configuration used k=4 and sigma=2.0 times the median positive numerical Gower dissimilarity. Cluster sizes were [918, 1055, 972, 806], numerical-Gower silhouette was 0.0279, spectral silhouette was 0.3919, and eigengap was 0.005372.

## 3. Resolution analysis
At the selected numeric bandwidth, k=2 produced [3748, 3], k=3 produced [1171, 1218, 1362], and k=4 produced [1055, 922, 973, 801].

## 4. Cross-representation comparison
Mixed labels source: `/Users/susannabaldo/Desktop/Machine_Learning_Project/Molecular-Bioresponse/reports/njw_mixed_baseline_k234_v2/tables/10_mixed_baseline_labels.csv`.
Binary labels source: `/Users/susannabaldo/Desktop/Machine_Learning_Project/Molecular-Bioresponse/models/binario/reports/njw_binary_only/tables/10_binary_labels.csv`.

- numeric_selected_vs_mixed_classical_selected: ARI=0.0802 (numeric k=4, reference k=2).
- numeric_selected_vs_mixed_balanced_selected: ARI=0.1141 (numeric k=4, reference k=4).
- numeric_k2_vs_mixed_balanced_k2: ARI=-0.0006 (numeric k=2, reference k=2).
- numeric_k3_vs_mixed_balanced_k3: ARI=0.2444 (numeric k=3, reference k=3).
- numeric_k4_vs_mixed_balanced_k4: ARI=0.1134 (numeric k=4, reference k=4).
- numeric_selected_vs_binary_selected: ARI=0.1137 (numeric k=4, reference k=4).
- numeric_k2_vs_binary_k2: ARI=-0.0006 (numeric k=2, reference k=2).
- numeric_k3_vs_binary_k3: ARI=0.2444 (numeric k=3, reference k=3).
- numeric_k4_vs_binary_k4: ARI=0.1130 (numeric k=4, reference k=4).

ARI and contingency tables are the valid cross-representation comparisons. Numerical-Gower, asymmetric-Jaccard, and mixed-Gower silhouette values belong to different geometries and must not be ranked directly against one another.