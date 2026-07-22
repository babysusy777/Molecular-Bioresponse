# Mixed-data Gower + NJW baseline: automatic interpretation

## 1. Dataset used
The analysis used **3751 observations** and **1523 retained descriptors**: **695 numerical (45.6%) and **828 binary (54.4%)**.
Input source: `/Users/federico.colangelo/Desktop/dmml_project/Molecular-Bioresponse/Dataset/train_filtered_no_activity.csv`. The Activity variable was excluded from all unsupervised steps.

## 2. Classical Gower baseline
The representative classical solution used k=2 and sigma=2.0 times the median positive Gower dissimilarity. It produced cluster sizes [1058, 2693], Gower silhouette 0.3238, spectral silhouette 0.7495, and eigengap 0.020651.

## 3. Why block weighting was explored
Classical Gower is retained as the standard reference, but its feature-wise aggregation does not explicitly control the total contribution of the numerical and binary descriptor blocks. The weighted analysis therefore asks whether the discovered structure is robust to moderate, controlled changes in block contribution; it is not an attempt to declare a different distance universally superior.

## 4. Weighted sensitivity and balanced representative
At the balanced representative resolution (k=4, sigma multiplier=0.5), the minimum pairwise ARI across alpha=0.4, 0.5, and 0.6 was 0.9929. The alpha=0.5 partition had cluster sizes [1056, 951, 1267, 477], Gower silhouette 0.1733, spectral silhouette 0.8376, and eigengap 0.074338.
The ARI between the selected classical and balanced partitions was 0.4070. This comparison is based on label agreement; silhouettes from the two Gower definitions are not ranked against one another because the underlying geometries differ.

## 5. Resolution analysis
Under alpha=0.5 and the selected weighted bandwidth, k=2 gave sizes [1428, 2323], whereas k=4 gave [1056, 1267, 477, 951]. The contingency table in `08_balanced_k2_vs_k4_counts.csv` shows whether the four clusters are nested refinements of the two macroclusters.

## 6. Baseline for the next analyses
The exported mixed-data labels provide the reference against which the same NJW procedure can be run on the numerical-only and binary-only blocks. ARI and contingency comparisons will then quantify which block reproduces the mixed organization and which clusters depend on the interaction between descriptor types.

## Balanced k=2 versus k=4 counts

```
              k4_cluster_0  k4_cluster_1  k4_cluster_2  k4_cluster_3
k2_cluster_0           951           477             0             0
k2_cluster_1             0             0          1056          1267
```