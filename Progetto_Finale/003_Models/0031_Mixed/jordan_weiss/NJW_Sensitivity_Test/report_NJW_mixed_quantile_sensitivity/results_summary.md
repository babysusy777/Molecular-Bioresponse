# Mixed-data quantile sensitivity analysis

## Input and representation

- Observations: 3751
- Numerical descriptors: 695
- Binary descriptors: 828
- Numerical transformation: feature-wise empirical quantile map to Uniform[0,1]
- Binary treatment: asymmetric Jaccard; joint zeros ignored
- Activity: excluded from all unsupervised operations and appended only to final labels

## Selected solutions

- Quantile-classical: k=2, sigma multiplier=2.0, sizes=[2320, 1431], Gower silhouette=0.2160, spectral silhouette=0.6549, eigengap=0.019032
- Quantile-balanced: k=4, sigma multiplier=0.5, sizes=[1056, 1267, 982, 446], Gower silhouette=0.1486, spectral silhouette=0.8026, eigengap=0.053911
- Weight sensitivity: minimum ARI=0.9731, mean ARI=0.9819

## Balanced resolution checks

- k=2: sizes=[2323, 1428], silhouette=0.1423
- k=3: sizes=[1056, 1428, 1267], silhouette=0.1451
- k=4: sizes=[1056, 1267, 982, 446], silhouette=0.1486

## Reference files

- No optional reference label files were available.

## Cross-representation ARI

- mixed_quantile_classical_selected vs mixed_quantile_balanced_selected: ARI=0.3216

## Interpretation rule

Silhouette values are compared only within the same transformed geometry. Cross-representation conclusions use matched-resolution ARI and contingency tables. This run remains a sensitivity analysis because quantile transformation changes the numerical distance geometry.