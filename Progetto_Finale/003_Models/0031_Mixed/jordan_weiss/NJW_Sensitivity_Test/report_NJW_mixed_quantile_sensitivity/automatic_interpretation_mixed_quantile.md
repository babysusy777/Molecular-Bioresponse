# Mixed quantile sensitivity analysis

## Representation

Only the numerical block was transformed feature-wise to an empirical Uniform[0,1] distribution. The asymmetric binary Jaccard block was left unchanged. This is a nonlinear sensitivity analysis, not standard Gower.

Objects: 3751; numerical descriptors: 695; binary descriptors: 828.

## Selected solutions

Quantile-classical: k=2, sigma multiplier=2.0, sizes=[2323, 1428], silhouette=0.2164.

Quantile-balanced: k=4, sigma multiplier=0.5, sizes=[1056, 982, 1267, 446], silhouette=0.1486.

Matched balanced resolutions:
- k=2: sizes=[1428, 2323], silhouette=0.1423, spectral silhouette=0.8535, eigengap=0.113852.
- k=3: sizes=[1267, 1428, 1056], silhouette=0.1451, spectral silhouette=0.6571, eigengap=0.036899.
- k=4: sizes=[1056, 1267, 446, 982], silhouette=0.1486, spectral silhouette=0.8026, eigengap=0.053911.

## Cross-representation agreement

- mixed_quantile_balanced_k2_vs_mixed_balanced_k2: ARI=1.0000.
- mixed_quantile_balanced_k3_vs_binary_k3: ARI=1.0000.
- mixed_quantile_balanced_k3_vs_mixed_balanced_k3: ARI=1.0000.
- mixed_quantile_balanced_k2_vs_binary_k2: ARI=1.0000.
- mixed_quantile_balanced_selected_vs_mixed_balanced_selected: ARI=0.9836.
- mixed_quantile_balanced_k4_vs_mixed_balanced_k4: ARI=0.9836.
- mixed_quantile_balanced_k4_vs_binary_k4: ARI=0.9763.
- mixed_quantile_balanced_selected_vs_binary_selected: ARI=0.9763.
- mixed_quantile_classical_selected_vs_mixed_classical_selected: ARI=0.6387.
- mixed_quantile_classical_selected_vs_numeric_quantile_selected: ARI=0.4646.
- mixed_quantile_balanced_selected_vs_mixed_classical_selected: ARI=0.4106.
- mixed_quantile_classical_selected_vs_mixed_quantile_balanced_selected: ARI=0.3220.
- mixed_quantile_classical_selected_vs_mixed_balanced_selected: ARI=0.3178.
- mixed_quantile_classical_selected_vs_binary_selected: ARI=0.3160.
- mixed_quantile_balanced_k3_vs_numeric_k3: ARI=0.2444.
- mixed_quantile_balanced_k3_vs_numeric_quantile_k3: ARI=0.2050.
- mixed_quantile_balanced_k2_vs_numeric_quantile_k2: ARI=0.1896.
- mixed_quantile_classical_selected_vs_numeric_selected: ARI=0.1531.
- mixed_quantile_balanced_selected_vs_numeric_quantile_selected: ARI=0.1317.
- mixed_quantile_balanced_selected_vs_numeric_selected: ARI=0.1158.
- mixed_quantile_balanced_k4_vs_numeric_k4: ARI=0.1151.
- mixed_quantile_balanced_k4_vs_numeric_quantile_k4: ARI=0.1142.
- mixed_quantile_balanced_k2_vs_numeric_k2: ARI=-0.0006.

## Interpretation rule

Silhouettes are comparable only within the same transformed geometry. Use ARI and contingency tables to compare the quantile-mixed analysis with the original mixed, binary-only, and numeric-only analyses.

The quantile route is supported only if it improves robustness and interpretability without relying on fragmented graphs or unstable partitions. It must remain explicitly labelled as a sensitivity analysis because the transform changes quantitative distances.