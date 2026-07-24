# Numeric-only quantile NJW results summary

## Data representation
- Objects: 3751
- Numerical descriptors used: 695
- Input file: `/Users/federico.colangelo/Desktop/dmml_project/Molecular-Bioresponse/Progetto_Finale/000_Dataset/train_numeric_only.csv`
- Numerical preprocessing: feature-wise empirical quantile transformation to Uniform[0,1]
- Distance: mean L1 dissimilarity in quantile-transformed space
- Analysis role: sensitivity analysis; it does not replace the min-max numeric baseline
- Activity: excluded from all unsupervised steps and used only post-hoc in the exported labels

## Selected numeric-quantile solution
- k = 2
- sigma multiplier = 2.0
- cluster sizes = [1474, 2277]
- quantile-L1 silhouette = 0.1935
- spectral silhouette = 0.6061
- eigengap = 0.015953

## Explicit resolutions at the selected bandwidth
- k=2 sizes = [1474, 2277]
- k=3 sizes = [1338, 867, 1546]
- k=4 sizes = [654, 962, 1339, 796]

## External references used
- Mixed labels: not available
- Binary labels: not available
- Min-max numeric labels: `/Users/federico.colangelo/Desktop/dmml_project/Molecular-Bioresponse/Progetto_Finale/003_Models/0033_Numeric/jordan_weiss/Report_njw_numeric_only/tables/10_numeric_labels.csv`

## Cross-representation ARI
- numeric_quantile_k4_vs_numeric_k4: ARI = 0.3977
- numeric_quantile_k3_vs_numeric_k3: ARI = 0.3217
- numeric_quantile_selected_vs_numeric_selected: ARI = 0.2307
- numeric_quantile_k2_vs_numeric_k2: ARI = 0.0009

## Interpretation rule
- Use ARI and contingency tables for cross-representation comparisons.
- Do not rank silhouettes directly across quantile-L1, min-max numerical Gower, asymmetric Jaccard and mixed Gower geometries.