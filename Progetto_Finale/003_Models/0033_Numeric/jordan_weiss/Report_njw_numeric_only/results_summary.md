# Numeric-only NJW results summary

## Data representation
- Objects: 3751
- Numerical descriptors used: 695
- Input file: `/Users/federico.colangelo/Desktop/dmml_project/Molecular-Bioresponse/Progetto_Finale/000_Dataset/train_numeric_only.csv`
- Numerical preprocessing: feature-wise min-max scaling to [0,1]
- Distance: numerical Gower = mean normalized absolute difference
- Activity: excluded from all unsupervised steps and used only post-hoc in the exported labels

## Selected numeric-only solution
- k = 4
- sigma multiplier = 2.0
- cluster sizes = [809, 970, 1053, 919]
- numerical-Gower silhouette = 0.0275
- spectral silhouette = 0.3919
- eigengap = 0.005372

## Explicit resolutions at the selected bandwidth
- k=2 sizes = [3748, 3]
- k=3 sizes = [1170, 1218, 1363]
- k=4 sizes = [809, 970, 1053, 919]

## External references used
- Mixed labels: not available
- Binary labels: not available

## Cross-representation ARI
- No external comparison was available.

## Interpretation rule
- Use ARI and contingency tables for cross-representation comparisons.
- Do not rank numerical-Gower, asymmetric-Jaccard and mixed-Gower silhouettes directly across different geometries.