\# UNSW-NB15 Dataset Audit



\## Files inspected



\- `UNSW\_NB15\_training-set.csv`

\- `UNSW\_NB15\_testing-set.csv`



\## Structural validation



| Property | Training | Testing |

|---|---:|---:|

| Rows | 175,341 | 82,332 |

| Columns | 45 | 45 |

| Missing values detected | 0 | 0 |

| Exact duplicate rows | 0 | 0 |

| Infinite numeric values | 0 | 0 |

| Unique IDs within file | 175,341 | 82,332 |



The training and testing files have matching column sets.



\## Binary label distribution



| Class | Training | Testing |

|---|---:|---:|

| Normal (`label = 0`) | 56,000 | 37,000 |

| Attack (`label = 1`) | 119,341 | 45,332 |



\## Attack-category validation



The `Normal` category maps to label 0. All observed attack categories map to label 1. No category/label conflicts were observed.



\## Initial modeling decisions



\- Exclude `id` from predictive features.

\- Use `label` as the binary classification target.

\- Exclude `attack\_cat` from binary classification inputs.

\- Preserve `attack\_cat` for attack-category analysis and a possible separate multiclass experiment.

\- Do not modify the original CSV files.



\## Limitations and pending checks



\- Cross-file ID overlap has not yet been checked.

\- Zero missing values and duplicate rows do not establish that the dataset is free of all quality issues or data leakage.

\- The supplied predefined split will be retained for the initial benchmark.

\- The provenance and exact version of the downloaded files should be recorded.

\- Feature distributions, constant columns, extreme values, and potential dataset artifacts still require investigation.



\## Status



Initial structural and label audit completed. No preprocessing or model training has been performed.
