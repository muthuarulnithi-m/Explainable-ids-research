# Phase 6.1 — CIC Scenario Group-Clustered Bootstrap Evaluation

## 1. Objective

This experiment evaluates the uncertainty and comparative performance of three previously trained multiclass XGBoost models on the unchanged CIC-IDS2017 binary test dataset.

The models are:

- XGBoost baseline
- XGBoost balanced
- XGBoost square-root balanced

A paired group-clustered bootstrap was used to estimate 95% confidence intervals and compare the models without retraining, refitting preprocessing, or tuning classification thresholds.

## 2. Experimental setup

The experiment uses the existing processed CIC-IDS2017 test dataset.

| Property | Value |
|---|---:|
| Test rows | 567,870 |
| Unique test groups | 503,927 |
| Model features | 78 |
| Total classes | 15 |
| Eligible classes | 12 |
| Excluded rare classes | 3 |
| Largest group | 6,814 flows |

The three classes excluded from headline macro-F1 are:

- Heartbleed
- Infiltration
- Web Attack - SQL Injection

These classes remain represented in the complete confusion matrices. Their exclusion from the headline metric is due to insufficient true support for reliable evaluation.

The 12 eligible classes were fixed according to the pre-registered evaluation protocol.

## 3. Bootstrap methodology

The bootstrap procedure used the following configuration:

| Parameter | Setting |
|---|---|
| Resampling unit | `group_id` |
| Bootstrap replicates | 5,000 |
| Random seed | 42 |
| Confidence level | 95% |
| Interval method | Percentile |
| Model comparison | Paired group resampling |

Groups were sampled with replacement. All flows belonging to a sampled group received the corresponding bootstrap multiplicity.

The same sampled groups were used for all three models, enabling paired comparisons.

The headline metric was macro-F1 over eligible classes with positive true support in each replicate. The complete 15-class confusion matrices were retained.

## 4. Point-estimate results

| Model | Test rows | Unique groups | Headline macro-F1 |
|---|---:|---:|---:|
| XGBoost baseline | 567,870 | 503,927 | 0.907296 |
| XGBoost balanced | 567,870 | 503,927 | 0.906487 |
| XGBoost sqrt-balanced | 567,870 | 503,927 | **0.912668** |

The square-root class-weighted model achieved the highest point estimate.

Its improvement over baseline was approximately 0.005372, while its improvement over the fully balanced model was approximately 0.006181.

These differences are descriptive and require the paired uncertainty analysis below.

## 5. Model confidence intervals

| Model | Point estimate | 95% CI lower | 95% CI upper |
|---|---:|---:|---:|
| XGBoost baseline | 0.907296 | 0.898629 | 0.915918 |
| XGBoost balanced | 0.906487 | 0.897659 | 0.915152 |
| XGBoost sqrt-balanced | 0.912668 | 0.903866 | 0.921029 |

The confidence intervals overlap. However, the paired model-difference intervals provide the more appropriate basis for evaluating differences between models.

## 6. Paired model comparisons

Differences are calculated as the first model minus the second model.

| Comparison | Point difference | 95% CI lower | 95% CI upper | Excludes zero |
|---|---:|---:|---:|---|
| Baseline − Balanced | +0.000809 | −0.007471 | +0.009282 | No |
| Baseline − Sqrt-balanced | −0.005372 | −0.013083 | +0.002111 | No |
| Balanced − Sqrt-balanced | −0.006181 | −0.013305 | +0.001112 | No |

None of the paired confidence intervals excludes zero.

Therefore, the experiment does not provide sufficient statistical evidence to establish that any one of the three weighting strategies is superior.

Although square-root weighting achieves the highest point estimate, its apparent advantage should be interpreted cautiously.

## 7. Largest-group sensitivity analysis

The largest test group contains 6,814 flows.

The experiment includes a sensitivity analysis that excludes this group and evaluates whether the primary results are materially affected.

The corresponding results are stored in:

`largest_group_sensitivity.csv`

This analysis is supplementary to the primary group-bootstrap estimates and does not replace them.

## 8. Reproducibility and integrity

The experiment records SHA-256 hashes for:

- Test dataset
- Imputer
- Label encoder
- Baseline XGBoost model
- Balanced XGBoost model
- Square-root-balanced XGBoost model

The hashes are stored in:

`group_bootstrap_config.json`

The point estimates were checked against the existing Phase 6.1 scenario evaluation results. All three values reproduced successfully.

The output directory contains:

- `point_estimates.csv`
- `group_bootstrap_confidence_intervals.csv`
- `paired_model_differences.csv`
- `group_bootstrap_distributions.csv`
- `largest_group_sensitivity.csv`
- Three model confusion matrices
- `group_bootstrap_config.json`

The automated test suite also passed all nine tests.

## 9. Limitations

1. The bootstrap is group-level, not capture-day-level.
2. Some group IDs span multiple capture files.
3. Three rare classes are excluded from headline macro-F1 because of insufficient support.
4. Predictions for excluded classes remain represented in the complete confusion matrices.
5. Bootstrap uncertainty is conditional on the fixed test dataset and existing trained models.
6. The bootstrap does not establish generalization to independent datasets, networks, or future attack scenarios.
7. The evaluation uses previously trained models and does not include retraining uncertainty.

## 10. Conclusion

The group-clustered bootstrap evaluation identified the square-root class-weighted XGBoost model as the strongest candidate based on its macro-F1 point estimate of 0.912668.

However, none of the paired 95% confidence intervals for model differences excluded zero. Consequently, the observed differences do not establish statistically significant superiority among the three weighting strategies.

The results support retaining square-root weighting as the leading candidate for subsequent evaluation, while avoiding claims of statistically demonstrated improvement.

This experiment provides a more appropriate assessment of uncertainty than treating individual flows as independent bootstrap observations. Nevertheless, its conclusions remain conditional on the selected CIC-IDS2017 test groups and trained models.

---

**Phase status:** Completed. The 5,000-replicate paired group bootstrap, model confidence intervals, paired comparisons, largest-group sensitivity analysis, reproducibility checks, and automated tests are complete.