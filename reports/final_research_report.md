# An Explainable and Generalization-Aware Machine Learning Framework for Network Intrusion Detection Under Class Imbalance

## Abstract

Network intrusion detection systems must identify malicious network activity while handling class imbalance, varying attack families, and differences between training and evaluation data. This research investigates an explainable and generalization-aware machine learning framework using CIC-IDS2017, with an additional dataset audit of UNSW-NB15.

The study evaluates binary and multiclass intrusion detection, compares class-imbalance strategies, examines model explanations using SHAP, and investigates generalization through group-clustered bootstrap evaluation and capture-day-ordered binary evaluation. Logistic Regression, Random Forest, and XGBoost are used in different experimental stages.

The results show that model performance depends on the evaluation design and metric. Although the square-root class-weighted XGBoost model achieved the highest point estimate in the CIC multiclass scenario evaluation, paired group-bootstrap confidence intervals did not establish a statistically significant difference between the three models. In the capture-day-ordered binary experiment, Logistic Regression achieved better threshold-based attack recall and F1-score than Random Forest, while Random Forest achieved higher average precision and ROC-AUC.

The findings emphasize that accuracy alone is insufficient for evaluating intrusion detection systems, particularly when attack classes are imbalanced or absent from training data.

**Keywords:** Network Intrusion Detection, Machine Learning, Class Imbalance, Explainable AI, SHAP, Generalization, CIC-IDS2017, XGBoost, Group Bootstrap.

---

## 1. Introduction

Network intrusion detection is an important component of network security. Machine learning methods can identify suspicious traffic patterns by learning from labelled network-flow data. However, intrusion detection datasets commonly contain class imbalance, where benign traffic and a few frequent attack types dominate the observations while some attack types have very limited representation.

A model can achieve high overall accuracy while failing to identify rare or previously unseen attack families. Evaluation results may also be overly optimistic when related network flows appear in both training and test partitions.

This research addresses these concerns by studying:

- Binary and multiclass network intrusion detection.
- The effect of class imbalance on model performance.
- Model interpretability using SHAP.
- Group-aware evaluation to reduce dependence between related observations.
- Capture-day-ordered evaluation to examine performance on attack families absent from the training period.

The research uses CIC-IDS2017 as the primary experimental dataset and includes an audit of UNSW-NB15.

## 2. Research Objectives

The main objectives are:

1. Audit network intrusion datasets and identify data-quality and label issues.
2. Develop binary intrusion detection baselines.
3. Compare class-imbalance handling strategies.
4. Evaluate multiclass intrusion detection performance.
5. Explain model predictions using SHAP.
6. Evaluate uncertainty using group-clustered bootstrap resampling.
7. Investigate capture-day-ordered generalization using a binary classification experiment.
8. Identify limitations and potential improvements for reliable intrusion detection.

## 3. Datasets and Preprocessing

### 3.1 CIC-IDS2017

CIC-IDS2017 is the primary dataset used in the experiments.

The raw CIC dataset contained eight CSV capture files and approximately 2,830,743 rows.

During preprocessing:

- Network-flow features were normalized.
- Identifier, IP-address, and timestamp columns were excluded from model features.
- Destination Port was retained.
- Numeric feature columns were used for model training.
- Conflicting labels associated with identical feature-value groups were excluded rather than reassigned through majority voting.

The resulting binary dataset contained:

| Dataset partition | Rows |
|---|---:|
| Training | 2,255,853 |
| Testing | 567,870 |
| Total | 2,823,723 |

A total of 7,020 rows belonging to 698 conflicting feature-value groups were excluded.

The processed dataset contained 78 numeric features. Missing values in Flow Bytes/s and Flow Packets/s were handled through training-fitted median imputation.

The binary split was group-aware, with no shared groups between the training and test partitions.

### 3.2 UNSW-NB15

UNSW-NB15 was also audited.

The official dataset partitions contained:

| Partition | Rows |
|---|---:|
| Training | 175,341 |
| Testing | 82,332 |

The dataset contains 45 columns.

The project includes an audit report at `reports/unsw_nb15_audit.md`. The experiments summarized in this report primarily use CIC-IDS2017; the UNSW-NB15 audit should not be interpreted as a completed cross-dataset model evaluation.

### 3.3 Preprocessing principles

The experimental pipeline follows these principles:

- Remove identifiers and other unsuitable metadata from model features.
- Fit imputation using training data only.
- Keep evaluation data separate from model fitting.
- Preserve group identities for group-aware splitting and resampling.
- Avoid tuning model thresholds on the final test data.
- Retain rare classes in confusion matrices even when they are excluded from selected headline metrics because of insufficient support.

## 4. Binary Classification Baselines

The binary classification task distinguishes benign traffic from attack traffic.

A stratified training sample of 300,000 observations was used for the baseline model comparison. The binary test partition contained 567,870 observations.

Two models were evaluated:

- Logistic Regression with median imputation and feature scaling.
- Random Forest with median imputation, 100 trees, maximum depth 20, and minimum leaf size 2.

The default classification threshold was used.

### 4.1 Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.926168 | 0.998771 |
| Macro F1-score | 0.878383 | 0.998054 |
| Attack precision | 0.848225 | 0.997919 |
| Attack recall | 0.760822 | 0.995829 |
| False-positive rate | 0.033339 | 0.000509 |
| Average Precision | 0.925757 | 0.999701 |

Random Forest achieved substantially stronger results than Logistic Regression in this baseline comparison.

**Evaluation caveat:** The binary test partition was used for the LR/RF baseline comparison. It is therefore not an untouched independent holdout for every subsequent binary experiment. Results from later experimental designs must be interpreted according to their own split and evaluation procedures.

## 5. Class-Imbalance Experiments

Class imbalance was investigated through six configurations:

- Logistic Regression baseline.
- Class-weighted Logistic Regression.
- SMOTE with Logistic Regression.
- Random Forest baseline.
- Balanced-subsample Random Forest.
- SMOTE with Random Forest.

The experiments used a group-aware training/validation split. The validation partition contained 442,245 observations. Model fitting used an inner stratified training sample of 300,000 observations.

The results showed that class-imbalance handling affected macro-level performance and attack detection. Random Forest with `balanced_subsample` provided the strongest overall macro-F1 and recall among the evaluated binary configurations.

Average Precision was used as the precision-recall summary metric. It should not be described as trapezoidal PR-AUC.

The experiment outputs are stored under `experiments/imbalance_experiments/`.

## 6. Explainability Analysis Using SHAP

SHAP was used to investigate the feature contributions of the balanced-subsample Random Forest model.

The analysis examined global feature importance and local predictions.

### 6.1 Global feature importance

The leading features included:

1. Destination Port.
2. Min Packet Length.
3. Init_Win_bytes_forward.
4. Bwd Packet Length Std.
5. Max Packet Length.
6. Average Packet Size.

The SHAP additivity check produced a maximum error of approximately \(2.72 \times 10^{-14}\), indicating that the explanation values satisfied the tested additivity relationship to numerical precision.

### 6.2 Local explanations

Local explanations were generated for true-positive, true-negative, false-positive, and false-negative examples.

The selected examples were the most confident cases in their respective categories. They should therefore be interpreted as illustrative high-confidence examples, not as representative samples of typical model behaviour.

### 6.3 Feature redundancy observation

The features `Fwd Header Length` and `Fwd Header Length.1` were found to be identical across all 2,255,853 training rows.

A feature-ablation experiment was deferred. Consequently, the effect of removing one of these duplicate features has not been established.

## 7. Multiclass CIC-IDS2017 Evaluation

The multiclass experiment used a group-aware training and validation split.

| Partition | Rows |
|---|---:|
| Training | 1,813,608 |
| Validation | 442,245 |

There were no shared groups between these partitions.

The experiment used 78 numeric features and 15 target classes. Model fitting used a stratified sample of 300,000 training observations.

The models included Random Forest and XGBoost.

### 7.1 Baseline results

| Model | Accuracy | Macro F1-score | Weighted F1-score |
|---|---:|---:|---:|
| Random Forest | 0.998216 | 0.701397 | 0.998116 |
| XGBoost | 0.999100 | 0.787566 | 0.999081 |

XGBoost achieved a higher macro F1-score than Random Forest in this experiment.

### 7.2 XGBoost imbalance strategies

Three XGBoost configurations were compared:

| Configuration | Accuracy | Macro F1-score | Weighted F1-score |
|---|---:|---:|---:|
| Baseline | 0.999100 | 0.787566 | 0.999081 |
| Balanced class weights | 0.998906 | 0.790589 | 0.998929 |
| Square-root-balanced weights | 0.999086 | 0.795563 | 0.999087 |

Square-root-balanced class weights achieved the highest point estimates for macro F1-score and weighted F1-score among these configurations.

The improvement in macro F1-score was modest. Minority-class performance remained weak, especially for classes with very low support.

The multiclass training and imbalance experiment outputs are stored under `experiments/multiclass_cic/`.

## 8. Phase 6.1 — CIC Scenario Group-Clustered Bootstrap Evaluation

### 8.1 Objective

Phase 6.1 examined the uncertainty of the three trained multiclass XGBoost configurations on the CIC scenario test partition.

The evaluation used the fixed trained models and test data. It did not refit models or tune thresholds.

### 8.2 Evaluation setup

- Test observations: 567,870.
- Unique test groups: 503,927.
- Numeric features: 78.
- Total target classes: 15.
- Eligible classes for headline macro F1-score: 12.
- Bootstrap repetitions: 5,000.
- Random seed: 42.
- Confidence interval: 95% percentile interval.

Heartbleed, Infiltration, and SQL Injection were excluded from the headline macro-F1 average because their test support was insufficient. They remained represented in the confusion matrices.

### 8.3 Point estimates

| Model | Headline macro F1-score |
|---|---:|
| XGBoost baseline | 0.907296 |
| XGBoost balanced | 0.906487 |
| XGBoost square-root-balanced | 0.912668 |

The square-root-balanced model achieved the highest point estimate.

### 8.4 Group-bootstrap confidence intervals

| Model | Lower 95% CI | Upper 95% CI |
|---|---:|---:|
| XGBoost baseline | 0.898629 | 0.915918 |
| XGBoost balanced | 0.897659 | 0.915152 |
| XGBoost square-root-balanced | 0.903866 | 0.921029 |

### 8.5 Paired model comparisons

The same sampled groups were used across models for paired comparisons.

| Comparison | Mean difference | Lower 95% CI | Upper 95% CI |
|---|---:|---:|---:|
| Baseline − balanced | +0.000809 | -0.007471 | +0.009282 |
| Baseline − square-root-balanced | -0.005372 | -0.013083 | +0.002111 |
| Balanced − square-root-balanced | -0.006181 | -0.013305 | +0.001112 |

All paired confidence intervals include zero. Therefore, the experiment does not establish a statistically significant difference between these models under this bootstrap procedure.

### 8.6 Reproducibility and limitations

The evaluation includes:

- Input and model SHA-256 integrity checks.
- Point-estimate reproduction checks against the earlier scenario evaluation.
- Group-level resampling.
- Largest-group sensitivity analysis.
- Saved confidence intervals, paired differences, and bootstrap distributions.

The bootstrap is a pooled group bootstrap, not a capture-day bootstrap. Some groups span multiple capture files. The uncertainty estimates are conditional on the fixed test partition and already-trained models.

The three low-support classes remain a limitation of the headline macro-F1 comparison.

Outputs are stored under `experiments/generalization/cic_scenario_group_bootstrap/`.

## 9. Phase 6.2 — Capture-Day-Ordered Binary Generalization Evaluation

### 9.1 Objective

Phase 6.2 investigated binary intrusion detection when models trained on earlier CIC capture files are evaluated on later capture files.

The capture files were ordered according to their documented dataset sequence. This is a capture-day-ordered evaluation, not proof of strict timestamp-based temporal validation.

### 9.2 Training and evaluation partitions

The first three capture files were used for training, and the remaining five were used for evaluation.

| Partition | Rows | Benign | Attack |
|---|---:|---:|---:|
| Training | 1,663,220 | 1,401,938 | 261,282 |
| Evaluation | 1,160,503 | 870,423 | 290,080 |

No attack families were shared between the training and evaluation partitions.

The experiment therefore investigates binary detection of traffic containing attack families absent from the training capture files. It does not evaluate multiclass identification of unseen attack families.

### 9.3 Feature-overlap audit

The audit identified 33,544 groups spanning multiple capture files across the complete binary dataset.

In the cutoff-3 evaluation partition:

- Evaluation rows: 1,160,503.
- Evaluation rows with feature vectors matching training: 59,210.
- Unique overlapping evaluation feature vectors: 23,658.
- All identified overlapping evaluation rows were benign.
- No attack rows had feature-vector overlap with training.

An overlap-excluded sensitivity evaluation retained 1,101,293 evaluation rows. Attack-family supports remained unchanged.

### 9.4 Model results on the natural evaluation partition

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.837451 | 0.819881 |
| Macro F1-score | 0.719129 | 0.665283 |
| Attack precision | 0.932780 | 0.995852 |
| Attack recall | 0.376858 | 0.280578 |
| Attack F1-score | 0.536829 | 0.437805 |
| False-positive rate | 0.009051 | 0.000389 |
| Average Precision | 0.653289 | 0.753515 |
| ROC-AUC | 0.741610 | 0.872789 |

At the default threshold, Logistic Regression achieved higher accuracy, macro F1-score, attack recall, and attack F1-score.

Random Forest achieved higher attack precision, Average Precision, and ROC-AUC, with a lower false-positive rate.

These metrics describe different aspects of performance. In particular, higher Average Precision or ROC-AUC does not necessarily imply higher attack recall at the selected classification threshold.

### 9.5 Overlap-excluded sensitivity results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.830846 | 0.810273 |
| Macro F1-score | 0.718159 | 0.661890 |
| Attack precision | 0.951875 | 0.996877 |
| Attack recall | 0.376858 | 0.280578 |
| Attack F1-score | 0.539945 | 0.437904 |
| False-positive rate | 0.006813 | 0.000314 |
| Average Precision | 0.674785 | 0.760106 |
| ROC-AUC | 0.751233 | 0.870784 |

The overlap-excluded results preserve the same broad trade-off between the models.

### 9.6 Attack-family recall

The family-level analysis showed substantial differences between attack types.

| Attack family | Evaluation support | LR detected | LR recall | RF detected | RF recall |
|---|---:|---:|---:|---:|---:|
| Bot | 1,966 | 5 | 0.002543 | 0 | 0.000000 |
| DDoS | 128,021 | 109,209 | 0.853055 | 81,244 | 0.634615 |
| Infiltration | 36 | 2 | 0.055556 | 0 | 0.000000 |
| PortScan | 157,877 | 97 | 0.000614 | 141 | 0.000893 |
| BruteForce | 1,507 | 1 | 0.000664 | 0 | 0.000000 |
| SQL Injection | 21 | 1 | 0.047619 | 3 | 0.142857 |
| XSS | 652 | 4 | 0.006135 | 2 | 0.003067 |

The models detected a substantial proportion of DDoS traffic but performed poorly on several other unseen attack families.

These family-level values are recall measurements. They should not be interpreted as family-level precision or F1-score.

### 9.7 Group-bootstrap uncertainty

The capture-day-ordered evaluation used 5,000 paired group-bootstrap repetitions with seed 42.

There were 1,017,491 unique evaluation groups, and the largest group contained 2,241 flows.

Selected 95% confidence intervals:

| Metric | LR lower | LR upper | RF lower | RF upper |
|---|---:|---:|---:|---:|
| Accuracy | 0.835624 | 0.839083 | 0.818564 | 0.821320 |
| Macro F1-score | 0.716911 | 0.721068 | 0.663825 | 0.666806 |
| Attack precision | 0.917196 | 0.942715 | 0.995333 | 0.996353 |
| Attack recall | 0.374612 | 0.379125 | 0.278630 | 0.282541 |
| Attack F1-score | 0.533449 | 0.539796 | 0.435435 | 0.440197 |
| False-positive rate | 0.007643 | 0.011300 | 0.000342 | 0.000439 |

Paired LR-minus-RF confidence intervals excluded zero for the reported metrics. Logistic Regression had higher accuracy, macro F1-score, attack recall, and attack F1-score. Random Forest had higher precision and lower false-positive rate.

Average Precision and ROC-AUC were not bootstrapped in this experiment.

### 9.8 Interpretation and limitations

The capture-day-ordered experiment demonstrates that strong random-split results do not guarantee strong detection of attack families absent from training.

The models struggled with Bot, PortScan, BruteForce, SQL Injection, XSS, and Infiltration traffic, despite strong performance on some other attack families.

The capture order follows the dataset's documented file sequence. It should not be described as a fully timestamp-validated temporal split. The experiment is also limited to CIC-IDS2017 and does not establish performance on independent network environments.

The detailed results are documented in `reports/phase_6_2_capture_day_generalization.md`.

## 10. Overall Discussion

The experiments demonstrate several important findings.

### 10.1 Accuracy is not sufficient

The baseline binary and multiclass experiments achieved very high accuracy with tree-based models. However, macro F1-score and attack-family recall revealed weaknesses that accuracy alone could conceal.

### 10.2 Imbalance handling affects minority-class performance

Balanced and square-root-balanced class weights changed multiclass macro-F1 performance. The square-root-balanced configuration achieved the highest point estimate in the multiclass validation and scenario evaluations, but the paired bootstrap results did not establish a statistically significant advantage over the other configurations.

### 10.3 Explainability provides feature-level insight

SHAP identified features that contributed strongly to model predictions. However, feature importance should not be interpreted as causal evidence or proof that a feature is universally useful across network environments.

### 10.4 Generalization depends on the evaluation design

The group-held-out scenario evaluation and capture-day-ordered binary evaluation answer different questions.

The group bootstrap quantifies uncertainty from resampling groups in a fixed test partition. The capture-day-ordered experiment investigates performance on later capture files and attack families not present in the training files.

The two experiments should not be merged into a single generalization score.

### 10.5 Unseen attack families remain difficult

The capture-day-ordered evaluation revealed weak recall for several attack families absent from training. This is a central limitation for deploying a model trained on a fixed collection of attack patterns.

## 11. Limitations and Threats to Validity

The following limitations must be considered:

1. **Dataset dependence:** The primary model experiments use CIC-IDS2017. Results may not transfer to different network environments.
2. **Rare classes:** Heartbleed, Infiltration, and SQL Injection have insufficient support for inclusion in the Phase 6.1 headline macro-F1 average.
3. **Group-bootstrap scope:** Bootstrap confidence intervals are conditional on the fixed test partition and trained models.
4. **Capture-day ordering:** The capture-day experiment follows documented file order and is not a strict timestamp-based temporal validation.
5. **Unseen attack detection:** Several unseen attack families have very low recall.
6. **Feature redundancy:** Two forward-header-length features were found to be identical, but feature ablation was deferred.
7. **SHAP example selection:** Local examples were selected from high-confidence cases and may not represent typical predictions.
8. **Binary baseline evaluation:** The original binary test partition was used for baseline comparison and is not an untouched independent holdout for every experiment.
9. **Cross-dataset validation:** UNSW-NB15 was audited, but a completed cross-dataset training and evaluation experiment is not established by the results summarized here.

These limitations restrict the conclusions that can be drawn about deployment readiness.

## 12. Conclusion

This research developed and evaluated a machine learning workflow for network intrusion detection under class imbalance, with attention to explainability and generalization.

The experiments compared binary and multiclass models, examined imbalance strategies, used SHAP for model interpretation, and evaluated uncertainty through group-clustered bootstrap procedures.

Square-root-balanced XGBoost achieved the highest multiclass point estimate in the CIC scenario evaluation. However, paired group-bootstrap confidence intervals included zero for all model comparisons, so the experiment did not establish a statistically significant winner.

In the capture-day-ordered binary experiment, Logistic Regression achieved stronger threshold-based attack recall and F1-score than Random Forest, while Random Forest achieved higher precision, Average Precision, and ROC-AUC. Both models showed poor recall for several attack families absent from training.

The main conclusion is that reliable network intrusion detection requires more than high accuracy on a random test split. Evaluation should consider class imbalance, related-flow grouping, uncertainty, attack-family coverage, and changes in the traffic distribution.

The framework provides a reproducible experimental foundation for further research. Future work should include independent cross-dataset evaluation, improved detection of rare and unseen attack families, feature-ablation studies, and more rigorous temporal validation.

---

## 13. Reproducibility and Project Artifacts

The project stores preprocessing, model, explainability, and evaluation artifacts in the repository.

Important report files:

- `reports/unsw_nb15_audit.md`
- `reports/phase_6_1_group_bootstrap.md`
- `reports/phase_6_2_capture_day_generalization.md`

Important Phase 6.1 outputs:

- `experiments/generalization/cic_scenario_group_bootstrap/point_estimates.csv`
- `experiments/generalization/cic_scenario_group_bootstrap/group_bootstrap_confidence_intervals.csv`
- `experiments/generalization/cic_scenario_group_bootstrap/paired_model_differences.csv`
- `experiments/generalization/cic_scenario_group_bootstrap/largest_group_sensitivity.csv`
- `experiments/generalization/cic_scenario_group_bootstrap/group_bootstrap_distributions.csv`
- `experiments/generalization/cic_scenario_group_bootstrap/group_bootstrap_config.json`

The Phase 6.1 test suite completed with **9 passing tests** using:

```powershell
python -m pytest -q
```

The project also uses input and model hashes in the scenario evaluation configuration to support artifact integrity checks.


### 8.6 Largest-group sensitivity analysis

The largest test group contained 6,814 observations. To examine whether this group disproportionately influenced the results, it was excluded and the headline macro F1-score was recalculated.

| Model | Remaining rows | Headline macro F1-score |
|---|---:|---:|
| XGBoost baseline | 561,056 | 0.907286 |
| XGBoost balanced | 561,056 | 0.906477 |
| XGBoost square-root-balanced | 561,056 | 0.912658 |

The resulting scores changed only slightly compared with the original point estimates. This indicates that the largest group alone did not materially determine the headline results. It does not establish generalization to independent network environments.