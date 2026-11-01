# Phase 6.2 — Capture-Day Ordered Binary Generalization

## 1. Objective

This experiment evaluates whether binary intrusion-detection models trained on earlier CIC-IDS2017 capture scenarios can detect attacks in later capture scenarios.

The experiment investigates generalization to attack families that were not represented in the training partition. It also examines the effect of overlapping feature vectors between training and evaluation data.

This is a capture-day ordered evaluation, not a true temporal evaluation, because reliable per-flow timestamps are not available in the processed dataset.

## 2. Dataset and capture order

The experiment uses the processed CIC-IDS2017 binary dataset.

- Total processed rows: 2,823,723
- Unique groups: 2,520,966
- Model features: 78
- Binary classes: BENIGN and ATTACK

The following capture-file order was used:

1. Monday-WorkingHours
2. Tuesday-WorkingHours
3. Wednesday-workingHours
4. Thursday-WorkingHours-Morning-WebAttacks
5. Thursday-WorkingHours-Afternoon-Infilteration
6. Friday-WorkingHours-Morning
7. Friday-WorkingHours-Afternoon-PortScan
8. Friday-WorkingHours-Afternoon-DDos

The order represents the dataset's documented capture scenarios. It must not be interpreted as exact per-flow chronological ordering.

## 3. Cutoff selection

Cutoff 3 was selected: the first three capture files form the training partition, while the remaining five files form the evaluation partition.

| Partition | Rows | BENIGN | ATTACK |
|---|---:|---:|---:|
| Training | 1,663,220 | 1,401,938 | 261,282 |
| Evaluation | 1,160,503 | 870,423 | 290,080 |

The training partition contains seven attack families:

- DoS GoldenEye
- DoS Hulk
- DoS Slowhttptest
- DoS slowloris
- FTP-Patator
- Heartbleed
- SSH-Patator

The evaluation partition contains seven attack families absent from training:

- Bot: 1,966
- DDoS: 128,021
- Infiltration: 36
- PortScan: 157,877
- Web Attack - Brute Force: 1,507
- Web Attack - SQL Injection: 21
- Web Attack - XSS: 652

There are zero attack families shared between the training and evaluation partitions.

Therefore, this experiment measures **binary detection of unseen attack families**, not multiclass recognition of unseen attack categories.

## 4. Leakage and overlap audits

### 4.1 Group and feature identity

The audit identified:

- Unique group IDs: 2,520,966
- Unique feature hashes: 2,520,966
- Groups mapping to multiple feature hashes: 0
- Feature hashes mapping to multiple group IDs: 0

A one-to-one relationship was observed at the 64-bit hash level. This is strong empirical evidence of identity consistency, but it is not a mathematical proof of exact equality because the audit uses finite-width hashes.

### 4.2 Evaluation feature-vector overlap

At cutoff 3:

- Evaluation rows: 1,160,503
- Rows with feature vectors matching training: 59,210
- Matching evaluation rows: 5.1021%
- Non-overlapping evaluation rows: 1,101,293
- Non-overlapping evaluation rows: 94.8979%

All 59,210 overlapping evaluation rows belong to the BENIGN class. No ATTACK evaluation rows were identified as overlapping.

The experiment therefore reports two evaluation conditions:

1. Natural evaluation: all 1,160,503 evaluation rows.
2. Overlap-excluded evaluation: 1,101,293 evaluation rows after removing matching feature vectors.

The overlap-excluded condition changes the benign evaluation population. It is a sensitivity analysis, not a replacement for the natural evaluation.

### 4.3 Group mapping

Prediction records were mapped back to their original group IDs using source-file and source-row identifiers.

- LR predictions mapped: 1,160,503 / 1,160,503
- RF predictions mapped: 1,160,503 / 1,160,503
- Unmatched predictions: 0
- Label mismatches: 0
- Unique evaluation groups: 1,017,491
- Largest evaluation group: 2,241 flows

The mapping audit passed.

## 5. Experimental setup

Two binary classifiers were evaluated.

### Logistic Regression

- Median imputation
- Standard scaling
- Logistic Regression
- Maximum iterations: 1,000
- No class weighting
- Random seed: 42

### Random Forest

- Median imputation
- 100 trees
- Maximum depth: 20
- Minimum samples per leaf: 2
- Class weighting: balanced_subsample
- Random seed: 42

Both models were trained using a deterministic class-stratified sample of 300,000 rows from the cutoff-3 training partition:

- BENIGN: 252,872
- ATTACK: 47,128

Imputation and scaling were fitted using training data only. Evaluation was performed in chunks of 100,000 rows.

The classification threshold was fixed at 0.5. No threshold tuning was performed.

## 6. Natural evaluation results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.837451 | 0.819881 |
| Macro F1 | 0.719129 | 0.665283 |
| Attack precision | 0.932780 | 0.995852 |
| Attack recall | 0.376858 | 0.280578 |
| Attack F1 | 0.536829 | 0.437805 |
| False-positive rate | 0.009051 | 0.000389 |
| Average Precision (AP) | 0.653289 | 0.753515 |
| ROC-AUC | 0.741610 | 0.872789 |

Average Precision is reported as AP. It should not be described as trapezoidal PR-AUC.

### Interpretation

Logistic Regression achieves higher accuracy, macro F1, attack recall, and attack F1 at the fixed threshold.

Random Forest achieves higher attack precision, lower false-positive rate, AP, and ROC-AUC.

The higher AP and ROC-AUC of Random Forest indicate stronger ranking performance, but do not imply better attack detection at the selected 0.5 threshold.

## 7. Overlap-excluded evaluation results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.830846 | 0.810273 |
| Macro F1 | 0.718159 | 0.661890 |
| Attack precision | 0.951875 | 0.996877 |
| Attack recall | 0.376858 | 0.280578 |
| Attack F1 | 0.539946 | 0.437904 |
| False-positive rate | 0.006813 | 0.000314 |
| Average Precision (AP) | 0.674785 | 0.760106 |
| ROC-AUC | 0.751233 | 0.870784 |

Attack recall remains identical for each model because the removed overlapping rows are all BENIGN.

The changes in precision, false-positive rate, accuracy, macro F1, AP, and ROC-AUC reflect the changed benign evaluation population. The overlap-excluded condition must not be interpreted as proof that removing overlap improves model generalization.

## 8. Attack-family recall

The following values are from the corrected family-level reports. Family-level precision and F1 are intentionally not reported because the family-specific subsets do not provide a valid basis for those binary precision calculations.

| Attack family | Support | LR detected | LR recall | RF detected | RF recall |
|---|---:|---:|---:|---:|---:|
| Bot | 1,966 | 5 | 0.002543 | 0 | 0.000000 |
| DDoS | 128,021 | 109,209 | 0.853055 | 81,244 | 0.634615 |
| Infiltration | 36 | 2 | 0.055556 | 0 | 0.000000 |
| PortScan | 157,877 | 97 | 0.000614 | 141 | 0.000893 |
| Brute Force | 1,507 | 1 | 0.000664 | 0 | 0.000000 |
| SQL Injection | 21 | 1 | 0.047619 | 3 | 0.142857 |
| XSS | 652 | 4 | 0.006135 | 2 | 0.003067 |

The models detect DDoS substantially better than most other unseen attack families. Detection is extremely weak for Bot, PortScan, Brute Force, and XSS.

Infiltration and SQL Injection have very small evaluation supports. Their recall estimates are consequently unstable and should not be treated as reliable measures of broad performance.

## 9. Group-clustered bootstrap analysis

A paired group-clustered bootstrap was performed using:

- Evaluation groups: 1,017,491
- Bootstrap replicates: 5,000
- Random seed: 42
- Confidence level: 95%

The resampling unit was the group, rather than the individual flow. The paired comparison resampled the same groups for both models.

### 9.1 Model confidence intervals

| Metric | Logistic Regression 95% CI | Random Forest 95% CI |
|---|---:|---:|
| Accuracy | 0.835624–0.839083 | 0.818564–0.821320 |
| Macro F1 | 0.716911–0.721068 | 0.663825–0.666806 |
| Attack precision | 0.917196–0.942715 | 0.995333–0.996353 |
| Attack recall | 0.374612–0.379125 | 0.278630–0.282541 |
| Attack F1 | 0.533449–0.539796 | 0.435435–0.440197 |
| False-positive rate | 0.007643–0.011300 | 0.000342–0.000439 |

### 9.2 Paired differences

Differences are calculated as Logistic Regression minus Random Forest.

| Metric | Difference 95% CI |
|---|---:|
| Accuracy | +0.015796 to +0.018720 |
| Macro F1 | +0.051840 to +0.055314 |
| Attack precision | -0.078577 to -0.053152 |
| Attack recall | +0.095147 to +0.097418 |
| Attack F1 | +0.096177 to +0.101227 |
| False-positive rate | +0.007255 to +0.010893 |

All six paired confidence intervals exclude zero.

Within this experiment, the bootstrap supports higher LR accuracy, macro F1, attack recall, and attack F1, while RF has higher attack precision and lower false-positive rate.

These are conditional comparisons for this dataset and evaluation design, not evidence of universal model superiority.

### 9.3 Bootstrap stability

The 5,000-replicate confidence intervals were compared with the earlier 1,000-replicate results. The interval boundaries changed only slightly, indicating that the reported uncertainty estimates are stable with respect to the increase in bootstrap repetitions.

## 10. Largest-group sensitivity analysis

The largest evaluation group contained 2,241 flows.

| Metric | LR all groups | LR largest excluded | RF all groups | RF largest excluded |
|---|---:|---:|---:|---:|
| Accuracy | 0.837451 | 0.837136 | 0.819881 | 0.819532 |
| Macro F1 | 0.719129 | 0.719013 | 0.665283 | 0.665160 |
| Attack precision | 0.932780 | 0.932780 | 0.995852 | 0.995852 |
| Attack recall | 0.376858 | 0.376858 | 0.280578 | 0.280578 |
| Attack F1 | 0.536829 | 0.536829 | 0.437805 | 0.437805 |
| False-positive rate | 0.009051 | 0.009074 | 0.000389 | 0.000390 |

The changes are small. The main comparison is not materially driven by this single largest evaluation group.

## 11. Main findings

1. Both models experience substantial performance degradation when evaluated on attack families absent from training.
2. Logistic Regression provides higher attack recall and attack F1 at the fixed 0.5 threshold.
3. Random Forest provides higher attack precision and substantially lower false-positive rates.
4. Random Forest has higher AP and ROC-AUC despite lower threshold-based attack recall.
5. Both models struggle with several unseen attack families, particularly Bot, PortScan, Brute Force, and XSS.
6. The feature-vector overlap audit found that all overlapping evaluation rows were BENIGN.
7. The overlap-excluded evaluation preserves attack support and attack recall but changes the benign population and other metrics.
8. Group-clustered bootstrap intervals support the observed metric trade-offs.
9. Excluding the largest evaluation group produces only small changes in the reported results.

## 12. Limitations

- The capture-file order is not a true timestamp-based temporal split.
- The experiment evaluates one selected cutoff rather than every possible cutoff.
- The training partition uses a 300,000-row stratified sample.
- The binary test dataset was previously used for baseline model comparison and is not an untouched independent holdout.
- The evaluation attack families are entirely unseen during training, so results concern unseen-family binary detection.
- Some attack families have very small evaluation supports.
- The overlap audit uses 64-bit feature hashes; theoretical hash collisions are possible.
- The overlap-excluded evaluation changes the class distribution.
- The threshold was fixed at 0.5 and was not optimized.
- AP and ROC-AUC are point estimates; the reported bootstrap intervals cover confusion-derived metrics, not ranking metrics.
- Bootstrap inference is conditional on the selected evaluation groups and does not establish generalization to other datasets or networks.

## 13. Conclusion

The capture-day ordered experiment demonstrates that strong performance on familiar CIC-IDS2017 attack scenarios does not guarantee reliable detection of unseen attack families.

Logistic Regression detects a larger proportion of attacks at the fixed decision threshold, while Random Forest is more conservative and produces fewer false positives. Neither model provides satisfactory detection across all unseen attack families.

The results highlight the importance of evaluating intrusion-detection models beyond random flow-level splits and familiar attack categories. They also show why accuracy and ranking metrics alone are insufficient when attack-family recall and false-positive control are important.

The findings should be interpreted as evidence from a capture-scenario-ordered, unseen-family binary detection experiment, not as proof of real-world temporal or cross-network generalization.

---

**Phase status:** Experimental execution, overlap audits, prediction-to-group mapping, corrected attack-family recall, 5,000-replicate group bootstrap, and largest-group sensitivity analysis completed. Final reproducibility checks and report review remain before formally closing Phase 6.2.