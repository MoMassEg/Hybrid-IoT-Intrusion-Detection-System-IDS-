# HybridShield Q1 Experiment Results

- Run ID: `20260601_102121`
- Dataset: `HAI-20.07-ICS`
- Source: `data\hai20_unified_dataset.parquet`
- Samples: `601200` (train=420840, val=60120, test=120240)
- RF trees: `220`, IF trees: `200`
- Model artifact: `C:\Users\Dell\Documents\IOT\ML_IOT\models\q1_hybridshield_20260601_102121.joblib`

## Methodological Notes

- HAI-20.07 is the ICS-native dataset used to address the critical-infrastructure reviewer request. NSL-KDD is retained only as a cross-dataset stress target when present.
- CPU% is process CPU time divided by wall time; values above 100% indicate multi-core aggregate usage from parallel model inference.
- Throughput latency is measured in batches, so larger event-rate batches amortize fixed model-call overhead and can show lower per-event latency.
- IF-only detections and profile suppression are reported as measured. If these rows are weak, the paper should frame IF as an auxiliary score-calibration signal rather than a source of many unique true positives.
- Any prior claim that profiling improves F1 by a fixed percentage must be replaced by the measured dataset-specific ablation result.

## Research Integrity Note

Rows with status requires_dataset or not_run are placeholders for reproducibility tracking, not fabricated performance claims.

## 1.1_new_ics_dataset_integration

| experiment | status | dataset |
| --- | --- | --- |
| 1.1_new_ics_dataset_integration | completed | HAI-20.07-ICS |
| 1.1_new_ics_dataset_integration | requires_dataset | SWaT |
| 1.1_new_ics_dataset_integration | requires_dataset | DS2OS |

## 1.2_full_pipeline_dataset

| experiment | status | dataset |
| --- | --- | --- |
| 1.2_full_pipeline_dataset | completed | HAI-20.07-ICS |

## 1.2_full_pipeline_run

| experiment | status | dataset | variant | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.2_full_pipeline_run | completed | HAI-20.07-ICS | RF_only |  | 0.9679 | 0.9532 | 0.9605 | 0.9999 | 0.993 | 73.0 | 108.0 | 0.0006 | 0.0468 | 0.5 |
| 1.2_full_pipeline_run | completed | HAI-20.07-ICS | IF_only |  | 0.0599 | 0.1101 | 0.0776 | 0.598 | 0.0322 | 3984.0 | 2054.0 | 0.0338 | 0.8899 | 0.55 |
| 1.2_full_pipeline_run | completed | HAI-20.07-ICS | HybridShield_fused | 0.45 | 0.9442 | 0.9311 | 0.9376 | 0.9982 | 0.9688 | 127.0 | 159.0 | 0.0011 | 0.0689 | 0.4 |

## 1.3_cross_dataset_generalization

| experiment | status | dataset | variant | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.3_cross_dataset_generalization | completed | NSL-KDD-Harmonized | HybridShield_cross_dataset | 0.45 | 0.4677 | 0.9984 | 0.637 | 0.5187 | 0.505 | 13324.0 | 19.0 | 0.9892 | 0.0016 | 0.4 |

## 2_hyperparameter_ablation

| experiment | status | dataset | alpha | value | window_size | warm_up | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold | profile_update_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.05 | 0.05 | 100.0 | 50.0 | 0.9446 | 0.9086 | 0.9262 | 0.9981 | 0.9628 | 123.0 | 211.0 | 0.001 | 0.0914 | 0.4 | 0.012398 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 0.1 | 100.0 | 50.0 | 0.9442 | 0.9311 | 0.9376 | 0.9982 | 0.9688 | 127.0 | 159.0 | 0.0011 | 0.0689 | 0.4 | 0.012269 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.15 | 0.15 | 100.0 | 50.0 | 0.9481 | 0.9259 | 0.9369 | 0.9979 | 0.9674 | 117.0 | 171.0 | 0.001 | 0.0741 | 0.41 | 0.011986 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.2 | 0.2 | 100.0 | 50.0 | 0.9266 | 0.9237 | 0.9251 | 0.9976 | 0.9615 | 169.0 | 176.0 | 0.0014 | 0.0763 | 0.4 | 0.012207 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.3 | 0.3 | 100.0 | 50.0 | 0.8924 | 0.8986 | 0.8955 | 0.9968 | 0.9441 | 250.0 | 234.0 | 0.0021 | 0.1014 | 0.4 | 0.012374 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 25.0 | 25.0 | 50.0 | 0.8391 | 0.8653 | 0.852 | 0.9966 | 0.9177 | 383.0 | 311.0 | 0.0032 | 0.1347 | 0.42 | 0.012459 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 50.0 | 50.0 | 50.0 | 0.9369 | 0.9203 | 0.9285 | 0.9981 | 0.9642 | 143.0 | 184.0 | 0.0012 | 0.0797 | 0.4 | 0.012489 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 100.0 | 100.0 | 50.0 | 0.9442 | 0.9311 | 0.9376 | 0.9982 | 0.9688 | 127.0 | 159.0 | 0.0011 | 0.0689 | 0.4 | 0.012371 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 150.0 | 150.0 | 50.0 | 0.9426 | 0.9181 | 0.9302 | 0.998 | 0.9639 | 129.0 | 189.0 | 0.0011 | 0.0819 | 0.4 | 0.012199 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 200.0 | 200.0 | 50.0 | 0.9435 | 0.9038 | 0.9232 | 0.9978 | 0.9611 | 125.0 | 222.0 | 0.0011 | 0.0962 | 0.4 | 0.012414 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 10.0 | 100.0 | 10.0 | 0.9442 | 0.9311 | 0.9376 | 0.9982 | 0.9686 | 127.0 | 159.0 | 0.0011 | 0.0689 | 0.4 | 0.01247 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 20.0 | 100.0 | 20.0 | 0.9442 | 0.9311 | 0.9376 | 0.9982 | 0.9686 | 127.0 | 159.0 | 0.0011 | 0.0689 | 0.4 | 0.012332 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 50.0 | 100.0 | 50.0 | 0.9442 | 0.9311 | 0.9376 | 0.9982 | 0.9688 | 127.0 | 159.0 | 0.0011 | 0.0689 | 0.4 | 0.012478 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 100.0 | 100.0 | 100.0 | 0.9442 | 0.9311 | 0.9376 | 0.9982 | 0.9688 | 127.0 | 159.0 | 0.0011 | 0.0689 | 0.4 | 0.012242 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 150.0 | 100.0 | 150.0 | 0.929 | 0.9129 | 0.9209 | 0.9978 | 0.9563 | 161.0 | 201.0 | 0.0014 | 0.0871 | 0.41 | 0.012276 |

## 3.1_component_ablation

| experiment | status | dataset | variant | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.1_component_ablation | completed | HAI-20.07-ICS | RF_only |  | 0.9679 | 0.9532 | 0.9605 | 0.9999 | 0.993 | 73.0 | 108.0 | 0.0006 | 0.0468 | 0.5 |
| 3.1_component_ablation | completed | HAI-20.07-ICS | IF_only |  | 0.0599 | 0.1101 | 0.0776 | 0.598 | 0.0322 | 3984.0 | 2054.0 | 0.0338 | 0.8899 | 0.55 |
| 3.1_component_ablation | completed | HAI-20.07-ICS | IF_RF_fused | 0.45 | 0.9442 | 0.9311 | 0.9376 | 0.9982 | 0.9688 | 127.0 | 159.0 | 0.0011 | 0.0689 | 0.4 |

## 3.2_zero_day_detection

| experiment | status | dataset | variant | held_out_attack | zero_day_recall | known_attack_recall | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | RF_only | HAI_P1 | 0.114 | 0.9973 | 0.11 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | IF_only | HAI_P1 | 0.086 | 0.1721 | 0.54 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | IF_RF_fused | HAI_P1 | 0.1739 | 0.9973 | 0.1 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | RF_only | HAI_P2 | 0.0379 | 0.9702 | 0.47 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | IF_only | HAI_P2 | 0.0114 | 0.1267 | 0.54 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | IF_RF_fused | HAI_P2 | 0.0341 | 0.9618 | 0.49 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | RF_only | HAI_P1_P2 | 0.3686 | 0.9749 | 0.43 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | IF_only | HAI_P1_P2 | 0.0763 | 0.1178 | 0.54 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | IF_RF_fused | HAI_P1_P2 | 0.3559 | 0.9667 | 0.41 |

## 3.3_if_contribution_breakdown

| experiment | status | dataset | contribution | percent_of_fused_alerts | attack_rate |
| --- | --- | --- | --- | --- | --- |
| 3.3_if_contribution_breakdown | completed | HAI-20.07-ICS | IF_only_detection | 0.0198 | 0.0222 |
| 3.3_if_contribution_breakdown | completed | HAI-20.07-ICS | RF_only_detection | 0.848 | 0.9777 |
| 3.3_if_contribution_breakdown | completed | HAI-20.07-ICS | consensus_detection | 0.1129 | 0.9844 |

## 3.4_if_anomaly_score_distribution

| experiment | status | dataset | group | if_score_mean | if_score_p50 | if_score_p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 3.4_if_anomaly_score_distribution | completed | HAI-20.07-ICS | benign | 0.2346 | 0.206 | 0.504 |
| 3.4_if_anomaly_score_distribution | completed | HAI-20.07-ICS | known_attack | 0.2941 | 0.2393 | 0.6483 |
| 3.4_if_anomaly_score_distribution | completed | HAI-20.07-ICS | rare_attack | 0.4751 | 0.5009 | 0.796 |

## 4.1_baseline_fp_measurement

| experiment | status | dataset | strategy | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.1_baseline_fp_measurement | completed | HAI-20.07-ICS | baseline | 0.45 | 0.9442 | 0.9311 | 0.9376 | 0.9982 | 0.9688 | 127.0 | 159.0 | 0.0011 | 0.0689 | 0.4 |

## 4.2_raise_tau_base

| experiment | status | dataset | strategy | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.2137 | 0.45 | 0.2097 | 0.9935 | 0.3463 | 0.9982 | 0.9688 | 8642.0 | 15.0 | 0.0733 | 0.0065 | 0.2137 |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.2200 | 0.45 | 0.229 | 0.9935 | 0.3722 | 0.9982 | 0.9688 | 7721.0 | 15.0 | 0.0655 | 0.0065 | 0.22 |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.2300 | 0.45 | 0.2596 | 0.9931 | 0.4116 | 0.9982 | 0.9688 | 6538.0 | 16.0 | 0.0554 | 0.0069 | 0.23 |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.2500 | 0.45 | 0.3327 | 0.9879 | 0.4977 | 0.9982 | 0.9688 | 4574.0 | 28.0 | 0.0388 | 0.0121 | 0.25 |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.2800 | 0.45 | 0.4482 | 0.9814 | 0.6154 | 0.9982 | 0.9688 | 2788.0 | 43.0 | 0.0236 | 0.0186 | 0.28 |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.3000 | 0.45 | 0.5245 | 0.977 | 0.6826 | 0.9982 | 0.9688 | 2044.0 | 53.0 | 0.0173 | 0.023 | 0.3 |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.4000 | 0.45 | 0.9442 | 0.9311 | 0.9376 | 0.9982 | 0.9688 | 127.0 | 159.0 | 0.0011 | 0.0689 | 0.4 |

## 4.3_reduce_fusion_alpha

| experiment | status | dataset | strategy | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.3_reduce_fusion_alpha | completed | HAI-20.07-ICS | alpha=0.45 | 0.45 | 0.9442 | 0.9311 | 0.9376 | 0.9982 | 0.9688 | 127.0 | 159.0 | 0.0011 | 0.0689 | 0.4 |
| 4.3_reduce_fusion_alpha | completed | HAI-20.07-ICS | alpha=0.40 | 0.4 | 0.9605 | 0.9376 | 0.9489 | 0.9989 | 0.9789 | 89.0 | 144.0 | 0.0008 | 0.0624 | 0.41 |
| 4.3_reduce_fusion_alpha | completed | HAI-20.07-ICS | alpha=0.35 | 0.35 | 0.9627 | 0.9406 | 0.9516 | 0.9993 | 0.9848 | 84.0 | 137.0 | 0.0007 | 0.0594 | 0.42 |
| 4.3_reduce_fusion_alpha | completed | HAI-20.07-ICS | alpha=0.30 | 0.3 | 0.9659 | 0.9445 | 0.9551 | 0.9996 | 0.9884 | 77.0 | 128.0 | 0.0007 | 0.0555 | 0.43 |
| 4.3_reduce_fusion_alpha | completed | HAI-20.07-ICS | alpha=0.25 | 0.25 | 0.9715 | 0.9437 | 0.9574 | 0.9997 | 0.9906 | 64.0 | 130.0 | 0.0005 | 0.0563 | 0.45 |

## 4.4_profile_guided_suppression

| experiment | status | dataset | strategy | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.4_profile_guided_suppression | completed | HAI-20.07-ICS | s_if<0.70_and_maha<2.0 | 0.9442 | 0.9307 | 0.9374 | 0.9982 | 0.9686 | 127.0 | 160.0 | 0.0011 | 0.0693 | 0.4 |

## 4.5_combined_mitigation

| experiment | status | dataset | strategy | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.5_combined_mitigation | completed | HAI-20.07-ICS | tau>=0.23_alpha=0.35_profile_suppression | 0.35 | 0.9627 | 0.9406 | 0.9516 | 0.9993 | 0.9848 | 84.0 | 137.0 | 0.0007 | 0.0594 | 0.42 |

## 5.1_beta_grid_search

_Showing top 30 rows for compactness. See CSV/JSON for full details._

| experiment | status | dataset | beta1 | beta2 | beta3 | is_best | validation_f1 | validation_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.025 | 0.025 | True | 0.9357 | 0.0006 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.025 | 0.05 | False | 0.9267 | 0.0011 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.025 | 0.1 | False | 0.8782 | 0.0033 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.025 | 0.15 | False | 0.8027 | 0.0073 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.05 | 0.025 | False | 0.9257 | 0.0013 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.05 | 0.05 | False | 0.9067 | 0.0022 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.05 | 0.1 | False | 0.8294 | 0.006 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.05 | 0.15 | False | 0.7564 | 0.0103 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.1 | 0.025 | False | 0.8216 | 0.0066 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.1 | 0.05 | False | 0.7876 | 0.0086 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.1 | 0.1 | False | 0.7177 | 0.0132 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.1 | 0.15 | False | 0.6298 | 0.0205 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.15 | 0.025 | False | 0.7114 | 0.0139 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.15 | 0.05 | False | 0.6797 | 0.0164 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.15 | 0.1 | False | 0.6005 | 0.0237 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.15 | 0.15 | False | 0.4679 | 0.0417 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.025 | 0.025 | False | 0.9238 | 0.0004 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.025 | 0.05 | False | 0.9267 | 0.0004 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.025 | 0.1 | False | 0.9203 | 0.0008 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.025 | 0.15 | False | 0.8752 | 0.003 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.05 | 0.025 | False | 0.9293 | 0.0005 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.05 | 0.05 | False | 0.9298 | 0.0005 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.05 | 0.1 | False | 0.9076 | 0.0016 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.05 | 0.15 | False | 0.839 | 0.0048 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.1 | 0.025 | False | 0.9236 | 0.0011 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.1 | 0.05 | False | 0.9023 | 0.0021 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.1 | 0.1 | False | 0.8233 | 0.006 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.1 | 0.15 | False | 0.7514 | 0.0102 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.15 | 0.025 | False | 0.809 | 0.007 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.15 | 0.05 | False | 0.7791 | 0.0088 |

## 5.2_static_vs_adaptive_threshold

| experiment | status | dataset | beta1 | beta2 | beta3 | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.2_static_vs_adaptive_threshold | completed | HAI-20.07-ICS |  |  |  | 0.9442 | 0.9311 | 0.9376 | 0.9982 | 0.9688 | 127.0 | 159.0 | 0.0011 | 0.0689 | 0.4 |
| 5.2_static_vs_adaptive_threshold | completed | HAI-20.07-ICS | 0.05 | 0.025 | 0.025 | 0.9626 | 0.9146 | 0.938 | 0.9982 | 0.9688 | 82.0 | 197.0 | 0.0007 | 0.0854 | 0.4125 |

## 5.3_time_of_day_effect

| experiment | status | dataset | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold | time_window |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.3_time_of_day_effect | completed | HAI-20.07-ICS | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 6.0 | 0.0 | 0.0002 | 0.0 | 0.4038 | 00-06 |
| 5.3_time_of_day_effect | completed | HAI-20.07-ICS | 0.9703 | 0.9196 | 0.9443 | 0.9988 | 0.9718 | 21.0 | 60.0 | 0.0007 | 0.0804 | 0.4183 | 06-12 |
| 5.3_time_of_day_effect | completed | HAI-20.07-ICS | 0.9681 | 0.9123 | 0.9394 | 0.9973 | 0.9764 | 47.0 | 137.0 | 0.0016 | 0.0877 | 0.4211 | 12-18 |
| 5.3_time_of_day_effect | completed | HAI-20.07-ICS | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 8.0 | 0.0 | 0.0003 | 0.0 | 0.407 | 18-24 |

## 5.4_device_risk_validation

| experiment | status | dataset | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold | device_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.4_device_risk_validation | completed | HAI-20.07-ICS | 0.9127 | 0.9759 | 0.9432 | 0.9975 | 0.9722 | 31.0 | 8.0 | 0.0021 | 0.0241 | 0.3947 | gateway |
| 5.4_device_risk_validation | completed | HAI-20.07-ICS | 0.9723 | 0.9044 | 0.9371 | 0.9986 | 0.9757 | 51.0 | 189.0 | 0.0005 | 0.0956 | 0.4151 | sensor |

## 6.1_extended_fusion_strategy

| experiment | status | dataset | fusion_method | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.1_extended_fusion_strategy | completed | HAI-20.07-ICS | weighted_linear | 0.45 | 0.9442 | 0.9311 | 0.9376 | 0.9982 | 0.9688 | 127.0 | 159.0 | 0.0011 | 0.0689 | 0.4 |
| 6.1_extended_fusion_strategy | completed | HAI-20.07-ICS | max | 0.45 | 0.9225 | 0.6863 | 0.7871 | 0.9892 | 0.8601 | 133.0 | 724.0 | 0.0011 | 0.3137 | 0.83 |
| 6.1_extended_fusion_strategy | completed | HAI-20.07-ICS | product | 0.45 | 0.8758 | 0.8956 | 0.8856 | 0.9989 | 0.9469 | 293.0 | 241.0 | 0.0025 | 0.1044 | 0.09 |
| 6.1_extended_fusion_strategy | completed | HAI-20.07-ICS | stacking | 0.45 | 0.9622 | 0.9606 | 0.9614 | 0.9999 | 0.9933 | 87.0 | 91.0 | 0.0007 | 0.0394 | 0.98 |
| 6.1_extended_fusion_strategy | completed | HAI-20.07-ICS | bayesian_reliability | 0.45 | 0.9598 | 0.9424 | 0.951 | 0.9992 | 0.9824 | 91.0 | 133.0 | 0.0008 | 0.0576 | 0.41 |
| 6.1_extended_fusion_strategy | completed | HAI-20.07-ICS | rank_based | 0.45 | 0.4539 | 0.2968 | 0.3589 | 0.7625 | 0.3011 | 824.0 | 1623.0 | 0.007 | 0.7032 | 0.92 |

## 6.2_fusion_alpha_sensitivity

| experiment | status | dataset | fusion_method | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.1 | 0.9749 | 0.9411 | 0.9577 | 0.9998 | 0.9926 | 56.0 | 136.0 | 0.0005 | 0.0589 | 0.5 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.2 | 0.974 | 0.9415 | 0.9575 | 0.9998 | 0.9916 | 58.0 | 135.0 | 0.0005 | 0.0585 | 0.47 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.3 | 0.9659 | 0.9445 | 0.9551 | 0.9996 | 0.9884 | 77.0 | 128.0 | 0.0007 | 0.0555 | 0.43 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.4 | 0.9605 | 0.9376 | 0.9489 | 0.9989 | 0.9789 | 89.0 | 144.0 | 0.0008 | 0.0624 | 0.41 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.45 | 0.9442 | 0.9311 | 0.9376 | 0.9982 | 0.9688 | 127.0 | 159.0 | 0.0011 | 0.0689 | 0.4 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.5 | 0.9634 | 0.8674 | 0.9129 | 0.9969 | 0.9491 | 76.0 | 306.0 | 0.0006 | 0.1326 | 0.44 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.6 | 0.686 | 0.6854 | 0.6857 | 0.9891 | 0.7829 | 724.0 | 726.0 | 0.0061 | 0.3146 | 0.45 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.7 | 0.3077 | 0.5269 | 0.3885 | 0.9637 | 0.4343 | 2736.0 | 1092.0 | 0.0232 | 0.4731 | 0.42 |

## 7_behavioral_profiling_ablation

| experiment | status | dataset | variant | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7_behavioral_profiling_ablation | completed | HAI-20.07-ICS | without_profile | 0.1 | 0.9839 | 0.9549 | 0.9692 | 0.9999 | 0.9956 | 36.0 | 104.0 | 0.0003 | 0.0451 | 0.52 |
| 7_behavioral_profiling_ablation | completed | HAI-20.07-ICS | ewma_zscore_only | 0.1 | 0.9684 | 0.9692 | 0.9688 | 0.9999 | 0.9952 | 73.0 | 71.0 | 0.0006 | 0.0308 | 0.46 |
| 7_behavioral_profiling_ablation | completed | HAI-20.07-ICS | mahalanobis_only | 0.1 | 0.9797 | 0.9415 | 0.9602 | 0.9999 | 0.9943 | 45.0 | 135.0 | 0.0004 | 0.0585 | 0.53 |
| 7_behavioral_profiling_ablation | completed | HAI-20.07-ICS | both_per_device | 0.1 | 0.9714 | 0.9411 | 0.956 | 0.9998 | 0.9914 | 64.0 | 136.0 | 0.0005 | 0.0589 | 0.49 |
| 7_behavioral_profiling_ablation | completed | HAI-20.07-ICS | both_global | 0.1 | 0.9711 | 0.9471 | 0.959 | 0.9999 | 0.9934 | 65.0 | 122.0 | 0.0006 | 0.0529 | 0.49 |

## 8.1_throughput_scaling

| experiment | status | dataset | event_rate_per_sec | mean_latency_ms | p95_latency_ms | p99_latency_ms | drop_rate | cpu_usage_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8.1_throughput_scaling | completed | HAI-20.07-ICS | 100.0 | 0.153102 | 0.206688 | 0.267928 | 0.0 | 163.29 |
| 8.1_throughput_scaling | completed | HAI-20.07-ICS | 500.0 | 0.152188 | 0.205453 | 0.266328 | 0.0 | 123.2 |
| 8.1_throughput_scaling | completed | HAI-20.07-ICS | 1000.0 | 0.151045 | 0.20391 | 0.264328 | 0.0 | 113.79 |
| 8.1_throughput_scaling | completed | HAI-20.07-ICS | 2000.0 | 0.097281 | 0.13133 | 0.170242 | 0.0 | 128.49 |
| 8.1_throughput_scaling | completed | HAI-20.07-ICS | 5000.0 | 0.065896 | 0.08896 | 0.115319 | 0.0 | 137.53 |

## 8.2_device_scaling

| experiment | status | dataset | estimated_profile_memory_mb | profile_update_latency_ms |
| --- | --- | --- | --- | --- |
| 8.2_device_scaling | completed | HAI-20.07-ICS | 0.022 | 0.012531 |
| 8.2_device_scaling | completed | HAI-20.07-ICS | 0.1099 | 0.012531 |
| 8.2_device_scaling | completed | HAI-20.07-ICS | 0.2197 | 0.012531 |
| 8.2_device_scaling | completed | HAI-20.07-ICS | 1.0986 | 0.012531 |
| 8.2_device_scaling | completed | HAI-20.07-ICS | 2.1973 | 0.012531 |

## 8.3_profile_warmup_impact

| experiment | status | dataset | fp_rate | fn_rate |
| --- | --- | --- | --- | --- |
| 8.3_profile_warmup_impact | completed | HAI-20.07-ICS | 0.0 | 0.1111 |
| 8.3_profile_warmup_impact | completed | HAI-20.07-ICS | 0.0011 | 0.0687 |

## model_artifact

| experiment | status | dataset | model_path |
| --- | --- | --- | --- |
| model_artifact | completed | HAI-20.07-ICS | C:\Users\Dell\Documents\IOT\ML_IOT\models\q1_hybridshield_20260601_102121.joblib |
