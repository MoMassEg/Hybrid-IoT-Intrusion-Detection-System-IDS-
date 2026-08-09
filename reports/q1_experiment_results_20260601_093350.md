# HybridShield Q1 Experiment Results

- Run ID: `20260601_093350`
- Dataset: `HAI-20.07-ICS`
- Source: `data\hai20_unified_dataset.parquet`
- Samples: `601200` (train=420840, val=60120, test=120240)
- RF trees: `180`, IF trees: `180`
- Model artifact: `C:\Users\Dell\Documents\IOT\ML_IOT\models\q1_hybridshield_20260601_093350.joblib`

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
| 1.2_full_pipeline_run | completed | HAI-20.07-ICS | RF_only |  | 0.9719 | 0.9437 | 0.9576 | 0.9998 | 0.9927 | 63.0 | 130.0 | 0.0005 | 0.0563 | 0.52 |
| 1.2_full_pipeline_run | completed | HAI-20.07-ICS | IF_only |  | 0.0564 | 0.1135 | 0.0753 | 0.5976 | 0.0315 | 4386.0 | 2046.0 | 0.0372 | 0.8865 | 0.54 |
| 1.2_full_pipeline_run | completed | HAI-20.07-ICS | HybridShield_fused | 0.45 | 0.9533 | 0.9198 | 0.9363 | 0.9981 | 0.9674 | 104.0 | 185.0 | 0.0009 | 0.0802 | 0.41 |

## 1.3_cross_dataset_generalization

| experiment | status | dataset | variant | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.3_cross_dataset_generalization | completed | Local-NSL-KDD-Harmonized | HybridShield_cross_dataset | 0.45 | 0.4804 | 0.9807 | 0.6449 | 0.5523 | 0.5319 | 12440.0 | 226.0 | 0.9236 | 0.0193 | 0.41 |

## 2_hyperparameter_ablation

| experiment | status | dataset | alpha | value | window_size | warm_up | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold | profile_update_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.05 | 0.05 | 100.0 | 50.0 | 0.9411 | 0.9064 | 0.9234 | 0.9979 | 0.9606 | 131.0 | 216.0 | 0.0011 | 0.0936 | 0.4 | 0.013159 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 0.1 | 100.0 | 50.0 | 0.9533 | 0.9198 | 0.9363 | 0.9981 | 0.9674 | 104.0 | 185.0 | 0.0009 | 0.0802 | 0.41 | 0.01303 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.15 | 0.15 | 100.0 | 50.0 | 0.9479 | 0.9216 | 0.9345 | 0.9978 | 0.9658 | 117.0 | 181.0 | 0.001 | 0.0784 | 0.41 | 0.012898 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.2 | 0.2 | 100.0 | 50.0 | 0.9385 | 0.9125 | 0.9253 | 0.9975 | 0.9598 | 138.0 | 202.0 | 0.0012 | 0.0875 | 0.41 | 0.013211 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.3 | 0.3 | 100.0 | 50.0 | 0.9082 | 0.8787 | 0.8932 | 0.9966 | 0.9416 | 205.0 | 280.0 | 0.0017 | 0.1213 | 0.41 | 0.013172 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 25.0 | 25.0 | 50.0 | 0.8645 | 0.8462 | 0.8553 | 0.9965 | 0.9176 | 306.0 | 355.0 | 0.0026 | 0.1538 | 0.43 | 0.013371 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 50.0 | 50.0 | 50.0 | 0.9472 | 0.9099 | 0.9282 | 0.998 | 0.9626 | 117.0 | 208.0 | 0.001 | 0.0901 | 0.41 | 0.013018 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 100.0 | 100.0 | 50.0 | 0.9533 | 0.9198 | 0.9363 | 0.9981 | 0.9674 | 104.0 | 185.0 | 0.0009 | 0.0802 | 0.41 | 0.013048 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 150.0 | 150.0 | 50.0 | 0.9491 | 0.9042 | 0.9261 | 0.9979 | 0.9621 | 112.0 | 221.0 | 0.0009 | 0.0958 | 0.41 | 0.01324 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 200.0 | 200.0 | 50.0 | 0.9404 | 0.9016 | 0.9206 | 0.9977 | 0.9591 | 132.0 | 227.0 | 0.0011 | 0.0984 | 0.4 | 0.013656 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 10.0 | 100.0 | 10.0 | 0.9533 | 0.9198 | 0.9363 | 0.9981 | 0.9672 | 104.0 | 185.0 | 0.0009 | 0.0802 | 0.41 | 0.013139 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 20.0 | 100.0 | 20.0 | 0.9533 | 0.9198 | 0.9363 | 0.9981 | 0.9672 | 104.0 | 185.0 | 0.0009 | 0.0802 | 0.41 | 0.012934 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 50.0 | 100.0 | 50.0 | 0.9533 | 0.9198 | 0.9363 | 0.9981 | 0.9674 | 104.0 | 185.0 | 0.0009 | 0.0802 | 0.41 | 0.013106 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 100.0 | 100.0 | 100.0 | 0.9533 | 0.9198 | 0.9363 | 0.9981 | 0.9674 | 104.0 | 185.0 | 0.0009 | 0.0802 | 0.41 | 0.01323 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 150.0 | 100.0 | 150.0 | 0.9421 | 0.9021 | 0.9216 | 0.9976 | 0.9553 | 128.0 | 226.0 | 0.0011 | 0.0979 | 0.42 | 0.013123 |

## 3.1_component_ablation

| experiment | status | dataset | variant | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.1_component_ablation | completed | HAI-20.07-ICS | RF_only |  | 0.9719 | 0.9437 | 0.9576 | 0.9998 | 0.9927 | 63.0 | 130.0 | 0.0005 | 0.0563 | 0.52 |
| 3.1_component_ablation | completed | HAI-20.07-ICS | IF_only |  | 0.0564 | 0.1135 | 0.0753 | 0.5976 | 0.0315 | 4386.0 | 2046.0 | 0.0372 | 0.8865 | 0.54 |
| 3.1_component_ablation | completed | HAI-20.07-ICS | IF_RF_fused | 0.45 | 0.9533 | 0.9198 | 0.9363 | 0.9981 | 0.9674 | 104.0 | 185.0 | 0.0009 | 0.0802 | 0.41 |

## 3.2_zero_day_detection

| experiment | status | dataset | variant | held_out_attack | zero_day_recall | known_attack_recall | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | RF_only | HAI_P1 | 0.093 | 0.9959 | 0.13 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | IF_only | HAI_P1 | 0.1076 | 0.1992 | 0.51 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | IF_RF_fused | HAI_P1 | 0.0975 | 0.9919 | 0.18 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | RF_only | HAI_P2 | 0.0303 | 0.9599 | 0.51 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | IF_only | HAI_P2 | 0.0114 | 0.1531 | 0.51 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | IF_RF_fused | HAI_P2 | 0.0303 | 0.9594 | 0.49 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | RF_only | HAI_P1_P2 | 0.3771 | 0.9754 | 0.42 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | IF_only | HAI_P1_P2 | 0.089 | 0.1424 | 0.51 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | IF_RF_fused | HAI_P1_P2 | 0.3475 | 0.9633 | 0.45 |

## 3.3_if_contribution_breakdown

| experiment | status | dataset | contribution | percent_of_fused_alerts | attack_rate |
| --- | --- | --- | --- | --- | --- |
| 3.3_if_contribution_breakdown | completed | HAI-20.07-ICS | IF_only_detection | 0.0153 | 0.0294 |
| 3.3_if_contribution_breakdown | completed | HAI-20.07-ICS | RF_only_detection | 0.8473 | 0.9799 |
| 3.3_if_contribution_breakdown | completed | HAI-20.07-ICS | consensus_detection | 0.119 | 0.9849 |

## 3.4_if_anomaly_score_distribution

| experiment | status | dataset | group | if_score_mean | if_score_p50 | if_score_p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 3.4_if_anomaly_score_distribution | completed | HAI-20.07-ICS | benign | 0.2321 | 0.203 | 0.505 |
| 3.4_if_anomaly_score_distribution | completed | HAI-20.07-ICS | known_attack | 0.2908 | 0.2371 | 0.6473 |
| 3.4_if_anomaly_score_distribution | completed | HAI-20.07-ICS | rare_attack | 0.4729 | 0.4971 | 0.7949 |

## 4.1_baseline_fp_measurement

| experiment | status | dataset | strategy | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.1_baseline_fp_measurement | completed | HAI-20.07-ICS | baseline | 0.45 | 0.9533 | 0.9198 | 0.9363 | 0.9981 | 0.9674 | 104.0 | 185.0 | 0.0009 | 0.0802 | 0.41 |

## 4.2_raise_tau_base

| experiment | status | dataset | strategy | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.2137 | 0.45 | 0.21 | 0.9944 | 0.3468 | 0.9981 | 0.9674 | 8634.0 | 13.0 | 0.0732 | 0.0056 | 0.2137 |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.2200 | 0.45 | 0.2276 | 0.9944 | 0.3704 | 0.9981 | 0.9674 | 7789.0 | 13.0 | 0.066 | 0.0056 | 0.22 |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.2300 | 0.45 | 0.2573 | 0.9922 | 0.4086 | 0.9981 | 0.9674 | 6610.0 | 18.0 | 0.056 | 0.0078 | 0.23 |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.2500 | 0.45 | 0.3305 | 0.9883 | 0.4954 | 0.9981 | 0.9674 | 4620.0 | 27.0 | 0.0392 | 0.0117 | 0.25 |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.2800 | 0.45 | 0.4466 | 0.9818 | 0.6139 | 0.9981 | 0.9674 | 2808.0 | 42.0 | 0.0238 | 0.0182 | 0.28 |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.3000 | 0.45 | 0.5204 | 0.9762 | 0.6789 | 0.9981 | 0.9674 | 2076.0 | 55.0 | 0.0176 | 0.0238 | 0.3 |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.4100 | 0.45 | 0.9533 | 0.9198 | 0.9363 | 0.9981 | 0.9674 | 104.0 | 185.0 | 0.0009 | 0.0802 | 0.41 |

## 4.3_reduce_fusion_alpha

| experiment | status | dataset | strategy | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.3_reduce_fusion_alpha | completed | HAI-20.07-ICS | alpha=0.45 | 0.45 | 0.9533 | 0.9198 | 0.9363 | 0.9981 | 0.9674 | 104.0 | 185.0 | 0.0009 | 0.0802 | 0.41 |
| 4.3_reduce_fusion_alpha | completed | HAI-20.07-ICS | alpha=0.40 | 0.4 | 0.9534 | 0.9398 | 0.9465 | 0.9988 | 0.9782 | 106.0 | 139.0 | 0.0009 | 0.0602 | 0.4 |
| 4.3_reduce_fusion_alpha | completed | HAI-20.07-ICS | alpha=0.35 | 0.35 | 0.96 | 0.9458 | 0.9529 | 0.9993 | 0.9843 | 91.0 | 125.0 | 0.0008 | 0.0542 | 0.41 |
| 4.3_reduce_fusion_alpha | completed | HAI-20.07-ICS | alpha=0.30 | 0.3 | 0.9672 | 0.9445 | 0.9557 | 0.9996 | 0.988 | 74.0 | 128.0 | 0.0006 | 0.0555 | 0.43 |
| 4.3_reduce_fusion_alpha | completed | HAI-20.07-ICS | alpha=0.25 | 0.25 | 0.9672 | 0.9463 | 0.9566 | 0.9997 | 0.9902 | 74.0 | 124.0 | 0.0006 | 0.0537 | 0.44 |

## 4.4_profile_guided_suppression

| experiment | status | dataset | strategy | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.4_profile_guided_suppression | completed | HAI-20.07-ICS | s_if<0.70_and_maha<2.0 | 0.9533 | 0.9194 | 0.936 | 0.9981 | 0.9671 | 104.0 | 186.0 | 0.0009 | 0.0806 | 0.41 |

## 4.5_combined_mitigation

| experiment | status | dataset | strategy | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.5_combined_mitigation | completed | HAI-20.07-ICS | tau>=0.23_alpha=0.35_profile_suppression | 0.35 | 0.96 | 0.9458 | 0.9529 | 0.9993 | 0.9843 | 91.0 | 125.0 | 0.0008 | 0.0542 | 0.41 |

## 5.1_beta_grid_search

_Showing top 30 rows for compactness. See CSV/JSON for full details._

| experiment | status | dataset | beta1 | beta2 | beta3 | is_best | validation_f1 | validation_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.025 | 0.025 | True | 0.9317 | 0.0006 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.025 | 0.05 | False | 0.926 | 0.0009 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.025 | 0.1 | False | 0.8783 | 0.003 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.025 | 0.15 | False | 0.8099 | 0.0065 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.05 | 0.025 | False | 0.925 | 0.0011 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.05 | 0.05 | False | 0.9089 | 0.0018 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.05 | 0.1 | False | 0.8408 | 0.0052 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.05 | 0.15 | False | 0.7707 | 0.0092 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.1 | 0.025 | False | 0.846 | 0.0055 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.1 | 0.05 | False | 0.8073 | 0.0076 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.1 | 0.1 | False | 0.743 | 0.0117 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.1 | 0.15 | False | 0.6446 | 0.0194 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.15 | 0.025 | False | 0.7406 | 0.0122 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.15 | 0.05 | False | 0.7068 | 0.0147 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.15 | 0.1 | False | 0.6261 | 0.0216 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.15 | 0.15 | False | 0.498 | 0.0373 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.025 | 0.025 | False | 0.9198 | 0.0004 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.025 | 0.05 | False | 0.9203 | 0.0004 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.025 | 0.1 | False | 0.9178 | 0.0006 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.025 | 0.15 | False | 0.8737 | 0.0027 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.05 | 0.025 | False | 0.9249 | 0.0005 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.05 | 0.05 | False | 0.9278 | 0.0005 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.05 | 0.1 | False | 0.9086 | 0.0014 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.05 | 0.15 | False | 0.8494 | 0.0042 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.1 | 0.025 | False | 0.9278 | 0.001 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.1 | 0.05 | False | 0.9115 | 0.0017 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.1 | 0.1 | False | 0.8414 | 0.0051 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.1 | 0.15 | False | 0.7707 | 0.0091 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.15 | 0.025 | False | 0.8373 | 0.0058 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.15 | 0.05 | False | 0.8013 | 0.0079 |

## 5.2_static_vs_adaptive_threshold

| experiment | status | dataset | beta1 | beta2 | beta3 | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.2_static_vs_adaptive_threshold | completed | HAI-20.07-ICS |  |  |  | 0.9533 | 0.9198 | 0.9363 | 0.9981 | 0.9674 | 104.0 | 185.0 | 0.0009 | 0.0802 | 0.41 |
| 5.2_static_vs_adaptive_threshold | completed | HAI-20.07-ICS | 0.05 | 0.025 | 0.025 | 0.9703 | 0.9068 | 0.9375 | 0.9981 | 0.9674 | 64.0 | 215.0 | 0.0005 | 0.0932 | 0.4183 |

## 5.3_time_of_day_effect

| experiment | status | dataset | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold | time_window |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.3_time_of_day_effect | completed | HAI-20.07-ICS | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.0 | 0.0 | 0.0001 | 0.0 | 0.4096 | 00-06 |
| 5.3_time_of_day_effect | completed | HAI-20.07-ICS | 0.9756 | 0.9129 | 0.9432 | 0.9987 | 0.9704 | 17.0 | 65.0 | 0.0006 | 0.0871 | 0.424 | 06-12 |
| 5.3_time_of_day_effect | completed | HAI-20.07-ICS | 0.9738 | 0.904 | 0.9376 | 0.9972 | 0.9757 | 38.0 | 150.0 | 0.0013 | 0.096 | 0.427 | 12-18 |
| 5.3_time_of_day_effect | completed | HAI-20.07-ICS | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.0 | 0.0 | 0.0002 | 0.0 | 0.4129 | 18-24 |

## 5.4_device_risk_validation

| experiment | status | dataset | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold | device_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.4_device_risk_validation | completed | HAI-20.07-ICS | 0.9774 | 0.8976 | 0.9358 | 0.9985 | 0.975 | 41.0 | 202.0 | 0.0004 | 0.1024 | 0.4201 | actuator |
| 5.4_device_risk_validation | completed | HAI-20.07-ICS | 0.961 | 0.9367 | 0.9487 | 0.9989 | 0.9722 | 3.0 | 5.0 | 0.0017 | 0.0633 | 0.4141 | camera |
| 5.4_device_risk_validation | completed | HAI-20.07-ICS | 0.9254 | 0.9688 | 0.9466 | 0.9968 | 0.9716 | 20.0 | 8.0 | 0.0015 | 0.0312 | 0.4048 | gateway |

## 6.1_extended_fusion_strategy

| experiment | status | dataset | fusion_method | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.1_extended_fusion_strategy | completed | HAI-20.07-ICS | weighted_linear | 0.45 | 0.9533 | 0.9198 | 0.9363 | 0.9981 | 0.9674 | 104.0 | 185.0 | 0.0009 | 0.0802 | 0.41 |
| 6.1_extended_fusion_strategy | completed | HAI-20.07-ICS | max | 0.45 | 0.8606 | 0.7171 | 0.7823 | 0.9889 | 0.8525 | 268.0 | 653.0 | 0.0023 | 0.2829 | 0.81 |
| 6.1_extended_fusion_strategy | completed | HAI-20.07-ICS | product | 0.45 | 0.8762 | 0.8895 | 0.8828 | 0.9989 | 0.9458 | 290.0 | 255.0 | 0.0025 | 0.1105 | 0.09 |
| 6.1_extended_fusion_strategy | completed | HAI-20.07-ICS | stacking | 0.45 | 0.9687 | 0.9519 | 0.9602 | 0.9998 | 0.993 | 71.0 | 111.0 | 0.0006 | 0.0481 | 0.99 |
| 6.1_extended_fusion_strategy | completed | HAI-20.07-ICS | bayesian_reliability | 0.45 | 0.9533 | 0.9454 | 0.9493 | 0.9991 | 0.9819 | 107.0 | 126.0 | 0.0009 | 0.0546 | 0.4 |
| 6.1_extended_fusion_strategy | completed | HAI-20.07-ICS | rank_based | 0.45 | 0.5283 | 0.2712 | 0.3584 | 0.7554 | 0.295 | 559.0 | 1682.0 | 0.0047 | 0.7288 | 0.93 |

## 6.2_fusion_alpha_sensitivity

| experiment | status | dataset | fusion_method | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.1 | 0.965 | 0.9558 | 0.9604 | 0.9998 | 0.9924 | 80.0 | 102.0 | 0.0007 | 0.0442 | 0.46 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.2 | 0.9726 | 0.9393 | 0.9557 | 0.9998 | 0.9913 | 61.0 | 140.0 | 0.0005 | 0.0607 | 0.47 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.3 | 0.9672 | 0.9445 | 0.9557 | 0.9996 | 0.988 | 74.0 | 128.0 | 0.0006 | 0.0555 | 0.43 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.4 | 0.9534 | 0.9398 | 0.9465 | 0.9988 | 0.9782 | 106.0 | 139.0 | 0.0009 | 0.0602 | 0.4 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.45 | 0.9533 | 0.9198 | 0.9363 | 0.9981 | 0.9674 | 104.0 | 185.0 | 0.0009 | 0.0802 | 0.41 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.5 | 0.9689 | 0.8505 | 0.9059 | 0.9967 | 0.9462 | 63.0 | 345.0 | 0.0005 | 0.1495 | 0.45 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.6 | 0.6525 | 0.6776 | 0.6648 | 0.9885 | 0.7658 | 833.0 | 744.0 | 0.0071 | 0.3224 | 0.45 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.7 | 0.2926 | 0.5581 | 0.3839 | 0.9627 | 0.4191 | 3114.0 | 1020.0 | 0.0264 | 0.4419 | 0.41 |

## 7_behavioral_profiling_ablation

| experiment | status | dataset | variant | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7_behavioral_profiling_ablation | completed | HAI-20.07-ICS | without_profile | 0.2 | 0.9762 | 0.9614 | 0.9688 | 0.9999 | 0.9947 | 54.0 | 89.0 | 0.0005 | 0.0386 | 0.47 |
| 7_behavioral_profiling_ablation | completed | HAI-20.07-ICS | ewma_zscore_only | 0.1 | 0.9657 | 0.9744 | 0.97 | 0.9999 | 0.9952 | 80.0 | 59.0 | 0.0007 | 0.0256 | 0.45 |
| 7_behavioral_profiling_ablation | completed | HAI-20.07-ICS | mahalanobis_only | 0.1 | 0.974 | 0.9567 | 0.9652 | 0.9999 | 0.9948 | 59.0 | 100.0 | 0.0005 | 0.0433 | 0.51 |
| 7_behavioral_profiling_ablation | completed | HAI-20.07-ICS | both_per_device | 0.1 | 0.9699 | 0.9354 | 0.9524 | 0.9998 | 0.9904 | 67.0 | 149.0 | 0.0006 | 0.0646 | 0.49 |
| 7_behavioral_profiling_ablation | completed | HAI-20.07-ICS | both_global | 0.1 | 0.9628 | 0.9532 | 0.958 | 0.9998 | 0.9922 | 85.0 | 108.0 | 0.0007 | 0.0468 | 0.47 |

## 8.1_throughput_scaling

| experiment | status | dataset | event_rate_per_sec | mean_latency_ms | p95_latency_ms | p99_latency_ms | drop_rate | cpu_usage_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8.1_throughput_scaling | completed | HAI-20.07-ICS | 100.0 | 0.138931 | 0.187556 | 0.243129 | 0.0 | 134.96 |
| 8.1_throughput_scaling | completed | HAI-20.07-ICS | 500.0 | 0.141068 | 0.190442 | 0.246869 | 0.0 | 143.99 |
| 8.1_throughput_scaling | completed | HAI-20.07-ICS | 1000.0 | 0.129633 | 0.175005 | 0.226858 | 0.0 | 120.53 |
| 8.1_throughput_scaling | completed | HAI-20.07-ICS | 2000.0 | 0.083143 | 0.112243 | 0.1455 | 0.0 | 122.15 |
| 8.1_throughput_scaling | completed | HAI-20.07-ICS | 5000.0 | 0.052688 | 0.071129 | 0.092205 | 0.0 | 136.42 |

## 8.2_device_scaling

| experiment | status | dataset | estimated_profile_memory_mb | profile_update_latency_ms |
| --- | --- | --- | --- | --- |
| 8.2_device_scaling | completed | HAI-20.07-ICS | 0.022 | 0.013154 |
| 8.2_device_scaling | completed | HAI-20.07-ICS | 0.1099 | 0.013154 |
| 8.2_device_scaling | completed | HAI-20.07-ICS | 0.2197 | 0.013154 |
| 8.2_device_scaling | completed | HAI-20.07-ICS | 1.0986 | 0.013154 |
| 8.2_device_scaling | completed | HAI-20.07-ICS | 2.1973 | 0.013154 |

## 8.3_profile_warmup_impact

| experiment | status | dataset | fp_rate | fn_rate |
| --- | --- | --- | --- | --- |
| 8.3_profile_warmup_impact | completed | HAI-20.07-ICS | 0.0 | 0.1111 |
| 8.3_profile_warmup_impact | completed | HAI-20.07-ICS | 0.0009 | 0.08 |

## model_artifact

| experiment | status | dataset | model_path |
| --- | --- | --- | --- |
| model_artifact | completed | HAI-20.07-ICS | C:\Users\Dell\Documents\IOT\ML_IOT\models\q1_hybridshield_20260601_093350.joblib |
