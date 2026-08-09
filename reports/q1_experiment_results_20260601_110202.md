# HybridShield Q1 Experiment Results

- Run ID: `20260601_110202`
- Dataset: `HAI-20.07-ICS`
- Source: `data\hai20_unified_dataset.parquet`
- Samples: `601200` (train=420840, val=60120, test=120240)
- RF trees: `220`, IF trees: `200`
- Model artifact: `C:\Users\Dell\Documents\IOT\ML_IOT\models\q1_hybridshield_20260601_110202.joblib`

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
| 1.2_full_pipeline_run | completed | HAI-20.07-ICS | IF_only |  | 0.0192 | 1.0 | 0.0377 | 0.4313 | 0.0177 | 117932.0 | 0.0 | 1.0 | 0.0 | 0.01 |
| 1.2_full_pipeline_run | completed | HAI-20.07-ICS | HybridShield_fused | 0.45 | 0.9662 | 0.9541 | 0.9601 | 0.9999 | 0.9931 | 77.0 | 106.0 | 0.0007 | 0.0459 | 0.72 |

## 1.3_cross_dataset_generalization

| experiment | status | dataset |
| --- | --- | --- |
| 1.3_cross_dataset_generalization | requires_dataset | HAI-20.07-ICS |

## 2_hyperparameter_ablation

| experiment | status | dataset | alpha | value | window_size | warm_up | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold | profile_update_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.05 | 0.05 | 100.0 | 50.0 | 0.9576 | 0.9584 | 0.958 | 0.9998 | 0.9921 | 98.0 | 96.0 | 0.0008 | 0.0416 | 0.7 | 0.013518 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 0.1 | 100.0 | 50.0 | 0.9662 | 0.9541 | 0.9601 | 0.9999 | 0.9931 | 77.0 | 106.0 | 0.0007 | 0.0459 | 0.72 | 0.013127 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.15 | 0.15 | 100.0 | 50.0 | 0.968 | 0.9445 | 0.9561 | 0.9998 | 0.9917 | 72.0 | 128.0 | 0.0006 | 0.0555 | 0.74 | 0.013035 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.2 | 0.2 | 100.0 | 50.0 | 0.9631 | 0.9272 | 0.9448 | 0.9997 | 0.9887 | 82.0 | 168.0 | 0.0007 | 0.0728 | 0.75 | 0.013163 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.3 | 0.3 | 100.0 | 50.0 | 0.9441 | 0.9151 | 0.9294 | 0.9995 | 0.9801 | 125.0 | 196.0 | 0.0011 | 0.0849 | 0.75 | 0.01631 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 25.0 | 25.0 | 50.0 | 0.8707 | 0.9077 | 0.8888 | 0.9992 | 0.9614 | 311.0 | 213.0 | 0.0026 | 0.0923 | 0.74 | 0.01378 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 50.0 | 50.0 | 50.0 | 0.9644 | 0.9398 | 0.9519 | 0.9998 | 0.9912 | 80.0 | 139.0 | 0.0007 | 0.0602 | 0.73 | 0.013241 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 100.0 | 100.0 | 50.0 | 0.9662 | 0.9541 | 0.9601 | 0.9999 | 0.9931 | 77.0 | 106.0 | 0.0007 | 0.0459 | 0.72 | 0.016531 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 150.0 | 150.0 | 50.0 | 0.9656 | 0.9484 | 0.9569 | 0.9998 | 0.992 | 78.0 | 119.0 | 0.0007 | 0.0516 | 0.72 | 0.013247 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 200.0 | 200.0 | 50.0 | 0.9622 | 0.948 | 0.955 | 0.9998 | 0.9915 | 86.0 | 120.0 | 0.0007 | 0.052 | 0.71 | 0.013263 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 10.0 | 100.0 | 10.0 | 0.9662 | 0.9536 | 0.9599 | 0.9999 | 0.9931 | 77.0 | 107.0 | 0.0007 | 0.0464 | 0.72 | 0.013045 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 20.0 | 100.0 | 20.0 | 0.9662 | 0.9536 | 0.9599 | 0.9999 | 0.9931 | 77.0 | 107.0 | 0.0007 | 0.0464 | 0.72 | 0.013189 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 50.0 | 100.0 | 50.0 | 0.9662 | 0.9541 | 0.9601 | 0.9999 | 0.9931 | 77.0 | 106.0 | 0.0007 | 0.0459 | 0.72 | 0.013087 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 100.0 | 100.0 | 100.0 | 0.9662 | 0.9541 | 0.9601 | 0.9999 | 0.9931 | 77.0 | 106.0 | 0.0007 | 0.0459 | 0.72 | 0.013036 |
| 2_hyperparameter_ablation | completed | HAI-20.07-ICS | 0.1 | 150.0 | 100.0 | 150.0 | 0.9614 | 0.9289 | 0.9449 | 0.9998 | 0.9897 | 86.0 | 164.0 | 0.0007 | 0.0711 | 0.74 | 0.013276 |

## 3.1_component_ablation

| experiment | status | dataset | variant | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.1_component_ablation | completed | HAI-20.07-ICS | RF_only |  | 0.9679 | 0.9532 | 0.9605 | 0.9999 | 0.993 | 73.0 | 108.0 | 0.0006 | 0.0468 | 0.5 |
| 3.1_component_ablation | completed | HAI-20.07-ICS | IF_only |  | 0.0192 | 1.0 | 0.0377 | 0.4313 | 0.0177 | 117932.0 | 0.0 | 1.0 | 0.0 | 0.01 |
| 3.1_component_ablation | completed | HAI-20.07-ICS | IF_RF_fused | 0.45 | 0.9662 | 0.9541 | 0.9601 | 0.9999 | 0.9931 | 77.0 | 106.0 | 0.0007 | 0.0459 | 0.72 |

## 3.2_zero_day_detection

| experiment | status | dataset | variant | held_out_attack | zero_day_recall | known_attack_recall | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | RF_only | HAI_P1 | 0.114 | 0.9973 | 0.11 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | IF_only | HAI_P1 | 1.0 | 1.0 | 0.01 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | IF_RF_fused | HAI_P1 | 0.1045 | 0.9946 | 0.47 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | RF_only | HAI_P2 | 0.0379 | 0.9702 | 0.47 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | IF_only | HAI_P2 | 1.0 | 1.0 | 0.01 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | IF_RF_fused | HAI_P2 | 0.0379 | 0.9702 | 0.52 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | RF_only | HAI_P1_P2 | 0.3686 | 0.9749 | 0.43 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | IF_only | HAI_P1_P2 | 1.0 | 1.0 | 0.01 |
| 3.2_zero_day_detection | completed | HAI-20.07-ICS | IF_RF_fused | HAI_P1_P2 | 0.3602 | 0.9715 | 0.61 |

## 3.3_if_contribution_breakdown

| experiment | status | dataset | contribution | percent_of_fused_alerts | attack_rate |
| --- | --- | --- | --- | --- | --- |
| 3.3_if_contribution_breakdown | completed | HAI-20.07-ICS | IF_only_detection | 0.0026 | 0.3333 |
| 3.3_if_contribution_breakdown | completed | HAI-20.07-ICS | RF_only_detection | 0.0 | 0.0 |
| 3.3_if_contribution_breakdown | completed | HAI-20.07-ICS | consensus_detection | 0.9974 | 0.9679 |

## 3.4_if_anomaly_score_distribution

| experiment | status | dataset | group | if_score_mean | if_score_p50 | if_score_p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 3.4_if_anomaly_score_distribution | completed | HAI-20.07-ICS | benign | 0.9966 | 0.9968 | 0.9984 |
| 3.4_if_anomaly_score_distribution | completed | HAI-20.07-ICS | known_attack | 0.9954 | 0.9965 | 0.9985 |
| 3.4_if_anomaly_score_distribution | completed | HAI-20.07-ICS | rare_attack | 0.9931 | 0.995 | 0.9982 |

## 4.1_baseline_fp_measurement

| experiment | status | dataset | strategy | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.1_baseline_fp_measurement | completed | HAI-20.07-ICS | baseline | 0.45 | 0.9662 | 0.9541 | 0.9601 | 0.9999 | 0.9931 | 77.0 | 106.0 | 0.0007 | 0.0459 | 0.72 |

## 4.2_raise_tau_base

| experiment | status | dataset | strategy | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.2137 | 0.45 | 0.0192 | 1.0 | 0.0377 | 0.9999 | 0.9931 | 117932.0 | 0.0 | 1.0 | 0.0 | 0.2137 |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.2200 | 0.45 | 0.0192 | 1.0 | 0.0377 | 0.9999 | 0.9931 | 117932.0 | 0.0 | 1.0 | 0.0 | 0.22 |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.2300 | 0.45 | 0.0192 | 1.0 | 0.0377 | 0.9999 | 0.9931 | 117932.0 | 0.0 | 1.0 | 0.0 | 0.23 |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.2500 | 0.45 | 0.0192 | 1.0 | 0.0377 | 0.9999 | 0.9931 | 117932.0 | 0.0 | 1.0 | 0.0 | 0.25 |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.2800 | 0.45 | 0.0192 | 1.0 | 0.0377 | 0.9999 | 0.9931 | 117932.0 | 0.0 | 1.0 | 0.0 | 0.28 |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.3000 | 0.45 | 0.0192 | 1.0 | 0.0377 | 0.9999 | 0.9931 | 117932.0 | 0.0 | 1.0 | 0.0 | 0.3 |
| 4.2_raise_tau_base | completed | HAI-20.07-ICS | tau=0.7200 | 0.45 | 0.9662 | 0.9541 | 0.9601 | 0.9999 | 0.9931 | 77.0 | 106.0 | 0.0007 | 0.0459 | 0.72 |

## 4.3_reduce_fusion_alpha

| experiment | status | dataset | strategy | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.3_reduce_fusion_alpha | completed | HAI-20.07-ICS | alpha=0.45 | 0.45 | 0.9662 | 0.9541 | 0.9601 | 0.9999 | 0.9931 | 77.0 | 106.0 | 0.0007 | 0.0459 | 0.72 |
| 4.3_reduce_fusion_alpha | completed | HAI-20.07-ICS | alpha=0.40 | 0.4 | 0.9761 | 0.938 | 0.9567 | 0.9999 | 0.9931 | 53.0 | 143.0 | 0.0004 | 0.062 | 0.72 |
| 4.3_reduce_fusion_alpha | completed | HAI-20.07-ICS | alpha=0.35 | 0.35 | 0.9662 | 0.9541 | 0.9601 | 0.9999 | 0.9931 | 77.0 | 106.0 | 0.0007 | 0.0459 | 0.67 |
| 4.3_reduce_fusion_alpha | completed | HAI-20.07-ICS | alpha=0.30 | 0.3 | 0.9683 | 0.9519 | 0.96 | 0.9999 | 0.9931 | 72.0 | 111.0 | 0.0006 | 0.0481 | 0.65 |
| 4.3_reduce_fusion_alpha | completed | HAI-20.07-ICS | alpha=0.25 | 0.25 | 0.9662 | 0.9541 | 0.9601 | 0.9999 | 0.9931 | 77.0 | 106.0 | 0.0007 | 0.0459 | 0.62 |

## 4.4_profile_guided_suppression

| experiment | status | dataset | strategy | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.4_profile_guided_suppression | completed | HAI-20.07-ICS | s_if<0.70_and_maha<2.0 | 0.9662 | 0.9541 | 0.9601 | 0.9999 | 0.9931 | 77.0 | 106.0 | 0.0007 | 0.0459 | 0.72 |

## 4.5_combined_mitigation

| experiment | status | dataset | strategy | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.5_combined_mitigation | completed | HAI-20.07-ICS | tau>=0.23_alpha=0.35_profile_suppression | 0.35 | 0.9662 | 0.9541 | 0.9601 | 0.9999 | 0.9931 | 77.0 | 106.0 | 0.0007 | 0.0459 | 0.67 |

## 5.1_beta_grid_search

_Showing top 30 rows for compactness. See CSV/JSON for full details._

| experiment | status | dataset | beta1 | beta2 | beta3 | is_best | validation_f1 | validation_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.025 | 0.025 | False | 0.9606 | 0.0005 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.025 | 0.05 | False | 0.9611 | 0.0006 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.025 | 0.1 | False | 0.9598 | 0.0006 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.025 | 0.15 | False | 0.959 | 0.0007 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.05 | 0.025 | False | 0.9608 | 0.0007 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.05 | 0.05 | False | 0.9604 | 0.0007 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.05 | 0.1 | False | 0.9605 | 0.0007 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.05 | 0.15 | False | 0.9606 | 0.0008 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.1 | 0.025 | False | 0.956 | 0.0011 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.1 | 0.05 | False | 0.9573 | 0.0011 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.1 | 0.1 | False | 0.9553 | 0.0012 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.1 | 0.15 | False | 0.9537 | 0.0013 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.15 | 0.025 | False | 0.9499 | 0.0016 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.15 | 0.05 | False | 0.9492 | 0.0016 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.15 | 0.1 | False | 0.9448 | 0.0018 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.05 | 0.15 | 0.15 | False | 0.8318 | 0.0073 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.025 | 0.025 | False | 0.9554 | 0.0003 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.025 | 0.05 | False | 0.9564 | 0.0003 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.025 | 0.1 | False | 0.9578 | 0.0003 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.025 | 0.15 | False | 0.9607 | 0.0004 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.05 | 0.025 | False | 0.9611 | 0.0004 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.05 | 0.05 | True | 0.9625 | 0.0004 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.05 | 0.1 | False | 0.9622 | 0.0005 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.05 | 0.15 | False | 0.9606 | 0.0005 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.1 | 0.025 | False | 0.9591 | 0.0007 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.1 | 0.05 | False | 0.9591 | 0.0007 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.1 | 0.1 | False | 0.9587 | 0.0007 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.1 | 0.15 | False | 0.9596 | 0.0007 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.15 | 0.025 | False | 0.9575 | 0.001 |
| 5.1_beta_grid_search | completed | HAI-20.07-ICS | 0.1 | 0.15 | 0.05 | False | 0.9584 | 0.001 |

## 5.2_static_vs_adaptive_threshold

| experiment | status | dataset | beta1 | beta2 | beta3 | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.2_static_vs_adaptive_threshold | completed | HAI-20.07-ICS |  |  |  | 0.9662 | 0.9541 | 0.9601 | 0.9999 | 0.9931 | 77.0 | 106.0 | 0.0007 | 0.0459 | 0.72 |
| 5.2_static_vs_adaptive_threshold | completed | HAI-20.07-ICS | 0.1 | 0.05 | 0.05 | 0.9795 | 0.9311 | 0.9547 | 0.9999 | 0.9931 | 45.0 | 159.0 | 0.0004 | 0.0689 | 0.7369 |

## 5.3_time_of_day_effect

| experiment | status | dataset | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold | time_window |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.3_time_of_day_effect | completed | HAI-20.07-ICS | 0.9815 | 0.9223 | 0.9509 | 0.9998 | 0.9945 | 13.0 | 58.0 | 0.0005 | 0.0777 | 0.7483 | 06-12 |
| 5.3_time_of_day_effect | completed | HAI-20.07-ICS | 0.9786 | 0.9353 | 0.9565 | 0.9995 | 0.9925 | 32.0 | 101.0 | 0.0011 | 0.0647 | 0.7543 | 12-18 |

## 5.4_device_risk_validation

| experiment | status | dataset | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold | device_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.4_device_risk_validation | completed | HAI-20.07-ICS | 0.9775 | 0.9255 | 0.9508 | 0.9998 | 0.9923 | 42.0 | 147.0 | 0.0004 | 0.0745 | 0.7403 | actuator |
| 5.4_device_risk_validation | completed | HAI-20.07-ICS | 0.9919 | 0.9684 | 0.98 | 0.9999 | 0.9972 | 2.0 | 8.0 | 0.0002 | 0.0316 | 0.7096 | gateway |
| 5.4_device_risk_validation | completed | HAI-20.07-ICS | 0.9873 | 0.9512 | 0.9689 | 0.9999 | 0.9965 | 1.0 | 4.0 | 0.0006 | 0.0488 | 0.7434 | sensor |

## 6.1_extended_fusion_strategy

| experiment | status | dataset | fusion_method | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.1_extended_fusion_strategy | completed | HAI-20.07-ICS | weighted_linear | 0.45 | 0.9662 | 0.9541 | 0.9601 | 0.9999 | 0.9931 | 77.0 | 106.0 | 0.0007 | 0.0459 | 0.72 |
| 6.1_extended_fusion_strategy | completed | HAI-20.07-ICS | max | 0.45 | 0.0192 | 1.0 | 0.0377 | 0.499 | 0.1164 | 117932.0 | 0.0 | 1.0 | 0.0 | 0.08 |
| 6.1_extended_fusion_strategy | completed | HAI-20.07-ICS | product | 0.45 | 0.9662 | 0.9541 | 0.9601 | 0.9999 | 0.9931 | 77.0 | 106.0 | 0.0007 | 0.0459 | 0.49 |
| 6.1_extended_fusion_strategy | completed | HAI-20.07-ICS | stacking | 0.45 | 0.9606 | 0.9606 | 0.9606 | 0.9999 | 0.9931 | 91.0 | 91.0 | 0.0008 | 0.0394 | 0.98 |
| 6.1_extended_fusion_strategy | completed | HAI-20.07-ICS | bayesian_reliability | 0.45 | 0.9679 | 0.9528 | 0.9603 | 0.9999 | 0.9931 | 73.0 | 109.0 | 0.0006 | 0.0472 | 0.65 |
| 6.1_extended_fusion_strategy | completed | HAI-20.07-ICS | rank_based | 0.45 | 0.2024 | 0.1096 | 0.1422 | 0.6223 | 0.1018 | 997.0 | 2055.0 | 0.0085 | 0.8904 | 0.94 |

## 6.2_fusion_alpha_sensitivity

| experiment | status | dataset | fusion_method | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.1 | 0.9683 | 0.9519 | 0.96 | 0.9999 | 0.9931 | 72.0 | 111.0 | 0.0006 | 0.0481 | 0.55 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.2 | 0.9765 | 0.9372 | 0.9564 | 0.9999 | 0.9931 | 52.0 | 145.0 | 0.0004 | 0.0628 | 0.63 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.3 | 0.9683 | 0.9519 | 0.96 | 0.9999 | 0.9931 | 72.0 | 111.0 | 0.0006 | 0.0481 | 0.65 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.4 | 0.9761 | 0.938 | 0.9567 | 0.9999 | 0.9931 | 53.0 | 143.0 | 0.0004 | 0.062 | 0.72 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.45 | 0.9662 | 0.9541 | 0.9601 | 0.9999 | 0.9931 | 77.0 | 106.0 | 0.0007 | 0.0459 | 0.72 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.5 | 0.9654 | 0.9558 | 0.9606 | 0.9999 | 0.9931 | 79.0 | 102.0 | 0.0007 | 0.0442 | 0.74 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.6 | 0.9647 | 0.9584 | 0.9615 | 0.9999 | 0.9931 | 81.0 | 96.0 | 0.0007 | 0.0416 | 0.79 |
| 6.2_fusion_alpha_sensitivity | completed | HAI-20.07-ICS | weighted_linear | 0.7 | 0.9774 | 0.9372 | 0.9569 | 0.9999 | 0.9931 | 50.0 | 145.0 | 0.0004 | 0.0628 | 0.86 |

## 7_behavioral_profiling_ablation

| experiment | status | dataset | variant | alpha | precision | recall | f1 | roc_auc | avg_precision | fp | fn | fpr | fnr | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7_behavioral_profiling_ablation | completed | HAI-20.07-ICS | without_profile | 0.1 | 0.9852 | 0.9489 | 0.9667 | 0.9999 | 0.9956 | 33.0 | 118.0 | 0.0003 | 0.0511 | 0.59 |
| 7_behavioral_profiling_ablation | completed | HAI-20.07-ICS | ewma_zscore_only | 0.5 | 0.9737 | 0.9636 | 0.9686 | 0.9999 | 0.9954 | 60.0 | 84.0 | 0.0005 | 0.0364 | 0.74 |
| 7_behavioral_profiling_ablation | completed | HAI-20.07-ICS | mahalanobis_only | 0.1 | 0.9739 | 0.9523 | 0.963 | 0.9999 | 0.9947 | 59.0 | 110.0 | 0.0005 | 0.0477 | 0.57 |
| 7_behavioral_profiling_ablation | completed | HAI-20.07-ICS | both_per_device | 0.6 | 0.9591 | 0.9549 | 0.957 | 0.9998 | 0.9919 | 94.0 | 104.0 | 0.0008 | 0.0451 | 0.79 |
| 7_behavioral_profiling_ablation | completed | HAI-20.07-ICS | both_global | 0.2 | 0.9662 | 0.9532 | 0.9597 | 0.9999 | 0.9936 | 77.0 | 108.0 | 0.0007 | 0.0468 | 0.59 |

## 8.1_throughput_scaling

| experiment | status | dataset | event_rate_per_sec | mean_latency_ms | p95_latency_ms | p99_latency_ms | drop_rate | cpu_usage_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8.1_throughput_scaling | completed | HAI-20.07-ICS | 100.0 | 0.050663 | 0.068395 | 0.08866 | 0.0 | 154.21 |
| 8.1_throughput_scaling | completed | HAI-20.07-ICS | 500.0 | 0.047942 | 0.064722 | 0.083899 | 0.0 | 228.14 |
| 8.1_throughput_scaling | completed | HAI-20.07-ICS | 1000.0 | 0.047468 | 0.064082 | 0.083069 | 0.0 | 98.75 |
| 8.1_throughput_scaling | completed | HAI-20.07-ICS | 2000.0 | 0.025495 | 0.034418 | 0.044617 | 0.0 | 153.22 |
| 8.1_throughput_scaling | completed | HAI-20.07-ICS | 5000.0 | 0.012009 | 0.016212 | 0.021015 | 0.0 | 286.25 |

## 8.2_device_scaling

| experiment | status | dataset | estimated_profile_memory_mb | profile_update_latency_ms |
| --- | --- | --- | --- | --- |
| 8.2_device_scaling | completed | HAI-20.07-ICS | 0.022 | 0.013281 |
| 8.2_device_scaling | completed | HAI-20.07-ICS | 0.1099 | 0.013281 |
| 8.2_device_scaling | completed | HAI-20.07-ICS | 0.2197 | 0.013281 |
| 8.2_device_scaling | completed | HAI-20.07-ICS | 1.0986 | 0.013281 |
| 8.2_device_scaling | completed | HAI-20.07-ICS | 2.1973 | 0.013281 |

## 8.3_profile_warmup_impact

| experiment | status | dataset | fp_rate | fn_rate |
| --- | --- | --- | --- | --- |
| 8.3_profile_warmup_impact | completed | HAI-20.07-ICS | 0.0 | 0.0 |
| 8.3_profile_warmup_impact | completed | HAI-20.07-ICS | 0.0007 | 0.0461 |

## model_artifact

| experiment | status | dataset | model_path |
| --- | --- | --- | --- |
| model_artifact | completed | HAI-20.07-ICS | C:\Users\Dell\Documents\IOT\ML_IOT\models\q1_hybridshield_20260601_110202.joblib |
