# Hybrid IoT Intrusion Detection System (IDS)

A comprehensive machine learning framework for detecting network intrusions in IoT environments using a hybrid approach combining unsupervised anomaly detection with supervised attack classification.

## 📋 Project Overview

This framework implements a 4-phase pipeline for IoT network security:

1. **Phase 1 - Data Loading & Preprocessing**: Acquires datasets (UNSW-NB15, CIC-IDS-2017) with automatic fallback to synthetic data generation
2. **Phase 2 - Behavioral Profiling**: Builds per-device statistical profiles using EWMA and Mahalanobis distance
3. **Phase 3 - ML Models**: Trains two parallel models:
   - Isolation Forest (unsupervised anomaly detection)
   - Random Forest (supervised attack classification)
4. **Phase 4 - Decision Fusion & Adaptive Thresholding**: Combines model outputs with context-aware alert generation

## 📁 Project Structure

```
hybrid_iot_ids/
├── config/
│   ├── config.py              # Configuration: datasets, features, hyperparameters
│   └── __init__.py
├── phase1_data/
│   ├── data_loader.py         # Data acquisition, preprocessing, feature harmonization
│   └── __init__.py
├── phase2_profiling/
│   ├── behavioral_profiler.py # Device profile management
│   └── __init__.py
├── phase3_models/
│   ├── isolation_forest_model.py  # Unsupervised anomaly detection
│   ├── random_forest_model.py     # Supervised attack classification
│   └── __init__.py
├── phase4_fusion/
│   ├── decision_fusion.py     # Score fusion engine
│   ├── adaptive_threshold.py  # Context-aware thresholding & alerting
│   └── __init__.py
├── utils/
│   ├── logger.py              # Logging utility
│   └── __init__.py
├── data/                      # Dataset cache directory (auto-created)
├── models/                    # Trained model serialization (auto-created)
├── logs/                      # Log files (auto-created)
├── reports/                   # Alert reports (auto-created)
├── main.py                    # Pipeline orchestrator
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd hybrid_iot_ids

# Create a Python virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
# Execute the full 4-phase pipeline with default settings
python main.py
```

**Default behavior:**
- Uses cached dataset if available (otherwise downloads UNSW-NB15 or CIC-IDS-2017, or generates synthetic data)
- Trains models without hyperparameter tuning (fast mode)
- Saves trained models to `models/` directory
- Generates alert log in `reports/phase4_alerts.json`

### 3. Advanced Options

Edit `main.py` at the bottom or modify parameters:

```python
final_results = run_pipeline(
    use_cache   = True,       # Use cached dataset (False to re-download)
    tune_models = True,       # Enable hyperparameter tuning (slower, ~5-10 min)
    save_models = True        # Save models to disk for reuse
)
```

## 📊 Dataset Support

The framework automatically attempts to load datasets in this priority order:

1. **Cached Dataset** (if `use_cache=True`): Loads from `data/unified_dataset_cache.parquet`
2. **UNSW-NB15** (Priority 1): Download from GitHub mirror
3. **CIC-IDS-2017** (Priority 2): Download from GitHub mirror
4. **Synthetic IoT Dataset** (Fallback): Auto-generated with 15,000 samples (85% normal, 15% attack)

### Supported Features

The framework harmonizes these feature categories across datasets:

- **Traffic Volume**: packet_rate, byte_rate, burst_length, inter_arrival_time
- **Connection Behavior**: conn_frequency, unique_dst_ips, port_diversity, protocol_distribution
- **Session Level**: response_time, session_duration, payload_size_mean, payload_size_std
- **Temporal**: time_of_day, day_of_week (cyclically encoded)

## 🔍 Phase Descriptions

### Phase 1: Data Loading & Preprocessing
- Downloads or loads cached datasets
- Harmonizes dataset-specific columns to unified schema
- Handles missing values, scaling (RobustScaler), and temporal encoding
- Applies SMOTE for class imbalance correction
- Output: Preprocessed feature matrix X, binary labels (0=normal, 1=attack), multi-class labels (attack types)

### Phase 2: Behavioral Profiling
- Creates per-device statistical profiles using Exponential Weighted Moving Average (EWMA)
- Computes Mahalanobis distance and Z-score deviations
- Augments feature matrix with deviation features for each device
- Output: Enhanced feature matrix with behavioral context

### Phase 3: ML Models
#### Isolation Forest (Unsupervised)
- Trained on normal traffic only
- Produces anomaly scores ∈ [0, 1]
- Hyperparameters: n_estimators=100, contamination=0.1, max_features=1.0

#### Random Forest (Supervised)
- Trained on balanced dataset (SMOTE-augmented)
- Multi-class attack classification
- Produces attack probability ∈ [0, 1]
- Hyperparameters: n_estimators=200, max_depth=None, class_weight="balanced"

### Phase 4: Decision Fusion & Adaptive Thresholding
#### Decision Fusion
- Combines IF and RF scores using 4 methods:
  - **weighted_linear**: Sfused = α·sIF + (1-α)·pRF
  - **max**: Sfused = max(sIF, pRF)
  - **product**: Sfused = sIF · pRF
  - **stacking**: Meta-learner (Logistic Regression)
- Optimizes fusion weight α on validation set

#### Adaptive Threshold
- Dynamic threshold: τ(t,d) = τ_base + β₁·L(t) - β₂·R(d) - β₃·T(t)
  - L(t): Network load factor
  - R(d): Device risk level
  - T(t): Time-of-day activity
- Alert levels: NONE / WARNING / CRITICAL
- Generates structured alerts with context

## 📈 Output & Logging

### Log Files
- Location: `logs/hybrid_ids_YYYYMMDD.log`
- Includes console (INFO+) and file (DEBUG+) handlers
- Tracks all pipeline phases and metrics

### Alert Reports
- Location: `reports/phase4_alerts.json`
- JSON format with complete alert context
- Fields: alert_id, device_id, timestamp, fused_score, threshold_used, alert_level, attack_type, context

### Final Report
Printed to console at pipeline completion with:
- Phase 1: Data source, total samples, feature count
- Phase 2: Device profiles, warmed-up count, augmented dimensions
- Phase 3: Model metrics (ROC-AUC, F1, FPR) for IF and RF
- Phase 4: Fusion method, calibrated threshold, final metrics, alert statistics

## ⚙️ Configuration

Edit `config/config.py` to customize:

```python
# Dataset URLs
DATASET_CONFIG = {
    "UNSW_NB15": {...},
    "CIC_IDS2017": {...},
    "SYNTHETIC_IOT": {...}
}

# Feature schema
FEATURE_COLUMNS = {...}

# Model hyperparameters
ISOLATION_FOREST_CONFIG = {...}
RANDOM_FOREST_CONFIG = {...}

# Fusion settings
FUSION_CONFIG = {
    "alpha": 0.5,                      # Fusion weight
    "fusion_method": "weighted_linear" # Fusion strategy
}

# Adaptive threshold
THRESHOLD_CONFIG = {
    "tau_base": 0.5,      # Base threshold
    "beta1": 0.1,         # Network load weight
    "beta2": 0.05,        # Device risk weight
    "beta3": 0.05,        # Time-of-day weight
    "warning_margin": 0.15
}

# Training split
TRAIN_CONFIG = {
    "test_size": 0.2,
    "val_size": 0.1,
    "cv_folds": 5
}
```

## 🔧 Hyperparameter Tuning

To enable grid search over model hyperparameters:

```python
# In main.py, set tune_models=True
final_results = run_pipeline(tune_models=True)
```

This searches over:
- **Isolation Forest**: n_estimators, contamination, max_features, max_samples
- **Random Forest**: Auto-tuned via cross-validation

**Note**: Tuning adds 5-10 minutes to execution time.

## 📦 Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **scikit-learn**: ML models and metrics
- **imbalanced-learn**: SMOTE oversampling
- **requests**: Dataset downloading
- **joblib**: Model serialization
- **scipy**: Statistical functions

See `requirements.txt` for pinned versions.

## 🎯 Example Usage

```python
from main import run_pipeline

# Standard execution
results = run_pipeline(
    use_cache=True,
    tune_models=False,
    save_models=True
)

# Access results
phase1_info = results["phase1"]  # Data stats
phase3_metrics = results["phase3"]["if_test"]  # IF metrics
phase4_alerts = results["phase4"]["alert_summary"]  # Alert stats
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Network download failures | Set `use_cache=False` to retry, or manually place CSV in `data/` |
| SMOTE K-neighbors error | Automatically handled for small datasets (k adjusted down) |
| Memory issues with large datasets | Reduce `n_samples` in SyntheticIoTDataset or use caching |
| Missing log files | Ensure `logs/` directory has write permissions |

## 📝 Citation & References

This framework implements techniques from:
- Isolation Forest: Liu et al. (2008)
- Random Forest: Breiman (2001)
- SMOTE: Chawla et al. (2002)
- Adaptive Thresholding: Context-aware anomaly detection for IoT

## 📄 License

Research framework for academic and internal use.

## 👤 Author

Hybrid IoT IDS Research Framework v1.0

---

**For questions or issues, review log files in `logs/` directory.**
