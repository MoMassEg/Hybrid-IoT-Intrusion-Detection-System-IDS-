# =============================================================================
# Configuration File - Hybrid IoT IDS Framework
# =============================================================================

import os

# ─────────────────────────────────────────────
# Dataset Configuration
# ─────────────────────────────────────────────
DATASET_CONFIG = {
    "UNSW_NB15": {
        "name": "UNSW-NB15",
        "urls": [
            "https://raw.githubusercontent.com/denysvitali/datasets/main/UNSW-NB15/UNSW_NB15_training-set.csv",
        ],
        "fallback_url": "https://people.ucsc.edu/~tkrose/datasets/UNSW_NB15_training-set.csv",
        "label_column": "label",
        "attack_column": "attack_cat",
        "separator": ",",
        "priority": 1
    },
    "CIC_IDS2017": {
        "name": "CIC-IDS-2017",
        "urls": [
            "https://raw.githubusercontent.com/imfaisalmalik/IDS-2017-Dataset/main/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        ],
        "label_column": " Label",
        "attack_column": " Label",
        "separator": ",",
        "priority": 2
    },
    "SYNTHETIC_IOT": {
        "name": "Synthetic-IoT",
        "urls": [],   # generated programmatically
        "label_column": "label",
        "attack_column": "attack_cat",
        "separator": ",",
        "priority": 3   # fallback
    }
}

# ─────────────────────────────────────────────
# Feature Schema
# ─────────────────────────────────────────────
FEATURE_COLUMNS = {
    "traffic_volume": [
        "packet_rate",
        "byte_rate",
        "burst_length",
        "inter_arrival_time"
    ],
    "connection_behavior": [
        "conn_frequency",
        "unique_dst_ips",
        "port_diversity",
        "protocol_distribution"
    ],
    "session_level": [
        "response_time",
        "session_duration",
        "payload_size_mean",
        "payload_size_std"
    ],
    "temporal": [
        "time_of_day",
        "day_of_week"
    ]
}

ALL_FEATURES = (
    FEATURE_COLUMNS["traffic_volume"] +
    FEATURE_COLUMNS["connection_behavior"] +
    FEATURE_COLUMNS["session_level"] +
    FEATURE_COLUMNS["temporal"]
)

# ─────────────────────────────────────────────
# Phase 3 - Model Configuration
# ─────────────────────────────────────────────
ISOLATION_FOREST_CONFIG = {
    "n_estimators": 100,
    "max_samples": "auto",
    "contamination": 0.1,       # 10% anomaly rate assumed
    "max_features": 1.0,
    "random_state": 42,
    "n_jobs": -1
}

RANDOM_FOREST_CONFIG = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1
}

# ─────────────────────────────────────────────
# Phase 4 - Fusion & Threshold Configuration
# ─────────────────────────────────────────────
FUSION_CONFIG = {
    "alpha": 0.5,              # Fusion weight: 0=RF only, 1=IF only
    "fusion_method": "weighted_linear",   # Options: weighted_linear, max, product, stacking
}

THRESHOLD_CONFIG = {
    "tau_base": 0.5,           # Base detection threshold
    "beta1": 0.1,              # Network load weight
    "beta2": 0.05,             # Device risk weight
    "beta3": 0.05,             # Time-of-day weight
    "warning_margin": 0.15,    # Buffer above tau for CRITICAL
}

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, REPORT_DIR]:
    os.makedirs(directory, exist_ok=True)

# ─────────────────────────────────────────────
# Training Configuration
# ─────────────────────────────────────────────
TRAIN_CONFIG = {
    "test_size": 0.2,
    "val_size": 0.1,
    "random_state": 42,
    "cv_folds": 5,
    "smote_random_state": 42
}
