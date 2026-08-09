# =============================================================================
# main.py — Full Pipeline Orchestrator
# Hybrid ML Framework for IoT Intrusion & Anomaly Detection
# Phases 1 → 2 → 3 → 4
# =============================================================================

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Internal Modules ──────────────────────────────────────────────────────────
from config.config import ALL_FEATURES, REPORT_DIR, MODEL_DIR, DATA_DIR
from utils.logger import get_logger

from phase1_data.data_loader import IoTDataLoader
from phase2_profiling.behavioral_profiler import BehavioralProfiler
from phase3_models.sgd_one_class_svm_model import SGDOneClassSVMModel
from phase3_models.random_forest_model import RandomForestModel
from phase4_fusion.decision_fusion import DecisionFusionEngine
from phase4_fusion.adaptive_threshold import AdaptiveThresholdEngine

logger = get_logger("MainPipeline")


# =============================================================================
# Pipeline Configuration
# =============================================================================
PIPELINE_CONFIG = {
    # ── Data ─────────────────────────────────────────────────────
    "use_cache":           True,      # Load from CSV cache if available
    "force_synthetic":     False,     # Skip online download, use synthetic

    # ── Phase 2 ──────────────────────────────────────────────────
    "profiler_warmup":     50,        # Min samples before profile activates
    "n_devices":           20,        # Simulated device count

    # ── Phase 3 ──────────────────────────────────────────────────
    "tune_if":             False,     # Run IF hyperparameter grid search
    "tune_rf":             False,     # Run RF hyperparameter grid search
    "use_feature_select":  True,      # RF: remove low-importance features
    "calibrate_proba":     True,      # RF: Platt scaling calibration
    "tune_threshold":      True,      # Both: tune decision threshold on val

    # ── Phase 4 ──────────────────────────────────────────────────
    "target_fpr":          0.05,      # Calibrate threshold for ≤5% FPR
    "fusion_method":       "auto",    # auto / weighted_linear / max / product
    "simulate_load":       True,      # Simulate varying network conditions

    # ── Output ───────────────────────────────────────────────────
    "save_models":         True,      # Serialize models to disk
    "save_reports":        True,      # Save JSON reports
    "model_version":       "v1",
}


# =============================================================================
# Utility: Print Section Banner
# =============================================================================
def _banner(title: str, char: str = "═", width: int = 58) -> None:
    line = char * width
    logger.info(f"\n{line}")
    logger.info(f"  {title}")
    logger.info(f"{line}")


# =============================================================================
# Utility: Save JSON Report
# =============================================================================
def _save_report(data: dict, filename: str) -> str:
    path = os.path.join(REPORT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Report saved: {path}")
    return path


# =============================================================================
# Utility: Convert numpy types for JSON serialization
# =============================================================================
def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# =============================================================================
# PHASE 1: Data Loading & Preprocessing
# =============================================================================
def run_phase1(cfg: dict, loader: IoTDataLoader) -> dict:
    """
    Phase 1: Dataset acquisition, harmonization, and preprocessing.

    Steps:
        1. Load dataset (cache → online → synthetic)
        2. Preprocess (scale, encode, split)
        3. SMOTE balancing
        4. Extract normal-only subset

    Returns:
        p1: dict with all arrays and metadata
    """
    _banner("PHASE 1: DATA & ENVIRONMENT SETUP")

    # ── 1.1 Load ─────────────────────────────────────────────────
    if cfg["force_synthetic"]:
        logger.info("force_synthetic=True → skipping online download")
        from phase1_data.data_loader import SyntheticIoTDataset
        df_raw = SyntheticIoTDataset(15000, 0.15).generate()
        loader.df_raw         = df_raw
        loader.dataset_source = "Synthetic (forced)"
        loader.downloader.save_cache(df_raw, "iot_ids_dataset")
    else:
        df_raw = loader.load_dataset(use_cache=cfg["use_cache"])

    logger.info(f"Dataset source : {loader.dataset_source}")
    logger.info(f"Dataset shape  : {df_raw.shape}")

    # ── 1.2 Preprocess ───────────────────────────────────────────
    logger.info("\n[1.2] Preprocessing...")
    X, y_binary, y_multi, feature_names = loader.preprocess(df_raw)

    # ── 1.3 Train / Val / Test Split ─────────────────────────────
    logger.info("\n[1.3] Splitting dataset...")
    splits = loader.train_val_test_split(X, y_binary)
    X_train = splits["X_train"]
    X_val   = splits["X_val"]
    X_test  = splits["X_test"]
    y_train = splits["y_train"]
    y_val   = splits["y_val"]
    y_test  = splits["y_test"]

    # ── 1.4 Normal-only for IF ────────────────────────────────────
    logger.info("\n[1.4] Extracting normal samples for SGD One-Class SVM...")
    X_normal = loader.get_normal_data(X_train, y_train)

    # ── 1.5 Device IDs ────────────────────────────────────────────
    n_total = len(X)
    if "device_id" in df_raw.columns:
        all_device_ids = df_raw["device_id"].values[:n_total]
    else:
        rng = np.random.RandomState(42)
        all_device_ids = np.array([
            f"device_{rng.randint(1, cfg['n_devices'] + 1):03d}"
            for _ in range(n_total)
        ])

    n_train = len(X_train)
    n_val   = len(X_val)
    n_test  = len(X_test)

    train_device_ids = all_device_ids[:n_train]
    val_device_ids   = all_device_ids[n_train: n_train + n_val]
    test_device_ids  = all_device_ids[n_train + n_val: n_train + n_val + n_test]

    # ── Summary ───────────────────────────────────────────────────
    label_dist = dict(zip(*np.unique(y_binary, return_counts=True)))
    attack_pct = label_dist.get(1, 0) / len(y_binary) * 100

    p1 = {
        # Raw
        "df_raw":           df_raw,
        "dataset_source":   loader.dataset_source,
        # Arrays
        "X":                X,
        "y_binary":         y_binary,
        "y_multi":          y_multi,
        "feature_names":    feature_names,
        # Splits
        "X_train":          X_train,
        "X_val":            X_val,
        "X_test":           X_test,
        "y_train":          y_train,
        "y_val":            y_val,
        "y_test":           y_test,
        # Normal
        "X_normal":         X_normal,
        # Device IDs
        "all_device_ids":   all_device_ids,
        "train_device_ids": train_device_ids,
        "val_device_ids":   val_device_ids,
        "test_device_ids":  test_device_ids,
        # Metadata
        "n_total":          n_total,
        "n_train":          n_train,
        "n_val":            n_val,
        "n_test":           n_test,
        "n_features":       len(feature_names),
        "n_normal_train":   len(X_normal),
        "label_dist":       {int(k): int(v) for k, v in label_dist.items()},
        "attack_pct":       round(attack_pct, 2),
    }

    logger.info("\n── Phase 1 Summary ──────────────────────────────────")
    logger.info(f"  Source      : {p1['dataset_source']}")
    logger.info(f"  Total rows  : {p1['n_total']:,}")
    logger.info(f"  Features    : {p1['n_features']}")
    logger.info(f"  Train       : {p1['n_train']:,}")
    logger.info(f"  Val         : {p1['n_val']:,}")
    logger.info(f"  Test        : {p1['n_test']:,}")
    logger.info(f"  Normal (IF) : {p1['n_normal_train']:,}")
    logger.info(f"  Attack %    : {p1['attack_pct']:.2f}%")
    logger.info(f"  Label dist  : {p1['label_dist']}")

    return p1


# =============================================================================
# PHASE 2: Behavioral Profiling
# =============================================================================
def run_phase2(cfg: dict, p1: dict) -> dict:
    """
    Phase 2: Build per-device behavioral profiles and augment features.

    Steps:
        1. Initialize BehavioralProfiler
        2. Fit profiles on normal training data
        3. Augment all splits with deviation features
        4. Re-apply SMOTE on augmented training data

    Returns:
        p2: dict with augmented arrays and profiler
    """
    _banner("PHASE 2: BEHAVIORAL PROFILING")

    loader   = IoTDataLoader()
    profiler = BehavioralProfiler(warm_up_samples=cfg["profiler_warmup"])

    feature_names = p1["feature_names"]
    X_train       = p1["X_train"]
    X_val         = p1["X_val"]
    X_test        = p1["X_test"]
    y_train       = p1["y_train"]

    # ── 2.1 Fit profiles on normal training data ──────────────────
    logger.info("[2.1] Building device profiles from normal traffic...")

    normal_mask  = y_train == 0
    X_normal_raw = X_train[normal_mask]
    dev_ids_norm = p1["train_device_ids"][normal_mask]

    df_normal = pd.DataFrame(X_normal_raw, columns=feature_names)
    df_normal["device_id"] = dev_ids_norm
    profiler.fit_profiles(df_normal, feature_names, device_col="device_id")

    n_warm = sum(1 for p in profiler.profiles.values() if p.is_warm)
    logger.info(
        f"  Profiles built : {len(profiler.profiles)} total | "
        f"{n_warm} warmed"
    )

    # ── 2.2 Augment all splits ────────────────────────────────────
    logger.info("\n[2.2] Augmenting feature matrices with deviation scores...")

    X_train_aug = profiler.compute_deviation_features(
        X_train,
        list(p1["train_device_ids"]),
        feature_names
    )
    X_val_aug = profiler.compute_deviation_features(
        X_val,
        list(p1["val_device_ids"]),
        feature_names
    )
    X_test_aug = profiler.compute_deviation_features(
        X_test,
        list(p1["test_device_ids"]),
        feature_names
    )
    X_normal_aug = profiler.compute_deviation_features(
        p1["X_normal"],
        list(p1["train_device_ids"][:len(p1["X_normal"])]),
        feature_names
    )

    aug_feature_names = feature_names + [
        "deviation_zscore_mean",
        "mahalanobis_dist"
    ]

    # ── 2.3 SMOTE on augmented data ───────────────────────────────
    logger.info("\n[2.3] Applying SMOTE on augmented training data...")
    X_train_aug_smote, y_train_smote = loader.apply_smote(
        X_train_aug, y_train
    )

    p2 = {
        "profiler":            profiler,
        "aug_feature_names":   aug_feature_names,
        # Augmented splits
        "X_train_aug":         X_train_aug,
        "X_val_aug":           X_val_aug,
        "X_test_aug":          X_test_aug,
        "X_normal_aug":        X_normal_aug,
        # SMOTE balanced
        "X_train_aug_smote":   X_train_aug_smote,
        "y_train_smote":       y_train_smote,
        # Stats
        "n_profiles":          len(profiler.profiles),
        "n_warmed":            n_warm,
        "n_aug_features":      len(aug_feature_names),
        "aug_shape":           X_train_aug.shape,
    }

    logger.info("\n── Phase 2 Summary ──────────────────────────────────")
    logger.info(f"  Device profiles  : {p2['n_profiles']}")
    logger.info(f"  Warmed profiles  : {p2['n_warmed']}")
    logger.info(f"  Original features: {p1['n_features']}")
    logger.info(f"  Augmented dims   : {p2['n_aug_features']}")
    logger.info(f"  Train (SMOTE)    : {X_train_aug_smote.shape[0]:,}")

    return p2


# =============================================================================
# PHASE 3: ML Model Development
# =============================================================================
def run_phase3(cfg: dict, p1: dict, p2: dict) -> dict:
    """
    Phase 3: Train SGD One-Class SVM + Random Forest.

    SGD One-Class SVM:
        - Trained on normal-only augmented data
        - Contamination estimated from label ratio
        - Threshold tuned on validation set

    Random Forest:
        - Trained on SMOTE-balanced binary labels
        - Probability calibration (Platt scaling)
        - Feature selection
        - Threshold tuned on validation set

    Returns:
        p3: dict with models, scores, and metrics
    """
    _banner("PHASE 3: ML MODELS DEVELOPMENT")

    aug_fn    = p2["aug_feature_names"]
    X_val_aug = p2["X_val_aug"]
    X_test_aug= p2["X_test_aug"]
    y_val     = p1["y_val"]
    y_test    = p1["y_test"]

    # ══════════════════════════════════════════════════════════════
    # 3A: SGD One-Class SVM
    # ══════════════════════════════════════════════════════════════
    _banner("3A: SGD One-Class SVM (Unsupervised)", char="─", width=50)

    if_model = SGDOneClassSVMModel()

    # Optional: full hyperparameter search
    if cfg["tune_if"]:
        logger.info("Running IF hyperparameter grid search...")
        best_cfg = if_model.tune_hyperparameters(
            p2["X_normal_aug"],
            X_val_aug,
            y_val,
            feature_names=aug_fn
        )
        if_model = SGDOneClassSVMModel(config=best_cfg)
    else:
        logger.info("Skipping IF tuning (tune_if=False)")

    # Train
    if_model.train(
        p2["X_normal_aug"],
        feature_names = aug_fn,
        y_all         = p1["y_train"],      # for contamination estimate
        X_val         = X_val_aug,          # for threshold tuning
        y_val         = y_val
    )

    # Evaluate
    logger.info("\n[IF] Validation set evaluation:")
    if_metrics_val  = if_model.evaluate(X_val_aug,  y_val,  "Validation")

    logger.info("\n[IF] Test set evaluation:")
    if_metrics_test = if_model.evaluate(X_test_aug, y_test, "Test")

    # Scores for Phase 4
    s_if_val  = if_model.predict_scores(X_val_aug)
    s_if_test = if_model.predict_scores(X_test_aug)

    logger.info(
        f"\n  IF Score Stats (test): "
        f"min={s_if_test.min():.3f} | "
        f"max={s_if_test.max():.3f} | "
        f"mean={s_if_test.mean():.3f} | "
        f"std={s_if_test.std():.3f}"
    )

    # ══════════════════════════════════════════════════════════════
    # 3B: Random Forest
    # ══════════════════════════════════════════════════════════════
    _banner("3B: Random Forest (Supervised)", char="─", width=50)

    # Use BINARY labels — avoids multi-class normal class confusion
    y_train_rf = p1["y_train"]
    y_val_rf   = p1["y_val"]
    y_test_rf  = p1["y_test"]

    # Re-apply SMOTE on augmented binary data
    logger.info("[RF] Re-applying SMOTE for binary RF training...")
    loader_tmp = IoTDataLoader()
    X_rf_smote, y_rf_smote = loader_tmp.apply_smote(
        p2["X_train_aug"], y_train_rf
    )
    logger.info(
        f"  RF training set after SMOTE: {X_rf_smote.shape[0]:,} samples"
    )

    rf_model = RandomForestModel()
    rf_model.train(
        X_rf_smote,
        y_rf_smote,
        X_val                = X_val_aug,
        y_val                = y_val_rf,
        feature_names        = aug_fn,
        use_feature_selection = cfg["use_feature_select"],
        calibrate_proba      = cfg["calibrate_proba"],
        tune_threshold       = cfg["tune_threshold"]
    )

    # Evaluate
    logger.info("\n[RF] Validation set evaluation:")
    rf_metrics_val  = rf_model.evaluate(X_val_aug,  y_val_rf,  "Validation")

    logger.info("\n[RF] Test set evaluation:")
    rf_metrics_test = rf_model.evaluate(X_test_aug, y_test_rf, "Test")

    # Scores for Phase 4
    p_rf_val  = rf_model.predict_proba_attack(X_val_aug)
    p_rf_test = rf_model.predict_proba_attack(X_test_aug)

    logger.info(
        f"\n  RF Score Stats (test): "
        f"min={p_rf_test.min():.3f} | "
        f"max={p_rf_test.max():.3f} | "
        f"mean={p_rf_test.mean():.3f} | "
        f"std={p_rf_test.std():.3f}"
    )

    # Attack type labels from RF prediction
    attack_labels_test = [
        "Attack" if p >= rf_model.optimal_threshold else "Normal"
        for p in p_rf_test
    ]

    # ── Save Models ───────────────────────────────────────────────
    if cfg["save_models"]:
        if_model.save(version=cfg["model_version"])
        rf_model.save(version=cfg["model_version"])

    # ── Comparative Table ─────────────────────────────────────────
    _print_model_comparison(if_metrics_test, rf_metrics_test)

    p3 = {
        # Models
        "if_model":           if_model,
        "rf_model":           rf_model,
        # Metrics
        "if_metrics_val":     if_metrics_val,
        "if_metrics_test":    if_metrics_test,
        "rf_metrics_val":     rf_metrics_val,
        "rf_metrics_test":    rf_metrics_test,
        # Scores
        "s_if_val":           s_if_val,
        "s_if_test":          s_if_test,
        "p_rf_val":           p_rf_val,
        "p_rf_test":          p_rf_test,
        # Labels
        "attack_labels_test": attack_labels_test,
        "y_val_rf":           y_val_rf,
        "y_test_rf":          y_test_rf,
        # Score stats
        "if_score_stats": {
            "min":  round(float(s_if_test.min()),  4),
            "max":  round(float(s_if_test.max()),  4),
            "mean": round(float(s_if_test.mean()), 4),
            "std":  round(float(s_if_test.std()),  4),
        },
        "rf_score_stats": {
            "min":  round(float(p_rf_test.min()),  4),
            "max":  round(float(p_rf_test.max()),  4),
            "mean": round(float(p_rf_test.mean()), 4),
            "std":  round(float(p_rf_test.std()),  4),
        },
    }

    return p3


# =============================================================================
# PHASE 4: Fusion & Adaptive Threshold
# =============================================================================
def run_phase4(cfg: dict, p1: dict, p2: dict, p3: dict) -> dict:
    """
    Phase 4: Decision fusion + adaptive threshold + alert generation.

    Steps:
        1. Optimize fusion weight alpha
        2. Train stacking meta-learner
        3. Compare all fusion methods → select best
        4. Calibrate base threshold for target FPR
        5. Process test set → generate alerts
        6. Compute final fused metrics

    Returns:
        p4: dict with fused scores, alerts, and metrics
    """
    _banner("PHASE 4: FUSION & ADAPTIVE THRESHOLD")

    X_val_aug  = p2["X_val_aug"]
    X_test_aug = p2["X_test_aug"]
    y_val      = p1["y_val"]
    y_test     = p1["y_test"]

    s_if_val  = p3["s_if_val"]
    s_if_test = p3["s_if_test"]
    p_rf_val  = p3["p_rf_val"]
    p_rf_test = p3["p_rf_test"]

    # ══════════════════════════════════════════════════════════════
    # 4A: Decision Fusion
    # ══════════════════════════════════════════════════════════════
    _banner("4A: Decision Fusion", char="─", width=50)

    fusion = DecisionFusionEngine()

    # 4A-1: Optimize alpha
    logger.info("[4A-1] Optimizing fusion weight (alpha)...")
    best_alpha = fusion.optimize_alpha(
        s_if_val, p_rf_val, y_val
    )

    # 4A-2: Train stacking meta-learner
    logger.info("\n[4A-2] Training stacking meta-learner...")
    fusion.train_stacking_meta_learner(
        s_if_val, p_rf_val, y_val
    )

    # 4A-3: Compare all fusion methods
    logger.info("\n[4A-3] Comparing fusion methods...")
    fusion_comparison = fusion.compare_fusion_methods(
        s_if_val, p_rf_val, y_val
    )

    # 4A-4: Select fusion method
    if cfg["fusion_method"] == "auto":
        selected_method = fusion.best_method
        logger.info(f"  Auto-selected method: {selected_method}")
    else:
        selected_method = cfg["fusion_method"]
        fusion.best_method = selected_method
        logger.info(f"  Manually selected method: {selected_method}")

    # 4A-5: Compute fused scores
    logger.info(f"\n[4A-5] Computing fused scores [{selected_method}]...")
    fused_val  = fusion.fuse(s_if_val,  p_rf_val,  method=selected_method)
    fused_test = fusion.fuse(s_if_test, p_rf_test, method=selected_method)

    logger.info(
        f"  Fused val  stats: "
        f"min={fused_val.min():.3f} | "
        f"max={fused_val.max():.3f} | "
        f"mean={fused_val.mean():.3f}"
    )
    logger.info(
        f"  Fused test stats: "
        f"min={fused_test.min():.3f} | "
        f"max={fused_test.max():.3f} | "
        f"mean={fused_test.mean():.3f}"
    )

    # ══════════════════════════════════════════════════════════════
    # 4B: Adaptive Threshold
    # ══════════════════════════════════════════════════════════════
    _banner("4B: Adaptive Threshold Calibration", char="─", width=50)

    threshold_engine = AdaptiveThresholdEngine()

    # 4B-1: Calibrate base threshold
    logger.info(
        f"[4B-1] Calibrating threshold "
        f"(target FPR ≤ {cfg['target_fpr']:.0%})..."
    )
    tau_calibrated = threshold_engine.calibrate_base_threshold(
        fused_val, y_val, target_fpr=cfg["target_fpr"]
    )
    logger.info(f"  Calibrated τ_base = {tau_calibrated:.4f}")

    # ══════════════════════════════════════════════════════════════
    # 4C: Alert Generation
    # ══════════════════════════════════════════════════════════════
    _banner("4C: Alert Generation", char="─", width=50)

    n_test          = len(fused_test)
    test_device_ids = list(p1["test_device_ids"][:n_test])
    attack_labels   = p3["attack_labels_test"][:n_test]

    # Simulate contextual factors
    rng = np.random.RandomState(42)
    if cfg["simulate_load"]:
        network_loads = rng.uniform(0.2, 0.9, n_test)
        hours         = rng.randint(0, 24, n_test)
        device_types  = rng.choice(
            ["sensor", "camera", "gateway", "actuator", "controller"],
            n_test,
            p=[0.4, 0.2, 0.2, 0.1, 0.1]
        )
    else:
        network_loads = np.full(n_test, 0.5)
        hours         = np.full(n_test, 12, dtype=int)
        device_types  = np.array(["sensor"] * n_test)

    logger.info(f"[4C] Processing {n_test:,} samples through alert engine...")

    alerts = threshold_engine.process_batch(
        fused_scores  = fused_test,
        s_if_scores   = s_if_test,
        p_rf_scores   = p_rf_test,
        device_ids    = test_device_ids,
        attack_labels = attack_labels,
        device_types  = list(device_types),
        network_load  = float(network_loads.mean()),
        hour          = int(hours.mean())
    )

    alert_summary = threshold_engine.get_alert_summary()
    logger.info(f"  Alert summary: {alert_summary}")

    # ══════════════════════════════════════════════════════════════
    # 4D: Final Fused System Evaluation
    # ══════════════════════════════════════════════════════════════
    _banner("4D: Fused System Evaluation", char="─", width=50)

    from sklearn.metrics import (
        f1_score, roc_auc_score,
        precision_score, recall_score,
        average_precision_score,
        confusion_matrix as sk_cm
    )

    y_test_bin   = y_test[:n_test].astype(int)
    fused_preds  = (fused_test >= threshold_engine.tau_base).astype(int)

    f1_fused     = f1_score(y_test_bin, fused_preds, zero_division=0)
    pre_fused    = precision_score(y_test_bin, fused_preds, zero_division=0)
    rec_fused    = recall_score(y_test_bin, fused_preds, zero_division=0)

    auc_fused    = 0.0
    ap_fused     = 0.0
    if len(np.unique(y_test_bin)) > 1:
        auc_fused = roc_auc_score(y_test_bin, fused_test)
        ap_fused  = average_precision_score(y_test_bin, fused_test)

    cm           = sk_cm(y_test_bin, fused_preds)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    fpr_fused    = fp / max(fp + tn, 1)
    dr_fused     = tp / max(tp + fn, 1)

    fused_metrics = {
        "f1_score":       round(float(f1_fused),  4),
        "roc_auc":        round(float(auc_fused), 4),
        "avg_precision":  round(float(ap_fused),  4),
        "precision":      round(float(pre_fused), 4),
        "recall":         round(float(rec_fused), 4),
        "detection_rate": round(float(dr_fused),  4),
        "fpr":            round(float(fpr_fused), 4),
        "best_alpha":     round(float(best_alpha),2),
        "best_method":    selected_method,
        "tau_base":       round(float(threshold_engine.tau_base), 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }

    logger.info("Fused system metrics:")
    for k, v in fused_metrics.items():
        logger.info(f"  {k:20s}: {v}")

    # ── Save Alerts ───────────────────────────────────────────────
    if cfg["save_reports"]:
        threshold_engine.save_alerts("phase4_alerts.json")

    p4 = {
        "fusion_engine":    fusion,
        "threshold_engine": threshold_engine,
        "fused_val":        fused_val,
        "fused_test":       fused_test,
        "fused_preds":      fused_preds,
        "alerts":           alerts,
        "alert_summary":    alert_summary,
        "fusion_comparison":fusion_comparison,
        "fused_metrics":    fused_metrics,
        "selected_method":  selected_method,
        "best_alpha":       best_alpha,
        "tau_base":         threshold_engine.tau_base,
    }

    return p4


# =============================================================================
# Final Report Printer
# =============================================================================
def _print_model_comparison(
    if_metrics: dict, rf_metrics: dict
) -> None:
    """Side-by-side model comparison table."""
    logger.info("\n┌─────────────────────────────────────────────────────┐")
    logger.info("│          MODEL COMPARISON (Test Set)                │")
    logger.info("├──────────────────────┬──────────────────┬───────────┤")
    logger.info("│ Metric               │ SGD One-Class SVM│ Rand Forest│")
    logger.info("├──────────────────────┼──────────────────┼───────────┤")

    metrics = ["roc_auc", "avg_precision", "f1_score",
               "precision", "recall", "fpr", "threshold"]
    for m in metrics:
        v_if = if_metrics.get(m, "N/A")
        v_rf = rf_metrics.get(m, "N/A")
        logger.info(f"│ {m:20s} │ {str(v_if):16s} │ {str(v_rf):9s} │")

    logger.info("└──────────────────────┴──────────────────┴───────────┘")


def _print_final_report(
    p1: dict, p2: dict, p3: dict, p4: dict
) -> None:
    """Print comprehensive final pipeline results."""

    logger.info("\n")
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║         HYBRID IoT IDS — FINAL PIPELINE RESULTS     ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    # ── Phase 1 ──────────────────────────────────────────────────
    logger.info("\n── Phase 1: Data ─────────────────────────────────────")
    logger.info(f"  Source         : {p1['dataset_source']}")
    logger.info(f"  Total samples  : {p1['n_total']:,}")
    logger.info(f"  Features       : {p1['n_features']}")
    logger.info(f"  Attack ratio   : {p1['attack_pct']:.2f}%")
    logger.info(f"  Train / Val / Test: "
                f"{p1['n_train']:,} / {p1['n_val']:,} / {p1['n_test']:,}")

    # ── Phase 2 ──────────────────────────────────────────────────
    logger.info("\n── Phase 2: Profiling ────────────────────────────────")
    logger.info(f"  Device profiles  : {p2['n_profiles']}")
    logger.info(f"  Warmed profiles  : {p2['n_warmed']}")
    logger.info(f"  Augmented dims   : {p2['n_aug_features']}")

    # ── Phase 3 ──────────────────────────────────────────────────
    logger.info("\n── Phase 3: Models ───────────────────────────────────")

    it = p3["if_metrics_test"]
    rt = p3["rf_metrics_test"]

    logger.info("  ┌──────────────────────┬──────────┬──────────┐")
    logger.info("  │ Metric               │    IF    │    RF    │")
    logger.info("  ├──────────────────────┼──────────┼──────────┤")

    for m in ["roc_auc", "avg_precision", "f1_score",
              "precision", "recall", "detection_rate", "fpr", "threshold"]:
        vi = str(it.get(m, "─"))
        vr = str(rt.get(m, "─"))
        logger.info(f"  │ {m:20s} │ {vi:8s} │ {vr:8s} │")

    logger.info("  └──────────────────────┴──────────┴──────────┘")

    # ── Phase 4 ──────────────────────────────────────────────────
    logger.info("\n── Phase 4: Fusion & Threshold ───────────────────────")
    fm = p4["fused_metrics"]
    al = p4["alert_summary"]

    logger.info(f"  Fusion method    : {fm['best_method']}")
    logger.info(f"  Best alpha       : {fm['best_alpha']}")
    logger.info(f"  Calibrated τ     : {fm['tau_base']}")
    logger.info("")
    logger.info("  ┌──────────────────────┬──────────┐")
    logger.info("  │ Fused Metric         │  Value   │")
    logger.info("  ├──────────────────────┼──────────┤")

    for m in ["f1_score", "roc_auc", "avg_precision",
              "precision", "recall", "detection_rate", "fpr"]:
        v = str(fm.get(m, "─"))
        # Flag if meets targets
        target_met = _check_target(m, fm.get(m, 0))
        flag = " ✓" if target_met else " ✗"
        logger.info(f"  │ {m:20s} │ {v:6s}   │{flag}")

    logger.info("  └──────────────────────┴──────────┘")

    logger.info("\n  ┌──────────────────┬────────┐")
    logger.info("  │ Alert Type       │ Count  │")
    logger.info("  ├──────────────────┼────────┤")
    logger.info(f"  │ Total alerts     │ {al.get('total_alerts', 0):6,} │")
    logger.info(f"  │ Critical         │ {al.get('critical_count', 0):6,} │")
    logger.info(f"  │ Warning          │ {al.get('warning_count', 0):6,} │")
    logger.info("  └──────────────────┴────────┘")

    # ── Confusion Matrix ──────────────────────────────────────────
    logger.info("\n  Fused System Confusion Matrix:")
    logger.info(f"    TP={fm.get('tp',0):,}  FP={fm.get('fp',0):,}")
    logger.info(f"    FN={fm.get('fn',0):,}  TN={fm.get('tn',0):,}")

    logger.info("\n╔══════════════════════════════════════════════════════╗")
    logger.info("║               PIPELINE COMPLETE ✓                   ║")
    logger.info("╚══════════════════════════════════════════════════════╝\n")


def _check_target(metric: str, value: float) -> bool:
    """Check if metric meets the target threshold from the plan."""
    targets = {
        "roc_auc":        0.97,
        "avg_precision":  0.93,
        "f1_score":       0.92,
        "precision":      0.90,
        "recall":         0.95,
        "detection_rate": 0.95,
        "fpr":            None,         # lower is better
    }
    if metric == "fpr":
        return float(value) <= 0.05
    t = targets.get(metric)
    if t is None:
        return True
    return float(value) >= t


# =============================================================================
# Main Entry Point
# =============================================================================
def run_pipeline(cfg: dict = None) -> dict:
    """
    Execute the complete Phase 1 → 2 → 3 → 4 pipeline.

    Args:
        cfg: Pipeline configuration dict.
             Defaults to PIPELINE_CONFIG if None.

    Returns:
        results: Nested dict with all phase outputs and metrics.
    """
    if cfg is None:
        cfg = PIPELINE_CONFIG

    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║   HYBRID IoT IDS — PIPELINE STARTING                ║")
    logger.info("╠══════════════════════════════════════════════════════╣")
    logger.info("║   Phases: Data → Profiling → Models → Fusion        ║")
    logger.info("╚══════════════════════════════════════════════════════╝")
    logger.info(f"\nConfiguration:")
    for k, v in cfg.items():
        logger.info(f"  {k:25s}: {v}")

    loader  = IoTDataLoader()
    results = {}

    # ── Phase 1 ──────────────────────────────────────────────────
    p1 = run_phase1(cfg, loader)
    results["phase1"] = {
        k: v for k, v in p1.items()
        if not isinstance(v, (np.ndarray, pd.DataFrame))
    }

    # ── Phase 2 ──────────────────────────────────────────────────
    p2 = run_phase2(cfg, p1)
    results["phase2"] = {
        k: v for k, v in p2.items()
        if not isinstance(v, (np.ndarray, BehavioralProfiler))
    }

    # ── Phase 3 ──────────────────────────────────────────────────
    p3 = run_phase3(cfg, p1, p2)
    results["phase3"] = {
        "if_metrics_val":  p3["if_metrics_val"],
        "if_metrics_test": p3["if_metrics_test"],
        "rf_metrics_val":  p3["rf_metrics_val"],
        "rf_metrics_test": p3["rf_metrics_test"],
        "if_score_stats":  p3["if_score_stats"],
        "rf_score_stats":  p3["rf_score_stats"],
    }

    # ── Phase 4 ──────────────────────────────────────────────────
    p4 = run_phase4(cfg, p1, p2, p3)
    results["phase4"] = {
        "fused_metrics":    p4["fused_metrics"],
        "alert_summary":    p4["alert_summary"],
        "fusion_comparison":p4["fusion_comparison"],
        "selected_method":  p4["selected_method"],
        "best_alpha":       p4["best_alpha"],
        "tau_base":         p4["tau_base"],
    }

    # ── Final Report ──────────────────────────────────────────────
    _print_final_report(p1, p2, p3, p4)

    # ── Save JSON Report ──────────────────────────────────────────
    if cfg["save_reports"]:
        _save_report(
            _to_serializable(results),
            "pipeline_results.json"
        )

    return results


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid IoT IDS Pipeline"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-download dataset (ignore cache)"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic dataset only"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run full hyperparameter tuning (slower)"
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default="auto",
        choices=["auto", "weighted_linear", "max", "product", "stacking"],
        help="Fusion method to use"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save models or reports"
    )
    args = parser.parse_args()

    # Override config with CLI arguments
    run_cfg = PIPELINE_CONFIG.copy()
    if args.no_cache:
        run_cfg["use_cache"]       = False
    if args.synthetic:
        run_cfg["force_synthetic"] = True
    if args.tune:
        run_cfg["tune_if"]         = True
        run_cfg["tune_rf"]         = True
    if args.fusion != "auto":
        run_cfg["fusion_method"]   = args.fusion
    if args.no_save:
        run_cfg["save_models"]     = False
        run_cfg["save_reports"]    = False

    final_results = run_pipeline(cfg=run_cfg)