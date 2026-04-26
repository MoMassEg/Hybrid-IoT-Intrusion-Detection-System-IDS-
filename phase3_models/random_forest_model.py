# =============================================================================
# Phase 3: Random Forest Classifier - FIXED VERSION
# Fix: Removed double transformation of X_val
# =============================================================================

import os
import numpy as np
import joblib
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, f1_score
)
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import RANDOM_FOREST_CONFIG, MODEL_DIR, TRAIN_CONFIG
from utils.logger import get_logger

logger = get_logger("RandomForest")


class RandomForestModel:
    """
    Improved Random Forest with:
    - Probability calibration
    - Feature selection
    - Correct binary probability extraction
    - Cross-validated threshold tuning
    """

    def __init__(self, config: dict = None):
        self.config  = config or RANDOM_FOREST_CONFIG
        self.model:  Optional[RandomForestClassifier] = None
        self.calibrated_model = None
        self.selector = None
        self.classes_: List = []
        self.feature_names: List[str] = []
        self.selected_feature_names: List[str] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.normal_class_index: Optional[int] = None
        self.optimal_threshold: float = 0.5
        self.is_trained = False

    # ──────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray = None,
        y_val:   np.ndarray = None,
        feature_names: List[str] = None,
        use_feature_selection: bool = True,
        calibrate_proba: bool = True,
        tune_threshold: bool = True
    ) -> "RandomForestModel":
        """
        Full training pipeline.
        
        CRITICAL: Does NOT pre-transform X_val in Step 2.
        All transformations handled by prediction methods.
        """
        self.feature_names = feature_names or [
            f"f{i}" for i in range(X_train.shape[1])
        ]

        logger.info("=" * 55)
        logger.info("TRAINING: Random Forest (Improved)")
        logger.info(f"  Input        : {X_train.shape}")
        logger.info(f"  Classes      : {np.unique(y_train)}")
        logger.info(f"  Class counts : {dict(zip(*np.unique(y_train, return_counts=True)))}")
        logger.info("=" * 55)

        # ── Step 1: Initial fit ───────────────────────────────────
        logger.info("[RF Step 1] Initial training...")
        self.model = RandomForestClassifier(**self.config)
        self.model.fit(X_train, y_train)
        self.classes_              = list(self.model.classes_)
        self.feature_importances_  = self.model.feature_importances_
        self.normal_class_index    = self._find_normal_class_index()
        
        # Set trained flag early
        self.is_trained = True

        logger.info(f"  Classes found      : {self.classes_}")
        logger.info(f"  Normal class index : {self.normal_class_index}")

        # ── Step 2: Feature Selection ─────────────────────────────
        X_train_selected = X_train
        if use_feature_selection:
            X_train_selected = self._select_features(X_train, y_train)
            # ✅ Do NOT transform X_val here!

        # ── Step 3: Retrain on selected features ──────────────────
        if use_feature_selection and self.selector is not None:
            logger.info("[RF Step 3] Retraining on selected features...")
            self.model = RandomForestClassifier(**self.config)
            self.model.fit(X_train_selected, y_train)
            self.classes_             = list(self.model.classes_)
            self.feature_importances_ = self.model.feature_importances_
            self.normal_class_index   = self._find_normal_class_index()

        # ── Step 4: Probability Calibration ──────────────────────
        if calibrate_proba:
            logger.info("[RF Step 4] Calibrating probabilities (Platt)...")
            try:
                cv_calib = min(3, min(np.bincount(y_train.astype(int))) // 2)
                cv_calib = max(cv_calib, 2)
                
                base_estimator = RandomForestClassifier(**self.config)
                self.calibrated_model = CalibratedClassifierCV(
                    base_estimator,
                    method="sigmoid",
                    cv=cv_calib
                )
                self.calibrated_model.fit(X_train_selected, y_train)
                logger.info("  Probability calibration complete")
            except Exception as e:
                logger.warning(f"  Calibration failed ({e}) — using raw RF")
                self.calibrated_model = None

        # ── Step 5: Cross-Validation ──────────────────────────────
        self._run_cross_validation(X_train_selected, y_train)

        # ── Step 6: Threshold Tuning ──────────────────────────────
        if tune_threshold and X_val is not None and y_val is not None:
            self._tune_threshold(X_val, y_val)  # Pass ORIGINAL X_val

        self._log_feature_importance(top_n=10)
        logger.info("Random Forest training complete.")
        return self

    def _select_features(
        self, X: np.ndarray, y: np.ndarray, threshold: str = "mean"
    ) -> np.ndarray:
        """Remove features below mean importance."""
        logger.info("[RF Step 2] Feature selection...")
        self.selector = SelectFromModel(
            self.model, threshold=threshold, prefit=True
        )
        X_sel = self.selector.transform(X)

        if X_sel.shape[1] < 5:
            logger.warning(
                f"  Too few features selected ({X_sel.shape[1]}) "
                f"→ disabling selection"
            )
            self.selector = None
            self.selected_feature_names = self.feature_names.copy()
            return X

        mask = self.selector.get_support()
        self.selected_feature_names = [
            n for n, m in zip(self.feature_names, mask) if m
        ]
        logger.info(
            f"  Features: {len(self.feature_names)} → {X_sel.shape[1]} selected"
        )
        return X_sel

    def _run_cross_validation(self, X: np.ndarray, y: np.ndarray) -> None:
        """5-fold stratified cross-validation."""
        logger.info("[RF Step CV] Cross-validation...")
        n_splits = min(TRAIN_CONFIG["cv_folds"], min(np.bincount(y.astype(int))))
        n_splits = max(n_splits, 2)

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(
            RandomForestClassifier(**self.config),
            X, y, cv=cv, scoring="f1_weighted", n_jobs=-1
        )
        logger.info(
            f"  CV F1 (weighted): {scores.mean():.4f} ± {scores.std():.4f}"
        )

    def _tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Find threshold that maximises F1 on validation set."""
        logger.info("[RF Step 5] Threshold tuning on validation set...")

        y_bin = self._to_binary(y_val)
        attack_proba = self.predict_proba_attack(X_val)  # Handles transform internally

        best_t  = 0.5
        best_f1 = -1.0
        results = []

        for t in np.arange(0.05, 0.96, 0.05):
            preds = (attack_proba >= t).astype(int)
            f1    = f1_score(y_bin, preds, zero_division=0)
            results.append((round(t, 2), round(f1, 4)))
            if f1 > best_f1:
                best_f1 = f1
                best_t  = t

        self.optimal_threshold = round(float(best_t), 2)
        logger.info(
            f"  Optimal threshold : {self.optimal_threshold:.2f} → F1={best_f1:.4f}"
        )

        top5 = sorted(results, key=lambda x: -x[1])[:5]
        logger.info("  Top-5 thresholds:")
        for t, f in top5:
            marker = " ← BEST" if abs(t - self.optimal_threshold) < 0.01 else ""
            logger.info(f"    τ={t:.2f} → F1={f:.4f}{marker}")

    # ──────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Apply feature selector if fitted."""
        if self.selector is not None:
            return self.selector.transform(X)
        return X

    def predict_proba_attack(self, X: np.ndarray) -> np.ndarray:
        """P(attack | x) ∈ [0, 1]."""
        self._assert_trained()
        X_sel = self.transform_features(X)  # Transforms here
        model = self.calibrated_model if self.calibrated_model else self.model
        proba = model.predict_proba(X_sel)

        if self.normal_class_index is not None:
            attack_proba = 1.0 - proba[:, self.normal_class_index]
        else:
            attack_proba = 1.0 - proba[:, 0]
        return np.clip(attack_proba, 0.0, 1.0)

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self._assert_trained()
        X_sel = self.transform_features(X)
        model = self.calibrated_model if self.calibrated_model else self.model
        return model.predict(X_sel)

    def predict_binary(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """Binary prediction using tuned threshold."""
        t = threshold if threshold is not None else self.optimal_threshold
        return (self.predict_proba_attack(X) >= t).astype(int)

    def predict_full(self, X: np.ndarray) -> Dict:
        """Full prediction output."""
        self._assert_trained()
        labels       = self.predict_labels(X)
        attack_proba = self.predict_proba_attack(X)
        X_sel        = self.transform_features(X)
        model        = self.calibrated_model if self.calibrated_model else self.model
        proba_matrix = model.predict_proba(X_sel)

        return {
            "labels":       labels,
            "attack_proba": attack_proba,
            "proba_matrix": proba_matrix,
            "class_names":  self.classes_,
            "threshold":    self.optimal_threshold,
        }

    # ──────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────
    def evaluate(
        self, X: np.ndarray, y_true: np.ndarray, dataset_name: str = "Test"
    ) -> Dict:
        """Comprehensive evaluation."""
        self._assert_trained()

        y_bin    = self._to_binary(y_true)
        y_pred   = self.predict_binary(X)
        atk_prob = self.predict_proba_attack(X)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_bin, y_pred, average="binary", zero_division=0
        )

        cm = confusion_matrix(y_bin, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        fpr = fp / max(fp + tn, 1)
        dr  = tp / max(tp + fn, 1)

        roc_auc = ap = 0.0
        if len(np.unique(y_bin)) > 1:
            roc_auc = roc_auc_score(y_bin, atk_prob)
            ap      = average_precision_score(y_bin, atk_prob)

        metrics = {
            "model":          "RandomForest",
            "dataset":        dataset_name,
            "threshold":      self.optimal_threshold,
            "roc_auc":        round(roc_auc, 4),
            "avg_precision":  round(ap, 4),
            "precision":      round(precision, 4),
            "recall":         round(recall, 4),
            "f1_score":       round(f1, 4),
            "detection_rate": round(dr, 4),
            "fpr":            round(fpr, 4),
            "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn),
        }

        logger.info(f"── RF Evaluation [{dataset_name}] ──")
        for k, v in metrics.items():
            if k not in ["model", "dataset"]:
                logger.info(f"  {k:20s}: {v}")

        y_pred_labels = self.predict_labels(X)
        logger.info(
            f"\n{classification_report(y_true, y_pred_labels, zero_division=0)}"
        )
        return metrics

    # ──────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────
    def save(self, version: str = "v1") -> str:
        self._assert_trained()
        path = os.path.join(MODEL_DIR, f"random_forest_{version}.joblib")
        joblib.dump({
            "model":                  self.model,
            "calibrated_model":       self.calibrated_model,
            "selector":               self.selector,
            "classes_":               self.classes_,
            "feature_names":          self.feature_names,
            "selected_feature_names": self.selected_feature_names,
            "feature_importances_":   self.feature_importances_,
            "normal_class_index":     self.normal_class_index,
            "optimal_threshold":      self.optimal_threshold,
            "config":                 self.config,
        }, path)
        logger.info(f"RF model saved: {path}")
        return path

    def load(self, version: str = "v1") -> "RandomForestModel":
        path = os.path.join(MODEL_DIR, f"random_forest_{version}.joblib")
        data = joblib.load(path)
        self.model                  = data["model"]
        self.calibrated_model       = data.get("calibrated_model")
        self.selector               = data.get("selector")
        self.classes_               = data["classes_"]
        self.feature_names          = data["feature_names"]
        self.selected_feature_names = data.get("selected_feature_names", [])
        self.feature_importances_   = data["feature_importances_"]
        self.normal_class_index     = data.get("normal_class_index")
        self.optimal_threshold      = data.get("optimal_threshold", 0.5)
        self.config                 = data["config"]
        self.is_trained             = True
        logger.info(f"RF model loaded: {path}")
        return self

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────
    def _find_normal_class_index(self) -> Optional[int]:
        """Find index of normal/benign class."""
        normal_labels = {
            0, "0", 0.0, "Normal", "normal", "NORMAL",
            "Benign", "benign", "BENIGN", "legitimate", "safe"
        }
        for i, cls in enumerate(self.classes_):
            if cls in normal_labels:
                logger.info(f"  Normal class found: '{cls}' at index {i}")
                return i
        if len(self.classes_) > 0:
            logger.warning(
                f"  Normal class not found. Assuming index 0 = '{self.classes_[0]}'"
            )
            return 0
        return None

    def _to_binary(self, y: np.ndarray) -> np.ndarray:
        """Convert to binary (0=normal, 1=attack)."""
        unique = np.unique(y)
        if set(unique).issubset({0, 1}):
            return y.astype(int)
        if np.issubdtype(y.dtype, np.integer):
            return (y != 0).astype(int)
        normal_set = {"0", "Normal", "normal", "NORMAL", "Benign", "benign", "BENIGN"}
        return np.array([0 if str(v) in normal_set else 1 for v in y], dtype=int)

    def _log_feature_importance(self, top_n: int = 10) -> None:
        if self.feature_importances_ is None:
            return
        names   = self.selected_feature_names or self.feature_names
        indices = np.argsort(self.feature_importances_)[::-1]
        logger.info(f"Top-{top_n} Feature Importances:")
        for rank, idx in enumerate(indices[:top_n], 1):
            name = names[idx] if idx < len(names) else f"f{idx}"
            logger.info(
                f"  {rank:2d}. {name:30s} {self.feature_importances_[idx]:.4f}"
            )

    def _assert_trained(self):
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call .train() first.")