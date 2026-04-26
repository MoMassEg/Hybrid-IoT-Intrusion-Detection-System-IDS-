# =============================================================================
# Phase 3: Isolation Forest - IMPROVED VERSION
# =============================================================================

import os
import numpy as np
import joblib
from typing import Dict, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix, f1_score
)
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import ISOLATION_FOREST_CONFIG, MODEL_DIR
from utils.logger import get_logger

logger = get_logger("IsolationForest")


class IsolationForestModel:
    """
    Improved Isolation Forest with:
    - Auto contamination estimation from data
    - Precision-Recall based threshold tuning
    - Better score normalization
    - Optimal decision threshold on validation set
    """

    def __init__(self, config: dict = None):
        self.config       = config or ISOLATION_FOREST_CONFIG.copy()
        self.model:       Optional[IsolationForest] = None
        self.score_scaler = MinMaxScaler(feature_range=(0, 1))
        self.best_params: Dict = {}
        self.optimal_threshold: float = 0.5
        self.is_trained   = False
        self.feature_names: list = []

    # ──────────────────────────────────────────────────────────────
    # Contamination Estimation
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def estimate_contamination(
        y_train: np.ndarray,
        min_val: float = 0.01,
        max_val: float = 0.45
    ) -> float:
        """
        Estimate contamination from known label ratio.
        Clips to [min_val, max_val] for IF stability.
        """
        attack_ratio = float((y_train > 0).mean())
        contamination = float(np.clip(attack_ratio, min_val, max_val))
        logger.info(
            f"  Estimated contamination: {contamination:.4f} "
            f"(attack ratio={attack_ratio:.4f})"
        )
        return contamination

    # ──────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────
    def train(
        self,
        X_normal: np.ndarray,
        feature_names: list = None,
        y_all: np.ndarray = None,        # full y for contamination estimate
        X_val: np.ndarray = None,        # for threshold tuning
        y_val: np.ndarray = None,
    ) -> "IsolationForestModel":
        """
        Train Isolation Forest.

        Steps:
            1. Auto-estimate contamination from y_all (if provided)
            2. Fit on X_normal (benign samples only)
            3. Fit score scaler on training scores
            4. Tune decision threshold on validation set (if provided)

        Args:
            X_normal:     Benign-only training data
            feature_names: Feature names for logging
            y_all:        Full label array for contamination estimation
            X_val:        Mixed validation set for threshold tuning
            y_val:        Validation labels (binary)
        """
        self.feature_names = feature_names or [
            f"f{i}" for i in range(X_normal.shape[1])
        ]

        logger.info("=" * 55)
        logger.info("TRAINING: Isolation Forest (Improved)")
        logger.info(f"  Normal samples : {X_normal.shape[0]:,}")
        logger.info(f"  Features       : {X_normal.shape[1]}")
        logger.info("=" * 55)

        # ── Step 1: Contamination ────────────────────────────────
        if y_all is not None:
            self.config["contamination"] = self.estimate_contamination(y_all)
        logger.info(f"  Contamination  : {self.config['contamination']}")

        # ── Step 2: Fit ──────────────────────────────────────────
        logger.info("[IF] Fitting Isolation Forest...")
        self.model = IsolationForest(**self.config)
        self.model.fit(X_normal)

        # ── Step 3: Score Scaler ─────────────────────────────────
        train_raw = self.model.decision_function(X_normal)
        # Negate: higher raw = more normal → negate for anomaly direction
        train_anomaly = -train_raw
        self.score_scaler.fit(train_anomaly.reshape(-1, 1))

        self.is_trained  = True
        self.best_params = self.config.copy()

        # ── Step 4: Threshold Tuning ─────────────────────────────
        if X_val is not None and y_val is not None:
            self._tune_threshold(X_val, y_val)
        else:
            logger.info(
                "  No validation set → using default threshold=0.5"
            )

        logger.info("Isolation Forest training complete.")
        return self

    def _tune_threshold(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> None:
        """
        Find threshold maximising F1 on validation set.
        Uses F1 + 0.5×precision to prefer fewer false positives.
        """
        logger.info("[IF] Threshold tuning on validation set...")
        y_bin   = (y_val > 0).astype(int)
        scores  = self.predict_scores(X_val)

        best_t    = 0.5
        best_score = -1.0
        results   = []

        for t in np.arange(0.05, 0.96, 0.05):
            preds = (scores >= t).astype(int)
            f1    = f1_score(y_bin, preds, zero_division=0)

            # Penalise high FPR
            fp_count = int(((preds == 1) & (y_bin == 0)).sum())
            tn_count = int(((preds == 0) & (y_bin == 0)).sum())
            fpr      = fp_count / max(fp_count + tn_count, 1)

            # Combined score: maximise F1, penalise FPR
            combined = f1 - 0.3 * fpr

            results.append((round(t, 2), round(f1, 4), round(fpr, 4)))
            if combined > best_score:
                best_score = combined
                best_t     = t

        self.optimal_threshold = round(float(best_t), 2)
        logger.info(
            f"  Optimal threshold: {self.optimal_threshold:.2f} "
            f"(score={best_score:.4f})"
        )

        top5 = sorted(results, key=lambda x: -(x[1] - 0.3 * x[2]))[:5]
        logger.info("  Top-5 thresholds (τ → F1, FPR):")
        for t, f1, fpr in top5:
            marker = " ← BEST" if abs(t - self.optimal_threshold) < 0.01 else ""
            logger.info(f"    τ={t:.2f} → F1={f1:.4f} | FPR={fpr:.4f}{marker}")

    # ──────────────────────────────────────────────────────────────
    # Hyperparameter Tuning
    # ──────────────────────────────────────────────────────────────
    def tune_hyperparameters(
        self,
        X_normal: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list = None
    ) -> Dict:
        """
        Grid search over IF hyperparameters.
        Uses AUC + Average Precision as joint criterion.
        """
        param_grid = {
            "n_estimators":  [100, 200, 300],
            "contamination": [0.05, 0.10, 0.15, 0.20, 0.25],
            "max_samples":   ["auto", 128, 256],
            "max_features":  [0.7, 1.0],
        }

        y_bin = (y_val > 0).astype(int)

        # Flatten grid
        from itertools import product as iterproduct
        keys   = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(iterproduct(*values))

        logger.info(
            f"IF Hyperparameter tuning: {len(combos)} combinations..."
        )

        best_score = -np.inf
        best_cfg   = self.config.copy()

        for i, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            cfg    = {**self.config, **params}
            try:
                m = IsolationForest(**cfg)
                m.fit(X_normal)
                raw    = m.decision_function(X_val)
                scores = -raw

                auc = roc_auc_score(y_bin, scores) \
                      if len(np.unique(y_bin)) > 1 else 0.5
                ap  = average_precision_score(y_bin, scores) \
                      if len(np.unique(y_bin)) > 1 else 0.5

                combined = 0.6 * auc + 0.4 * ap
                if combined > best_score:
                    best_score = combined
                    best_cfg   = cfg.copy()

                if (i + 1) % 15 == 0:
                    logger.info(
                        f"  [{i+1:3d}/{len(combos)}] "
                        f"n_est={params['n_estimators']} "
                        f"cont={params['contamination']:.2f} "
                        f"AUC={auc:.4f} AP={ap:.4f}"
                    )
            except Exception as e:
                logger.debug(f"  Combo failed: {params} → {e}")

        logger.info(f"Best IF config: {best_cfg}")
        logger.info(f"Best score    : {best_score:.4f}")
        self.config = best_cfg
        return best_cfg

    # ──────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Normalized anomaly scores ∈ [0, 1].
        Higher = more anomalous.
        """
        self._assert_trained()
        raw    = self.model.decision_function(X)
        anom   = -raw
        scores = self.score_scaler.transform(
            anom.reshape(-1, 1)
        ).flatten()
        return np.clip(scores, 0.0, 1.0)

    def predict_binary(
        self, X: np.ndarray, threshold: float = None
    ) -> np.ndarray:
        """Binary prediction using tuned threshold."""
        t = threshold if threshold is not None else self.optimal_threshold
        return (self.predict_scores(X) >= t).astype(int)

    # ──────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────
    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        dataset_name: str = "Test"
    ) -> Dict:
        """Evaluate using tuned threshold."""
        self._assert_trained()

        y_bin   = (y_true > 0).astype(int)
        scores  = self.predict_scores(X)
        y_pred  = self.predict_binary(X)

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
            roc_auc = roc_auc_score(y_bin, scores)
            ap      = average_precision_score(y_bin, scores)

        metrics = {
            "model":          "IsolationForest",
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

        logger.info(f"── IF Evaluation [{dataset_name}] ──")
        for k, v in metrics.items():
            if k not in ["model", "dataset"]:
                logger.info(f"  {k:20s}: {v}")
        return metrics

    # ──────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────
    def save(self, version: str = "v1") -> str:
        self._assert_trained()
        path = os.path.join(MODEL_DIR, f"isolation_forest_{version}.joblib")
        joblib.dump({
            "model":             self.model,
            "score_scaler":      self.score_scaler,
            "config":            self.config,
            "best_params":       self.best_params,
            "feature_names":     self.feature_names,
            "optimal_threshold": self.optimal_threshold,
        }, path)
        logger.info(f"IF model saved: {path}")
        return path

    def load(self, version: str = "v1") -> "IsolationForestModel":
        path = os.path.join(MODEL_DIR, f"isolation_forest_{version}.joblib")
        data = joblib.load(path)
        self.model             = data["model"]
        self.score_scaler      = data["score_scaler"]
        self.config            = data["config"]
        self.best_params       = data["best_params"]
        self.feature_names     = data.get("feature_names", [])
        self.optimal_threshold = data.get("optimal_threshold", 0.5)
        self.is_trained        = True
        logger.info(f"IF model loaded: {path}")
        return self

    def _assert_trained(self):
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call .train() first.")