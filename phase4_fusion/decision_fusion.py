# =============================================================================
# Phase 4: Decision Fusion Engine
# =============================================================================

import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import FUSION_CONFIG
from utils.logger import get_logger

logger = get_logger("DecisionFusion")


class DecisionFusionEngine:
    """
    Combines Isolation Forest anomaly scores and Random Forest
    attack probabilities into a single fused detection score.
    
    Input:
        sIF  ∈ [0, 1]  — Isolation Forest anomaly score
        PRF  ∈ [0, 1]  — Random Forest attack probability
    
    Output:
        Sfused ∈ [0, 1] — Unified detection score
    """

    METHODS = ["weighted_linear", "max", "product", "stacking"]

    def __init__(self, config: dict = None):
        self.config = config or FUSION_CONFIG
        self.alpha = self.config.get("alpha", 0.5)
        self.method = self.config.get("fusion_method", "weighted_linear")
        self.meta_learner: Optional[LogisticRegression] = None
        self.best_method = self.method
        self.best_alpha = self.alpha
        logger.info(
            f"FusionEngine initialized: method={self.method}, alpha={self.alpha}"
        )

    # ──────────────────────────────────────────────────────────────
    # Core Fusion
    # ──────────────────────────────────────────────────────────────
    def fuse(
        self,
        s_if: np.ndarray,
        p_rf: np.ndarray,
        method: str = None,
        alpha: float = None
    ) -> np.ndarray:
        """
        Fuse two score arrays into a single detection score.
        
        Args:
            s_if:   Isolation Forest scores ∈ [0, 1]
            p_rf:   Random Forest attack probabilities ∈ [0, 1]
            method: Fusion strategy (overrides config if provided)
            alpha:  Fusion weight (overrides config if provided)
        
        Returns:
            fused_scores: np.ndarray ∈ [0, 1]
        """
        s_if = np.clip(np.asarray(s_if, dtype=float), 0, 1)
        p_rf = np.clip(np.asarray(p_rf, dtype=float), 0, 1)
        method = method or self.best_method
        alpha  = alpha  if alpha is not None else self.best_alpha

        if method == "weighted_linear":
            fused = alpha * s_if + (1 - alpha) * p_rf

        elif method == "max":
            fused = np.maximum(s_if, p_rf)

        elif method == "product":
            fused = s_if * p_rf

        elif method == "stacking":
            if self.meta_learner is None:
                logger.warning(
                    "Meta-learner not trained. Falling back to weighted_linear."
                )
                fused = alpha * s_if + (1 - alpha) * p_rf
            else:
                stack_input = np.column_stack([s_if, p_rf])
                fused = self.meta_learner.predict_proba(stack_input)[:, 1]

        else:
            raise ValueError(
                f"Unknown fusion method: {method}. "
                f"Choose from {self.METHODS}"
            )

        return np.clip(fused, 0, 1)

    # ──────────────────────────────────────────────────────────────
    # Optimization
    # ──────────────────────────────────────────────────────────────
    def optimize_alpha(
        self,
        s_if_val: np.ndarray,
        p_rf_val: np.ndarray,
        y_val: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """
        Grid-search optimal alpha on validation set using F1-score.
        
        Returns:
            best_alpha: Optimal fusion weight
        """
        logger.info("Optimizing fusion weight alpha...")
        best_alpha = 0.5
        best_f1 = -1.0
        results = []

        for alpha in np.arange(0.0, 1.05, 0.05):
            fused = self.fuse(s_if_val, p_rf_val, method="weighted_linear",
                              alpha=alpha)
            y_pred = (fused >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            results.append((round(alpha, 2), round(f1, 4)))

            if f1 > best_f1:
                best_f1 = f1
                best_alpha = alpha

        self.best_alpha = round(best_alpha, 2)
        logger.info(
            f"Best alpha = {self.best_alpha:.2f} → F1 = {best_f1:.4f}"
        )
        logger.info("Alpha search results (alpha → F1):")
        for a, f in results:
            marker = " ← BEST" if abs(a - self.best_alpha) < 0.01 else ""
            logger.info(f"  alpha={a:.2f} → F1={f:.4f}{marker}")
        return self.best_alpha

    def compare_fusion_methods(
        self,
        s_if_val: np.ndarray,
        p_rf_val: np.ndarray,
        y_val: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """
        Evaluate all fusion methods on validation data.
        Selects the best method automatically.
        
        Returns:
            results: Dict of method → metrics
        """
        logger.info("Comparing all fusion methods...")
        results = {}

        for method in self.METHODS:
            try:
                if method == "stacking" and self.meta_learner is None:
                    logger.info("  Skipping stacking (not trained yet)")
                    continue

                fused = self.fuse(
                    s_if_val, p_rf_val,
                    method=method, alpha=self.best_alpha
                )
                y_pred = (fused >= threshold).astype(int)
                f1 = f1_score(y_val, y_pred, zero_division=0)

                auc = 0.0
                if len(np.unique(y_val)) > 1:
                    auc = roc_auc_score(y_val, fused)

                fp_rate = (
                    np.sum((y_pred == 1) & (y_val == 0)) /
                    max(np.sum(y_val == 0), 1)
                )
                results[method] = {
                    "f1": round(f1, 4),
                    "roc_auc": round(auc, 4),
                    "fpr": round(fp_rate, 4)
                }
                logger.info(
                    f"  {method:20s} → F1={f1:.4f} | "
                    f"AUC={auc:.4f} | FPR={fp_rate:.4f}"
                )
            except Exception as e:
                logger.warning(f"  {method}: failed → {e}")

        # Select best method by F1
        if results:
            self.best_method = max(results, key=lambda m: results[m]["f1"])
            logger.info(f"Selected best fusion method: {self.best_method}")

        return results

    def train_stacking_meta_learner(
        self,
        s_if_train: np.ndarray,
        p_rf_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:
        """Train logistic regression meta-learner for stacking fusion."""
        logger.info("Training stacking meta-learner (Logistic Regression)...")
        X_stack = np.column_stack([s_if_train, p_rf_train])
        self.meta_learner = LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=500
        )
        self.meta_learner.fit(X_stack, y_train)
        logger.info("Meta-learner training complete.")
