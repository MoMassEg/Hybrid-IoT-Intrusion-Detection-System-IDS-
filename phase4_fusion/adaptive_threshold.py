# =============================================================================
# Phase 4: Adaptive Threshold Engine
# =============================================================================

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import json
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import THRESHOLD_CONFIG, REPORT_DIR
from utils.logger import get_logger

logger = get_logger("AdaptiveThreshold")


# =============================================================================
# Alert Data Structure
# =============================================================================
@dataclass
class AlertObject:
    """Structured alert output from Phase 4."""
    alert_id:       str
    device_id:      str
    timestamp:      str
    fused_score:    float
    threshold_used: float
    alert_level:    str          # NONE / WARNING / CRITICAL
    attack_type:    str
    s_if_score:     float
    p_rf_score:     float
    context:        Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "alert_id":       self.alert_id,
            "device_id":      self.device_id,
            "timestamp":      self.timestamp,
            "fused_score":    round(self.fused_score, 4),
            "threshold_used": round(self.threshold_used, 4),
            "alert_level":    self.alert_level,
            "attack_type":    self.attack_type,
            "s_if_score":     round(self.s_if_score, 4),
            "p_rf_score":     round(self.p_rf_score, 4),
            "context":        self.context
        }

    def __str__(self) -> str:
        return (
            f"[{self.alert_level}] {self.device_id} | "
            f"score={self.fused_score:.3f} | "
            f"τ={self.threshold_used:.3f} | "
            f"attack={self.attack_type}"
        )


# =============================================================================
# Adaptive Threshold Engine
# =============================================================================
class AdaptiveThresholdEngine:
    """
    Dynamically adjusts detection threshold based on network context.
    
    Formula:
        τ(t,d) = τ_base + β1×L(t) - β2×R(d) - β3×T(t)
    
    Input:
        Sfused ∈ [0,1]         — Fused detection score
        L(t)   ∈ [0,1]         — Normalized network load
        R(d)   ∈ {0.0 ... 1.0} — Device risk level
        T(t)   ∈ [0,1]         — Time-of-day activity factor
    
    Output:
        AlertObject with level: NONE / WARNING / CRITICAL
    """

    # Device risk levels
    DEVICE_RISK = {
        "sensor":       0.3,
        "actuator":     0.5,
        "camera":       0.6,
        "gateway":      0.9,
        "controller":   1.0,
        "unknown":      0.5
    }

    def __init__(self, config: dict = None):
        self.config = config or THRESHOLD_CONFIG
        self.tau_base       = self.config["tau_base"]
        self.beta1          = self.config["beta1"]
        self.beta2          = self.config["beta2"]
        self.beta3          = self.config["beta3"]
        self.warning_margin = self.config["warning_margin"]

        self.alert_log: List[AlertObject] = []
        self._alert_counter = 0

        logger.info(
            f"AdaptiveThreshold: τ_base={self.tau_base} | "
            f"β1={self.beta1} | β2={self.beta2} | β3={self.beta3}"
        )

    # ──────────────────────────────────────────────────────────────
    # Threshold Computation
    # ──────────────────────────────────────────────────────────────
    def compute_threshold(
        self,
        network_load: float = 0.5,
        device_risk: float = 0.5,
        time_factor: float = 0.5
    ) -> float:
        """
        Compute adaptive threshold using contextual factors.
        
        Args:
            network_load: Current normalized network load ∈ [0,1]
                          (High load → raise threshold to reduce FP)
            device_risk:  Device risk level ∈ [0,1]
                          (High risk → lower threshold → stricter)
            time_factor:  Time-of-day activity ∈ [0,1]
                          (Low activity (night) → lower threshold)
        
        Returns:
            tau: Adaptive threshold ∈ [0.1, 0.9]
        """
        tau = (
            self.tau_base
            + self.beta1 * network_load
            - self.beta2 * device_risk
            - self.beta3 * (1.0 - time_factor)     # low activity → stricter
        )
        tau = float(np.clip(tau, 0.1, 0.9))
        return tau

    def get_time_factor(self, hour: Optional[int] = None) -> float:
        """
        Compute time-of-day activity factor.
        Business hours → high activity (factor near 1.0)
        Night hours    → low activity  (factor near 0.0)
        """
        if hour is None:
            hour = datetime.now().hour
        # Smooth cosine curve peaking at hour 12
        factor = 0.5 * (1 + np.cos(np.pi * (hour - 12) / 12))
        return float(np.clip(factor, 0, 1))

    def get_device_risk(self, device_type: str) -> float:
        """Look up risk level for a device type."""
        return self.DEVICE_RISK.get(
            device_type.lower(),
            self.DEVICE_RISK["unknown"]
        )

    # ──────────────────────────────────────────────────────────────
    # Alert Classification
    # ──────────────────────────────────────────────────────────────
    def classify_alert(
        self,
        fused_score: float,
        tau: float
    ) -> str:
        """
        Classify alert level based on fused score vs threshold.
        
        Returns:
            "NONE" | "WARNING" | "CRITICAL"
        """
        if fused_score < tau:
            return "NONE"
        elif fused_score < tau + self.warning_margin:
            return "WARNING"
        else:
            return "CRITICAL"

    # ──────────────────────────────────────────────────────────────
    # Main Processing Interface
    # ──────────────────────────────────────────────────────────────
    def process(
        self,
        fused_score: float,
        s_if_score: float,
        p_rf_score: float,
        device_id: str,
        attack_type: str = "Unknown",
        device_type: str = "unknown",
        network_load: float = 0.5,
        hour: Optional[int] = None
    ) -> AlertObject:
        """
        Process a single detection event and produce an AlertObject.
        
        Args:
            fused_score:   Combined score from fusion engine
            s_if_score:    Raw Isolation Forest score
            p_rf_score:    Raw Random Forest probability
            device_id:     Device MAC or identifier
            attack_type:   Predicted attack category (from RF)
            device_type:   Device category (sensor, gateway, etc.)
            network_load:  Current normalized network load ∈ [0,1]
            hour:          Current hour (None = use system time)
        
        Returns:
            AlertObject with full context
        """
        time_factor  = self.get_time_factor(hour)
        device_risk  = self.get_device_risk(device_type)
        tau = self.compute_threshold(network_load, device_risk, time_factor)
        alert_level  = self.classify_alert(fused_score, tau)

        self._alert_counter += 1
        alert_id = f"ALERT-{self._alert_counter:06d}"
        timestamp = datetime.utcnow().isoformat() + "Z"

        alert = AlertObject(
            alert_id       = alert_id,
            device_id      = device_id,
            timestamp      = timestamp,
            fused_score    = float(fused_score),
            threshold_used = float(tau),
            alert_level    = alert_level,
            attack_type    = attack_type,
            s_if_score     = float(s_if_score),
            p_rf_score     = float(p_rf_score),
            context        = {
                "network_load":  round(network_load, 3),
                "device_risk":   round(device_risk, 3),
                "time_factor":   round(time_factor, 3),
                "hour":          hour if hour else datetime.now().hour,
                "device_type":   device_type
            }
        )

        if alert_level != "NONE":
            self.alert_log.append(alert)
            logger.warning(str(alert))
        else:
            logger.debug(
                f"[NONE] {device_id} | score={fused_score:.3f} | τ={tau:.3f}"
            )

        return alert

    def process_batch(
        self,
        fused_scores:  np.ndarray,
        s_if_scores:   np.ndarray,
        p_rf_scores:   np.ndarray,
        device_ids:    list,
        attack_labels: list,
        device_types:  list = None,
        network_load:  float = 0.5,
        hour: Optional[int] = None
    ) -> List[AlertObject]:
        """
        Process a batch of detection events.
        
        Returns:
            List of AlertObjects (one per sample)
        """
        n = len(fused_scores)
        if device_types is None:
            device_types = ["unknown"] * n

        alerts = []
        for i in range(n):
            alert = self.process(
                fused_score  = float(fused_scores[i]),
                s_if_score   = float(s_if_scores[i]),
                p_rf_score   = float(p_rf_scores[i]),
                device_id    = str(device_ids[i]),
                attack_type  = str(attack_labels[i]),
                device_type  = str(device_types[i]),
                network_load = network_load,
                hour         = hour
            )
            alerts.append(alert)
        return alerts

    # ──────────────────────────────────────────────────────────────
    # Calibration
    # ──────────────────────────────────────────────────────────────
    def calibrate_base_threshold(
        self,
        fused_scores_val: np.ndarray,
        y_val: np.ndarray,
        target_fpr: float = 0.05
    ) -> float:
        """
        Calibrate τ_base on validation set to meet target FPR.
        
        Args:
            fused_scores_val: Fused scores on validation set
            y_val: True binary labels
            target_fpr: Maximum acceptable false positive rate
        
        Returns:
            Calibrated tau_base
        """
        logger.info(f"Calibrating threshold for target FPR ≤ {target_fpr:.2f}...")
        normal_scores = fused_scores_val[y_val == 0]

        # Find threshold where FPR = target_fpr
        percentile = (1 - target_fpr) * 100
        tau_calibrated = float(np.percentile(normal_scores, percentile))
        tau_calibrated = float(np.clip(tau_calibrated, 0.1, 0.9))

        self.tau_base = tau_calibrated
        logger.info(f"Calibrated τ_base = {self.tau_base:.4f}")
        return self.tau_base

    # ──────────────────────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────────────────────
    def get_alert_summary(self) -> Dict:
        """Return summary statistics of all generated alerts."""
        if not self.alert_log:
            return {"total_alerts": 0}

        levels = [a.alert_level for a in self.alert_log]
        attacks = [a.attack_type for a in self.alert_log]
        scores = [a.fused_score for a in self.alert_log]

        return {
            "total_alerts":     len(self.alert_log),
            "critical_count":   levels.count("CRITICAL"),
            "warning_count":    levels.count("WARNING"),
            "mean_score":       round(float(np.mean(scores)), 4),
            "max_score":        round(float(np.max(scores)), 4),
            "attack_type_dist": {
                t: attacks.count(t) for t in set(attacks)
            }
        }

    def save_alerts(self, filename: str = "alert_log.json") -> str:
        """Save all alerts to JSON file."""
        path = os.path.join(REPORT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [a.to_dict() for a in self.alert_log],
                f, indent=2, default=str
            )
        logger.info(f"Alert log saved: {path} ({len(self.alert_log)} alerts)")
        return path
