# =============================================================================
# Phase 2: Behavioral Profiler
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger("BehavioralProfiler")


class DeviceProfile:
    """Statistical behavioral profile for a single IoT device."""

    def __init__(self, device_id: str, window_size: int = 100, alpha: float = 0.1):
        self.device_id = device_id
        self.window_size = window_size
        self.alpha = alpha        # EWMA smoothing factor
        self.n_updates = 0

        # Running statistics
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.cov_matrix: Optional[np.ndarray] = None
        self.buffer = []          # rolling window buffer
        self.is_warm = False      # True after warm-up period

    def update(self, feature_vector: np.ndarray) -> None:
        """Update profile with new observation using EWMA."""
        self.buffer.append(feature_vector.copy())
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        self.n_updates += 1

        if len(self.buffer) >= 10:
            data = np.array(self.buffer)
            new_mean = data.mean(axis=0)
            new_std  = data.std(axis=0) + 1e-8

            if self.mean is None:
                self.mean = new_mean
                self.std  = new_std
            else:
                # EWMA update
                self.mean = self.alpha * new_mean + (1 - self.alpha) * self.mean
                self.std  = self.alpha * new_std  + (1 - self.alpha) * self.std

            if len(self.buffer) >= 20:
                self.cov_matrix = np.cov(data.T) + np.eye(data.shape[1]) * 1e-6
                self.is_warm = True

    def z_score(self, x: np.ndarray) -> np.ndarray:
        """Compute per-feature Z-score deviation."""
        if self.mean is None:
            return np.zeros(len(x))
        return (x - self.mean) / (self.std + 1e-8)

    def mahalanobis_distance(self, x: np.ndarray) -> float:
        """Compute Mahalanobis distance from device baseline."""
        if self.cov_matrix is None or self.mean is None:
            return 0.0
        try:
            cov_inv = pinv(self.cov_matrix)
            diff = x - self.mean
            dist = float(np.sqrt(diff @ cov_inv @ diff))
            return dist
        except Exception:
            return 0.0

    def deviation_score(self, x: np.ndarray) -> float:
        """Aggregate scalar deviation score for fusion engine."""
        z = self.z_score(x)
        return float(np.mean(np.abs(z)))


class BehavioralProfiler:
    """
    Manages per-device behavioral profiles across the IoT network.
    """

    def __init__(self, warm_up_samples: int = 50):
        self.profiles: Dict[str, DeviceProfile] = {}
        self.warm_up_samples = warm_up_samples
        logger.info(f"Profiler initialized (warm-up: {warm_up_samples} samples)")

    def get_or_create(self, device_id: str) -> DeviceProfile:
        if device_id not in self.profiles:
            self.profiles[device_id] = DeviceProfile(device_id)
            logger.debug(f"New profile created: {device_id}")
        return self.profiles[device_id]

    def fit_profiles(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        device_col: str = "device_id"
    ) -> None:
        """
        Build profiles from historical normal-traffic DataFrame.
        
        Args:
            df: DataFrame with device_id and feature columns
            feature_cols: List of feature column names
            device_col: Column containing device identifiers
        """
        logger.info(f"Building profiles for {df[device_col].nunique()} devices...")

        for device_id, group in df.groupby(device_col):
            profile = self.get_or_create(str(device_id))
            for _, row in group[feature_cols].iterrows():
                profile.update(row.values.astype(float))

        warmed = sum(1 for p in self.profiles.values() if p.is_warm)
        logger.info(
            f"Profiles built: {len(self.profiles)} total, "
            f"{warmed} warmed up"
        )

    def compute_deviation_features(
        self,
        X: np.ndarray,
        device_ids: list,
        feature_names: list
    ) -> np.ndarray:
        """
        Augment feature matrix with per-device deviation scores.
        
        Returns:
            X_augmented: Original features + [z_score_mean, mahalanobis]
        """
        z_scores = []
        maha_scores = []

        for i, (x, dev_id) in enumerate(zip(X, device_ids)):
            profile = self.get_or_create(str(dev_id))
            profile.update(x)
            z_mean = float(np.mean(np.abs(profile.z_score(x))))
            m_dist = profile.mahalanobis_distance(x)
            z_scores.append(z_mean)
            maha_scores.append(m_dist)

        deviation_matrix = np.column_stack([
            np.array(z_scores).reshape(-1, 1),
            np.array(maha_scores).reshape(-1, 1)
        ])
        X_augmented = np.hstack([X, deviation_matrix])
        logger.info(
            f"Deviation features added: X {X.shape} → {X_augmented.shape}"
        )
        return X_augmented
