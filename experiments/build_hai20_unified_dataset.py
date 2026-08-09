"""
Build a HybridShield unified-schema dataset from HAI 20.07 ICS files.

Input files are the public HAI train/test CSV.GZ files from:
https://github.com/icsdataset/hai

The mapping converts multivariate SCADA/process telemetry into the existing
HybridShield unified feature schema using process-signal statistics and
time-derived context. It does not use attack labels as features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_COLUMNS = {"attack", "attack_P1", "attack_P2", "attack_P3"}


def process_group_columns(columns, prefix):
    return [c for c in columns if c.startswith(prefix)]


def load_hai_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if "time" not in df.columns or "attack" not in df.columns:
        raise ValueError(f"{path} is not a HAI CSV file.")
    return df


def attack_category(df: pd.DataFrame) -> np.ndarray:
    labels = []
    p1 = df.get("attack_P1", pd.Series(np.zeros(len(df), dtype=int))).values
    p2 = df.get("attack_P2", pd.Series(np.zeros(len(df), dtype=int))).values
    p3 = df.get("attack_P3", pd.Series(np.zeros(len(df), dtype=int))).values
    attack = df["attack"].values
    for a, a1, a2, a3 in zip(attack, p1, p2, p3):
        if int(a) == 0:
            labels.append("Normal")
            continue
        parts = []
        if int(a1) == 1:
            parts.append("P1")
        if int(a2) == 1:
            parts.append("P2")
        if int(a3) == 1:
            parts.append("P3")
        labels.append("HAI_" + "_".join(parts) if parts else "HAI_System")
    return np.array(labels, dtype=object)


def dominant_process_device(df: pd.DataFrame, numeric_cols) -> np.ndarray:
    groups = {
        "P1_controller": process_group_columns(numeric_cols, "P1_"),
        "P2_controller": process_group_columns(numeric_cols, "P2_"),
        "P3_controller": process_group_columns(numeric_cols, "P3_"),
        "P4_controller": process_group_columns(numeric_cols, "P4_"),
    }
    diffs = df[numeric_cols].diff().abs().fillna(0)
    group_scores = pd.DataFrame(index=df.index)
    for name, cols in groups.items():
        group_scores[name] = diffs[cols].mean(axis=1) if cols else 0.0
    return group_scores.idxmax(axis=1).values.astype(object)


def harmonize_hai(df: pd.DataFrame) -> pd.DataFrame:
    timestamps = pd.to_datetime(df["time"], errors="coerce")
    numeric_cols = [
        c
        for c in df.columns
        if c not in LABEL_COLUMNS and c != "time"
    ]
    values = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median(numeric_only=True)).fillna(0)
    diffs = values.diff().abs().fillna(0)

    group_prefixes = ["P1_", "P2_", "P3_", "P4_"]
    group_change = []
    for prefix in group_prefixes:
        cols = process_group_columns(numeric_cols, prefix)
        if cols:
            group_change.append(diffs[cols].mean(axis=1).values)
        else:
            group_change.append(np.zeros(len(df)))
    group_change = np.vstack(group_change).T

    elapsed = timestamps.diff().dt.total_seconds().fillna(1).clip(lower=1)
    elapsed = elapsed.replace(0, 1)

    change_threshold = diffs.quantile(0.75).replace(0, 1e-9)
    change_flags = diffs.gt(change_threshold, axis=1)
    change_count = change_flags.sum(axis=1).astype(float)

    unified = pd.DataFrame(
        {
            "packet_rate": change_count / elapsed.values,
            "byte_rate": values.abs().sum(axis=1),
            "burst_length": values.std(axis=1),
            "inter_arrival_time": elapsed.values,
            "conn_frequency": change_count.rolling(30, min_periods=1).mean(),
            "unique_dst_ips": (group_change > 0).sum(axis=1),
            "port_diversity": change_count / max(len(numeric_cols), 1),
            "protocol_distribution": np.argmax(group_change, axis=1) + 1,
            "response_time": diffs.mean(axis=1),
            "session_duration": (
                (timestamps - timestamps.iloc[0]).dt.total_seconds().fillna(0) % 3600
            ),
            "payload_size_mean": values.mean(axis=1),
            "payload_size_std": values.std(axis=1),
            "time_of_day": timestamps.dt.hour.fillna(0).astype(int),
            "day_of_week": timestamps.dt.dayofweek.fillna(0).astype(int),
            "label": df["attack"].astype(int),
            "attack_cat": attack_category(df),
            "device_id": dominant_process_device(df, numeric_cols),
        }
    )
    unified["source_dataset"] = "HAI-20.07"
    return unified.replace([np.inf, -np.inf], np.nan).fillna(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HAI 20.07 unified parquet.")
    parser.add_argument("--train", required=True, help="HAI train CSV.GZ path.")
    parser.add_argument("--test", required=True, help="HAI test CSV.GZ path.")
    parser.add_argument(
        "--out",
        default="data/hai20_unified_dataset.parquet",
        help="Output parquet path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    for path in [Path(args.train), Path(args.test)]:
        print(f"Loading {path}")
        frames.append(harmonize_hai(load_hai_file(path)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {out} rows={len(df)} attacks={int(df['label'].sum())}")
    print(df["attack_cat"].value_counts().to_string())


if __name__ == "__main__":
    main()
