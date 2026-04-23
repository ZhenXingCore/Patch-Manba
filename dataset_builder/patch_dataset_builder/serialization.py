"""Serialization helpers for prebuilt datasets."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .datasets import (
    PatchForecastDataset,
    PatchForecastRolloutDataset,
    pack_rollout_samples_to_batch,
    pack_samples_to_batch,
)
from .utils import json_to_ndarray, ndarray_to_json


def samples_to_dataframe(samples: List[Dict]) -> pd.DataFrame:
    """Convert flat pseudo-recursive samples to a tabular DataFrame."""
    rows = []
    for s in samples:
        rows.append(
            {
                "source_name": s["source_name"],
                "track_id": s["track_id"],
                "sample_type": s["sample_type"],
                "recursive_step": s["recursive_step"],
                "window_start_ts": s["window_start_ts"],
                "window_end_ts": s["window_end_ts"],
                "future_time_ts": s["future_time_ts"],
                "future_interp_flag": s["future_interp_flag"],
                "feedback_point_count": s["feedback_point_count"],
                "input_point_count": s["input_point_count"],
                "data_sequence_json": ndarray_to_json(np.round(s["data_sequence"], 5)),
                "patch_index_json": ndarray_to_json(s["patch_index"].astype(np.int64)),
                "patch_mask_json": ndarray_to_json(np.round(s["patch_mask"], 5)),
                "label_json": ndarray_to_json(np.round(s["label"], 5)),
                "restore_info_json": ndarray_to_json(np.round(s["restore_info"], 6)),
            }
        )
    return pd.DataFrame(rows)


def save_samples_to_csv(samples: List[Dict], output_csv: str) -> None:
    """Save flat pseudo-recursive samples to CSV."""
    folder = os.path.dirname(output_csv)
    if folder:
        os.makedirs(folder, exist_ok=True)
    df = samples_to_dataframe(samples)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")


def load_saved_samples_from_csv(saved_csv: str) -> List[Dict]:
    """Load flat pseudo-recursive samples from a prebuilt CSV file."""
    df = pd.read_csv(saved_csv, encoding="utf-8-sig")
    samples = []
    for _, row in df.iterrows():
        data_sequence = json_to_ndarray(row["data_sequence_json"], dtype=np.float32)
        patch_index = json_to_ndarray(row["patch_index_json"], dtype=np.int64)
        patch_mask = json_to_ndarray(row["patch_mask_json"], dtype=np.float32)
        label = json_to_ndarray(row["label_json"], dtype=np.float32)
        restore_info = json_to_ndarray(row["restore_info_json"], dtype=np.float32)
        samples.append(
            {
                "source_name": str(row["source_name"]),
                "track_id": int(row["track_id"]),
                "sample_type": str(row["sample_type"]),
                "recursive_step": int(row["recursive_step"]),
                "window_start_ts": float(row["window_start_ts"]),
                "window_end_ts": float(row["window_end_ts"]),
                "future_time_ts": float(row["future_time_ts"]),
                "future_interp_flag": float(row["future_interp_flag"]),
                "feedback_point_count": int(row["feedback_point_count"]),
                "input_point_count": int(row["input_point_count"]),
                "data_sequence": data_sequence.astype(np.float32),
                "patch_index": patch_index.astype(np.int64),
                "patch_mask": patch_mask.astype(np.float32),
                "label": label.astype(np.float32),
                "restore_info": restore_info.astype(np.float32),
            }
        )
    return samples


def load_saved_dataset_from_csv(saved_csv: str, pad_value: float = 0.0):
    """Load a flat pseudo-recursive dataset directly from a prebuilt CSV file."""
    samples = load_saved_samples_from_csv(saved_csv)
    batch_data = pack_samples_to_batch(samples, pad_value=pad_value)
    dataset = PatchForecastDataset(batch_data)
    return samples, batch_data, dataset


def rollout_samples_to_dataframe(samples: List[Dict]) -> pd.DataFrame:
    """Convert rollout samples to a tabular DataFrame."""
    rows = []
    for s in samples:
        rows.append(
            {
                "source_name": s["source_name"],
                "track_id": s["track_id"],
                "cut_time_ts": s["cut_time_ts"],
                "observed_points6_json": ndarray_to_json(np.round(s["observed_points6"], 6)),
                "future_points6_json": ndarray_to_json(np.round(s["future_points6"], 6)),
                "future_labels_json": ndarray_to_json(np.round(s["future_labels"], 5)),
                "future_model_labels_json": ndarray_to_json(np.round(s["future_model_labels"], 5)),
                "restore_info_json": ndarray_to_json(np.round(s["restore_info"], 6)),
            }
        )
    return pd.DataFrame(rows)


def save_rollout_samples_to_csv(samples: List[Dict], output_csv: str) -> None:
    """Save rollout samples to CSV."""
    folder = os.path.dirname(output_csv)
    if folder:
        os.makedirs(folder, exist_ok=True)
    df = rollout_samples_to_dataframe(samples)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")


def load_rollout_samples_from_csv(saved_csv: str) -> List[Dict]:
    """Load rollout samples from a prebuilt CSV file."""
    df = pd.read_csv(saved_csv, encoding="utf-8-sig")
    samples = []
    for _, row in df.iterrows():
        observed_points6 = json_to_ndarray(row["observed_points6_json"], dtype=np.float32)
        future_points6 = json_to_ndarray(row["future_points6_json"], dtype=np.float32)
        future_labels = json_to_ndarray(row["future_labels_json"], dtype=np.float32)
        future_model_labels = json_to_ndarray(row["future_model_labels_json"], dtype=np.float32)
        restore_info = json_to_ndarray(row["restore_info_json"], dtype=np.float32)
        samples.append(
            {
                "source_name": str(row["source_name"]),
                "track_id": int(row["track_id"]),
                "cut_time_ts": float(row["cut_time_ts"]),
                "observed_points6": observed_points6.astype(np.float32),
                "future_points6": future_points6.astype(np.float32),
                "future_labels": future_labels.astype(np.float32),
                "future_model_labels": future_model_labels.astype(np.float32),
                "restore_info": restore_info.astype(np.float32),
            }
        )
    return samples


def load_rollout_dataset_from_csv(saved_csv: str):
    """Load a rollout dataset directly from a prebuilt CSV file."""
    samples = load_rollout_samples_from_csv(saved_csv)
    batch_data = pack_rollout_samples_to_batch(samples)
    dataset = PatchForecastRolloutDataset(batch_data)
    return samples, batch_data, dataset
