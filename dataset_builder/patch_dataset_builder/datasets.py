"""Dataset objects and batch packing utilities."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


def pack_samples_to_batch(samples: List[Dict], pad_value: float = 0.0) -> Dict[str, np.ndarray]:
    """Pack flat pseudo-recursive samples into padded batch arrays.

    The packed representation is designed to serve both direct model input usage and
    later serialization. In particular, the function keeps the original 10D feature
    sequence while also exposing a 6D `model_input` view and a 5D `model_label` view.
    """
    if len(samples) == 0:
        return {
            "data_sequence": np.empty((0, 0, 10), dtype=np.float32),
            "model_input": np.empty((0, 0, 6), dtype=np.float32),
            "sequence_mask": np.empty((0, 0), dtype=np.float32),
            "patch_index": np.empty((0, 0), dtype=np.int64),
            "patch_mask": np.empty((0, 0), dtype=np.float32),
            "label": np.empty((0, 10), dtype=np.float32),
            "model_label": np.empty((0, 5), dtype=np.float32),
            "restore_info": np.empty((0, 4), dtype=np.float32),
            "track_id": np.empty((0,), dtype=np.int64),
            "source_name": np.empty((0,), dtype=object),
        }

    N = len(samples)
    Lmax = max(s["data_sequence"].shape[0] for s in samples)
    P = len(samples[0]["patch_mask"])

    data_sequence = np.full((N, Lmax, 10), pad_value, dtype=np.float32)
    model_input = np.full((N, Lmax, 6), pad_value, dtype=np.float32)
    sequence_mask = np.zeros((N, Lmax), dtype=np.float32)
    patch_index = np.zeros((N, Lmax), dtype=np.int64)
    patch_mask = np.zeros((N, P), dtype=np.float32)
    label = np.zeros((N, 10), dtype=np.float32)
    model_label = np.zeros((N, 5), dtype=np.float32)
    restore_info = np.zeros((N, 4), dtype=np.float32)
    track_id = np.zeros((N,), dtype=np.int64)
    source_name = np.empty((N,), dtype=object)

    for i, s in enumerate(samples):
        L = s["data_sequence"].shape[0]
        if L > 0:
            data_sequence[i, :L] = s["data_sequence"]
            model_input[i, :L] = s["data_sequence"][:, :6]
            sequence_mask[i, :L] = 1.0
            patch_index[i, :L] = s["patch_index"]

        patch_mask[i] = s["patch_mask"]
        label[i] = s["label"]
        model_label[i] = s["label"][:5]
        restore_info[i] = s["restore_info"]
        track_id[i] = int(s.get("track_id", i))
        source_name[i] = str(s.get("source_name", "unknown"))

    return {
        "data_sequence": data_sequence,
        "model_input": model_input,
        "sequence_mask": sequence_mask,
        "patch_index": patch_index,
        "patch_mask": patch_mask,
        "label": label,
        "model_label": model_label,
        "restore_info": restore_info,
        "track_id": track_id,
        "source_name": source_name,
    }


class PatchForecastDataset(Dataset):
    """PyTorch dataset for flat pseudo-recursive training samples."""

    def __init__(self, batch_data: Dict[str, np.ndarray]):
        self.data_sequence = torch.tensor(batch_data["data_sequence"], dtype=torch.float32)
        self.model_input = torch.tensor(batch_data["model_input"], dtype=torch.float32)
        self.sequence_mask = torch.tensor(batch_data["sequence_mask"], dtype=torch.float32)
        self.patch_index = torch.tensor(batch_data["patch_index"], dtype=torch.long)
        self.patch_mask = torch.tensor(batch_data["patch_mask"], dtype=torch.float32)
        self.label = torch.tensor(batch_data["label"], dtype=torch.float32)
        self.model_label = torch.tensor(batch_data["model_label"], dtype=torch.float32)
        self.restore_info = torch.tensor(batch_data["restore_info"], dtype=torch.float32)
        self.track_id = torch.tensor(batch_data.get("track_id", np.zeros((len(self.data_sequence),), dtype=np.int64)), dtype=torch.long)
        self.source_name = list(batch_data.get("source_name", ["unknown"] * len(self.data_sequence)))

    def __len__(self) -> int:
        return len(self.data_sequence)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "data_sequence": self.data_sequence[idx],
            "model_input": self.model_input[idx],
            "sequence_mask": self.sequence_mask[idx],
            "patch_index": self.patch_index[idx],
            "patch_mask": self.patch_mask[idx],
            "label": self.label[idx],
            "model_label": self.model_label[idx],
            "restore_info": self.restore_info[idx],
            "track_id": self.track_id[idx],
            "source_name": self.source_name[idx],
        }


def pack_rollout_samples_to_batch(samples: List[Dict]) -> Dict[str, np.ndarray]:
    """Pack true recursive rollout samples into padded batch arrays.

    Each rollout sample contains an observed history segment and a sequence of future
    fixed-step supervision targets. The packed representation preserves both the raw
    future points and the encoded per-step labels.
    """
    if len(samples) == 0:
        return {
            "observed_points6": np.empty((0, 0, 6), dtype=np.float32),
            "observed_points6_mask": np.empty((0, 0), dtype=np.float32),
            "future_points6": np.empty((0, 0, 6), dtype=np.float32),
            "future_labels": np.empty((0, 0, 10), dtype=np.float32),
            "future_model_labels": np.empty((0, 0, 5), dtype=np.float32),
            "rollout_mask": np.empty((0, 0), dtype=np.float32),
            "restore_info": np.empty((0, 4), dtype=np.float32),
            "cut_time_ts": np.empty((0,), dtype=np.float64),
            "source_name": np.empty((0,), dtype=object),
            "track_id": np.empty((0,), dtype=np.int64),
        }

    N = len(samples)
    Omax = max(s["observed_points6"].shape[0] for s in samples)
    Tmax = max(s["future_points6"].shape[0] for s in samples)

    observed_points6 = np.zeros((N, Omax, 6), dtype=np.float32)
    observed_points6_mask = np.zeros((N, Omax), dtype=np.float32)
    future_points6 = np.zeros((N, Tmax, 6), dtype=np.float32)
    future_labels = np.zeros((N, Tmax, 10), dtype=np.float32)
    future_model_labels = np.zeros((N, Tmax, 5), dtype=np.float32)
    rollout_mask = np.zeros((N, Tmax), dtype=np.float32)
    restore_info = np.zeros((N, 4), dtype=np.float32)
    cut_time_ts = np.zeros((N,), dtype=np.float64)
    source_name = np.empty((N,), dtype=object)
    track_id = np.zeros((N,), dtype=np.int64)

    for i, s in enumerate(samples):
        O = s["observed_points6"].shape[0]
        T = s["future_points6"].shape[0]
        if O > 0:
            observed_points6[i, :O] = s["observed_points6"]
            observed_points6_mask[i, :O] = 1.0
        if T > 0:
            future_points6[i, :T] = s["future_points6"]
            future_labels[i, :T] = s["future_labels"]
            future_model_labels[i, :T] = s["future_model_labels"]
            rollout_mask[i, :T] = 1.0
        restore_info[i] = s["restore_info"]
        cut_time_ts[i] = s["cut_time_ts"]
        source_name[i] = s["source_name"]
        track_id[i] = s["track_id"]

    return {
        "observed_points6": observed_points6,
        "observed_points6_mask": observed_points6_mask,
        "future_points6": future_points6,
        "future_labels": future_labels,
        "future_model_labels": future_model_labels,
        "rollout_mask": rollout_mask,
        "restore_info": restore_info,
        "cut_time_ts": cut_time_ts,
        "source_name": source_name,
        "track_id": track_id,
    }


class PatchForecastRolloutDataset(Dataset):
    """PyTorch dataset for true recursive rollout samples."""

    def __init__(self, batch_data: Dict[str, np.ndarray]):
        self.observed_points6 = torch.tensor(batch_data["observed_points6"], dtype=torch.float32)
        self.observed_points6_mask = torch.tensor(batch_data["observed_points6_mask"], dtype=torch.float32)
        self.future_points6 = torch.tensor(batch_data["future_points6"], dtype=torch.float32)
        self.future_labels = torch.tensor(batch_data["future_labels"], dtype=torch.float32)
        self.future_model_labels = torch.tensor(batch_data["future_model_labels"], dtype=torch.float32)
        self.rollout_mask = torch.tensor(batch_data["rollout_mask"], dtype=torch.float32)
        self.restore_info = torch.tensor(batch_data["restore_info"], dtype=torch.float32)
        self.cut_time_ts = torch.tensor(batch_data["cut_time_ts"], dtype=torch.float64)
        self.source_name = list(batch_data["source_name"])
        self.track_id = torch.tensor(batch_data["track_id"], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.future_model_labels)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "observed_points6": self.observed_points6[idx],
            "observed_points6_mask": self.observed_points6_mask[idx],
            "future_points6": self.future_points6[idx],
            "future_labels": self.future_labels[idx],
            "future_model_labels": self.future_model_labels[idx],
            "rollout_mask": self.rollout_mask[idx],
            "restore_info": self.restore_info[idx],
            "cut_time_ts": self.cut_time_ts[idx],
            "track_id": self.track_id[idx],
            "source_name": self.source_name[idx],
        }
