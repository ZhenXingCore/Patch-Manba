"""High-level dataset construction logic.

This module contains the core dataset building pipeline for both pseudo-recursive
single-step samples and true recursive rollout samples.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .constants import EPS
from .datasets import (
    PatchForecastDataset,
    PatchForecastRolloutDataset,
    pack_rollout_samples_to_batch,
    pack_samples_to_batch,
)
from .encoding import (
    append_interp_flag,
    build_recursive_mixed_points,
    collect_input_patches_as_feat10,
    encode_raw_point_to_feat10,
    generate_future_fixed_points_from_raw,
)
from .parsing import load_tracks_from_csv_raw_single_source
from .serialization import save_rollout_samples_to_csv, save_samples_to_csv
from .utils import build_output_csv_path, default_window_configs, get_track_restore_info


def build_patch_forecast_dataset_from_raw_tracks_pseudo(
    tracks_raw: List[np.ndarray],
    source_name: str = "unknown",
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    strict: bool = False,
    pad_value: float = 0.0,
    future_step_minutes: int = 5,
    sample_stride_minutes: int = 5,
    min_total_input_points: int = 1,
    max_future_steps: int | None = None,
):
    """Build pseudo-recursive flat training samples from raw tracks.

    This is the original training-oriented sample construction mode. The input side
    may include generated fixed-step feedback points, but each label is still derived
    independently from the real trajectory to avoid recursive label drift.
    """
    samples = []

    patch_sec = int(patch_minutes * 60)
    input_horizon_sec = input_patch_num * patch_sec
    future_step_sec = int(future_step_minutes * 60)
    stride_sec = int(sample_stride_minutes * 60)

    for track_id, raw_arr in enumerate(tracks_raw):
        if raw_arr is None or len(raw_arr) <= 1:
            continue

        raw_arr = np.asarray(raw_arr, dtype=np.float32)
        raw_arr = raw_arr[np.argsort(raw_arr[:, 4])]

        track_start = raw_arr[0, 4]
        track_end = raw_arr[-1, 4]
        latest_base_ws = track_end - input_horizon_sec - future_step_sec
        if latest_base_ws < track_start:
            continue

        restore_info = get_track_restore_info(raw_arr)
        real_points6 = append_interp_flag(raw_arr)
        base_window_starts = np.arange(track_start, latest_base_ws + EPS, stride_sec, dtype=np.float64)

        for base_ws in base_window_starts:
            base_we = base_ws + input_horizon_sec
            observed_points6 = real_points6[real_points6[:, 4] <= base_we + EPS]

            future_points = generate_future_fixed_points_from_raw(
                raw_arr=raw_arr,
                cut_time_ts=base_we,
                future_step_minutes=future_step_minutes,
                future_end_time_ts=track_end,
            )
            if len(future_points) == 0:
                continue

            if max_future_steps is not None:
                future_points = future_points[: int(max_future_steps)]
                if len(future_points) == 0:
                    continue

            for s in range(len(future_points)):
                cur_ws = base_ws + s * future_step_sec
                cur_we = cur_ws + input_horizon_sec
                cur_label_point = future_points[s]
                prev_generated = future_points[:s] if s > 0 else np.empty((0, 6), dtype=np.float32)

                mixed_points = build_recursive_mixed_points(observed_points6=observed_points6, generated_points6=prev_generated)
                window_mask = (mixed_points[:, 4] >= cur_ws - EPS) & (mixed_points[:, 4] <= cur_we + EPS)
                window_points = mixed_points[window_mask]

                data_sequence, patch_index, patch_mask, point_count = collect_input_patches_as_feat10(
                    points_arr=window_points,
                    window_start_ts=cur_ws,
                    input_patch_num=input_patch_num,
                    patch_minutes=patch_minutes,
                    restore_info=restore_info,
                )

                if point_count < min_total_input_points:
                    continue
                if strict and np.sum(patch_mask) < input_patch_num:
                    continue

                label = encode_raw_point_to_feat10(
                    raw_point=cur_label_point[:5],
                    sample_start_ts=cur_ws,
                    restore_info=restore_info,
                )

                samples.append(
                    {
                        "source_name": str(source_name),
                        "data_sequence": data_sequence.astype(np.float32),
                        "patch_index": patch_index.astype(np.int64),
                        "patch_mask": patch_mask.astype(np.float32),
                        "label": label.astype(np.float32),
                        "restore_info": restore_info.astype(np.float32),
                        "track_id": int(track_id),
                        "sample_type": "normal" if s == 0 else "recursive",
                        "recursive_step": int(s),
                        "window_start_ts": float(cur_ws),
                        "window_end_ts": float(cur_we),
                        "future_time_ts": float(cur_label_point[4]),
                        "future_interp_flag": float(cur_label_point[5]),
                        "feedback_point_count": int(len(prev_generated)),
                        "input_point_count": int(point_count),
                    }
                )

    batch_data = pack_samples_to_batch(samples, pad_value=pad_value)
    dataset = PatchForecastDataset(batch_data)
    return samples, batch_data, dataset


def build_patch_rollout_dataset_from_raw_tracks(
    tracks_raw: List[np.ndarray],
    source_name: str = "unknown",
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    strict: bool = False,
    pad_value: float = 0.0,
    future_step_minutes: int = 5,
    sample_stride_minutes: int = 5,
    min_total_input_points: int = 1,
    max_future_steps: int | None = None,
):
    """Build true recursive rollout samples from raw tracks.

    In this mode, each sample stores one observed history segment and an entire
    future fixed-step sequence. The result is suitable for recursive training or
    rollout-based evaluation, where the model must behave consistently across
    multiple autoregressive steps.
    """
    samples = []

    history_sec = int(input_patch_num * patch_minutes * 60)
    future_step_sec = int(future_step_minutes * 60)
    stride_sec = int(sample_stride_minutes * 60)

    for track_id, raw_arr in enumerate(tracks_raw):
        if raw_arr is None or len(raw_arr) <= 1:
            continue

        raw_arr = np.asarray(raw_arr, dtype=np.float32)
        raw_arr = raw_arr[np.argsort(raw_arr[:, 4])]
        real_points6 = append_interp_flag(raw_arr)
        restore_info = get_track_restore_info(raw_arr)

        track_start = raw_arr[0, 4]
        track_end = raw_arr[-1, 4]
        earliest_cut = track_start + history_sec
        latest_cut = track_end - future_step_sec
        if latest_cut < earliest_cut:
            continue

        cut_times = np.arange(earliest_cut, latest_cut + EPS, stride_sec, dtype=np.float64)

        for cut_time_ts in cut_times:
            observed_points6 = real_points6[real_points6[:, 4] <= cut_time_ts + EPS]
            future_points = generate_future_fixed_points_from_raw(
                raw_arr=raw_arr,
                cut_time_ts=cut_time_ts,
                future_step_minutes=future_step_minutes,
                future_end_time_ts=track_end,
            )
            if len(future_points) == 0:
                continue

            if max_future_steps is not None:
                future_points = future_points[: int(max_future_steps)]
                if len(future_points) == 0:
                    continue

            init_ws = cut_time_ts - history_sec
            init_we = cut_time_ts
            init_mask = (observed_points6[:, 4] >= init_ws - EPS) & (observed_points6[:, 4] <= init_we + EPS)
            init_window_points = observed_points6[init_mask]

            _, _, init_patch_mask, point_count = collect_input_patches_as_feat10(
                points_arr=init_window_points,
                window_start_ts=init_ws,
                input_patch_num=input_patch_num,
                patch_minutes=patch_minutes,
                restore_info=restore_info,
            )

            if point_count < min_total_input_points:
                continue
            if strict and np.sum(init_patch_mask) < input_patch_num:
                continue

            future_labels = []
            future_model_labels = []
            for step_idx in range(len(future_points)):
                current_cut = cut_time_ts + step_idx * future_step_sec
                current_ws = current_cut - history_sec
                label = encode_raw_point_to_feat10(
                    raw_point=future_points[step_idx][:5],
                    sample_start_ts=current_ws,
                    restore_info=restore_info,
                )
                future_labels.append(label.astype(np.float32))
                future_model_labels.append(label[:5].astype(np.float32))

            samples.append(
                {
                    "source_name": str(source_name),
                    "track_id": int(track_id),
                    "cut_time_ts": float(cut_time_ts),
                    "observed_points6": observed_points6.astype(np.float32),
                    "future_points6": future_points.astype(np.float32),
                    "future_labels": np.asarray(future_labels, dtype=np.float32),
                    "future_model_labels": np.asarray(future_model_labels, dtype=np.float32),
                    "restore_info": restore_info.astype(np.float32),
                }
            )

    batch_data = pack_rollout_samples_to_batch(samples)
    dataset = PatchForecastRolloutDataset(batch_data)
    return samples, batch_data, dataset


def build_patch_forecast_dataset_from_raw_tracks(
    tracks_raw: List[np.ndarray],
    source_name: str = "unknown",
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    strict: bool = False,
    pad_value: float = 0.0,
    future_step_minutes: int = 5,
    sample_stride_minutes: int = 5,
    min_total_input_points: int = 1,
    max_future_steps: int | None = None,
    training_mode: str = "pseudo_recursive",
):
    """Unified entrypoint for building datasets from raw tracks.

    Parameters
    ----------
    training_mode:
        Either `"pseudo_recursive"` or `"recursive"`.
    """
    if training_mode == "pseudo_recursive":
        return build_patch_forecast_dataset_from_raw_tracks_pseudo(
            tracks_raw=tracks_raw,
            source_name=source_name,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            strict=strict,
            pad_value=pad_value,
            future_step_minutes=future_step_minutes,
            sample_stride_minutes=sample_stride_minutes,
            min_total_input_points=min_total_input_points,
            max_future_steps=max_future_steps,
        )
    if training_mode == "recursive":
        return build_patch_rollout_dataset_from_raw_tracks(
            tracks_raw=tracks_raw,
            source_name=source_name,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            strict=strict,
            pad_value=pad_value,
            future_step_minutes=future_step_minutes,
            sample_stride_minutes=sample_stride_minutes,
            min_total_input_points=min_total_input_points,
            max_future_steps=max_future_steps,
        )
    raise ValueError(f"Unsupported training_mode: {training_mode}")


def build_patch_forecast_dataset_from_csv_single_source(
    csv_path: str = "data.csv",
    source_name: str = "AIS",
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    strict: bool = False,
    pad_value: float = 0.0,
    future_step_minutes: int = 5,
    sample_stride_minutes: int = 5,
    min_total_input_points: int = 1,
    max_future_steps: int | None = None,
    training_mode: str = "pseudo_recursive",
):
    """Build a dataset directly from one source column of a raw CSV file."""
    tracks_raw = load_tracks_from_csv_raw_single_source(csv_path, source_name=source_name)
    return build_patch_forecast_dataset_from_raw_tracks(
        tracks_raw=tracks_raw,
        source_name=source_name,
        input_patch_num=input_patch_num,
        patch_minutes=patch_minutes,
        strict=strict,
        pad_value=pad_value,
        future_step_minutes=future_step_minutes,
        sample_stride_minutes=sample_stride_minutes,
        min_total_input_points=min_total_input_points,
        max_future_steps=max_future_steps,
        training_mode=training_mode,
    )


def build_and_save_source_multiscale(
    csv_path: str = "data.csv",
    output_dir: str = "prebuilt_source_csv",
    source_name: str = "AIS",
    window_configs=None,
    strict: bool = False,
    pad_value: float = 0.0,
    future_step_minutes: int = 5,
    sample_stride_minutes: int = 5,
    min_total_input_points: int = 1,
    max_future_steps: int = 12,
    training_mode: str = "pseudo_recursive",
):
    """Build and save multiple window configurations for one source.

    This function is intended for offline preprocessing. It creates a deterministic
    prebuilt dataset CSV for each requested temporal window configuration and also
    exports a compact summary table.
    """
    import os
    import pandas as pd

    if window_configs is None:
        window_configs = default_window_configs()

    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []

    for cfg in window_configs:
        name = str(cfg["name"])
        input_patch_num = int(cfg["input_patch_num"])
        patch_minutes = int(cfg["patch_minutes"])

        samples, batch_data, _ = build_patch_forecast_dataset_from_csv_single_source(
            csv_path=csv_path,
            source_name=source_name,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            strict=strict,
            pad_value=pad_value,
            future_step_minutes=future_step_minutes,
            sample_stride_minutes=sample_stride_minutes,
            min_total_input_points=min_total_input_points,
            max_future_steps=max_future_steps,
            training_mode=training_mode,
        )

        output_csv = build_output_csv_path(
            output_dir=output_dir,
            source_name=source_name,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            future_step_minutes=future_step_minutes,
            training_mode=training_mode,
        )

        if training_mode == "pseudo_recursive":
            save_samples_to_csv(samples, output_csv)
            normal_cnt = sum(s["sample_type"] == "normal" for s in samples)
            recursive_cnt = sum(s["sample_type"] == "recursive" for s in samples)
        else:
            save_rollout_samples_to_csv(samples, output_csv)
            normal_cnt = len(samples)
            recursive_cnt = len(samples)

        summary_rows.append(
            {
                "source_name": source_name,
                "config_name": name,
                "training_mode": training_mode,
                "input_patch_num": input_patch_num,
                "patch_minutes": patch_minutes,
                "history_minutes": input_patch_num * patch_minutes,
                "future_step_minutes": future_step_minutes,
                "max_future_steps": max_future_steps,
                "output_csv": output_csv,
                "sample_count": len(samples),
                "normal_count": normal_cnt,
                "recursive_count": recursive_cnt,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, f"summary_{str(source_name).lower()}_{training_mode}.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    return summary_df
