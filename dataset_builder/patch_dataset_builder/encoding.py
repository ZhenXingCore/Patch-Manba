"""Feature encoding and patch extraction utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .constants import EPS
from .geometry import project_point_by_sog_cog


def encode_raw_point_to_feat10(raw_point: np.ndarray, sample_start_ts: float, restore_info: np.ndarray) -> np.ndarray:
    """Encode one raw point into the standard 10-dimensional feature interface.

    The encoding follows the convention:

    `[lon_norm, lat_norm, sog_div10, cog_sin, cog_cos, relative_time_min,
      lon_min_raw, lon_max_raw, lat_min_raw, lat_max_raw]`

    Parameters
    ----------
    raw_point:
        One raw point with at least the first five columns
        `[lon, lat, sog, cog, timestamp_sec]`.
    sample_start_ts:
        Reference timestamp used to compute relative time in minutes.
    restore_info:
        Restoration statistics `[lon_min, lon_max, lat_min, lat_max]`.

    Returns
    -------
    numpy.ndarray
        A `(10,)` encoded feature vector.
    """
    lon, lat, sog, cog, ts = raw_point[:5]
    lon_min, lon_max, lat_min, lat_max = restore_info

    lon_norm = 0.0 if abs(lon_max - lon_min) < 1e-8 else (lon - lon_min) / (lon_max - lon_min)
    lat_norm = 0.0 if abs(lat_max - lat_min) < 1e-8 else (lat - lat_min) / (lat_max - lat_min)

    sog_div10 = sog / 10.0
    cog_rad = np.deg2rad(cog)
    cog_sin = np.sin(cog_rad)
    cog_cos = np.cos(cog_rad)
    rel_time_min = (ts - sample_start_ts) / 60.0

    feat = np.array(
        [lon_norm, lat_norm, sog_div10, cog_sin, cog_cos, rel_time_min, lon_min, lon_max, lat_min, lat_max],
        dtype=np.float32,
    )
    return np.round(feat, 5).astype(np.float32)


def append_interp_flag(raw_arr: np.ndarray) -> np.ndarray:
    """Append an interpolation flag column to a raw trajectory array.

    Real observed points are marked with `interp_flag = 0`. Future interpolated or
    forward-projected points may later use `interp_flag = 1`.
    """
    raw_arr = np.asarray(raw_arr, dtype=np.float32)
    return np.concatenate([raw_arr, np.zeros((len(raw_arr), 1), dtype=np.float32)], axis=1)


def sort_points6(points_arr: Optional[np.ndarray]) -> np.ndarray:
    """Sort a `(N, 6)` point array by timestamp.

    The sixth column is assumed to be the interpolation flag, while the timestamp is
    stored in column index 4.
    """
    if points_arr is None or len(points_arr) == 0:
        return np.empty((0, 6), dtype=np.float32)
    points_arr = np.asarray(points_arr, dtype=np.float32)
    return points_arr[np.argsort(points_arr[:, 4])]


def build_recursive_mixed_points(observed_points6: np.ndarray, generated_points6: np.ndarray) -> np.ndarray:
    """Merge observed history points and generated feedback points for recursion.

    This helper is used in pseudo-recursive training sample construction, where the
    input side may contain previously generated fixed-step points while the labels are
    still derived independently from the real trajectory.
    """
    observed_points6 = sort_points6(observed_points6)
    generated_points6 = sort_points6(generated_points6)

    if len(observed_points6) == 0 and len(generated_points6) == 0:
        return np.empty((0, 6), dtype=np.float32)
    if len(observed_points6) == 0:
        return generated_points6.astype(np.float32)
    if len(generated_points6) == 0:
        return observed_points6.astype(np.float32)

    mixed = np.concatenate([observed_points6, generated_points6], axis=0)
    return sort_points6(mixed).astype(np.float32)


def generate_future_fixed_points_from_raw(
    raw_arr: np.ndarray,
    cut_time_ts: float,
    future_step_minutes: int = 5,
    future_end_time_ts: Optional[float] = None,
) -> np.ndarray:
    """Generate fixed-step future points from a raw trajectory.

    This function intentionally implements the **label-stable** logic used in the
    original pseudo-recursive pipeline:

    - for each target timestamp, the function finds the latest real point at or before
      that timestamp;
    - if the timestamp exactly matches a real point, the real point is used;
    - otherwise, the point is forward-projected from that latest real point.

    This design prevents label drift because later labels do not depend on earlier
    generated labels.
    """
    raw_arr = np.asarray(raw_arr, dtype=np.float32)
    raw_arr = raw_arr[np.argsort(raw_arr[:, 4])]

    if future_end_time_ts is None:
        future_end_time_ts = raw_arr[-1, 4]

    step_sec = int(future_step_minutes * 60)
    if future_end_time_ts - cut_time_ts < step_sec - EPS:
        return np.empty((0, 6), dtype=np.float32)

    target_times = np.arange(cut_time_ts + step_sec, future_end_time_ts + EPS, step_sec, dtype=np.float64)
    out = []
    for tgt_ts in target_times:
        src_idx = np.searchsorted(raw_arr[:, 4], tgt_ts, side="right") - 1
        if src_idx < 0:
            continue

        base_lon, base_lat, base_sog, base_cog, base_ts = raw_arr[src_idx]
        if abs(base_ts - tgt_ts) < EPS:
            lon_t, lat_t = base_lon, base_lat
            sog_t, cog_t = base_sog, base_cog
            interp_flag = 0.0
        else:
            lon_t, lat_t = project_point_by_sog_cog(base_lon, base_lat, base_sog, base_cog, tgt_ts - base_ts)
            sog_t, cog_t = base_sog, base_cog
            interp_flag = 1.0
        out.append([lon_t, lat_t, sog_t, cog_t, tgt_ts, interp_flag])

    if len(out) == 0:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def collect_input_patches_as_feat10(
    points_arr: Optional[np.ndarray],
    window_start_ts: float,
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    restore_info: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Collect windowed points into patch-indexed 10D feature sequences.

    Parameters
    ----------
    points_arr:
        Point array with shape `(N, 5)` or `(N, 6)`. The first five columns must be
        `[lon, lat, sog, cog, timestamp_sec]`.
    window_start_ts:
        Timestamp of the beginning of the full history window.
    input_patch_num:
        Number of temporal patches.
    patch_minutes:
        Duration of each patch in minutes.
    restore_info:
        Restoration statistics used to encode normalized coordinates.

    Returns
    -------
    tuple
        `(data_sequence, patch_index, patch_mask, point_count)` where:

        - `data_sequence` has shape `(L, 10)`;
        - `patch_index` has shape `(L,)` with values in `1..input_patch_num`;
        - `patch_mask` has shape `(input_patch_num,)`;
        - `point_count` is the total number of retained points.
    """
    patch_sec = int(patch_minutes * 60)
    all_feats = []
    all_patch_ids = []
    patch_mask = np.zeros((input_patch_num,), dtype=np.float32)

    if points_arr is None or len(points_arr) == 0:
        return np.empty((0, 10), dtype=np.float32), np.empty((0,), dtype=np.int64), patch_mask, 0

    points_arr = np.asarray(points_arr, dtype=np.float32)
    points_arr = points_arr[np.argsort(points_arr[:, 4])]

    for p in range(input_patch_num):
        left = window_start_ts + p * patch_sec
        right = left + patch_sec
        if p < input_patch_num - 1:
            mask = (points_arr[:, 4] >= left - EPS) & (points_arr[:, 4] < right - EPS)
        else:
            mask = (points_arr[:, 4] >= left - EPS) & (points_arr[:, 4] <= right + EPS)

        patch_points = points_arr[mask]
        if len(patch_points) > 0:
            patch_mask[p] = 1.0
            for point in patch_points:
                feat10 = encode_raw_point_to_feat10(
                    raw_point=point[:5],
                    sample_start_ts=window_start_ts,
                    restore_info=restore_info,
                )
                all_feats.append(feat10)
                all_patch_ids.append(p + 1)

    if len(all_feats) == 0:
        return np.empty((0, 10), dtype=np.float32), np.empty((0,), dtype=np.int64), patch_mask, 0

    data_sequence = np.stack(all_feats, axis=0).astype(np.float32)
    patch_index = np.asarray(all_patch_ids, dtype=np.int64)
    return data_sequence, patch_index, patch_mask, len(data_sequence)
