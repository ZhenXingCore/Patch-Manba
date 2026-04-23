"""General helper utilities.

The functions in this module are intentionally lightweight and dependency-minimal.
They are shared by parsing, encoding, serialization, and dataset construction code.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List

import numpy as np


def clean_time_string(time_str: Any) -> str:
    """Normalize a timestamp string extracted from a trajectory cell.

    Raw CSV trajectory cells may contain duplicated quotes or mixed quote styles
    after spreadsheet export or manual editing. This function removes surrounding
    quote artifacts while preserving the timestamp content itself.

    Parameters
    ----------
    time_str:
        The raw timestamp value stored in a trajectory point. The input may already
        be a string, but non-string values are also accepted and converted.

    Returns
    -------
    str
        A cleaned timestamp string suitable for `datetime.strptime`.
    """
    s = str(time_str).strip()
    s = s.replace('""', '"')
    s = s.strip('"').strip("'").strip()
    return s


def inverse_minmax(norm_value: Any, x_min: Any, x_max: Any) -> np.ndarray:
    """Invert min-max normalization.

    The function supports both scalar min/max values and vectorized array inputs,
    which makes it reusable in both per-sample restoration and batched metric
    computation.

    Parameters
    ----------
    norm_value:
        Normalized value or array in the nominal [0, 1] range.
    x_min:
        Minimum raw value used during normalization.
    x_max:
        Maximum raw value used during normalization.

    Returns
    -------
    numpy.ndarray
        Restored raw-scale values, rounded for stable serialization.
    """
    norm_value = np.asarray(norm_value, dtype=np.float32)

    if np.isscalar(x_min) and np.isscalar(x_max):
        if abs(x_max - x_min) < 1e-8:
            raw = np.full_like(norm_value, fill_value=x_min, dtype=np.float32)
        else:
            raw = norm_value * (x_max - x_min) + x_min
        return np.round(raw, 6).astype(np.float32)

    x_min = np.asarray(x_min, dtype=np.float32)
    x_max = np.asarray(x_max, dtype=np.float32)
    diff = x_max - x_min
    raw = np.where(np.abs(diff) < 1e-8, x_min, norm_value * diff + x_min)
    return np.round(raw, 6).astype(np.float32)


def restore_pred_lonlat(pred_norm_xy: Any, restore_info: Any) -> np.ndarray:
    """Restore normalized longitude/latitude predictions to raw coordinates.

    Parameters
    ----------
    pred_norm_xy:
        Normalized coordinates with shape `(2,)` or `(N, 2)`.
    restore_info:
        Per-sample restoration statistics with shape `(4,)` or `(N, 4)`, following
        the convention `[lon_min, lon_max, lat_min, lat_max]`.

    Returns
    -------
    numpy.ndarray
        Raw longitude/latitude coordinates with the same leading shape as the input.
    """
    pred_norm_xy = np.asarray(pred_norm_xy, dtype=np.float32)
    restore_info = np.asarray(restore_info, dtype=np.float32)

    if pred_norm_xy.ndim == 1:
        lon_norm = pred_norm_xy[0]
        lat_norm = pred_norm_xy[1]
        lon_min, lon_max, lat_min, lat_max = restore_info
        lon_raw = inverse_minmax(lon_norm, lon_min, lon_max)
        lat_raw = inverse_minmax(lat_norm, lat_min, lat_max)
        return np.array([lon_raw, lat_raw], dtype=np.float32)

    lon_norm = pred_norm_xy[:, 0]
    lat_norm = pred_norm_xy[:, 1]
    lon_min = restore_info[:, 0]
    lon_max = restore_info[:, 1]
    lat_min = restore_info[:, 2]
    lat_max = restore_info[:, 3]
    lon_raw = inverse_minmax(lon_norm, lon_min, lon_max)
    lat_raw = inverse_minmax(lat_norm, lat_min, lat_max)
    return np.stack([lon_raw, lat_raw], axis=1).astype(np.float32)


def get_track_restore_info(raw_arr: np.ndarray) -> np.ndarray:
    """Extract longitude/latitude restoration statistics from one raw track.

    Parameters
    ----------
    raw_arr:
        Raw trajectory array with shape `(M, 5)` and columns
        `[lon, lat, sog, cog, timestamp_sec]`.

    Returns
    -------
    numpy.ndarray
        A `(4,)` array storing `[lon_min, lon_max, lat_min, lat_max]`.
    """
    lon = raw_arr[:, 0]
    lat = raw_arr[:, 1]
    return np.array([np.min(lon), np.max(lon), np.min(lat), np.max(lat)], dtype=np.float32)


def normalize_source_name(source_name: Any) -> str:
    """Normalize user-facing source names to stable internal aliases.

    This helper is primarily used when building filenames or cross-window matching
    keys. The normalization intentionally supports a few multilingual aliases that
    frequently occur in trajectory datasets.
    """
    s = str(source_name).strip().lower()
    mapping = {
        "ais": "ais",
        "radar": "radar",
        "rader": "radar",
        "bd": "bd",
        "beidou": "bd",
        "北斗": "bd",
        "雷达": "radar",
    }
    return mapping.get(s, s)


def build_output_csv_path(
    output_dir: str,
    source_name: str,
    input_patch_num: int,
    patch_minutes: int,
    future_step_minutes: int,
    training_mode: str = "pseudo_recursive",
) -> str:
    """Build a deterministic output filename for prebuilt dataset CSV files."""
    source_alias = {"AIS": "ais", "radar": "radar", "bd": "bd"}
    source_key = source_alias.get(source_name, str(source_name).lower())
    suffix = "pseudo" if training_mode == "pseudo_recursive" else "recursive"
    return os.path.join(
        output_dir,
        f"{source_key}_{input_patch_num}batch_{patch_minutes}min_{future_step_minutes}min_{suffix}.csv",
    )


def default_window_configs() -> List[Dict[str, int]]:
    """Return the default set of temporal window configurations.

    The preset list mirrors the common multi-scale configurations used in the
    accompanying experiments and provides a convenient baseline for external users.
    """
    return [
        {"name": "win15_12x15", "input_patch_num": 12, "patch_minutes": 15},
        {"name": "win10_18x10", "input_patch_num": 18, "patch_minutes": 10},
        {"name": "win20_9x20", "input_patch_num": 9, "patch_minutes": 20},
        {"name": "win30_6x30", "input_patch_num": 6, "patch_minutes": 30},
        {"name": "win25_7x25", "input_patch_num": 7, "patch_minutes": 25},
    ]


def ndarray_to_json(arr: Any) -> str:
    """Serialize a numpy-compatible array to a JSON string."""
    arr = np.asarray(arr)
    return json.dumps(arr.tolist(), ensure_ascii=False)


def json_to_ndarray(s: str, dtype=np.float32) -> np.ndarray:
    """Deserialize a JSON-encoded array string to a numpy array."""
    return np.asarray(json.loads(s), dtype=dtype)
