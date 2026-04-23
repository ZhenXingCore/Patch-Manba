"""CSV parsing and raw trajectory loading utilities."""

from __future__ import annotations

import ast
from datetime import datetime
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from .utils import clean_time_string


def parse_track_cell_raw(cell: Any) -> np.ndarray | None:
    """Parse one trajectory cell into a raw trajectory array.

    Each valid trajectory point is expected to follow the schema
    `[longitude, latitude, sog, cog, timestamp_string]`. Invalid points are skipped
    rather than causing the full cell to fail.

    Parameters
    ----------
    cell:
        A CSV cell that stores a Python-style list of trajectory points.

    Returns
    -------
    numpy.ndarray or None
        A sorted raw trajectory array with shape `(M, 5)` where columns are
        `[lon, lat, sog, cog, timestamp_sec]`. Returns `None` when parsing fails
        or fewer than two valid points are present.
    """
    if pd.isna(cell):
        return None

    if isinstance(cell, str):
        cell = cell.strip()
        if cell == "":
            return None
        cell = cell.replace('""', '"')

    try:
        traj = ast.literal_eval(cell) if isinstance(cell, str) else cell
    except Exception:
        return None

    if not isinstance(traj, (list, tuple)) or len(traj) == 0:
        return None

    parsed = []
    for point in traj:
        if not isinstance(point, (list, tuple)) or len(point) < 5:
            continue
        try:
            lon = float(point[0])
            lat = float(point[1])
            sog = float(point[2])
            cog = float(point[3])
            time_str = clean_time_string(point[4])
            ts = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").timestamp()
            parsed.append([lon, lat, sog, cog, ts])
        except Exception:
            continue

    if len(parsed) <= 1:
        return None

    arr = np.asarray(parsed, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 4])]
    return arr


def read_csv_auto_encoding(csv_path: str) -> Tuple[pd.DataFrame, str]:
    """Read a CSV file using a small fallback list of common encodings.

    This function exists because public trajectory datasets are frequently produced
    by mixed spreadsheet or desktop tooling, and file encodings can vary across
    environments.
    """
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]
    df = None
    used_encoding = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            used_encoding = enc
            break
        except UnicodeDecodeError:
            continue
        except Exception:
            continue

    if df is None:
        raise ValueError("Failed to read the CSV file with the supported encodings.")

    df.columns = [str(c).strip() for c in df.columns]
    return df, used_encoding


def resolve_source_column(df: pd.DataFrame, source_name: str) -> str:
    """Resolve the actual column name for a requested source.

    The helper supports the common `radar`/`rader` naming inconsistency found in
    some historical data files.
    """
    source_name = str(source_name).strip()
    actual_columns = list(df.columns)

    if source_name in actual_columns:
        return source_name
    if source_name == "radar" and "rader" in actual_columns:
        return "rader"
    raise ValueError(f"Column '{source_name}' does not exist. Available columns: {actual_columns}")


def load_tracks_from_csv_raw_single_source(csv_path: str, source_name: str) -> List[np.ndarray]:
    """Load all valid raw trajectories from one source column of a CSV file.

    Parameters
    ----------
    csv_path:
        Path to the raw CSV file.
    source_name:
        Name of the trajectory source column, for example `AIS`, `radar`, or `bd`.

    Returns
    -------
    list of numpy.ndarray
        A list of raw trajectory arrays with shape `(M_i, 5)`.
    """
    df, _ = read_csv_auto_encoding(csv_path)
    col = resolve_source_column(df, source_name)

    all_tracks = []
    for _, row in df.iterrows():
        raw_arr = parse_track_cell_raw(row[col])
        if raw_arr is not None and len(raw_arr) > 1:
            all_tracks.append(raw_arr.astype(np.float32))
    return all_tracks
