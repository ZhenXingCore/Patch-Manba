"""Patch Dataset Builder.

This package provides modular utilities for building patch-based maritime trajectory
forecasting datasets under both pseudo-recursive and true recursive training modes.
"""

from .builders import (
    build_patch_forecast_dataset_from_csv_single_source,
    build_patch_forecast_dataset_from_raw_tracks,
    build_patch_forecast_dataset_from_raw_tracks_pseudo,
    build_patch_rollout_dataset_from_raw_tracks,
    build_and_save_source_multiscale,
)
from .serialization import (
    load_saved_dataset_from_csv,
    load_rollout_dataset_from_csv,
    save_samples_to_csv,
    save_rollout_samples_to_csv,
)
from .utils import default_window_configs

__all__ = [
    "build_patch_forecast_dataset_from_csv_single_source",
    "build_patch_forecast_dataset_from_raw_tracks",
    "build_patch_forecast_dataset_from_raw_tracks_pseudo",
    "build_patch_rollout_dataset_from_raw_tracks",
    "build_and_save_source_multiscale",
    "load_saved_dataset_from_csv",
    "load_rollout_dataset_from_csv",
    "save_samples_to_csv",
    "save_rollout_samples_to_csv",
    "default_window_configs",
]
