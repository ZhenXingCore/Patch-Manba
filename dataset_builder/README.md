# Patch Dataset Builder

**Patch Dataset Builder** is a modular, research-oriented preprocessing toolkit for building patch-based maritime trajectory forecasting datasets from raw CSV trajectory records.

This package reorganizes the dataset construction logic into independent modules for:

- raw CSV parsing,
- coordinate and trajectory geometry,
- feature encoding and restoration,
- pseudo-recursive single-step sample construction,
- true recursive rollout sample construction,
- dataset serialization and reload,
- multi-scale prebuilding for single-source experiments.

The code is written in a reviewer-friendly style with explicit function boundaries, detailed docstrings, and stable public entrypoints.

## Package structure

```text
dataset_builder/
├── README.md
├── build_prebuilt_datasets.py
└── patch_dataset_builder/
    ├── __init__.py
    ├── builders.py
    ├── constants.py
    ├── datasets.py
    ├── encoding.py
    ├── geometry.py
    ├── parsing.py
    ├── serialization.py
    └── utils.py
```

## Public entrypoints

The main high-level functions are:

- `build_patch_forecast_dataset_from_csv_single_source(...)`
- `build_patch_forecast_dataset_from_raw_tracks(...)`
- `build_and_save_source_multiscale(...)`
- `load_saved_dataset_from_csv(...)`
- `load_rollout_dataset_from_csv(...)`

## Training modes

This package supports two dataset construction modes:

1. **Pseudo-recursive mode** (`training_mode="pseudo_recursive"`)  
   Each training sample corresponds to one future step. The input may include previously generated fixed-step points, but the label at each step is generated independently from the real trajectory.

2. **Recursive mode** (`training_mode="recursive"`)  
   Each training sample corresponds to a multi-step rollout segment. The sample contains the observed history, the future fixed-step sequence, and per-step encoded labels for true recursive training.

## Expected CSV format

The input CSV should contain one or more source columns, for example:

- `AIS`
- `radar`
- `bd`

Each cell in a source column should store a Python-style list of trajectory points:

```python
[
    [longitude, latitude, sog, cog, "YYYY-mm-dd HH:MM:SS"],
    [longitude, latitude, sog, cog, "YYYY-mm-dd HH:MM:SS"],
    ...
]
```

Example:

```python
[
    [120.1234, 36.1234, 10.5, 85.0, "2021-10-11 11:53:07"],
    [120.1240, 36.1238, 10.7, 86.2, "2021-10-11 11:58:07"],
    [120.1248, 36.1243, 10.8, 86.0, "2021-10-11 12:03:07"],
]
```

## Example

```python
from patch_dataset_builder import build_patch_forecast_dataset_from_csv_single_source

samples, batch_data, dataset = build_patch_forecast_dataset_from_csv_single_source(
    csv_path="data41.csv",
    source_name="AIS",
    input_patch_num=12,
    patch_minutes=15,
    future_step_minutes=5,
    max_future_steps=12,
    training_mode="pseudo_recursive",
)
```

## Design notes

- All future-point construction is performed in raw geographic space before final feature encoding.
- The standard encoded point uses a 10-dimensional interface:

```text
[lon_norm, lat_norm, sog_div10, cog_sin, cog_cos, relative_time_min,
 lon_min_raw, lon_max_raw, lat_min_raw, lat_max_raw]
```

- The rollout representation stores both raw future points and encoded labels so that training and evaluation code can choose the desired supervision interface.

