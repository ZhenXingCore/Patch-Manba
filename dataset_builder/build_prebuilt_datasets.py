"""Example script for exporting prebuilt dataset CSV files.

This script is intentionally small and explicit. It demonstrates how external users
can call the modular dataset builder from a standalone entry file, which is a common
pattern in open-source research repositories.
"""

from patch_dataset_builder import build_and_save_source_multiscale, default_window_configs


if __name__ == "__main__":
    summary_df = build_and_save_source_multiscale(
        csv_path="data41.csv",
        output_dir="prebuilt_source_csv",
        source_name="AIS",
        window_configs=default_window_configs(),
        strict=False,
        pad_value=0.0,
        future_step_minutes=5,
        sample_stride_minutes=5,
        min_total_input_points=1,
        max_future_steps=12,
        training_mode="pseudo_recursive",
    )
    print(summary_df)
