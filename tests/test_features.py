import pandas as pd

from swiss_electricity_load.features import (
    build_training_dataset,
    find_model_input_file,
)


def test_find_model_input_file_prefers_parquet(tmp_path):
    parquet_path = tmp_path / "model_input_quarter_hourly.parquet"
    csv_path = tmp_path / "model_input_quarter_hourly.csv"

    pd.DataFrame({"timestamp": ["2026-01-01 00:00:00"], "load": [1.0]}).to_parquet(parquet_path, index=False)
    pd.DataFrame({"timestamp": ["2026-01-01 00:00:00"], "load": [1.0]}).to_csv(csv_path, index=False)

    resolved = find_model_input_file(tmp_path)
    assert resolved == parquet_path


def test_build_training_dataset_adds_calendar_and_lags(tmp_path):
    input_path = tmp_path / "model_input_quarter_hourly.csv"
    output_path = tmp_path / "training_dataset_quarter_hourly.csv"

    timestamps = pd.date_range("2026-01-01 00:00:00", periods=12, freq="15min")
    df = pd.DataFrame({"timestamp": timestamps, "load": range(12), "temp_weighted": [5.0] * 12})
    df.to_csv(input_path, index=False)

    featured = build_training_dataset(
        input_path=input_path,
        output_path=output_path,
        target_column="load",
        lag_steps=(1, 4),
        drop_na_rows=True,
    )

    assert output_path.exists()
    assert "hour" in featured.columns
    assert "day_of_week" in featured.columns
    assert "is_weekend" in featured.columns
    assert "load_lag_1" in featured.columns
    assert "load_lag_4" in featured.columns
    assert len(featured) == 8


def test_build_training_dataset_writes_parquet(tmp_path):
    input_path = tmp_path / "model_input_quarter_hourly.parquet"
    output_path = tmp_path / "training_dataset_quarter_hourly.parquet"

    timestamps = pd.date_range("2026-01-01 00:00:00", periods=6, freq="15min")
    df = pd.DataFrame({"timestamp": timestamps, "load": [10, 11, 12, 13, 14, 15]})
    df.to_parquet(input_path, index=False)

    build_training_dataset(
        input_path=input_path,
        output_path=output_path,
        target_column="load",
        lag_steps=(1,),
        drop_na_rows=True,
    )

    assert output_path.exists()
