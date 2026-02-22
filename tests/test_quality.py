import pandas as pd
import pytest

from swiss_electricity_load.quality import (
    validate_model_input,
    validate_swissgrid_dataset,
    validate_weather_hourly_dataset,
)


def test_validate_swissgrid_dataset_passes_for_regular_quarter_hour_data():
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=8, freq="15min")
    df = pd.DataFrame({"timestamp": timestamps, "load": range(8)})
    validate_swissgrid_dataset(df)


def test_validate_swissgrid_dataset_fails_on_duplicate_timestamps():
    timestamps = pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:00:00"])
    df = pd.DataFrame({"timestamp": timestamps, "load": [1, 2]})
    with pytest.raises(ValueError, match="duplicate timestamps"):
        validate_swissgrid_dataset(df)


def test_validate_weather_hourly_dataset_fails_on_missing_feature():
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=3, freq="1h")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temp_weighted": [1.0, None, 3.0],
            "HDH": [0.0, 0.0, 0.0],
            "CDH": [0.0, 0.0, 0.0],
            "temp_72h": [1.0, 2.0, 3.0],
            "extreme_cold": [0, 0, 0],
        }
    )
    with pytest.raises(ValueError, match="missing values"):
        validate_weather_hourly_dataset(df)


def test_validate_model_input_fails_when_weather_columns_missing():
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=2, freq="15min")
    df = pd.DataFrame({"timestamp": timestamps, "load": [10, 12]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_model_input(df)
