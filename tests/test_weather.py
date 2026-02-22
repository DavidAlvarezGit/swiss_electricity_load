import pandas as pd

from swiss_electricity_load.weather import align_weather_to_timestamps


def test_align_weather_to_timestamps_forward_fills_hourly_data():
    weather_hourly = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 01:00:00"]),
            "temp_weighted": [10.0, 14.0],
            "HDH": [8.0, 4.0],
            "CDH": [0.0, 0.0],
            "temp_72h": [10.0, 12.0],
            "extreme_cold": [0, 0],
        }
    )

    target_timestamps = pd.to_datetime(
        ["2026-01-01 00:00:00", "2026-01-01 00:15:00", "2026-01-01 00:30:00", "2026-01-01 01:00:00"]
    )
    aligned = align_weather_to_timestamps(weather_hourly, target_timestamps)

    assert len(aligned) == 4
    assert aligned.iloc[1]["temp_weighted"] == 10.0
    assert aligned.iloc[3]["temp_weighted"] == 14.0
