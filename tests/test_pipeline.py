import pandas as pd
import pytest

from swiss_electricity_load.pipeline import run_pipeline


def test_run_pipeline_skip_weather_writes_swissgrid(monkeypatch, tmp_path):
    def fake_build_swissgrid_dataset(**kwargs):
        timestamps = pd.date_range("2026-01-01 00:00:00", periods=4, freq="15min")
        df = pd.DataFrame({"timestamp": timestamps, "load": [1, 2, 3, 4]})
        df.to_csv(kwargs["output_csv"], index=False)
        return df

    monkeypatch.setattr("swiss_electricity_load.pipeline.build_swissgrid_dataset", fake_build_swissgrid_dataset)

    run_pipeline(raw_dir=tmp_path, processed_dir=tmp_path, skip_weather=True)

    assert (tmp_path / "swissgrid_quarter_hourly.csv").exists()
    assert not (tmp_path / "weather_hourly.csv").exists()


def test_run_pipeline_raises_if_weather_missing_after_merge(monkeypatch, tmp_path):
    def fake_build_swissgrid_dataset(**kwargs):
        timestamps = pd.date_range("2026-01-01 00:00:00", periods=4, freq="15min")
        df = pd.DataFrame({"timestamp": timestamps, "load": [1, 2, 3, 4]})
        df.to_csv(kwargs["output_csv"], index=False)
        return df

    def fake_build_weather_hourly_dataset(**kwargs):
        timestamps = pd.date_range("2026-01-01 00:00:00", periods=1, freq="1h")
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "temp_weighted": [None],
                "HDH": [0.0],
                "CDH": [0.0],
                "temp_72h": [0.0],
                "extreme_cold": [0],
            }
        )

    def fake_align(weather_hourly, timestamps):
        # Return one row less to force missing weather after merge.
        t = pd.to_datetime(pd.Series(timestamps)).sort_values().reset_index(drop=True)
        t = t.iloc[:-1]
        return pd.DataFrame(
            {
                "timestamp": t,
                "temp_weighted": [1.0] * len(t),
                "HDH": [0.0] * len(t),
                "CDH": [0.0] * len(t),
                "temp_72h": [1.0] * len(t),
                "extreme_cold": [0] * len(t),
            }
        )

    monkeypatch.setattr("swiss_electricity_load.pipeline.build_swissgrid_dataset", fake_build_swissgrid_dataset)
    monkeypatch.setattr("swiss_electricity_load.pipeline.build_weather_hourly_dataset", fake_build_weather_hourly_dataset)
    monkeypatch.setattr("swiss_electricity_load.pipeline.align_weather_to_timestamps", fake_align)

    with pytest.raises(ValueError, match="missing values"):
        run_pipeline(raw_dir=tmp_path, processed_dir=tmp_path, skip_weather=False)
