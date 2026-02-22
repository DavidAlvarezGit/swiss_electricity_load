import json

import pandas as pd

from swiss_electricity_load.model import train_models


def test_train_models_creates_outputs(tmp_path):
    input_path = tmp_path / "training_dataset_quarter_hourly.csv"

    timestamps = pd.date_range("2026-01-01 00:00:00", periods=40, freq="15min")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load": list(range(40)),
            "load_lag_1": [0] + list(range(39)),
            "temp_weighted": [5.0] * 40,
            "hour": [t.hour for t in timestamps],
        }
    )
    df.to_csv(input_path, index=False)

    report, predictions = train_models(
        input_path=input_path,
        output_dir=tmp_path,
        target_column="load",
        test_fraction=0.25,
    )

    assert report["target_column"] == "load"
    assert report["n_rows_test"] == 10
    assert "mae" in report["baseline_metrics"]
    assert "mae" in report["linear_metrics"]

    assert (tmp_path / "model_report.json").exists()
    assert (tmp_path / "model_predictions.csv").exists()
    assert (tmp_path / "model_predictions.parquet").exists()

    assert "baseline_pred" in predictions.columns
    assert "linear_pred" in predictions.columns


def test_train_models_quantile_adds_interval_columns(tmp_path):
    input_path = tmp_path / "training_dataset_quarter_hourly.parquet"

    timestamps = pd.date_range("2026-01-01 00:00:00", periods=60, freq="15min")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load": [100 + i for i in range(60)],
            "load_lag_1": [100] + [100 + i for i in range(59)],
            "temp_weighted": [3.0] * 60,
            "hour": [t.hour for t in timestamps],
        }
    )
    df.to_parquet(input_path, index=False)

    report, predictions = train_models(
        input_path=input_path,
        output_dir=tmp_path,
        target_column="load",
        make_quantile_interval=True,
        alpha=0.1,
    )

    assert report["quantile"]["enabled"] is True
    assert "linear_q_lower" in predictions.columns
    assert "linear_q_upper" in predictions.columns

    saved_report = json.loads((tmp_path / "model_report.json").read_text(encoding="utf-8"))
    assert saved_report["quantile"]["alpha"] == 0.1


def test_train_models_with_lightgbm_columns_and_metrics(monkeypatch, tmp_path):
    input_path = tmp_path / "training_dataset_quarter_hourly.csv"

    timestamps = pd.date_range("2026-01-01 00:00:00", periods=30, freq="15min")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load": list(range(30)),
            "load_lag_1": [0] + list(range(29)),
            "temp_weighted": [4.0] * 30,
            "hour": [t.hour for t in timestamps],
        }
    )
    df.to_csv(input_path, index=False)

    def fake_point_model(train_df, test_df, feature_columns, target_column, random_state=42):
        _ = (train_df, feature_columns, target_column, random_state)
        preds = test_df["load_lag_1"].to_numpy(dtype=float)
        return object(), preds

    monkeypatch.setattr("swiss_electricity_load.model.train_lightgbm_point_model", fake_point_model)

    report, predictions = train_models(
        input_path=input_path,
        output_dir=tmp_path,
        target_column="load",
        use_lightgbm=True,
        save_models=False,
    )

    assert "lightgbm_pred" in predictions.columns
    assert report["lightgbm_metrics"] is not None
    assert "mae" in report["lightgbm_metrics"]


def test_train_models_with_lightgbm_quantiles(monkeypatch, tmp_path):
    input_path = tmp_path / "training_dataset_quarter_hourly.parquet"

    timestamps = pd.date_range("2026-01-01 00:00:00", periods=40, freq="15min")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load": [50 + i for i in range(40)],
            "load_lag_1": [50] + [50 + i for i in range(39)],
            "temp_weighted": [2.0] * 40,
            "hour": [t.hour for t in timestamps],
        }
    )
    df.to_parquet(input_path, index=False)

    def fake_quantile_models(train_df, test_df, feature_columns, target_column, quantiles=(0.1, 0.5, 0.9), random_state=42):
        _ = (train_df, feature_columns, target_column, quantiles, random_state)
        base = test_df["load_lag_1"].to_numpy(dtype=float)
        return {0.1: object(), 0.5: object(), 0.9: object()}, {0.1: base - 1.0, 0.5: base, 0.9: base + 1.0}

    monkeypatch.setattr("swiss_electricity_load.model.train_lightgbm_quantile_models", fake_quantile_models)

    report, predictions = train_models(
        input_path=input_path,
        output_dir=tmp_path,
        target_column="load",
        use_lightgbm_quantiles=True,
        save_models=False,
    )

    assert report["lightgbm_quantiles"]["enabled"] is True
    assert "lightgbm_q10" in predictions.columns
    assert "lightgbm_q50" in predictions.columns
    assert "lightgbm_q90" in predictions.columns


def test_train_models_with_cv(monkeypatch, tmp_path):
    input_path = tmp_path / "training_dataset_quarter_hourly.csv"

    timestamps = pd.date_range("2026-01-01 00:00:00", periods=80, freq="15min")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load": list(range(80)),
            "load_lag_1": [0] + list(range(79)),
            "temp_weighted": [6.0] * 80,
            "hour": [t.hour for t in timestamps],
        }
    )
    df.to_csv(input_path, index=False)

    report, _ = train_models(
        input_path=input_path,
        output_dir=tmp_path,
        target_column="load",
        cv_folds=3,
    )

    assert report["time_series_cv"] is not None
    assert report["time_series_cv"]["n_folds"] >= 2
    assert "mae" in report["time_series_cv"]["mean_linear_metrics"]


def test_train_models_saves_lightgbm_artifacts(monkeypatch, tmp_path):
    input_path = tmp_path / "training_dataset_quarter_hourly.parquet"

    timestamps = pd.date_range("2026-01-01 00:00:00", periods=50, freq="15min")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load": [200 + i for i in range(50)],
            "load_lag_1": [200] + [200 + i for i in range(49)],
            "temp_weighted": [1.0] * 50,
            "hour": [t.hour for t in timestamps],
        }
    )
    df.to_parquet(input_path, index=False)

    class FakeBooster:
        def save_model(self, path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("fake model")

    class FakeModel:
        def __init__(self):
            self.booster_ = FakeBooster()

    def fake_point_model(train_df, test_df, feature_columns, target_column, random_state=42):
        _ = (train_df, feature_columns, target_column, random_state)
        preds = test_df["load_lag_1"].to_numpy(dtype=float)
        return FakeModel(), preds

    monkeypatch.setattr("swiss_electricity_load.model.train_lightgbm_point_model", fake_point_model)

    report, _ = train_models(
        input_path=input_path,
        output_dir=tmp_path,
        target_column="load",
        use_lightgbm=True,
        save_models=True,
    )

    assert len(report["saved_models"]) == 1
    assert (tmp_path / "models" / "lightgbm_point.txt").exists()
