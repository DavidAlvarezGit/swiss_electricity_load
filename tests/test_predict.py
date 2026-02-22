import json

import pandas as pd

from swiss_electricity_load.predict import run_inference


def test_run_inference_writes_outputs(monkeypatch, tmp_path):
    input_path = tmp_path / "training_dataset_quarter_hourly.parquet"
    report_path = tmp_path / "model_report.json"
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="15min"),
            "f1": [1, 2, 3, 4, 5, 6],
            "f2": [10, 10, 10, 10, 10, 10],
        }
    )
    df.to_parquet(input_path, index=False)

    report = {"feature_columns": ["f1", "f2"]}
    report_path.write_text(json.dumps(report), encoding="utf-8")

    # Create empty model files so path checks pass.
    (model_dir / "lightgbm_point.txt").write_text("x", encoding="utf-8")
    (model_dir / "lightgbm_quantile_q10.txt").write_text("x", encoding="utf-8")
    (model_dir / "lightgbm_quantile_q50.txt").write_text("x", encoding="utf-8")
    (model_dir / "lightgbm_quantile_q90.txt").write_text("x", encoding="utf-8")

    class FakeBooster:
        def __init__(self, value):
            self.value = value

        def predict(self, x):
            return [self.value] * len(x)

    def fake_loader(path):
        name = str(path)
        if "q10" in name:
            return FakeBooster(1.0)
        if "q50" in name:
            return FakeBooster(2.0)
        if "q90" in name:
            return FakeBooster(3.0)
        return FakeBooster(2.5)

    monkeypatch.setattr("swiss_electricity_load.predict.load_lightgbm_booster", fake_loader)

    preds, csv_path, parquet_path = run_inference(
        input_path=input_path,
        report_path=report_path,
        model_dir=model_dir,
        output_dir=tmp_path,
        output_prefix="preds",
        last_n=4,
    )

    assert len(preds) == 4
    assert "lightgbm_pred" in preds.columns
    assert "lightgbm_q10" in preds.columns
    assert "lightgbm_q50" in preds.columns
    assert "lightgbm_q90" in preds.columns
    assert csv_path.exists()
    assert parquet_path.exists()
