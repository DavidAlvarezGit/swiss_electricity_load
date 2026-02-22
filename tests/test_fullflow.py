from swiss_electricity_load.fullflow import run_full_flow


def test_run_full_flow_calls_all_steps(monkeypatch, tmp_path):
    called = {
        "pipeline": False,
        "features": False,
        "train": False,
        "predict": False,
    }

    def fake_pipeline(**kwargs):
        called["pipeline"] = True
        _ = kwargs

    def fake_features(**kwargs):
        called["features"] = True
        _ = kwargs
        return None

    def fake_train(**kwargs):
        called["train"] = True
        _ = kwargs
        return {"ok": True}, None

    def fake_predict(**kwargs):
        called["predict"] = True
        _ = kwargs
        return [], tmp_path / "inference_predictions.csv", tmp_path / "inference_predictions.parquet"

    monkeypatch.setattr("swiss_electricity_load.fullflow.run_pipeline", fake_pipeline)
    monkeypatch.setattr("swiss_electricity_load.fullflow.build_training_dataset", fake_features)
    monkeypatch.setattr("swiss_electricity_load.fullflow.train_models", fake_train)
    monkeypatch.setattr("swiss_electricity_load.fullflow.run_inference", fake_predict)

    preds, csv_path, parquet_path = run_full_flow(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        predict_last_n=24,
    )

    assert called["pipeline"] is True
    assert called["features"] is True
    assert called["train"] is True
    assert called["predict"] is True
    assert str(csv_path).endswith("inference_predictions.csv")
    assert str(parquet_path).endswith("inference_predictions.parquet")
    assert preds == []
