import json

import pandas as pd

from swiss_electricity_load.dashboard import load_report, load_table


def test_dashboard_load_table_prefers_parquet(tmp_path):
    p_parquet = tmp_path / "demo.parquet"
    p_csv = tmp_path / "demo.csv"

    df = pd.DataFrame({"a": [1, 2]})
    df.to_parquet(p_parquet, index=False)
    df.to_csv(p_csv, index=False)

    loaded, used = load_table(tmp_path, "demo")

    assert loaded is not None
    assert str(used).endswith("demo.parquet")
    assert len(loaded) == 2


def test_dashboard_load_report(tmp_path):
    report_path = tmp_path / "model_report.json"
    report = {"n_rows_train": 10, "n_rows_test": 5}
    report_path.write_text(json.dumps(report), encoding="utf-8")

    loaded, used = load_report(tmp_path)

    assert loaded["n_rows_train"] == 10
    assert str(used).endswith("model_report.json")


def test_dashboard_load_report_missing(tmp_path):
    loaded, used = load_report(tmp_path)
    assert loaded is None
    assert used is None
