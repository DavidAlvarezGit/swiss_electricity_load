from pathlib import Path

import pandas as pd

from swiss_electricity_load.swissgrid import find_datetime_column, get_year_from_filename


def test_get_year_from_filename_reads_year():
    path = Path("EnergieUebersichtCH-2024.xlsx")
    assert get_year_from_filename(path) == 2024


def test_find_datetime_column_chooses_best_column():
    df = pd.DataFrame(
        {
            "bad_col": ["x", "y", "z"],
            "Unnamed: 0": ["2026-01-01 00:00:00", "2026-01-01 00:15:00", "2026-01-01 00:30:00"],
        }
    )
    col, parsed = find_datetime_column(df)
    assert col == "Unnamed: 0"
    assert int(parsed.isna().sum()) == 0
