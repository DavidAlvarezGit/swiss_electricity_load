from pathlib import Path
from io import BytesIO

import pandas as pd
import requests

SWISSGRID_URL_TEMPLATE = (
    "https://www.swissgrid.ch/content/dam/dataimport/energy-statistic/"
    "EnergieUebersichtCH-{year}.{ext}"
)


def read_correct_sheet(file_obj):
    """Read the Swissgrid time-series sheet if present, else first sheet."""
    xls = pd.ExcelFile(file_obj)
    for sheet_name in xls.sheet_names:
        name = sheet_name.lower().replace(" ", "")
        if "zeitreihen" in name:
            return pd.read_excel(file_obj, sheet_name=sheet_name, header=0)
    return pd.read_excel(file_obj, sheet_name=0, header=0)


def get_year_from_filename(file_path):
    """Try to read a 4-digit year from a file name."""
    text = file_path.stem
    number = ""
    for ch in text:
        if ch.isdigit():
            number += ch
            if len(number) == 4:
                year = int(number)
                if 1990 <= year <= 2100:
                    return year
        else:
            number = ""
    return None


def download_swissgrid_year(year, timeout=90):
    """Download one year from Swissgrid (.xlsx first, then .xls)."""
    for ext in ["xlsx", "xls"]:
        url = SWISSGRID_URL_TEMPLATE.format(year=year, ext=ext)
        response = requests.get(url, timeout=timeout)
        if response.status_code == 404:
            continue
        response.raise_for_status()
        if response.content:
            return read_correct_sheet(BytesIO(response.content))
    raise ValueError(f"No Swissgrid file available for year {year}")


def find_datetime_column(df):
    """Pick the column that looks most like datetime."""
    candidate_cols = [df.columns[0]]
    for col in df.columns:
        lower = str(col).lower()
        if "date" in lower or "time" in lower or lower.startswith("unnamed"):
            if col not in candidate_cols:
                candidate_cols.append(col)

    best_col = candidate_cols[0]
    best_parsed = pd.to_datetime(df[best_col], errors="coerce", format="mixed")
    best_ratio = best_parsed.notna().mean()

    for col in candidate_cols[1:]:
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        ratio = parsed.notna().mean()
        if ratio > best_ratio:
            best_col = col
            best_parsed = parsed
            best_ratio = ratio

    if best_ratio < 0.95:
        raise ValueError("Could not find a reliable datetime column")

    return best_col, best_parsed


def build_swissgrid_dataset(
    raw_dir,
    output_csv,
    start_year=2009,
    end_year=None,
    download_missing_years=False,
):
    """Build one clean quarter-hour Swissgrid CSV."""
    raw_dir = Path(raw_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(list(raw_dir.glob("*.xls")) + list(raw_dir.glob("*.xlsx")))
    if not files:
        raise ValueError(f"No Excel files found in {raw_dir}")

    # Load local files by year.
    local_by_year = {}
    for file_path in files:
        year = get_year_from_filename(file_path)
        if year is not None:
            local_by_year[year] = read_correct_sheet(file_path)

    if not local_by_year:
        raise ValueError(f"No year-coded Swissgrid files found in {raw_dir}")

    if end_year is None:
        end_year = max(local_by_year.keys())

    frames = []
    base_columns = None

    for year in range(start_year, end_year + 1):
        if year in local_by_year:
            frame = local_by_year[year]
        elif download_missing_years:
            frame = download_swissgrid_year(year)
        else:
            continue

        if base_columns is None:
            base_columns = frame.columns

        frame = frame.reindex(columns=base_columns)
        frames.append(frame)

    if not frames:
        raise ValueError("No Swissgrid data found for the requested year range")

    merged = pd.concat(frames, ignore_index=True)

    datetime_col, datetime_values = find_datetime_column(merged)
    merged[datetime_col] = datetime_values
    merged = merged.dropna(subset=[datetime_col])
    merged = merged.rename(columns={datetime_col: "timestamp"})
    merged = merged.sort_values("timestamp")
    merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
    merged = merged.reset_index(drop=True)

    merged.to_csv(output_csv, index=False)
    return merged
