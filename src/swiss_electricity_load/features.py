from pathlib import Path

import pandas as pd


def find_model_input_file(input_path=None):
    """
    Resolve input file path for model input data.

    Priority:
    1) explicit file path (if provided)
    2) explicit directory path + expected file names
    3) default data/processed/model_input_quarter_hourly.{parquet,csv}
    """
    if input_path is None:
        base_dir = Path("data/processed")
    else:
        candidate = Path(input_path)
        if candidate.is_file():
            return candidate
        base_dir = candidate

    parquet_file = base_dir / "model_input_quarter_hourly.parquet"
    csv_file = base_dir / "model_input_quarter_hourly.csv"

    if parquet_file.exists():
        return parquet_file
    if csv_file.exists():
        return csv_file

    raise FileNotFoundError(
        "Could not find model_input_quarter_hourly.parquet or model_input_quarter_hourly.csv"
    )


def load_table(path):
    """Load CSV or Parquet based on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def pick_target_column(df, target_column=None):
    """
    Pick target column for lag features.

    If target_column is given, use it.
    Else choose the first numeric column that is not 'timestamp'.
    """
    if target_column is not None:
        if target_column not in df.columns:
            raise ValueError(f"Target column not found: {target_column}")
        return target_column

    numeric_cols = []
    for col in df.columns:
        if col == "timestamp":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    if not numeric_cols:
        raise ValueError("No numeric column found to use as target")

    return numeric_cols[0]


def add_calendar_features(df):
    """Add simple calendar features from timestamp."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])

    out["hour"] = out["timestamp"].dt.hour
    out["minute"] = out["timestamp"].dt.minute
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["day_of_month"] = out["timestamp"].dt.day
    out["month"] = out["timestamp"].dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    return out


def add_lag_features(df, target_column, lag_steps):
    """Add lag features for the chosen target column."""
    out = df.copy()
    for lag in lag_steps:
        out[f"{target_column}_lag_{lag}"] = out[target_column].shift(lag)
    return out


def save_table(df, output_path):
    """Save output as CSV or Parquet based on extension."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError("Output file must end with .csv or .parquet")


def build_training_dataset(
    input_path=None,
    output_path=None,
    target_column=None,
    lag_steps=(1, 4, 96),
    drop_na_rows=True,
):
    """
    Build training dataset with calendar + lag features.

    Default lag_steps for quarter-hour data:
    - 1: previous 15 minutes
    - 4: previous 1 hour
    - 96: previous 24 hours
    """
    source_path = find_model_input_file(input_path)
    df = load_table(source_path)

    if "timestamp" not in df.columns:
        raise ValueError("Input dataset must contain a 'timestamp' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    target = pick_target_column(df, target_column=target_column)
    featured = add_calendar_features(df)
    featured = add_lag_features(featured, target_column=target, lag_steps=lag_steps)

    if drop_na_rows:
        featured = featured.dropna().reset_index(drop=True)

    if output_path is None:
        output_path = Path("data/processed/training_dataset_quarter_hourly.parquet")

    save_table(featured, output_path)
    return featured
