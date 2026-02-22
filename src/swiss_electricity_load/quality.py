import pandas as pd


WEATHER_FEATURE_COLUMNS = ["temp_weighted", "HDH", "CDH", "temp_72h", "extreme_cold"]


def require_columns(df, columns, dataset_name):
    """Stop if required columns are missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}. "
            "Please check your input data and column names."
        )


def require_no_null_timestamps(df, dataset_name):
    """Stop if any timestamp is empty/null."""
    require_columns(df, ["timestamp"], dataset_name)
    null_count = int(df["timestamp"].isna().sum())
    if null_count > 0:
        raise ValueError(
            f"{dataset_name} has {null_count} null timestamps. "
            "Every row must have a valid timestamp."
        )


def require_no_duplicate_timestamps(df, dataset_name):
    """Stop if the same timestamp appears more than once."""
    require_columns(df, ["timestamp"], dataset_name)
    duplicate_count = int(df["timestamp"].duplicated().sum())
    if duplicate_count > 0:
        raise ValueError(
            f"{dataset_name} has {duplicate_count} duplicate timestamps. "
            "Each timestamp should be unique."
        )


def require_regular_frequency(df, dataset_name, expected_frequency):
    """Stop if timestamps are not continuous at the expected frequency."""
    require_columns(df, ["timestamp"], dataset_name)
    if len(df) < 2:
        return

    timestamps = pd.to_datetime(df["timestamp"]).sort_values().drop_duplicates()
    full_index = pd.date_range(start=timestamps.iloc[0], end=timestamps.iloc[-1], freq=expected_frequency)
    missing = full_index.difference(timestamps)
    if len(missing) > 0:
        raise ValueError(
            f"{dataset_name} has missing timestamps for frequency {expected_frequency}. "
            f"Missing count: {len(missing)}. "
            "There are gaps in time."
        )


def require_no_missing_values(df, columns, dataset_name):
    """Stop if any required feature column contains missing values."""
    require_columns(df, columns, dataset_name)
    for col in columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            raise ValueError(
                f"{dataset_name} has {null_count} missing values in column '{col}'. "
                "Fill or fix missing values before modeling."
            )


def validate_swissgrid_dataset(df):
    """Checks for Swissgrid quarter-hour dataset."""
    require_no_null_timestamps(df, "swissgrid")
    require_no_duplicate_timestamps(df, "swissgrid")
    require_regular_frequency(df, "swissgrid", "15min")


def validate_weather_hourly_dataset(df):
    """Checks for weather hourly dataset."""
    require_no_null_timestamps(df, "weather_hourly")
    require_no_duplicate_timestamps(df, "weather_hourly")
    require_regular_frequency(df, "weather_hourly", "1h")
    require_no_missing_values(df, WEATHER_FEATURE_COLUMNS, "weather_hourly")


def validate_weather_quarter_hourly_dataset(df):
    """Checks for weather quarter-hour dataset."""
    require_no_null_timestamps(df, "weather_quarter_hourly")
    require_no_duplicate_timestamps(df, "weather_quarter_hourly")
    require_regular_frequency(df, "weather_quarter_hourly", "15min")
    require_no_missing_values(df, WEATHER_FEATURE_COLUMNS, "weather_quarter_hourly")


def validate_model_input(df):
    """Checks for final merged model input dataset."""
    require_no_null_timestamps(df, "model_input")
    require_no_duplicate_timestamps(df, "model_input")
    require_regular_frequency(df, "model_input", "15min")
    require_no_missing_values(df, WEATHER_FEATURE_COLUMNS, "model_input")
