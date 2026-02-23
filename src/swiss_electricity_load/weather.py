from datetime import datetime, timedelta
from pathlib import Path
import time

import pandas as pd
import requests

CITIES = {
    "zurich": (47.3769, 8.5417),
    "geneva": (46.2044, 6.1432),
    "basel": (47.5596, 7.5886),
    "bern": (46.9480, 7.4474),
    "lausanne": (46.5197, 6.6323),
    "lugano": (46.0037, 8.9511),
}

WEIGHTS = {
    "zurich": 0.30,
    "geneva": 0.18,
    "basel": 0.15,
    "bern": 0.15,
    "lausanne": 0.12,
    "lugano": 0.10,
}

NASA_URL = (
    "https://power.larc.nasa.gov/api/temporal/hourly/point"
    "?parameters=T2M"
    "&community=RE"
    "&longitude={lon}"
    "&latitude={lat}"
    "&start={start}"
    "&end={end}"
    "&format=JSON"
)


def fetch_json_with_retry(url, timeout=60, retries=3, wait_seconds=2):
    """Simple retry wrapper for API calls."""
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(wait_seconds)
    raise RuntimeError(f"Request failed after {retries} tries: {url}") from last_error


def fetch_weather_hourly(start_year=2009, end_date=None):
    """Fetch weather and create weighted features at hourly frequency."""
    if end_date is None:
        end_date = datetime.now().date() - timedelta(days=1)

    city_frames = []

    for city, (lat, lon) in CITIES.items():
        year_frames = []

        for year in range(start_year, end_date.year + 1):
            start = f"{year}0101"
            end = f"{year}1231"

            if year == end_date.year:
                end = end_date.strftime("%Y%m%d")

            url = NASA_URL.format(lon=lon, lat=lat, start=start, end=end)
            payload = fetch_json_with_retry(url)

            values = payload.get("properties", {}).get("parameter", {}).get("T2M", {})
            if not values:
                continue

            year_df = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(list(values.keys()), format="%Y%m%d%H"),
                    city: list(values.values()),
                }
            )
            year_df[city] = year_df[city].replace(-999, pd.NA)
            year_frames.append(year_df)

        if not year_frames:
            raise ValueError(f"No weather data fetched for city: {city}")

        city_df = pd.concat(year_frames, ignore_index=True)
        city_frames.append(city_df)

    weather = city_frames[0]
    for city_df in city_frames[1:]:
        weather = weather.merge(city_df, on="timestamp", how="inner")

    for city in CITIES:
        weather[city] = pd.to_numeric(weather[city], errors="coerce")

    weather = weather.dropna().sort_values("timestamp").reset_index(drop=True)

    weather["temp_weighted"] = sum(weather[city] * WEIGHTS[city] for city in CITIES)
    weather["HDH"] = (18 - weather["temp_weighted"]).clip(lower=0)
    weather["CDH"] = (weather["temp_weighted"] - 22).clip(lower=0)
    weather["temp_72h"] = weather["temp_weighted"].rolling(72, min_periods=1).mean()
    weather["extreme_cold"] = (weather["temp_weighted"] < -5).astype(int)

    return weather


def align_weather_to_timestamps(weather_hourly, timestamps):
    """Use forward-fill to map hourly weather onto quarter-hour timestamps.

    Only align up to the last available weather timestamp to avoid extending
    beyond fetched data.
    """
    weather_hourly = weather_hourly.sort_values("timestamp").set_index("timestamp")

    target = pd.to_datetime(pd.Series(timestamps))
    target = target.dropna().sort_values().drop_duplicates()
    last_weather_ts = weather_hourly.index.max()
    if pd.notna(last_weather_ts):
        target = target[target <= last_weather_ts]

    aligned = weather_hourly.reindex(target, method="ffill")
    aligned = aligned.reset_index().rename(columns={"index": "timestamp"})
    return aligned


def build_weather_hourly_dataset(output_csv, start_year=2009, end_date=None):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    weather = fetch_weather_hourly(start_year=start_year, end_date=end_date)
    weather.to_csv(output_csv, index=False)
    return weather
