import argparse
from pathlib import Path

from swiss_electricity_load.quality import (
    validate_model_input,
    validate_swissgrid_dataset,
    validate_weather_hourly_dataset,
    validate_weather_quarter_hourly_dataset,
)
from swiss_electricity_load.swissgrid import build_swissgrid_dataset
from swiss_electricity_load.weather import build_weather_hourly_dataset, align_weather_to_timestamps


def run_pipeline(
    raw_dir,
    processed_dir,
    swissgrid_start_year=2009,
    swissgrid_end_year=None,
    download_missing_swissgrid_years=False,
    weather_start_year=2009,
    skip_weather=False,
):
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    swissgrid_path = processed_dir / "swissgrid_quarter_hourly.csv"
    swissgrid = build_swissgrid_dataset(
        raw_dir=raw_dir,
        output_csv=swissgrid_path,
        start_year=swissgrid_start_year,
        end_year=swissgrid_end_year,
        download_missing_years=download_missing_swissgrid_years,
    )
    # Quality checks (fail fast if data is broken)
    validate_swissgrid_dataset(swissgrid)
    print(f"Swissgrid rows: {len(swissgrid):,} -> {swissgrid_path}")

    if skip_weather:
        return

    weather_hourly_path = processed_dir / "weather_hourly.csv"
    weather_hourly = build_weather_hourly_dataset(
        output_csv=weather_hourly_path,
        start_year=weather_start_year,
    )
    # Quality checks on raw weather time series
    validate_weather_hourly_dataset(weather_hourly)
    print(f"Weather hourly rows: {len(weather_hourly):,} -> {weather_hourly_path}")

    weather_qh = align_weather_to_timestamps(weather_hourly, swissgrid["timestamp"])
    # Quality checks after resampling/alignment
    validate_weather_quarter_hourly_dataset(weather_qh)
    weather_qh_path = processed_dir / "weather_quarter_hourly.csv"
    weather_qh.to_csv(weather_qh_path, index=False)
    print(f"Weather quarter-hourly rows: {len(weather_qh):,} -> {weather_qh_path}")

    model_input = swissgrid.merge(weather_qh, on="timestamp", how="left")
    # Final quality checks before saving model input
    validate_model_input(model_input)
    model_path = processed_dir / "model_input_quarter_hourly.csv"
    model_input.to_csv(model_path, index=False)

    missing_weather = int(model_input["temp_weighted"].isna().sum())
    print(f"Model input rows: {len(model_input):,} -> {model_path}")
    print(f"Rows missing weather features after merge: {missing_weather:,}")


def build_parser():
    parser = argparse.ArgumentParser(description="Build Swiss electricity load and weather datasets")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--swissgrid-start-year", type=int, default=2009)
    parser.add_argument("--swissgrid-end-year", type=int, default=None)
    parser.add_argument("--download-missing-swissgrid-years", action="store_true")
    parser.add_argument("--weather-start-year", type=int, default=2009)
    parser.add_argument("--skip-weather", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    run_pipeline(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        swissgrid_start_year=args.swissgrid_start_year,
        swissgrid_end_year=args.swissgrid_end_year,
        download_missing_swissgrid_years=args.download_missing_swissgrid_years,
        weather_start_year=args.weather_start_year,
        skip_weather=args.skip_weather,
    )


if __name__ == "__main__":
    main()
