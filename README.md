# swiss_electricity_load

Production data ingestion pipeline for Swissgrid load/covariates and Swiss weather features.

## What is included
- `src/swiss_electricity_load/swissgrid.py`: builds quarter-hourly Swissgrid dataset from local Excel files (optionally downloading missing years).
- `src/swiss_electricity_load/weather.py`: fetches hourly weather from NASA POWER and computes weighted weather features.
- `src/swiss_electricity_load/pipeline.py`: end-to-end orchestrator and CLI.

## Outputs
Running the full pipeline writes:
- `data/processed/swissgrid_quarter_hourly.csv`
- `data/processed/weather_hourly.csv`
- `data/processed/weather_quarter_hourly.csv`
- `data/processed/model_input_quarter_hourly.csv`

## Run
Install dependencies:

```bash
poetry install
```

Run full pipeline:

```bash
poetry run swiss-load-pipeline
```

Run Swissgrid-only build:

```bash
poetry run swiss-load-pipeline --skip-weather
```

Optional arguments:
- `--raw-dir data/raw`
- `--processed-dir data/processed`
- `--swissgrid-start-year 2009`
- `--swissgrid-end-year 2026`
- `--download-missing-swissgrid-years`
- `--weather-start-year 2009`

## Build Training Features
Create a training dataset with calendar and lag features from `model_input_quarter_hourly.*`:

```bash
poetry run python -c "from swiss_electricity_load.features import build_training_dataset; build_training_dataset()"
```

Custom example:

```bash
poetry run python -c "from swiss_electricity_load.features import build_training_dataset; build_training_dataset(input_path='data/processed/model_input_quarter_hourly.parquet', output_path='data/processed/training_dataset_quarter_hourly.parquet', target_column='load', lag_steps=(1,4,96))"
```
