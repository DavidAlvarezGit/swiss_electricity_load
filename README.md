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


## Train First Models
Train two beginner-friendly models on `training_dataset_quarter_hourly.*`:
- baseline model: uses previous value (lag-1)
- advanced model: simple linear regression

```bash
poetry run swiss-load-train
```

Optional quantile interval (10%-90% by default):

```bash
poetry run swiss-load-train --quantile --alpha 0.1
```

Train with LightGBM (SOTA tabular model):

```bash
poetry run swiss-load-train --use-lightgbm
```

Train LightGBM quantile models (P10/P50/P90):

```bash
poetry run swiss-load-train --use-lightgbm --use-lightgbm-quantiles
```

Use rolling time-series cross-validation (example: 3 folds):

```bash
poetry run swiss-load-train --use-lightgbm --cv-folds 3 --cv-test-fraction 0.1 --cv-min-train-fraction 0.4
```

Model files are saved to `data/processed/models/` by default.
Disable saving model artifacts:

```bash
poetry run swiss-load-train --use-lightgbm --no-save-models
```


## Predict From Saved Models
Run inference from saved LightGBM models:

```bash
poetry run swiss-load-predict --input-path data/processed/training_dataset_quarter_hourly.parquet --last-n 96
```

## One Command Full Flow
Run everything end-to-end in one command:

```bash
poetry run swiss-load-fullflow
```

Custom full-flow example:

```bash
poetry run swiss-load-fullflow --cv-folds 3 --cv-test-fraction 0.1 --cv-min-train-fraction 0.4 --predict-last-n 96
```
