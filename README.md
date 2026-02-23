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

By default, training uses a **safe feature policy** (lags + calendar + weather) to reduce leakage risk.
If you explicitly want all numeric columns (not recommended for forecasting), add:

```bash
poetry run swiss-load-train --allow-contemporaneous-features
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


## Streamlit Dashboard
Launch dashboard for metrics, predictions, CV, and inference outputs:

```bash
poetry run swiss-load-dashboard
```

You can also point to a custom processed folder:

```bash
poetry run python -m swiss_electricity_load.dashboard --processed-dir data/processed
```

## Azure Deployment (Container Apps)
This repo includes a production-ready container and deployment automation for the Streamlit dashboard.

### 1) Local one-command Azure deploy
Use the deployment script:

```powershell
.\deploy\azure\deploy.ps1 -AcrName "<your-unique-acr-name>"
```

Useful optional flags:

```powershell
.\deploy\azure\deploy.ps1 `
  -AcrName "<your-unique-acr-name>" `
  -ResourceGroup "rg-swiss-load" `
  -Location "westeurope" `
  -ContainerAppsEnv "cae-swiss-load" `
  -ContainerAppName "swiss-load-dashboard" `
  -ImageTag "latest" `
  -ProcessedDir "data/processed"
```

The script will:
- create/update Resource Group
- create/update ACR
- build and push the dashboard image with `az acr build`
- create/update Container Apps environment
- create/update the Container App and print the public URL

### 2) GitHub Actions auto-deploy on `main`
Workflow file:
- `.github/workflows/deploy-azure-dashboard.yml`

Set these GitHub repository **secrets**:
- `AZURE_CLIENT_ID`
- `AZURE_TENANT_ID`
- `AZURE_SUBSCRIPTION_ID`

Set these GitHub repository **variables**:
- `AZURE_RESOURCE_GROUP`
- `AZURE_LOCATION`
- `AZURE_ACR_NAME`
- `AZURE_CONTAINERAPPS_ENV`
- `AZURE_CONTAINER_APP_NAME`

Notes:
- Use OIDC with `azure/login` for secure auth from GitHub Actions.
- The workflow deploys on push to `main` and supports manual `workflow_dispatch`.

### 3) Data path in Azure
The container reads dashboard artifacts from `PROCESSED_DIR` (default: `data/processed`).
If your processed files are external (recommended), mount storage and set:

```text
PROCESSED_DIR=/mounted/path/to/processed
```
