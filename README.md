# Swiss Electricity Demand Forecasting Platform

Automated forecasting platform for Swiss electricity demand, from data collection to a cloud-hosted executive dashboard.

**Live Demo:** https://swiss-energy-forecast.orangedune-8851111d.germanywestcentral.azurecontainerapps.io/

## What This Project Does
- Builds a 24h-ahead electricity demand forecast for Switzerland.
- Produces point forecasts and prediction intervals (P10/P50/P90).
- Serves results in a business-facing dashboard deployed on Azure.

## Data Ingestion (Automated)
The pipeline is designed to run with minimal manual intervention:
- Ingests Swissgrid source files from local CSV/Excel inputs.
- Automatically processes and standardizes quarter-hourly load/covariate data.
- Fetches weather features from external API sources (NASA POWER).
- Merges historical files + API covariates into a unified modeling dataset.

In short: local source files + API enrichment are handled end-to-end by the pipeline.

## Modeling Pipeline
- Feature engineering: lag, calendar, and weather predictors.
- Models: baseline + LightGBM (point and quantile models).
- Evaluation: baseline vs LightGBM performance tracking.
- Inference: latest forecasts exported for dashboard consumption.

## Dashboard
The 24h dashboard includes:
- Forecast vs actual comparison
- Prediction interval shading (P10-P90)
- Baseline vs LightGBM KPI view
- Variable importance (readable feature names)

The app is hosted in Azure, so it remains accessible even when the local machine is offline.

## Tech Stack
- Language & runtime:
  - Python 3.12
- Data processing:
  - pandas
  - NumPy
- ML & forecasting:
  - scikit-learn (baseline evaluation utilities and metrics)
  - LightGBM (point forecast + quantile models for P10/P50/P90)
- Visualization & app layer:
  - Streamlit (dashboard UI)
- Data sources:
  - Swissgrid historical load/covariate files (CSV/Excel)
  - NASA POWER API (weather covariates)
- Packaging & dependency management:
  - Poetry
  - pip/requirements export for container builds
- Containerization:
  - Docker (Python slim base image)
- Cloud infrastructure:
  - Azure Container Apps (app hosting)
  - Azure Container Registry (image storage)
  - Azure Files (persistent mounted data/artifact storage)
- CI/CD & automation:
  - GitHub Actions (build/deploy workflow)
  - Azure CLI + PowerShell deployment scripts

## Repository Structure
- `src/swiss_electricity_load/pipeline.py`: automated ingestion and orchestration
- `src/swiss_electricity_load/features.py`: feature engineering
- `src/swiss_electricity_load/model.py`: training and evaluation
- `src/swiss_electricity_load/predict.py`: inference
- `src/swiss_electricity_load/dashboard.py`: dashboard app
- `deploy/azure/deploy.ps1`: deployment automation
- `.github/workflows/deploy-azure-dashboard.yml`: CI/CD deployment workflow
