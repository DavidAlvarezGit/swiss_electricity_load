# Swiss Electricity Demand Forecasting Platform

Production forecasting platform for Swiss electricity demand, built end-to-end from data ingestion to a cloud-hosted decision dashboard.

**Live Demo:** https://swiss-energy-forecast.orangedune-8851111d.germanywestcentral.azurecontainerapps.io/

## Overview
This project delivers a full machine learning workflow for next-day (24h) electricity demand forecasting:
- ingest and standardize Swissgrid load data and weather covariates,
- build forecasting-ready feature sets,
- train and evaluate baseline and LightGBM models,
- generate probabilistic forecasts (P10/P50/P90),
- serve results in a business-facing dashboard.

## Key Capabilities
- Time-series feature engineering (lag, calendar, weather)
- Point and quantile forecasting with LightGBM
- Baseline-vs-model performance tracking
- Forecast uncertainty visualization (prediction intervals)
- Model driver analysis (feature importance with readable labels)
- Cloud deployment on Azure Container Apps with persistent storage

## System Architecture
- **Data ingestion:** Swissgrid load/covariates + NASA POWER weather
- **Feature layer:** quarter-hourly model input with lag/calendar/weather features
- **Model layer:** baseline + LightGBM point/quantile models
- **Serving layer:** Streamlit dashboard (24h executive view)
- **Infrastructure:** Docker, Azure Container Apps, Azure Files, GitHub Actions CI/CD

## Dashboard
The dashboard is focused on operational readability for stakeholders:
- 24h forecast versus actual
- P10-P90 interval shading with median forecast
- Baseline and LightGBM KPI comparison
- Variable importance with domain-friendly naming

The dashboard is hosted in Azure and remains accessible even when local development machines are offline.

## Repository Structure
- `src/swiss_electricity_load/pipeline.py`: ingestion orchestration
- `src/swiss_electricity_load/features.py`: feature engineering
- `src/swiss_electricity_load/model.py`: model training/evaluation
- `src/swiss_electricity_load/predict.py`: inference from saved models
- `src/swiss_electricity_load/dashboard.py`: Streamlit application
- `deploy/azure/deploy.ps1`: Azure deployment automation
- `.github/workflows/deploy-azure-dashboard.yml`: CI/CD workflow

## Notes
- Primary user interface is the dashboard.
- CLI modules are retained for reproducibility, automation, and diagnostics.
