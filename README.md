# Swiss Electricity Demand Forecasting Platform

Production-grade forecasting project for Swiss electricity load, built end-to-end from ingestion to executive dashboard.

**Live Demo:** https://swiss-energy-forecast.orangedune-8851111d.germanywestcentral.azurecontainerapps.io/

## Project Highlights
- Built a complete ML product workflow: data ingestion, feature engineering, model training, probabilistic inference, and visualization.
- Designed a 24h-ahead forecasting dashboard for business stakeholders (accuracy, uncertainty bands, and model drivers).
- Deployed to Azure Container Apps with containerization, CI/CD, and persistent cloud storage for model artifacts.

## What This Demonstrates (Skills)
- Time-series forecasting (lag features, calendar effects, weather covariates, horizon modeling)
- ML engineering (train/infer separation, model persistence, quantile forecasts, feature importance)
- Data engineering (raw-to-processed pipelines, reproducible datasets, quality checks)
- Cloud deployment (Docker, Azure Container Apps, Azure Files, GitHub Actions OIDC)
- Product thinking (executive-facing dashboard design, uncertainty communication, UX simplification)

## Architecture
- Data ingestion:
  - Swissgrid load and covariates
  - NASA POWER weather + weighted weather features
- Feature layer:
  - Quarter-hourly target dataset
  - Lag + calendar + weather predictors
- Modeling:
  - Baseline and LightGBM point forecast
  - LightGBM quantile models (P10/P50/P90)
- App layer:
  - Streamlit dashboard focused on 24h forecast performance and risk bands
- Deployment:
  - Docker container
  - Azure Container Apps
  - Azure Files mounted as persistent `processed` store

## Dashboard (Business View)
The dashboard presents:
- 24h forecast vs actual
- Prediction interval shading (P10-P90) with median forecast
- Baseline vs LightGBM performance
- Feature importance with readable business labels

Live app URL can be found in your Azure Container App output after deployment.
Because the app is hosted on Azure (not on your local machine), it remains accessible even when your computer is turned off.

## Repository Structure
- `src/swiss_electricity_load/pipeline.py`: ingestion orchestration
- `src/swiss_electricity_load/features.py`: feature engineering
- `src/swiss_electricity_load/model.py`: model training and evaluation
- `src/swiss_electricity_load/predict.py`: inference from saved models
- `src/swiss_electricity_load/dashboard.py`: Streamlit dashboard
- `deploy/azure/deploy.ps1`: Azure deployment automation
- `.github/workflows/deploy-azure-dashboard.yml`: CI/CD deployment workflow

## Notes
- Primary user-facing interface is the dashboard.
- CLI internals remain available for experimentation, but are intentionally not the main usage path.
