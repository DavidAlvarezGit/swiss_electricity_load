# Deploy To Azure (Simple)

This uses Azure Container Apps and a container image in Azure Container Registry (ACR).

## 1) Prerequisites

- Azure CLI installed
- Docker Desktop installed and running
- Logged in:

```bash
az login
```

## 2) Set variables

Use PowerShell:

```powershell
$RG = "rg-swiss-load"
$LOC = "westeurope"
$ACR = "acrswissload12345"   # must be globally unique, letters/numbers only
$ENV = "env-swiss-load"
$APP = "app-swiss-load"
$IMG = "swiss-dashboard"
$TAG = "v1"
```

## 3) Build and push image

```powershell
az group create -n $RG -l $LOC
az acr create -g $RG -n $ACR --sku Basic
az acr login -n $ACR
docker build -t "$ACR.azurecr.io/$IMG`:$TAG" .
docker push "$ACR.azurecr.io/$IMG`:$TAG"
```

## 4) Create Container Apps environment

```powershell
az containerapp env create -g $RG -n $ENV -l $LOC
```

## 5) Deploy app

```powershell
az containerapp create `
  -g $RG `
  -n $APP `
  --environment $ENV `
  --image "$ACR.azurecr.io/$IMG`:$TAG" `
  --target-port 8501 `
  --ingress external `
  --registry-server "$ACR.azurecr.io" `
  --query properties.configuration.ingress.fqdn
```

The command returns your public URL.

## 6) Update app later

When you build a new image tag:

```powershell
$TAG = "v2"
docker build -t "$ACR.azurecr.io/$IMG`:$TAG" .
docker push "$ACR.azurecr.io/$IMG`:$TAG"

az containerapp update `
  -g $RG `
  -n $APP `
  --image "$ACR.azurecr.io/$IMG`:$TAG"
```

## Notes

- This setup includes `data/processed` inside the image at build time.
- If you regenerate model/data artifacts, rebuild and redeploy the image.
