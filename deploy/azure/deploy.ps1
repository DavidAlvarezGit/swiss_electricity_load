param(
    [Parameter(Mandatory = $true)]
    [string]$AcrName,
    [string]$ResourceGroup = "rg-swiss-load",
    [string]$Location = "westeurope",
    [string]$ContainerAppsEnv = "cae-swiss-load",
    [string]$ContainerAppName = "swiss-load-dashboard",
    [string]$ImageRepository = "swiss-load-dashboard",
    [string]$ImageTag = "latest",
    [string]$ProcessedDir = "data/processed",
    [switch]$SkipLogin
)

$ErrorActionPreference = "Stop"

function Test-AzCli {
    if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
        throw "Azure CLI (az) is not installed or not in PATH."
    }
}

function Ensure-AzureLogin {
    param([switch]$Skip)
    if ($Skip) {
        return
    }

    try {
        az account show 1>$null 2>$null
    }
    catch {
        Write-Host "No active Azure session detected. Opening az login..."
        az login 1>$null
    }
}

function Ensure-ResourceGroup {
    param(
        [string]$Name,
        [string]$Region
    )
    Write-Host "Ensuring resource group: $Name ($Region)"
    az group create --name $Name --location $Region 1>$null
}

function Ensure-Acr {
    param(
        [string]$Name,
        [string]$Rg
    )
    $existsCount = az acr list --resource-group $Rg --query "[?name=='$Name'] | length(@)" -o tsv
    if ($existsCount -eq "0") {
        Write-Host "Creating Azure Container Registry: $Name"
        az acr create --name $Name --resource-group $Rg --sku Basic --admin-enabled true 1>$null
    }
    else {
        Write-Host "ACR already exists: $Name"
        az acr update --name $Name --resource-group $Rg --admin-enabled true 1>$null
    }
}

function Build-And-PushImage {
    param(
        [string]$Acr,
        [string]$Rg,
        [string]$Repo,
        [string]$Tag
    )
    Write-Host "Building image in ACR: ${Repo}:$Tag"
    az acr build --registry $Acr --resource-group $Rg --image "$Repo`:$Tag" --image "$Repo`:latest" .
}

function Ensure-ContainerAppsEnvironment {
    param(
        [string]$Name,
        [string]$Rg,
        [string]$Region
    )
    $existsCount = az containerapp env list --resource-group $Rg --query "[?name=='$Name'] | length(@)" -o tsv
    if ($existsCount -eq "0") {
        Write-Host "Creating Container Apps environment: $Name"
        az containerapp env create --name $Name --resource-group $Rg --location $Region 1>$null
    }
    else {
        Write-Host "Container Apps environment already exists: $Name"
    }
}

function Ensure-ContainerApp {
    param(
        [string]$Name,
        [string]$Rg,
        [string]$EnvName,
        [string]$Image,
        [string]$RegistryServer,
        [string]$RegistryUsername,
        [string]$RegistryPassword,
        [string]$ProcessedPath
    )
    $existsCount = az containerapp list --resource-group $Rg --query "[?name=='$Name'] | length(@)" -o tsv

    if ($existsCount -eq "0") {
        Write-Host "Creating Container App: $Name"
        az containerapp create `
            --name $Name `
            --resource-group $Rg `
            --environment $EnvName `
            --image $Image `
            --ingress external `
            --target-port 8501 `
            --cpu 1.0 `
            --memory 2.0Gi `
            --registry-server $RegistryServer `
            --registry-username $RegistryUsername `
            --registry-password $RegistryPassword `
            --env-vars "PROCESSED_DIR=$ProcessedPath" 1>$null
    }
    else {
        Write-Host "Updating Container App image: $Name"
        az containerapp registry set `
            --name $Name `
            --resource-group $Rg `
            --server $RegistryServer `
            --username $RegistryUsername `
            --password $RegistryPassword 1>$null

        az containerapp update `
            --name $Name `
            --resource-group $Rg `
            --image $Image `
            --set-env-vars "PROCESSED_DIR=$ProcessedPath" 1>$null
    }
}

Test-AzCli
Ensure-AzureLogin -Skip:$SkipLogin
Ensure-ResourceGroup -Name $ResourceGroup -Region $Location
Ensure-Acr -Name $AcrName -Rg $ResourceGroup
Build-And-PushImage -Acr $AcrName -Rg $ResourceGroup -Repo $ImageRepository -Tag $ImageTag
Ensure-ContainerAppsEnvironment -Name $ContainerAppsEnv -Rg $ResourceGroup -Region $Location

$loginServer = az acr show --name $AcrName --resource-group $ResourceGroup --query "loginServer" -o tsv
$acrUsername = az acr credential show --name $AcrName --resource-group $ResourceGroup --query "username" -o tsv
$acrPassword = az acr credential show --name $AcrName --resource-group $ResourceGroup --query "passwords[0].value" -o tsv
$image = "$loginServer/$ImageRepository`:$ImageTag"

Ensure-ContainerApp `
    -Name $ContainerAppName `
    -Rg $ResourceGroup `
    -EnvName $ContainerAppsEnv `
    -Image $image `
    -RegistryServer $loginServer `
    -RegistryUsername $acrUsername `
    -RegistryPassword $acrPassword `
    -ProcessedPath $ProcessedDir

$url = az containerapp show --name $ContainerAppName --resource-group $ResourceGroup --query "properties.configuration.ingress.fqdn" -o tsv

Write-Host ""
Write-Host "Deployment complete."
Write-Host "Dashboard URL: https://$url"
Write-Host "Processed data path inside container: $ProcessedDir"
