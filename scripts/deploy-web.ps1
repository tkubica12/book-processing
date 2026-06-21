param(
    [string]$ResourceGroup = "rg-book-processing-site",
    [string]$Location = "westeurope",
    [string]$ContainerAppName = "ca-book-processing-site",
    [string]$EnvironmentName = "cae-book-processing-site",
    [string]$StorageAccountName = "",
    [string]$BlobContainerName = "books",
    [string]$RegistryName = "",
    [string]$Image = "",
    [string]$CustomHostname = "books.tomasonline.net",
    [string]$DnsZoneName = "tomasonline.net",
    [string]$DnsZoneResourceGroup = "rg-base",
    [string]$GitHubOAuthClientId = $env:GITHUB_OAUTH_CLIENT_ID,
    [string]$GitHubOAuthClientSecret = $env:GITHUB_OAUTH_CLIENT_SECRET,
    [string]$GitHubOAuthCookieSecret = $env:GITHUB_OAUTH_COOKIE_SECRET,
    [string]$AllowedGitHubLogin = "tkubica12",
    [string]$AllowedGitHubEmail = "tkubica12@gmail.com",
    [string]$OutputDir = "output",
    [switch]$SkipUpload
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

function Test-AzResource {
    param([scriptblock]$Command)
    try {
        & $Command *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Get-AzCopyPath {
    $existing = Get-Command azcopy -ErrorAction SilentlyContinue
    if ($existing) {
        return $existing.Source
    }

    $downloadDir = Join-Path $env:TEMP "azcopy-book-processing"
    $zipPath = Join-Path $downloadDir "azcopy.zip"
    New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null
    $downloaded = Get-ChildItem $downloadDir -Recurse -Filter azcopy.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($downloaded) {
        return $downloaded.FullName
    }
    Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
    Invoke-WebRequest -Uri "https://aka.ms/downloadazcopy-v10-windows" -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $downloadDir -Force
    $azCopy = Get-ChildItem $downloadDir -Recurse -Filter azcopy.exe | Select-Object -First 1
    if (-not $azCopy) {
        throw "AzCopy download did not contain azcopy.exe"
    }
    return $azCopy.FullName
}

$suffix = (az account show --query id -o tsv).Replace("-", "").Substring(0, 10).ToLowerInvariant()
$tenantId = az account show --query tenantId -o tsv
$env:PYTHONIOENCODING = "utf-8"
$env:AZURE_CORE_NO_COLOR = "true"
if (-not $StorageAccountName) {
    $StorageAccountName = "booksite$suffix"
}
if (-not $RegistryName) {
    $RegistryName = "acrbooksite$suffix"
}
if (-not $Image) {
    $Image = "ghcr.io/tkubica12/book-processing/book-processing-site:latest"
}

Write-Host "Generate local index pages"
uv run --no-sync python -c "from book_processing.site_generator import main; main()"

Write-Host "Create resource group"
az group create --name $ResourceGroup --location $Location --output none

Write-Host "Create private Blob Storage"
if (-not (Test-AzResource { az storage account show --name $StorageAccountName --resource-group $ResourceGroup })) {
    az storage account create `
        --name $StorageAccountName `
        --resource-group $ResourceGroup `
        --location $Location `
        --sku Standard_LRS `
        --kind StorageV2 `
        --allow-blob-public-access false `
        --allow-shared-key-access false `
        --output none
}

az storage container-rm create `
    --resource-group $ResourceGroup `
    --storage-account $StorageAccountName `
    --name $BlobContainerName `
    --public-access off `
    --output none

$storageId = az storage account show `
    --name $StorageAccountName `
    --resource-group $ResourceGroup `
    --query id `
    -o tsv

if ($Image.StartsWith("ghcr.io/")) {
    Write-Host "Use public GHCR image $Image"
    $registryId = ""
} else {
    Write-Host "Create ACR and build web image"
    if (-not (Test-AzResource { az acr show --name $RegistryName --resource-group $ResourceGroup })) {
        az acr create `
            --name $RegistryName `
            --resource-group $ResourceGroup `
            --location $Location `
            --sku Basic `
            --admin-enabled false `
            --output none
    }

    $loginServer = az acr show --name $RegistryName --resource-group $ResourceGroup --query loginServer -o tsv
    $Image = "$loginServer/book-processing-site:latest"
    az acr build `
        --registry $RegistryName `
        --image "book-processing-site:latest" `
        --file Dockerfile.web `
        . `
        --output none
    $registryId = az acr show --name $RegistryName --resource-group $ResourceGroup --query id -o tsv
}

Write-Host "Create Container Apps environment"
if (-not (Test-AzResource { az containerapp env show --name $EnvironmentName --resource-group $ResourceGroup })) {
    az containerapp env create `
        --name $EnvironmentName `
        --resource-group $ResourceGroup `
        --location $Location `
        --output none
}

Write-Host "Deploy web container app"
if (Test-AzResource { az containerapp show --name $ContainerAppName --resource-group $ResourceGroup }) {
    az containerapp delete `
        --name $ContainerAppName `
        --resource-group $ResourceGroup `
        --yes `
        --output none
}

az containerapp create `
    --name $ContainerAppName `
    --resource-group $ResourceGroup `
    --environment $EnvironmentName `
    --image mcr.microsoft.com/azuredocs/containerapps-helloworld:latest `
    --ingress external `
    --target-port 80 `
    --min-replicas 0 `
    --max-replicas 2 `
    --cpu 0.25 `
    --memory 0.5Gi `
    --system-assigned `
    --output none

$principalId = az containerapp show `
    --name $ContainerAppName `
    --resource-group $ResourceGroup `
    --query identity.principalId `
    -o tsv

if ($registryId) {
    az role assignment create --assignee $principalId --role AcrPull --scope $registryId --output none 2>$null
}
az role assignment create --assignee $principalId --role "Storage Blob Data Reader" --scope $storageId --output none 2>$null
Start-Sleep -Seconds 45

if ($registryId) {
    az containerapp registry set `
        --name $ContainerAppName `
        --resource-group $ResourceGroup `
        --server $loginServer `
        --identity system `
        --output none
}

az containerapp update `
    --name $ContainerAppName `
    --resource-group $ResourceGroup `
    --image $Image `
    --set-env-vars "STORAGE_ACCOUNT_NAME=$StorageAccountName" "BLOB_CONTAINER_NAME=$BlobContainerName" `
    --min-replicas 1 `
    --max-replicas 2 `
    --output none

az containerapp ingress update `
    --name $ContainerAppName `
    --resource-group $ResourceGroup `
    --target-port 8000 `
    --transport http `
    --output none

$fqdn = az containerapp show `
    --name $ContainerAppName `
    --resource-group $ResourceGroup `
    --query properties.configuration.ingress.fqdn `
    -o tsv

if ($CustomHostname) {
    Write-Host "Configure custom domain $CustomHostname"
    $verificationId = az containerapp show `
        --name $ContainerAppName `
        --resource-group $ResourceGroup `
        --query properties.customDomainVerificationId `
        -o tsv
    $recordName = $CustomHostname.Substring(0, $CustomHostname.Length - $DnsZoneName.Length - 1)
    az network dns record-set cname set-record `
        --resource-group $DnsZoneResourceGroup `
        --zone-name $DnsZoneName `
        --record-set-name $recordName `
        --cname $fqdn `
        --ttl 300 `
        --output none
    az network dns record-set txt create `
        --resource-group $DnsZoneResourceGroup `
        --zone-name $DnsZoneName `
        --record-set-name "asuid.$recordName" `
        --ttl 300 `
        --output none 2>$null
    az network dns record-set txt add-record `
        --resource-group $DnsZoneResourceGroup `
        --zone-name $DnsZoneName `
        --record-set-name "asuid.$recordName" `
        --value $verificationId `
        --output none 2>$null
    az containerapp hostname add `
        --name $ContainerAppName `
        --resource-group $ResourceGroup `
        --hostname $CustomHostname `
        --output none 2>$null
    az containerapp hostname bind `
        --name $ContainerAppName `
        --resource-group $ResourceGroup `
        --hostname $CustomHostname `
        --environment $EnvironmentName `
        --validation-method CNAME `
        --output none
}

if (-not $GitHubOAuthClientId -or -not $GitHubOAuthClientSecret -or -not $GitHubOAuthCookieSecret) {
    throw "GitHub OAuth settings are required. Set GITHUB_OAUTH_CLIENT_ID, GITHUB_OAUTH_CLIENT_SECRET, and GITHUB_OAUTH_COOKIE_SECRET or pass the matching parameters."
}

$publicBaseUrl = if ($CustomHostname) { "https://$CustomHostname" } else { "https://$fqdn" }
Write-Host "Configure app-level GitHub OAuth for $AllowedGitHubLogin"
az containerapp secret set `
    --name $ContainerAppName `
    --resource-group $ResourceGroup `
    --secrets "github-oauth-client-secret=$GitHubOAuthClientSecret" "github-oauth-cookie-secret=$GitHubOAuthCookieSecret" `
    --output none

az containerapp update `
    --name $ContainerAppName `
    --resource-group $ResourceGroup `
    --set-env-vars `
        "STORAGE_ACCOUNT_NAME=$StorageAccountName" `
        "BLOB_CONTAINER_NAME=$BlobContainerName" `
        "PUBLIC_BASE_URL=$publicBaseUrl" `
        "GITHUB_OAUTH_CLIENT_ID=$GitHubOAuthClientId" `
        "GITHUB_OAUTH_CLIENT_SECRET=secretref:github-oauth-client-secret" `
        "GITHUB_OAUTH_COOKIE_SECRET=secretref:github-oauth-cookie-secret" `
        "ALLOWED_GITHUB_LOGIN=$AllowedGitHubLogin" `
        "ALLOWED_GITHUB_EMAIL=$AllowedGitHubEmail" `
    --output none

Write-Host "Disable Container Apps Easy Auth (GitHub OAuth is enforced in the app)"
az containerapp auth update `
    --name $ContainerAppName `
    --resource-group $ResourceGroup `
    --enabled false `
    --yes `
    --output none

if (-not $SkipUpload) {
    Write-Host "Grant current user upload rights"
    $signedInUserObjectId = az ad signed-in-user show --query id -o tsv
    az role assignment create --assignee $signedInUserObjectId --role "Storage Blob Data Contributor" --scope $storageId --output none 2>$null

    Write-Host "Upload output to private Blob Storage"
    $azCopyPath = Get-AzCopyPath
    $destination = "https://$StorageAccountName.blob.core.windows.net/$BlobContainerName"
    & $azCopyPath sync $OutputDir $destination --recursive=true --delete-destination=true
    if ($LASTEXITCODE -ne 0) {
        throw "AzCopy sync failed with exit code $LASTEXITCODE. Run: $azCopyPath login --tenant-id $tenantId"
    }
    & $azCopyPath set-properties $destination --block-blob-tier=Cold --recursive=true
    if ($LASTEXITCODE -ne 0) {
        throw "AzCopy Cold tier update failed with exit code $LASTEXITCODE."
    }
}

az containerapp update `
    --name $ContainerAppName `
    --resource-group $ResourceGroup `
    --min-replicas 0 `
    --output none

Write-Host ""
Write-Host "DONE"
Write-Host "URL: https://$fqdn"
if ($CustomHostname) {
    Write-Host "Custom URL: https://$CustomHostname"
}
