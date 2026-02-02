# =========================================
# deploy-and-smoke-test.ps1
# =========================================

# ------------------------------
# Configuration
# ------------------------------
$ImageName = "cats-vs-dogs"
$Tag = "latest"
$ContainerName = "cats-vs-dogs-service"
$Registry = ""   # Set to "localhost:5000" if using a local registry
$ServiceURL = "http://localhost:8000"
$MaxRetries = 12
$Delay = 5  # seconds between retries

# ------------------------------
# Step 1: Build Docker image
# ------------------------------
Write-Host "Building Docker image..."
# Dockerfile is in parent folder of scripts/
docker build -t "${ImageName}:${Tag}" -f "..\Dockerfile" ..

# ------------------------------
# Step 2: Optional - Push to local registry
# ------------------------------
if ($Registry -ne "") {
    Write-Host "Tagging image for local registry..."
    docker tag "${ImageName}:${Tag}" "${Registry}/${ImageName}:${Tag}"
    
    Write-Host "Pushing image to local registry..."
    docker push "${Registry}/${ImageName}:${Tag}"

    $ImageToRun = "${Registry}/${ImageName}:${Tag}"
} else {
    $ImageToRun = "${ImageName}:${Tag}"
}

# ------------------------------
# Step 3: Stop & remove existing container
# ------------------------------
$existing = docker ps -aq -f "name=$ContainerName"
if ($existing) {
    Write-Host "Stopping and removing existing container..."
    docker rm -f $ContainerName
}

# ------------------------------
# Step 4: Start new container
# ------------------------------
Write-Host "Starting new container..."
docker run -d -p 8000:8000 `
    -v "${PWD}/logs:/app/logs" `
    -v "${PWD}/predictions:/app/predictions" `
    --name $ContainerName `
    $ImageToRun

# ------------------------------
# Step 5: Wait for health endpoint
# ------------------------------
$HealthOK = $false
for ($i = 0; $i -lt $MaxRetries; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri "$ServiceURL/health" -UseBasicParsing -TimeoutSec 5
        if ($resp.StatusCode -eq 200) {
            Write-Host "Health endpoint OK"
            $HealthOK = $true
            break
        }
    } catch {}
    Write-Host "Waiting for container to be ready..."
    Start-Sleep -Seconds $Delay
}

if (-not $HealthOK) {
    Write-Host "Health check failed. Exiting."
    exit 1
}

# ------------------------------
# Step 6: Smoke test - single image prediction using curl.exe (robust)
# ------------------------------
$ImageFolder = Join-Path $PSScriptRoot "..\data\raw\PetImages\Cat"
$ImageFolder = Resolve-Path $ImageFolder
$ImagePath = Get-ChildItem -Path $ImageFolder -File | Select-Object -First 1

if (-not $ImagePath) {
    Write-Host "No images found in $ImageFolder"
    exit 1
}

Write-Host "Using image: $($ImagePath.FullName)"

$PredictOK = $false
for ($i = 0; $i -lt $MaxRetries; $i++) {

    # Arguments array for Start-Process (handles spaces in paths)
    $Args = @(
        "-s", "-X", "POST", "$ServiceURL/predict",
        "-H", "accept: application/json",
        "-F", "file=@`"$($ImagePath.FullName)`"",
        "-F", "true_label=Cat"
    )

    try {
        # Execute curl.exe
        $proc = Start-Process -FilePath "curl.exe" -ArgumentList $Args -NoNewWindow -Wait -PassThru -RedirectStandardOutput "stdout.json"
        $OutputJson = Get-Content "stdout.json" | ConvertFrom-Json

        Write-Host "Prediction endpoint OK"
        Write-Host "Predicted result: $($OutputJson.predicted_label) with probability $($OutputJson.probability)"
        $PredictOK = $true
        break
    } catch {
        Write-Host "Prediction endpoint not ready or failed, retrying in $Delay seconds..."
        Start-Sleep -Seconds $Delay
    }
}

if (-not $PredictOK) {
    Write-Host "Prediction endpoint failed after retries"
    exit 1
}

Write-Host "All smoke tests passed!"
exit 0
