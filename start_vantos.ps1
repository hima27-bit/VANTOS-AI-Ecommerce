# ==========================================
# VANTOS AI E-Commerce Launcher
# ==========================================

$LLAMA_SERVER = "C:\Users\Hima Sai\Downloads\llama-b10896-bin-win-cuda-12.4-x64\llama-server.exe"

$QWEN_MODEL = "C:\Users\Hima Sai\Desktop\VANTOS\models\Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

$QWEN_URL = "http://127.0.0.1:8080"
$API_URL = "http://127.0.0.1:8000"


Write-Host ""
Write-Host "==========================================" 
Write-Host "       VANTOS AI E-Commerce System"
Write-Host "=========================================="
Write-Host ""


# ------------------------------------------
# Check llama-server
# ------------------------------------------

if (!(Test-Path $LLAMA_SERVER)) {

    Write-Host "ERROR: llama-server.exe not found."
    Write-Host "Update the `$LLAMA_SERVER path in start_vantos.ps1"
    exit 1
}


# ------------------------------------------
# Start Qwen
# ------------------------------------------

Write-Host "Starting Qwen server..."

Start-Process `
    -FilePath $LLAMA_SERVER `
    -ArgumentList "-m `"$QWEN_MODEL`" -ngl 99 -c 8192 --host 127.0.0.1 --port 8080" `
    -WindowStyle Normal


# ------------------------------------------
# Wait for Qwen
# ------------------------------------------

Write-Host "Waiting for Qwen..."

$qwenReady = $false

for ($i = 1; $i -le 30; $i++) {

    try {

        $response = Invoke-WebRequest `
            -Uri "$QWEN_URL/health" `
            -TimeoutSec 2

        if ($response.StatusCode -eq 200) {

            $qwenReady = $true
            break
        }

    } catch {

        Start-Sleep -Seconds 1
    }
}


# ------------------------------------------
# Start FastAPI
# ------------------------------------------

Write-Host ""
Write-Host "Starting FastAPI..."
Write-Host ""

python -m uvicorn backend.main:app --reload