# MemEIC Environment Setup Script for PowerShell
# This script cleans and reinstalls the memeic conda environment with correct package versions

param(
    [string]$EnvName = "memeic",
    [string]$ProjectPath = $PSScriptRoot
)

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "MemEIC Environment Setup - Clean Install" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if conda is available
$condaCheck = conda --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Conda not found. Please install Miniconda or Anaconda." -ForegroundColor Red
    exit 1
}

Write-Host "[1/5] Removing existing $EnvName environment..."
conda remove -n $EnvName --all -y 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Could not remove environment (may not exist)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[2/5] Creating new conda environment with Python 3.10..."
conda create -n $EnvName python=3.10 -y
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create environment" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[3/5] Installing PyTorch (CUDA-enabled)..."
conda activate $EnvName
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: PyTorch installation may have issues, trying alternative..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio
}

Write-Host ""
Write-Host "[4/5] Installing project dependencies from requirements_pinned.txt..."
$reqFile = Join-Path $ProjectPath "requirements_pinned.txt"
if (Test-Path $reqFile) {
    pip install -r $reqFile -q
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Some packages failed to install, trying verbose mode..." -ForegroundColor Yellow
        pip install -r $reqFile
    }
} else {
    $origReqFile = Join-Path $ProjectPath "requirements.txt"
    if (Test-Path $origReqFile) {
        Write-Host "Using original requirements.txt..." -ForegroundColor Yellow
        pip install -r $origReqFile -q
    } else {
        Write-Host "ERROR: No requirements file found!" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "[5/5] Verifying installation..."
conda activate $EnvName

# Verify imports
python -c "from easyeditor.trainer.blip2_models.Qformer import apply_chunking_to_forward; print('✓ Qformer import successful')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Qformer import verification failed" -ForegroundColor Red
    exit 1
}

python -c "import transformers; print(f'✓ Transformers version: {transformers.__version__}')"
python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}')"
python -c "import timm; print(f'✓ Timm version: {timm.__version__}')"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "SUCCESS! Environment is ready for MemEIC" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Activate environment: conda activate $EnvName" -ForegroundColor White
Write-Host "  2. Run test: python test_compositional_edit.py test_LLaVA_FT_comp" -ForegroundColor White
Write-Host "  3. Or use provided configs in hparams/ directory" -ForegroundColor White
Write-Host ""
Write-Host "For more details, see FIX_GUIDE.md" -ForegroundColor Cyan
Write-Host ""
