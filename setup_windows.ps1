#!/usr/bin/env pwsh
<#
  setup_windows.ps1

  One-shot environment setup script for:
  - Python 3.11 virtual environment
  - requirements.txt installation
  - Jupyter kernel registration
  - Jupyter Lab launch

  Usage (from PowerShell):
    cd <project-root>
    # (Optional) if scripts are blocked:
    #   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\setup_windows.ps1

  Optionally override the Python binary via:
    $env:PYTHON_BIN = "C:\Path\To\Python311\python.exe"
    .\setup_windows.ps1
#>

$ErrorActionPreference = "Stop"

# Resolve project root (directory containing this script)
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $PROJECT_ROOT

Write-Host "===================================================" 
Write-Host " Optimal Fertilizer Prediction - setup (Windows)"
Write-Host " Project root : $PROJECT_ROOT"
Write-Host "===================================================" 

# ------------------------------------------------------
# Locate a Python 3.11 interpreter
# Priority:
#   1. $env:PYTHON_BIN (full path or command)
#   2. 'py -3.11' launcher
#   3. 'python' if it is 3.11.x
# ------------------------------------------------------
function Get-Python311 {
    if ($env:PYTHON_BIN) {
        Write-Host "[INFO] Using PYTHON_BIN from environment: $env:PYTHON_BIN"
        return $env:PYTHON_BIN
    }

    # Try 'py -3.11'
    try {
        $version = & py -3.11 -V 2>$null
        if ($LASTEXITCODE -eq 0 -and $version -match "3\.11") {
            Write-Host "[INFO] Found Python 3.11 via 'py -3.11'"
            return "py -3.11"
        }
    } catch { }

    # Try plain 'python'
    try {
        $version = & python -V 2>$null
        if ($LASTEXITCODE -eq 0 -and $version -match "3\.11") {
            Write-Host "[INFO] Found Python 3.11 via 'python'"
            return "python"
        }
    } catch { }

    throw "ERROR: Could not find a Python 3.11 interpreter. Set PYTHON_BIN or install Python 3.11 and retry."
}

$pythonCmd = Get-Python311
Write-Host "[INFO] Using base Python: $pythonCmd"
& $pythonCmd -V

# ------------------------------------------------------
# Create virtual environment if not present
# ------------------------------------------------------
$venvPath = Join-Path $PROJECT_ROOT ".venv"

if (-Not (Test-Path $venvPath)) {
    Write-Host "[INFO] Creating virtual environment in .venv ..."
    & $pythonCmd -m venv ".venv"
} else {
    Write-Host "[INFO] Virtual environment .venv already exists. Reusing."
}

# Python inside the venv
$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-Not (Test-Path $venvPython)) {
    throw "ERROR: venv Python not found at $venvPython"
}

Write-Host "[INFO] Using Python from venv: $venvPython"
& $venvPython -V

# ------------------------------------------------------
# Upgrade pip
# ------------------------------------------------------
Write-Host "[INFO] Upgrading pip ..."
& $venvPython -m pip install --upgrade pip

# ------------------------------------------------------
# Install requirements
# ------------------------------------------------------
$requirementsFile = Join-Path $PROJECT_ROOT "requirements.txt"
if (Test-Path $requirementsFile) {
    Write-Host "[INFO] Installing Python dependencies from requirements.txt ..."
    & $venvPython -m pip install -r $requirementsFile
} else {
    Write-Warning "[WARN] requirements.txt not found in project root. Skipping requirements install."
}

# ------------------------------------------------------
# Ensure Jupyter + kernel are available
# ------------------------------------------------------
Write-Host "[INFO] Installing Jupyter + IPython kernel ..."
& $venvPython -m pip install jupyter ipykernel

# Register named kernel for this env
$KERNEL_NAME = "fertilizer_py311"
Write-Host "[INFO] Registering Jupyter kernel '$KERNEL_NAME' ..."
& $venvPython -m ipykernel install --user --name $KERNEL_NAME --display-name "Python 3.11 (Fertilizer)"

Write-Host "===================================================" 
Write-Host " Setup complete."
Write-Host " Virtual env : .venv"
Write-Host " Kernel name : $KERNEL_NAME"
Write-Host "===================================================" 
Write-Host ""
Write-Host "To activate the environment manually in PowerShell:"
Write-Host "  `.` .venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "To run Jupyter Lab manually (after activation):"
Write-Host "  jupyter lab"
Write-Host ""
Write-Host "Available notebooks:"
Write-Host "  - OptimalFertilizer_Approach1.ipynb (Ensemble with EDA - trains models)"
Write-Host "  - OptimalFertilizer_Approach2.ipynb (XGBoost with Feature Engineering)"
Write-Host "  - OptimalFertilizer_GUI.ipynb (Interactive GUI Application)"
Write-Host ""
Write-Host "Note: Run Approach1 first to generate model files for the GUI"
Write-Host ""
Write-Host "[INFO] Launching Jupyter Lab now ..."
& $venvPython -m jupyter lab

