@echo off
REM ==========================================================
REM setup_env.bat - SkillSight Environment Setup
REM Run with: call setup_env.bat
REM ==========================================================

cd /d "%~dp0"

if not exist "requirements.txt" (
  echo [ERROR] requirements.txt not found in: %cd%
  exit /b 1
)

REM ---- Create venv if missing ----
if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating virtual environment in .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
  )
)

REM ---- Activate venv ----
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate virtual environment.
  exit /b 1
)

REM ---- Upgrade pip ----
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM ---- Install PyTorch with CUDA (GPU support) ----
echo [INFO] Installing PyTorch with CUDA support...
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126 --quiet

REM ---- Install other requirements ----
echo [INFO] Installing requirements...
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
  echo [ERROR] Installation failed.
  exit /b 1
)

REM ---- Download models from Hugging Face ----
echo [INFO] Downloading models from Hugging Face...
python download_models.py
if errorlevel 1 (
  echo [WARN] Model download failed. You can retry later with: python download_models.py
)

echo.
echo [OK] Environment ready!
echo [OK] VENV is active.
echo.
echo To run skill extraction:
echo   python analyze_resume.py --text "your text" --model deberta
echo.
