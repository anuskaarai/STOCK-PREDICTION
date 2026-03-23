@echo off
echo ==============================================
echo Quant-Edge Startup Script
echo ==============================================

echo Checking if Python is installed...
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH. Please install Python to run the backend.
    pause
    exit /b
)

echo.
echo Installing requirements...
pip install -r "minor project\requirements.txt" || pip install -r "requirements.txt"

echo.
echo Starting FastAPI Backend...
start "Quant-Edge API" cmd /c "cd minor project && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000" || start "Quant-Edge API" cmd /c "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"

echo.
echo Please wait 5 seconds for the backend to start...
timeout /t 5 >nul

echo.
echo Opening Frontend in your default browser...
IF EXIST "minor project\frontend\index.html" (
    start "" "minor project\frontend\index.html"
) ELSE IF EXIST "frontend\index.html" (
    start "" "frontend\index.html"
) ELSE (
    echo "Could not find frontend\index.html."
)

echo.
echo Site is now working! You can use the dashboard.
pause
