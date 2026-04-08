@echo off
echo Starting Quant-Edge Stock Prediction...

:: Open the frontend website in the default browser
start "" "%~dp0minor project\frontend\index.html"

:: Change directory to the backend project folder
cd /d "%~dp0minor project"

:: Activate the virtual environment
call .venv\Scripts\activate.bat

:: Start the backend server
echo Starting Backend Server...
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pause
