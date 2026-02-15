@echo off
echo Starting Volatility Risk Analyzer (VRA)...

cd /d "%~dp0"

echo Installing requirements...
pip install -r requirements.txt

echo Starting Backend Server...
start "VRA Backend" cmd /k "cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000"

echo Opening Frontend...
timeout /t 3
start frontend/index.html

echo VRA System is running!
echo Backend: http://localhost:8000/docs
echo Frontend: Opened in your default browser.
pause
