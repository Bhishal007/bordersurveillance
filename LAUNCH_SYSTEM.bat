@echo off
title SSB C2 Node Initialization
color 0B

echo =========================================================
echo  SSB BORDER SURVEILLANCE C2 NODE - AUTOMATED BOOTUP
echo =========================================================
echo.

echo [*] STEP 1: Waking up Omni-Agent (Local LLM)...
start "Omni-Agent" cmd /c "1-start-ollama-server.bat"
timeout /t 3 /nobreak >nul

echo [*] STEP 2: Booting Core Services (Docker, Mongo, Node, React)...
start "Docker Services" cmd /c "docker compose up -d"
timeout /t 5 /nobreak >nul

echo [*] STEP 3: Initializing Python ML Backend (Port 8000)...
start "FastAPI Backend" cmd /c "cd border-surveillance-ml && venv\Scripts\activate && uvicorn backend.main:app --host 0.0.0.0 --port 8000"
timeout /t 3 /nobreak >nul

echo [*] STEP 4: Initializing YOLO11 CV Engine (Port 5001)...
start "Computer Vision Engine" cmd /c "cd border-surveillance-ml && venv\Scripts\activate && python stream.py"
timeout /t 3 /nobreak >nul

echo [*] STEP 5: Launching Tactical Dashboard...
start http://localhost:5173

echo.
echo =========================================================
echo  SYSTEM ONLINE. You can now close this setup window.
echo =========================================================
pause