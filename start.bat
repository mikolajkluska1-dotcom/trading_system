@echo off
REM REDLINE Trading System - Startup Script for Windows
REM Double-click this file to start both backend and frontend

echo ========================================
echo REDLINE Trading System - Startup Script
echo ========================================
echo.

echo Starting Backend Server...
start "REDLINE Backend" cmd /k "cd /d %~dp0 && uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"

timeout /t 2 /nobreak >nul

echo Starting Frontend Dev Server...
start "REDLINE Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ========================================
echo Services Started!
echo ========================================
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Close the terminal windows to stop services.
pause
