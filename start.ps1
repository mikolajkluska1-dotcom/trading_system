# REDLINE Startup Script for Windows PowerShell
# Run this script from the trading_system directory

Write-Host "🚀 REDLINE Trading System - Startup Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Start Backend
Write-Host "📡 Starting Backend Server..." -ForegroundColor Yellow
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"

# Wait a moment for backend to initialize
Start-Sleep -Seconds 2

# Start Frontend
Write-Host "🎨 Starting Frontend Dev Server..." -ForegroundColor Yellow
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\frontend'; npm run dev"

Write-Host ""
Write-Host "✅ Services Started!" -ForegroundColor Green
Write-Host ""
Write-Host "Backend:  http://localhost:8000" -ForegroundColor White
Write-Host "Frontend: http://localhost:5173" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C in each terminal to stop services." -ForegroundColor Gray
