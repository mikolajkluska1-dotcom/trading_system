# =========================================
# REDLINE TRADING SYSTEM - AUTO INSTALLER
# =========================================
# Ten skrypt zainstaluje wszystkie wymagane biblioteki
# Uruchom jako: .\INSTALL_ALL.ps1

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "  REDLINE SYSTEM - AUTO INSTALLER         " -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# Sprawdzenie czy jestesmy w glownym katalogu
if (-not (Test-Path "requirements.txt")) {
    Write-Host "BLAD: Uruchom ten skrypt z glownego katalogu projektu!" -ForegroundColor Red
    Write-Host "Przejdz do: C:\Users\Mikolaj\trading_system" -ForegroundColor Yellow
    pause
    exit 1
}

# =========================================
# KROK 1: Sprawdzenie Python
# =========================================
Write-Host "Sprawdzam Pythona..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "OK: $pythonVersion" -ForegroundColor Green
Write-Host ""

# =========================================
# KROK 2: Instalacja bibliotek (BEZ PyTorch)
# =========================================
Write-Host "INSTALUJE BIBLIOTEKI (to zajmie ~5 minut)..." -ForegroundColor Yellow
Write-Host ""

pip install --upgrade pip

Write-Host "Instaluje FastAPI i serwery..." -ForegroundColor Cyan
pip install fastapi uvicorn pydantic python-multipart requests python-dotenv aiohttp psutil

Write-Host "Instaluje biblioteki do danych i finansow..." -ForegroundColor Cyan
pip install numpy pandas yfinance ccxt ta gspread oauth2client

Write-Host "Instaluje scikit-learn i textblob..." -ForegroundColor Cyan
pip install scikit-learn textblob

Write-Host "Instaluje Streamlit i Plotly..." -ForegroundColor Cyan
pip install streamlit plotly

Write-Host "Instaluje biblioteki bezpieczenstwa..." -ForegroundColor Cyan
pip install cryptography websockets

Write-Host "Instaluje biblioteki bazodanowe..." -ForegroundColor Cyan
pip install psycopg2-binary sqlalchemy asyncpg aiopg

Write-Host ""
Write-Host "BIBLIOTEKI PODSTAWOWE ZAINSTALOWANE!" -ForegroundColor Green
Write-Host ""

# =========================================
# KROK 3: PyTorch (opcjonalnie)
# =========================================
Write-Host "PYTORCH - Ta biblioteka jest ogromna (~3GB)" -ForegroundColor Yellow
Write-Host "Czy chcesz zainstalowac PyTorch z CUDA (dla karty NVIDIA)?" -ForegroundColor Yellow
$choice = Read-Host "Wpisz T (tak) lub N (nie)"

if ($choice -eq "T" -or $choice -eq "t") {
    Write-Host "Instaluje PyTorch z CUDA... (to moze zajac 10-20 minut)" -ForegroundColor Cyan
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    Write-Host "PyTorch zainstalowany!" -ForegroundColor Green
}
else {
    Write-Host "Pominieto PyTorch (mozesz zainstalowac pozniej)" -ForegroundColor Yellow
}

Write-Host ""

# =========================================
# KROK 4: Tworzenie pliku .env
# =========================================
Write-Host "Tworze plik .env..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "Plik .env juz istnieje, pomijam..." -ForegroundColor Yellow
}
else {
    Copy-Item ".env.example" ".env"
    Write-Host "Plik .env utworzony!" -ForegroundColor Green
    Write-Host "Pamietaj aby wypelnic klucze API w pliku .env" -ForegroundColor Cyan
}

Write-Host ""

# =========================================
# KROK 5: Weryfikacja
# =========================================
Write-Host "WERYFIKACJA INSTALACJI..." -ForegroundColor Yellow
Write-Host ""

$packages = @("fastapi", "uvicorn", "pandas", "numpy", "ccxt", "yfinance", "streamlit", "sqlalchemy")
foreach ($pkg in $packages) {
    $installed = pip show $pkg 2>$null
    if ($installed) {
        Write-Host "  OK: $pkg" -ForegroundColor Green
    }
    else {
        Write-Host "  BRAK: $pkg" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "===========================================" -ForegroundColor Green
Write-Host "     INSTALACJA ZAKONCZONA!               " -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Green
Write-Host ""
Write-Host "NASTEPNE KROKI:" -ForegroundColor Cyan
Write-Host "1. Uruchom Docker Desktop" -ForegroundColor White
Write-Host "2. Wypelnij plik .env swoimi kluczami API" -ForegroundColor White
Write-Host "3. Uruchom system: docker-compose up -d" -ForegroundColor White
Write-Host ""

pause
