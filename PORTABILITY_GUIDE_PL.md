# Instrukcja Przeniesienia Projektu (Redline System)

Ten dokument opisuje, jak przenieść projekt na inny komputer, abyś mógł go otworzyć w Google Antigravity (lub dowolnym innym środowisku) i kontynuować pracę.

## 1. Co musisz skopiować?

Aby projekt działał w pełni, musisz przenieść dwa główne elementy:

1.  **Folder projektu**: `c:\Users\user\Desktop\trading_system` (wszystkie pliki `.py`, `.js`, `.json`, `Dockerfile`, itp.).
2.  **Folder danych (KRYTYCZNE)**: Cały folder `R:/Redline_Data`. 
    *   To tam znajduje się baza danych PostgreSQL (TimescaleDB).
    *   Tam są zapisane nauczone modele AI (`.pth`).
    *   Bez tego folderu projekt uruchomi się "pusty" (bez historii handlu i bez wiedzy AI).

> [!IMPORTANT]
> Jeśli na nowym komputerze nie masz napędu `R:`, możesz skopiować folder `Redline_Data` w dowolne miejsce (np. `D:/Redline_Data`) i zmienić ścieżki w pliku `docker-compose.yml` (linia 83) oraz w skryptach AI (np. `turbo_training.py`).

## 2. Wymagania Techniczne

Na nowym komputerze zainstaluj:
1.  **Docker Desktop** (najprostszy sposób na bazę danych i n8n).
2.  **Python 3.10+**.
3.  **Node.js 18+**.
4.  **Google Antigravity** (załaduj folder projektu jako workspace).

## 3. Instalacja Bibliotek

Po otwarciu projektu w nowym terminalu:

### Backend (Python)
```powershell
pip install -r requirements.txt
```

### Frontend (React)
Przejdź do folderu frontendu:
```powershell
cd agents/Frontend/frontend
npm install
```

## 4. Uruchamianie Systemu

### Opcja A: Docker (Zalecane)
W głównym folderze projektu:
```powershell
docker-compose up -d
```
To uruchomi:
*   Backend (Port 8000)
*   Frontend (Port 3000)
*   Bazę danych (Port 5435)
*   n8n (Port 5678)

### Opcja B: Ręczne (Dla deweloperów)
1.  Uruchom tylko bazę w Dockerze: `docker-compose up timescaledb -d`
2.  Uruchom Backend: `python -m agents.BackendAPI.backend.main`
3.  Uruchom Frontend: `cd agents/Frontend/frontend` -> `npm run dev`

## 5. Checklist Przed Wyjściem
- [x] Plik `requirements.txt` jest aktualny.
- [x] Plik `package.json` ma wszystkie biblioteki.
- [x] Folder `R:/Redline_Data` został zarchiwizowany/skopiowany.
- [x] Wszystkie zmiany w kodzie są zapisane (Zapisz pliki w Antigravity!).

---
*Przygotowane przez Antigravity AI.*
