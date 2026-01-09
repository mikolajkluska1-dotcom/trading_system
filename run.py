import subprocess
import sys
import os
import time

def run_system():
    # Ustalanie ścieżek
    root_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(root_dir, "frontend")
    
    # Wykrywanie systemu (Windows vs Mac/Linux) dla komendy npm
    is_windows = os.name == 'nt'
    npm_cmd = "npm.cmd" if is_windows else "npm"
    python_cmd = sys.executable  # Używa tego samego pythona, co skrypt

    print("\n🚀 REDLINE TRADING SYSTEM — LAUNCH SEQUENCE")
    print("=============================================")

    processes = []

    try:
        # 1. URUCHOMIENIE BACKENDU
        print("🔌 [Backend] Starting FastAPI (Port 8000)...")
        # Używamy 'python -m uvicorn' dla pewności ścieżek
        backend = subprocess.Popen(
            [python_cmd, "-m", "uvicorn", "backend.main:app", "--reload"],
            cwd=root_dir,
            env=os.environ.copy()
        )
        processes.append(backend)
        
        # Dajemy chwilę na start backendu, żeby nie mieszały się logi
        time.sleep(2) 

        # 2. URUCHOMIENIE FRONTENDU
        print("💻 [Frontend] Starting Vite (Port 5173)...")
        frontend = subprocess.Popen(
            [npm_cmd, "run", "dev"],
            cwd=frontend_dir,
            env=os.environ.copy()
        )
        processes.append(frontend)

        print("\n✅ SYSTEM ONLINE")
        print("   -> App:      http://localhost:5173")
        print("   -> API Docs: http://localhost:8000/docs")
        print("   (Press Ctrl+C to stop all services)\n")
        
        # Czekamy na zakończenie procesów (lub przerwanie przez usera)
        backend.wait()
        frontend.wait()

    except KeyboardInterrupt:
        print("\n\n🛑 SHUTTING DOWN...")
        for p in processes:
            try:
                p.terminate()
            except:
                pass
        print("   All systems offline. Bye!\n")

if __name__ == "__main__":
    run_system()