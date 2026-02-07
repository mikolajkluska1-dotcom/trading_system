"""
🚀 AIBrain Transfer Script - DevOps Edition
Kopiuje kluczowe elementy projektu tradingowego na dysk zewnętrzny D:
"""
import os
import shutil
import datetime

# ============ KONFIGURACJA ============
SOURCE_DIR = r"C:\Users\user\Desktop\trading_system"
DEST_DRIVE = "D:/"
DEST_DIR = os.path.join(DEST_DRIVE, "AIBrain_Transfer")

# Wzorce do ignorowania
IGNORE_PATTERNS = shutil.ignore_patterns(
    '__pycache__', 
    '*.pyc', 
    '.git', 
    'venv', 
    'env',
    '.env',  # NIE kopiuj .env z kluczami!
    'node_modules',  # Nie kopiuj node_modules (można zainstalować przez npm)
    '*.log',
    '.vscode',
    '.idea'
)

def get_dir_size(path):
    """Oblicza rozmiar katalogu w MB"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except Exception as e:
        print(f"⚠️ Błąd przy obliczaniu rozmiaru {path}: {e}")
    return total / (1024 * 1024)  # Konwersja na MB

def copy_project():
    """Główna funkcja kopiująca projekt"""
    print("=" * 60)
    print("🚀 AIBrain Transfer Script - DevOps Edition")
    print("=" * 60)
    print(f"📂 ŹRÓDŁO: {SOURCE_DIR}")
    print(f"📦 CEL: {DEST_DIR}")
    print(f"⏰ Data: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Sprawdź czy dysk D: istnieje
    if not os.path.exists(DEST_DRIVE):
        print(f"❌ BŁĄD: Dysk {DEST_DRIVE} nie istnieje!")
        return
    
    # Usuń stary backup jeśli istnieje
    if os.path.exists(DEST_DIR):
        print(f"🗑️  Usuwanie starej wersji backupu: {DEST_DIR}")
        try:
            shutil.rmtree(DEST_DIR)
            print("✅ Stary backup usunięty")
        except Exception as e:
            print(f"❌ Błąd przy usuwaniu: {e}")
            return
    
    print(f"\n📦 Rozpoczynam kopiowanie...\n")
    
    # ============ KOPIOWANIE FOLDERÓW ============
    # Foldery do skopiowania (dostosowane do rzeczywistej struktury)
    dirs_to_copy = [
        'agents',      # Główny folder z agentami (AIBrain, BackendAPI, Database, etc.)
        'assets',      # Zasoby
        'n8n',         # Konfiguracja n8n
        'archive'      # Archiwum (jeśli potrzebne)
    ]
    
    copied_dirs = []
    for item in dirs_to_copy:
        source_path = os.path.join(SOURCE_DIR, item)
        dest_path = os.path.join(DEST_DIR, item)
        
        if os.path.exists(source_path):
            try:
                print(f"📁 Kopiuję folder: {item}...", end=" ")
                shutil.copytree(source_path, dest_path, ignore=IGNORE_PATTERNS)
                size_mb = get_dir_size(dest_path)
                print(f"✅ ({size_mb:.2f} MB)")
                copied_dirs.append(item)
            except Exception as e:
                print(f"❌ BŁĄD: {e}")
        else:
            print(f"⚠️  Folder nie istnieje: {item}")
    
    # ============ KOPIOWANIE PLIKÓW Z ROOTA ============
    print("\n📄 Kopiuję pliki z głównego katalogu...\n")
    
    files_to_copy = [
        'START_AI_TRAINING.py',
        'START_TURBO_TRAINING_AT_22.py',
        'START_VOLUME_HUNTER_AT_22.py',
        'deep_diver.py',
        'requirements.txt',
        '.env.example',  # Przykładowy .env (BEZ kluczy)
        '.gitignore',
        'docker-compose.yml',
        'Dockerfile.backend',
        'Dockerfile.frontend',
        'SYSTEM.md',
        'PORTABILITY_GUIDE_PL.md'
    ]
    
    # Utwórz katalog docelowy jeśli nie istnieje
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
    
    copied_files = []
    for filename in files_to_copy:
        source_path = os.path.join(SOURCE_DIR, filename)
        dest_path = os.path.join(DEST_DIR, filename)
        
        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, dest_path)
                size_kb = os.path.getsize(dest_path) / 1024
                print(f"✅ {filename} ({size_kb:.2f} KB)")
                copied_files.append(filename)
            except Exception as e:
                print(f"❌ Błąd przy kopiowaniu {filename}: {e}")
        else:
            print(f"⚠️  Plik nie istnieje: {filename}")
    
    # ============ PODSUMOWANIE ============
    print("\n" + "=" * 60)
    print("📊 PODSUMOWANIE TRANSFERU")
    print("=" * 60)
    print(f"✅ Skopiowane foldery ({len(copied_dirs)}):")
    for d in copied_dirs:
        print(f"   • {d}")
    
    print(f"\n✅ Skopiowane pliki ({len(copied_files)}):")
    for f in copied_files:
        print(f"   • {f}")
    
    # Oblicz całkowity rozmiar
    total_size = get_dir_size(DEST_DIR)
    print(f"\n📦 Całkowity rozmiar transferu: {total_size:.2f} MB")
    print(f"📍 Lokalizacja: {DEST_DIR}")
    
    print("\n" + "=" * 60)
    print("🎉 SUKCES! Projekt gotowy do transferu na serwer!")
    print("=" * 60)
    print("\n💡 NASTĘPNE KROKI:")
    print("1. Przenieś dysk D: na serwer obliczeniowy (Desktop)")
    print("2. Skopiuj folder AIBrain_Transfer na serwer")
    print("3. Zainstaluj zależności: pip install -r requirements.txt")
    print("4. Skonfiguruj plik .env z kluczami API")
    print("5. Uruchom: python START_AI_TRAINING.py")
    print("=" * 60)

if __name__ == "__main__":
    try:
        copy_project()
    except KeyboardInterrupt:
        print("\n\n⚠️  Transfer przerwany przez użytkownika!")
    except Exception as e:
        print(f"\n\n❌ KRYTYCZNY BŁĄD: {e}")
        import traceback
        traceback.print_exc()
