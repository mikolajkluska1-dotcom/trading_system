import os

# Nazwa pliku wynikowego
OUTPUT_FILE = "redline_scan.txt"

# Czego NIE chcemy skanować (śmieci)
IGNORE_DIRS = {
    'node_modules', '.git', '__pycache__', 'venv', '.venv', 'env', 
    'build', 'dist', '.vscode', '.idea'
}
# Jakie pliki nas interesują (kod)
INCLUDE_EXT = {'.py', '.js', '.jsx', '.json', '.css', '.html', '.md'}

def scan_project():
    root_dir = os.getcwd()
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"=== REDLINE PROJECT SCAN ===\n")
        f.write(f"Root: {root_dir}\n\n")

        # 1. Rysujemy strukturę folderów
        f.write("=== FILE STRUCTURE ===\n")
        for root, dirs, files in os.walk(root_dir):
            # Filtrowanie folderów
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            level = root.replace(root_dir, '').count(os.sep)
            indent = ' ' * 4 * (level)
            f.write(f"{indent}[{os.path.basename(root)}/]\n")
            subindent = ' ' * 4 * (level + 1)
            for file in files:
                f.write(f"{subindent}{file}\n")
        
        f.write("\n\n=== FILE CONTENTS ===\n")

        # 2. Zgrywamy treść plików kodu
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in INCLUDE_EXT and file != "package-lock.json" and file != OUTPUT_FILE:
                    path = os.path.join(root, file)
                    rel_path = os.path.relpath(path, root_dir)
                    
                    f.write(f"\n\n{'='*50}\n")
                    f.write(f"FILE: {rel_path}\n")
                    f.write(f"{'='*50}\n")
                    
                    try:
                        with open(path, 'r', encoding='utf-8') as source:
                            content = source.read()
                            # Jeśli plik jest ogromny (np. baza danych json), utnij go
                            if len(content) > 50000: 
                                f.write("[CONTENT TOO LARGE - SKIPPED]\n")
                            else:
                                f.write(content)
                    except Exception as e:
                        f.write(f"[ERROR READING FILE: {e}]\n")

    print(f"✅ Gotowe! Utworzono plik: {OUTPUT_FILE}")
    print("Wrzuć ten plik na czat.")

if __name__ == "__main__":
    scan_project()