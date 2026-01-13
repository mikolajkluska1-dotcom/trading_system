#!/usr/bin/env python3
"""
REDLINE Trading System - Startup Launcher
Run both backend and frontend with a single command: python start.py
"""

import subprocess
import sys
import os
import time
import platform
from pathlib import Path

def print_banner():
    print("\n" + "="*60)
    print("🚀 REDLINE Trading System - Startup Launcher")
    print("="*60 + "\n")

def get_base_dir():
    """Get the project root directory"""
    return Path(__file__).parent.absolute()

def start_services():
    """Launch backend and frontend in separate processes"""
    base_dir = get_base_dir()
    frontend_dir = base_dir / "frontend"
    is_windows = platform.system() == "Windows"
    
    print("📡 Starting Backend Server...")
    print(f"   Directory: {base_dir}")
    print(f"   Command: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000\n")
    
    # Start backend
    backend_cmd = ["uvicorn", "backend.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
    backend_process = subprocess.Popen(
        backend_cmd if not is_windows else " ".join(backend_cmd),
        cwd=str(base_dir),
        shell=is_windows
    )
    
    # Wait a moment for backend to initialize
    time.sleep(2)
    
    print("🎨 Starting Frontend Dev Server...")
    print(f"   Directory: {frontend_dir}")
    print(f"   Command: npm run dev\n")
    
    # Start frontend (npm on Windows needs shell=True)
    frontend_cmd = ["npm", "run", "dev"]
    frontend_process = subprocess.Popen(
        frontend_cmd if not is_windows else " ".join(frontend_cmd),
        cwd=str(frontend_dir),
        shell=is_windows
    )
    
    print("\n" + "="*60)
    print("✅ Services Started!")
    print("="*60)
    print("\n📍 Access Points:")
    print("   Backend API:  http://localhost:8000")
    print("   Frontend UI:  http://localhost:5173")
    print("\n💡 Press Ctrl+C to stop both services\n")
    
    try:
        # Keep the script running and monitor both processes
        while True:
            # Check if processes are still running
            backend_status = backend_process.poll()
            frontend_status = frontend_process.poll()
            
            if backend_status is not None:
                print(f"\n⚠️  Backend process exited with code {backend_status}")
                frontend_process.terminate()
                break
            
            if frontend_status is not None:
                print(f"\n⚠️  Frontend process exited with code {frontend_status}")
                backend_process.terminate()
                break
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services...")
        backend_process.terminate()
        frontend_process.terminate()
        
        # Wait for graceful shutdown
        backend_process.wait(timeout=5)
        frontend_process.wait(timeout=5)
        
        print("✅ Services stopped successfully!")
        sys.exit(0)

def main():
    print_banner()
    
    # Check if we're in the right directory
    base_dir = get_base_dir()
    if not (base_dir / "backend").exists():
        print("❌ Error: 'backend' directory not found!")
        print(f"   Current path: {base_dir}")
        print("   Please run this script from the trading_system root directory.")
        sys.exit(1)
    
    if not (base_dir / "frontend").exists():
        print("❌ Error: 'frontend' directory not found!")
        print(f"   Current path: {base_dir}")
        sys.exit(1)
    
    # Check if uvicorn is installed
    try:
        subprocess.run(["uvicorn", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: uvicorn not found!")
        print("   Install it with: pip install uvicorn")
        sys.exit(1)
    
    # Check if npm is installed
    try:
        is_windows = platform.system() == "Windows"
        subprocess.run(
            "npm --version" if is_windows else ["npm", "--version"],
            capture_output=True,
            check=True,
            shell=is_windows
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: npm not found!")
        print("   Please install Node.js and npm first.")
        sys.exit(1)
    
    # Start the services
    start_services()

if __name__ == "__main__":
    main()
