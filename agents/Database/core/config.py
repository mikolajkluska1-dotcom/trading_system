# ================================================================
# REDLINE SYSTEM CONFIGURATION
# ================================================================

# --- SYSTEM DEPENDENCIES (FIX BŁĘDU IMPORTU) ---
REQUIRED_LIBS = [
    "streamlit",
    "pandas",
    "numpy",
    "ccxt",
    "yfinance",
    "ta",
    "textblob",
    "plotly",
    "torch",
    "sklearn"
]

# --- SECURITY ---
SECURITY_SALT = "QUANTUM_ENTROPY_V1"
SECURITY_OVERRIDE_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" # Empty hash placeholder

# --- UI THEMES ---
# 'p': Primary, 's': Secondary, 't': Text, 'bg': Background, 'err': Error, 'warn': Warning
THEMES = {
    'MATRIX': {
        'p': '#00ff41',    # Matrix Green
        's': '#003b00',    # Dark Green (Secondary)
        't': '#e0e0e0',    # Light Text
        'bg': '#0d0d0d',   # Black Background
        'err': '#ff3b30',  # Red
        'warn': '#ffcc00', # Yellow
        'info': '#00aaff'  # Blue
    },
    'CYBERPUNK': {
        'p': '#ffee00',    # Yellow
        's': '#3d3d00',    # Dark Yellow
        't': '#00f0ff',    # Cyan Text
        'bg': '#0b0b15',   # Deep Blue Bg
        'err': '#ff003c',  # Pink/Red
        'warn': '#ffee00',
        'info': '#00f0ff'
    },
    'CRIMSON': {
        'p': '#ff3b30',    # Red
        's': '#4a0e0b',    # Dark Red
        't': '#ffffff',    # White
        'bg': '#0a0a0a',   # Black
        'err': '#ff0000',
        'warn': '#ff9500',
        'info': '#5ac8fa'
    }
}
