import streamlit as st
import psutil, os
from datetime import datetime
from core.config import THEMES
from trading.wallet import WalletManager

def render_hud():
    """Displays top status HUD in hacker terminal style."""
    theme = THEMES[st.session_state['sys'].get('theme', 'MATRIX')]
    user = st.session_state['sys'].get('user', 'anon')
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    wallet = WalletManager.get_balance()
    pid = os.getpid()
    clock = datetime.now().strftime("%H:%M:%S")

    st.markdown(f"""
    <div class="hud-bar">
        <div>NODE: <span style="color:{theme['p']}">REDLINE_V68</span></div>
        <div>USER: <span style="color:#fff">{user}</span> (PID {pid})</div>
        <div>CPU: {cpu}% | MEM: {ram}%</div>
        <div>FUNDS: <span style="color:{theme['p']}">${wallet:,.2f}</span></div>
        <div>TIME: {clock}</div>
    </div>
    """, unsafe_allow_html=True)
