import streamlit as st

# =====================================================
# 1. PAGE CONFIG (MUSI BYĆ PIERWSZE)
# =====================================================
st.set_page_config(
    page_title="REDLINE V68",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================
# 2. CORE IMPORTS
# =====================================================
from core.bootstrap import init_session, session_watchdog
from ui.styles import load_styles
from ui.auth import render_auth
from ui.dashboard import render_dashboard
from ui.hud import render_hud
from ui.admin import render_admin
from ui.wallet import render_wallet
from ui.scanner_view import render_scanner_view
from ui.training_view import render_training_view

# =====================================================
# 3. INIT SESSION
# =====================================================
init_session()
session_watchdog()

# =====================================================
# 4. LOAD STYLES
# =====================================================
load_styles(
    mode=st.session_state.get("ui_mode", "ops"),
    system_name="REDLINE V68",
    system_tagline="PRIVATE TRADING & RISK TERMINAL",
)

# =====================================================
# 5. GLOBAL EVENT SYSTEM (JS – INLINE, BEZ alert())
# =====================================================
st.components.v1.html(
    """
    <script>
    console.log("EVENT SYSTEM INITIALIZING");

    const ws = new WebSocket("ws://127.0.0.1:8002/ws/events?scope=OPS");

    ws.onopen = () => {
        console.log("EVENT WS CONNECTED");
    };

    ws.onmessage = (e) => {
        const ev = JSON.parse(e.data);
        console.log("EVENT RECEIVED:", ev);

        let container = document.getElementById("event-overlay");
        if (!container) {
            container = document.createElement("div");
            container.id = "event-overlay";
            container.style.position = "fixed";
            container.style.top = "80px";
            container.style.right = "20px";
            container.style.zIndex = "9999";
            container.style.display = "flex";
            container.style.flexDirection = "column";
            container.style.gap = "10px";
            document.body.appendChild(container);
        }

        const card = document.createElement("div");
        card.style.background = "rgba(0,0,0,0.85)";
        card.style.color = "#e5e7eb";
        card.style.border = "1px solid #1f2937";
        card.style.borderRadius = "8px";
        card.style.padding = "10px 14px";
        card.style.fontSize = "12px";
        card.style.minWidth = "260px";
        card.style.boxShadow = "0 6px 18px rgba(0,0,0,0.45)";
        card.style.opacity = "0";
        card.style.transform = "translateY(-6px)";
        card.style.transition = "all 0.25s ease-out";

        if (ev.level === "warning") card.style.borderColor = "#f59e0b";
        if (ev.level === "success") card.style.borderColor = "#10b981";
        if (ev.level === "error") card.style.borderColor = "#ef4444";

        card.innerHTML = `
            <div style="font-weight:600; letter-spacing:1px;">${ev.type}</div>
            <div style="opacity:0.85; margin-top:4px;">${ev.message}</div>
        `;

        container.appendChild(card);

        requestAnimationFrame(() => {
            card.style.opacity = "1";
            card.style.transform = "translateY(0)";
        });

        setTimeout(() => {
            card.style.opacity = "0";
            card.style.transform = "translateY(-6px)";
            setTimeout(() => card.remove(), 300);
        }, 5000);
    };

    ws.onerror = (e) => {
        console.error("EVENT WS ERROR", e);
    };

    ws.onclose = () => {
        console.warn("EVENT WS CLOSED");
    };
    </script>
    """,
    height=0,
)

# =====================================================
# 6. AUTH GATE
# =====================================================
if not st.session_state["sys"].get("auth", False):
    render_auth()

else:
    # =================================================
    # HUD (TOP BAR)
    # =================================================
    render_hud()

    # =================================================
    # MAIN NAVIGATION
    # =================================================
    tabs = st.tabs(
        [
            " Dashboard",
            " Wallet",
            " Scanner",
            " Training",
            " System",
        ]
    )

    with tabs[0]:
        render_dashboard()

    with tabs[1]:
        render_wallet()

    with tabs[2]:
        render_scanner_view()

    with tabs[3]:
        render_training_view()

    with tabs[4]:
        render_admin()
