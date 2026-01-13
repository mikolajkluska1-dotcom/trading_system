import streamlit as st

@st.cache_data
def _render_css(css: str):
    st.markdown(css, unsafe_allow_html=True)

def load_styles(
    mode: str = "ops",
    system_name: str = "REDLINE V68",
    system_tagline: str = "PRIVATE TRADING & RISK TERMINAL",
):
    mode = mode.lower()
    if mode not in {"ops", "user"}:
        raise ValueError("Invalid UI mode")

    # ================= COLORS =================
    if mode == "user":
        bg = "#0e1117"
        panel = "#121826"
        text = "#e5e7eb"
        muted = "#9ca3af"
        prim = "#d4b46a"
        accent = "#1f2937"
        alert = "#92400e"
        error = "#7f1d1d"
        glow = "none"
    else:
        bg = "#05070c"
        panel = "#0b0f19"
        text = "#d1d5db"
        muted = "#6b7280"
        prim = "#7aa2f7"
        accent = "#111827"
        alert = "#92400e"
        error = "#7f1d1d"
        glow = "0 0 12px rgba(122,162,247,0.15)"

    css = f"""
    <style>
    :root {{
        --bg: {bg};
        --panel: {panel};
        --text: {text};
        --muted: {muted};
        --prim: {prim};
        --accent: {accent};
        --alert: {alert};
        --error: {error};
    }}

    html, body, [data-testid="stAppViewContainer"] {{
        background: var(--bg) !important;
        color: var(--text);
        font-family: Inter, sans-serif;
    }}

    [data-testid="stMainBlockContainer"] {{
        max-width: 1400px;
        margin: auto;
        background: var(--panel);
        padding: 28px;
        border-radius: 12px;
        box-shadow: {glow};
    }}

    .brand-bar {{
        text-align: center;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--accent);
    }}

    .brand-title {{
        color: var(--prim);
        font-size: 18px;
        letter-spacing: 2px;
    }}

    .brand-tagline {{
        color: var(--muted);
        font-size: 11px;
        letter-spacing: 1.5px;
    }}

    /* ================= HUD ================= */
    .hud-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 18px;
        padding: 8px 14px;
        margin-bottom: 18px;
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--accent);
        border-radius: 10px;
        font-size: 12px;
        color: var(--muted);
    }}

    .hud-item {{
        display: flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
    }}

    .hud-ok {{ color: var(--text); }}
    .hud-warn {{ color: var(--alert); }}
    .hud-crit {{ color: var(--error); }}

    .hud-icon {{
        width: 12px;
        height: 12px;
        opacity: 0.7;
    }}

    #MainMenu, header, footer {{
        visibility: hidden;
    }}
    </style>

    <div class="brand-bar">
        <div class="brand-title">{system_name}</div>
        <div class="brand-tagline">{system_tagline}</div>
    </div>
    """

    _render_css(css)
