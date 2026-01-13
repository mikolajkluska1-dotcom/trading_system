import streamlit as st
import pandas as pd
import plotly.express as px
from core.auto_rotator import AutoRotator
from core.config import THEMES

def render_scanner_view():
    """Widok Skanera Rynku V5-PRO"""
    theme = THEMES[st.session_state['sys'].get('theme', 'MATRIX')]

    st.markdown("###  DEEP FIELD SCANNER V5")
    st.caption("Pro-Grade Screening: HTF Trend + Regime Filter + Risk-Adjusted Scoring")

    # Pasek narzędzi
    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        timeframe = st.selectbox("TIMEFRAME (LTF)", ["15m", "1h", "4h"], index=1, key="scan_tf")

    with c2:
        st.write("")
        st.write("")
        # Przycisk uruchamia skanowanie
        run_btn = st.button(" SCAN TOP 50 LIQUID PAIRS", use_container_width=True)

    # Logika Skanowania
    if run_btn:
        with st.status("RUNNING QUANTUM SCAN PROTOCOL...", expanded=True):
            st.write(" Fetching Market Data & Checking 4H HTF Trends...")
            st.write(" Calculating Market Regime (MQS) & Filtering Chop...")
            st.write(" DeepBrain Inference (Threshold > 60%)...")

            # Uruchomienie skanera
            result_df = AutoRotator.run_scan(timeframe)
            st.session_state['scan_result'] = result_df

            if not result_df.empty:
                st.success(f"SCAN COMPLETE. Found {len(result_df)} opportunities.")
            else:
                st.warning("NO OPPORTUNITIES FOUND (Market might be choppy).")

    # Wyświetlanie wyników
    if 'scan_result' in st.session_state and not st.session_state['scan_result'].empty:
        df = st.session_state['scan_result']

        # Top Picks
        top_picks = df.head(10)

        st.markdown(f"#### ELITE SIGNALS (Top {len(top_picks)})")

        # Formatowanie tabeli
        # Dodajemy kolory do trendu HTF
        def highlight_htf(val):
            color = theme['p'] if val == "BULLISH" else ('red' if val == "BEARISH" else 'grey')
            return f'color: {color}; font-weight: bold'

        styled_df = top_picks[['symbol', 'signal', 'htf_trend', 'mqs', 'conf', 'growth_%', 'score']].style.applymap(
            highlight_htf, subset=['htf_trend']
        ).format({
            "conf": "{:.2f}",
            "growth_%": "{:+.2f}%",
            "score": "{:.1f}",
            "mqs": "{:.0f}"
        })

        st.dataframe(styled_df, use_container_width=True)

        # Wizualizacja V5
        c_chart1, c_chart2 = st.columns(2)

        with c_chart1:
            # Wykres Potencjału vs Ryzyka
            fig = px.scatter(
                top_picks, x='mqs', y='score',
                size='conf', color='htf_trend', hover_name='symbol',
                title="Market Quality vs AI Score",
                color_discrete_map={"BULLISH": theme['p'], "BEARISH": "red", "NEUTRAL": "grey"}
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

        with c_chart2:
            # Ranking słupkowy
            fig2 = px.bar(
                top_picks, x='symbol', y='score',
                color='signal',
                title="Final Edge Score (Risk-Adjusted)",
                color_discrete_map={"BUY": theme['p'], "SELL": "red"}
            )
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig2, use_container_width=True)

    elif 'scan_result' in st.session_state:
        st.info("System filtered out all assets. Current market conditions do not meet 'PRO' entry criteria.")
