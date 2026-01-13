import streamlit as st
import pandas as pd
import time
from core.config import THEMES
from ml.brain import DeepBrain
from ml.knowledge import KnowledgeBase
from data.feed import DataFeed

def render_training_view():
    """
    WIDOK TRENINGOWY (AI LAB).
    Pozwala douczać model na błędach i zarządzać bazą wiedzy.
    """
    # Pobieramy motyw z sesji (bezpiecznie)
    theme_name = st.session_state['sys'].get('theme', 'MATRIX')
    theme = THEMES[theme_name]

    st.markdown(f"### NEURAL LABS")
    st.caption("Reinforcement Learning & Pattern Correction")

    # Linia, która powodowała błąd (teraz bezpieczna dzięki poprawionemu config.py)
    st.markdown(f"<hr style='border-color:{theme['s']}'>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])

    with c1:
        st.info("MANUAL OVERRIDE TRAINING")
        symbol = st.text_input("SYMBOL", "BTC/USDT")

        if st.button("FETCH & ANALYZE"):
            df = DataFeed.get_market_data(symbol, "1h", limit=100)
            if not df.empty:
                st.session_state['train_df'] = df
                st.success("DATA LOADED")
            else:
                st.error("NO DATA")

    with c2:
        if 'train_df' in st.session_state:
            df = st.session_state['train_df']

            # Brain Inference
            brain = DeepBrain()
            price, conf, signal = brain.predict(df)

            st.markdown(f"#### AI PREDICTION: <span style='color:{theme['p']}'>{signal}</span> ({conf*100:.1f}%)", unsafe_allow_html=True)
            st.line_chart(df['close'])

            st.write("Was this prediction correct?")

            b1, b2 = st.columns(2)
            if b1.button(" YES (REWARD)"):
                snap = {"close": df['close'].iloc[-1], "rsi": 50} # Uproszczony snapshot
                KnowledgeBase.save_pattern("OPERATOR", snap, 1.0)
                st.success("Model Rewarded (+1.0)")

            if b2.button(" NO (PUNISH)"):
                snap = {"close": df['close'].iloc[-1], "rsi": 50}
                KnowledgeBase.save_pattern("OPERATOR", snap, -1.0)
                st.error("Model Punished (-1.0)")
