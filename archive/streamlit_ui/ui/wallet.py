import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# CORE & TRADING IMPORTS
from core.config import THEMES
from trading.wallet import WalletManager
from trading.execution import ExecutionEngine # <-- Tu była stara nazwa TradeExecutor

def render_wallet():
    """
    WIDOK PORTFELA (BANKING).
    Wersja V2: Zintegrowana z WalletManager i ExecutionEngine.
    """
    theme = THEMES[st.session_state['sys'].get('theme', 'MATRIX')]

    # Pobieramy dane z portfela
    wallet = WalletManager.get_wallet_data()
    balance = wallet.get('balance', 0.0)
    assets = wallet.get('assets', [])
    history = wallet.get('history', [])

    # Nagłówek
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"## 💳 WALLET OPS")
        st.caption("Capital Allocation & Asset Tracking")
    with c2:
        st.metric("AVAILABLE CASH", f"${balance:,.2f}")

    st.divider()

    # --- SEKCJA 1: AKTYWA (ASSETS) ---
    st.markdown("### 🏛️ ACTIVE HOLDINGS")

    if assets:
        # Konwersja do DataFrame dla ładnego wyświetlania
        df_assets = pd.DataFrame(assets)

        # Obliczamy bieżącą wartość (symulacja, bo w assets mamy cenę wejścia)
        # W wersji PRO tutaj pobieralibyśmy aktualną cenę z DataFeed,
        # ale na potrzeby widoku portfela wystarczy podgląd wejścia.

        # Formatowanie tabeli
        display_df = df_assets.copy()

        # Jeśli mamy kolumny z datami w ISO, sformatujmy je
        if 'ts' in display_df.columns:
            display_df['entry_time'] = display_df['ts'] # Uproszczenie

        # Wybór kolumn do wyświetlenia
        cols_to_show = ['sym', 'amt', 'entry', 'cost', 'ts']
        # Filtrujemy tylko te, które istnieją
        final_cols = [c for c in cols_to_show if c in display_df.columns]

        st.dataframe(
            display_df[final_cols].style.format({
                "entry": "${:.2f}",
                "cost": "${:.2f}",
                "amt": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Wykres kołowy alokacji
        if len(assets) > 0:
            fig = px.pie(df_assets, values='cost', names='sym', title='Exposure Allocation',
                         color_discrete_sequence=[theme['p'], '#444', '#666'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No active positions. Capital is 100% liquid.")

    st.divider()

    # --- SEKCJA 2: HISTORIA (LOGI) ---
    st.markdown("### 📜 TRANSACTION LEDGER")

    if history:
        # Odwracamy kolejność (najnowsze na górze)
        df_hist = pd.DataFrame(history).iloc[::-1]

        st.dataframe(
            df_hist,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.caption("No transaction history recorded yet.")

    # --- SEKCJA 3: DEBUG / MANUAL ACTIONS ---
    with st.expander("🛠️ TELLER OPERATIONS (Manual Deposit/Withdraw)", expanded=False):
        c_dep1, c_dep2 = st.columns(2)
        amount = c_dep1.number_input("Amount ($)", min_value=10.0, step=100.0, value=1000.0)

        if c_dep2.button("DEPOSIT FUNDS"):
            new_bal = balance + amount
            wallet['balance'] = round(new_bal, 2)

            # Log
            if 'history' not in wallet: wallet['history'] = []
            wallet['history'].append({
                "date": datetime.now().strftime('%Y-%m-%d'),
                "action": "DEPOSIT",
                "desc": f"MANUAL TRANSFER +${amount}",
                "pnl_val": 0
            })

            WalletManager.save_wallet_data(wallet)
            st.success(f"Deposited ${amount}. New Balance: ${new_bal:.2f}")
            st.rerun()

        if c_dep2.button("WITHDRAW FUNDS"):
            if balance >= amount:
                new_bal = balance - amount
                wallet['balance'] = round(new_bal, 2)
                 # Log
                if 'history' not in wallet: wallet['history'] = []
                wallet['history'].append({
                    "date": datetime.now().strftime('%Y-%m-%d'),
                    "action": "WITHDRAW",
                    "desc": f"MANUAL TRANSFER -${amount}",
                    "pnl_val": 0
                })
                WalletManager.save_wallet_data(wallet)
                st.success(f"Withdrawn ${amount}. New Balance: ${new_bal:.2f}")
                st.rerun()
            else:
                st.error("Insufficient funds.")
