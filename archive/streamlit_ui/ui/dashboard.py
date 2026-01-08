import streamlit as st
import time
import plotly.graph_objects as go

# =====================================================
# CORE
# =====================================================
from core.logger import log_event

# DATA & ML
from data.feed import DataFeed
from ml.brain import DeepBrain
from ml.regime import MarketRegime
from ml.knowledge import KnowledgeBase

# TRADING LOGIC
from trading.decision import DecisionEngine
from trading.execution import ExecutionEngine
from trading.analytics import TradeAnalytics


# =====================================================
# UI HELPERS
# =====================================================
def card_open(title: str = None):
    st.markdown(
        f"""
        <div style="
            background: var(--panel);
            border: 1px solid var(--accent);
            border-radius: 10px;
            padding: 18px;
            margin-bottom: 18px;
        ">
        {f"<div style='font-size:13px; color:var(--muted); margin-bottom:8px; letter-spacing:1px'>{title}</div>" if title else ""}
        """,
        unsafe_allow_html=True,
    )


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# MAIN DASHBOARD
# =====================================================
def render_dashboard():
    """
    PREMIUM DASHBOARD
    - OPS MODE (ADMIN / ROOT)
    - INVESTOR MODE (GUEST / USER)
    - REPORT LOOK (read-only)
    """

    sys = st.session_state.get("sys", {})
    role = sys.get("role", "GUEST")
    user = sys.get("user", "viewer")

    is_ops = role in {"ADMIN", "ROOT"}
    is_investor = not is_ops

    decision_engine = DecisionEngine(mode="PAPER")
    executor = ExecutionEngine(mode="PAPER")

    # =====================================================
    # 🔌 JS EVENT SYSTEM (ROLE-BASED)
    # =====================================================
    if is_ops:
        st.components.v1.html(
            """
            <script src="/static/event_renderer.js"></script>
            <script src="/static/events_ops.js"></script>
            """,
            height=0,
        )
    else:
        st.components.v1.html(
            """
            <script src="/static/event_renderer.js"></script>
            <script src="/static/events_investor.js"></script>
            """,
            height=0,
        )

    # =====================================================
    # HUD — SYSTEM PERFORMANCE
    # =====================================================
    card_open("SYSTEM PERFORMANCE (PAPER)")

    stats = TradeAnalytics.generate_report()

    hud = st.columns(4)
    hud[0].metric("Win Rate", f"{stats['win_rate']}%", f"{stats['total_trades']} trades")
    hud[1].metric("Profit Factor", f"{stats['profit_factor']:.2f}")
    hud[2].metric("Expectancy (EV)", f"${stats['expectancy']:.2f}")
    hud[3].metric("Net PnL", f"${stats['net_profit']:.2f}", f"DD {stats['max_drawdown']:.2f}")

    card_close()

    # =====================================================
    # INVESTOR MODE — REPORT VIEW
    # =====================================================
    if is_investor:
        card_open("INVESTOR REPORT")

        st.caption("Read-only performance overview")

        st.markdown(
            f"""
            <div style="font-size:14px; margin-bottom:12px;">
                Strategy Status: <b>ACTIVE (Paper)</b><br>
                Risk Profile: <b>Conservative / Systematic</b><br>
                Last Trade: <b>{stats.get("last_trade", "N/A")}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        card_close()
        return  # INVESTOR NIE WIDZI OPS

    # =====================================================
    # OPS MODE — ACTIVE OPERATIONS
    # =====================================================
    left, right = st.columns([1.2, 2.2], gap="large")

    # =============================
    # LEFT — OPS CONTROL
    # =============================
    with left:
        card_open("NEURAL OPS")

        symbol = st.selectbox(
            "Market",
            ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
        )
        tf = st.selectbox("Timeframe", ["15m", "1h", "4h"], index=1)

        if st.button("RUN MARKET SCAN", use_container_width=True):

            brain = DeepBrain()

            with st.status("Running inference pipeline...", expanded=True):
                df = DataFeed.get_market_data(symbol, tf)

                if df.empty:
                    st.warning("Market data unavailable.")
                    card_close()
                    return

                ai_price, conf, signal = brain.predict(df)
                mqs, regime = MarketRegime.analyze(df)

            price = df["close"].iloc[-1]
            upside = (ai_price - price) / price * 100
            ev = (upside * conf) - abs(upside) * (1 - conf) * 0.5

            card_open("AI SIGNAL")

            st.markdown(
                f"""
                <div style="text-align:center;">
                    <div style="font-size:26px; font-weight:600;">{signal}</div>
                    <div style="font-size:12px; opacity:0.8;">
                        CONF {conf*100:.1f}% | EV {ev:.2f} | MQS {mqs}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            card_close()

            card_open("RISK GATE")

            candidate = {
                "symbol": symbol,
                "signal": signal,
                "conf": conf,
                "mqs": mqs,
                "current_price": price,
            }

            approved, reason, size = decision_engine.evaluate_entry(candidate)

            if approved:
                st.info(f"Approved. Position size ${size:.2f}")

                if st.button(f"EXECUTE {signal}", use_container_width=True):
                    res = executor.execute_order(symbol, "BUY", size)
                    if res.get("status") == "FILLED":
                        st.success(f"Filled @ {res.get('avg_price'):.2f}")
                        log_event(f"MANUAL EXEC {symbol}", "TRADE")
                        time.sleep(0.6)
                        st.rerun()
                    else:
                        st.warning(res.get("reason", "Execution issue"))
            else:
                st.warning(f"Blocked: {reason}")

            card_close()

    # =============================
    # RIGHT — MARKET MAP
    # =============================
    with right:
        card_open("MARKET MAP")

        df_chart = DataFeed.get_market_data(symbol, tf)

        if not df_chart.empty:
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=df_chart["time"],
                        open=df_chart["open"],
                        high=df_chart["high"],
                        low=df_chart["low"],
                        close=df_chart["close"],
                        increasing_line_color="#7aa2f7",
                        decreasing_line_color="#9ca3af",
                    )
                ]
            )

            fig.update_layout(
                height=560,
                template="plotly_dark",
                margin=dict(t=20, b=20, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(rangeslider=dict(visible=False)),
            )

            st.plotly_chart(fig, use_container_width=True)

        card_close()

    # =====================================================
    # REPORT DISCLAIMER
    # =====================================================
    card_open("REPORT SUMMARY")

    st.markdown(
        """
        <div style="font-size:12px; opacity:0.75;">
            This system operates under predefined risk rules.
            Historical performance does not guarantee future results.
        </div>
        """,
        unsafe_allow_html=True,
    )

    card_close()
