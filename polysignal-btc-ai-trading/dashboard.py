"""
dashboard.py — PolySignal Performance Dashboard

Run locally after downloading the DB from each session:
    streamlit run dashboard.py

Requires: pip install streamlit plotly pandas
"""

import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime

DB_PATH = "polysignal_btc.db"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PolySignal Dashboard",
    page_icon="₿",
    layout="wide",
)

st.title("₿ PolySignal — Performance Dashboard")
st.caption("BTC 5-Min Prediction Market · Multi-Agent Ensemble")

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_data():
    conn = sqlite3.connect(DB_PATH)

    trades = pd.read_sql_query("""
        SELECT * FROM trades
        WHERE resolved = 1
        ORDER BY resolved_at ASC
    """, conn)

    pending_trades = pd.read_sql_query("""
        SELECT * FROM trades WHERE resolved = 0
    """, conn)

    evals = pd.read_sql_query("""
        SELECT * FROM eval_records ORDER BY scored_at ASC
    """, conn)

    account = pd.read_sql_query("SELECT * FROM account", conn)

    conn.close()

    # Parse timestamps
    if not trades.empty:
        trades["resolved_at"] = pd.to_datetime(trades["resolved_at"])
        trades["created_at"]  = pd.to_datetime(trades["created_at"])

    if not evals.empty:
        evals["scored_at"] = pd.to_datetime(evals["scored_at"])
        evals["was_correct"] = evals["was_correct"].astype(bool)
        evals["rolling_acc"] = evals["was_correct"].expanding().mean() * 100

    return trades, pending_trades, evals, account

trades, pending_trades, evals, account = load_data()

# ── Top-level metrics ─────────────────────────────────────────────────────────

initial_balance = float(account["initial_balance"].iloc[0]) if not account.empty else 100_000
current_balance = float(account["balance"].iloc[0])         if not account.empty else 100_000
total_pnl       = current_balance - initial_balance
pnl_pct         = (total_pnl / initial_balance) * 100

total_trades    = len(trades)
wins            = int((trades["pnl"] > 0).sum()) if not trades.empty else 0
losses          = int((trades["pnl"] <= 0).sum()) if not trades.empty else 0
win_rate        = (wins / total_trades * 100) if total_trades > 0 else 0

total_scored    = len(evals)
accuracy        = (evals["was_correct"].mean() * 100) if not evals.empty else 0
brier           = evals["brier_score"].mean()          if not evals.empty else 0

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Account Balance",  f"${current_balance:,.2f}",
            f"{'+' if total_pnl >= 0 else ''}{total_pnl:,.2f} ({pnl_pct:+.2f}%)")
col2.metric("Total Trades",     f"{total_trades}",
            f"{len(pending_trades)} pending")
col3.metric("Win Rate",         f"{win_rate:.1f}%",
            f"{wins}W / {losses}L")
col4.metric("Predictions",      f"{total_scored} scored")
col5.metric("Directional Acc.", f"{accuracy:.1f}%",
            f"{'▲' if accuracy > 50 else '▼'} vs 50% baseline")
col6.metric("Brier Score",      f"{brier:.4f}",
            "↓ lower is better (0.25 = random)")

st.divider()

# ── Row 1: Account balance + P&L bar ─────────────────────────────────────────

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Account Balance Over Time")

    if trades.empty:
        st.info("No settled trades yet — run the pipeline to generate data.")
    else:
        # Build balance history including initial point
        balance_history = pd.concat([
            pd.DataFrame({
                "time":    [trades["created_at"].min() - pd.Timedelta(minutes=5)],
                "balance": [initial_balance],
                "pnl":     [0.0],
                "won":     [None],
                "direction": ["—"],
            }),
            trades[["resolved_at", "balance_after", "pnl", "direction"]].rename(
                columns={"resolved_at": "time", "balance_after": "balance"}
            ).assign(won=trades["pnl"] > 0),
        ], ignore_index=True)

        fig_balance = go.Figure()

        # Balance line
        fig_balance.add_trace(go.Scatter(
            x=balance_history["time"],
            y=balance_history["balance"],
            mode="lines",
            line=dict(color="#00bfff", width=2.5),
            name="Balance",
            hovertemplate="<b>%{x|%b %d %H:%M}</b><br>Balance: $%{y:,.2f}<extra></extra>",
        ))

        # Win markers
        wins_df = balance_history[balance_history["won"] == True]
        fig_balance.add_trace(go.Scatter(
            x=wins_df["time"],
            y=wins_df["balance"],
            mode="markers",
            marker=dict(color="#00e676", size=10, symbol="circle",
                        line=dict(color="white", width=1)),
            name="Win",
            hovertemplate="<b>WIN</b><br>%{x|%b %d %H:%M}<br>Balance: $%{y:,.2f}<extra></extra>",
        ))

        # Loss markers
        losses_df = balance_history[balance_history["won"] == False]
        fig_balance.add_trace(go.Scatter(
            x=losses_df["time"],
            y=losses_df["balance"],
            mode="markers",
            marker=dict(color="#ff5252", size=10, symbol="circle",
                        line=dict(color="white", width=1)),
            name="Loss",
            hovertemplate="<b>LOSS</b><br>%{x|%b %d %H:%M}<br>Balance: $%{y:,.2f}<extra></extra>",
        ))

        # Baseline reference
        fig_balance.add_hline(
            y=initial_balance,
            line_dash="dash", line_color="gray", line_width=1,
            annotation_text="Starting $100k", annotation_position="bottom right",
        )

        fig_balance.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(tickprefix="$", tickformat=",.0f"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_balance, use_container_width=True)

with col_right:
    st.subheader("P&L Per Trade")

    if trades.empty:
        st.info("No data yet.")
    else:
        fig_pnl = go.Figure(go.Bar(
            x=list(range(1, len(trades) + 1)),
            y=trades["pnl"],
            marker_color=["#00e676" if p > 0 else "#ff5252" for p in trades["pnl"]],
            hovertemplate="Trade #%{x}<br>P&L: $%{y:,.2f}<extra></extra>",
        ))
        fig_pnl.add_hline(y=0, line_color="gray", line_width=1)
        fig_pnl.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Trade #",
            yaxis=dict(tickprefix="$"),
            showlegend=False,
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

st.divider()

# ── Row 2: Rolling accuracy + Calibration + Direction split ───────────────────

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.subheader("Rolling Accuracy")

    if evals.empty:
        st.info("No scored predictions yet.")
    else:
        fig_acc = go.Figure()

        fig_acc.add_trace(go.Scatter(
            x=evals["scored_at"],
            y=evals["rolling_acc"],
            mode="lines",
            line=dict(color="#ffd740", width=2.5),
            name="Rolling Accuracy",
            hovertemplate="%{x|%b %d %H:%M}<br>Accuracy: %{y:.1f}%<extra></extra>",
            fill="tozeroy",
            fillcolor="rgba(255,215,64,0.1)",
        ))

        fig_acc.add_hline(
            y=50, line_dash="dash", line_color="gray", line_width=1,
            annotation_text="50% baseline", annotation_position="bottom right",
        )

        fig_acc.update_layout(
            template="plotly_dark",
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(ticksuffix="%", range=[30, 80]),
            showlegend=False,
        )
        st.plotly_chart(fig_acc, use_container_width=True)

with col_b:
    st.subheader("Confidence Calibration")

    if evals.empty:
        st.info("No data yet.")
    else:
        bins    = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 1.01)]
        labels  = ["50–55%", "55–60%", "60–65%", "65%+"]
        act_acc = []
        avg_con = []
        counts  = []

        for lo, hi in bins:
            mask = (evals["confidence"] >= lo) & (evals["confidence"] < hi)
            subset = evals[mask]
            if len(subset) > 0:
                act_acc.append(subset["was_correct"].mean() * 100)
                avg_con.append(subset["confidence"].mean() * 100)
                counts.append(len(subset))
            else:
                act_acc.append(None)
                avg_con.append(None)
                counts.append(0)

        fig_cal = go.Figure()

        fig_cal.add_trace(go.Bar(
            name="Actual Accuracy",
            x=labels, y=act_acc,
            marker_color="#00bfff",
            hovertemplate="%{x}<br>Actual: %{y:.1f}%<extra></extra>",
        ))
        fig_cal.add_trace(go.Bar(
            name="Avg Confidence",
            x=labels, y=avg_con,
            marker_color="#ffd740",
            hovertemplate="%{x}<br>Confidence: %{y:.1f}%<extra></extra>",
        ))
        fig_cal.add_hline(y=50, line_dash="dash", line_color="gray", line_width=1)

        fig_cal.update_layout(
            template="plotly_dark",
            height=280,
            barmode="group",
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(ticksuffix="%", range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_cal, use_container_width=True)

with col_c:
    st.subheader("UP vs DOWN Calls")

    if evals.empty:
        st.info("No data yet.")
    else:
        up_correct   = int(evals[evals["direction"] == "UP"]["was_correct"].sum())
        up_wrong     = int((evals["direction"] == "UP").sum()) - up_correct
        down_correct = int(evals[evals["direction"] == "DOWN"]["was_correct"].sum())
        down_wrong   = int((evals["direction"] == "DOWN").sum()) - down_correct

        fig_dir = go.Figure(go.Bar(
            x=["UP ✓", "UP ✗", "DOWN ✓", "DOWN ✗"],
            y=[up_correct, up_wrong, down_correct, down_wrong],
            marker_color=["#00e676", "#ff5252", "#00e676", "#ff5252"],
            hovertemplate="%{x}: %{y}<extra></extra>",
        ))
        fig_dir.update_layout(
            template="plotly_dark",
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_dir, use_container_width=True)

st.divider()

# ── Row 3: Trade history table ────────────────────────────────────────────────

st.subheader("Trade History")

if trades.empty:
    st.info("No settled trades yet.")
else:
    display = trades[["resolved_at", "direction", "odds_price",
                       "bet_amount", "pnl", "balance_after"]].copy()
    display["resolved_at"] = display["resolved_at"].dt.strftime("%b %d %H:%M")
    display["odds_price"]  = display["odds_price"].map(lambda x: f"{x:.1%}")
    display["bet_amount"]  = display["bet_amount"].map(lambda x: f"${x:,.0f}")
    display["pnl"]         = display["pnl"].map(
        lambda x: f"+${x:,.2f}" if x > 0 else f"-${abs(x):,.2f}"
    )
    display["balance_after"] = display["balance_after"].map(lambda x: f"${x:,.2f}")
    display.columns = ["Time (UTC)", "Direction", "Odds", "Bet", "P&L", "Balance After"]
    display = display.iloc[::-1].reset_index(drop=True)  # newest first

    st.dataframe(
        display,
        use_container_width=True,
        height=300,
        column_config={
            "P&L": st.column_config.TextColumn("P&L"),
        },
    )

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"DB: `{DB_PATH}`  ·  "
    f"Last refreshed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  ·  "
    f"Auto-refreshes every 30s"
)
