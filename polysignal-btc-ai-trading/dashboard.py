"""
dashboard.py — PolySignal Performance Dashboard

Run:
    python3 -m streamlit run dashboard.py
"""

import sqlite3
import json
import re
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# ── DB resolution ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
for _candidate in [
    _HERE / "polysignal_btc.db",
    _HERE / "data" / "polysignal_final.db",
]:
    if _candidate.exists():
        DB_PATH = str(_candidate)
        break
else:
    DB_PATH = str(_HERE / "polysignal_btc.db")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PolySignal · BTC AI Trading",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens (theme-neutral trace colors) ────────────────────────────────
C = {
    "green":  "#10b981",
    "red":    "#f43f5e",
    "blue":   "#38bdf8",
    "yellow": "#f59e0b",
    "purple": "#818cf8",
}

# Transparent chart backgrounds + neutral colors that read on both themes
GRID  = "rgba(128,128,128,0.15)"
MUTED = "#888"

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color=MUTED, size=12),
    margin=dict(l=10, r=10, t=10, b=10),
)

# ── Custom CSS — uses Streamlit CSS variables so it adapts to any theme ───────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
#MainMenu, footer, header   { visibility: hidden; }

/* ── Hero banner — adapts via prefers-color-scheme ── */
.ps-hero {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 16px;
    padding: 22px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
@media (prefers-color-scheme: dark) {
    .ps-hero {
        background: linear-gradient(135deg, var(--background-color) 0%, #1a1040 55%, var(--background-color) 100%);
        border-color: rgba(129,140,248,.2);
    }
}
@media (prefers-color-scheme: light) {
    .ps-hero {
        background: linear-gradient(135deg, var(--background-color) 0%, #eff6ff 55%, var(--background-color) 100%);
        border-color: rgba(99,102,241,.15);
    }
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.12);
    border-radius: 12px;
    padding: 18px 20px !important;
    transition: border-color .2s, box-shadow .2s;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(128,128,128,0.3);
    box-shadow: 0 0 0 1px rgba(128,128,128,0.15);
}
[data-testid="stMetricLabel"] p {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    color: var(--text-color) !important;
    opacity: 0.45;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: var(--text-color) !important;
    letter-spacing: -0.5px;
}
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── Section labels ── */
.sec-label {
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--text-color);
    opacity: 0.35;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(128,128,128,0.12);
}

/* ── Misc ── */
hr                              { border-color: rgba(128,128,128,0.12) !important; margin: 20px 0 !important; }
[data-testid="stDataFrame"]     { border: 1px solid rgba(128,128,128,0.12) !important; border-radius: 12px !important; }
.stAlert                        { border: 1px solid rgba(128,128,128,0.12) !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero banner ───────────────────────────────────────────────────────────────
now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
st.markdown(f"""
<div class="ps-hero">
  <div>
    <div style="
        font-size:1.9rem;font-weight:700;letter-spacing:-0.6px;line-height:1.15;
        background:linear-gradient(90deg,#818cf8 0%,#38bdf8 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
      ₿ PolySignal
    </div>
    <div style="font-size:.82rem;color:var(--text-color);opacity:.45;margin-top:5px;">
      BTC 5-Min Prediction Market &nbsp;·&nbsp; Multi-Agent AI Ensemble
    </div>
  </div>
  <div style="text-align:right;">
    <div style="
        background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.3);
        color:#10b981;padding:5px 14px;border-radius:20px;
        font-size:.72rem;font-weight:700;letter-spacing:.6px;display:inline-block;">
      ● LIVE
    </div>
    <div style="font-size:.7rem;color:var(--text-color);opacity:.3;margin-top:5px;">{now_str}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_data():
    conn = sqlite3.connect(DB_PATH)

    trades = pd.read_sql_query(
        "SELECT * FROM trades WHERE resolved=1 ORDER BY resolved_at ASC", conn)
    pending = pd.read_sql_query(
        "SELECT * FROM trades WHERE resolved=0", conn)
    evals = pd.read_sql_query(
        "SELECT * FROM eval_records ORDER BY scored_at ASC", conn)
    account = pd.read_sql_query("SELECT * FROM account", conn)

    conn.close()

    if not trades.empty:
        trades["resolved_at"] = pd.to_datetime(trades["resolved_at"])
        trades["created_at"]  = pd.to_datetime(trades["created_at"])

    if not evals.empty:
        evals["scored_at"]   = pd.to_datetime(evals["scored_at"])
        evals["was_correct"] = evals["was_correct"].astype(bool)
        evals["rolling_acc"] = evals["was_correct"].expanding().mean() * 100

    return trades, pending, evals, account

trades, pending_trades, evals, account = load_data()

# ── KPIs ──────────────────────────────────────────────────────────────────────
initial_balance = float(account["initial_balance"].iloc[0]) if not account.empty else 100_000
current_balance = float(account["balance"].iloc[0])         if not account.empty else 100_000
total_pnl       = current_balance - initial_balance
pnl_pct         = (total_pnl / initial_balance) * 100

total_trades = len(trades)
wins         = int((trades["pnl"] > 0).sum())  if not trades.empty else 0
losses       = int((trades["pnl"] <= 0).sum()) if not trades.empty else 0
win_rate     = (wins / total_trades * 100)      if total_trades > 0 else 0

total_scored = len(evals)
accuracy     = (evals["was_correct"].mean() * 100) if not evals.empty else 0
brier        = evals["brier_score"].mean()          if not evals.empty else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Portfolio Value",    f"${current_balance:,.2f}",
          f"{'+' if total_pnl >= 0 else ''}{total_pnl:,.2f} ({pnl_pct:+.2f}%)")
c2.metric("Total Trades",       f"{total_trades}",
          f"{len(pending_trades)} pending")
c3.metric("Win Rate",           f"{win_rate:.1f}%",
          f"{wins}W  /  {losses}L")
c4.metric("Predictions Scored", f"{total_scored}")
c5.metric("Directional Acc.",   f"{accuracy:.1f}%",
          f"{'▲' if accuracy > 50 else '▼'} vs 50% baseline")
c6.metric("Avg Brier Score",    f"{brier:.4f}",
          "↓ lower is better")

st.divider()

# ── Row 1: Balance curve + P&L bar ───────────────────────────────────────────
col_l, col_r = st.columns([2, 1])

with col_l:
    st.markdown('<div class="sec-label">Portfolio Performance</div>', unsafe_allow_html=True)

    if trades.empty:
        st.info("No settled trades yet — run the pipeline to generate data.")
    else:
        bh = pd.concat([
            pd.DataFrame({
                "time":    [trades["created_at"].min() - pd.Timedelta(minutes=5)],
                "balance": [initial_balance],
                "pnl":     [0.0],
                "won":     [None],
            }),
            trades[["resolved_at", "balance_after", "pnl"]].rename(
                columns={"resolved_at": "time", "balance_after": "balance"}
            ).assign(won=trades["pnl"] > 0),
        ], ignore_index=True)

        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(
            x=bh["time"], y=bh["balance"],
            mode="lines",
            line=dict(color=C["blue"], width=2.5),
            fill="tozeroy", fillcolor="rgba(56,189,248,0.06)",
            name="Balance",
            hovertemplate="<b>%{x|%b %d %H:%M}</b><br>Balance: $%{y:,.2f}<extra></extra>",
        ))
        wins_df = bh[bh["won"] == True]
        fig_b.add_trace(go.Scatter(
            x=wins_df["time"], y=wins_df["balance"], mode="markers",
            marker=dict(color=C["green"], size=8, symbol="circle",
                        line=dict(color="rgba(0,0,0,0)", width=1.5)),
            name="Win",
            hovertemplate="<b>WIN</b><br>%{x|%b %d %H:%M}<br>$%{y:,.2f}<extra></extra>",
        ))
        loss_df = bh[bh["won"] == False]
        fig_b.add_trace(go.Scatter(
            x=loss_df["time"], y=loss_df["balance"], mode="markers",
            marker=dict(color=C["red"], size=8, symbol="circle",
                        line=dict(color="rgba(0,0,0,0)", width=1.5)),
            name="Loss",
            hovertemplate="<b>LOSS</b><br>%{x|%b %d %H:%M}<br>$%{y:,.2f}<extra></extra>",
        ))
        fig_b.add_hline(
            y=initial_balance,
            line_dash="dot", line_color=GRID, line_width=1.5,
            annotation_text="Starting balance",
            annotation_position="bottom right",
            annotation_font=dict(color=MUTED, size=11),
        )
        _bal_min = bh["balance"].min()
        _bal_max = bh["balance"].max()
        _pad     = (_bal_max - _bal_min) * 0.15 or 2000
        fig_b.update_layout(
            **BASE_LAYOUT, height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
            yaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor=GRID,
                       linecolor=GRID, zeroline=False,
                       range=[_bal_min - _pad, _bal_max + _pad]),
            xaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False),
            hovermode="x unified",
        )
        st.plotly_chart(fig_b, use_container_width=True)

with col_r:
    st.markdown('<div class="sec-label">P&L Per Trade</div>', unsafe_allow_html=True)

    if trades.empty:
        st.info("No data yet.")
    else:
        fig_pnl = go.Figure(go.Bar(
            x=list(range(1, len(trades) + 1)),
            y=trades["pnl"],
            marker_color=[C["green"] if p > 0 else C["red"] for p in trades["pnl"]],
            marker_opacity=0.85,
            hovertemplate="Trade #%{x}<br>P&L: $%{y:,.2f}<extra></extra>",
        ))
        fig_pnl.add_hline(y=0, line_color=GRID, line_width=1.5)
        fig_pnl.update_layout(
            **BASE_LAYOUT, height=360,
            xaxis_title="Trade #",
            xaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False),
            yaxis=dict(tickprefix="$", gridcolor=GRID, linecolor=GRID, zeroline=False),
            showlegend=False,
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

st.divider()

# ── Row 2: Rolling acc + Calibration + Direction split ────────────────────────
ca, cb, cc = st.columns(3)

with ca:
    st.markdown('<div class="sec-label">Rolling Directional Accuracy</div>', unsafe_allow_html=True)

    if evals.empty:
        st.info("No scored predictions yet.")
    else:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=evals["scored_at"], y=evals["rolling_acc"],
            mode="lines",
            line=dict(color=C["yellow"], width=2.5),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.07)",
            hovertemplate="%{x|%b %d %H:%M}<br>Accuracy: %{y:.1f}%<extra></extra>",
        ))
        fig_acc.add_hline(
            y=50, line_dash="dot", line_color=GRID, line_width=1.5,
            annotation_text="50% baseline",
            annotation_position="bottom right",
            annotation_font=dict(color=MUTED, size=11),
        )
        fig_acc.update_layout(
            **BASE_LAYOUT, height=290,
            xaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False),
            yaxis=dict(ticksuffix="%", range=[30, 80], gridcolor=GRID,
                       linecolor=GRID, zeroline=False),
            showlegend=False,
        )
        st.plotly_chart(fig_acc, use_container_width=True)

with cb:
    st.markdown('<div class="sec-label">Confidence Calibration</div>', unsafe_allow_html=True)

    if evals.empty:
        st.info("No data yet.")
    else:
        bins   = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 1.01)]
        labels = ["50–55%", "55–60%", "60–65%", "65%+"]
        act_acc, avg_con = [], []
        for lo, hi in bins:
            mask = (evals["confidence"] >= lo) & (evals["confidence"] < hi)
            sub  = evals[mask]
            if len(sub) > 0:
                act_acc.append(sub["was_correct"].mean() * 100)
                avg_con.append(sub["confidence"].mean() * 100)
            else:
                act_acc.append(None); avg_con.append(None)

        fig_cal = go.Figure()
        fig_cal.add_trace(go.Bar(
            name="Actual Accuracy", x=labels, y=act_acc,
            marker_color=C["blue"], marker_opacity=0.85,
            hovertemplate="%{x}<br>Actual: %{y:.1f}%<extra></extra>",
        ))
        fig_cal.add_trace(go.Bar(
            name="Avg Confidence", x=labels, y=avg_con,
            marker_color=C["yellow"], marker_opacity=0.7,
            hovertemplate="%{x}<br>Confidence: %{y:.1f}%<extra></extra>",
        ))
        fig_cal.add_hline(y=50, line_dash="dot", line_color=GRID, line_width=1.5)
        fig_cal.update_layout(
            **BASE_LAYOUT, height=290,
            barmode="group",
            xaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False),
            yaxis=dict(ticksuffix="%", range=[0, 100], gridcolor=GRID,
                       linecolor=GRID, zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_cal, use_container_width=True)

with cc:
    st.markdown('<div class="sec-label">UP vs DOWN Calls</div>', unsafe_allow_html=True)

    if evals.empty:
        st.info("No data yet.")
    else:
        up_c = int(evals[evals["direction"] == "UP"]["was_correct"].sum())
        up_w = int((evals["direction"] == "UP").sum()) - up_c
        dn_c = int(evals[evals["direction"] == "DOWN"]["was_correct"].sum())
        dn_w = int((evals["direction"] == "DOWN").sum()) - dn_c

        fig_dir = go.Figure(go.Bar(
            x=["UP ✓", "UP ✗", "DOWN ✓", "DOWN ✗"],
            y=[up_c, up_w, dn_c, dn_w],
            marker_color=[C["green"], C["red"], C["green"], C["red"]],
            marker_opacity=0.85,
            text=[up_c, up_w, dn_c, dn_w],
            textposition="outside",
            textfont=dict(size=12, color=MUTED),
            hovertemplate="%{x}: %{y}<extra></extra>",
        ))
        fig_dir.update_layout(
            **BASE_LAYOUT, height=290,
            xaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False),
            yaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False),
            showlegend=False,
        )
        st.plotly_chart(fig_dir, use_container_width=True)

st.divider()

# ── Row 3: Per-model accuracy ─────────────────────────────────────────────────
st.markdown('<div class="sec-label">Per-Model Performance</div>', unsafe_allow_html=True)

@st.cache_data(ttl=30)
def load_per_model():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT p.direction, p.confidence, p.actual_outcome, p.reasoning, p.key_factors
        FROM predictions p
        JOIN eval_records e ON p.prediction_id = e.prediction_id
        ORDER BY p.created_at ASC
    """).fetchall()
    conn.close()

    models = {}

    def _record(label, direction, confidence, actual):
        correct = (direction.upper() == actual)
        b = (confidence - (1.0 if correct else 0.0)) ** 2
        if label not in models:
            models[label] = {"total": 0, "correct": 0, "brier_sum": 0.0}
        models[label]["total"]     += 1
        models[label]["correct"]   += int(correct)
        models[label]["brier_sum"] += b

    for direction, confidence, actual, reasoning, kf_raw in rows:
        if not actual:
            continue
        if reasoning and reasoning.startswith("[ENSEMBLE"):
            m = re.search(r'Votes: (\[.*?\])', reasoning)
            if m:
                try:
                    for vote in json.loads(m.group(1)):
                        _record(vote.get("model", "?"),
                                str(vote.get("direction", "")),
                                float(vote.get("confidence", 0.5)), actual)
                except Exception:
                    pass
        elif reasoning and reasoning.startswith("[DEBATE"):
            try:
                kf = json.loads(kf_raw) if kf_raw else []
            except Exception:
                kf = []
            for item in kf:
                fm = re.match(r'(xAI|OpenAI) final: (UP|DOWN) @ (\d+)%', str(item))
                if fm:
                    lbl = "xAI (Grok)" if fm.group(1) == "xAI" else "OpenAI (GPT-4o-mini)"
                    _record(lbl, fm.group(2), float(fm.group(3)) / 100, actual)
            _record("Claude (Judge)", direction, float(confidence), actual)

    out = []
    for label, d in sorted(models.items()):
        t, c = d["total"], d["correct"]
        acc  = c / t if t > 0 else 0.0
        out.append({
            "Model": label, "Votes": t, "Correct": c,
            "Accuracy": acc,
            "Avg Brier": round(d["brier_sum"] / t, 4) if t > 0 else 0.0,
        })
    return pd.DataFrame(out)

pm_df = load_per_model()
if pm_df.empty:
    st.info("No scored predictions yet.")
else:
    col_pm_chart, col_pm_table = st.columns([3, 2])

    with col_pm_chart:
        fig_pm = go.Figure(go.Bar(
            x=pm_df["Model"],
            y=pm_df["Accuracy"] * 100,
            marker_color=[
                C["green"] if a >= 0.55 else C["yellow"] if a >= 0.50 else C["red"]
                for a in pm_df["Accuracy"]
            ],
            marker_opacity=0.85,
            text=[f"{a:.1%}" for a in pm_df["Accuracy"]],
            textposition="outside",
            textfont=dict(size=12, color=MUTED),
            hovertemplate="%{x}<br>Accuracy: %{y:.1f}%<br>Votes: %{customdata}<extra></extra>",
            customdata=pm_df["Votes"],
        ))
        fig_pm.add_hline(
            y=50, line_dash="dot", line_color=GRID, line_width=1.5,
            annotation_text="50% baseline",
            annotation_position="bottom right",
            annotation_font=dict(color=MUTED, size=11),
        )
        fig_pm.update_layout(
            **{**BASE_LAYOUT, "margin": dict(l=10, r=10, t=30, b=10)},
            height=280,
            xaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False),
            yaxis=dict(ticksuffix="%", range=[0, 90], gridcolor=GRID,
                       linecolor=GRID, zeroline=False),
            showlegend=False,
        )
        st.plotly_chart(fig_pm, use_container_width=True)

    with col_pm_table:
        disp_pm = pm_df.copy()
        disp_pm["Accuracy"] = disp_pm["Accuracy"].map(lambda x: f"{x:.1%}")
        st.dataframe(disp_pm, use_container_width=True, hide_index=True, height=280)

st.divider()

# ── Row 4: Trade history ──────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Trade History</div>', unsafe_allow_html=True)

if trades.empty:
    st.info("No settled trades yet.")
else:
    disp = trades[["resolved_at", "direction", "odds_price",
                   "bet_amount", "pnl", "balance_after"]].copy()
    disp["resolved_at"]   = disp["resolved_at"].dt.strftime("%b %d  %H:%M")
    disp["odds_price"]    = disp["odds_price"].map(lambda x: f"{x:.1%}")
    disp["bet_amount"]    = disp["bet_amount"].map(lambda x: f"${x:,.0f}")
    disp["pnl"]           = disp["pnl"].map(
        lambda x: f"+${x:,.2f}" if x > 0 else f"-${abs(x):,.2f}")
    disp["balance_after"] = disp["balance_after"].map(lambda x: f"${x:,.2f}")
    disp.columns = ["Time (UTC)", "Direction", "Odds", "Bet", "P&L", "Balance After"]
    disp = disp.iloc[::-1].reset_index(drop=True)

    st.dataframe(
        disp,
        use_container_width=True,
        height=300,
        column_config={"P&L": st.column_config.TextColumn("P&L")},
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Dashboard")
    st.caption(f"DB: `{Path(DB_PATH).name}`")
    st.caption(f"Updated: {datetime.utcnow().strftime('%H:%M:%S UTC')}")

    if st.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("**Session**")
    st.write(f"Settled trades:  **{total_trades}**")
    st.write(f"Pending trades:  **{len(pending_trades)}**")
    st.write(f"Scored preds:    **{total_scored}**")
    st.write(f"Win rate:        **{win_rate:.1f}%**")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="text-align:center;font-size:.68rem;opacity:.3;margin-top:12px;">'
    f'PolySignal · {Path(DB_PATH).name} · auto-refresh every 30s · {now_str}'
    f'</div>',
    unsafe_allow_html=True,
)
