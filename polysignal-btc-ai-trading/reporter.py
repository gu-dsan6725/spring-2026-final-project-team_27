"""
reporter.py — Accuracy metrics for BTC 5-Min predictions.

Computes:
  - Directional accuracy overall and by confidence bin
  - Brier score (calibration quality)
  - Profit-equivalent analysis (if you had traded at your stated confidence)
"""
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from storage import get_all_evals, get_recent_predictions, db_summary, INITIAL_BALANCE

console = Console()
REPORTS = Path("reports")

BINS = [
    ("50–55%", 0.50, 0.55),
    ("55–60%", 0.55, 0.60),
    ("60–65%", 0.60, 0.65),
    ("65%+",   0.65, 1.01),
]


def compute_metrics(evals: list[dict]) -> dict:
    if not evals:
        return {}

    total   = len(evals)
    correct = sum(1 for e in evals if e["was_correct"])
    brier   = sum(e["brier_score"] for e in evals) / total
    avg_chg = sum(abs(e["price_change_pct"]) for e in evals) / total

    # Direction distribution
    up_calls   = sum(1 for e in evals if e["direction"] == "UP")
    down_calls = total - up_calls

    # Confidence bins
    bins = {}
    for label, lo, hi in BINS:
        bucket = [e for e in evals if lo <= e["confidence"] < hi]
        if not bucket:
            continue
        t   = len(bucket)
        c   = sum(1 for e in bucket if e["was_correct"])
        avg = sum(e["confidence"] for e in bucket) / t
        bins[label] = {
            "total":    t,
            "correct":  c,
            "accuracy": round(c/t, 3),
            "avg_conf": round(avg, 3),
            "gap":      round(abs(c/t - avg), 3),
        }

    return {
        "total":               total,
        "correct":             correct,
        "accuracy":            round(correct/total, 3),
        "avg_brier":           round(brier, 4),
        "avg_price_move_pct":  round(avg_chg, 4),
        "up_calls":            up_calls,
        "down_calls":          down_calls,
        "bins":                bins,
    }


def print_report(metrics: dict, summary: dict):
    if not metrics:
        console.print(Panel(
            "[dim]No scored predictions yet.\n"
            "Predictions are scored automatically once their 5-min window closes.\n"
            "Run the system for a few cycles to accumulate data.[/dim]",
            title="Accuracy Report", border_style="yellow"
        ))
        return

    acc = metrics["accuracy"]
    col = "green" if acc >= 0.55 else "yellow" if acc >= 0.50 else "red"

    # ── Overall ───────────────────────────────────────────────────────────────
    t = Table(title="BTC 5-Min Prediction Accuracy", box=box.ROUNDED)
    t.add_column("Metric",  style="cyan",       width=30)
    t.add_column("Value",   style="bold white",  width=20)
    t.add_column("Note",    style="dim",          width=32)

    t.add_row("Windows predicted",  str(metrics["total"]),  "")
    t.add_row("Correct calls",      str(metrics["correct"]), "")
    t.add_row("Directional accuracy",
              f"[{col}]{acc:.1%}[/{col}]",
              "Coin-flip baseline = 50%")
    t.add_row("Avg Brier score",
              f"{metrics['avg_brier']:.4f}",
              "Perfect=0  Random=0.25")
    t.add_row("Avg actual price move",
              f"{metrics['avg_price_move_pct']:.3f}%",
              "Typical 5-min volatility")
    t.add_row("UP calls",   str(metrics["up_calls"]),   "")
    t.add_row("DOWN calls", str(metrics["down_calls"]), "")
    console.print(t)
    console.print()

    # ── Confidence bins ───────────────────────────────────────────────────────
    if metrics["bins"]:
        bt = Table(title="Confidence Calibration", box=box.SIMPLE_HEAVY, show_lines=True)
        bt.add_column("Confidence",  style="cyan",  width=12)
        bt.add_column("Predictions", width=13, justify="center")
        bt.add_column("Correct",     width=9,  justify="center")
        bt.add_column("Actual %",    width=10, justify="center")
        bt.add_column("Avg Conf",    width=10, justify="center")
        bt.add_column("Gap",         width=8,  justify="center")
        bt.add_column("Calibration", width=18, justify="center")

        for label, s in metrics["bins"].items():
            g   = s["gap"]
            cal = (
                "[green]Well calibrated[/green]" if g < 0.10 else
                "[yellow]Slightly off[/yellow]"  if g < 0.20 else
                "[red]Poorly calibrated[/red]"
            )
            bt.add_row(label, str(s["total"]), str(s["correct"]),
                       f"{s['accuracy']:.1%}", f"{s['avg_conf']:.1%}",
                       f"{g:.1%}", cal)
        console.print(bt)
        console.print()

    # ── Pipeline ──────────────────────────────────────────────────────────────
    st = Table(title="Pipeline Summary", box=box.ROUNDED)
    st.add_column("Metric", style="cyan", width=26)
    st.add_column("Value",  style="bold white", width=12)
    st.add_row("Total predictions",  str(summary.get("total_predictions", 0)))
    st.add_row("Pending (open)",     str(summary.get("pending", 0)))
    st.add_row("Scored",             str(summary.get("scored", 0)))
    st.add_row("Correct",            str(summary.get("correct", 0)))
    console.print(st)


def save_report(metrics: dict, summary: dict, newly_scored: list) -> Path:
    REPORTS.mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS / f"btc_report_{ts}.md"

    lines = [
        "# PolySignal BTC 5-Min — Accuracy Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Pipeline Summary",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total predictions | {summary.get('total_predictions', 0)} |",
        f"| Pending | {summary.get('pending', 0)} |",
        f"| Scored | {summary.get('scored', 0)} |",
        f"| Correct | {summary.get('correct', 0)} |",
        "",
    ]

    if not metrics:
        lines.append("_No scored predictions yet._")
    else:
        lines += [
            "## Overall Accuracy",
            f"| Metric | Value | Note |",
            f"|--------|-------|------|",
            f"| Directional accuracy | {metrics['accuracy']:.1%} | Coin-flip = 50% |",
            f"| Avg Brier score | {metrics['avg_brier']:.4f} | Perfect=0, Random=0.25 |",
            f"| UP calls | {metrics['up_calls']} | |",
            f"| DOWN calls | {metrics['down_calls']} | |",
            "",
            "## Confidence Calibration",
            "| Confidence | Count | Correct | Actual % | Gap | Calibration |",
            "|------------|-------|---------|----------|-----|-------------|",
        ]
        for label, s in metrics["bins"].items():
            g   = s["gap"]
            cal = "Well calibrated" if g < 0.10 else "Slightly off" if g < 0.20 else "Poorly calibrated"
            lines.append(
                f"| {label} | {s['total']} | {s['correct']} | "
                f"{s['accuracy']:.1%} | {g:.1%} | {cal} |"
            )

    if newly_scored:
        lines += [
            "",
            "## Scored This Run",
            "| Direction | Actual | Correct | BTC Δ | Brier |",
            "|-----------|--------|---------|-------|-------|",
        ]
        for ev in newly_scored:
            tick = "✓" if ev.was_correct else "✗"
            lines.append(
                f"| {ev.direction} ({ev.confidence:.0%}) | {ev.actual_outcome} "
                f"| {tick} | {ev.price_change_pct:+.3f}% | {ev.brier_score:.3f} |"
            )

    path.write_text("\n".join(lines))
    return path


def print_sim_report(sim: dict):
    """Print simulated account P&L to console."""
    if not sim:
        console.print(Panel(
            "[dim]No settled trades yet.\n"
            "Trades are settled automatically once the 5-min window closes.[/dim]",
            title="Simulated Account", border_style="yellow"
        ))
        return

    pnl_color = "green" if sim["total_pnl"] >= 0 else "red"
    roi_color = "green" if sim["roi_pct"] >= 0 else "red"

    t = Table(title="Simulated Account — $100k Starting Capital", box=box.ROUNDED)
    t.add_column("Metric",  style="cyan",      width=28)
    t.add_column("Value",   style="bold white", width=22)
    t.add_column("Note",    style="dim",        width=28)

    t.add_row("Current balance",
              f"[{pnl_color}]${sim['balance']:,.2f}[/{pnl_color}]",
              f"Started at ${INITIAL_BALANCE:,.0f}")
    t.add_row("Total P&L",
              f"[{pnl_color}]{sim['total_pnl']:+,.2f}[/{pnl_color}]", "")
    t.add_row("ROI",
              f"[{roi_color}]{sim['roi_pct']:+.3f}%[/{roi_color}]", "")
    t.add_row("Trades settled",   str(sim["total_trades"]), "")
    t.add_row("Win / Loss",
              f"[green]{sim['wins']}[/green] / [red]{sim['losses']}[/red]",
              f"Win rate {sim['win_rate']:.1%}")
    t.add_row("Avg win P&L",  f"[green]+${sim['avg_win']:,.2f}[/green]",  "per winning trade")
    t.add_row("Avg loss P&L", f"[red]-${abs(sim['avg_loss']):,.2f}[/red]", "per losing trade")
    t.add_row("Total wagered", f"${sim['total_wagered']:,.0f}",
              f"${500:.0f}/trade flat bet")
    console.print(t)
    console.print()

    # Last 5 trades
    recent = sim.get("trades", [])[:5]
    if recent:
        rt = Table(title="Last 5 Trades", box=box.SIMPLE_HEAVY, show_lines=True)
        rt.add_column("Direction",  width=10)
        rt.add_column("Actual",     width=8)
        rt.add_column("Entry odds", width=12, justify="center")
        rt.add_column("Bet",        width=8,  justify="right")
        rt.add_column("Payout",     width=10, justify="right")
        rt.add_column("P&L",        width=12, justify="right")
        rt.add_column("Balance",    width=14, justify="right")

        for tr in recent:
            won      = tr["direction"] == tr["actual_outcome"]
            pnl_str  = f"[green]+${tr['pnl']:,.2f}[/green]" if won else f"[red]-${abs(tr['pnl']):,.2f}[/red]"
            rt.add_row(
                f"[bold]{tr['direction']}[/bold]",
                tr["actual_outcome"] or "-",
                f"{tr['odds_price']:.2%}",
                f"${tr['bet_amount']:,.0f}",
                f"${tr['actual_payout']:,.2f}",
                pnl_str,
                f"${tr['balance_after']:,.2f}",
            )
        console.print(rt)
        console.print()


def save_analysis_log() -> Path:
    """Save the last 12 full analyzer reasoning entries to latest_analysis.md."""
    rows = get_recent_predictions(limit=12)
    lines = [
        "# PolySignal — Last 12 Analyzer Reasonings",
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for i, r in enumerate(rows, 1):
        direction  = r.get("direction", "?")
        confidence = float(r.get("confidence", 0))
        question   = r.get("market_question", "")
        reasoning  = r.get("reasoning", "")
        factors    = r.get("key_factors", "[]")
        btc        = float(r.get("btc_price_at_call", 0))
        created    = r.get("created_at", "")[:16]
        resolved   = r.get("resolved", 0)
        actual     = r.get("actual_outcome", "pending")
        correct    = None
        if resolved:
            correct = (direction == actual)

        result_str = ""
        if resolved:
            result_str = f"  →  {'✓ CORRECT' if correct else '✗ WRONG'}  (actual: {actual})"

        lines += [
            f"---",
            f"## {i}. {question}",
            f"**Called:** {direction}  |  **Confidence:** {confidence:.0%}  |  "
            f"**BTC:** ${btc:,.2f}  |  **Time:** {created} UTC{result_str}",
            "",
            "### Reasoning",
            reasoning,
            "",
        ]

        try:
            import json
            factors_list = json.loads(factors)
            if factors_list:
                lines.append("### Key Factors")
                for f in factors_list:
                    lines.append(f"- {f}")
                lines.append("")
        except Exception:
            pass

    path = Path("latest_analysis.md")
    path.write_text("\n".join(lines))
    return path
