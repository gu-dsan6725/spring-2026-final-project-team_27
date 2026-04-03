"""
run.py — PolySignal BTC 5-Min entry point.

Commands:
  python run.py          # run one cycle: collect → analyze → score → report
  python run.py --live   # continuous mode (runs every 5 minutes)
  python run.py --report # print accuracy report from DB
  python run.py --pending # show unscored predictions
"""
import asyncio
import argparse
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from loguru import logger

load_dotenv()
console = Console()


def check_env():
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key.startswith("your_"):
        console.print(Panel(
            "[red]ANTHROPIC_API_KEY not set.[/red]\n\n"
            "1. Copy [cyan].env.example[/cyan] to [cyan].env[/cyan]\n"
            "2. Add: [cyan]ANTHROPIC_API_KEY=sk-ant-...[/cyan]",
            title="Missing API Key", border_style="red"
        ))
        sys.exit(1)
    return key


async def run_once(api_key: str):
    from collector import BTCCollector
    from ensemble  import EnsembleAgent
    from evaluator import EvaluatorAgent
    from simulator import TradingSimulator, compute_sim_metrics
    from reporter  import compute_metrics, print_report, save_report, save_analysis_log, print_sim_report
    from storage   import init, save_prediction, get_all_evals, db_summary

    init()

    # ── Step 1: Score any pending predictions from previous runs ──────────────
    console.rule("[bold cyan]Step 1 — Evaluator[/bold cyan]")
    evaluator    = EvaluatorAgent()
    newly_scored = await evaluator.score_pending()

    simulator = TradingSimulator()
    if newly_scored:
        for ev in newly_scored:
            tick = "[green]✓ CORRECT[/green]" if ev.was_correct else "[red]✗ WRONG[/red]"
            console.print(
                f"  {tick}  {ev.direction} call  "
                f"actual={ev.actual_outcome}  Δ={ev.price_change_pct:+.3f}%  "
                f"Brier={ev.brier_score:.3f}"
            )
            # Settle the corresponding simulated trade
            result = simulator.settle(ev.prediction_id, ev.actual_outcome)
            if result:
                pnl_str = f"[green]+${result['pnl']:,.2f}[/green]" if result["won"] else f"[red]-${abs(result['pnl']):,.2f}[/red]"
                console.print(
                    f"  [dim]  Trade settled: {pnl_str}  "
                    f"(entry {result['odds_price']:.2%}) → "
                    f"balance ${result['balance_after']:,.2f}[/dim]"
                )
    else:
        console.print("  [dim]No pending predictions to score yet.[/dim]")

    # ── Step 2: Collect current market window ─────────────────────────────────
    console.rule("[bold cyan]Step 2 — Collector[/bold cyan]")
    collector = BTCCollector()
    await collector.start()

    window = await collector.build_window()
    await collector.close()

    if not window:
        console.print("  [yellow]No active 5-min market found right now.[/yellow]")
        console.print("  [dim]Markets run continuously — try again in a minute.[/dim]")
    else:
        console.print(f"  [green]✓[/green] Window: {window.market_question}")
        console.print(
            f"  BTC=${window.btc_price_now:,.2f}  "
            f"UP={window.up_price:.0%}  DOWN={window.down_price:.0%}  "
            f"RSI={window.rsi_14:.1f}" if window.rsi_14 else
            f"  BTC=${window.btc_price_now:,.2f}  "
            f"UP={window.up_price:.0%}  DOWN={window.down_price:.0%}"
        )
        if window.price_change_1m is not None:
            console.print(
                f"  1m Δ={window.price_change_1m:+.3f}%  "
                f"5m Δ={window.price_change_5m:+.3f}%" if window.price_change_5m else
                f"  1m Δ={window.price_change_1m:+.3f}%"
            )

        # ── Step 3: Ensemble Analyze ──────────────────────────────────────────
        console.rule("[bold cyan]Step 3 — Ensemble Analyzer[/bold cyan]")
        ensemble   = EnsembleAgent(api_key=api_key)
        prediction = await ensemble.analyze(window)

        if prediction:
            save_prediction(prediction)
            dir_color = "green" if prediction.direction == "UP" else "red"
            # Show individual votes
            for factor in prediction.key_factors:
                console.print(f"  [dim]  {factor}[/dim]")
            console.print(
                f"  [{dir_color}]▶ FINAL: {prediction.direction}[/{dir_color}]  "
                f"confidence={prediction.confidence:.0%}"
            )
            console.print(
                f"  [dim]Prediction saved. Will be scored after "
                f"{window.window_end.strftime('%H:%M UTC')}[/dim]"
            )
            # Place simulated trade
            trade = simulator.place_trade(prediction, window)
            if trade:
                console.print(
                    f"  [dim]  Trade: buy {trade.direction} @ {trade.odds_price:.2%}  "
                    f"bet=${trade.bet_amount:.0f}  "
                    f"potential=${trade.potential_payout:.2f}[/dim]"
                )
        else:
            console.print("  [red]Ensemble failed — check logs[/red]")

    # ── Step 4: Sim Report ────────────────────────────────────────────────────
    console.rule("[bold cyan]Step 4 — Simulated Account[/bold cyan]")
    sim_metrics = compute_sim_metrics()
    print_sim_report(sim_metrics)

    # ── Step 5: Accuracy Report ───────────────────────────────────────────────
    console.rule("[bold cyan]Step 5 — Accuracy Report[/bold cyan]")
    evals   = get_all_evals()
    metrics = compute_metrics(evals)
    summary = db_summary()

    print_report(metrics, summary)

    report_path   = save_report(metrics, summary, newly_scored)
    analysis_path = save_analysis_log()
    console.print(f"\n[dim]Report   → [cyan]{report_path}[/cyan][/dim]")
    console.print(f"[dim]Analysis → [cyan]{analysis_path}[/cyan][/dim]")
    console.print(f"[dim]DB       → [cyan]polysignal_btc.db[/cyan][/dim]\n")


async def run_live(api_key: str, hours: float = 1.0):
    import math
    from datetime import timezone, timedelta

    COST_PER_RUN = 0.006          # ~$0.03 / 5 runs
    total_secs   = int(hours * 3600)
    max_runs     = math.ceil(total_secs / 300)
    est_cost     = max_runs * COST_PER_RUN

    console.print(Panel(
        f"[bold green]Live mode — BTC 5-Min[/bold green]\n"
        f"Duration : [cyan]{hours:.1f}h[/cyan]  "
        f"({max_runs} windows)\n"
        f"Est. cost: [cyan]~${est_cost:.2f}[/cyan]  "
        f"(${COST_PER_RUN:.3f}/run)\n"
        f"[dim]Press Ctrl+C to stop early.[/dim]",
        border_style="green"
    ))

    session_start = datetime.now(tz=timezone.utc)
    runs          = 0
    last_window   = None   # track window slug to avoid double-runs

    while runs < max_runs:
        try:
            now = datetime.now(tz=timezone.utc)

            # Compute current 5-min window slug
            floored = (now.minute // 5) * 5
            current_window = now.replace(minute=floored, second=0, microsecond=0)
            window_slug = int(current_window.timestamp())

            if window_slug == last_window:
                # Same window — sleep until next boundary
                next_boundary = current_window + timedelta(minutes=5)
                wait = max(0, (next_boundary - now).total_seconds()) + 2
                console.print(f"  [dim]Window already processed — next in {int(wait)}s[/dim]")
                await asyncio.sleep(wait)
                continue

            last_window = window_slug
            runs += 1
            elapsed = (datetime.now(tz=timezone.utc) - session_start).total_seconds() / 60

            console.rule(
                f"[dim]Run {runs}/{max_runs}  |  "
                f"+{elapsed:.0f}m elapsed  |  "
                f"{max_runs - runs} runs left  |  "
                f"~${runs * COST_PER_RUN:.3f} spent[/dim]"
            )
            await run_once(api_key)

            if runs >= max_runs:
                break

            # Sleep until just after the next 5-min boundary
            now2          = datetime.now(tz=timezone.utc)
            next_boundary = current_window + timedelta(minutes=5)
            wait = max(0, (next_boundary - now2).total_seconds()) + 2
            console.print(f"  [dim]Next window in {int(wait)}s[/dim]")
            await asyncio.sleep(wait)

        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped early.[/yellow]")
            break

    spent = runs * COST_PER_RUN
    console.print(Panel(
        f"[bold]Session complete[/bold]\n"
        f"Runs     : {runs}\n"
        f"Spent    : ~${spent:.3f}\n"
        f"Duration : {(datetime.now(tz=timezone.utc) - session_start).total_seconds()/60:.1f}m",
        border_style="cyan"
    ))


def cmd_report():
    from storage  import init, get_all_evals, db_summary
    from reporter import compute_metrics, print_report, save_report
    init()
    evals   = get_all_evals()
    metrics = compute_metrics(evals)
    summary = db_summary()
    print_report(metrics, summary)
    path = save_report(metrics, summary, [])
    console.print(f"\n[dim]Report → [cyan]{path}[/cyan][/dim]")


def cmd_analysis():
    from storage  import init, get_recent_predictions
    import json
    init()
    rows = get_recent_predictions(limit=12)
    if not rows:
        console.print("[dim]No predictions yet.[/dim]")
        return
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
        correct    = (direction == actual) if resolved else None

        dir_color  = "green" if direction == "UP" else "red"
        result_str = ""
        if resolved:
            result_str = "  [green]✓ CORRECT[/green]" if correct else "  [red]✗ WRONG[/red]"

        console.rule(f"[dim]{i}/12  {created} UTC[/dim]")
        console.print(
            f"[bold {dir_color}]{direction}[/bold {dir_color}]  "
            f"[cyan]{confidence:.0%}[/cyan]  "
            f"BTC=${btc:,.2f}{result_str}"
        )
        console.print(f"[dim]{question}[/dim]\n")
        console.print(reasoning)
        try:
            for f in json.loads(factors):
                console.print(f"  [yellow]•[/yellow] {f}")
        except Exception:
            pass
        console.print()


def cmd_pending():
    from storage import init, get_pending_predictions
    init()
    rows = get_pending_predictions()
    if not rows:
        console.print("[dim]No pending predictions.[/dim]")
        return
    t = Table(title=f"{len(rows)} Pending Predictions", box=box.ROUNDED, show_lines=True)
    t.add_column("ID",         width=10, style="dim")
    t.add_column("Direction",  width=10)
    t.add_column("Confidence", width=12, justify="center")
    t.add_column("BTC at call", width=14, justify="right")
    t.add_column("Window end", width=14, justify="center")
    for r in rows:
        t.add_row(
            r["prediction_id"][:8],
            f"[bold]{r['direction']}[/bold]",
            f"{float(r['confidence']):.0%}",
            f"${float(r['btc_price_at_call']):,.2f}",
            r["window_end"][:16],
        )
    console.print(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PolySignal BTC 5-Min")
    parser.add_argument("--live",     action="store_true", help="Run continuously")
    parser.add_argument("--hours",    type=float, default=1.0, help="Hours to run in live mode (default: 1)")
    parser.add_argument("--report",   action="store_true", help="Print accuracy report")
    parser.add_argument("--analysis", action="store_true", help="Show last 12 full analyzer reasonings")
    parser.add_argument("--pending",  action="store_true", help="List open predictions")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    logger.add("polysignal_btc.log", level="DEBUG",
               rotation="10 MB", retention="7 days")

    console.print(Panel.fit(
        "[bold cyan]PolySignal — BTC 5-Min Forecaster[/bold cyan]\n"
        "[dim]Collector · Analyzer · Evaluator  |  No wallet. Pure accuracy.[/dim]",
        border_style="cyan"
    ))

    if args.report:
        cmd_report(); sys.exit(0)
    if args.analysis:
        cmd_analysis(); sys.exit(0)
    if args.pending:
        cmd_pending(); sys.exit(0)

    api_key = check_env()
    if args.live:
        asyncio.run(run_live(api_key, hours=args.hours))
    else:
        asyncio.run(run_once(api_key))
