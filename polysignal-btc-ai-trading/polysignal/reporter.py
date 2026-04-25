"""
reporter.py — Accuracy metrics for BTC 5-Min predictions.

Computes:
  - Directional accuracy overall and by confidence bin
  - Brier score (calibration quality)
  - Baseline comparisons (coin flip, always-UP, always-DOWN, crowd-following)
  - Per-model accuracy breakdown (parsed from ensemble reasoning field)
  - 95% confidence intervals on all accuracy estimates
"""
import json
import math
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from .storage import get_all_evals, get_recent_predictions, db_summary, INITIAL_BALANCE

console = Console()
REPORTS = Path("reports")

BINS = [
    ("50–55%", 0.50, 0.55),
    ("55–60%", 0.55, 0.60),
    ("60–65%", 0.60, 0.65),
    ("65%+",   0.65, 1.01),
]


def _ci95(p: float, n: int) -> float:
    """Wilson-approximation 95% confidence interval half-width."""
    if n == 0:
        return 0.0
    return 1.96 * math.sqrt(p * (1 - p) / n)


def compute_baselines(resolved_preds: list[dict]) -> dict:
    """
    Accuracy and Brier score for four dumb baselines, computed on the same
    resolved predictions the model was scored on.

    Baselines:
      coin_flip    — always predict 50/50 (theoretical reference)
      always_up    — predict UP every time, confidence = 1.0
      always_down  — predict DOWN every time, confidence = 1.0
      crowd        — predict whichever direction has Polymarket odds > 0.5,
                     using the raw odds price as the predicted probability
    """
    if not resolved_preds:
        return {}

    n = len(resolved_preds)
    up_outcomes = sum(1 for r in resolved_preds if r["actual_outcome"] == "UP")
    up_rate = up_outcomes / n

    def _brier_constant(pred_prob: float, actual_outcome_col: str) -> float:
        return sum(
            (pred_prob - (1.0 if r["actual_outcome"] == actual_outcome_col else 0.0)) ** 2
            for r in resolved_preds
        ) / n

    always_up_acc   = up_rate
    always_up_brier = _brier_constant(1.0, "UP")

    always_down_acc   = 1.0 - up_rate
    always_down_brier = _brier_constant(1.0, "DOWN")

    # Crowd: directional accuracy uses majority odds; Brier uses actual odds probability
    crowd_correct = sum(
        1 for r in resolved_preds
        if (r["market_up_odds"] > 0.5 and r["actual_outcome"] == "UP") or
           (r["market_up_odds"] <= 0.5 and r["actual_outcome"] == "DOWN")
    )
    crowd_acc   = crowd_correct / n
    crowd_brier = sum(
        (r["market_up_odds"] - (1.0 if r["actual_outcome"] == "UP" else 0.0)) ** 2
        for r in resolved_preds
    ) / n

    return {
        "n":           n,
        "up_rate":     round(up_rate, 3),
        "coin_flip":   {"accuracy": 0.500, "brier": 0.2500, "ci": 0.0},
        "always_up":   {
            "accuracy": round(always_up_acc, 3),
            "brier":    round(always_up_brier, 4),
            "ci":       round(_ci95(always_up_acc, n), 3),
        },
        "always_down": {
            "accuracy": round(always_down_acc, 3),
            "brier":    round(always_down_brier, 4),
            "ci":       round(_ci95(always_down_acc, n), 3),
        },
        "crowd": {
            "accuracy": round(crowd_acc, 3),
            "brier":    round(crowd_brier, 4),
            "ci":       round(_ci95(crowd_acc, n), 3),
        },
    }


def _record(models: dict, label: str, direction: str, confidence: float, actual: str):
    correct = (direction.upper() == actual)
    brier   = (confidence - (1.0 if correct else 0.0)) ** 2
    if label not in models:
        models[label] = {"total": 0, "correct": 0, "brier_sum": 0.0}
    models[label]["total"]     += 1
    models[label]["correct"]   += int(correct)
    models[label]["brier_sum"] += brier


def compute_per_model_accuracy(resolved_preds: list[dict]) -> dict:
    """
    Parse individual model contributions from both Ensemble and Debate predictions.

    Ensemble format — parses Votes JSON embedded in reasoning:
      Haiku (Momentum), Sonnet (Technical), Sonnet 4.6 (Contrarian)

    Debate format — parses key_factors and prediction direction:
      xAI (Grok), OpenAI (GPT-4o-mini), Claude (Judge)
    """
    models: dict = {}

    for r in resolved_preds:
        reasoning = r.get("reasoning", "")
        actual    = r.get("actual_outcome")
        if not actual:
            continue

        # ── Ensemble predictions ──────────────────────────────────────────────
        if reasoning.startswith("[ENSEMBLE"):
            m = re.search(r'Votes: (\[.*?\])', reasoning)
            if not m:
                continue
            try:
                votes = json.loads(m.group(1))
            except Exception:
                continue
            for vote in votes:
                _record(models,
                        label      = vote.get("model", "unknown"),
                        direction  = str(vote.get("direction", "")),
                        confidence = float(vote.get("confidence", 0.5)),
                        actual     = actual)

        # ── Debate predictions ────────────────────────────────────────────────
        elif reasoning.startswith("[DEBATE"):
            kf_raw = r.get("key_factors", "[]")
            try:
                kf_list = json.loads(kf_raw) if isinstance(kf_raw, str) else kf_raw
            except Exception:
                kf_list = []

            for item in kf_list:
                # "xAI final: DOWN @ 55%" or "OpenAI final: UP @ 52%"
                fm = re.match(r'(xAI|OpenAI) final: (UP|DOWN) @ (\d+)%', str(item))
                if fm:
                    label      = "xAI (Grok)" if fm.group(1) == "xAI" else "OpenAI (GPT-4o-mini)"
                    direction  = fm.group(2)
                    confidence = float(fm.group(3)) / 100.0
                    _record(models, label, direction, confidence, actual)

            # Claude judge = the final prediction itself
            _record(models,
                    label      = "Claude (Judge)",
                    direction  = r.get("direction", ""),
                    confidence = float(r.get("confidence", 0.5)),
                    actual     = actual)

    result = {}
    for label, d in models.items():
        t = d["total"]
        c = d["correct"]
        acc = c / t if t > 0 else 0.0
        result[label] = {
            "total":     t,
            "correct":   c,
            "accuracy":  round(acc, 3),
            "avg_brier": round(d["brier_sum"] / t, 4) if t > 0 else 0.0,
            "ci":        round(_ci95(acc, t), 3),
        }
    return result


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

    acc = correct / total
    return {
        "total":               total,
        "correct":             correct,
        "accuracy":            round(acc, 3),
        "accuracy_ci":         round(_ci95(acc, total), 3),
        "avg_brier":           round(brier, 4),
        "avg_price_move_pct":  round(avg_chg, 4),
        "up_calls":            up_calls,
        "down_calls":          down_calls,
        "bins":                bins,
    }


def print_report(metrics: dict, summary: dict,
                 baselines: Optional[dict] = None,
                 per_model: Optional[dict] = None):
    if not metrics:
        console.print(Panel(
            "[dim]No scored predictions yet.\n"
            "Predictions are scored automatically once their 5-min window closes.\n"
            "Run the system for a few cycles to accumulate data.[/dim]",
            title="Accuracy Report", border_style="yellow"
        ))
        return

    acc = metrics["accuracy"]
    ci  = metrics.get("accuracy_ci", 0.0)
    col = "green" if acc >= 0.55 else "yellow" if acc >= 0.50 else "red"

    # ── Overall ───────────────────────────────────────────────────────────────
    t = Table(title="BTC 5-Min Prediction Accuracy", box=box.ROUNDED)
    t.add_column("Metric",  style="cyan",       width=30)
    t.add_column("Value",   style="bold white",  width=22)
    t.add_column("Note",    style="dim",          width=32)

    t.add_row("Windows predicted",  str(metrics["total"]),  "")
    t.add_row("Correct calls",      str(metrics["correct"]), "")
    t.add_row("Directional accuracy",
              f"[{col}]{acc:.1%} ± {ci:.1%}[/{col}]",
              "95% CI  |  coin-flip = 50%")
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

    # ── Baseline comparison ───────────────────────────────────────────────────
    if baselines:
        model_acc   = metrics["accuracy"]
        model_brier = metrics["avg_brier"]
        model_ci    = metrics.get("accuracy_ci", 0.0)

        bt = Table(title="Baseline Comparison", box=box.ROUNDED)
        bt.add_column("Strategy",    style="cyan",       width=22)
        bt.add_column("Accuracy",    width=18, justify="center")
        bt.add_column("Brier Score", width=14, justify="center")
        bt.add_column("Notes",       style="dim",        width=26)

        def _acc_str(a, ci=0.0, bold=False):
            col = "green" if a >= 0.55 else "yellow" if a >= 0.50 else "red"
            s = f"{a:.1%}" + (f" ± {ci:.1%}" if ci else "")
            return f"[bold {col}]{s}[/bold {col}]" if bold else f"[{col}]{s}[/{col}]"

        bt.add_row("PolySignal (ours)",
                   _acc_str(model_acc, model_ci, bold=True),
                   f"[bold]{model_brier:.4f}[/bold]",
                   "Multi-agent ensemble")
        bt.add_row("Coin flip",
                   _acc_str(0.50),
                   "0.2500",
                   "Theoretical lower bound")
        bt.add_row("Always UP",
                   _acc_str(baselines["always_up"]["accuracy"],
                            baselines["always_up"]["ci"]),
                   f"{baselines['always_up']['brier']:.4f}",
                   f"UP rate = {baselines['up_rate']:.1%}")
        bt.add_row("Always DOWN",
                   _acc_str(baselines["always_down"]["accuracy"],
                            baselines["always_down"]["ci"]),
                   f"{baselines['always_down']['brier']:.4f}",
                   "")
        bt.add_row("Crowd (Polymarket odds)",
                   _acc_str(baselines["crowd"]["accuracy"],
                            baselines["crowd"]["ci"]),
                   f"{baselines['crowd']['brier']:.4f}",
                   "Follow majority odds")
        console.print(bt)
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

    # ── Per-model breakdown ───────────────────────────────────────────────────
    if per_model:
        mt = Table(title="Per-Model Accuracy (Ensemble Votes)", box=box.ROUNDED)
        mt.add_column("Model",    style="cyan",       width=28)
        mt.add_column("Votes",    width=7,  justify="center")
        mt.add_column("Correct",  width=9,  justify="center")
        mt.add_column("Accuracy", width=16, justify="center")
        mt.add_column("Brier",    width=10, justify="center")

        for label, d in sorted(per_model.items()):
            a   = d["accuracy"]
            ci  = d.get("ci", 0.0)
            col = "green" if a >= 0.55 else "yellow" if a >= 0.50 else "red"
            mt.add_row(
                label,
                str(d["total"]),
                str(d["correct"]),
                f"[{col}]{a:.1%} ± {ci:.1%}[/{col}]",
                f"{d['avg_brier']:.4f}",
            )
        console.print(mt)
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


def save_report(metrics: dict, summary: dict, newly_scored: list,
                baselines: Optional[dict] = None,
                per_model: Optional[dict] = None) -> Path:
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
        ci = metrics.get("accuracy_ci", 0.0)
        lines += [
            "## Overall Accuracy",
            f"| Metric | Value | Note |",
            f"|--------|-------|------|",
            f"| Directional accuracy | {metrics['accuracy']:.1%} ± {ci:.1%} | 95% CI  \\|  Coin-flip = 50% |",
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

        if baselines:
            b = baselines
            lines += [
                "",
                "## Baseline Comparison",
                "| Strategy | Accuracy | Brier Score | Notes |",
                "|----------|----------|-------------|-------|",
                f"| **PolySignal (ours)** | **{metrics['accuracy']:.1%} ± {ci:.1%}** | **{metrics['avg_brier']:.4f}** | Multi-agent ensemble |",
                f"| Coin flip | 50.0% | 0.2500 | Theoretical lower bound |",
                f"| Always UP | {b['always_up']['accuracy']:.1%} ± {b['always_up']['ci']:.1%} | {b['always_up']['brier']:.4f} | UP rate = {b['up_rate']:.1%} |",
                f"| Always DOWN | {b['always_down']['accuracy']:.1%} ± {b['always_down']['ci']:.1%} | {b['always_down']['brier']:.4f} | |",
                f"| Crowd (Polymarket odds) | {b['crowd']['accuracy']:.1%} ± {b['crowd']['ci']:.1%} | {b['crowd']['brier']:.4f} | Follow majority odds |",
            ]

        if per_model:
            lines += [
                "",
                "## Per-Model Accuracy (Ensemble Votes)",
                "| Model | Votes | Correct | Accuracy | Brier |",
                "|-------|-------|---------|----------|-------|",
            ]
            for label, d in sorted(per_model.items()):
                lines.append(
                    f"| {label} | {d['total']} | {d['correct']} | "
                    f"{d['accuracy']:.1%} ± {d.get('ci', 0):.1%} | {d['avg_brier']:.4f} |"
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
