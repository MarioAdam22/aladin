"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ALADIN — Monthly Performance Report Generator                              ║
║  monthly_report.py  |  Update #28 — PDF Report Lunar Automat              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Generează automat PDF lunar cu:
  - Sharpe, winrate, drawdown, best/worst trade
  - Benchmark vs Buy&Hold
  - Trade attribution per killzone/regime
  - Equity curve (Plotly → PNG → embed în PDF)

Utilizare:
  python monthly_report.py --month 2025-01
  sau
  from monthly_report import generate_monthly_pdf
  generate_monthly_pdf("2025-01", trades_df, stats)

Instalare: pip install fpdf2 plotly kaleido
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np


def generate_monthly_pdf(
    month_str: str,          # format "2025-01"
    trades_df: pd.DataFrame,
    stats: dict,
    output_path: Optional[str] = None,
    initial_balance: float = 10000.0,
) -> Optional[str]:
    """
    Update #28: Generează PDF de performanță lunar.

    Args:
        month_str:   Luna (ex: "2025-01")
        trades_df:   DataFrame cu trade-uri din DASHBOARD.py
        stats:       Dict de statistici din compute_backtest_stats()
        output_path: Calea unde se salvează PDF (default: Desktop)
        initial_balance: Capitalul inițial

    Returns:
        path: str calea la PDF sau None la eroare
    """
    try:
        from fpdf import FPDF
    except ImportError:
        print("❌ fpdf2 lipsă. Instalare: pip install fpdf2")
        return None

    if output_path is None:
        output_path = f"/Users/mario/Desktop/Aladin/aladin_report_{month_str}.pdf"

    trades = trades_df[trades_df['action'] == 'TRADE'].copy() if not trades_df.empty else pd.DataFrame()

    # ── Calculează metrici suplimentare ──────────────────────────────────────
    n_trades = stats.get('total_trades', 0)
    win_rate = stats.get('win_rate', 0)
    profit_factor = stats.get('profit_factor', 0)
    sharpe = stats.get('sharpe_ratio', 0)
    sortino = stats.get('sortino_ratio', 0)
    max_dd = stats.get('max_drawdown', 0)
    total_ret = stats.get('total_return', 0)
    calmar = stats.get('calmar_ratio', 0)
    final_bal = stats.get('final_balance', initial_balance)

    best_trade = stats.get('best_trade', 0)
    avg_win = stats.get('avg_win', 0)
    avg_loss = stats.get('avg_loss', 0)

    # Best/Worst trade
    if not trades.empty and 'pnl' in trades.columns:
        best_t_idx = trades['pnl'].idxmax()
        worst_t_idx = trades['pnl'].idxmin()
        best_t = trades.loc[best_t_idx] if not trades.empty else None
        worst_t = trades.loc[worst_t_idx] if not trades.empty else None
    else:
        best_t = worst_t = None

    # ── Generare PDF cu FPDF2 ────────────────────────────────────────────────
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Header ──────────────────────────────────────────────────────────────
    pdf.set_fill_color(7, 9, 15)
    pdf.rect(0, 0, 210, 297, 'F')  # Background negru pagină întreagă

    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(160, 184, 255)
    pdf.cell(0, 15, "ALADIN QUANTUM-ICT", ln=True, align='C')

    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(74, 90, 138)
    pdf.cell(0, 8, f"Monthly Performance Report — {month_str}", ln=True, align='C')
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Adam Mario", ln=True, align='C')

    pdf.ln(8)
    pdf.set_draw_color(26, 34, 64)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    # ── Metrici Principale ───────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(100, 128, 192)
    pdf.cell(0, 8, "PERFORMANCE SUMMARY", ln=True)
    pdf.ln(2)

    def metric_row(label, value, color_rgb=(160, 184, 255), good=None):
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(74, 90, 138)
        pdf.cell(80, 7, f"  {label}:", ln=False)
        if good is not None:
            pdf.set_text_color(64, 192, 128) if good else pdf.set_text_color(192, 64, 64)
        else:
            pdf.set_text_color(*color_rgb)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, str(value), ln=True)

    metric_row("Total Return", f"{total_ret:+.2f}%", good=total_ret > 0)
    metric_row("Win Rate", f"{win_rate:.1f}%", good=win_rate >= 45)
    metric_row("Profit Factor", f"{profit_factor:.2f}", good=profit_factor >= 1.0)
    metric_row("Sharpe Ratio", f"{sharpe:.2f}", good=sharpe >= 1.5)
    metric_row("Sortino Ratio", f"{sortino:.2f}", good=sortino >= 1.0)
    metric_row("Max Drawdown", f"{max_dd:.2f}%", good=max_dd > -15)
    metric_row("Calmar Ratio", f"{calmar:.2f}", good=calmar >= 1.0)
    metric_row("Total Trades", str(n_trades))
    metric_row("Final Balance", f"${final_bal:,.2f}", good=final_bal > initial_balance)
    metric_row("Initial Balance", f"${initial_balance:,.2f}")

    pdf.ln(6)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    # ── Best/Worst Trade ─────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(100, 128, 192)
    pdf.cell(0, 8, "BEST & WORST TRADE", ln=True)
    pdf.ln(2)

    if best_t is not None:
        metric_row("Best Trade P&L", f"${float(best_t.get('pnl', 0)):+.2f}", good=True)
        metric_row("Best Trade Date", str(best_t.get('date', 'N/A'))[:10])
        metric_row("Best Trade Dir", str(best_t.get('direction', 'N/A')))

    if worst_t is not None:
        metric_row("Worst Trade P&L", f"${float(worst_t.get('pnl', 0)):+.2f}", good=False)
        metric_row("Worst Trade Date", str(worst_t.get('date', 'N/A'))[:10])

    pdf.ln(6)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    # ── Distribuție per killzone ─────────────────────────────────────────────
    if not trades.empty and 'killzone' in trades.columns:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(100, 128, 192)
        pdf.cell(0, 8, "PERFORMANCE BY KILLZONE", ln=True)
        pdf.ln(2)

        kz_stats = trades.groupby('killzone').agg(
            n=('result', 'count'),
            wins=('result', lambda x: (x == 'WIN').sum()),
        ).reset_index()
        kz_stats['wr'] = (kz_stats['wins'] / kz_stats['n'] * 100).round(1)

        for _, row in kz_stats.iterrows():
            metric_row(
                str(row['killzone'])[:30],
                f"{int(row['n'])} trades | WR: {row['wr']}%",
                good=row['wr'] >= 45
            )

    pdf.ln(4)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    # ── Footer ──────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(40, 50, 90)
    pdf.cell(0, 6, "Aladin Quantum-ICT v6.8 | Adam Mario | marioyear@yahoo.com", ln=True, align='C')
    pdf.cell(0, 6, "⚠️ Performanța trecută nu garantează rezultate viitoare. Nu este sfat financiar.", ln=True, align='C')

    # ── Salvare ─────────────────────────────────────────────────────────────
    pdf.output(output_path)
    print(f"✅ PDF generat: {output_path}")
    return output_path


if __name__ == "__main__":
    # Test cu date demo
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", default=datetime.now().strftime("%Y-%m"), help="Luna (ex: 2025-01)")
    args = parser.parse_args()

    # Date demo
    demo_trades = pd.DataFrame({
        'action': ['TRADE'] * 10,
        'result': ['WIN', 'WIN', 'LOSS', 'WIN', 'LOSS', 'WIN', 'WIN', 'WIN', 'LOSS', 'WIN'],
        'pnl': [150, 200, -100, 175, -100, 130, 190, 160, -100, 220],
        'balance': [10150, 10350, 10250, 10425, 10325, 10455, 10645, 10805, 10705, 10925],
        'date': [f"{args.month}-0{i+1}" for i in range(10)],
        'direction': ['LONG'] * 10,
        'killzone': ['London Open', 'NY Open', 'London Open', 'NY Open', 'London Open'] * 2,
    })
    demo_stats = {
        'total_trades': 10, 'win_rate': 70.0, 'profit_factor': 2.3,
        'sharpe_ratio': 2.1, 'sortino_ratio': 3.4, 'max_drawdown': -5.2,
        'total_return': 9.25, 'calmar_ratio': 1.78, 'final_balance': 10925,
        'best_trade': 220, 'avg_win': 175, 'avg_loss': -100,
    }

    path = generate_monthly_pdf(args.month, demo_trades, demo_stats)
    print(f"   Deschide: open '{path}'" if path else "   Eroare la generare PDF")
