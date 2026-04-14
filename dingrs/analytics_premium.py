"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ALADIN — Analytics Premium                                                 ║
║  analytics_premium.py  |  Updates #55-58 — Analiză avansată performanță   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Update #55: Benchmark comparison vs Buy & Hold QQQ
Update #56: Trade attribution analysis (ce % profit vine din fiecare semnal)
Update #57: Drawdown analysis detaliat
Update #58: Slippage & cost analysis
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime
from typing import Optional

PATH_DB = "/Users/mario/Desktop/Aladin/mario_trading.db"


# =============================================================================
# UPDATE #55 — Benchmark Comparison
# =============================================================================
def compare_vs_benchmark(
    trades_df: pd.DataFrame,
    initial_balance: float = 10000.0,
    instrument: str = "QQQ",
) -> dict:
    """
    Update #55: Compară performanța cu Buy & Hold QQQ.
    Dacă botul nu bate Buy & Hold pe 12 luni → alertă internă.
    """
    if trades_df.empty:
        return {}

    trades = trades_df[trades_df['action'] == 'TRADE'].copy()
    if trades.empty:
        return {}

    start_date = trades['date'].min() if 'date' in trades.columns else None
    end_date = trades['date'].max() if 'date' in trades.columns else None

    # Obține prețul QQQ la start/end din DB
    bnh_return = 0.0
    try:
        conn = sqlite3.connect(PATH_DB)
        if start_date and end_date:
            df_prices = pd.read_sql_query(
                "SELECT date(timestamp) as d, close FROM market_data "
                "WHERE date(timestamp) BETWEEN ? AND ? "
                "GROUP BY date(timestamp) ORDER BY d",
                conn, params=(start_date, end_date)
            )
            conn.close()
            if len(df_prices) >= 2:
                start_price = float(df_prices['close'].iloc[0])
                end_price = float(df_prices['close'].iloc[-1])
                bnh_return = (end_price - start_price) / start_price * 100
        else:
            conn.close()
    except Exception as e:
        bnh_return = 0.0

    # Performanța botului
    final_balance = float(trades['balance'].iloc[-1]) if 'balance' in trades.columns else initial_balance
    bot_return = (final_balance - initial_balance) / initial_balance * 100

    beats_bnh = bot_return > bnh_return
    alpha = round(bot_return - bnh_return, 2)  # Alpha față de benchmark

    return {
        'bot_return': round(bot_return, 2),
        'bnh_return': round(bnh_return, 2),
        'alpha': alpha,
        'beats_benchmark': beats_bnh,
        'verdict': f"✅ Bot bate Buy&Hold cu {alpha:.1f}%" if beats_bnh else f"⚠️ Bot sub Buy&Hold cu {abs(alpha):.1f}%",
        'start_date': start_date,
        'end_date': end_date,
    }


# =============================================================================
# UPDATE #56 — Trade Attribution Analysis
# =============================================================================
def trade_attribution_analysis(trades_df: pd.DataFrame) -> dict:
    """
    Update #56: Trade attribution — din profitul total, câte % vine din fiecare semnal.
    Analizează corelația între semnalele active (smt, fvg, killzone) și trade-urile câștigătoare.
    """
    if trades_df.empty:
        return {}

    trades = trades_df[trades_df['action'] == 'TRADE'].copy()
    wins = trades[trades['result'] == 'WIN']

    if trades.empty or wins.empty:
        return {}

    total_profit = wins['pnl'].sum() if 'pnl' in wins.columns else 0

    attribution = {}

    # Analiza per killzone
    if 'killzone' in trades.columns:
        kz_analysis = trades.groupby('killzone').agg(
            n_trades=('result', 'count'),
            n_wins=('result', lambda x: (x == 'WIN').sum()),
            total_pnl=('pnl', 'sum')
        ).reset_index()
        kz_analysis['win_rate'] = (kz_analysis['n_wins'] / kz_analysis['n_trades'] * 100).round(1)
        kz_analysis['pct_of_profit'] = (kz_analysis['total_pnl'] / total_profit * 100).round(1) if total_profit != 0 else 0
        attribution['by_killzone'] = kz_analysis.to_dict('records')

    # Analiza per regime
    if 'regime' in trades.columns:
        reg_analysis = trades.groupby('regime').agg(
            n_trades=('result', 'count'),
            n_wins=('result', lambda x: (x == 'WIN').sum()),
            total_pnl=('pnl', 'sum')
        ).reset_index()
        reg_analysis['win_rate'] = (reg_analysis['n_wins'] / reg_analysis['n_trades'] * 100).round(1)
        attribution['by_regime'] = reg_analysis.to_dict('records')

    # Analiza SMT vs non-SMT
    if 'smt' in trades.columns:
        for flag, label in [('smt', 'ICT_SMT'), ('fvg', 'ICT_FVG')]:
            if flag in trades.columns:
                flag_wins = trades[(trades[flag] == True) & (trades['result'] == 'WIN')]
                flag_total = trades[trades[flag] == True]
                noflag_wins = trades[(trades[flag] == False) & (trades['result'] == 'WIN')]
                noflag_total = trades[trades[flag] == False]

                wr_with = len(flag_wins) / len(flag_total) * 100 if len(flag_total) > 0 else 0
                wr_without = len(noflag_wins) / len(noflag_total) * 100 if len(noflag_total) > 0 else 0
                pnl_with = flag_wins['pnl'].sum() if not flag_wins.empty else 0

                attribution[label] = {
                    'win_rate_with': round(wr_with, 1),
                    'win_rate_without': round(wr_without, 1),
                    'pnl_contribution': round(pnl_with, 2),
                    'pct_of_profit': round(pnl_with / total_profit * 100, 1) if total_profit != 0 else 0,
                }

    attribution['total_profit'] = round(total_profit, 2)
    return attribution


# =============================================================================
# UPDATE #57 — Drawdown Analysis Detaliat
# =============================================================================
def detailed_drawdown_analysis(trades_df: pd.DataFrame, initial_balance: float = 10000.0) -> dict:
    """
    Update #57: Drawdown analysis detaliat.
    - Cât durează în medie să recuperezi
    - Frecvența drawdown-urilor >5%
    - Worst consecutive loss streak
    """
    if trades_df.empty:
        return {}

    trades = trades_df[trades_df['action'] == 'TRADE'].copy()
    if trades.empty or 'balance' not in trades.columns:
        return {}

    balances = trades['balance'].astype(float).values
    peak = np.maximum.accumulate(balances)
    dd_pct = (balances - peak) / peak * 100

    # Drawdown-uri >5%
    dd_above_5 = (dd_pct < -5.0).sum()
    dd_above_10 = (dd_pct < -10.0).sum()

    # Perioadele de drawdown
    in_dd = dd_pct < -0.5  # threshold minim 0.5%
    dd_periods = []
    dd_start = None

    for i, is_down in enumerate(in_dd):
        if is_down and dd_start is None:
            dd_start = i
        elif not is_down and dd_start is not None:
            depth = abs(float(dd_pct[dd_start:i].min()))
            duration = i - dd_start
            dd_periods.append({'start': dd_start, 'end': i, 'depth': depth, 'duration': duration})
            dd_start = None

    avg_dd_depth = round(np.mean([d['depth'] for d in dd_periods]), 2) if dd_periods else 0
    avg_dd_duration = round(np.mean([d['duration'] for d in dd_periods]), 1) if dd_periods else 0
    max_dd_depth = round(max([d['depth'] for d in dd_periods]), 2) if dd_periods else 0

    # Worst consecutive loss streak
    results = list(trades['result']) if 'result' in trades.columns else []
    max_streak = cur_streak = 0
    for r in results:
        if r == 'LOSS':
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0

    return {
        'n_drawdown_periods': len(dd_periods),
        'avg_dd_depth_pct': avg_dd_depth,
        'avg_dd_duration_bars': avg_dd_duration,
        'max_dd_depth_pct': max_dd_depth,
        'dd_above_5pct': int(dd_above_5),
        'dd_above_10pct': int(dd_above_10),
        'worst_consec_loss': max_streak,
        'dd_periods': dd_periods[:10],  # primele 10 pentru display
    }


# =============================================================================
# UPDATE #58 — Slippage & Cost Analysis
# =============================================================================
def slippage_cost_analysis(trades_df: pd.DataFrame) -> dict:
    """
    Update #58: Profit net real după: comisioane + slippage + data feed costs.
    Nu prezenta profit brut ca profit net.
    """
    if trades_df.empty:
        return {}

    trades = trades_df[trades_df['action'] == 'TRADE'].copy()
    if trades.empty:
        return {}

    n_trades = len(trades)

    # Costuri estimate (dacă nu avem transaction_cost în date)
    commission_per_trade = 0.50  # $0.50
    slippage_per_trade = 0.50  # ~$0.50 medie pe trade (0.05% dintr-un trade tipic)
    data_feed_monthly = 10.0  # $10/lună IB data feed

    total_commission = n_trades * commission_per_trade
    total_slippage = n_trades * slippage_per_trade

    # Dacă avem transaction_cost real (din Update #4)
    if 'transaction_cost' in trades.columns:
        actual_costs = trades['transaction_cost'].sum()
    else:
        actual_costs = total_commission + total_slippage

    gross_pnl = trades['pnl'].sum() if 'pnl' in trades.columns else 0

    # Data feed costs proporțional (presupunem o lună dacă avem date)
    if 'date' in trades.columns and len(trades) > 0:
        try:
            days_range = (pd.to_datetime(trades['date'].max()) - pd.to_datetime(trades['date'].min())).days
            months = max(1, days_range / 30)
            data_cost = months * data_feed_monthly
        except Exception:
            data_cost = data_feed_monthly
    else:
        data_cost = data_feed_monthly

    total_costs = actual_costs + data_cost
    net_pnl = gross_pnl - total_costs

    return {
        'gross_pnl': round(gross_pnl, 2),
        'total_commission': round(total_commission, 2),
        'total_slippage': round(total_slippage, 2),
        'data_feed_cost': round(data_cost, 2),
        'total_costs': round(total_costs, 2),
        'net_pnl': round(net_pnl, 2),
        'cost_ratio_pct': round(abs(total_costs) / abs(gross_pnl) * 100, 1) if gross_pnl != 0 else 0,
        'verdict': f"Net P&L: ${net_pnl:,.2f} (costuri: ${total_costs:,.2f} = {round(abs(total_costs)/abs(gross_pnl)*100,1) if gross_pnl != 0 else 0}%)",
    }


if __name__ == "__main__":
    print("✅ analytics_premium.py — toate funcțiile disponibile")
    print("   - compare_vs_benchmark(trades_df)")
    print("   - trade_attribution_analysis(trades_df)")
    print("   - detailed_drawdown_analysis(trades_df)")
    print("   - slippage_cost_analysis(trades_df)")
