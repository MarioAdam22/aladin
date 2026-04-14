"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     SIMULARE PROP FIRM — LUCID TRADING RULES v2                             ║
║     Setup: NY session + conf >= 0.60 (mario_bot.json)                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  REGULI:                                                                     ║
║  • Cont: $50,000                                                             ║
║  • Trailing DD: $2,000 (urmărește peak-ul în sus, floor max = $50,000)       ║
║  • Eval target: +$3,000 (reach $53,000)                                      ║
║  • Payout eligibilitate: 5 zile cu profit >$150 cumulate de la ultima       ║
║    retragere (sau de la funded start)                                        ║
║  • Prima retragere: trebuie +$2,000 net ȘI 5 zile eligibile                 ║
║    → scoate tot peste $51,000, rămân $51,000 în cont                        ║
║  • Retrageri ulterioare: orice peste $51,000 + 5 zile noi eligibile         ║
║  • Floor blocat la $50,000 după prima retragere                              ║
║  • BLOWN dacă balance ≤ floor                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from collections import defaultdict

INITIAL_BALANCE     = 50_000.0
TRAILING_DD         = 2_000.0
EVAL_TARGET_PROFIT  = 3_000.0    # +$3k pentru eval
FIRST_PAYOUT_PROFIT = 2_000.0    # +$2k pe funded pentru prima retragere
PAYOUT_BUFFER       = 1_000.0    # $1,000 rămân mereu în cont
WIN_DAY_MIN         = 150.0      # zi de win = profit zilnic > $150
WIN_DAYS_REQUIRED   = 5          # câte zile win trebuie cumulate

TRADES_CSV = "backtest_mario_bot_trades.csv"


def compute_daily_pnl(trades_df: pd.DataFrame) -> dict:
    """Returnează dict {date_str: pnl_usd} pentru toate zilele."""
    trades_df = trades_df.copy()
    trades_df['date'] = pd.to_datetime(trades_df['timestamp']).dt.date
    return trades_df.groupby('date')['pnl_usd'].sum().to_dict()


def run_sim(df_trades: pd.DataFrame) -> dict:
    trades = df_trades.sort_values('timestamp').reset_index(drop=True)

    # ── State ──────────────────────────────────────────────────────────────
    balance       = INITIAL_BALANCE
    peak          = INITIAL_BALANCE
    floor         = INITIAL_BALANCE - TRAILING_DD   # $48,000
    floor_locked  = False
    eval_passed   = False

    first_payout_done   = False
    win_days_count      = 0       # zile cu >$150 cumulate de la ultima retragere
    payout_eligible     = False   # True când win_days_count >= 5
    daily_pnl_in_run    = defaultdict(float)  # pnl per zi în run-ul curent

    total_payouts   = 0
    total_withdrawn = 0.0
    n_blown         = 0
    attempts        = []

    run_trades      = []
    run_start_ts    = None
    prev_date       = None

    log_lines = []

    def reset_run():
        nonlocal balance, peak, floor, floor_locked, eval_passed
        nonlocal first_payout_done, win_days_count, payout_eligible
        nonlocal daily_pnl_in_run, run_trades, run_start_ts, prev_date
        balance             = INITIAL_BALANCE
        peak                = INITIAL_BALANCE
        floor               = INITIAL_BALANCE - TRAILING_DD
        floor_locked        = False
        eval_passed         = False
        first_payout_done   = False
        win_days_count      = 0
        payout_eligible     = False
        daily_pnl_in_run    = defaultdict(float)
        run_trades          = []
        run_start_ts        = None
        prev_date           = None

    reset_run()

    for _, trade in trades.iterrows():
        pnl    = float(trade['pnl_usd'])
        ts     = pd.to_datetime(trade['timestamp'])
        date   = ts.date()

        if run_start_ts is None:
            run_start_ts = ts

        # ── Execută trade ───────────────────────────────────────────────────
        balance += pnl
        run_trades.append(trade)
        daily_pnl_in_run[date] += pnl

        # ── Detectează zi nouă → actualizează win_days_count ────────────────
        if eval_passed and prev_date is not None and date != prev_date:
            day_pnl = daily_pnl_in_run.get(prev_date, 0.0)
            if day_pnl > WIN_DAY_MIN:
                win_days_count += 1
                if win_days_count >= WIN_DAYS_REQUIRED and not payout_eligible:
                    payout_eligible = True
                    log_lines.append(
                        f"  📅 ELIGIBIL RETRAGERE  win_days={win_days_count}  "
                        f"balance=${balance:,.0f}  [{date}]"
                    )
        prev_date = date

        # ── Update peak + floor ─────────────────────────────────────────────
        if balance > peak:
            peak = balance
        if not floor_locked:
            floor = min(peak - TRAILING_DD, INITIAL_BALANCE)
            if floor >= INITIAL_BALANCE:
                floor = INITIAL_BALANCE
                floor_locked = True

        # ── CHECK EVAL ──────────────────────────────────────────────────────
        if not eval_passed:
            if balance >= INITIAL_BALANCE + EVAL_TARGET_PROFIT:
                eval_passed = True
                log_lines.append(
                    f"  ✅ EVAL PASSED  balance=${balance:,.0f}  "
                    f"floor=${floor:,.0f}  [{ts.date()}]"
                )

        # ── CHECK PAYOUT (funded + eligibil) ────────────────────────────────
        if eval_passed and payout_eligible:
            if not first_payout_done:
                if balance >= INITIAL_BALANCE + FIRST_PAYOUT_PROFIT:
                    payout = balance - (INITIAL_BALANCE + PAYOUT_BUFFER)
                    if payout > 0:
                        balance         -= payout
                        total_withdrawn += payout
                        total_payouts   += 1
                        first_payout_done = True
                        floor           = INITIAL_BALANCE
                        floor_locked    = True
                        peak            = max(peak, balance)
                        # Reset win days counter
                        win_days_count  = 0
                        payout_eligible = False
                        log_lines.append(
                            f"  💸 PAYOUT #{total_payouts:<3} +${payout:>7,.0f}  "
                            f"balance=${balance:,.0f}  floor=${floor:,.0f}  "
                            f"[{ts.date()}]"
                        )
            else:
                if balance > INITIAL_BALANCE + PAYOUT_BUFFER:
                    payout = balance - (INITIAL_BALANCE + PAYOUT_BUFFER)
                    balance         -= payout
                    total_withdrawn += payout
                    total_payouts   += 1
                    floor           = INITIAL_BALANCE
                    peak            = max(peak, balance)
                    win_days_count  = 0
                    payout_eligible = False
                    log_lines.append(
                        f"  💸 PAYOUT #{total_payouts:<3} +${payout:>7,.0f}  "
                        f"balance=${balance:,.0f}  floor=${floor:,.0f}  "
                        f"[{ts.date()}]"
                    )

        # ── CHECK BLOWN ──────────────────────────────────────────────────────
        if balance <= floor:
            n_blown += 1
            phase = "FUNDED" if eval_passed else "EVAL"
            log_lines.append(
                f"  💥 BLOWN #{n_blown}  phase={phase}  "
                f"balance=${balance:,.0f}  floor=${floor:,.0f}  "
                f"payouts_in_run={sum(1 for l in log_lines if 'PAYOUT' in l and f'BLOWN #{n_blown}' not in l) - (0 if n_blown==1 else sum(1 for a in attempts for _ in range(a.get('payouts',0))))}"
                f"  [{ts.date()}]"
            )
            attempts.append(dict(
                run=n_blown, eval_passed=eval_passed,
                trades=len(run_trades), blown_bal=balance,
                blown_floor=floor, ts_blown=ts.date(),
            ))
            reset_run()

    log_lines.append(
        f"\n  🏁 FIN TRADES  balance=${balance:,.0f}  "
        f"eval={'PASS' if eval_passed else 'FAIL'}  "
        f"win_days={win_days_count}  floor=${floor:,.0f}"
    )

    return dict(
        n_blown=n_blown, total_payouts=total_payouts,
        total_withdrawn=total_withdrawn, attempts=attempts,
        log=log_lines, final_balance=balance,
        eval_passed_final=eval_passed,
    )


def main():
    df = pd.read_csv(TRADES_CSV)
    best = df[(df['session'] == 'NY') & (df['confidence'] >= 0.60)].copy()
    best['pnl_usd'] = best['pnl_usd'].astype(float)
    best = best.sort_values('timestamp').reset_index(drop=True)

    print("═" * 70)
    print("  PROP FIRM SIM v2 — LUCID TRADING  |  mario_bot NY conf≥0.60")
    print("═" * 70)
    print(f"  Cont:              ${INITIAL_BALANCE:,.0f}")
    print(f"  Trailing DD:       ${TRAILING_DD:,.0f}")
    print(f"  Eval target:       +${EVAL_TARGET_PROFIT:,.0f}")
    print(f"  Payout eligib.:    {WIN_DAYS_REQUIRED} zile cu >${WIN_DAY_MIN:.0f}/zi (cumulate)")
    print(f"  Prima retragere:   +${FIRST_PAYOUT_PROFIT:,.0f} profit + eligibil")
    print(f"  Buffer permanent:  ${PAYOUT_BUFFER:,.0f}")
    print(f"  Trades input:      {len(best):,}  ({best['timestamp'].iloc[0][:10]} → {best['timestamp'].iloc[-1][:10]})")
    print("─" * 70)

    res = run_sim(best)

    print("\n  JURNAL EVENIMENTE:")
    for line in res['log']:
        print(line)

    attempts = res['attempts']
    if attempts:
        print("\n  BLOWN ATTEMPTS:")
        print(f"  {'#':>3}  {'Faza':<10}  {'Balance':<12}  {'Floor':<12}  {'Trades':>7}  Data")
        print("  " + "-" * 62)
        for a in attempts:
            faza = "FUNDED" if a['eval_passed'] else "EVAL"
            print(f"  {a['run']:>3}  {faza:<10}  ${a['blown_bal']:>10,.0f}  "
                  f"${a['blown_floor']:>10,.0f}  {a['trades']:>7}  {a['ts_blown']}")

    n_total   = len(attempts) + 1
    n_pass    = sum(1 for a in attempts if a['eval_passed']) + (1 if res['eval_passed_final'] else 0)
    n_fail    = n_total - n_pass
    blown_eval   = sum(1 for a in attempts if not a['eval_passed'])
    blown_funded = sum(1 for a in attempts if a['eval_passed'])

    print("\n" + "═" * 70)
    print("  SUMAR FINAL")
    print("═" * 70)
    print(f"  Total runs:           {n_total}")
    print(f"  Eval PASS:            {n_pass} ({n_pass/n_total*100:.0f}%)")
    print(f"  Blown în EVAL:        {blown_eval}")
    print(f"  Blown pe FUNDED:      {blown_funded}")
    print(f"  Total retrageri:      {res['total_payouts']}")
    print(f"  Total scos:           ${res['total_withdrawn']:,.0f}")
    print(f"  Balance final:        ${res['final_balance']:,.0f}")

    if res['total_payouts'] > 0:
        print(f"  Media/retragere:      ${res['total_withdrawn']/res['total_payouts']:,.0f}")

    print()
    # Business case per eval cost
    print("  BUSINESS CASE (cost eval per cont):")
    for eval_cost in [150, 250, 350, 500]:
        total_cost = blown_eval * eval_cost + (blown_funded + 1) * eval_cost
        # blown_eval = n_evals fără payout, blown_funded = n_evals cu payout
        # cost total = toate run-urile × cost eval
        total_cost_all = n_total * eval_cost
        net = res['total_withdrawn'] - total_cost_all
        print(f"    Eval ${eval_cost:>3}/cont:  "
              f"cost total ${total_cost_all:>6,}  |  "
              f"net profit ${net:>8,}  |  "
              f"ROI {net/total_cost_all*100:>+.0f}%")
    print("═" * 70)


if __name__ == "__main__":
    main()
