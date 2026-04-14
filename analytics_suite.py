"""
ALADIN v14 — Analytics Suite
═══════════════════════════════════════════════════════════════════════════════
Suite completă de evaluare statistică — folosită de toate cele 3 modele:
  • train_breakout_model.py
  • train_mario_ai.py
  • train_reversal_model.py

Include:
  1. Correlation Matrix          — identifică features corelate/redundante
  2. Threshold Analysis          — precision/recall/EV la fiecare threshold
  3. Calibration Analysis        — sunt probabilitățile modelului bine calibrate?
  4. Equity Curve Analysis       — simulare P&L cu RM din mario_rag.py
  5. Monte Carlo Simulation      — distribuție outcomes pe N simulări bootstrap
  6. Permutation Significance    — e AUC semnificativ statistic vs random?
  7. Permutation Feature Import. — importanță reală vs native XGBoost
  8. Walk-Forward Validation     — stabilitate OOS pe 5 ferestre temporale
  9. run_full_analysis()         — masterfunction care apelează toate

RM folosit în simulări (exact mario_rag.py):
  0.5R  → move SL to BE
  0.85R → lock +0.5R
  1R    → exit 50% position
  2R    → exit 25% position
  trail remaining 25% (avg target ~3R)
  → avg win = 0.50×1R + 0.25×2R + 0.25×3R = 1.625R
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss

# ── CONTRACT CONFIG ───────────────────────────────────────────────────────────
MICRO_NQ_MULT  = 5.0   # Micro NQ futures: $5 per point
FULL_NQ_MULT   = 20.0  # Full NQ futures: $20 per point
DEFAULT_MULT   = FULL_NQ_MULT   # Mini NQ ($20/pt) — 1 contract

# RM trailing stop — 1 contract Mini NQ (fără partial exits)
# Logică: SL→BE la +0.5R, trail activ la +0.85R, NQ face 30-40pts real
# Avg capturat cu trail 10-12pts: ~22-25pts = 1.8-2.1R → folosim 2.0R
WIN_R_AVG      = 2.0    # avg winner cu trailing stop pe 1 contract (NQ 30-40pt move)
LOSS_R_AVG     = 1.0    # full stop loss = 1R (dacă BE nu e atins)

TARGET_DAILY   = 1200.0          # $1200/zi target
TARGET_ANNUAL  = TARGET_DAILY * 252


# ══════════════════════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════════════════════
def _section(title: str, w: int = 64):
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")


def _bar(val: float, scale: float = 200, max_w: int = 30) -> str:
    return '█' * min(int(abs(val) * scale), max_w)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
def analyze_feature_correlations(X: pd.DataFrame, features: list,
                                  threshold: float = 0.80,
                                  save_csv: str = None) -> pd.DataFrame:
    """
    Calculează matricea de corelație și identifică perechile cu |r| ≥ threshold.
    Perechile cu r > 0.95 sunt candidate pentru eliminare (redundante).
    """
    _section("FEATURE CORRELATION MATRIX")

    _X = X[features].fillna(0)
    corr = _X.corr()

    pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                pairs.append((features[i], features[j], round(float(r), 4)))

    pairs.sort(key=lambda x: -abs(x[2]))

    print(f"\n   Features: {len(features)} | Perechi cu |r|≥{threshold}: {len(pairs)}")

    if pairs:
        print(f"\n   {'Feature A':<32} {'Feature B':<32} {'r':>7}")
        print(f"   {'-'*74}")
        for a, b, r in pairs[:25]:
            flag = " ⚠️  REDUNDANT" if abs(r) > 0.95 else ""
            print(f"   {a:<32} {b:<32} {r:>7.4f}{flag}")
        if len(pairs) > 25:
            print(f"   ... +{len(pairs)-25} alte perechi")
    else:
        print(f"   ✅ Nicio pereche cu |r|≥{threshold} — features independente")

    redundant = [a for a, b, r in pairs if abs(r) > 0.95]
    if redundant:
        print(f"\n   ⚠️  Candidați eliminare (r>0.95): {list(set(redundant))}")

    if save_csv:
        corr.to_csv(save_csv)
        print(f"   💾 Corelații salvate: {save_csv}")

    return corr


# ══════════════════════════════════════════════════════════════════════════════
# 2. THRESHOLD ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def threshold_analysis(y_true: np.ndarray, y_proba: np.ndarray,
                        thresholds: list = None,
                        risk_pts: float = 10.0,
                        label: str = "") -> dict:
    """
    Precision / Recall / EV (Expected Value în R) la fiecare threshold.
    EV = precision × WIN_R - (1-precision) × LOSS_R
    """
    _section(f"THRESHOLD ANALYSIS  {label}")

    if thresholds is None:
        thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

    base  = float(y_true.mean())
    n     = len(y_true)

    print(f"\n   Base rate REAL: {base:.1%}  |  Test samples: {n:,}")
    print(f"   Win R (partial exits): +{WIN_R_AVG}R  |  Loss R: -{LOSS_R_AVG}R\n")

    hdr = f"   {'Thr':>5} {'N sig':>7} {'Prec':>7} {'Recall':>8} {'Lift':>6} {'EV/R':>7} {'EV pts':>8} {'Sig/yr':>8}"
    print(hdr)
    print(f"   {'-'*66}")

    best = {'thr': None, 'ev': -999}
    results = {}

    for thr in thresholds:
        mask  = y_proba >= thr
        n_sig = int(mask.sum())
        if n_sig < 3:
            continue

        prec   = float(y_true[mask].mean())
        recall = float(y_true[mask].sum() / max(y_true.sum(), 1))
        lift   = prec / max(base, 1e-8)
        ev_r   = prec * WIN_R_AVG - (1 - prec) * LOSS_R_AVG
        ev_pts = ev_r * risk_pts
        sig_yr = n_sig / max(n / (252 * 390), 1) * 252  # approx signals/year

        tag = " ← BEST EV" if ev_r > best['ev'] and n_sig >= 5 else ""
        if ev_r > best['ev'] and n_sig >= 5:
            best = {'thr': thr, 'ev': ev_r, 'prec': prec, 'n': n_sig}

        print(f"   {thr:>5.2f} {n_sig:>7} {prec:>7.1%} {recall:>8.1%} "
              f"{lift:>6.2f}× {ev_r:>7.3f} {ev_pts:>8.1f} {sig_yr:>8.0f}{tag}")
        results[thr] = {'prec': prec, 'recall': recall, 'ev_r': ev_r, 'n_sig': n_sig}

    if best['thr']:
        print(f"\n   🎯 Threshold optim: {best['thr']:.2f} "
              f"(Prec={best['prec']:.1%}, EV={best['ev']:.3f}R = {best['ev']*risk_pts:.1f}pts/trade)")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. CALIBRATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def calibration_analysis(y_true: np.ndarray, y_proba: np.ndarray,
                          n_bins: int = 10, label: str = ""):
    """
    Verifică dacă probabilitățile modelului sunt bine calibrate.
    Model calibrat: când zice P=0.60 → ~60% din cazuri sunt pozitive.
    Brier Score: 0=perfect, 0.25=uninformativ (random).
    """
    _section(f"CALIBRATION ANALYSIS  {label}")

    brier = brier_score_loss(y_true, y_proba)
    brier_skill = 1.0 - brier / 0.25

    print(f"\n   Brier Score:  {brier:.4f}  (0=perfect, 0.25=random)")
    print(f"   Brier Skill:  {brier_skill:.1%}  față de baseline uninformativ")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    print(f"\n   {'Interval':>14} {'Pred center':>12} {'Actual rate':>12} "
          f"{'N':>8} {'Δ cal':>8}")
    print(f"   {'-'*60}")

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_proba >= lo) & (y_proba < hi)
        n_b = mask.sum()
        if n_b < 3:
            continue
        actual = float(y_true[mask].mean())
        center = (lo + hi) / 2.0
        delta  = actual - center
        flag   = ("⬆️ overconf" if delta < -0.10 else
                  ("⬇️ underconf" if delta > 0.10 else "✅ ok"))
        print(f"   [{lo:.1f}-{hi:.1f}]        {center:>12.2f} {actual:>12.1%} "
              f"{n_b:>8,} {delta:>+8.2f}  {flag}")

    verdict = "✅ Model bine calibrat" if brier_skill > 0.05 else "⚠️  Calibrare slabă"
    print(f"\n   {verdict}  (skill={brier_skill:.1%})")


# ══════════════════════════════════════════════════════════════════════════════
# 4. EQUITY CURVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def equity_curve_analysis(y_true: np.ndarray, y_proba: np.ndarray,
                           threshold: float = 0.55,
                           risk_pts: float = 10.0,
                           mult: float = DEFAULT_MULT,
                           label: str = ""):
    """
    Simulează curba de equity din predicțiile modelului.
    Folosește RM-ul din mario_rag.py (partial exits).
    """
    _section(f"EQUITY CURVE ANALYSIS @ {threshold:.0%}  {label}")

    mask  = y_proba >= threshold
    y_sel = y_true[mask]
    n_sel = len(y_sel)

    if n_sel < 3:
        print(f"   ⚠️  Prea puține semnale ({n_sel}) la {threshold:.0%}")
        return

    pnl = np.where(y_sel == 1,
                   risk_pts * WIN_R_AVG * mult,
                   -risk_pts * LOSS_R_AVG * mult)

    equity = np.cumsum(pnl)
    wins   = int((y_sel == 1).sum())
    losses = int((y_sel == 0).sum())
    wr     = wins / max(n_sel, 1)
    total  = float(equity[-1]) if len(equity) else 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(equity)
    dd     = running_max - equity
    max_dd = float(dd.max()) if len(dd) else 0.0

    # Profit factor
    gp = float(pnl[pnl > 0].sum())
    gl = float(abs(pnl[pnl < 0].sum()))
    pf = gp / max(gl, 1.0)

    # Sharpe
    sharpe = (pnl.mean() / max(pnl.std(), 1e-8)) * np.sqrt(252)

    # Max consecutive losses
    max_cl = cl = 0
    for p in pnl:
        if p < 0:
            cl += 1; max_cl = max(max_cl, cl)
        else:
            cl = 0

    print(f"\n   Trades: {n_sel}  |  Win Rate: {wr:.1%}  ({wins}W / {losses}L)")
    print(f"   Total P&L:    ${total:>10,.0f}")
    print(f"   Max Drawdown: ${max_dd:>10,.0f}")
    print(f"   Profit Factor: {pf:.2f}  |  Sharpe: {sharpe:.2f}")
    print(f"   Max Consec Losses: {max_cl}")
    print(f"   Avg Win: ${risk_pts*WIN_R_AVG*mult:.0f}  |  Avg Loss: ${risk_pts*LOSS_R_AVG*mult:.0f}")

    # ASCII equity curve
    if len(equity) >= 5:
        steps  = min(60, len(equity))
        idx    = np.linspace(0, len(equity) - 1, steps, dtype=int)
        eq_s   = equity[idx]
        eq_min = eq_s.min(); eq_max = eq_s.max()
        rng    = max(eq_max - eq_min, 1.0)
        H      = 7
        print(f"\n   Equity curve  [${eq_min:,.0f} → ${eq_max:,.0f}]:")
        for row in range(H, -1, -1):
            thr_v = eq_min + rng * row / H
            line  = "   |"
            for v in eq_s:
                line += "█" if v >= thr_v else " "
            suffix = f"  ${eq_max:,.0f}" if row == H else (f"  ${eq_min:,.0f}" if row == 0 else "")
            print(line + suffix)
        print(f"   {'─'*(steps+3)}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
def monte_carlo_simulation(y_true: np.ndarray, y_proba: np.ndarray,
                            threshold: float = 0.55,
                            risk_pts: float = 10.0,
                            mult: float = DEFAULT_MULT,
                            n_sims: int = 1000,
                            trades_per_year: int = 200,
                            label: str = "") -> dict:
    """
    Monte Carlo bootstrap: resample trades de N ori, calculează distribuția
    de annual return, max drawdown, Sharpe ratio.

    Parametri RM (mario_rag.py):
      Win  → avg +1.625R  (50%@1R + 25%@2R + 25%@3R)
      Loss → -1.0R
    """
    _section(f"MONTE CARLO SIMULATION  {label}  ({n_sims:,} simulări)")

    mask  = y_proba >= threshold
    y_sel = y_true[mask]
    n_sel = len(y_sel)

    if n_sel < 5:
        print(f"   ⚠️  Prea puține semnale ({n_sel}) la {threshold:.0%}")
        return {}

    wr   = float((y_sel == 1).mean())
    outcomes = np.where(y_sel == 1,
                        risk_pts * WIN_R_AVG * mult,
                        -risk_pts * LOSS_R_AVG * mult)

    print(f"\n   Threshold: {threshold:.0%}  |  Semnale: {n_sel}  |  WR baza: {wr:.1%}")
    print(f"   Win: +${risk_pts*WIN_R_AVG*mult:.0f}  |  Loss: -${risk_pts*LOSS_R_AVG*mult:.0f}")
    print(f"   Trades/an simulat: {trades_per_year}  |  Simulări: {n_sims:,}\n")

    rng = np.random.RandomState(42)
    ann_ret, max_dds, sharpes = [], [], []

    for _ in range(n_sims):
        samp   = rng.choice(outcomes, size=min(trades_per_year, n_sel), replace=True)
        equity = np.cumsum(samp)
        rm     = np.maximum.accumulate(equity)
        dd     = float((rm - equity).max())
        sh     = float((samp.mean() / max(samp.std(), 1e-8)) * np.sqrt(252))
        ann_ret.append(float(equity[-1]))
        max_dds.append(dd)
        sharpes.append(sh)

    ann_ret = np.array(ann_ret)
    max_dds = np.array(max_dds)
    sharpes = np.array(sharpes)

    pcts = [5, 25, 50, 75, 95]
    print(f"   {'Metric':<24} {'P5':>10} {'P25':>10} {'P50':>10} {'P75':>10} {'P95':>10}")
    print(f"   {'-'*68}")
    print(f"   {'Annual Return ($)':<24} " +
          " ".join(f"{np.percentile(ann_ret, p):>10,.0f}" for p in pcts))
    print(f"   {'Max Drawdown ($)':<24} " +
          " ".join(f"{np.percentile(max_dds, p):>10,.0f}" for p in pcts))
    print(f"   {'Sharpe Ratio':<24} " +
          " ".join(f"{np.percentile(sharpes, p):>10.2f}" for p in pcts))

    prob_profit  = float((ann_ret > 0).mean() * 100)
    prob_target  = float((ann_ret >= TARGET_ANNUAL).mean() * 100)
    prob_sharpe1 = float((sharpes > 1.0).mean() * 100)
    p50_ret      = float(np.percentile(ann_ret, 50))
    p50_dd       = float(np.percentile(max_dds, 50))

    print(f"\n   Prob. an profitabil:       {prob_profit:.1f}%")
    print(f"   Prob. Sharpe > 1.0:        {prob_sharpe1:.1f}%")
    print(f"   🎯 Target ${TARGET_ANNUAL:,.0f}/an:")
    print(f"      P50 return: ${p50_ret:,.0f} ({p50_ret/TARGET_ANNUAL*100:.0f}% din target)")
    print(f"      Prob. ≥ target: {prob_target:.1f}%")

    # Risk of ruin (max DD > 50% of hypothetical $50K account)
    ruin_thresh  = 25_000
    prob_ruin    = float((max_dds > ruin_thresh).mean() * 100)
    print(f"   ⚠️  Prob. DD > ${ruin_thresh:,}: {prob_ruin:.1f}%")

    return {
        'threshold': threshold, 'n_signals': n_sel, 'base_wr': wr,
        'p50_annual': p50_ret, 'p5_annual': float(np.percentile(ann_ret, 5)),
        'p95_annual': float(np.percentile(ann_ret, 95)),
        'p50_max_dd': p50_dd, 'p50_sharpe': float(np.percentile(sharpes, 50)),
        'prob_profitable': prob_profit, 'prob_above_target': prob_target,
        'prob_sharpe_gt1': prob_sharpe1,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. PERMUTATION SIGNIFICANCE TEST
# ══════════════════════════════════════════════════════════════════════════════
def permutation_significance_test(model, X_test: np.ndarray, y_test: np.ndarray,
                                   n_permutations: int = 200,
                                   label: str = "") -> float:
    """
    Testează dacă AUC-ul modelului e semnificativ statistic față de shuffling aleator.
    p-value = fracția din AUC-urile permutate care depășesc AUC-ul real.
    p < 0.01 → model semnificativ statistic.
    """
    _section(f"PERMUTATION SIGNIFICANCE TEST  {label}  ({n_permutations} permutări)")

    y_proba   = model.predict_proba(X_test)[:, 1]
    real_auc  = roc_auc_score(y_test, y_proba)

    rng = np.random.RandomState(42)
    perm_aucs = []
    for _ in range(n_permutations):
        y_shuf = rng.permutation(y_test)
        try:
            perm_aucs.append(roc_auc_score(y_shuf, y_proba))
        except Exception:
            perm_aucs.append(0.5)

    perm_aucs = np.array(perm_aucs)
    p_value   = float((perm_aucs >= real_auc).mean())

    print(f"\n   AUC real:            {real_auc:.4f}")
    print(f"   AUC permutări (avg): {perm_aucs.mean():.4f} ± {perm_aucs.std():.4f}")
    print(f"   p-value:             {p_value:.4f}")

    if p_value < 0.01:
        print(f"   ✅ SEMNIFICATIV STATISTIC (p<0.01) — modelul învaţă pattern-uri reale")
    elif p_value < 0.05:
        print(f"   ✅ Semnificativ marginal (p<0.05)")
    else:
        print(f"   ❌ NU semnificativ (p={p_value:.3f}) — posibil overfitting / noise")

    return p_value


# ══════════════════════════════════════════════════════════════════════════════
# 7. PERMUTATION FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
def permutation_feature_importance(model, X_test: np.ndarray, y_test: np.ndarray,
                                    features: list, n_repeats: int = 10,
                                    label: str = "") -> pd.DataFrame:
    """
    Importanță prin permutare: cu cât scade AUC când un feature e shuffled?
    Importanță negativă = feature adaugă noise (mai bine fără el).
    """
    _section(f"PERMUTATION FEATURE IMPORTANCE  {label}  ({n_repeats} repetiții/feature)")

    # Convert to numpy so column indexing [:, fi] works regardless of DataFrame/ndarray input
    X_np     = X_test.values if hasattr(X_test, 'values') else np.asarray(X_test)
    y_proba  = model.predict_proba(X_np)[:, 1]
    base_auc = roc_auc_score(y_test, y_proba)
    rng      = np.random.RandomState(42)
    rows     = []

    for fi, feat in enumerate(features):
        drops = []
        for _ in range(n_repeats):
            Xp = X_np.copy()
            Xp[:, fi] = rng.permutation(Xp[:, fi])
            try:
                pa = roc_auc_score(y_test, model.predict_proba(Xp)[:, 1])
            except Exception:
                pa = base_auc
            drops.append(base_auc - pa)
        rows.append({'feature': feat,
                     'importance': float(np.mean(drops)),
                     'std': float(np.std(drops))})

    df = (pd.DataFrame(rows)
            .sort_values('importance', ascending=False)
            .reset_index(drop=True))

    print(f"\n   Base AUC: {base_auc:.4f}\n")
    print(f"   {'#':<4} {'Feature':<35} {'ΔAUC':>9} {'Std':>7}  Bar")
    print(f"   {'-'*70}")

    for i, row in df.head(20).iterrows():
        bar  = _bar(row['importance'], scale=300)
        flag = "  ❌ NOISE" if row['importance'] < -0.005 else ""
        print(f"   {i+1:<4} {row['feature']:<35} {row['importance']:>9.4f} "
              f"{row['std']:>7.4f}  {bar}{flag}")

    neg = df[df['importance'] < -0.01]
    if len(neg):
        print(f"\n   ⚠️  Features cu importanță negativă (candidați eliminare):")
        for _, r in neg.iterrows():
            print(f"      {r['feature']}: {r['importance']:.4f}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 8. WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def walk_forward_validation(X: np.ndarray, y: np.ndarray,
                             model_factory,
                             n_splits: int = 5,
                             purge: int = 30,
                             label: str = "") -> list:
    """
    Purged walk-forward: train pe trecut, test pe viitor cu gap de purge bare.
    Fiecare fold test = perioadă nevăzută de model.
    Verifică stabilitatea: dacă AUC variază mult între folds → overfitting.
    """
    _section(f"WALK-FORWARD VALIDATION  {label}  ({n_splits} folds, purge={purge})")

    n         = len(X)
    fold_size = n // (n_splits + 1)
    results   = []

    print(f"\n   {'Fold':<5} {'Train N':>10} {'Test N':>8} {'AUC':>8} "
          f"{'Prec@55':>9} {'EV@55':>8} {'Sig':>6}")
    print(f"   {'-'*62}")

    for fold in range(n_splits):
        tr_end   = fold_size * (fold + 1)
        te_start = tr_end + purge
        te_end   = min(te_start + fold_size, n)

        if te_end - te_start < 20:
            continue

        Xtr, ytr = X[:tr_end], y[:tr_end]
        Xte, yte = X[te_start:te_end], y[te_start:te_end]

        n_pos = (ytr == 1).sum()
        n_neg = (ytr == 0).sum()
        spw   = max(n_neg / max(n_pos, 1), 0.5)

        model = model_factory(scale_pos_weight=spw)
        model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

        yp = model.predict_proba(Xte)[:, 1]
        try:
            auc = roc_auc_score(yte, yp)
        except Exception:
            auc = 0.5

        mask55  = yp >= 0.55
        n55     = mask55.sum()
        prec55  = float(yte[mask55].mean()) if n55 > 0 else 0.0
        ev55    = prec55 * WIN_R_AVG - (1 - prec55) * LOSS_R_AVG

        results.append({'fold': fold + 1, 'auc': auc,
                        'prec_55': prec55, 'ev_55': ev55, 'n_sig': n55})

        flag = " ✅" if auc > 0.60 else " ⚠️"
        print(f"   {fold+1:<5} {tr_end:>10,} {te_end-te_start:>8,} {auc:>8.4f} "
              f"{prec55:>9.1%} {ev55:>8.3f} {n55:>6}{flag}")

    if results:
        m_auc   = np.mean([r['auc'] for r in results])
        s_auc   = np.std([r['auc'] for r in results])
        m_prec  = np.mean([r['prec_55'] for r in results])
        m_ev    = np.mean([r['ev_55'] for r in results])
        print(f"\n   {'MEAN':<5} {'':>10} {'':>8} {m_auc:>8.4f} {m_prec:>9.1%} {m_ev:>8.3f}")
        print(f"   {'STD':<5} {'':>10} {'':>8} {s_auc:>8.4f}")
        stab = "STABILĂ ✅" if s_auc < 0.05 else ("VARIABILĂ ⚠️" if s_auc < 0.10 else "INSTABILĂ ❌")
        print(f"\n   Stabilitate AUC: {stab}  (std={s_auc:.4f})")
        print(f"   AUC mediu: {m_auc:.4f}  |  EV mediu@0.55: {m_ev:.3f}R")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 9. MASTER — run_full_analysis
# ══════════════════════════════════════════════════════════════════════════════
def run_full_analysis(model,
                      X_train: np.ndarray, X_test: np.ndarray,
                      y_train: np.ndarray, y_test: np.ndarray,
                      features: list,
                      risk_pts: float = 10.0,
                      threshold: float = 0.55,
                      label: str = "",
                      save_dir: str = None,
                      model_factory=None,
                      monte_carlo_sims: int = 1000,
                      run_perm_imp: bool = True,
                      run_walk_fwd: bool = True,
                      n_perm_imp_repeats: int = 5):
    """
    Master function — rulează TOATE testele pe un model binar antrenat.

    Parametri:
      model          : model XGBoost antrenat
      X_train/X_test : numpy arrays
      y_train/y_test : numpy arrays (0=FAKE/NEG, 1=REAL/POS)
      features       : lista cu nume features (același ordin ca X)
      risk_pts       : SL mediu în puncte NQ (default 10)
      threshold      : threshold principal pentru equity/MC
      label          : prefix afișat în toate secțiunile
      save_dir       : director salvare CSVuri (corelații, importanță)
      model_factory  : funcție (scale_pos_weight=X) → model nou (pt walk-fwd)
      monte_carlo_sims: număr simulări MC
      run_perm_imp   : rulează permutation feature importance
      run_walk_fwd   : rulează walk-forward (mai lent)
      n_perm_imp_repeats: repetiții per feature pt permutation importance
    """
    print(f"\n{'╔'+'═'*62+'╗'}")
    print(f"║  {'ANALYTICS SUITE — ' + label:<60}  ║")
    print(f"{'╚'+'═'*62+'╝'}")

    y_proba = model.predict_proba(X_test)[:, 1]

    # Paths pentru salvare
    _dir = Path(save_dir) if save_dir else None

    # ── 1. Correlatie ─────────────────────────────────────────────────────────
    X_test_df = pd.DataFrame(X_test, columns=features)
    corr_csv  = str(_dir / f"corr_{label.lower()}.csv") if _dir else None
    analyze_feature_correlations(X_test_df, features, threshold=0.80,
                                  save_csv=corr_csv)

    # ── 2. Threshold analysis ─────────────────────────────────────────────────
    thr_results = threshold_analysis(y_test, y_proba,
                                      risk_pts=risk_pts, label=label)

    # ── 3. Calibration ────────────────────────────────────────────────────────
    calibration_analysis(y_test, y_proba, label=label)

    # ── 4. Equity curve ───────────────────────────────────────────────────────
    equity_curve_analysis(y_test, y_proba, threshold=threshold,
                           risk_pts=risk_pts, label=label)

    # ── 5. Monte Carlo ────────────────────────────────────────────────────────
    mc_result = monte_carlo_simulation(y_test, y_proba, threshold=threshold,
                                        risk_pts=risk_pts, n_sims=monte_carlo_sims,
                                        label=label)

    # ── 6. Permutation significance ───────────────────────────────────────────
    permutation_significance_test(model, X_test, y_test,
                                   n_permutations=200, label=label)

    # ── 7. Permutation feature importance ─────────────────────────────────────
    if run_perm_imp:
        df_imp = permutation_feature_importance(
            model, X_test, y_test, features,
            n_repeats=n_perm_imp_repeats, label=label)
        if _dir:
            df_imp.to_csv(_dir / f"perm_imp_{label.lower()}.csv", index=False)

    # ── 8. Walk-forward ───────────────────────────────────────────────────────
    if run_walk_fwd and model_factory is not None:
        X_all = np.vstack([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])
        walk_forward_validation(X_all, y_all, model_factory,
                                 n_splits=5, purge=30, label=label)
    elif run_walk_fwd and model_factory is None:
        print(f"\n   ⏸️  Walk-forward: SKIP (model_factory nespecificat)")

    # ── Summary ───────────────────────────────────────────────────────────────
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n{'─'*64}")
    print(f"  ANALYTICS SUMMARY — {label}")
    print(f"{'─'*64}")
    print(f"  AUC test:         {auc:.4f}")
    if threshold in thr_results:
        r = thr_results[threshold]
        print(f"  Prec @ {threshold:.0%}:       {r['prec']:.1%}  ({r['n_sig']} semnale în test)")
        print(f"  EV  @ {threshold:.0%}:       {r['ev_r']:.3f}R = {r['ev_r']*risk_pts:.1f} pts/trade")
    if mc_result:
        print(f"  MC P50 return:    ${mc_result.get('p50_annual',0):,.0f}/an")
        print(f"  MC prob profit:   {mc_result.get('prob_profitable',0):.1f}%")
    print(f"{'═'*64}\n")

    return {'auc': auc, 'mc': mc_result, 'threshold_results': thr_results}
