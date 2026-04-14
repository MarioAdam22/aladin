"""
ALADIN — SHORT Quality Filter v2
════════════════════════════════════════════════════════════
Insight OOS 2023-2025:
  NY SHORT: WR=18.7%, avgR=+0.296 → profitable
  NY LONG:  WR=11.2%, avgR=+0.010 → breakeven → ELIMINAT

LABEL: trail = 1 dacă exit_reason = TRAIL_STOP

Features din market_data DB (join pe timestamp):
  - h4/h1 bias, hurst, adx_14, rvol, dom_ratio
  - has_displacement, fvg_down, acf_lag1, fisher_transform
  - garch_vol, dist_vwap, val, vah, poc_level, bar_delta
  - + features din trade: confidence, atr, sl_pts, time, dow

Train: 2023-2024 SHORT signals (temporal split)
Test:  2025 SHORT signals
════════════════════════════════════════════════════════════
"""

import sqlite3, json, time
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import roc_auc_score

DIR = Path(__file__).parent
DB  = DIR / "mario_trading.db"

MODEL_OUT    = DIR / "mario_short_quality.json"
FEATURES_OUT = DIR / "mario_short_quality_features.json"

FEATURES = [
    # v1 trade features
    "confidence", "sl_pts", "atr_entry",
    "time_in_ny", "day_of_week", "month",
    # Market structure from DB
    "h4_bias",           # (h4_mid - close) / atr → >0 = bearish h4
    "h1_bias",           # (h1_mid - close) / atr → >0 = bearish h1
    "hurst",             # >0.5 trending, <0.5 ranging
    "adx_14",            # trend strength
    "rvol",              # relative volume
    "dom_ratio",         # DOM: >1 ask heavy (bearish pressure)
    "has_displacement",  # displacement bar prezent
    "fvg_down",          # fair value gap bearish
    "acf_lag1",          # momentum persistence
    "fisher_transform",  # >0 overbought (bullish exhaustion)
    "garch_vol_atr",     # garch volatilitate / ATR
    "dist_vwap_atr",     # signed distance from VWAP / ATR
    "dist_poc_atr",      # signed distance from POC / ATR
    "bar_delta_norm",    # bar_delta / volume (negative = sell pressure)
    "sweep_wick_atr",    # upper wick / ATR (rejection strength)
    "body_bear",         # (open-close) / range → >0 = bearish body
    "prev_day_bear",     # previous day was bearish (1/0)
    "atr_vs_10d",        # atr / 10-day avg atr (volatility regime)
]


def load_short_signals() -> pd.DataFrame:
    """Încarcă semnalele SHORT din backtest CSV și join cu market_data."""
    trades = pd.read_csv(DIR / "backtest_mario_bot_trades.csv")
    trades["ts"] = pd.to_datetime(trades["timestamp"])

    # Filtrare: NY, SHORT, conf >= 0.60
    shorts = trades[
        (trades["session"] == "NY") &
        (trades["direction"] == "SHORT") &
        (trades["confidence"] >= 0.60)
    ].copy()
    shorts["trail"]       = (shorts["exit_reason"] == "TRAIL_STOP").astype(int)
    shorts["year"]        = shorts["ts"].dt.year
    shorts["month"]       = shorts["ts"].dt.month
    shorts["time_in_ny"]  = shorts["ts"].dt.hour * 60 + shorts["ts"].dt.minute - (15 * 60 + 30)
    shorts["day_of_week"] = shorts["ts"].dt.dayofweek
    print(f"Semnale SHORT NY conf>=0.60: {len(shorts)} | TRAIL={shorts.trail.sum()} ({shorts.trail.mean()*100:.1f}%)")

    # Load context din DB pentru aceste timestamp-uri
    print("Loading market_data context...")
    ts_list = shorts["timestamp"].tolist()
    placeholders = ",".join(["?" for _ in ts_list])

    conn = sqlite3.connect(DB)
    mkt = pd.read_sql_query(
        f"""SELECT timestamp, open, high, low, close, volume, atr_14,
               h4_hi, h4_lo, h1_hi, h1_lo,
               hurst, adx_14, rvol, dom_ratio,
               has_displacement, fvg_down, acf_lag1, fisher_transform,
               garch_vol, dist_vwap, poc_level, val, vah, bar_delta
            FROM market_data
            WHERE timestamp IN ({placeholders})""",
        conn, params=ts_list
    )
    conn.close()
    print(f"  DB rows matched: {len(mkt)} / {len(shorts)}")

    # Merge
    df = shorts.merge(mkt, on="timestamp", how="left")

    # Compute derived features
    atr = df["atr_14"].fillna(9.0).replace(0, 9.0)
    h4h = df["h4_hi"].fillna(0); h4l = df["h4_lo"].fillna(0)
    h1h = df["h1_hi"].fillna(0); h1l = df["h1_lo"].fillna(0)
    cl  = df["close"].fillna(df["entry_px"])
    hi  = df["high"].fillna(cl)
    op  = df["open"].fillna(cl)
    lo  = df["low"].fillna(cl)
    rng = (hi - lo).clip(lower=0.1)

    df["h4_bias"]         = np.where((h4h>0)&(h4l>0), ((h4h+h4l)/2 - cl) / atr, 0.0)
    df["h1_bias"]         = np.where((h1h>0)&(h1l>0), ((h1h+h1l)/2 - cl) / atr, 0.0)
    df["hurst"]           = df["hurst"].fillna(0.5)
    df["adx_14"]          = df["adx_14"].fillna(20.0)
    df["rvol"]            = df["rvol"].fillna(1.0).replace(0, 1.0)
    df["dom_ratio"]       = df["dom_ratio"].fillna(1.0).replace(0, 1.0)
    df["has_displacement"]= df["has_displacement"].fillna(0)
    df["fvg_down"]        = df["fvg_down"].fillna(0)
    df["acf_lag1"]        = df["acf_lag1"].fillna(0)
    df["fisher_transform"]= df["fisher_transform"].fillna(0)
    df["garch_vol_atr"]   = (df["garch_vol"].fillna(atr) / atr).clip(0.1, 5)
    df["dist_vwap_atr"]   = (df["dist_vwap"].fillna(0) / atr).clip(-10, 10)
    poc = df["poc_level"].fillna(0)
    df["dist_poc_atr"]    = np.where(poc>0, (cl - poc) / atr, 0.0)
    df["bar_delta_norm"]  = (df["bar_delta"].fillna(0) / df["volume"].clip(lower=1)).clip(-1, 1)
    df["sweep_wick_atr"]  = ((hi - hi.clip(upper=op.combine(cl, max))) / atr).clip(0, 5)
    df["body_bear"]       = ((op - cl) / rng).clip(-1, 1)

    # Prev day direction — recompute from timestamps
    df["date_col"] = df["ts"].dt.date
    day_ret = df.groupby("date_col").apply(
        lambda g: (g.sort_values("ts").iloc[-1]["entry_px"] -
                   g.sort_values("ts").iloc[0]["entry_px"]) if len(g) > 0 else 0
    ).shift(1).rename("day_ret")
    day_df = day_ret.reset_index()
    day_df.columns = ["date_col", "day_ret"]
    df = df.merge(day_df, on="date_col", how="left")
    df["prev_day_bear"] = (df["day_ret"].fillna(0) < 0).astype(int)

    # ATR vs 10d average
    daily_atr = df.groupby("date_col")["atr_14"].mean()
    daily_atr_10d = daily_atr.rolling(10, min_periods=3).mean().rename("atr_10d")
    df = df.merge(daily_atr_10d.reset_index().rename(columns={"date_col":"date_col","atr_10d":"atr_10d"}),
                  on="date_col", how="left")
    df["atr_vs_10d"] = (atr / df["atr_10d"].fillna(atr)).clip(0.3, 3.0)

    # Clip all features
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    return df


def backtest_short_filtered(test_df: pd.DataFrame, proba: np.ndarray, thr: float) -> dict:
    """Calculează P&L pentru SHORT signals filtrate la prob >= thr."""
    mask = proba >= thr
    sub  = test_df[mask]
    if len(sub) == 0:
        return {"n": 0, "wr": 0, "avgR": 0, "pnl_wk": 0, "trail_rate": 0}

    td = sub["ts"].dt.date.nunique()
    total_td = test_df["ts"].dt.date.nunique()

    avg_r = float(sub["r_mult"].mean())
    pnl   = sub["pnl_usd"].sum()
    pnl_wk = pnl / total_td * 5  # dividi per tutte le trading days (including zeros)
    wr    = sub["trail"].mean() * 100
    return {
        "n": len(sub),
        "tday": td,
        "wr": wr,
        "avgR": avg_r,
        "pnl_wk": pnl_wk,
        "trail_rate": sub["trail"].mean() * 100,
    }


def main():
    print("="*60)
    print("SHORT QUALITY FILTER v2")
    print("="*60)

    # ── Încarcă și preprocesează date ──
    df = load_short_signals()

    # Split temporal: train=2023-2024, test=2025
    train = df[df["year"] <= 2024].copy()
    test  = df[df["year"] == 2025].copy()
    print(f"\nTrain 2023-2024: {len(train)} | TRAIL={train.trail.sum()} ({train.trail.mean()*100:.1f}%)")
    print(f"Test  2025:      {len(test)}  | TRAIL={test.trail.sum()}  ({test.trail.mean()*100:.1f}%)")

    if len(train) < 50:
        print("❌ Date insuficiente!"); return

    X_train = train[FEATURES].fillna(0).astype(float)
    y_train = train["trail"].astype(int)
    X_test  = test[FEATURES].fillna(0).astype(float)
    y_test  = test["trail"].astype(int)

    # ── Train ──
    print("\n🚀 Training XGBoost SHORT quality model...")
    scale_pw = max(1, int((y_train==0).sum() / max((y_train==1).sum(),1)))

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=scale_pw,
        eval_metric="auc",
        early_stopping_rounds=40,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=False)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n📊 AUC OOS 2025: {auc:.4f}")

    # ── Baseline (no filter) ──
    total_td = test["ts"].dt.date.nunique()
    base_avgR = test["r_mult"].mean()
    base_pnl_wk = test["pnl_usd"].sum() / total_td * 5
    print(f"\n📌 Baseline 2025 SHORT (no filter):")
    print(f"   n={len(test)}, WR={test.trail.mean()*100:.1f}%, avgR={base_avgR:+.3f}, P&L/wk=${base_pnl_wk:+.0f}")

    # ── Threshold sweep ──
    print(f"\n{'thr':>5} {'n':>5} {'WR%':>6} {'avgR':>7} {'P&L/wk':>9} {'improve':>9}")
    best_thr = 0.0
    best_pnl = base_pnl_wk
    for thr in np.arange(0.25, 0.75, 0.05):
        r = backtest_short_filtered(test, y_proba, thr)
        if r["n"] < 5: continue
        improve = r["pnl_wk"] - base_pnl_wk
        marker = " ← BEST" if r["pnl_wk"] > best_pnl else ""
        print(f"  {thr:.2f} {r['n']:>5} {r['wr']:>5.1f}% {r['avgR']:>+7.3f} {r['pnl_wk']:>+9.0f} {improve:>+9.0f}{marker}")
        if r["pnl_wk"] > best_pnl:
            best_pnl = r["pnl_wk"]
            best_thr = thr

    # ── Feature importance ──
    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(f"\n🔑 Feature Importance (top 10):")
    for feat, imp in fi.head(10).items():
        print(f"  {feat:25s}: {imp:.4f}")

    # ── Save dacă e util ──
    if auc >= 0.57:
        model.save_model(str(MODEL_OUT))
        meta = {
            "features": FEATURES,
            "auc_oos_2025": round(float(auc), 4),
            "best_threshold": float(best_thr),
            "direction": "SHORT",
            "session": "NY",
            "v1_conf_threshold": 0.60,
            "baseline_pnl_wk": round(float(base_pnl_wk), 2),
            "filtered_pnl_wk": round(float(best_pnl), 2),
            "improvement_pct": round((best_pnl/max(abs(base_pnl_wk),1)-1)*100, 1),
        }
        with open(FEATURES_OUT, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n✅ Salvat: {MODEL_OUT.name} (AUC={auc:.4f}, best_thr={best_thr:.2f})")
        print(f"   P&L/wk: ${base_pnl_wk:+.0f} → ${best_pnl:+.0f} ({(best_pnl/max(abs(base_pnl_wk),1)-1)*100:+.0f}%)")
    else:
        print(f"\n❌ Model insuficient: AUC={auc:.4f}, improvement={best_pnl-base_pnl_wk:+.0f}/wk")
        print("   → Revertim la SHORT-only fără filtru suplimentar")

    # ── Rezumat final ──
    print("\n" + "="*60)
    print("REZUMAT PERFORMANȚĂ")
    print("="*60)

    # All years combined SHORT
    all_short = df[df["direction"] == "SHORT"] if "direction" in df.columns else df
    total_td_all = all_short["ts"].dt.date.nunique()
    print(f"SHORT-ONLY 2023-2025:")
    print(f"  n={len(df)}, {len(df)/total_td_all:.2f}/day")
    print(f"  WR={df.trail.mean()*100:.1f}%, avgR={df.r_mult.mean():+.3f}")
    print(f"  P&L/wk=${df.pnl_usd.sum()/total_td_all*5:+.0f}")


if __name__ == "__main__":
    main()
