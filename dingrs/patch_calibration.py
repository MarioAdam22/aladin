"""
PATCH: Recalibrare mario_bot_calibrated.pkl fără reantrenare XGBoost.

Problema: IsotonicRegression antrenat pe cal_X cu 95% WAIT
→ colapsează SHORT/LONG la 0 pentru că nu vede destule exemple.

Fix: Oversampling SHORT/LONG în setul de calibrare → IsotonicRegression
vede distribuție echilibrată → probabilitățile SHORT/LONG sunt corecte.

Durată: ~3-5 minute (nu reantrenăm XGBoost, doar calibrarea)
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
ALADIN_DIR  = "/Users/mario/Desktop/Aladin"
DB_PATH     = f"{ALADIN_DIR}/mario_trading.db"
MODEL_JSON  = f"{ALADIN_DIR}/mario_bot.json"
FEAT_JSON   = f"{ALADIN_DIR}/mario_features.json"
OUT_PKL     = f"{ALADIN_DIR}/mario_bot_calibrated.pkl"

print("=" * 65)
print("  PATCH: Recalibrare Isotonic — fără reantrenare XGBoost")
print("=" * 65)

# ── 1. Import PlattModel din scriptul original ─────────────────────────────
sys.path.insert(0, ALADIN_DIR)
# Importăm doar clasa PlattModel + funcțiile de target din train_mario_ai.py
from train_mario_ai import PlattModel, generate_atr_target, add_sniper_conditions
print("✅ PlattModel + funcții target importate din train_mario_ai.py")

# ── 2. Încarcă modelul XGBoost existent ────────────────────────────────────
import xgboost as xgb
model_xgb = xgb.XGBClassifier()
model_xgb.load_model(MODEL_JSON)
print(f"✅ XGBoost încărcat: {MODEL_JSON}")

# ── 3. Încarcă features din mario_features.json ────────────────────────────
import json
with open(FEAT_JSON) as f:
    feat_meta = json.load(f)
FEATURES = feat_meta["features"]
print(f"✅ {len(FEATURES)} features încărcate")

# ── 4. Încarcă DOAR ultimii 20% din date (test set) ───────────────────────
print("\n⏳ Încărcare date din DB (ultimii 20%)...")
conn = sqlite3.connect(DB_PATH)
total_rows = pd.read_sql("SELECT COUNT(*) as n FROM market_data", conn).iloc[0, 0]
skip_rows  = int(total_rows * 0.80)

df = pd.read_sql(
    f"SELECT * FROM market_data LIMIT {total_rows - skip_rows} OFFSET {skip_rows}",
    conn
)
conn.close()
print(f"✅ {len(df):,} rânduri încărcate (test set)")

# ── 5. Calculează target (la fel ca în training) ────────────────────────────
print("⏳ Calculare target ATR + condiții sniper...")
df = add_sniper_conditions(df)
df['target_atr'] = generate_atr_target(df, horizon=60, atr_multiplier=2.0)
df['target'] = 0

# Aceleași condiții sniper ca în train
short_cond = (
    df.get('sweep_high', pd.Series(False, index=df.index)) &
    (df['target_atr'] == 1)
)
long_cond = (
    df.get('sweep_low', pd.Series(False, index=df.index)) &
    (df['target_atr'] == 2)
)

# Fallback dacă sweep_high/sweep_low nu există — folosim target_atr direct
if short_cond.sum() < 100:
    short_cond = (df['target_atr'] == 1) & df.get('has_displacement', pd.Series(True, index=df.index)).astype(bool)
if long_cond.sum() < 100:
    long_cond  = (df['target_atr'] == 2) & df.get('has_displacement', pd.Series(True, index=df.index)).astype(bool)

df.loc[short_cond, 'target'] = 1
df.loc[long_cond,  'target'] = 2
# Elimină semnale consecutive (deduplication)
df.loc[df['target'].shift(1) == df['target'], 'target'] = 0

counts = df['target'].value_counts()
print(f"   WAIT={counts.get(0,0):,}  SHORT={counts.get(1,0):,}  LONG={counts.get(2,0):,}")

# ── 6. Calculează features derivate care lipsesc din DB ───────────────────
# Acestea sunt calculate on-the-fly în training, nu sunt stocate în market_data
print("⏳ Calculare features derivate (slope, momentum, body_dir, wick_ratio)...")
df['slope_h1']    = (df['close'] - df['close'].shift(60))  / (df['close'].shift(60).abs()  + 1e-8)
df['slope_h4']    = (df['close'] - df['close'].shift(240)) / (df['close'].shift(240).abs() + 1e-8)
df['momentum_15'] = (df['close'] - df['close'].shift(15))  / (df['close'].shift(15).abs()  + 1e-8)
df['body_dir']    = np.sign(df['close'] - df['open'])
df['wick_ratio']  = (df['high'] - df['low']) / ((df['close'] - df['open']).abs() + 1e-8)
print("✅ Features derivate calculate: slope_h1, slope_h4, momentum_15, body_dir, wick_ratio")

# ── 7. Pregătește X, y ─────────────────────────────────────────────────────
available_feats = [f for f in FEATURES if f in df.columns]
missing = set(FEATURES) - set(available_feats)
if missing:
    print(f"   ⚠️  Features încă lipsă (vor fi 0): {missing}")
    for m in missing:
        df[m] = 0.0

df_clean = df[available_feats + ['target']].dropna()
X = df_clean[available_feats].astype(np.float32)
y = df_clean['target'].astype(int)

print(f"✅ X shape: {X.shape} | y distribuție: {dict(y.value_counts())}")

# ── 7. Split pentru calibrare / evaluare ──────────────────────────────────
cal_size  = len(X) // 2
cal_X, eval_X = X.iloc[:cal_size],  X.iloc[cal_size:]
cal_y, eval_y = y.iloc[:cal_size],  y.iloc[cal_size:]

print(f"\n📊 Cal set (pre-oversample): {dict(cal_y.value_counts())}")

# ── 8. ✨ FIX: Oversample SHORT + LONG în setul de calibrare ──────────────
print("⚡ Oversampling SHORT/LONG pentru calibrare echilibrată...")
from sklearn.utils import resample

cal_df = pd.concat([cal_X, cal_y], axis=1)
target_col = cal_y.name if hasattr(cal_y, 'name') else 'target'
cal_df.columns = list(available_feats) + ['target']

wait_df  = cal_df[cal_df['target'] == 0]
short_df = cal_df[cal_df['target'] == 1]
long_df  = cal_df[cal_df['target'] == 2]

# Targetul: SHORT și LONG la 15% din WAIT (nu 1:1:1, ci proporție realistă)
# Asta dă Isotonic-ului destule exemple fără să exagerăm
n_minority_target = max(len(short_df), len(long_df), int(len(wait_df) * 0.15))

short_up = resample(short_df, replace=True,  n_samples=n_minority_target, random_state=42)
long_up  = resample(long_df,  replace=True,  n_samples=n_minority_target, random_state=42)
wait_down = resample(wait_df, replace=False, n_samples=min(len(wait_df), n_minority_target * 5), random_state=42)

cal_balanced = pd.concat([wait_down, short_up, long_up]).sample(frac=1, random_state=42)
cal_X_bal = cal_balanced[available_feats]
cal_y_bal = cal_balanced['target'].astype(int)

print(f"✅ Cal set (post-oversample): {dict(pd.Series(cal_y_bal).value_counts())}")

# ── 9. Reantrenăm IsotonicRegression pe setul BALANSAT ────────────────────
print("\n⏳ Reantrenare IsotonicRegression pe date echilibrate...")
from sklearn.isotonic import IsotonicRegression

raw_proba_cal = model_xgb.predict_proba(cal_X_bal)
cal_y_arr     = np.array(cal_y_bal)

_iso_calibrators = []
for _c in range(3):
    _y_bin = (cal_y_arr == _c).astype(float)
    pos_rate = _y_bin.mean()
    _iso = IsotonicRegression(out_of_bounds='clip')
    _iso.fit(raw_proba_cal[:, _c], _y_bin)
    _iso_calibrators.append(_iso)
    print(f"   Clasa {_c}: {int(_y_bin.sum()):,} pozitive ({pos_rate*100:.1f}%) — fitted OK")

model_calibrated = PlattModel(model_xgb, _iso_calibrators)

# ── 10. Evaluare pe setul de test (distribuție reală) ─────────────────────
print("\n📊 Evaluare pe test set real (fără oversample)...")
from sklearn.metrics import accuracy_score, classification_report

y_proba_cal = model_calibrated.predict_proba(eval_X)
y_pred_cal  = np.argmax(y_proba_cal, axis=1)
acc_cal     = accuracy_score(eval_y, y_pred_cal)

print(f"   Accuracy: {acc_cal:.4f}")
print(classification_report(eval_y, y_pred_cal, target_names=['WAIT','SHORT','LONG'], zero_division=0))

# Verificăm că SHORT și LONG nu mai sunt 0
report_lines = classification_report(eval_y, y_pred_cal, target_names=['WAIT','SHORT','LONG'],
                                     zero_division=0, output_dict=True)
short_f1 = report_lines.get('SHORT', {}).get('f1-score', 0)
long_f1  = report_lines.get('LONG',  {}).get('f1-score', 0)

if short_f1 > 0.01 and long_f1 > 0.01:
    print(f"\n✅ PATCH REUȘIT! SHORT f1={short_f1:.3f}, LONG f1={long_f1:.3f}")
    print("   (Modelul calibrat acum dă semnale SHORT și LONG reale)")
else:
    print(f"\n⚠️  SHORT f1={short_f1:.3f}, LONG f1={long_f1:.3f}")
    print("   Dacă sunt încă mici — normal pe distribuție 95% WAIT, dar > 0 e bine.")

# ── 11. Verificare probabilități medii ────────────────────────────────────
print("\n📈 Distribuție probabilități calibrate (sample 1000 rânduri):")
sample_proba = model_calibrated.predict_proba(eval_X.iloc[:1000])
print(f"   WAIT  avg prob: {sample_proba[:, 0].mean():.3f}")
print(f"   SHORT avg prob: {sample_proba[:, 1].mean():.3f}")
print(f"   LONG  avg prob: {sample_proba[:, 2].mean():.3f}")
print(f"   HIGH CONF SHORT (>60%): {(sample_proba[:, 1] > 0.6).sum()} semnale")
print(f"   HIGH CONF LONG  (>60%): {(sample_proba[:, 2] > 0.6).sum()} semnale")

# ── 12. Salvare ────────────────────────────────────────────────────────────
# Backup modelul vechi
import shutil
backup_path = OUT_PKL.replace('.pkl', '_BACKUP_pre_patch.pkl')
if os.path.exists(OUT_PKL):
    shutil.copy(OUT_PKL, backup_path)
    print(f"\n💾 Backup salvat: {backup_path}")

with open(OUT_PKL, 'wb') as f:
    pickle.dump(model_calibrated, f)
print(f"✅ Model calibrat salvat: {OUT_PKL}")

print("\n" + "=" * 65)
print("  PATCH COMPLET!")
print(f"  Fișier:   {OUT_PKL}")
print(f"  Backup:   {backup_path}")
print(f"  SHORT f1: {short_f1:.3f}  |  LONG f1: {long_f1:.3f}")
print("=" * 65)
