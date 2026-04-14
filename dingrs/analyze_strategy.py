import pandas as pd

# 1. Încărcăm dataset-ul MASTER îmbunătățit
path = "/Users/mario/Desktop/Aladin/AI_MASTER_DATASET.csv"
df = pd.read_csv(path)

# 2. Filtrăm doar momentele în care ai dat semnal
trades = df[df['user_signal'] != 'NEUTRAL'].copy()

print("📊 ANALIZA STRATEGIEI TALE (BACKTEST CU CONTEXT 15 MIN)")
print("-" * 60)

if trades.empty:
    print("⚠️ Nu am găsit semnale în coloana 'user_signal'.")
else:
    for index, row in trades.iterrows():
        print(f"📅 Trade la ora: {row['timestamp']}")
        print(f"   🔹 Tip: {row['user_signal']}")
        
        # Verificăm contextul (dacă a fost un sweep/fvg în ultimele 15 min)
        sweep_l = "✅ DA" if row['context_sweep_low'] == 1 else "❌ NU"
        sweep_h = "✅ DA" if row['context_sweep_high'] == 1 else "❌ NU"
        fvg_bull = "✅ DA" if row['context_fvg_bull'] == 1 else "❌ NU"
        fvg_bear = "✅ DA" if row['context_fvg_bear'] == 1 else "❌ NU"
        
        print(f"   🔹 Sweep London Low (Recent): {sweep_l}")
        print(f"   🔹 Sweep London High (Recent): {sweep_h}")
        print(f"   🔹 FVG Bullish (Recent): {fvg_bull}")
        print(f"   🔹 FVG Bearish (Recent): {fvg_bear}")
        
        dist_pdl = row['close'] - row['pdl']
        print(f"   🔹 Distanța față de PDL: {dist_pdl:.2f} puncte")
        print("-" * 40)

# 3. Statistici Generale bazate pe CONTEXT
print("\n📈 STATISTICI PENTRU AI (ACURATEȚE PATTERN):")
fvg_count = (trades['context_fvg_bull'].sum() + trades['context_fvg_bear'].sum())
sweep_count = (trades['context_sweep_low'].sum() + trades['context_sweep_high'].sum())

fvg_rate = (fvg_count / len(trades)) * 100 if len(trades) > 0 else 0
sweep_rate = (sweep_count / len(trades)) * 100 if len(trades) > 0 else 0

print(f"🔥 Corelație cu FVG (Context): {fvg_rate:.1f}%")
print(f"🔥 Corelație cu London Sweeps (Context): {sweep_rate:.1f}%")
print("-" * 60)