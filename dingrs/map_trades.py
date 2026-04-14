import pandas as pd
import numpy as np
import os

# --- CONFIGURARE ---
# Folosim fișierul MASTER 24h pe care l-am auditatt
PATH_RAW = "/Users/mario/Desktop/Aladin/QQQ_24h_ICT_MASTER.csv" 
PATH_JURNAL = "/Users/mario/Desktop/Aladin/mario_journal.csv"
OUTPUT_PATH = "/Users/mario/Desktop/Aladin/final_training_data.csv"

if not os.path.exists(PATH_RAW):
    print(f"❌ Nu gasesc fisierul MASTER la calea specificata!")
else:
    print("🚀 Incep procesarea tabelului master (Context 24h -> Trading Window)...")
    
    # 1. Incarcam datele (ATENTIE la separatorul ';')
    df = pd.read_csv(PATH_RAW, sep=';')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Europe/Bucharest')
    df['date'] = df['timestamp'].dt.date

    # Stergem coloanele vechi pentru a evita dublurile (Overlap)
    cols_to_clean = ['pdh', 'pdl', 'context_sweep_high', 'context_sweep_low', 'signal']
    df = df.drop(columns=[c for c in cols_to_clean if c in df.columns], errors='ignore')

    # 2. PDH / PDL (Calculat pe baza intregului istoric 24h din fisier)
    print("📈 Calculam PDH/PDL reale...")
    daily_stats = df.groupby('date').agg({'high': 'max', 'low': 'min'}).shift(1)
    daily_stats.columns = ['pdh', 'pdl']
    df = df.join(daily_stats, on='date')

    # 3. CONTEXT SWEEPS (Bazat pe nivelele de 24h)
    df['context_sweep_high'] = (df['high'] > df['pdh']).astype(int)
    df['context_sweep_low'] = (df['low'] < df['pdl']).astype(int)

    # 4. FILTRARE INTERVAL TRADING (09:00 - 18:00)
    # Taiem restul minutelor inutile pentru antrenare, dar dupa ce am extras contextul de 24h
    df = df.set_index('timestamp')
    final_df = df.between_time('09:00', '18:00').reset_index()

    # 5. AUTO-LABELING (Strategia ICT)
    final_df['signal'] = 'NEUTRAL'
    print("🤖 Generam semnale automate (Sweep + MSS)...")
    
    # Long: Sweep PDL + MSS Bullish
    final_df.loc[(final_df['context_sweep_low'] == 1) & (final_df['mss_bullish'] == 1), 'signal'] = 'AUTO_LONG'
    # Short: Sweep PDH + MSS Bearish
    final_df.loc[(final_df['context_sweep_high'] == 1) & (final_df['mss_bearish'] == 1), 'signal'] = 'AUTO_SHORT'

    # 6. MAPARE JURNAL (Daca exista)
    if os.path.exists(PATH_JURNAL):
        print("📝 Sincronizam cu jurnalul tau personal...")
        try:
            df_j = pd.read_csv(PATH_JURNAL, sep=';', encoding='utf-8-sig', skiprows=1)
            df_j.columns = df_j.columns.str.strip()
            for i, trade in df_j.iterrows():
                t_date = pd.to_datetime(trade['date']).strftime('%Y-%m-%d')
                t_hour = str(trade['hour_min']).strip()
                # Cautam minutul exact in tabelul mare
                mask = (final_df['timestamp'].dt.strftime('%Y-%m-%d') == t_date) & \
                       (final_df['timestamp'].dt.strftime('%H:%M') == t_hour)
                
                if mask.any():
                    final_df.loc[mask, 'signal'] = str(trade['signal']).strip()
        except Exception as e:
            print(f"⚠️ Eroare jurnal: {e}")

    # 7. SALVARE FINALA (CSV standard pentru antrenare)
    final_df.to_csv(OUTPUT_PATH, index=False)
    
    total_auto = final_df[final_df['signal'].str.contains('AUTO')].shape[0]
    print("-" * 50)
    print(f"✅ GATA! Fisierul de training a fost salvat: {OUTPUT_PATH}")
    print(f"🎯 Setup-uri identificate (09:00-18:00): {total_auto}")
    print(f"📊 Total randuri finale: {len(final_df):,}")
    print("-" * 50)