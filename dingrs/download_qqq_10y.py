import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame
import datetime
import time
import os

# CONFIGURARE
API_KEY = "PKR25GKASP4V24KDCEEXVB6UZZ"
SECRET_KEY = "8DK8T1MMj6ojoUQdZvqFPqC1wUNHBmfaKY6CMLqNaEWv"
rest_api = REST(API_KEY, SECRET_KEY, base_url='https://paper-api.alpaca.markets')

def download_raw_data():
    # Definim activele necesare pentru SMT (QQQ vs SPY)
    symbols = ["QQQ", "SPY"]
    
    # Perioada extinsă: 2019 - Prezent
    start_year = 2019 
    end_year = 2026
    
    print(f"🚀 Pornesc descărcarea datelor BRUTE pentru QQQ și SPY (2019-2026)...")

    for symbol in symbols:
        output_path = f"/Users/mario/Desktop/Aladin/{symbol}_24h_ICT_MASTER.csv"
        all_chunks = []
        print(f"\n📦 Colectez date pentru: {symbol}")

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Oprim descărcarea la data curentă (Februarie 2026)
                if year == 2026 and month > 2: break
                
                start_dt = f"{year}-{month:02d}-01"
                if month == 12:
                    end_dt = f"{year}-12-31"
                else:
                    end_dt = (datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

                print(f"  📅 {year}-{month:02d}...", end=" ", flush=True)
                
                try:
                    # Descărcăm doar OHLCV brute
                    bars = rest_api.get_bars(symbol, TimeFrame.Minute, start_dt, end_dt, adjustment='split').df
                    
                    if not bars.empty:
                        # Aliniere DST automată (București)
                        bars.index = bars.index.tz_convert('Europe/Bucharest')
                        
                        # Păstrăm doar coloanele esențiale
                        bars = bars[['open', 'high', 'low', 'close', 'volume']]
                        all_chunks.append(bars)
                        print(f"✅")
                    else:
                        print("⚠️ Gol")
                    
                    # Sleep scurt pentru a respecta limitele Alpaca Free Tier
                    time.sleep(0.5) 
                    
                except Exception as e:
                    print(f"❌ Eroare la {symbol}: {e}")
                    continue

        if all_chunks:
            final_df = pd.concat(all_chunks)
            final_df.reset_index(inplace=True)
            
            # Redenumim indexul în timestamp
            if 'timestamp' not in final_df.columns:
                final_df.rename(columns={'index': 'timestamp'}, inplace=True)
                
            # Salvare Brută cu separator ";"
            final_df.to_csv(output_path, index=False, sep=';')
            
            print(f"✨ MASTER {symbol} SALVAT: {output_path}")
            print(f"📊 Total rânduri: {len(final_df):,}")
        else:
            print(f"❌ Nu s-a putut colecta nicio dată pentru {symbol}.")

if __name__ == "__main__":
    download_raw_data()