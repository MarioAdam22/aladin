"""
python3 /Users/mario/Desktop/Aladin/debug_levels.py
"""
import sqlite3, pandas as pd

conn = sqlite3.connect("/Users/mario/Desktop/Aladin/mario_trading.db")

# Toate coloanele
cols = [c[1] for c in conn.execute("PRAGMA table_info(market_data)").fetchall()]
print("=== COLOANE market_data ===")
for c in cols:
    print(f"  {c}")

# Un rand real cu toate valorile
print("\n=== EXEMPLU RAND (2023-11-10 15:30) ===")
row = pd.read_sql_query(
    "SELECT * FROM market_data WHERE timestamp LIKE '2023-11-10 15:30%' LIMIT 1",
    conn
)
if not row.empty:
    for col in row.columns:
        print(f"  {col}: {row.iloc[0][col]}")
else:
    print("  Nu exista timestamp exact - primul rand disponibil:")
    row = pd.read_sql_query("SELECT * FROM market_data LIMIT 1", conn)
    for col in row.columns:
        print(f"  {col}: {row.iloc[0][col]}")

conn.close()
