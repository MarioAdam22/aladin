"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ALADIN — stitch_continuous_nq.py                                           ║
║  Lipește 40 contracte quarterly NQ într-un continuous contract              ║
║  Back-adjustment (Panama Canal method) la fiecare roll date                 ║
║  Output: data/NQ_continuous.parquet                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Metodă:
  - Determină roll date pentru fiecare contract (cu ~8 zile înainte de expirare)
  - Expirarea NQ = a 3-a vineri din luna trimestrială (Mar/Jun/Sep/Dec)
  - La fiecare roll calculează gap-ul de preț și ajustează backward toată historia
  - Rezultat: serie continuă fără gap-uri artificiale la expirare

Utilizare:
  python3 stitch_continuous_nq.py
  python3 stitch_continuous_nq.py --dingrs /cale/catre/fisiere --out data/NQ_continuous.parquet
"""

import os
import re
import glob
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ALADIN_DIR = Path(__file__).parent
DINGRS_DIR = ALADIN_DIR / "dingrs"
OUTPUT_DIR = ALADIN_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "NQ_continuous.parquet"

# Roll cu 8 zile înainte de expirare (standard pentru NQ)
ROLL_DAYS_BEFORE_EXPIRY = 8


# ─── HELPER: a 3-a vineri din lună ───────────────────────────────────────────
def third_friday(year: int, month: int) -> datetime:
    """Returnează data celei de-a 3-a vineri din luna dată."""
    # Găsim prima zi a lunii
    first = datetime(year, month, 1)
    # Vineri = weekday 4
    # Câte zile până la prima vineri?
    days_to_friday = (4 - first.weekday()) % 7
    first_friday = first + timedelta(days=days_to_friday)
    third = first_friday + timedelta(weeks=2)
    return third


def get_expiry(contract_str: str) -> datetime:
    """
    Din string-ul contractului (ex: '03-26', '12-25') returnează data expirării.
    Format: MM-YY (luna-an pe 2 cifre)
    """
    month_str, year_str = contract_str.split('-')
    month = int(month_str)
    year = 2000 + int(year_str)
    return third_friday(year, month)


def get_roll_date(contract_str: str) -> datetime:
    """Roll date = expiry - ROLL_DAYS_BEFORE_EXPIRY zile."""
    expiry = get_expiry(contract_str)
    return expiry - timedelta(days=ROLL_DAYS_BEFORE_EXPIRY)


# ─── CITIRE FIȘIER NT8 ────────────────────────────────────────────────────────
def read_nt8_file(file_path: str, date_from: datetime = None,
                   date_to: datetime = None) -> pd.DataFrame:
    """
    Citește fișier NT8: YYYYMMDD HHMMSS;open;high;low;close;volume
    Optimizat: filtrează la nivel de string înainte de parse pentru viteză maximă.
    """
    try:
        # Pre-filter rapid cu grep-like pe string date (format YYYYMMDD)
        if date_from is not None or date_to is not None:
            rows = []
            from_str = date_from.strftime('%Y%m%d') if date_from else '00000000'
            to_str   = date_to.strftime('%Y%m%d')   if date_to   else '99999999'
            with open(file_path, 'r') as f:
                for line in f:
                    date_part = line[:8]
                    if from_str <= date_part <= to_str:
                        rows.append(line)
            if not rows:
                return pd.DataFrame()
            from io import StringIO
            df = pd.read_csv(
                StringIO(''.join(rows)),
                sep=';',
                header=None,
                names=['ts_raw', 'open', 'high', 'low', 'close', 'volume'],
                dtype={'ts_raw': str, 'open': float, 'high': float,
                       'low': float, 'close': float, 'volume': float}
            )
        else:
            df = pd.read_csv(
                file_path,
                sep=';',
                header=None,
                names=['ts_raw', 'open', 'high', 'low', 'close', 'volume'],
                dtype={'ts_raw': str, 'open': float, 'high': float,
                       'low': float, 'close': float, 'volume': float}
            )

        df = df.dropna(subset=['ts_raw'])
        df['timestamp'] = pd.to_datetime(df['ts_raw'], format='%Y%m%d %H%M%S', errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.drop(columns=['ts_raw'])
        df['volume'] = df['volume'].fillna(0).astype(int)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  ❌ Eroare {file_path}: {e}")
        return pd.DataFrame()


# ─── IDENTIFICARE CONTRACTE ───────────────────────────────────────────────────
def find_nq_files(dingrs_dir: Path) -> list:
    """
    Găsește toate fișierele NQ și returnează lista sortată cu (contract_str, file_path, expiry).
    """
    pattern = re.compile(r'NQ[_ ](\d{2}-\d{2})\.Last\.txt', re.IGNORECASE)
    files = []

    for f in sorted(dingrs_dir.glob('NQ*.txt')):
        m = pattern.search(f.name)
        if m:
            contract_str = m.group(1)
            try:
                expiry = get_expiry(contract_str)
                files.append({
                    'contract': contract_str,
                    'file': str(f),
                    'expiry': expiry,
                    'roll_date': get_roll_date(contract_str)
                })
            except Exception as e:
                print(f"  ⚠️  Skip {f.name}: {e}")

    # Sort by expiry
    files = sorted(files, key=lambda x: x['expiry'])
    return files


# ─── MAIN STITCHING ───────────────────────────────────────────────────────────
def stitch_contracts(files: list, verbose: bool = True) -> pd.DataFrame:
    """
    Panama Canal back-adjustment:
    1. Procesăm contractele de la cel mai recent spre cel mai vechi
    2. La fiecare roll calculăm gap-ul și ajustăm datele mai vechi
    """
    if not files:
        raise ValueError("Nu s-au găsit fișiere NQ!")

    if verbose:
        print(f"\n📦 Procesez {len(files)} contracte NQ...\n")

    # Citim fișierele — doar perioada activă per contract (optimizat)
    contract_dfs = {}
    for i, info in enumerate(files):
        # Perioada activă = de la roll_date(prev) - 2 zile până la roll_date(current) + 2 zile
        if i > 0:
            date_from = files[i-1]['roll_date'] - timedelta(days=2)
        else:
            date_from = None  # primul contract — citim tot de la început
        date_to = info['roll_date'] + timedelta(days=2)

        if verbose:
            from_str = date_from.strftime('%Y-%m-%d') if date_from else 'start'
            print(f"  📄 {info['contract']} (roll: {info['roll_date'].strftime('%Y-%m-%d')}) "
                  f"| citesc {from_str} → {date_to.strftime('%Y-%m-%d')}")

        df = read_nt8_file(info['file'], date_from=date_from, date_to=date_to)
        if not df.empty:
            df['contract'] = info['contract']
            df['expiry'] = info['expiry']
            contract_dfs[info['contract']] = df

    if not contract_dfs:
        raise ValueError("Niciun fișier NQ valid găsit!")

    # Construim continuous contract de la cel mai recent spre cel mai vechi
    # Folosim intervalele active: fiecare contract e activ de la roll_date(prev) la roll_date(current)

    all_segments = []
    cumulative_adjustment = 0.0

    # Iterăm de la cel mai recent spre cel mai vechi
    for i in range(len(files) - 1, -1, -1):
        info = files[i]
        contract_str = info['contract']

        if contract_str not in contract_dfs:
            continue

        df = contract_dfs[contract_str].copy()

        # Determinăm intervalul activ pentru acest contract
        # De la roll_date(i-1) până la roll_date(i)
        if i > 0:
            prev_info = files[i - 1]
            start_date = prev_info['roll_date']
        else:
            start_date = df['timestamp'].min()

        end_date = info['roll_date']

        # Filtrăm bara activă
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] < end_date)
        segment = df[mask].copy()

        if segment.empty:
            if verbose:
                print(f"  ⚠️  {contract_str}: segment gol între {start_date.date()} și {end_date.date()}")
            continue

        # Calculăm gap față de contractul următor (dacă există)
        if i < len(files) - 1:
            next_contract_str = files[i + 1]['contract']
            if next_contract_str in contract_dfs:
                next_df = contract_dfs[next_contract_str]

                # Găsim prima bară din contractul următor după roll date
                next_first = next_df[next_df['timestamp'] >= end_date].head(1)
                # Găsim ultima bară din contractul curent înainte de roll
                current_last = segment.tail(1)

                if not next_first.empty and not current_last.empty:
                    # Gap = open contractul nou - close contractul vechi la roll
                    gap = next_first['open'].iloc[0] - current_last['close'].iloc[0]
                    cumulative_adjustment += gap

                    if verbose and abs(gap) > 0.25:
                        print(f"  🔄 Roll {contract_str}→{next_contract_str}: "
                              f"gap={gap:+.2f} | adj cumulativ={cumulative_adjustment:+.2f}")

        # Aplicăm ajustarea cumulativă pe segmentul curent
        if cumulative_adjustment != 0:
            for col in ['open', 'high', 'low', 'close']:
                segment[col] = segment[col] + cumulative_adjustment

        all_segments.append(segment)

    if not all_segments:
        raise ValueError("Nu s-au generat segmente valide!")

    # Concatenăm și sortăm
    continuous = pd.concat(all_segments, ignore_index=True)
    continuous = continuous.sort_values('timestamp').reset_index(drop=True)

    # Eliminăm duplicate de timestamp (pot apărea la granițele roll)
    continuous = continuous.drop_duplicates(subset=['timestamp'], keep='last')
    continuous = continuous.sort_values('timestamp').reset_index(drop=True)

    if verbose:
        print(f"\n✅ Contract continuu generat:")
        print(f"   Bare totale: {len(continuous):,}")
        print(f"   Interval: {continuous['timestamp'].min()} → {continuous['timestamp'].max()}")
        print(f"   Preț range: {continuous['close'].min():.2f} → {continuous['close'].max():.2f}")

    return continuous


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Stitch NQ continuous contract')
    parser.add_argument('--dingrs', type=str, default=str(DINGRS_DIR),
                        help='Director cu fișierele NQ*.Last.txt')
    parser.add_argument('--out', type=str, default=str(OUTPUT_FILE),
                        help='Output parquet path')
    parser.add_argument('--quiet', action='store_true', help='Fără output verbose')
    args = parser.parse_args()

    dingrs = Path(args.dingrs)
    out_path = Path(args.out)

    print("=" * 65)
    print("  ALADIN — NQ Continuous Contract Builder")
    print("=" * 65)

    # Găsim fișierele
    files = find_nq_files(dingrs)
    if not files:
        print(f"❌ Nu s-au găsit fișiere NQ în {dingrs}")
        return 1

    print(f"\n📁 Fișiere găsite: {len(files)} contracte")
    print(f"   Cel mai vechi: {files[0]['contract']} (exp: {files[0]['expiry'].strftime('%Y-%m-%d')})")
    print(f"   Cel mai nou:   {files[-1]['contract']} (exp: {files[-1]['expiry'].strftime('%Y-%m-%d')})")

    # Stitching
    continuous = stitch_contracts(files, verbose=not args.quiet)

    # Salvăm
    out_path.parent.mkdir(parents=True, exist_ok=True)
    continuous.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n💾 Salvat: {out_path} ({size_mb:.1f} MB)")
    print(f"   Coloane: {list(continuous.columns)}")

    return 0


if __name__ == '__main__':
    exit(main())
