"""Build sweep_dataset _v3 parquets one year at a time."""
import sys, time
sys.path.insert(0, '/sessions/dreamy-youthful-turing/mnt/Aladin')
from train_sweep_v2 import build_dataset, BASE
import pandas as pd

year = int(sys.argv[1])
out = BASE / f'sweep_dataset_{year}_v3.parquet'
if out.exists():
    df = pd.read_parquet(out)
    print(f"Already exists: {out.name} ({len(df)} rows)")
    sys.exit(0)

t0 = time.time()
df = build_dataset([year])
df.to_parquet(out, index=False)
print(f"Saved {out.name}: {len(df)} rows in {time.time()-t0:.1f}s")
