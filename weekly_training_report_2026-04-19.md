# Aladin weekly training — run report

Scheduled task: `aladin-weekly-training`
Run at: 2026-04-19 (Sunday)
Command attempted: `bash ~/Desktop/Aladin/refresh_model.sh --train`

## Result: FAILED — script is broken

Training did not run. The script fails before the training step because three paths inside it are stale (files were moved or deleted during a recent reorg). Even if the earlier steps were fixed, the training step itself points at a file that no longer exists.

## What's broken in `refresh_model.sh`

| Line | Expected path | Actual path on disk | Status |
| --- | --- | --- | --- |
| 14 | `$ALADIN_DIR/NQ_06-26.Last.txt` | `$ALADIN_DIR/nt8/NQ_06-26.Last.txt` | Moved |
| 15 | `$ALADIN_DIR/AladinExport/NQ_06-26.Last.txt` | `$ALADIN_DIR/nt8/AladinExport/NQ_06-26.Last.txt` | Moved |
| 64 | `$ALADIN_DIR/import_nt8_nq.py` | `$ALADIN_DIR/nt8/import_nt8_nq.py` | Moved |
| 96 | `$ALADIN_DIR/train_mario_ai.py` | **does not exist anywhere** | Deleted |

`git status` in `~/Desktop/Aladin` confirms `import_nt8_nq.py`, `train_mario_ai.py`, and `retrain_weekly.py` are all tracked-but-deleted from the working tree (along with ~25 other files — `backfill_features.py`, `analyze_losses.py`, `backtest_*`, `mario_bot_*.json` etc).

The last successful run of `refresh_model.sh` (2026-04-17) was import-only — `--train` was NOT passed. Import succeeded (6,120 NQ bars, 2026-03-30 → 2026-04-03), but the log line `ℹ️  Training sarit` means the weekly-training path hasn't actually been exercised since the reorg.

## What training scripts DO exist now

The `train/` subdirectory contains six newer training scripts that appear to have replaced the old monolithic `train_mario_ai.py`:

```
train/train_quality_v6.py          (Apr 18 11:23)
train/train_quality_ny_v3.py       (Apr 18 11:23)
train/train_quality_ts_lon_v1.py   (Apr 18 12:41)
train/train_quality_ts_ny_v1.py    (Apr 18 12:48)
train/train_scorer_v4_1.py         (Apr 18 09:52)
train/train_mario_bot_open.py      (Apr 15 08:04)
```

Each one produces its own model artefact (e.g. `mario_quality_v6.json`, `mario_quality_ny_v3.json`, `mario_quality_ts_ny_v1.json`, `ict_setup_scorer_v4_1.pkl` — all present in Aladin root, dated Apr 18). So the current training workflow seems to be "run each `train/train_*.py` individually" rather than one `train_mario_ai.py` entry point.

I did NOT run any of these — the task file pins the exact command `refresh_model.sh --train`, and I don't have the context to know which of the six scripts constitute "the weekly retrain" or what order they should run in.

## Additional blocker even if paths were fixed

The scheduled task runs in a Linux sandbox, not on your Mac directly. Two things would still fail from here:

1. `xgboost` and `lightgbm` are not installed in the sandbox Python (`pandas` and `numpy` are). The training scripts import both.
2. `pkill -f "bridge_api.py"` and `bash start.sh` operate on the sandbox process table, not your Mac — the real `bridge_api.py` running on your Mac wouldn't get paused or restarted.

So weekly retraining needs to either run on your Mac directly (e.g. launchd/cron on macOS rather than a Cowork scheduled task), or the sandbox needs xgboost/lightgbm installed and the script needs to understand it's not the one managing bridge lifecycle.

## Recommended next steps (for you to decide)

1. Rewrite `refresh_model.sh` to point at the moved paths (`nt8/` prefix) and replace the `train_mario_ai.py` call with the actual set of `train/train_*.py` scripts you want run weekly.
2. Decide whether the scheduled task should run on your Mac (launchd) instead of via Cowork — the bridge-pause/restart logic only works locally.
3. If you want me to wire this up, tell me which subset of `train/train_*.py` scripts makes up "weekly retrain" and whether this should stay as a Cowork scheduled task or move to launchd.

No model files were modified. No training ran.
