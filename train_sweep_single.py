"""Train a single regime from pre-built _v3 parquets. Usage: python3 train_sweep_single.py ALL [n_trials]"""
import sys, os
sys.path.insert(0, '/sessions/dreamy-youthful-turing/mnt/Aladin')

# Override constants before importing
import train_sweep_v2 as tsv
regime_arg  = sys.argv[1] if len(sys.argv) > 1 else 'ALL'
n_trials    = int(sys.argv[2]) if len(sys.argv) > 2 else 80

tsv.OPTUNA_TRIALS  = n_trials
tsv.ACTIVE_REGIMES = [regime_arg]
tsv.log.info(f"Single-regime run: {regime_arg}, {n_trials} trials")
tsv.train_and_save()
