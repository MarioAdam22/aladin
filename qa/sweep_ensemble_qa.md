# Sweep Ensemble QA — 2026-04-25 16:23

**OOS Period:** 2025+  |  **Threshold:** 0.65  |  **Dataset:** sweep_dataset_2022_2023_2024_2025_v3.parquet

**Old:** `sweep_REGIME.pkl` (single model)  
**New:** `sweep_REGIME_ensemble.pkl` (3-seed ensemble)  

## Per-Regime Results

| Regime | n_oos | base_wr | n_sig_old | hr_old | auc_old | n_sig_ens | hr_ens | auc_ens | Δhit_rate | Δn_signals |
|--------|-------|---------|-----------|--------|---------|-----------|--------|---------|-----------|------------|
| ALL             |   997 |    0.46 |       186 |  0.699 |  0.6468 |        80 |  0.825 |  0.6569 |    +0.126 |       -106 |
| PRE_EXPANSION   |   217 |   0.516 |        96 |  0.583 |  0.5595 |        70 |    0.7 |  0.7072 |    +0.117 |        -26 |
| EXPANSION       |   151 |   0.543 |         0 |    N/A |  0.5936 |        25 |   0.84 |   0.707 |       N/A |         25 |
| RETRACEMENT     |   148 |   0.439 |         0 |    N/A |  0.5245 |         8 |  0.875 |  0.6105 |       N/A |          8 |
| CONSOLIDATION   |   432 |   0.421 |       182 |  0.571 |  0.6391 |        63 |   0.73 |  0.6838 |    +0.159 |       -119 |

## Score Distribution (median / p75)

| Regime | old_p50 | old_p75 | ens_p50 | ens_p75 |
|--------|---------|---------|---------|----------|
| ALL             |   0.448 |   0.448 |   0.461 |    0.533 |
| PRE_EXPANSION   |     0.5 |   0.667 |   0.485 |    0.679 |
| EXPANSION       |   0.562 |   0.583 |   0.455 |     0.58 |
| RETRACEMENT     |     0.1 |   0.214 |   0.246 |    0.442 |
| CONSOLIDATION   |   0.552 |     0.7 |   0.451 |    0.559 |

## Session Breakdown (ALL regime model)

| Session | n_oos | base_wr | n_sig_old | hr_old | n_sig_ens | hr_ens | Δhit_rate | auc_old | auc_ens |
|---------|-------|---------|-----------|--------|-----------|--------|-----------|---------|----------|
| LON     |   345 |   0.435 |        24 |  0.708 |         3 |    1.0 |    +0.292 |  0.6238 |   0.6482 |
| NY      |   652 |   0.474 |       162 |  0.698 |        77 |  0.818 |    +0.121 |  0.6629 |   0.6686 |

## Summary

**ALL regime (full OOS):**  
- Old model: 186 signals @ 0.65, hit_rate = 0.699, AUC = 0.6468  
- Ensemble:  80 signals @ 0.65, hit_rate = 0.825, AUC = 0.6569  
- Δ hit_rate = **+0.126** | Δ signals = -106  

**Best improvement:** CONSOLIDATION (+0.159 hit_rate)  
**Worst:** PRE_EXPANSION (+0.117 hit_rate)  

*Generated: 2026-04-25 16:23*
