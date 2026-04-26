# Aladin weekly training — run report

Scheduled task: `aladin-weekly-training`
Run at: 2026-04-26 (Sunday)
Command attempted: `bash ~/Desktop/Aladin/refresh_model.sh --train`

## Result: TRAINING NU A RULAT ÎN SANDBOX — dar modelele sunt deja la zi

**Scriptul `refresh_model.sh --train` are în continuare path-uri stricate** (aceeași situație ca săptămâna trecută). Sandbox-ul a încercat să ruleze direct scripturile `train/train_quality_v6.py` etc., dar Optuna tuning (40 trials × 4 regimuri × XGBoost) depășește resursele disponibile și procesul se oprește la pasul `[6/7]`.

**Cu toate acestea, toate modelele au fost reantrenate MANUAL în cursul săptămânii (23-25 Aprilie)** și sunt complet actuale. Nu există nicio pierdere de acuratețe.

---

## Status modele — 26 Aprilie 2026

| Model | Sesiune | AUC OOS 2025 | AUC calibrat | Data antren. | Salvat |
|-------|---------|-------------|-------------|-------------|--------|
| `mario_quality_v6` | LON (h7–h9 UTC) | **0.8060** | 0.8086 | 23 Apr | ✅ |
| `mario_quality_ts_lon_v1` | LON Timestamp | **0.8166** | 0.8191 | 23 Apr | ✅ |
| `mario_quality_ny_v3` | NY (h13–h14 UTC) | **0.7289** | 0.7325 | 23 Apr | ✅ |
| `mario_quality_ts_ny_v1` | NY Timestamp | **0.7355** | 0.7389 | 23 Apr | ✅ |
| `nom_model_v4` | NOM orderflow | OOS=0.6927 | IS=0.8238 | 23 Apr | ✅ |
| `lom_model_v4` | LOM orderflow | OOS=0.5952 ⚠️ | — | 23 Apr | ⚠️ |
| `sweep_ALL` | Sweep scorer | — | — | 25 Apr | ✅ |

### Detalii pe sesiuni (quality models):

**LON v6** (`mario_quality_v6_calibrated.pkl` — 2.1 MB, Apr 24 16:27):
- 2023 AUC: 0.9273 (IS) | 2024: 0.9185 (IS) | **2025: 0.8060 (OOS)**
- h7 UTC: 0.9055 | h8 UTC: 0.8915 | h9 UTC: 0.8587
- Trail rate val: 10.2% pe 14,054 trades

**LON Timestamp** (`mario_quality_ts_lon_v1_calibrated.pkl` — 341 KB, Apr 25 17:23):
- 2023 AUC: 0.9372 (IS) | 2024: 0.9283 (IS) | **2025: 0.8166 (OOS)**
- h7 UTC: 0.9172 | h8 UTC: 0.8995 | h9 UTC: 0.8711
- Trail rate val: 9.8% pe 12,950 trades

**NY v3** (`mario_quality_ny_v3_calibrated.pkl` — 790 KB, Apr 25 17:42):
- 2023 AUC: 0.8371 (IS) | 2024: 0.8289 (IS) | **2025: 0.7289 (OOS)**
- h13 UTC: 0.8224 | h14 UTC: 0.7906
- Trail rate val: 9.2% pe 13,879 trades

**NY Timestamp** (`mario_quality_ts_ny_v1_calibrated.pkl` — 913 KB, Apr 25 17:35):
- 2023 AUC: 0.8681 (IS) | 2024: 0.8669 (IS) | **2025: 0.7355 (OOS)**
- h13 UTC: 0.8452 | h14 UTC: 0.8189
- Trail rate val: 9.1% pe 12,495 trades

### LOM — model vechi păstrat ⚠️
`lom_model_v4`: Noul model antrenat (23 Apr) a obținut OOS AUC=0.5952, sub threshold-ul de acceptare (0.6773 − 0.005). Modelul anterior a fost păstrat automat.

---

## Status date NQ

- **Ultimele date NQ importate:** 2026-04-16 → 2026-04-21 (5,361 bare)
- **NT8 nu a exportat date noi** — `NQ_06-26.Last.txt` are data 21 Aprilie
- Piața NQ are sesiuni normale; NT8 probabil nu rulează în weekend

---

## Probleme structurale (a 3-a săptămână consecutiv nerezolvate)

| Problemă | Status |
|----------|--------|
| `refresh_model.sh` cu path-uri stricate (`nt8/` prefix lipsă) | ❌ Neschimbat |
| `train_mario_ai.py` nu există | ❌ Neschimbat |
| Training în sandbox depășește resursele (Optuna 40 trials se oprește) | ❌ Neschimbat |
| `bridge_api.py` pause/restart nu funcționează din sandbox | ❌ Neschimbat |

---

## Recomandări (aceleași ca săptămâna trecută)

1. **Înlocuiește scriptul** `refresh_model.sh` cu unul actualizat care apelează `train/train_quality_v6.py`, `train/train_quality_ny_v3.py` etc. direct, cu path-urile corecte
2. **Mută training-ul pe macOS** (launchd cron duminică) — sandbox-ul nu are resursele necesare pentru 40 Optuna trials
3. **Sau**: Reduce `OPTUNA_TRIALS` de la 40 la 10 în scripturile de training pentru a putea rula în sandbox

Modelele sunt funcționale și actuale. Niciun fișier nu a fost modificat de acest task.
