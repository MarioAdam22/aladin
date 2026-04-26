# ALADIN QUANTUM-ICT — Changelog & Roadmap

## v6.8 — 2026-03-13 (Current)

### 🔴 Critical Fixes
- **Fix #1**: XGBoost feature mismatch rezolvat — 5 features lipsă adăugate (`slope_h1`, `slope_h4`, `momentum_15`, `body_dir`, `wick_ratio`)
- **Fix Orderflow**: Ponderea orderflow redusă 40%→25%, logica ICT sweep corectată (v6.7)
- **Fix bug**: `check_max_drawdown_breaker()` keyword argument corectat

### 🟡 Model Improvements
- **#9**: SMOTE pentru echilibrare clase (imbalanced-learn)
- **#10**: Feature importance analysis — top 15 features afișate după antrenare
- **#11**: LightGBM antrenat în paralel cu XGBoost
- **#12**: Calibrare probabilități Platt Scaling
- **#13**: Ensemble model (XGBoost + LightGBM + RandomForest)
- **#14**: Online learning — funcție `online_learning_weekly()` pentru reantrenare automată

### 🟡 Backtest Robust
- **#4**: Transaction costs în backtest ($0.50 comision + 0.05% slippage)
- **#5**: Walk-forward testing (Q3/Q4 per an)
- **#6**: Out-of-sample testing 2024 (date nevăzute de model)
- **#7**: Monte Carlo simulation (1000 scenarii)
- **#8**: Metrici profesionale: Sharpe, Sortino, Calmar, Max Consecutive Loss

### 🟡 Semnale Avansate
- **#15**: Volatility compression filter (ATR percentile >80% → skip)
- **#16**: Volume trend boost (+0.1 conviction pentru volume spike)
- **#17**: VIX filter (VIX>25 → sizing 50%, VIX<15 → sizing 125%)
- **#18**: FinBERT sentiment analysis pe știri financiare
- **#43**: Synthetic DXY calculator din 6 perechi forex
- **#53**: Options flow Put/Call ratio sentiment
- **#54**: FRED API macro filter (Fed Rate, CPI, Unemployment)

### 🔵 Infrastructure
- **#19**: Git + .gitignore
- **#20**: FastAPI REST endpoint (`api.py`)
- **#21**: MLflow experiment tracking (`mlflow_tracker.py`)
- **#22**: Telegram alerts pentru semnale DIAMOND
- **#36**: Email alerts via SMTP (`email_alerts.py`)
- **#38**: Terms of Service & Disclaimer
- **#39**: Security config (bcrypt, env vars, secrets management)
- **#40**: GDPR Privacy Policy

### 🔴 Risk Management
- **#44**: Order Management System (`oms.py`)
- **#45**: Partial fills handling
- **#46**: Execution slippage monitoring (alertă >0.10%)
- **#47**: Portfolio heat monitoring (max 5% total risk)
- **#48**: Correlation filter (blochează NQ+ES simultan)
- **#49**: Daily loss circuit breaker (-3%/zi → stop)
- **#50**: Max drawdown circuit breaker (-15% → stop)
- **#51**: Kelly Criterion position sizing (Quarter Kelly)

### 🟢 Analytics Premium
- **#27**: Audit trail complet per semnal în UI
- **#28**: Monthly PDF report
- **#33**: Backtester interactiv extins (per sesiune, per regim)
- **#34**: Strategy builder (slidere ponderi în DASHBOARD)
- **#52**: Regime-specific models (`regime_models.py`)
- **#55-58**: Analytics premium: benchmark, attribution, drawdown, slippage (`analytics_premium.py`)

---

## 🗺️ Roadmap v7.0 — Planificat

### Necesită Conturi/Infrastructură Externă
- **#23**: Cloud deployment Oracle Cloud Free Tier
- **#24**: UptimeRobot monitoring
- **#25**: Dashboard React rebuild
- **#26**: Live track record public
- **#29**: JWT authentication per user
- **#30**: Multi-broker support (IBKR + Alpaca)
- **#31**: Risk settings per user
- **#32**: Stripe subscription management
- **#35**: Multi-instrument simultan (NQ+ES+XAU+BTC)
- **#37**: Mobile PWA
- **#41**: IBKR live data integration
- **#59**: Affiliate program
- **#60**: Free tier / 14 zile trial
- **#61**: Discord community

---

*Aladin Quantum-ICT Engine | Developer: Adam Mario | marioyear@yahoo.com*
