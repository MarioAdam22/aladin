-- ============================================================
-- ALADIN — Supabase Tables  (UPDATE #1)
-- Rulează în: Supabase Dashboard → SQL Editor → Run
-- ============================================================

-- ──────────────────────────────────────────────────────────
-- 1. TABELA SIGNALS — fiecare semnal generat de aladin_engine
-- ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS signals (
    id               BIGSERIAL PRIMARY KEY,
    ts               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol           TEXT        NOT NULL DEFAULT 'NQ',
    direction        TEXT        NOT NULL,          -- LONG / SHORT / WAIT
    score_pct        NUMERIC(6,2),
    ai_score         NUMERIC(6,2),
    verdict          TEXT,
    ict_component    NUMERIC(8,4),
    q_component      NUMERIC(8,4),
    sentiment_score  NUMERIC(8,4),
    sentiment_mult   NUMERIC(8,4),
    vix_mult         NUMERIC(8,4),
    macro_mult       NUMERIC(8,4),
    regime           TEXT,
    killzone         TEXT,
    live_mode        BOOLEAN      DEFAULT FALSE,
    raw_score        NUMERIC(8,4),
    extra            TEXT         -- JSON string cu date suplimentare
);

-- Index pentru queries rapide pe timp + symbol
CREATE INDEX IF NOT EXISTS idx_signals_ts     ON signals (ts DESC);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals (symbol, ts DESC);

-- Row Level Security — anon poate INSERT + SELECT
ALTER TABLE signals ENABLE ROW LEVEL SECURITY;
CREATE POLICY "anon_insert_signals" ON signals FOR INSERT TO anon WITH CHECK (true);
CREATE POLICY "anon_select_signals" ON signals FOR SELECT TO anon USING (true);


-- ──────────────────────────────────────────────────────────
-- 2. TABELA TRADES — trade-uri deschise + închise
-- ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    id           BIGSERIAL PRIMARY KEY,
    ts_open      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ts_close     TIMESTAMPTZ,
    symbol       TEXT        NOT NULL DEFAULT 'NQ',
    direction    TEXT        NOT NULL,          -- LONG / SHORT
    score_pct    NUMERIC(6,2),
    ai_score     NUMERIC(6,2),
    entry_price  NUMERIC(12,2),
    exit_price   NUMERIC(12,2),
    sl_price     NUMERIC(12,2),
    tp_price     NUMERIC(12,2),
    qty          NUMERIC(8,2) DEFAULT 1,
    risk_usd     NUMERIC(10,2),
    pnl          NUMERIC(10,2) DEFAULT 0,
    status       TEXT         DEFAULT 'OPEN',  -- OPEN / CLOSED / SL_HIT / TP_HIT / MANUAL
    live_mode    BOOLEAN      DEFAULT FALSE,
    note         TEXT
);

-- Index
CREATE INDEX IF NOT EXISTS idx_trades_ts     ON trades (ts_open DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol, ts_open DESC);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status);

-- Row Level Security
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
CREATE POLICY "anon_insert_trades" ON trades FOR INSERT TO anon WITH CHECK (true);
CREATE POLICY "anon_select_trades" ON trades FOR SELECT TO anon USING (true);
CREATE POLICY "anon_update_trades" ON trades FOR UPDATE TO anon USING (true);


-- ──────────────────────────────────────────────────────────
-- 3. VIEW STATISTICI — win rate, PnL etc.
-- ──────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW trade_stats AS
SELECT
    symbol,
    COUNT(*)                                            AS total_trades,
    COUNT(*) FILTER (WHERE pnl > 0)                    AS wins,
    COUNT(*) FILTER (WHERE pnl <= 0)                   AS losses,
    ROUND(COUNT(*) FILTER (WHERE pnl > 0)::NUMERIC
          / NULLIF(COUNT(*), 0) * 100, 1)               AS win_rate_pct,
    ROUND(SUM(pnl), 2)                                  AS total_pnl,
    ROUND(AVG(pnl), 2)                                  AS avg_pnl,
    MAX(ts_open)                                        AS last_trade_ts
FROM trades
WHERE status != 'OPEN'
GROUP BY symbol;

-- ──────────────────────────────────────────────────────────
-- DONE — 2 tabele + 1 view create cu succes!
-- ──────────────────────────────────────────────────────────
