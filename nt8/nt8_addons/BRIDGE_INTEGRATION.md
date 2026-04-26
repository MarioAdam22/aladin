# AladinBridge Integration — Big Trades + Absorption v8.1

## Ce trebuie adăugat în AladinBridge.cs (NT8 addon existent)

### 1. În metoda care construiește JSON-ul pentru POST /nt8_data:

```csharp
// ── v8.1: Big Trades + Absorption ──
// Citește datele din AladinAbsorption indicator (shared static)

// Big Trades (ultimele 60 sec, ordine >= 50 contracts)
List<object> bigTrades = new List<object>();
lock (AladinAbsorption.DataLock)
{
    foreach (var bt in AladinAbsorption.RecentBigTrades)
    {
        bigTrades.Add(new {
            price = bt.Price,
            size = bt.Size,
            side = bt.Side,
            timestamp = bt.Timestamp
        });
    }
}

// Absorption Score (0-100, 0 = nu e detectat)
double absorptionScore = 0;
string absorptionSide = "";
lock (AladinAbsorption.DataLock)
{
    absorptionScore = AladinAbsorption.CurrentAbsorptionScore;
    absorptionSide = AladinAbsorption.CurrentAbsorptionSide;
}

// Adaugă în JSON payload:
// "big_trades": bigTrades,
// "absorption_score": absorptionScore,
// "absorption_side": absorptionSide
```

### 2. Exemplu JSON trimis la bridge:

```json
{
  "symbol": "NQ",
  "timestamp": "2026-04-01T14:30:00",
  "price": { "open": 19500, "high": 19520, "low": 19490, "close": 19515 },
  "orderflow": {
    "cum_delta": 350.0,
    "bar_buy_vol": 1200,
    "bar_sell_vol": 800,
    "imbalance_pct": 15.5
  },
  "big_trades": [
    {"price": 19510.25, "size": 75, "side": "BUY", "timestamp": "2026-04-01T14:29:45Z"},
    {"price": 19508.50, "size": 120, "side": "BUY", "timestamp": "2026-04-01T14:29:52Z"}
  ],
  "absorption_score": 72.5,
  "absorption_side": "BID"
}
```

## IMPORTANT — Fallback fără addon

Dacă AladinAbsorption NU este instalat pe chart:
- `big_trades` = [] (gol)
- `absorption_score` = 0
- `absorption_side` = ""

mario_rag.py funcționează oricum — calculează absorption din:
- bar_buy_vol / bar_sell_vol (delta absorption)
- bar volume vs average + body size (volume absorption proxy)

NT8 addon-ul oferă doar precizie mai mare cu date tick-by-tick.
