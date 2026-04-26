// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  ALADIN BRIDGE v2.4 — NinjaTrader 8 Strategy                               ║
// ║  Trimite OHLCV + OrderFlow complet către Mac (port 8000)                    ║
// ║  Ascultă comenzi BUY/SELL/CLOSE de la Mac (port 8002)                      ║
// ║                                                                              ║
// ║  OrderFlow transmis:                                                        ║
// ║    • Volume Profile: POC, VAH, VAL (sesiune curentă)                        ║
// ║    • Cumulative Delta: cumpărători vs vânzători (calculat din tick data)    ║
// ║    • DOM Snapshot: top 5 niveluri Bid + Ask cu volume                       ║
// ║    • Footprint Imbalances: bara curentă buy/sell imbalance %                ║
// ║    • Session Stats: VWAP, total volume sesiune, tick count                  ║
// ║    • Bar History: ultimele 20 bare OHLCV + delta complete (AI context)      ║
// ║    • ATR(14): Average True Range nativ NT8                                  ║
// ║    • HTF Bias: H4 + H1 OHLC real (High/Low pentru bias filter)             ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

#region Using declarations
using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.DrawingTools;
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class AladinBridge : Strategy
    {
        // ── Configurare ──────────────────────────────────────────────────────
        private string MacIP        = "172.16.233.1";    // IP Mac pe rețeaua VMware NAT — fix, nu se schimbă la restart
        private int    MacPort      = 8000;              // FastAPI server (bridge_api.py)
        private int    ListenPort   = 8002;              // HttpListener comenzi execuție

        // ── HTTP Client (singleton — nu recrea pe fiecare tick) ──────────────
        // IMPORTANT: static = o singură instanță pentru procesul NT8
        // NU se dispune niciodată (singleton pe durata procesului)
        private static readonly HttpClient _http = new HttpClient
        {
            Timeout = TimeSpan.FromMilliseconds(15000)   // 15s — analiza mario_rag durează 5-10s
        };
        private static bool _httpInitialized = false;

        // ── Execution Listener ───────────────────────────────────────────────
        private HttpListener  _listener;
        private Thread        _listenerThread;
        private volatile bool _listenerRunning = false;

        // ── OrderFlow State ──────────────────────────────────────────────────
        private double _cumDelta       = 0.0;   // Cumulative Delta sesiune
        private double _sessionVolume  = 0.0;   // Volume total sesiune
        private double _vwapNumerator  = 0.0;   // Σ(price × volume) pentru VWAP
        private long   _tickCount      = 0;     // Număr tick-uri sesiune
        private double _barBuyVolume   = 0.0;   // Buy volume bara curentă
        private double _barSellVolume  = 0.0;   // Sell volume bara curentă
        private double _lastPrice      = 0.0;

        // ── Big Trades + Tape Speed (native, fără AladinAbsorption) ──────────
        // BIG_TRADE_THRESHOLD: 20 contracte NQ = footprint instituțional minim
        // La 50+ = bloc instituțional clar. Ajustează după lichiditate zilnică.
        private const double BIG_TRADE_THRESHOLD = 20.0;
        private int    _barBigBuyCount  = 0;     // nr trades >= 20c pe buy în bara curentă
        private int    _barBigSellCount = 0;     // nr trades >= 20c pe sell în bara curentă
        private double _barMaxTradeSize = 0.0;   // cel mai mare trade individual în bara curentă
        private long   _barTickCount    = 0;     // tick-uri în bara curentă (≠ _tickCount sesiune)
        private DateTime _barOpenTime   = DateTime.UtcNow;  // start bară pentru calcul tape speed

        // ── Delta la High/Low barei ──────────────────────────────────────────
        // Captează delta cumulativă în momentul în care prețul atinge High/Low bara
        // Delta negativă la High = vânzători absorb exact acolo = BSL sweep, nu breakout real
        // Delta pozitivă la Low  = cumpărători absorb exact acolo = retracement, nu reversal
        private double _deltaAtHigh     = 0.0;   // cum_delta când price == barHigh
        private double _deltaAtLow      = 0.0;   // cum_delta când price == barLow
        private double _currentBarHigh  = double.MinValue;  // high curent pentru tracking
        private double _currentBarLow   = double.MaxValue;  // low curent pentru tracking
        private double _lastBid        = 0.0;
        private double _lastAsk        = 0.0;

        // ── VWAP Standard Deviation (σ) — Institutional Reference Bands ─────────
        // Formula numerically-stable: Var = E[X²] − E[X]² = Σ(vol·price²)/Σvol − VWAP²
        // ±1σ = mean-reversion zone; ±2σ = extremă statistică (overextension)
        private double _vwapSumPriceSq = 0.0;  // Σ(volume × price²) — reset la sesiune nouă

        // ── OF Consolidation Metrics (native NT8, trimise în payload) ────────────
        // Calculăm direct din datele OF tick-by-tick — mult mai precis decât recalculat din cache
        // 1. Delta Oscillation Index: autocorrelation lag-1 pe bar_delta history
        //    Negativ = mean-reversion (consolidare), Pozitiv = momentum (trend)
        // 2. Bilateral Absorption: absorption la AMBELE extreme (high+low) = range instituțional
        // 3. Big Trade Balance: buy/(buy+sell) pe big trades; 0.35-0.65 = echilibrat = consolidare
        // 4. Profile Shape D-count: consecutive D-shapes = balanced VP = consolidare
        private Queue<double> _barDeltaHistory   = new Queue<double>();   // ultimele 25 bar_delta
        private const int     BarDeltaHistSize   = 25;
        private Queue<double> _dahHistory        = new Queue<double>();   // delta_at_high per bară
        private Queue<double> _dalHistory        = new Queue<double>();   // delta_at_low per bară
        private const int     DahDalHistSize     = 10;
        private Queue<string> _shapeHistory      = new Queue<string>();   // P/b/D per bară
        private const int     ShapeHistSize      = 10;

        // Metricile calculate (actualizate la fiecare bar close)
        private double _ofConsolDeltaOscIdx    = 0.0;    // [-1, +1]
        private bool   _ofConsolBilateralAbsorb = false;
        private double _ofConsolBigTradeBalance = 0.5;   // [0, 1]
        private int    _ofConsolDShapeCount     = 0;      // consecutive D-shapes
        private string _currentProfileShape     = "D";    // P/b/D — trimis în payload

        // ── Previous Session Volume Profile Levels ───────────────────────────────
        // POC/VAH/VAL sesiunea anterioară = niveluri instituționale primare pentru ziua curentă
        // Stocate înainte de reset VP la sesiune nouă
        private double _prevSessionPoc = 0.0;
        private double _prevSessionVah = 0.0;
        private double _prevSessionVal = 0.0;

        // ── Volume Profile (calculat manual din bara curentă) ────────────────
        // Stocăm distribuția volumului pe niveluri de preț (tick size granularitate)
        private Dictionary<double, double> _vpBuyMap  = new Dictionary<double, double>();
        private Dictionary<double, double> _vpSellMap = new Dictionary<double, double>();

        // ── DOM Snapshot ─────────────────────────────────────────────────────
        private SortedDictionary<double, long> _domBids = new SortedDictionary<double, long>(new DescendingComparer());
        private SortedDictionary<double, long> _domAsks = new SortedDictionary<double, long>();
        private readonly object _domLock = new object();

        // ── Bar History — ultimele 20 bare complete pentru AI context ─────────
        private Queue<string> _barHistory    = new Queue<string>();
        private const int     BarHistorySize = 20;

        // ── Historical Orderflow — date per-bară pentru filtre elite ──────────
        // POC History: urmărim migrarea POC-ului (POC rising = bullish pressure)
        private Queue<double> _pocHistory      = new Queue<double>();
        private const int     PocHistorySize   = 10;   // ultimele 10 bare
        // DOM Ratio History: urmărim trendul bid/ask ratio (spike vs trend)
        private Queue<double> _domRatioHistory = new Queue<double>();
        private const int     DomRatioHistSize = 20;   // ultimele 20 bare

        // ── Throttle (nu trimitem pe fiecare tick, ci la interval) ───────────
        private DateTime _lastSend   = DateTime.MinValue;
        private int      SendIntervalMs = 250;  // 4 pachete/secundă max

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Aladin Bridge v2.2 — OrderFlow + Execution + SL/TP + OnPositionUpdate";
                Name        = "AladinBridge";
                Calculate   = Calculate.OnEachTick;   // Tick-by-tick pentru CumDelta
                IsOverlay   = false;
                BarsRequiredToTrade = 20;
            }
            else if (State == State.Configure)
            {
                // Activează Market Depth pentru DOM
                AddDataSeries(BarsPeriodType.Tick, 1);
                // ── HTF Bias: H4 (index 2) și H1 (index 3) ─────────────────
                // Folosit de mario_rag pentru filtrele instituționale HTF
                // H4 high/low = bias major, H1 high/low = bias intraday
                AddDataSeries(BarsPeriodType.Minute, 240);  // H4  → BarsArray[2]
                AddDataSeries(BarsPeriodType.Minute, 60);   // H1  → BarsArray[3]
                AddDataSeries(BarsPeriodType.Minute, 15);   // M15 → BarsArray[4]
            }
            else if (State == State.Realtime)
            {
                StartExecutionListener();
                Print($"[ALADIN] Bridge pornit → Mac: {MacIP}:{MacPort}  Listen: {ListenPort}");
                Print($"[ALADIN] HttpClient ready: {!_http.Equals(null)}  BaseAddr: {_http.BaseAddress?.ToString() ?? "none"}");
                _httpInitialized = true;
                // Test imediat de conectivitate
                _ = PostAsync($"http://{MacIP}:{MacPort}/ping_nt8", $"{{\"status\":\"hello\",\"strategy\":\"AladinBridge\",\"ts\":\"{DateTime.UtcNow:O}\"}}");
            }
            else if (State == State.Terminated)
            {
                StopExecutionListener();
                // NU dispune _http — e static, trăiește pe durata procesului NT8
                // _http.Dispose() ← BUGFIX: această linie distrugea HttpClient-ul pentru instanțele viitoare
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // MARKET DATA — fiecare tick: calculăm CumDelta, VWAP, VP
        // ════════════════════════════════════════════════════════════════════
        protected override void OnMarketData(MarketDataEventArgs e)
        {
            if (e.MarketDataType == MarketDataType.Last && e.Volume > 0)
            {
                double price  = e.Price;
                double vol    = e.Volume;
                double bid    = _lastBid > 0 ? _lastBid : (price - TickSize);
                double ask    = _lastAsk > 0 ? _lastAsk : (price + TickSize);

                // Cumulative Delta: up-tick = BUY, down-tick = SELL
                if (price >= ask)
                {
                    _cumDelta      += vol;
                    _barBuyVolume  += vol;
                    // Big trade native tracking
                    if (vol >= BIG_TRADE_THRESHOLD) _barBigBuyCount++;
                }
                else if (price <= bid)
                {
                    _cumDelta      -= vol;
                    _barSellVolume += vol;
                    // Big trade native tracking
                    if (vol >= BIG_TRADE_THRESHOLD) _barBigSellCount++;
                }

                // Max trade size și per-bar tick count
                if (vol > _barMaxTradeSize) _barMaxTradeSize = vol;
                _barTickCount++;

                // Delta la High/Low — captează delta în momentul atingerii extremei
                if (price > _currentBarHigh)
                {
                    _currentBarHigh = price;
                    _deltaAtHigh    = _cumDelta;   // snapshot delta la noul High
                }
                if (price < _currentBarLow)
                {
                    _currentBarLow = price;
                    _deltaAtLow    = _cumDelta;    // snapshot delta la noul Low
                }

                // VWAP + Variance accumulator
                _vwapNumerator  += price * vol;
                _vwapSumPriceSq += price * price * vol;   // Σ(vol·price²) pentru σ
                _sessionVolume  += vol;
                _tickCount++;

                // Volume Profile (niveluri de preț rotunjite la TickSize)
                double level = Math.Round(price / TickSize) * TickSize;
                if (price >= ask)
                {
                    if (!_vpBuyMap.ContainsKey(level))  _vpBuyMap[level]  = 0;
                    _vpBuyMap[level] += vol;
                }
                else if (price <= bid)
                {
                    if (!_vpSellMap.ContainsKey(level)) _vpSellMap[level] = 0;
                    _vpSellMap[level] += vol;
                }

                _lastPrice = price;
            }
            else if (e.MarketDataType == MarketDataType.Bid)
            {
                _lastBid = e.Price;
            }
            else if (e.MarketDataType == MarketDataType.Ask)
            {
                _lastAsk = e.Price;
            }

            // Throttle — trimite la fiecare SendIntervalMs ms
            if ((DateTime.Now - _lastSend).TotalMilliseconds >= SendIntervalMs)
            {
                SendMarketData();
                _lastSend = DateTime.Now;
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // DOM — Market Depth (Bid/Ask levels)
        // ════════════════════════════════════════════════════════════════════
        protected override void OnMarketDepth(MarketDepthEventArgs e)
        {
            lock (_domLock)
            {
                if (e.MarketDataType == MarketDataType.Bid)
                {
                    if (e.Operation == Operation.Add || e.Operation == Operation.Update)
                        _domBids[e.Price] = e.Volume;
                    else if (e.Operation == Operation.Remove)
                        _domBids.Remove(e.Price);
                }
                else if (e.MarketDataType == MarketDataType.Ask)
                {
                    if (e.Operation == Operation.Add || e.Operation == Operation.Update)
                        _domAsks[e.Price] = e.Volume;
                    else if (e.Operation == Operation.Remove)
                        _domAsks.Remove(e.Price);
                }
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // BAR UPDATE — reset variabile intra-bar + resetăm VP la deschiderea zilei
        // ════════════════════════════════════════════════════════════════════
        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0) return;

            // ── CAPTURĂM date bara completată ÎNAINTE de orice reset ────────────
            // La OnBarUpdate, _barBuyVolume/_barSellVolume conțin datele barei
            // care tocmai s-a închis — după reset ele devin 0 pentru bara nouă.
            double completedBarDelta = _barBuyVolume - _barSellVolume;

            // ── Salvează bara COMPLETĂ în bar history (bara anterioară = [1]) ──
            if (CurrentBar > 0 && State == State.Realtime)
            {
                try
                {
                    double prevAtr = ATR(14)[1];

                    // ── POC History: salvăm POC-ul barei ce se închide ──────────
                    // Urmărim dacă POC-ul migrează (up = bullish pressure, down = bearish)
                    double pocNow = 0, vahNow = 0, valNow = 0;
                    ComputeVolumeProfile(out pocNow, out vahNow, out valNow);
                    if (pocNow > 0)
                    {
                        _pocHistory.Enqueue(pocNow);
                        if (_pocHistory.Count > PocHistorySize)
                            _pocHistory.Dequeue();
                    }

                    // ── DOM Ratio History: salvăm bid/ask ratio per bară ────────
                    // Urmărim dacă presiunea instituțională crește sau scade
                    long bidTot = 0, askTot = 0;
                    lock (_domLock)
                    {
                        foreach (var kv in _domBids) bidTot += kv.Value;
                        foreach (var kv in _domAsks) askTot += kv.Value;
                    }
                    double domRatioNow = askTot > 0 ? (double)bidTot / askTot : 1.0;
                    _domRatioHistory.Enqueue(domRatioNow);
                    if (_domRatioHistory.Count > DomRatioHistSize)
                        _domRatioHistory.Dequeue();

                    // ── Bar History JSON — include delta per bară (câmp "d") ────
                    // "d" = buy_vol - sell_vol barei completate
                    // Folosit pentru Delta Divergence filter (preț sus dar delta jos = distribuție)
                    string barJson = string.Format(
                        System.Globalization.CultureInfo.InvariantCulture,
                        "{{\"o\":{0:F2},\"h\":{1:F2},\"l\":{2:F2},\"c\":{3:F2},\"v\":{4:F0},\"atr\":{5:F4},\"d\":{6:F0},\"t\":\"{7:yyyy-MM-ddTHH:mm:ssZ}\"}}",
                        SafeD(Open[1]), SafeD(High[1]), SafeD(Low[1]), SafeD(Close[1]), SafeD(Volume[1]), SafeD(prevAtr),
                        SafeD(completedBarDelta),
                        Time[1].ToUniversalTime()
                    );
                    _barHistory.Enqueue(barJson);
                    if (_barHistory.Count > BarHistorySize)
                        _barHistory.Dequeue();

                    // ══════════════════════════════════════════════════════════════
                    // OF CONSOLIDATION METRICS — calculate nativ la fiecare bar close
                    // Trimise în payload, bridge-ul le citește direct fără recalculare
                    // ══════════════════════════════════════════════════════════════

                    // ── Salvăm bar_delta, delta_at_high/low, shape în history ──
                    _barDeltaHistory.Enqueue(completedBarDelta);
                    if (_barDeltaHistory.Count > BarDeltaHistSize)
                        _barDeltaHistory.Dequeue();

                    _dahHistory.Enqueue(_deltaAtHigh);
                    if (_dahHistory.Count > DahDalHistSize)
                        _dahHistory.Dequeue();

                    _dalHistory.Enqueue(_deltaAtLow);
                    if (_dalHistory.Count > DahDalHistSize)
                        _dalHistory.Dequeue();

                    // Profile shape: P (bullish), b (bearish), D (balanced/consolidare)
                    string barShape = "D";
                    if (pocNow > 0)
                    {
                        double range = SafeD(High[1]) - SafeD(Low[1]);
                        if (range > 0)
                        {
                            double pocPos = (pocNow - SafeD(Low[1])) / range;
                            if (pocPos > 0.66)      barShape = "P";  // POC în top 1/3 = bullish
                            else if (pocPos < 0.33)  barShape = "b";  // POC în bottom 1/3 = bearish
                            // else D = balanced
                        }
                    }
                    _currentProfileShape = barShape;   // salvăm pentru JSON payload
                    _shapeHistory.Enqueue(barShape);
                    if (_shapeHistory.Count > ShapeHistSize)
                        _shapeHistory.Dequeue();

                    // ── 1. DELTA OSCILLATION INDEX (autocorrelation lag-1) ──────
                    // Măsoară dacă delta oscilează (mean-revert → consolidare) sau persistă (trend)
                    // Negativ = mean-reversion, Pozitiv = momentum
                    _ofConsolDeltaOscIdx = 0.0;
                    if (_barDeltaHistory.Count >= 10)
                    {
                        double[] deltas = new double[_barDeltaHistory.Count];
                        _barDeltaHistory.CopyTo(deltas, 0);
                        int n = deltas.Length;
                        double mean = 0;
                        for (int i = 0; i < n; i++) mean += deltas[i];
                        mean /= n;

                        double num = 0, den = 0;
                        for (int i = 1; i < n; i++)
                            num += (deltas[i] - mean) * (deltas[i - 1] - mean);
                        for (int i = 0; i < n; i++)
                            den += (deltas[i] - mean) * (deltas[i] - mean);

                        _ofConsolDeltaOscIdx = den > 0 ? num / den : 0.0;
                        // Clamp la [-1, 1]
                        _ofConsolDeltaOscIdx = Math.Max(-1.0, Math.Min(1.0, _ofConsolDeltaOscIdx));
                    }

                    // ── 2. BILATERAL ABSORPTION ────────────────────────────────
                    // Absorption la AMBELE extreme = instituționalii joacă range-ul
                    // delta_at_high < -30 (vânzători absorb la highs) ȘI delta_at_low > 30 (cumpărători absorb la lows)
                    _ofConsolBilateralAbsorb = false;
                    if (_dahHistory.Count >= 3 && _dalHistory.Count >= 3)
                    {
                        int absHighCount = 0, absLowCount = 0;
                        double[] dahArr = new double[_dahHistory.Count];
                        double[] dalArr = new double[_dalHistory.Count];
                        _dahHistory.CopyTo(dahArr, 0);
                        _dalHistory.CopyTo(dalArr, 0);

                        // Verificăm ultimele 5 bare (sau câte avem)
                        int checkBars = Math.Min(5, Math.Min(dahArr.Length, dalArr.Length));
                        for (int i = dahArr.Length - checkBars; i < dahArr.Length; i++)
                        {
                            if (dahArr[i] < -30) absHighCount++;
                        }
                        for (int i = dalArr.Length - checkBars; i < dalArr.Length; i++)
                        {
                            if (dalArr[i] > 30)  absLowCount++;
                        }
                        // Bilateral = absorption la ambele capete în cel puțin 2 din ultimele 5 bare
                        _ofConsolBilateralAbsorb = (absHighCount >= 2 && absLowCount >= 2);
                    }

                    // ── 3. BIG TRADE BALANCE ───────────────────────────────────
                    // buy/(buy+sell) pe big trades bara completată
                    // 0.35-0.65 = echilibrat (consolidare), <0.30 sau >0.70 = direcțional (trend)
                    {
                        int totalBig = _barBigBuyCount + _barBigSellCount;
                        _ofConsolBigTradeBalance = totalBig > 0
                            ? (double)_barBigBuyCount / totalBig
                            : 0.5;
                    }

                    // ── 4. D-SHAPE CONSECUTIVE COUNT ───────────────────────────
                    // Numărăm D-shapes consecutive de la coadă (cele mai recente)
                    // 3+ consecutive = confirmare solidă de consolidare
                    _ofConsolDShapeCount = 0;
                    {
                        string[] shapes = new string[_shapeHistory.Count];
                        _shapeHistory.CopyTo(shapes, 0);
                        for (int i = shapes.Length - 1; i >= 0; i--)
                        {
                            if (shapes[i] == "D") _ofConsolDShapeCount++;
                            else break;
                        }
                    }
                }
                catch { /* ignoră dacă nu sunt suficiente bare */ }
            }

            // Reset sesiune la prima bară a zilei
            if (Bars.IsFirstBarOfSession)
            {
                // ── Salvăm nivelurile sesiunii anterioare ÎNAINTE de reset VP ──
                // Aceste niveluri sunt magistrale instituționale pentru ziua nouă
                double _sPoc, _sVah, _sVal;
                ComputeVolumeProfile(out _sPoc, out _sVah, out _sVal);
                if (_sPoc > 0)
                {
                    _prevSessionPoc = _sPoc;
                    _prevSessionVah = _sVah;
                    _prevSessionVal = _sVal;
                    Print($"[ALADIN] Prev session: POC={_prevSessionPoc:F2} VAH={_prevSessionVah:F2} VAL={_prevSessionVal:F2}");
                }

                _cumDelta       = 0;
                _sessionVolume  = 0;
                _vwapNumerator  = 0;
                _vwapSumPriceSq = 0;   // reset VWAP variance accumulator
                _tickCount      = 0;
                _vpBuyMap.Clear();
                _vpSellMap.Clear();
                // Reset OF consolidation histories la sesiune nouă
                _barDeltaHistory.Clear();
                _dahHistory.Clear();
                _dalHistory.Clear();
                _shapeHistory.Clear();
                _ofConsolDeltaOscIdx     = 0.0;
                _ofConsolBilateralAbsorb = false;
                _ofConsolBigTradeBalance = 0.5;
                _ofConsolDShapeCount     = 0;
                Print($"[ALADIN] Sesiune nouă — reset OrderFlow state. Bar history: {_barHistory.Count} bare");
            }

            // Reset la bara nouă (DUPĂ ce am salvat datele barei completate)
            _barBuyVolume    = 0;
            _barSellVolume   = 0;
            _barBigBuyCount  = 0;
            _barBigSellCount = 0;
            _barMaxTradeSize = 0;
            _barTickCount    = 0;
            _barOpenTime     = DateTime.UtcNow;
            _deltaAtHigh     = _cumDelta;
            _deltaAtLow      = _cumDelta;
            _currentBarHigh  = double.MinValue;
            _currentBarLow   = double.MaxValue;

            // Fallback: trimite date pe fiecare bară (garantat, chiar dacă OnMarketData nu primește ticks în DEMO)
            if (State == State.Realtime && CurrentBar > 20)
                SendMarketData();
        }

        // ════════════════════════════════════════════════════════════════════
        // SEND — construim JSON-ul complet și îl trimitem async
        // ════════════════════════════════════════════════════════════════════
        private void SendMarketData()
        {
            if (State != State.Realtime || CurrentBars[0] < BarsRequiredToTrade) return;

            try
            {
                // ── Price data ────────────────────────────────────────────
                double open   = Open[0];
                double high   = High[0];
                double low    = Low[0];
                double close  = Close[0];
                double volume = Volume[0];

                // ── VWAP ──────────────────────────────────────────────────
                double vwap = _sessionVolume > 0
                    ? _vwapNumerator / _sessionVolume
                    : close;

                // ── Volume Profile: POC, VAH, VAL ─────────────────────────
                double poc = 0, vah = 0, val = 0;
                ComputeVolumeProfile(out poc, out vah, out val);

                // ── Imbalance bara curentă ────────────────────────────────
                double totalBarVol = _barBuyVolume + _barSellVolume;
                double imbalancePct = totalBarVol > 0
                    ? (_barBuyVolume - _barSellVolume) / totalBarVol * 100.0
                    : 0.0;   // >0 = buy dominance, <0 = sell dominance

                // ── DOM snapshot (top 5 bid + ask) ────────────────────────
                string domJson = BuildDomJson(5);

                // ── Spread & Liquidity ────────────────────────────────────
                // Fix v10.5: bid/ask pot fi 0 la startup → spread fals = prețul întreg → 422 Pydantic
                double spread = (_lastBid > 0 && _lastAsk > 0)
                    ? Math.Max(0, _lastAsk - _lastBid)
                    : 0.0;
                long   domBidTotal = 0, domAskTotal = 0;
                lock (_domLock)
                {
                    foreach (var kv in _domBids) domBidTotal += kv.Value;
                    foreach (var kv in _domAsks) domAskTotal += kv.Value;
                }

                // ── ATR(14) nativ NT8 ─────────────────────────────────────
                double atr14 = 0;
                try { atr14 = ATR(14)[0]; } catch { }

                // ── VWAP Standard Deviation Bands (±1σ, ±2σ) ─────────────
                // Formula: σ² = Σ(vol·price²)/Σvol − VWAP²  (numerically stable)
                // ±1σ = zone de mean-reversion instituționale
                // ±2σ = extremă statistică (extins → probabilitate revenire >80%)
                double vwapSd = 0, vwapSd1Hi = 0, vwapSd1Lo = 0, vwapSd2Hi = 0, vwapSd2Lo = 0;
                if (_sessionVolume > 0 && _vwapSumPriceSq > 0)
                {
                    double vwapVar = (_vwapSumPriceSq / _sessionVolume) - (vwap * vwap);
                    vwapSd   = Math.Sqrt(Math.Max(vwapVar, 0));
                    vwapSd1Hi = vwap + vwapSd;
                    vwapSd1Lo = vwap - vwapSd;
                    vwapSd2Hi = vwap + 2.0 * vwapSd;
                    vwapSd2Lo = vwap - 2.0 * vwapSd;
                }

                // ── LVN / HVN Detection din Volume Profile curent ────────
                // HVN (High Volume Node): preț magnet — prețul stagnează, TP zone
                // LVN (Low Volume Node): zonă subțire — prețul traversează rapid, entry zone
                string hvnJson = "[]", lvnJson = "[]";
                try
                {
                    var totalVp = new Dictionary<double, double>();
                    foreach (var kv in _vpBuyMap)
                    {
                        if (!totalVp.ContainsKey(kv.Key)) totalVp[kv.Key] = 0;
                        totalVp[kv.Key] += kv.Value;
                    }
                    foreach (var kv in _vpSellMap)
                    {
                        if (!totalVp.ContainsKey(kv.Key)) totalVp[kv.Key] = 0;
                        totalVp[kv.Key] += kv.Value;
                    }

                    if (totalVp.Count >= 5)
                    {
                        // Sortăm descendent după volum
                        var sortedVp = new List<KeyValuePair<double, double>>(totalVp);
                        sortedVp.Sort((a, b) => b.Value.CompareTo(a.Value));

                        // HVN = top 3 niveluri cu cel mai mare volum
                        var hvnLevels = new List<double>();
                        for (int i = 0; i < Math.Min(3, sortedVp.Count); i++)
                            hvnLevels.Add(sortedVp[i].Key);
                        hvnLevels.Sort();

                        // LVN = niveluri cu volum < 25% din medie, în zona ±2ATR de close
                        double totalVolSum = 0;
                        foreach (var kv in totalVp) totalVolSum += kv.Value;
                        double avgVol  = totalVp.Count > 0 ? totalVolSum / totalVp.Count : 0;
                        double atrZone = Math.Max(atr14, TickSize * 20);
                        double zLo = close - 2.0 * atrZone;
                        double zHi = close + 2.0 * atrZone;

                        var lvnCandidates = new List<double>();
                        for (int i = sortedVp.Count - 1; i >= 0 && lvnCandidates.Count < 5; i--)
                        {
                            double lvlPx  = sortedVp[i].Key;
                            double lvlVol = sortedVp[i].Value;
                            if (avgVol > 0 && lvlVol < avgVol * 0.25 && lvlPx > zLo && lvlPx < zHi)
                                lvnCandidates.Add(lvlPx);
                        }
                        lvnCandidates.Sort();
                        var lvnLevels = new List<double>();
                        for (int i = 0; i < Math.Min(3, lvnCandidates.Count); i++)
                            lvnLevels.Add(lvnCandidates[i]);

                        var _vnic = System.Globalization.CultureInfo.InvariantCulture;
                        var hvnParts = new List<string>();
                        foreach (double p in hvnLevels) hvnParts.Add(p.ToString("F2", _vnic));
                        hvnJson = "[" + string.Join(",", hvnParts) + "]";

                        var lvnParts = new List<string>();
                        foreach (double p in lvnLevels) lvnParts.Add(p.ToString("F2", _vnic));
                        lvnJson = "[" + string.Join(",", lvnParts) + "]";
                    }
                }
                catch { }

                // ── HTF Bias: H4, H1, M15 OHLC real ──────────────────────
                // BarsArray[2] = H4 (minute 240), BarsArray[3] = H1 (minute 60)
                // BarsArray[4] = M15 (minute 15) — refinare entry + VP/OF confluence
                double h4Hi = 0, h4Lo = 0, h4Open = 0, h4Close = 0;
                double h1Hi = 0, h1Lo = 0, h1Open = 0, h1Close = 0;
                double m15Hi = 0, m15Lo = 0, m15Open = 0, m15Close = 0;
                try
                {
                    if (BarsArray.Length > 2 && BarsArray[2].Count > 1)
                    {
                        h4Hi    = Highs[2][0];
                        h4Lo    = Lows[2][0];
                        h4Open  = Opens[2][0];
                        h4Close = Closes[2][0];
                    }
                }
                catch { }
                try
                {
                    if (BarsArray.Length > 3 && BarsArray[3].Count > 1)
                    {
                        h1Hi    = Highs[3][0];
                        h1Lo    = Lows[3][0];
                        h1Open  = Opens[3][0];
                        h1Close = Closes[3][0];
                    }
                }
                catch { }
                try
                {
                    if (BarsArray.Length > 4 && BarsArray[4].Count > 1)
                    {
                        m15Hi    = Highs[4][0];
                        m15Lo    = Lows[4][0];
                        m15Open  = Opens[4][0];
                        m15Close = Closes[4][0];
                    }
                }
                catch { }

                // ── Bar History JSON ──────────────────────────────────────
                string barHistoryJson = "[" + string.Join(",", _barHistory) + "]";

                // ── POC History JSON ──────────────────────────────────────
                // Ultimele 10 valori POC — pentru detectarea migrării (drift)
                var pocSb = new System.Text.StringBuilder("[");
                bool pocFirst = true;
                foreach (double p in _pocHistory)
                {
                    if (!pocFirst) pocSb.Append(",");
                    pocSb.Append(SafeD(p).ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
                    pocFirst = false;
                }
                pocSb.Append("]");
                string pocHistJson = pocSb.ToString();

                // ── DOM Ratio History JSON ────────────────────────────────
                // Ultimele 20 valori bid/ask ratio — pentru detectarea trendului
                var domSb = new System.Text.StringBuilder("[");
                bool domFirst = true;
                foreach (double r in _domRatioHistory)
                {
                    if (!domFirst) domSb.Append(",");
                    domSb.Append(SafeD(r).ToString("F3", System.Globalization.CultureInfo.InvariantCulture));
                    domFirst = false;
                }
                domSb.Append("]");
                string domRatioHistJson = domSb.ToString();

                // ══════════════════════════════════════════════════════════════
                // ADVANCED ORDERFLOW ANALYTICS (computed from VP maps + DOM)
                // ══════════════════════════════════════════════════════════════

                // ── 1. STACKED IMBALANCES ────────────────────────────────────
                // 3+ niveluri consecutive cu buy/sell ratio > 3:1 (sau invers)
                // Semnal puternic de presiune direcțională instituțională
                int stackedBullLevels = 0;
                int stackedBearLevels = 0;
                string stackedSide = "";
                try
                {
                    // Sortăm nivelurile de preț crescător
                    var allLevels = new SortedSet<double>();
                    foreach (var k in _vpBuyMap.Keys)  allLevels.Add(k);
                    foreach (var k in _vpSellMap.Keys) allLevels.Add(k);

                    int bullRun = 0, bearRun = 0;
                    int maxBullRun = 0, maxBearRun = 0;
                    foreach (double lvl in allLevels)
                    {
                        double bv = _vpBuyMap.ContainsKey(lvl) ? _vpBuyMap[lvl] : 0;
                        double sv = _vpSellMap.ContainsKey(lvl) ? _vpSellMap[lvl] : 0;
                        double total = bv + sv;
                        if (total < 5) { bullRun = 0; bearRun = 0; continue; }

                        double ratio = bv / Math.Max(sv, 1);
                        if (ratio >= 3.0)       // buy dominant
                        { bullRun++; bearRun = 0; if (bullRun > maxBullRun) maxBullRun = bullRun; }
                        else if (ratio <= 0.33)  // sell dominant (1/3)
                        { bearRun++; bullRun = 0; if (bearRun > maxBearRun) maxBearRun = bearRun; }
                        else
                        { bullRun = 0; bearRun = 0; }
                    }
                    stackedBullLevels = maxBullRun;
                    stackedBearLevels = maxBearRun;
                    if (maxBullRun >= 3 && maxBullRun > maxBearRun)
                        stackedSide = "BULL";
                    else if (maxBearRun >= 3 && maxBearRun > maxBullRun)
                        stackedSide = "BEAR";
                }
                catch { }

                // ── 2. UNFINISHED BUSINESS (UB) ─────────────────────────────
                // Niveluri unde a existat trading doar pe o parte (only buy SAU only sell)
                // Prețul tinde să revină la aceste niveluri → magnet
                string ubJson = "[]";
                try
                {
                    var ubLevels = new List<string>();
                    var allUbLevels = new SortedSet<double>();
                    foreach (var k in _vpBuyMap.Keys)  allUbLevels.Add(k);
                    foreach (var k in _vpSellMap.Keys) allUbLevels.Add(k);

                    var _ubic = System.Globalization.CultureInfo.InvariantCulture;
                    foreach (double lvl in allUbLevels)
                    {
                        double bv = _vpBuyMap.ContainsKey(lvl) ? _vpBuyMap[lvl] : 0;
                        double sv = _vpSellMap.ContainsKey(lvl) ? _vpSellMap[lvl] : 0;
                        if ((bv > 20 && sv == 0) || (sv > 20 && bv == 0))
                        {
                            string ubSide = bv > 0 ? "BUY_ONLY" : "SELL_ONLY";
                            string ubPrice = SafeD(lvl).ToString("F2", _ubic);
                            string ubVol   = ((long)SafeD(Math.Max(bv, sv))).ToString(_ubic);
                            ubLevels.Add("{\"price\":" + ubPrice + ",\"side\":\"" + ubSide + "\",\"vol\":" + ubVol + "}");
                        }
                    }
                    if (ubLevels.Count > 0)
                        ubJson = "[" + string.Join(",", ubLevels) + "]";
                }
                catch { }

                // ── 3. ICEBERG DETECTION ─────────────────────────────────────
                // Compară volumul executat la un nivel (din VP) cu volumul vizibil pe DOM
                // Dacă executed >> visible → ordin iceberg ascuns la acel nivel
                double icebergScore = 0;
                string icebergSide = "";
                try
                {
                    lock (_domLock)
                    {
                        // Verificăm top 3 bid levels
                        int bidCheck = 0;
                        foreach (var kv in _domBids)
                        {
                            if (bidCheck >= 3) break;
                            double lvl = kv.Key;
                            long domVol = kv.Value;
                            double execVol = (_vpBuyMap.ContainsKey(lvl) ? _vpBuyMap[lvl] : 0)
                                           + (_vpSellMap.ContainsKey(lvl) ? _vpSellMap[lvl] : 0);
                            // Executed > 5x visible = iceberg probabil
                            if (domVol > 0 && execVol > domVol * 5)
                            {
                                double ratio = execVol / domVol;
                                if (ratio > icebergScore)
                                {
                                    icebergScore = Math.Min(ratio, 20);  // cap la 20x
                                    icebergSide = "BID";  // iceberg pe bid = buyer ascuns = bullish
                                }
                            }
                            bidCheck++;
                        }
                        // Verificăm top 3 ask levels
                        int askCheck = 0;
                        foreach (var kv in _domAsks)
                        {
                            if (askCheck >= 3) break;
                            double lvl = kv.Key;
                            long domVol = kv.Value;
                            double execVol = (_vpBuyMap.ContainsKey(lvl) ? _vpBuyMap[lvl] : 0)
                                           + (_vpSellMap.ContainsKey(lvl) ? _vpSellMap[lvl] : 0);
                            if (domVol > 0 && execVol > domVol * 5)
                            {
                                double ratio = execVol / domVol;
                                if (ratio > icebergScore)
                                {
                                    icebergScore = Math.Min(ratio, 20);
                                    icebergSide = "ASK";  // iceberg pe ask = seller ascuns = bearish
                                }
                            }
                            askCheck++;
                        }
                    }
                }
                catch { }

                // ── Big Trades + Absorption (din AladinAbsorption shared static) ──
                string bigTradesJson = "[]";
                double absScore = 0;
                string absSide = "";
                try
                {
                    lock (NinjaTrader.NinjaScript.Indicators.AladinAbsorption.DataLock)
                    {
                        absScore = NinjaTrader.NinjaScript.Indicators.AladinAbsorption.CurrentAbsorptionScore;
                        absSide  = NinjaTrader.NinjaScript.Indicators.AladinAbsorption.CurrentAbsorptionSide ?? "";
                        var trades = NinjaTrader.NinjaScript.Indicators.AladinAbsorption.RecentBigTrades;
                        if (trades != null && trades.Count > 0)
                        {
                            var btSb = new System.Text.StringBuilder("[");
                            bool btFirst = true;
                            foreach (var bt in trades)
                            {
                                if (!btFirst) btSb.Append(",");
                                var _btic = System.Globalization.CultureInfo.InvariantCulture;
                                btSb.Append("{\"price\":" + SafeD(bt.Price).ToString("F2", _btic)
                                    + ",\"size\":" + bt.Size.ToString(_btic)
                                    + ",\"side\":\"" + bt.Side
                                    + "\",\"ts\":\"" + bt.Timestamp + "\"}");
                                btFirst = false;
                            }
                            btSb.Append("]");
                            bigTradesJson = btSb.ToString();
                        }
                    }
                }
                catch { /* AladinAbsorption nu e încărcat — trimitem valori default */ }

                // ── Delta Profile per nivel de preț ───────────────────────
                // delta[level] = buy_vol[level] - sell_vol[level]
                // Pozitiv = cumpărători dominau la acel nivel (suport real)
                // Negativ = vânzători dominau (rezistență reală)
                // Trimitem top 20 niveluri cu cel mai mare |delta| (cele mai semnificative)
                string deltaProfileJson = "[]";
                try
                {
                    var allDpLevels = new SortedSet<double>();
                    foreach (var k in _vpBuyMap.Keys)  allDpLevels.Add(k);
                    foreach (var k in _vpSellMap.Keys) allDpLevels.Add(k);

                    if (allDpLevels.Count > 0)
                    {
                        // Calculăm delta per nivel și sortăm după |delta| descendent
                        var dpList = new List<KeyValuePair<double, double>>();
                        foreach (double lvl in allDpLevels)
                        {
                            double bv = _vpBuyMap.ContainsKey(lvl)  ? _vpBuyMap[lvl]  : 0;
                            double sv = _vpSellMap.ContainsKey(lvl) ? _vpSellMap[lvl] : 0;
                            double totalLvl = bv + sv;
                            if (totalLvl < 5) continue;   // ignorăm niveluri cu volum neglijabil
                            dpList.Add(new KeyValuePair<double, double>(lvl, bv - sv));
                        }
                        // Sortăm după |delta| descendent — cele mai contestate niveluri primele
                        dpList.Sort((a, b) => Math.Abs(b.Value).CompareTo(Math.Abs(a.Value)));

                        var _dpic = System.Globalization.CultureInfo.InvariantCulture;
                        var dpSb  = new System.Text.StringBuilder("[");
                        bool dpFirst = true;
                        int dpCount  = 0;
                        foreach (var kv in dpList)
                        {
                            if (dpCount++ >= 20) break;   // top 20 niveluri
                            if (!dpFirst) dpSb.Append(",");
                            dpSb.Append("{\"p\":" + SafeD(kv.Key).ToString("F2", _dpic)
                                + ",\"d\":" + SafeD(kv.Value).ToString("F0", _dpic) + "}");
                            dpFirst = false;
                        }
                        dpSb.Append("]");
                        deltaProfileJson = dpSb.ToString();
                    }
                }
                catch { }

                // ── JSON payload ──────────────────────────────────────────
                var ic = System.Globalization.CultureInfo.InvariantCulture;
                string posStr = Position.MarketPosition == MarketPosition.Long ? "LONG"
                              : Position.MarketPosition == MarketPosition.Short ? "SHORT" : "FLAT";
                double bidAskRatio = domAskTotal > 0 ? (double)domBidTotal / domAskTotal : 1.0;

                string json =
                  "{\n" +
                  "  \"symbol\":       \"" + Instrument.MasterInstrument.Name + "\",\n" +
                  "  \"timestamp\":   \"" + DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ", ic) + "\",\n" +
                  "  \"price\":       {\n" +
                  "    \"open\":  "  + SafeD(open).ToString("F2", ic)   + ",\n" +
                  "    \"high\":  "  + SafeD(high).ToString("F2", ic)   + ",\n" +
                  "    \"low\":   "  + SafeD(low).ToString("F2", ic)    + ",\n" +
                  "    \"close\": "  + SafeD(close).ToString("F2", ic)  + ",\n" +
                  "    \"volume\": "  + SafeD(volume).ToString("F0", ic) + ",\n" +
                  "    \"bid\":   "  + SafeD(_lastBid).ToString("F2", ic) + ",\n" +
                  "    \"ask\":   "  + SafeD(_lastAsk).ToString("F2", ic) + ",\n" +
                  "    \"spread\": " + SafeD(spread).ToString("F2", ic) + "\n" +
                  "  },\n" +
                  "  \"orderflow\": {\n" +
                  "    \"cum_delta\":     " + SafeD(_cumDelta).ToString("F0", ic)      + ",\n" +
                  "    \"bar_buy_vol\":   " + SafeD(_barBuyVolume).ToString("F0", ic)  + ",\n" +
                  "    \"bar_sell_vol\":  " + SafeD(_barSellVolume).ToString("F0", ic) + ",\n" +
                  "    \"imbalance_pct\": " + SafeD(imbalancePct).ToString("F2", ic)   + ",\n" +
                  "    \"session_vol\":   " + SafeD(_sessionVolume).ToString("F0", ic) + ",\n" +
                  "    \"tick_count\":      " + _tickCount.ToString(ic)                          + ",\n" +
                  "    \"vwap\":           " + SafeD(vwap).ToString("F2", ic)                   + ",\n" +
                  "    \"big_buy_count\":  " + _barBigBuyCount.ToString(ic)                     + ",\n" +
                  "    \"big_sell_count\": " + _barBigSellCount.ToString(ic)                    + ",\n" +
                  "    \"max_trade_size\": " + SafeD(_barMaxTradeSize).ToString("F0", ic)       + ",\n" +
                  "    \"bar_tick_count\": " + _barTickCount.ToString(ic)                       + ",\n" +
                  "    \"tape_speed\":     " + SafeD(_barTickCount / Math.Max((DateTime.UtcNow - _barOpenTime).TotalSeconds, 1.0)).ToString("F2", ic) + ",\n" +
                  "    \"bar_delta\":      " + SafeD(_barBuyVolume - _barSellVolume).ToString("F0", ic) + ",\n" +
                  "    \"delta_at_high\":  " + SafeD(_deltaAtHigh).ToString("F0", ic)           + ",\n" +
                  "    \"delta_at_low\":   " + SafeD(_deltaAtLow).ToString("F0", ic)            + "\n" +
                  "  },\n" +
                  "  \"volume_profile\": {\n" +
                  "    \"poc\":          " + SafeD(poc).ToString("F2", ic)             + ",\n" +
                  "    \"vah\":          " + SafeD(vah).ToString("F2", ic)             + ",\n" +
                  "    \"val\":          " + SafeD(val).ToString("F2", ic)             + ",\n" +
                  "    \"prev_poc\":     " + SafeD(_prevSessionPoc).ToString("F2", ic) + ",\n" +
                  "    \"prev_vah\":     " + SafeD(_prevSessionVah).ToString("F2", ic) + ",\n" +
                  "    \"prev_val\":     " + SafeD(_prevSessionVal).ToString("F2", ic) + ",\n" +
                  "    \"hvn\":          " + hvnJson                                   + ",\n" +
                  "    \"lvn\":          " + lvnJson                                   + ",\n" +
                  "    \"vwap_sd\":      " + SafeD(vwapSd).ToString("F4", ic)          + ",\n" +
                  "    \"vwap_sd1_hi\":  " + SafeD(vwapSd1Hi).ToString("F2", ic)       + ",\n" +
                  "    \"vwap_sd1_lo\":  " + SafeD(vwapSd1Lo).ToString("F2", ic)       + ",\n" +
                  "    \"vwap_sd2_hi\":  " + SafeD(vwapSd2Hi).ToString("F2", ic)       + ",\n" +
                  "    \"vwap_sd2_lo\":  " + SafeD(vwapSd2Lo).ToString("F2", ic)       + ",\n" +
                  "    \"delta_profile\": " + deltaProfileJson                          + "\n" +
                  "  },\n" +
                  "  \"dom\": " + domJson + ",\n" +
                  "  \"dom_liquidity\": {\n" +
                  "    \"total_bid_size\": " + domBidTotal.ToString(ic) + ",\n" +
                  "    \"total_ask_size\": " + domAskTotal.ToString(ic) + ",\n" +
                  "    \"bid_ask_ratio\":  " + SafeD(bidAskRatio).ToString("F3", ic) + "\n" +
                  "  },\n" +
                  "  \"bar_history\": " + barHistoryJson + ",\n" +
                  "  \"atr_14\":  " + SafeD(atr14).ToString("F4", ic) + ",\n" +
                  "  \"bar_index\": " + CurrentBar.ToString(ic) + ",\n" +
                  "  \"position_status\": \"" + posStr + "\",\n" +
                  "  \"position_qty\": " + Position.Quantity.ToString(ic) + ",\n" +
                  "  \"poc_history\": " + pocHistJson + ",\n" +
                  "  \"dom_ratio_history\": " + domRatioHistJson + ",\n" +
                  "  \"htf\": {\n" +
                  "    \"h4_hi\":    " + SafeD(h4Hi).ToString("F2", ic)    + ",\n" +
                  "    \"h4_lo\":    " + SafeD(h4Lo).ToString("F2", ic)    + ",\n" +
                  "    \"h4_open\":  " + SafeD(h4Open).ToString("F2", ic)  + ",\n" +
                  "    \"h4_close\": " + SafeD(h4Close).ToString("F2", ic) + ",\n" +
                  "    \"h1_hi\":    " + SafeD(h1Hi).ToString("F2", ic)    + ",\n" +
                  "    \"h1_lo\":    " + SafeD(h1Lo).ToString("F2", ic)    + ",\n" +
                  "    \"h1_open\":  " + SafeD(h1Open).ToString("F2", ic)   + ",\n" +
                  "    \"h1_close\": " + SafeD(h1Close).ToString("F2", ic)  + ",\n" +
                  "    \"m15_hi\":   " + SafeD(m15Hi).ToString("F2", ic)    + ",\n" +
                  "    \"m15_lo\":   " + SafeD(m15Lo).ToString("F2", ic)    + ",\n" +
                  "    \"m15_open\": " + SafeD(m15Open).ToString("F2", ic)  + ",\n" +
                  "    \"m15_close\":" + SafeD(m15Close).ToString("F2", ic) + "\n" +
                  "  },\n" +
                  "  \"big_trades\": " + bigTradesJson + ",\n" +
                  "  \"absorption_score\": " + SafeD(absScore).ToString("F1", ic) + ",\n" +
                  "  \"absorption_side\":  \"" + absSide + "\",\n" +
                  "  \"stacked_imbalances\": {\n" +
                  "    \"bull_levels\": " + stackedBullLevels.ToString(ic) + ",\n" +
                  "    \"bear_levels\": " + stackedBearLevels.ToString(ic) + ",\n" +
                  "    \"side\":        \"" + stackedSide + "\"\n" +
                  "  },\n" +
                  "  \"unfinished_business\": " + ubJson + ",\n" +
                  "  \"iceberg\": {\n" +
                  "    \"score\": " + SafeD(icebergScore).ToString("F1", ic) + ",\n" +
                  "    \"side\":  \"" + icebergSide + "\"\n" +
                  "  },\n" +
                  "  \"profile_shape\": \"" + _currentProfileShape + "\",\n" +
                  "  \"of_consol_metrics\": {\n" +
                  "    \"delta_oscillation_idx\": " + SafeD(_ofConsolDeltaOscIdx).ToString("F4", ic) + ",\n" +
                  "    \"bilateral_absorption\":  " + (_ofConsolBilateralAbsorb ? "true" : "false") + ",\n" +
                  "    \"big_trade_balance\":     " + SafeD(_ofConsolBigTradeBalance).ToString("F4", ic) + ",\n" +
                  "    \"consol_d_shape_count\":  " + _ofConsolDShapeCount.ToString(ic) + "\n" +
                  "  },\n" +
                  "  \"account\": {\n" +
                  "    \"cash\":               " + SafeD(Account.Get(AccountItem.CashValue,            Currency.UsDollar)).ToString("F2", ic) + ",\n" +
                  "    \"realized_pnl_today\": " + SafeD(Account.Get(AccountItem.RealizedProfitLoss,   Currency.UsDollar)).ToString("F2", ic) + ",\n" +
                  "    \"open_pnl\":           " + SafeD(Account.Get(AccountItem.UnrealizedProfitLoss, Currency.UsDollar)).ToString("F2", ic) + ",\n" +
                  "    \"net_liquidation\":    " + SafeD(Account.Get(AccountItem.NetLiquidation,       Currency.UsDollar)).ToString("F2", ic) + "\n" +
                  "  }\n" +
                  "}";

                // Trimite async fără await — non-blocking pe thread-ul NT8
                _ = PostAsync($"http://{MacIP}:{MacPort}/nt8_data", json);
            }
            catch (Exception ex)
            {
                Print($"[ALADIN] SendMarketData error: {ex.Message}");
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // VOLUME PROFILE — calculăm POC, VAH, VAL din distribuția intra-sesiune
        // ════════════════════════════════════════════════════════════════════
        private void ComputeVolumeProfile(out double poc, out double vah, out double val)
        {
            poc = 0; vah = 0; val = 0;
            if (_vpBuyMap.Count == 0 && _vpSellMap.Count == 0) return;

            // Combină buy + sell pentru total volume per nivel
            var totalVP = new Dictionary<double, double>();
            foreach (var kv in _vpBuyMap)
            {
                if (!totalVP.ContainsKey(kv.Key)) totalVP[kv.Key] = 0;
                totalVP[kv.Key] += kv.Value;
            }
            foreach (var kv in _vpSellMap)
            {
                if (!totalVP.ContainsKey(kv.Key)) totalVP[kv.Key] = 0;
                totalVP[kv.Key] += kv.Value;
            }

            // POC = nivelul cu cel mai mare volum
            double maxVol = 0, pocLevel = 0;
            double totalVol = 0;
            foreach (var kv in totalVP)
            {
                totalVol += kv.Value;
                if (kv.Value > maxVol) { maxVol = kv.Value; pocLevel = kv.Key; }
            }
            poc = pocLevel;

            // VAH/VAL = 70% din volumul total (Value Area)
            double vaTarget = totalVol * 0.70;
            var sorted = new List<KeyValuePair<double, double>>(totalVP);
            sorted.Sort((a, b) => b.Value.CompareTo(a.Value));  // descending by volume

            double accumulated = 0;
            double vaLow = double.MaxValue, vaHigh = double.MinValue;
            foreach (var kv in sorted)
            {
                accumulated += kv.Value;
                if (kv.Key < vaLow)  vaLow  = kv.Key;
                if (kv.Key > vaHigh) vaHigh = kv.Key;
                if (accumulated >= vaTarget) break;
            }
            val = vaLow  < double.MaxValue ? vaLow  : poc - TickSize * 10;
            vah = vaHigh > double.MinValue ? vaHigh : poc + TickSize * 10;
        }

        // ════════════════════════════════════════════════════════════════════
        // DOM JSON — top N niveluri bid + ask
        // ════════════════════════════════════════════════════════════════════
        // ── NaN/Infinity guard — JSON nu acceptă NaN sau Infinity ────────────
        private static double SafeD(double v) =>
            (double.IsNaN(v) || double.IsInfinity(v)) ? 0.0 : v;

        private string BuildDomJson(int levels)
        {
            var bids = new List<string>();
            var asks = new List<string>();

            lock (_domLock)
            {
                int i = 0;
                foreach (var kv in _domBids)
                {
                    if (i++ >= levels) break;
                    bids.Add($"{{\"price\":{kv.Key:F2},\"size\":{kv.Value}}}");
                }
                i = 0;
                foreach (var kv in _domAsks)
                {
                    if (i++ >= levels) break;
                    asks.Add($"{{\"price\":{kv.Key:F2},\"size\":{kv.Value}}}");
                }
            }

            return $"{{\"bids\":[{string.Join(",", bids)}],\"asks\":[{string.Join(",", asks)}]}}";
        }

        // ════════════════════════════════════════════════════════════════════
        // HTTP POST — async fire-and-forget
        // ════════════════════════════════════════════════════════════════════
        private static int _httpErrorCount = 0;
        private static DateTime _lastHttpError = DateTime.MinValue;

        private async Task PostAsync(string url, string json)
        {
            try
            {
                using var content = new StringContent(json, Encoding.UTF8, "application/json");
                using var response = await _http.PostAsync(url, content).ConfigureAwait(false);
                // Resetăm contor erori la succes
                if (_httpErrorCount > 0)
                {
                    Print($"[ALADIN] ✅ Conexiune restabilită → {url}");
                    _httpErrorCount = 0;
                }
            }
            catch (TaskCanceledException)
            {
                // Timeout — logăm o dată la 30 secunde să nu spamăm Output Window
                if ((DateTime.Now - _lastHttpError).TotalSeconds > 30)
                {
                    Print($"[ALADIN] ⚠️ TIMEOUT POST {url} (>{_http.Timeout.TotalMilliseconds}ms) — bridge offline?");
                    _lastHttpError = DateTime.Now;
                    _httpErrorCount++;
                }
            }
            catch (ObjectDisposedException ode)
            {
                Print($"[ALADIN] ❌ HttpClient DISPOSED — bug detectat: {ode.ObjectName}. Repornește NT8.");
            }
            catch (Exception ex)
            {
                if ((DateTime.Now - _lastHttpError).TotalSeconds > 10)
                {
                    Print($"[ALADIN] ❌ HTTP error [{ex.GetType().Name}]: {ex.Message}");
                    Print($"[ALADIN] ❌ Target: {url}");
                    _lastHttpError = DateTime.Now;
                    _httpErrorCount++;
                }
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // EXECUTION LISTENER — ascultă comenzi BUY/SELL/CLOSE pe port 8002
        // ════════════════════════════════════════════════════════════════════
        private void StartExecutionListener()
        {
            try
            {
                _listener        = new HttpListener();
                _listener.Prefixes.Add($"http://+:{ListenPort}/");
                _listener.Start();
                _listenerRunning = true;
                _listenerThread  = new Thread(ListenLoop) { IsBackground = true, Name = "AladinExecListener" };
                _listenerThread.Start();
                Print($"[ALADIN] Execution listener pornit pe port {ListenPort}");
            }
            catch (Exception ex)
            {
                Print($"[ALADIN] Listener start error: {ex.Message}  (poate necesită 'netsh http add urlacl')");
            }
        }

        private void StopExecutionListener()
        {
            _listenerRunning = false;
            try { _listener?.Stop(); _listener?.Close(); } catch { }
        }

        private void ListenLoop()
        {
            while (_listenerRunning && _listener != null && _listener.IsListening)
            {
                try
                {
                    var ctx     = _listener.GetContext();
                    var request = ctx.Request;

                    if (request.HttpMethod == "POST")
                    {
                        string body;
                        using (var reader = new System.IO.StreamReader(request.InputStream, Encoding.UTF8))
                            body = reader.ReadToEnd();

                        // Parsăm comanda (JSON simplu: {"action":"BUY","qty":1,"price":25000})
                        ParseAndExecute(body);
                    }

                    // Răspuns 200 OK
                    ctx.Response.StatusCode = 200;
                    using var writer = new System.IO.StreamWriter(ctx.Response.OutputStream);
                    writer.Write("{\"status\":\"ok\"}");
                }
                catch (HttpListenerException) { break; }  // Oprit normal
                catch (Exception ex)
                {
                    Print($"[ALADIN] ListenLoop error: {ex.Message}");
                }
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // EXECUTE COMMAND — parsăm JSON și executăm pe thread-ul NT8
        // ════════════════════════════════════════════════════════════════════
        private void ParseAndExecute(string json)
        {
            // Parser minimal fără dependențe externe
            string action = ExtractJsonString(json, "action")?.ToUpper() ?? "";
            int    qty    = int.TryParse(ExtractJsonString(json, "qty"), out int q) ? q : 1;
            string signal = ExtractJsonString(json, "signal") ?? "";
            double sl     = double.TryParse(ExtractJsonString(json, "sl"),
                                System.Globalization.NumberStyles.Any,
                                System.Globalization.CultureInfo.InvariantCulture, out double slVal) ? slVal : 0;
            double tp     = double.TryParse(ExtractJsonString(json, "tp"),
                                System.Globalization.NumberStyles.Any,
                                System.Globalization.CultureInfo.InvariantCulture, out double tpVal) ? tpVal : 0;

            Print($"[ALADIN] COMMAND RECEIVED: {action} qty={qty} sl={sl:F2} tp={tp:F2} signal={signal}");

            // NT8 execuție — trebuie pe GUI thread prin Dispatcher
            Dispatcher.InvokeAsync(() =>
            {
                try
                {
                    switch (action)
                    {
                        case "BUY":
                        case "LONG":
                            if (Position.MarketPosition == MarketPosition.Short)
                                ExitShort(qty, "AladinClose", "");
                            EnterLong(qty, "AladinLong");
                            // SL/TP bracket orders reale
                            if (sl > 0) SetStopLoss("AladinLong", CalculationMode.Price, sl, false);
                            if (tp > 0) SetProfitTarget("AladinLong", CalculationMode.Price, tp);
                            Print($"[ALADIN] ✅ LONG {qty} | SL={sl:F2} TP={tp:F2}");
                            break;

                        case "SELL":
                        case "SHORT":
                            if (Position.MarketPosition == MarketPosition.Long)
                                ExitLong(qty, "AladinClose", "");
                            EnterShort(qty, "AladinShort");
                            // SL/TP bracket orders reale
                            if (sl > 0) SetStopLoss("AladinShort", CalculationMode.Price, sl, false);
                            if (tp > 0) SetProfitTarget("AladinShort", CalculationMode.Price, tp);
                            Print($"[ALADIN] ✅ SHORT {qty} | SL={sl:F2} TP={tp:F2}");
                            break;

                        case "CLOSE":
                        case "EXIT":
                            if (Position.MarketPosition == MarketPosition.Long)
                                ExitLong("AladinClose");
                            else if (Position.MarketPosition == MarketPosition.Short)
                                ExitShort("AladinClose");
                            Print($"[ALADIN] ✅ CLOSE executat");
                            break;

                        // ADD TO POSITION — adaugă la poziția existentă (scaled entry ICT)
                        case "ADD":
                        {
                            int addQty = qty > 0 ? qty : 1;
                            if (Position.MarketPosition == MarketPosition.Long)
                            {
                                EnterLong(addQty, "AladinScaled");
                                if (sl > 0) SetStopLoss("AladinScaled", CalculationMode.Price, sl, false);
                                if (tp > 0) SetProfitTarget("AladinScaled", CalculationMode.Price, tp);
                                Print($"[ALADIN] ➕ ADD LONG +{addQty} @ {Close[0]:F2}");
                            }
                            else if (Position.MarketPosition == MarketPosition.Short)
                            {
                                EnterShort(addQty, "AladinScaled");
                                if (sl > 0) SetStopLoss("AladinScaled", CalculationMode.Price, sl, false);
                                if (tp > 0) SetProfitTarget("AladinScaled", CalculationMode.Price, tp);
                                Print($"[ALADIN] ➕ ADD SHORT +{addQty} @ {Close[0]:F2}");
                            }
                            break;
                        }

                        // PARTIAL CLOSE la 1R — închide jumătate din poziție
                        case "REDUCE":
                        {
                            int reduceQty = qty > 0 ? qty : (Position.Quantity / 2);
                            if (Position.MarketPosition == MarketPosition.Long && reduceQty > 0)
                            {
                                ExitLong(reduceQty, "AladinPartial", "AladinLong");
                                Print($"[ALADIN] ✅ PARTIAL CLOSE LONG -{reduceQty} @ {Close[0]:F2}");
                            }
                            else if (Position.MarketPosition == MarketPosition.Short && reduceQty > 0)
                            {
                                ExitShort(reduceQty, "AladinPartial", "AladinShort");
                                Print($"[ALADIN] ✅ PARTIAL CLOSE SHORT -{reduceQty} @ {Close[0]:F2}");
                            }
                            break;
                        }

                        // MOVE_SL — mută stop loss la preț nou (breakeven sau trailing)
                        // Fix v10.6: InvalidPrice guard — folosim GetCurrentBid/Ask (preț LIVE),
                        // NU Close[0] care e close-ul ultimei bare completate (STALE în Dispatcher.InvokeAsync).
                        // Close[0] poate fi cu 10+ puncte în urmă pe NQ → falsă detecție "breached" → EXIT prematur.
                        // Logică: pentru LONG, SL invalid = SL >= currentBid (SL deasupra prețului curent)
                        //         pentru SHORT, SL invalid = SL <= currentAsk (SL sub prețul curent)
                        case "MOVE_SL":
                        {
                            if (sl > 0)
                            {
                                double liveBid = GetCurrentBid();
                                double liveAsk = GetCurrentAsk();
                                // Fallback: dacă bid/ask nu sunt disponibile, folosim Close[0] cu marjă de siguranță
                                double livePrice = (liveBid > 0) ? liveBid : Close[0];
                                double livePriceShort = (liveAsk > 0) ? liveAsk : Close[0];

                                if (Position.MarketPosition == MarketPosition.Long)
                                {
                                    // LONG: SL trebuie să fie SUB prețul curent
                                    // Invalid doar dacă SL >= currentBid (SL deasupra prețului)
                                    if (sl >= livePrice)
                                    {
                                        // SL deasupra prețului curent → InvalidPrice dacă setăm → CLOSE direct
                                        ExitLong("AladinSL_Hit", "AladinLong");
                                        Print($"[ALADIN] ⚡ MOVE_SL LONG breached: sl={sl:F2} >= Bid={livePrice:F2} → InvalidPrice prevenit → EXIT LONG direct");
                                    }
                                    else
                                    {
                                        SetStopLoss("AladinLong", CalculationMode.Price, sl, false);
                                        Print($"[ALADIN] 🔄 SL mutat → {sl:F2} (LONG) [Bid={livePrice:F2}]");
                                    }
                                }
                                else if (Position.MarketPosition == MarketPosition.Short)
                                {
                                    // SHORT: SL trebuie să fie DEASUPRA prețului curent
                                    // Invalid doar dacă SL <= currentAsk (SL sub prețul)
                                    if (sl <= livePriceShort)
                                    {
                                        // SL sub prețul curent → InvalidPrice dacă setăm → CLOSE direct
                                        ExitShort("AladinSL_Hit", "AladinShort");
                                        Print($"[ALADIN] ⚡ MOVE_SL SHORT breached: sl={sl:F2} <= Ask={livePriceShort:F2} → InvalidPrice prevenit → EXIT SHORT direct");
                                    }
                                    else
                                    {
                                        SetStopLoss("AladinShort", CalculationMode.Price, sl, false);
                                        Print($"[ALADIN] 🔄 SL mutat → {sl:F2} (SHORT) [Ask={livePriceShort:F2}]");
                                    }
                                }
                            }
                            break;
                        }

                        default:
                            Print($"[ALADIN] ⚠️ Comandă necunoscută: {action}");
                            break;
                    }

                    // Confirmă execuția înapoi la Mac — DOAR pentru BUY/SELL/CLOSE, nu MOVE_SL
                    // MOVE_SL nu e un trade event, nu trimitem confirm pentru el
                    if (action == "BUY" || action == "LONG" ||
                        action == "SELL" || action == "SHORT" ||
                        action == "CLOSE" || action == "EXIT")
                    {
                        // Pentru CLOSE folosim prețul poziției dacă e disponibil, altfel Close[0]
                        double confirmPrice = (action == "CLOSE" || action == "EXIT")
                            ? (Position.MarketPosition != MarketPosition.Flat
                               ? Position.AveragePrice   // poziție încă deschisă → avg price
                               : Close[0])               // deja flat → close bar curent
                            : Close[0];
                        _ = PostAsync($"http://{MacIP}:{MacPort}/execution_confirm",
                            $"{{\"action\":\"{action}\",\"qty\":{qty},\"price\":{confirmPrice:F2},\"ts\":\"{DateTime.UtcNow:O}\"}}");
                    }
                }
                catch (Exception ex)
                {
                    Print($"[ALADIN] Execute error: {ex.Message}");
                }
            });
        }

        // ════════════════════════════════════════════════════════════════════
        // POSITION UPDATE — trimite CLOSE confirm când poziția se închide
        // (SL hit, TP hit, ExitOnSessionClose, sau manual din NT8)
        // ════════════════════════════════════════════════════════════════════
        protected override void OnPositionUpdate(Cbi.Position position, double averagePrice,
                                                  int quantity, Cbi.MarketPosition marketPosition)
        {
            if (marketPosition == MarketPosition.Flat && State == State.Realtime)
            {
                // averagePrice poate fi 0 dacă NT8 nu a procesat fill-ul încă — folosim Close[0] ca fallback
                double exitPrice = (averagePrice > 0) ? averagePrice : Close[0];
                if (exitPrice <= 0) return;  // dacă amândouă sunt 0, ignorăm complet
                Print($"[ALADIN] 📤 Poziție închisă @ {exitPrice:F2} — trimit CLOSE confirm");
                _ = PostAsync($"http://{MacIP}:{MacPort}/execution_confirm",
                    $"{{\"action\":\"CLOSE\",\"qty\":{quantity},\"price\":{exitPrice:F2},\"ts\":\"{DateTime.UtcNow:O}\"}}");
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // HELPER — JSON parser minimal (fără System.Text.Json dependency)
        // ════════════════════════════════════════════════════════════════════
        private string ExtractJsonString(string json, string key)
        {
            string search = $"\"{key}\"";
            int idx = json.IndexOf(search, StringComparison.OrdinalIgnoreCase);
            if (idx < 0) return null;
            idx += search.Length;
            while (idx < json.Length && (json[idx] == ':' || json[idx] == ' ')) idx++;
            if (idx >= json.Length) return null;
            if (json[idx] == '"')
            {
                idx++;
                int end = json.IndexOf('"', idx);
                return end > 0 ? json.Substring(idx, end - idx) : null;
            }
            else
            {
                int end = idx;
                while (end < json.Length && json[end] != ',' && json[end] != '}') end++;
                return json.Substring(idx, end - idx).Trim();
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // HELPER — Descending comparer pentru SortedDictionary DOM bids
        // ════════════════════════════════════════════════════════════════════
        private class DescendingComparer : IComparer<double>
        {
            public int Compare(double x, double y) => y.CompareTo(x);
        }
    }
}

/*
 ╔══════════════════════════════════════════════════════════════════════════════╗
 ║  INSTRUCȚIUNI DE SETUP                                                       ║
 ╠══════════════════════════════════════════════════════════════════════════════╣
 ║  1. Copiază fișierul în:                                                     ║
 ║     Documents\NinjaTrader 8\bin\Custom\Strategies\                          ║
 ║                                                                              ║
 ║  2. Compilează din NT8: New > NinjaScript Editor > Compile                  ║
 ║                                                                              ║
 ║  3. Setează IP-ul Mac-ului: schimbă MacIP = "192.168.1.100"                ║
 ║     (găsești IP-ul Mac din System Preferences > Network)                   ║
 ║                                                                              ║
 ║  4. Pe Windows (Parallels), rulează în PowerShell ca Admin:                 ║
 ║     netsh http add urlacl url=http://+:8002/ user=Everyone                  ║
 ║     (permite HttpListener să asculte pe port 8002)                          ║
 ║                                                                              ║
 ║  5. Activează strategia pe graficul NQ sau ES (1min sau tick)               ║
 ║                                                                              ║
 ║  6. Verifică în Output Window NT8:                                           ║
 ║     [ALADIN] Bridge pornit → Mac: 192.168.x.x:8001  Listen: 8002           ║
 ╚══════════════════════════════════════════════════════════════════════════════╝
*/
