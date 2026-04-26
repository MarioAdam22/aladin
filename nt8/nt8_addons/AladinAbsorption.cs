// ═══════════════════════════════════════════════════════════════════
// AladinAbsorption v8.1 — Big Trades + Absorption Detection for NT8
// ═══════════════════════════════════════════════════════════════════
//
// INSTALARE:
// 1. Copiază acest fișier în: Documents\NinjaTrader 8\bin\Custom\AddOns\
// 2. În NinjaTrader → New → NinjaScript Editor → Compilează (F5)
// 3. Atașează indicatorul AladinAbsorption pe chart-ul NQ 1-min
// 4. AladinBridge.cs va citi automat datele din shared memory
//
// CE FACE:
// - Monitorizează Time & Sales pentru ordine >= 50 contracte (Big Trades)
// - Detectează absorption: volum mare la un nivel dar prețul nu se mișcă
// - Trimite datele prin AladinBridge la Python bridge_api.py
//
// ALTERNATIVĂ SIMPLĂ (fără acest addon):
// mario_rag.py calculează absorption din bar_buy_vol/bar_sell_vol existente
// Acest addon oferă precizie mai mare cu date tick-by-tick
// ═══════════════════════════════════════════════════════════════════

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.DrawingTools;
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class AladinAbsorption : Indicator
    {
        // ── Parametri Configurabili ──
        [NinjaScriptProperty]
        [Range(10, 500)]
        [Display(Name = "Big Trade Threshold", Order = 1, GroupName = "Aladin")]
        public int BigTradeThreshold { get; set; }

        [NinjaScriptProperty]
        [Range(3, 30)]
        [Display(Name = "Absorption Lookback Bars", Order = 2, GroupName = "Aladin")]
        public int AbsorptionLookback { get; set; }

        [NinjaScriptProperty]
        [Range(1.5, 5.0)]
        [Display(Name = "Volume Spike Multiplier", Order = 3, GroupName = "Aladin")]
        public double VolSpikeMultiplier { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, 0.5)]
        [Display(Name = "Small Body ATR Ratio", Order = 4, GroupName = "Aladin")]
        public double SmallBodyRatio { get; set; }

        // ── Shared Data (citite de AladinBridge) ──
        // Aceste proprietăți sunt citite de AladinBridge.cs și trimise la Python
        public static List<BigTradeEntry> RecentBigTrades = new List<BigTradeEntry>();
        public static double CurrentAbsorptionScore = 0;
        public static string CurrentAbsorptionSide = "";  // "BID" or "ASK"
        public static readonly object DataLock = new object();

        // ── Internal ──
        private List<BigTradeEntry> _bigTrades = new List<BigTradeEntry>();
        private double _avgVolume = 0;
        private int _tickCount = 0;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Aladin v8.1 — Big Trades + Absorption Detection";
                Name = "AladinAbsorption";
                Calculate = Calculate.OnEachTick;
                IsOverlay = true;

                BigTradeThreshold = 50;        // ordine >= 50 contracts = big trade
                AbsorptionLookback = 10;       // ultimele 10 bare pt avg volume
                VolSpikeMultiplier = 2.0;      // vol > 2x avg = spike
                SmallBodyRatio = 0.3;          // body < 0.3 * ATR = small body
            }
            else if (State == State.Realtime)
            {
                // Subscrie la Time & Sales pentru big trade detection
                // NT8 trimite automat MarketDataType.Last pe fiecare tick
            }
        }

        protected override void OnMarketData(MarketDataEventArgs marketDataUpdate)
        {
            // ── Big Trade Detection din Time & Sales ──
            if (marketDataUpdate.MarketDataType == MarketDataType.Last)
            {
                double price = marketDataUpdate.Price;
                long volume = marketDataUpdate.Volume;

                if (volume >= BigTradeThreshold)
                {
                    // Determinăm BUY vs SELL din relația cu bid/ask
                    string side = "UNKNOWN";
                    double bid = GetCurrentBid();
                    double ask = GetCurrentAsk();

                    if (price >= ask)
                        side = "BUY";   // lifted the offer → aggressive buyer
                    else if (price <= bid)
                        side = "SELL";  // hit the bid → aggressive seller
                    else
                        side = price > (bid + ask) / 2 ? "BUY" : "SELL";

                    var entry = new BigTradeEntry
                    {
                        Price = price,
                        Size = (int)volume,
                        Side = side,
                        Timestamp = DateTime.UtcNow.ToString("o")
                    };

                    _bigTrades.Add(entry);

                    // Păstrăm doar ultimele 60 secunde de big trades
                    var cutoff = DateTime.UtcNow.AddSeconds(-60);
                    _bigTrades.RemoveAll(bt => DateTime.Parse(bt.Timestamp) < cutoff);

                    // Update shared data
                    lock (DataLock)
                    {
                        RecentBigTrades = new List<BigTradeEntry>(_bigTrades);
                    }

                    // Visual marker on chart
                    if (side == "BUY")
                        Draw.TriangleUp(this, "BT_" + _tickCount, false, 0, price - TickSize * 5, Brushes.Lime);
                    else
                        Draw.TriangleDown(this, "BT_" + _tickCount, false, 0, price + TickSize * 5, Brushes.Red);

                    _tickCount++;
                }
            }
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBars[0] < AbsorptionLookback + 2) return;

            // ── Absorption Detection pe bara închisă ──
            // Calculăm: volum mare + body mic = cineva absoarbe
            double avgVol = 0;
            for (int i = 1; i <= AbsorptionLookback; i++)
                avgVol += Volume[i];
            avgVol /= AbsorptionLookback;

            double curVol = Volume[0];
            double curBody = Math.Abs(Close[0] - Open[0]);
            double curRange = High[0] - Low[0];
            double atr = 0;

            // ATR simplu pe 14 bare
            if (CurrentBars[0] >= 15)
            {
                double sumTR = 0;
                for (int i = 1; i <= 14; i++)
                {
                    double tr = Math.Max(High[i] - Low[i],
                                Math.Max(Math.Abs(High[i] - Close[i + 1]),
                                         Math.Abs(Low[i] - Close[i + 1])));
                    sumTR += tr;
                }
                atr = sumTR / 14.0;
            }

            // ── Condiții Absorption ──
            bool volSpike = curVol > avgVol * VolSpikeMultiplier;
            bool smallBody = atr > 0 && curBody < atr * SmallBodyRatio;

            double absScore = 0;
            string absSide = "";

            if (volSpike && smallBody)
            {
                // Absorption detectată! Determinăm direcția din wicks
                double lowerWick = Math.Min(Open[0], Close[0]) - Low[0];
                double upperWick = High[0] - Math.Max(Open[0], Close[0]);

                // Scor bazat pe cât de puternic e spike-ul
                absScore = Math.Min((curVol / Math.Max(avgVol, 1)) * 25, 100);

                if (lowerWick > upperWick * 1.5)
                {
                    // Lower wick dominant → bid absorption (buyers absorbing sellers)
                    absSide = "BID";
                    // Green square on chart
                    Draw.Square(this, "ABS_" + CurrentBar, false, 0, Low[0] - TickSize * 3, Brushes.DodgerBlue);
                }
                else if (upperWick > lowerWick * 1.5)
                {
                    // Upper wick dominant → ask absorption (sellers absorbing buyers)
                    absSide = "ASK";
                    Draw.Square(this, "ABS_" + CurrentBar, false, 0, High[0] + TickSize * 3, Brushes.Orange);
                }
                else
                {
                    // Ambiguous — use recent delta direction
                    // Default to BID if close > open, ASK otherwise
                    absSide = Close[0] >= Open[0] ? "BID" : "ASK";
                    Draw.Diamond(this, "ABS_" + CurrentBar, false, 0,
                        absSide == "BID" ? Low[0] - TickSize * 3 : High[0] + TickSize * 3,
                        Brushes.Yellow);
                }
            }

            // Update shared data pentru AladinBridge
            lock (DataLock)
            {
                CurrentAbsorptionScore = absScore;
                CurrentAbsorptionSide = absSide;
            }

            _avgVolume = avgVol;
        }

        #region Properties
        [Browsable(false)]
        [XmlIgnore]
        public double AvgVolume { get { return _avgVolume; } }
        #endregion
    }

    // ── Data Model ──
    public class BigTradeEntry
    {
        public double Price { get; set; }
        public int Size { get; set; }
        public string Side { get; set; }  // "BUY" or "SELL"
        public string Timestamp { get; set; }
    }
}
