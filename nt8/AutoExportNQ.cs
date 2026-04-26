// AutoExportNQ.cs — NT8 Add-On
// Instalare: Documents\NinjaTrader 8\bin\Custom\AddOns\AutoExportNQ.cs
// Compilare: NinjaScript Editor → F5
// Fereastra se deschide automat la pornirea NT8

#region Using declarations
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui.Tools;
#endregion

namespace NinjaTrader.NinjaScript.AddOns
{
    public class AutoExportNQ : AddOnBase
    {
        // ── SETĂRI ────────────────────────────────────────────────────────────
        // Schimbă calea dacă folosești Parallels shared folder:
        //   ex: @"\\Mac\Home\Desktop\Aladin\"
        private static readonly string OUTPUT_DIR = Environment.GetFolderPath(Environment.SpecialFolder.Desktop) + @"\Aladin\AladinExport\";

        // DAILY UPDATE: NQ 06-26 — trage de la 1 aprilie 2026 pana azi
        private static readonly string[] SYMBOLS = new string[]
        {
            "NQ 06-26",
        };

        // ── UI ────────────────────────────────────────────────────────────────
        private NTWindow    _window;
        private TextBox     _txtDir;
        private TextBox     _txtLog;
        private Button      _btnStart;
        private Button      _btnStop;
        private ProgressBar _progress;
        private volatile bool _running;
        private volatile bool _stop;
        private const int AUTO_EXPORT_INTERVAL_MIN = 5; // re-export automat la fiecare 5 minute

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name        = "AutoExportNQ";
                Description = "Export automat date istorice NQ/ES 1-minut";
            }
            else if (State == State.Active)
            {
                // Deschidem fereastra automat la pornirea NT8
                Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                {
                    BuildWindow();
                    _window.Show();
                }));

                // AUTO-START: exportam automat dupa 45 secunde (NT8 are timp sa se incarce)
                Thread autoThread = new Thread(() =>
                {
                    Thread.Sleep(45000); // 45s delay initial
                    Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        if (_btnStart != null && _btnStart.IsEnabled)
                        {
                            Log("⚡ AUTO-START: export automat la pornire NT8...");
                            OnStart(null, null);
                        }
                    }));

                    // PERIODIC RE-EXPORT: la fiecare AUTO_EXPORT_INTERVAL_MIN minute
                    while (true)
                    {
                        Thread.Sleep(AUTO_EXPORT_INTERVAL_MIN * 60 * 1000);
                        Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                        {
                            if (!_running) // nu porni dacă deja rulează
                            {
                                Log("🔄 AUTO-REFRESH: re-export periodic (" + AUTO_EXPORT_INTERVAL_MIN + " min)...");
                                OnStart(null, null);
                            }
                        }));
                    }
                });
                autoThread.IsBackground = true;
                autoThread.Start();
            }
        }

        private void BuildWindow()
        {
            _window = new NTWindow
            {
                Title  = "AutoExportNQ — Export date NQ/ES",
                Width  = 680,
                Height = 580,
            };

            StackPanel root = new StackPanel { Margin = new Thickness(12) };

            root.Children.Add(new Label
            {
                Content    = "AutoExportNQ — Export automat date istorice NQ/ES 1-minut",
                FontWeight = FontWeights.Bold,
                FontSize   = 14,
                Margin     = new Thickness(0, 0, 0, 8),
            });

            root.Children.Add(new Label { Content = "Folder output (unde se salvează fișierele CSV):" });

            _txtDir = new TextBox
            {
                Text   = OUTPUT_DIR,
                Margin = new Thickness(0, 0, 0, 4),
            };
            root.Children.Add(_txtDir);

            root.Children.Add(new Label
            {
                Content  = "Parallels shared folder: \\\\Mac\\Home\\Desktop\\Aladin\\",
                Foreground = System.Windows.Media.Brushes.Gray,
                FontSize = 11,
                Margin   = new Thickness(0, 0, 0, 10),
            });

            StackPanel btns = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                Margin      = new Thickness(0, 0, 0, 8),
            };

            _btnStart = new Button
            {
                Content    = "▶  START EXPORT",
                Width      = 160,
                Height     = 34,
                Margin     = new Thickness(0, 0, 10, 0),
                Background = System.Windows.Media.Brushes.DarkGreen,
                Foreground = System.Windows.Media.Brushes.White,
                FontWeight = FontWeights.Bold,
            };
            _btnStart.Click += OnStart;

            _btnStop = new Button
            {
                Content   = "⏹  STOP",
                Width     = 100,
                Height    = 34,
                IsEnabled = false,
            };
            _btnStop.Click += (s, ev) => { _stop = true; };

            btns.Children.Add(_btnStart);
            btns.Children.Add(_btnStop);
            root.Children.Add(btns);

            _progress = new ProgressBar
            {
                Minimum = 0,
                Maximum = SYMBOLS.Length,
                Value   = 0,
                Height  = 18,
                Margin  = new Thickness(0, 0, 0, 8),
            };
            root.Children.Add(_progress);

            root.Children.Add(new Label { Content = "Log:" });

            _txtLog = new TextBox
            {
                IsReadOnly                  = true,
                AcceptsReturn               = true,
                VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
                Height                      = 280,
                FontFamily                  = new System.Windows.Media.FontFamily("Consolas"),
                FontSize                    = 11,
                Background                  = System.Windows.Media.Brushes.Black,
                Foreground                  = System.Windows.Media.Brushes.LimeGreen,
            };
            root.Children.Add(_txtLog);

            _window.Content = root;
        }

        private void Log(string msg)
        {
            string line = string.Format("[{0:HH:mm:ss}] {1}\n", DateTime.Now, msg);
            Application.Current.Dispatcher.BeginInvoke(new Action(() =>
            {
                if (_txtLog == null) return;
                _txtLog.AppendText(line);
                _txtLog.ScrollToEnd();
            }));
        }

        private void SetProgress(int val)
        {
            Application.Current.Dispatcher.BeginInvoke(new Action(() =>
            {
                if (_progress != null) _progress.Value = val;
            }));
        }

        private void SetButtons(bool running)
        {
            Application.Current.Dispatcher.BeginInvoke(new Action(() =>
            {
                if (_btnStart != null) _btnStart.IsEnabled = !running;
                if (_btnStop  != null) _btnStop.IsEnabled  =  running;
            }));
        }

        private void OnStart(object sender, RoutedEventArgs e)
        {
            if (_running) return;
            _running = true;
            _stop    = false;
            SetButtons(true);

            string outDir = _txtDir.Text.Trim();
            Thread t = new Thread(() => RunExport(outDir));
            t.IsBackground = true;
            t.Start();
        }

        private void RunExport(string outDir)
        {
            try { Directory.CreateDirectory(outDir); }
            catch (Exception ex)
            {
                Log("EROARE folder: " + ex.Message);
                _running = false;
                SetButtons(false);
                return;
            }

            Log("=== START EXPORT ===");
            Log("Folder: " + outDir);
            Log("Total contracte: " + SYMBOLS.Length);
            Log("");

            int done = 0, skip = 0;

            for (int i = 0; i < SYMBOLS.Length; i++)
            {
                if (_stop) { Log("⏹ Oprit."); break; }
                SetProgress(i);
                ExportSymbol(SYMBOLS[i], outDir, ref done, ref skip);
                Thread.Sleep(300);
            }

            SetProgress(SYMBOLS.Length);
            Log("");
            Log("=== FINALIZAT ===");
            Log("Exportate: " + done + " | Sarite: " + skip);
            Log("Pasul urmator pe Mac Terminal:");
            Log("  bash ~/Desktop/Aladin/RUN_IMPORT_NQ.sh");

            _running = false;
            SetButtons(false);
        }

        private void ExportSymbol(string symbol, string outDir, ref int done, ref int skip)
        {
            string fileName = symbol.Replace(" ", "_") + ".Last.txt";
            string filePath = Path.Combine(outDir, fileName);

            // GAP FILL: nu skipăm — suprascrie fișierul existent
            if (File.Exists(filePath))
            {
                Log("  [OVERWRITE] " + symbol + " — suprascrie fișier existent");
            }

            Log("  [...] " + symbol + " — se solicita date...");

            Instrument instr = Instrument.GetInstrument(symbol);
            if (instr == null)
            {
                Log("  [MISS] " + symbol + " — negasit in NT8");
                skip++;
                return;
            }

            ManualResetEvent evt   = new ManualResetEvent(false);
            List<string>     lines = new List<string>();
            string           err   = null;

            // ROLLING 30 ZILE: trage ultimele 30 zile pentru a acoperi gap-uri
            BarsRequest req = new BarsRequest(instr,
                DateTime.Now.AddDays(-30).Date,
                DateTime.Now);

            req.BarsPeriod = new BarsPeriod
            {
                BarsPeriodType = BarsPeriodType.Minute,
                Value          = 1,
            };

            req.Request((request, errorCode, errorMessage) =>
            {
                if (errorCode != ErrorCode.NoError)
                {
                    err = errorCode.ToString() + ": " + errorMessage;
                    evt.Set();
                    return;
                }

                Bars bars = request.Bars;
                if (bars == null || bars.Count == 0)
                {
                    evt.Set();
                    return;
                }

                for (int j = 0; j < bars.Count; j++)
                {
                    DateTime ts = bars.GetTime(j);
                    double   o  = bars.GetOpen(j);
                    double   h  = bars.GetHigh(j);
                    double   l  = bars.GetLow(j);
                    double   c  = bars.GetClose(j);
                    long     v  = (long)bars.GetVolume(j);
                    lines.Add(string.Format("{0:yyyyMMdd HHmmss};{1};{2};{3};{4};{5}",
                        ts, o, h, l, c, v));
                }

                evt.Set();
            });

            evt.WaitOne(120000);

            if (err != null)
            {
                Log("  [ERR] " + symbol + " — " + err);
                skip++;
                return;
            }

            if (lines.Count == 0)
            {
                Log("  [SKIP] " + symbol + " — 0 bare disponibile");
                skip++;
                return;
            }

            File.WriteAllLines(filePath, lines);
            Log(string.Format("  [OK] {0} — {1:N0} bare salvate", symbol, lines.Count));
            done++;
        }
    }
}
