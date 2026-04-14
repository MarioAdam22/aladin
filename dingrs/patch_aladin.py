"""
python3 /Users/mario/Desktop/Aladin/patch_aladin.py
- Nimic nou in mario_rag.py - e deja corect
- Doar reverificam cu limita SL corecta (< 5% din pret)
"""
import sys, importlib
sys.path.insert(0, "/Users/mario/Desktop")
import mario_rag
importlib.reload(mario_rag)

print(f"{'Timestamp':<22} {'Dir':<6} {'Close':>7} {'SL':>7} {'TP':>7} {'distSL':>7} {'dist%':>6} {'RR':>5} {'OK'}")
print("-" * 80)
for ts in ["2023-11-10 15:30", "2019-04-18 09:30", "2022-03-10 09:30", "2020-03-20 14:00", "2021-06-15 09:30"]:
    r  = mario_rag.aladin_engine(ts)
    d  = r.get("trade_direction", "?")
    ri = r.get("risk", {})
    sl = ri.get("sl", 0)
    tp = ri.get("tp", 0)
    cl = r.get("close", 0)
    dsl = abs(cl - sl)
    dtp = abs(cl - tp)
    pct = dsl / cl * 100 if cl > 0 else 0
    rr  = round(dtp / dsl, 1) if dsl > 0 else 0
    ok_dir  = (sl < cl < tp) if d == "LONG" else (tp < cl < sl)
    ok_sane = 0.3 < pct < 5.0   # SL intre 0.3% si 5% din pret
    ok_rr   = rr >= 2.0
    ok = ok_dir and ok_sane and ok_rr
    flags = []
    if not ok_sane: flags.append(f"SL={pct:.1f}%!")
    if not ok_rr:   flags.append(f"RR={rr}!")
    if not ok_dir:  flags.append("DIR!")
    print(f"{ts:<22} {d:<6} {cl:>7.2f} {sl:>7.2f} {tp:>7.2f} {dsl:>7.2f} {pct:>5.1f}% {rr:>5} {'✅' if ok else '❌'} {' '.join(flags)}")

print()
print("SL sanity: intre 0.3% si 5% din pret = OK pentru QQQ intraday")
print("Daca toate OK → reporneste Streamlit si ruleaza backtest!")
