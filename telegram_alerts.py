"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ALADIN — Telegram Alerts Module                                            ║
║  telegram_alerts.py  |  Update #22 — Alertă la semnal DIAMOND             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Setup:
  1. Creează bot pe Telegram cu @BotFather → obții BOT_TOKEN
  2. Trimite un mesaj bot-ului, apoi accesează:
     https://api.telegram.org/bot<TOKEN>/getUpdates
     → găsești CHAT_ID
  3. Creează fișier .env sau setează variabile de mediu:
     TELEGRAM_BOT_TOKEN=your_token_here
     TELEGRAM_CHAT_ID=your_chat_id_here

Utilizare:
  from telegram_alerts import send_diamond_alert, send_signal_summary
  send_diamond_alert(result_dict)  # apelat din aladin_engine()
"""

import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("aladin-telegram")

# Configurare — citește din env vars sau fișier .env
BOT_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID       = os.getenv("TELEGRAM_CHAT_ID", "")

# Bot separat pentru alerte geopolitice (nu deranjează chat-ul principal Aladin)
GEO_BOT_TOKEN = os.getenv("TELEGRAM_GEO_BOT_TOKEN", "")
GEO_CHAT_ID   = os.getenv("TELEGRAM_GEO_CHAT_ID",   "")


def _load_env_config():
    """Încearcă să încarce .env dacă python-dotenv e disponibil."""
    global BOT_TOKEN, CHAT_ID, GEO_BOT_TOKEN, GEO_CHAT_ID
    if BOT_TOKEN and CHAT_ID and GEO_BOT_TOKEN:
        return
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(env_path, override=True)
        BOT_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "")
        CHAT_ID       = os.getenv("TELEGRAM_CHAT_ID", "")
        GEO_BOT_TOKEN = os.getenv("TELEGRAM_GEO_BOT_TOKEN", "")
        GEO_CHAT_ID   = os.getenv("TELEGRAM_GEO_CHAT_ID", "")
    except ImportError:
        pass


def send_telegram_message(message: str, parse_mode: str = "HTML") -> bool:
    """
    Trimite un mesaj Telegram.
    Returnează True dacă a reușit, False altfel.
    """
    _load_env_config()

    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("Telegram nu configurat (BOT_TOKEN/CHAT_ID lipsă). "
                       "Setează TELEGRAM_BOT_TOKEN și TELEGRAM_CHAT_ID în .env")
        return False

    try:
        import requests
        url  = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = {
            "chat_id":    CHAT_ID,
            "text":       message,
            "parse_mode": parse_mode,
        }
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code == 200:
            logger.info("✅ Telegram: mesaj trimis cu succes")
            return True
        else:
            logger.error(f"❌ Telegram error: {resp.status_code} — {resp.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Telegram exception: {e}")
        return False


def send_geo_telegram_message(message: str, parse_mode: str = "HTML") -> bool:
    """
    Trimite alertă geopolitică pe bot-ul separat de geo news.
    Folosește GEO_BOT_TOKEN în loc de BOT_TOKEN principal,
    astfel încât alertele geo nu deranjează chat-ul Aladin.
    """
    _load_env_config()
    _geo_chat = GEO_CHAT_ID or CHAT_ID
    if not _geo_chat:
        logger.warning("Telegram geo: CHAT_ID lipsă.")
        return False
    try:
        import requests
        url  = f"https://api.telegram.org/bot{GEO_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id":    _geo_chat,
            "text":       message,
            "parse_mode": parse_mode,
        }
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code == 200:
            logger.info("✅ Geo Telegram: mesaj trimis cu succes")
            return True
        else:
            logger.error(f"❌ Geo Telegram error: {resp.status_code} — {resp.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Geo Telegram exception: {e}")
        return False


def send_diamond_alert(result: dict) -> bool:
    """
    Update #22: Alertă Telegram când scoring engine returnează DIAMOND (>80%).
    Apelat automat din aladin_engine() când conviction == 'DIAMOND'.
    """
    conviction = result.get('conviction', '')
    if conviction != 'DIAMOND':
        return False

    score_pct  = round(result.get('score', 0) * 100, 1)
    direction  = result.get('trade_direction', 'LONG')
    regime     = result.get('regime', 'UNKNOWN')
    killzone   = result.get('killzone', 'Outside')
    verdict    = result.get('verdict', '')
    risk_obj   = result.get('risk', {})
    sl         = risk_obj.get('sl', 0)
    tp         = risk_obj.get('tp', 0)
    risk_usd   = risk_obj.get('risk_usd', 0)

    dir_emoji  = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
    now_str    = datetime.now().strftime("%Y-%m-%d %H:%M")

    message = f"""💎 <b>ALADIN DIAMOND SIGNAL</b>

📅 <b>{now_str}</b>
{dir_emoji} — Score: <b>{score_pct}%</b>

🎯 <b>Verdict:</b> {verdict}
⚡ <b>Regime:</b> {regime}
🕐 <b>Killzone:</b> {killzone or 'Outside'}

💰 <b>Risk:</b> ${risk_usd:.0f}
🛑 <b>SL:</b> {sl:.2f}
✅ <b>TP:</b> {tp:.2f}

<i>Aladin Quantum-ICT v6.8 | marioyear@yahoo.com</i>"""

    return send_telegram_message(message)


def send_signal_summary(result: dict) -> bool:
    """
    Trimite rezumat semnal (pentru toate nivelurile de conviction).
    """
    score_pct = round(result.get('score', 0) * 100, 1)
    direction = result.get('trade_direction', 'LONG')
    conviction = result.get('conviction', 'LOW')
    now_str   = datetime.now().strftime("%H:%M")

    dir_emoji = "🟢" if direction == "LONG" else "🔴"

    message = f"""⚛️ <b>Aladin Signal [{now_str}]</b>
{dir_emoji} {direction} | Score: {score_pct}% | {conviction}"""

    return send_telegram_message(message)


def send_daily_summary(stats: dict) -> bool:
    """
    Trimite rezumat zilnic de performanță.
    """
    total     = stats.get('total_signals', 0)
    win_rate  = stats.get('win_rate_est', 0)
    avg_score = stats.get('avg_score', 0)
    now_str   = datetime.now().strftime("%Y-%m-%d")

    message = f"""📊 <b>Aladin Daily Summary [{now_str}]</b>

🎯 Semnale totale: {total}
📈 Winrate estimat: {win_rate}%
⚡ Scor mediu: {avg_score}

<i>Aladin Quantum-ICT v6.8</i>"""

    return send_telegram_message(message)


def send_trade_executed(direction: str, score: float, sl: float, tp: float,
                        risk_usd: float, strategy: str = "", price: float = 0,
                        trades_today: int = 0, max_trades: int = 0,
                        is_scale_in: bool = False, scale_in_qty: int = 0,
                        scale_in_total_qty: int = 0, avg_entry: float = 0,
                        component_scores: dict = None, ict_signals: dict = None,
                        vp_context: dict = None, delta_exhaustion: str = "",
                        ict_ml_score: float = 0.0) -> bool:
    """
    Notificare când Aladin execută un trade real.
    Apelat din bridge_api._auto_execute după execuție.
    Fix v9.1: is_scale_in=True → afișăm SCALE IN (nu TRADE NOU)
    v10.0: + score breakdown, ICT signals, VP context, delta exhaustion
    """
    dir_emoji  = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
    now_str    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    strat_str  = f"\n📋 <b>Strategie:</b> {strategy}" if strategy else ""
    trades_str = f"\n📊 <b>Trade #{trades_today}/{max_trades} azi</b>" if max_trades else ""

    # Calculăm SL/TP în puncte și R potențial
    sl_pts  = abs(price - sl) if price and sl else 0
    tp_pts  = abs(tp - price) if price and tp else 0
    r_ratio = round(tp_pts / sl_pts, 1) if sl_pts > 0 else 0

    # ── Score breakdown pe componente ────────────────────────────────────────
    score_breakdown = ""
    if component_scores:
        cs = component_scores
        score_breakdown = (
            f"\n\n🧩 <b>Breakdown scor:</b>\n"
            f"   🤖 AI: <b>{cs.get('ai', 0)*100:.0f}%</b> | "
            f"📐 ICT: <b>{cs.get('ict', 0)*100:.0f}%</b> | "
            f"🌊 OF: <b>{cs.get('orderflow', 0)*100:.0f}%</b>\n"
            f"   📊 VP: <b>{cs.get('volume_profile', 0)*100:.0f}%</b> | "
            f"💪 RS: <b>{cs.get('rel_strength', 0)*100:.0f}%</b>"
        )

    # ── ICT semnale active ────────────────────────────────────────────────────
    ict_line = ""
    if ict_signals:
        def _tf(v): return "✅" if v else "❌"
        ict_line = (
            f"\n\n🏛 <b>ICT Signals:</b>\n"
            f"   H4:{_tf(ict_signals.get('h4'))} "
            f"H1:{_tf(ict_signals.get('h1'))} "
            f"M15:{_tf(ict_signals.get('m15'))} "
            f"KZ:{_tf(ict_signals.get('kz'))} "
            f"FVG:{_tf(ict_signals.get('fvg'))} "
            f"SMT:{_tf(ict_signals.get('smt'))}"
        )

    # ── Volume Profile context ────────────────────────────────────────────────
    vp_line = ""
    if vp_context:
        rvol    = vp_context.get('rvol', 0)
        shape   = vp_context.get('shape', '')
        poc_dist = vp_context.get('poc_dist', 0)
        poc_dir  = "deasupra" if poc_dist >= 0 else "sub"
        shape_emoji = {"P": "🅿️", "b": "🔵", "D": "🔻"}.get(shape, "")
        vp_line = (
            f"\n\n📊 <b>Volume Profile:</b>\n"
            f"   RVOL: <b>{rvol:.1f}x</b> | "
            f"Shape: {shape_emoji}<b>{shape}</b> | "
            f"Entry {poc_dir} POC cu <b>{abs(poc_dist):.1f} pts</b>"
        )

    # ── Delta exhaustion ─────────────────────────────────────────────────────
    exhaust_line = ""
    if delta_exhaustion and delta_exhaustion not in ("NONE", ""):
        exhaust_emoji = "🔴" if "SHORT" in delta_exhaustion else "🟢"
        exhaust_line  = f"\n⚡ <b>Delta Exhaustion:</b> {exhaust_emoji} {delta_exhaustion}"

    # ── ML Setup Score ────────────────────────────────────────────────────────
    ml_line = ""
    if ict_ml_score > 0:
        ml_emoji = "🟢" if ict_ml_score >= 0.40 else ("🟡" if ict_ml_score >= 0.25 else "🔴")
        ml_wr    = "~61%" if ict_ml_score >= 0.40 else ("~53%" if ict_ml_score >= 0.25 else "~35%")
        ml_line  = f"\n🤖 <b>ML Setup Score:</b> {ml_emoji} <b>{ict_ml_score:.2f}</b>  (WR estimat {ml_wr})"

    if is_scale_in:
        avg_str = f"\n📊 <b>Entry mediu:</b> <code>{avg_entry:.2f}</code>" if avg_entry > 0 else ""
        message = (
            f"➕ <b>ALADIN — SCALE IN</b>\n\n"
            f"📅 {now_str}\n"
            f"{dir_emoji} | +{scale_in_qty or 1} contract(e){strat_str}\n\n"
            f"💵 <b>Preț adăugat:</b> <code>{price:.2f}</code>{avg_str}\n"
            f"🛑 <b>SL:</b> <code>{sl:.2f}</code>  ({sl_pts:.0f} pts)\n"
            f"✅ <b>TP:</b> <code>{tp:.2f}</code>  ({tp_pts:.0f} pts)\n"
            f"📦 <b>Total poziție:</b> {scale_in_total_qty or 2} contracte\n"
            f"💰 <b>Risc total:</b> ${risk_usd:.0f}{trades_str}"
            f"{score_breakdown}{ict_line}{vp_line}{exhaust_line}\n\n"
            f"<i>🤖 Aladin Quantum-ICT</i>"
        )
    else:
        message = (
            f"⚡ <b>ALADIN — TRADE EXECUTAT</b>\n\n"
            f"📅 {now_str}\n"
            f"{dir_emoji} | Score: <b>{round(score, 1)}%</b>{strat_str}\n\n"
            f"💵 <b>Entry:</b> <code>{price:.2f}</code>\n"
            f"🛑 <b>SL:</b> <code>{sl:.2f}</code>  ({sl_pts:.0f} pts)\n"
            f"✅ <b>TP:</b> <code>{tp:.2f}</code>  ({tp_pts:.0f} pts)\n"
            f"📐 <b>R:R</b> 1:{r_ratio}\n"
            f"💰 <b>Risc:</b> ${risk_usd:.0f}{trades_str}"
            f"{score_breakdown}{ict_line}{vp_line}{exhaust_line}{ml_line}\n\n"
            f"<i>🤖 Aladin Quantum-ICT</i>"
        )

    return send_telegram_message(message)


def send_status_reply(state_snapshot: dict) -> bool:
    """
    Răspunde la /status sau /trade cu statusul curent al sistemului.
    Include PnL live, drawdown sesiune, max profit.
    """
    auto      = "✅ ON" if state_snapshot.get("autotrade") else "❌ OFF"
    mode      = "📄 PAPER" if state_snapshot.get("paper_mode") else "💰 LIVE"
    strat     = state_snapshot.get("strategy") or "—"
    score     = state_snapshot.get("score", 0)
    t_today   = state_snapshot.get("trades_today", 0)
    t_max     = state_snapshot.get("max_trades", 0)
    in_win    = "✅ în fereastră" if state_snapshot.get("in_window") else "⏳ în așteptare"
    signal    = state_snapshot.get("last_signal", "—")
    open_tr   = state_snapshot.get("open_trade")
    now_str   = datetime.now().strftime("%H:%M")

    # PnL sesiune
    realized_pnl  = state_snapshot.get("daily_profit_usd", 0.0) - state_snapshot.get("daily_loss_usd", 0.0)
    open_pnl      = state_snapshot.get("open_pnl_usd", 0.0)
    total_pnl     = realized_pnl + open_pnl
    max_profit    = state_snapshot.get("session_max_profit", 0.0)
    max_drawdown  = state_snapshot.get("session_max_drawdown", 0.0)

    pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
    pnl_sign  = "+" if total_pnl >= 0 else ""

    pnl_line = (
        f"\n\n💰 <b>PnL Sesiune:</b>\n"
        f"   Realizat: <b>${realized_pnl:+.0f}</b>\n"
        f"   Open:     <b>${open_pnl:+.0f}</b>\n"
        f"   {pnl_emoji} Total:    <b>${pnl_sign}{total_pnl:.0f}</b>\n"
        f"   📈 Max profit: <b>${max_profit:.0f}</b>\n"
        f"   📉 Max drawdown: <b>${max_drawdown:.0f}</b>"
    )

    # Trade deschis
    trade_line = ""
    if open_tr:
        d        = "🟢 LONG" if open_tr.get("direction") == "LONG" else "🔴 SHORT"
        entry    = open_tr.get("entry", 0)
        sl       = open_tr.get("sl", 0)
        tp       = open_tr.get("tp", 0)
        sl_pts   = abs(entry - sl) if entry and sl else 0
        tp_pts   = abs(tp - entry) if entry and tp else 0
        rr       = round(tp_pts / sl_pts, 1) if sl_pts > 0 else 0
        trade_line = (
            f"\n\n📌 <b>Trade deschis: {d}</b>\n"
            f"   Entry: <code>{entry:.2f}</code>\n"
            f"   SL: <code>{sl:.2f}</code> ({sl_pts:.0f}pts)\n"
            f"   TP: <code>{tp:.2f}</code> ({tp_pts:.0f}pts)\n"
            f"   R:R 1:{rr}"
        )
    else:
        trade_line = "\n\n💤 <b>Niciun trade deschis</b>"

    strat_line = f"\n📋 Strategie: {strat}" if strat and strat != "—" else ""

    message = f"""📊 <b>Aladin Status [{now_str}]</b>

🤖 AutoTrade: {auto}
{mode}{strat_line}
⏰ Sesiune: {in_win}
📊 Trades azi: {t_today}/{t_max}

⚡ Scor live: <b>{score:.1f}%</b>
🧭 Semnal: {signal}{trade_line}{pnl_line}"""

    return send_telegram_message(message)


async def telegram_poll_loop(get_state_fn, interval: int = 5, command_callback=None):
    """
    Polling loop async — verifică mesajele noi și răspunde la comenzi.
    get_state_fn() returnează un dict cu statusul curent al sistemului.
    command_callback(cmd: str) → opțional, handler pentru comenzi speciale.
    Pornit ca background task în bridge_api.on_startup().
    Comenzi suportate: /status /trade /help /geo on /geo off
    """
    import asyncio
    _load_env_config()
    if not BOT_TOKEN or not CHAT_ID:
        return

    import requests as _req
    offset = 0
    base   = f"https://api.telegram.org/bot{BOT_TOKEN}"

    while True:
        try:
            r = _req.get(f"{base}/getUpdates", params={"offset": offset, "timeout": 3}, timeout=6)
            if r.status_code == 200:
                updates = r.json().get("result", [])
                for upd in updates:
                    offset = upd["update_id"] + 1
                    msg    = upd.get("message", {})
                    text   = msg.get("text", "").strip().lower()
                    chat   = str(msg.get("chat", {}).get("id", ""))

                    # Răspunde doar la CHAT_ID-ul tău
                    if chat != str(CHAT_ID):
                        continue

                    if text in ("/status", "/trade", "status", "trade", "status?", "ce faci"):
                        snap = get_state_fn()
                        send_status_reply(snap)

                    elif text in ("/geo on", "geo on"):
                        if command_callback:
                            command_callback("geo_on")
                        else:
                            send_telegram_message("🌍 <b>GEO RISK MODE</b>: nicio funcție callback setată.")

                    elif text in ("/geo off", "geo off"):
                        if command_callback:
                            command_callback("geo_off")
                        else:
                            send_telegram_message("✅ <b>GEO RISK MODE</b>: nicio funcție callback setată.")

                    elif text in ("/geo", "geo"):
                        snap = get_state_fn()
                        geo_active = snap.get("geo_risk_active", False)
                        status_icon = "🔴 ACTIV" if geo_active else "✅ INACTIV"
                        send_telegram_message(
                            f"🌍 <b>GEO RISK MODE:</b> {status_icon}\n\n"
                            f"Utilizare:\n"
                            f"/geo on — activează manual\n"
                            f"/geo off — dezactivează manual\n\n"
                            f"<i>Se activează automat la știri geopolitice critice</i>"
                        )

                    elif text in ("/help", "help", "comenzi"):
                        send_telegram_message(
                            "🤖 <b>Comenzi disponibile:</b>\n\n"
                            "/status — status curent sistem\n"
                            "/trade — trade deschis + statistici\n"
                            "/geo — status Geo Risk Mode\n"
                            "/geo on — activează Geo Risk manual\n"
                            "/geo off — dezactivează Geo Risk\n"
                            "/help — această listă"
                        )
        except Exception:
            pass
        await asyncio.sleep(interval)


def send_trade_closed(direction: str, entry: float, exit_price: float,
                      pnl_usd: float, result: str, r_mult: float = 0.0,
                      exit_reason: str = "", strategy: str = "",
                      daily_net: float = 0.0, duration_min: float = 0.0,
                      mae_pts: float = 0.0, mfe_pts: float = 0.0) -> bool:
    """
    Notificare Telegram când Aladin iese dintr-un trade (TP, SL, trailing, time exit).
    v10.0: + durată trade, MAE (max adverse excursion), MFE (max favorable excursion)
    """
    now_str    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dir_emoji  = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
    strat_str  = f"\n📋 <b>Strategie:</b> {strategy}" if strategy else ""

    if result == "WIN":
        result_emoji = "✅ WIN"
    elif result == "LOSS":
        result_emoji = "❌ LOSS"
    else:
        result_emoji = "⚖️ BE"

    pnl_sign   = "+" if pnl_usd >= 0 else ""
    r_str      = f"  ({pnl_sign}{r_mult:.2f}R)" if r_mult != 0 else ""
    reason_str = f"\n🚪 <b>Motiv ieșire:</b> {exit_reason}" if exit_reason else ""
    daily_str  = f"\n💼 <b>Net zi:</b> <code>${daily_net:+.0f}</code>" if daily_net != 0 else ""

    # ── Durată trade ─────────────────────────────────────────────────────────
    dur_str = ""
    if duration_min > 0:
        if duration_min < 60:
            dur_str = f"\n⏱ <b>Durată:</b> {int(duration_min)} min"
        else:
            dur_str = f"\n⏱ <b>Durată:</b> {duration_min/60:.1f} ore"

    # ── MAE / MFE — cât a mers contra ta și cât a mers în favoarea ta ────────
    excursion_str = ""
    if mae_pts > 0 or mfe_pts > 0:
        mae_str = f"📉 MAE: <b>{mae_pts:.0f} pts</b> contra" if mae_pts > 0 else ""
        mfe_str = f"📈 MFE: <b>{mfe_pts:.0f} pts</b> favorabil" if mfe_pts > 0 else ""
        parts   = [p for p in [mae_str, mfe_str] if p]
        excursion_str = "\n" + " | ".join(parts) if parts else ""

    message = (
        f"{result_emoji} <b>ALADIN — TRADE ÎNCHIS</b>\n\n"
        f"📅 {now_str}\n"
        f"{dir_emoji}{strat_str}\n\n"
        f"💵 <b>Entry:</b> <code>{entry:.2f}</code>\n"
        f"🏁 <b>Exit:</b>  <code>{exit_price:.2f}</code>\n"
        f"💰 <b>PnL:</b>   <code>${pnl_sign}{pnl_usd:.0f}{r_str}</code>"
        f"{reason_str}{daily_str}{dur_str}{excursion_str}\n\n"
        f"<i>🤖 Aladin Quantum-ICT</i>"
    )

    return send_telegram_message(message)


def send_trade_skipped_summary(reason: str, score: float) -> bool:
    """
    Opțional: notificare când un semnal bun e blocat (ex: volatilitate).
    Dezactivat implicit — activează manual dacă vrei.
    """
    return False  # dezactivat


# ─── NEWS ALERT: 15 minute înainte de eveniment ───────────────────────────────

# Probabilități istorice orientative per eveniment (% de surpriză pozitivă/negativă/inline)
_NEWS_PROBABILITIES = {
    "NFP":             {"bullish": 45, "bearish": 35, "inline": 20,
                        "desc": "Non-Farm Payrolls — cel mai volatil eveniment lunar pentru NQ"},
    "FOMC":            {"bullish": 35, "bearish": 40, "inline": 25,
                        "desc": "Fed Rate Decision — poate schimba direcția zilei"},
    "CPI":             {"bullish": 40, "bearish": 40, "inline": 20,
                        "desc": "Inflație — surpriză la scădere = bullish NQ, surpriză la creștere = bearish"},
    "PPI":             {"bullish": 42, "bearish": 38, "inline": 20,
                        "desc": "Producer Price Index — indicator leading pentru CPI"},
    "PCE":             {"bullish": 42, "bearish": 38, "inline": 20,
                        "desc": "PCE Deflator — metrica preferată a Fed pentru inflație"},
    "JOLTS":           {"bullish": 38, "bearish": 42, "inline": 20,
                        "desc": "Job Openings — piață muncă puternică = Fed mai hawkish = bearish NQ"},
    "ISM Manufacturing":{"bullish": 45, "bearish": 35, "inline": 20,
                        "desc": "PMI Manufacturier — peste 50 = expansiune"},
    "ISM Services":    {"bullish": 45, "bearish": 35, "inline": 20,
                        "desc": "PMI Servicii — sector dominant în SUA"},
    "Retail Sales":    {"bullish": 48, "bearish": 32, "inline": 20,
                        "desc": "Consum retail — surpriză pozitivă = creștere economică = bullish NQ"},
    "GDP":             {"bullish": 45, "bearish": 35, "inline": 20,
                        "desc": "Creștere economică — surprise pozitivă = bullish NQ/ES"},
    "ADP Employment":  {"bullish": 43, "bearish": 37, "inline": 20,
                        "desc": "Angajări private — pre-cursor NFP"},
    "Consumer Confidence": {"bullish": 47, "bearish": 33, "inline": 20,
                        "desc": "Sentiment consumator — indicat leading pt cheltuieli"},
    "Jobless Claims":  {"bullish": 40, "bearish": 40, "inline": 20,
                        "desc": "Cereri șomaj săptămânale — impact moderat"},
}


def send_news_alert_15min(event_name: str, release_time: str, symbol: str = "NQ") -> bool:
    """
    Alertă Telegram cu 15 minute înainte de un eveniment economic major.
    Include probabilități istorice de surpriză bullish/bearish/inline.
    """
    probs = _NEWS_PROBABILITIES.get(event_name, {"bullish": 40, "bearish": 40, "inline": 20, "desc": "Eveniment economic"})

    bull_bar = "🟢" * (probs["bullish"] // 10) + "⬜" * (10 - probs["bullish"] // 10)
    bear_bar = "🔴" * (probs["bearish"] // 10) + "⬜" * (10 - probs["bearish"] // 10)
    inline_bar = "🟡" * (probs["inline"] // 10) + "⬜" * (10 - probs["inline"] // 10)

    msg = (
        f"⏰ <b>NEWS ALERT — 15 MIN</b>\n\n"
        f"📅 <b>{event_name}</b> la <b>{release_time} UTC</b>\n"
        f"📊 <i>{probs['desc']}</i>\n\n"
        f"<b>Probabilități istorice ({symbol}):</b>\n"
        f"🟢 Bullish: <b>{probs['bullish']}%</b>  {bull_bar}\n"
        f"🔴 Bearish: <b>{probs['bearish']}%</b>  {bear_bar}\n"
        f"🟡 Inline:  <b>{probs['inline']}%</b>  {inline_bar}\n\n"
        f"⚡ <b>Aladin intră în News Trade Mode după 2 min de la release</b>\n"
        f"🛑 Trading blocat în fereastra spike ({release_time} + 2 min)\n\n"
        f"<i>🤖 Aladin Engine — Auto Alert</i>"
    )
    return send_telegram_message(msg)


def send_circuit_breaker_alert(reason: str, details: str, daily_loss: float = 0.0,
                                consecutive: int = 0, account_drawdown_pct: float = 0.0) -> bool:
    """
    Alertă Telegram când robotul se oprește din cauza unui circuit breaker.
    Motive: daily loss limit, consecutive losses, account drawdown, profit target hit.
    """
    emoji_map = {
        "daily_loss":         "🛑",
        "consecutive_losses": "⛔",
        "account_drawdown":   "💀",
        "profit_target":      "🏆",
        "forced_pause":       "⏸️",
    }
    emoji = emoji_map.get(reason, "⚠️")

    reason_labels = {
        "daily_loss":         "Daily Loss Limit atins",
        "consecutive_losses": "Losses Consecutive — Pauză forțată",
        "account_drawdown":   "Account Drawdown Critic",
        "profit_target":      "Profit Target Zilnic Atins",
        "forced_pause":       "Pauză Forțată Activată",
    }
    label = reason_labels.get(reason, reason)

    msg = (
        f"{emoji} <b>CIRCUIT BREAKER ACTIVAT</b>\n\n"
        f"<b>Motiv:</b> {label}\n"
        f"<b>Detalii:</b> {details}\n\n"
    )

    if daily_loss > 0:
        msg += f"📉 <b>Pierdere zilnică:</b> -${daily_loss:.0f}\n"
    if consecutive > 0:
        msg += f"🔴 <b>Losses consecutive:</b> {consecutive}\n"
    if account_drawdown_pct > 0:
        msg += f"💀 <b>Drawdown cont:</b> -{account_drawdown_pct:.1f}%\n"

    msg += (
        f"\n⏹️ <b>Robotul s-a OPRIT automat</b>\n"
        f"🔄 Resetul va fi posibil mâine la deschidere\n\n"
        f"<i>🤖 Aladin Engine — Risk Manager</i>"
    )
    return send_telegram_message(msg)


def send_geo_alert(headline: str, source: str, severity: str, keywords_found: list,
                   geo_risk_active: bool = True) -> bool:
    """
    Alertă Telegram pentru știri geopolitice/politice care pot mișca piața.
    severity: "CRITICAL" | "HIGH" | "MEDIUM"
    """
    severity_cfg = {
        "CRITICAL": ("🚨", "ALERTĂ CRITICĂ",  "Trading BLOCAT automat"),
        "HIGH":     ("⚠️",  "IMPACT RIDICAT",   "Aladin în mod prudență — scor +10% necesar"),
        "MEDIUM":   ("👁️",  "DE URMĂRIT",       "Aladin monitorizează — impact posibil"),
    }
    emoji, label, action = severity_cfg.get(severity, ("⚠️", "ȘTIRE", "Monitorizare"))
    kw_str = " · ".join(f"<b>{k}</b>" for k in keywords_found[:5])

    msg = (
        f"{emoji} <b>GEO NEWS — {label}</b>\n\n"
        f"📰 {headline}\n"
        f"🌐 <i>Sursă: {source}</i>\n\n"
        f"🔍 Keywords: {kw_str}\n\n"
        f"📊 <b>Impact piață:</b> {action}\n"
        f"{'🛑 <b>Robotul a redus expunerea automat</b>' if geo_risk_active else ''}\n\n"
        f"<i>🤖 Aladin — Geo News Monitor</i>"
    )
    return send_geo_telegram_message(msg)


def send_geo_risk_update(active: bool, reason: str = "") -> bool:
    """Notificare când geo_risk_mode se activează sau dezactivează manual."""
    if active:
        msg = (
            f"🌍 <b>GEO RISK MODE ACTIVAT</b>\n\n"
            f"Motiv: {reason}\n\n"
            f"⚡ Aladin operează cu prag ridicat (+10% scor necesar)\n"
            f"📉 Sizing redus automat cu 30%\n\n"
            f"Dezactivează manual: /geo off\n"
            f"<i>🤖 Aladin — Risk Manager</i>"
        )
    else:
        msg = (
            f"✅ <b>GEO RISK MODE DEZACTIVAT</b>\n\n"
            f"{reason}\n\n"
            f"Aladin revine la parametri normali.\n"
            f"<i>🤖 Aladin — Risk Manager</i>"
        )
    return send_geo_telegram_message(msg)


def send_daily_report(snapshot: dict) -> bool:
    """
    Raport zilnic automat trimis la închiderea sesiunii NY (22:00 UTC).
    snapshot = dict cu statisticile zilei din bridge_api.
    Chei așteptate:
      trades_today, wins, losses, pnl_usd, best_trade, worst_trade,
      win_rate, avg_score, circuit_open, geo_risk_active, geo_sentiment,
      consecutive_losses, consecutive_wins, daily_loss_usd, daily_profit_usd,
      date_str, sharpe_day, best_hour, top_component, pnl_list
    """
    date_str     = snapshot.get("date_str", "—")
    trades       = snapshot.get("trades_today", 0)
    wins         = snapshot.get("wins", 0)
    losses       = snapshot.get("losses", 0)
    pnl          = snapshot.get("pnl_usd", 0.0)
    best         = snapshot.get("best_trade", 0.0)
    worst        = snapshot.get("worst_trade", 0.0)
    win_rate     = snapshot.get("win_rate", 0.0)
    avg_score    = snapshot.get("avg_score", 0.0)
    circuit      = snapshot.get("circuit_open", False)
    geo_active   = snapshot.get("geo_risk_active", False)
    geo_sent     = snapshot.get("geo_sentiment", "NEUTRAL")
    cons_losses  = snapshot.get("consecutive_losses", 0)
    cons_wins    = snapshot.get("consecutive_wins", 0)
    daily_loss   = snapshot.get("daily_loss_usd", 0.0)
    daily_profit = snapshot.get("daily_profit_usd", 0.0)
    sharpe_day   = snapshot.get("sharpe_day", None)
    best_hour    = snapshot.get("best_hour", "")       # ex: "10:00-11:00 (2W 0L)"
    top_comp     = snapshot.get("top_component", "")   # ex: "ict (avg 0.82)"

    # PnL emoji
    pnl_emoji = "🟢" if pnl >= 0 else "🔴"
    pnl_str   = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

    # Win rate bar vizual
    wr_pct    = int(win_rate * 100)
    wr_filled = wr_pct // 10
    wr_bar    = "🟢" * wr_filled + "⬜" * (10 - wr_filled)

    # Scor mediu bar
    sc_filled = int(avg_score) // 10
    sc_bar    = "🔵" * sc_filled + "⬜" * (10 - sc_filled)

    # Status circuit breaker
    cb_line = "🔴 CIRCUIT BREAKER ACTIV" if circuit else "✅ Nicio limitare activă"

    # Geo status
    geo_line = ""
    if geo_active:
        geo_line = f"🌍 GEO RISK activ | Sentiment: {geo_sent}\n"

    # Verdict zi
    if trades == 0:
        verdict_day = "📭 Nicio tranzacție azi"
    elif pnl > 0:
        verdict_day = f"💚 Zi profitabilă — {wins}W / {losses}L"
    elif pnl < 0:
        verdict_day = f"🔴 Zi în pierdere — {wins}W / {losses}L"
    else:
        verdict_day = f"🟡 Zi neutră — {wins}W / {losses}L"

    # ── Streak curent ─────────────────────────────────────────────────────────
    streak_line = ""
    if cons_wins > 1:
        streak_line = f"\n🔥 <b>Streak:</b> {cons_wins} WIN-uri la rând"
    elif cons_losses > 1:
        streak_line = f"\n❄️ <b>Streak:</b> {cons_losses} LOSS-uri la rând — fii atent"

    # ── Sharpe zilnic ─────────────────────────────────────────────────────────
    sharpe_line = ""
    if sharpe_day is not None and trades >= 2:
        sh_emoji = "🟢" if sharpe_day >= 1.0 else "🟡" if sharpe_day >= 0 else "🔴"
        sharpe_line = f"\n📐 <b>Sharpe zi:</b> {sh_emoji} <b>{sharpe_day:.2f}</b>"

    # ── Cea mai bună oră ──────────────────────────────────────────────────────
    best_hour_line = ""
    if best_hour:
        best_hour_line = f"\n⏰ <b>Ora cea mai bună:</b> {best_hour}"

    # ── Componenta cu cel mai mare impact ─────────────────────────────────────
    top_comp_line = ""
    if top_comp:
        top_comp_line = f"\n🏅 <b>Top componentă:</b> {top_comp}"

    msg = (
        f"📊 <b>RAPORT ZILNIC ALADIN</b> — {date_str}\n"
        f"{'─' * 30}\n\n"
        f"{verdict_day}\n\n"
        f"<b>📈 Tranzacții:</b> {trades} total | {wins} WIN | {losses} LOSS\n"
        f"<b>💰 PnL Zi:</b> {pnl_emoji} <b>{pnl_str}</b>\n"
        f"<b>🏆 Best trade:</b>  +${best:.2f}\n"
        f"<b>💀 Worst trade:</b> -${abs(worst):.2f}\n\n"
        f"<b>🎯 Win Rate:</b> {wr_pct}%  {wr_bar}\n"
        f"<b>🧠 Scor mediu:</b> {avg_score:.1f}%  {sc_bar}\n"
        f"{sharpe_line}{streak_line}{best_hour_line}{top_comp_line}\n\n"
        f"<b>⛔ Circuit Breaker:</b> {cb_line}\n"
        f"<b>📉 Pierdere zi:</b> ${daily_loss:.2f} | "
        f"<b>Profit zi:</b> ${daily_profit:.2f}\n"
        f"<b>🔁 Loss consecutiv:</b> {cons_losses}\n"
        f"{geo_line}\n"
        f"<i>🤖 Aladin — Raport automat 22:00 UTC</i>"
    )
    return send_telegram_message(msg)


def send_weekly_report(snapshot: dict) -> bool:
    """
    Raport săptămânal trimis duminică seara (~21:00 UTC).
    snapshot = dict cu statisticile săptămânii agregate din bridge_api.
    Chei așteptate:
      week_str, total_trades, wins, losses, pnl_usd, win_rate,
      avg_score, sharpe_week, best_day, worst_day, pnl_per_day (dict),
      top_component, best_hour_week
    """
    week_str    = snapshot.get("week_str", "—")
    trades      = snapshot.get("total_trades", 0)
    wins        = snapshot.get("wins", 0)
    losses      = snapshot.get("losses", 0)
    pnl         = snapshot.get("pnl_usd", 0.0)
    win_rate    = snapshot.get("win_rate", 0.0)
    avg_score   = snapshot.get("avg_score", 0.0)
    sharpe_w    = snapshot.get("sharpe_week", None)
    best_day    = snapshot.get("best_day", "")    # ex: "Marți +$320"
    worst_day   = snapshot.get("worst_day", "")   # ex: "Joi -$180"
    pnl_per_day = snapshot.get("pnl_per_day", {}) # {"Luni": 120, "Marți": 320, ...}
    top_comp    = snapshot.get("top_component", "")
    best_hour_w = snapshot.get("best_hour_week", "")

    pnl_emoji = "🟢" if pnl >= 0 else "🔴"
    pnl_str   = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
    wr_pct    = int(win_rate * 100)
    wr_filled = wr_pct // 10
    wr_bar    = "🟢" * wr_filled + "⬜" * (10 - wr_filled)

    # Sharpe săptămânal
    sharpe_line = ""
    if sharpe_w is not None and trades >= 3:
        sh_emoji    = "🟢" if sharpe_w >= 1.0 else "🟡" if sharpe_w >= 0 else "🔴"
        sharpe_line = f"\n📐 <b>Sharpe săptămână:</b> {sh_emoji} <b>{sharpe_w:.2f}</b>"

    # PnL per zi — mini grafic
    day_lines = ""
    day_order = ["Luni", "Marți", "Miercuri", "Joi", "Vineri"]
    for day in day_order:
        if day in pnl_per_day:
            v = pnl_per_day[day]
            emoji = "🟢" if v > 0 else "🔴" if v < 0 else "⬜"
            day_lines += f"   {emoji} {day}: <code>${v:+.0f}</code>\n"

    best_day_line  = f"\n🏆 <b>Cea mai bună zi:</b> {best_day}" if best_day else ""
    worst_day_line = f"\n💀 <b>Cea mai slabă zi:</b> {worst_day}" if worst_day else ""
    top_comp_line  = f"\n🏅 <b>Top componentă:</b> {top_comp}" if top_comp else ""
    best_hour_line = f"\n⏰ <b>Ora cea mai bună:</b> {best_hour_w}" if best_hour_w else ""

    # Verdict săptămână
    if trades == 0:
        verdict = "📭 Nicio tranzacție săptămâna asta"
    elif pnl > 0:
        verdict = f"💚 Săptămână profitabilă — {wins}W / {losses}L"
    elif pnl < 0:
        verdict = f"🔴 Săptămână în pierdere — {wins}W / {losses}L"
    else:
        verdict = f"🟡 Săptămână neutră — {wins}W / {losses}L"

    msg = (
        f"📅 <b>RAPORT SĂPTĂMÂNAL ALADIN</b>\n"
        f"📆 {week_str}\n"
        f"{'─' * 30}\n\n"
        f"{verdict}\n\n"
        f"<b>📈 Tranzacții:</b> {trades} | {wins}W / {losses}L\n"
        f"<b>💰 PnL Total:</b> {pnl_emoji} <b>{pnl_str}</b>\n"
        f"<b>🎯 Win Rate:</b> {wr_pct}%  {wr_bar}\n"
        f"<b>🧠 Scor mediu:</b> {avg_score:.1f}%"
        f"{sharpe_line}{best_day_line}{worst_day_line}\n\n"
        f"<b>📊 PnL pe zile:</b>\n{day_lines}"
        f"{top_comp_line}{best_hour_line}\n\n"
        f"<i>🤖 Aladin — Raport automat Duminică 21:00 UTC</i>"
    )
    return send_telegram_message(msg)


if __name__ == "__main__":
    # Test rapid
    print("🔔 Test Telegram Alert...")
    test_result = {
        'conviction': 'DIAMOND',
        'score': 0.87,
        'trade_direction': 'LONG',
        'regime': 'TRENDING UP',
        'killzone': 'London Open',
        'verdict': '💎 SNIPER ENTRY CONFIRMED — DIAMOND',
        'risk': {'sl': 480.50, 'tp': 485.00, 'risk_usd': 150},
    }
    ok = send_diamond_alert(test_result)
    print(f"   Rezultat: {'✅ Trimis' if ok else '❌ Eșuat (verifică .env)'}")
