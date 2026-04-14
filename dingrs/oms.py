"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ALADIN OMS — Order Management System                                        ║
║  Updates #44 (OMS), #45 (Partial fills), #46 (Slippage monitoring)          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Update #44: Gestionează ordinele active: entry, SL, TP, modificări, anulări.
Update #45: Partial fills handling — gestionează poziții parțiale.
Update #46: Execution slippage monitoring — trackează diferența semnal vs execuție.
"""

import json
import os
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

OMS_PATH          = "/Users/mario/Desktop/Aladin/aladin_open_trades.json"
SLIPPAGE_LOG_PATH = "/Users/mario/Desktop/Aladin/aladin_slippage_log.csv"
SLIPPAGE_ALERT_PCT = 0.10  # alertă dacă slippage >0.10%


# =============================================================================
# UPDATE #44 — ORDER MANAGEMENT SYSTEM
# =============================================================================
class AladinOMS:
    """
    Order Management System (OMS) pentru Aladin.
    Trackează ciclul de viață complet al fiecărui ordin:
    PENDING → FILLED (parțial sau total) → CLOSED (SL/TP/Manual)
    """

    def __init__(self, oms_path: str = OMS_PATH):
        self.oms_path = oms_path
        self._load()

    def _load(self):
        """Încarcă ordinele active din fișier JSON."""
        if os.path.exists(self.oms_path):
            try:
                with open(self.oms_path, 'r') as f:
                    self.orders = json.load(f)
            except Exception:
                self.orders = []
        else:
            self.orders = []

    def _save(self):
        """Salvează ordinele active în fișier JSON."""
        try:
            with open(self.oms_path, 'w') as f:
                json.dump(self.orders, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"OMS save error: {e}")

    def open_order(
        self,
        signal_ts:     str,
        direction:     str,
        signal_price:  float,
        sl:            float,
        tp:            float,
        risk_usd:      float,
        conviction:    str = "HIGH",
        instrument:    str = "QQQ",
        score:         float = 0.0,
    ) -> dict:
        """
        Update #44: Deschide un ordin nou.
        Status inițial: PENDING (nu e executat încă).
        """
        order = {
            "order_id":     f"ALN-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "status":       "PENDING",
            "direction":    direction,
            "instrument":   instrument,
            "signal_ts":    signal_ts,
            "signal_price": round(signal_price, 4),
            "entry_price":  None,   # setat la fill
            "sl":           round(sl, 4),
            "tp":           round(tp, 4),
            "risk_usd":     round(risk_usd, 2),
            "conviction":   conviction,
            "score":        round(score, 4),
            # Update #45: Partial fills
            "filled_qty":   0.0,
            "total_qty":    1.0,    # normalizat la 1 lot
            "fills":        [],     # lista fill-urilor parțiale
            # Update #46: Slippage
            "slippage_pts": None,
            "slippage_pct": None,
            "created_at":   datetime.now().isoformat(),
            "closed_at":    None,
            "close_reason": None,
            "pnl_usd":      None,
        }
        self.orders.append(order)
        self._save()
        logger.info(f"   📋 OMS: Ordin deschis {order['order_id']} {direction} {instrument} @ signal {signal_price}")
        return order

    def fill_order(
        self,
        order_id:      str,
        fill_price:    float,
        fill_qty:      float = 1.0,
        fill_ts:       str   = None,
    ) -> Optional[dict]:
        """
        Update #44 + #45: Marchează un ordin ca executat (total sau parțial).
        Calculează slippage față de prețul semnalului.
        """
        order = self._find(order_id)
        if not order:
            logger.warning(f"OMS fill: ordin {order_id} negăsit")
            return None

        fill_ts = fill_ts or datetime.now().isoformat()

        # Update #45: Partial fill
        fill_record = {
            "fill_price": round(fill_price, 4),
            "fill_qty":   round(fill_qty, 4),
            "fill_ts":    fill_ts,
        }
        order["fills"].append(fill_record)
        order["filled_qty"] = round(order["filled_qty"] + fill_qty, 4)

        # Calculează preț mediu de intrare
        total_cost = sum(f["fill_price"] * f["fill_qty"] for f in order["fills"])
        total_qty  = sum(f["fill_qty"] for f in order["fills"])
        order["entry_price"] = round(total_cost / total_qty, 4) if total_qty > 0 else fill_price

        # Update #46: Slippage monitoring
        if order["signal_price"] and order["entry_price"]:
            slip_pts = abs(order["entry_price"] - order["signal_price"])
            slip_pct = slip_pts / order["signal_price"] * 100
            order["slippage_pts"] = round(slip_pts, 4)
            order["slippage_pct"] = round(slip_pct, 4)

            if slip_pct > SLIPPAGE_ALERT_PCT:
                logger.warning(f"   ⚠️ SLIPPAGE ALERT: {order_id} — {slip_pct:.3f}% > {SLIPPAGE_ALERT_PCT}%")
                _log_slippage(order)

        # Status
        if order["filled_qty"] >= order["total_qty"] * 0.99:
            order["status"] = "FILLED"
        else:
            order["status"] = "PARTIAL"

        self._save()
        logger.info(f"   ✅ OMS: Fill {order_id} @ {fill_price} (qty {fill_qty}) slippage={order.get('slippage_pct', 0):.3f}%")
        return order

    def close_order(
        self,
        order_id:     str,
        close_price:  float,
        close_reason: str = "TP",  # TP / SL / MANUAL / TIME_STOP
        close_ts:     str = None,
    ) -> Optional[dict]:
        """Update #44: Închide un ordin și calculează P&L."""
        order = self._find(order_id)
        if not order:
            return None

        close_ts = close_ts or datetime.now().isoformat()
        order["status"]       = "CLOSED"
        order["closed_at"]    = close_ts
        order["close_reason"] = close_reason

        # Calculează P&L
        if order["entry_price"] and close_price:
            entry = order["entry_price"]
            risk  = order["risk_usd"]
            sl    = order["sl"]
            tp    = order["tp"]

            if order["direction"] == "LONG":
                if close_reason == "TP":
                    order["pnl_usd"] = round(risk * abs(tp - entry) / abs(entry - sl), 2)
                else:
                    order["pnl_usd"] = round(-risk, 2)
            else:  # SHORT
                if close_reason == "TP":
                    order["pnl_usd"] = round(risk * abs(entry - tp) / abs(sl - entry), 2)
                else:
                    order["pnl_usd"] = round(-risk, 2)

        # Scoate din lista de ordine active
        self.orders = [o for o in self.orders if o["order_id"] != order_id]
        self._save()

        logger.info(f"   🔒 OMS: Ordin {order_id} închis ({close_reason}) P&L: ${order.get('pnl_usd', 0)}")
        return order

    def modify_order(self, order_id: str, new_sl: float = None, new_tp: float = None) -> Optional[dict]:
        """Update #44: Modifică SL/TP al unui ordin activ."""
        order = self._find(order_id)
        if not order:
            return None
        if new_sl:
            order["sl"] = round(new_sl, 4)
        if new_tp:
            order["tp"] = round(new_tp, 4)
        self._save()
        logger.info(f"   ✏️  OMS: Ordin {order_id} modificat SL={new_sl} TP={new_tp}")
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Update #44: Anulează un ordin PENDING."""
        order = self._find(order_id)
        if not order or order["status"] != "PENDING":
            return False
        self.orders = [o for o in self.orders if o["order_id"] != order_id]
        self._save()
        logger.info(f"   ❌ OMS: Ordin {order_id} anulat")
        return True

    def get_open_orders(self) -> list:
        """Returnează toate ordinele active (PENDING / PARTIAL / FILLED)."""
        return [o for o in self.orders if o["status"] in ("PENDING", "PARTIAL", "FILLED")]

    def get_stats(self) -> dict:
        """Statistici OMS: număr ordine, slippage mediu, etc."""
        open_orders = self.get_open_orders()
        return {
            "n_open":           len(open_orders),
            "total_risk_usd":   round(sum(o["risk_usd"] for o in open_orders), 2),
            "avg_slippage_pct": round(
                float(np.mean([o["slippage_pct"] for o in open_orders if o.get("slippage_pct") is not None])),
                4
            ) if any(o.get("slippage_pct") for o in open_orders) else 0.0,
        }

    def _find(self, order_id: str) -> Optional[dict]:
        for o in self.orders:
            if o["order_id"] == order_id:
                return o
        return None


# =============================================================================
# UPDATE #46 — SLIPPAGE LOG
# =============================================================================
def _log_slippage(order: dict):
    """Salvează slippage excesiv în CSV pentru analiză ulterioară."""
    try:
        row = {
            "timestamp":    datetime.now().isoformat(),
            "order_id":     order.get("order_id"),
            "direction":    order.get("direction"),
            "signal_price": order.get("signal_price"),
            "entry_price":  order.get("entry_price"),
            "slippage_pts": order.get("slippage_pts"),
            "slippage_pct": order.get("slippage_pct"),
            "alert":        order.get("slippage_pct", 0) > SLIPPAGE_ALERT_PCT,
        }
        df_row = pd.DataFrame([row])
        if os.path.exists(SLIPPAGE_LOG_PATH):
            df_row.to_csv(SLIPPAGE_LOG_PATH, mode='a', header=False, index=False)
        else:
            df_row.to_csv(SLIPPAGE_LOG_PATH, index=False)
    except Exception as e:
        logger.warning(f"Slippage log error: {e}")


def get_slippage_report() -> dict:
    """
    Update #46: Raport de slippage.
    Returnează statistici: medie, max, % ordine cu slippage excesiv.
    """
    try:
        if not os.path.exists(SLIPPAGE_LOG_PATH):
            return {"avg_slippage": 0.0, "max_slippage": 0.0, "alert_rate": 0.0, "n_records": 0}
        df = pd.read_csv(SLIPPAGE_LOG_PATH)
        if df.empty:
            return {"avg_slippage": 0.0, "max_slippage": 0.0, "alert_rate": 0.0, "n_records": 0}
        return {
            "avg_slippage": round(float(df["slippage_pct"].mean()), 4),
            "max_slippage": round(float(df["slippage_pct"].max()), 4),
            "alert_rate":   round(float(df["alert"].mean() * 100), 1),
            "n_records":    len(df),
        }
    except Exception as e:
        return {"error": str(e)}


# Instanță globală
oms = AladinOMS()


if __name__ == "__main__":
    print("🔷 ALADIN OMS — Test")
    test_order = oms.open_order(
        signal_ts="2025-01-15 09:30:00",
        direction="LONG",
        signal_price=490.50,
        sl=488.00,
        tp=496.00,
        risk_usd=50.0,
        conviction="DIAMOND",
        score=0.72,
    )
    print(f"  Ordin creat: {test_order['order_id']}")
    oms.fill_order(test_order["order_id"], fill_price=490.65, fill_qty=1.0)
    print(f"  Fill @ 490.65 | Slippage: {test_order.get('slippage_pct', 'N/A')}%")
    stats = oms.get_stats()
    print(f"  Stats: {stats}")
    print("✅ OMS Test OK")
