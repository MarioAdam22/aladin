"""
╔══════════════════════════════════════════════════════════════════╗
║   ALADIN SENTIMENT ENGINE — UPDATE #6                            ║
║   FinBERT (știri Yahoo Finance) + Stocktwits ($NQ, $ES)          ║
║   Gratuit, fără API key, rulează local                           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import logging
import time as _time
import requests

logger = logging.getLogger('ALADIN')

# ── Cache intern (5 minute) ───────────────────────────────────────
_SENT_CACHE: dict = {}
_SENT_TTL: float  = 300.0  # 5 minute

def _scache_get(key: str):
    e = _SENT_CACHE.get(key)
    if e and (_time.time() - e['ts']) < _SENT_TTL:
        return e['val']
    return None

def _scache_set(key: str, val):
    _SENT_CACHE[key] = {'val': val, 'ts': _time.time()}


# ═══════════════════════════════════════════════════════════════════
# 1. FinBERT — analiză știri Yahoo Finance
# ═══════════════════════════════════════════════════════════════════
_finbert_pipe = None

def _get_finbert():
    """Încarcă FinBERT o singură dată (~440MB, cached local după prima rulare)."""
    global _finbert_pipe
    if _finbert_pipe is None:
        try:
            from transformers import pipeline
            _finbert_pipe = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                top_k=None,
                device=-1,   # CPU — fără GPU necesar
                truncation=True,
                max_length=512,
            )
            logger.info("   📰 FinBERT încărcat (ProsusAI/finbert)")
        except Exception as e:
            logger.warning(f"   FinBERT load error: {e}")
    return _finbert_pipe


def get_news_sentiment(symbols: list = None) -> dict:
    """
    Analizează titlurile de știri de pe Yahoo Finance cu FinBERT.
    symbols: lista de tickers (default: NQ futures + SPY + QQQ)
    Returnează: {'score': float -1→+1, 'label': str, 'n_articles': int}
    """
    cached = _scache_get('finbert')
    if cached is not None:
        return cached

    if symbols is None:
        symbols = ["NQ=F", "ES=F", "QQQ", "SPY"]

    try:
        import yfinance as yf

        titles = []
        for sym in symbols:
            try:
                news = yf.Ticker(sym).news or []
                titles += [n.get('title', '') for n in news[:5] if n.get('title')]
            except Exception:
                pass

        if not titles:
            result = {'score': 0.0, 'label': 'NEUTRAL', 'n_articles': 0, 'source': 'no_data'}
            _scache_set('finbert', result)
            return result

        finbert = _get_finbert()
        if finbert is None:
            result = {'score': 0.0, 'label': 'NEUTRAL', 'n_articles': 0, 'source': 'finbert_unavailable'}
            _scache_set('finbert', result)
            return result

        # Analizăm titlurile
        raw = finbert(titles[:20])   # max 20 titluri
        total_score = 0.0
        for item in raw:
            for r in item:
                if r['label'] == 'positive':
                    total_score += r['score']
                elif r['label'] == 'negative':
                    total_score -= r['score']

        avg_score = total_score / max(len(titles[:20]), 1)
        label = 'BULLISH' if avg_score > 0.10 else 'BEARISH' if avg_score < -0.10 else 'NEUTRAL'

        result = {
            'score':      round(avg_score, 4),
            'label':      label,
            'n_articles': len(titles),
            'source':     'finbert_yfinance',
            'top_titles': titles[:3],
        }
        _scache_set('finbert', result)
        logger.info(f"   📰 FinBERT: {avg_score:.3f} → {label} ({len(titles)} articole)")
        return result

    except Exception as e:
        logger.warning(f"   FinBERT news error: {e}")
        result = {'score': 0.0, 'label': 'NEUTRAL', 'n_articles': 0, 'source': 'error'}
        _scache_set('finbert', result)
        return result


# ═══════════════════════════════════════════════════════════════════
# 2. Stocktwits — sentiment traderi ($NQ, $ES)
# ═══════════════════════════════════════════════════════════════════

def get_stocktwits_sentiment(symbols: list = None) -> dict:
    """
    Sentiment de pe Stocktwits — gratuit, fără autentificare.
    Traderii postează $NQ, $ES cu tag bullish/bearish explicit.
    Returnează: {'score': float -1→+1, 'bull_pct': float, 'bear_pct': float}
    """
    cached = _scache_get('stocktwits')
    if cached is not None:
        return cached

    if symbols is None:
        symbols = ["NQ", "ES", "QQQ"]

    total_bull = 0
    total_bear = 0
    total_posts = 0

    for sym in symbols:
        try:
            url  = f"https://api.stocktwits.com/api/2/streams/symbol/{sym}.json"
            resp = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            if resp.status_code != 200:
                continue

            data     = resp.json()
            messages = data.get('messages', [])

            for m in messages:
                sent = m.get('entities', {}).get('sentiment', {})
                if not sent:
                    continue
                basic = sent.get('basic', '')
                if basic == 'Bullish':
                    total_bull += 1
                elif basic == 'Bearish':
                    total_bear += 1
                total_posts += 1

        except Exception as e:
            logger.debug(f"   Stocktwits {sym} error: {e}")

    total_tagged = total_bull + total_bear
    if total_tagged == 0:
        result = {
            'score':    0.0,
            'bull_pct': 50.0,
            'bear_pct': 50.0,
            'n_posts':  total_posts,
            'source':   'stocktwits_no_tags',
        }
        _scache_set('stocktwits', result)
        return result

    bull_pct = (total_bull / total_tagged) * 100
    bear_pct = (total_bear / total_tagged) * 100
    score    = (bull_pct - 50.0) / 50.0   # -1 → +1

    result = {
        'score':    round(score, 4),
        'bull_pct': round(bull_pct, 1),
        'bear_pct': round(bear_pct, 1),
        'n_posts':  total_posts,
        'n_tagged': total_tagged,
        'source':   'stocktwits',
    }
    _scache_set('stocktwits', result)
    logger.info(f"   🐦 Stocktwits: {score:.3f} → {bull_pct:.0f}% bull / {bear_pct:.0f}% bear ({total_posts} posts)")
    return result


# ═══════════════════════════════════════════════════════════════════
# 3. Combined — FinBERT 60% + Stocktwits 40%
# ═══════════════════════════════════════════════════════════════════

def get_combined_sentiment() -> dict:
    """
    Combină FinBERT (știri) + Stocktwits (social) într-un scor final.
    Returnează: {
        'combined_score': float -1→+1,
        'label':          str (BULLISH/BEARISH/NEUTRAL),
        'sentiment_mult': float (0.80→1.20, multiplicator pentru scor final),
        'news':           dict,
        'social':         dict,
    }
    """
    cached = _scache_get('combined')
    if cached is not None:
        return cached

    news   = get_news_sentiment()
    social = get_stocktwits_sentiment()

    # Ponderi: 60% FinBERT (știri instituționale) + 40% Stocktwits (retail)
    combined = news['score'] * 0.60 + social['score'] * 0.40

    label = 'BULLISH' if combined > 0.10 else 'BEARISH' if combined < -0.10 else 'NEUTRAL'

    # Multiplicator pentru scorul final Aladin:
    # combined=+0.5 → mult=1.15 (+15% boost pe scor)
    # combined=-0.5 → mult=0.85 (-15% reducere scor)
    # Limitat la [0.80, 1.20] pentru a nu domina semnalul
    sentiment_mult = max(0.80, min(1.20, 1.0 + combined * 0.40))

    result = {
        'combined_score': round(combined, 4),
        'label':          label,
        'sentiment_mult': round(sentiment_mult, 3),
        'news':           news,
        'social':         social,
    }
    _scache_set('combined', result)

    logger.info(
        f"   🧠 Sentiment combinat: FinBERT={news['score']:.3f} "
        f"Stocktwits={social['score']:.3f} → {combined:.3f} {label} "
        f"(mult×{sentiment_mult:.2f})"
    )
    return result


# ═══════════════════════════════════════════════════════════════════
# Test standalone
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    print("\n" + "="*60)
    print("  ALADIN SENTIMENT ENGINE — Test")
    print("="*60)

    print("\n📰 FinBERT News Sentiment...")
    news = get_news_sentiment()
    print(f"   Score: {news['score']:.3f} | Label: {news['label']} | Articole: {news['n_articles']}")
    if news.get('top_titles'):
        for t in news['top_titles']:
            print(f"   • {t[:80]}")

    print("\n🐦 Stocktwits Social Sentiment...")
    social = get_stocktwits_sentiment()
    print(f"   Score: {social['score']:.3f} | Bull: {social['bull_pct']:.0f}% | Bear: {social['bear_pct']:.0f}%")
    print(f"   Posts analizate: {social['n_posts']}")

    print("\n🧠 Sentiment Combinat...")
    combined = get_combined_sentiment()
    print(f"   Score: {combined['combined_score']:.3f} | Label: {combined['label']}")
    print(f"   Multiplicator scor Aladin: ×{combined['sentiment_mult']:.2f}")
    print("="*60)
