#!/usr/bin/env python3
"""Swing-trade report: odds of +10% in a week, market regime, screen and earnings dates.

One script, one HTML output. It:
  1. downloads ~5y of daily prices for a liquid US universe (Yahoo Finance via yfinance)
  2. backtests common swing setups: enter next open, exit at +TARGET, -STOP or after HOLD days
  3. splits results by market regime (SPY above/below its 50-day average)
  4. studies what SPY did after past sharp selloffs
  5. screens the latest bar for setups and builds a watchlist
  6. fetches next/last earnings dates for every suggested stock and flags those that
     fall inside (or just after) the coming holding window, NYSE holidays included
  7. writes a self-contained, theme-aware HTML report

Usage:
  pip install --require-hashes --no-deps -r requirements.txt
  python swing_report.py                       # writes swing_report.html
  python swing_report.py -o out.html --account 2500 --risk 0.01

Educational analysis only, not financial advice.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import html
import json
import math
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

UNIVERSE = sorted(set("""
AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO AMD NFLX ORCL CRM ADBE INTC QCOM MU
AMAT LRCX KLAC MRVL ARM SMCI TSM ASML PLTR SNOW CRWD PANW ZS NET DDOG MDB SHOP
UBER ABNB DASH COIN MSTR HOOD SOFI AFRM UPST XYZ PYPL RBLX U DKNG RIVN LCID NIO
ROKU TTD APP IONQ RKLB SOUN AI PATH CVNA CELH ENPH FSLR ON MRNA BNTX LLY NVO
JPM GS BAC XOM CVX OXY BA CAT DE GE F GM DIS NKE SBUX WMT COST HD TGT LULU
MARA RIOT CLSK VST CEG OKLO SMR HIMS NU GME AMC
""".split()))

TARGET, STOP, HOLD = 0.10, 0.05, 5
COIN_FLIP_UP10 = 0.30   # stocks that touch +10% in >=30% of 5-day windows
GROUP_SIZE = 8
SOON_DAYS = 5           # trading days after the window still flagged "soon"
RECENT_DAYS = 14        # calendar days: "just reported"

# --------------------------------------------------------------------------- data


def load_prices(period: str) -> dict[str, pd.DataFrame]:
    tick = UNIVERSE + ["SPY", "QQQ", "^VIX"]
    df = yf.download(tick, period=period, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    out = {}
    for t in tick:
        try:
            d = df[t].dropna()
        except KeyError:
            continue
        if len(d) > 300:
            out[t] = d
    for t in [t for t in tick if t not in out]:  # batch downloads sometimes drop a ticker; retry singly
        try:
            d = yf.download([t], period=period, auto_adjust=True, progress=False, group_by="ticker")[t].dropna()
            if len(d) > 300:
                out[t] = d
        except Exception:
            pass
    missing = [t for t in tick if t not in out]
    if missing:
        print("No/short data, skipped:", " ".join(missing), file=sys.stderr)
    return out


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def indicators(d: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    x = pd.DataFrame(index=d.index)
    x["sma20"], x["sma50"], x["sma200"] = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    x["rsi2"], x["rsi14"] = rsi(c, 2), rsi(c, 14)
    x["hi20"] = h.rolling(20).max().shift(1)
    x["volr"] = v / v.rolling(20).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    x["atrp"] = tr.rolling(14).mean() / c
    spy = spy_close.reindex(c.index)
    x["rs63"] = (c / c.shift(63)) / (spy / spy.shift(63))
    x["ret5"] = c / c.shift(5) - 1
    return x


# --------------------------------------------------------------------------- setups
# key -> (label, note, signal function)
SETUPS = {
    "random": ("Random entry (base rate)", "", lambda d, x: pd.Series(True, index=d.index)),
    "oversold": ("Oversold pullback in an uptrend", "RSI(2) under 10, price above 200-day avg",
                 lambda d, x: (x["rsi2"] < 10) & (d["Close"] > x["sma200"])),
    "momentum": ("Momentum leader", "Beating SPY by 30%+ over 3 months, above 20/50-day",
                 lambda d, x: (x["rs63"] > 1.3) & (d["Close"] > x["sma20"]) & (x["sma20"] > x["sma50"])),
    "breakout": ("20-day breakout on 1.5× volume", "Close above prior 20-day high, above 50-day",
                 lambda d, x: (d["Close"] > x["hi20"]) & (x["volr"] > 1.5) & (d["Close"] > x["sma50"])),
    "highvol": ("High-volatility names only", "Average daily range above 5%",
                lambda d, x: x["atrp"] > 0.05),
    "bigdrop": ("Bounce after a 15%+ drop in 5 days", "", lambda d, x: x["ret5"] < -0.15),
}


def simulate(d: pd.DataFrame, sig: pd.Series):
    """Yield (signal_index, return, outcome, touched_up10, touched_dn10) per trade."""
    o, h, l, c = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    s = sig.fillna(False).to_numpy(dtype=bool)
    n = len(d)
    for i in np.flatnonzero(s[:-1]):
        e = i + 1
        if e + HOLD > n:
            break
        entry = o[e]
        tgt, stp = entry * (1 + TARGET), entry * (1 - STOP)
        outcome, ret = "time", c[e + HOLD - 1] / entry - 1
        for j in range(e, e + HOLD):
            if l[j] <= stp:  # conservative: stop assumed first if both hit on one bar
                outcome, ret = "stop", min(o[j], stp) / entry - 1
                break
            if h[j] >= tgt:
                outcome, ret = "target", max(o[j], tgt) / entry - 1
                break
        yield i, ret, outcome, h[e:e + HOLD].max() >= tgt, l[e:e + HOLD].min() <= entry * 0.90


def summarize(trades):
    if not trades:
        return dict(trades=0, target=np.nan, stop=np.nan, time=np.nan, avg=np.nan, win=np.nan)
    a = np.array([t[0] for t in trades])
    oc = [t[1] for t in trades]
    n = len(a)
    return dict(trades=n, target=oc.count("target") / n, stop=oc.count("stop") / n,
                time=oc.count("time") / n, avg=float(a.mean()), win=float((a > 0).mean()))


def backtest(data, ind, spy_below50):
    results, per_ticker = {}, {}
    for key, (_, _, fn) in SETUPS.items():
        allt, below, above = [], [], []
        for t, x in ind.items():
            d = data[t]
            reg = spy_below50.reindex(d.index).fillna(False).to_numpy(dtype=bool)
            tr = list(simulate(d, fn(d, x)))
            if key == "random":
                per_ticker[t] = dict(up10=float(np.mean([z[3] for z in tr])),
                                     dn10=float(np.mean([z[4] for z in tr])))
            for i, ret, oc, *_ in tr:
                allt.append((ret, oc))
                (below if reg[i] else above).append((ret, oc))
        results[key] = dict(all=summarize(allt), below=summarize(below), above=summarize(above))
        r = results[key]["all"]
        print(f"  {SETUPS[key][0]:40s} n={r['trades']:7d} target={r['target']:.1%} stop={r['stop']:.1%} avg={r['avg']:+.2%}")
    return results, per_ticker


def selloff_study(period="max"):
    s = yf.download(["SPY", "^VIX"], period=period, auto_adjust=True, progress=False)["Close"].dropna()
    spy, vix = s["SPY"], s["^VIX"]
    r5 = spy / spy.shift(5) - 1
    f5, f20 = spy.shift(-5) / spy - 1, spy.shift(-20) / spy - 1
    hits, last = [], None
    for d in spy.index[((r5 < -0.07) & (vix > 30)).fillna(False)]:
        if last is None or (d - last).days > 30:
            hits.append(d)
            last = d
    ev = pd.DataFrame({"f5": f5[hits], "f20": f20[hits]})
    done = ev.dropna()
    return dict(
        events=len(ev), with_data=len(done), since=str(spy.index[0].year),
        points=[(d.date(), float(v)) for d, v in done.f5.items()],
        f5_pos=float((done.f5 > 0).mean()), f5_mean=float(done.f5.mean()), f5_med=float(done.f5.median()),
        f5_best=float(done.f5.max()), f5_best_d=done.f5.idxmax().year,
        f5_worst=float(done.f5.min()), f5_worst_d=done.f5.idxmin().year,
        f20_pos=float((done.f20 > 0).mean()), f20_mean=float(done.f20.mean()),
        f20_worst=float(done.f20.min()), f20_worst_d=done.f20.idxmin().strftime("%B %Y"),
        normal_f5=float(f5.mean()),
    )


# --------------------------------------------------------------------------- calendar

def _nth_weekday(y, m, wd, n):
    d = dt.date(y, m, 1)
    d += dt.timedelta((wd - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def _last_weekday(y, m, wd):
    d = dt.date(y, m + 1, 1) - dt.timedelta(1) if m < 12 else dt.date(y, 12, 31)
    return d - dt.timedelta((d.weekday() - wd) % 7)


def _easter(y):
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return dt.date(y, month, day + 1)


def _observed(d):
    return d - dt.timedelta(1) if d.weekday() == 5 else d + dt.timedelta(1) if d.weekday() == 6 else d


def nyse_holidays(y):
    """Full-day closures (name by date) and early 1pm closes."""
    full = {
        _observed(dt.date(y, 1, 1)): "New Year's Day",
        _nth_weekday(y, 1, 0, 3): "Martin Luther King Jr. Day",
        _nth_weekday(y, 2, 0, 3): "Presidents' Day",
        _easter(y) - dt.timedelta(2): "Good Friday",
        _last_weekday(y, 5, 0): "Memorial Day",
        _observed(dt.date(y, 6, 19)): "Juneteenth",
        _observed(dt.date(y, 7, 4)): "Independence Day",
        _nth_weekday(y, 9, 0, 1): "Labor Day",
        _nth_weekday(y, 11, 3, 4): "Thanksgiving",
        _observed(dt.date(y, 12, 25)): "Christmas",
    }
    full = {d: n for d, n in full.items() if d.year == y}  # NYSE skips Sat-New-Year observance
    early = {_nth_weekday(y, 11, 3, 4) + dt.timedelta(1): "day after Thanksgiving"}
    for d, n in ((dt.date(y, 12, 24), "Christmas Eve"), (dt.date(y, 7, 3), "July 3")):
        if d.weekday() < 5 and d not in full:
            early[d] = n
    return full, early


def next_sessions(after: dt.date, n: int):
    full, early = {}, {}
    for y in (after.year, after.year + 1):
        f, e = nyse_holidays(y)
        full.update(f)
        early.update(e)
    days, skipped, d = [], [], after
    while len(days) < n:
        d += dt.timedelta(1)
        if d.weekday() >= 5:
            continue
        if d in full:
            skipped.append((d, full[d]))
            continue
        days.append(d)
    halfs = [(x, early[x]) for x in days if x in early]
    return days, skipped, halfs


# --------------------------------------------------------------------------- earnings

def earnings_dates(tickers, asof: dt.date):
    out = {}
    now = pd.Timestamp(asof)
    for t in tickers:
        nxt = last = None
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=8)
            if ed is not None and len(ed):
                idx = ed.index.tz_localize(None) if ed.index.tz is not None else ed.index
                idx = idx.normalize()
                fut, past = idx[idx > now], idx[idx <= now]
                nxt = fut.min().date() if len(fut) else None
                last = past.max().date() if len(past) else None
        except Exception:
            pass
        if nxt is None:
            try:
                cal = yf.Ticker(t).calendar
                dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
                if dates:
                    cand = [pd.Timestamp(x).date() for x in dates if pd.Timestamp(x).date() > asof]
                    nxt = min(cand) if cand else None
            except Exception:
                pass
        out[t] = dict(next=nxt, last=last)
    return out


# --------------------------------------------------------------------------- screen

def screen(data, ind, per_ticker):
    rows = []
    for t, x in ind.items():
        d = data[t]
        hits = [k for k, (_, _, fn) in SETUPS.items() if k != "random" and bool(fn(d, x).iloc[-1])]
        if hits:
            rows.append(dict(t=t, close=float(d["Close"].iloc[-1]), setups=hits,
                             ret5=float(x["ret5"].iloc[-1]), atrp=float(x["atrp"].iloc[-1]),
                             above200=bool(d["Close"].iloc[-1] > x["sma200"].iloc[-1]),
                             **per_ticker[t]))
    return rows


def build_groups(rows, results, below50):
    reg = "below" if below50 else "above"
    reg_word = "below" if below50 else "above"

    def stats(key):
        r = results[key][reg]
        if not r["trades"]:
            return "No history in this market regime."
        return (f"Historically {pct(r['avg'])} per trade when SPY was {reg_word} its 50-day average "
                f"({r['target']:.0%} hit target, {r['stop']:.0%} hit stop, {r['trades']:,} trades).")

    defs = [
        ("oversold", "Oversold, trend intact", "Pulled back but still above the 200-day",
         lambda r: "oversold" in r["setups"], lambda r: r["ret5"]),
        ("coin", "Coin flips", "Most likely to move 10% in a week",
         lambda r: r["up10"] >= COIN_FLIP_UP10 and ({"bigdrop", "highvol"} & set(r["setups"])),
         lambda r: -r["up10"]),
        ("bigdrop", "Selloff bounce", "Dropped 15%+ in five days",
         lambda r: "bigdrop" in r["setups"], lambda r: r["ret5"]),
        ("momentum", "Momentum leader", "Leading the market, trend up",
         lambda r: "momentum" in r["setups"], lambda r: -r["ret5"]),
        ("breakout", "Breakout", "New 20-day high on heavy volume",
         lambda r: "breakout" in r["setups"], lambda r: -r["ret5"]),
    ]
    used, groups = set(), []
    for key, tag, title, flt, sort in defs:
        pick = sorted([r for r in rows if r["t"] not in used and flt(r)], key=sort)[:GROUP_SIZE]
        if not pick:
            continue
        used |= {r["t"] for r in pick}
        if key == "coin":
            desc = ("Each touched +10% in at least 30% of past 5-day windows, and fell 10% about as often. "
                    "Size these smallest.")
        else:
            desc = stats(key)
        groups.append(dict(key=key, tag=tag, title=title, desc=desc, rows=pick))
    return groups


# --------------------------------------------------------------------------- day movers

MOVERS_EACH = 6         # top gainers and top losers on the latest session
NEWS_LOOKBACK = 3       # calendar days of headlines considered "why it moved"


GENERIC_WORDS = {"advanced", "american", "united", "general", "international", "first", "the", "global",
                 "applied", "taiwan", "eli", "home", "bank", "micro", "super"}
NAME_SUFFIX = re.compile(r"[,.]?\s+(inc|incorporated|corp|corporation|co|company|holdings?|plc|ltd|limited|n\.?v|s\.?a|group|class [a-c]|technologies|platforms)\.?\b.*$", re.I)


MOVE_WORDS = re.compile(r"\b(falls?|fell|drops?|dropped|plunge[sd]?|plummet\w*|sinks?|sank|slides?|slid|tumbles?|tumbled|"
                        r"slumps?|crash\w*|rout|sell-?off|selloff|surges?|surged|jumps?|jumped|soars?|soared|rall(y|ies|ied)|"
                        r"climbs?|climbed|gains?|spikes?|pops?|rebounds?|why .* stock|shares? (are|is) (up|down))\b", re.I)
CATALYST_WORDS = re.compile(r"\b(earnings|results|guidance|outlook|forecast|downgrade[sd]?|upgrade[sd]?|price target|"
                            r"probe|investigation|sec\b|doj|lawsuit|ceo|resign\w*|ousted|acquisition|acquire[sd]?|merger|deal|"
                            r"fda|approval|tariffs?|export|ban|bitcoin|offering|dilution|contract|beats?|miss(es|ed)?)\b", re.I)
BOILERPLATE = re.compile(r"(zacks\.com users|most watched|etfs? (sold|bought)|\d+ etfs|is it time to buy|should you buy|"
                         r"here's what you need to know before|stock of the day|motley fool|better buy|"
                         r"buy what they know|retirees|dividend income)", re.I)


def company_keywords(t: str):
    """Words that show a headline is actually about this company: ticker, cleaned name, distinctive first word."""
    kws = {t.lower()}
    try:
        info = yf.Ticker(t).info or {}
        name = (info.get("shortName") or info.get("longName") or "").strip()
    except Exception:
        name = ""
    clean = NAME_SUFFIX.sub("", name).strip(" ,.").lower()
    if clean:
        kws.add(clean)
        first = re.sub(r"[^a-z0-9]", "", clean.split()[0])
        if len(first) >= 4 and first not in GENERIC_WORDS:
            kws.add(first)
    return kws


def _mentions(text: str, kws) -> bool:
    low = text.lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", low) for k in kws)


def fetch_news(t: str, asof: dt.date, n: int = 2):
    """Recent Yahoo Finance headlines that name the company, most relevant then newest first."""
    tk = yf.Ticker(t)
    try:
        items = tk.get_news(count=40) or []
    except Exception:
        try:
            items = tk.news or []
        except Exception:
            return []
    kws = company_keywords(t)
    lo = pd.Timestamp(asof - dt.timedelta(NEWS_LOOKBACK), tz="UTC")
    hi = pd.Timestamp(asof + dt.timedelta(2), tz="UTC")
    arts = []
    for it in items:
        c = it.get("content", it)
        title = (c.get("title") or "").strip()
        if not title:
            continue
        try:
            when = pd.Timestamp(c.get("pubDate") or c.get("displayTime"))
            when = when.tz_localize("UTC") if when.tzinfo is None else when
        except Exception:
            continue
        if not lo <= when <= hi:
            continue
        tickers = {s.get("symbol") for s in ((c.get("finance") or {}).get("stockTickers") or [])}
        url = ((c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url") or "")
        summary = re.sub(r"^\s*key\s*takeaways\s*", "", (c.get("summary") or c.get("description") or ""), flags=re.I).strip()
        in_title, in_summary = _mentions(title, kws), _mentions(summary, kws)
        if not (in_title or in_summary):
            continue  # tagged-only articles are usually about something else
        both = f"{title} {summary}"
        score = (3 * in_title + in_summary + (t in tickers and len(tickers) <= 2)
                 + 2 * bool(MOVE_WORDS.search(both)) + 2 * bool(CATALYST_WORDS.search(both))
                 + (when.tz_convert("America/New_York").date() == asof)
                 - 4 * bool(BOILERPLATE.search(both)))
        arts.append(dict(title=title, summary=summary if in_summary else "", when=when, score=score,
                         source=(c.get("provider") or {}).get("displayName", ""),
                         url=url if url.startswith("https://") else ""))
    arts.sort(key=lambda a: (-a["score"], -a["when"].timestamp()))
    return arts[:n]


def day_movers(data, ind, asof):
    rows = []
    for t, x in ind.items():
        d = data[t]
        if d.index[-1].date() != asof or len(d) < 21:
            continue
        c = d["Close"]
        rows.append(dict(t=t, close=float(c.iloc[-1]), r1=float(c.iloc[-1] / c.iloc[-2] - 1),
                         volr=float(x["volr"].iloc[-1]), ret5=float(x["ret5"].iloc[-1])))
    rows.sort(key=lambda r: r["r1"])
    losers = [r for r in rows if r["r1"] < 0][:MOVERS_EACH]
    gainers = [r for r in rows[::-1] if r["r1"] > 0][:MOVERS_EACH]
    return gainers, losers


def explain_move(m, spy_1d, earn, asof):
    """Short plain-language reason, built from headlines plus price and volume context."""
    parts, chips = [], []
    last = (earn or {}).get("last")
    if last and 0 <= (asof - last).days <= 3:
        parts.append(f"Reported earnings on {fmt_d(last, True)}.")
        chips.append("Earnings")
    news = m.get("news") or []
    if news:
        a = news[0]
        text = a["summary"] or (a["title"].rstrip(".") + ".")
        if len(text) > 240:
            text = text[:237].rsplit(" ", 1)[0] + "…"
        parts.append(text)
    else:
        same_way = (m["r1"] < 0) == (spy_1d < 0) and abs(spy_1d) > 0.01
        parts.append("No company-specific headline in Yahoo Finance's feed. "
                     + (f"The move went the same way as the broad market (SPY {pct(spy_1d)})." if same_way
                        else "The cause isn't clear from public headlines."))
    rel = m["r1"] - spy_1d
    chips.append(f"{pct(rel)} vs SPY")
    if m["volr"] >= 1.5:
        chips.append(f"Volume {m['volr']:.1f}× avg")
    return " ".join(parts), chips


# --------------------------------------------------------------------------- html

def pct(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x * 100:+.{d}f}%".replace("-", "−") if sign else f"{x * 100:.{d}f}%"


def cls(x):
    return "up" if x > 0 else "down" if x < 0 else ""


def fmt_d(d: dt.date | None, long=False):
    if d is None:
        return "n/a"
    return d.strftime("%a %b %-d" if long else "%b %-d")


def q(term_id: str, label: str = "") -> str:
    """Small '?' link that jumps to a glossary entry."""
    return (f'<a class="q" href="#g-{term_id}" aria-label="What does {html.escape(label or term_id)} mean?" '
            f'title="What does this mean?">?</a>')


# --------------------------------------------------------------------------- glossary
# (category, id, term, definition, example-or-tip). Text only; rendered escaped.
GLOSSARY = [
    ("Market and indexes", "ticker", "Ticker",
     "The short code a stock or fund trades under on an exchange.",
     "NVDA is Nvidia, MSFT is Microsoft, SPY is an S&P 500 fund."),
    ("Market and indexes", "sp500", "S&P 500",
     "An index of about 500 of the largest US companies, weighted by size. It's the most common measure of "
     "\"the US stock market.\"", ""),
    ("Market and indexes", "etf", "ETF (exchange-traded fund)",
     "A fund that holds a basket of stocks and trades on the exchange like a single stock. One share gives you a "
     "slice of everything in the basket.", ""),
    ("Market and indexes", "spy", "SPY",
     "The ticker of the SPDR S&P 500 ETF, which holds the S&P 500 companies. This report uses it as \"the market.\"",
     "\"SPY −8.6% in 5 days\" means large US stocks as a whole fell about 8.6% that week."),
    ("Market and indexes", "qqq", "QQQ",
     "An ETF that tracks the Nasdaq-100, the 100 largest companies on the Nasdaq exchange. It's heavy in "
     "technology, so it usually swings more than SPY.", ""),
    ("Market and indexes", "vix", "VIX (the \"fear gauge\")",
     "The Cboe Volatility Index. It uses S&P 500 option prices to measure how much movement traders expect over the "
     "next 30 days, as a yearly percentage. It rises when traders rush to buy protection, which usually happens when "
     "stocks are falling.",
     "Rule of 16: divide the VIX by 16 to get the expected daily move. VIX 36 ≈ ±2.25% a day for SPY. "
     "Zones used here: under 15 calm, 15–20 normal, 20–30 elevated, 30–40 fear, 40+ panic."),
    ("Market and indexes", "fear-greed", "VIX vs the Fear & Greed Index",
     "They're related but not the same. The VIX measures only expected volatility, so it shows fear well. A low VIX "
     "means calm, which can go along with greed but doesn't measure it. CNN's Fear & Greed Index is a separate 0–100 "
     "score that combines seven signals: market momentum, stocks at 52-week highs vs lows, market breadth, put/call "
     "option demand, junk bond demand, the VIX, and demand for safe havens like bonds.",
     "This report shows the VIX because it's an official, real-time number that's easy to get."),
    ("Market and indexes", "regime", "Market regime",
     "Whether the overall market is trending up or down. Here it's set by SPY being above or below its 50-day "
     "average. Setups often behave differently in each regime.",
     "Breakouts lost money on average when SPY was below its 50-day average."),
    ("Market and indexes", "selloff", "Selloff",
     "A sharp, broad drop in prices. In this report's history study, a selloff week means SPY fell more than 7% in "
     "5 days with the VIX above 30.", ""),

    ("Price and trend", "ma", "Moving average (20, 50, 200-day)",
     "The average closing price over the last N trading days, updated daily so it \"moves\" with the price. "
     "20 days ≈ 1 month, 50 ≈ 2½ months, 200 ≈ 10 months.", ""),
    ("Price and trend", "vs50", "vs 50-day avg",
     "How far today's price is above (+) or below (−) its 50-day average, in percent. It's a quick check on the "
     "short-to-medium-term trend.",
     "SPY 689.81 vs a 50-day average of 760.03 → 689.81 ÷ 760.03 − 1 = −9.2%. About ±2–3% is normal; −9% is a big "
     "gap."),
    ("Price and trend", "vs200", "vs 200-day avg",
     "The same idea over about 10 months, so it shows the long-term trend. Many investors treat \"above the 200-day\" "
     "as a healthy market.", ""),
    ("Price and trend", "trend", "Uptrend and downtrend",
     "An uptrend is a series of higher highs and higher lows; a downtrend is the opposite. In this report: "
     "above both the 50- and 200-day averages = uptrend; below both = downtrend; between them = pullback or recovery.",
     ""),
    ("Price and trend", "support", "Support and resistance",
     "Price levels where buying (support) or selling (resistance) has tended to show up before. Moving averages and "
     "recent highs and lows often act as these levels because many traders watch them.", ""),
    ("Price and trend", "ret5", "5d (5-day change)",
     "The percent change over the last 5 trading sessions, which is about one week.", ""),
    ("Price and trend", "gap", "Gap",
     "When a stock opens well above or below the previous close, with no trading in between, often after earnings or "
     "big news overnight.", "A stop at −5% can fill at −15% if the stock gaps down past it."),

    ("Indicators", "rsi", "RSI (Relative Strength Index)",
     "A 0–100 score comparing recent gains with recent losses. RSI(14) uses 14 days: under 30 is \"oversold,\" over "
     "70 is \"overbought.\" RSI(2) uses only 2 days and reacts fast: under 10 means a sharp short-term drop.",
     "Not the same as relative strength vs SPY."),
    ("Indicators", "atr", "ATR / average daily range %",
     "Average True Range: how much a stock typically moves in a day over the last 14 days, shown here as a percent "
     "of its price. Higher means more volatile.", "Above 5% a day counts as a high-volatility name here."),
    ("Indicators", "volr", "Volume ratio (1.5× avg)",
     "Today's number of shares traded divided by the 20-day average. Big moves on heavy volume are taken more "
     "seriously than moves on light volume.", ""),
    ("Indicators", "rs", "Relative strength vs SPY",
     "A stock's 3-month return compared with SPY's. Above 1.3 means it beat the market by 30% or more.", ""),

    ("Setups", "swing", "Swing trade",
     "A trade held for a few days to a few weeks, trying to catch one price swing.", ""),
    ("Setups", "setup", "Setup",
     "A defined pattern or rule that flags a possible trade.", ""),
    ("Setups", "oversold", "Oversold pullback in an uptrend",
     "RSI(2) under 10 while the price is still above its 200-day average: a sharp short-term dip in a stock whose "
     "long-term trend is up.", "Lowest stop-out rate of the setups tested here."),
    ("Setups", "momentum", "Momentum leader",
     "A stock beating SPY by 30%+ over 3 months and trading above its rising 20- and 50-day averages.", ""),
    ("Setups", "breakout", "Breakout",
     "A close above the highest price of the last 20 days on at least 1.5× normal volume, with the price above its "
     "50-day average.", ""),
    ("Setups", "bigdrop", "Bounce after a big drop",
     "Buying a stock that fell 15% or more in 5 days, betting on a rebound.",
     "Worked best when the market was below its 50-day average, but stops were hit 58% of the time."),
    ("Setups", "highvol", "High-volatility names",
     "Stocks whose average daily range is above 5%. They hit +10% more often, and −5% even more often.", ""),
    ("Setups", "coin", "Coin flips (watchlist group)",
     "Stocks that touched +10% in at least 30% of past 5-day windows and fell 10% about as often. Big moves are "
     "likely; the direction isn't.", ""),

    ("Backtest and odds", "backtest", "Backtest",
     "Testing a trading rule on past prices to see how it would have done.",
     "Past results don't guarantee future ones, but they show the odds a rule has had."),
    ("Backtest and odds", "base-rate", "Base rate (random entry)",
     "What happens if you buy any stock on any day with the same exit rules. It's the benchmark every setup must beat.",
     ""),
    ("Backtest and odds", "tst", "Target, stop and time exit",
     "The three ways a test trade ends: price reaches +10% (target), price falls to −5% (stop), or 5 trading days "
     "pass (time exit), whichever comes first.", ""),
    ("Backtest and odds", "win-rate", "Win rate",
     "The share of trades that ended with any gain.", ""),
    ("Backtest and odds", "avg-trade", "Avg / trade (expectancy)",
     "The average result per trade, wins and losses together. Above zero means the rule made money on average, "
     "before costs.", ""),
    ("Backtest and odds", "breakeven", "Breakeven hit rate",
     "With a +10% target and a −5% stop, one win pays for two losses. So the target has to be hit in at least a "
     "third of the trades that hit either level.", ""),
    ("Backtest and odds", "odds10", "↑10% / ↓10% odds",
     "In the watchlist: the share of past 5-day windows in which the stock touched +10% (↑) or −10% (↓) at any "
     "point. It describes how the stock has behaved, not a forecast.", ""),
    ("Backtest and odds", "survivorship", "Survivorship bias",
     "Testing only on stocks that are still around and popular today makes past results look better than they were, "
     "because the ones that collapsed or were delisted are missing.", ""),
    ("Backtest and odds", "costs", "Slippage, spread and commissions",
     "Real trading costs the backtest leaves out. The spread is the gap between buy and sell prices; slippage is "
     "getting a worse fill than expected. On volatile small stocks these can cost 0.2–0.5% per round trip.", ""),

    ("Orders and risk", "stop-order", "Stop-loss order",
     "An order that sells automatically if the price falls to a set level. It limits losses but can't protect "
     "against gaps.", ""),
    ("Orders and risk", "limit-order", "Limit order",
     "An order to buy or sell only at a set price or better.", "Used here to take profit at +10%."),
    ("Orders and risk", "risk-trade", "Risk per trade",
     "How much of the account you're willing to lose if the stop is hit. This report uses 2%.",
     "$1,000 × 2% = $20 at risk."),
    ("Orders and risk", "position", "Position size",
     "How much money goes into one trade, worked out from the risk per trade and the stop distance.",
     "$20 risk ÷ 5% stop = a $400 position."),
    ("Orders and risk", "reversal", "Reversal day (waiting for a turn)",
     "A day that closes above the previous day's high after a drop. It's a sign buyers are stepping in, used here "
     "as the cue to enter instead of buying the drop itself.", ""),
    ("Orders and risk", "pdt", "Pattern day trader (PDT) rule",
     "A US rule for margin accounts under $25,000: no more than 3 same-day buy-and-sell round trips in 5 business "
     "days. Holding overnight, as swing trades do, avoids it.", ""),
    ("Orders and risk", "t1", "T+1 settlement",
     "Stock sales finish settling one business day after the trade. In a cash account you have to wait for that "
     "before the money can be reused freely.", ""),

    ("Earnings and calendar", "earnings", "Earnings (results) date",
     "The day a company reports its quarterly results. Stocks often jump or drop sharply on the news.",
     "Upcoming dates are often estimates until the company confirms them."),
    ("Earnings and calendar", "hold-window", "Hold window",
     "The next 5 trading sessions, the length of a test trade here. Market holidays are skipped.", ""),
    ("Earnings and calendar", "during-hold", "Reports during / just after the hold",
     "Red: the company reports inside the hold window, so a gap risk is built in. Amber: it reports within 5 "
     "sessions after the window, so a late entry would bring it inside.", ""),
    ("Earnings and calendar", "just-reported", "Just reported",
     "The company reported in the last 14 days, so that news is already in the price.", ""),
    ("Earnings and calendar", "early-close", "Market holiday / early close",
     "US exchanges close for set holidays and close early (1pm New York) on a few days, such as the day after "
     "Thanksgiving.", ""),

    ("Report labels", "vs-spy", "\"vs SPY\" chip",
     "How much better or worse a stock did than the market on the same day, in percentage points.",
     "WMT +1.5% on a day SPY fell 3.6% → +5.2% vs SPY."),
    ("Report labels", "vol-chip", "\"Volume ×\" chip",
     "Shown when a stock traded at least 1.5× its normal volume, a sign the move had real participation.", ""),
]


def render_glossary() -> str:
    e = html.escape
    cats: dict[str, list] = {}
    for cat, tid, term, defn, ex in GLOSSARY:
        cats.setdefault(cat, []).append((tid, term, defn, ex))
    blocks = []
    for cat, items in cats.items():
        terms = "".join(
            f'<div class="term" id="g-{tid}"><dt>{e(term)}</dt><dd>{e(defn)}</dd>'
            + (f'<dd class="ex">{e(ex)}</dd>' if ex else "") + "</div>" for tid, term, defn, ex in items)
        blocks.append(f'<div class="gcat"><h3>{e(cat)}</h3><dl class="terms">{terms}</dl></div>')
    return "".join(blocks)


# --------------------------------------------------------------------------- visuals
# All charts are HTML/CSS plus inline SVG paths, so no chart library and no external requests.
# Lines use a 1000x100 viewBox stretched to the plot (non-scaling strokes); text, dots and
# axis labels are HTML positioned in percent, so they stay crisp and readable at any width.

VIX_ZONES = [("Calm", 10, 15, "var(--good)", "<15"), ("Normal", 15, 20, "var(--normal)", "15–20"),
             ("Elevated", 20, 30, "var(--warning)", "20–30"), ("Fear", 30, 40, "var(--serious)", "30–40"),
             ("Panic", 40, 50, "var(--critical)", "40+")]


def vix_zone(v: float):
    for z in VIX_ZONES:
        if v < z[2]:
            return z
    return VIX_ZONES[-1]


def nice_ticks(lo: float, hi: float, n: int = 4):
    span = max(hi - lo, 1e-9)
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    t0, t1 = math.floor(lo / step) * step, math.ceil(hi / step) * step
    ticks = [t0 + i * step for i in range(int(round((t1 - t0) / step)) + 1)]
    return t0, t1, ticks


def viz_line(title: str, sub: str, dates, series, fmt="price", height=200, refs=(), aria=""):
    """Line chart. series: [{"n": name, "c": "var(--s1)", "v": [floats]}] (first series is primary)."""
    e = html.escape
    n = len(dates)
    vals = [v for s in series for v in s["v"] if v is not None and not math.isnan(v)]
    vals += [r[0] for r in refs]
    lo, hi, ticks = nice_ticks(min(vals), max(vals))
    y = lambda v: (1 - (v - lo) / (hi - lo)) * 100
    x = lambda i: i / (n - 1) * 100
    paths = []
    for s in series:
        pts = [f"{x(i) * 10:.1f},{y(v):.2f}" for i, v in enumerate(s["v"]) if v is not None and not math.isnan(v)]
        paths.append(f'<path style="stroke:{s["c"]}" d="M{" L".join(pts)}"/>')
    tick_fmt = (lambda t: f"{t:,.0f}") if fmt == "price" else (lambda t: f"{t:g}")
    grid = "".join(f'<div class="gl" style="top:{y(t):.2f}%"></div><span class="yt" style="top:{y(t):.2f}%">{tick_fmt(t)}</span>'
                   for t in ticks)
    ref_html = "".join(f'<div class="ref" style="top:{y(v):.2f}%">{f"<span>{e(lab)}</span>" if lab else ""}</div>'
                       for v, lab in refs)
    # month ticks on the x axis
    xt, last_x = [], -100.0
    for i in range(1, n):
        if dates[i].month != dates[i - 1].month and x(i) - last_x > 9:
            xt.append(f'<span style="left:{x(i):.2f}%">{dates[i]:%b}</span>')
            last_x = x(i)
    # end labels: value with a line key; legend names the series. Drop all but the primary on collision.
    ends = [(y(s["v"][-1]) / 100 * height, k, s) for k, s in enumerate(series)]
    ends_sorted = sorted(ends)
    collide = any(b[0] - a[0] < 16 for a, b in zip(ends_sorted, ends_sorted[1:]))
    vf = (lambda v: f"{v:,.2f}") if fmt == "price" else (lambda v: f"{v:.1f}")
    end_html = "".join(
        f'<span class="end" style="top:{y(s["v"][-1]):.2f}%"><i class="lk" style="--c:{s["c"]}"></i><b>{vf(s["v"][-1])}</b></span>'
        for _, k, s in ends if not collide or k == 0)
    dot = f'<span class="dotm" style="--c:{series[0]["c"]};left:100%;top:{y(series[0]["v"][-1]):.2f}%"></span>'
    legend = ("" if len(series) < 2 else '<div class="legend">' + "".join(
        f'<span><i class="lk" style="--c:{s["c"]}"></i>{e(s["n"])}</span>' for s in series) + "</div>")
    data = json.dumps({"dates": [f"{d:%a %b %-d, %Y}" for d in dates], "lo": lo, "hi": hi, "fmt": fmt,
                       "series": [{"n": s["n"], "c": s["c"],
                                   "v": [None if (v is None or math.isnan(v)) else round(float(v), 2) for v in s["v"]]}
                                  for s in series]}, separators=(",", ":"))
    return f"""<figure class="lc" data-d="{e(data, quote=True)}">
  <figcaption><h3>{e(title)}</h3>{f'<span class="note">{sub}</span>' if sub else ''}</figcaption>
  {legend}
  <div class="plot" style="height:{height}px" tabindex="0" role="img" aria-label="{e(aria or title)}">{grid}{ref_html}
    <svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true">{''.join(paths)}</svg>{dot}{end_html}
  </div>
  <div class="xt">{''.join(xt)}</div>
</figure>"""


def viz_gauge(vix: float, prev: float, compact=False) -> str:
    e = html.escape
    lo, hi = 10, 50
    pos = lambda v: (min(max(v, lo), hi) - lo) / (hi - lo) * 100
    z = vix_zone(vix)
    zp = vix_zone(prev)
    segs = "".join(f'<span class="g-seg{" on" if zz is z else ""}" style="flex:{zz[2] - zz[1]};--zc:{zz[3]}"></span>'
                   for zz in VIX_ZONES)
    labs = "".join(f'<span style="flex:{zz[2] - zz[1]}"><b>{zz[0]}</b><br>{zz[4]}</span>' for zz in VIX_ZONES)
    daily = vix / math.sqrt(252) / 100
    return f"""<div class="gauge" role="img" aria-label="VIX {vix:.1f}, {z[0]} zone. A week ago {prev:.1f}, {zp[0]}.">
  <div class="g-head"><span class="g-val">{vix:.1f}</span><span class="g-zone" style="--zc:{z[3]}"><i></i>{z[0]}</span>
    <span class="note">VIX {q('vix', 'VIX')} · {prev:.1f} a week ago ({zp[0].lower()})</span></div>
  <div class="g-track">{segs}
    <span class="g-mark prev" style="left:{pos(prev):.1f}%" title="A week ago: {prev:.1f}"></span>
    <span class="g-mark" style="left:{pos(vix):.1f}%"><span class="lab">now</span></span>
  </div>
  <div class="g-labels">{labs}</div>
  {'' if compact else f'<p class="note">Traders expect SPY to move about <b>±{daily:.1%}</b> a day (VIX ÷ 16). In calm markets that\'s under ±1%.</p>'}
</div>"""


def trend_state(reg) -> tuple[str, str]:
    a50, a200 = reg["spy"] > reg["sma50"], reg["spy"] > reg["sma200"]
    if a50 and a200:
        return "Uptrend", "up"
    if not a50 and not a200:
        return "Downtrend", "dn"
    return ("Pullback in an uptrend", "dn") if a200 else ("Recovering", "up")


def viz_pills(reg) -> str:
    def pill(label, val, tid):
        d = "up" if val > 0 else "dn"
        arrow = "▲" if val > 0 else "▼"
        word = "above" if val > 0 else "below"
        return (f'<span class="pill {d}"><span class="ar" aria-hidden="true">{arrow}</span>{label} '
                f'<b>{pct(val)}</b><span class="note">{word}</span>{q(tid, label)}</span>')
    t, td = trend_state(reg)
    return ('<div class="pills">'
            f'<span class="pill {td}"><span class="ar" aria-hidden="true">{"▲" if td == "up" else "▼"}</span><b>{t}</b>{q("trend", "trend")}</span>'
            + pill("SPY vs 50-day", reg["spy"] / reg["sma50"] - 1, "vs50")
            + pill("vs 200-day", reg["spy"] / reg["sma200"] - 1, "vs200")
            + f'<span class="pill {"up" if reg["spy_5d"] > 0 else "dn"}"><span class="ar" aria-hidden="true">{"▲" if reg["spy_5d"] > 0 else "▼"}</span>SPY 5-day <b>{pct(reg["spy_5d"])}</b></span>'
            + "</div>")


def viz_strip(events, mean: float) -> str:
    """Dot strip of SPY's next-week return after each past selloff."""
    if not events:
        return ""
    m = max(abs(v) for _, v in events)
    M = math.ceil(m * 100 / 2) * 2 / 100 or 0.02
    xp = lambda v: 3 + (v + M) / (2 * M) * 94
    rows: list[float] = []
    dots = []
    for d, v in sorted(events, key=lambda z: z[1]):
        xv = xp(v)
        r = next((i for i, last in enumerate(rows) if xv - last > 3.4), None)
        if r is None:
            rows.append(xv)
            r = len(rows) - 1
        else:
            rows[r] = xv
        c = "var(--gain)" if v > 0 else "var(--loss)"
        dots.append(f'<span class="d" tabindex="0" style="--c:{c};left:{xv:.2f}%;top:{r * 14 + 7}px" '
                    f'data-tip="{d:%b %Y}|SPY next week: {pct(v)}" aria-label="{d:%B %Y}: SPY next week {pct(v)}"></span>')
    h = max(len(rows), 1) * 14
    ax = "".join(f'<span class="ax" style="left:{xp(t):.2f}%">{pct(t, 0) if t else "0%"}</span>'
                 for t in (-M, -M / 2, 0, M / 2, M))
    return f"""<div class="strip" style="height:{h}px">
  <span class="zero" style="left:{xp(0):.2f}%"></span>
  <span class="avg" style="left:{xp(mean):.2f}%"><span>avg {pct(mean)}</span></span>
  {''.join(dots)}{ax}
</div>"""


def viz_movers(movers, spy_1d: float) -> str:
    rows_ = sorted(movers, key=lambda m: -m["r1"])
    if not rows_:
        return ""
    M = max([abs(m["r1"]) for m in rows_] + [abs(spy_1d)]) or 0.01
    xp = lambda v: 50 + v / M * 40
    out = [f'<span></span><div class="trk hdr"><span class="spyl" style="left:{xp(spy_1d):.2f}%">SPY {pct(spy_1d)}</span></div>']
    for m in rows_:
        v = m["r1"]
        x0, x1 = sorted((50, xp(v)))
        c = "var(--gain)" if v > 0 else "var(--loss)"
        lab = (f'<span class="v" style="left:calc({x1:.2f}% + 6px)">{pct(v)}</span>' if v > 0 else
               f'<span class="v" style="left:calc({x0:.2f}% - 6px);transform:translate(-100%,-50%)">{pct(v)}</span>')
        out.append(f'<span class="tk">{html.escape(m["t"])}</span><div class="trk">'
                   f'<span class="zero" style="left:50%"></span><span class="spy" style="left:{xp(spy_1d):.2f}%"></span>'
                   f'<span class="b {"pos" if v > 0 else "neg"}" style="--c:{c};left:{x0:.2f}%;width:{x1 - x0:.2f}%"></span>{lab}</div>')
    return f'<div class="mvbars" role="img" aria-label="One-day change of the biggest movers">{"".join(out)}</div>'


def viz_odds(up: float, dn: float) -> str:
    w = lambda p: min(p, 0.5) / 0.5 * 50
    return (f'<span class="odds" title="↓10% in {dn:.0%} of weeks, ↑10% in {up:.0%}">'
            f'<span>{dn:.0%}</span><span class="otr"><i class="odn" style="width:{w(dn):.1f}%"></i>'
            f'<i class="oup" style="width:{w(up):.1f}%"></i></span><span>{up:.0%}</span></span>')


def viz_timeline(asof: dt.date, window, closed, items, days=35) -> str:
    """Earnings dates over the next few weeks with the hold window shaded. items: [(ticker, date, flag)]."""
    end = asof + dt.timedelta(days)
    xp = lambda d: (d - asof).days / days * 100
    inside = sorted([(t, d, f) for t, d, f in items if d and asof < d <= end], key=lambda z: z[1])
    later = len([1 for t, d, f in items if d and d > end])
    rows: list[float] = []
    evs = []
    for t, d, f in inside:
        xv = xp(d)
        r = next((i for i, last in enumerate(rows) if xv - last > 9), None)
        if r is None:
            rows.append(xv)
            r = len(rows) - 1
        else:
            rows[r] = xv
        c = {"now": "var(--critical)", "soon": "var(--warning)"}.get(f, "var(--muted)")
        evs.append(f'<span class="ev" tabindex="0" style="--c:{c};left:{xv:.2f}%;bottom:{r * 30 + 6}px" '
                   f'data-tip="{html.escape(t)}|Reports {fmt_d(d, True)}"><b>{html.escape(t)}</b><i></i></span>')
    h = max(len(rows), 1) * 30 + 30
    hx0, hx1 = xp(window[0]) - 100 / days / 2, xp(window[-1]) + 100 / days / 2
    ticks = "".join(f'<span class="tick" style="left:{xp(d):.2f}%">{d:%b %-d}</span>'
                    for d in (asof + dt.timedelta(i) for i in range(1, days)) if d.weekday() == 0)
    shut = "".join(f'<span class="shut" style="left:{xp(d):.2f}%" title="Market closed: {html.escape(nm)}"></span>'
                   for d, nm in closed if asof < d <= end)
    tail = f'<p class="note">{later} more suggested stocks report after {fmt_d(end)}. They are listed in the table below.</p>' if later else ""
    return f"""<div class="tl" style="height:{h}px" role="img" aria-label="Earnings dates over the next {days} days">
  <span class="hold" style="left:{hx0:.2f}%;width:{hx1 - hx0:.2f}%"><span>Hold window</span></span>
  {''.join(evs)}<span class="axis"></span>{shut}{ticks}
</div>{tail}"""


CSS = """
*,*::before,*::after{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0}
[hidden]{display:none!important}
img{max-width:100%}
/* Layout: single reading column, a trading-desk "tape" strip up top, then scoreboard, watchlist, plan */
:root{
  --bg:#f5f6f8; --surface:#ffffff; --fg:#17202b; --muted:#5b6675; --line:#dde1e7;
  --accent:#2747b8; --up:#0b6e79; --down:#b23c1c; --flat:#b4bcc8; --warn-bg:#fff4e0; --warn-fg:#7a4a00;
  /* chart marks: gain/loss pair and series slots validated for color-vision deficiency */
  --gain:#0f7f8c; --loss:#d4502b; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  --grid:#e8ebf0; --axis:#c3c9d2; --track:#e3e7ec;
  --good:#0ca30c; --normal:#7b8899; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b;
  --display:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --body:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  color-scheme:light;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0f141b; --surface:#171e27; --fg:#e6ebf1; --muted:#97a3b3; --line:#2a3440;
  --accent:#8aa6ff; --up:#5cc4d3; --down:#f08a6a; --flat:#4a5566; --warn-bg:#3a2a0e; --warn-fg:#ffd28a;
  --gain:#2a9db0; --loss:#e5673f; --s1:#3987e5; --s2:#d95926; --s3:#199e70;
  --grid:#222c39; --axis:#3a4656; --track:#2a3440; --normal:#8794a6; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0f141b; --surface:#171e27; --fg:#e6ebf1; --muted:#97a3b3; --line:#2a3440;
  --accent:#8aa6ff; --up:#5cc4d3; --down:#f08a6a; --flat:#4a5566; --warn-bg:#3a2a0e; --warn-fg:#ffd28a;
  --gain:#2a9db0; --loss:#e5673f; --s1:#3987e5; --s2:#d95926; --s3:#199e70;
  --grid:#222c39; --axis:#3a4656; --track:#2a3440; --normal:#8794a6; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--body);font-size:16px;line-height:1.55;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:32px 64px}
.wrap{max-width:880px;margin:0 auto;display:flex;flex-direction:column;gap:44px}
h1,h2,h3{font-family:var(--display);text-wrap:balance;margin:0;line-height:1.2}
h1{font-size:2.1rem;font-weight:700;letter-spacing:-.01em}
h2{font-size:1.35rem;font-weight:600}
h3{font-size:1rem;font-weight:600}
p{margin:0;max-width:68ch}
section{display:flex;flex-direction:column;gap:14px}
.eyebrow{font-family:var(--mono);font-size:.75rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
td.n,.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.up{color:var(--up)} .down{color:var(--down)}
header{display:flex;flex-direction:column;gap:10px}
.lede{font-size:1.1rem}
.verdict{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px 20px;display:flex;flex-direction:column;gap:8px}
.verdict ul{margin:0;padding-left:1.2em;display:flex;flex-direction:column;gap:6px}
.tape{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.tape div{background:var(--surface);padding:12px 14px;display:flex;flex-direction:column;gap:2px}
.tape .k{font-family:var(--mono);font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.tape .v{font-family:var(--mono);font-size:1.3rem;font-variant-numeric:tabular-nums}
.tape .s{font-size:.82rem;color:var(--muted)}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-family:var(--mono);font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:500}
tr:last-child td{border-bottom:0}
td.n,th.n{text-align:right}
td.name{white-space:normal;min-width:190px}
tr.base td{background:color-mix(in srgb,var(--accent) 7%,transparent)}
.bar{display:flex;gap:2px;height:10px;width:130px}
.bar i{display:block;height:100%;min-width:0}
.bar i:first-child{border-radius:5px 0 0 5px} .bar i:last-child{border-radius:0 5px 5px 0}
.bar .t{background:var(--gain)} .bar .x{background:var(--flat)} .bar .s{background:var(--loss)}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:.82rem;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{width:10px;height:10px;border-radius:2px;display:inline-block}
.legend i.lk{width:14px;height:2px;border-radius:1px}
.legend i.dotkey{border-radius:50%}
.note{font-size:.88rem;color:var(--muted)}
.groups{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:16px}
.group{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;display:flex;flex-direction:column;gap:10px;min-width:0}
.group p{font-size:.88rem;color:var(--muted)}
.tag{align-self:flex-start;font-family:var(--mono);font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;padding:3px 8px;border-radius:999px;border:1px solid var(--line);color:var(--muted)}
.mini{width:100%;font-size:.84rem}
.mini th,.mini td{padding:6px 4px}
.ed{font-family:var(--mono);font-size:.8rem;font-variant-numeric:tabular-nums;color:var(--muted)}
.ed.soon,.ed.now{color:var(--warn-fg);background:var(--warn-bg);border-radius:4px;padding:1px 5px;font-weight:500}
.ed.now{color:var(--down);background:color-mix(in srgb,var(--down) 14%,transparent)}
.callout{background:var(--warn-bg);color:var(--warn-fg);border-radius:10px;padding:14px 16px;font-size:.92rem;display:flex;flex-direction:column;gap:8px}
.callout p{max-width:none}
.plan{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}
.plan div{border-top:2px solid var(--accent);padding-top:10px;display:flex;flex-direction:column;gap:6px}
.plan p{font-size:.92rem}
.math{font-family:var(--mono);font-size:.9rem;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;overflow-x:auto;white-space:pre;line-height:1.7}
/* app shell: menu in a left pane, one view at a time on the right */
.shell{display:grid;grid-template-columns:220px minmax(0,1fr);gap:48px;max-width:1160px;margin:0 auto;align-items:start}
.side{position:sticky;top:calc(env(safe-area-inset-top,0px) + 24px);display:flex;flex-direction:column;gap:18px}
.brand{display:flex;flex-direction:column;gap:2px;padding-inline:12px}
.brand b{font-family:var(--display);font-size:1.15rem}
.brand span{font-family:var(--mono);font-size:.72rem;color:var(--muted)}
.menu{display:flex;flex-direction:column;gap:2px}
.menu a{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:8px 12px;border-radius:8px;color:var(--fg);text-decoration:none;font-size:.95rem;white-space:nowrap}
.menu a:hover{background:color-mix(in srgb,var(--surface) 70%,transparent)}
.menu a[aria-current="page"]{background:var(--surface);box-shadow:inset 3px 0 0 var(--accent);font-weight:600}
.menu a:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.count{font-family:var(--mono);font-size:.72rem;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:0 7px;line-height:1.6}
.count.warn{color:var(--warn-fg);background:var(--warn-bg);border-color:transparent}
main{min-width:0;max-width:880px}
.view{display:flex;flex-direction:column;gap:14px}
.view > .group-head{display:flex;flex-direction:column;gap:10px}
.movers{display:flex;flex-direction:column;gap:10px}
.mv{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;display:grid;grid-template-columns:110px minmax(0,1fr);gap:16px}
.mv .tk{display:flex;flex-direction:column;gap:2px}
.mv .tk b{font-family:var(--mono);font-size:1.05rem}
.mv .tk .r{font-family:var(--mono);font-size:1.35rem;font-variant-numeric:tabular-nums}
.mv .tk small{font-family:var(--mono);color:var(--muted);font-size:.75rem}
.mv .why{display:flex;flex-direction:column;gap:8px;min-width:0}
.mv .why p{font-size:.93rem}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{font-family:var(--mono);font-size:.72rem;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--muted)}
.chip.hot{color:var(--warn-fg);background:var(--warn-bg);border-color:transparent}
.src{font-size:.8rem;color:var(--muted);display:flex;flex-direction:column;gap:2px}
.src a{color:var(--accent);text-decoration:none}
.src a:hover{text-decoration:underline}
/* ---- visuals */
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;display:flex;flex-direction:column;gap:14px;min-width:0}
.mood{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px;align-items:start}
.lk{display:inline-block;width:14px;height:2px;border-radius:1px;background:var(--c);flex:none}
a.q{display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:50%;border:1px solid var(--line);
  font:500 .62rem/1 var(--body);color:var(--muted);text-decoration:none;margin-left:5px;vertical-align:middle;letter-spacing:0;text-transform:none;flex:none}
a.q:hover,a.q:focus-visible{color:var(--accent);border-color:var(--accent);outline:none}
.gauge{display:flex;flex-direction:column;gap:6px;min-width:0}
.g-head{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.g-val{font-size:2rem;font-weight:600;line-height:1}
.g-zone{display:inline-flex;align-items:center;gap:6px;font-weight:600}
.g-zone i{width:10px;height:10px;border-radius:50%;background:var(--zc)}
.g-track{position:relative;display:flex;gap:2px;height:12px;margin-top:20px}
.g-seg{background:var(--track);min-width:0}
.g-seg:first-child{border-radius:6px 0 0 6px} .g-seg:last-child{border-radius:0 6px 6px 0}
.g-seg.on{background:var(--zc)}
.g-mark{position:absolute;top:-5px;width:4px;height:22px;margin-left:-2px;background:var(--fg);border-radius:2px;box-shadow:0 0 0 2px var(--surface)}
.g-mark.prev{width:10px;height:10px;top:1px;margin-left:-5px;border-radius:50%;background:var(--surface);box-shadow:none;border:2px solid var(--muted)}
.g-mark .lab{position:absolute;bottom:calc(100% + 2px);left:50%;transform:translateX(-50%);font:.7rem var(--mono);color:var(--fg)}
.g-labels{display:flex;gap:2px;font-size:.7rem;line-height:1.3;color:var(--muted)}
.g-labels span{min-width:0;text-align:center}
.g-labels b{font-weight:600;color:var(--fg)}
.pills{display:flex;flex-wrap:wrap;gap:8px}
.pill{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;font-size:.85rem;background:var(--track)}
.pill b{font-family:var(--mono);font-weight:600}
.pill .ar{font-size:.7rem}
.pill.dn{background:color-mix(in srgb,var(--loss) 14%,var(--surface))} .pill.dn .ar{color:var(--down)}
.pill.up{background:color-mix(in srgb,var(--gain) 14%,var(--surface))} .pill.up .ar{color:var(--up)}
.lc{margin:0;display:flex;flex-direction:column;gap:10px;min-width:0}
.lc figcaption{display:flex;flex-direction:column;gap:2px}
.lc .plot{position:relative;margin:6px 84px 0 40px;touch-action:pan-y;outline:none}
.lc .plot:focus-visible{box-shadow:0 0 0 2px var(--accent);border-radius:2px}
.lc svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.lc path{fill:none;stroke-width:2px;stroke-linejoin:round;stroke-linecap:round;vector-effect:non-scaling-stroke}
.lc .gl{position:absolute;left:0;right:0;height:1px;background:var(--grid)}
.lc .yt{position:absolute;right:calc(100% + 8px);transform:translateY(-50%);font:.7rem var(--mono);color:var(--muted);font-variant-numeric:tabular-nums}
.lc .xt{position:relative;height:16px;margin:0 84px 0 40px}
.lc .xt span{position:absolute;transform:translateX(-50%);font:.7rem var(--mono);color:var(--muted)}
.lc .end{position:absolute;left:calc(100% + 12px);transform:translateY(-50%);display:inline-flex;align-items:center;gap:5px;font-size:.75rem;white-space:nowrap}
.lc .end b{font-family:var(--mono);font-weight:500}
.lc .ref{position:absolute;left:0;right:0;border-top:1px solid var(--axis)}
.lc .ref span{position:absolute;left:4px;bottom:2px;font-size:.68rem;color:var(--muted);background:var(--surface);padding:0 4px;border-radius:3px}
.dotm{position:absolute;width:10px;height:10px;margin:-5px 0 0 -5px;border-radius:50%;background:var(--c);box-shadow:0 0 0 2px var(--surface);pointer-events:none}
.xh{position:absolute;top:0;bottom:0;width:1px;background:var(--axis);pointer-events:none}
#tip{position:fixed;z-index:50;pointer-events:none;background:var(--surface);color:var(--fg);border:1px solid var(--line);border-radius:8px;
  padding:8px 10px;font-size:.8rem;box-shadow:0 6px 20px rgba(0,0,0,.14);display:flex;flex-direction:column;gap:3px;max-width:260px}
#tip .tr{display:flex;align-items:center;gap:8px}
#tip .th{font-weight:600}
#tip .tv{font-family:var(--mono);font-weight:600}
#tip .tn{color:var(--muted)}
.strip{position:relative;margin:30px 6px 26px}
.strip .zero{position:absolute;top:-10px;bottom:-10px;width:1px;background:var(--axis)}
.strip .avg{position:absolute;top:-10px;width:0;height:0}
.strip .avg span{position:absolute;bottom:2px;left:0;transform:translateX(-50%);font:.7rem var(--mono);color:var(--muted);white-space:nowrap}
.strip .avg::after{content:"";position:absolute;left:-1px;top:0;width:2px;height:6px;background:var(--fg)}
.strip .d{position:absolute;width:10px;height:10px;margin:-5px 0 0 -5px;border-radius:50%;background:var(--c);box-shadow:0 0 0 2px var(--surface);outline:none}
.strip .d::before{content:"";position:absolute;inset:-7px}
.strip .d:hover,.strip .d:focus-visible{box-shadow:0 0 0 2px var(--surface),0 0 0 4px var(--c)}
.strip .ax{position:absolute;top:calc(100% + 10px);transform:translateX(-50%);font:.7rem var(--mono);color:var(--muted)}
.mvbars{display:grid;grid-template-columns:auto minmax(0,1fr);gap:4px 10px;align-items:center}
.mvbars .tk{font:.8rem var(--mono)}
.mvbars .trk{position:relative;height:18px}
.mvbars .trk.hdr{height:16px}
.mvbars .b{position:absolute;top:3px;height:12px;background:var(--c)}
.mvbars .b.pos{border-radius:0 4px 4px 0} .mvbars .b.neg{border-radius:4px 0 0 4px}
.mvbars .v{position:absolute;top:50%;transform:translateY(-50%);font:.72rem var(--mono);white-space:nowrap}
.mvbars .zero{position:absolute;top:-3px;bottom:-3px;width:1px;background:var(--axis)}
.mvbars .spy{position:absolute;top:-3px;bottom:-3px;border-left:1px solid var(--muted);opacity:.6}
.mvbars .spyl{position:absolute;bottom:0;transform:translateX(-50%);font:.7rem var(--mono);color:var(--muted);white-space:nowrap}
.odds{display:inline-grid;grid-template-columns:2.4em 60px 2.4em;align-items:center;gap:5px;font:.76rem var(--mono);font-variant-numeric:tabular-nums}
.odds > span:first-child{text-align:right}
.otr{position:relative;height:8px}
.otr::after{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--axis)}
.otr i{position:absolute;top:0;height:8px}
.otr .odn{right:50%;margin-right:1px;background:var(--loss);border-radius:4px 0 0 4px}
.otr .oup{left:50%;margin-left:1px;background:var(--gain);border-radius:0 4px 4px 0}
.tl{position:relative;margin:4px 8px 30px}
.tl .hold{position:absolute;top:0;bottom:0;background:color-mix(in srgb,var(--accent) 9%,transparent);border-radius:4px}
.tl .hold span{position:absolute;top:4px;left:6px;font-size:.68rem;color:var(--muted);white-space:nowrap}
.tl .axis{position:absolute;left:0;right:0;bottom:0;height:1px;background:var(--axis)}
.tl .tick{position:absolute;top:calc(100% + 6px);transform:translateX(-50%);font:.68rem var(--mono);color:var(--muted);white-space:nowrap}
.tl .shut{position:absolute;bottom:-3px;width:6px;height:6px;margin-left:-3px;background:var(--muted);transform:rotate(45deg)}
.tl .ev{position:absolute;transform:translateX(-50%);display:flex;flex-direction:column;align-items:center;gap:3px;outline:none}
.tl .ev b{font:500 .72rem var(--mono)}
.tl .ev i{width:10px;height:10px;border-radius:50%;background:var(--c);box-shadow:0 0 0 2px var(--surface)}
.tl .ev:focus-visible i,.tl .ev:hover i{box-shadow:0 0 0 2px var(--surface),0 0 0 4px var(--c)}
.dotkey{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--c)}
details.data summary{cursor:pointer;font-size:.82rem;color:var(--muted)}
details.data table{margin-top:8px;font-size:.82rem}
/* glossary */
.gtools{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.gtools input{flex:1;min-width:0;font:inherit;padding:9px 12px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--fg)}
.gtools input:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.gcat{display:flex;flex-direction:column;gap:10px}
.terms{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:10px;margin:0}
.term{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px;display:flex;flex-direction:column;gap:5px;scroll-margin:90px;transition:box-shadow .3s}
.term dt{font-weight:600}
.term dd{margin:0;font-size:.9rem}
.term dd.ex{font-size:.84rem;color:var(--muted)}
.term.flash{box-shadow:0 0 0 2px var(--accent)}
@media (prefers-reduced-motion:reduce){.term{transition:none}}
@media (max-width:760px){
  .mood{grid-template-columns:minmax(0,1fr)}
  .lc .plot,.lc .xt{margin-right:70px}
  .group{padding:14px 12px}
  .odds{grid-template-columns:2.2em 44px 2.2em;gap:4px}
  .mini th,.mini td{padding:6px 3px}
  .shell{grid-template-columns:minmax(0,1fr);gap:20px}
  .side{top:env(safe-area-inset-top,0px);z-index:5;background:var(--bg);padding-block:8px;gap:8px;border-bottom:1px solid var(--line)}
  body::before{content:"";position:fixed;inset:0 0 auto;height:env(safe-area-inset-top,0px);z-index:10;background:var(--bg)}
  .brand{padding-inline:0}
  .menu{flex-direction:row;overflow-x:auto;gap:4px;scrollbar-width:none}
  .menu a{padding:6px 10px}
  .menu a[aria-current="page"]{box-shadow:inset 0 -2px 0 var(--accent)}
  .mv{grid-template-columns:minmax(0,1fr);gap:8px}
  .mv .tk{flex-direction:row;align-items:baseline;gap:10px;flex-wrap:wrap}
}
"""

JS = """
(function(){
  var $=function(s,r){return [].slice.call((r||document).querySelectorAll(s));};
  var views=$('.view'), links=$('.menu a');

  /* ---- views: #name shows a section; #g-term opens the glossary at that term */
  function show(scroll){
    var h=location.hash.slice(1), id=h, term=null;
    if(h.indexOf('g-')===0 && document.getElementById(h)){ id='glossary'; term=document.getElementById(h); }
    /* views use data-view (not id) so the browser never jumps to them on its own */
    if(!views.some(function(v){return v.getAttribute('data-view')===id;})) id='overview';
    views.forEach(function(v){ v.hidden = v.getAttribute('data-view')!==id; });
    links.forEach(function(a){ if(a.getAttribute('href')==='#'+id) a.setAttribute('aria-current','page'); else a.removeAttribute('aria-current'); });
    hideTip();
    if(term){
      if(filter && filter.value){ filter.value=''; applyFilter(); }
      requestAnimationFrame(function(){ term.scrollIntoView({block:'center'}); term.classList.add('flash');
        setTimeout(function(){ term.classList.remove('flash'); },1800); });
    } else if(scroll) window.scrollTo(0,0);
  }

  /* ---- glossary search */
  var filter=document.getElementById('gloss-filter'), count=document.getElementById('gloss-count');
  function applyFilter(){
    var q=(filter.value||'').trim().toLowerCase();
    $('.term').forEach(function(t){ t.hidden = !!q && t.textContent.toLowerCase().indexOf(q)<0; });
    $('.gcat').forEach(function(c){ c.hidden = !c.querySelector('.term:not([hidden])'); });
    var n=$('.term:not([hidden])').length; count.textContent = n+(n===1?' term':' terms');
  }
  if(filter) filter.addEventListener('input',applyFilter);

  /* ---- one shared tooltip; content set with textContent only */
  var tip=document.createElement('div'); tip.id='tip'; tip.hidden=true; tip.setAttribute('role','status'); document.body.appendChild(tip);
  function hideTip(){ tip.hidden=true; }
  function fill(rows){
    tip.textContent='';
    rows.forEach(function(r){
      var d=document.createElement('div'); d.className=r.h?'th':'tr';
      if(r.h){ d.textContent=r.h; tip.appendChild(d); return; }
      if(r.c){ var k=document.createElement('span'); k.className='lk'; k.style.setProperty('--c',r.c); d.appendChild(k); }
      if(r.v!=null){ var v=document.createElement('span'); v.className='tv'; v.textContent=r.v; d.appendChild(v); }
      if(r.n){ var n=document.createElement('span'); n.className='tn'; n.textContent=r.n; d.appendChild(n); }
      tip.appendChild(d);
    });
    tip.hidden=false;
  }
  function place(x,y){
    var w=tip.offsetWidth, h=tip.offsetHeight, l=x+14, t=y-h-12;
    if(l+w>innerWidth-8) l=x-14-w; if(l<8) l=8; if(t<8) t=y+18;
    tip.style.left=l+'px'; tip.style.top=t+'px';
  }
  $('[data-tip]').forEach(function(el){
    function on(ev){
      var parts=el.getAttribute('data-tip').split('|');
      fill([{h:parts[0]}].concat(parts.slice(1).map(function(s){ return {n:s}; })));
      var r=el.getBoundingClientRect(), mouse=ev.type!=='focus';
      place(mouse?ev.clientX:r.left+r.width/2, mouse?ev.clientY:r.top);
    }
    el.addEventListener('pointerenter',on); el.addEventListener('pointermove',on); el.addEventListener('focus',on);
    el.addEventListener('pointerleave',function(ev){ if(ev.pointerType!=='touch') hideTip(); });
    el.addEventListener('blur',hideTip);
  });

  /* ---- line charts: crosshair snaps to the nearest day; every series in one readout */
  $('.lc').forEach(function(fig){
    var d=JSON.parse(fig.getAttribute('data-d')), plot=fig.querySelector('.plot'), n=d.dates.length, idx=n-1;
    var xh=document.createElement('div'); xh.className='xh'; xh.hidden=true; plot.appendChild(xh);
    var dots=d.series.map(function(s){ var m=document.createElement('span'); m.className='dotm hv'; m.style.setProperty('--c',s.c); m.hidden=true; plot.appendChild(m); return m; });
    function y(v){ return (1-(v-d.lo)/(d.hi-d.lo))*100; }
    function fmt(v){ return d.fmt==='price' ? v.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}) : v.toFixed(1); }
    function at(i,cx,cy){
      idx=Math.max(0,Math.min(n-1,i)); var x=idx/(n-1)*100, rows=[{h:d.dates[idx]}];
      xh.style.left=x+'%'; xh.hidden=false;
      d.series.forEach(function(s,k){ var v=s.v[idx]; dots[k].hidden=(v==null);
        if(v!=null){ dots[k].style.left=x+'%'; dots[k].style.top=y(v)+'%'; }
        rows.push({c:s.c, v:v==null?'–':fmt(v), n:s.n}); });
      fill(rows); var r=plot.getBoundingClientRect();
      place(cx!=null?cx:r.left+r.width*x/100, cy!=null?cy:r.top+24);
    }
    function off(){ xh.hidden=true; dots.forEach(function(m){ m.hidden=true; }); hideTip(); }
    plot.addEventListener('pointermove',function(ev){ var r=plot.getBoundingClientRect(); at(Math.round((ev.clientX-r.left)/r.width*(n-1)),ev.clientX,ev.clientY); });
    plot.addEventListener('pointerdown',function(ev){ var r=plot.getBoundingClientRect(); at(Math.round((ev.clientX-r.left)/r.width*(n-1)),ev.clientX,ev.clientY); });
    plot.addEventListener('pointerleave',function(ev){ if(ev.pointerType!=='touch') off(); });
    plot.addEventListener('focus',function(){ at(idx); });
    plot.addEventListener('blur',off);
    plot.addEventListener('keydown',function(ev){
      if(ev.key==='ArrowLeft'||ev.key==='ArrowRight'){ ev.preventDefault(); at(idx+(ev.key==='ArrowLeft'?-1:1)); }
      if(ev.key==='Escape') off();
    });
  });
  document.addEventListener('pointerdown',function(ev){ if(!ev.target.closest('.plot,[data-tip]')) { hideTip(); $('.xh,.dotm.hv').forEach(function(m){ m.hidden=true; }); } });
  window.addEventListener('scroll',hideTip,{passive:true});

  window.addEventListener('hashchange',function(){ show(true); });
  show(true);  /* also on first load, so a #view link never opens with its heading under the sticky menu */
})();
"""


def render(ctx) -> str:
    e = html.escape
    R, reg, st, groups, ed = ctx["results"], ctx["regime"], ctx["study"], ctx["groups"], ctx["earnings"]
    below = reg["below50"]
    base = R["random"]["all"]
    win_days = ctx["window"]
    win_start, win_end = win_days[0], win_days[-1]
    soon_end = ctx["soon_end"]

    # --- earnings flags
    def ed_flag(t):
        nxt = ed.get(t, {}).get("next")
        if nxt is None:
            return "", "n/a"
        if win_start <= nxt <= win_end:
            return "now", fmt_d(nxt)
        if win_end < nxt <= soon_end:
            return "soon", fmt_d(nxt)
        return "", fmt_d(nxt)

    suggested = [r["t"] for g in groups for r in g["rows"]]
    in_win = [(t, ed[t]["next"]) for t in suggested if ed_flag(t)[0] == "now"]
    soon = [(t, ed[t]["next"]) for t in suggested if ed_flag(t)[0] == "soon"]
    recent = sorted([(t, ed[t]["last"]) for t in suggested if ed[t].get("last")
                     and (ctx["asof"] - ed[t]["last"]).days <= RECENT_DAYS], key=lambda z: z[1], reverse=True)

    # --- header & verdict
    best_reg = max((k for k in SETUPS if k != "random"), key=lambda k: np.nan_to_num(R[k]["below" if below else "above"]["avg"], nan=-9))
    breg = R[best_reg]["below" if below else "above"]
    selloff = reg["spy_5d"] < -0.05 and reg["vix"] > 30
    if selloff:
        mkt_head = "A sharp selloff with fear spiking"
        mkt_bullet = ("The market is in a sharp selloff right now. Historically that has meant slightly better bounce odds "
                      "and much wider swings both ways.")
    elif below:
        mkt_head = "The market is trading under its 50-day average"
        mkt_bullet = "SPY is below its 50-day average, a regime where breakouts have tended to fail."
    else:
        mkt_head = "The market is in an uptrend"
        mkt_bullet = "SPY is above its 50-day average, a regime where pullbacks to support have tended to work best."
    pos = ctx["account"] * ctx["risk"] / STOP
    risk_amt = ctx["account"] * ctx["risk"]

    # Security: a complete document whose Content-Security-Policy allows exactly one script (ours, by
    # hash), no network requests of any kind, no forms, no plugins and no <base> rewriting. Any markup
    # that slipped into the page from third-party text (news headlines) could not run code.
    script_hash = base64.b64encode(hashlib.sha256(JS.encode("utf-8")).digest()).decode()
    csp = ("default-src 'none'; "
           f"script-src 'sha256-{script_hash}'; "
           "style-src 'unsafe-inline'; img-src data:; "
           "base-uri 'none'; form-action 'none'; object-src 'none'")
    out = ['<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
           f'<meta http-equiv="Content-Security-Policy" content="{csp}">',
           '<meta name="referrer" content="no-referrer">',
           '<meta name="robots" content="noindex, nofollow">',
           "<title>10% Week Odds</title>",
           f"<style>{CSS}</style>\n</head>\n<body>"]
    gainers, losers = ctx["gainers"], ctx["losers"]
    top_g = gainers[0] if gainers else None
    top_l = losers[0] if losers else None
    menu = [("overview", "Overview", ""), ("market", "Market", ""),
            ("movers", "Day movers", str(len(gainers) + len(losers))),
            ("watchlist", "Watchlist", str(len(suggested))),
            ("earnings", "Earnings", (f"!{len(in_win)}" if in_win else str(len(in_win) + len(soon)))),
            ("setups", "Setup scoreboard", ""), ("playbook", "Playbook", ""),
            ("glossary", "Glossary", str(len(GLOSSARY))), ("method", "Method", "")]
    hist = ctx["hist"]
    zone = vix_zone(reg["vix"])
    trend_word, _ = trend_state(reg)
    spy_chart = viz_line(
        "SPY and its moving averages", "Last 6 months of daily closes. Hover, tap, or use the arrow keys to read any day.",
        hist["dates"], [{"n": "SPY close", "c": "var(--s1)", "v": hist["spy"]},
                        {"n": "50-day average", "c": "var(--s2)", "v": hist["sma50"]},
                        {"n": "200-day average", "c": "var(--s3)", "v": hist["sma200"]}],
        fmt="price", height=210,
        aria=f"SPY closed at {reg['spy']:.2f}, {pct(reg['spy'] / reg['sma50'] - 1)} vs its 50-day average and "
             f"{pct(reg['spy'] / reg['sma200'] - 1)} vs its 200-day average.")
    vix_chart = viz_line(
        "VIX over the last 6 months", "The darker lines at 20 and 30 mark where the elevated and fear zones begin.",
        hist["dates"], [{"n": "VIX", "c": "var(--s1)", "v": hist["vix"]}], fmt="num", height=150,
        refs=[(20, ""), (30, "")], aria=f"VIX is {reg['vix']:.1f}, in the {zone[0]} zone.")
    last10 = list(zip(hist["dates"], hist["spy"], hist["sma50"], hist["sma200"], hist["vix"]))[-10:][::-1]
    data_table = ('<details class="data"><summary>Show the last 10 sessions as a table</summary><div class="scroll"><table>'
                  '<thead><tr><th>Date</th><th class="n">SPY</th><th class="n">50-day</th><th class="n">200-day</th><th class="n">VIX</th></tr></thead><tbody>'
                  + "".join(f'<tr><td>{d:%a %b %-d}</td><td class="n">{s:,.2f}</td><td class="n">{a:,.2f}</td>'
                            f'<td class="n">{b:,.2f}</td><td class="n">{v:.1f}</td></tr>' for d, s, a, b, v in last10)
                  + "</tbody></table></div></details>")
    nav = "".join(
        f'<a href="#{k}"{" aria-current=page" if k == "overview" else ""}>{lbl}'
        + (f'<span class="count{" warn" if c.startswith("!") else ""}">{c.lstrip("!")}</span>' if c else "")
        + "</a>" for k, lbl, c in menu)
    out.append(f"""<div class="shell">
<aside class="side">
  <div class="brand"><b>10% Week Odds</b><span>as of {ctx['asof']:%d %b %Y}</span></div>
  <nav class="menu" aria-label="Report sections">{nav}</nav>
</aside>
<main>""")
    mover_tile = lambda m, k: (f'<div><span class="k">{k}</span><span class="v {cls(m["r1"])}">{e(m["t"])} {pct(m["r1"])}</span>'
                               f'<span class="s"><a href="#movers" style="color:inherit">See why it moved</a></span></div>') if m else ""
    out.append(f"""
<section data-view="overview" class="view">
<header>
  <span class="eyebrow">Swing trading backtest · data through {ctx['asof'].strftime('%a %-d %b %Y')}</span>
  <h1>What are the odds of +{TARGET:.0%} in a week?</h1>
  <p class="lede">I tested {ctx['n_univ']} liquid US stocks over {ctx['period']} of daily data. Every trade enters at the next open, takes profit at +{TARGET:.0%}, stops out at −{STOP:.0%}, and exits after {HOLD} trading days if neither is hit. That is {base['trades']:,} simulated trades.</p>
</header>
<div class="card">
  <h3>Market mood at a glance</h3>
  <div class="mood">
    {viz_gauge(reg['vix'], reg['vix_5d_ago'], compact=True)}
    <div style="display:flex;flex-direction:column;gap:10px;min-width:0">
      {viz_pills(reg)}
      <p class="note">{e(trend_word)} with {e(zone[0].lower())} in the options market. <a href="#market" style="color:var(--accent)">See the charts</a></p>
    </div>
  </div>
</div>
<div class="verdict">
  <h3>Bottom line</h3>
  <ul>
    <li>Picking a stock at random, the +{TARGET:.0%} target hit <b>{base['target']:.1%}</b> of the time and the −{STOP:.0%} stop hit <b>{base['stop']:.1%}</b> of the time.</li>
    <li>No common setup turns that into a reliable edge. The setups that hit +{TARGET:.0%} more often also hit the stop more often.</li>
    <li>{mkt_bullet} In this regime the best setup so far has been <b>{e(SETUPS[best_reg][0].lower())}</b> at {pct(breg['avg'], 2)} per trade.</li>
    <li>With ${ctx['account']:,.0f}, risk about ${risk_amt:,.0f} per trade, which means positions of roughly ${pos:,.0f}. Expect 10% weeks to be rare wins, not the plan.</li>
  </ul>
</div>
<h3 style="margin-top:8px">Coming up</h3>
<div class="tape">
  <div><span class="k">Next hold window</span><span class="v">{fmt_d(win_start)} → {fmt_d(win_end)}</span><span class="s">{len(win_days)} sessions{' · ' + ', '.join(n for _, n in ctx['skipped']) + ' closed' if ctx['skipped'] else ''}</span></div>
  <div><span class="k">Earnings in window</span><span class="v {'down' if in_win else ''}">{len(in_win)}</span><span class="s">{e(', '.join(t for t, _ in in_win)) or 'none of the suggested stocks'} · <a href="#earnings" style="color:inherit">calendar</a></span></div>
  {mover_tile(top_g, 'Top gainer today')}
  {mover_tile(top_l, 'Top loser today')}
</div>
</section>""")

    # --- market
    movers = ", ".join(f"{t} {pct(r)}" for t, r in ctx["movers"])
    out.append(f"""
<section data-view="market" class="view" hidden>
  <span class="eyebrow">Market right now</span>
  <h2>{mkt_head}</h2>
  {viz_pills(reg)}
  <div class="tape">
    <div><span class="k">SPY close{q('spy', 'SPY')}</span><span class="v">{reg['spy']:.2f}</span><span class="s {cls(reg['spy_1d'])}">{pct(reg['spy_1d'])} on the day</span></div>
    <div><span class="k">SPY 5-day{q('ret5', '5-day change')}</span><span class="v {cls(reg['spy_5d'])}">{pct(reg['spy_5d'])}</span><span class="s">QQQ {pct(reg['qqq_5d'])}{q('qqq', 'QQQ')}</span></div>
    <div><span class="k">VIX{q('vix', 'VIX')}</span><span class="v">{reg['vix']:.1f}</span><span class="s">{reg['vix_5d_ago']:.1f} a week ago</span></div>
    <div><span class="k">vs 50-day avg{q('vs50', 'vs 50-day avg')}</span><span class="v {cls(reg['spy'] / reg['sma50'] - 1)}">{pct(reg['spy'] / reg['sma50'] - 1)}</span><span class="s">50d at {reg['sma50']:.2f}</span></div>
    <div><span class="k">vs 200-day avg{q('vs200', 'vs 200-day avg')}</span><span class="v {cls(reg['spy'] / reg['sma200'] - 1)}">{pct(reg['spy'] / reg['sma200'] - 1)}</span><span class="s">200d at {reg['sma200']:.2f}</span></div>
  </div>
  <div class="card">{spy_chart}{data_table}</div>
  <div class="card"><h3>Fear gauge</h3>{viz_gauge(reg['vix'], reg['vix_5d_ago'])}{vix_chart}</div>
  <p>{ctx['n_drop15']} of the {ctx['n_univ']} stocks are down 15% or more in five days. Biggest 5-day moves: {e(movers)}.</p>
  <h3 style="margin-top:8px">What SPY did the week after sharp selloffs</h3>
  <p>Since {st['since']} there have been {st['events']} separate times SPY fell more than 7% in five days with the VIX above 30. {'The current week qualifies.' if selloff else 'The current week does not qualify, so treat this as background.'}</p>
  <div class="card">
    <h3>SPY's next-week return after each past selloff</h3>
    <span class="note">Each dot is one selloff since {st['since']}. Teal = SPY rose the next week, orange = it fell. Hover or tap a dot for the date.</span>
    {viz_strip(st['points'], st['f5_mean'])}
  </div>
  <div class="tape">
    <div><span class="k">Up next week</span><span class="v">{st['f5_pos']:.0%}</span><span class="s">of {st['with_data']} with data</span></div>
    <div><span class="k">Avg next week</span><span class="v {cls(st['f5_mean'])}">{pct(st['f5_mean'])}</span><span class="s">normal week {pct(st['normal_f5'])}</span></div>
    <div><span class="k">Best / worst</span><span class="v">{pct(st['f5_best'])} / {pct(st['f5_worst'])}</span><span class="s">{st['f5_best_d']} vs {st['f5_worst_d']}</span></div>
    <div><span class="k">Up a month later</span><span class="v">{st['f20_pos']:.0%}</span><span class="s">avg {pct(st['f20_mean'])}</span></div>
  </div>
  <p class="note">The worst month after one of these selloffs was {e(st['f20_worst_d'])}, when SPY fell another {pct(-st['f20_worst'], sign=False)}.</p>
</section>""")

    # --- scoreboard
    rk = "below" if below else "above"
    setup_term = {"random": "base-rate", "oversold": "oversold", "momentum": "momentum", "breakout": "breakout",
                  "highvol": "highvol", "bigdrop": "bigdrop"}
    rows = []
    for k, (label, note, _) in SETUPS.items():
        a, r = R[k]["all"], R[k][rk]
        rows.append(
            f'<tr{" class=base" if k == "random" else ""}><td class="name">{e(label)}{q(setup_term[k], label)}'
            f'{f"<br><span class=note>{e(note)}</span>" if note else ""}</td>'
            f'<td class="n">{a["trades"]:,}</td>'
            f'<td><div class="bar" role="img" aria-label="{a["target"]:.0%} target, {a["time"]:.0%} time exit, {a["stop"]:.0%} stop">'
            f'<i class="t" style="flex:{a["target"]:.4f}"></i><i class="x" style="flex:{a["time"]:.4f}"></i><i class="s" style="flex:{a["stop"]:.4f}"></i></div></td>'
            f'<td class="n">{a["target"]:.1%}</td><td class="n">{a["stop"]:.1%}</td>'
            f'<td class="n {cls(a["avg"])}">{pct(a["avg"], 2)}</td>'
            f'<td class="n {cls(r["avg"])}">{pct(r["avg"], 2)}</td></tr>')
    bd = R["bigdrop"]["all"]
    be = bd["target"] / (bd["target"] + bd["stop"])
    breakeven = STOP / (TARGET + STOP)
    out.append(f"""
<section data-view="setups" class="view" hidden>
  <span class="eyebrow">Setup scoreboard · +{TARGET:.0%} target / −{STOP:.0%} stop / {HOLD} days</span>
  <h2>How common swing setups performed</h2>
  <div class="legend"><span><i style="background:var(--gain)"></i>Hit +{TARGET:.0%} target</span><span><i style="background:var(--flat)"></i>Neither, exited day {HOLD}</span><span><i style="background:var(--loss)"></i>Hit −{STOP:.0%} stop</span></div>
  <div class="scroll"><table>
    <thead><tr><th>Setup</th><th class="n">Trades</th><th>Outcome mix{q('tst', 'target, stop and time exit')}</th><th class="n">Target</th><th class="n">Stop</th><th class="n">Avg / trade{q('avg-trade', 'avg per trade')}</th><th class="n">{'In down markets' if below else 'In up markets'}*</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table></div>
  <p class="note">*Average per trade on days SPY was {'below' if below else 'above'} its 50-day average, which is the situation now. Returns are before commissions, spreads and slippage, which on volatile small caps can easily cost 0.2–0.5% per round trip.</p>
  <h3 style="margin-top:8px">Why a {TARGET:.0%} target with a {STOP:.0%} stop still barely pays</h3>
  <div class="math">Bounce setup:  {bd['target']:.1%} × (+{TARGET:.0%})  +  {bd['stop']:.1%} × (−{STOP:.0%})  +  {bd['time']:.1%} × (day-{HOLD} exit)  ≈ {pct(bd['avg'], 2)} per trade
Breakeven needs: target hits ≥ {breakeven:.0%} of the trades that hit target or stop
Reality:         {bd['target']:.1%} / ({bd['target']:.1%} + {bd['stop']:.1%}) = {be:.0%}</div>
  <p>Raising the odds of hitting +{TARGET:.0%} means choosing stocks that swing more, and those also hit −{STOP:.0%} more often. The averages are close to zero in every row.</p>
</section>""")

    # --- watchlist
    gh = []
    for g in groups:
        trs = []
        for r in g["rows"]:
            fl, txt = ed_flag(r["t"])
            last = ed.get(r["t"], {}).get("last")
            title = f' title="Last reported {fmt_d(last)}"' if last else ""
            trs.append(f'<tr><td class="mono">{e(r["t"])}</td><td><span class="ed {fl}"{title}>{txt}</span></td>'
                       f'<td class="n {cls(r["ret5"])}">{pct(r["ret5"])}</td>'
                       f'<td class="n">{viz_odds(r["up10"], r["dn10"])}</td></tr>')
        gh.append(f"""<div class="group"><span class="tag">{e(g['tag'])}</span><h3>{e(g['title'])}</h3><p>{e(g['desc'])}</p>
<table class="mini"><thead><tr><th>Ticker</th><th>Earnings{q('earnings', 'earnings date')}</th><th class="n">5d</th><th class="n">10% odds{q('odds10', '10% odds')}</th></tr></thead><tbody>{''.join(trs)}</tbody></table></div>""")

    skipped = "; ".join(f"closed {fmt_d(d, True)} ({n})" for d, n in ctx["skipped"])
    halfs = "; ".join(f"early close {fmt_d(d, True)} ({n})" for d, n in ctx["halfs"])
    cal_note = " ".join(x for x in (skipped and f"Markets are {skipped}.", halfs and f"There is an {halfs}.") if x)
    lst = lambda xs: ", ".join(f"<b>{e(t)} ({fmt_d(d, True)})</b>" for t, d in xs)
    callout = [f"<p><b>Earnings and the next hold.</b> A {HOLD}-day hold entered {fmt_d(win_start, True)} runs to {fmt_d(win_end, True)}. {cal_note}</p>"]
    callout.append(f"<p>Reporting inside that window: {lst(in_win)}. An earnings report can gap a stock 15–25% past your stop overnight, and a stop order won't protect you from a gap.</p>"
                   if in_win else "<p>None of the suggested stocks reports inside that window.</p>")
    if soon:
        callout.append(f"<p>Reporting within {SOON_DAYS} trading days after it, so a late entry puts them inside: {lst(soon)}.</p>")
    if recent:
        callout.append(f"<p>Reported in the last {RECENT_DAYS} days, so that news is already in the price: "
                       + ", ".join(f"{e(t)} ({fmt_d(d)})" for t, d in recent[:8]) + ".</p>")
    out.append(f"""
<section data-view="watchlist" class="view" hidden>
  <span class="eyebrow">Watchlist · flagged by the screen on {fmt_d(ctx['asof'])} close</span>
  <h2>Names in play for the next {HOLD} sessions</h2>
  <p>These names trigger a setup from the Setup scoreboard. The odds bar shows how often each stock fell 10% (orange, left) or rose 10% (teal, right) within any 5-day window over the test period. A balanced bar means big moves both ways. These describe the stock's behavior. They are not forecasts.</p>
  <div class="legend"><span><span class="ed now">date</span> reports during the hold</span><span><span class="ed soon">date</span> reports just after it</span>
    <span>{viz_odds(0.30, 0.12)} example: falls 10% in 12% of weeks, rises 10% in 30%</span></div>
  <div class="groups">{''.join(gh)}</div>
  <p class="note">Every date with its status is listed in <a href="#earnings" style="color:var(--accent)">Earnings</a>.</p>
</section>""")

    # --- earnings calendar for suggested stocks
    group_of = {r["t"]: g["tag"] for g in groups for r in g["rows"]}
    status_txt = {"now": "Reports during the hold", "soon": "Reports just after the hold", "": ""}
    closures = []
    for yy in (ctx["asof"].year, ctx["asof"].year + 1):
        closures += sorted(nyse_holidays(yy)[0].items())
    def sort_key(t):
        n = ed.get(t, {}).get("next")
        return n or dt.date.max
    erows = []
    for t in sorted(suggested, key=sort_key):
        nx, last = ed.get(t, {}).get("next"), ed.get(t, {}).get("last")
        fl, _ = ed_flag(t)
        away = f"{(nx - ctx['asof']).days} days" if nx else "–"
        st_ = status_txt[fl] or ("Just reported" if last and (ctx["asof"] - last).days <= RECENT_DAYS else "Clear")
        erows.append(f'<tr><td class="mono">{e(t)}</td><td>{e(group_of[t])}</td>'
                     f'<td><span class="ed {fl}">{fmt_d(nx, True) if nx else "n/a"}</span></td><td class="n">{away}</td>'
                     f'<td><span class="ed">{fmt_d(last) if last else "n/a"}</span></td><td>{st_}</td></tr>')
    out.append(f"""
<section data-view="earnings" class="view" hidden>
  <span class="eyebrow">Earnings calendar · {len(suggested)} suggested stocks</span>
  <h2>When the suggested stocks report</h2>
  <div class="callout">{''.join(callout)}</div>
  <div class="card">
    <h3>Next 5 weeks</h3>
    <div class="legend"><span><i class="dotkey" style="--c:var(--critical)"></i>Reports during the hold</span>
      <span><i class="dotkey" style="--c:var(--warning)"></i>Reports just after it</span>
      <span><i class="dotkey" style="--c:var(--muted)"></i>Later</span>
      <span><i class="dotkey" style="--c:var(--muted);border-radius:0;transform:rotate(45deg) scale(.6)"></i>Market closed</span></div>
    {viz_timeline(ctx['asof'], win_days, closures, [(t, ed.get(t, {}).get('next'), ed_flag(t)[0]) for t in suggested])}
  </div>
  <div class="scroll"><table>
    <thead><tr><th>Ticker</th><th>Group</th><th>Next earnings</th><th class="n">Away</th><th>Last reported</th><th>Status</th></tr></thead>
    <tbody>{''.join(erows)}</tbody>
  </table></div>
  <p class="note">Earnings dates come from Yahoo Finance as of {fmt_d(ctx['asof'])}. Upcoming dates are often estimates until the company confirms them, so check each company's investor-relations page before you trade.</p>
</section>""")

    # --- day movers
    def mover_cards(ms):
        cards = []
        for m in ms:
            src = "".join(
                f'<span>{e(a["source"])} · {a["when"]:%b %-d}: '
                + (f'<a href="{e(a["url"])}" target="_blank" rel="noopener noreferrer nofollow">{e(a["title"])}</a>' if a["url"] else e(a["title"]))
                + "</span>" for a in m["news"])
            chips = "".join(f'<span class="chip{" hot" if c == "Earnings" or c.startswith("Volume") else ""}">{e(c)}</span>' for c in m["chips"])
            wl = f'<span class="chip">On watchlist · {e(group_of[m["t"]])}</span>' if m["t"] in group_of else ""
            cards.append(f"""<div class="mv"><div class="tk"><b>{e(m['t'])}</b><span class="r {cls(m['r1'])}">{pct(m['r1'])}</span><small>${m['close']:,.2f} · 5d {pct(m['ret5'])}</small></div>
<div class="why"><p>{e(m['why'])}</p><div class="chips">{chips}{wl}</div>{f'<div class="src">{src}</div>' if src else ''}</div></div>""")
        return "".join(cards) or '<p class="note">No stocks in the universe moved this way today.</p>'

    out.append(f"""
<section data-view="movers" class="view" hidden>
  <span class="eyebrow">Day movers · {ctx['asof']:%a %-d %b} session</span>
  <h2>Biggest moves of the day</h2>
  <p>The largest one-day gains and losses among the {ctx['n_univ']} stocks in the universe. SPY moved {pct(reg['spy_1d'])} on the day. Each explanation comes from the most relevant recent Yahoo Finance headline that names the company, plus earnings timing and volume.</p>
  <div class="card">
    <h3>One-day change</h3>
    <span class="note">Bars start at zero. The thin gray line marks SPY's move, so bars past it beat or lagged the market.</span>
    {viz_movers(gainers + losers, reg['spy_1d'])}
  </div>
  <h3 style="margin-top:6px">Gainers</h3>
  <div class="movers">{mover_cards(gainers)}</div>
  <h3 style="margin-top:6px">Losers</h3>
  <div class="movers">{mover_cards(losers)}</div>
  <p class="note">Headlines are pulled when the report runs and may include later news. One headline can't fully explain a stock's move, so read the linked article before trading on it.</p>
</section>""")

    # --- plan & footer
    left = ctx["account"] * (1 - ctx["risk"]) ** 10
    out.append(f"""
<section data-view="playbook" class="view" hidden>
  <span class="eyebrow">Playbook for a ${ctx['account']:,.0f} account</span>
  <h2>How to size and run it</h2>
  <div class="plan">
    <div><h3>Risk ${risk_amt:,.0f} per trade</h3><p>That is {ctx['risk']:.0%} of the account. With a {STOP:.0%} stop, ${risk_amt:,.0f} ÷ {STOP:.0%} = <b>${pos:,.0f} position</b>. Ten straight losers would leave you with about ${left:,.0f}, not $0.</p></div>
    <div><h3>Two positions at most</h3><p>In a selloff stocks move together. Three semiconductor stocks are really one trade. Pick names from different groups.</p></div>
    <div><h3>Wait for a turn</h3><p>Don't buy the open on a drop alone. Wait for a day that closes above the prior day's high, and set your stop under that day's low.</p></div>
    <div><h3>Decide your exits first</h3><p>Place the stop and the +{TARGET:.0%} limit order when you enter. If neither triggers by day {HOLD}, sell. Keep a log so you can see your own hit rate after 20 trades.</p></div>
  </div>
  <p class="note">If you trade on margin and have under $25,000, US pattern-day-trader rules limit you to 3 same-day round trips in 5 business days. Holding overnight avoids that. In a cash account, sale proceeds settle the next business day (T+1).</p>
</section>
<section data-view="glossary" class="view" hidden>
<span class="eyebrow">Glossary · {len(GLOSSARY)} terms</span>
<h2>What the terms mean</h2>
<p>Plain-English definitions for every term in this report. The small <a class="q" href="#g-vix" aria-label="Example glossary link">?</a> links next to labels jump straight to the matching entry.</p>
<div class="gtools"><input id="gloss-filter" type="search" placeholder="Search terms, e.g. VIX, RSI, stop" aria-label="Search the glossary" autocomplete="off"><span id="gloss-count" class="note">{len(GLOSSARY)} terms</span></div>
{render_glossary()}
</section>
<section data-view="method" class="view" hidden>
<span class="eyebrow">Method</span>
<h2>How this report is built</h2>
<footer>
  <span>Method: daily split-adjusted prices from Yahoo Finance, {ctx['start']} to {ctx['asof']}, {ctx['n_univ']} tickers. If a stop and target fall on the same day, the stop is assumed to fill first. Gaps through the stop fill at the open. Generated by swing_report.py on {dt.date.today()}.</span>
  <span>Limits: the ticker list was chosen recently, so it favors stocks that survived and stayed popular (survivorship bias). Costs and slippage are not included. Past hit rates do not predict next week.</span>
  <span>Day movers: the largest one-day percentage moves in the universe on the last session. The "why" text is the summary of the highest-ranked Yahoo Finance article from the last {NEWS_LOOKBACK} days that names the company (articles describing a price move or a catalyst such as earnings, guidance, analyst calls or probes rank higher; boilerplate ranks lower), with an earnings note if the company reported in the last 3 days.</span>
  <span>This is educational analysis, not personalized financial advice.</span>
</footer>
</section>
</main>
</div>
<script>{JS}</script>
</body>
</html>""")
    return "\n".join(out)


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="swing_report.html", help="output HTML path")
    ap.add_argument("--period", default="5y", help="price history to backtest (yfinance period)")
    ap.add_argument("--account", type=float, default=1000, help="account size in $")
    ap.add_argument("--risk", type=float, default=0.02, help="fraction of account risked per trade")
    a = ap.parse_args()

    print("Downloading prices…")
    data = load_prices(a.period)
    spy = data["SPY"]["Close"]
    asof = spy.index[-1].date()
    ind = {t: indicators(data[t], spy) for t in UNIVERSE if t in data}
    below50 = spy < spy.rolling(50).mean()

    print("Backtesting setups…")
    results, per_ticker = backtest(data, ind, below50)

    print("Studying past selloffs…")
    study = selloff_study()

    vix, qqq = data["^VIX"]["Close"], data["QQQ"]["Close"]
    n_hist = 126  # about 6 months of sessions for the charts
    vix_on_spy = vix.reindex(spy.index).ffill()
    hist = dict(dates=[d.date() for d in spy.index[-n_hist:]],
                spy=[float(v) for v in spy.iloc[-n_hist:]],
                sma50=[float(v) for v in spy.rolling(50).mean().iloc[-n_hist:]],
                sma200=[float(v) for v in spy.rolling(200).mean().iloc[-n_hist:]],
                vix=[float(v) for v in vix_on_spy.iloc[-n_hist:]])
    regime = dict(spy=float(spy.iloc[-1]), sma50=float(spy.rolling(50).mean().iloc[-1]),
                  sma200=float(spy.rolling(200).mean().iloc[-1]), below50=bool(below50.iloc[-1]),
                  spy_1d=float(spy.iloc[-1] / spy.iloc[-2] - 1), spy_5d=float(spy.iloc[-1] / spy.iloc[-6] - 1),
                  qqq_5d=float(qqq.iloc[-1] / qqq.iloc[-6] - 1),
                  vix=float(vix.iloc[-1]), vix_5d_ago=float(vix.iloc[-6]))

    print("Screening latest bar…")
    rows = screen(data, ind, per_ticker)
    groups = build_groups(rows, results, regime["below50"])
    suggested = [r["t"] for g in groups for r in g["rows"]]

    gainers, losers = day_movers(data, ind, asof)
    mover_t = [m["t"] for m in gainers + losers]
    need = list(dict.fromkeys(suggested + mover_t))
    print(f"Fetching earnings dates for {len(need)} stocks…")
    earn = earnings_dates(need, asof)

    print(f"Fetching headlines for {len(mover_t)} day movers…")
    for m in gainers + losers:
        m["news"] = fetch_news(m["t"], asof)
        m["why"], m["chips"] = explain_move(m, regime["spy_1d"], earn.get(m["t"]), asof)

    window, skipped, halfs = next_sessions(asof, HOLD)
    soon_end = next_sessions(window[-1], SOON_DAYS)[0][-1]
    ret5 = sorted(((t, float(x["ret5"].iloc[-1])) for t, x in ind.items()), key=lambda z: z[1])
    movers = ret5[:3] if regime["spy_5d"] < 0 else ret5[-3:][::-1]

    ctx = dict(results=results, regime=regime, study=study, groups=groups, earnings=earn, asof=asof,
               gainers=gainers, losers=losers, hist=hist,
               window=window, skipped=skipped, halfs=halfs, soon_end=soon_end, movers=movers,
               n_univ=len(ind), n_drop15=sum(r < -0.15 for _, r in ret5), period=a.period.replace("y", " years"),
               start=spy.index[0].date(), account=a.account, risk=a.risk)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(render(ctx))

    print(f"\nHold window: {window[0]} → {window[-1]}" + (f" (closed: {', '.join(f'{d} {n}' for d, n in skipped)})" if skipped else ""))
    for g in groups:
        print(f"\n[{g['tag']}]")
        for r in g["rows"]:
            n = earn[r["t"]]["next"]
            flag = " <-- EARNINGS IN HOLD WINDOW" if n and window[0] <= n <= window[-1] else \
                   " <-- earnings soon after" if n and window[-1] < n <= soon_end else ""
            print(f"  {r['t']:6s} 5d {r['ret5']:+6.1%}  up10 {r['up10']:4.0%}  dn10 {r['dn10']:4.0%}  next earnings {n}{flag}")
    print("\n[Day movers]")
    for m in gainers + losers:
        print(f"  {m['t']:6s} {m['r1']:+6.1%}  {m['why'][:110]}")
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
