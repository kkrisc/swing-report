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


CSS = """
*,*::before,*::after{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0}
[hidden]{display:none!important}
img{max-width:100%}
/* Layout: single reading column, a trading-desk "tape" strip up top, then scoreboard, watchlist, plan */
:root{
  --bg:#f5f6f8; --surface:#ffffff; --fg:#17202b; --muted:#5b6675; --line:#dde1e7;
  --accent:#2747b8; --up:#1f7a4d; --down:#b4372f; --flat:#a3acb9; --warn-bg:#fff4e0; --warn-fg:#7a4a00;
  --display:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --body:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  color-scheme:light;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0f141b; --surface:#171e27; --fg:#e6ebf1; --muted:#97a3b3; --line:#2a3440;
  --accent:#8aa6ff; --up:#4cc38a; --down:#f07a70; --flat:#566273; --warn-bg:#3a2a0e; --warn-fg:#ffd28a; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0f141b; --surface:#171e27; --fg:#e6ebf1; --muted:#97a3b3; --line:#2a3440;
  --accent:#8aa6ff; --up:#4cc38a; --down:#f07a70; --flat:#566273; --warn-bg:#3a2a0e; --warn-fg:#ffd28a; color-scheme:dark}
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
td.name{white-space:normal;min-width:200px}
tr.base td{background:color-mix(in srgb,var(--accent) 7%,transparent)}
.bar{display:flex;height:10px;width:180px;border-radius:3px;overflow:hidden}
.bar i{display:block;height:100%}
.bar .t{background:var(--up)} .bar .x{background:var(--flat)} .bar .s{background:var(--down)}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:.82rem;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{width:10px;height:10px;border-radius:2px;display:inline-block}
.note{font-size:.88rem;color:var(--muted)}
.groups{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:16px}
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
@media (max-width:760px){
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
  var views=[].slice.call(document.querySelectorAll('.view')),links=[].slice.call(document.querySelectorAll('.menu a'));
  function show(scroll){
    var id=location.hash.slice(1), ok=views.some(function(v){return v.id===id});
    if(!ok) id='overview';
    views.forEach(function(v){v.hidden=v.id!==id});
    links.forEach(function(a){ if(a.getAttribute('href')==='#'+id) a.setAttribute('aria-current','page'); else a.removeAttribute('aria-current'); });
    if(scroll) window.scrollTo(0,0);
  }
  window.addEventListener('hashchange',function(){show(true)});
  show(false);
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
            ("setups", "Setup scoreboard", ""), ("playbook", "Playbook", ""), ("method", "Method", "")]
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
<section id="overview" class="view">
<header>
  <span class="eyebrow">Swing trading backtest · data through {ctx['asof'].strftime('%a %-d %b %Y')}</span>
  <h1>What are the odds of +{TARGET:.0%} in a week?</h1>
  <p class="lede">I tested {ctx['n_univ']} liquid US stocks over {ctx['period']} of daily data. Every trade enters at the next open, takes profit at +{TARGET:.0%}, stops out at −{STOP:.0%}, and exits after {HOLD} trading days if neither is hit. That is {base['trades']:,} simulated trades.</p>
</header>
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
<section id="market" class="view" hidden>
  <span class="eyebrow">Market right now</span>
  <h2>{mkt_head}</h2>
  <div class="tape">
    <div><span class="k">SPY close</span><span class="v">{reg['spy']:.2f}</span><span class="s {cls(reg['spy_1d'])}">{pct(reg['spy_1d'])} on the day</span></div>
    <div><span class="k">SPY 5-day</span><span class="v {cls(reg['spy_5d'])}">{pct(reg['spy_5d'])}</span><span class="s">QQQ {pct(reg['qqq_5d'])}</span></div>
    <div><span class="k">VIX</span><span class="v {'down' if reg['vix'] > 25 else ''}">{reg['vix']:.1f}</span><span class="s">{reg['vix_5d_ago']:.1f} a week ago</span></div>
    <div><span class="k">vs 50-day avg</span><span class="v {cls(reg['spy'] / reg['sma50'] - 1)}">{pct(reg['spy'] / reg['sma50'] - 1)}</span><span class="s">50d at {reg['sma50']:.2f}</span></div>
    <div><span class="k">vs 200-day avg</span><span class="v {cls(reg['spy'] / reg['sma200'] - 1)}">{pct(reg['spy'] / reg['sma200'] - 1)}</span><span class="s">200d at {reg['sma200']:.2f}</span></div>
  </div>
  <p>{ctx['n_drop15']} of the {ctx['n_univ']} stocks are down 15% or more in five days. Biggest 5-day moves: {e(movers)}.</p>
  <h3 style="margin-top:8px">What SPY did the week after sharp selloffs</h3>
  <p>Since {st['since']} there have been {st['events']} separate times SPY fell more than 7% in five days with the VIX above 30. {'The current week qualifies.' if selloff else 'The current week does not qualify, so treat this as background.'}</p>
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
    rows = []
    for k, (label, note, _) in SETUPS.items():
        a, r = R[k]["all"], R[k][rk]
        rows.append(
            f'<tr{" class=base" if k == "random" else ""}><td class="name">{e(label)}'
            f'{f"<br><span class=note>{e(note)}</span>" if note else ""}</td>'
            f'<td class="n">{a["trades"]:,}</td>'
            f'<td><div class="bar"><i class="t" style="width:{a["target"]*100:.1f}%"></i><i class="x" style="width:{a["time"]*100:.1f}%"></i><i class="s" style="width:{a["stop"]*100:.1f}%"></i></div></td>'
            f'<td class="n">{a["target"]:.1%}</td><td class="n">{a["stop"]:.1%}</td>'
            f'<td class="n {cls(a["avg"])}">{pct(a["avg"], 2)}</td>'
            f'<td class="n {cls(r["avg"])}">{pct(r["avg"], 2)}</td></tr>')
    bd = R["bigdrop"]["all"]
    be = bd["target"] / (bd["target"] + bd["stop"])
    breakeven = STOP / (TARGET + STOP)
    out.append(f"""
<section id="setups" class="view" hidden>
  <span class="eyebrow">Setup scoreboard · +{TARGET:.0%} target / −{STOP:.0%} stop / {HOLD} days</span>
  <h2>How common swing setups performed</h2>
  <div class="legend"><span><i style="background:var(--up)"></i>Hit +{TARGET:.0%} target</span><span><i style="background:var(--flat)"></i>Neither, exited day {HOLD}</span><span><i style="background:var(--down)"></i>Hit −{STOP:.0%} stop</span></div>
  <div class="scroll"><table>
    <thead><tr><th>Setup</th><th class="n">Trades</th><th>Outcome mix</th><th class="n">Target</th><th class="n">Stop</th><th class="n">Avg / trade</th><th class="n">{'In down markets' if below else 'In up markets'}*</th></tr></thead>
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
                       f'<td class="n">{r["up10"]:.0%}</td><td class="n">{r["dn10"]:.0%}</td></tr>')
        gh.append(f"""<div class="group"><span class="tag">{e(g['tag'])}</span><h3>{e(g['title'])}</h3><p>{e(g['desc'])}</p>
<table class="mini"><thead><tr><th>Ticker</th><th>Earnings</th><th class="n">5d</th><th class="n">↑10%</th><th class="n">↓10%</th></tr></thead><tbody>{''.join(trs)}</tbody></table></div>""")

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
<section id="watchlist" class="view" hidden>
  <span class="eyebrow">Watchlist · flagged by the screen on {fmt_d(ctx['asof'])} close</span>
  <h2>Names in play for the next {HOLD} sessions</h2>
  <p>These names trigger a setup from the table above. The odds columns show how often each stock moved 10% up, or 10% down, within any 5-day window over the test period. They describe the stock's behavior. They are not forecasts. Hover an earnings date to see when the company last reported.</p>
  <div class="legend"><span><span class="ed now">date</span> reports during the hold</span><span><span class="ed soon">date</span> reports just after it</span></div>
  <div class="groups">{''.join(gh)}</div>
  <p class="note">Every date with its status is listed in <a href="#earnings" style="color:var(--accent)">Earnings</a>.</p>
</section>""")

    # --- earnings calendar for suggested stocks
    group_of = {r["t"]: g["tag"] for g in groups for r in g["rows"]}
    status_txt = {"now": "Reports during the hold", "soon": "Reports just after the hold", "": ""}
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
<section id="earnings" class="view" hidden>
  <span class="eyebrow">Earnings calendar · {len(suggested)} suggested stocks</span>
  <h2>When the suggested stocks report</h2>
  <div class="callout">{''.join(callout)}</div>
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
<section id="movers" class="view" hidden>
  <span class="eyebrow">Day movers · {ctx['asof']:%a %-d %b} session</span>
  <h2>Biggest moves of the day</h2>
  <p>The largest one-day gains and losses among the {ctx['n_univ']} stocks in the universe. SPY moved {pct(reg['spy_1d'])} on the day. Each explanation comes from the most relevant recent Yahoo Finance headline that names the company, plus earnings timing and volume.</p>
  <h3 style="margin-top:6px">Gainers</h3>
  <div class="movers">{mover_cards(gainers)}</div>
  <h3 style="margin-top:6px">Losers</h3>
  <div class="movers">{mover_cards(losers)}</div>
  <p class="note">Headlines are pulled when the report runs and may include later news. One headline can't fully explain a stock's move, so read the linked article before trading on it.</p>
</section>""")

    # --- plan & footer
    left = ctx["account"] * (1 - ctx["risk"]) ** 10
    out.append(f"""
<section id="playbook" class="view" hidden>
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
<section id="method" class="view" hidden>
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
               gainers=gainers, losers=losers,
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
