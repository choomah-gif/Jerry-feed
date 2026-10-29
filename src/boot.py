"""boot.py — starts the Jerry desk with two live fixes, no edits to the other files.
1) NHL: MoneyPuck spells ice time 'icetime' in some CSVs and 'iceTime' in others -> both names always exist.
2) MLB: adds an MLB model to the hourly run (negative binomial on runs scored/allowed + starter FIP + home edge),
   priced against Pinnacle. Unvalidated, so calls are capped at Lean.
"""
import json, sqlite3, sys, runpy, traceback
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests
from scipy.stats import nbinom

UTC = timezone.utc

# ---------- fix 1: ice time column name
_read_csv = pd.read_csv
def _patched_read_csv(*a, **k):
    df = _read_csv(*a, **k)
    if isinstance(df, pd.DataFrame):
        c = set(df.columns)
        if "icetime" in c and "iceTime" not in c: df["iceTime"] = df["icetime"]
        elif "iceTime" in c and "icetime" not in c: df["icetime"] = df["iceTime"]
    return df
pd.read_csv = _patched_read_csv

# ---------- fix 2: MLB model
API = "https://statsapi.mlb.com/api/v1"
SHARP = ("pinnacle", "lowvig", "betonlineag")
R_DISP = 4.0          # negative binomial dispersion for MLB runs
STARTER_SHARE = 0.55  # share of innings a starter covers
HOME_EDGE = 1.02

def _get(path, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=20); r.raise_for_status(); return r.json()

def _team_rates(season):
    js = _get("/standings", leagueId="103,104", season=season, standingsTypes="regularSeason")
    rows = {}
    for rec in js["records"]:
        for t in rec["teamRecords"]:
            gp = t.get("gamesPlayed") or (t["wins"] + t["losses"])
            rows[t["team"]["name"]] = (t["runsScored"] / gp, t["runsAllowed"] / gp)
    lg = float(np.mean([v[0] for v in rows.values()]))
    return {k: (v[0] / lg, v[1] / lg) for k, v in rows.items()}, lg

def _ip(s):
    s = str(s); w, _, f = s.partition("."); return int(w) + (int(f) if f else 0) / 3

def _fip(pid, season):
    js = _get(f"/people/{pid}/stats", stats="season", group="pitching", season=season)
    st = js["stats"][0]["splits"][0]["stat"]
    ip = _ip(st["inningsPitched"])
    raw = (13 * st["homeRuns"] + 3 * (st["baseOnBalls"] + st.get("hitByPitch", 0)) - 2 * st["strikeOuts"]) / max(ip, 1) + 3.10
    return raw, ip

def _schedule(day):
    js = _get("/schedule", sportId=1, date=day, hydrate="probablePitcher")
    return [g for d in js.get("dates", []) for g in d["games"]]

def _quotes(con, eid):
    """Find the desk's odds table by its columns and return latest sharp prices {(market,outcome): (line, decimal)}."""
    cands = []
    for (t,) in con.execute("select name from sqlite_master where type='table'"):
        cols = [c[1] for c in con.execute(f"pragma table_info('{t}')")]
        s = set(cols)
        if not ({"event_id", "market", "outcome", "book"} <= s): continue
        price = next((c for c in ("price", "odds", "decimal", "dec", "price_dec") if c in s), None)
        ts = next((c for c in ("ts", "t", "time", "fetched", "fetched_at", "pulled", "snap_ts", "created") if c in s), None)
        if not price: continue
        line = next((c for c in ("point", "line", "handicap") if c in s), None)
        rank = 0 if any(x in t.lower() for x in ("quote", "snap", "tape", "odds", "price")) else 1
        cands.append((rank, t, price, ts, line, "player" in s))
    if not cands:
        raise RuntimeError("no odds table found")
    cands.sort()
    _, t, price, ts, line, has_player = cands[0]
    for book in SHARP:
        q = f"select market, outcome, {line or 'null'} as line, {price} as price from {t} where event_id=? and book=?"
        if has_player: q += " and coalesce(player,'')=''"
        if ts: q += f" order by {ts} desc"
        out = {}
        for m, o, l, p in con.execute(q, (eid, book)):
            if (m, o) in out or p is None: continue
            p = float(p)
            if p > 20 or p < 0:  # american odds
                p = 1 + (p / 100 if p > 0 else 100 / -p)
            out[(m, o)] = (l, p)
        if out:
            return out, book
    return {}, None

def _pmf(mu):
    k = np.arange(0, 31); p = R_DISP / (R_DISP + mu)
    v = nbinom.pmf(k, R_DISP, p); return v / v.sum()

def _model(mu_h, mu_a):
    ph, pa = _pmf(mu_h), _pmf(mu_a)
    J = np.outer(ph, pa)  # J[h, a]
    diff = np.subtract.outer(np.arange(31), np.arange(31))
    tot = np.add.outer(np.arange(31), np.arange(31))
    return J, diff, tot

def _prob(J, mask_win, mask_push):
    w, push = J[mask_win].sum(), J[mask_push].sum()
    return w / max(1 - push, 1e-9)

POSTSEASON_RUN_FACTOR = 0.88   # postseason scoring runs roughly 10-12% below the regular season


def refresh_mlb(db_path):
    now = datetime.now(UTC)
    con = sqlite3.connect(db_path, timeout=60); con.row_factory = None
    con.execute("""create table if not exists model_probs(event_id text, market text, outcome text, player text, line real,
                   model_prob real, fair_prob real, edge real, call text, note text, ts text,
                   primary key(event_id, market, outcome, player))""")
    evs = []
    for eid, home, away, start in con.execute("select event_id, home, away, commence from events where sport='MLB'"):
        st = datetime.fromisoformat(start.replace("Z", "+00:00"))
        if now - timedelta(minutes=5) < st < now + timedelta(hours=36):
            evs.append((eid, home, away, st))
    if not evs:
        con.close(); return 0
    season = now.year
    rates, lg = _team_rates(season)
    games = []
    for d in {(e[3] - timedelta(hours=6)).strftime("%Y-%m-%d") for e in evs} | {e[3].strftime("%Y-%m-%d") for e in evs}:
        games += _schedule(d)
    n = 0
    for eid, home, away, st in evs:
        g = next((g for g in games if g["teams"]["home"]["team"]["name"] == home and g["teams"]["away"]["team"]["name"] == away
                  and abs(datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00")) - st) < timedelta(hours=6)), None)
        sp = {}
        for side in ("home", "away"):
            pp = (g or {}).get("teams", {}).get(side, {}).get("probablePitcher")
            if pp:
                fip, ip = _fip(pp["id"], season)
                rel = fip / (lg * 0.92)
                w = ip / (ip + 60)  # shrink small samples toward average
                sp[side] = (pp.get("fullName", "?"), fip, w * rel + (1 - w) * 1.0)
            else:
                sp[side] = ("TBD", None, 1.0)
        oh, dh = rates.get(home, (1, 1)); oa, da = rates.get(away, (1, 1))
        def_h = STARTER_SHARE * sp["home"][2] + (1 - STARTER_SHARE) * dh
        def_a = STARTER_SHARE * sp["away"][2] + (1 - STARTER_SHARE) * da
        mu_h = lg * oh * def_a * HOME_EDGE
        mu_a = lg * oa * def_h / HOME_EDGE
        # October: aces start, bullpens are shortened and deployed aggressively, and run
        # scoring drops. A transparent prior, not a fitted coefficient — logged in detail.
        post = st.month >= 10
        if post:
            mu_h *= POSTSEASON_RUN_FACTOR; mu_a *= POSTSEASON_RUN_FACTOR
        J, diff, tot = _model(mu_h, mu_a)
        tie = J[diff == 0].sum()
        p_home = J[diff > 0].sum() + tie * 0.52
        q, book = _quotes(con, eid)
        if not q:
            continue
        _jr = sys.modules.get("jerry_reason")
        try:
            if _jr:
                _tl = next((v[0] for (m, o), v in q.items() if m == "totals"), None)
                _sl = q.get(("spreads", home), (None, None))[0]
                _jr.record_numbers(con, event_id=eid, sport="MLB", home=home, away=away, commence=st.isoformat(),
                                   margin=mu_h - mu_a, total=mu_h + mu_a, p_home=p_home,
                                   mkt_margin=None, mkt_total=None if _tl is None else float(_tl),
                                   checks={"starters_known": sp["home"][0] != "TBD" and sp["away"][0] != "TBD"},
                                   detail=f"runs {away} {mu_a:.2f} / {home} {mu_h:.2f}; {'postseason x%.2f; ' % POSTSEASON_RUN_FACTOR if post else ''}"
                                          f"SP {sp['away'][0]} @ {sp['home'][0]}")
        except Exception:
            pass
        def _sp(side):
            name, fip, _ = sp[side]
            return f"{name} ({'n/a' if fip is None else f'{fip:.2f}'} FIP)"
        sp_txt = f"SP {_sp('away')} @ {_sp('home')}"
        base = f"model runs {away} {mu_a:.2f} / {home} {mu_h:.2f}; {sp_txt}; vs {book}; MLB model unvalidated, capped at Lean"
        rows = []
        _jr = sys.modules.get("jerry_reason")
        for market in ("h2h", "spreads", "totals"):
            outs = [(o, v) for (m, o), v in q.items() if m == market]
            if len(outs) != 2: continue
            # power de-vig (falls back to proportional if the module is missing)
            _pf = _jr.power_devig([v[1] for _, v in outs]) if _jr else None
            inv = sum(1 / v[1] for _, v in outs)
            calc = []
            for idx, (o, (line, price)) in enumerate(outs):
                fair = _pf[idx] if _pf else (1 / price) / inv
                if market == "h2h":
                    mp = p_home if o == home else 1 - p_home
                elif market == "spreads":
                    L = float(line)
                    d = diff if o == home else -diff
                    mp = _prob(J, d + L > 0, d + L == 0)
                else:
                    L = float(line)
                    mp = _prob(J, tot > L, tot == L) if o == "Over" else _prob(J, tot < L, tot == L)
                calc.append((o, line, mp, fair, mp - fair))
            best = max(calc, key=lambda r: r[4])
            for o, line, mp, fair, edge in calc:
                reasoning = None
                if o == best[0]:
                    # totals get a real error band (Poisson sd of the run total); sides stay at the old edge rule
                    if market == "totals" and _jr and line is not None:
                        _band = math.sqrt(max(mu_h + mu_a, 1.0))
                        _ratio = _jr.edge_ratio((mu_h + mu_a) - float(line), _band)
                        _tag, _stk, _ = _jr.size_from_ratio(_ratio, edge, cap_lean=True)
                        call = f"{_tag} {o}" if _tag != "Sit" else "Sit"
                        _comp = {"mu": round(mu_h + mu_a, 3), "sd": round(_band, 3), "model_number": round(mu_h + mu_a, 3),
                                 "market_number": float(line), "runs_home": round(mu_h, 3), "runs_away": round(mu_a, 3)}
                        reasoning = _jr.render_reasoning(market="totals", player="", selection=o, line=float(line),
                                                         model_prob=float(mp), fair_prob=float(fair), components=_comp,
                                                         opponent="", status="", n_games=None, sport="MLB")
                        reasoning += f" Starters: {sp_txt}. The MLB model is unvalidated with no graded history."
                    else:
                        call = f"Lean {o}" if edge >= 0.02 else "Sit"
                        if _jr:
                            reasoning = (f"Negative-binomial runs model: {away} {mu_a:.2f}, {home} {mu_h:.2f} from season run rates, "
                                         f"starter FIP ({sp_txt}) and home edge. Makes {o} {mp:.0%} versus {fair:.0%} at fair. "
                                         f"No backtest and no graded history — colour at most. "
                                         f"Falsifier: the other side scores first and leads after five innings.")
                else:
                    call = "see other row"
                note = base + (f"; total {mu_h+mu_a:.1f} vs {line}" if market == "totals" else "")
                flag = plain = None
                if _jr and call.startswith(("Lean", "Play")):
                    try:
                        call, reasoning, _sg = _jr.sharp_gate(con, eid, market, call, "", reasoning or "")
                        flag = _sg.get("label") if _sg else None
                        call, plain = _jr.finalize_call(con, eid, market, call, "", float(mp), float(fair),
                                                        None if line is None else float(line), o)
                    except Exception:
                        pass
                rows.append((eid, market, o, "", None if line is None else float(line), float(mp), float(fair), float(edge), call, note, now.isoformat(), reasoning, flag, plain))
        # name the columns: the desk now carries a 12th column (reasoning) and a positional insert would fail
        try:
            cols = {r[1] for r in con.execute("pragma table_info(model_probs)")}
            if "reasoning" not in cols:
                con.execute("alter table model_probs add column reasoning text")
            if "flag" not in cols:
                con.execute("alter table model_probs add column flag text")
            if "plain" not in cols:
                con.execute("alter table model_probs add column plain text")
        except sqlite3.Error:
            pass
        con.executemany("""insert or replace into model_probs(event_id,market,outcome,player,line,model_prob,fair_prob,edge,call,note,ts,reasoning,flag,plain)
                           values(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
        con.commit(); n += len(rows)
    con.close()
    return n

import os
import math

# ---------- modules the desk and the live models now import.
# Embedded here rather than shipped as separate files so the Dockerfile's COPY
# line never has to change: they are registered in sys.modules before jerry_live
# or jerry_desk are loaded, so "import jerry_reason" resolves to this code.
import types as _types

_REASON_SRC = r'''"""
jerry_reason.py — the decision and reasoning layer.

Everything here is about turning a raw model number into a sized, auditable pick.
Nothing in this file predicts anything; it decides what a prediction is worth and
records why, in fields rather than prose, so the reasoning can be graded later.

Four jobs:
  1. De-vig properly (power method) instead of proportionally.
  2. Size an edge against the model's own error band, not against zero.
  3. Measure CLV in probability space so line moves stop being discarded.
  4. Attach evidence class, correlation group, falsifier and steelman to every pick.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from collections import defaultdict

SQRT_2PI = math.sqrt(2 * math.pi)

# ---------------------------------------------------------------- de-vigging

def power_devig(prices, tol=1e-9, max_iter=80):
    """De-vig a set of decimal prices by the power method.

    Proportional de-vig (p_i / sum p) assumes the bookmaker's margin is spread
    evenly across outcomes. It is not: margin is concentrated on longshots. That
    bias makes every long price look like value, which is exactly how a 4.60
    anytime-TD quote gets scored +7.9% EV when it is roughly fair.

    The power method solves for k in sum(p_i ** k) == 1, which takes more margin
    out of the longshots than the favourites. Reduces to proportional when the
    book is balanced, so it is safe to use everywhere.
    """
    raw = [1.0 / p for p in prices if p and p > 1.0]
    if len(raw) < 2:
        return None
    if abs(sum(raw) - 1.0) < tol:
        return list(raw)
    lo, hi = 0.2, 3.0
    for _ in range(max_iter):
        k = (lo + hi) / 2
        s = sum(r ** k for r in raw)
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    out = [r ** k for r in raw]
    tot = sum(out)
    return [o / tot for o in out]


def fair_two_way_power(price_a, price_b):
    """Fair probability of the first outcome, power-de-vigged. Drop-in for fair_two_way."""
    f = power_devig([price_a, price_b])
    if not f:
        return None
    return f[0]


def strip_single_side_vig(price, half_vig=0.022):
    """Approximate fair probability when only one side of a market is visible."""
    if not price or price <= 1.0:
        return None
    return (1.0 / price) / (1.0 + half_vig)


# ------------------------------------------------------- CLV in probability space

# Probability change per one unit of line movement, used when the model's own sd
# is unavailable. Derived from typical NFL distributions.
LINE_SLOPE = {"totals": 0.016, "spreads": 0.028, "h2h": 0.0}

# Which way a unit of line movement pushes the probability of each side winning.
#   +1 : a HIGHER line makes this selection MORE likely (spread handicaps, Under)
#   -1 : a HIGHER line makes this selection LESS likely (Over, Yes)
def _line_sign(market, selection):
    sel = (selection or "").strip().lower()
    if market == "spreads":
        return +1
    if sel in ("under", "no"):
        return +1
    if sel in ("over", "yes"):
        return -1
    return 0


def slope_for(market, sd=None, mu=None, line=None):
    """d(probability)/d(line). Uses the model's own sd when we stored one."""
    if sd and sd > 0:
        # density of the projection at the line: the local exchange rate between
        # points of line and points of probability
        if mu is not None and line is not None:
            z = (float(line) - float(mu)) / sd
            return math.exp(-0.5 * z * z) / (sd * SQRT_2PI)
        return 1.0 / (sd * SQRT_2PI)
    if market.startswith("player_"):
        return 0.045
    return LINE_SLOPE.get(market, 0.02)


def clv_pp(bet_line, bet_price, close_line, close_price, market, selection,
           sd=None, mu=None, close_fair=None):
    """Closing line value in probability points, valid across line movement.

    The old rule only recorded CLV when close_line == bet_line, which threw away
    every bet whose number moved — i.e. precisely the bets carrying information.
    Here the closing price is translated back to the line actually bet, so a bet
    taken at -3.5 and closing at -4.5 scores the move it earned.

    Positive = the market moved toward you.
    """
    if not bet_price or bet_price <= 1.0:
        return None
    # The price you actually got is the price you pay: no vig to strip from your own ticket.
    # (Stripping 2.2% here inflated every CLV by ~1.0-1.2pp; fixed 2026-10-26.)
    p_bet = 1.0 / float(bet_price)
    p_close = close_fair if close_fair is not None else strip_single_side_vig(close_price)
    if p_bet is None or p_close is None:
        return None
    if bet_line is not None and close_line is not None:
        d = float(close_line) - float(bet_line)
        if abs(d) > 1e-9:
            sign = _line_sign(market, selection)
            if sign == 0:
                return None          # cannot translate: unknown market geometry
            p_close = p_close + sign * slope_for(market, sd, mu, bet_line) * d
            p_close = min(max(p_close, 0.001), 0.999)
    return round((p_close - p_bet) * 100, 2)


# ------------------------------------------------- sizing against the error band

# Fallback dispersion of the model's own prediction, in the units of the market,
# used until the ledger has enough graded rows to measure it directly.
DEFAULT_MODEL_MAE = {"totals": 9.1, "spreads": 9.5, "h2h": 0.0}


def measure_model_error(con, sport, market, min_n=25):
    """Mean absolute error of the model's own number vs the realised result.

    Returns None until there is enough history, so callers fall back to the
    defaults rather than trusting a two-row estimate.
    """
    try:
        rows = con.execute(
            """select b.line, b.home_score, b.away_score, b.comp_json
                 from bets b
                where b.paper=1 and b.sport=? and b.market=?
                  and b.result in ('W','L','P')
                  and b.home_score is not null and b.away_score is not null""",
            (sport, market)).fetchall()
    except sqlite3.Error:
        return None
    errs = []
    for r in rows:
        try:
            comp = json.loads(r["comp_json"] or "{}")
        except Exception:
            comp = {}
        proj = comp.get("model_number")
        if proj is None:
            continue
        if market == "totals":
            actual = float(r["home_score"]) + float(r["away_score"])
        elif market == "spreads":
            actual = float(r["home_score"]) - float(r["away_score"])
        else:
            continue
        errs.append(abs(float(proj) - actual))
    if len(errs) < min_n:
        return None
    return sum(errs) / len(errs)


def edge_ratio(disagreement, model_mae):
    """How large the model's disagreement is relative to how wrong the model usually is.

    A 2.9 point total disagreement against a 9.1 point MAE is 0.32 — noise. This
    single number is what should drive sizing; raw edge in probability points
    hides the model's own dispersion and makes noise look like signal.
    """
    if not model_mae or model_mae <= 0:
        return None
    return abs(float(disagreement)) / float(model_mae)


# Ratio thresholds. Deliberately strict: below 0.75 the model and the market are
# not meaningfully apart, whatever the probability edge says.
RATIO_SIT, RATIO_LEAN, RATIO_PLAY = 0.75, 1.00, 1.50


def size_from_ratio(ratio, prob_edge, cap_lean=False):
    """Returns (tag, stake_units, why). Stake is the honest output, tag is the label."""
    if ratio is None:
        # no error band available: fall back to probability edge, capped at Lean
        tag = "Lean" if abs(prob_edge) >= 0.03 else "Sit"
        return (tag, 0.5 if tag == "Lean" else 0.0,
                "no measured error band for this market — capped at Lean")
    if ratio < RATIO_SIT:
        return ("Sit", 0.0,
                f"disagreement is {ratio:.2f} of the model's own average error — "
                f"model and market are effectively tied")
    if ratio < RATIO_LEAN:
        return ("Lean", 0.25,
                f"disagreement is {ratio:.2f} error bands — real but small")
    if ratio < RATIO_PLAY or cap_lean:
        return ("Lean", 0.5,
                f"disagreement is {ratio:.2f} error bands")
    return ("Play", 1.0,
            f"disagreement is {ratio:.2f} error bands — past the point where the "
            f"model's typical error explains it")


# ---------------------------------------------------------- evidence classes

EVIDENCE = {
    "price":  "soft-book price vs sharp fair — no prediction required",
    "model":  "model disagreement with the market — requires the model to be right",
    "hybrid": "model and price edge agree independently",
}


def evidence_record(con, sport, market_family, kind):
    """Live track record for this evidence class, so a pick carries its own warning."""
    try:
        rows = con.execute(
            "select clv_pp, result, pnl, stake from bets where paper=1 and sport=?",
            (sport,)).fetchall()
    except sqlite3.Error:
        return None
    clv = [r["clv_pp"] for r in rows if r["clv_pp"] is not None]
    done = [r for r in rows if r["result"] in ("W", "L", "P")]
    if not clv and not done:
        return None
    beat = sum(1 for c in clv if c > 0) / len(clv) if clv else None
    units = sum((r["pnl"] or 0) for r in done)
    return {
        "n": len(rows),
        "beat_close_pct": round(beat, 3) if beat is not None else None,
        "clv_pp_avg": round(sum(clv) / len(clv), 2) if clv else None,
        "graded": len(done),
        "units": round(units, 2),
        "kind": kind,
        "what_it_means": EVIDENCE.get(kind, ""),
    }


# ---------------------------------------------------- correlation and exposure

def correlation_group(sport, event_id, market, selection, player=""):
    """Legs that win and lose together share a group key.

    A receiver's yardage Over and his game's Over are one bet wearing two hats.
    Flat per-leg staking treats them as independent and silently doubles exposure.
    """
    sel = (selection or "").strip().lower()
    direction = "up" if sel in ("over", "yes") else "down" if sel in ("under", "no") else sel
    if market.startswith("player_") or market == "totals":
        return f"{sport}:{event_id}:scoring:{direction}"
    return f"{sport}:{event_id}:{market}"


MAX_UNITS_PER_GAME = 0.5   # per-game exposure cap (2026-10-26; was 1.5)
CORRELATION_HAIRCUT = 0.6     # applied to every leg after the first in a group


def apply_correlation_caps(picks):
    """Scale stakes down so correlated legs cannot quietly become one oversized bet.

    Mutates and returns the picks, each gaining `stake_final` and `corr_note`.
    """
    by_group = defaultdict(list)
    for p in picks:
        by_group[p["corr_group"]].append(p)
    for group, legs in by_group.items():
        legs.sort(key=lambda x: -(x.get("ratio") or 0))
        for i, leg in enumerate(legs):
            if i == 0:
                leg["stake_final"] = leg["stake"]
                leg["corr_note"] = ""
            else:
                leg["stake_final"] = round(leg["stake"] * CORRELATION_HAIRCUT, 3)
                leg["corr_note"] = (f"correlated with {legs[0]['label']} — "
                                    f"stake cut {int((1-CORRELATION_HAIRCUT)*100)}%")
    by_game = defaultdict(list)
    for p in picks:
        by_game[(p["sport"], p["event_id"])].append(p)
    for (sport, eid), legs in by_game.items():
        total = sum(l.get("stake_final", l["stake"]) for l in legs)
        if total > MAX_UNITS_PER_GAME and total > 0:
            scale = MAX_UNITS_PER_GAME / total
            for l in legs:
                l["stake_final"] = round(l.get("stake_final", l["stake"]) * scale, 3)
                l["corr_note"] = (l.get("corr_note") or "") + \
                    f" | game exposure capped at {MAX_UNITS_PER_GAME}u"
    return picks


# ----------------------------------------------- what the model cannot see

BLIND_SPOTS = {
    "totals": ["starting-QB change", "weather and wind", "pace and pass-rate shifts",
               "red-zone personnel", "coaching tendency with a lead"],
    "spreads": ["starting-QB change", "offensive-line injuries", "motivation and rest",
                "weather"],
    "player_receptions": ["target redistribution after an injury", "snap share",
                          "route participation", "shadow coverage"],
    "player_reception_yds": ["target redistribution after an injury", "air-yards role",
                             "shadow coverage", "game script"],
    "player_rush_yds": ["game script and lead protection", "goal-line committee",
                        "offensive-line injuries"],
    "player_pass_yds": ["game script", "pass rate over expectation", "pressure rate"],
    "player_anytime_td": ["red-zone role", "goal-line committee", "depth-chart position"],
}


SPORT_BLIND_SPOTS = {
    ("NHL", "totals"): ["confirmed starting goalies", "back-to-back fatigue", "line changes and scratches"],
    ("NHL", "h2h"): ["confirmed starting goalies", "back-to-back fatigue", "line changes and scratches"],
    ("NHL", "spreads"): ["confirmed starting goalies", "empty-net tendencies", "back-to-back fatigue"],
    ("MLB", "totals"): ["bullpen availability", "lineups and platoon splits", "weather and park", "postseason run environment"],
    ("MLB", "spreads"): ["bullpen availability", "lineups", "postseason run environment"],
    ("MLB", "h2h"): ["bullpen availability", "lineups", "postseason run environment"],
    ("NBA", "totals"): ["rest and back-to-backs", "late scratches", "pace changes"],
    ("NBA", "spreads"): ["rest and back-to-backs", "late scratches", "minutes restrictions"],
    ("NBA", "h2h"): ["rest and back-to-backs", "late scratches", "minutes restrictions"],
}
EXTREME_TOTAL = {"NFL": (40, 52), "NBA": (212, 245), "NHL": (5.5, 6.5), "MLB": (6.5, 10.0)}


def steelman(market, model_number, market_number, selection, extra=None):
    """Why the market may be right. Written before the pick is taken, not after.

    The market number is set by people who can see everything in BLIND_SPOTS and
    the model cannot. Stating that explicitly is the only reliable guard against
    writing a confident story around a number the model produced by itself.
    """
    blind = BLIND_SPOTS.get(market, ["information the model does not ingest"])
    bits = [f"The market is at {market_number}; the model is at {model_number}."]
    bits.append("A sharp book setting that number can see " + ", ".join(blind[:3]) +
                " — none of which is an input to this model.")
    if market == "totals" and market_number is not None:
        try:
            mn = float(market_number)
            if mn >= 52:
                bits.append("This total is already near the top of the league's range, "
                            "so the model is claiming the market under-reacted on a "
                            "number it has already pushed to an extreme.")
            if mn <= 40:
                bits.append("This total is already near the bottom of the range, so the "
                            "model is claiming further room in a direction the market "
                            "has already taken a long way.")
        except (TypeError, ValueError):
            pass
    if extra:
        bits.append(extra)
    return " ".join(bits)


# --------------------------------------------------------------- falsifiers

def falsifier(market, selection, line, model_number=None, player=""):
    """One observable that, if seen, means this pick was wrong for the reason given.

    Recorded at pick time and graded afterwards. This is what separates 'right for
    the right reason' from 'right by luck', and it makes the reasoning layer itself
    measurable rather than decorative.
    """
    sel = (selection or "").strip().lower()
    if market == "totals":
        if sel == "over":
            return ("Either team leads by 14+ at halftime, or combined first-half "
                    "points are under 40% of the line.")
        return ("Combined first-half points exceed 55% of the line, or both teams "
                "score on their opening two drives.")
    if market == "spreads":
        return (f"The {selection} side trails by more than {abs(float(line or 0)) + 7:.0f} "
                f"at any point, or loses the turnover margin by 2+.")
    if market == "player_anytime_td":
        who = player or "the player"
        if sel == "yes":
            return f"{who} has fewer than two touches inside the 20 by halftime."
        return f"{who} scores."
    if market.startswith("player_"):
        who = player or "the player"
        if sel in ("over", "yes"):
            return (f"{who} is under 50% of his usual snap share, or has fewer than "
                    f"3 touches/targets by halftime.")
        return (f"{who} exceeds his usual snap share and clears the line before the "
                f"fourth quarter.")
    return "The stated mechanism does not appear in the box score."


def grade_falsifier(fired, result):
    """Classify the pick after the fact: was it right for the stated reason?"""
    if result not in ("W", "L", "P"):
        return None
    if result == "P":
        return "push"
    if result == "W":
        return "right_for_the_reason" if not fired else "right_by_luck"
    return "wrong_as_predicted" if fired else "wrong_for_another_reason"


# ----------------------------------------------------------- language control

# Fixed vocabulary per probability band. Intensifiers are banned outright: a 56%
# call and a 70% call must not be describable with the same words, and no pick
# should ever be called the best, biggest or cleanest anything.
BANNED_WORDS = ("best", "biggest", "cleanest", "huge", "massive", "lock", "smash",
                "hammer", "free money", "can't lose", "slam", "max")

def language_band(prob):
    if prob is None:
        return "unquantified"
    p = abs(float(prob))
    if p < 0.52:
        return "a coin flip"
    if p < 0.55:
        return "a slight lean"
    if p < 0.58:
        return "a modest edge"
    if p < 0.62:
        return "a clear edge"
    return "a large edge (verify before trusting)"


def check_language(text):
    """Returns the banned words found. Used to gate written output."""
    low = (text or "").lower()
    return [w for w in BANNED_WORDS if w in low]


# ------------------------------------------------------------ pick assembly

def build_pick(con, *, sport, event_id, market, selection, player="", line=None,
               price=None, book="", model_prob=None, fair_prob=None,
               model_number=None, market_number=None, components=None,
               kind="model", cap_lean=False, label=""):
    """Assemble one pick as structured fields. Prose is generated from this, never
    the other way round, so what is written can always be checked against what was
    computed."""
    components = dict(components or {})
    disagreement = None
    if model_number is not None and market_number is not None:
        disagreement = float(model_number) - float(market_number)

    mae = measure_model_error(con, sport, market) or DEFAULT_MODEL_MAE.get(market)
    ratio = edge_ratio(disagreement, mae) if disagreement is not None else None
    prob_edge = (model_prob - fair_prob) if (model_prob is not None and fair_prob is not None) else 0.0

    if kind == "price":
        # a price edge needs no model to be right, so it is not sized off the error band
        tag, stake, why = ("Play", 1.0, "soft-book price beats sharp fair — no prediction required")
    else:
        tag, stake, why = size_from_ratio(ratio, prob_edge, cap_lean=cap_lean)

    rec = {
        "sport": sport, "event_id": event_id, "market": market, "selection": selection,
        "player": player, "line": line, "price": price, "book": book,
        "label": label or f"{player or selection} {line if line is not None else ''}".strip(),
        "model_prob": model_prob, "fair_prob": fair_prob, "prob_edge": round(prob_edge, 4),
        "model_number": model_number, "market_number": market_number,
        "disagreement": None if disagreement is None else round(disagreement, 2),
        "model_mae": mae, "ratio": None if ratio is None else round(ratio, 2),
        "tag": tag, "stake": stake, "sizing_reason": why,
        "evidence_kind": kind,
        "evidence_record": evidence_record(con, sport, market, kind),
        "corr_group": correlation_group(sport, event_id, market, selection, player),
        "steelman": steelman(market, model_number, market_number, selection),
        "falsifier": falsifier(market, selection, line, model_number, player),
        "blind_spots": BLIND_SPOTS.get(market, []),
        "confidence_language": language_band(model_prob),
        "components": components,
    }
    return rec


def picks_to_json(picks):
    return json.dumps(picks, default=str, indent=2)

# ----------------------------------------------------------- reasoning text

_STAT_WORD = {
    "player_pass_yds": "passing yards", "player_rush_yds": "rushing yards",
    "player_reception_yds": "receiving yards", "player_receptions": "receptions",
    "player_anytime_td": "an anytime touchdown", "player_pass_tds": "passing touchdowns",
}


def render_reasoning(*, market, player, selection, line, model_prob, fair_prob,
                     components, opponent="", status="", n_games=None, sport="NFL"):
    """One paragraph, generated from the fields, explaining why the pick exists.

    This is the prose layer of the reasoning rules: it says what the model
    computed, how large the disagreement is against the model's own dispersion,
    what the market can see that the model cannot, and what would falsify it.
    It contains no causal story about why the model is right, and it never uses
    a superlative, because neither is evidence.
    """
    c = dict(components or {})
    sel = (selection or "").strip()
    mu, sd = c.get("mu", c.get("model_number")), c.get("sd")
    mkt = c.get("market_number", line)
    try:
        disagreement = float(mu) - float(mkt)
    except (TypeError, ValueError):
        disagreement = None
    band = sd if (sd and sd > 0) else DEFAULT_MODEL_MAE.get(market)
    ratio = edge_ratio(disagreement, band) if (disagreement is not None and band) else None
    pe = (model_prob - fair_prob) if (model_prob is not None and fair_prob is not None) else None
    who = player or sel
    games = n_games or c.get("games")
    opp_f = c.get("opp_factor")
    out = []

    # 1. what was computed
    if market.startswith("player_anytime_td"):
        touches = c.get("touches"); prior = c.get("td_prior")
        out.append(f"{who} projects {float(mu):.2f} touchdowns per game over his last {games} "
                   f"games on about {touches:.0f} touches a week" + (f", shrunk toward a role prior of {prior:.2f}" if prior is not None else "")
                   + f". That is {float(model_prob):.0%} to score versus a de-vigged market of {float(fair_prob):.0%}.")
    elif market.startswith("player_"):
        word = _STAT_WORD.get(market, market.replace("player_", "").replace("_", " "))
        out.append(f"{who} projects {float(mu):.1f} {word} against a line of {mkt} — a recency-weighted average of his "
                   f"last {games} games" + (f" scaled by a {opp_f:.2f} opponent factor" if opp_f and abs(opp_f - 1) > 0.015 else "")
                   + f". The model makes the {sel} {float(model_prob):.0%} after anchoring, versus {float(fair_prob):.0%} at fair.")
        vol, eff, dp, wp = c.get("volume") or {}, c.get("efficiency") or {}, c.get("decomp_proj"), c.get("whole_stat_proj")
        stat = c.get("stat", "")
        if dp is not None and wp is not None:
            if stat in ("receiving_yards", "receptions"):
                vtxt = f"{vol.get('targets', 0):.1f} targets"; etxt = (f"{eff.get('ypt', 0):.1f} yards per target" if stat == "receiving_yards"
                                                                        else f"a {eff.get('catch', 0):.0%} catch rate")
            elif stat == "rushing_yards":
                vtxt = f"{vol.get('carries', 0):.1f} carries"; etxt = f"{eff.get('ypc', 0):.1f} yards per carry"
            elif stat == "passing_yards":
                vtxt = f"{vol.get('attempts', 0):.1f} attempts"; etxt = f"{eff.get('ypa', 0):.1f} yards per attempt"
            else:
                vtxt = etxt = ""
            if vtxt:
                out.append(f"Decomposed: {vtxt} at {etxt} (shrunk toward the position average) gives {float(dp):.1f}; "
                           f"the whole-stat average gives {float(wp):.1f}; the projection is the blend.")
                if abs(float(dp) - float(wp)) > 0.25 * max(abs(float(wp)), 1.0):
                    out.append("The two halves disagree by a wide margin, which usually means recent volume and recent "
                               "efficiency are telling different stories — a reason to trust this projection less, not more.")
    elif market == "totals":
        src = ("expected goals" if "xg_home" in c else "run rates and starter FIP" if "runs_home" in c
               else "team ratings and pace" if "games_h" in c else "both teams' season EPA per play shrunk toward average, with rest")
        out.append(f"The model's total is {float(mu):.1f} against a market of {mkt}, from {src}. "
                   f"It makes the {sel} {float(model_prob):.0%} versus {float(fair_prob):.0%} at fair.")
        it, idet = c.get("injury_total_adj"), c.get("injury_detail")
        if it:
            out.append(f"Includes an injury adjustment of {float(it):+.1f} to the total ({idet}).")
    elif market == "spreads":
        out.append(f"The model's margin is {float(mu):+.1f} against a line of {float(mkt):+.1f}, from EPA per play and rest. "
                   f"It makes {who} covering {float(model_prob):.0%} versus {float(fair_prob):.0%} at fair.")
        ia, idet = c.get("injury_margin_adj"), c.get("injury_detail")
        if ia:
            out.append(f"That margin already includes an injury adjustment of {float(ia):+.1f} points ({idet}) — a transparent "
                       f"prior, not a fitted coefficient, applied because the regression itself has no injury input.")
    else:
        out.append(f"Model {float(model_prob):.0%} versus fair {float(fair_prob):.0%}.")

    # 2. how big that actually is
    if ratio is not None:
        if ratio < RATIO_SIT:
            out.append(f"The disagreement is {ratio:.2f} of the projection's own error band, which is inside normal noise — "
                       f"the model and the market are effectively tied, and this is a Sit regardless of the {pe*100:+.1f}-point probability edge."
                       if pe is not None else
                       f"The disagreement is {ratio:.2f} of the projection's own error band — inside normal noise, so a Sit.")
        elif ratio < RATIO_LEAN:
            out.append(f"The disagreement is {ratio:.2f} error bands: real but small, which caps this at a quarter-unit Lean.")
        elif ratio < RATIO_PLAY:
            out.append(f"The disagreement is {ratio:.2f} error bands — larger than the model's typical miss, a half-unit Lean.")
        else:
            out.append(f"The disagreement is {ratio:.2f} error bands, past the point where the model's usual error explains it.")

    # 3. what the market can see that the model cannot
    blind = SPORT_BLIND_SPOTS.get((sport, market)) or BLIND_SPOTS.get(market, [])
    if blind:
        out.append("The number on the other side was set by a book that can see " + ", ".join(blind[:3]) +
                   " — none of which this model ingests" + (f", and {who} is listed Questionable" if status == "Q" else "") + ".")
    if market == "totals" and mkt is not None and sport in EXTREME_TOTAL:
        lo, hi = EXTREME_TOTAL[sport]
        try:
            m = float(mkt)
            if m >= hi and sel == "Over":
                out.append("This total already sits near the top of the league's range, so the model is claiming the market under-reacted on a number it has already pushed to an extreme.")
            elif m <= lo and sel == "Under":
                out.append("This total already sits near the bottom of the range, so the model is claiming further room in a direction the market has already taken a long way.")
        except (TypeError, ValueError):
            pass

    # 4. what would prove it wrong
    out.append("Falsifier: " + falsifier(market, sel, line, mu, player))
    text = " ".join(out)
    bad = check_language(text)
    if bad:   # should never happen from a template, but the rule is enforced on output
        for w in bad:
            text = text.replace(w, "notable")
    return text


# ===================================================================== sharp side
#
# Jerry has no ticket or handle percentages — The Odds API doesn't sell them — so
# "public" and "sharp" are inferred from prices, which is where they show up anyway:
#
#   sharp side   the side Pinnacle / LowVig / BetOnline have moved TOWARD since open,
#                plus Steam/Sharp alerts (sharp book leading, several books following)
#   public side  the side retail books SHADE — charge more for than Pinnacle does.
#                Books shade the side the public loads up on; that shade is the tax.
#
# The house edge is real: vig plus shading. The only systematic way around it is to
# be where sharp books are moving and retail books are discounting — which is the
# same thing as beating the closing line. A pick on the public side with sharp money
# moving the other way is demoted to Sit, whatever the model says.

SHARP_BOOKS = ("pinnacle", "lowvig", "betonlineag", "betfair_ex_eu", "matchbook", "smarkets")
RETAIL_BOOKS = ("draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet", "fanatics", "betrivers", "hardrockbet", "ballybet")


def _imp(price):
    try:
        return 1.0 / float(price) if price and float(price) > 1.0 else None
    except (TypeError, ValueError):
        return None


def _support(market, outcome, line_from, line_to, p_from, p_to):
    """+1 if this outcome gained support (shorter price / tougher line), -1 if it lost it, 0 if flat."""
    if line_from is not None and line_to is not None and abs(float(line_from) - float(line_to)) > 1e-9:
        lf, lt = float(line_from), float(line_to)
        if outcome in ("Over", "Under") or market == "totals" or market.startswith("player_"):
            d = 1 if lt > lf else -1          # Over gained support when the number rose
            return d if outcome in ("Over", "Yes") else -d
        return 1 if lt < lf else -1           # spread: -3 -> -3.5 means more support
    a, b = _imp(p_from), _imp(p_to)
    if a is None or b is None or abs(b - a) < 0.004:
        return 0
    return 1 if b > a else -1


def sharp_signal(con, event_id, market, outcome, player=""):
    """Where the sharp money and the public money are on this outcome.

    Returns {verdict, score, pin_dir, steam_for, steam_against, shade_pp, text}.
    verdict: 'sharp' (with the sharp side), 'against' (sharp moved the other way),
             'public' (public side, sharp not with it), or 'neutral'.
    """
    player = player or ""
    out = {"verdict": "neutral", "score": 0, "pin_dir": 0, "steam_for": 0, "steam_against": 0,
           "shade_pp": None, "text": "", "is_prop": bool(player)}
    try:
        # 1. sharp-book move since open on this outcome
        for bk in SHARP_BOOKS:
            o = con.execute("select line, price from opens where event_id=? and book=? and market=? and outcome=? and player=?",
                            (event_id, bk, market, outcome, player)).fetchone()
            n = con.execute("select line, price from last_quote where event_id=? and book=? and market=? and outcome=? and player=?",
                            (event_id, bk, market, outcome, player)).fetchone()
            if o and n:
                out["pin_dir"] = _support(market, outcome, o[0], n[0], o[1], n[1])
                # props: Pinnacle's prop limits are low, so only a move of 1.5pp+ (or a line
                # change) counts. 2026-10-26: Jaren Jackson Jr. was SHARP on a sub-1pp move and
                # scored 6 in limited minutes.
                if player and out["pin_dir"] and (o[0] == n[0] or o[0] is None):
                    a_, b_ = _imp(o[1]), _imp(n[1])
                    if a_ is not None and b_ is not None and abs(b_ - a_) < 0.015:
                        out["pin_dir"] = 0
                out["pin_book"] = bk
                break
        # 2. steam / sharp alerts: for this outcome (+dir) and against it (other outcomes' +dir)
        # Time-decayed (half-life 6h) and de-duplicated: one market move logs an alert on BOTH
        # sides, and DAL@PHI had 255 raw "Steam" rows. Count each (book, 15-min bucket) once,
        # weight by recency, so an opener move days ago no longer outvotes the last 90 minutes.
        _now = datetime.now(timezone.utc)
        _seen = set()
        _for = _against = 0.0
        for o_name, d, kind, ts, book in con.execute(
                """select outcome, dir, kind, ts, book from alerts where event_id=? and market=? and player=?
                   and kind in ('Steam','Sharp') order by ts desc limit 400""", (event_id, market, player)):
            same = (o_name == outcome)
            supports = (d > 0) == same
            try:
                _t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                if _t.tzinfo is None:
                    _t = _t.replace(tzinfo=timezone.utc)
                age_h = max(0.0, (_now - _t).total_seconds() / 3600)
            except Exception:
                age_h = 24.0
            key = (book, supports, str(ts)[:15])          # book + direction + 10-min-ish bucket
            if key in _seen:
                continue
            _seen.add(key)
            w = (2 if kind == "Steam" else 1) * 0.5 ** (age_h / 6.0)
            if supports:
                _for += w
            else:
                _against += w
        out["steam_for"], out["steam_against"] = round(_for, 1), round(_against, 1)
        # 3. public shade: retail implied minus sharp implied, this side vs the other side
        rows = con.execute("""select book, outcome, line, price from last_quote where event_id=? and market=? and player=?""",
                           (event_id, market, player)).fetchall()
        pin = {r[1]: (r[2], r[3]) for r in rows if r[0] == out.get("pin_book", "pinnacle")}
        if outcome in pin and len(pin) == 2:
            other = [k for k in pin if k != outcome][0]
            def shade(side):
                L = pin[side][0]
                ret = [_imp(r[3]) for r in rows if r[0] in RETAIL_BOOKS and r[1] == side
                       and ((L is None and r[2] is None) or (L is not None and r[2] is not None and abs(float(r[2]) - float(L)) < 1e-9))]
                ret = [x for x in ret if x]
                return (sum(ret) / len(ret) - _imp(pin[side][1])) if ret and _imp(pin[side][1]) else None
            s1, s2 = shade(outcome), shade(other)
            if s1 is not None and s2 is not None:
                out["shade_pp"] = round((s1 - s2) * 100, 2)
    except Exception as e:                       # noqa: BLE001 - a signal failure must never block a refresh
        out["text"] = f"sharp signal unavailable ({str(e)[:60]})"
        return out

    # 4. DraftKings splits: handle running ahead of tickets = bigger bettors on this side;
    #    tickets running ahead of handle = lots of small bets, the public
    out["dk_handle"] = out["dk_bets"] = None
    try:
        import jerry_splits as _js
        spl = _js.splits_for(con, event_id, market, outcome) if not player else None
        if spl:
            out["dk_handle"], out["dk_bets"] = spl[0], spl[1]
    except Exception:
        pass
    gap = (out["dk_handle"] - out["dk_bets"]) if out["dk_handle"] is not None else None
    out["dk_gap"] = gap
    dk_sharp = gap is not None and gap >= 10
    dk_public = gap is not None and (gap <= -10 or (out["dk_bets"] >= 65 and gap <= 0))
    # reverse line movement: the public is on this side and the sharp book moved away from it anyway
    out["rlm"] = bool(out["dk_bets"] is not None and out["dk_bets"] >= 60 and out["pin_dir"] < 0)

    _net = out["steam_for"] - out["steam_against"]
    net_steam = 0 if abs(_net) < 1.0 else max(-2, min(2, _net))   # under one fresh alert of net weight = noise
    public = (out["shade_pp"] is not None and out["shade_pp"] >= 1.5) or dk_public
    out["score"] = out["pin_dir"] + (1 if net_steam > 0 else -1 if net_steam < 0 else 0) - (1 if public else 0)
    sharp_with = out["pin_dir"] > 0 or net_steam > 0 or dk_sharp
    sharp_against = out["pin_dir"] < 0 or net_steam < 0
    if sharp_against and not sharp_with:
        out["verdict"] = "against"
    elif public and not sharp_with:
        out["verdict"] = "public"
    elif sharp_with and not sharp_against:
        out["verdict"] = "sharp"
    bits = []
    if out["pin_dir"]:
        bits.append(f"{out.get('pin_book', 'pinnacle')} has moved {'toward' if out['pin_dir'] > 0 else 'away from'} it since open")
    if out["steam_for"] or out["steam_against"]:
        bits.append(f"sharp/steam alerts {out['steam_for']} for, {out['steam_against']} against")
    if out["shade_pp"] is not None:
        if out["shade_pp"] >= 1.5:
            bits.append(f"retail books shade it {out['shade_pp']:.1f}pp — this is the public side")
        elif out["shade_pp"] <= -1.5:
            bits.append(f"retail books discount it {abs(out['shade_pp']):.1f}pp — the public is on the other side")
    if out["dk_handle"] is not None:
        bits.append(f"DraftKings {out['dk_bets']:.0f}% of bets / {out['dk_handle']:.0f}% of money")
    if out["rlm"]:
        bits.append("reverse line movement — the public is on it and the sharp book moved away")
    out["text"] = "; ".join(bits) if bits else "no sharp or public signal yet"
    out["label"] = flag_label(out)
    return out


def flag_label(sig):
    """Short label for display. Always shown; never used to remove a pick."""
    if not sig:
        return "NO READ"
    tags = []
    if sig.get("rlm"):
        tags.append("RLM FADE")
    v = sig.get("verdict")
    if v == "sharp":
        tags.append("SHARP")
    elif v == "against":
        tags.append("AGAINST SHARP")
    elif v == "public":
        tags.append("PUBLIC")
    if v != "public" and ((sig.get("shade_pp") or 0) >= 1.5 or (sig.get("dk_gap") is not None and sig["dk_gap"] <= -10)):
        tags.append("PUBLIC")
    if (sig.get("dk_gap") or 0) >= 10 and "SHARP" not in tags:
        tags.append("BIG MONEY")
    # retail discounting this side = retail needs action here = the public is on the other side.
    # That IS the fade-the-public side; it used to fall through to NEUTRAL (2026-10-26).
    if (sig.get("shade_pp") is not None and sig["shade_pp"] <= -1.5) and v != "against":
        tags.append("FADE PUBLIC")
    if sig.get("is_prop") and any(t in tags for t in ("SHARP", "AGAINST SHARP", "RLM FADE")):
        tags.append("THIN MARKET")      # a prop money read is worth less than a game-line one
    return " · ".join(dict.fromkeys(tags)) or "NEUTRAL"


BET_BOOKS = ("draftkings", "fanduel", "fanatics")   # the books KRONOS picks must be placeable at
_MKT_WORD = {"player_pass_yds": "passing yards", "player_rush_yds": "rushing yards", "player_reception_yds": "receiving yards",
             "player_receptions": "receptions", "player_pass_tds": "passing TDs", "player_points": "points",
             "player_rebounds": "rebounds", "player_assists": "assists", "player_threes": "threes",
             "player_shots_on_goal": "shots on goal", "player_goals": "goals", "player_anytime_td": "anytime TD"}


def plain_pick(market, selection, player, line, tag, model_prob=None, fair_prob=None):
    """The pick in words a bettor can place: 'Lean: George Pickens Over 64.5 receiving yards'."""
    word = _MKT_WORD.get(market, market.replace("player_", "").replace("_", " "))
    pct = (f" — KRONOS {model_prob:.0%} vs market {fair_prob:.0%}"
           if model_prob is not None and fair_prob is not None else "")
    ln = "" if line is None else f" {float(line):+g}" if market == "spreads" else f" {float(line):g}"
    if market == "player_anytime_td":
        return f"{tag}: {player} to score a TD{pct}"
    if player:
        return f"{tag}: {player} {selection}{ln} {word}{pct}"
    if market == "totals":
        return f"{tag}: {selection}{ln} total points{pct}"
    if market == "spreads":
        return f"{tag}: {selection}{ln}{pct}"
    if market == "h2h":
        return f"{tag}: {selection} to win{pct}"
    return f"{tag}: {selection}{ln}{pct}"


def bettable(con, event_id, market, selection, player=""):
    """True if that exact side has a price at one of BET_BOOKS right now."""
    try:
        q = f"""select 1 from last_quote where event_id=? and market=? and outcome=? and player=?
                and book in ({','.join('?' * len(BET_BOOKS))}) limit 1"""
        return con.execute(q, (event_id, market, selection, player or "", *BET_BOOKS)).fetchone() is not None
    except Exception:
        return True          # never let a lookup failure hide a pick


def finalize_call(con, event_id, market, call, player="", model_prob=None, fair_prob=None, line=None, outcome=None):
    """Turn a model call into something you can actually bet, plus a plain-English line.

    - Anytime TD 'No' is never a pick: books like DraftKings and FanDuel don't offer it.
      It becomes 'AVOID Yes' — information (don't parlay him), not a bet.
    - Any other pick whose exact side has no price at DraftKings, FanDuel or Fanatics is
      demoted to 'Sit (not offered at your books)'.
    model_prob / fair_prob are for the row's `outcome`; they're flipped when the pick is the other side.
    Returns (call, plain).
    """
    if not call or not str(call).startswith(("Lean", "Play")):
        return call, None
    sel = _selection_from_call(call)
    tag = str(call).split()[0]
    mp, fp = model_prob, fair_prob
    if outcome is not None and sel is not None and sel != outcome and mp is not None and fp is not None:
        mp, fp = 1 - mp, 1 - fp
    if market == "player_anytime_td" and sel == "No":
        yes_m = None if model_prob is None else (model_prob if outcome == "Yes" else 1 - model_prob)
        yes_f = None if fair_prob is None else (fair_prob if outcome == "Yes" else 1 - fair_prob)
        pct = f" — market {yes_f:.0%}, KRONOS {yes_m:.0%}" if yes_m is not None and yes_f is not None else ""
        return "AVOID Yes", f"Avoid: {player} to score a TD{pct}. Overpriced to score — not a bet; keep him out of parlays."
    # a spread row is stored under one team's line; picking the other team flips its sign
    # (2026-10-27: a Montreal pick printed as -1.5 when the Canadiens side was +1.5)
    if market == "spreads" and line is not None and outcome is not None and sel is not None and sel != outcome:
        line = -float(line)
    if sel and not bettable(con, event_id, market, sel, player):
        return "Sit (not offered at your books)", plain_pick(market, sel, player, line, "Not offered", mp, fp)
    return call, plain_pick(market, sel, player, line, tag, mp, fp)


def _selection_from_call(call):
    """'Lean Dallas Cowboys' -> 'Dallas Cowboys'; 'Lean Over 3.5' -> 'Over'; 'Sit' -> None."""
    if not call or call.startswith("Sit") or call.startswith("see "):
        return None
    words = call.split()
    if words and words[0] in ("Lean", "Play"):
        words = words[1:]
    if not words:
        return None
    if words[0] in ("Over", "Under", "Yes", "No"):
        return words[0]
    try:
        float(words[-1]); words = words[:-1]
    except ValueError:
        pass
    return " ".join(words) or None


def sharp_gate(con, event_id, market, call, player="", reasoning=""):
    """Label every pick with where the sharp and public money is. Returns (call, reasoning, signal).

    This used to demote picks to Sit when they were on the public side or against
    sharp movement. It no longer removes anything: the call stands, and the flag
    (SHARP / PUBLIC / AGAINST SHARP / RLM FADE / BIG MONEY / NEUTRAL) travels with
    it, so every pick shows which side of the money it is on. The flags are logged
    on paper bets so their value is measured, not assumed.
    """
    sel = _selection_from_call(call)
    if not sel:
        return call, reasoning, None
    sig = sharp_signal(con, event_id, market, sel, player)
    lab = sig.get("label", "NEUTRAL")
    tags = set(lab.split(" · "))                 # exact tags: 'SHARP' must not match 'AGAINST SHARP'
    note = {"AGAINST SHARP": " Caution: sharp money has moved the other way.",
            "PUBLIC": " Caution: this is the public side.",
            "SHARP": " Sharp money agrees.",
            "BIG MONEY": " Bigger bettors are on this side at DraftKings."}
    extra = "".join(v for k, v in note.items() if k in tags)
    if "RLM FADE" in tags:
        extra += " Reverse line movement against this side — the classic fade signal."
    return call, (reasoning or "") + f" Market read [{lab}]: {sig['text']}.{extra}", sig


# ================================================================ Jerry's numbers

# Each model's own error band for margin and total, in that sport's units. Priors
# until measured by grade_spins().
SPORT_MAE = {("NFL", "margin"): 9.5, ("NFL", "total"): 9.1, ("NBA", "margin"): 10.0, ("NBA", "total"): 14.0,
             ("NHL", "margin"): 1.8, ("NHL", "total"): 1.9, ("MLB", "margin"): 3.2, ("MLB", "total"): 3.3}
EXPLOIT_RATIO = 0.75


def _ensure_numbers(con):
    con.execute("""create table if not exists jerry_numbers(event_id text primary key, sport text, home text, away text,
                   commence text, home_pts real, away_pts real, margin real, total real, p_home real,
                   mkt_margin real, mkt_total real, mkt_p_home real, checks text, detail text, ts text)""")
    con.execute("""create table if not exists final_scores(event_id text primary key, sport text, home_score real,
                   away_score real, ts text)""")


def record_numbers(con, *, event_id, sport, home, away, commence, margin, total, p_home=None,
                   mkt_margin=None, mkt_total=None, mkt_p_home=None, checks=None, detail=""):
    """Store Jerry's own projection for a game — every game, not only the ones with an edge."""
    _ensure_numbers(con)
    if margin is None or total is None:
        return
    hp, ap = (float(total) + float(margin)) / 2.0, (float(total) - float(margin)) / 2.0
    from datetime import datetime, timezone
    con.execute("""insert or replace into jerry_numbers values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (event_id, sport, home, away, str(commence), round(hp, 2), round(ap, 2), round(float(margin), 3),
                 round(float(total), 3), None if p_home is None else round(float(p_home), 4),
                 None if mkt_margin is None else float(mkt_margin), None if mkt_total is None else float(mkt_total),
                 None if mkt_p_home is None else round(float(mkt_p_home), 4),
                 json.dumps(checks or {}), detail[:400], datetime.now(timezone.utc).isoformat(timespec="seconds")))


def spin(con, row):
    """Jerry's spin for one game: his own score next to the market's, the gap, and
    whether the gap is an exploit candidate (large, blind spots clear, sharp side not against)."""
    sp = row["sport"]
    mm, mt = row["mkt_margin"], row["mkt_total"]
    out = {"sport": sp, "home": row["home"], "away": row["away"], "commence": row["commence"],
           "jerry": (row["home_pts"], row["away_pts"]), "market": None, "gap_margin": None, "gap_total": None,
           "r_margin": None, "r_total": None, "exploit": None, "why": ""}
    if mm is not None and mt is not None:
        out["market"] = ((mt + mm) / 2.0, (mt - mm) / 2.0)
    if mm is not None:
        out["gap_margin"] = row["margin"] - mm
        out["r_margin"] = abs(out["gap_margin"]) / SPORT_MAE[(sp, "margin")]
    if mt is not None:
        out["gap_total"] = row["total"] - mt
        out["r_total"] = abs(out["gap_total"]) / SPORT_MAE[(sp, "total")]
    checks = json.loads(row["checks"] or "{}")
    blocked = [k for k, ok in checks.items() if ok is False]
    cand = []
    if (out["r_margin"] or 0) >= EXPLOIT_RATIO:
        side = row["home"] if out["gap_margin"] > 0 else row["away"]
        cand.append(("spreads", side, out["r_margin"]))
    if (out["r_total"] or 0) >= EXPLOIT_RATIO:
        cand.append(("totals", "Over" if out["gap_total"] > 0 else "Under", out["r_total"]))
    if not cand:
        out["why"] = "no gap large enough to be a mispricing"
        return out
    mkt, side, r = max(cand, key=lambda c: c[2])
    if blocked:
        # A failed blind-spot check means the gap is most likely missing information,
        # not a mispricing: report it as a gap, never as an EXPLOIT (2026-10-26).
        out["why"] = (f"gap {side} {mkt} {r:.2f} error bands — NOT an exploit: blind-spot checks failed "
                      f"({', '.join(blocked)}), the gap is probably missing information")
        return out
    sig = sharp_signal(con, row["event_id"], mkt, side)
    lab = sig.get("label", "NEUTRAL")
    warn = ""
    out["exploit"] = {"market": mkt, "side": side, "ratio": round(r, 2), "flag": lab, "checks_clear": not blocked}
    out["why"] = f"EXPLOIT CANDIDATE [{lab}]: {side} {mkt}, {r:.2f} error bands; {sig['text']}{warn}"
    return out


def grade_spins(con):
    """Jerry's spin error vs the market's, per sport, on completed games."""
    _ensure_numbers(con)
    rows = con.execute("""select n.sport, n.margin, n.total, n.mkt_margin, n.mkt_total, f.home_score, f.away_score
                          from jerry_numbers n join final_scores f using(event_id)""").fetchall()
    agg = {}
    for sp, m, t, mm, mt, hs, as_ in rows:
        a = agg.setdefault(sp, {"n": 0, "jm": 0.0, "mm": 0.0, "jt": 0.0, "mt": 0.0, "nm": 0, "nt": 0})
        a["n"] += 1
        am, at = hs - as_, hs + as_
        if mm is not None:
            a["jm"] += abs(m - am); a["mm"] += abs(mm - am); a["nm"] += 1
        if mt is not None:
            a["jt"] += abs(t - at); a["mt"] += abs(mt - at); a["nt"] += 1
    out = {}
    for sp, a in agg.items():
        out[sp] = {"games": a["n"],
                   "margin_mae_jerry": round(a["jm"] / a["nm"], 2) if a["nm"] else None,
                   "margin_mae_market": round(a["mm"] / a["nm"], 2) if a["nm"] else None,
                   "total_mae_jerry": round(a["jt"] / a["nt"], 2) if a["nt"] else None,
                   "total_mae_market": round(a["mt"] / a["nt"], 2) if a["nt"] else None}
    return out
'''

_GRADE_SRC = r'''"""
jerry_grade.py — grading for everything the team score cannot settle.

The desk's settle() takes team scores, so every player prop fell through to None
and stayed pending forever: 216 of 290 paper rows ungraded. This module supplies
the missing half — a player result feed, a name resolver, and settlement rules.

Design rules:
  - A player who did not play is a VOID, never a loss.
  - A stat landing exactly on a whole-number line is a PUSH.
  - A name that cannot be resolved is QUARANTINED and reported, never silently
    skipped. Silent skipping is how the ledger got into this state.
"""
from __future__ import annotations

import math
import os
import re
import sqlite3

DATA_DIR = os.environ.get("JERRY_DATA_DIR", "/data")
CACHE = os.path.join(DATA_DIR, "cache")

# nflverse moved/retired some release assets; try each in turn rather than
# letting a single 404 silently stop the model (which is what happened on
# 2026-10-26 with schedules/games.csv).
# CSV first so pandas alone can read these — no pyarrow, no requirements change.
NFL_PLAYER_STATS_URLS = [
    "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{season}.csv.gz",
    "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv.gz",
    "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv.gz",
    "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{season}.parquet",
]
NFL_SCHEDULE_URLS = [
    "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv",
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
]

# market -> (column in the player-stats frame, settlement style)
NFL_PROP_COLUMN = {
    "player_pass_yds":        ("passing_yards",   "line"),
    "player_rush_yds":        ("rushing_yards",   "line"),
    "player_reception_yds":   ("receiving_yards", "line"),
    "player_receptions":      ("receptions",      "line"),
    "player_pass_tds":        ("passing_tds",     "line"),
    "player_anytime_td":      ("_anytime_td",     "yesno"),
}



_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _strip_suffix(n):
    """'paul reed jr' -> 'paul reed'. The odds feed and ESPN disagree on suffixes (2026-10-28:
    'Paul Reed Jr' vs 'Paul Reed' meant the role-change rule never found his history)."""
    parts = n.split()
    while len(parts) > 2 and parts[-1] in _SUFFIXES:
        parts = parts[:-1]
    return " ".join(parts)


def _norm(s):
    """Match the desk's own normaliser so the two agree on what a name is (suffixes dropped)."""
    return _strip_suffix(re.sub(r"[^a-z ]", "", str(s).lower().replace(".", "").replace("-", " ")).strip())


def _initial_last(s):
    """'KC Concepcion' -> 'k concepcion'. Books abbreviate; feeds usually do not."""
    parts = _norm(s).split()
    if len(parts) < 2:
        return _norm(s)
    return f"{parts[0][0]} {parts[-1]}"


def _read_any(url):
    """Read a nflverse asset whatever format it is published in today."""
    import pandas as pd
    if url.endswith(".parquet"):
        return pd.read_parquet(url)
    return pd.read_csv(url, low_memory=False)


def _fetch_first(urls, season, reader=None):
    last = None
    for u in urls:
        try:
            return (reader or _read_any)(u.format(season=season)), u
        except Exception as e:               # noqa: BLE001 - any failure means try the next
            last = e
            continue
    raise RuntimeError(f"no nflverse source reachable: {last}")


def load_nfl_player_week(season, week):
    """Weekly player box scores. Returns a DataFrame or raises."""
    import pandas as pd
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"player_week_{season}.csv.gz")
    df = None
    if os.path.exists(cache):
        try:
            df = pd.read_csv(cache, low_memory=False)
        except Exception:
            df = None
    if df is None or week not in set(df["week"].tolist() if "week" in df.columns else []):
        df, _src = _fetch_first(NFL_PLAYER_STATS_URLS, season)
        try:
            df.to_csv(cache, index=False, compression="gzip")
        except Exception:
            pass
    df = df[df["season"] == season] if "season" in df.columns else df
    return df[df["week"] == week] if "week" in df.columns else df


class NameResolver:
    """Resolves a sportsbook player string to a row in the results frame.

    Keeps an alias table in the desk database so a name only has to be resolved
    once, and quarantines anything it cannot match instead of dropping it.
    """

    def __init__(self, con):
        self.con = con
        con.execute("""create table if not exists player_alias(
            sport text, book_name text, feed_name text, resolved_ts text,
            primary key(sport, book_name))""")
        con.execute("""create table if not exists grade_quarantine(
            bet_id integer primary key, sport text, player text, market text,
            reason text, seen_ts text)""")

    def _alias(self, sport, book_name):
        r = self.con.execute("select feed_name from player_alias where sport=? and book_name=?",
                             (sport, _norm(book_name))).fetchone()
        return r[0] if r else None

    def remember(self, sport, book_name, feed_name, now_iso):
        self.con.execute("insert or replace into player_alias values(?,?,?,?)",
                         (sport, _norm(book_name), feed_name, now_iso))

    def resolve(self, sport, book_name, candidates, now_iso):
        """candidates: iterable of feed-side display names. Returns a name or None."""
        known = self._alias(sport, book_name)
        if known:
            return known
        want, want_il = _norm(book_name), _initial_last(book_name)
        by_norm, by_il = {}, {}
        for c in candidates:
            by_norm.setdefault(_norm(c), c)
            by_il.setdefault(_initial_last(c), c)
        hit = by_norm.get(want) or by_il.get(want_il) or by_il.get(want)
        if hit:
            self.remember(sport, book_name, hit, now_iso)
            return hit
        # last resort: unique surname match
        surname = want.split()[-1] if want.split() else ""
        matches = [c for c in candidates if _norm(c).split()[-1:] == [surname]]
        if len(matches) == 1:
            self.remember(sport, book_name, matches[0], now_iso)
            return matches[0]
        return None

    def quarantine(self, bet_id, sport, player, market, reason, now_iso):
        self.con.execute("insert or replace into grade_quarantine values(?,?,?,?,?,?)",
                         (bet_id, sport, player, market, reason, now_iso))

    def quarantined(self):
        return [dict(r) for r in self.con.execute(
            "select * from grade_quarantine order by seen_ts desc limit 200")]


def settle_prop(market, selection, line, value, played=True):
    """W / L / P / V for one player prop.

    played=False returns 'V' (void). A scratch is not a losing bet, and grading it
    as one poisons every downstream hit rate.
    """
    if not played:
        return "V"
    if value is None:
        return None
    sel = (selection or "").strip().lower()
    style = NFL_PROP_COLUMN.get(market, (None, "line"))[1]
    if style == "yesno":
        hit = float(value) > 0
        if sel in ("yes", "over"):
            return "W" if hit else "L"
        if sel in ("no", "under"):
            return "L" if hit else "W"
        return None
    if line is None:
        return None
    v, L = float(value), float(line)
    if abs(v - L) < 1e-9:
        return "P"                       # exact landing on a whole number
    if sel in ("over", "yes"):
        return "W" if v > L else "L"
    if sel in ("under", "no"):
        return "W" if v < L else "L"
    return None


def _anytime_td(row):
    """Books settle anytime-TD on offensive scores only: rushing + receiving.

    A passing TD does not count for the quarterback, a two-point conversion does
    not count, and return/defensive scores are excluded by most books.
    """
    tot = 0.0
    for c in ("rushing_tds", "receiving_tds"):
        v = row.get(c)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            tot += float(v)
    return tot


def _played(row):
    """Did the player actually take part? Used to void rather than lose."""
    for c in ("offense_snaps", "attempts", "carries", "targets", "receptions"):
        v = row.get(c)
        if v is not None and not (isinstance(v, float) and math.isnan(v)) and float(v) > 0:
            return True
    for c in ("passing_yards", "rushing_yards", "receiving_yards"):
        v = row.get(c)
        if v is not None and not (isinstance(v, float) and math.isnan(v)) and float(v) != 0:
            return True
    return False



def nfl_week_of(commence):
    """(season, week) for a kickoff timestamp. Season opens the Thursday after Labor Day."""
    from datetime import datetime, timedelta, timezone
    if isinstance(commence, str):
        try:
            dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
        except ValueError:
            return (None, None)
    else:
        dt = commence
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    y = dt.year if dt.month >= 3 else dt.year - 1
    sept = datetime(y, 9, 1, tzinfo=timezone.utc)
    labor = sept + timedelta(days=(7 - sept.weekday()) % 7)
    kickoff = labor + timedelta(days=3)
    return y, max(1, min(22, ((dt - kickoff).days // 7) + 1))


def repair_misgraded_props(con, now_iso):
    """One-time repair of the first grading pass, which had no week filter.

    Only touches rows this grader itself settled (it stamps ' | settled X vs Y'
    into the note), so nothing graded from team scores is disturbed.
    """
    cur = con.execute("""update bets set result='pending', pnl=null, graded_ts=null
                           where sport='NFL' and market like 'player_%'
                             and result in ('W','L','P','V')
                             and note like '%| settled %'""")
    n = cur.rowcount
    con.execute("""update bets set note=replace(note, ' | settled', ' | voidstamp')
                    where note like '%| settled %'""")
    con.commit()
    return n

def grade_nfl_props(con, season, week, now_iso, limit=500):
    """Grade every pending NFL prop for one week. Returns a report dict."""
    res = NameResolver(con)
    pend = [dict(r) for r in con.execute(
        """select * from bets where result='pending' and sport='NFL'
             and market like 'player_%' order by id limit ?""", (limit,))]
    # Only grade bets that belong to THIS week. Without this filter every pending
    # prop is settled against whichever week's box scores happen to be loaded,
    # which silently produces a full ledger of meaningless results.
    pend = [b for b in pend if nfl_week_of(b["commence"]) == (season, week)]
    if not pend:
        return {"graded": 0, "voided": 0, "quarantined": 0, "note": "nothing pending for this week"}

    try:
        df = load_nfl_player_week(season, week)
    except Exception as e:                     # noqa: BLE001
        return {"graded": 0, "error": f"player stats unavailable: {e}"}

    name_col = "player_display_name" if "player_display_name" in df.columns else "player_name"
    try:
        import jerry_feeds as _jf
        _jf.nfl_player_team_from_frame(con, df, now_iso)
    except Exception:
        pass
    rows = {r[name_col]: r for r in df.to_dict("records")}
    candidates = list(rows.keys())

    graded = voided = quar = 0
    for b in pend:
      try:
        col, _style = NFL_PROP_COLUMN.get(b["market"], (None, None))
        if not col:
            res.quarantine(b["id"], "NFL", b["player"], b["market"],
                           "market not in NFL_PROP_COLUMN", now_iso); quar += 1
            continue
        feed = res.resolve("NFL", b["player"], candidates, now_iso)
        if not feed:
            res.quarantine(b["id"], "NFL", b["player"], b["market"],
                           "player name did not resolve", now_iso); quar += 1
            continue
        row = rows.get(feed)
        if row is None:
            # the alias cache resolves names across weeks, but a player with no
            # stat line this week simply did not play: void, never a loss.
            r, value = "V", None
        else:
            value = _anytime_td(row) if col == "_anytime_td" else row.get(col)
            r = settle_prop(b["market"], b["selection"], b["line"], value, played=_played(row))
        if r is None:
            res.quarantine(b["id"], "NFL", b["player"], b["market"],
                           "stat missing from feed", now_iso); quar += 1
            continue
        pnl = {"W": b["stake"] * (b["price"] - 1), "L": -b["stake"], "P": 0.0, "V": 0.0}[r]
        con.execute("""update bets set result=?, pnl=?, graded_ts=?,
                       note=coalesce(note,'')||? where id=?""",
                    (r, round(pnl, 4), now_iso,
                     f" | settled {value} vs {b['line']}", b["id"]))
        graded += 1
        voided += (r == "V")
      except Exception as _e:   # noqa: BLE001 - never let one row stop the pass
        res.quarantine(b["id"], "NFL", b["player"], b["market"],
                       f"grading error: {str(_e)[:120]}", now_iso); quar += 1
    con.commit()
    return {"graded": graded, "voided": voided, "quarantined": quar,
            "week": week, "season": season}
'''

def _register(name, src):
    m = _types.ModuleType(name)
    m.__file__ = f"<{name} embedded in boot.py>"
    sys.modules[name] = m
    try:
        exec(compile(src, m.__file__, "exec"), m.__dict__)
    except Exception:
        del sys.modules[name]
        raise
    return m

_FEEDS_SRC = r'''"""
jerry_feeds.py — result feeds for NBA and NHL.

Three jobs from the same box scores:
  1. Grade NBA/NHL player props (points, rebounds, assists, threes; points, SOG, goals).
  2. Keep a player -> team roster cache, used to reject props The Odds API has
     attached to the wrong game (the "game label" problem).
  3. Store completed games (score, possessions) for the NBA model.

Sources are public, keyless endpoints:
  NBA  ESPN   site.api.espn.com  (scoreboard + summary)
  NHL  NHL    api-web.nhle.com   (score by date + gamecenter boxscore)

These were written against the documented response shapes but could not be
exercised live from the build environment. Every parse is defensive; anything
unexpected quarantines the bet rather than guessing.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import urllib.request
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
UA = {"User-Agent": "Mozilla/5.0 (Jerry desk; prop grading)"}

ESPN_NBA = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
NHL_API = "https://api-web.nhle.com/v1"

# market -> stat key in the normalised box line
NBA_PROP_STAT = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast", "player_threes": "fg3m",
                 "player_blocks": "blk"}
NHL_PROP_STAT = {"player_points": "points", "player_shots_on_goal": "sog", "player_goals": "goals"}

# NHL API uses abbreviations; The Odds API uses full names.
NHL_ABBR = {
    "ANA": "Anaheim Ducks", "BOS": "Boston Bruins", "BUF": "Buffalo Sabres", "CGY": "Calgary Flames",
    "CAR": "Carolina Hurricanes", "CHI": "Chicago Blackhawks", "COL": "Colorado Avalanche",
    "CBJ": "Columbus Blue Jackets", "DAL": "Dallas Stars", "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers", "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild", "MTL": "Montreal Canadiens",
    "NSH": "Nashville Predators", "NJD": "New Jersey Devils", "NYI": "New York Islanders", "NYR": "New York Rangers",
    "OTT": "Ottawa Senators", "PHI": "Philadelphia Flyers", "PIT": "Pittsburgh Penguins", "SJS": "San Jose Sharks",
    "SEA": "Seattle Kraken", "STL": "St Louis Blues", "TBL": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Mammoth", "VAN": "Vancouver Canucks", "VGK": "Vegas Golden Knights", "WSH": "Washington Capitals",
    "WPG": "Winnipeg Jets",
}


def _get_json(url, timeout=25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))



_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _strip_suffix(n):
    """'paul reed jr' -> 'paul reed'. The odds feed and ESPN disagree on suffixes (2026-10-28:
    'Paul Reed Jr' vs 'Paul Reed' meant the role-change rule never found his history)."""
    parts = n.split()
    while len(parts) > 2 and parts[-1] in _SUFFIXES:
        parts = parts[:-1]
    return " ".join(parts)


def _norm(s):
    return _strip_suffix(re.sub(r"[^a-z ]", "", " ".join(str(s).lower().replace(".", " ").replace("-", " ").split())).strip())


def _team_key(name):
    """Last word of a team name, lower — 'Los Angeles Lakers' -> 'lakers'. Enough to match feeds."""
    n = _norm(name).replace("st louis", "st. louis")
    return n.split()[-1] if n.split() else n


def _ensure(con):
    con.execute("""create table if not exists player_team(sport text, player_norm text, team text, seen_ts text,
                   primary key(sport, player_norm))""")
    con.execute("""create table if not exists game_results(sport text, game_date text, home text, away text,
                   home_score int, away_score int, poss real, source_id text, seen_ts text,
                   primary key(sport, source_id))""")
    con.execute("""create table if not exists box_lines(sport text, game_date text, source_id text, team text,
                   player text, player_norm text, stats text, primary key(sport, source_id, player_norm))""")


# ----------------------------------------------------------------- NBA (ESPN)

def espn_nba_games(day):
    """Completed NBA games for a date (YYYY-MM-DD). Returns [{id, home, away, hs, as}]."""
    d = day.replace("-", "")
    data = _get_json(f"{ESPN_NBA}/scoreboard?dates={d}")
    out = []
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        if not comp.get("status", {}).get("type", {}).get("completed"):
            continue
        teams = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        h, a = teams.get("home"), teams.get("away")
        if not h or not a:
            continue
        try:
            out.append({"id": str(ev["id"]), "home": h["team"]["displayName"], "away": a["team"]["displayName"],
                        "hs": int(float(h["score"])), "as": int(float(a["score"]))})
        except (KeyError, ValueError, TypeError):
            continue
    return out


def espn_nba_box(game_id):
    """Player lines for one game. Returns {team_name: [{player, pts, reb, ast, fg3m, min, played}]} and a possessions estimate."""
    data = _get_json(f"{ESPN_NBA}/summary?event={game_id}")
    box = data.get("boxscore", {})
    out = {}
    team_tot = {}
    for t in box.get("players", []):
        team = t.get("team", {}).get("displayName", "")
        lines = []
        for grp in t.get("statistics", []):
            labels = [str(x).upper() for x in grp.get("labels", [])]
            idx = {lab: i for i, lab in enumerate(labels)}
            for ath in grp.get("athletes", []):
                name = ath.get("athlete", {}).get("displayName", "")
                st = ath.get("stats", [])
                def g(lab):
                    i = idx.get(lab)
                    if i is None or i >= len(st):
                        return None
                    v = str(st[i])
                    if "-" in v and lab in ("FG", "3PT", "FT"):
                        return v
                    try:
                        return float(v)
                    except ValueError:
                        return None
                fg3 = g("3PT"); fg3m = None
                if isinstance(fg3, str) and "-" in fg3:
                    try:
                        fg3m = float(fg3.split("-")[0])
                    except ValueError:
                        fg3m = None
                mins = g("MIN")
                dnp = bool(ath.get("didNotPlay")) or (mins is None and not st)
                lines.append({"player": name, "pts": g("PTS"), "reb": g("REB"), "ast": g("AST"), "fg3m": fg3m, "blk": g("BLK"),
                              "min": mins, "played": (not dnp) and (mins or 0) > 0})
        out[team] = lines
    # possessions ~ FGA - OREB + TOV + 0.44*FTA, from team totals when ESPN provides them
    try:
        for t in box.get("teams", []):
            team = t.get("team", {}).get("displayName", "")
            stats = {s.get("name") or s.get("label"): s.get("displayValue") for s in t.get("statistics", [])}
            def num(k):
                v = stats.get(k)
                if v is None:
                    return None
                v = str(v)
                if "-" in v:
                    return float(v.split("-")[1])   # attempts from "made-attempted"
                return float(v)
            fga = num("fieldGoalsMade-fieldGoalsAttempted") or num("FG")
            fta = num("freeThrowsMade-freeThrowsAttempted") or num("FT")
            oreb = num("offensiveRebounds") or num("OREB")
            tov = num("turnovers") or num("TO")
            if None not in (fga, fta, oreb, tov):
                team_tot[team] = fga - oreb + tov + 0.44 * fta
    except Exception:
        team_tot = {}
    poss = (sum(team_tot.values()) / 2.0) if len(team_tot) == 2 else None
    return out, poss


# ----------------------------------------------------------------- NHL (NHL API)

def nhl_games(day):
    """Completed NHL games for a date. Returns [{id, home, away, hs, as}]."""
    data = _get_json(f"{NHL_API}/score/{day}")
    out = []
    for g in data.get("games", []):
        if str(g.get("gameState", "")).upper() not in ("OFF", "FINAL"):
            continue
        ht, at = g.get("homeTeam", {}), g.get("awayTeam", {})
        try:
            out.append({"id": str(g["id"]), "home": NHL_ABBR.get(ht.get("abbrev"), ht.get("abbrev", "")),
                        "away": NHL_ABBR.get(at.get("abbrev"), at.get("abbrev", "")),
                        "hs": int(ht.get("score", 0)), "as": int(at.get("score", 0))})
        except (KeyError, ValueError, TypeError):
            continue
    return out


def nhl_box(game_id):
    """Skater lines for one game. Returns {team_name: [{player, goals, assists, points, sog, played}]}."""
    data = _get_json(f"{NHL_API}/gamecenter/{game_id}/boxscore")
    out = {}
    pbs = data.get("playerByGameStats", {})
    for side in ("homeTeam", "awayTeam"):
        abbr = data.get(side, {}).get("abbrev", "")
        team = NHL_ABBR.get(abbr, abbr)
        lines = []
        for grp in ("forwards", "defense"):
            for p in pbs.get(side, {}).get(grp, []):
                nm = p.get("name", {})
                name = nm.get("default", "") if isinstance(nm, dict) else str(nm)
                toi = str(p.get("toi", "0:00"))
                try:
                    mins = int(toi.split(":")[0]) + int(toi.split(":")[1]) / 60.0
                except (ValueError, IndexError):
                    mins = 0.0
                lines.append({"player": name, "goals": p.get("goals"), "assists": p.get("assists"),
                              "points": p.get("points"), "sog": p.get("sog", p.get("shots")), "played": mins > 0})
        out[team] = lines
    return out


# ------------------------------------------------------------ ingest + grade


def box_complete(sport, box, g):
    """(ok, reason). Players' points (NBA) or goals (NHL) must reconcile with the final score.

    NHL allows the winner one goal short: a shootout winner is credited to the team, not a player.
    """
    if not box or len(box) < 2:
        return False, "fewer than two teams in box"
    key = "pts" if sport == "NBA" else "goals"
    finals = {_team_key(g["home"]): g["hs"], _team_key(g["away"]): g["as"]}
    for team, lines in box.items():
        tk = _team_key(team)
        if tk not in finals:
            return False, f"unexpected team {team}"
        tot = sum(float(l.get(key) or 0) for l in lines)
        diff = finals[tk] - tot
        allow = 1 if (sport == "NHL" and finals[tk] > min(finals.values())) else 0
        if diff < 0 or diff > allow:
            return False, f"{team}: players sum to {tot:.0f}, team scored {finals[tk]}"
    return True, ""

def ingest_day(con, sport, day, now_iso):
    """Pull every completed game for a date, store results, box lines and roster entries."""
    _ensure(con)
    games = espn_nba_games(day) if sport == "NBA" else nhl_games(day)
    n = 0
    for g in games:
        if con.execute("select 1 from game_results where sport=? and source_id=?", (sport, g["id"])).fetchone():
            continue
        try:
            if sport == "NBA":
                box, poss = espn_nba_box(g["id"])
            else:
                box, poss = nhl_box(g["id"]), None
        except Exception as e:                       # noqa: BLE001
            con.execute("insert or replace into meta(k,v) values(?,?)", (f"feed_error_{sport}", f"{now_iso} {g['id']} {str(e)[:120]}"))
            continue
        # Completeness check (2026-10-28): a box score whose players don't add up to the team's
        # score is missing players. Grading from it turns real results into fake voids, as the
        # manual grade did for Paul Reed (22 pts) and Tage Thompson. Hold the game; retry next cycle.
        ok_box, why_box = box_complete(sport, box, g)
        if not ok_box:
            con.execute("insert or replace into meta(k,v) values(?,?)",
                        (f"feed_incomplete_{sport}_{g['id']}", f"{now_iso} {g['away']} @ {g['home']}: {why_box}"))
            continue
        con.execute("insert or replace into game_results values(?,?,?,?,?,?,?,?,?)",
                    (sport, day, g["home"], g["away"], g["hs"], g["as"], poss, g["id"], now_iso))
        for team, lines in box.items():
            for ln in lines:
                pn = _norm(ln["player"])
                if not pn:
                    continue
                con.execute("insert or replace into box_lines values(?,?,?,?,?,?,?)",
                            (sport, day, g["id"], team, ln["player"], pn, json.dumps(ln)))
                con.execute("insert or replace into player_team values(?,?,?,?)", (sport, pn, team, now_iso))
        n += 1
    con.commit()
    return n


def _initial_last(s):
    parts = _norm(s).split()
    return f"{parts[0][0]} {parts[-1]}" if len(parts) >= 2 else _norm(s)


def grade_props(con, sport, now_iso, lookback_days=3, limit=400):
    """Settle pending NBA/NHL props from stored box lines. Void on DNP, push on exact line."""
    _ensure(con)
    import jerry_grade as jg
    res = jg.NameResolver(con)
    statmap = NBA_PROP_STAT if sport == "NBA" else NHL_PROP_STAT
    pend = [dict(r) for r in con.execute(
        """select * from bets where result='pending' and sport=? and market like 'player_%'
             and commence < ? order by id limit ?""", (sport, now_iso, limit))]
    graded = voided = quar = 0
    for b in pend:
        try:
            stat = statmap.get(b["market"])
            if not stat:
                res.quarantine(b["id"], sport, b["player"], b["market"], "market not gradeable", now_iso); quar += 1
                continue
            day = (b["commence"] or "")[:10]
            # the game's box lines: match on the two teams in the bet's game string, same date (+/-1 for UTC rollover)
            try:
                away, home = [x.strip() for x in (b["game"] or "").split("@")]
            except ValueError:
                res.quarantine(b["id"], sport, b["player"], b["market"], "bad game string", now_iso); quar += 1
                continue
            # Bets carry a UTC kickoff; the feeds date games by US local date. An 8pm ET start
            # is the next UTC day, so look one day either side and pin the exact matchup
            # (2026-10-27: every game from 8pm ET on was silently left ungraded).
            d0 = datetime.fromisoformat(day)
            lo, hi = (d0 - timedelta(days=1)).strftime("%Y-%m-%d"), (d0 + timedelta(days=1)).strftime("%Y-%m-%d")
            hk, ak = _team_key(home), _team_key(away)
            games = [g for g in con.execute("""select source_id, game_date, home, away from game_results
                                               where sport=? and game_date between ? and ?""", (sport, lo, hi))
                     if {_team_key(g["home"]), _team_key(g["away"])} == {hk, ak}]
            if not games:
                continue                                   # game not ingested yet; stays pending
            games.sort(key=lambda g: abs((datetime.fromisoformat(g["game_date"]) - d0).days))
            rows = con.execute("select * from box_lines where sport=? and source_id=?", (sport, games[0]["source_id"])).fetchall()
            if not rows:
                continue                                   # game not ingested yet; stays pending
            cands = {r["player"]: r for r in rows}
            feed = res.resolve(sport, b["player"], list(cands.keys()), now_iso)
            if not feed:
                res.quarantine(b["id"], sport, b["player"], b["market"], "player not in either team's box", now_iso); quar += 1
                continue
            ln = json.loads(cands[feed]["stats"])
            val = ln.get(stat)
            r = jg.settle_prop(b["market"] if sport == "NFL" else "player_line", b["selection"], b["line"], val,
                               played=bool(ln.get("played")))
            if r is None:
                res.quarantine(b["id"], sport, b["player"], b["market"], "stat missing in box", now_iso); quar += 1
                continue
            pnl = {"W": b["stake"] * (b["price"] - 1), "L": -b["stake"], "P": 0.0, "V": 0.0}[r]
            con.execute("""update bets set result=?, pnl=?, graded_ts=?, note=coalesce(note,'')||? where id=?""",
                        (r, round(pnl, 4), now_iso, f" | settled {val} vs {b['line']}", b["id"]))
            graded += 1; voided += (r == "V")
        except Exception as e:                           # noqa: BLE001
            res.quarantine(b["id"], sport, b["player"], b["market"], f"grading error: {str(e)[:100]}", now_iso); quar += 1
    con.commit()
    return {"graded": graded, "voided": voided, "quarantined": quar}


# ---------------------------------------------------- roster cross-check (#5)

def player_team(con, sport, player):
    _ensure(con)
    pn = _norm(player)
    r = con.execute("select team from player_team where sport=? and player_norm=?", (sport, pn)).fetchone()
    if r:
        return r[0]
    il = _initial_last(player)
    best = None
    for row in con.execute("select player_norm, team, seen_ts from player_team where sport=? order by seen_ts desc", (sport,)):
        if _norm(row[0]) == pn:          # keys stored before suffixes were dropped
            return row[1]
        if best is None and _initial_last(row[0]) == il:
            best = row[1]
    return best


NBA_MIN_MINUTES = 12.0     # below this a rotation player's counting-stat line is mostly noise


def active_status(con, sport, player):
    """Did this player play real minutes in his team's most recent game?

    Returns (status, minutes): 'active', 'low' (NBA under NBA_MIN_MINUTES), 'dnp'
    (on the team but absent from, or did-not-play in, the last box score), or
    'unknown' (no team on file — never blocks, only tags).

    Built after 2026-10-26, when three of ten card bets (Gui Santos, Ryan Rollins,
    Ben Jones) were on players who did not play. A soft-book edge on a bench player
    often exists because sharper books already know he is out.
    """
    _ensure(con)
    team = player_team(con, sport, player)
    if not team:
        return "unknown", None
    tk = _team_key(team)
    last = None
    for sid, gd, h, a in con.execute("""select source_id, game_date, home, away from game_results where sport=?
                                        order by game_date desc limit 80""", (sport,)):
        if tk in (_team_key(h), _team_key(a)):
            last = sid
            break
    if not last:
        return "unknown", None
    pn, il = _norm(player), _initial_last(player)
    row = None
    for r in con.execute("select player_norm, stats from box_lines where sport=? and source_id=?", (sport, last)):
        if r[0] == pn or _norm(r[0]) == pn or _initial_last(r[0]) == il:
            row = r
            break
    if not row:
        return "dnp", None
    try:
        st = json.loads(row[1])
    except Exception:
        return "unknown", None
    if not st.get("played"):
        return "dnp", 0.0
    mins = st.get("min")
    if sport == "NBA" and mins is not None and float(mins) < NBA_MIN_MINUTES:
        return "low", float(mins)
    return "active", (float(mins) if mins is not None else None)


def label_ok(con, sport, player, home, away):
    """True if the player's last known team is one of the two in the event, or unknown.

    Unknown returns True — never block a bet on missing data — but the caller
    should mark it as unverified. A known mismatch returns False.
    """
    t = player_team(con, sport, player)
    if not t:
        return None
    return _team_key(t) in (_team_key(home), _team_key(away))


def nfl_player_team_from_frame(con, df, now_iso):
    """Populate the roster cache for NFL from the nflverse weekly frame already loaded for grading."""
    _ensure(con)
    name_col = "player_display_name" if "player_display_name" in df.columns else "player_name"
    if "recent_team" in df.columns:
        tcol = "recent_team"
    elif "team" in df.columns:
        tcol = "team"
    else:
        return 0
    latest = df.sort_values("week").groupby(name_col).tail(1)
    n = 0
    for _, r in latest.iterrows():
        con.execute("insert or replace into player_team values(?,?,?,?)", ("NFL", _norm(r[name_col]), str(r[tcol]), now_iso))
        n += 1
    con.commit()
    return n



ESPN_NHL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"
OUT_STATUSES = ("out", "suspension", "suspended", "injured reserve", "ir")
DOUBT_STATUSES = ("doubtful", "questionable", "day-to-day", "game time decision", "gtd")


def ingest_injuries(con, sport, now_iso):
    """Today's injury report into table `injuries`. Shapes are parsed defensively; anything
    unexpected is recorded in meta 'injuries_error_<sport>' rather than guessed."""
    _ensure(con)
    con.execute("""create table if not exists injuries(sport text, player_norm text, player text, team text,
                   status text, detail text, ts text, primary key(sport, player_norm))""")
    base = ESPN_NBA if sport == "NBA" else ESPN_NHL
    data = _get_json(f"{base}/injuries")
    teams = data.get("injuries") or data.get("teams") or []
    n = 0
    con.execute("delete from injuries where sport=?", (sport,))
    for t in teams:
        tname = (t.get("team") or {}).get("displayName") or t.get("displayName") or ""
        for inj in t.get("injuries", []) or []:
            ath = inj.get("athlete") or {}
            name = ath.get("displayName") or ath.get("fullName") or ""
            status = str(inj.get("status") or (inj.get("type") or {}).get("description") or "").strip()
            if not name or not status:
                continue
            detail = str(inj.get("shortComment") or inj.get("longComment") or "")[:160]
            team = tname or ((ath.get("team") or {}).get("displayName") or "")
            con.execute("insert or replace into injuries values(?,?,?,?,?,?,?)",
                        (sport, _norm(name), name, team, status, detail, now_iso))
            n += 1
    con.commit()
    return n


def injury_status(con, sport, player):
    """'out', 'doubtful' (incl. questionable / day-to-day) or None."""
    try:
        r = con.execute("select status from injuries where sport=? and player_norm=?", (sport, _norm(player))).fetchone()
    except Exception:
        return None
    if not r:
        return None
    s = str(r[0]).lower()
    if any(x in s for x in OUT_STATUSES):
        return "out"
    if any(x in s for x in DOUBT_STATUSES):
        return "doubtful"
    return None


# ---------- C. minutes-and-role projection (NBA) / per-game rate (NHL) ----------
PROJ_STAT = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast", "player_threes": "fg3m", "player_blocks": "blk",
             "player_shots_on_goal": "sog", "player_goals": "goals"}


def _recent_lines(con, sport, player, n=6):
    pn, il = _norm(player), _initial_last(player)
    out = []
    for gd, st, pnorm in con.execute("""select gr.game_date, bl.stats, bl.player_norm from box_lines bl join game_results gr
                                         using(sport, source_id) where bl.sport=? order by gr.game_date desc""", (sport,)):
        if pnorm == pn or _norm(pnorm) == pn or _initial_last(pnorm) == il:
            try:
                out.append(json.loads(st))
            except Exception:
                continue
            if len(out) >= n:
                break
    return out


def prop_check(con, sport, player, market, selection, line):
    """Does the player's recent role support this side? Returns dict:
       proj, p_side, verdict ('agrees' | 'neutral' | 'disagrees' | None), games, minutes, role_change(bool), note.

    NBA: recency-weighted minutes x per-minute rate. NHL: recency-weighted per-game rate.
    role_change = the line sits far above what this player's recent role produces — the market is
    pricing news (a new starter, a teammate out) that recent box scores can't see. Built after
    Paul Reed: 12.5-point line on a ~4-point role, the Under looked like value, he started and scored 22.
    """
    stat = PROJ_STAT.get(market)
    res = {"proj": None, "p_side": None, "verdict": None, "games": 0, "minutes": None, "role_change": False, "note": ""}
    if not stat or line is None:
        return res
    lines = [l for l in _recent_lines(con, sport, player) if l.get("played")]
    res["games"] = len(lines)
    if len(lines) < 3:
        res["note"] = f"only {len(lines)} recent games on file"
        return res
    w = [0.8 ** i for i in range(len(lines))]
    vals = [float(l.get(stat) or 0) for l in lines]
    per_game = sum(v * wi for v, wi in zip(vals, w)) / sum(w)
    if sport == "NBA":
        mins = [float(l.get("min") or 0) for l in lines]
        m = sum(x * wi for x, wi in zip(mins, w)) / sum(w)
        tot_min = sum(mins)
        rate = sum(vals) / tot_min if tot_min > 0 else 0.0
        proj = m * rate
        res["minutes"] = round(m, 1)
    else:
        proj = per_game
    res["proj"] = round(proj, 2)
    L = float(line)
    if per_game > 0 and L >= max(1.6 * per_game, per_game + 4.0) and stat == "pts":
        res["role_change"] = True
    elif per_game > 0 and L >= max(1.6 * per_game, per_game + 2.0) and stat in ("reb", "ast", "sog", "blk"):
        res["role_change"] = True
    # count stats ~ Poisson-ish; points ~ normal with sd ~ 0.35 * mean
    import math as _m
    if stat == "pts":
        sd = max(0.35 * proj, 2.5)
        z = (L + 0.5 - proj) / sd if abs(L - round(L)) < 1e-9 else (L - proj) / sd
        p_over = 1 - 0.5 * (1 + _m.erf(z / _m.sqrt(2)))
    else:
        lam = max(proj, 0.05); k = int(_m.floor(L))
        cdf = sum(_m.exp(-lam) * lam ** i / _m.factorial(i) for i in range(k + 1))
        p_over = 1 - cdf
    sel = (selection or "").lower()
    p_side = p_over if sel in ("over", "yes") else 1 - p_over
    res["p_side"] = round(p_side, 3)
    res["verdict"] = "agrees" if p_side >= 0.53 else "disagrees" if p_side < 0.47 else "neutral"
    res["note"] = (f"recent role: {res['minutes']} min, " if res["minutes"] is not None else "recent role: ") + \
                  f"{per_game:.1f} {stat}/game -> proj {proj:.1f} vs line {L:g}"
    return res

SEASON_START = {"NBA": (10, 15), "NHL": (10, 1)}   # (month, day): safely before opening night


def run(con, now, sports=("NBA", "NHL"), days_back=2):
    """Ingest the last few days and grade. Returns a report dict.

    The first run of a season backfills every day since opening night (once, marked in
    meta), so the NBA model fits on the whole season instead of the last two days."""
    rep = {}
    for sp in sports:
        try:
            got = 0
            span = days_back
            yr = now.year if now.month >= 8 else now.year - 1
            key = f"feeds_backfill_{sp}_{yr}"
            done = con.execute("select v from meta where k=?", (key,)).fetchone()
            if not done:
                m, d = SEASON_START.get(sp, (10, 1))
                span = max(days_back, (now - datetime(yr, m, d, tzinfo=UTC)).days)
            for k in range(span, -1, -1):
                day = (now - timedelta(days=k)).strftime("%Y-%m-%d")
                got += ingest_day(con, sp, day, now.isoformat(timespec="seconds"))
            try:
                inj_n = ingest_injuries(con, sp, now.isoformat(timespec="seconds"))
            except Exception as _e:                      # noqa: BLE001
                inj_n = f"error {str(_e)[:60]}"
                con.execute("insert or replace into meta(k,v) values(?,?)", (f"injuries_error_{sp}", f"{now.isoformat(timespec='seconds')} {str(_e)[:140]}"))
            g = grade_props(con, sp, now.isoformat(timespec="seconds"))
            if not done:
                con.execute("insert or replace into meta(k,v) values(?,?)", (key, f"{now.isoformat(timespec='seconds')} {got} games over {span} days"))
                con.commit()
            rep[sp] = {"games_ingested": got, "backfill_days": span if not done else 0, "injuries": inj_n, **g}
        except Exception as e:                           # noqa: BLE001
            rep[sp] = {"error": str(e)[:160]}
            try:
                con.execute("insert or replace into meta(k,v) values(?,?)", (f"feed_error_{sp}", f"{now.isoformat(timespec='seconds')} {str(e)[:150]}"))
                con.commit()
            except sqlite3.Error:
                pass
    return rep
'''

_SIGNALS_SRC = r'''"""
jerry_signals.py — the learning layer for KRONOS (2026-10-29).

KRONOS's reasoning used to be rules Claude wrote. This module makes it measure itself:

  capture()        every signal's value on every edge and paper bet (stored as JSON on the bet)
  fit_weights()    per-signal weights learned from graded results, starting at ZERO; a signal
                   only moves a price once it has 20+ graded spots and shows an effect
  adjust()         market fair price -> shifted by measured signal weights (capped)
  reports          signal scorecard, calibration, loss autopsy, timing, contradictions
  sim_prop()       NBA game-script simulation (blowout / overtime / pace) for props
  data feeds       dropping odds (own tape), DK public-betting-trends articles, nflverse snap
                   counts + injuries, NFL positional matchups, Open-Meteo weather, ESPN soccer grading

Nothing here changes a pick on day one: unproven signals have weight 0 by construction.
"""
from __future__ import annotations

import csv
import html as _html
import io
import json
import math
import random
import re
import urllib.request
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
UA = {"User-Agent": "Mozilla/5.0 (KRONOS desk)"}
GRADUATE_N = 20          # owner's rule: 20 spots before a feature can make a play
MAX_SHIFT_LOGIT = 0.25   # total adjustment cap (~6pp at a 50% price)
L2 = 20.0                # strong shrinkage: weights must be earned


def _get(url, timeout=25, text=False):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        b = r.read().decode("utf-8", "replace")
    return b if text else json.loads(b)


def _norm(s):
    s = re.sub(r"[^a-z ]", "", " ".join(str(s).lower().replace(".", " ").replace("-", " ").split())).strip()
    p = s.split()
    while len(p) > 2 and p[-1] in ("jr", "sr", "ii", "iii", "iv", "v"):
        p = p[:-1]
    return " ".join(p)


def _imp(dec):
    try:
        d = float(dec)
        return 1.0 / d if d > 1 else None
    except (TypeError, ValueError):
        return None


def _american(dec):
    d = float(dec)
    return (d - 1) * 100 if d >= 2 else -100 / (d - 1)


def _cents(dec_a, dec_b):
    """Line-movement 'cents' the way bettors count them: +100 and -100 are the same price, so
    +110 -> -105 is 15 cents, not 215."""
    x, y = _american(dec_a), _american(dec_b)
    if (x >= 100) == (y >= 100):
        return abs(x - y)
    return abs(abs(x) - 100) + abs(abs(y) - 100)


def _ensure(con):
    con.executescript("""
    create table if not exists nfl_snaps(season int, week int, team text, player_norm text, player text,
        position text, off_pct real, primary key(season, week, team, player_norm));
    create table if not exists nfl_inj(season int, week int, team text, player_norm text, player text,
        status text, primary key(season, week, team, player_norm));
    create table if not exists nfl_pos_def(season int, team text, position text, stat text, allowed_pg real,
        league_pg real, n int, primary key(season, team, position, stat));
    create table if not exists weather(event_id text primary key, ts text, wind_mph real, precip_mm real,
        temp_c real, dome int);
    create table if not exists soccer_box(source_id text, game_date text, team text, player_norm text,
        player text, goals real, assists real, primary key(source_id, team, player_norm));
    create table if not exists soccer_results(source_id text primary key, league text, game_date text,
        home text, away text, hs int, as_ int, ts text);
    """)
    cols = {r[1] for r in con.execute("pragma table_info(bets)")}
    if "signals" not in cols:
        con.execute("alter table bets add column signals text")


def _meta(con, k, v):
    con.execute("insert or replace into meta(k,v) values(?,?)", (k, v if isinstance(v, str) else json.dumps(v)))


def _meta_get(con, k):
    r = con.execute("select v from meta where k=?", (k,)).fetchone()
    return r[0] if r else None


# ======================================================================= data: dropping odds
def dropping(con, event_id, market, outcome, player, line):
    """Average across books of the change in implied probability since open, for this side.
    Positive = the price has shortened (money came in on this side). Returns (pct_drop, cents, n_books)."""
    rows = con.execute("""select o.book, o.price op, l.price np from opens o join last_quote l
                          on l.event_id=o.event_id and l.book=o.book and l.market=o.market and l.outcome=o.outcome
                          and l.player=o.player and coalesce(l.line,-999)=coalesce(o.line,-999)
                          where o.event_id=? and o.market=? and o.outcome=? and o.player=?
                          and coalesce(o.line,-999)=coalesce(?, -999)""",
                       (event_id, market, outcome, player or "", line)).fetchall()
    drops, cents = [], None
    for bk, op, np_ in rows:
        a, b = _imp(op), _imp(np_)
        if a and b:
            drops.append((b - a) / a)
            if bk == "pinnacle":
                try:
                    cents = _cents(op, np_) * (1 if b > a else -1)
                except ZeroDivisionError:
                    pass
    if not drops:
        return None, cents, 0
    return sum(drops) / len(drops), cents, len(drops)


# ======================================================================= data: DK public betting trends
_PCT = re.compile(r"(\d{1,3})%\s*/\s*(\d{1,3})%")


def parse_trends(page):
    """DK Network 'Public Betting Trends' article -> [(team_a, team_b, market, bets_a, bets_b, handle_a, handle_b)].
    Table header: Market | Team A (ABR) | Team B (ABR) | Bets % | Handle %; cells like '44% / 56%'."""
    out = []
    for tbl in re.findall(r"<table.*?</table>", page, flags=re.S | re.I):
        rows = [[_html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, flags=re.S | re.I)]
                for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, flags=re.S | re.I)]
        if not rows or len(rows[0]) < 5 or "bets" not in rows[0][3].lower():
            continue
        ta, tb = re.sub(r"\s*\(.*?\)", "", rows[0][1]).strip(), re.sub(r"\s*\(.*?\)", "", rows[0][2]).strip()
        for r in rows[1:]:
            if len(r) < 5:
                continue
            m1, m2 = _PCT.search(r[3]), _PCT.search(r[4])
            if not (m1 and m2):
                continue
            mk = {"moneyline": "h2h", "spread": "spreads", "total": "totals", "puck line": "spreads",
                  "run line": "spreads"}.get(r[0].strip().lower())
            if mk:
                out.append((ta, tb, mk, float(m1.group(1)), float(m1.group(2)), float(m2.group(1)), float(m2.group(2))))
    return out


def ingest_trends(con, now):
    """Pull recent 'public betting trends' posts from DK Network (WordPress REST) into the splits table."""
    after = (now - timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        posts = _get(f"https://dknetwork.draftkings.com/wp-json/wp/v2/posts?search=public%20betting%20trends"
                     f"&per_page=20&after={after}&_fields=title,link,content,modified")
    except Exception as e:                                        # noqa: BLE001
        _meta(con, "trends_status", f"{now.isoformat(timespec='seconds')} fetch failed: {str(e)[:120]}")
        return 0
    if not isinstance(posts, list):
        _meta(con, "trends_status", f"{now.isoformat(timespec='seconds')} unexpected reply from DK Network")
        return 0
    con.execute("""create table if not exists splits(event_id text, market text, outcome text, line real,
                   source text, handle_pct real, bets_pct real, ts text, url text,
                   primary key(event_id, market, outcome, source))""")
    lo, hi = (now - timedelta(hours=4)).isoformat(timespec="seconds"), (now + timedelta(hours=40)).isoformat(timespec="seconds")
    evs = list(con.execute("select event_id, home, away from events where commence>? and commence<?", (lo, hi)))
    nick = lambda t: _norm(t).split()[-1] if _norm(t) else ""
    n = 0
    for p in posts:
        for ta, tb, mk, ba, bb, ha, hb in parse_trends(p.get("content", {}).get("rendered", "")):
            na, nb = nick(ta), nick(tb)
            for eid, home, away in evs:
                if {nick(home), nick(away)} == {na, nb}:
                    side_a = home if nick(home) == na else away
                    side_b = away if side_a == home else home
                    if mk == "totals":
                        side_a, side_b = "Over", "Under"
                    for o, bets, handle in ((side_a, ba, ha), (side_b, bb, hb)):
                        con.execute("insert or replace into splits values(?,?,?,?,?,?,?,?,?)",
                                    (eid, mk, o, None, "dk_trends", handle, bets, now.isoformat(timespec="seconds"), p.get("link")))
                        n += 1
    _meta(con, "trends_status", f"{now.isoformat(timespec='seconds')} {len(posts)} posts, {n} split rows matched")
    return n


# ======================================================================= data: NFL usage + injuries + matchups
NFLV = "https://github.com/nflverse/nflverse-data/releases/download"


def _csv(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode("utf-8", "replace"))))


def ingest_nfl(con, season, now):
    """Snap counts, injury reports and positional defense from nflverse (free). Every 6 hours."""
    last = _meta_get(con, "nfl_usage_ts")
    if last and (now - datetime.fromisoformat(last)).total_seconds() < 6 * 3600:
        return _meta_get(con, "nfl_usage_status")
    _meta(con, "nfl_usage_ts", now.isoformat(timespec="seconds"))
    rep = []
    try:
        rows = _csv(f"{NFLV}/snap_counts/snap_counts_{season}.csv")
        for r in rows:
            try:
                con.execute("insert or replace into nfl_snaps values(?,?,?,?,?,?,?)",
                            (int(r["season"]), int(r["week"]), r["team"], _norm(r["player"]), r["player"],
                             r.get("position"), float(r.get("offense_pct") or 0)))
            except (KeyError, ValueError):
                continue
        rep.append(f"snaps {len(rows)}")
    except Exception as e:                                        # noqa: BLE001
        rep.append(f"snaps failed: {str(e)[:60]}")
    try:
        rows = _csv(f"{NFLV}/injuries/injuries_{season}.csv")
        for r in rows:
            try:
                con.execute("insert or replace into nfl_inj values(?,?,?,?,?,?)",
                            (int(r["season"]), int(r["week"]), r["team"], _norm(r["full_name"]), r["full_name"],
                             r.get("report_status") or ""))
            except (KeyError, ValueError):
                continue
        rep.append(f"injuries {len(rows)}")
    except Exception as e:                                        # noqa: BLE001
        rep.append(f"injuries failed: {str(e)[:60]}")
    try:
        rows = _csv(f"{NFLV}/stats_player/stats_player_week_{season}.csv")
        agg = {}
        for r in rows:
            pos, opp = r.get("position"), r.get("opponent_team")
            if pos not in ("QB", "RB", "WR", "TE") or not opp:
                continue
            for stat in ("passing_yards", "rushing_yards", "receiving_yards", "receptions"):
                try:
                    v = float(r.get(stat) or 0)
                except ValueError:
                    continue
                k = (opp, pos, stat)
                a = agg.setdefault(k, [0.0, set()])
                a[0] += v; a[1].add(r.get("week"))
        league = {}
        for (opp, pos, stat), (tot, wks) in agg.items():
            league.setdefault((pos, stat), []).append(tot / max(len(wks), 1))
        for (opp, pos, stat), (tot, wks) in agg.items():
            lg = league[(pos, stat)]
            con.execute("insert or replace into nfl_pos_def values(?,?,?,?,?,?,?)",
                        (season, opp, pos, stat, tot / max(len(wks), 1), sum(lg) / len(lg), len(wks)))
        rep.append(f"positional defense {len(agg)} rows")
    except Exception as e:                                        # noqa: BLE001
        rep.append(f"matchups failed: {str(e)[:60]}")
    st = f"{now.isoformat(timespec='seconds')} " + " | ".join(rep)
    _meta(con, "nfl_usage_status", st)
    con.commit()
    return st


def nfl_role(con, player):
    """(avg offense snap % over the last two weeks on file, latest injury status) for a player."""
    pn = _norm(player)
    snaps = [r[0] for r in con.execute("""select off_pct from nfl_snaps where player_norm=?
                                          order by season desc, week desc limit 2""", (pn,))]
    inj = con.execute("select status from nfl_inj where player_norm=? order by season desc, week desc limit 1", (pn,)).fetchone()
    return (sum(snaps) / len(snaps) if snaps else None), (inj[0] if inj else None)


NFL_STAT = {"player_pass_yds": "passing_yards", "player_rush_yds": "rushing_yards",
            "player_reception_yds": "receiving_yards", "player_receptions": "receptions"}



NFL_ABBR = {"Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
            "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
            "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
            "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
            "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
            "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
            "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
            "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}


def nfl_player_context(con, player, home, away):
    """(position, opponent abbreviation) for an NFL player from the latest snap-count row."""
    r = con.execute("select team, position from nfl_snaps where player_norm=? order by season desc, week desc limit 1",
                    (_norm(player),)).fetchone()
    if not r:
        return None, None
    ha, aa = NFL_ABBR.get(home), NFL_ABBR.get(away)
    opp = aa if r[0] == ha else ha if r[0] == aa else None
    return r[1], opp

def nfl_matchup(con, season, opp_team_abbr, position, market):
    stat = NFL_STAT.get(market)
    if not stat or not opp_team_abbr or not position:
        return None
    r = con.execute("select allowed_pg, league_pg, n from nfl_pos_def where season=? and team=? and position=? and stat=?",
                    (season, opp_team_abbr, position, stat)).fetchone()
    if not r or not r[1] or r[2] < 3:
        return None
    return r[0] / r[1]


# ======================================================================= data: NFL weather
STADIUM = {   # team: (lat, lon, dome/closed roof usually)
    "Arizona Cardinals": (33.528, -112.263, 1), "Atlanta Falcons": (33.755, -84.401, 1), "Baltimore Ravens": (39.278, -76.623, 0),
    "Buffalo Bills": (42.774, -78.787, 0), "Carolina Panthers": (35.226, -80.853, 0), "Chicago Bears": (41.862, -87.617, 0),
    "Cincinnati Bengals": (39.095, -84.516, 0), "Cleveland Browns": (41.506, -81.700, 0), "Dallas Cowboys": (32.748, -97.093, 1),
    "Denver Broncos": (39.744, -105.020, 0), "Detroit Lions": (42.340, -83.046, 1), "Green Bay Packers": (44.501, -88.062, 0),
    "Houston Texans": (29.685, -95.411, 1), "Indianapolis Colts": (39.760, -86.164, 1), "Jacksonville Jaguars": (30.324, -81.637, 0),
    "Kansas City Chiefs": (39.049, -94.484, 0), "Las Vegas Raiders": (36.091, -115.184, 1), "Los Angeles Chargers": (33.953, -118.339, 1),
    "Los Angeles Rams": (33.953, -118.339, 1), "Miami Dolphins": (25.958, -80.239, 0), "Minnesota Vikings": (44.974, -93.258, 1),
    "New England Patriots": (42.091, -71.264, 0), "New Orleans Saints": (29.951, -90.081, 1), "New York Giants": (40.813, -74.074, 0),
    "New York Jets": (40.813, -74.074, 0), "Philadelphia Eagles": (39.901, -75.168, 0), "Pittsburgh Steelers": (40.447, -80.016, 0),
    "San Francisco 49ers": (37.403, -121.970, 0), "Seattle Seahawks": (47.595, -122.332, 0), "Tampa Bay Buccaneers": (27.976, -82.503, 0),
    "Tennessee Titans": (36.166, -86.771, 0), "Washington Commanders": (38.908, -76.864, 0)}


def weather_for(con, event_id, home, commence, now):
    """Wind (mph), precipitation (mm) and temperature at kickoff from Open-Meteo (free), cached 3 hours."""
    st = STADIUM.get(home)
    if not st:
        return None
    lat, lon, dome = st
    r = con.execute("select ts, wind_mph, precip_mm, temp_c, dome from weather where event_id=?", (event_id,)).fetchone()
    if r and (now - datetime.fromisoformat(r[0])).total_seconds() < 3 * 3600:
        return {"wind_mph": r[1], "precip_mm": r[2], "temp_c": r[3], "dome": r[4]}
    if dome:
        con.execute("insert or replace into weather values(?,?,?,?,?,?)", (event_id, now.isoformat(timespec="seconds"), 0, 0, None, 1))
        return {"wind_mph": 0.0, "precip_mm": 0.0, "temp_c": None, "dome": 1}
    try:
        kick = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
        if (kick - now).days > 14:
            return None
        d = _get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                 f"&hourly=wind_speed_10m,precipitation,temperature_2m&wind_speed_unit=mph&timezone=UTC"
                 f"&start_date={kick.date().isoformat()}&end_date={kick.date().isoformat()}")
        hrs = d["hourly"]["time"]
        i = min(range(len(hrs)), key=lambda j: abs(datetime.fromisoformat(hrs[j]).replace(tzinfo=UTC) - kick))
        w = {"wind_mph": float(d["hourly"]["wind_speed_10m"][i]), "precip_mm": float(d["hourly"]["precipitation"][i]),
             "temp_c": float(d["hourly"]["temperature_2m"][i]), "dome": 0}
        con.execute("insert or replace into weather values(?,?,?,?,?,?)",
                    (event_id, now.isoformat(timespec="seconds"), w["wind_mph"], w["precip_mm"], w["temp_c"], 0))
        return w
    except Exception:                                             # noqa: BLE001
        return None


# ======================================================================= data: soccer results + prop grading
ESPN_SOCCER = {"EPL": "eng.1", "UCL": "uefa.champions"}


def _soccer_stat(stats, names):
    for s in stats or []:
        if str(s.get("name", "")).lower() in names or str(s.get("abbreviation", "")).lower() in names:
            try:
                return float(s.get("value") if s.get("value") is not None else s.get("displayValue"))
            except (TypeError, ValueError):
                return None
    return None


def ingest_soccer_day(con, league, day):
    """One date of finished matches. A match whose players' goals don't reconcile with the score is
    held (own goals and missing players both cause that), so a prop is never graded from a bad box."""
    code = ESPN_SOCCER[league]
    sb = _get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard?dates={day.replace('-', '')}")
    got = 0
    for ev in sb.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        if not (comp.get("status", {}).get("type", {}).get("completed")):
            continue
        teams = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        if "home" not in teams or "away" not in teams:
            continue
        home, away = teams["home"]["team"]["displayName"], teams["away"]["team"]["displayName"]
        hs, as_ = int(teams["home"].get("score") or 0), int(teams["away"].get("score") or 0)
        sm = _get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/summary?event={ev['id']}")
        lines, sums = [], {}
        for ros in sm.get("rosters", []):
            tname = (ros.get("team") or {}).get("displayName", "")
            for p in ros.get("roster", []):
                nm = (p.get("athlete") or {}).get("displayName")
                if not nm:
                    continue
                g = _soccer_stat(p.get("stats"), ("totalgoals", "goals", "g"))
                a = _soccer_stat(p.get("stats"), ("goalassists", "assists", "a"))
                lines.append((tname, nm, g or 0.0, a or 0.0))
                sums[tname] = sums.get(tname, 0.0) + (g or 0.0)
        if sums.get(home, -1) != hs or sums.get(away, -1) != as_:
            _meta(con, f"soccer_incomplete_{ev['id']}", f"{day} {away} @ {home}: player goals {sums} vs score {hs}-{as_}")
            continue
        con.execute("insert or replace into soccer_results values(?,?,?,?,?,?,?,?)",
                    (ev["id"], league, day, home, away, hs, as_, datetime.now(UTC).isoformat(timespec="seconds")))
        for tname, nm, g, a in lines:
            con.execute("insert or replace into soccer_box values(?,?,?,?,?,?,?)", (ev["id"], day, tname, _norm(nm), nm, g, a))
        got += 1
    return got


def grade_soccer(con, now):
    """Settle pending EPL/UCL anytime-goalscorer and assists props from reconciled ESPN boxes."""
    _ensure(con)
    graded = 0
    for lg in ESPN_SOCCER:
        for k in range(3, -1, -1):
            day = (now - timedelta(days=k)).date().isoformat()
            try:
                ingest_soccer_day(con, lg, day)
            except Exception as e:                                # noqa: BLE001
                _meta(con, f"soccer_feed_error_{lg}", f"{now.isoformat(timespec='seconds')} {str(e)[:120]}")
    for b in con.execute("""select * from bets where sport in ('EPL','UCL') and result='pending'
                            and market in ('player_goal_scorer_anytime','player_assists') and commence<?""",
                         ((now - timedelta(hours=3)).isoformat(timespec="seconds"),)).fetchall():
        day = str(b["commence"])[:10]
        d0 = datetime.fromisoformat(day)
        lo, hi = (d0 - timedelta(days=1)).date().isoformat(), (d0 + timedelta(days=1)).date().isoformat()
        try:
            away, home = [x.strip() for x in str(b["game"]).split("@")]
        except ValueError:
            continue
        nick = lambda t: _norm(t).split()[-1] if _norm(t) else ""
        m = [r for r in con.execute("select * from soccer_results where game_date between ? and ?", (lo, hi))
             if {nick(r["home"]), nick(r["away"])} == {nick(home), nick(away)}]
        if not m:
            continue
        row = con.execute("select * from soccer_box where source_id=? and player_norm=?", (m[0]["source_id"], _norm(b["player"]))).fetchone()
        if not row:
            continue                                             # not in a reconciled box: unknown, stays pending
        if b["market"] == "player_goal_scorer_anytime":
            res = "W" if (row["goals"] >= 1) == (b["selection"] == "Yes") else "L"
            val = row["goals"]
        else:
            v, L = row["assists"], float(b["line"] or 0.5)
            res = "P" if v == L else ("W" if (v > L) == (b["selection"] == "Over") else "L")
            val = v
        pnl = b["stake"] * (b["price"] - 1) if res == "W" else (-b["stake"] if res == "L" else 0.0)
        con.execute("update bets set result=?, pnl=?, graded_ts=?, note=coalesce(note,'')||? where id=?",
                    (res, pnl, now.isoformat(timespec="seconds"), f" | settled {val:g} (ESPN soccer)", b["id"]))
        graded += 1
    con.commit()
    return graded


# ======================================================================= NBA game-script simulation
def sim_prop(proj, line, side, spread=None, total=None, stat="pts", n=4000, seed=7):
    """Probability the side wins once game script is simulated: margin ~ N(spread, 12), total ~ N(total, 16).
    Blowouts (|margin| > 18) cut a starter's minutes ~15%; overtime (|margin| < 1, ~6% of games) adds ~10%;
    a higher-scoring game lifts counting stats in proportion. Points ~ Normal, counts ~ Poisson."""
    if proj is None or line is None:
        return None
    rnd = random.Random(seed)
    base_total = total or 225.0
    wins = 0
    for _ in range(n):
        m = rnd.gauss(spread or 0.0, 12.0)
        t = rnd.gauss(base_total, 16.0)
        f = 1.0
        if abs(m) > 18:
            f *= 0.85
        elif abs(m) < 1 and rnd.random() < 0.5:
            f *= 1.10
        f *= max(0.7, min(1.3, t / base_total))
        lam = max(proj * f, 0.05)
        if stat == "pts":
            x = rnd.gauss(lam, max(0.35 * lam, 2.5))
        else:
            # Poisson draw
            L_, k, p = math.exp(-lam), 0, 1.0
            while True:
                p *= rnd.random()
                if p <= L_:
                    break
                k += 1
            x = k
        over = x > line
        wins += 1 if (over if side in ("Over", "Yes") else not over) else 0
    return wins / n


# ======================================================================= signal capture
SIGNALS = ["drop_pct", "tape_cents", "steam", "rlm", "dk_gap", "shade_pp", "proj_p", "sim_p", "role_change",
           "injury_q", "snap_pct", "matchup", "wind_mph", "precip_mm", "hours", "edge", "n_books", "is_fav", "is_home"]


def capture(con, ev, market, outcome, player, line, price, fair, extra=None):
    """Every signal's value for one edge. Stored on the paper bet so its value can be measured later."""
    extra = extra or {}
    s = {k: None for k in SIGNALS}
    try:
        dp, cents, nb = dropping(con, ev["event_id"], market, outcome, player, line)
        s["drop_pct"], s["tape_cents"] = (None if dp is None else round(dp, 4)), (None if cents is None else round(cents, 1))
    except Exception:                                             # noqa: BLE001
        pass
    sig = extra.get("sig") or {}
    s["steam"] = (sig.get("steam_for") or 0) - (sig.get("steam_against") or 0) if sig else None
    s["rlm"] = 1 if sig.get("rlm") else (0 if sig else None)
    s["dk_gap"] = sig.get("dk_gap")
    s["shade_pp"] = sig.get("shade_pp")
    pc = extra.get("pc") or {}
    s["proj_p"] = pc.get("p_side")
    s["role_change"] = 1 if pc.get("role_change") else (0 if pc else None)
    s["injury_q"] = 1 if extra.get("injury") == "doubtful" else 0
    s["snap_pct"] = extra.get("snap_pct")
    s["matchup"] = extra.get("matchup")
    w = extra.get("weather") or {}
    s["wind_mph"], s["precip_mm"] = w.get("wind_mph"), w.get("precip_mm")
    s["sim_p"] = extra.get("sim_p")
    try:
        st = datetime.fromisoformat(str(ev["commence"]).replace("Z", "+00:00"))
        s["hours"] = round((st - datetime.now(UTC)).total_seconds() / 3600, 2)
    except Exception:                                             # noqa: BLE001
        pass
    s["edge"] = round(float(price) * float(fair) - 1, 4) if fair else None
    s["n_books"] = extra.get("n_books")
    if not player and market in ("h2h", "spreads"):
        s["is_home"] = 1 if outcome == ev.get("home") else 0
        s["is_fav"] = 1 if (fair or 0) > 0.5 else 0
    return s


# ======================================================================= learning
def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _graded_rows(con):
    rows = []
    for b in con.execute("""select result, fair_prob, model_prob, price, clv_pp, signals, sport, market, evidence_kind, hours_to_start
                            from bets where paper=1 and result in ('W','L') and signals is not null"""):
        try:
            sg = json.loads(b["signals"])
        except Exception:                                         # noqa: BLE001
            continue
        f = b["fair_prob"] or b["model_prob"]
        if not f:
            continue
        rows.append((1 if b["result"] == "W" else 0, float(f), sg, b))
    return rows


def _fit(rows, keys, stats, l2, iters=600, lr=1.0):
    """Logistic regression with the market fair price as offset, ridge-penalised. Returns (weights, se)."""
    w = {k: 0.0 for k in keys}
    n = max(len(rows), 1)
    def x(sg, k):
        v = sg.get(k)
        return 0.0 if v is None else (v - stats[k]["mu"]) / stats[k]["sd"]
    for _ in range(iters):
        grad = {k: 0.0 for k in keys}
        for y, f, sg, _b in rows:
            z = _logit(f) + sum(w[k] * x(sg, k) for k in keys)
            p = 1 / (1 + math.exp(-max(min(z, 30), -30)))
            for k in keys:
                grad[k] += (p - y) * x(sg, k)
        for k in keys:
            w[k] -= lr * (grad[k] + l2 * w[k]) / n
    info = {k: 0.0 for k in keys}
    for y, f, sg, _b in rows:
        z = _logit(f) + sum(w[k] * x(sg, k) for k in keys)
        p = 1 / (1 + math.exp(-max(min(z, 30), -30)))
        for k in keys:
            info[k] += p * (1 - p) * x(sg, k) ** 2
    se = {k: (1 / math.sqrt(info[k])) if info[k] > 0 else float("inf") for k in keys}
    return w, se


def fit_weights(con, now):
    """A signal graduates only when it has GRADUATE_N graded spots, actually varies, AND its effect on
    results is statistically clear (|z| >= 2 on its own, with the market price as the baseline).
    Graduated signals are then fitted together, ridge-shrunk; everything else stays at weight 0."""
    rows = _graded_rows(con)
    stats = {}
    for k in SIGNALS:
        vals = [r[2].get(k) for r in rows if r[2].get(k) is not None]
        mu = sum(vals) / len(vals) if vals else 0.0
        sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
        stats[k] = {"n": len(vals), "mu": mu, "sd": sd if sd > 1e-9 else 1.0, "varies": sd > 1e-9}
    status, zs = {}, {}
    for k in SIGNALS:
        if stats[k]["n"] < GRADUATE_N:
            status[k] = f"probation {stats[k]['n']}/{GRADUATE_N}"; continue
        if not stats[k]["varies"]:
            status[k] = "constant (no information)"; continue
        sub = [r for r in rows if r[2].get(k) is not None]
        w1, se1 = _fit(sub, [k], stats, l2=1.0)
        zs[k] = w1[k] / se1[k] if se1[k] not in (0, float("inf")) else 0.0
        status[k] = "graduated" if abs(zs[k]) >= 2.0 else f"no clear effect yet (z {zs[k]:+.1f})"
    live = [k for k in SIGNALS if status[k] == "graduated"]
    w = {k: 0.0 for k in SIGNALS}
    if live:
        wl, _se = _fit(rows, live, stats, l2=L2)
        w.update(wl)
    out = {"ts": now.isoformat(timespec="seconds"), "n_rows": len(rows),
           "weights": {k: round(w[k], 4) for k in SIGNALS}, "z": {k: round(v, 2) for k, v in zs.items()},
           "stats": stats, "status": status}
    _meta(con, "signal_weights", out)
    return out


def adjust(fair, signals, model=None):
    """Market fair -> fair shifted by measured signal weights. Zero weights = no change. Capped."""
    if not fair or not signals or not model:
        return fair
    w, st = model.get("weights", {}), model.get("stats", {})
    shift = 0.0
    for k, wk in w.items():
        v = signals.get(k)
        if wk and v is not None and k in st:
            shift += wk * ((v - st[k]["mu"]) / (st[k]["sd"] or 1.0))
    shift = max(-MAX_SHIFT_LOGIT, min(MAX_SHIFT_LOGIT, shift))
    z = _logit(fair) + shift
    return 1 / (1 + math.exp(-z))


def load_model(con):
    try:
        return json.loads(_meta_get(con, "signal_weights") or "null")
    except Exception:                                             # noqa: BLE001
        return None


# ======================================================================= reports
def signal_scorecard(con):
    """Per signal: graded spots where it was 'on' vs 'off', win rate, ROI and CLV for each."""
    rows = _graded_rows(con)
    out = {}
    for k in SIGNALS:
        vals = [r[2].get(k) for r in rows if r[2].get(k) is not None]
        if not vals:
            out[k] = {"n": 0}
            continue
        med = sorted(vals)[len(vals) // 2]
        on = [r for r in rows if r[2].get(k) is not None and r[2][k] > med]
        off = [r for r in rows if r[2].get(k) is not None and r[2][k] <= med]
        def summ(rs):
            if not rs:
                return None
            wins = sum(r[0] for r in rs)
            pnl = sum((r[3]["price"] - 1) if r[0] else -1 for r in rs)
            clv = [r[3]["clv_pp"] for r in rs if r[3]["clv_pp"] is not None]
            return {"n": len(rs), "win": round(wins / len(rs), 3), "roi": round(pnl / len(rs), 3),
                    "clv": round(sum(clv) / len(clv), 2) if clv else None}
        out[k] = {"n": len(vals), "split_at": med, "high": summ(on), "low": summ(off)}
    return out


def calibration(con):
    """Bets grouped by the fair probability at bet time: does a 60% price win 60% of the time?"""
    rows = _graded_rows(con)
    buckets = {}
    for y, f, _sg, _b in rows:
        k = min(int(f * 10), 9)
        a = buckets.setdefault(k, [0, 0, 0.0])
        a[0] += 1; a[1] += y; a[2] += (f - y) ** 2
    out = []
    for k in sorted(buckets):
        n, w, se = buckets[k]
        out.append({"bucket": f"{k * 10}-{k * 10 + 10}%", "n": n, "hit": round(w / n, 3)})
    brier = round(sum(b[2] for b in buckets.values()) / max(sum(b[0] for b in buckets.values()), 1), 4)
    return {"buckets": out, "brier": brier}


def autopsy(con, days=14):
    """Every graded paper bet in the window, sorted into a cause."""
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")
    causes = {}
    for b in con.execute("""select * from bets where paper=1 and result in ('W','L') and graded_ts>?""", (since,)):
        sg = {}
        try:
            sg = json.loads(b["signals"] or "{}")
        except Exception:                                         # noqa: BLE001
            pass
        note = str(b["note"] or "")
        if b["result"] == "W":
            c = "win: value held" if (b["clv_pp"] or 0) >= 0 else "win: despite losing value"
        elif b["clv_pp"] is not None and b["clv_pp"] < -1.0:
            c = "loss: bad price (lost value to the close)"
        elif sg.get("role_change") == 1 or (sg.get("snap_pct") is not None and sg["snap_pct"] < 40):
            c = "loss: role / minutes"
        elif "settled" in note and b["line"] is not None and re.search(r"settled ([\d.]+) vs", note):
            v = float(re.search(r"settled ([\d.]+) vs", note).group(1))
            c = "loss: narrow (variance)" if abs(v - float(b["line"])) <= max(1.0, 0.1 * abs(float(b["line"]))) else "loss: clear miss"
        else:
            c = "loss: other / game script"
        causes[c] = causes.get(c, 0) + 1
    return causes


def timing(con):
    """CLV and ROI by how early the edge was taken."""
    out = {}
    for b in con.execute("""select hours_to_start h, clv_pp, result, price, market from bets
                            where paper=1 and result in ('W','L') and hours_to_start is not null"""):
        bucket = "6h+" if b["h"] >= 6 else "2-6h" if b["h"] >= 2 else "under 2h"
        kind = "props" if str(b["market"]).startswith("player_") else "game lines"
        a = out.setdefault(f"{kind} {bucket}", {"n": 0, "pnl": 0.0, "clv": []})
        a["n"] += 1
        a["pnl"] += (b["price"] - 1) if b["result"] == "W" else -1
        if b["clv_pp"] is not None:
            a["clv"].append(b["clv_pp"])
    return {k: {"n": v["n"], "roi": round(v["pnl"] / v["n"], 3),
                "clv": round(sum(v["clv"]) / len(v["clv"]), 2) if v["clv"] else None} for k, v in out.items()}


def contradictions(con, now):
    """Picks that contradict KRONOS's own numbers (2026-10-29 rule, owner's checklist):
      - spread: KRONOS's projected margin must cover the line on the picked side;
      - moneyline: judged on value, not on who is projected to win — a plus-money underdog KRONOS
        rates above its price is consistent; a pick KRONOS itself rates at or below the price is not.
    One entry per game and market."""
    out, seen = [], set()
    hi = (now + timedelta(hours=36)).isoformat(timespec="seconds")
    for m in con.execute("""select m.*, e.home, e.away from model_probs m join events e using(event_id)
                            where e.commence>? and e.commence<? and (m.call like 'Lean%' or m.call like 'Play%')
                            and m.market in ('spreads','h2h')""", (now.isoformat(timespec="seconds"), hi)):
        key = (m["event_id"], m["market"])
        if key in seen:
            continue
        side = str(m["call"]).split(" ", 1)[1].split(" [")[0].strip()
        if side not in (m["home"], m["away"]):
            continue
        game = f"{m['away']} @ {m['home']}"
        if m["market"] == "spreads":
            n = con.execute("select margin from jerry_numbers where event_id=?", (m["event_id"],)).fetchone()
            if not n or n[0] is None or m["line"] is None:
                continue
            line = float(m["line"]) if side == m["outcome"] else -float(m["line"])
            proj = float(n[0]) if side == m["home"] else -float(n[0])   # picked side's projected margin
            if proj + line <= 0:
                out.append(f"{game}: spread pick {side} {line:+g} but KRONOS projects {side} by {proj:+.1f}, which doesn't cover")
                seen.add(key)
        else:
            if m["model_prob"] is None or m["fair_prob"] is None:
                continue
            p = m["model_prob"] if side == m["outcome"] else 1 - m["model_prob"]
            f = m["fair_prob"] if side == m["outcome"] else 1 - m["fair_prob"]
            if p <= f:
                out.append(f"{game}: moneyline pick {side} but KRONOS rates it {p:.0%}, at or below the market's {f:.0%}")
                seen.add(key)
    return out


def run(con, now):
    """Refresh the learning layer and its reports (every 6 hours) and grade soccer props."""
    _ensure(con)
    rep = {}
    last = _meta_get(con, "learn_ts")
    if not last or (now - datetime.fromisoformat(last)).total_seconds() >= 6 * 3600:
        _meta(con, "learn_ts", now.isoformat(timespec="seconds"))
        model = fit_weights(con, now)
        rep["weights"] = {k: v for k, v in model["weights"].items() if v}
        _meta(con, "learn_report", {"scorecard": signal_scorecard(con), "calibration": calibration(con),
                                    "autopsy": autopsy(con), "timing": timing(con),
                                    "graduated": [k for k, v in model["status"].items() if v == "graduated"]})
    _meta(con, "contradictions", contradictions(con, now))
    try:
        rep["soccer_graded"] = grade_soccer(con, now)
    except Exception as e:                                        # noqa: BLE001
        rep["soccer_graded"] = f"error {str(e)[:80]}"
    try:
        rep["trends"] = ingest_trends(con, now)
    except Exception as e:                                        # noqa: BLE001
        rep["trends"] = f"error {str(e)[:80]}"
    season = now.year if now.month >= 8 else now.year - 1
    try:
        rep["nfl"] = ingest_nfl(con, season, now)
    except Exception as e:                                        # noqa: BLE001
        rep["nfl"] = f"error {str(e)[:80]}"
    con.commit()
    return rep
'''

_SPLITS_SRC = r'''"""
jerry_splits.py — free DraftKings betting splits (bets % and handle %) for KRONOS.

DraftKings Network publishes splits for free in two places:
  1. the live splits table — the page HTML is a shell; the numbers load from a
     data feed the browser calls. The feed URL isn't documented, so this module
     DISCOVERS it: it scans the page and its scripts for candidate endpoints, tries
     them, and records what it found in meta 'splits_discovery' so the parser can
     be pinned down from the feed summary.
  2. per-game "betting splits" articles — plain HTML tables (Market | Side |
     Handle % | Bets %). Fully parsed here. They are snapshots, not live.

DraftKings is the most recreational book in the US, which is the point: these
numbers are the cleanest available read on where the public is.
"""
from __future__ import annotations

import html as _html
import json
import re
import urllib.request
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
DKN = "https://dknetwork.draftkings.com"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36", "Accept": "text/html,application/json;q=0.9,*/*;q=0.8"}

MARKET_MAP = {"moneyline": "h2h", "money line": "h2h", "run line": "spreads", "puck line": "spreads",
              "spread": "spreads", "point spread": "spreads", "total": "totals", "game total": "totals"}

MULTIWORD = ("red sox", "white sox", "blue jays", "maple leafs", "red wings", "golden knights",
             "trail blazers", "blue jackets", "st louis")


def _get(url, timeout=25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _ensure(con):
    con.execute("""create table if not exists splits(event_id text, market text, outcome text, line real,
                   source text, handle_pct real, bets_pct real, ts text, url text,
                   primary key(event_id, market, outcome, source))""")


def _meta(con, k, v):
    try:
        con.execute("insert or replace into meta(k,v) values(?,?)", (k, str(v)[:600]))
    except Exception:
        pass


def _nick(team):
    t = re.sub(r"[^a-z ]", "", str(team).lower()).strip()
    for m in MULTIWORD:
        if t.endswith(m):
            return m
    return t.split()[-1] if t.split() else t


# --------------------------------------------------------------- article tables

def parse_article(content_html):
    """Return [(market, side_text, handle_pct, bets_pct)] from a DK Network splits article."""
    h = content_html
    i = h.lower().find("betting splits</h2>")
    if i < 0:
        i = h.lower().find("draftkings betting splits")
    if i < 0:
        return []
    seg = h[i:i + 6000]
    end = seg.lower().find("</table>")
    seg = seg[:end] if end > 0 else seg
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", seg, flags=re.S | re.I):
        cells = [_html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S | re.I)]
        if len(cells) < 4 or not cells[2].endswith("%"):
            continue
        try:
            out.append((cells[0], cells[1], float(cells[2].rstrip("%")), float(cells[3].rstrip("%"))))
        except ValueError:
            continue
    return out


def _title_teams(title):
    t = _html.unescape(re.sub(r"<[^>]+>", "", title))
    t = re.sub(r"^\s*(?:first|final|early|late)?\s*look\s+at\s+", "", t, flags=re.I)   # "First Look at X vs. Y"
    t = re.split(r"\s+Betting Splits", t, flags=re.I)[0]
    for sep in (r"\s+vs\.?\s+", r"\s+@\s+", r"\s+at\s+"):          # 'vs.' first: 'at' also appears in prose
        parts = re.split(sep, t, maxsplit=1, flags=re.I)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    return None, None


def _match_event(con, a, b, now):
    if not a or not b:
        return None
    na, nb = _nick(a), _nick(b)
    lo, hi = (now - timedelta(hours=4)).isoformat(timespec="seconds"), (now + timedelta(hours=40)).isoformat(timespec="seconds")
    for eid, home, away in con.execute("select event_id, home, away from events where commence>? and commence<?", (lo, hi)):
        hn, an = _nick(home), _nick(away)
        if {na, nb} == {hn, an}:
            return eid, home, away
    return None


def _side_to_outcome(side, home, away):
    """'MIL Brewers -1.5' -> ('Milwaukee Brewers', -1.5); 'Over 8' -> ('Over', 8.0)."""
    s = side.strip()
    m = re.match(r"^(Over|Under)\s+([\d.]+)", s, flags=re.I)
    if m:
        return m.group(1).title(), float(m.group(2))
    line = None
    mm = re.search(r"([+-]\d+(?:\.\d+)?)\s*$", s)
    if mm:
        line = float(mm.group(1)); s = s[:mm.start()].strip()
    sn = _nick(s)
    for team in (home, away):
        if _nick(team) == sn or sn in team.lower():
            return team, line
    return None, line


def from_articles(con, now, limit=60):
    """Pull recent DK Network splits articles via the WordPress REST API, falling back
    to the category listing if REST is closed. Returns number of rows stored."""
    after = (now - timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%S")
    posts = []
    try:
        raw = _get(f"{DKN}/wp-json/wp/v2/posts?search=betting%20splits&per_page={min(limit, 100)}"
                   f"&after={after}&_fields=title,link,content,modified")
        posts = [(p["title"]["rendered"], p["link"], p["content"]["rendered"], p.get("modified", "")) for p in json.loads(raw)]
        _meta(con, "splits_articles_source", f"{now.isoformat(timespec='seconds')} wp-json {len(posts)} posts")
    except Exception as e:                       # noqa: BLE001
        _meta(con, "splits_articles_source", f"{now.isoformat(timespec='seconds')} wp-json failed: {str(e)[:120]}")
        try:
            listing = _get(f"{DKN}/category/draftkings-sportsbook-insights/")
            links = sorted(set(re.findall(r'href="(https://dknetwork\.draftkings\.com/\d{4}/\d{2}/\d{2}/[^"]*betting-splits[^"]*)"', listing)))[:limit]
            for u in links:
                try:
                    page = _get(u)
                    t = re.search(r"<title>(.*?)</title>", page, flags=re.S)
                    posts.append((t.group(1) if t else "", u, page, ""))
                except Exception:
                    continue
            _meta(con, "splits_articles_source", f"{now.isoformat(timespec='seconds')} category fallback {len(posts)} pages")
        except Exception as e2:                  # noqa: BLE001
            _meta(con, "splits_articles_source", f"{now.isoformat(timespec='seconds')} category failed too: {str(e2)[:120]}")
            return 0
    n = 0
    for title, link, content, modified in posts:
        a, b = _title_teams(title)
        ev = _match_event(con, a, b, now)
        if not ev:
            continue
        eid, home, away = ev
        for mkt_txt, side, handle, bets in parse_article(content):
            market = MARKET_MAP.get(mkt_txt.strip().lower())
            if not market:
                continue
            outcome, line = _side_to_outcome(side, home, away)
            if not outcome:
                continue
            con.execute("insert or replace into splits values(?,?,?,?,?,?,?,?,?)",
                        (eid, market, outcome, line, "dk_article", handle, bets, modified or now.isoformat(timespec="seconds"), link))
            n += 1
    return n


# ------------------------------------------------------------- live-table feed

def discover_live_feed(con, now):
    """Find the data feed behind the live splits table. Records candidates and a
    sample of whatever JSON comes back, so the exact parser can be written from the
    feed summary. Returns the working URL or None. Parsing is generic until then."""
    found = []
    try:
        page = _get(f"{DKN}/draftkings-sportsbook-betting-splits/")
    except Exception as e:                       # noqa: BLE001
        _meta(con, "splits_discovery", f"{now.isoformat(timespec='seconds')} page fetch failed: {str(e)[:150]}")
        return None
    cands = set(re.findall(r'["\'](https?://[^"\']*(?:split|betting-splits|bettingsplits)[^"\']*)["\']', page, flags=re.I))
    cands |= {DKN + u for u in re.findall(r'["\'](/wp-json/[^"\']+)["\']', page)}
    for s in re.findall(r'<script[^>]+src="([^"]+)"', page)[:25]:
        if "split" in s.lower() or "dkn" in s.lower():
            try:
                js = _get(s if s.startswith("http") else DKN + s)
                cands |= set(re.findall(r'["\'](https?://[^"\']+)["\']', js)) if "split" in js.lower() else set()
                cands |= {DKN + u for u in re.findall(r'["\'](/wp-json/[^"\']*split[^"\']*)["\']', js, flags=re.I)}
            except Exception:
                continue
    cands = [c for c in cands if not c.endswith((".css", ".png", ".jpg", ".svg", ".webp", ".woff2"))][:20]
    working = None
    for c in cands:
        try:
            body = _get(c)
            data = json.loads(body)
            keys = _sample_keys(data)
            found.append(f"{c} -> json keys {keys[:12]}")
            if any("handle" in k.lower() for k in keys) and any("bet" in k.lower() for k in keys):
                working = c
                break
        except Exception as e:                   # noqa: BLE001
            found.append(f"{c} -> {type(e).__name__}")
    _meta(con, "splits_discovery", f"{now.isoformat(timespec='seconds')} candidates={len(cands)} working={working} | " + " || ".join(found[:6]))
    if working:
        _meta(con, "splits_feed_url", working)
    return working


def _sample_keys(obj, depth=0, out=None):
    out = [] if out is None else out
    if depth > 5 or len(out) > 60:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in out:
                out.append(k)
            _sample_keys(v, depth + 1, out)
    elif isinstance(obj, list):
        for v in obj[:3]:
            _sample_keys(v, depth + 1, out)
    return out



# ------------------------------------------------------------ the table itself
#
# The splits page serves its table in the HTML (a search engine's cached copy shows
# it as plain text). Layout per game, read top to bottom:
#   Moneyline | Odds | % Handle | % Bets | <side> [odds] <handle>% <bets>% | <side> ...
#   Run Line / Puck Line / Spread ... | Total ...
# The page also shows "Betting splits are temporarily unavailable" during outages.

_MKT_HDR = {"moneyline": "h2h", "run line": "spreads", "puck line": "spreads", "spread": "spreads",
            "point spread": "spreads", "total": "totals"}


def _text_lines(page):
    page = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", page, flags=re.S | re.I)
    page = re.sub(r"<br\s*/?>|</(div|p|li|tr|td|th|span|h\d|a|strong|b)>", "\n", page, flags=re.I)
    txt = _html.unescape(re.sub(r"<[^>]+>", " ", page))
    return [l.strip() for l in txt.split("\n") if l.strip()]


def parse_table(page):
    """Return [{market, rows:[(side, handle, bets)]}] blocks in page order."""
    lines = _text_lines(page)
    blocks, cur, pend_side = [], None, None
    pct = re.compile(r"^(\d{1,3})%$")
    i = 0
    while i < len(lines):
        l = lines[i]; low = l.lower()
        if low in _MKT_HDR:
            cur = {"market": _MKT_HDR[low], "rows": []}; blocks.append(cur); pend_side = None
            i += 1; continue
        if cur is None or low in ("odds", "% handle", "% bets", "handle %", "bets %"):
            i += 1; continue
        m = pct.match(l)
        if m and pend_side is not None:
            # next percentage after the side is handle, the one after is bets
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            m2 = pct.match(nxt)
            if m2:
                cur["rows"].append((pend_side, float(m.group(1)), float(m2.group(1))))
                pend_side = None; i += 2; continue
        if not m and not re.match(r"^[+-−]?\d+(\.\d+)?$", l.replace("−", "-")):
            pend_side = l           # a side label ("PHI Phillies", "TB Rays -1.5", "Over 9")
        i += 1
    return [b for b in blocks if len(b["rows"]) >= 2]


def from_table(con, now, url=None):
    """Read the live splits table and store rows matched to KRONOS events."""
    url = url or f"{DKN}/draftkings-sportsbook-betting-splits/"
    page = _get(url)
    if "temporarily unavailable" in page.lower() and "% handle" not in page.lower():
        _meta(con, "splits_table", f"{now.isoformat(timespec='seconds')} DraftKings reports the table temporarily unavailable")
        return 0
    blocks = parse_table(page)
    lo, hi = (now - timedelta(hours=4)).isoformat(timespec="seconds"), (now + timedelta(hours=40)).isoformat(timespec="seconds")
    evs = list(con.execute("select event_id, home, away from events where commence>? and commence<?", (lo, hi)))
    n = games = 0
    ev = None
    for b in blocks:
        if b["market"] == "h2h":                    # a moneyline block starts a new game
            names = {_nick(r[0]) for r in b["rows"][:2]}
            ev = next(((e, h, a) for e, h, a in evs if {_nick(h), _nick(a)} == names), None)
            games += 1
        if not ev:
            continue
        eid, home, away = ev
        for side, handle, bets in b["rows"][:2]:
            outcome, line = _side_to_outcome(side, home, away)
            if not outcome:
                continue
            con.execute("insert or replace into splits values(?,?,?,?,?,?,?,?,?)",
                        (eid, b["market"], outcome, line, "dk_table", handle, bets, now.isoformat(timespec="seconds"), url))
            n += 1
    _meta(con, "splits_table", f"{now.isoformat(timespec='seconds')} {games} games on page, {n} rows matched to tracked events")
    return n

def run(con, now=None):
    """One refresh: try the live feed (discovery only for now), then the articles."""
    now = now or datetime.now(UTC)
    _ensure(con)
    rep = {}
    try:
        rep["table_rows"] = from_table(con, now)
    except Exception as e:                       # noqa: BLE001
        rep["table_rows"] = f"error {str(e)[:80]}"
        _meta(con, "splits_table", f"{now.isoformat(timespec='seconds')} error {str(e)[:150]}")
    try:
        rep["article_rows"] = from_articles(con, now)
    except Exception as e:                       # noqa: BLE001
        rep["article_rows"] = f"error {str(e)[:80]}"
    con.commit()
    return rep


def splits_for(con, event_id, market, outcome):
    """(handle_pct, bets_pct, source) for an outcome, or None."""
    try:
        _ensure(con)
        r = con.execute("""select handle_pct, bets_pct, source from splits where event_id=? and market=? and outcome=?
                           order by ts desc limit 1""", (event_id, market, outcome)).fetchone()
        return (r[0], r[1], r[2]) if r else None
    except Exception:
        return None
'''

_NBA_SRC = r'''"""
jerry_nba.py — a first NBA model.

Ratings are fitted from completed games in `game_results` (filled by jerry_feeds):
a ridge regression of margin on home/away team indicators plus a home-court
term, and a separate pace rating from points totals. Both are shrunk hard toward
league average, because in October there are eight games of data per team.

This model is deliberately conservative. It writes rows with reasoning for every
tracked game, but emits a non-Sit call only once both teams have MIN_GAMES
played and the disagreement clears the error band. There is no backtest and no
graded history; it begins life as colour.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np

UTC = timezone.utc

MIN_GAMES = 8          # per team before any non-Sit call
HOME_PRIOR = 2.5       # points of home court, prior
RIDGE = 6.0            # shrinkage on team ratings (games-worth of league average)
MARGIN_SD = 12.5       # NBA margin dispersion
TOTAL_SD = 18.0
MAE_SPREAD = 10.0      # model error band priors until measured
MAE_TOTAL = 14.0
LEAGUE_PPG_PRIOR = 114.0


def _norm_team(s):
    import re
    n = re.sub(r"[^a-z ]", "", str(s).lower()).strip()
    return n.split()[-1] if n.split() else n


def fit(con, season_start=None):
    """Return ratings dict {team_key: {off, def, net, games}} plus league/home numbers."""
    rows = con.execute("select home, away, home_score, away_score, game_date from game_results where sport='NBA'").fetchall()
    if season_start:
        rows = [r for r in rows if r["game_date"] >= season_start]
    if not rows:
        return None
    teams = sorted({_norm_team(r["home"]) for r in rows} | {_norm_team(r["away"]) for r in rows})
    ix = {t: i for i, t in enumerate(teams)}
    n, k = len(rows), len(teams)
    X = np.zeros((n, k + 1)); y = np.zeros(n); tot = np.zeros(n)
    for i, r in enumerate(rows):
        X[i, ix[_norm_team(r["home"])]] = 1.0
        X[i, ix[_norm_team(r["away"])]] = -1.0
        X[i, k] = 1.0
        y[i] = r["home_score"] - r["away_score"]
        tot[i] = r["home_score"] + r["away_score"]
    # ridge with a prior on home court; team ratings shrink to zero
    lam = np.full(k + 1, RIDGE); lam[k] = 0.5
    prior = np.zeros(k + 1); prior[k] = HOME_PRIOR
    A = X.T @ X + np.diag(lam)
    b = X.T @ y + lam * prior
    beta = np.linalg.solve(A, b)
    # pace / scoring rating: team's average total vs league, shrunk
    lg_total = float(tot.mean())
    games = {t: 0 for t in teams}; ptot = {t: 0.0 for t in teams}
    for i, r in enumerate(rows):
        for t in (_norm_team(r["home"]), _norm_team(r["away"])):
            games[t] += 1; ptot[t] += tot[i]
    out = {}
    for t in teams:
        g = games[t]
        avg_tot = (ptot[t] + lg_total * RIDGE) / (g + RIDGE)
        out[t] = {"net": float(beta[ix[t]]), "total_tilt": float(avg_tot - lg_total), "games": g}
    return {"teams": out, "home": float(beta[k]), "lg_total": lg_total, "n_games": n}


def predict(fit_, home, away):
    th, ta = fit_["teams"].get(_norm_team(home)), fit_["teams"].get(_norm_team(away))
    if not th or not ta:
        return None
    margin = th["net"] - ta["net"] + fit_["home"]
    total = fit_["lg_total"] + 0.5 * (th["total_tilt"] + ta["total_tilt"])
    return {"margin": margin, "total": total, "games_h": th["games"], "games_a": ta["games"]}


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def refresh(db_path, now=None):
    """Write NBA model rows (spreads, totals, h2h) with reasoning for games in the next 36h."""
    import jerry_reason as jr
    now = now or datetime.now(UTC)
    con = sqlite3.connect(db_path, timeout=60); con.row_factory = sqlite3.Row
    try:
        cols = {r[1] for r in con.execute("pragma table_info(model_probs)")}
        if "reasoning" not in cols:
            con.execute("alter table model_probs add column reasoning text")
        season_start = f"{now.year if now.month >= 9 else now.year - 1}-10-01"
        f = fit(con, season_start)
        if not f:
            return 0
        evs = con.execute("""select event_id, home, away, commence from events where sport='NBA'
                             and commence>? and commence<?""",
                          (now.isoformat(timespec="seconds"), (now + timedelta(hours=36)).isoformat(timespec="seconds"))).fetchall()
        n = 0
        for ev in evs:
            p = predict(f, ev["home"], ev["away"])
            if not p:
                continue
            q = {}
            for r in con.execute("""select market, outcome, line, price from last_quote
                                    where event_id=? and player='' and book='pinnacle'""", (ev["event_id"],)):
                q.setdefault(r["market"], {})[r["outcome"]] = (r["line"], r["price"])
            if not q:
                continue
            gated = min(p["games_h"], p["games_a"]) < MIN_GAMES
            rows = []
            try:
                _sl = q.get("spreads", {}).get(ev["home"], (None, None))[0]
                _tl = q.get("totals", {}).get("Over", (None, None))[0]
                jr.record_numbers(con, event_id=ev["event_id"], sport="NBA", home=ev["home"], away=ev["away"], commence=ev["commence"],
                                  margin=p["margin"], total=p["total"], p_home=_ncdf(p["margin"] / MARGIN_SD),
                                  mkt_margin=None if _sl is None else -float(_sl), mkt_total=None if _tl is None else float(_tl),
                                  checks={"enough_games": not gated},
                                  detail=f"ridge ratings; games {p['games_h']}/{p['games_a']}")
            except Exception:
                pass
            # ---- spread
            sp = q.get("spreads", {})
            if ev["home"] in sp and ev["away"] in sp:
                hl, hp = sp[ev["home"]]; al, ap = sp[ev["away"]]
                fair = jr.power_devig([hp, ap]) or [1 / hp, 1 / ap]
                p_home = _ncdf((p["margin"] + float(hl)) / MARGIN_SD)
                e = p_home - fair[0]
                ratio = jr.edge_ratio(p["margin"] - (-float(hl)), MAE_SPREAD)
                tag, _s, _w = jr.size_from_ratio(ratio, e, cap_lean=True)
                side = ev["home"] if e > 0 else ev["away"]
                call = "Sit" if (gated or tag == "Sit") else f"{tag} {side}"
                comp = {"model_number": round(p["margin"], 2), "market_number": round(-float(hl), 2),
                        "games_h": p["games_h"], "games_a": p["games_a"], "raw_prob": round(p_home, 4)}
                why = jr.render_reasoning(market="spreads", player="", selection=side, line=hl if e > 0 else al,
                                          model_prob=p_home if e > 0 else 1 - p_home, fair_prob=fair[0] if e > 0 else fair[1],
                                          components=comp, opponent="", status="", n_games=None, sport="NBA")
                if gated:
                    why += (f" Gated: only {min(p['games_h'], p['games_a'])} games on record for the thinner team; "
                            f"the model does not make calls below {MIN_GAMES}.")
                why += " NBA model has no backtest and no graded history."
                rows.append((ev["event_id"], "spreads", ev["home"], "", hl, p_home, fair[0], e, call,
                             f"NBA ratings margin {p['margin']:+.1f} vs line {-float(hl):+.1f}; ratio {ratio:.2f}; games {p['games_h']}/{p['games_a']}",
                             now.isoformat(timespec="seconds"), why))
                rows.append((ev["event_id"], "spreads", ev["away"], "", al, 1 - p_home, fair[1], -e, "see home row", "", now.isoformat(timespec="seconds"), None))
            # ---- total
            to = q.get("totals", {})
            if "Over" in to and "Under" in to:
                ol, op = to["Over"]; ul, up = to["Under"]
                fair = jr.power_devig([up, op]) or [1 / up, 1 / op]
                p_under = _ncdf((float(ol) - p["total"]) / TOTAL_SD)
                e = p_under - fair[0]
                ratio = jr.edge_ratio(p["total"] - float(ol), MAE_TOTAL)
                tag, _s, _w = jr.size_from_ratio(ratio, e, cap_lean=True)
                side = "Under" if e > 0 else "Over"
                call = "Sit" if (gated or tag == "Sit") else f"{tag} {side}"
                comp = {"model_number": round(p["total"], 2), "market_number": float(ol), "raw_prob": round(p_under, 4),
                        "games_h": p["games_h"], "games_a": p["games_a"]}
                why = jr.render_reasoning(market="totals", player="", selection=side, line=ol,
                                          model_prob=p_under if e > 0 else 1 - p_under, fair_prob=fair[0] if e > 0 else fair[1],
                                          components=comp, opponent="", status="", n_games=None, sport="NBA")
                if gated:
                    why += f" Gated below {MIN_GAMES} games per team."
                why += " NBA model has no backtest and no graded history."
                rows.append((ev["event_id"], "totals", "Under", "", ul, p_under, fair[0], e, call,
                             f"NBA model total {p['total']:.1f} vs {ol}; ratio {ratio:.2f}", now.isoformat(timespec="seconds"), why))
                rows.append((ev["event_id"], "totals", "Over", "", ol, 1 - p_under, fair[1], -e, "see Under row", "", now.isoformat(timespec="seconds"), None))
            # ---- moneyline (colour only)
            ml = q.get("h2h", {})
            if ev["home"] in ml and ev["away"] in ml:
                _, hp = ml[ev["home"]]; _, ap = ml[ev["away"]]
                fair = jr.power_devig([hp, ap]) or [1 / hp, 1 / ap]
                p_home = _ncdf(p["margin"] / MARGIN_SD)
                e = p_home - fair[0]
                side = ev["home"] if e > 0 else ev["away"]
                call = "Sit" if (gated or abs(e) < 0.04) else f"Lean {side}"
                why = (f"Ratings margin {p['margin']:+.1f} implies {side} {max(p_home, 1 - p_home):.0%} to win versus "
                       f"{max(fair):.0%} at fair. Moneyline is colour only — no backtest, no graded history, and "
                       f"{min(p['games_h'], p['games_a'])} games of data on the thinner side.")
                rows.append((ev["event_id"], "h2h", ev["home"], "", None, p_home, fair[0], e, call,
                             f"NBA ratings margin {p['margin']:+.1f}", now.isoformat(timespec="seconds"), why))
                rows.append((ev["event_id"], "h2h", ev["away"], "", None, 1 - p_home, fair[1], -e, "see home row", "", now.isoformat(timespec="seconds"), None))
            gated_rows = []
            for r_ in rows:
                r_ = list(r_) + [None, None]
                if r_[8].startswith(("Lean", "Play")):
                    try:
                        r_[8], r_[11], _sg = jr.sharp_gate(con, r_[0], r_[1], r_[8], "", r_[11] or "")
                        r_[12] = _sg.get("label") if _sg else None
                        r_[8], r_[13] = jr.finalize_call(con, r_[0], r_[1], r_[8], "", r_[5], r_[6], r_[4], r_[2])
                    except Exception:
                        pass
                gated_rows.append(tuple(r_))
            try:
                _cols = {c[1] for c in con.execute("pragma table_info(model_probs)")}
                for _c in ("flag", "plain"):
                    if _c not in _cols:
                        con.execute(f"alter table model_probs add column {_c} text")
            except Exception:
                pass
            con.executemany("""insert or replace into model_probs(event_id,market,outcome,player,line,model_prob,fair_prob,edge,call,note,ts,reasoning,flag,plain)
                               values(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", gated_rows)
            n += len(rows)
        con.commit()
        return n
    finally:
        con.close()
'''

try:
    jerry_reason = _register("jerry_reason", _REASON_SRC)
    jerry_grade = _register("jerry_grade", _GRADE_SRC)
    jerry_feeds = _register("jerry_feeds", _FEEDS_SRC)
    jerry_nba = _register("jerry_nba", _NBA_SRC)
    jerry_splits = _register("jerry_splits", _SPLITS_SRC)
    jerry_signals = _register("jerry_signals", _SIGNALS_SRC)
    print("[boot] KRONOS: reasoning + grading + feeds + nba + splits + signals modules registered", flush=True)
except Exception:
    traceback.print_exc()
    print("[boot] WARNING reasoning/grading modules failed to load", flush=True)


def backfill_props(db_path, weeks=8):
    """Grade the player props the old grader could never settle.

    Runs once per week-number on start-up, so no manual backfill step is needed.
    Anything whose name will not resolve is quarantined and counted, never
    silently dropped.
    """
    import jerry_grade as _jg
    con = sqlite3.connect(db_path, timeout=60); con.row_factory = sqlite3.Row
    out = []
    try:
        now = datetime.now(UTC)
        season, week = _desk_season_week(now)
        if not con.execute("select v from meta where k='props_week_repair'").fetchone():
            fixed = _jg.repair_misgraded_props(con, now.isoformat(timespec="seconds"))
            con.execute("insert or replace into meta(k,v) values(?,?)",
                        ("props_week_repair", f"{now.isoformat(timespec='seconds')} reset {fixed}"))
            con.execute("delete from meta where k like 'backfill_props_%'")
            con.commit()
            print(f"[boot] reset {fixed} props mis-graded without a week filter", flush=True)
        for w in range(max(1, week - weeks + 1), week + 1):
            key = f"backfill_props_{season}_{w}"
            seen = con.execute("select v from meta where k=?", (key,)).fetchone()
            if seen and w != week:          # always re-run the current week
                continue
            try:
                rep = _jg.grade_nfl_props(con, season, w, now.isoformat(timespec="seconds"))
            except Exception as e:          # noqa: BLE001
                rep = {"error": str(e)[:160]}
            out.append({"week": w, **rep})
            con.execute("insert or replace into meta(k,v) values(?,?)", (key, json.dumps(rep)[:400]))
        con.commit()
    finally:
        con.close()
    return out


def _desk_season_week(now):
    y = now.year if now.month >= 3 else now.year - 1
    sept = datetime(y, 9, 1, tzinfo=UTC)
    labor = sept + timedelta(days=(7 - sept.weekday()) % 7)
    kickoff = labor + timedelta(days=3)
    return y, max(1, min(22, ((now - kickoff).days // 7) + 1))



# ---------- jerry.py's schedule URL 404s (nflverse retired that release asset).
# _cached_csv has no fallback, so a failed refresh raises and the NFL model runs
# on whatever stale copy is already cached. Wrap it with alternates.
_JM_FALLBACKS = {
    "schedules/games.csv": [
        "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
        "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv",
    ],
}

def _install_nflverse_fallbacks():
    import jerry as _JM
    if getattr(_JM, "_fallbacks_installed", False):
        return
    _orig = _JM._cached_csv

    def _cached_csv(url, name, refresh=False):
        try:
            return _orig(url, name, refresh)
        except Exception as first:
            alts = []
            for key, urls in _JM_FALLBACKS.items():
                if url.endswith(key):
                    alts = [u for u in urls if u != url]
            for alt in alts:
                try:
                    df = _orig(alt, name, refresh)
                    print(f"[boot] nflverse fallback used for {name}: {alt}", flush=True)
                    return df
                except Exception:
                    continue
            try:                      # last resort: whatever is already on disk
                p = _JM.CACHE / name
                if p.exists():
                    print(f"[boot] nflverse unreachable for {name}, using cached copy", flush=True)
                    return _JM.pd.read_csv(p, low_memory=False)
            except Exception:
                pass
            raise first

    _JM._cached_csv = _cached_csv
    _JM._fallbacks_installed = True
    print("[boot] nflverse URL fallbacks installed", flush=True)

try:
    _install_nflverse_fallbacks()
except Exception:
    traceback.print_exc()

# ---------- hook into the desk and start it
def _import_live():
    """jerry.py creates its cache folder at import time; make any missing data folders first."""
    for _ in range(5):
        try:
            import jerry_live
            return jerry_live
        except FileNotFoundError as e:
            if not e.filename:
                raise
            os.makedirs(e.filename, exist_ok=True)
    import jerry_live
    return jerry_live

try:
    L = _import_live()
except Exception:
    traceback.print_exc()
    L = None

def recompute_clv_once(db_path):
    """One-time repair (2026-10-26): every stored clv_pp was inflated ~1pp because the bet
    price was de-vigged. Recompute all rows with the fixed formula and a two-sided close."""
    import jerry_reason as _jr
    c = sqlite3.connect(db_path, timeout=60); c.row_factory = sqlite3.Row
    try:
        if c.execute("select 1 from meta where k='clv_fix_2026_10_26'").fetchone():
            return "already done"
        rows = c.execute("""select id, event_id, market, selection, player, line, price, commence, close_book,
                            close_line, close_price, comp_json from bets where close_price is not null""").fetchall()
        n = 0
        for b in rows:
            fair = None
            try:
                ol = (-b["close_line"] if (b["market"] == "spreads" and b["close_line"] is not None) else b["close_line"])
                r = c.execute("""select price from ticks where event_id=? and market=? and player=? and book=? and outcome<>?
                                 and line is ? and ts<=? order by ts desc limit 1""",
                              (b["event_id"], b["market"], b["player"] or "", b["close_book"], b["selection"], ol,
                               b["commence"])).fetchone()
                if r and r["price"]:
                    fair = _jr.fair_two_way_power(float(b["close_price"]), float(r["price"]))
            except Exception:
                fair = None
            try:
                comp = json.loads(b["comp_json"] or "{}")
            except Exception:
                comp = {}
            v = _jr.clv_pp(b["line"], b["price"], b["close_line"], b["close_price"], b["market"], b["selection"],
                           sd=comp.get("sd"), mu=comp.get("mu"), close_fair=fair)
            c.execute("update bets set clv_pp=? where id=?", (v, b["id"])); n += 1
        c.execute("insert or replace into meta(k,v) values('clv_fix_2026_10_26', ?)", (f"{datetime.now(UTC).isoformat()} {n} rows",))
        c.commit()
        return f"{n} rows recomputed"
    finally:
        c.close()


_orig_refresh_all = L.refresh_all if L else None
def refresh_all(db_path, sports=None):
    try:
        print(f"[boot] CLV repair: {recompute_clv_once(db_path)}", flush=True)
    except Exception as e:   # noqa: BLE001
        print(f"[boot] CLV repair failed: {e}", flush=True)
    _spl = None
    try:
        import jerry_splits as _js
        _c = sqlite3.connect(db_path, timeout=60); _c.row_factory = sqlite3.Row
        try:
            _spl = _js.run(_c)
        finally:
            _c.close()
    except Exception as e:   # noqa: BLE001 - splits must never block a refresh
        _spl = f"error: {e}"
    res = _orig_refresh_all(db_path) if sports is None else _orig_refresh_all(db_path, sports=tuple(s for s in sports if s != "MLB"))
    res["SPLITS"] = _spl
    # learning layer: measured signal weights, calibration, autopsy, timing, contradictions,
    # plus soccer grading, DK public-betting-trends splits and nflverse snaps/injuries/matchups
    try:
        import jerry_signals as _jsig
        _c = sqlite3.connect(db_path, timeout=60); _c.row_factory = sqlite3.Row
        try:
            res["LEARN"] = _jsig.run(_c, datetime.now(UTC))
        finally:
            _c.close()
    except Exception as e:   # noqa: BLE001 - never block a refresh
        res["LEARN"] = f"error: {e}"
    if sports is None or "MLB" in sports:
        try:
            res["MLB"] = refresh_mlb(db_path)
            try:
                c = sqlite3.connect(db_path, timeout=60); c.execute("delete from meta where k='error_model_MLB'"); c.commit(); c.close()
            except Exception:
                pass
        except Exception as e:
            res["MLB"] = f"error: {e}"
            traceback.print_exc()
            try:
                c = sqlite3.connect(db_path, timeout=60)
                c.execute("insert or replace into meta(k,v) values(?,?)", ("error_model_MLB", f"{datetime.now(UTC).isoformat()} {e}"[:200]))
                c.commit(); c.close()
            except Exception:
                pass
    try:
        if hasattr(L, "health_check"):
            L.health_check(db_path, res)
    except Exception:
        traceback.print_exc()
    try:
        res["GRADE"] = backfill_props(db_path)
    except Exception as e:   # noqa: BLE001 - grading must never stop a refresh
        res["GRADE"] = f"error: {e}"
        traceback.print_exc()
    # NBA/NHL box scores: grade their props, fill the roster cache, store results
    try:
        import jerry_feeds as _jf
        _c = sqlite3.connect(db_path, timeout=60); _c.row_factory = sqlite3.Row
        try:
            res["FEEDS"] = _jf.run(_c, datetime.now(UTC))
        finally:
            _c.close()
    except Exception as e:   # noqa: BLE001
        res["FEEDS"] = f"error: {e}"
    # NBA model, from the results the feed just stored
    try:
        import jerry_nba as _jn
        res["NBA"] = _jn.refresh(db_path)
    except Exception as e:   # noqa: BLE001
        res["NBA"] = f"error: {e}"
        traceback.print_exc()
    return res
if L:
    L.refresh_all = refresh_all
    print("[boot] fixes active: NHL icetime alias, MLB model, reasoning layer, prop grading", flush=True)
else:
    print("[boot] fixes NOT active, starting desk unchanged", flush=True)


# ---------- feed: push the desk summary to a public GitHub repo every 10 minutes, so Claude can read it fresh
import base64, html as _html, re, threading, time
FEED_REPO = os.environ.get("FEED_REPO", "choomah-gif/jerry-feed")
FEED_FILE = "summary.txt"
FEED_EVERY = 600

def _summary_text():
    port = os.environ.get("PORT", "8000")
    page = requests.get(f"http://127.0.0.1:{port}/", timeout=30).text
    for block in re.findall(r"<pre[^>]*>(.*?)</pre>", page, flags=re.S):
        txt = _html.unescape(re.sub(r"<[^>]+>", "", block))
        if "KRONOS DESK SUMMARY" in txt or "JERRY DESK SUMMARY" in txt:
            return txt
    raise RuntimeError("summary block not found on desk page")

def _put(path, txt, token, msg):
    api = f"https://api.github.com/repos/{FEED_REPO}/contents/{path}"
    h = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    cur = requests.get(api, headers=h, timeout=20)
    sha = cur.json().get("sha") if cur.status_code == 200 else None
    body = {"message": msg, "content": base64.b64encode(txt.encode()).decode()}
    if sha:
        body["sha"] = sha
    requests.put(api, headers=h, json=body, timeout=30).raise_for_status()

def push_source():
    token = os.environ.get("FEED_TOKEN")
    if not token:
        return "no FEED_TOKEN set"
    here = os.path.dirname(os.path.abspath(__file__))
    done = []
    for f in ("jerry.py", "jerry_live.py", "jerry_desk.py", "boot.py", "JERRY_PROMPT.md", "JERRY PROMPT.md"):
        fp = os.path.join(here, f)
        if os.path.exists(fp):
            _put(f"src/{f}", open(fp, encoding="utf-8").read(), token, f"source {f}")
            done.append(f)
    return f"source published: {done}"


def _quarantine_block():
    """Append the unresolved player names to the pushed summary.

    Quarantined rows are unsettled, not wrong — but they are invisible unless the
    names are published, and invisible is how the ledger got stuck the first time.
    """
    try:
        db = os.path.join(os.environ.get("JERRY_DATA_DIR", "/data"), "desk.db")
        con = sqlite3.connect(db, timeout=30); con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(
            """select player, market, reason, count(*) n from grade_quarantine
                group by player, market, reason order by n desc limit 60""")]
        con.close()
    except Exception as e:   # noqa: BLE001
        return f"\n\nQUARANTINE: unavailable ({e})\n"
    if not rows:
        return "\n\nQUARANTINE: none\n"
    out = ["", "", f"QUARANTINE ({len(rows)} unresolved player/market pairs; these bets are unsettled, not lost):"]
    for r in rows:
        out.append(f"  {r['player']} | {r['market']} | {r['reason']}" + (f" x{r['n']}" if r["n"] > 1 else ""))
    return "\n".join(out) + "\n"



# ================================================================= every-game signal check (2026-10-29)
# Owner's rule: the 19 KRONOS signals run on EVERY tracked game, not only on price edges.
# Before this, jerry_signals.capture() ran only inside the price-edge scan, so game sides and model
# calls carried no signals, the learning layer had nothing to learn from ("graded rows with signals: 0"),
# and the feed showed no per-game metrics. This block:
#   1. computes all 19 signals for both sides of every game line (moneyline, spread, total) on every
#      tracked game starting in the next 36 hours, at every feed push (10 min);
#   2. stores the latest values in game_signals, and freezes a pick-time copy 90 minutes before start;
#   3. back-fills signals onto every pending paper bet that has none, so the learning layer can grade them;
#   4. publishes the whole check in the feed as GAME SIGNALS, with KRONOS's projection, the de-vigged
#      fair price, the best Connecticut price, the sharp/public read and the contradiction check.
# Reporting and data capture only: no pick, price, stake or model weight is changed here.
GS_HORIZON_H = 36
GS_FREEZE_MIN = 90
GS_FAIR_BOOKS = ("pinnacle", "lowvig", "betonlineag")
GS_SD = {("NFL", "margin"): 13.0, ("NFL", "total"): 13.5, ("NBA", "margin"): 12.0, ("NBA", "total"): 17.0}


def _gs_imp(p):
    try:
        p = float(p)
        return 1.0 / p if p > 1 else None
    except (TypeError, ValueError):
        return None


def _gs_power_fair(prices):
    """Power de-vig across all outcomes of one market. Returns list of fair probs or None."""
    imps = [_gs_imp(p) for p in prices]
    if not imps or any(i is None for i in imps):
        return None
    lo, hi = 0.5, 3.0
    for _ in range(80):
        k = (lo + hi) / 2
        s = sum(i ** k for i in imps)
        if s > 1:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    return [i ** k for i in imps]


def _gs_ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _gs_poisson_pmf(lam, k):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _gs_sim_p(sport, market, outcome, line, home, nums):
    """Probability of this outcome from KRONOS's own projected score.
    NHL: independent Poisson goals (ties go to OT/shootout as a coin flip for the moneyline).
    NFL/NBA: normal around KRONOS's margin and total (approximation; NFL's own cover % uses the
    empirical key-number distribution inside the model)."""
    if not nums or nums["home_pts"] is None or nums["away_pts"] is None:
        return None
    hp, ap = float(nums["home_pts"]), float(nums["away_pts"])
    is_home = outcome == home
    if sport == "NHL":
        ph = [_gs_poisson_pmf(hp, k) for k in range(16)]
        pa = [_gs_poisson_pmf(ap, k) for k in range(16)]
        def p_margin(cond):
            return sum(ph[i] * pa[j] for i in range(16) for j in range(16) if cond(i - j, i + j))
        if market == "h2h":
            ph_win = p_margin(lambda m, t: m > 0) + 0.5 * p_margin(lambda m, t: m == 0)
            return ph_win if is_home else 1 - ph_win
        if market == "totals" and line is not None:
            over = p_margin(lambda m, t: t > float(line))
            return over if outcome == "Over" else p_margin(lambda m, t: t < float(line))
        if market == "spreads" and line is not None:
            L = float(line)
            return p_margin((lambda m, t: m + L > 0) if is_home else (lambda m, t: -m + L > 0))
        return None
    sdm, sdt = GS_SD.get((sport, "margin")), GS_SD.get((sport, "total"))
    if not sdm:
        return None
    m = hp - ap
    if market == "h2h":
        p = 1 - _gs_ncdf((0 - m) / sdm)
        return p if is_home else 1 - p
    if market == "spreads" and line is not None:
        L = float(line)
        p = 1 - _gs_ncdf((-L - m) / sdm) if is_home else _gs_ncdf((L - m) / sdm)
        return p
    if market == "totals" and line is not None:
        p_over = 1 - _gs_ncdf((float(line) - (hp + ap)) / sdt)
        return p_over if outcome == "Over" else 1 - p_over
    return None


def _gs_ensure(con):
    con.execute("""create table if not exists game_signals(event_id text, market text, outcome text, line real,
                   price real, fair real, best_book text, best_price real, model_prob real, sim_p real,
                   sharp text, signals text, ts text, primary key(event_id, market, outcome))""")
    con.execute("""create table if not exists game_signals_pick(event_id text, market text, outcome text, line real,
                   price real, fair real, best_book text, best_price real, model_prob real, sim_p real,
                   sharp text, signals text, ts text, primary key(event_id, market, outcome))""")
    cols = {r[1] for r in con.execute("pragma table_info(bets)")}
    if "signals" not in cols:
        con.execute("alter table bets add column signals text")


def _gs_fmt(k, v):
    if v is None:
        return f"{k} n/a"
    if k in ("drop_pct", "edge"):
        return f"{k} {v * 100:+.1f}%"
    if k in ("proj_p", "sim_p"):
        return f"{k} {v * 100:.1f}%"
    if k in ("shade_pp", "dk_gap"):
        return f"{k} {v:+.1f}"
    if isinstance(v, float):
        return f"{k} {v:+.2f}" if k in ("steam", "tape_cents") else f"{k} {v:g}"
    return f"{k} {v}"


def game_signals(db_path=None, now=None):
    """Compute, store and render the 19-signal check for every tracked game. Returns feed text."""
    import jerry_signals as _js
    import jerry_reason as _jr
    now = now or datetime.now(UTC)
    db_path = db_path or os.path.join(os.environ.get("JERRY_DATA_DIR", "/data"), "desk.db")
    con = sqlite3.connect(db_path, timeout=60); con.row_factory = sqlite3.Row
    out, n_sides, n_backfill, errors = [], 0, 0, []
    try:
        _gs_ensure(con)
        model = None
        try:
            model = _js.load_model(con)
        except Exception:                                             # noqa: BLE001
            model = None
        live_w = {k: v for k, v in ((model or {}).get("weights") or {}).items() if v}
        hi = (now + timedelta(hours=GS_HORIZON_H)).isoformat(timespec="seconds")
        lo = now.isoformat(timespec="seconds")
        evs = [dict(r) for r in con.execute("""select * from events where commence>? and commence<?
                                                order by commence, sport""", (lo, hi))]
        try:
            contra = json.loads(con.execute("select v from meta where k='contradictions'").fetchone()[0] or "[]")
        except Exception:                                             # noqa: BLE001
            contra = []
        for ev in evs:
            eid, sport = ev["event_id"], ev["sport"]
            nums = con.execute("select * from jerry_numbers where event_id=?", (eid,)).fetchone()
            game_lines = []
            hours = None
            try:
                hours = (datetime.fromisoformat(ev["commence"].replace("Z", "+00:00")) - now).total_seconds() / 3600
            except Exception:                                         # noqa: BLE001
                pass
            for market in ("h2h", "spreads", "totals"):
                ref = None
                for bk in GS_FAIR_BOOKS:
                    q = [dict(r) for r in con.execute("""select outcome, line, price from last_quote where event_id=? and book=?
                                                         and market=? and coalesce(player,'')='' order by outcome""",
                                                      (eid, bk, market))]
                    if len(q) >= 2:
                        ref = (bk, q)
                        break
                if not ref:
                    continue
                bk, q = ref
                fairs = _gs_power_fair([r["price"] for r in q])
                if not fairs:
                    continue
                n_books = con.execute("""select count(distinct book) from last_quote where event_id=? and market=?
                                         and coalesce(player,'')=''""", (eid, market)).fetchone()[0]
                for r, fair in zip(q, fairs):
                    outcome, line, price = r["outcome"], r["line"], r["price"]
                    best = con.execute(f"""select book, price from last_quote where event_id=? and market=? and outcome=?
                                           and coalesce(player,'')='' and coalesce(line,-999)=coalesce(?,-999)
                                           and book in ({",".join("?" * len(BET_BOOKS_GS))}) order by price desc limit 1""",
                                       (eid, market, outcome, line, *BET_BOOKS_GS)).fetchone()
                    mp = con.execute("""select model_prob, call from model_probs where event_id=? and market=? and outcome=?
                                        and coalesce(player,'')=''""", (eid, market, outcome)).fetchone()
                    model_prob = mp["model_prob"] if mp else None
                    if model_prob is None and market == "h2h" and nums is not None and nums["p_home"] is not None:
                        model_prob = nums["p_home"] if outcome == ev["home"] else (1 - nums["p_home"] if len(q) == 2 else None)
                    sim_p = None
                    try:
                        sim_p = _gs_sim_p(sport, market, outcome, line, ev["home"], nums)
                    except Exception:                                 # noqa: BLE001
                        sim_p = None
                    extra = {"n_books": n_books, "sim_p": round(sim_p, 4) if sim_p is not None else None}
                    try:
                        extra["sig"] = _jr.sharp_signal(con, eid, market, outcome, "")
                    except Exception as e:                            # noqa: BLE001
                        extra["sig"] = {}
                        errors.append(f"sharp {str(e)[:60]}")
                    if model_prob is not None:
                        extra["pc"] = {"p_side": round(float(model_prob), 4)}
                    if sport == "NFL":
                        try:
                            extra["weather"] = _js.weather_for(con, eid, ev["home"], ev["commence"], now)
                        except Exception:                             # noqa: BLE001
                            extra["weather"] = {}
                    try:
                        # edge is measured at the best price you can actually bet (CT books), else the sharp price
                        s = _js.capture(con, ev, market, outcome, "", line, (best["price"] if best else price), fair, extra)
                    except Exception as e:                            # noqa: BLE001
                        errors.append(f"capture {str(e)[:60]}")
                        continue
                    s["hours"] = round(hours, 2) if hours is not None else s.get("hours")
                    adj = None
                    if live_w:
                        try:
                            adj = _js.adjust(fair, s, model)
                        except Exception:                             # noqa: BLE001
                            adj = None
                    verdict = (extra.get("sig") or {}).get("verdict", "neutral")
                    row = (eid, market, outcome, line, price, round(fair, 4), best["book"] if best else None,
                           best["price"] if best else None, model_prob, sim_p, verdict, json.dumps(s),
                           now.isoformat(timespec="seconds"))
                    con.execute("insert or replace into game_signals values(?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
                    if hours is not None and hours * 60 <= GS_FREEZE_MIN and not con.execute(
                            "select 1 from game_signals_pick where event_id=? and market=? and outcome=?",
                            (eid, market, outcome)).fetchone():
                        con.execute("insert into game_signals_pick values(?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
                    n_sides += 1
                    game_lines.append((market, outcome, line, price, fair, best, model_prob, sim_p, verdict, s, adj,
                                       mp["call"] if mp else None))
            if not game_lines:
                continue
            head = f"{sport} {ev['commence'][:16]}Z {ev['away']} @ {ev['home']}"
            if nums is not None and nums["home_pts"] is not None:
                head += (f" | KRONOS {ev['away']} {float(nums['away_pts']):.1f} @ {ev['home']} {float(nums['home_pts']):.1f}"
                         + (f" | market margin {nums['mkt_margin']:+.1f} total {nums['mkt_total']:.1f}"
                            if nums["mkt_margin"] is not None and nums["mkt_total"] is not None else ""))
            else:
                head += " | KRONOS: no projection for this game"
            out.append(head)
            for (market, outcome, line, price, fair, best, model_prob, sim_p, verdict, s, adj, call) in game_lines:
                ln = f" {line:+g}" if (market == "spreads" and line is not None) else (f" {line:g}" if line is not None else "")
                bp = f"{best['book']} {best['price']:g} (EV {(best['price'] * fair - 1) * 100:+.1f}%)" if best else "no CT price"
                mtxt = f"KRONOS {model_prob * 100:.1f}%" if model_prob is not None else "KRONOS n/a"
                if call:
                    mtxt += f" [{call}]"
                parts = [f"  {market} {outcome}{ln} | sharp book {price:g} fair {fair * 100:.1f}% | {mtxt} | "
                         f"sim (raw projection) {sim_p * 100:.1f}%" if sim_p is not None else
                         f"  {market} {outcome}{ln} | sharp book {price:g} fair {fair * 100:.1f}% | {mtxt} | sim n/a"]
                parts.append(f"best CT {bp} | money [{verdict.upper()}]")
                if adj is not None and abs(adj - fair) > 1e-6:
                    parts.append(f"LEARNED {(adj - fair) * 100:+.1f}pp")
                out.append(" | ".join(parts))
                out.append("      signals: " + ", ".join(_gs_fmt(k, s.get(k)) for k in _js.SIGNALS))
            g = f"{ev['away']} @ {ev['home']}"
            hits = [c for c in contra if c.startswith(g) or g in c]
            out.append("      contradiction check: " + ("; ".join(hits) if hits else "clear"))
        # signals onto every pending paper bet that has none (the learning layer grades these)
        cols = {r[1] for r in con.execute("pragma table_info(bets)")}
        fair_col = "fair_prob" if "fair_prob" in cols else None
        for b in con.execute(f"""select b.id, b.event_id, b.market, b.selection, b.player, b.line, b.price, b.model_prob
                                 {', b.' + fair_col if fair_col else ''}
                                 from bets b join events e on e.event_id=b.event_id
                                 where b.paper=1 and b.result='pending' and b.signals is null and e.commence>?""",
                             (lo,)).fetchall():
            ev = dict(con.execute("select * from events where event_id=?", (b["event_id"],)).fetchone())
            f = (b[fair_col] if fair_col else None) or b["model_prob"]
            extra = {}
            try:
                extra["sig"] = _jr.sharp_signal(con, b["event_id"], b["market"], b["selection"], b["player"] or "")
            except Exception:                                         # noqa: BLE001
                extra["sig"] = {}
            if b["model_prob"] is not None:
                extra["pc"] = {"p_side": b["model_prob"]}
            try:
                s = _js.capture(con, ev, b["market"], b["selection"], b["player"] or "", b["line"], b["price"], f, extra)
                con.execute("update bets set signals=? where id=?", (json.dumps(s), b["id"]))
                n_backfill += 1
            except Exception as e:                                    # noqa: BLE001
                errors.append(f"backfill {str(e)[:60]}")
        con.commit()
    finally:
        con.close()
    hdr = ["", "", f"GAME SIGNALS (all {19} KRONOS signals on both sides of every game line, every tracked game in the next "
           f"{GS_HORIZON_H}h; {n_sides} sides; computed {now.isoformat(timespec='seconds')}; signals still at weight 0 "
           f"until graduated, so they inform but do not move prices; pick-time copy frozen {GS_FREEZE_MIN} min before start; "
           f"{n_backfill} pending paper bets given signals this pass" + (f"; errors: {'; '.join(errors[:3])}" if errors else "") + "):"]
    return "\n".join(hdr + out) + "\n"


BET_BOOKS_GS = ("draftkings", "fanduel", "fanatics")


def _game_signals_block():
    try:
        return game_signals()
    except Exception as e:                                            # noqa: BLE001
        return f"\n\nGAME SIGNALS: unavailable ({str(e)[:200]})\n"


def push_feed():
    token = os.environ.get("FEED_TOKEN")
    if not token:
        return "no FEED_TOKEN set"
    txt = _summary_text() + _game_signals_block() + _quarantine_block()
    api = f"https://api.github.com/repos/{FEED_REPO}/contents/{FEED_FILE}"
    h = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    cur = requests.get(api, headers=h, timeout=20)
    sha = cur.json().get("sha") if cur.status_code == 200 else None
    body = {"message": f"summary {datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}Z",
            "content": base64.b64encode(txt.encode()).decode()}
    if sha:
        body["sha"] = sha
    r = requests.put(api, headers=h, json=body, timeout=30)
    r.raise_for_status()
    return f"pushed {len(txt)} chars"

def _feed_loop():
    time.sleep(150)  # let the desk start and run its first model refresh
    try:
        print(f"[feed] {push_source()}", flush=True)
    except Exception as e:
        print(f"[feed] source publish failed: {e}", flush=True)
    while True:
        try:
            print(f"[feed] {push_feed()}", flush=True)
        except Exception as e:
            print(f"[feed] failed: {e}", flush=True)
        time.sleep(FEED_EVERY)

threading.Thread(target=_feed_loop, daemon=True).start()

if __name__ == "__main__":
    sys.argv = ["jerry_desk.py"] + sys.argv[1:]
    runpy.run_path("jerry_desk.py", run_name="__main__")
