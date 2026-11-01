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
    # Only grade props whose game is over (kickoff + 4h). The week's stat file fills in game by game:
    # on Sunday morning it holds Thursday's game only, and every Sunday player looked like a scratch
    # and was voided (2026-11-01: 169 false voids). Unplayed games stay pending.
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    _UTC = _tz.utc
    _cut = _dt.now(_UTC) - _td(hours=4)
    def _done(c):
        try:
            t = _dt.fromisoformat(str(c).replace("Z", "+00:00"))
            return (t if t.tzinfo else t.replace(tzinfo=_UTC)) <= _cut
        except Exception:                                   # noqa: BLE001
            return False
    pend = [b for b in pend if _done(b["commence"])]
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

_SOCCER_SRC = r'''"""
kronos_soccer — KRONOS soccer model for EPL and UCL (soccer-v1, 2026-11-01). Paper only, capped at Lean.

Data (free): football-data.co.uk season files for the Premier League, Championship and nine other
European leagues (results, shots on target, corners), plus ESPN day-by-day results for the three UEFA
competitions and the latest English results. Europe and promotion link the leagues together.

Model:
  * Poisson team ratings: log(goals) = base + home + league prior + team attack − (league prior + team defence)
    fitted with time decay (half-life 150 days) and ridge shrinkage (teams toward their league,
    leagues toward zero). The target blends goals with shots on target x the conversion rate, because
    goals alone are too noisy over a few months.
  * Dixon-Coles low-score correction (draws are under-predicted by plain Poisson).
  * Matchup priors, logged in the reasoning, not fitted yet: short rest (<=3 days since last match in any
    competition), set-piece edge (corners won vs corners conceded), both capped small.
  * Anchored to Pinnacle: 35% model / 65% market for EPL, 30/70 for UCL, until the ledger shows the model
    beats the market. Calls are capped at Lean while unvalidated.
Outputs: model_probs rows (h2h home/draw/away, totals at Pinnacle's line, Asian handicap at Pinnacle's
line), Jerry's spin numbers (expected goals each side), paper rows via the normal path.
"""
import json, math, os, re, unicodedata
from datetime import datetime, timezone, timedelta, date

UTC = timezone.utc
BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
DOMESTIC = ["eng.1", "eng.2", "esp.1", "ger.1", "ita.1", "fra.1", "por.1", "ned.1", "bel.1", "tur.1", "sco.1"]
EUROPE = ["uefa.champions", "uefa.europa", "uefa.europa.conf"]
CUPS = ["eng.fa", "eng.league_cup"]
LEAGUES = DOMESTIC + EUROPE + CUPS
SPORT_LEAGUE = {"EPL": "eng.1", "UCL": "uefa.champions"}
BACKFILL_FROM = date(2025, 7, 1)
HALF_LIFE_D = 150.0
MODEL_WEIGHT = {"EPL": 0.35, "UCL": 0.30}
RHO = -0.08                      # Dixon-Coles
SOT_BLEND = 0.4                  # weight on shots-on-target x conversion in the fitting target
REST_DAYS = 3
REST_ATT, REST_DEF = 0.96, 1.04  # short-rest prior: score 4% less, concede 4% more
CORNER_CAP = 0.06                # set-piece prior: at most +/-6% on expected goals
LEAN_EDGE = 0.02
VERSION = "soccer-v1-2026-11-01"
ALIASES = {"inter milan": "internazionale", "inter": "internazionale", "sporting lisbon": "sporting cp",
           "sporting": "sporting cp", "atletico madrid": "atletico madrid", "psg": "paris saint germain",
           "bayern munchen": "bayern munich", "man city": "manchester city", "man utd": "manchester united",
           "man united": "manchester united", "spurs": "tottenham hotspur", "wolves": "wolverhampton wanderers",
           "nottm forest": "nottingham forest", "brighton": "brighton and hove albion",
           "ac milan": "ac milan", "milan": "ac milan", "fc copenhagen": "fc copenhagen", "kobenhavn": "fc copenhagen",
           "red star belgrade": "crvena zvezda", "slavia praha": "slavia prague", "qarabag fk": "qarabag"}
STOP = {"fc", "afc", "cf", "sc", "the", "and", "de", "club", "calcio", "ac", "as", "ssc", "sv", "fk", "sk", "1", "04", "cp"}


# --------------------------------------------------------------------------- names
def norm(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ").replace("-", " ").replace(".", " ").replace("'", "")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return ALIASES.get(s, s)


def _tokens(s):
    t = set(norm(s).split())
    core = t - STOP
    return core or t


def name_score(a, b):
    na, nb = norm(a), norm(b)
    if na == nb:
        return 1.0
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    j = len(ta & tb) / len(ta | tb)
    if ta <= tb or tb <= ta:
        j = max(j, 0.75)
    return j


def best_name(name, candidates, min_score=0.5):
    best, sc = None, 0.0
    for c in candidates:
        s = name_score(name, c)
        if s > sc:
            best, sc = c, s
    return (best, sc) if sc >= min_score else (None, sc)


# --------------------------------------------------------------------------- data
def _get(url, timeout=25):
    import requests
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (KRONOS desk)"})
    r.raise_for_status()
    return r.json()


def ensure(con):
    con.executescript("""
    create table if not exists soccer_matches(source_id text primary key, league text, game_date text, home text,
        away text, hg real, ag real, hsot real, asot real, hsh real, ash real, hc real, ac real, neutral int,
        detail text, ts text);
    create index if not exists soccer_matches_d on soccer_matches(game_date);
    create table if not exists soccer_fetch(league text, month text, ts text, n int, primary key(league, month));
    """)


def _stat(stats, *names):
    for s in stats or []:
        if str(s.get("name", "")).lower() in names or str(s.get("abbreviation", "")).lower() in names:
            try:
                v = s.get("value") if s.get("value") is not None else s.get("displayValue")
                return float(str(v).replace("%", ""))
            except (TypeError, ValueError):
                return None
    return None


def parse_scoreboard(js, league):
    out = []
    for ev in js.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        st = (comp.get("status") or ev.get("status") or {}).get("type", {})
        if not st.get("completed"):
            continue
        teams = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        if "home" not in teams or "away" not in teams:
            continue
        h, a = teams["home"], teams["away"]
        try:
            hg, ag = float(h.get("score")), float(a.get("score"))
        except (TypeError, ValueError):
            continue
        hs, as_ = h.get("statistics") or [], a.get("statistics") or []
        out.append(dict(source_id=str(ev.get("id")), league=league, game_date=str(ev.get("date", ""))[:10],
                        home=(h.get("team") or {}).get("displayName", ""), away=(a.get("team") or {}).get("displayName", ""),
                        hg=hg, ag=ag, hsot=_stat(hs, "shotsontarget", "sot"), asot=_stat(as_, "shotsontarget", "sot"),
                        hsh=_stat(hs, "totalshots", "shots"), ash=_stat(as_, "totalshots", "shots"),
                        hc=_stat(hs, "woncorners", "corners", "cornerkicks"), ac=_stat(as_, "woncorners", "corners", "cornerkicks"),
                        neutral=1 if comp.get("neutralSite") else 0, detail=str(st.get("detail") or st.get("shortDetail") or "")))
    return out


# Data sources (2026-11-01 v2): ESPN's soccer scoreboard rejects date ranges (400 Bad Request) and only
# answers single days, so league history now comes from football-data.co.uk season files (one request per
# league-season, with shots on target and corners), and ESPN is used day by day only where it is needed:
# UEFA match days (Tue/Wed/Thu) and the last few days of the English leagues and Europe.
FD_CODES = {"E0": "eng.1", "E1": "eng.2", "SP1": "esp.1", "D1": "ger.1", "I1": "ita.1", "F1": "fra.1",
            "P1": "por.1", "N1": "ned.1", "B1": "bel.1", "T1": "tur.1", "SC0": "sco.1"}
FD_SEASONS = ("2526", "2627")
CURRENT_FD = "2627"
FD_ALIAS = {   # football-data short names -> the full names ESPN and the Odds API use
    "man city": "Manchester City", "man united": "Manchester United", "nottm forest": "Nottingham Forest",
    "newcastle": "Newcastle United", "wolves": "Wolverhampton Wanderers", "tottenham": "Tottenham Hotspur",
    "west ham": "West Ham United", "brighton": "Brighton & Hove Albion", "leeds": "Leeds United",
    "leicester": "Leicester City", "coventry": "Coventry City", "hull": "Hull City", "ipswich": "Ipswich Town",
    "luton": "Luton Town", "norwich": "Norwich City", "stoke": "Stoke City", "swansea": "Swansea City",
    "cardiff": "Cardiff City", "sheffield weds": "Sheffield Wednesday", "west brom": "West Bromwich Albion",
    "qpr": "Queens Park Rangers", "derby": "Derby County", "oxford": "Oxford United", "plymouth": "Plymouth Argyle",
    "preston": "Preston North End", "blackburn": "Blackburn Rovers", "birmingham": "Birmingham City",
    "charlton": "Charlton Athletic", "wrexham": "Wrexham", "bournemouth": "AFC Bournemouth",
    "ath madrid": "Atletico Madrid", "ath bilbao": "Athletic Club", "betis": "Real Betis", "sociedad": "Real Sociedad",
    "celta": "Celta Vigo", "espanol": "Espanyol", "vallecano": "Rayo Vallecano", "sevilla": "Sevilla",
    "bayern munich": "Bayern Munich", "dortmund": "Borussia Dortmund", "leverkusen": "Bayer Leverkusen",
    "m gladbach": "Borussia Monchengladbach", "mgladbach": "Borussia Monchengladbach", "ein frankfurt": "Eintracht Frankfurt",
    "rb leipzig": "RB Leipzig", "stuttgart": "VfB Stuttgart", "hoffenheim": "TSG Hoffenheim", "wolfsburg": "VfL Wolfsburg",
    "freiburg": "SC Freiburg", "inter": "Internazionale", "milan": "AC Milan", "roma": "AS Roma", "napoli": "Napoli",
    "juventus": "Juventus", "atalanta": "Atalanta", "lazio": "Lazio", "paris sg": "Paris Saint-Germain",
    "marseille": "Marseille", "lyon": "Lyon", "monaco": "AS Monaco", "lille": "Lille", "nice": "Nice", "lens": "Lens",
    "sp lisbon": "Sporting CP", "benfica": "Benfica", "porto": "FC Porto", "sp braga": "SC Braga",
    "psv eindhoven": "PSV Eindhoven", "ajax": "Ajax Amsterdam", "feyenoord": "Feyenoord", "az alkmaar": "AZ Alkmaar",
    "club brugge": "Club Brugge", "st gilloise": "Union St.-Gilloise", "galatasaray": "Galatasaray",
    "fenerbahce": "Fenerbahce", "besiktas": "Besiktas", "celtic": "Celtic", "rangers": "Rangers",
}


def canon(name):
    """One identity per club across football-data, ESPN and the Odds API."""
    n = norm(name)
    if n in FD_ALIAS:
        return FD_ALIAS[n]
    return str(name).strip()


def _get_text(url, timeout=30):
    import requests
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (KRONOS desk)"})
    r.raise_for_status()
    return r.text


def fd_ingest(con, season, code, get_text=_get_text):
    import csv as _csv, io as _io
    txt = get_text(f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv")
    league, n = FD_CODES[code], 0
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for r in _csv.DictReader(_io.StringIO(txt.lstrip("\ufeff"))):
        try:
            d = datetime.strptime(r["Date"], "%d/%m/%Y" if len(r["Date"]) > 8 else "%d/%m/%y").date()
            hg, ag = float(r["FTHG"]), float(r["FTAG"])
        except (KeyError, ValueError, TypeError):
            continue
        f = lambda k: float(r[k]) if r.get(k) not in (None, "") else None
        h, a = canon(r["HomeTeam"]), canon(r["AwayTeam"])
        con.execute("""insert or replace into soccer_matches values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (f"fd:{code}:{d.isoformat()}:{norm(h)}:{norm(a)}", league, d.isoformat(), h, a, hg, ag, f("HST"), f("AST"),
                     f("HS"), f("AS"), f("HC"), f("AC"), 0, "FT", now))
        n += 1
    return n


def ingest_day(con, league, day, get=_get):
    js = get(f"{BASE}/{league}/scoreboard?dates={day.strftime('%Y%m%d')}")
    rows = parse_scoreboard(js, league)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for r in rows:
        h, a = canon(r["home"]), canon(r["away"])
        # the same domestic match can arrive from football-data: keep one row per date + teams
        con.execute("delete from soccer_matches where game_date=? and home=? and away=? and source_id like 'fd:%'",
                    (r["game_date"], h, a))
        con.execute("""insert or replace into soccer_matches values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (f"espn:{r['source_id']}", r["league"], r["game_date"], h, a, r["hg"], r["ag"], r["hsot"],
                     r["asot"], r["hsh"], r["ash"], r["hc"], r["ac"], r["neutral"], r["detail"], now))
    return len(rows)


def backfill(con, now=None, budget=40, get=_get, get_text=_get_text):
    """football-data seasons (last season once, current season every 12h) + ESPN single days for Europe
    (Tue/Wed/Thu since Sept 2025) and the last 4 days of the English leagues and Europe."""
    ensure(con)
    now = now or datetime.now(UTC)
    calls, rows, errs = 0, 0, []
    for season in FD_SEASONS:
        for code in FD_CODES:
            key = f"fd:{season}"
            last = con.execute("select ts from soccer_fetch where league=? and month=?", (code, key)).fetchone()
            fresh = last and (season != CURRENT_FD or datetime.fromisoformat(last[0]) > now - timedelta(hours=12))
            if fresh:
                continue
            try:
                n = fd_ingest(con, season, code, get_text)
                rows += n
                con.execute("insert or replace into soccer_fetch values(?,?,?,?)", (code, key, now.isoformat(timespec="seconds"), n))
                con.commit()
            except Exception as e:                             # noqa: BLE001
                errs.append(f"football-data {code} {season}: {str(e)[:60]}")
    con.commit()
    days = []
    d = date(2025, 9, 1)
    while d <= now.date():
        if d.weekday() in (1, 2, 3):
            for lg in EUROPE:
                days.append((lg, d))
        d += timedelta(days=1)
    recent = [(lg, now.date() - timedelta(days=k)) for k in range(0, 4) for lg in ["eng.1", "eng.2"] + EUROPE]
    todo = []
    for lg, d in recent + list(reversed(days)):
        key = d.isoformat()
        done = con.execute("select ts from soccer_fetch where league=? and month=?", (lg, key)).fetchone()
        if done and d < now.date() - timedelta(days=3):
            continue
        if done and datetime.fromisoformat(done[0]) > now - timedelta(hours=3):
            continue
        todo.append((lg, d))
    seen = set()
    for lg, d in todo:
        if (lg, d) in seen:
            continue
        seen.add((lg, d))
        if calls >= budget:
            break
        calls += 1
        try:
            rows += ingest_day(con, lg, d, get)
            con.execute("insert or replace into soccer_fetch values(?,?,?,?)", (lg, d.isoformat(), now.isoformat(timespec="seconds"), 0))
            con.commit()
        except Exception as e:                                 # noqa: BLE001
            errs.append(f"{lg} {d}: {str(e)[:60]}")
    con.commit()
    left = sum(1 for lg, d in days if not con.execute("select 1 from soccer_fetch where league=? and month=?",
                                                       (lg, d.isoformat())).fetchone())
    total = con.execute("select count(*) from soccer_matches").fetchone()[0]
    return {"espn_calls": calls, "rows": rows, "europe_days_left": left, "matches_on_file": total, "errors": errs[:3]}


# --------------------------------------------------------------------------- fit
def fit(con, now=None):
    import numpy as np
    from scipy.optimize import minimize
    now = now or datetime.now(UTC)
    lo = (now - timedelta(days=540)).date().isoformat()
    raw = [dict(r) for r in con.execute("select * from soccer_matches where game_date>=? order by game_date", (lo,))]
    rows, seen = [], set()
    for r in raw:                      # one identity per club (norm of the canonical name), one row per match
        r["home"], r["away"] = norm(canon(r["home"])), norm(canon(r["away"]))
        k = (r["game_date"], r["home"], r["away"])
        if k in seen:
            continue
        seen.add(k); rows.append(r)
    if len(rows) < 200:
        return None
    teams = sorted({r["home"] for r in rows} | {r["away"] for r in rows})
    ti = {t: i for i, t in enumerate(teams)}
    # each team's league = its most recent domestic league; teams seen only in Europe/cups get 'other'
    tl = {}
    for r in rows:
        if r["league"] in DOMESTIC:
            tl[r["home"]] = r["league"]; tl[r["away"]] = r["league"]
    leagues = sorted(set(tl.values()) | {"other"})
    li = {l: i for i, l in enumerate(leagues)}
    team_lg = np.array([li[tl.get(t, "other")] for t in teams])
    # conversion rate for the shots-on-target half of the target
    g = sum(r["hg"] + r["ag"] for r in rows if r["hsot"] is not None and r["asot"] is not None)
    s = sum(r["hsot"] + r["asot"] for r in rows if r["hsot"] is not None and r["asot"] is not None)
    conv = (g / s) if s else 0.3
    H = np.array([ti[r["home"]] for r in rows]); A = np.array([ti[r["away"]] for r in rows])
    def target(goals, sot):
        return goals if sot is None else (1 - SOT_BLEND) * goals + SOT_BLEND * sot * conv
    yh = np.array([target(r["hg"], r["hsot"]) for r in rows]); ya = np.array([target(r["ag"], r["asot"]) for r in rows])
    age = np.array([(now.date() - date.fromisoformat(r["game_date"])).days for r in rows], dtype=float)
    w = 0.5 ** (age / HALF_LIFE_D)
    home_on = np.array([0.0 if r["neutral"] else 1.0 for r in rows])
    nT, nL = len(teams), len(leagues)
    RT, RL = 3.0, 1.0

    def unpack(x):
        return x[0], x[1], x[2:2 + nT], x[2 + nT:2 + 2 * nT], x[2 + 2 * nT:2 + 2 * nT + nL], x[2 + 2 * nT + nL:]

    def f(x):
        mu, hm, at, de, la, ld = unpack(x)
        ath, ata = at[H] + la[team_lg[H]], at[A] + la[team_lg[A]]
        deh, dea = de[H] + ld[team_lg[H]], de[A] + ld[team_lg[A]]
        eh = mu + hm * home_on + ath - dea
        ea = mu + ata - deh
        lh, lam = np.exp(eh), np.exp(ea)
        nll = np.sum(w * (lh - yh * eh)) + np.sum(w * (lam - ya * ea))
        nll += RT * (np.sum(at ** 2) + np.sum(de ** 2)) + RL * (np.sum(la ** 2) + np.sum(ld ** 2))
        rh, ra = w * (lh - yh), w * (lam - ya)
        gmu = rh.sum() + ra.sum(); ghm = (rh * home_on).sum()
        gat = np.zeros(nT); gde = np.zeros(nT); gla = np.zeros(nL); gld = np.zeros(nL)
        np.add.at(gat, H, rh); np.add.at(gat, A, ra)
        np.add.at(gde, A, -rh); np.add.at(gde, H, -ra)
        np.add.at(gla, team_lg[H], rh); np.add.at(gla, team_lg[A], ra)
        np.add.at(gld, team_lg[A], -rh); np.add.at(gld, team_lg[H], -ra)
        gat += 2 * RT * at; gde += 2 * RT * de; gla += 2 * RL * la; gld += 2 * RL * ld
        return nll, np.concatenate([[gmu, ghm], gat, gde, gla, gld])

    x0 = np.zeros(2 + 2 * nT + 2 * nL); x0[0] = math.log(1.35); x0[1] = 0.2
    res = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    mu, hm, at, de, la, ld = unpack(res.x)
    # corners for / against per team (decayed), and last match date per team
    cf, ca, cw, last = {}, {}, {}, {}
    for r, wi in zip(rows, w):
        last[r["home"]] = max(last.get(r["home"], ""), r["game_date"]); last[r["away"]] = max(last.get(r["away"], ""), r["game_date"])
        if r["hc"] is None or r["ac"] is None:
            continue
        for t, f_, a_ in ((r["home"], r["hc"], r["ac"]), (r["away"], r["ac"], r["hc"])):
            cf[t] = cf.get(t, 0) + wi * f_; ca[t] = ca.get(t, 0) + wi * a_; cw[t] = cw.get(t, 0) + wi
    games = {}
    for r in rows:
        games[r["home"]] = games.get(r["home"], 0) + 1; games[r["away"]] = games.get(r["away"], 0) + 1
    model = {"ts": now.isoformat(timespec="seconds"), "mu": float(mu), "home": float(hm), "conv": conv,
             "att": {t: float(at[i] + la[team_lg[i]]) for t, i in ti.items()},
             "def": {t: float(de[i] + ld[team_lg[i]]) for t, i in ti.items()},
             "league_of": {t: leagues[team_lg[i]] for t, i in ti.items()},
             "corners_for": {t: cf[t] / cw[t] for t in cw if cw[t] > 0.5},
             "corners_against": {t: ca[t] / cw[t] for t in cw if cw[t] > 0.5},
             "last": last, "games": games, "n_matches": len(rows), "converged": bool(res.success)}
    vals = list(model["corners_for"].values())
    model["corners_avg"] = sum(vals) / len(vals) if vals else None
    con.execute("insert or replace into meta(k,v) values(?,?)", ("soccer_model", json.dumps(model)))
    con.commit()
    return model


def load(con):
    r = con.execute("select v from meta where k='soccer_model'").fetchone()
    return json.loads(r[0]) if r else None


# --------------------------------------------------------------------------- probabilities
def score_matrix(lh, la, rho=RHO, n=11):
    def pois(l, k):
        return math.exp(-l) * l ** k / math.factorial(k)
    M = [[pois(lh, i) * pois(la, j) for j in range(n)] for i in range(n)]
    M[0][0] *= max(0.0, 1 - lh * la * rho); M[0][1] *= max(0.0, 1 + lh * rho)
    M[1][0] *= max(0.0, 1 + la * rho); M[1][1] *= max(0.0, 1 - rho)
    s = sum(map(sum, M))
    return [[v / s for v in row] for row in M]


def _split(line):
    """A quarter line is half on each neighbouring half-goal line."""
    q = round(float(line) * 4) % 2
    return [line - 0.25, line + 0.25] if q == 1 else [line]


def p_1x2(M):
    n = len(M)
    h = sum(M[i][j] for i in range(n) for j in range(n) if i > j)
    d = sum(M[i][i] for i in range(n))
    return h, d, 1 - h - d


def p_cover(M, f):
    """Effective win probability of a bet whose result per score is f(i,j) -> margin (>0 win, 0 push, <0 lose)."""
    n = len(M)
    win = sum(M[i][j] for i in range(n) for j in range(n) if f(i, j) > 1e-9)
    lose = sum(M[i][j] for i in range(n) for j in range(n) if f(i, j) < -1e-9)
    return win / (win + lose) if win + lose > 0 else 0.5


def p_over(M, line):
    return sum(p_cover(M, lambda i, j, L=L: (i + j) - L) for L in _split(line)) / len(_split(line))


def p_home_ah(M, line):
    return sum(p_cover(M, lambda i, j, L=L: (i - j) + L) for L in _split(line)) / len(_split(line))


def market_lambdas(fh, fa, f_over, line):
    """Expected goals implied by the market's 1X2 and total (used for the market side of Jerry's spin)."""
    from scipy.optimize import minimize
    def loss(x):
        lh, la = math.exp(x[0]), math.exp(x[1])
        M = score_matrix(lh, la)
        h, d, a = p_1x2(M)
        e = (h - fh) ** 2 + (a - fa) ** 2
        if f_over is not None and line is not None:
            e += (p_over(M, line) - f_over) ** 2
        return e
    r = minimize(loss, [math.log(1.4), math.log(1.1)], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-9})
    return math.exp(r.x[0]), math.exp(r.x[1])


def _devig(prices):
    inv = [1 / p for p in prices]
    s = sum(inv)
    return [x / s for x in inv]


def _pin(con, eid):
    """Sharp quotes {market: {outcome: (line, price)}} from the first sharp book that has them."""
    out = {}
    for bk in ("pinnacle", "lowvig", "betonlineag"):
        for r in con.execute("""select market, outcome, line, price from last_quote where event_id=? and book=?
                                and coalesce(player,'')=''""", (eid, bk)):
            out.setdefault(r[0], {}).setdefault(r[1], (r[2], r[3]))
    return out


def _days_since(last, commence):
    try:
        return (date.fromisoformat(str(commence)[:10]) - date.fromisoformat(last)).days
    except Exception:                                          # noqa: BLE001
        return None


def predict(model, home_e, away_e, commence, neutral=False):
    """Expected goals for an ESPN-named fixture, with the matchup priors and a note of each."""
    mu, hm = model["mu"], model["home"]
    att, de = model["att"], model["def"]
    lh = math.exp(mu + (0 if neutral else hm) + att[home_e] - de[away_e])
    la = math.exp(mu + att[away_e] - de[home_e])
    notes = []
    for side, team in (("h", home_e), ("a", away_e)):
        ds = _days_since(model["last"].get(team, ""), commence)
        if ds is not None and 0 < ds <= REST_DAYS:
            if side == "h":
                lh *= REST_ATT; la *= REST_DEF
            else:
                la *= REST_ATT; lh *= REST_DEF
            notes.append(f"{team} on {ds} days' rest")
    cavg = model.get("corners_avg")
    cfor, cag = model.get("corners_for", {}), model.get("corners_against", {})
    if cavg:
        for side, t, o in (("h", home_e, away_e), ("a", away_e, home_e)):
            if t in cfor and o in cag:
                idx = (cfor[t] / cavg) * (cag[o] / cavg)
                adj = max(-CORNER_CAP, min(CORNER_CAP, 0.10 * (idx - 1)))
                if abs(adj) >= 0.02:
                    if side == "h":
                        lh *= 1 + adj
                    else:
                        la *= 1 + adj
                    notes.append(f"set pieces: {t} {cfor[t]:.1f} corners/game vs {o} conceding {cag[o]:.1f} ({adj * 100:+.0f}% xG)")
    return lh, la, notes


def likely_scorers(con, team_key, lam, n=2):
    """Top recent scorers for a team from ESPN boxes the grader stored, with anytime chances."""
    try:
        names = [r[0] for r in con.execute("select distinct team from soccer_box")]
        team_espn, _sc = best_name(team_key, names, 0.6)
        if not team_espn:
            return []
        rows = con.execute("""select player, sum(goals) g from soccer_box where team=? group by player_norm
                              order by g desc limit 6""", (team_espn,)).fetchall()
        tot = con.execute("select sum(goals) from soccer_box where team=?", (team_espn,)).fetchone()[0] or 0
    except Exception:                                          # noqa: BLE001
        return []
    if not tot or tot < 3:
        return []
    out = []
    for r in rows[:n]:
        share = r[1] / tot
        out.append(f"{r[0]} {1 - math.exp(-lam * share):.0%}")
    return out


def refresh(con, now=None, L=None, jr=None, fit_every_h=6):
    """Hourly: top up data, refit every few hours, write model rows for EPL/UCL matches in the next 48h."""
    ensure(con)
    now = now or datetime.now(UTC)
    status = {"data": None, "fit": None, "rows": 0, "matched": 0, "unmatched": []}
    try:
        status["data"] = backfill(con, now)
    except Exception as e:                                     # noqa: BLE001
        status["data"] = f"error: {str(e)[:100]}"
    model = load(con)
    stale = (not model) or (datetime.fromisoformat(model["ts"]) < now - timedelta(hours=fit_every_h))
    if stale:
        try:
            model = fit(con, now) or model
            status["fit"] = "refit" if model else "not enough data yet"
        except Exception as e:                                 # noqa: BLE001
            status["fit"] = f"error: {str(e)[:100]}"
    else:
        status["fit"] = f"cached {model['ts']}"
    if not model:
        return status
    teams = list(model["att"].keys())
    evs = [dict(r) for r in con.execute("""select * from events where sport in ('EPL','UCL') and commence>? and commence<?
                                           order by commence""", (now.isoformat(timespec="seconds"),
                                                                  (now + timedelta(hours=48)).isoformat(timespec="seconds")))]
    rows = []
    for ev in evs:
        he, _s1 = best_name(norm(canon(ev["home"])), teams, 0.6); ae, _s2 = best_name(norm(canon(ev["away"])), teams, 0.6)
        if not he or not ae or he == ae:
            status["unmatched"].append(f"{ev['away']} @ {ev['home']}")
            continue
        status["matched"] += 1
        q = _pin(con, ev["event_id"])
        h2h = q.get("h2h", {})
        lh, la, notes = predict(model, he, ae, ev["commence"])
        M = score_matrix(lh, la)
        ph, pd, pa = p_1x2(M)
        wgt = MODEL_WEIGHT.get(ev["sport"], 0.3)
        if L is not None and hasattr(L, "fitted_anchor"):
            try:
                wgt = L.fitted_anchor(con, ev["sport"], "h2h", default=wgt)
            except Exception:                                  # noqa: BLE001
                pass
        ts = now.isoformat(timespec="seconds")
        detail = (f"xG {lh:.2f}-{la:.2f}; ratings att/def {ev['home']} {model['att'][he]:+.2f}/{model['def'][he]:+.2f}, "
                  f"{ev['away']} {model['att'][ae]:+.2f}/{model['def'][ae]:+.2f}; games {model['games'].get(he, 0)}/{model['games'].get(ae, 0)}"
                  + (f"; {'; '.join(notes)}" if notes else ""))
        sc_h, sc_a = likely_scorers(con, he, lh), likely_scorers(con, ae, la)
        if sc_h or sc_a:
            detail += f"; likely scorers {', '.join(sc_h + sc_a)}"
        mkt = {}
        if ev["home"] in h2h and ev["away"] in h2h and "Draw" in h2h:
            fh, fd, fa = _devig([h2h[ev["home"]][1], h2h["Draw"][1], h2h[ev["away"]][1]])
            mkt["h2h"] = (fh, fd, fa)
            for oc, pm, pf in ((ev["home"], ph, fh), ("Draw", pd, fd), (ev["away"], pa, fa)):
                pb = wgt * pm + (1 - wgt) * pf
                e = pb - pf
                call = f"Lean {oc}" if e >= LEAN_EDGE else "Sit"
                rows.append(dict(event_id=ev["event_id"], market="h2h", outcome=oc, player="", line=None,
                                 model_prob=round(pb, 4), fair_prob=round(pf, 4), edge=round(e, 4), call=call,
                                 note=f"{VERSION} raw {pm:.3f}", reasoning=f"KRONOS soccer: {detail}; model weight {wgt:.2f}",
                                 model_number=round(lh - la, 3), market_number=None, components={"lh": lh, "la": la}))
        tot = q.get("totals", {})
        if "Over" in tot and "Under" in tot and tot["Over"][0] is not None and abs(tot["Over"][0] - (tot["Under"][0] or 0)) < 1e-9:
            L_ = float(tot["Over"][0])
            fo, fu = _devig([tot["Over"][1], tot["Under"][1]])
            mkt["tot"] = (fo, L_)
            pm = p_over(M, L_)
            pb = wgt * pm + (1 - wgt) * fo
            e = pb - fo
            call = "Lean Over" if e >= LEAN_EDGE else "Lean Under" if e <= -LEAN_EDGE else "Sit"
            rows.append(dict(event_id=ev["event_id"], market="totals", outcome="Over", player="", line=L_,
                             model_prob=round(pb, 4), fair_prob=round(fo, 4), edge=round(e, 4), call=call,
                             note=f"{VERSION} raw {pm:.3f}", reasoning=f"KRONOS soccer: {detail}",
                             model_number=round(lh + la, 3), market_number=L_, components={"lh": lh, "la": la}))
        sp = q.get("spreads", {})
        if ev["home"] in sp and ev["away"] in sp and sp[ev["home"]][0] is not None:
            hl = float(sp[ev["home"]][0])
            fh_, fa_ = _devig([sp[ev["home"]][1], sp[ev["away"]][1]])
            pm = p_home_ah(M, hl)
            pb = wgt * pm + (1 - wgt) * fh_
            e = pb - fh_
            call = f"Lean {ev['home']}" if e >= LEAN_EDGE else f"Lean {ev['away']}" if e <= -LEAN_EDGE else "Sit"
            rows.append(dict(event_id=ev["event_id"], market="spreads", outcome=ev["home"], player="", line=hl,
                             model_prob=round(pb, 4), fair_prob=round(fh_, 4), edge=round(e, 4), call=call,
                             note=f"{VERSION} raw {pm:.3f}", reasoning=f"KRONOS soccer: {detail}",
                             model_number=round(lh - la, 3), market_number=-hl, components={"lh": lh, "la": la}))
        if jr is not None:
            try:
                mlh = mla = None
                if "h2h" in mkt:
                    fo_, L2 = mkt.get("tot", (None, None))
                    mlh, mla = market_lambdas(mkt["h2h"][0], mkt["h2h"][2], fo_, L2)
                jr.record_numbers(con, event_id=ev["event_id"], sport=ev["sport"], home=ev["home"], away=ev["away"],
                                  commence=ev["commence"], margin=lh - la, total=lh + la, p_home=ph,
                                  mkt_margin=None if mlh is None else mlh - mla, mkt_total=None if mlh is None else mlh + mla,
                                  mkt_p_home=mkt["h2h"][0] if "h2h" in mkt else None,
                                  checks={"team_data": min(model["games"].get(he, 0), model["games"].get(ae, 0)) >= 8},
                                  detail=detail)
            except Exception:                                  # noqa: BLE001
                pass
    if rows and L is not None:
        try:
            L.write_probs(con, rows, now); con.commit()
        except Exception as e:                                 # noqa: BLE001
            status["write_error"] = str(e)[:100]
        try:
            L.paper_log(con, rows, now); con.commit()
        except Exception as e:                                 # noqa: BLE001
            status["paper_error"] = str(e)[:100]
    status["rows"] = len(rows)
    con.execute("insert or replace into meta(k,v) values(?,?)", ("soccer_status", json.dumps(status)[:2000]))
    con.commit()
    return status
'''

_LEDGER_SRC = r'''"""
kronos_ledger — one KRONOS prediction for every tracked match, logged before kickoff, graded for free.
(ledger-v1, 2026-11-01)

Every event in the desk (NFL, NBA, NHL, EPL, UCL, UFC) gets a paper pick in each game-line market it
has (moneyline/1X2, total, spread/handicap), logged LEDGER_LEAD minutes before kickoff so it uses the
latest lineup-driven prices. Never "no bet": every market gets a side.

Side choice: the sport model's probability vs Pinnacle's de-vigged fair (model_probs). With no model
row, the market model: the side sharp money is on (sharp_signal score), else the market favourite.
Grades/stakes (paper units): A 1u (edge >= 5%), B 0.5u (>= 3%), C 0.25u (>= 1.5%), D 0.1u (otherwise and
every market-model pick). Unvalidated models are capped at B; a model-vs-market gap over 8% is capped at C and tagged
"check team news" (the biggest gaps have been the biggest misses).
Ledger rows are stored as paper=2 so they stay out of the desk's existing model/price-edge report and
are reported in their own KRONOS LEDGER block.
Price: best of DraftKings/FanDuel/Fanatics at the same line; if none, Pinnacle's price, marked
"no CT price"; with no price anywhere the pick is logged at zero stake, marked "no posted line".
Every row carries the 22 KRONOS signals, so the learning layer can measure them.

Grading: ESPN public scoreboards (free, no Odds API credits) for every sport; closing line value from
the desk's own Pinnacle tape. Extra-time / penalty finishes in soccer are left for the desk's grader
(1X2 settles on 90 minutes).
"""
import json, math, os, re, unicodedata
from datetime import datetime, timezone, timedelta

UTC = timezone.utc
VERSION = "ledger-v1-2026-11-01"
LEDGER_LEAD = int(os.environ.get("LEDGER_LEAD_MIN", "90"))
CT_BOOKS = ("draftkings", "fanduel", "fanatics")
SHARP = ("pinnacle", "lowvig", "betonlineag")
MARKETS = {"NFL": ("h2h", "spreads", "totals"), "NBA": ("h2h", "spreads", "totals"), "NHL": ("h2h", "spreads", "totals"),
           "EPL": ("h2h", "spreads", "totals"), "UCL": ("h2h", "spreads", "totals"), "UFC": ("h2h",),
           "MLB": ("h2h", "spreads", "totals")}
GRADES = ((0.05, "A", 1.0), (0.03, "B", 0.5), (0.015, "C", 0.25))
D_STAKE = 0.1
BIG_GAP = float(os.environ.get("LEDGER_BIG_GAP", "0.08"))
NEW_BETS = []          # BET picks logged since the caller last drained this list (for Telegram)
CAP_GRADE = os.environ.get("LEDGER_CAP_GRADE", "B")
ESPN = {"NFL": "football/nfl", "NBA": "basketball/nba", "NHL": "hockey/nhl", "MLB": "baseball/mlb",
        "EPL": "soccer/eng.1", "UCL": "soccer/uefa.champions", "UFC": "mma/ufc"}
GRADE_AFTER_MIN = {"NFL": 170, "NBA": 120, "NHL": 120, "MLB": 150, "EPL": 105, "UCL": 105, "UFC": 30}


def _iso(t):
    return t.isoformat(timespec="seconds")


def _ts(s):
    t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def _same(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) < 1e-9


def _devig(prices):
    inv = [1 / float(p) for p in prices]
    s = sum(inv)
    return [x / s for x in inv]


# --------------------------------------------------------------------------- market state
def sides(con, eid, market):
    """Current sharp sides for a market: [(outcome, line, price, book)], same-line pairs only."""
    for bk in SHARP:
        rows = [tuple(r) for r in con.execute("""select outcome, line, price from last_quote where event_id=? and book=?
                                                 and market=? and coalesce(player,'')=''""", (eid, bk, market))]
        if not rows:
            continue
        if market == "totals":
            o = [r for r in rows if r[0] == "Over"]; u = [r for r in rows if r[0] == "Under"]
            if o and u and _same(o[0][1], u[0][1]):
                return [(o[0][0], o[0][1], o[0][2], bk), (u[0][0], u[0][1], u[0][2], bk)]
            continue
        if market == "spreads":
            if len(rows) == 2 and rows[0][1] is not None and rows[1][1] is not None and _same(rows[0][1], -rows[1][1]):
                return [(r[0], r[1], r[2], bk) for r in rows]
            continue
        return [(r[0], r[1], r[2], bk) for r in rows]
    # no sharp book: any book that has the full market
    rows = [tuple(r) for r in con.execute("""select outcome, line, price, book from last_quote where event_id=? and market=?
                                             and coalesce(player,'')='' order by book""", (eid, market))]
    by = {}
    for r in rows:
        by.setdefault(r[3], []).append(r)
    for bk, rs in by.items():
        if market == "h2h" or len(rs) == 2:
            return [(r[0], r[1], r[2], bk) for r in rs]
    return []


def best_ct(con, eid, market, outcome, line):
    rows = con.execute(f"""select book, line, price from last_quote where event_id=? and market=? and outcome=?
                           and coalesce(player,'')='' and book in ({','.join('?' * len(CT_BOOKS))})""",
                       (eid, market, outcome, *CT_BOOKS)).fetchall()
    rows = [r for r in rows if market == "h2h" or _same(r[1], line)]
    if not rows:
        return None
    r = max(rows, key=lambda r: r[2])
    return {"book": r[0], "line": r[1], "price": float(r[2])}


def model_probs_for(con, eid, market, outs):
    """{outcome: model prob} from model_probs, completing two-way markets from one stored row."""
    rows = {r[0]: (r[1], r[2]) for r in con.execute("""select outcome, model_prob, line from model_probs where event_id=?
                                                       and market=? and coalesce(player,'')='' and model_prob is not null""",
                                                    (eid, market))}
    if not rows:
        return {}
    out = {}
    for o, line, *_ in outs:
        if o in rows:
            out[o] = float(rows[o][0])
    if len(outs) == 2 and len(out) == 1:
        o1 = next(iter(out)); o2 = [o for o, *_ in outs if o != o1][0]
        out[o2] = 1 - out[o1]
    if market == "spreads":
        # a stored spread row at a different line than the current one is stale: drop it
        for o, line, *_ in outs:
            if o in rows and rows[o][1] is not None and line is not None and not _same(rows[o][1], line):
                return {}
    return out


def grade_for(edge, has_model):
    if not has_model:
        return "D", D_STAKE
    for th, g, st in GRADES:
        if edge >= th:
            order = "ABCD"
            if order.index(g) < order.index(CAP_GRADE):
                g = CAP_GRADE; st = dict((x[1], x[2]) for x in GRADES)[g]
            return g, st
    return "D", D_STAKE


def model_name(L, sport, market):
    try:
        return L.model_version(sport, market) if L is not None else f"{sport.lower()}-model"
    except Exception:                                          # noqa: BLE001
        return f"{sport.lower()}-model"


# --------------------------------------------------------------------------- logging
def ensure(con):
    cols = {r[1] for r in con.execute("pragma table_info(bets)")}
    for c, t in (("paper", "int"), ("signals", "text"), ("model_version", "text"), ("hours_to_start", "real"),
                 ("edge", "real"), ("fair_prob", "real"), ("evidence_kind", "text"), ("logged_call", "text"),
                 ("sharp_flag", "text"), ("card_tags", "text"), ("comp_json", "text")):
        if c not in cols:
            con.execute(f"alter table bets add column {c} {t}")
    con.execute("create index if not exists bets_ledger on bets(evidence_kind, event_id, market)")


def log_event(con, ev, now, L=None, jr=None, js=None):
    """Log one ledger pick per game-line market for this event. Returns the number logged."""
    n = 0
    for market in MARKETS.get(ev["sport"], ("h2h",)):
        if con.execute("select 1 from bets where evidence_kind='ledger' and event_id=? and market=?",
                       (ev["event_id"], market)).fetchone():
            continue
        outs = sides(con, ev["event_id"], market)
        if len(outs) < 2:
            continue
        fair = _devig([o[2] for o in outs])
        fp = {o[0]: f for o, f in zip(outs, fair)}
        mp = model_probs_for(con, ev["event_id"], market, outs)
        has_model = len(mp) == len(outs)
        why = []
        # price every side at its best CT price (Pinnacle if no CT price) and pick on EV at that price (2026-11-01)
        quotes = {}
        for o in outs:
            q = best_ct(con, ev["event_id"], market, o[0], o[1])
            quotes[o[0]] = (q["price"], q["book"], True) if q else (float(o[2]), o[3], False)
        def _ev(o, use_model):
            pr = quotes[o[0]][0]
            return ((mp[o[0]] if use_model else fp[o[0]]) * pr) - 1
        price_edges = [o for o in outs if quotes[o[0]][2] and _ev(o, False) >= 0.015]
        if has_model:
            pick = max(outs, key=lambda o: _ev(o, True))
            edge = mp[pick[0]] - fp[pick[0]]
            why.append(f"model {mp[pick[0]]:.1%} vs fair {fp[pick[0]]:.1%}")
            kind = "model"
        elif price_edges:
            pick = max(price_edges, key=lambda o: _ev(o, False))
            edge = 0.0
            why.append("no model row: CT price beats Pinnacle fair")
            kind = "market"
        else:
            scores = {}
            for o in outs:
                try:
                    s_ = jr.sharp_signal(con, ev["event_id"], market, o[0]) if jr else {}
                except Exception:                              # noqa: BLE001
                    s_ = {}
                scores[o[0]] = (s_.get("score") or 0, fp[o[0]])
            pick = max(outs, key=lambda o: scores[o[0]])
            edge = 0.0
            why.append(f"no model row: market model, {'sharp side' if scores[pick[0]][0] > 0 else 'market favourite'}")
            kind = "market"
        sel, line = pick[0], pick[1]
        price, book, from_ct = quotes[sel]
        tags = [] if from_ct else ["no CT price"]
        try:
            sig = jr.sharp_signal(con, ev["event_id"], market, sel) if jr else {}
        except Exception:                                      # noqa: BLE001
            sig = {}
        try:
            import kronos_edge as _ke
            is_bet, bet_why, ev_mod, ev_mkt = _ke.bet_check(price, from_ct, fp[sel], mp.get(sel) if has_model else None,
                                                            flag_label=(sig or {}).get("label", ""), gap=edge if has_model else None)
            grade, stake = _ke.grade_from_ev(ev_mod if ev_mod is not None else -1, has_model, edge if has_model else None)
        except Exception:                                      # noqa: BLE001
            is_bet, bet_why, ev_mod, ev_mkt = False, "", None, price * fp[sel] - 1
            grade, stake = grade_for(edge, has_model)
        if has_model and edge > BIG_GAP:
            tags.append(f"big gap {edge:.0%}: check team news")
        ev_pct = price * fp[sel] - 1
        if ev_mod is not None and has_model:
            why.append(f"model EV {ev_mod:+.1%} at this price")
        if is_bet:
            why.append("BET: " + bet_why)
        signals = None
        if js is not None:
            try:
                extra = {"sig": sig}
                if has_model:
                    extra["pc"] = {"p_side": mp[sel]}
                signals = json.dumps(js.capture(con, ev, market, sel, "", line, price, fp[sel], extra))
            except Exception:                                  # noqa: BLE001
                signals = None
        mver = model_name(L, ev["sport"], market) if has_model else "market"
        try:
            hrs = round((_ts(ev["commence"]) - now).total_seconds() / 3600, 2)
        except Exception:                                      # noqa: BLE001
            hrs = None
        note = (f"KRONOS ledger | grade {grade} | {kind} ({mver}) | " + "; ".join(why) +
                f" | EV {ev_pct * 100:+.1f}% at {book}" + (f" | {', '.join(tags)}" if tags else ""))
        try:
            import kronos_master as _km
            ctx = _km.event_context(con, ev)
            note += f" | roster/schedule: {ctx['text']}" if ctx.get("text") else " | roster/schedule: nothing flagged"
        except Exception:                                      # noqa: BLE001
            pass
        con.execute("""insert into bets(placed_ts, sport, event_id, game, commence, market, selection, player, line, price,
                       book, stake, model_prob, note, paper, signals, model_version, hours_to_start, edge, fair_prob,
                       evidence_kind, logged_call, sharp_flag, card_tags)
                       values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,2,?,?,?,?,?,?,?,?,?)""",
                    (_iso(now), ev["sport"], ev["event_id"], f"{ev['away']} @ {ev['home']}", ev["commence"], market, sel, "",
                     line, price, book, stake, round(mp[sel], 4) if has_model else round(fp[sel], 4), note[:600],
                     signals, f"{VERSION}+{mver}", hrs, round(edge, 4), round(fp[sel], 4), "ledger",
                     (f"BET · Grade {grade}" if is_bet else f"Grade {grade}"), (sig or {}).get("label"),
                     json.dumps({"grade": grade, "kind": kind, "tags": tags, "bet": is_bet,
                                 "ev_model": None if ev_mod is None else round(ev_mod, 4), "ev_market": round(ev_pct, 4)})))
        if is_bet:
            ln = "" if line is None else (f" {line:+g}" if market == "spreads" else f" {line:g}")
            NEW_BETS.append(f"KRONOS BET: {ev['away']} @ {ev['home']} ({ev['sport']})\n{market}: {sel}{ln} at "
                            f"{book} {price:.2f}, grade {grade} {stake:g}u\n{bet_why}")
        n += 1
    return n


def log_due(con, now=None, L=None, jr=None, js=None, lead_min=None):
    """Every event starting within the lead window (and not yet started) gets its ledger picks."""
    ensure(con)
    now = now or datetime.now(UTC)
    lead = LEDGER_LEAD if lead_min is None else lead_min
    evs = [dict(r) for r in con.execute("select * from events where commence>? and commence<=? order by commence",
                                        (_iso(now), _iso(now + timedelta(minutes=lead))))]
    n = 0
    for ev in evs:
        try:
            n += log_event(con, ev, now, L, jr, js)
        except Exception as e:                                 # noqa: BLE001
            con.execute("insert or replace into meta(k,v) values(?,?)",
                        ("ledger_error", f"{_iso(now)} {ev.get('event_id')} {str(e)[:150]}"))
    con.commit()
    return n


# --------------------------------------------------------------------------- free grading (ESPN)
def _norm(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ").replace("-", " ").replace(".", " ").replace("'", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


_STOP = {"fc", "afc", "cf", "sc", "the", "and", "de", "club", "calcio", "ac", "as", "fk", "sk", "cp"}
_ALIAS = {"inter milan": "internazionale", "la clippers": "los angeles clippers", "la lakers": "los angeles lakers",
          "sporting lisbon": "sporting cp", "paris saint germain": "paris saint germain"}


def _tok(s):
    n = _ALIAS.get(_norm(s), _norm(s))
    t = set(n.split())
    return (t - _STOP) or t


def _match(a, b):
    ta, tb = _tok(a), _tok(b)
    if not ta or not tb:
        return 0.0
    j = len(ta & tb) / len(ta | tb)
    return max(j, 0.75) if (ta <= tb or tb <= ta) else j


def _get(url, timeout=20):
    import requests
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (KRONOS desk)"})
    r.raise_for_status()
    return r.json()


def espn_finals(sport, day, get=_get):
    """[(home, away, hs, as, detail)] for team sports, [(fighter_a, fighter_b, winner, detail)] for UFC."""
    path = ESPN[sport]
    js = get(f"https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard?dates={day.replace('-', '')}&limit=300")
    out = []
    for ev in js.get("events", []):
        for comp in ev.get("competitions", []) or []:
            st = (comp.get("status") or ev.get("status") or {}).get("type", {})
            if not st.get("completed"):
                continue
            detail = str(st.get("detail") or st.get("shortDetail") or st.get("description") or "")
            cs = comp.get("competitors", [])
            if sport == "UFC":
                names = [((c.get("athlete") or {}).get("displayName") or (c.get("team") or {}).get("displayName", "")) for c in cs]
                win = [n for n, c in zip(names, cs) if c.get("winner")]
                if len(names) == 2:
                    out.append(("ufc", names[0], names[1], win[0] if win else None, detail))
                continue
            t = {c.get("homeAway"): c for c in cs}
            if "home" in t and "away" in t:
                try:
                    out.append(("team", t["home"]["team"]["displayName"], t["away"]["team"]["displayName"],
                                float(t["home"]["score"]), float(t["away"]["score"]), detail))
                except (KeyError, TypeError, ValueError):
                    pass
    return out


def settle(market, selection, line, home, away, hs, as_):
    if hs is None or as_ is None:
        return None
    if market == "h2h" and selection == "Draw":
        return "W" if hs == as_ else "L"
    mine = hs if selection == home else as_ if selection == away else None
    theirs = as_ if selection == home else hs if selection == away else None
    if market == "h2h" and mine is not None:
        return "W" if mine > theirs else "L" if mine < theirs else "P"
    if market == "spreads" and mine is not None and line is not None:
        parts = [float(line) - 0.25, float(line) + 0.25] if round(float(line) * 4) % 2 == 1 else [float(line)]
        res = [("W" if mine + p - theirs > 0 else "L" if mine + p - theirs < 0 else "P") for p in parts]
        return _combine(res)
    if market == "totals" and line is not None and selection in ("Over", "Under"):
        t = hs + as_
        parts = [float(line) - 0.25, float(line) + 0.25] if round(float(line) * 4) % 2 == 1 else [float(line)]
        res = [("P" if t == p else ("W" if (t > p) == (selection == "Over") else "L")) for p in parts]
        return _combine(res)
    return None


def _combine(res):
    if len(res) == 1:
        return res[0]
    s = set(res)
    if s == {"W"}: return "W"
    if s == {"L"}: return "L"
    if s == {"W", "P"}: return "HW"      # half win (quarter line)
    if s == {"L", "P"}: return "HL"      # half loss
    return "P"


def pnl_for(res, stake, price):
    stake = float(stake or 0); price = float(price or 1)
    return {"W": stake * (price - 1), "L": -stake, "P": 0.0, "HW": stake * (price - 1) / 2, "HL": -stake / 2}[res]


def closing(con, b):
    """Pinnacle's last pre-kickoff price on this side (same line if possible), plus a de-vigged close."""
    for extra, args in (("and line is ?", (b["line"],)), ("", ())):
        r = con.execute(f"""select book, line, price, ts from ticks where event_id=? and market=? and outcome=? and player=''
                            and ts<=? and book in ('pinnacle','lowvig','betonlineag') {extra}
                            order by (book='pinnacle') desc, ts desc limit 1""",
                        (b["event_id"], b["market"], b["selection"], b["commence"], *args)).fetchone()
        if r:
            break
    else:
        return None
    others = []
    for o in con.execute("""select distinct outcome from ticks where event_id=? and market=? and player='' and outcome<>?""",
                         (b["event_id"], b["market"], b["selection"])):
        ol = r[1]
        if b["market"] == "spreads" and ol is not None:
            ol = -float(ol)
        q = con.execute("""select price from ticks where event_id=? and market=? and outcome=? and player='' and book=?
                           and line is ? and ts<=? order by ts desc limit 1""",
                        (b["event_id"], b["market"], o[0], r[0], None if b["market"] == "h2h" else ol, b["commence"])).fetchone()
        if q:
            others.append(float(q[0]))
    fair = None
    if others:
        fair = _devig([float(r[2])] + others)[0]
    return {"book": r[0], "line": r[1], "price": float(r[2]), "fair": fair}


def grade_pending(con, now=None, jr=None, get=_get):
    """Grade every pending game-line bet (ledger, model, price-edge, real) from ESPN finals."""
    now = now or datetime.now(UTC)
    pend = [dict(r) for r in con.execute("""select * from bets where result='pending' and coalesce(player,'')=''
                                            and market in ('h2h','spreads','totals') and commence<=?""", (_iso(now),))]
    due = [b for b in pend if _ts(b["commence"]) + timedelta(minutes=GRADE_AFTER_MIN.get(b["sport"], 150)) <= now]
    cache, graded, held = {}, 0, 0
    for b in due:
        if b["sport"] not in ESPN:
            continue
        ev = con.execute("select home, away from events where event_id=?", (b["event_id"],)).fetchone()
        if not ev:
            continue
        home, away = ev[0], ev[1]
        d0 = _ts(b["commence"])
        hit = None
        for day in sorted({(d0 - timedelta(hours=8)).date().isoformat(), d0.date().isoformat()}):
            k = (b["sport"], day)
            if k not in cache:
                try:
                    cache[k] = espn_finals(b["sport"], day, get=get)
                except Exception:                              # noqa: BLE001
                    cache[k] = []
            for g in cache[k]:
                if g[0] == "ufc":
                    s = min(_match(home, g[1]), _match(away, g[2])) if True else 0
                    s2 = min(_match(home, g[2]), _match(away, g[1]))
                    if max(s, s2) >= 0.5:
                        hit = g; break
                else:
                    if min(_match(home, g[1]), _match(away, g[2])) >= 0.5:
                        hit = g; break
            if hit:
                break
        if not hit:
            continue
        if hit[0] == "ufc":
            winner = hit[3]
            if winner is None:
                res = "P"
            else:
                res = "W" if _match(b["selection"], winner) >= 0.5 else "L"
            hs = as_ = None
        else:
            _, eh, ea, hs, as_, detail = hit
            if b["sport"] in ("EPL", "UCL") and re.search(r"aet|pen|extra", detail, re.I):
                held += 1
                continue                                       # 1X2 settles on 90 minutes: leave for the desk / manual
            res = settle(b["market"], b["selection"], b["line"], home, away, hs, as_)
        if not res:
            continue
        c = closing(con, b)
        clv = None
        if c and jr is not None:
            try:
                clv = jr.clv_pp(b["line"], b["price"], c["line"], c["price"], b["market"], b["selection"], close_fair=c["fair"])
            except Exception:                                  # noqa: BLE001
                clv = None
        stored = {"HW": "W", "HL": "L"}.get(res, res)
        con.execute("""update bets set result=?, pnl=?, home_score=?, away_score=?, close_book=?, close_line=?, close_price=?,
                       clv_pp=coalesce(clv_pp, ?), graded_ts=?, note=coalesce(note,'')||? where id=? and result='pending'""",
                    (stored, round(pnl_for(res, b["stake"], b["price"]), 4), hs, as_, c["book"] if c else None,
                     c["line"] if c else None, c["price"] if c else None, clv, _iso(now),
                     (" | half " + ("win" if res == "HW" else "loss") if res in ("HW", "HL") else "") + " | graded (ESPN)",
                     b["id"]))
        if hs is not None:
            try:
                con.execute("insert or replace into final_scores values(?,?,?,?,?)", (b["event_id"], b["sport"], hs, as_, _iso(now)))
            except Exception:                                  # noqa: BLE001
                pass
        graded += 1
    con.commit()
    return {"due": len(due), "graded": graded, "held_extra_time": held}


# --------------------------------------------------------------------------- learning dedupe
def install(js):
    """The learning layer counts each (event, market, side) once, preferring the ledger row, so a game
    that has both a model paper row and a ledger row is not double-counted."""
    if js is None or getattr(js, "_ledger_installed", False):
        return False

    def _graded_rows(con):
        rows, seen = [], {}
        for b in con.execute("""select result, fair_prob, model_prob, price, clv_pp, signals, sport, market, evidence_kind,
                                       hours_to_start, event_id, selection, coalesce(player,'') player, graded_ts
                                from bets where paper in (1, 2) and result in ('W','L') and signals is not null
                                order by (evidence_kind='ledger') desc, id"""):
            key = (b["event_id"], b["market"], b["selection"], b["player"])
            if key in seen:
                continue
            seen[key] = 1
            try:
                sg = json.loads(b["signals"])
            except Exception:                                  # noqa: BLE001
                continue
            f = b["fair_prob"] or b["model_prob"]
            if not f:
                continue
            rows.append((1 if b["result"] == "W" else 0, float(f), sg, b))
        return rows

    js._graded_rows = _graded_rows
    js._ledger_installed = True
    return True


# --------------------------------------------------------------------------- reporting
def coverage(con, now=None, L=None):
    """For every tracked sport: events in the next 48h, how many have model rows, and ledger status."""
    now = now or datetime.now(UTC)
    out = {}
    for sp in ("NFL", "NBA", "NHL", "EPL", "UCL", "UFC"):
        evs = [r[0] for r in con.execute("select event_id from events where sport=? and commence>? and commence<?",
                                         (sp, _iso(now), _iso(now + timedelta(hours=48))))]
        with_model = 0
        for e in evs:
            if con.execute("select 1 from model_probs where event_id=? and coalesce(player,'')='' limit 1", (e,)).fetchone():
                with_model += 1
        out[sp] = {"events_48h": len(evs), "with_model_rows": with_model,
                   "model": model_name(L, sp, "h2h") if L is not None else "?"}
    return out


def feed_block(con, now=None, L=None):
    now = now or datetime.now(UTC)
    lines = ["", "", f"KRONOS LEDGER ({VERSION}: one paper pick per market for every tracked match, logged {LEDGER_LEAD} min "
             "before kickoff, graded free from ESPN; grades A 1u / B 0.5u / C 0.25u / D 0.1u, models capped at "
             f"{CAP_GRADE} until validated):"]
    try:
        cov = coverage(con, now, L)
        lines.append("  model coverage next 48h: " + "; ".join(
            f"{sp} {c['with_model_rows']}/{c['events_48h']} games ({c['model']})" for sp, c in cov.items()))
    except Exception as e:                                     # noqa: BLE001
        lines.append(f"  coverage unavailable ({str(e)[:80]})")
    try:
        rows = con.execute("""select sport, coalesce(json_extract(card_tags,'$.kind'),'?') kind, count(*) n,
                                     sum(result='pending') p, sum(result='W') w, sum(result='L') l, sum(result='P') pu,
                                     round(sum(case when result in ('W','L','P') then pnl else 0 end),2) u,
                                     round(sum(case when result in ('W','L','P') then stake else 0 end),2) st,
                                     round(avg(clv_pp),2) clv, sum(clv_pp>0) bc, sum(clv_pp is not null) nc
                              from bets where evidence_kind='ledger' group by sport, kind order by sport, kind""").fetchall()
        if not rows:
            lines.append("  no ledger picks yet (the first ones log 90 min before the next kickoff)")
        for r in rows:
            roi = f"{(r['u'] / r['st']) * 100:+.1f}%" if r["st"] else "n/a"
            bc = f"{r['bc'] / r['nc']:.0%}" if r["nc"] else "n/a"
            lines.append(f"  {r['sport']} {r['kind']}: {r['n']} picks, {r['p']} pending | {r['w']}-{r['l']}-{r['pu']} | "
                         f"{r['u']:+.2f}u on {r['st']:.2f}u ({roi}) | CLV avg {r['clv'] if r['clv'] is not None else 'n/a'}pp, beat close {bc}")
        g = con.execute("""select coalesce(json_extract(card_tags,'$.grade'),'?') g, count(*) n, sum(result='W') w,
                                  sum(result='L') l, round(sum(case when result in ('W','L','P') then pnl else 0 end),2) u
                           from bets where evidence_kind='ledger' and result in ('W','L','P') group by g order by g""").fetchall()
        if g:
            lines.append("  by grade: " + "; ".join(f"{r['g']} {r['w']}-{r['l']} {r['u']:+.2f}u" for r in g))
        up = con.execute("""select sport, game, market, selection, line, price, book, stake, logged_call, commence
                            from bets where evidence_kind='ledger' and result='pending' order by commence limit 40""").fetchall()
        for r in up:
            ln = "" if r["line"] is None else (f" {r['line']:+g}" if r["market"] == "spreads" else f" {r['line']:g}")
            lines.append(f"  pick {r['sport']} {str(r['commence'])[:16]}Z {r['game']} | {r['market']} {r['selection']}{ln} "
                         f"{r['price']:.2f} {r['book']} | {r['logged_call']} {r['stake']:g}u")
        st = con.execute("select v from meta where k='ledger_grade'").fetchone()
        if st:
            lines.append(f"  last grading pass: {st[0]}")
        er = con.execute("select v from meta where k='ledger_error'").fetchone()
        if er:
            lines.append(f"  last ledger error: {er[0]}")
        ss = con.execute("select v from meta where k='soccer_status'").fetchone()
        if ss:
            lines.append(f"  soccer model: {ss[0][:400]}")
    except Exception as e:                                     # noqa: BLE001
        lines.append(f"  ledger report unavailable ({str(e)[:100]})")
    return "\n".join(lines) + "\n"
'''

_EDGE_SRC = r'''"""
kronos_edge — where a Connecticut bettor's edge actually is (edge-v1, 2026-11-01).

1. BET tier (rule in one place, used by the ledger, alt lines and the early scan):
   a pick is a BET only when the price is at DraftKings / FanDuel / Fanatics and either
     * the price beats Pinnacle's de-vigged fair by >= +1.5%  (a pure price edge), or
     * the model's probability gives >= +2% EV at that price (props/alt lines >= +4%) and the price
       is no worse than -1.5% against Pinnacle's fair (the model may not overrule the market by more),
   and the picked side is not the public side and the model gap is under 8%. Grades are now set on EV
   at the bettable price, not on the model's edge against fair.
2. News trigger: ESPN injury lists for NFL / NBA / NHL every 10 minutes for teams playing in the next
   4 hours. A player newly listed Out / Doubtful re-pulls that game's lines and props through the desk's
   own pull code (credits accounted the normal way) and alerts any CT price left stale.
3. Boost checker: send the Telegram bot "boost <bet> <odds>" (legs joined with +) and it replies with
   Pinnacle's fair odds and the boost's EV. "bets" lists today's BET picks. "help" lists commands.
4. Alternate lines (NFL, NBA): CT books' alternate spreads and totals priced off Pinnacle's main line
   with a normal margin/total model; edges >= +4% are logged as paper picks and alerted.
5. Early-week NFL scan: next week's games once lines are posted, logged as 'early' paper picks so their
   closing line value shows whether early numbers are softer.
6. Preflight: on boot, every external call and database write is tested once and reported.
"""
import json, math, os, re, time
from datetime import datetime, timezone, timedelta

UTC = timezone.utc
VERSION = "edge-v1-2026-11-01"
CT = ("draftkings", "fanduel", "fanatics")
SHARP = ("pinnacle", "lowvig", "betonlineag")
BOOKN = {"draftkings": "DraftKings", "fanduel": "FanDuel", "fanatics": "Fanatics", "pinnacle": "Pinnacle"}
PRICE_EDGE = float(os.environ.get("BET_PRICE_EDGE", "0.015"))
MODEL_EV_MAIN = float(os.environ.get("BET_MODEL_EV", "0.02"))
MODEL_EV_PROP = float(os.environ.get("BET_MODEL_EV_PROP", "0.04"))
MKT_FLOOR = float(os.environ.get("BET_MKT_FLOOR", "-0.015"))
BIG_GAP = 0.08
SD = {("NFL", "spreads"): 13.5, ("NFL", "totals"): 10.0, ("NBA", "spreads"): 12.0, ("NBA", "totals"): 17.5}
ALT_MAX_DIST = {"NFL": 7.0, "NBA": 10.0}   # far-off alt lines are where the normal model is least reliable
ALT_MIN_EV = float(os.environ.get("ALT_MIN_EV", "0.04"))
ALT_MIN_EV_NFL = float(os.environ.get("ALT_MIN_EV_NFL", "0.06"))   # NFL key numbers (3, 7) make the normal model rougher
ALT_MAX_CALLS_DAY = int(os.environ.get("ALT_MAX_CALLS_DAY", "60"))
NEWS_MAX_PULLS_DAY = int(os.environ.get("NEWS_MAX_PULLS_DAY", "30"))
ESPN = {"NFL": "football/nfl", "NBA": "basketball/nba", "NHL": "hockey/nhl"}
OUT_WORDS = ("out", "doubtful", "injured reserve", "suspension", "suspended")


def _iso(t):
    return t.isoformat(timespec="seconds")


def _ts(s):
    t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def _meta(con, k, v=None):
    if v is None:
        r = con.execute("select v from meta where k=?", (k,)).fetchone()
        return r[0] if r else None
    con.execute("insert or replace into meta(k,v) values(?,?)", (k, v))


def _match(a, b):
    try:
        import kronos_ledger as kl
        return kl._match(a, b)
    except Exception:                                          # noqa: BLE001
        return 1.0 if str(a).lower() == str(b).lower() else 0.0


def ensure(con):
    con.executescript("""
    create table if not exists news_snap(sport text, team text, player text, status text, ts text,
        primary key(sport, team, player));
    create table if not exists edge_log(ts text, kind text, sport text, event_id text, text text);
    create index if not exists edge_log_ts on edge_log(ts);
    """)


def log(con, kind, sport, eid, text):
    con.execute("insert into edge_log values(?,?,?,?,?)", (_iso(datetime.now(UTC)), kind, sport, eid, str(text)[:600]))


# --------------------------------------------------------------------------- the BET rule
def grade_from_ev(ev, has_model, gap):
    if not has_model:
        g, st = ("C", 0.25) if ev >= PRICE_EDGE else ("D", 0.1)
    elif ev >= 0.05:
        g, st = "B", 0.5            # A is capped at B while the models are unvalidated
    elif ev >= 0.03:
        g, st = "B", 0.5
    elif ev >= 0.015:
        g, st = "C", 0.25
    else:
        g, st = "D", 0.1
    if has_model and gap is not None and gap > BIG_GAP and g == "B":
        g, st = "C", 0.25
    return g, st


def bet_check(price, from_ct, fair_p, model_p=None, prop=False, flag_label="", gap=None):
    """Returns (is_bet, reason, ev_model, ev_market)."""
    ev_mkt = fair_p * price - 1 if fair_p else None
    ev_mod = (model_p * price - 1) if model_p is not None else ev_mkt
    if not from_ct:
        return False, "no CT price", ev_mod, ev_mkt
    if "PUBLIC" in str(flag_label).upper():
        return False, "public side", ev_mod, ev_mkt
    if gap is not None and gap > BIG_GAP:
        return False, "model gap over 8%: check news", ev_mod, ev_mkt
    if ev_mkt is not None and ev_mkt >= PRICE_EDGE:
        return True, f"price beats Pinnacle fair by {ev_mkt:+.1%}", ev_mod, ev_mkt
    need = MODEL_EV_PROP if prop else MODEL_EV_MAIN
    if model_p is not None and ev_mod >= need and (ev_mkt is None or ev_mkt >= MKT_FLOOR):
        return True, f"model EV {ev_mod:+.1%} at this price, market {ev_mkt:+.1%}", ev_mod, ev_mkt
    return False, "no edge at a CT price", ev_mod, ev_mkt


# --------------------------------------------------------------------------- desk handle (reuse the desk's own pull code)
def desk():
    import sys
    d = getattr(sys.modules.get("__main__"), "DESK", None)
    if d is not None and hasattr(d, "pull_props"):
        return d
    import gc
    for o in gc.get_objects():
        try:
            if type(o).__name__ == "Desk" and hasattr(o, "pull_props"):
                return o
        except Exception:                                      # noqa: BLE001
            continue
    return None


def _sport_key(sport):
    import sys
    m = sys.modules.get("__main__")
    try:
        return m.SPORTS[sport]["key"]
    except Exception:                                          # noqa: BLE001
        return {"NFL": "americanfootball_nfl", "NBA": "basketball_nba", "NHL": "icehockey_nhl",
                "EPL": "soccer_epl", "UCL": "soccer_uefa_champs_league", "UFC": "mma_mixed_martial_arts"}[sport]


def pull_event(con, sport, eid, markets, books=None, label="news", now=None):
    """One event-odds request through the desk (credit accounting, storage, tape) — same pattern as pull_props."""
    d = desk()
    if d is None:
        raise RuntimeError("desk not found")
    import sys
    m = sys.modules.get("__main__")
    now = now or datetime.now(UTC)
    nbooks = len((books or getattr(m, "BOOKS", "")).split(",")) or 20
    cost = len(markets.split(",")) * max(1, -(-nbooks // 10))
    if not d.can_spend(con, cost, now, floor=0.0, priority="tape"):
        raise RuntimeError(f"credits: can't spend {cost} now")
    params = {"bookmakers": books or getattr(m, "BOOKS"), "markets": markets, "oddsFormat": "decimal"}
    if not d.lock.acquire(timeout=90):
        raise RuntimeError("desk busy")
    try:
        data, hdr = d.get(f"/sports/{_sport_key(sport)}/events/{eid}/odds", params)
        d._record(con, sport, label, hdr, now)
        rows = []
        for b in data.get("bookmakers", []):
            for mk in b.get("markets", []):
                for o in mk.get("outcomes", []):
                    rows.append((data, b["key"], mk["key"], o["name"], o.get("description", ""), o.get("point"), o["price"]))
        if label != "alt":
            d.store(con, sport, rows, now, False)
        con.commit()
        return data, rows
    finally:
        d.lock.release()


# --------------------------------------------------------------------------- news trigger
def _espn_injuries(sport):
    import requests
    r = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/{ESPN[sport]}/injuries", timeout=20,
                     headers={"User-Agent": "Mozilla/5.0 (KRONOS desk)"})
    r.raise_for_status()
    out = []
    for team in r.json().get("injuries", []) or []:
        tname = (team.get("team") or {}).get("displayName") or team.get("displayName", "")
        for inj in team.get("injuries", []) or []:
            ath = inj.get("athlete") or {}
            name = ath.get("displayName") or ath.get("fullName") or ""
            st = str(inj.get("status") or "").strip()
            if name and st:
                out.append((tname, name, st))
    return out


def stale_sides(con, eid, min_ev=0.02):
    """Sides where a CT book now beats the fresh Pinnacle fair price."""
    import kronos_ledger as kl
    out = []
    for mk in ("h2h", "spreads", "totals"):
        sides = kl.sides(con, eid, mk)
        if len(sides) < 2:
            continue
        fair = kl._devig([s[2] for s in sides])
        for (o, ln, pr, bk), f in zip(sides, fair):
            q = kl.best_ct(con, eid, mk, o, ln)
            if q and q["price"] * f - 1 >= min_ev:
                out.append((mk, o, ln, q["book"], q["price"], q["price"] * f - 1, f))
    return out


def news_scan(con, tg=None, now=None):
    ensure(con)
    now = now or datetime.now(UTC)
    hi = _iso(now + timedelta(hours=4))
    report = {}
    for sp in ESPN:
        evs = [dict(r) for r in con.execute("select * from events where sport=? and commence>? and commence<?",
                                            (sp, _iso(now), hi))]
        if not evs:
            continue
        try:
            rows = _espn_injuries(sp)
        except Exception as e:                                 # noqa: BLE001
            report[sp] = f"error {str(e)[:60]}"
            continue
        first = not con.execute("select 1 from news_snap where sport=? limit 1", (sp,)).fetchone()
        prev = {(r[0], r[1]): r[2] for r in con.execute("select team, player, status from news_snap where sport=?", (sp,))}
        changed = []
        for team, player, st in rows:
            was = prev.get((team, player))
            is_out = any(w in st.lower() for w in OUT_WORDS)
            was_out = was is not None and any(w in was.lower() for w in OUT_WORDS)
            if is_out and not was_out and not first:
                changed.append((team, player, st))
        con.execute("delete from news_snap where sport=?", (sp,))
        con.executemany("insert or replace into news_snap values(?,?,?,?,?)", [(sp, t, p, s, _iso(now)) for t, p, s in rows])
        con.commit()
        report[sp] = {"listed": len(rows), "new_outs": len(changed)}
        for team, player, st in changed:
            ev = next((e for e in evs if max(_match(team, e["home"]), _match(team, e["away"])) >= 0.5), None)
            if not ev:
                continue
            day = now.strftime("%Y%m%d")
            used = int(_meta(con, f"news_pulls:{day}") or 0)
            k = f"news:{ev['event_id']}:{player}"
            if _meta(con, k):
                continue
            _meta(con, k, _iso(now))
            msg = [f"KRONOS NEWS: {player} ({team}) now listed {st}", f"{ev['away']} @ {ev['home']}, {_et(ev['commence'])}"]
            if used < NEWS_MAX_PULLS_DAY:
                try:
                    import sys
                    m = sys.modules.get("__main__")
                    pull_event(con, sp, ev["event_id"], "h2h,spreads,totals", label="news", now=now)
                    props = ",".join(m.SPORTS[sp].get("props", [])) if getattr(m, "SPORTS", None) else ""
                    if props:
                        pull_event(con, sp, ev["event_id"], props, label="news-props", now=now)
                    _meta(con, f"news_pulls:{day}", str(used + 1))
                    st_sides = stale_sides(con, ev["event_id"])
                    if st_sides:
                        msg.append("Stale CT prices after the re-pull:")
                        for mk, o, ln, bk, pr, ev_, f in st_sides[:5]:
                            lns = "" if ln is None else (f" {ln:+g}" if mk == "spreads" else f" {ln:g}")
                            msg.append(f"• {o}{lns} ({mk}) {BOOKN.get(bk, bk)} {pr:.2f}, fair {1 / f:.2f}, EV {ev_:+.1%}")
                    else:
                        msg.append("Prices re-pulled: no CT book is stale on game lines right now.")
                except Exception as e:                         # noqa: BLE001
                    msg.append(f"Couldn't re-pull prices: {str(e)[:80]}")
            else:
                msg.append(f"Daily re-pull limit ({NEWS_MAX_PULLS_DAY}) reached; check prices by hand.")
            log(con, "news", sp, ev["event_id"], " | ".join(msg))
            con.commit()
            if tg:
                tg("\n".join(msg))
    _meta(con, "news_status", json.dumps({"ts": _iso(now), **report})[:500])
    con.commit()
    return report


def _et(c):
    try:
        t = _ts(c).astimezone(timezone(timedelta(hours=-5)))
        return t.strftime("%a %-I:%M %p ET")
    except Exception:                                          # noqa: BLE001
        return str(c)


# --------------------------------------------------------------------------- alternate lines
def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _phinv(p):
    # Acklam-free: bisection is plenty here
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def alt_prob(sport, market, main_line, main_fair, alt_line, side_is_main_side=True):
    """P(win) at alt_line given the main line and its fair probability (same side), normal model."""
    sd = SD[(sport, market)]
    if market == "spreads":
        # side covers if margin_side + line > 0; P = Phi((mu + line)/sd) where mu = side's expected margin
        mu = sd * _phinv(main_fair) - main_line
        return _phi((mu + alt_line) / sd)
    # totals, Over side: P(T > L) = 1 - Phi((L - mu)/sd)
    mu = main_line + sd * _phinv(main_fair)
    return 1 - _phi((alt_line - mu) / sd)


def alt_scan(con, tg=None, now=None):
    """NFL + NBA games starting within 6h: CT alternate lines vs Pinnacle's main line, once per game per 2h."""
    import kronos_ledger as kl
    now = now or datetime.now(UTC)
    day = now.strftime("%Y%m%d")
    used = int(_meta(con, f"alt_calls:{day}") or 0)
    found = 0
    for sp in ("NFL", "NBA"):
        evs = [dict(r) for r in con.execute("select * from events where sport=? and commence>? and commence<?",
                                            (sp, _iso(now + timedelta(minutes=20)), _iso(now + timedelta(hours=6))))]
        for ev in evs:
            if used >= ALT_MAX_CALLS_DAY:
                break
            k = f"alt_pull:{ev['event_id']}"
            last = _meta(con, k)
            if last and _ts(last) > now - timedelta(hours=2):
                continue
            try:
                _d, rows = pull_event(con, sp, ev["event_id"], "alternate_spreads,alternate_totals",
                                      books=",".join(CT), label="alt", now=now)
            except Exception as e:                             # noqa: BLE001
                log(con, "alt_error", sp, ev["event_id"], str(e)[:120]); con.commit()
                continue
            used += 1
            _meta(con, f"alt_calls:{day}", str(used)); _meta(con, k, _iso(now)); con.commit()
            for mk_main, mk_alt in (("spreads", "alternate_spreads"), ("totals", "alternate_totals")):
                main = kl.sides(con, ev["event_id"], mk_main)
                if len(main) != 2 or not str(main[0][3]) in SHARP:
                    continue
                fair = kl._devig([s[2] for s in main])
                ref = {s[0]: (s[1], f) for s, f in zip(main, fair)}
                for _g, bk, mk, oc, _pl, ln, pr in rows:
                    if mk != mk_alt or ln is None or oc not in ref:
                        continue
                    mline, mfair = ref[oc]
                    if mk_main == "totals":
                        over_line, over_fair = ref["Over"] if "Over" in ref else (None, None)
                        if over_line is None:
                            continue
                        p_over = alt_prob(sp, "totals", float(over_line), over_fair, float(ln))
                        p = p_over if oc == "Over" else 1 - p_over
                        dist = abs(float(ln) - float(over_line))
                    else:
                        p = alt_prob(sp, "spreads", float(mline), mfair, float(ln))
                        dist = abs(float(ln) - float(mline))
                    if dist < 0.5 or dist > ALT_MAX_DIST[sp]:
                        continue
                    evv = p * float(pr) - 1
                    if evv < (ALT_MIN_EV_NFL if sp == "NFL" else ALT_MIN_EV):
                        continue
                    key = (ev["event_id"], mk_main, oc, float(ln))
                    if con.execute("""select 1 from bets where evidence_kind='alt' and event_id=? and market=? and selection=?
                                      and line=?""", key).fetchone():
                        continue
                    note = (f"KRONOS alt line | {BOOKN.get(bk, bk)} {mk_alt} {oc} {ln:+g} at {pr:.2f} | fair {p:.1%} from Pinnacle's "
                            f"main {mline:+g} ({mfair:.1%}) with a normal model (sd {SD[(sp, mk_main)]}) | EV {evv:+.1%} | "
                            "approximation: ignores key numbers; check the price before betting")
                    con.execute("""insert into bets(placed_ts, sport, event_id, game, commence, market, selection, player, line, price,
                                   book, stake, model_prob, note, paper, model_version, edge, fair_prob, evidence_kind, logged_call, card_tags)
                                   values(?,?,?,?,?,?,?,'',?,?,?,?,?,?,2,?,?,?,'alt',?,?)""",
                                (_iso(now), sp, ev["event_id"], f"{ev['away']} @ {ev['home']}", ev["commence"], mk_main, oc,
                                 float(ln), float(pr), bk, 0.25, round(p, 4), note, VERSION, round(evv, 4), round(p, 4),
                                 "ALT EDGE", json.dumps({"grade": "C", "kind": "alt", "bet": True})))
                    log(con, "alt", sp, ev["event_id"], note)
                    con.commit()
                    found += 1
                    if tg:
                        lns = f"{ln:+g}" if mk_main == "spreads" else f"{ln:g}"
                        tg(f"KRONOS ALT LINE: {ev['away']} @ {ev['home']} ({_et(ev['commence'])})\n{oc} {lns} at "
                           f"{BOOKN.get(bk, bk)} {pr:.2f}\nFair {1 / p:.2f} (from Pinnacle's main line), EV {evv:+.1%}\n"
                           "Rough model: check the number before betting.")
    return {"alt_calls_today": used, "edges": found}


# --------------------------------------------------------------------------- early-week NFL scan
def early_scan(con, L=None, jr=None, js=None, tg=None, now=None):
    import kronos_ledger as kl
    now = now or datetime.now(UTC)
    evs = [dict(r) for r in con.execute("select * from events where sport='NFL' and commence>? and commence<?",
                                        (_iso(now + timedelta(hours=30)), _iso(now + timedelta(days=8))))]
    n = bets = 0
    for ev in evs:
        for mk in ("h2h", "spreads", "totals"):
            if con.execute("select 1 from bets where evidence_kind='early' and event_id=? and market=?",
                           (ev["event_id"], mk)).fetchone():
                continue
            outs = kl.sides(con, ev["event_id"], mk)
            if len(outs) < 2:
                continue
            fair = kl._devig([o[2] for o in outs]); fp = {o[0]: f for o, f in zip(outs, fair)}
            mp = kl.model_probs_for(con, ev["event_id"], mk, outs)
            best = None
            for o in outs:
                q = kl.best_ct(con, ev["event_id"], mk, o[0], o[1])
                if not q:
                    continue
                ok, why, evm, evk = bet_check(q["price"], True, fp[o[0]], mp.get(o[0]))
                score = evm if evm is not None else -9
                if best is None or score > best[0]:
                    best = (score, o, q, ok, why, evk)
            if not best:
                continue
            score, o, q, ok, why, evk = best
            note = (f"KRONOS early | {why} | model EV {score:+.1%}, market EV {evk:+.1%} | logged {(_ts(ev['commence']) - now).days}d out "
                    "to measure whether early numbers beat the close")
            con.execute("""insert into bets(placed_ts, sport, event_id, game, commence, market, selection, player, line, price,
                           book, stake, model_prob, note, paper, model_version, edge, fair_prob, evidence_kind, logged_call, card_tags)
                           values(?,?,?,?,?,?,?,'',?,?,?,?,?,?,2,?,?,?,'early',?,?)""",
                        (_iso(now), "NFL", ev["event_id"], f"{ev['away']} @ {ev['home']}", ev["commence"], mk, o[0], o[1],
                         q["price"], q["book"], 0.25 if ok else 0.1, round(mp.get(o[0], fp[o[0]]), 4), note[:600], VERSION,
                         round(score, 4), round(fp[o[0]], 4), "BET" if ok else "early lean",
                         json.dumps({"kind": "early", "bet": ok})))
            n += 1
            if ok:
                bets += 1
                if tg:
                    ln = "" if o[1] is None else (f" {o[1]:+g}" if mk == "spreads" else f" {o[1]:g}")
                    tg(f"KRONOS EARLY BET: {ev['away']} @ {ev['home']} ({_et(ev['commence'])})\n{mk}: {o[0]}{ln} at "
                       f"{BOOKN.get(q['book'], q['book'])} {q['price']:.2f}\n{why}")
    con.commit()
    return {"logged": n, "bets": bets}


# --------------------------------------------------------------------------- boost checker (Telegram)
MK_WORDS = [
    (r"pass(ing)? ?(yds|yards)", "player_pass_yds"), (r"pass(ing)? ?(td|tds|touchdowns)", "player_pass_tds"),
    (r"rush(ing)? ?(yds|yards)?", "player_rush_yds"), (r"rec(eiving)? ?(yds|yards)", "player_reception_yds"),
    (r"receptions|catches", "player_receptions"), (r"anytime|td scorer|to score a td|touchdown", "player_anytime_td"),
    (r"\bpoints\b|\bpts\b", "player_points"), (r"rebounds|\breb\b", "player_rebounds"), (r"assists|\bast\b", "player_assists"),
    (r"blocks|\bblk\b", "player_blocks"), (r"shots on goal|\bsog\b", "player_shots_on_goal"),
    (r"to score|goal ?scorer|anytime goal", "player_goal_scorer_anytime"),
]


def _parse_odds(tok):
    tok = tok.strip()
    if re.fullmatch(r"[+-]\d{3,4}", tok):
        a = int(tok)
        return 1 + (a / 100 if a > 0 else 100 / -a)
    try:
        v = float(tok)
        return v if v > 1.0 else None
    except ValueError:
        return None


def _devig_pair(a, b):
    ia, ib = 1 / a, 1 / b
    return ia / (ia + ib)


def price_leg(con, text, now=None):
    """Find the bet described in `text` on the board and return its fair probability."""
    now = now or datetime.now(UTC)
    t = text.lower().strip()
    lo, hi = _iso(now - timedelta(hours=4)), _iso(now + timedelta(hours=72))
    evs = [dict(r) for r in con.execute("select * from events where commence>? and commence<?", (lo, hi))]
    num = re.findall(r"[+-]?\d+(?:\.\d+)?", t)
    line = float(num[-1]) if num else None
    side_ou = "Over" if re.search(r"\bover\b|\bo\d|\+$", t) else "Under" if re.search(r"\bunder\b|\bu\d", t) else None
    market = next((m for pat, m in MK_WORDS if re.search(pat, t)), None)
    stop = {"ml", "moneyline", "to", "win", "wins", "over", "under", "o", "u", "pass", "passing", "rush", "rushing", "rec",
            "receiving", "yds", "yards", "yard", "receptions", "catches", "anytime", "td", "tds", "touchdown", "touchdowns",
            "scorer", "score", "a", "points", "pts", "rebounds", "reb", "assists", "ast", "blocks", "blk", "shots", "on", "goal",
            "sog", "goals", "spread", "total", "alt", "the", "and", "plus", "+"}
    words = " ".join(w for w in re.sub(r"[^a-z' .-]", " ", t).split() if w not in stop and not re.fullmatch(r"[\d.]+", w))

    def score(name):
        q = set(words.split()); c = set(re.sub(r"[^a-z' .-]", " ", str(name).lower()).split())
        if not q or not c:
            return 0.0
        return len(q & c) / len(q) + 0.01 * len(q & c) / len(q | c)
    # player props
    if market:
        best = None
        for ev in evs:
            for r in con.execute("""select distinct player from last_quote where event_id=? and market=? and player<>''""",
                                 (ev["event_id"], market)):
                s_ = score(r[0])
                if best is None or s_ > best[0]:
                    best = (s_, ev, r[0])
        if not best or best[0] < 0.5:
            return {"ok": False, "why": "couldn't find that player prop on the board"}
        _s, ev, player = best
        q = {}
        for r in con.execute("""select book, outcome, line, price from last_quote where event_id=? and market=? and player=?""",
                             (ev["event_id"], market, player)):
            q.setdefault(r[0], []).append((r[1], r[2], r[3]))
        want = "Yes" if (market.endswith("anytime_td") or market.endswith("scorer_anytime")) else (side_ou or "Over")
        for bk in SHARP:
            rows = q.get(bk, [])
            mine = [r for r in rows if r[0] == want and (line is None or want == "Yes" or r[1] is None or abs(float(r[1]) - line) < 1e-9)]
            other = [r for r in rows if r[0] != want and (mine and r[1] == mine[0][1])]
            if mine and other:
                f = _devig_pair(mine[0][2], other[0][2])
                return {"ok": True, "desc": f"{player} {want} {'' if mine[0][1] is None else mine[0][1]} {market.replace('player_', '').replace('_', ' ')}",
                        "fair": f, "src": BOOKN.get(bk, bk), "event": ev}
        # one-sided (anytime TD etc.): average implied across books, minus a typical 6% prop margin
        imps = [1 / r[2] for rows in q.values() for r in rows if r[0] == want and
                (line is None or want == "Yes" or r[1] is None or abs(float(r[1]) - line) < 1e-9)]
        if imps:
            f = sum(imps) / len(imps) / 1.06
            return {"ok": True, "desc": f"{player} {want} {market.replace('player_', '').replace('_', ' ')}", "fair": f,
                    "src": f"{len(imps)}-book average, margin removed (approximate)", "event": ev}
        return {"ok": False, "why": f"found {player} but no price at that line"}
    # game lines: team ML / spread / total
    best = None
    for ev in evs:
        for team in (ev["home"], ev["away"]):
            s_ = score(team)
            if best is None or s_ > best[0]:
                best = (s_, ev, team)
    if not best or best[0] < 0.5:
        return {"ok": False, "why": "couldn't find that game on the board"}
    _s, ev, team = best
    import kronos_ledger as kl
    if side_ou:
        sides = kl.sides(con, ev["event_id"], "totals")
        if len(sides) == 2:
            fair = kl._devig([s[2] for s in sides]); d = {s[0]: (s[1], f) for s, f in zip(sides, fair)}
            ml, f = d[side_ou]
            if line is not None and ml is not None and abs(line - float(ml)) > 1e-9 and (ev["sport"], "totals") in SD:
                fo = d["Over"][1]
                p_over = alt_prob(ev["sport"], "totals", float(d["Over"][0]), fo, line)
                f = p_over if side_ou == "Over" else 1 - p_over
                return {"ok": True, "desc": f"{ev['away']} @ {ev['home']} {side_ou} {line:g}", "fair": f,
                        "src": f"Pinnacle total {ml:g}, converted to {line:g} (approximate)", "event": ev}
            return {"ok": True, "desc": f"{ev['away']} @ {ev['home']} {side_ou} {ml:g}", "fair": f, "src": "Pinnacle", "event": ev}
    if line is not None and re.search(r"[+-]\d", t) and not re.search(r"\bml\b|moneyline|to win", t):
        sides = kl.sides(con, ev["event_id"], "spreads")
        if len(sides) != 2:
            return {"ok": False, "why": "no spread on the board for that game yet"}
        if len(sides) == 2:
            fair = kl._devig([s[2] for s in sides]); d = {s[0]: (s[1], f) for s, f in zip(sides, fair)}
            if team in d:
                ml, f = d[team]
                if abs(line - float(ml)) > 1e-9 and (ev["sport"], "spreads") in SD:
                    f = alt_prob(ev["sport"], "spreads", float(ml), f, line)
                    return {"ok": True, "desc": f"{team} {line:+g}", "fair": f,
                            "src": f"Pinnacle spread {ml:+g}, converted (approximate)", "event": ev}
                return {"ok": True, "desc": f"{team} {ml:+g}", "fair": f, "src": "Pinnacle", "event": ev}
    sides = kl.sides(con, ev["event_id"], "h2h")
    if sides:
        fair = kl._devig([s[2] for s in sides]); d = {s[0]: f for s, f in zip(sides, fair)}
        if team in d:
            return {"ok": True, "desc": f"{team} to win", "fair": d[team], "src": "Pinnacle", "event": ev}
    return {"ok": False, "why": "found the game but not that market"}


def boost_reply(con, text, now=None):
    body = re.sub(r"^/?boost\s*", "", text.strip(), flags=re.I)
    toks = body.split()
    if not toks:
        return "Send: boost <bet> <odds>. Example: boost Patriots ML + Bills ML 3.5"
    odds = _parse_odds(toks[-1])
    if odds is None:
        return "I need the boosted odds at the end, e.g. 'boost Josh Allen over 249.5 pass yds 2.10'."
    legs = [l.strip() for l in re.split(r"\s\+\s|\s&\s|\n", " ".join(toks[:-1])) if l.strip()]
    priced, p = [], 1.0
    for l in legs:
        r = price_leg(con, l, now)
        if not r["ok"]:
            return f"Leg '{l}': {r['why']}. Try the player or team name as it appears on the books."
        priced.append(r); p *= r["fair"]
    evv = p * odds - 1
    lines = ["KRONOS BOOST CHECK"]
    for r in priced:
        lines.append(f"• {r['desc']}: fair {r['fair']:.1%} ({1 / r['fair']:.2f}) from {r['src']}")
    if len(priced) > 1:
        lines.append(f"Combined fair {p:.1%} ({1 / p:.2f}), treating legs as independent: same-game legs that move together make this optimistic or pessimistic.")
    verdict = "TAKE IT" if evv >= 0.05 else "small edge" if evv > 0 else "PASS"
    lines.append(f"Boost {odds:.2f} vs fair {1 / p:.2f}: EV {evv:+.1%}. {verdict}.")
    log(con, "boost", priced[0]["event"]["sport"], priced[0]["event"]["event_id"], " | ".join(lines)); con.commit()
    return "\n".join(lines)


def bets_reply(con, now=None):
    now = now or datetime.now(UTC)
    rows = con.execute("""select sport, game, market, selection, line, price, book, stake, logged_call, commence, note from bets
                          where paper=2 and commence>? and commence<? and (logged_call like 'BET%' or logged_call='ALT EDGE')
                          order by commence""", (_iso(now - timedelta(hours=3)), _iso(now + timedelta(hours=30)))).fetchall()
    if not rows:
        return "No BET picks right now. Everything on today's board is priced at or below fair at your books."
    out = ["KRONOS BETS"]
    for r in rows:
        ln = "" if r["line"] is None else (f" {r['line']:+g}" if r["market"] == "spreads" else f" {r['line']:g}")
        out.append(f"• {_et(r['commence'])} {r['game']}: {r['selection']}{ln} at {BOOKN.get(r['book'], r['book'])} {r['price']:.2f} "
                   f"({r['logged_call']}, {r['stake']:g}u)")
    return "\n".join(out)


HELP = ("KRONOS commands:\n• boost <bet> <odds> — prices a boost, e.g. 'boost Patriots ML + Bills ML 3.5' or "
        "'boost Josh Allen over 249.5 pass yds 2.10'\n• bets — today's BET picks\n• help — this list")


def telegram_poll(get_con, tg_send):
    """Long-poll the bot for commands from your own chat only."""
    import requests
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        return
    c = get_con()
    try:
        off = int(_meta(c, "tg_offset") or 0)
    finally:
        c.close()
    while True:
        try:
            r = requests.get(f"https://api.telegram.org/bot{tok}/getUpdates", params={"timeout": 25, "offset": off + 1}, timeout=35)
            for u in r.json().get("result", []):
                off = max(off, u["update_id"])
                m = u.get("message") or {}
                if str((m.get("chat") or {}).get("id")) != str(chat):
                    continue
                txt = str(m.get("text") or "").strip()
                low = txt.lower()
                c = get_con()
                try:
                    if low.startswith(("boost", "/boost")):
                        reply = boost_reply(c, txt)
                    elif low in ("bets", "/bets"):
                        reply = bets_reply(c)
                    elif low in ("help", "/help", "/start"):
                        reply = HELP
                    else:
                        reply = None
                    _meta(c, "tg_offset", str(off)); c.commit()
                finally:
                    c.close()
                if reply:
                    tg_send(reply)
        except Exception as e:                                 # noqa: BLE001
            print(f"[telegram poll] {e}", flush=True)
            time.sleep(15)


# --------------------------------------------------------------------------- preflight
def preflight(con):
    import requests
    checks = []
    def chk(name, fn):
        try:
            fn(); checks.append((name, True, ""))
        except Exception as e:                                 # noqa: BLE001
            checks.append((name, False, str(e)[:120]))
    chk("database write/read", lambda: (con.execute("create table if not exists preflight_t(x int)"),
                                         con.execute("insert into preflight_t values(1)"),
                                         con.execute("select count(*) from preflight_t").fetchone(),
                                         con.execute("drop table preflight_t"), con.commit()))
    chk("ESPN injuries (NFL)", lambda: _espn_injuries("NFL"))
    chk("ESPN scoreboard (NHL)", lambda: requests.get("https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
                                                      timeout=20).raise_for_status())
    chk("football-data.co.uk", lambda: requests.get("https://www.football-data.co.uk/mmz4281/2627/E0.csv", timeout=30,
                                                    headers={"User-Agent": "Mozilla/5.0"}).raise_for_status())
    tok = os.environ.get("TELEGRAM_BOT_TOKEN")
    if tok:
        chk("Telegram bot", lambda: (lambda r: r.raise_for_status() or (r.json().get("ok") or (_ for _ in ()).throw(Exception(r.text[:80]))))(
            requests.get(f"https://api.telegram.org/bot{tok}/getMe", timeout=15)))
    def _desk():
        d = desk()
        if d is None:
            raise RuntimeError("desk object not found: news re-pulls and alt lines are off")
        if d.remaining is None:
            raise RuntimeError("desk has not read the credit balance yet")
    chk("desk handle + credits", _desk)
    ok = all(c[1] for c in checks)
    _meta(con, "preflight", json.dumps({"ts": _iso(datetime.now(UTC)), "ok": ok,
                                        "checks": [{"name": n, "ok": o, "error": e} for n, o, e in checks]}))
    con.commit()
    return ok, checks


# --------------------------------------------------------------------------- feed
def feed_block(con, now=None):
    now = now or datetime.now(UTC)
    L = ["", "", f"KRONOS EDGE ({VERSION}: BET tier on EV at CT prices, news re-pulls, boost checker, alternate lines, early NFL scan):"]
    try:
        pf = _meta(con, "preflight")
        if pf:
            p = json.loads(pf)
            bad = [c for c in p["checks"] if not c["ok"]]
            L.append(f"  preflight {p['ts']}: " + ("all checks passed" if not bad else "FAILED: " + "; ".join(f"{c['name']}: {c['error']}" for c in bad)))
        L.append("  " + bets_reply(con, now).replace("\n", "\n  "))
        ns = _meta(con, "news_status")
        if ns:
            L.append(f"  news scan: {ns}")
        for r in con.execute("select ts, kind, sport, text from edge_log order by ts desc limit 12"):
            L.append(f"  {r[0]} {r[1]} {r[2]}: {r[3][:300]}")
        g = con.execute("""select evidence_kind, count(*) n, sum(result='W') w, sum(result='L') l,
                                  round(sum(case when result in ('W','L','P') then pnl else 0 end),2) u, round(avg(clv_pp),2) clv
                           from bets where evidence_kind in ('alt','early') group by evidence_kind""").fetchall()
        for r in g:
            L.append(f"  {r[0]}: {r[1]} picks, {r[2] or 0}-{r[3] or 0}, {r[4] or 0:+.2f}u, CLV {r[5]}")
    except Exception as e:                                     # noqa: BLE001
        L.append(f"  edge report unavailable ({str(e)[:100]})")
    return "\n".join(L) + "\n"
'''

_MASTER_SRC = r'''"""
kronos_master — the shared KRONOS engine, batch 1 (master-v1, 2026-11-01).

One rulebook and one learning loop for every sport. This batch adds:
  * Roster layer: ESPN injury reports for NFL, NBA, NHL (MLB when tracked), refreshed hourly, matched to
    each event. Signals out_diff (opponent outs minus our outs), out_total. Names go into the pick note.
  * Schedule layer: each team's recent games from ESPN scoreboards -> rest days, back-to-backs.
    Signals rest_diff, rest_min, b2b_flag.
  * Learning upgrades: autopsy covers the ledger (paper=2) and is split by sport; a decay check demotes a
    graduated signal whose last-60-day effect has faded; a layer scorecard shows, for every signal, the
    win rate and CLV when it is high vs low; a mistake memory turns repeating losing patterns
    (sport x market x flag, n>=20, ROI < -10%, CLV < -1pp) into candidate rules on probation, never
    applied automatically.
  * Backtest harness (soccer first): football-data.co.uk seasons with Pinnacle closing odds, walk-forward
    weekly refits, model vs market log loss and CLV of model picks. Runs nightly; /api/backtest?sport=EPL.
  * Rulebook: the universal decision rules, printed in the feed so every report quotes the same ones.
The matchup layer (player vs defender / scheme) is batch 2 and is NOT in this file.
"""
import csv, io, json, math, os, re
from datetime import datetime, timezone, timedelta, date

UTC = timezone.utc
VERSION = "master-v1-2026-11-01"
ESPN = {"NFL": "football/nfl", "NBA": "basketball/nba", "NHL": "hockey/nhl", "MLB": "baseball/mlb"}
NEW_SIGNALS = ["out_diff", "out_total", "rest_diff", "rest_min", "b2b_flag"]
OUT_STATUSES = ("out", "doubtful", "injured reserve", "ir", "suspension", "suspended")
RULEBOOK = [
    "R1 Market first: Pinnacle's de-vigged price is the baseline; the model moves off it only where the ledger shows it has earned to.",
    "R2 Every tracked match gets a pick in every game-line market, logged 90 min before start. Never 'no bet'; no line = zero-stake pick.",
    "R3 Grades: A 1u (edge >= 5%), B 0.5u (>= 3%), C 0.25u (>= 1.5%), D 0.1u. Unvalidated models capped at B; gaps > 8% capped at C (check news).",
    "R4 Prices: best of DraftKings / FanDuel / Fanatics at the same line; otherwise Pinnacle, marked 'no CT price'. Decimal odds only.",
    "R5 Every signal starts at weight 0. It graduates after 20 graded picks with a clear effect (|z| >= 2) against the market baseline, and is demoted when its last 60 days fade.",
    "R6 Closing line value judges a pick before the result does. A model earns weight after 150 graded picks that beat the close.",
    "R7 Dropping odds: only a sharp-led drop is evidence; retail-only drops are public money; priced-in drops are not chased; act on stale CT prices only.",
    "R8 Every pick states which layers drove it (market, team strength, roster, matchup, game flow) and what each one lacked.",
    "R9 Repeating losing patterns become candidate rules on probation (mistake memory); nothing is applied until it is proven.",
    "R10 Soccer reports use Sports Mole previews; soccer 1X2 settles on 90 minutes; props need confirmed lineups first.",
]


def _iso(t):
    return t.isoformat(timespec="seconds")


def _get(url, timeout=20):
    import requests
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (KRONOS desk)"})
    r.raise_for_status()
    return r.json()


def _match(a, b):
    try:
        import kronos_ledger as kl
        return kl._match(a, b)
    except Exception:                                          # noqa: BLE001
        return 1.0 if str(a).lower() == str(b).lower() else 0.0


def ensure(con):
    # injuries uses the desk's existing table (jerry_feeds already fills NBA + NHL hourly); this module
    # only adds NFL (and MLB when tracked) in the same shape. Fixed 2026-11-01 after a schema clash.
    con.executescript("""
    create table if not exists injuries(sport text, player_norm text, player text, team text,
        status text, detail text, ts text, primary key(sport, player_norm));
    create table if not exists team_games(sport text, team text, game_date text, opponent text, home int,
        primary key(sport, team, game_date));
    create table if not exists master_fetch(k text primary key, ts text);
    """)


# --------------------------------------------------------------------------- roster layer
OWN_INJURY_SPORTS = ("NFL", "MLB")     # NBA + NHL injuries are already ingested by jerry_feeds


def _pnorm(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def pull_injuries(con, sport, get=_get):
    if sport not in OWN_INJURY_SPORTS:
        r = con.execute("select count(*) from injuries where sport=?", (sport,)).fetchone()
        return r[0] if r else 0
    js = get(f"https://site.api.espn.com/apis/site/v2/sports/{ESPN[sport]}/injuries")
    rows = []
    for team in js.get("injuries", []) or js.get("teams", []) or []:
        tname = (team.get("team") or {}).get("displayName") or team.get("displayName", "")
        for inj in team.get("injuries", []) or []:
            ath = inj.get("athlete") or {}
            name = ath.get("displayName") or ath.get("fullName") or ""
            status = str(inj.get("status") or (inj.get("type") or {}).get("description") or "").strip()
            if not name or not status:
                continue
            det = inj.get("details") or {}
            rows.append((sport, _pnorm(name), name, tname, status,
                         str(det.get("type") or inj.get("shortComment") or "")[:160]))
    now = _iso(datetime.now(UTC))
    con.execute("delete from injuries where sport=?", (sport,))
    con.executemany("insert or replace into injuries values(?,?,?,?,?,?,?)", [r + (now,) for r in rows])
    con.execute("insert or replace into master_fetch values(?,?)", (f"inj:{sport}", now))
    con.commit()
    return len(rows)


def team_outs(con, sport, team_name):
    """Players listed Out / Doubtful / IR for the team that best matches this name."""
    teams = [r[0] for r in con.execute("select distinct team from injuries where sport=?", (sport,))]
    if not teams:
        return None
    best, sc = None, 0.0
    for t in teams:
        s = _match(team_name, t)
        if s > sc:
            best, sc = t, s
    if sc < 0.5:
        return None
    outs, q = [], []
    for r in con.execute("select player, status, detail from injuries where sport=? and team=?", (sport, best)):
        st = str(r[1]).lower()
        if any(k in st for k in OUT_STATUSES):
            outs.append(r[0])
        elif "quest" in st or "day" in st or "gtd" in st:
            q.append(r[0])
    return {"team": best, "out": outs, "questionable": q}


# --------------------------------------------------------------------------- schedule layer
def pull_schedule(con, sport, days_back=10, get=_get):
    """Recent completed games per team (for rest days / back-to-backs), one ESPN call per day."""
    now = datetime.now(UTC)
    last = con.execute("select ts from master_fetch where k=?", (f"sched:{sport}",)).fetchone()
    start = (now - timedelta(days=days_back)).date()
    if last:
        start = max(start, (datetime.fromisoformat(last[0]) - timedelta(days=2)).date())
    d, calls = start, 0
    while d <= now.date():
        try:
            js = get(f"https://site.api.espn.com/apis/site/v2/sports/{ESPN[sport]}/scoreboard?dates={d.strftime('%Y%m%d')}")
            calls += 1
            for ev in js.get("events", []):
                comp = (ev.get("competitions") or [{}])[0]
                st = (comp.get("status") or {}).get("type", {})
                if not st.get("completed"):
                    continue
                t = {c.get("homeAway"): (c.get("team") or {}).get("displayName", "") for c in comp.get("competitors", [])}
                gd = str(ev.get("date", ""))[:10]
                if "home" in t and "away" in t:
                    con.execute("insert or replace into team_games values(?,?,?,?,?)", (sport, t["home"], gd, t["away"], 1))
                    con.execute("insert or replace into team_games values(?,?,?,?,?)", (sport, t["away"], gd, t["home"], 0))
            con.commit()
        except Exception:                                      # noqa: BLE001
            pass
        d += timedelta(days=1)
    con.execute("insert or replace into master_fetch values(?,?)", (f"sched:{sport}", _iso(now)))
    con.commit()
    return calls


def rest_days(con, sport, team_name, commence):
    teams = [r[0] for r in con.execute("select distinct team from team_games where sport=?", (sport,))]
    best, sc = None, 0.0
    for t in teams:
        s = _match(team_name, t)
        if s > sc:
            best, sc = t, s
    if sc < 0.5:
        return None
    try:
        day = datetime.fromisoformat(str(commence).replace("Z", "+00:00")).astimezone(timezone(timedelta(hours=-5))).date()
    except Exception:                                          # noqa: BLE001
        return None
    r = con.execute("select max(game_date) from team_games where sport=? and team=? and game_date<?",
                    (sport, best, day.isoformat())).fetchone()
    if not r or not r[0]:
        return None
    return (day - date.fromisoformat(r[0])).days


def event_context(con, ev):
    """Roster + schedule read for an event: {'text':..., 'home':{...}, 'away':{...}}."""
    sp = ev["sport"]
    out = {"home": {}, "away": {}, "text": ""}
    if sp not in ESPN:
        return out
    bits = []
    for side in ("home", "away"):
        o = team_outs(con, sp, ev[side])
        rd = rest_days(con, sp, ev[side], ev["commence"])
        out[side] = {"out": (o or {}).get("out", []), "q": (o or {}).get("questionable", []), "rest": rd}
        if o and (o["out"] or o["questionable"]):
            s = f"{ev[side]}: out {', '.join(o['out'][:4])}" if o["out"] else f"{ev[side]}:"
            if len(o["out"]) > 4:
                s += f" +{len(o['out']) - 4}"
            if o["questionable"]:
                s += f"; questionable {', '.join(o['questionable'][:3])}"
            bits.append(s)
        if rd is not None and rd <= 1:
            bits.append(f"{ev[side]} on a back-to-back" if rd == 1 else f"{ev[side]} played today")
    out["text"] = "; ".join(bits)
    return out


def side_signals(con, ev, market, outcome):
    """The five roster/schedule signals, relative to the picked side (totals: team-neutral versions)."""
    ctx = event_context(con, ev)
    h, a = ctx["home"], ctx["away"]
    oh, oa = len(h.get("out") or []), len(a.get("out") or [])
    rh, ra = h.get("rest"), a.get("rest")
    have_out = bool(h or a) and ev["sport"] in ESPN and con.execute("select 1 from injuries where sport=? limit 1", (ev["sport"],)).fetchone()
    sig = {"out_total": (oh + oa) if have_out else None,
           "rest_min": min(x for x in (rh, ra) if x is not None) if (rh is not None or ra is not None) else None,
           "out_diff": None, "rest_diff": None, "b2b_flag": None}
    if market in ("h2h", "spreads") and outcome in (ev["home"], ev["away"]):
        mine_h = outcome == ev["home"]
        if have_out:
            sig["out_diff"] = (oa - oh) if mine_h else (oh - oa)
        mine, theirs = (rh, ra) if mine_h else (ra, rh)
        if rh is not None and ra is not None:
            sig["rest_diff"] = mine - theirs
        if mine is not None and mine <= 1:
            sig["b2b_flag"] = 1
        elif theirs is not None and theirs <= 1:
            sig["b2b_flag"] = -1
        elif mine is not None and theirs is not None:
            sig["b2b_flag"] = 0
    return sig, ctx


def refresh(con, now=None, get=_get):
    """Hourly: injuries for every ESPN sport, recent schedules."""
    ensure(con)
    now = now or datetime.now(UTC)
    st = {}
    for sp in ESPN:
        if not con.execute("select 1 from events where sport=? and commence>? limit 1", (sp, _iso(now - timedelta(days=1)))).fetchone():
            continue
        try:
            st[sp] = {"injuries": pull_injuries(con, sp, get), "sched_calls": pull_schedule(con, sp, get=get)}
        except Exception as e:                                 # noqa: BLE001
            st[sp] = f"error: {str(e)[:80]}"
    con.execute("insert or replace into meta(k,v) values(?,?)", ("master_status", json.dumps({"ts": _iso(now), **st})[:1500]))
    con.commit()
    return st


# --------------------------------------------------------------------------- learning upgrades
def autopsy(con, days=14):
    """Every graded paper bet (model rows and ledger rows) sorted into a cause, overall and by sport."""
    since = _iso(datetime.now(UTC) - timedelta(days=days))
    causes, by_sport = {}, {}
    for b in con.execute("""select * from bets where paper in (1,2) and result in ('W','L') and graded_ts>?""", (since,)):
        try:
            sg = json.loads(b["signals"] or "{}")
        except Exception:                                      # noqa: BLE001
            sg = {}
        note = str(b["note"] or "")
        if b["result"] == "W":
            c = "win: value held" if (b["clv_pp"] or 0) >= 0 else "win: despite losing value"
        elif b["clv_pp"] is not None and b["clv_pp"] < -1.0:
            c = "loss: bad price (lost value to the close)"
        elif sg.get("role_change") == 1 or (sg.get("snap_pct") is not None and sg["snap_pct"] < 40):
            c = "loss: role / minutes"
        elif sg.get("b2b_flag") == 1:
            c = "loss: schedule spot (back-to-back)"
        elif (sg.get("out_diff") or 0) < 0:
            c = "loss: roster (more outs than the opponent)"
        elif "settled" in note and b["line"] is not None and re.search(r"settled ([\d.]+) vs", note):
            v = float(re.search(r"settled ([\d.]+) vs", note).group(1))
            c = "loss: narrow (variance)" if abs(v - float(b["line"])) <= max(1.0, 0.1 * abs(float(b["line"]))) else "loss: clear miss"
        else:
            c = "loss: other / game script"
        causes[c] = causes.get(c, 0) + 1
        d = by_sport.setdefault(b["sport"], {})
        d[c] = d.get(c, 0) + 1
    con.execute("insert or replace into meta(k,v) values(?,?)", ("autopsy_by_sport", json.dumps(by_sport)))
    con.commit()
    return causes


def layer_scorecard(con, js):
    """For each signal: picks, win rate and CLV when the signal is above its median vs below."""
    rows = js._graded_rows(con)
    out = {}
    for k in js.SIGNALS:
        vals = [(r[2].get(k), r[0], r[3]["clv_pp"]) for r in rows if r[2].get(k) is not None]
        if len(vals) < 10:
            out[k] = {"n": len(vals)}
            continue
        xs = sorted(v[0] for v in vals)
        med = xs[len(xs) // 2]
        hi = [v for v in vals if v[0] > med]; lo = [v for v in vals if v[0] <= med]
        if not hi or not lo:
            out[k] = {"n": len(vals), "note": "no spread"}
            continue
        def summ(g):
            clv = [v[2] for v in g if v[2] is not None]
            return {"n": len(g), "win": round(sum(v[1] for v in g) / len(g), 3),
                    "clv": round(sum(clv) / len(clv), 2) if clv else None}
        out[k] = {"n": len(vals), "median": round(med, 3) if isinstance(med, float) else med, "high": summ(hi), "low": summ(lo)}
    con.execute("insert or replace into meta(k,v) values(?,?)", ("layer_scorecard", json.dumps(out)))
    con.commit()
    return out


def decay_check(con, js, weights, days=60):
    """Demote a graduated signal whose effect over the last `days` has faded (|z| < 1)."""
    since = _iso(datetime.now(UTC) - timedelta(days=days))
    rows = [r for r in js._graded_rows(con) if (r[3]["graded_ts"] or "") > since] if js._graded_rows(con) else []
    status = weights.get("status", {})
    changed = []
    for k, st in list(status.items()):
        if st != "graduated":
            continue
        sub = [r for r in rows if r[2].get(k) is not None]
        if len(sub) < js.GRADUATE_N:
            continue
        try:
            w1, se1 = js._fit(sub, [k], weights["stats"], l2=1.0)
            z = w1[k] / se1[k] if se1[k] not in (0, float("inf")) else 0.0
        except Exception:                                      # noqa: BLE001
            continue
        if abs(z) < 1.0:
            status[k] = f"demoted: faded over last {days}d (z {z:+.1f})"
            weights["weights"][k] = 0.0
            changed.append(k)
    if changed:
        js._meta(con, "signal_weights", weights)
    return changed


def mistake_memory(con):
    """Repeating losing patterns -> candidate rules on probation (reported, never applied)."""
    pats = {}
    for b in con.execute("""select sport, market, result, price, clv_pp, signals, card_tags, logged_call from bets
                            where paper in (1,2) and result in ('W','L')"""):
        try:
            sg = json.loads(b["signals"] or "{}")
        except Exception:                                      # noqa: BLE001
            sg = {}
        flags = []
        if sg.get("is_fav") is not None:
            flags.append("favorite" if sg["is_fav"] else "underdog")
        if sg.get("is_home") is not None:
            flags.append("home" if sg["is_home"] else "away")
        if sg.get("b2b_flag") == 1:
            flags.append("on back-to-back")
        if (sg.get("hours") or 0) >= 6:
            flags.append("taken 6h+ early")
        if sg.get("pin_led") == -1:
            flags.append("retail-only drop")
        kind = "props" if str(b["market"]).startswith("player_") else b["market"]
        for f in flags:
            k = (b["sport"], kind, f)
            p = pats.setdefault(k, {"n": 0, "w": 0, "pnl": 0.0, "clv": [], "imp": []})
            p["n"] += 1; p["w"] += b["result"] == "W"
            p["pnl"] += (float(b["price"]) - 1) if b["result"] == "W" else -1.0
            if b["clv_pp"] is not None:
                p["clv"].append(b["clv_pp"])
            if b["price"]:
                p["imp"].append(1 / float(b["price"]))
    rules = []
    for (sp, mk, f), p in pats.items():
        if p["n"] < 20:
            continue
        roi = p["pnl"] / p["n"]
        clv = sum(p["clv"]) / len(p["clv"]) if p["clv"] else None
        if roi < -0.10 and (clv is None or clv < -1.0):
            rules.append({"rule": f"{sp} {mk} picks that are {f}", "n": p["n"], "record": f"{p['w']}-{p['n'] - p['w']}",
                          "roi": round(roi, 3), "clv": None if clv is None else round(clv, 2),
                          "status": "candidate: review before any action"})
    rules.sort(key=lambda r: r["roi"])
    con.execute("insert or replace into meta(k,v) values(?,?)", ("candidate_rules", json.dumps(rules[:12])))
    con.commit()
    return rules


def install(js):
    """Wrap jerry_signals: new signals in capture, ledger-aware autopsy, decay check after every weight fit."""
    if js is None or getattr(js, "_master_installed", False):
        return False
    for k in NEW_SIGNALS:
        if k not in js.SIGNALS:
            js.SIGNALS.append(k)
    _cap = js.capture

    def capture(con, ev, market, outcome, player, line, price, fair, extra=None):
        s = _cap(con, ev, market, outcome, player, line, price, fair, extra)
        try:
            if not player:
                sig, _ctx = side_signals(con, ev, market, outcome)
                s.update(sig)
        except Exception:                                      # noqa: BLE001
            pass
        return s

    js.capture = capture
    js.autopsy = autopsy
    _fit_w = js.fit_weights

    def fit_weights(con, now):
        out = _fit_w(con, now)
        try:
            ch = decay_check(con, js, out)
            if ch:
                out["demoted"] = ch
            layer_scorecard(con, js)
            mistake_memory(con)
        except Exception:                                      # noqa: BLE001
            pass
        return out

    js.fit_weights = fit_weights
    js._master_installed = True
    return True


# --------------------------------------------------------------------------- backtest harness (soccer)
FD = {"EPL": ("E0", "eng.1")}


def _fd_csv(season, code, get_text):
    return get_text(f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv")


def _get_text(url, timeout=30):
    import requests
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (KRONOS desk)"})
    r.raise_for_status()
    return r.text


def backtest_soccer(sport="EPL", seasons=("2425", "2526", "2627"), get_text=_get_text, model_weight=0.35):
    """Walk-forward: refit weekly on everything before the week, predict the week, score vs Pinnacle closing.
    Returns log loss (model, market, blend), model picks' CLV vs the close and flat-stake ROI at close."""
    import sqlite3
    import kronos_soccer as ks
    code, league = FD[sport]
    rows = []
    for s in seasons:
        try:
            txt = _fd_csv(s, code, get_text)
        except Exception:                                      # noqa: BLE001
            continue
        for r in csv.DictReader(io.StringIO(txt)):
            try:
                d = datetime.strptime(r["Date"], "%d/%m/%Y").date()
                rows.append(dict(date=d, home=r["HomeTeam"], away=r["AwayTeam"], hg=float(r["FTHG"]), ag=float(r["FTAG"]),
                                 hst=float(r["HST"]) if r.get("HST") else None, ast=float(r["AST"]) if r.get("AST") else None,
                                 hc=float(r["HC"]) if r.get("HC") else None, ac=float(r["AC"]) if r.get("AC") else None,
                                 oh=float(r["PSH"]), od=float(r["PSD"]), oa=float(r["PSA"]),
                                 ch=float(r["PSCH"]), cd=float(r["PSCD"]), ca=float(r["PSCA"])))
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda r: r["date"])
    if len(rows) < 300:
        return {"error": f"only {len(rows)} usable rows"}
    con = sqlite3.connect(":memory:"); con.row_factory = sqlite3.Row
    ks.ensure(con); con.execute("create table if not exists meta(k text primary key, v text)")
    def put(r, i):
        con.execute("insert or replace into soccer_matches values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (f"bt{i}", league, r["date"].isoformat(), r["home"], r["away"], r["hg"], r["ag"], r["hst"], r["ast"],
                     None, None, r["hc"], r["ac"], 0, "FT", "bt"))
    n_pred = 0; ll_m = ll_k = ll_b = 0.0; picks = []
    i = 0
    weeks = sorted({r["date"] - timedelta(days=r["date"].weekday()) for r in rows})
    fed = 0
    for wk in weeks:
        # feed everything before this week
        while fed < len(rows) and rows[fed]["date"] < wk:
            put(rows[fed], fed); fed += 1
        con.commit()
        if fed < 200:
            continue
        model = ks.fit(con, datetime.combine(wk, datetime.min.time(), UTC))
        if not model:
            continue
        week_rows = [r for r in rows if wk <= r["date"] < wk + timedelta(days=7)]
        for r in week_rows:
            hk, ak = ks.norm(ks.canon(r["home"])), ks.norm(ks.canon(r["away"]))
            if hk not in model["att"] or ak not in model["att"]:
                continue
            lh, la, _n = ks.predict(model, hk, ak, r["date"].isoformat())
            ph, pd, pa = ks.p_1x2(ks.score_matrix(lh, la))
            fo = ks._devig([r["oh"], r["od"], r["oa"]]); fc = ks._devig([r["ch"], r["cd"], r["ca"]])
            res = 0 if r["hg"] > r["ag"] else 1 if r["hg"] == r["ag"] else 2
            pm = [ph, pd, pa]; pb = [model_weight * m + (1 - model_weight) * k for m, k in zip(pm, fc)]
            eps = 1e-6
            ll_m -= math.log(max(pm[res], eps)); ll_k -= math.log(max(fc[res], eps)); ll_b -= math.log(max(pb[res], eps))
            n_pred += 1
            # model pick vs the opening price: which side has edge >= 2% at open, measured against the close
            edges = [m - o for m, o in zip(pm, fo)]
            j = max(range(3), key=lambda x: edges[x])
            if edges[j] >= 0.02:
                open_price = [r["oh"], r["od"], r["oa"]][j]; close_fair = fc[j]
                clv = (close_fair - 1 / open_price) * 100
                pnl = (open_price - 1) if res == j else -1.0
                picks.append((clv, pnl))
    con.close()
    if not n_pred:
        return {"error": "nothing predicted"}
    out = {"sport": sport, "seasons": list(seasons), "matches": n_pred,
           "log_loss": {"model": round(ll_m / n_pred, 4), "market_close": round(ll_k / n_pred, 4), "blend": round(ll_b / n_pred, 4)},
           "picks_vs_open": {"n": len(picks), "clv_avg_pp": round(sum(p[0] for p in picks) / len(picks), 2) if picks else None,
                             "beat_close": round(sum(p[0] > 0 for p in picks) / len(picks), 3) if picks else None,
                             "roi_flat": round(sum(p[1] for p in picks) / len(picks), 3) if picks else None},
           "read": None, "ts": _iso(datetime.now(UTC))}
    m, k, b = out["log_loss"]["model"], out["log_loss"]["market_close"], out["log_loss"]["blend"]
    out["read"] = ("blend beats the closing market" if b < k else "market alone is still better than the blend") + \
                  f"; raw model {'worse' if m > k else 'better'} than market by {abs(m - k):.4f} log loss"
    return out


def run_backtests(con, now=None):
    now = now or datetime.now(UTC)
    res = {}
    for sp in FD:
        try:
            res[sp] = backtest_soccer(sp)
        except Exception as e:                                 # noqa: BLE001
            res[sp] = {"error": str(e)[:120]}
        con.execute("insert or replace into meta(k,v) values(?,?)", (f"backtest_{sp}", json.dumps(res[sp])))
    con.commit()
    return res


# --------------------------------------------------------------------------- feed
def feed_block(con, js=None):
    L = ["", "", f"KRONOS MASTER ({VERSION}: shared rulebook, roster + schedule layers, learning upgrades, backtests):"]
    L.append("  rulebook: " + " | ".join(RULEBOOK))
    try:
        st = con.execute("select v from meta where k='master_status'").fetchone()
        L.append(f"  roster/schedule data: {st[0] if st else 'not pulled yet'}")
        n = con.execute("select sport, count(*) from injuries group by sport").fetchall()
        if n:
            L.append("  injury lists: " + ", ".join(f"{r[0]} {r[1]} players" for r in n))
        bs = con.execute("select v from meta where k='autopsy_by_sport'").fetchone()
        if bs:
            L.append(f"  autopsy by sport (14d): {bs[0][:600]}")
        cr = con.execute("select v from meta where k='candidate_rules'").fetchone()
        if cr:
            rules = json.loads(cr[0])
            L.append("  mistake memory: " + ("; ".join(f"{r['rule']} {r['record']} ROI {r['roi']:+.2f} CLV {r['clv']}" for r in rules[:6]) if rules else "no repeating losing pattern with n>=20 yet"))
        sc = con.execute("select v from meta where k='layer_scorecard'").fetchone()
        if sc:
            s = json.loads(sc[0])
            parts = []
            for k, v in s.items():
                if v.get("high") and v.get("low"):
                    parts.append(f"{k}: high {v['high']['win']:.0%}/{v['high']['clv']}pp vs low {v['low']['win']:.0%}/{v['low']['clv']}pp (n{v['n']})")
            L.append("  layer scorecard (win rate / CLV when signal high vs low): " + ("; ".join(parts) if parts else "needs 10+ graded picks per signal"))
        for sp in FD:
            bt = con.execute("select v from meta where k=?", (f"backtest_{sp}",)).fetchone()
            if bt:
                L.append(f"  backtest {sp}: {bt[0][:500]}")
    except Exception as e:                                     # noqa: BLE001
        L.append(f"  master report unavailable ({str(e)[:100]})")
    return "\n".join(L) + "\n"
'''

_DROPS_SRC = r'''"""
kronos_drops — dropping odds for KRONOS (2026-10-31).

KRONOS's own version of a dropping-odds board, built from the desk's 20-book tape (ticks table),
so no third-party site is scraped. For every tracked game-line side it measures:
  * odds drop % since open, and over the last 1h / 6h / 24h (decimal-odds change, (old-new)/old),
    for Pinnacle alone and averaged across all books at the same line
  * who led: SHARP-LED (Pinnacle/sharp book moved first), RETAIL-ONLY (retail moved, Pinnacle did not),
    MIXED, or NONE
  * line moves on spreads/totals (e.g. -1.0 -> -0.75)
  * whether the drop is already priced in at CT books (best CT price vs Pinnacle de-vigged fair)
  * STALE CT PRICE: Pinnacle dropped but a CT book still offers a +EV price -> Telegram alert

Reasoning rules (shown in the feed and applied to sharp_signal text/tags; they do NOT change verdicts
or stakes until the learning layer graduates the new signals):
  R1 A drop is sharp evidence only if it is SHARP-LED (Pinnacle/sharp book first or alone).
  R2 A RETAIL-ONLY drop (retail >=5%, Pinnacle <1%) is public money.
  R3 If the price is already below fair after the drop (best CT EV < 0), it is PRICED IN: do not chase.
  R4 The only actionable drop is a STALE CT PRICE: Pinnacle dropped >= DROP_ALERT_PCT in the last hour
     and a CT book still pays >= DROP_ALERT_EV over the new fair price.
  R5 A drop that follows confirmed team news is information, not an edge, unless a CT price is stale.
"""
import json, os
from datetime import datetime, timezone, timedelta

UTC = timezone.utc
SHARP = ("pinnacle", "lowvig", "betonlineag")
CT_BOOKS = ("draftkings", "fanduel", "fanatics")
GAME_MARKETS = ("h2h", "spreads", "totals")
WINDOWS_H = (1, 6, 24)
DROP_FLAG = float(os.environ.get("DROP_FLAG_PCT", "5"))        # "Dropping"
DROP_HEAVY = float(os.environ.get("DROP_HEAVY_PCT", "10"))     # "Heavy drop"
ALERT_PCT = float(os.environ.get("DROP_ALERT_PCT", "5"))       # Pinnacle drop in last 60 min
ALERT_EV = float(os.environ.get("DROP_ALERT_EV", "1"))         # CT EV % still left
HORIZON_H = float(os.environ.get("DROP_HORIZON_H", "48"))
BOOK_NAMES = {"draftkings": "DraftKings", "fanduel": "FanDuel", "fanatics": "Fanatics", "pinnacle": "Pinnacle"}


def _ts(s):
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=UTC)
    except Exception:                                              # noqa: BLE001
        return None


def _same(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) < 1e-9


def _drop(old, new):
    """Decimal-odds drop in %, positive = price shortened (money came in). OddsPortal convention."""
    try:
        old, new = float(old), float(new)
        return (old - new) / old * 100 if old > 1 else None
    except Exception:                                              # noqa: BLE001
        return None


def _devig(prices):
    """Proportional de-vig, list of decimal prices -> list of fair probs."""
    inv = [1 / p for p in prices if p and p > 1]
    if len(inv) != len(prices):
        return None
    s = sum(inv)
    return [x / s for x in inv]


def _fair(con, event_id, market, outcome, line):
    """Pinnacle (or the first sharp book with a full market) de-vigged fair prob for this side at this line."""
    for bk in SHARP:
        rows = con.execute("""select outcome, line, price from last_quote where event_id=? and book=? and market=?
                              and coalesce(player,'')=''""", (event_id, bk, market)).fetchall()
        if not rows:
            continue
        if market == "h2h":
            side = [r for r in rows]
        elif market == "totals":
            side = [r for r in rows if _same(r[1], line)]
        else:   # spreads: this side at line, the other side at -line
            side = [r for r in rows if (r[0] == outcome and _same(r[1], line)) or
                    (r[0] != outcome and line is not None and r[1] is not None and _same(r[1], -float(line)))]
        names = [r[0] for r in side]
        if outcome not in names or len(side) < 2 or len(set(names)) != len(names):
            continue
        f = _devig([r[2] for r in side])
        if f:
            return f[names.index(outcome)], bk
    return None, None


def side_drops(con, event_id, market, outcome, player="", line=None, now=None):
    """Full dropping-odds read for one side. line = the side's current sharp line (None for h2h)."""
    now = now or datetime.now(UTC)
    player = player or ""
    ticks = con.execute("""select book, ts, line, price from ticks where event_id=? and market=? and outcome=?
                           and player=? order by ts""", (event_id, market, outcome, player)).fetchall()
    by_book = {}
    for bk, ts, ln, pr in ticks:
        t = _ts(ts)
        if t is None or pr is None:
            continue
        by_book.setdefault(bk, []).append((t, ln, float(pr)))
    out = {"event_id": event_id, "market": market, "outcome": outcome, "player": player, "line": line,
           "books": 0, "pin": {}, "avg": {}, "led": "NONE", "line_move": None, "fair": None,
           "best_ct": None, "best_ct_book": None, "ct_ev": None, "level": None, "tags": [], "stale_ct": False}
    if not by_book:
        return out
    win = {"open": None, **{f"{h}h": now - timedelta(hours=h) for h in WINDOWS_H}}
    per_book = {}
    first_move = {}
    for bk, seq in by_book.items():
        at_line = [x for x in seq if _same(x[1], line)] if market != "h2h" else seq
        if not at_line:
            continue
        cur = at_line[-1][2]
        d = {}
        for k, t0 in win.items():
            if t0 is None:
                ref = at_line[0][2]
            else:
                before = [x for x in at_line if x[0] <= t0]
                ref = before[-1][2] if before else at_line[0][2]
            d[k] = _drop(ref, cur)
        per_book[bk] = d
        # when did this book first shorten in the last 6h (for who-led)
        t6 = win["6h"]
        before = [x for x in at_line if x[0] <= t6]
        ref = before[-1][2] if before else at_line[0][2]
        mv = next((x[0] for x in at_line if x[0] > t6 and x[2] < ref - 1e-9), None)
        if mv:
            first_move[bk] = mv
    if not per_book:
        return out
    out["books"] = len(per_book)
    pin_bk = next((b for b in SHARP if b in per_book), None)
    if pin_bk:
        out["pin"] = {k: (round(v, 1) if v is not None else None) for k, v in per_book[pin_bk].items()}
        out["pin_book"] = pin_bk
    for k in win:
        vals = [d[k] for d in per_book.values() if d.get(k) is not None]
        out["avg"][k] = round(sum(vals) / len(vals), 1) if vals else None
    # line move (spreads/totals) from Pinnacle's opening line
    if market in ("spreads", "totals") and pin_bk:
        o = con.execute("""select line from opens where event_id=? and book=? and market=? and outcome=? and player=?""",
                        (event_id, pin_bk, market, outcome, player)).fetchone()
        if o and o[0] is not None and line is not None and not _same(o[0], line):
            lf, lt = float(o[0]), float(line)
            out["line_move"] = (lf, lt)
            if market == "totals":
                toward = (lt > lf) if outcome == "Over" else (lt < lf)
            else:
                toward = lt < lf          # spread: +0.5 -> +0.25 or -3 -> -3.5 = more support for this side
            out["line_toward"] = bool(toward)
    # who led
    pin6 = (out["pin"].get("6h") or 0) if pin_bk else 0
    retail = [d.get("6h") for b, d in per_book.items() if b not in SHARP and d.get("6h") is not None]
    ret6 = sum(retail) / len(retail) if retail else 0
    sharp_first = [first_move[b] for b in SHARP if b in first_move]
    retail_first = [first_move[b] for b in first_move if b not in SHARP]
    if pin6 >= 1.0 and (not retail_first or (sharp_first and min(sharp_first) <= min(retail_first))):
        out["led"] = "SHARP-LED"
    elif ret6 >= DROP_FLAG and pin6 < 1.0:
        out["led"] = "RETAIL-ONLY"
    elif pin6 >= 1.0 or ret6 >= 1.0:
        out["led"] = "MIXED"
    # level, from the bigger of Pinnacle's and the average move since open
    big = max([v for v in (out["pin"].get("open"), out["avg"].get("open")) if v is not None] or [0])
    out["level"] = "HEAVY DROP" if big >= DROP_HEAVY else "DROPPING" if big >= DROP_FLAG else None
    # priced in / stale CT price
    fair, _fb = _fair(con, event_id, market, outcome, line)
    out["fair"] = round(fair, 4) if fair else None
    cts = con.execute(f"""select book, line, price from last_quote where event_id=? and market=? and outcome=?
                          and player=? and book in ({','.join('?' * len(CT_BOOKS))})""",
                      (event_id, market, outcome, player, *CT_BOOKS)).fetchall()
    cts = [r for r in cts if market == "h2h" or _same(r[1], line)]
    if cts:
        best = max(cts, key=lambda r: r[2])
        out["best_ct"], out["best_ct_book"] = float(best[2]), best[0]
        if fair:
            out["ct_ev"] = round((float(best[2]) * fair - 1) * 100, 1)
    if out["level"] or (out["pin"].get("1h") or 0) >= ALERT_PCT:
        if out["ct_ev"] is not None and out["ct_ev"] < 0:
            out["tags"].append("PRICED IN")
        if (out["pin"].get("1h") or 0) >= ALERT_PCT and out["ct_ev"] is not None and out["ct_ev"] >= ALERT_EV:
            out["stale_ct"] = True
            out["tags"].append("STALE CT PRICE")
    if out.get("line_toward"):
        out["tags"].append("LINE MOVE")
    if out["level"]:
        out["tags"].insert(0, out["level"])
    if out["led"] in ("SHARP-LED", "RETAIL-ONLY") and (out["level"] or out["led"] == "RETAIL-ONLY"):
        out["tags"].insert(1 if out["level"] else 0, out["led"] + " DROP")
    return out


def _sides(con, now):
    lo, hi = now.isoformat(timespec="seconds"), (now + timedelta(hours=HORIZON_H)).isoformat(timespec="seconds")
    evs = [dict(r) for r in con.execute("select * from events where commence>? and commence<? order by commence",
                                        (lo, hi))]
    for ev in evs:
        for m in GAME_MARKETS:
            rows = None
            for bk in SHARP:
                rows = con.execute("""select outcome, line from last_quote where event_id=? and book=? and market=?
                                      and coalesce(player,'')=''""", (ev["event_id"], bk, m)).fetchall()
                if rows:
                    break
            if not rows:   # no sharp book: fall back to any book's lines
                rows = con.execute("""select outcome, line from last_quote where event_id=? and market=? and
                                      coalesce(player,'')='' group by outcome""", (ev["event_id"], m)).fetchall()
            for o, ln in rows:
                yield ev, m, o, (None if m == "h2h" else ln)


def board(con, now=None, min_pct=None, limit=60):
    """All dropping sides in the next HORIZON_H hours, ranked by the biggest drop."""
    now = now or datetime.now(UTC)
    min_pct = DROP_FLAG if min_pct is None else min_pct
    rows = []
    for ev, m, o, ln in _sides(con, now):
        try:
            d = side_drops(con, ev["event_id"], m, o, "", ln, now)
        except Exception as e:                                     # noqa: BLE001
            continue
        score = max([v for v in (d["pin"].get("open"), d["avg"].get("open")) if v is not None] or [0])
        hot1 = max([v for v in (d["pin"].get("1h"), d["avg"].get("1h")) if v is not None] or [0])
        if score >= min_pct or hot1 >= 3 or d["stale_ct"] or d["led"] == "RETAIL-ONLY" or d.get("line_toward"):
            d.update({"sport": ev["sport"], "home": ev["home"], "away": ev["away"], "commence": ev["commence"],
                      "score": round(score, 1)})
            rows.append(d)
    rows.sort(key=lambda r: (not r["stale_ct"], -r["score"]))
    return rows[:limit]


def _side_label(d):
    if d["market"] == "h2h":
        return f"{d['outcome']} to win" if d["outcome"] != "Draw" else "Draw"
    ln = d["line"]
    if d["market"] == "totals":
        return f"{d['outcome']} {ln:g}" if ln is not None else d["outcome"]
    return f"{d['outcome']} {ln:+g}" if ln is not None else d["outcome"]


def _pct(v):
    return "n/a" if v is None else f"{v:+.1f}%"


def _kick(commence):
    t = _ts(commence)
    if not t:
        return str(commence)
    try:
        from zoneinfo import ZoneInfo
        t = t.astimezone(ZoneInfo("America/New_York"))
        return t.strftime("%a %-I:%M%p ET")
    except Exception:                                              # noqa: BLE001
        return t.strftime("%a %H:%MZ")


def feed_block(con, now=None):
    now = now or datetime.now(UTC)
    rows = board(con, now)
    out = ["", "", f"DROPPING ODDS (KRONOS's own 20-book tape, next {HORIZON_H:g}h; odds drop % = (old-new)/old, "
           f"positive = shortened; flag >={DROP_FLAG:g}%, heavy >={DROP_HEAVY:g}%; computed {now.isoformat(timespec='seconds')}; "
           f"{len(rows)} sides):",
           "  rules: R1 only SHARP-LED drops count as sharp evidence; R2 RETAIL-ONLY drops are public money; "
           "R3 PRICED IN (best CT EV < 0) = do not chase; R4 act only on STALE CT PRICE (Pinnacle "
           f">={ALERT_PCT:g}% in 1h, CT EV >= +{ALERT_EV:g}%); R5 a drop on confirmed team news is information, not an edge"]
    if not rows:
        out.append("  no side has dropped past the flag yet")
    for d in rows:
        lm = f" | line {d['line_move'][0]:g}->{d['line_move'][1]:g}" if d.get("line_move") else ""
        ct = (f"best CT {BOOK_NAMES.get(d['best_ct_book'], d['best_ct_book'])} {d['best_ct']:.2f} EV {d['ct_ev']:+.1f}%"
              if d.get("best_ct") and d.get("ct_ev") is not None else "no CT price")
        out.append(f"{d['sport']} {_kick(d['commence'])} {d['away']} @ {d['home']} | {d['market']} {_side_label(d)} | "
                   f"Pinnacle open {_pct(d['pin'].get('open'))} 1h {_pct(d['pin'].get('1h'))} 6h {_pct(d['pin'].get('6h'))} "
                   f"24h {_pct(d['pin'].get('24h'))} | all books open {_pct(d['avg'].get('open'))} 1h {_pct(d['avg'].get('1h'))} "
                   f"({d['books']} book{'s' if d['books'] != 1 else ''}) | {d['led']}{lm} | {ct} | [{' · '.join(d['tags']) or 'WATCH'}]")
    try:
        st = con.execute("select v from meta where k='telegram_status'").fetchone()
        out.append(f"  telegram: {st[0] if st else 'no message sent yet'}")
    except Exception:                                              # noqa: BLE001
        pass
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- Telegram
def tg_send(text, con=None):
    """Send to Telegram and record whether it worked (the old sender never checked the reply)."""
    import requests
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        status = "not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)"
        ok = False
    else:
        try:
            r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                              json={"chat_id": chat, "text": text[:3900], "disable_web_page_preview": True}, timeout=15)
            js = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            ok = bool(js.get("ok"))
            status = "ok" if ok else f"error {r.status_code}: {js.get('description', r.text[:120])}"
        except Exception as e:                                     # noqa: BLE001
            ok, status = False, f"error: {str(e)[:120]}"
    if con is not None:
        try:
            con.execute("insert or replace into meta(k,v) values(?,?)",
                        ("telegram_status", f"{datetime.now(UTC).isoformat(timespec='seconds')} {status}"))
            con.commit()
        except Exception:                                          # noqa: BLE001
            pass
    print(f"[telegram] {status}", flush=True)
    return ok


def alert_text(d, base_url=""):
    pin_book = BOOK_NAMES.get(d.get("pin_book"), d.get("pin_book"))
    fair_price = (1 / d["fair"]) if d.get("fair") else None
    lines = [f"KRONOS DROPPING ODDS: {d['sport']} {d['away']} @ {d['home']} ({_kick(d['commence'])})",
             f"{d['market']}: {_side_label(d)}",
             f"{pin_book} dropped {d['pin'].get('1h'):.1f}% in the last hour ({d['pin'].get('open'):.1f}% since open), {d['led'].lower()}",
             f"{BOOK_NAMES.get(d['best_ct_book'], d['best_ct_book'])} still {d['best_ct']:.2f}"
             + (f" vs fair {fair_price:.2f}" if fair_price else "") + f", EV {d['ct_ev']:+.1f}%"]
    if d.get("line_move"):
        lines.append(f"line moved {d['line_move'][0]:g} -> {d['line_move'][1]:g}")
    lines.append("Stale CT price, R4. Check the lineup and team news before betting; the price can move any minute.")
    if base_url:
        lines.append(f"{base_url}/dropping")
    return "\n".join(lines)


def alert_scan(con, now=None, base_url=""):
    """Send one Telegram per side per 6h when a CT price is stale after a Pinnacle drop."""
    now = now or datetime.now(UTC)
    sent = 0
    for d in board(con, now, min_pct=0):
        if not d["stale_ct"]:
            continue
        k = f"dropalert:{d['event_id']}:{d['market']}:{d['outcome']}:{d['line']}:{now.strftime('%Y%m%d')}{now.hour // 6}"
        if con.execute("select 1 from meta where k=?", (k,)).fetchone():
            continue
        if tg_send(alert_text(d, base_url), con):
            con.execute("insert or replace into meta(k,v) values(?,?)", (k, now.isoformat(timespec="seconds")))
            con.commit()
            sent += 1
    return sent


# --------------------------------------------------------------------------- reasoning + learning hooks
def signal_values(con, event_id, market, outcome, player, line):
    """The three new learning-layer signals for one side."""
    try:
        d = side_drops(con, event_id, market, outcome, player or "", None if market == "h2h" else line)
    except Exception:                                              # noqa: BLE001
        return {"drop_1h": None, "drop_6h": None, "pin_led": None}
    p = d["pin"] or d["avg"]
    led = {"SHARP-LED": 1, "RETAIL-ONLY": -1, "MIXED": 0, "NONE": 0}.get(d["led"], 0)
    return {"drop_1h": None if p.get("1h") is None else round(p["1h"] / 100, 4),
            "drop_6h": None if p.get("6h") is None else round(p["6h"] / 100, 4),
            "pin_led": led if d["books"] else None}


def install(jr=None, js=None):
    """Wrap jerry_reason.sharp_signal / flag_label and jerry_signals.capture. Verdicts are untouched."""
    done = []
    if jr is not None and not getattr(jr, "_drops_installed", False):
        _orig_sig, _orig_flag = jr.sharp_signal, jr.flag_label

        def sharp_signal(con, event_id, market, outcome, player=""):
            out = _orig_sig(con, event_id, market, outcome, player)
            try:
                ln = None
                if market != "h2h":
                    r = con.execute("""select line from last_quote where event_id=? and market=? and outcome=? and player=?
                                       and book in ('pinnacle','lowvig','betonlineag') limit 1""",
                                    (event_id, market, outcome, player or "")).fetchone()
                    ln = r[0] if r else None
                d = side_drops(con, event_id, market, outcome, player or "", ln)
                out["drops"] = {k: d[k] for k in ("pin", "avg", "led", "level", "tags", "ct_ev", "best_ct",
                                                  "best_ct_book", "line_move", "stale_ct")}
                p = d["pin"] or d["avg"]
                if d["level"] or d["led"] == "RETAIL-ONLY" or (p.get("1h") or 0) >= 3:
                    bit = (f"odds dropping: Pinnacle {_pct(d['pin'].get('open'))} since open, "
                           f"{_pct(d['pin'].get('1h'))} in 1h, {d['led'].lower()}")
                    if "PRICED IN" in d["tags"]:
                        bit += "; already priced in at CT books, do not chase"
                    if d["stale_ct"]:
                        bit += f"; {BOOK_NAMES.get(d['best_ct_book'], d['best_ct_book'])} still stale at {d['best_ct']:.2f}"
                    if d["led"] == "RETAIL-ONLY":
                        bit += "; retail-only drop = public money"
                    t = out.get("text") or ""
                    out["text"] = (t + "; " + bit) if t and t != "no sharp or public signal yet" else bit
                out["label"] = flag_label(out)
            except Exception:                                      # noqa: BLE001 - never block a refresh
                pass
            return out

        def flag_label(sig):
            base = _orig_flag(sig)
            d = (sig or {}).get("drops") or {}
            extra = [t for t in (d.get("tags") or []) if t]
            if not extra:
                return base
            tags = ([] if base in ("NEUTRAL", "NO READ") else base.split(" · ")) + extra
            return " · ".join(dict.fromkeys(tags))

        jr.sharp_signal, jr.flag_label = sharp_signal, flag_label
        jr._drops_installed = True
        done.append("reasoning")
    if js is not None and not getattr(js, "_drops_installed", False):
        for k in ("drop_1h", "drop_6h", "pin_led"):
            if k not in js.SIGNALS:
                js.SIGNALS.append(k)
        _orig_cap = js.capture

        def capture(con, ev, market, outcome, player, line, price, fair, extra=None):
            s = _orig_cap(con, ev, market, outcome, player, line, price, fair, extra)
            try:
                s.update(signal_values(con, ev["event_id"], market, outcome, player, line))
            except Exception:                                      # noqa: BLE001
                pass
            return s

        js.capture = capture
        js._drops_installed = True
        done.append("signals")
    return done


# --------------------------------------------------------------------------- desk page
PAGE_CSS = """
:root{--bg:#F3F5F2;--card:#fff;--ink:#14213A;--muted:#5A6476;--line:#D9DED8;--up:#22714A;--warn:#B7791F;--bad:#B83A32;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#0F1A16;--card:#16231E;--ink:#E7EDE9;--muted:#9AA8A1;--line:#2A3A33;--up:#5BC48A;--warn:#E5B05A;--bad:#EC7D74;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,Segoe UI,Roboto,Arial,sans-serif;
padding:env(safe-area-inset-top,0) 0 env(safe-area-inset-bottom,0)}.wrap{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:26px;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 12px}.rules{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:10px 14px;margin-bottom:14px;font-size:14px}.rules li{margin:2px 0}
.tw{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}table{border-collapse:collapse;width:100%;min-width:900px}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-size:12px;color:var(--muted);font-weight:600}
td.n{text-align:right;font-variant-numeric:tabular-nums}.up{color:var(--up);font-weight:600}.tag{display:inline-block;border-radius:4px;padding:1px 6px;
margin:1px 2px 1px 0;font-size:12px;font-weight:600;background:var(--line)}.t-STALE{background:var(--up);color:#fff}.t-HEAVY{background:var(--bad);color:#fff}
.t-PRICED{background:var(--warn);color:#fff}tr.stale td{background:rgba(34,113,74,.08)}a{color:inherit}
"""


def page_html(con, now=None):
    import html as _h
    now = now or datetime.now(UTC)
    rows = board(con, now)
    trs = []
    for d in rows:
        tags = "".join(f'<span class="tag t-{_h.escape(t.split()[0])}">{_h.escape(t)}</span>' for t in d["tags"]) or '<span class="tag">WATCH</span>'
        lm = f"<br><small>line {d['line_move'][0]:g} → {d['line_move'][1]:g}</small>" if d.get("line_move") else ""
        ct = (f"{BOOK_NAMES.get(d['best_ct_book'], d['best_ct_book'])} {d['best_ct']:.2f}<br><small>EV {d['ct_ev']:+.1f}%</small>"
              if d.get("best_ct") and d.get("ct_ev") is not None else "—")
        trs.append(f"""<tr class="{'stale' if d['stale_ct'] else ''}"><td>{_h.escape(d['sport'])}<br><small>{_h.escape(_kick(d['commence']))}</small></td>
<td>{_h.escape(d['away'])} @ {_h.escape(d['home'])}</td><td>{_h.escape(d['market'])}<br><b>{_h.escape(_side_label(d))}</b>{lm}</td>
<td class="n up">{_pct(d['pin'].get('open'))}</td><td class="n">{_pct(d['pin'].get('1h'))}</td><td class="n">{_pct(d['pin'].get('6h'))}</td>
<td class="n">{_pct(d['avg'].get('open'))}<br><small>{d['books']} book{'s' if d['books'] != 1 else ''}</small></td><td>{_h.escape(d['led'])}</td><td class="n">{ct}</td><td>{tags}</td></tr>""")
    body = "".join(trs) or '<tr><td colspan="10">No side has dropped past the flag yet.</td></tr>'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta http-equiv="refresh" content="300"><title>KRONOS dropping odds</title><style>{PAGE_CSS}</style></head><body><div class="wrap">
<p><a href="/">← Desk</a></p><h1>Dropping odds</h1>
<p class="sub">Next {HORIZON_H:g} hours · decimal odds · odds drop % = (old − new) ÷ old · refreshes every 5 minutes · {now.strftime('%Y-%m-%d %H:%M')} UTC</p>
<ol class="rules"><li>Only a sharp-led drop (Pinnacle or another sharp book first) counts as sharp evidence.</li>
<li>A retail-only drop is public money.</li><li>Priced in (best CT price below fair) means don’t chase it.</li>
<li>Act only on a stale CT price: Pinnacle down {ALERT_PCT:g}%+ in the last hour and a CT book still +{ALERT_EV:g}% EV. These go to Telegram.</li>
<li>A drop on confirmed team news is information, not an edge.</li></ol>
<div class="tw"><table><thead><tr><th>Sport</th><th>Game</th><th>Side</th><th class="n">Pinnacle since open</th><th class="n">1h</th><th class="n">6h</th>
<th class="n">All books since open</th><th>Who led</th><th class="n">Best CT</th><th>Tags</th></tr></thead><tbody>{body}</tbody></table></div>
</div></body></html>"""
'''

try:
    jerry_reason = _register("jerry_reason", _REASON_SRC)
    jerry_grade = _register("jerry_grade", _GRADE_SRC)
    jerry_feeds = _register("jerry_feeds", _FEEDS_SRC)
    jerry_nba = _register("jerry_nba", _NBA_SRC)
    jerry_splits = _register("jerry_splits", _SPLITS_SRC)
    jerry_signals = _register("jerry_signals", _SIGNALS_SRC)
    kronos_drops = _register("kronos_drops", _DROPS_SRC)
    print(f"[boot] KRONOS dropping odds installed: {kronos_drops.install(jerry_reason, jerry_signals)}", flush=True)
    kronos_soccer = _register("kronos_soccer", _SOCCER_SRC)
    kronos_ledger = _register("kronos_ledger", _LEDGER_SRC)
    kronos_ledger.install(jerry_signals)
    kronos_master = _register("kronos_master", _MASTER_SRC)
    kronos_edge = _register("kronos_edge", _EDGE_SRC)
    kronos_master.install(jerry_signals)
    # soccer error bands for Jerry's spin (goals): typical miss on margin / total per match
    jerry_reason.SPORT_MAE.update({("EPL", "margin"): 1.3, ("EPL", "total"): 1.4,
                                   ("UCL", "margin"): 1.4, ("UCL", "total"): 1.5})
    try:
        _rc = sqlite3.connect(os.path.join(os.environ.get("JERRY_DATA_DIR", "/data"), "desk.db"), timeout=60)
        _fix = []
        for _id, _c, _g in _rc.execute("""select id, commence, graded_ts from bets where sport='NFL' and market like 'player_%'
                                          and result='V' and graded_ts is not null""").fetchall():
            try:
                _ct = datetime.fromisoformat(str(_c).replace("Z", "+00:00")); _ct = _ct if _ct.tzinfo else _ct.replace(tzinfo=UTC)
                _gt = datetime.fromisoformat(str(_g).replace("Z", "+00:00")); _gt = _gt if _gt.tzinfo else _gt.replace(tzinfo=UTC)
                if _gt < _ct + timedelta(hours=3):
                    _fix.append(_id)
            except Exception:                                   # noqa: BLE001
                pass
        for _id in _fix:
            _rc.execute("""update bets set result='pending', pnl=null, graded_ts=null,
                           note=replace(note, ' | settled', ' | unvoided') where id=?""", (_id,))
        _rc.execute("create table if not exists meta(k text primary key, v text)")
        # props from earlier weeks that went back to pending need their week re-run by the backfill
        if _fix or not _rc.execute("select 1 from meta where k='nfl_false_void_rerun'").fetchone():
            _rc.execute("delete from meta where k like 'backfill_props_%'")
            _rc.execute("insert or replace into meta(k,v) values('nfl_false_void_rerun', ?)", (datetime.now(UTC).isoformat(timespec='seconds'),))
        _rc.execute("insert or replace into meta(k,v) values(?,?)", ("nfl_false_void_repair",
                    f"{datetime.now(UTC).isoformat(timespec='seconds')} reset {len(_fix)} props voided before their game ended"))
        _rc.commit(); _rc.close()
        print(f"[boot] NFL false-void repair: {len(_fix)} props back to pending", flush=True)
    except Exception as _e:                                     # noqa: BLE001
        print(f"[boot] NFL false-void repair skipped: {_e}", flush=True)
    print(f"[boot] KRONOS soccer model + prediction ledger + master engine registered ({len(jerry_signals.SIGNALS)} signals)", flush=True)
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
        _c = sqlite3.connect(db_path, timeout=60); _c.row_factory = sqlite3.Row; _c.execute('pragma busy_timeout=60000')
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
        _c = sqlite3.connect(db_path, timeout=60); _c.row_factory = sqlite3.Row; _c.execute('pragma busy_timeout=60000')
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
        _c = sqlite3.connect(db_path, timeout=60); _c.row_factory = sqlite3.Row; _c.execute('pragma busy_timeout=60000')
        try:
            res["FEEDS"] = _jf.run(_c, datetime.now(UTC))
        finally:
            _c.close()
    except Exception as e:   # noqa: BLE001
        res["FEEDS"] = f"error: {e}"
    # master engine: ESPN injury reports + recent schedules (rest / back-to-backs)
    try:
        import kronos_master as _km
        _c = sqlite3.connect(db_path, timeout=60); _c.row_factory = sqlite3.Row; _c.execute('pragma busy_timeout=60000')
        try:
            res["ROSTER"] = _km.refresh(_c, datetime.now(UTC))
        finally:
            _c.close()
    except Exception as e:   # noqa: BLE001
        res["ROSTER"] = f"error: {e}"
    # soccer model (EPL + UCL): ESPN data top-up, refit every 6h, model rows + spins + paper rows
    try:
        import kronos_soccer as _ks
        _c = sqlite3.connect(db_path, timeout=60); _c.row_factory = sqlite3.Row; _c.execute('pragma busy_timeout=60000')
        try:
            res["SOCCER"] = _ks.refresh(_c, datetime.now(UTC), L=L, jr=sys.modules.get("jerry_reason"))
        finally:
            _c.close()
    except Exception as e:   # noqa: BLE001 - never block a refresh
        res["SOCCER"] = f"error: {e}"
        traceback.print_exc()
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
    try:
        L.MODEL_VERSIONS[("EPL", "model")] = "epl-soccer-v1-2026-11-01"
        L.MODEL_VERSIONS[("UCL", "model")] = "ucl-soccer-v1-2026-11-01"
    except Exception:
        pass
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
    hdr = ["", "", f"GAME SIGNALS (all {len(_js.SIGNALS)} KRONOS signals on both sides of every game line, every tracked game in the next "
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
    txt = _summary_text() + _edge_block() + _ledger_block() + _master_block() + _dropping_block() + _game_signals_block() + _quarantine_block()
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


# ---------- dropping odds (2026-10-31): feed block, desk page, Telegram alerts
DESK_URL = os.environ.get("DESK_URL", "https://jerrybot-production.up.railway.app").rstrip("/")
DROP_SCAN_SEC = int(os.environ.get("DROP_SCAN_SEC", "300"))


def _desk_db():
    return os.path.join(os.environ.get("JERRY_DATA_DIR", "/data"), "desk.db")


def _drops_con():
    c = sqlite3.connect(_desk_db(), timeout=60); c.row_factory = sqlite3.Row
    try:
        c.execute("pragma busy_timeout=60000")
    except Exception:                                                 # noqa: BLE001
        pass
    return c


# WAL lets the desk keep reading while the new KRONOS jobs write (2026-11-01: 'database is locked').
try:
    _wc = sqlite3.connect(_desk_db(), timeout=60)
    print(f"[boot] desk.db journal mode: {_wc.execute('pragma journal_mode=WAL').fetchone()[0]}", flush=True)
    _wc.close()
except Exception as _e:                                               # noqa: BLE001
    print(f"[boot] WAL not set: {_e}", flush=True)


def _dropping_block():
    try:
        import kronos_drops as _kd
        c = _drops_con()
        try:
            return _kd.feed_block(c)
        finally:
            c.close()
    except Exception as e:                                            # noqa: BLE001
        return f"\n\nDROPPING ODDS: unavailable ({str(e)[:200]})\n"


# Every Telegram message now goes through one sender that checks Telegram's reply and records
# the result (the feed shows it). Steam alerts are off unless TELEGRAM_STEAM=1: one busy match
# logged 139 steam events today, which would bury the dropping-odds alerts.
if L is not None and hasattr(L, "telegram"):
    # Health checks (2026-11-01): only problems you need to act on reach Telegram, in plain English,
    # once per problem per 12h. Known, already-handled conditions stay in the feed only.
    _HEALTH_RULES = [
        # (pattern, alert?, plain-English text)
        ("not a mounted volume", True, "Storage problem: the desk's data folder isn't on a saved volume, so a redeploy would wipe it. Check the Railway volume is attached to /data."),
        ("model error", True, "A sport model crashed on its last run: {p}. KRONOS keeps logging market-model picks for that sport until it's fixed. Send me 'KRONOS check'."),
        ("goalie parser broken", True, "NHL goalie feed is broken, so every NHL total is a Sit until it's fixed. Send me 'KRONOS check'."),
        ("0 confirmed goalies", False, ""),
        ("goalie feed has not been read", False, ""),
        ("PAPER ONLY rows tagged Play", True, "A paper-only model produced a live Play call. Don't bet it. Send me 'KRONOS check'."),
        ("prices present but the model produced 0 rows", True, "{p}. The model ran but made no predictions. Send me 'KRONOS check'."),
        ("no prices on the board", False, ""),     # normal before lines are posted
        ("edge above", False, ""),                 # handled: the ledger caps big gaps at grade C
        ("DK splits: 0 rows", False, ""),
        ("database is locked", None, "The desk's database was busy {n} times in the last 24 hours. It recovers on its own; tell me if this keeps coming."),
    ]

    def _health_filter(text):
        lines = [l[2:].strip() for l in str(text).splitlines()[1:] if l.startswith("- ")]
        try:
            c = _drops_con()
        except Exception:                                             # noqa: BLE001
            return None
        out = []
        try:
            now = datetime.now(UTC)
            for p in lines:
                rule = next((r for r in _HEALTH_RULES if r[0] in p), None)
                if rule is None:
                    alert, msg = True, f"{p}. Send me 'KRONOS check' if it repeats."
                else:
                    alert, msg = rule[1], rule[2]
                if alert is None:              # count it; alert only when it keeps happening
                    k = "tg_lock_seen"
                    seen = [t for t in json.loads((c.execute("select v from meta where k=?", (k,)).fetchone() or ["[]"])[0])
                            if datetime.fromisoformat(t) > now - timedelta(hours=24)]
                    seen.append(now.isoformat(timespec="seconds"))
                    c.execute("insert or replace into meta(k,v) values(?,?)", (k, json.dumps(seen[-50:])))
                    alert = len(seen) >= 4
                    msg = msg.format(n=len(seen))
                if not alert:
                    continue
                key = "tg_health:" + rule[0] if rule else "tg_health:" + p[:60]
                last = c.execute("select v from meta where k=?", (key,)).fetchone()
                if last and datetime.fromisoformat(last[0]) > now - timedelta(hours=12):
                    continue
                c.execute("insert or replace into meta(k,v) values(?,?)", (key, now.isoformat(timespec="seconds")))
                out.append("• " + msg.format(p=p))
            c.commit()
        finally:
            c.close()
        return ("KRONOS needs attention\n" + "\n".join(out)) if out else None

    def _tg(text):
        if str(text).startswith("STEAM") and os.environ.get("TELEGRAM_STEAM", "0") != "1":
            return False
        if str(text).startswith("KRONOS HEALTH CHECK"):
            if os.environ.get("TELEGRAM_HEALTH", "1") == "0":
                return False
            text = _health_filter(text)
            if not text:
                return False
        try:
            import kronos_drops as _kd
            c = _drops_con()
            try:
                return _kd.tg_send(text, c)
            finally:
                c.close()
        except Exception as e:                                        # noqa: BLE001
            print(f"[telegram] failed: {e}", flush=True)
            return False
    L.telegram = _tg


def _drops_loop():
    time.sleep(120)
    try:
        import kronos_drops as _kd
        if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
            c = _drops_con()
            try:
                if c.execute("select v from meta where k='telegram_status' and v like '% ok'").fetchone():
                    raise StopIteration          # already connected once: no hello on every deploy
                _kd.tg_send("KRONOS is connected. You'll get a message here when Pinnacle drops a price and a "
                            "CT book is still stale.\n" + DESK_URL + "/dropping", c)
            finally:
                c.close()
    except StopIteration:
        pass
    except Exception as e:                                            # noqa: BLE001
        print(f"[drops] hello failed: {e}", flush=True)
    while True:
        try:
            import kronos_drops as _kd
            c = _drops_con()
            try:
                n = _kd.alert_scan(c, base_url=DESK_URL)
                c.execute("insert or replace into meta(k,v) values(?,?)",
                          ("drops_scan", f"{datetime.now(UTC).isoformat(timespec='seconds')} sent {n}"))
                c.commit()
            finally:
                c.close()
            if n:
                print(f"[drops] sent {n} alert(s)", flush=True)
        except Exception as e:                                        # noqa: BLE001
            print(f"[drops] scan failed: {e}", flush=True)
        time.sleep(DROP_SCAN_SEC)


threading.Thread(target=_drops_loop, daemon=True).start()


# ---------- prediction ledger (2026-11-01): a pick for every market of every match before kickoff, graded free
LEDGER_EVERY_SEC = int(os.environ.get("LEDGER_EVERY_SEC", "300"))


def _ledger_block():
    try:
        import kronos_ledger as _kl
        c = _drops_con()
        try:
            return _kl.feed_block(c, L=L)
        finally:
            c.close()
    except Exception as e:                                            # noqa: BLE001
        return f"\n\nKRONOS LEDGER: unavailable ({str(e)[:200]})\n"


def _ledger_loop():
    time.sleep(180)
    k = 0
    while True:
        try:
            import kronos_ledger as _kl
            c = _drops_con()
            try:
                n = _kl.log_due(c, L=L, jr=sys.modules.get("jerry_reason"), js=sys.modules.get("jerry_signals"))
                if n:
                    print(f"[ledger] logged {n} picks", flush=True)
                while _kl.NEW_BETS:
                    _m = _kl.NEW_BETS.pop(0)
                    try:
                        L.telegram(_m) if L is not None else None
                    except Exception:                                   # noqa: BLE001
                        pass
                if k % 2 == 0:
                    g = _kl.grade_pending(c, jr=sys.modules.get("jerry_reason"))
                    c.execute("insert or replace into meta(k,v) values(?,?)",
                              ("ledger_grade", f"{datetime.now(UTC).isoformat(timespec='seconds')} {json.dumps(g)}"))
                    c.commit()
                    if g.get("graded"):
                        print(f"[ledger] graded {g['graded']}", flush=True)
            finally:
                c.close()
        except Exception as e:                                        # noqa: BLE001
            print(f"[ledger] pass failed: {e}", flush=True)
        k += 1
        time.sleep(LEDGER_EVERY_SEC)


threading.Thread(target=_ledger_loop, daemon=True).start()


# ---------- master engine (2026-11-01): feed block + nightly backtests
def _master_block():
    try:
        import kronos_master as _km
        c = _drops_con()
        try:
            return _km.feed_block(c, sys.modules.get("jerry_signals"))
        finally:
            c.close()
    except Exception as e:                                            # noqa: BLE001
        return f"\n\nKRONOS MASTER: unavailable ({str(e)[:200]})\n"


def _backtest_loop():
    time.sleep(900)
    last_day = None
    while True:
        try:
            today = datetime.now(UTC).date()
            if last_day != today and (last_day is None or datetime.now(UTC).hour >= 9):
                import kronos_master as _km
                c = _drops_con()
                try:
                    r = _km.run_backtests(c)
                    print(f"[backtest] {json.dumps(r)[:300]}", flush=True)
                finally:
                    c.close()
                last_day = today
        except Exception as e:                                        # noqa: BLE001
            print(f"[backtest] failed: {e}", flush=True)
        time.sleep(1800)


threading.Thread(target=_backtest_loop, daemon=True).start()


# ---------- edge layer (2026-11-01): BET tier, news re-pulls, boost checker, alternate lines, early NFL scan, preflight
EDGE_EVERY_SEC = int(os.environ.get("EDGE_EVERY_SEC", "600"))


def _edge_block():
    try:
        import kronos_edge as _ke
        c = _drops_con()
        try:
            _ke.ensure(c)
            return _ke.feed_block(c)
        finally:
            c.close()
    except Exception as e:                                            # noqa: BLE001
        return f"\n\nKRONOS EDGE: unavailable ({str(e)[:200]})\n"


def _tg_out(msg):
    try:
        if L is not None:
            L.telegram(msg)
    except Exception:                                                 # noqa: BLE001
        pass


def _edge_loop():
    time.sleep(200)
    try:
        import kronos_edge as _ke
        c = _drops_con()
        try:
            _ke.ensure(c)
            ok, checks = _ke.preflight(c)
            print(f"[preflight] {'ok' if ok else 'FAILED'}: {checks}", flush=True)
            if not ok:
                _tg_out("KRONOS needs attention\nStartup check failed:\n" +
                        "\n".join(f"• {n}: {e}" for n, o, e in checks if not o) + "\nSend me 'KRONOS check'.")
        finally:
            c.close()
    except Exception as e:                                            # noqa: BLE001
        print(f"[preflight] crashed: {e}", flush=True)
    k = 0
    while True:
        try:
            import kronos_edge as _ke
            c = _drops_con()
            try:
                _ke.ensure(c)
                r1 = _ke.news_scan(c, tg=_tg_out)
                r2 = _ke.alt_scan(c, tg=_tg_out)
                if k % 6 == 0:
                    r3 = _ke.early_scan(c, L=L, jr=sys.modules.get("jerry_reason"), js=sys.modules.get("jerry_signals"), tg=_tg_out)
                else:
                    r3 = None
                c.execute("insert or replace into meta(k,v) values(?,?)",
                          ("edge_status", json.dumps({"ts": datetime.now(UTC).isoformat(timespec="seconds"),
                                                      "news": r1, "alt": r2, "early": r3})[:900]))
                c.commit()
            finally:
                c.close()
        except Exception as e:                                        # noqa: BLE001
            print(f"[edge] pass failed: {e}", flush=True)
        k += 1
        time.sleep(EDGE_EVERY_SEC)


def _tg_poll_loop():
    time.sleep(60)
    try:
        import kronos_edge as _ke
        import kronos_drops as _kd
        def _send(text):
            c = _drops_con()
            try:
                return _kd.tg_send(text, c)
            finally:
                c.close()
        _ke.telegram_poll(_drops_con, _send)
    except Exception as e:                                            # noqa: BLE001
        print(f"[telegram poll] stopped: {e}", flush=True)


threading.Thread(target=_edge_loop, daemon=True).start()
threading.Thread(target=_tg_poll_loop, daemon=True).start()

# Desk routes /dropping and /api/dropping, added without editing jerry_desk.py: the desk builds
# its server from http.server.ThreadingHTTPServer, so wrap that class to put a route layer in
# front of the desk's own handler (and add a Dropping odds link to the home page).
import http.server as _hs
import urllib.parse as _up
_OrigTHS = _hs.ThreadingHTTPServer


class _KronosServer(_OrigTHS):
    def __init__(self, addr, handler, *a, **k):
        class _H(handler):
            def do_GET(self):
                p = _up.urlparse(self.path).path
                if p in ("/api/backtest", "/api/roster", "/api/rulebook"):
                    try:
                        import kronos_master as _km
                        c = _drops_con()
                        try:
                            if p == "/api/rulebook":
                                return self.send(json.dumps(_km.RULEBOOK))
                            if p == "/api/roster":
                                rows = [dict(r) for r in c.execute("select * from injuries order by sport, team, player")]
                                return self.send(json.dumps(rows))
                            qs = _up.parse_qs(_up.urlparse(self.path).query)
                            sp = (qs.get("sport") or ["EPL"])[0]
                            r = c.execute("select v from meta where k=?", (f"backtest_{sp}",)).fetchone()
                            return self.send(r[0] if r else json.dumps({"status": "no backtest stored yet for " + sp}))
                        finally:
                            c.close()
                    except Exception as e:                            # noqa: BLE001
                        return self.send(f"master error: {e}", "text/plain")
                if p in ("/dropping", "/api/dropping"):
                    try:
                        import kronos_drops as _kd
                        c = _drops_con()
                        try:
                            if p == "/dropping":
                                return self.send(_kd.page_html(c), "text/html; charset=utf-8")
                            return self.send(json.dumps(_kd.board(c), default=str))
                        finally:
                            c.close()
                    except Exception as e:                            # noqa: BLE001
                        return self.send(f"dropping odds error: {e}", "text/plain")
                self._kronos_home = (p == "/")
                return super().do_GET()

            def send(self, body, ctype="application/json"):
                if getattr(self, "_kronos_home", False) and str(ctype).startswith("text/html"):
                    link = '<a href="/dropping" style="display:inline-block;margin:8px 0;font-weight:600">Dropping odds</a>'
                    if isinstance(body, bytes):
                        body = body.decode("utf-8", "replace")
                    i = body.find("<body")
                    j = body.find(">", i) if i >= 0 else -1
                    body = body[:j + 1] + link + body[j + 1:] if j >= 0 else link + body
                    self._kronos_home = False
                return super().send(body, ctype)
        super().__init__(addr, _H, *a, **k)


_hs.ThreadingHTTPServer = _KronosServer

if __name__ == "__main__":
    sys.argv = ["jerry_desk.py"] + sys.argv[1:]
    runpy.run_path("jerry_desk.py", run_name="__main__")
