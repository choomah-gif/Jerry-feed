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
        J, diff, tot = _model(mu_h, mu_a)
        tie = J[diff == 0].sum()
        p_home = J[diff > 0].sum() + tie * 0.52
        q, book = _quotes(con, eid)
        if not q:
            continue
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
                                                         opponent="", status="", n_games=None)
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
                rows.append((eid, market, o, "", None if line is None else float(line), float(mp), float(fair), float(edge), call, note, now.isoformat(), reasoning))
        # name the columns: the desk now carries a 12th column (reasoning) and a positional insert would fail
        try:
            cols = {r[1] for r in con.execute("pragma table_info(model_probs)")}
            if "reasoning" not in cols:
                con.execute("alter table model_probs add column reasoning text")
        except sqlite3.Error:
            pass
        con.executemany("""insert or replace into model_probs(event_id,market,outcome,player,line,model_prob,fair_prob,edge,call,note,ts,reasoning)
                           values(?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
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
    p_bet = strip_single_side_vig(bet_price)
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


MAX_UNITS_PER_GAME = 1.5
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
                     components, opponent="", status="", n_games=None):
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
    blind = BLIND_SPOTS.get(market, [])
    if blind:
        out.append("The number on the other side was set by a book that can see " + ", ".join(blind[:3]) +
                   " — none of which this model ingests" + (f", and {who} is listed Questionable" if status == "Q" else "") + ".")
    if market == "totals" and mkt is not None:
        try:
            m = float(mkt)
            if m >= 52:
                out.append("This total already sits near the top of the league's range, so the model is claiming the market under-reacted on a number it has already pushed to an extreme.")
            elif m <= 40:
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


def _norm(s):
    """Match the desk's own normaliser so the two agree on what a name is."""
    return re.sub(r"[^a-z ]", "", str(s).lower().replace(".", "").replace("-", " ")).strip()


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
NBA_PROP_STAT = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast", "player_threes": "fg3m"}
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


def _norm(s):
    return re.sub(r"[^a-z ]", "", str(s).lower().replace(".", " ").replace("-", " ")).strip()


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
                lines.append({"player": name, "pts": g("PTS"), "reb": g("REB"), "ast": g("AST"), "fg3m": fg3m,
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
            rows = con.execute("""select bl.* from box_lines bl join game_results gr using(sport, source_id)
                                  where bl.sport=? and gr.game_date between ? and ?""",
                               (sport, day, (datetime.fromisoformat(day) + timedelta(days=1)).strftime("%Y-%m-%d"))).fetchall()
            hk, ak = _team_key(home), _team_key(away)
            rows = [r for r in rows if _team_key(r["team"]) in (hk, ak)]
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
    r = con.execute("select team from player_team where sport=? and player_norm=?", (sport, _norm(player))).fetchone()
    if r:
        return r[0]
    il = _initial_last(player)
    for row in con.execute("select player_norm, team from player_team where sport=?", (sport,)):
        if _initial_last(row[0]) == il:
            return row[1]
    return None


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


def run(con, now, sports=("NBA", "NHL"), days_back=2):
    """Ingest the last few days and grade. Returns a report dict."""
    rep = {}
    for sp in sports:
        try:
            got = 0
            for k in range(days_back, -1, -1):
                day = (now - timedelta(days=k)).strftime("%Y-%m-%d")
                got += ingest_day(con, sp, day, now.isoformat(timespec="seconds"))
            g = grade_props(con, sp, now.isoformat(timespec="seconds"))
            rep[sp] = {"games_ingested": got, **g}
        except Exception as e:                           # noqa: BLE001
            rep[sp] = {"error": str(e)[:160]}
            try:
                con.execute("insert or replace into meta(k,v) values(?,?)", (f"feed_error_{sp}", f"{now.isoformat(timespec='seconds')} {str(e)[:150]}"))
                con.commit()
            except sqlite3.Error:
                pass
    return rep
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
                                          components=comp, opponent="", status="", n_games=None)
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
                                          components=comp, opponent="", status="", n_games=None)
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
            con.executemany("""insert or replace into model_probs(event_id,market,outcome,player,line,model_prob,fair_prob,edge,call,note,ts,reasoning)
                               values(?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
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
    print("[boot] reasoning + grading + feeds + nba modules registered", flush=True)
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

_orig_refresh_all = L.refresh_all if L else None
def refresh_all(db_path, sports=None):
    res = _orig_refresh_all(db_path) if sports is None else _orig_refresh_all(db_path, sports=tuple(s for s in sports if s != "MLB"))
    if sports is None or "MLB" in sports:
        try:
            res["MLB"] = refresh_mlb(db_path)
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
        if "JERRY DESK SUMMARY" in txt:
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
    for f in ("jerry.py", "jerry_live.py", "jerry_desk.py", "boot.py", "JERRY_PROMPT.md"):
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

def push_feed():
    token = os.environ.get("FEED_TOKEN")
    if not token:
        return "no FEED_TOKEN set"
    txt = _summary_text() + _quarantine_block()
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
