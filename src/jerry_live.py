"""jerry_live.py — the glue between the desk (live prices in desk.db) and Jerry's models (jerry.py).

Runs inside jerry_desk.py on a timer. Every hour it:
  * NFL   — fits the market-anchored EPA model on all history and prices every game on the board
             (spread / total / moneyline), flags Out/Doubtful players from the official injury report
  * NFL props — projects every player prop the desk has a price for (pass/rush/rec yards, receptions,
             anytime TD) from nflverse usage, adjusted for the opponent's defense; excludes injured players
  * NHL   — expected goals from MoneyPuck 5v5 xG blended with finishing talent, power-play xG, confirmed
             goalies (DailyFaceoff, with a manual override), back-to-backs and 3-in-4 from the schedule
  * UFC   — the fight model on every numbered-card fight the desk tracks
and writes model_prob / fair_prob / edge / call into desk.db so the board, the summary and Telegram
alerts can use them. Everything is wrapped so a failure in one sport never stops the desk.

Calls: NFL sides/totals use the jerry.py thresholds (Lean 1.5%, Play 3%). NHL and all props are capped at
Lean until they have beaten the close in a backtest — the desk is collecting the closing lines for that.
"""
import json
import math
import os
import re
import sqlite3
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm, poisson

import jerry as JM

UTC = timezone.utc
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"

NFL_TEAMS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
NHL_TEAMS = {
    "Anaheim Ducks": "ANA", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF", "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI", "Colorado Avalanche": "COL", "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL", "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM", "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN", "Montréal Canadiens": "MTL", "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH", "New Jersey Devils": "NJD", "New York Islanders": "NYI", "New York Rangers": "NYR",
    "Ottawa Senators": "OTT", "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT", "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA", "St Louis Blues": "STL", "St. Louis Blues": "STL", "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR", "Utah Hockey Club": "UTA", "Utah Mammoth": "UTA", "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK", "Washington Capitals": "WSH", "Winnipeg Jets": "WPG",
}
PROP_STAT = {   # Odds API market -> (nflverse stat column, distribution)
    "player_pass_yds": ("passing_yards", "normal"), "player_rush_yds": ("rushing_yards", "normal"),
    "player_reception_yds": ("receiving_yards", "normal"), "player_receptions": ("receptions", "count"),
    "player_anytime_td": ("td", "poisson"),
}
LEAN, PLAY = 0.015, 0.03          # NFL sides/totals (jerry.py thresholds)
UFC_LEAN, UFC_PLAY = 0.05, 0.08


def log(msg):
    print(f"[live] {msg}", flush=True)


def nfl_season(now):
    return now.year if now.month >= 9 else now.year - 1


def nhl_season(now):
    return now.year if now.month >= 9 else now.year - 1


def _norm(s):
    return re.sub(r"[^a-z ]", "", str(s).lower().replace(".", "").replace("-", " ")).strip()


def fair_two_way(pa, pb):
    ia, ib = 1 / pa, 1 / pb
    return ia / (ia + ib)


def to_american(p):
    if p <= 0 or p >= 1:
        return None
    d = 1 / p
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def call_for(edge, lean, play, cap_lean=False):
    e = abs(edge)
    tag = "Play" if e >= play else "Lean" if e >= lean else "Sit"
    if cap_lean and tag == "Play":
        tag = "Lean"
    return tag


# ---------------------------------------------------------------- desk access
def desk_quotes(con, sport, markets):
    q = f"""select q.*, e.home, e.away, e.commence from last_quote q join events e using(event_id)
            where e.sport=? and e.commence>? and q.market in ({",".join("?" * len(markets))})"""
    return pd.DataFrame([dict(r) for r in con.execute(q, (sport, datetime.now(UTC).isoformat(timespec="seconds"), *markets))])


def pinnacle_line(df, event_id, market, outcome):
    """Pinnacle's line/price for an outcome; falls back to the median line across books."""
    sub = df[(df.event_id == event_id) & (df.market == market) & (df.outcome == outcome) & (df.player == "")]
    if sub.empty:
        return None, None
    p = sub[sub.book == "pinnacle"]
    if not p.empty:
        return p.iloc[0].line, p.iloc[0].price
    return sub.line.median(), sub.price.median()


def write_probs(con, rows, now):
    for r in rows:
        con.execute("""insert into model_probs(event_id,market,outcome,player,line,model_prob,fair_prob,edge,call,note,ts)
                       values(?,?,?,?,?,?,?,?,?,?,?) on conflict(event_id,market,outcome,player) do update set
                       line=excluded.line, model_prob=excluded.model_prob, fair_prob=excluded.fair_prob, edge=excluded.edge,
                       call=excluded.call, note=excluded.note, ts=excluded.ts""",
                    (r["event_id"], r["market"], r["outcome"], r.get("player", ""), r.get("line"), r["model_prob"],
                     r.get("fair_prob"), r.get("edge"), r["call"], r.get("note", ""), now.isoformat(timespec="seconds")))


# ---------------------------------------------------------------- NFL sides / totals
def nfl_injuries(season, week):
    try:
        inj = JM._cached_csv(f"{NFLVERSE}/injuries/injuries_{season}.csv", f"nfl_injuries_{season}.csv", refresh=True)
    except Exception as e:
        log(f"injuries unavailable: {e}"); return pd.DataFrame(columns=["team", "full_name", "position", "report_status"])
    inj = inj[inj.week == week]
    return inj[inj.report_status.isin(["Out", "Doubtful", "Questionable"])][["team", "full_name", "position", "report_status"]]


def current_week(games, now):
    g = games[(games.season == nfl_season(now)) & (games.gameday >= pd.Timestamp(now.date()) - pd.Timedelta(days=1))]
    return int(g.week.min()) if len(g) else None


def refresh_nfl(con, now):
    g = JM.games(refresh=True)
    season = nfl_season(now)
    week = current_week(g, now)
    if week is None:
        return []
    stats = JM.team_week_stats(list(range(2015, season + 1)), refresh=True)
    frame = JM.build_frame(g, stats)
    train = frame[frame.result.notna()]
    model = JM.NFLModel().fit(train)

    quotes = desk_quotes(con, "NFL", ["h2h", "spreads", "totals"])
    if quotes.empty:
        return []
    # upcoming nflverse games this/next week, lines filled from the desk (Pinnacle) so we never wait on nflverse
    up = g[(g.season == season) & (g.week.isin([week, week + 1])) & g.result.isna()].copy()
    std = JM._season_to_date(stats)
    rows, out = [], []
    inj = nfl_injuries(season, week)
    for _, ev in quotes.drop_duplicates("event_id").iterrows():
        h, a = NFL_TEAMS.get(ev.home), NFL_TEAMS.get(ev.away)
        if not h or not a:
            continue
        nv = up[(up.home_team == h) & (up.away_team == a)]
        if nv.empty:
            continue
        nv = nv.iloc[0]
        hline, hprice = pinnacle_line(quotes, ev.event_id, "spreads", ev.home)
        aline, aprice = pinnacle_line(quotes, ev.event_id, "spreads", ev.away)
        oline, oprice = pinnacle_line(quotes, ev.event_id, "totals", "Over")
        uline, uprice = pinnacle_line(quotes, ev.event_id, "totals", "Under")
        _, hmlp = pinnacle_line(quotes, ev.event_id, "h2h", ev.home)
        _, amlp = pinnacle_line(quotes, ev.event_id, "h2h", ev.away)
        if hline is None or oline is None:
            continue
        hs = std[(std.season == season) & (std.team == h)].sort_values("week").iloc[-1:]
        as_ = std[(std.season == season) & (std.team == a)].sort_values("week").iloc[-1:]
        if hs.empty or as_.empty:
            continue
        row = {"game_id": nv.game_id, "season": season, "week": int(nv.week), "home_team": h, "away_team": a,
               "spread_line": -float(hline), "total_line": float(oline),
               "h_off": float(hs.off_epa_play_std.iloc[0]), "h_def": float(hs.def_epa_play_std.iloc[0]),
               "a_off": float(as_.off_epa_play_std.iloc[0]), "a_def": float(as_.def_epa_play_std.iloc[0]),
               "h_rest_edge": float(np.clip((7 if pd.isna(nv.home_rest) else nv.home_rest) - (7 if pd.isna(nv.away_rest) else nv.away_rest), -7, 7))}
        pred = model.predict(pd.DataFrame([row])).iloc[0]
        # fair probs from Pinnacle two-way prices
        f_hcov = fair_two_way(hprice, aprice) if hprice and aprice else 0.5
        f_under = fair_two_way(uprice, oprice) if uprice and oprice else 0.5
        f_hml = fair_two_way(hmlp, amlp) if hmlp and amlp else None
        inj_h = inj[inj.team == h]; inj_a = inj[inj.team == a]
        note = "; ".join(f"{t}: " + ", ".join(f"{r.full_name} ({r.position}) {r.report_status}" for _, r in d.iterrows() if r.report_status != "Questionable")
                         for t, d in ((h, inj_h), (a, inj_a)) if (d.report_status != "Questionable").any())
        qs = [r for _, r in pd.concat([inj_h, inj_a]).iterrows() if r.report_status == "Questionable" and r.position in ("QB", "RB", "WR", "TE")]
        if qs:
            note += (" | " if note else "") + "Questionable: " + ", ".join(f"{r.full_name} ({r.team})" for r in qs[:6])
        e = pred.p_home_cover - f_hcov
        out.append(dict(event_id=ev.event_id, market="spreads", outcome=ev.home, line=hline, model_prob=pred.p_home_cover, fair_prob=f_hcov, edge=e,
                        call=f"{call_for(e, LEAN, PLAY, cap_lean=True)} {ev.home if e > 0 else ev.away}" if abs(e) >= LEAN else "Sit",
                        note=f"model margin {pred.pred_margin:+.1f} (home) vs line {-hline:+.1f}; P(|margin|=3) {pred.abs_margin_3:.0%}, =7 {pred.abs_margin_7:.0%}. {note}"))
        out.append(dict(event_id=ev.event_id, market="spreads", outcome=ev.away, line=aline, model_prob=1 - pred.p_home_cover, fair_prob=1 - f_hcov, edge=-e, call="see home row"))
        eu = pred.p_under - f_under
        big = abs(pred.pred_total - float(oline)) >= 1.5          # the only NFL signal the backtest supported
        out.append(dict(event_id=ev.event_id, market="totals", outcome="Under", line=uline, model_prob=pred.p_under, fair_prob=f_under, edge=eu,
                        call=f"{call_for(eu, LEAN, PLAY, cap_lean=not big)} {'Under' if eu > 0 else 'Over'}" if abs(eu) >= LEAN else "Sit",
                        note=f"model total {pred.pred_total:.1f} vs {oline}; backtest: 1.5+ pt disagreements went 58.5% (n=159)"))
        out.append(dict(event_id=ev.event_id, market="totals", outcome="Over", line=oline, model_prob=1 - pred.p_under, fair_prob=1 - f_under, edge=-eu, call="see Under row"))
        if f_hml is not None:
            em = pred.p_home_ml - f_hml
            out.append(dict(event_id=ev.event_id, market="h2h", outcome=ev.home, model_prob=pred.p_home_ml, fair_prob=f_hml, edge=em,
                            call=f"{call_for(em, LEAN, PLAY, cap_lean=True)} {ev.home if em > 0 else ev.away}" if abs(em) >= LEAN else "Sit"))
            out.append(dict(event_id=ev.event_id, market="h2h", outcome=ev.away, model_prob=1 - pred.p_home_ml, fair_prob=1 - f_hml, edge=-em, call="see home row"))
    write_probs(con, out, now)
    return out


# ---------------------------------------------------------------- NFL props
def _player_rates(pw, team_week, season, week):
    """Per-player recent rates (last 6 games, recency-weighted) and game-to-game spread."""
    pw = pw[(pw.season_type == "REG") & (pw.week < week)].copy()
    pw["td"] = pw.rushing_tds + pw.receiving_tds
    pw["games"] = 1
    stats = ["passing_yards", "rushing_yards", "receiving_yards", "receptions", "td", "targets", "carries", "attempts"]
    out = {}
    pw[stats] = pw[stats].fillna(0)
    pw["usage"] = pw.attempts + pw.carries + pw.targets
    pw = pw[pw.usage > 0]
    for (pid, name, team, pos), d in pw.groupby(["player_id", "player_display_name", "team", "position"]):
        # drop games where he barely played (injury exit, benching): under 40% of his usual touches
        d = d[d.usage >= 0.4 * d.usage.median()]
        d = d.sort_values("week").tail(6)
        if d.empty:
            continue
        w = np.array([0.8 ** (len(d) - 1 - i) for i in range(len(d))]); w /= w.sum()   # gentle recency, not last-game-heavy
        mean = {s: float((d[s].values * w).sum()) for s in stats}
        mean["td"] = (mean["td"] * len(d) + 0.35 * 4) / (len(d) + 4)   # shrink TD rate: 6 games is nothing
        sd = {s: float(d[s].std(ddof=0)) if len(d) > 1 else float("nan") for s in stats}
        out[(name, team)] = {"pos": pos, "n": len(d), "mean": mean, "sd": sd, "last_week": int(d.week.max())}
    return out


def _defense_factors(team_week, season, week):
    """How much each defense allows vs league average, per stat family (shrunk)."""
    tw = team_week[(team_week.season == season) & (team_week.week < week) & (team_week.season_type == "REG")]
    # yards allowed = opponent's yards gained that week
    off = tw[["week", "team", "opponent_team", "passing_yards", "rushing_yards", "receptions", "passing_tds", "rushing_tds", "receiving_tds"]].copy()
    off["td"] = off.rushing_tds + off.receiving_tds
    allowed = off.rename(columns={"team": "offense", "opponent_team": "team"})
    lg = {s: allowed[s].mean() for s in ["passing_yards", "rushing_yards", "receptions", "td"]}
    fac = {}
    for t, d in allowed.groupby("team"):
        n = len(d)
        fac[t] = {s: ((d[s].sum() + lg[s] * 4) / (n + 4)) / lg[s] for s in lg}   # shrink 4 games toward league
    return fac


def refresh_nfl_props(con, now):
    g = JM.games()
    season = nfl_season(now); week = current_week(g, now)
    if week is None:
        return []
    quotes = desk_quotes(con, "NFL", list(PROP_STAT))
    if quotes.empty:
        return []
    pw = JM._cached_csv(f"{NFLVERSE}/stats_player/stats_player_week_{season}.csv", f"nfl_player_week_{season}.csv", refresh=True)
    tw = JM._cached_csv(f"{NFLVERSE}/stats_team/stats_team_week_{season}.csv", f"nfl_team_week_{season}.csv", refresh=True)
    rates = _player_rates(pw, tw, season, week)
    dfac = _defense_factors(tw, season, week)
    inj = nfl_injuries(season, week)
    out_players = {_norm(n) for n in inj[inj.report_status.isin(["Out", "Doubtful"])].full_name}
    quest = {_norm(n) for n in inj[inj.report_status == "Questionable"].full_name}
    by_norm = {_norm(k[0]): (k, v) for k, v in rates.items()}
    out = []
    for (eid, market, player), d in quotes.groupby(["event_id", "market", "player"]):
        if not player:
            continue
        hit = by_norm.get(_norm(player))
        if not hit:
            continue
        (name, team), r = hit
        home, away = d.iloc[0].home, d.iloc[0].away
        opp = NFL_TEAMS.get(away) if team == NFL_TEAMS.get(home) else NFL_TEAMS.get(home)
        stat, kind = PROP_STAT[market]
        fam = {"passing_yards": "passing_yards", "rushing_yards": "rushing_yards", "receiving_yards": "passing_yards",
               "receptions": "receptions", "td": "td"}[stat]
        f = dfac.get(opp, {}).get(fam, 1.0) ** 0.5          # dampen the defense adjustment
        mu = r["mean"][stat] * f
        sd = r["sd"][stat]
        if r["n"] < 3 or mu <= 0:
            continue
        # consensus line + Pinnacle/median prices
        line = d.line.median() if d.line.notna().any() else None
        over = d[d.outcome.isin(["Over", "Yes"])]; under = d[d.outcome.isin(["Under", "No"])]
        po = over.price.median() if len(over) else None; pu = under.price.median() if len(under) else None
        if kind == "poisson":                       # anytime TD: Yes price only
            p_yes = 1 - math.exp(-mu)
            fair = 1 / po if po else None
            if fair and pu:
                fair = fair_two_way(po, pu)
            if not fair:
                continue
            edge = p_yes - fair
            o_name, p_model = "Yes", p_yes
        else:
            if line is None or not po or not pu:
                continue
            hist = sd if sd == sd else 0
            if kind == "count":
                sd_use = min(max(hist, 1.1 * math.sqrt(mu), 1.2), 0.9 * mu + 1)
            elif stat == "passing_yards":
                sd_use = min(max(hist, 0.25 * mu, 35.0), 0.45 * mu)
            else:
                sd_use = min(max(hist, 0.5 * mu, 12.0), 0.9 * mu)
            p_over = 1 - norm.cdf(line, mu, sd_use)
            fair = fair_two_way(po, pu)
            edge = p_over - fair
            o_name, p_model = "Over", p_over
        # market anchor: start from the de-vigged price, move only PROPS_ANCHOR of the way toward the model
        raw = p_model
        p_model = fair + PROPS_ANCHOR * (raw - fair)
        edge = p_model - fair
        status = "OUT" if _norm(name) in out_players else "Q" if _norm(name) in quest else ""
        note = f"proj {mu:.1f} (sd {sd_use if kind != 'poisson' else 0:.1f}) from last {r['n']} games x opp factor {f:.2f}; raw model {raw:.1%}, anchored {PROPS_ANCHOR:.0%}" + (" — Questionable" if status == "Q" else "")
        off_line = line is not None and kind != "poisson" and line > 0 and not (0.6 <= mu / line <= 1.6)
        if status == "OUT":
            tag = "Sit"
            note = f"{name} listed Out/Doubtful — no bet"
        elif off_line:
            tag = "Sit"
            note += " — projection far from the line (role change or data gap), no bet"
        elif abs(edge) > 0.06:
            tag = "Sit"
            note += " — edge too large to trust"
        else:
            tag = call_for(edge, 0.03, 0.05, cap_lean=True)   # props: capped at Lean until validated by live CLV
        side = o_name if edge > 0 else ("Under" if o_name == "Over" else "No")
        out.append(dict(event_id=eid, market=market, outcome=o_name, player=player, line=line, model_prob=p_model, fair_prob=fair, edge=edge,
                        call=f"{tag} {side} {line if line is not None else ''}".strip() if tag != "Sit" else "Sit", note=note))
    write_probs(con, out, now)
    return out


# ---------------------------------------------------------------- NHL
def dailyfaceoff_goalies():
    """Confirmed starters from DailyFaceoff (best effort: parses the page's embedded JSON). Returns {team_abbr: (goalie, status)}."""
    try:
        html = requests.get("https://www.dailyfaceoff.com/starting-goalies", timeout=20,
                            headers={"User-Agent": "Mozilla/5.0"}).text
    except Exception as e:
        log(f"dailyfaceoff unavailable: {e}"); return {}
    out = {}
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    blob = m.group(1) if m else html
    # look for goalie objects: {"name": "...", "team": {"abbreviation": "BOS"}, "status": "Confirmed"} in any nesting
    for mm in re.finditer(r'"name"\s*:\s*"([^"]+)"[^{}]{0,400}?"abbreviation"\s*:\s*"([A-Z]{2,3})"[^{}]{0,400}?"(?:status|newsStrength)"\s*:\s*"([^"]+)"', blob):
        name, abbr, status = mm.groups()
        if abbr not in out or "onfirm" in status:
            out[abbr] = (name, status)
    return out


def goalie_overrides(con):
    rows = con.execute("select k, v from meta where k like 'goalie_%'").fetchall()
    out = {}
    for r in rows:
        try:
            v = json.loads(r["v"])
            if datetime.fromisoformat(v["until"]) > datetime.now(UTC):
                out[r["k"][7:]] = (v["name"], "Manual")
        except Exception:
            pass
    return out


def schedule_flags(con, team_name, commence):
    """back-to-back / 3-in-4 from the desk's own events table."""
    d0 = datetime.fromisoformat(commence.replace("Z", "+00:00"))
    lo = (d0 - timedelta(days=4)).isoformat(); hi = (d0 - timedelta(hours=6)).isoformat()
    rows = con.execute("select commence from events where sport='NHL' and (home=? or away=?) and commence>=? and commence<?",
                       (team_name, team_name, lo, hi)).fetchall()
    days = {datetime.fromisoformat(r["commence"].replace("Z", "+00:00")).date() for r in rows}
    b2b = (d0 - timedelta(days=1)).date() in days
    three_in_four = len(days) >= 2
    return b2b, three_in_four


NHL_ANCHOR = 0.35
PROPS_ANCHOR = 0.35   # same idea for NFL props until 150 graded prop bets   # weight on the model vs the market price until NHL has 150 graded bets


def _gsax60(gsax, ice_seconds, prior_hours=25.0, cap=0.35):
    """Goals saved above expected per 60, shrunk toward 0 (about 25 hours of average goaltending) and capped."""
    return float(np.clip(float(gsax) / (float(ice_seconds) / 3600 + prior_hours), -cap, cap))


def refresh_nhl(con, now):
    season = nhl_season(now)
    teams_all = JM._daily(f"{JM.MP_BASE}/{season}/regular/teams.csv", f"mp_teams_raw_{season}")
    goalies = JM.goalies(season)
    quotes = desk_quotes(con, "NHL", ["h2h", "spreads", "totals"])
    if quotes.empty:
        return []
    t5 = teams_all[teams_all.situation == "5on5"].set_index("team")
    pp = teams_all[teams_all.situation == "5on4"].set_index("team")
    pk = teams_all[teams_all.situation == "4on5"].set_index("team")
    hrs = lambda d: d["iceTime"] / 3600.0
    lg_xg5 = (t5.xGoalsFor / hrs(t5)).mean()
    lg_pp = (pp.xGoalsFor / hrs(pp)).mean(); lg_pk = (pk.xGoalsAgainst / hrs(pk)).mean()
    pp_minutes = 4.5  # expected PP time per team per game, league average
    confirmed = dailyfaceoff_goalies(); confirmed.update(goalie_overrides(con))
    out = []; pend = []
    for _, ev in quotes.drop_duplicates("event_id").iterrows():
        h, a = NHL_TEAMS.get(ev.home), NHL_TEAMS.get(ev.away)
        if not h or not a or h not in t5.index or a not in t5.index:
            continue
        def rates(t):
            # shrink toward league average with ~10 hours (about 12 games) of average 5v5 play
            r = t5.loc[t]; gp = max(r.games_played, 1); H = r.iceTime / 3600; H0 = 10.0
            xgf = (r.xGoalsFor + lg_xg5 * H0) / (H + H0); xga = (r.xGoalsAgainst + lg_xg5 * H0) / (H + H0)
            gf = (r.goalsFor + lg_xg5 * H0) / (H + H0); ga = (r.goalsAgainst + lg_xg5 * H0) / (H + H0)
            w = min(gp, 20) / 20 * 0.35                 # finishing talent: trust actual goals more as the season goes on
            return {"xgf_60": (1 - w) * xgf + w * gf, "xga_60": (1 - w) * xga + w * ga}
        rh, ra = rates(h), rates(a)
        def goalie_term(t):
            g = confirmed.get(t)
            if g:
                cand = goalies[(goalies.team == t)]
                hit = cand[cand.name.str.lower().str.contains(_norm(g[0]).split(" ")[-1])]
                if not hit.empty:
                    return _gsax60(hit.iloc[0].gsax, hit.iloc[0].iceTime), f"{g[0]} ({g[1]})"
            tg = goalies[goalies.team == t]
            if tg.empty:
                return 0.0, "goalie unconfirmed (league avg)"
            return _gsax60(tg.gsax.sum(), tg.iceTime.sum()), "goalie unconfirmed (team avg)"
        gh, gh_txt = goalie_term(h); ga_, ga_txt = goalie_term(a)
        hb2b, h34 = schedule_flags(con, ev.home, ev.commence); ab2b, a34 = schedule_flags(con, ev.away, ev.commence)
        lam_h, lam_a = JM.expected_goals(pd.Series(rh), pd.Series(ra), lg_xg5, gh, ga_, hb2b, ab2b)
        # power play: own 5on4 xGF/60 vs opponent 4on5 xGA/60, scaled to expected PP minutes
        def rel(df, team, col, lg):  # special-teams rate vs league, shrunk with 2 hours of average play, capped
            H = hrs(df).loc[team]
            return float(np.clip((df.loc[team][col] + lg * 2.0) / (H + 2.0) / lg, 0.6, 1.6))
        pph = rel(pp, h, "xGoalsFor", lg_pp) * rel(pk, a, "xGoalsAgainst", lg_pk) * lg_pp * pp_minutes / 60
        ppa = rel(pp, a, "xGoalsFor", lg_pp) * rel(pk, h, "xGoalsAgainst", lg_pk) * lg_pp * pp_minutes / 60
        base_pp = lg_pp * pp_minutes / 60
        lam_h += (pph - base_pp); lam_a += (ppa - base_pp)
        if h34: lam_h *= 0.98
        if a34: lam_a *= 0.98
        pend.append((ev, h, a, lam_h, lam_a, gh_txt, ga_txt, hb2b, ab2b))
    # slate calibration: scale every game so the model's average total matches the market's average total
    # (matched on Under probability, not raw goals: a 6.5 line at even money means about 6.1 expected goals)
    cal = []
    for ev, h, a, lam_h, lam_a, *_ in pend:
        ol, op = pinnacle_line(quotes, ev.event_id, "totals", "Over"); _ul, up = pinnacle_line(quotes, ev.event_id, "totals", "Under")
        if ol is not None and op and up:
            cal.append((lam_h, lam_a, float(ol), fair_two_way(up, op)))
    k = 1.0
    if cal:
        def gap(kk):
            return np.mean([JM.nhl_summarize(lh * kk, la * kk, ln)["p_under"] - fu for lh, la, ln, fu in cal])
        lo, hi = 0.80, 1.25
        if gap(lo) <= 0: k = lo
        elif gap(hi) >= 0: k = hi
        else:
            for _ in range(30):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if gap(mid) > 0 else (lo, mid)
            k = (lo + hi) / 2
    for ev, h, a, lam_h, lam_a, gh_txt, ga_txt, hb2b, ab2b in pend:
        lam_h *= k; lam_a *= k
        oline, oprice = pinnacle_line(quotes, ev.event_id, "totals", "Over"); uline, uprice = pinnacle_line(quotes, ev.event_id, "totals", "Under")
        hml_l, hmlp = pinnacle_line(quotes, ev.event_id, "h2h", ev.home); _, amlp = pinnacle_line(quotes, ev.event_id, "h2h", ev.away)
        hpl, hplp = pinnacle_line(quotes, ev.event_id, "spreads", ev.home); _, aplp = pinnacle_line(quotes, ev.event_id, "spreads", ev.away)
        total = float(oline) if oline is not None else 6.0
        s = JM.nhl_summarize(lam_h, lam_a, total, pl_line=abs(float(hpl)) if hpl is not None else 1.5)
        unconf = "unconfirmed" in gh_txt or "unconfirmed" in ga_txt
        n0 = len(out)
        note = f"xG {lam_h:.2f}-{lam_a:.2f}; {h} G: {gh_txt}; {a} G: {ga_txt}" + ("; B2B " + ",".join(t for t, b in ((h, hb2b), (a, ab2b)) if b) if (hb2b or ab2b) else "")
        if hmlp and amlp:
            f = fair_two_way(hmlp, amlp); e = s["p_home_ml"] - f
            out.append(dict(event_id=ev.event_id, market="h2h", outcome=ev.home, model_prob=s["p_home_ml"], fair_prob=f, edge=e,
                            call=f"{call_for(e, 0.03, 0.05, cap_lean=True)} {ev.home if e > 0 else ev.away}" if abs(e) >= 0.03 else "Sit", note=note))
            out.append(dict(event_id=ev.event_id, market="h2h", outcome=ev.away, model_prob=1 - s["p_home_ml"], fair_prob=1 - f, edge=-e, call="see home row"))
        if oprice and uprice:
            f = fair_two_way(uprice, oprice); e = s["p_under"] - f
            tag = "Sit (goalie unconfirmed)" if unconf else (f"{call_for(e, 0.03, 0.05, cap_lean=True)} {'Under' if e > 0 else 'Over'}" if abs(e) >= 0.03 else "Sit")
            out.append(dict(event_id=ev.event_id, market="totals", outcome="Under", line=uline, model_prob=s["p_under"], fair_prob=f, edge=e, call=tag,
                            note=f"exp total {s['exp_total']:.2f} vs {total}; one-goal {s['one_goal_share']:.0%}, OT {s['p_ot_so']:.0%}. {note}"))
            out.append(dict(event_id=ev.event_id, market="totals", outcome="Over", line=oline, model_prob=1 - s["p_under"], fair_prob=1 - f, edge=-e, call="see Under row"))
        if hplp and aplp and hpl is not None:
            f = fair_two_way(hplp, aplp)
            # home -1.5 -> home wins by 2+ ; home +1.5 -> anything but an away win by 2+
            p = s["p_home_by_2plus"] if float(hpl) < 0 else 1 - s["p_away_by_2plus"]
            e = p - f
            out.append(dict(event_id=ev.event_id, market="spreads", outcome=ev.home, line=hpl, model_prob=p, fair_prob=f, edge=e,
                            call=f"{call_for(e, 0.03, 0.05, cap_lean=True)} {ev.home if e > 0 else ev.away}" if abs(e) >= 0.03 else "Sit", note=note))
            out.append(dict(event_id=ev.event_id, market="spreads", outcome=ev.away, line=-float(hpl), model_prob=1 - p, fair_prob=1 - f, edge=-e, call="see home row"))
        # sanity gate: real NHL games sit in this range; anything outside means bad inputs, not an edge
        bad = not (4.5 <= lam_h + lam_a <= 7.5) or abs(lam_h - lam_a) > 1.6
        # market anchor: start from Pinnacle's fair price and move only NHL_ANCHOR of the way toward the model
        for i in range(n0, len(out), 2):
            r, pair = out[i], out[i + 1]
            raw = r["model_prob"]
            r["model_prob"] = r["fair_prob"] + NHL_ANCHOR * (raw - r["fair_prob"])
            r["edge"] = r["model_prob"] - r["fair_prob"]
            pair["model_prob"] = 1 - r["model_prob"]; pair["edge"] = -r["edge"]
            e = r["edge"]
            if r["market"] == "totals":
                side = "Under" if e > 0 else "Over"
            else:
                side = ev.home if e > 0 else ev.away
            if bad:
                r["call"] = "Sit (sanity check failed)"
            elif abs(e) > 0.06:
                r["call"] = "Sit (edge too large to trust)"
            elif r["market"] == "totals" and unconf:
                r["call"] = "Sit (goalie unconfirmed)"
            elif abs(e) >= 0.03:
                r["call"] = f"{call_for(e, 0.03, 0.05, cap_lean=True)} {side}"
            else:
                r["call"] = "Sit"
            r["note"] = (r.get("note") or "") + f"; raw model {raw:.1%}, anchored {NHL_ANCHOR:.0%} to market, slate scale {k:.2f}"
    write_probs(con, out, now)
    return out


# ---------------------------------------------------------------- UFC
def card_flag(a, b):
    """five_rd flag from cards/*.json (title fights and main events are 5 rounds)."""
    import glob
    cards_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cards")
    key = {JM.norm_name(a).split(" ")[-1], JM.norm_name(b).split(" ")[-1]}
    for path in glob.glob(os.path.join(cards_dir, "*.json")):
        try:
            raw = json.load(open(path))
        except Exception:
            continue
        for f in (raw.get("fights", []) if isinstance(raw, dict) else raw):
            if {JM.norm_name(f["a"]).split(" ")[-1], JM.norm_name(f["b"]).split(" ")[-1]} == key:
                return int(f.get("five_rd", 0) or f.get("title", 0) or 0)
    return 0


def refresh_ufc(con, now):
    quotes = desk_quotes(con, "UFC", ["h2h"])
    if quotes.empty:
        return []
    fights = []
    for _, ev in quotes.drop_duplicates("event_id").iterrows():
        _, hp = pinnacle_line(quotes, ev.event_id, "h2h", ev.home); _, ap = pinnacle_line(quotes, ev.event_id, "h2h", ev.away)
        if not hp or not ap:
            continue
        dec_to_am = lambda d: round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))
        five = card_flag(ev.home, ev.away)
        fights.append({"a": ev.home, "b": ev.away, "a_odds": dec_to_am(hp), "b_odds": dec_to_am(ap), "five_rd": five, "event_id": ev.event_id})
    if not fights:
        return []
    try:
        T = JM.ufc_card([{k: v for k, v in f.items() if k != "event_id"} for f in fights])
    except KeyError as e:
        log(f"ufc name not in history: {e}")
        return []
    out = []
    for f, (_, r) in zip(fights, T.iterrows()):
        e = r.p_model - r.p_mkt_a
        thin = min(r.a_n, r.b_n) < 4
        tag = call_for(e, UFC_LEAN, UFC_PLAY, cap_lean=thin)
        out.append(dict(event_id=f["event_id"], market="h2h", outcome=f["a"], model_prob=r.p_model, fair_prob=r.p_mkt_a, edge=e,
                        call=f"{tag} {f['a'] if e > 0 else f['b']}" if tag != "Sit" else "Sit",
                        note=f"stats-only {r.p_stats:.0%}; P(decision) {r.p_dec:.0%}; UFC fights {r.a_n}/{r.b_n}" + (" — thin data" if thin else "")))
        out.append(dict(event_id=f["event_id"], market="h2h", outcome=f["b"], model_prob=1 - r.p_model, fair_prob=1 - r.p_mkt_a, edge=-e, call="see other row"))
    write_probs(con, out, now)
    return out


# ---------------------------------------------------------------- alerts
def telegram(text):
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        return False
    try:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", json={"chat_id": chat, "text": text[:3900]}, timeout=15)
        return True
    except Exception as e:
        log(f"telegram failed: {e}"); return False


def alert_new_plays(con, rows):
    sent = {r["k"] for r in con.execute("select k from slots where k like 'alert:%'")}
    msgs = []
    for r in rows:
        if not str(r["call"]).startswith("Play"):
            continue
        k = f"alert:{r['event_id']}:{r['market']}:{r.get('player','')}"
        if k in sent:
            continue
        ev = con.execute("select sport, home, away, commence from events where event_id=?", (r["event_id"],)).fetchone()
        if not ev:
            continue
        msgs.append(f"{ev['sport']} {ev['away']} @ {ev['home']} ({ev['commence'][:16]}Z)\n{r['call']} {r.get('player','')} {r.get('line') or ''} — "
                    f"model {r['model_prob']:.0%} vs fair {r['fair_prob']:.0%} (edge {r['edge']*100:+.1f})\n{r.get('note','')}")
        con.execute("insert or ignore into slots values(?,?)", (k, datetime.now(UTC).isoformat()))
    if msgs:
        telegram("JERRY — new Play edges\n\n" + "\n\n".join(msgs[:8]))


# ---------------------------------------------------------------- entry point
def refresh_all(db_path, sports=("NFL", "NFL_PROPS", "NHL", "UFC")):
    now = datetime.now(UTC)
    con = sqlite3.connect(db_path, timeout=60); con.row_factory = sqlite3.Row
    con.execute("""create table if not exists model_probs(event_id text, market text, outcome text, player text, line real,
                   model_prob real, fair_prob real, edge real, call text, note text, ts text,
                   primary key(event_id, market, outcome, player))""")
    results = {}
    for s in sports:
        try:
            fn = {"NFL": refresh_nfl, "NFL_PROPS": refresh_nfl_props, "NHL": refresh_nhl, "UFC": refresh_ufc}[s]
            rows = fn(con, now)
            con.commit()
            results[s] = len(rows)
            alert_new_plays(con, rows)
            con.commit()
        except Exception as e:
            con.rollback()
            results[s] = f"error: {e}"
            log(f"{s} failed: {e}")
            con.execute("insert into meta(k,v) values(?,?) on conflict(k) do update set v=excluded.v",
                        (f"error_model_{s}", f"{now.isoformat(timespec='seconds')} {e}"[:200]))
            con.commit()
    con.close()
    return results
