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
# NFL props (model) failed its 150-bet review on 2026-10-26: 65-85, -17.2u, -11.8% ROI, beat close 22%.
# Calls keep paper-logging as evidence but are tagged [PAPER ONLY] and must not go on a card.
NFL_PROPS_PAPER_ONLY = True
NFL_ANCHOR = 0.25      # NFL model moves 25% from Pinnacle fair (backtest: calibrated, no side edge); fixed until 150 graded per market
NFL_SIT_RAW = 0.06     # raw (pre-anchor) NFL edge above this = missing information, not a bet (same rule as NHL/props)
UFC_LEAN, UFC_PLAY = 0.05, 0.08


def log(msg):
    print(f"[live] {msg}", flush=True)


def nfl_season(now):
    return now.year if now.month >= 9 else now.year - 1


def nhl_season(now):
    return now.year if now.month >= 9 else now.year - 1


import jerry_reason as jr


def _norm(s):
    return re.sub(r"[^a-z ]", "", str(s).lower().replace(".", "").replace("-", " ")).strip()


def fair_two_way(pa, pb):
    """Power de-vig. Proportional de-vig concentrates its error on longshots and
    manufactures value on every long price — which is how a 4.60 anytime-TD quote
    scored +7.9% EV. Reduces to proportional on a balanced two-way market."""
    f = jr.fair_two_way_power(pa, pb)
    if f is not None:
        return f
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
    try:
        ensure_paper_schema(con)
    except Exception:
        pass
    for r in rows:
        # fade the public: no model call survives if sharp money is on the other side,
        # or if it is the public side with no sharp support
        c = r.get("call") or ""
        if c.startswith(("Lean", "Play")):
            try:
                r["call"], r["reasoning"], _sig = jr.sharp_gate(con, r["event_id"], r["market"], c,
                                                                 r.get("player", "") or "", r.get("reasoning") or "")
                r["flag"] = _sig.get("label") if _sig else None
                r["_sig"] = _sig
            except Exception:
                pass
            # a pick must be placeable: TD 'No' becomes an AVOID note, and sides with no price at
            # DraftKings/FanDuel/Fanatics are demoted (2026-10-27, after 'Pickens TD Lean No')
            try:
                r["call"], r["plain"] = jr.finalize_call(con, r["event_id"], r["market"], r["call"], r.get("player", "") or "",
                                                         r.get("model_prob"), r.get("fair_prob"), r.get("line"), r.get("outcome"))
            except Exception:
                pass
        con.execute("""insert into model_probs(event_id,market,outcome,player,line,model_prob,fair_prob,edge,call,note,ts,reasoning,flag,plain)
                       values(?,?,?,?,?,?,?,?,?,?,?,?,?,?) on conflict(event_id,market,outcome,player) do update set
                       line=excluded.line, model_prob=excluded.model_prob, fair_prob=excluded.fair_prob, edge=excluded.edge,
                       call=excluded.call, note=excluded.note, ts=excluded.ts, reasoning=excluded.reasoning, flag=excluded.flag,
                       plain=excluded.plain""",
                    (r["event_id"], r["market"], r["outcome"], r.get("player", ""), r.get("line"), r["model_prob"],
                     r.get("fair_prob"), r.get("edge"), r["call"], r.get("note", ""), now.isoformat(timespec="seconds"),
                     r.get("reasoning"), r.get("flag"), r.get("plain")))


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



# Injury adjustment to the margin model. The regression has no injury input, so a
# team missing its starting QB is modelled as if he were playing — exactly the case
# where the market knows more than the model. These are transparent priors, not
# fitted coefficients; they are logged in components so they can be audited.
QB_OUT_PTS = 4.0          # points of margin for a confirmed-Out starting QB
STARTER_OUT_PTS = 0.5     # per Out starter at RB/WR/TE/OL, capped below
STARTER_OUT_CAP = 2.0
QB_OUT_TOTAL = 2.0        # a backup QB also lowers the total

def _starting_qbs(pw, season, week, lookback=3):
    """Team -> (QB with the most pass attempts over the last few weeks, played_last_game).

    A starter shut down for the season leaves the weekly injury report entirely, so
    the injury list alone misses him. If he has no attempts in his team's most recent
    game, he is treated as out.
    """
    d = pw[(pw.season == season) & (pw.week < week) & (pw.week >= week - lookback) & (pw.position == "QB")]
    if d.empty:
        return {}
    att = d.groupby(["team", "player_display_name"])["attempts"].sum().reset_index()
    out = {}
    for t, g in att.groupby("team"):
        qb = g.sort_values("attempts", ascending=False).iloc[0].player_display_name
        team_last = d[d.team == t].week.max()
        played = ((d.team == t) & (d.week == team_last) & (d.player_display_name == qb) & (d.attempts > 5)).any()
        # consecutive most-recent team games he missed (2+ = shut down / IR, even if off the report)
        missed = 0
        for wk in sorted(d[d.team == t].week.unique(), reverse=True):
            if ((d.team == t) & (d.week == wk) & (d.player_display_name == qb) & (d.attempts > 5)).any():
                break
            missed += 1
        out[t] = (qb, bool(played), missed)
    return out

def injury_adjustment(inj, team, qb):
    """(margin_pts, total_pts, detail) — positive margin_pts means this team is WEAKER by that much.
    qb is (name, played_last_game) from _starting_qbs, or None."""
    # QB status comes from the official designation, not last week's box score (2026-10-26).
    #   Out/Doubtful                        -> full QB_OUT_PTS
    #   missed 2+ straight team games, unlisted (IR / shut down) -> full
    #   Questionable                        -> half
    #   missed only the last game, not on this week's report yet -> half, flagged UNVERIFIED
    #   otherwise                           -> 0
    if qb and isinstance(qb, tuple):
        qb_name, qb_played = qb[0], qb[1]
        qb_missed = qb[2] if len(qb) > 2 else (0 if qb[1] else 1)
    else:
        qb_name, qb_played, qb_missed = (qb, True, 0) if qb else (None, True, 0)
    d = inj[inj.team == team] if inj is not None and not inj.empty else None
    out = d[d.report_status.isin(["Out", "Doubtful"])] if d is not None and not d.empty else None
    pts = tot = 0.0; bits = []
    listed_out = qb_name is not None and out is not None and (out.full_name == qb_name).any()
    listed_q = (qb_name is not None and d is not None and not d.empty
                and ((d.full_name == qb_name) & (d.report_status == "Questionable")).any())
    team_reported = d is not None and not d.empty
    share, why = 0.0, ""
    if qb_name and listed_out:
        share, why = 1.0, "listed Out/Doubtful"
    elif qb_name and qb_missed >= 2:
        share, why = 1.0, f"missed {qb_missed} straight games"
    elif qb_name and listed_q:
        share, why = 0.5, "Questionable"
    elif qb_name and not qb_played:
        share, why = (0.0, "missed last game, not on this week's report (presumed back)") if team_reported \
            else (0.5, "missed last game, status UNVERIFIED — no report yet")
    if share:
        pts += QB_OUT_PTS * share; tot += QB_OUT_TOTAL * share
        bits.append(f"starting QB {qb_name} {why} ({QB_OUT_PTS*share:+.1f})")
    elif why:
        bits.append(f"starting QB {qb_name} {why}")
    if out is None:
        return pts, tot, "; ".join(bits)
    starters = out[out.position.isin(["RB", "WR", "TE", "T", "G", "C", "OL"])]
    k = min(len(starters), int(STARTER_OUT_CAP / STARTER_OUT_PTS))
    if k:
        pts += STARTER_OUT_PTS * k; bits.append(f"{len(starters)} skill/OL out ({STARTER_OUT_PTS*k:+.1f})")
    return pts, tot, "; ".join(bits)


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
    try:
        _pw = JM._cached_csv(f"{NFLVERSE}/stats_player/stats_player_week_{season}.csv", f"nfl_player_week_{season}.csv", refresh=False)
        qbs = _starting_qbs(_pw, season, week)
    except Exception as _e:   # noqa: BLE001
        qbs = {}
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
        # ---- injury adjustment (#4): shift margin/total, then recompute the probabilities
        adj_h, tot_h, det_h = injury_adjustment(inj, h, qbs.get(h))
        adj_a, tot_a, det_a = injury_adjustment(inj, a, qbs.get(a))
        inj_margin = adj_a - adj_h            # away team weaker -> home margin up
        inj_total = -(tot_h + tot_a)
        inj_detail = "; ".join(x for x in (f"{h}: {det_h}" if det_h else "", f"{a}: {det_a}" if det_a else "") if x)
        if inj_margin or inj_total:
            pred = pred.copy()
            pred["pred_margin"] = float(pred.pred_margin) + inj_margin
            pred["pred_total"] = float(pred.pred_total) + inj_total
            pred["p_home_cover"] = float(norm.cdf((pred.pred_margin + float(hline)) / 13.5))
            pred["p_under"] = float(norm.cdf((float(oline) - pred.pred_total) / 13.0))
            if "p_home_ml" in pred.index:
                pred["p_home_ml"] = float(norm.cdf(pred.pred_margin / 13.5))
        try:
            _qbh, _qba = qbs.get(h), qbs.get(a)
            jr.record_numbers(con, event_id=ev.event_id, sport="NFL", home=ev.home, away=ev.away, commence=ev.commence,
                              margin=float(pred.pred_margin), total=float(pred.pred_total),
                              mkt_margin=-float(hline), mkt_total=float(oline),
                              checks={"qb_status_known": bool(_qbh) and bool(_qba)},
                              detail=f"EPA ridge + rest; injury adj {inj_margin:+.1f} margin {inj_total:+.1f} total; {inj_detail}")
        except Exception:
            pass
        # fair probs from Pinnacle two-way prices
        f_hcov = fair_two_way(hprice, aprice) if hprice and aprice else 0.5
        f_under = fair_two_way(uprice, oprice) if uprice and oprice else 0.5
        f_hml = fair_two_way(hmlp, amlp) if hmlp and amlp else None
        # ---- market anchor (2026-10-26): NFL had none, so a crude input produced +10.9% ML edges
        pred = pred.copy()
        raw_cov, raw_under = float(pred.p_home_cover), float(pred.p_under)
        raw_ml = float(pred.p_home_ml) if "p_home_ml" in pred.index else None
        pred["p_home_cover"] = f_hcov + NFL_ANCHOR * (raw_cov - f_hcov)
        pred["p_under"] = f_under + NFL_ANCHOR * (raw_under - f_under)
        if raw_ml is not None and f_hml is not None:
            pred["p_home_ml"] = f_hml + NFL_ANCHOR * (raw_ml - f_hml)
        def _too_big(raw, fair):
            return fair is not None and raw is not None and abs(raw - fair) > NFL_SIT_RAW
        inj_h = inj[inj.team == h]; inj_a = inj[inj.team == a]
        note = "; ".join(f"{t}: " + ", ".join(f"{r.full_name} ({r.position}) {r.report_status}" for _, r in d.iterrows() if r.report_status != "Questionable")
                         for t, d in ((h, inj_h), (a, inj_a)) if (d.report_status != "Questionable").any())
        qs = [r for _, r in pd.concat([inj_h, inj_a]).iterrows() if r.report_status == "Questionable" and r.position in ("QB", "RB", "WR", "TE")]
        if qs:
            note += (" | " if note else "") + "Questionable: " + ", ".join(f"{r.full_name} ({r.team})" for r in qs[:6])
        e = pred.p_home_cover - f_hcov
        sp_comp = {"model_number": round(float(pred.pred_margin), 2), "market_number": round(float(-hline), 2),
                   "key3": round(float(pred.abs_margin_3), 3), "key7": round(float(pred.abs_margin_7), 3),
                   "injuries": note[:240], "raw_prob": round(float(pred.p_home_cover), 4),
                   "injury_margin_adj": round(float(inj_margin), 2), "injury_detail": inj_detail}
        sp_ratio = jr.edge_ratio(pred.pred_margin - (-hline), jr.DEFAULT_MODEL_MAE["spreads"])
        sp_tag, _sp_stake, _ = jr.size_from_ratio(sp_ratio, e, cap_lean=True)
        sp_side = ev.home if e > 0 else ev.away
        sp_comp["raw_prob_pre_anchor"] = round(raw_cov, 4)
        if _too_big(raw_cov, f_hcov):
            sp_tag = "Sit (edge too large — check inputs)"
        out.append(dict(event_id=ev.event_id, market="spreads", outcome=ev.home, line=hline, model_prob=pred.p_home_cover, fair_prob=f_hcov, edge=e,
                        call=f"{sp_tag} {sp_side}" if not sp_tag.startswith("Sit") else sp_tag,
                        note=f"model margin {pred.pred_margin:+.1f} (home) vs line {-hline:+.1f}; ratio {sp_ratio:.2f} of a {jr.DEFAULT_MODEL_MAE['spreads']}-pt error band; P(|margin|=3) {pred.abs_margin_3:.0%}, =7 {pred.abs_margin_7:.0%}. {note}",
                        components=sp_comp, model_number=sp_comp["model_number"], market_number=sp_comp["market_number"],
                        reasoning=jr.render_reasoning(market="spreads", player="", selection=sp_side, line=hline if e > 0 else aline,
                                                      model_prob=pred.p_home_cover if e > 0 else 1 - pred.p_home_cover,
                                                      fair_prob=f_hcov if e > 0 else 1 - f_hcov, components=sp_comp,
                                                      opponent=ev.away if e > 0 else ev.home, status="", n_games=None)))
        out.append(dict(event_id=ev.event_id, market="spreads", outcome=ev.away, line=aline, model_prob=1 - pred.p_home_cover, fair_prob=1 - f_hcov, edge=-e, call="see home row"))
        eu = pred.p_under - f_under
        tot_comp = {"model_number": round(float(pred.pred_total), 2), "market_number": round(float(oline), 2),
                    "raw_prob": round(float(pred.p_under), 4), "injury_total_adj": round(float(inj_total), 2),
                    "injury_detail": inj_detail}
        tot_ratio = jr.edge_ratio(pred.pred_total - float(oline), jr.DEFAULT_MODEL_MAE["totals"])
        tot_tag, _t_stake, _ = jr.size_from_ratio(tot_ratio, eu)
        tot_side = "Under" if eu > 0 else "Over"
        tot_comp["raw_prob_pre_anchor"] = round(raw_under, 4)
        if _too_big(raw_under, f_under):
            tot_tag = "Sit (edge too large — check inputs)"
        out.append(dict(event_id=ev.event_id, market="totals", outcome="Under", line=uline, model_prob=pred.p_under, fair_prob=f_under, edge=eu,
                        call=f"{tot_tag} {tot_side}" if not tot_tag.startswith("Sit") else tot_tag,
                        note=f"model total {pred.pred_total:.1f} vs {oline}; ratio {tot_ratio:.2f} of a {jr.DEFAULT_MODEL_MAE['totals']}-pt error band "
                             f"(backtest 58.5% at n=159 has a 95% interval of 50.8-66.2%, not significant vs 52.4% break-even)",
                        components=tot_comp, model_number=tot_comp["model_number"], market_number=tot_comp["market_number"],
                        reasoning=jr.render_reasoning(market="totals", player="", selection=tot_side, line=oline,
                                                      model_prob=pred.p_under if eu > 0 else 1 - pred.p_under,
                                                      fair_prob=f_under if eu > 0 else 1 - f_under, components=tot_comp,
                                                      opponent="", status="", n_games=None)))
        out.append(dict(event_id=ev.event_id, market="totals", outcome="Over", line=oline, model_prob=1 - pred.p_under, fair_prob=1 - f_under, edge=-eu, call="see Under row"))
        if f_hml is not None:
            em = pred.p_home_ml - f_hml
            ml_call = ("Sit (edge too large — check inputs)" if _too_big(raw_ml, f_hml)
                       else f"{call_for(em, LEAN, PLAY, cap_lean=True)} {ev.home if em > 0 else ev.away}" if abs(em) >= LEAN else "Sit")
            out.append(dict(event_id=ev.event_id, market="h2h", outcome=ev.home, model_prob=pred.p_home_ml, fair_prob=f_hml, edge=em,
                            call=ml_call,
                            note=f"raw model {raw_ml:.1%} vs fair {f_hml:.1%}, anchored {NFL_ANCHOR:.0%} toward market. {inj_detail}"[:400]))
            out.append(dict(event_id=ev.event_id, market="h2h", outcome=ev.away, model_prob=1 - pred.p_home_ml, fair_prob=1 - f_hml, edge=-em, call="see home row"))
    write_probs(con, out, now)
    return out


# ---------------------------------------------------------------- NFL props

def _td_prior(touches, pos=""):
    """Prior TD rate per game given weekly touches. Bench players stay near zero;
    only a full workload earns the starter's 0.35."""
    if pos == "QB":
        return 0.04 + 0.16 * min(1.0, max(0.0, (touches - 2.0) / 6.0))      # rushing TDs only
    return 0.03 + 0.32 * min(1.0, max(0.0, (touches - 3.0) / 10.0)) ** 1.2


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
    def _eff_of(df):
        t, c, a = max(df.targets.sum(), 1), max(df.carries.sum(), 1), max(df.attempts.sum(), 1)
        return {"ypt": df.receiving_yards.sum() / t, "catch": df.receptions.sum() / t,
                "ypc": df.rushing_yards.sum() / c, "ypa": df.passing_yards.sum() / a}
    _lg_eff = {"_all": _eff_of(pw)}
    for _pos, _g in pw.groupby("position"):
        _lg_eff[_pos] = _eff_of(_g)
    for (pid, name, team, pos), d in pw.groupby(["player_id", "player_display_name", "team", "position"]):
        # drop games where he barely played (injury exit, benching): under 40% of his usual touches
        d = d[d.usage >= 0.4 * d.usage.median()]
        d = d.sort_values("week").tail(6)
        if d.empty:
            continue
        w = np.array([0.8 ** (len(d) - 1 - i) for i in range(len(d))]); w /= w.sum()   # gentle recency, not last-game-heavy
        mean = {s: float((d[s].values * w).sum()) for s in stats}
        # ---- decomposition (#3): project VOLUME and EFFICIENCY separately, then recombine.
        # A whole-stat average cannot respond to a role change or a tough matchup on
        # efficiency; this can, and both halves are logged so a miss can be attributed.
        vol_t, vol_c, vol_a = mean["targets"], mean["carries"], mean["attempts"]
        tot_t, tot_c, tot_a = float(d.targets.sum()), float(d.carries.sum()), float(d.attempts.sum())
        K = 30.0   # opportunities of league-average efficiency to shrink toward
        lg = _lg_eff.get(pos, _lg_eff.get("_all"))
        eff = {
            "ypt": (float(d.receiving_yards.sum()) + lg["ypt"] * K) / (tot_t + K),
            "catch": (float(d.receptions.sum()) + lg["catch"] * K) / (tot_t + K),
            "ypc": (float(d.rushing_yards.sum()) + lg["ypc"] * K) / (tot_c + K),
            "ypa": (float(d.passing_yards.sum()) + lg["ypa"] * K) / (tot_a + K),
        }
        decomp = {"receiving_yards": vol_t * eff["ypt"], "receptions": vol_t * eff["catch"],
                  "rushing_yards": vol_c * eff["ypc"], "passing_yards": vol_a * eff["ypa"]}
        raw_mean = dict(mean)
        for k_, v_ in decomp.items():
            mean[k_] = 0.5 * raw_mean[k_] + 0.5 * v_   # blend until either half is validated
        mean["_decomp"] = {k_: round(v_, 2) for k_, v_ in decomp.items()}
        mean["_eff"] = {k_: round(v_, 3) for k_, v_ in eff.items()}
        mean["_vol"] = {"targets": round(vol_t, 2), "carries": round(vol_c, 2), "attempts": round(vol_a, 2)}
        mean["_raw"] = {k_: round(raw_mean[k_], 2) for k_ in decomp}
        # Shrink the TD rate toward a prior that depends on the player's ROLE, not a
        # flat starter's 0.35. The flat prior handed every bench player 0.2 TD/game
        # on a three-game sample (0 + 0.35*4)/(3+4) and the TD board filled with
        # longshots. Touches = carries + targets; QBs use carries only.
        touches = mean["carries"] + (0.0 if pos == "QB" else mean["targets"])
        mean["touches"] = touches
        prior = _td_prior(touches, pos)
        mean["td_prior"] = prior
        mean["td"] = (mean["td"] * len(d) + prior * 4) / (len(d) + 4)
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
            touches = r["mean"].get("touches", 0.0)
            # role gate: a player who does not touch the ball cannot be a TD edge,
            # whatever the Poisson on a tiny sample says
            if touches < 5.0 or r["n"] < 4:
                continue
            p_yes = 1 - math.exp(-mu)
            fair = 1 / po if po else None
            if fair and pu:
                fair = fair_two_way(po, pu)
            if not fair:
                continue
            # market-plausibility gate: below ~12% the market considers him an
            # incidental scorer, and small-sample Poisson noise dominates the edge
            if fair < 0.12:
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
        # the projection decomposed, so a miss can be attributed to volume or to
        # efficiency instead of being recorded as one undifferentiated loss
        if kind == "poisson":
            # Poisson rate has sd sqrt(mu/n) on the mean; the market-implied rate is
            # -ln(1-fair). This gives TD picks a real error band to size against.
            td_sd = math.sqrt(max(mu, 0.02) / max(r["n"], 1))
            comp = {"mu": round(float(mu), 3), "sd": round(float(td_sd), 3),
                    "opp_factor": round(float(f), 3), "games": int(r["n"]),
                    "raw_prob": round(float(raw), 4), "anchor": PROPS_ANCHOR,
                    "stat": stat, "touches": round(float(r["mean"].get("touches", 0.0)), 1),
                    "td_prior": round(float(r["mean"].get("td_prior", 0.0)), 3),
                    "model_number": round(float(mu), 3),
                    "market_number": round(float(-math.log(max(1e-6, 1 - fair))), 3)}
        else:
            comp = {"mu": round(float(mu), 3),
                    "sd": round(float(sd_use), 3),
                    "opp_factor": round(float(f), 3), "games": int(r["n"]),
                    "raw_prob": round(float(raw), 4), "anchor": PROPS_ANCHOR,
                    "stat": stat, "touches": round(float(r["mean"].get("touches", 0.0)), 1),
                    "model_number": round(float(mu), 3),
                    "market_number": None if line is None else float(line),
                    "volume": r["mean"].get("_vol", {}), "efficiency": r["mean"].get("_eff", {}),
                    "decomp_proj": r["mean"].get("_decomp", {}).get(stat), "whole_stat_proj": r["mean"].get("_raw", {}).get(stat)}
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
            # sized against the projection's own dispersion, not a flat edge threshold
            _ratio = jr.edge_ratio(float(comp["model_number"]) - float(comp["market_number"]), comp.get("sd") or 1.0) \
                     if comp.get("market_number") is not None else None
            tag, _stk, _why = jr.size_from_ratio(_ratio, edge, cap_lean=True)
            note += f"; ratio {_ratio:.2f}" if _ratio is not None else ""
        side = o_name if edge > 0 else ("Under" if o_name == "Over" else "No")
        reasoning = jr.render_reasoning(market=market, player=player, selection=side, line=line,
                                        model_prob=p_model, fair_prob=fair, components=comp,
                                        opponent=opp, status=status, n_games=r["n"])
        out.append(dict(event_id=eid, market=market, outcome=o_name, player=player, line=line, model_prob=p_model, fair_prob=fair, edge=edge,
                        call=(f"{tag} {side} {line if line is not None else ''}".strip()
                              + (" [PAPER ONLY]" if NFL_PROPS_PAPER_ONLY else "")) if tag != "Sit" else "Sit", note=note,
                        components=comp, model_number=comp["model_number"], market_number=comp["market_number"],
                        reasoning=reasoning))
    # Clear this sport's prop rows for upcoming games before writing. write_probs only
    # upserts, so a row the new gates now reject (a backup's TD, a prop that became a
    # Sit) would otherwise linger forever carrying the old call.
    con.execute("""delete from model_probs where market like 'player_%' and event_id in
                   (select event_id from events where sport='NFL' and commence>?)""", (now.isoformat(timespec="seconds"),))
    write_probs(con, out, now)
    return out


# ---------------------------------------------------------------- NHL
def dailyfaceoff_goalies():
    """Confirmed starters from DailyFaceoff. Returns {team_abbr: (goalie, status)}.

    Parses the page's __NEXT_DATA__ JSON and walks it. The old regex used [^{}] spans,
    which can never step into the nested "team": {...} object, so it matched nothing
    and every NHL total ran as a goalie-unconfirmed Sit (fixed 2026-10-26).
    """
    try:
        html = requests.get("https://www.dailyfaceoff.com/starting-goalies", timeout=20,
                            headers={"User-Agent": "Mozilla/5.0"}).text
    except Exception as e:
        log(f"dailyfaceoff unavailable: {e}"); return {}
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        log("dailyfaceoff: no __NEXT_DATA__ on page"); return {}
    try:
        blob = json.loads(m.group(1))
    except Exception as e:
        log(f"dailyfaceoff: bad JSON {e}"); return {}
    out = {}
    rank = lambda st: 2 if "onfirm" in st else 1 if "ikel" in st or "xpect" in st else 0

    def team_abbr(d, side=""):
        for k in (f"{side}TeamAbbreviation", f"{side}TeamSlug", f"{side}TeamAbbr"):
            v = d.get(k)
            if isinstance(v, str) and 2 <= len(v) <= 3:
                return v.upper()
        t = d.get(f"{side}Team") if side else d.get("team")
        for k in (f"{side}TeamName", "teamName"):
            if isinstance(d.get(k), str) and d[k] in NHL_TEAMS:
                return NHL_TEAMS[d[k]]
        if isinstance(t, str) and t in NHL_TEAMS:
            return NHL_TEAMS[t]
        if isinstance(t, dict):
            if isinstance(t.get("name"), str) and t["name"] in NHL_TEAMS:
                return NHL_TEAMS[t["name"]]
            for k in ("abbreviation", "abbr", "triCode", "shortName"):
                v = t.get(k)
                if isinstance(v, str) and 2 <= len(v) <= 3:
                    return v.upper()
        return None

    alias = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL", "T.B": "TBL", "N.J": "NJD", "L.A": "LAK",
             "S.J": "SJS", "WAS": "WSH", "MON": "MTL", "CLB": "CBJ", "VEG": "VGK", "UTAH": "UTA", "ARI": "UTA"}

    def put(abbr, name, status):
        abbr = alias.get(abbr, abbr) if abbr else abbr
        if abbr and name and (abbr not in out or rank(status) > rank(out[abbr][1])):
            out[abbr] = (name, status or "")

    def walk(o):
        if isinstance(o, dict):
            # layout A: one object per game with home*/away* goalie fields
            for side in ("home", "away"):
                name = o.get(f"{side}GoalieName")
                g = o.get(f"{side}Goalie")
                if not isinstance(name, str) and isinstance(g, dict):
                    name = g.get("name") or g.get("fullName")
                if isinstance(name, str):
                    st = o.get(f"{side}NewsStrengthName") or o.get(f"{side}GoalieStatus") or o.get(f"{side}Status") or ""
                    put(team_abbr(o, side), name, str(st))
            # layout B: one object per goalie with a nested team
            if isinstance(o.get("name"), str) and ("team" in o) and any(k in o for k in ("status", "newsStrength", "newsStrengthName")):
                st = o.get("status") or o.get("newsStrengthName") or o.get("newsStrength") or ""
                put(team_abbr(o), o["name"], str(st))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(blob)
    global LAST_GOALIES
    LAST_GOALIES = (datetime.now(UTC), len(out), sum(1 for v in out.values() if "onfirm" in v[1]))
    log(f"dailyfaceoff: {len(out)} teams, {LAST_GOALIES[2]} confirmed")
    return out


LAST_GOALIES = None   # (ts, teams parsed, confirmed) from the latest DailyFaceoff read, for the health check


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


def fitted_anchor(con, sport, market, default=0.35, min_n=40):
    """Weight on the model vs the market, derived from which one has been closer.

    A hardcoded 0.35 asserts the model deserves a third of the say before anything
    has been measured. When the model's error exceeds the market's, the correct
    weight is zero, and this returns it.
    """
    try:
        import jerry_reason as _jr
        mae_model = _jr.measure_model_error(con, sport, market, min_n=min_n)
    except Exception:
        return default
    if not mae_model:
        return default
    base = _jr.DEFAULT_MODEL_MAE.get(market) or mae_model
    if mae_model >= base:
        return 0.0                      # model is no better than the market: no weight
    return max(0.0, min(default, 1.0 - (mae_model / base)))


NHL_ANCHOR = 0.35
PROPS_ANCHOR = 0.35   # same idea for NFL props until 150 graded prop bets   # weight on the model vs the market price until NHL has 150 graded bets


def _gsax60(gsax, ice_seconds, prior_hours=25.0, cap=0.35):
    """Goals saved above expected per 60, shrunk toward 0 (about 25 hours of average goaltending) and capped."""
    return float(np.clip(float(gsax) / (float(ice_seconds) / 3600 + prior_hours), -cap, cap))


def refresh_nhl(con, now):
    season = nhl_season(now)
    teams_all = JM._daily(f"{JM.MP_BASE}/{season}/regular/teams.csv", f"mp_teams_raw_{season}")
    try:   # last season's rates are the prior for early-season shrinkage (a flat league average if unavailable)
        prev_all = JM._daily(f"{JM.MP_BASE}/{season - 1}/regular/teams.csv", f"mp_teams_raw_{season - 1}")
        t5_prev = prev_all[prev_all.situation == "5on5"].set_index("team")
    except Exception as e:
        log(f"NHL prior season unavailable: {e}"); t5_prev = None
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
            r = t5.loc[t]; gp = max(r.games_played, 1); H = r.iceTime / 3600; H0 = 12.0
            pf = pa = lg_xg5
            if t5_prev is not None and t in t5_prev.index:   # 70% last season, 30% league, as the prior
                p = t5_prev.loc[t]; Hp = max(p.iceTime / 3600, 1)
                pf = 0.7 * p.xGoalsFor / Hp + 0.3 * lg_xg5; pa = 0.7 * p.xGoalsAgainst / Hp + 0.3 * lg_xg5
            xgf = (r.xGoalsFor + pf * H0) / (H + H0); xga = (r.xGoalsAgainst + pa * H0) / (H + H0)
            gf = (r.goalsFor + pf * H0) / (H + H0); ga = (r.goalsAgainst + pa * H0) / (H + H0)
            w = min(gp, 20) / 20 * 0.35                 # finishing talent: trust actual goals more as the season goes on
            return {"xgf_60": (1 - w) * xgf + w * gf, "xga_60": (1 - w) * xga + w * ga}
        rh, ra = rates(h), rates(a)
        def goalie_term(t):
            g = confirmed.get(t)
            if g:
                cand = goalies[(goalies.team == t)]
                hit = cand[cand.name.str.lower().str.contains(_norm(g[0]).split(" ")[-1])]
                if not hit.empty:
                    sure = ("onfirm" in g[1]) or g[1] == "Manual"
                    # a Likely/Expected starter is used for the number but still counts as unconfirmed (totals Sit)
                    return _gsax60(hit.iloc[0].gsax, hit.iloc[0].iceTime), (f"{g[0]} ({g[1]})" if sure else f"{g[0]} ({g[1]}, unconfirmed)")
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
            _mc = {"xg_home": round(float(lam_h), 3), "xg_away": round(float(lam_a), 3), "raw_prob": round(float(s["p_home_ml"]), 4)}
            _side = ev.home if e > 0 else ev.away
            out.append(dict(event_id=ev.event_id, market="h2h", outcome=ev.home, model_prob=s["p_home_ml"], fair_prob=f, edge=e,
                            call=f"{call_for(e, 0.03, 0.05, cap_lean=True)} {_side}" if abs(e) >= 0.03 else "Sit", note=note,
                            components=_mc,
                            reasoning=None))
            out.append(dict(event_id=ev.event_id, market="h2h", outcome=ev.away, model_prob=1 - s["p_home_ml"], fair_prob=1 - f, edge=-e, call="see home row"))
        if oprice and uprice:
            f = fair_two_way(uprice, oprice); e = s["p_under"] - f
            tag = "Sit (goalie unconfirmed)" if unconf else (f"{call_for(e, 0.03, 0.05, cap_lean=True)} {'Under' if e > 0 else 'Over'}" if abs(e) >= 0.03 else "Sit")
            _nc = {"mu": round(float(s["exp_total"]), 3), "sd": round(float(math.sqrt(max(lam_h + lam_a, 0.5))), 3),
                   "model_number": round(float(s["exp_total"]), 3), "market_number": float(total),
                   "xg_home": round(float(lam_h), 3), "xg_away": round(float(lam_a), 3), "goalie_unconfirmed": bool(unconf)}
            out.append(dict(event_id=ev.event_id, market="totals", outcome="Under", line=uline, model_prob=s["p_under"], fair_prob=f, edge=e, call=tag,
                            note=f"exp total {s['exp_total']:.2f} vs {total}; one-goal {s['one_goal_share']:.0%}, OT {s['p_ot_so']:.0%}. {note}",
                            components=_nc, model_number=_nc["model_number"], market_number=_nc["market_number"],
                            reasoning=jr.render_reasoning(market="totals", player="", selection="Under" if e > 0 else "Over", line=total,
                                                          model_prob=s["p_under"] if e > 0 else 1 - s["p_under"],
                                                          fair_prob=f if e > 0 else 1 - f, components=_nc, opponent="", status="", n_games=None)
                                      + (" Goalie unconfirmed at the time of the run, which alone makes this a Sit." if unconf else "")))
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
            elif unconf:
                r["call"] = "Sit (goalie unconfirmed)"
            elif abs(e) >= 0.03:
                r["call"] = f"{call_for(e, 0.03, 0.05, cap_lean=True)} {side}"
            else:
                r["call"] = "Sit"
            r["note"] = (r.get("note") or "") + f"; raw model {raw:.1%}, anchored {NHL_ANCHOR:.0%} to market, slate scale {k:.2f}"
            # reasoning is written here, after anchoring, so it quotes the number the call uses
            gtxt = "goalies unconfirmed — both run at team average" if unconf else f"goalies {gh_txt} / {ga_txt}"
            if r["market"] == "h2h":
                pm = r["model_prob"] if e > 0 else 1 - r["model_prob"]; pf = r["fair_prob"] if e > 0 else 1 - r["fair_prob"]
                r["reasoning"] = (f"Expected goals {lam_h:.2f}-{lam_a:.2f} (home first) from 5v5 and special-teams xG shrunk toward "
                                  f"league average; {gtxt}. Raw model {raw:.0%} home, anchored {NHL_ANCHOR:.0%} toward the market: "
                                  f"{side} {pm:.0%} versus {pf:.0%} at fair. Moneylines have no backtest; colour only. "
                                  f"Falsifier: the other side leads by two after the first period.")
            elif r["market"] == "totals":
                _nc = {"mu": round(float(lam_h + lam_a), 3), "sd": round(float(math.sqrt(max(lam_h + lam_a, 0.5))), 3),
                       "model_number": round(float(lam_h + lam_a), 3), "market_number": float(total), "xg_home": round(float(lam_h), 3)}
                pm = r["model_prob"] if e > 0 else 1 - r["model_prob"]; pf = r["fair_prob"] if e > 0 else 1 - r["fair_prob"]
                r["reasoning"] = jr.render_reasoning(market="totals", player="", selection=side, line=total, model_prob=pm,
                                                     fair_prob=pf, components=_nc, sport="NHL") + f" Goaltending: {gtxt}."
        try:
            ph = next((x["model_prob"] for x in out[n0:] if x["market"] == "h2h" and x["outcome"] == ev.home), None)
            fh = next((x["fair_prob"] for x in out[n0:] if x["market"] == "h2h" and x["outcome"] == ev.home), None)
            jr.record_numbers(con, event_id=ev.event_id, sport="NHL", home=ev.home, away=ev.away, commence=ev.commence,
                              margin=float(lam_h - lam_a), total=float(lam_h + lam_a), p_home=ph,
                              mkt_margin=None, mkt_total=float(total) if oline is not None else None, mkt_p_home=fh,
                              checks={"goalies_confirmed": not unconf, "sanity": not bad},
                              detail=f"xG {lam_h:.2f}-{lam_a:.2f}; {gh_txt} / {ga_txt}")
        except Exception:
            pass
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
        telegram("KRONOS — new Play edges\n\n" + "\n\n".join(msgs[:8]))


# ---------------------------------------------------------------- entry point
# ---------------------------------------------------------------- paper ledger, price edges, health checks
PAPER_BOOKS = ("pinnacle", "lowvig", "betonlineag", "draftkings", "fanduel", "fanatics", "betmgm", "williamhill_us", "espnbet", "betrivers",
               "betfair_ex_eu", "matchbook", "smarkets", "marathonbet", "williamhill", "hardrockbet", "ballybet", "bovada", "betus", "mybookieag")
PRICE_BOOKS = tuple(b.strip() for b in os.environ.get("PRICE_BOOKS", "draftkings,fanduel,fanatics").split(",") if b.strip())
PRICE_EDGE_MIN = 0.015     # +EV vs Pinnacle fair needed to flag a soft-book price
PRICE_EDGE_MIN_ADJ = 0.025 # higher bar when the fair had to be translated across a line gap

# Version tag on every paper row, so a rebuild never mixes with the evidence of the old
# model. Bump the string whenever a model's logic changes; the 150-bet clock is per version.
MODEL_VERSIONS = {
    ("NFL", "model"): "nfl-v3-anchor25-qbstatus-2026-10-26",
    ("NFL_PROPS", "model"): "nflprops-v2-paperonly-2026-10-26",
    ("NHL", "model"): "nhl-v3-goaliejson-2026-10-26",
    ("UFC", "model"): "ufc-v1-2026-10-24",
    ("NBA", "model"): "nba-v1-2026-10-26",
    ("MLB", "model"): "mlb-v1-2026-10-24",
    ("ANY", "price"): "price-v5-20books-2026-10-29",
}


def model_version(sport, market, kind="model"):
    if kind == "price":
        return MODEL_VERSIONS[("ANY", "price")]
    key = "NFL_PROPS" if (sport == "NFL" and str(market).startswith("player_")) else sport
    return MODEL_VERSIONS.get((key, kind), f"{key.lower()}-unversioned")


def _hours_to(commence, now):
    try:
        t = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
        return round((t - now).total_seconds() / 3600, 2)
    except Exception:
        return None


NFL_KEY = (3.0, 7.0, 10.0, 14.0)
# probability per point near the middle of the distribution (normal density at the mean)
GAME_SLOPE = {("NFL", "spreads"): 0.028, ("NFL", "totals"): 0.016,
              ("NBA", "spreads"): 0.031, ("NBA", "totals"): 0.021}


def _translate_fair(sport, market, outcome, pin_line, soft_line):
    """Change in fair probability for `outcome` when moving from Pinnacle's line to the soft
    book's line, or None when a linear translation would be dishonest."""
    d = soft_line - pin_line
    if abs(d) > 1.0:
        return None
    if market.startswith("player_"):
        # Not yet: the flat 0.045/pt prop slope is wrong in both directions (rebounds ~0.13/pt,
        # yards ~0.016/pt), which would manufacture edges. Queued until per-market dispersion
        # is measured from graded props.
        return None
    elif market == "spreads":
        if sport == "NFL" and any(min(abs(pin_line), abs(soft_line)) <= k <= max(abs(pin_line), abs(soft_line)) for k in NFL_KEY):
            return None
        if sport in ("NHL", "MLB"):
            return None          # puck/run lines are fixed at 1.5; a different number is a different bet
    elif market == "totals":
        if sport in ("NHL", "MLB"):
            return None          # half a goal/run is far from linear
    else:
        return None
    sign = jr._line_sign(market, outcome)
    if sign == 0:
        return None
    slope = GAME_SLOPE.get((sport, market))
    if slope is None:
        return None
    return sign * slope * d
BIG_EDGE = 0.08            # a model edge above this is treated as a bug, not a bet


def _is_call(c):
    return bool(c) and (str(c).startswith("Lean") or str(c).startswith("Play"))


def ensure_paper_schema(con):
    cols = {r[1] for r in con.execute("pragma table_info(bets)")}
    if "paper" not in cols:
        con.execute("alter table bets add column paper int default 0")
    con.execute("""create table if not exists value_edges(event_id text, market text, outcome text, player text, line real,
                   book text, price real, fair real, ev real, ts text, primary key(event_id, market, outcome, player, book))""")
    con.execute("create index if not exists bets_paper on bets(paper, event_id, market, selection, player)")
    # flags travel with every edge and every paper bet so their value can be measured
    for tbl, col, typ in (("value_edges", "flag", "text"), ("bets", "sharp_flag", "text"),
                          ("bets", "dk_bets_pct", "real"), ("bets", "dk_handle_pct", "real"),
                          ("model_probs", "flag", "text"), ("model_probs", "reasoning", "text"),
                          ("bets", "model_version", "text"), ("bets", "hours_to_start", "real"),
                          ("bets", "edge", "real"), ("bets", "fair_prob", "real"),
                          ("model_probs", "plain", "text"), ("value_edges", "activity", "text")):
        try:
            if col not in {r[1] for r in con.execute(f"pragma table_info({tbl})")}:
                con.execute(f"alter table {tbl} add column {col} {typ}")
        except Exception:
            pass


def _quote_for(con, event_id, market, outcome, player, line, books=PAPER_BOOKS):
    """Best available quote for a selection, preferring sharp books and the exact line."""
    rows = [dict(r) for r in con.execute("select book, line, price from last_quote where event_id=? and market=? and outcome=? and player=?",
                                         (event_id, market, outcome, player or ""))]
    if not rows:
        return None
    rank = {b: i for i, b in enumerate(books)}
    same = [r for r in rows if line is None or r["line"] is None or abs(float(r["line"]) - float(line)) < 1e-9]
    pool = same or rows
    pool.sort(key=lambda r: rank.get(r["book"], 99))
    return pool[0]


def _selection(row):
    """(selection, selection line, model prob for that selection) from a model row's call."""
    call = str(row["call"]); market = row["market"]
    words = call.split(" ")[1:]
    if market in ("totals",) or market.startswith("player_"):
        sel = words[0] if words else row["outcome"]
    else:
        sel = " ".join(words)
    mine = sel == row["outcome"]
    p = row["model_prob"] if mine else (1 - row["model_prob"] if row["model_prob"] is not None else None)
    line = row.get("line")
    if market == "spreads" and not mine and line is not None:
        line = -float(line)
    return sel, line, p


def paper_log(con, rows, now):
    """Record every Lean/Play as a paper bet at the price available when the call was first made."""
    ensure_paper_schema(con)
    n = 0
    for r in rows:
        # Sits are logged as well, at zero stake. Without them the ledger contains
        # only large edges and cannot answer whether edge size predicts anything —
        # the exact question raised when the three largest edges were the three
        # largest misses.
        is_bet = _is_call(r.get("call"))
        sel, line, p = _selection(r)
        player = r.get("player", "") or ""
        if con.execute("select 1 from bets where paper=1 and event_id=? and market=? and selection=? and player=? and note not like 'price edge%'",
                       (r["event_id"], r["market"], sel, player)).fetchone():
            continue
        ev = con.execute("select sport, home, away, commence from events where event_id=?", (r["event_id"],)).fetchone()
        if not ev or ev["commence"] <= now.isoformat(timespec="seconds"):
            continue
        q = _quote_for(con, r["event_id"], r["market"], sel, player, line)
        if not q:
            continue
        import json as _json
        comp = r.get("components") or {}
        mnum, knum = r.get("model_number"), r.get("market_number")
        ratio = None
        if mnum is not None and knum is not None:
            mae = jr.measure_model_error(con, ev["sport"], r["market"]) or \
                  jr.DEFAULT_MODEL_MAE.get(r["market"])
            ratio = jr.edge_ratio(float(mnum) - float(knum), mae)
        stake = 0.0 if not is_bet else (jr.size_from_ratio(ratio, r.get("edge") or 0.0,
                                        cap_lean=r["market"].startswith("player_"))[1] or 1.0)
        con.execute("""insert into bets(placed_ts, sport, event_id, game, commence, market, selection, player, line, price, book, stake,
                       model_prob, note, paper, comp_json, ratio, model_number, market_number,
                       evidence_kind, corr_group, falsifier, steelman, stake_pre_corr, logged_call, sharp_flag, dk_bets_pct, dk_handle_pct)
                       values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (now.isoformat(timespec="seconds"), ev["sport"], r["event_id"], f"{ev['away']} @ {ev['home']}", ev["commence"],
                     r["market"], sel, player, q["line"] if q["line"] is not None else line, q["price"], q["book"], stake,
                     None if p is None else round(float(p), 4), f"{r['call']} | {r.get('note') or ''}"[:300],
                     _json.dumps(comp), ratio, mnum, knum, "model",
                     jr.correlation_group(ev["sport"], r["event_id"], r["market"], sel, player),
                     jr.falsifier(r["market"], sel, line, mnum, player),
                     jr.steelman(r["market"], mnum, knum, sel), stake, r.get("call"),
                     r.get("flag"), (r.get("_sig") or {}).get("dk_bets"), (r.get("_sig") or {}).get("dk_handle")))
        _fp = r.get("fair_prob")
        if _fp is not None and sel != r.get("outcome"):
            _fp = 1 - _fp
        _ed = r.get("edge")
        if _ed is not None and sel != r.get("outcome"):
            _ed = -_ed
        con.execute("update bets set model_version=?, hours_to_start=?, edge=?, fair_prob=? where id=last_insert_rowid()",
                    (model_version(ev["sport"], r["market"]), _hours_to(ev["commence"], now), _ed, _fp))
        n += 1
    return n



CONSENSUS_W = {"pinnacle": 3.0, "betfair_ex_eu": 3.0, "lowvig": 2.0, "betonlineag": 2.0,
               "matchbook": 2.0, "smarkets": 2.0, "marathonbet": 1.5}


def _consensus_fair(rows, outcome, line, exclude_book):
    """De-vigged fair probability for `outcome` averaged across every book quoting both sides at
    the same line, excluding the book offering the edge. Pinnacle weighs 3, other sharp books 2,
    retail 1. Returns (fair, n_books) or (None, 0).

    2026-10-28: Pinnacle alone is a weak referee on props (low limits, softer lines) — NFL price-edge
    props beat Pinnacle's close 69% yet went 12-22 — so a prop edge must survive the whole market.
    """
    by_book = {}
    for r in rows:
        if r["book"] == exclude_book or not r["price"]:
            continue
        by_book.setdefault(r["book"], {})[r["outcome"]] = r
    num = den = 0.0
    n = 0
    for bk, sides in by_book.items():
        if outcome not in sides or len(sides) != 2:
            continue
        me = sides[outcome]
        other = [v for k, v in sides.items() if k != outcome][0]
        if line is not None:
            if me["line"] is None or abs(float(me["line"]) - float(line)) > 1e-9:
                continue
        f = fair_two_way(me["price"], other["price"])
        w = CONSENSUS_W.get(bk, 1.0)
        num += f * w; den += w; n += 1
    if n < 3 or den <= 0:
        return None, n
    return num / den, n


def price_edges(con, now):
    """Soft-book prices that beat Pinnacle's de-vigged fair price: an edge that needs no model."""
    ensure_paper_schema(con)
    lim = (now + timedelta(hours=48)).isoformat(timespec="seconds")
    evs = [dict(r) for r in con.execute("select event_id, sport, home, away, commence from events where commence>? and commence<?",
                                        (now.isoformat(timespec="seconds"), lim))]
    con.execute("delete from value_edges where event_id in (select event_id from events where commence>?)", (now.isoformat(timespec="seconds"),))
    out = []
    for ev in evs:
        q = [dict(r) for r in con.execute("select * from last_quote where event_id=?", (ev["event_id"],))]
        if not q:
            continue
        groups = {}
        for r in q:
            groups.setdefault((r["market"], r["player"] or ""), []).append(r)
        for (market, player), rows in groups.items():
            # referee: Pinnacle, then BetOnline/LowVig, then the exchanges (Betfair is the sharpest soccer price)
            sharp = ([r for r in rows if r["book"] == "pinnacle"] or [r for r in rows if r["book"] in ("lowvig", "betonlineag")]
                     or [r for r in rows if r["book"] in ("betfair_ex_eu", "matchbook", "smarkets")])
            outs = {r["outcome"] for r in sharp}
            three_way = market == "h2h" and len(outs) == 3          # soccer: home / draw / away
            if len(outs) != 2 and not three_way:
                continue
            by = {}
            for r in sharp:
                by.setdefault(r["outcome"], r)
            if three_way:
                ks = list(by)
                fw = jr.power_devig([by[k]["price"] for k in ks])
                if not fw:
                    continue
                fair = {k: fw[i] for i, k in enumerate(ks)}
                lines = {k: None for k in ks}
            else:
                o1, o2 = list(by.values())[:2]
                if o1["line"] is not None and o2["line"] is not None and market != "spreads" and abs(float(o1["line"]) - float(o2["line"])) > 1e-9:
                    continue
                fair = {o1["outcome"]: fair_two_way(o1["price"], o2["price"]), o2["outcome"]: fair_two_way(o2["price"], o1["price"])}
                lines = {o1["outcome"]: o1["line"], o2["outcome"]: o2["line"]}
            for r in rows:
                if r["book"] not in PRICE_BOOKS or r["outcome"] not in fair or not r["price"]:
                    continue
                L = lines[r["outcome"]]
                if (L is None) != (r["line"] is None):
                    continue
                f_use, adj = fair[r["outcome"]], 0.0
                if L is not None and abs(float(L) - float(r["line"])) > 1e-9:
                    # Line-translated fair (2026-10-26): retail lines often sit a half-point off
                    # Pinnacle and were all skipped. Translate only where a linear slope is honest.
                    adj = _translate_fair(ev["sport"], market, r["outcome"], float(L), float(r["line"]))
                    if adj is None:
                        continue
                    f_use = min(max(f_use + adj, 0.01), 0.99)
                evv = float(r["price"]) * f_use - 1
                if adj and evv < PRICE_EDGE_MIN_ADJ:
                    continue
                if evv >= PRICE_EDGE_MIN and player:
                    # roster cross-check (#5): The Odds API sometimes returns a player's prop
                    # under the wrong event. If his last known team is neither side, drop it.
                    try:
                        import jerry_feeds as _jf
                        ok = _jf.label_ok(con, ev["sport"], player, ev["home"], ev["away"])
                    except Exception:
                        ok = None
                    if ok is False:
                        con.execute("insert or replace into meta(k,v) values(?,?)",
                                    (f"label_reject_{ev['sport']}_{player}"[:120], f"{now.isoformat(timespec='seconds')} {ev['away']} @ {ev['home']}"))
                        continue
                # active-player gate (2026-10-27): 3 of 10 card bets on 10-26 were players who did not play.
                # Did not play last game -> no edge. Under 12 NBA minutes -> tagged LOW MINUTES. No team
                # on file -> tagged MINUTES UNVERIFIED. NFL has no box-score feed here, so it is not gated.
                activity = None
                if evv >= PRICE_EDGE_MIN and player and ev["sport"] in ("NBA", "NHL"):
                    try:
                        import jerry_feeds as _jf
                        st_, mins_ = _jf.active_status(con, ev["sport"], player)
                    except Exception:
                        st_, mins_ = "unknown", None
                    if st_ == "dnp":
                        con.execute("insert or replace into meta(k,v) values(?,?)",
                                    (f"inactive_skip_{ev['sport']}_{player}"[:120], f"{now.isoformat(timespec='seconds')} {ev['away']} @ {ev['home']}"))
                        continue
                    activity = {"low": f"LOW MINUTES ({mins_:.0f} last game)" if mins_ is not None else "LOW MINUTES",
                                "unknown": "MINUTES UNVERIFIED"}.get(st_)
                # ---- 2026-10-28 checks: consensus price, injuries, role change, projection ----
                tags = [activity] if activity else []
                if evv >= PRICE_EDGE_MIN and not adj and market != "spreads":
                    f_con, n_con = _consensus_fair(rows, r["outcome"], r["line"], r["book"])
                    if f_con is not None:
                        f_use = f_con
                        evv = float(r["price"]) * f_use - 1
                        tags.append(f"CONSENSUS {n_con} BOOKS")
                        if evv < PRICE_EDGE_MIN:
                            con.execute("insert or replace into meta(k,v) values(?,?)",
                                        ("consensus_reject_last", f"{now.isoformat(timespec='seconds')} {player} {market} {r['outcome']}"))
                            continue
                if evv >= PRICE_EDGE_MIN and player and ev["sport"] in ("NBA", "NHL"):
                    try:
                        import jerry_feeds as _jf
                        inj = _jf.injury_status(con, ev["sport"], player)
                    except Exception:
                        inj = None
                    if inj == "out":
                        continue
                    if inj == "doubtful":
                        tags.append("QUESTIONABLE")
                    try:
                        pc = _jf.prop_check(con, ev["sport"], player, market, r["outcome"], r["line"])
                    except Exception:
                        pc = {"verdict": None, "role_change": False}
                    # A line far above the player's recent role is the market pricing news the box
                    # scores can't see. Never fade it (Paul Reed U12.5, 2026-10-27: started, scored 22).
                    if pc.get("role_change"):
                        if r["outcome"] in ("Under", "No"):
                            con.execute("insert or replace into meta(k,v) values(?,?)",
                                        (f"role_change_skip_{player}"[:120], f"{now.isoformat(timespec='seconds')} {pc.get('note')}"))
                            continue
                        tags.append("ROLE CHANGE")
                    elif pc.get("verdict") == "disagrees":
                        con.execute("insert or replace into meta(k,v) values(?,?)",
                                    (f"projection_skip_{player}"[:120], f"{now.isoformat(timespec='seconds')} {pc.get('note')}"))
                        continue
                    elif pc.get("verdict") == "agrees":
                        tags.append(f"PROJECTION AGREES ({pc['p_side']:.0%})")
                    elif pc.get("verdict") == "neutral":
                        tags.append("PROJECTION NEUTRAL")
                    else:
                        tags.append("NO PROJECTION")
                # line move since open on a prop: a large move is news, not a price to fade
                if evv >= PRICE_EDGE_MIN and player and r["line"] is not None:
                    o_ = con.execute("""select line from opens where event_id=? and market=? and outcome=? and player=?
                                        and book in ('pinnacle','draftkings','fanduel') and line is not null limit 1""",
                                     (ev["event_id"], market, r["outcome"], player)).fetchone()
                    if o_ and o_[0] is not None:
                        mv = float(r["line"]) - float(o_[0])
                        big = abs(mv) >= max(1.0, 0.12 * abs(float(o_[0])))
                        against = (mv > 0 and r["outcome"] in ("Under", "No")) or (mv < 0 and r["outcome"] in ("Over", "Yes"))
                        if big and against:
                            continue
                        if big:
                            tags.append(f"LINE MOVED {float(o_[0]):g}->{float(r['line']):g}")
                activity = " · ".join(t for t in tags if t) or None
                if evv >= PRICE_EDGE_MIN:
                    row = dict(event_id=ev["event_id"], market=market, outcome=r["outcome"], player=player, line=r["line"], book=r["book"],
                               price=r["price"], fair=round(f_use, 4), ev=round(evv, 4), sport=ev["sport"], home=ev["home"], away=ev["away"],
                               commence=ev["commence"], line_adj=bool(adj), pin_line=L, activity=activity)
                    try:
                        _sig = jr.sharp_signal(con, ev["event_id"], market, r["outcome"], player)
                    except Exception:
                        _sig = None
                    _lab = _sig.get("label") if _sig else "NO READ"
                    # 'FADE PUBLIC' on a price edge measures the discount that *is* the edge — no information
                    _lab = " · ".join(t for t in _lab.split(" · ") if t != "FADE PUBLIC") or "NEUTRAL"
                    row["flag"] = _lab
                    row["flag_text"] = _sig.get("text") if _sig else ""
                    row["_sig"] = _sig
                    out.append(row)
                    con.execute("""insert or replace into value_edges(event_id,market,outcome,player,line,book,price,fair,ev,ts,flag,activity)
                                   values(?,?,?,?,?,?,?,?,?,?,?,?)""",
                                (row["event_id"], market, r["outcome"], player, r["line"], r["book"], r["price"], row["fair"], row["ev"],
                                 now.isoformat(timespec="seconds"), row["flag"], row.get("activity")))
    # paper-trade them too (one per outcome, first book seen)
    n = 0
    for row in out:
        if con.execute("select 1 from bets where paper=1 and event_id=? and market=? and selection=? and player=? and note like 'price edge%'",
                       (row["event_id"], row["market"], row["outcome"], row["player"])).fetchone():
            continue
        con.execute("""insert into bets(placed_ts, sport, event_id, game, commence, market, selection, player, line, price, book, stake,
                       model_prob, note, paper) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
                    (now.isoformat(timespec="seconds"), row["sport"], row["event_id"], f"{row['away']} @ {row['home']}", row["commence"],
                     row["market"], row["outcome"], row["player"], row["line"], row["price"], row["book"], 1.0, row["fair"],
                     (f"price edge (line-adj from {row['pin_line']}) {row['ev']:+.1%} vs translated fair {row['fair']:.1%}"
                      if row.get("line_adj") else f"price edge {row['ev']:+.1%} vs pinnacle fair {row['fair']:.1%}")))
        _s = row.get("_sig") or {}
        con.execute("""update bets set evidence_kind=?, corr_group=?, sharp_flag=?, dk_bets_pct=?, dk_handle_pct=?,
                       model_version=?, hours_to_start=?, edge=?, fair_prob=?
                       where id=last_insert_rowid()""",
                    ("price_adj" if row.get("line_adj") else "price",
                     jr.correlation_group(row["sport"], row["event_id"], row["market"], row["outcome"], row["player"]),
                     row.get("flag"), _s.get("dk_bets"), _s.get("dk_handle"),
                     model_version(row["sport"], row["market"], "price"), _hours_to(row["commence"], now),
                     row["ev"], row["fair"]))
        n += 1
    return out, n



# ----------------------------------------------------------------- KRONOS shadow card
CARD_SIZE = 10
CARD_MAX_PER_GAME = 2
CARD_MAX_SAME_DIRECTION = 0.6
CARD_LOCK_MIN = 90          # a pick locks 90 minutes before its start, from the freshest numbers
CARD_PRIOR_N = 100          # shrink toward zero by n/(n+100): the biggest raw edges are the likeliest errors
CARD_EXCLUDE = ("ROLE CHANGE", "MINUTES UNVERIFIED", "QUESTIONABLE", "NO PROJECTION", "STALE?")


def _category_record(con, sport, is_prop):
    q = """select count(*), sum(case when result in ('W','L','P') then pnl else 0 end),
                  sum(case when result in ('W','L','P') then stake else 0 end)
           from bets where paper=1 and sport=? and evidence_kind in ('price','price_adj') and result in ('W','L','P')
           and market """ + ("like 'player_%'" if is_prop else "not like 'player_%'")
    n, u, st = con.execute(q, (sport,)).fetchone()
    return int(n or 0), (float(u) / float(st)) if st else None


def build_card(con, now, edges):
    """KRONOS's own nightly card, chosen by rule rather than by narrative, logged as a shadow so its
    results can be compared with the full flat-stake price-edge ledger and with Claude's hand-built cards.

    Rank = EV x n/(n+100) (category reliability), halved for a category losing money over 50+ graded.
    Edges more than twice the night's median for their sport/market type are tagged STALE? and skipped —
    on 2026-10-27 the two largest edges (Reed +7.0%, Shamet +6.9%) both lost.
    """
    try:
        cols = {r[1] for r in con.execute("pragma table_info(bets)")}
        for c, t in (("card", "int"), ("card_rank", "int"), ("card_score", "real"), ("card_ts", "text"), ("card_tags", "text")):
            if c not in cols:
                con.execute(f"alter table bets add column {c} {t}")
    except Exception:
        pass
    import statistics as _st
    from zoneinfo import ZoneInfo as _Z
    et = _Z("America/New_York")
    today = now.astimezone(et).date()
    cand = []
    groups = {}
    for e in edges:
        groups.setdefault((e["sport"], str(e["market"]).startswith("player_")), []).append(e["ev"])
    med = {k: _st.median(v) for k, v in groups.items() if v}
    rel_cache = {}
    seen = set()
    for e in sorted(edges, key=lambda x: -x["ev"]):
        key = (e["event_id"], e["market"], e["outcome"], e["player"])
        if key in seen:
            continue                      # one book per outcome: the best-priced one sorts first
        seen.add(key)
        try:
            st = datetime.fromisoformat(str(e["commence"]).replace("Z", "+00:00"))
        except ValueError:
            continue
        if st.astimezone(et).date() != today or st <= now:
            continue
        tags = (e.get("activity") or "")
        is_prop = str(e["market"]).startswith("player_")
        m = med.get((e["sport"], is_prop))
        if m and e["ev"] > 2 * m and len(groups[(e["sport"], is_prop)]) >= 5:
            tags = (tags + " · STALE?").strip(" ·")
        if any(x in tags for x in CARD_EXCLUDE):
            continue
        ck = (e["sport"], is_prop)
        if ck not in rel_cache:
            n, roi = _category_record(con, *ck)
            # a small floor so an untested category still ranks by edge instead of tying at zero
            r_ = (n + 10) / (n + 10 + CARD_PRIOR_N)
            if n >= 50 and roi is not None and roi < 0:
                r_ *= 0.5
            rel_cache[ck] = r_
        score = e["ev"] * rel_cache[ck]
        cand.append((score, st, e, tags))
    cand.sort(key=lambda c: -c[0])
    locked = [dict(r) for r in con.execute("select * from bets where paper=1 and card=1 and substr(card_ts,1,10)>=?",
                                            ((now - timedelta(hours=30)).date().isoformat(),))]
    locked = [b for b in locked if datetime.fromisoformat(str(b["commence"]).replace("Z", "+00:00")).astimezone(et).date() == today]
    per_game, dirs = {}, {"up": 0, "down": 0}
    for b in locked:
        per_game[b["event_id"]] = per_game.get(b["event_id"], 0) + 1
        dirs["up" if b["selection"] in ("Over", "Yes") else "down"] += 1
    slots = CARD_SIZE - len(locked)
    picked = 0
    for score, st, e, tags in cand:
        if slots <= 0:
            break
        if per_game.get(e["event_id"], 0) >= CARD_MAX_PER_GAME:
            continue
        d = "up" if e["outcome"] in ("Over", "Yes") else "down"
        total = sum(dirs.values()) + 1
        if total >= 5 and (dirs[d] + 1) / total > CARD_MAX_SAME_DIRECTION:
            continue
        # rank among the whole remaining night, lock only when close to start
        if (st - now).total_seconds() > CARD_LOCK_MIN * 60:
            per_game[e["event_id"]] = per_game.get(e["event_id"], 0) + 1
            dirs[d] += 1
            slots -= 1                    # reserve its slot so later games can't crowd it out
            continue
        row = con.execute("""select id from bets where paper=1 and event_id=? and market=? and selection=? and player=?
                             and note like 'price edge%' and (card is null or card=0) order by id desc limit 1""",
                          (e["event_id"], e["market"], e["outcome"], e["player"])).fetchone()
        if not row:
            continue
        con.execute("update bets set card=1, card_rank=?, card_score=?, card_ts=?, card_tags=? where id=?",
                    (len(locked) + picked + 1, round(score, 5), now.isoformat(timespec="seconds"), tags, row[0]))
        per_game[e["event_id"]] = per_game.get(e["event_id"], 0) + 1
        dirs[d] += 1
        slots -= 1
        picked += 1
    con.commit()
    return picked

def health_check(db_path, results):
    """After every model run: catch broken data before it reaches the board. Returns the list of problems."""
    now = datetime.now(UTC)
    con = sqlite3.connect(db_path, timeout=60); con.row_factory = sqlite3.Row
    probs = []
    d = os.environ.get("JERRY_DATA_DIR", "/data")
    try:
        if os.path.exists(d) and os.stat(d).st_dev == os.stat("/").st_dev:
            probs.append(f"{d} is not a mounted volume: data will be lost on the next redeploy")
    except Exception:
        pass
    for s, v in (results or {}).items():
        if isinstance(v, str):
            probs.append(f"{s} model error: {v}")
    t0 = now.isoformat(timespec="seconds"); t1 = (now + timedelta(hours=36)).isoformat(timespec="seconds")
    for sport in ("NFL", "NHL", "MLB", "UFC", "NBA"):
        ev = [r[0] for r in con.execute("select event_id from events where sport=? and commence>? and commence<?", (sport, t0, t1))]
        if not ev:
            continue
        marks = ",".join("?" * len(ev))
        nq = con.execute(f"select count(*) from last_quote where event_id in ({marks})", ev).fetchone()[0]
        nm = con.execute(f"select count(*) from model_probs where event_id in ({marks})", ev).fetchone()[0]
        if not nq:
            probs.append(f"{sport}: {len(ev)} games within 36h but no prices on the board")
        elif not nm and sport in (results or {}) and sport != "NBA":
            probs.append(f"{sport}: prices present but the model produced 0 rows")
    # NHL goalie feed: a game within 90 minutes with zero confirmed starters league-wide means the parser broke
    t90 = (now + timedelta(minutes=90)).isoformat(timespec="seconds")
    nhl_soon = con.execute("select count(*) from events where sport='NHL' and commence>? and commence<?", (t0, t90)).fetchone()[0]
    if nhl_soon:
        if LAST_GOALIES is None:
            probs.append("NHL: games within 90 min but the goalie feed has not been read this session")
        elif LAST_GOALIES[1] == 0:
            probs.append("NHL: DailyFaceoff parsed 0 teams — goalie parser broken, every total is a Sit")
        elif LAST_GOALIES[2] == 0:
            probs.append(f"NHL: {nhl_soon} games within 90 min and 0 confirmed goalies ({LAST_GOALIES[1]} teams parsed)")
    spl = (results or {}).get("SPLITS")
    if isinstance(spl, dict) and not (spl.get("table_rows") or spl.get("article_rows")):
        probs.append("DK splits: 0 rows — public-money input is empty; flags rest on Pinnacle-vs-retail only")
    # a [PAPER ONLY] model must never reach Telegram/card as a live Play
    leak = con.execute("""select count(*) from model_probs m join events e using(event_id) where e.commence>?
                          and m.call like 'Play%' and m.call like '%PAPER ONLY%'""", (t0,)).fetchone()[0]
    if leak:
        probs.append(f"{leak} PAPER ONLY rows tagged Play")
    big = con.execute("""select count(*) from model_probs m join events e using(event_id) where e.commence>? and abs(coalesce(m.edge,0))>?
                         and (m.call like 'Lean%' or m.call like 'Play%')""", (t0, BIG_EDGE)).fetchone()[0]
    if big:
        probs.append(f"{big} live calls with edge above {BIG_EDGE:.0%}: model output suspect")
    for k, v in con.execute("select k, v from meta where k like 'error_%'"):
        try:
            if now - datetime.fromisoformat(v[:19]).replace(tzinfo=UTC) < timedelta(hours=1):
                probs.append(f"{k}: {v[20:120]}")
        except Exception:
            pass
    con.execute("insert into meta(k,v) values(?,?) on conflict(k) do update set v=excluded.v",
                ("health", f"{t0} " + ("ok" if not probs else " || ".join(probs))[:900]))
    if probs:
        key = str(abs(hash(" | ".join(sorted(probs)))))
        last = con.execute("select v from meta where k='health_sent'").fetchone()
        if not last or last[0].split(" ", 1)[0] != key or now - datetime.fromisoformat(last[0].split(" ", 1)[1]).replace(tzinfo=UTC) > timedelta(hours=6):
            telegram("KRONOS HEALTH CHECK\n" + "\n".join(f"- {p}" for p in probs))
            con.execute("insert into meta(k,v) values(?,?) on conflict(k) do update set v=excluded.v", ("health_sent", f"{key} {t0}"))
    con.commit(); con.close()
    log(f"health: {'ok' if not probs else probs}")
    return probs


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
            con.execute("delete from meta where k=?", (f"error_model_{s}",))   # clear a stale error once the model runs clean
            alert_new_plays(con, rows)
            con.commit()
            try:
                paper_log(con, rows, now); con.commit()
            except Exception as e:
                con.rollback(); log(f"paper log failed for {s}: {e}")
        except Exception as e:
            con.rollback()
            results[s] = f"error: {e}"
            log(f"{s} failed: {e}")
            con.execute("insert into meta(k,v) values(?,?) on conflict(k) do update set v=excluded.v",
                        (f"error_model_{s}", f"{now.isoformat(timespec='seconds')} {e}"[:200]))
            con.commit()
    try:
        edges_, n_new = price_edges(con, now); con.commit()
        results["PRICE"] = len(edges_)
        try:
            results["CARD"] = build_card(con, now, edges_)
        except Exception as _e:   # noqa: BLE001 - the shadow card must never block a refresh
            results["CARD"] = f"error: {_e}"
    except Exception as e:
        con.rollback(); results["PRICE"] = f"error: {e}"; log(f"price edges failed: {e}")
    con.close()
    return results
