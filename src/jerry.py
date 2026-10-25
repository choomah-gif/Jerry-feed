#!/usr/bin/env python3
"""
jerry.py — Jerry v2.3 (models, ledger, backtests). Live odds now come from jerry_desk.py. Locked 2026-10-21; UFC module 2026-10-24; budget-aware odds tracker + World Series 2026-10-24.

Three layers: data (pull) -> model (probabilities, backtested vs the close) -> Jerry/Claude (verify, write, shade <= 3%).
Scored on closing-line value, Brier, and ROI with a confidence interval. Weights frozen until 150 graded bets per market.

  pip install pandas numpy scipy scikit-learn requests
  export ODDS_API_KEY=...        # the-odds-api.com, free tier

  python jerry.py backtest-nfl                    # walk-forward 2022-2025 vs nflverse closing lines
  python jerry.py nfl-week 2026 7                 # model output for a week (needs lines in nflverse or your snapshot)
  python jerry.py usage                           # Odds API credits used/left this month + projection (free call)
  python jerry.py budget                          # planned monthly credit spend vs your quota
  python jerry.py snapshot [NFL NHL MLB WS UFC]   # lines -> odds_snapshots.csv; skips sports with no events in 48h
  python jerry.py close-watch                     # cron every 5 min: grabs closing lines just before starts
  python jerry.py close                           # manual close pull, all sports
  python jerry.py backfill NHL 2023-10-10 2024-04-18 [--dry-run]   # historical closes for backtests (resumable)
  python jerry.py nhl BOS TOR 6.0 --hg 0.12 --ag -0.05 --away-b2b
  python jerry.py mlb "Los Angeles Dodgers" "New York Yankees" 8.5 --home-fip 3.2 --away-fip 3.9
  python jerry.py bet --game_date 2026-10-26 --sport NFL --game "DAL@PHI" --market spread --selection "PHI -6.5" \
                      --line -6.5 --odds 1.91 --book draftkings --model-prob 0.52 --jerry-prob 0.53 --override-reason "..."
  python jerry.py grade B00001 W --close-odds 1.87 --close-line -7 --close-book pinnacle
  python jerry.py report                          # ROI/CLV/Brier/calibration + are Jerry's overrides helping?
  python jerry.py backtest-ufc                    # walk-forward 2016-2026 vs historical UFC odds
  python jerry.py ufc-card ufc333.json            # model every fight on a card vs posted odds

Data sources (all public): nflverse (scores + closing lines 1999->, team EPA), The Odds API (live/close lines),
MoneyPuck (NHL xG, goalie GSAx), MLB Stats API (runs, starters),
UFCStats via github.com/Greco1899/scrape_ufc_stats (every UFC fight + round stats), historical UFC odds via
github.com/shortlikeafox/ultimate_ufc_dataset. Jerry instructions: see JERRY_PROMPT.md.
"""
import argparse, json, os, re, sys, time, unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm, poisson, nbinom
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler



# ======================================================================
# jerry/config.py
# ======================================================================
"""Central config. Secrets come from environment variables, never from code."""

ROOT = Path(os.environ.get("JERRY_DATA_DIR") or Path(__file__).resolve().parent)   # Railway: the volume
CACHE = ROOT / "cache"
DATA = ROOT / "data"
CACHE.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

# The Odds API (https://the-odds-api.com). Free tier = 500 CREDITS/month, reset on the 1st.
# Cost of one /odds call = (number of markets) x (number of regions). Up to 10 named bookmakers
# count as ONE region. So we name books instead of using regions=us — that also brings in Pinnacle,
# which lives in the EU region and was missing from v2 (the CLV reference book never got pulled).
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_BOOKMAKERS = "pinnacle,draftkings,fanduel,betmgm,williamhill_us,betrivers,espnbet,fanatics,bovada,betonlineag"  # 10 = 1 region
ODDS_MONTHLY_CREDITS = int(os.environ.get("ODDS_MONTHLY_CREDITS", "20000"))  # 20K plan
ODDS_RESERVE_FOR_CLOSES = 1000   # snapshot()/backfill refuse to spend below this; close-watch may use it
CLV_REFERENCE_BOOK = "pinnacle"

LEDGER_PATH = DATA / "ledger.csv"
ODDS_SNAPSHOT_PATH = DATA / "odds_snapshots.csv"

# How many graded bets per market before weights may change. Not 20.
RECALIBRATION_MIN_BETS = 150
# Max shade Jerry (LLM) may apply to a model probability, absolute.
LLM_OVERRIDE_CAP = 0.03


# ======================================================================
# jerry/eval/metrics.py
# ======================================================================


def brier(p, y) -> float:
    p = np.asarray(p, float); y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y, eps=1e-6) -> float:
    p = np.clip(np.asarray(p, float), eps, 1 - eps); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_table(p, y, bins=(0, .4, .45, .5, .55, .6, 1.0)) -> pd.DataFrame:
    df = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float)})
    df["bucket"] = pd.cut(df["p"], bins=list(bins), include_lowest=True)
    t = df.groupby("bucket", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).reset_index()
    t["bucket"] = t["bucket"].astype(str)
    return t


def implied_prob(decimal_odds: float) -> float:
    return 1.0 / decimal_odds


def devig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Multiplicative de-vig of a two-way market, decimal odds in -> fair probs out."""
    a, b = 1 / odds_a, 1 / odds_b
    return a / (a + b), b / (a + b)


def clv_cents(bet_decimal: float, close_decimal: float) -> float:
    """Closing-line value in 'cents' of American odds, positive = you beat the close."""
    def to_american(d):
        return (d - 1) * 100 if d >= 2 else -100 / (d - 1)
    return float(to_american(bet_decimal) - to_american(close_decimal))


def roi_ci(pnl_units: np.ndarray, stakes: np.ndarray, n_boot=5000, seed=0) -> tuple[float, float, float]:
    """ROI with bootstrap 90% CI. Shows how little 50 bets tells you."""
    rng = np.random.default_rng(seed)
    pnl = np.asarray(pnl_units, float); st = np.asarray(stakes, float)
    roi = pnl.sum() / st.sum()
    idx = rng.integers(0, len(pnl), size=(n_boot, len(pnl)))
    boots = pnl[idx].sum(1) / st[idx].sum(1)
    return float(roi), float(np.percentile(boots, 5)), float(np.percentile(boots, 95))


# ======================================================================
# jerry/models/margin.py
# ======================================================================
"""Empirical margin distributions. Key numbers (3, 7 in NFL; 1 goal / OT in NHL)
come out of history, not out of a rule telling a language model to 'raise mass'."""


class EmpiricalMargin:
    """Fit on residuals (actual margin - predicted margin). Then P(cover) is just
    the share of historical residuals that would push the predicted margin past the line.
    Residuals are rounded to integers so integer key numbers keep their mass."""

    def __init__(self, residuals: np.ndarray):
        r = np.asarray(residuals, dtype=float)
        self.res = np.round(r[~np.isnan(r)]).astype(int)
        self.sigma = float(self.res.std())

    def margin_pmf(self, pred_margin: float) -> pd.Series:
        m = np.round(pred_margin + self.res).astype(int)
        return pd.Series(m).value_counts(normalize=True).sort_index()

    def p_home_cover(self, pred_margin: float, spread_line: float) -> tuple[float, float]:
        """spread_line in nflverse convention (positive = home favored).
        Returns (p_cover, p_push)."""
        m = pred_margin + self.res
        cover = (m > spread_line).mean()
        push = (np.round(m) == spread_line).mean() if float(spread_line).is_integer() else 0.0
        return float(cover), float(push)

    def p_home_win(self, pred_margin: float) -> float:
        m = pred_margin + self.res
        return float((m > 0).mean() + 0.5 * (np.round(m) == 0).mean())

    def key_mass(self, pred_margin: float, keys=(3, 7)) -> dict:
        pmf = self.margin_pmf(pred_margin)
        return {f"abs_margin_{k}": float(pmf.get(k, 0) + pmf.get(-k, 0)) for k in keys}

    def iqr(self, pred_margin: float) -> tuple[float, float]:
        m = pred_margin + self.res
        return float(np.percentile(m, 25)), float(np.percentile(m, 75))


# ======================================================================
# jerry/data/nflverse.py
# ======================================================================
"""nflverse public data: schedules with closing lines (1999-present) and team-week stats.

No API key. Files are GitHub release assets, cached locally.
spread_line convention (nflverse): positive = HOME team favored by that many points.
result = home_score - away_score. Home covers iff result > spread_line.
"""

NFLVERSE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"


def _cached_csv(url: str, name: str, refresh: bool = False) -> pd.DataFrame:
    p = CACHE / name
    if p.exists() and not refresh:
        return pd.read_csv(p, low_memory=False)
    r = requests.get(url, timeout=60); r.raise_for_status()   # requests follows GitHub's redirect to release-assets
    p.write_bytes(r.content)
    return pd.read_csv(p, low_memory=False)


def games(refresh: bool = False) -> pd.DataFrame:
    """All games 1999-present incl. spread_line, total_line, moneylines (closing, per nflverse)."""
    g = _cached_csv(f"{NFLVERSE_BASE}/schedules/games.csv", "nfl_games.csv", refresh)
    g["gameday"] = pd.to_datetime(g["gameday"])
    return g


def team_week_stats(seasons, refresh: bool = False) -> pd.DataFrame:
    """Offensive team-week box + EPA. Defensive EPA is derived by joining opponent rows."""
    frames = []
    for s in seasons:
        frames.append(_cached_csv(f"{NFLVERSE_BASE}/stats_team/stats_team_week_{s}.csv", f"nfl_team_week_{s}.csv", refresh))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["season_type"] == "REG"].copy()
    plays = df["attempts"] + df["carries"] + df["sacks_suffered"]
    df["off_epa_play"] = (df["passing_epa"] + df["rushing_epa"]) / plays.clip(lower=1)
    df["off_plays"] = plays
    df["points_for"] = 0  # filled from games() in features
    # Defensive EPA/play = opponent's offensive EPA/play in the same game
    opp = df[["season", "week", "team", "off_epa_play"]].rename(
        columns={"team": "opponent_team", "off_epa_play": "def_epa_play"})
    df = df.merge(opp, on=["season", "week", "opponent_team"], how="left")
    return df[["season", "week", "team", "opponent_team", "off_epa_play", "def_epa_play", "off_plays"]]


# ======================================================================
# jerry/data/odds_api.py
# ======================================================================
"""The Odds API client, budget-aware. Every call's cost is logged to data/odds_usage.csv.

    python jerry.py usage                 # credits used / left this month + projection (free call)
    python jerry.py snapshot [SPORTS...]  # only sports with events in the next 48h are pulled
    python jerry.py close-watch           # run every 5 min by cron; pulls a sport ONCE per start-time
                                          # cluster, 2-15 min before first pitch/puck/kickoff/main card
    python jerry.py budget                # planned monthly spend for the cadence below

/sports and /events do not cost credits (per The Odds API docs) — used for free scheduling.
"""

ODDS_BASE = "https://api.the-odds-api.com/v4"
# label: (sport_key, markets, is_outright)
SPORTS = {
    "NFL": ("americanfootball_nfl", "h2h,spreads,totals", False),
    "NHL": ("icehockey_nhl", "h2h,spreads,totals", False),
    "MLB": ("baseball_mlb", "h2h,spreads,totals", False),            # World Series GAMES live here
    "WS":  ("baseball_mlb_world_series_winner", "outrights", True),  # World Series WINNER (series price)
    "UFC": ("mma_mixed_martial_arts", "h2h", False),
}
ODDS_USAGE_PATH = DATA / "odds_usage.csv"
CLOSE_LOG_PATH = DATA / "close_pulls.csv"


def _cost(sport):
    return len(SPORTS[sport][1].split(","))   # x 1 region (10 named books)


def _log_usage(sport, endpoint, headers):
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sport": sport, "endpoint": endpoint,
           "cost": headers.get("x-requests-last"), "used": headers.get("x-requests-used"),
           "remaining": headers.get("x-requests-remaining")}
    pd.DataFrame([row]).to_csv(ODDS_USAGE_PATH, mode="a", header=not ODDS_USAGE_PATH.exists(), index=False)
    return row


def _get(path, params, sport="-", endpoint="-"):
    if not ODDS_API_KEY:
        raise RuntimeError("Set ODDS_API_KEY in your environment.")
    r = requests.get(f"{ODDS_BASE}{path}", params={"apiKey": ODDS_API_KEY, **params}, timeout=30)
    r.raise_for_status()
    _log_usage(sport, endpoint, r.headers)
    return r


def usage() -> dict:
    """Current quota from response headers. Uses /sports, which costs 0 credits."""
    r = _get("/sports", {}, endpoint="sports")
    used, left = int(r.headers.get("x-requests-used", 0)), int(r.headers.get("x-requests-remaining", 0))
    today = date.today()
    nxt = date(today.year + (today.month == 12), today.month % 12 + 1, 1)
    days_left = (nxt - today).days
    out = {"used": used, "remaining": left, "quota": used + left, "resets": nxt.isoformat(), "days_to_reset": days_left}
    if ODDS_USAGE_PATH.exists():
        u = pd.read_csv(ODDS_USAGE_PATH, parse_dates=["ts"])
        u = u[u.ts >= pd.Timestamp(today.replace(day=1), tz="UTC")]
        u["cost"] = pd.to_numeric(u["cost"], errors="coerce").fillna(0)
        days = max((pd.Timestamp.now("UTC") - u.ts.min()).days, 1) if len(u) else 1
        out["avg_per_day"] = round(u.cost.sum() / days, 1)
        out["projected_month_end_remaining"] = round(left - out["avg_per_day"] * days_left)
        out["by_sport"] = u.groupby("sport").cost.sum().to_dict()
    return out


def events(sport) -> list[dict]:
    """Upcoming events, no odds — free."""
    key, _, outright = SPORTS[sport]
    if outright:
        return []
    return _get(f"/sports/{key}/events", {}, sport, "events").json()


def fetch(sport: str, odds_format="decimal") -> list[dict]:
    key, markets, _ = SPORTS[sport]
    r = _get(f"/sports/{key}/odds", {"bookmakers": ODDS_BOOKMAKERS, "markets": markets, "oddsFormat": odds_format},
             sport, "odds")
    print(f"[odds] {sport}: {len(r.json())} events | cost {r.headers.get('x-requests-last')} | "
          f"{r.headers.get('x-requests-remaining')} credits left")
    return r.json()


def flatten(sport: str, evs: list[dict], is_close: bool = False) -> pd.DataFrame:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for ev in evs:
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for o in mk.get("outcomes", []):
                    rows.append({
                        "snapshot_ts": ts, "is_close": int(is_close), "sport": sport,
                        "event_id": ev["id"], "commence_time": ev["commence_time"],
                        "home_team": ev.get("home_team", ""), "away_team": ev.get("away_team", ""),
                        "book": bk["key"], "market": mk["key"], "outcome": o["name"],
                        "line": o.get("point"), "decimal_odds": o["price"],
                    })
    return pd.DataFrame(rows)


def _append(df):
    if not df.empty:
        df.to_csv(ODDS_SNAPSHOT_PATH, mode="a", header=not ODDS_SNAPSHOT_PATH.exists(), index=False)


def _remaining() -> int | None:
    if not ODDS_USAGE_PATH.exists():
        return None
    u = pd.read_csv(ODDS_USAGE_PATH)
    v = pd.to_numeric(u["remaining"], errors="coerce").dropna()
    return int(v.iloc[-1]) if len(v) else None


def snapshot(sports=("NFL", "NHL", "MLB", "WS", "UFC"), is_close=False, within_hours=48, force=False) -> pd.DataFrame:
    """Pull odds only for sports that have an event inside the window (checked for free first)."""
    frames, now = [], pd.Timestamp.now("UTC")
    for s in sports:
        try:
            if not SPORTS[s][2] and not force:
                evs = events(s)
                soon = [e for e in evs if pd.Timestamp(e["commence_time"]) - now <= pd.Timedelta(hours=within_hours)
                        and pd.Timestamp(e["commence_time"]) > now]
                if not soon:
                    print(f"[odds] {s}: nothing in the next {within_hours}h — skipped, 0 credits"); continue
            left = _remaining()
            if left is not None and not is_close and left - _cost(s) < ODDS_RESERVE_FOR_CLOSES:
                print(f"[odds] {s}: skipped — would dip into the {ODDS_RESERVE_FOR_CLOSES}-credit close reserve"); continue
            frames.append(flatten(s, fetch(s), is_close))
        except Exception as e:
            print(f"[odds] {s} failed: {e}")
        time.sleep(1)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    _append(df)
    return df


def close_watch(sports=("NFL", "NHL", "MLB", "UFC"), lead_min=(2, 15)) -> None:
    """Cron every 5 minutes. For each sport, if an event starts in 2-15 min and that start time
    hasn't been closed yet, pull the sport once (one call covers every game in that cluster) and
    mark ALL events starting within 30 min of it as closed. UFC: run once per card at main-card
    start; prelim prices are taken from the earlier snapshot.
    Exhausted quota -> it stops; the ledger then falls back to the last snapshot for CLV."""
    done = set(pd.read_csv(CLOSE_LOG_PATH)["event_id"]) if CLOSE_LOG_PATH.exists() else set()
    now = pd.Timestamp.now("UTC")
    for s in sports:
        try:
            evs = events(s)
        except Exception as e:
            print(f"[close] {s} events failed: {e}"); continue
        due = [e for e in evs if e["id"] not in done and
               pd.Timedelta(minutes=lead_min[0]) <= pd.Timestamp(e["commence_time"]) - now <= pd.Timedelta(minutes=lead_min[1])]
        if not due:
            continue
        first = min(pd.Timestamp(e["commence_time"]) for e in due)
        cluster = [e["id"] for e in evs if e["id"] not in done and
                   abs(pd.Timestamp(e["commence_time"]) - first) <= pd.Timedelta(minutes=30)]
        if s == "UFC":   # one close per card: mark the whole card
            cluster = [e["id"] for e in evs if e["id"] not in done and
                       abs(pd.Timestamp(e["commence_time"]) - first) <= pd.Timedelta(hours=8)]
        try:
            _append(flatten(s, fetch(s), is_close=True))
            pd.DataFrame({"event_id": cluster, "sport": s, "ts": now.isoformat()}).to_csv(
                CLOSE_LOG_PATH, mode="a", header=not CLOSE_LOG_PATH.exists(), index=False)
            print(f"[close] {s}: closed {len(cluster)} events starting ~{first}")
        except Exception as e:
            print(f"[close] {s} failed: {e}")


# Planned cadence on the 20K plan. Edit to match how you bet; `python jerry.py budget` prints it.
# Line movement is a model input (UFC backtest: money moving against a fighter predicts results),
# so the 20K plan buys frequent snapshots, not just opens and closes.
BUDGET_PLAN = [
    # sport, pulls per month, note
    ("NFL", 4.3 * 6 * 8, "every 3h (upper bound; only pulls when a game is within 48h)"),
    ("NFL", 4.3 * 12, "hourly on Sunday"),
    ("NFL", 4.3 * 5, "closes: TNF, Sun 1pm, Sun 4pm, SNF, MNF"),
    ("NHL", 30 * 12, "every 2h, every day of the season"),
    ("NHL", 30 * 3, "closes: 7pm, 8pm-9pm, 10pm clusters"),
    ("MLB", 6 * 12, "World Series games: hourly on game days (up to 6)"),
    ("MLB", 6 * 1, "World Series closes"),
    ("WS", 8 * 4, "World Series winner price, every 6h"),
    ("UFC", 4 * 7 * 8, "every 3h for the week before each card (~4 cards)"),
    ("UFC", 4 * 1, "main-card close"),
]
BACKFILL_PLAN = [
    # one-time historical closing lines for backtests: 10 x markets x 1 region per daily snapshot
    ("NHL", 2 * 190, "h2h,totals", "2023-24 + 2024-25 seasons, one 7pm-ET snapshot per game day"),
    ("MLB", 186, "h2h,totals", "2025 season, one 7pm-ET snapshot per game day"),
]


def budget() -> pd.DataFrame:
    rows = [{"sport": s, "pulls": round(n, 1), "credits_each": _cost(s), "credits": round(n * _cost(s)), "what": w}
            for s, n, w in BUDGET_PLAN]
    return pd.DataFrame(rows)


def backfill_cost(n_days, markets) -> int:
    return int(n_days * 10 * len(markets.split(",")))


def backfill(sport, start, end, hour_utc=23, markets="h2h,totals", dry_run=False) -> pd.DataFrame:
    """Historical odds (paid plans; data from mid-2020) for backtesting NHL/MLB/UFC against the close.
    One snapshot per day at hour_utc (23 UTC = 7pm ET = just before most NHL/MLB starts; later games get
    a ~3h-early number, fine for backtests, flagged is_close=2). Costs 10 x markets per day.
    Resumable: days already in the file are skipped. Stops at the close reserve."""
    key = SPORTS[sport][0]
    days = pd.date_range(start, end, freq="D")
    path = DATA / f"hist_odds_{sport}.csv"
    have = set(pd.read_csv(path, usecols=["snap_day"])["snap_day"]) if path.exists() else set()
    todo = [d for d in days if d.date().isoformat() not in have]
    est = backfill_cost(len(todo), markets)
    print(f"[backfill] {sport}: {len(todo)} days to pull, ~{est} credits")
    if dry_run:
        return pd.DataFrame()
    out = []
    for d in todo:
        left = _remaining()
        if left is not None and left - 10 * len(markets.split(",")) < ODDS_RESERVE_FOR_CLOSES:
            print("[backfill] stopping at the close reserve; rerun after the 1st — it resumes"); break
        ts = (d + pd.Timedelta(hours=hour_utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            r = _get(f"/historical/sports/{key}/odds", {"bookmakers": ODDS_BOOKMAKERS, "markets": markets,
                                                        "oddsFormat": "decimal", "date": ts}, sport, "historical")
        except Exception as e:
            print(f"[backfill] {d.date()} failed: {e}"); continue
        body = r.json()
        df = flatten(sport, [e for e in body.get("data", []) if pd.Timestamp(e["commence_time"]) >= pd.Timestamp(ts)])
        df["snapshot_ts"] = body.get("timestamp", ts); df["is_close"] = 2; df["snap_day"] = d.date().isoformat()
        cols = ["snapshot_ts", "is_close", "sport", "event_id", "commence_time", "home_team", "away_team",
                "book", "market", "outcome", "line", "decimal_odds", "snap_day"]
        if df.empty:   # mark the day done even with no games, so a resume skips it
            df = pd.DataFrame([{"snapshot_ts": ts, "is_close": 2, "sport": sport, "snap_day": d.date().isoformat()}])
        df = df.reindex(columns=cols)
        df.to_csv(path, mode="a", header=not path.exists(), index=False)
        out.append(df)
        time.sleep(0.3)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def load_snapshots() -> pd.DataFrame:
    if not ODDS_SNAPSHOT_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(ODDS_SNAPSHOT_PATH, parse_dates=["snapshot_ts", "commence_time"])


def closing_line(event_id: str, market: str, outcome: str, book: str = "pinnacle") -> dict | None:
    """Last snapshot before commence_time for this book/market/outcome; falls back to any book's last."""
    s = load_snapshots()
    if s.empty:
        return None
    sel = s[(s.event_id == event_id) & (s.market == market) & (s.outcome == outcome)]
    sel = sel[sel.snapshot_ts <= sel.commence_time]
    if sel.empty:
        return None
    pref = sel[sel.book == book]
    row = (pref if not pref.empty else sel).sort_values("snapshot_ts").iloc[-1]
    return {"book": row.book, "line": row.line, "decimal_odds": row.decimal_odds, "ts": row.snapshot_ts}


# ======================================================================
# jerry/data/moneypuck.py
# ======================================================================
"""MoneyPuck public CSVs (no key): team xG rates and goalie GSAx.
Downloads the current season's team and goalie files; cached per day.

Files (verify at com/data.htm — paths occasionally change):
  https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/teams.csv
  https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/goalies.csv
Season 2026 == the 2026-27 season.
"""

MP_BASE = "https://moneypuck.com/moneypuck/playerData/seasonSummary"


def _daily(url: str, name: str) -> pd.DataFrame:
    p = CACHE / f"{name}_{date.today().isoformat()}.csv"
    if p.exists():
        return pd.read_csv(p)
    df = pd.read_csv(url)
    df.to_csv(p, index=False)
    return df


def teams(season: int, situation: str = "5on5") -> pd.DataFrame:
    df = _daily(f"{MP_BASE}/{season}/regular/teams.csv", f"mp_teams_{season}")
    df = df[df["situation"] == situation].copy()
    # per-60 rates; MoneyPuck 'iceTime' is seconds
    hrs = df["iceTime"] / 3600.0
    df["xgf_60"] = df["xGoalsFor"] / hrs
    df["xga_60"] = df["xGoalsAgainst"] / hrs
    df["gf_60"] = df["goalsFor"] / hrs
    df["ga_60"] = df["goalsAgainst"] / hrs
    df["hd_against_60"] = df["highDangerShotsAgainst"] / hrs
    return df[["team", "games_played", "xgf_60", "xga_60", "gf_60", "ga_60", "hd_against_60"]]


def goalies(season: int, situation: str = "all") -> pd.DataFrame:
    df = _daily(f"{MP_BASE}/{season}/regular/goalies.csv", f"mp_goalies_{season}")
    df = df[df["situation"] == situation].copy()
    df["gsax"] = df["xGoals"] - df["goals"]           # goals saved above expected (season)
    df["gsax_per_60"] = df["gsax"] / (df["iceTime"] / 3600.0)
    df["sv_pct"] = 1 - df["goals"] / df["unblocked_shot_attempts"].clip(lower=1)
    return df[["name", "team", "games_played", "gsax", "gsax_per_60", "sv_pct", "iceTime"]]


# ======================================================================
# jerry/data/mlb_stats.py
# ======================================================================
"""MLB Stats API (statsapi.mlb.com) — public, no key. Team run scoring/allowed and
probable starters. Enough for a runs model in a short series; park factors and
bullpen state are the obvious next adds."""

MLB_BASE = "https://statsapi.mlb.com/api/v1"


def team_runs(season: int) -> pd.DataFrame:
    r = requests.get(f"{MLB_BASE}/teams/stats", params={"season": season, "group": "hitting", "stats": "season", "sportId": 1}, timeout=30)
    r.raise_for_status()
    hit = pd.DataFrame([{"team_id": s["team"]["id"], "team": s["team"]["name"], "g": s["stat"]["gamesPlayed"], "runs": s["stat"]["runs"]}
                        for s in r.json()["stats"][0]["splits"]])
    r = requests.get(f"{MLB_BASE}/teams/stats", params={"season": season, "group": "pitching", "stats": "season", "sportId": 1}, timeout=30)
    r.raise_for_status()
    pit = pd.DataFrame([{"team_id": s["team"]["id"], "runs_allowed": s["stat"]["runs"], "era": float(s["stat"]["era"])}
                        for s in r.json()["stats"][0]["splits"]])
    df = hit.merge(pit, on="team_id")
    df["rs_g"] = df["runs"] / df["g"]
    df["ra_g"] = df["runs_allowed"] / df["g"]
    return df


def probable_pitchers(date_iso: str) -> pd.DataFrame:
    r = requests.get(f"{MLB_BASE}/schedule", params={"sportId": 1, "date": date_iso, "hydrate": "probablePitcher(note),team"}, timeout=30)
    r.raise_for_status()
    rows = []
    for d in r.json().get("dates", []):
        for g in d["games"]:
            for side in ("home", "away"):
                t = g["teams"][side]
                pp = t.get("probablePitcher", {})
                rows.append({"game_pk": g["gamePk"], "side": side, "team": t["team"]["name"],
                             "pitcher": pp.get("fullName"), "pitcher_id": pp.get("id")})
    return pd.DataFrame(rows)


def pitcher_season(pitcher_id: int, season: int) -> dict:
    r = requests.get(f"{MLB_BASE}/people/{pitcher_id}/stats", params={"stats": "season", "group": "pitching", "season": season}, timeout=30)
    r.raise_for_status()
    sp = r.json()["stats"][0]["splits"]
    if not sp:
        return {}
    st = sp[0]["stat"]
    ip = float(st["inningsPitched"].replace(".1", ".333").replace(".2", ".667"))
    # FIP without the constant; league constant ~3.1 added in model
    fip_core = (13 * st["homeRuns"] + 3 * (st["baseOnBalls"] + st["hitByPitch"]) - 2 * st["strikeOuts"]) / max(ip, 1)
    return {"ip": ip, "era": float(st["era"]), "fip": fip_core + 3.1, "starts": st["gamesStarted"]}


# ======================================================================
# jerry/models/nfl.py
# ======================================================================
"""NFL side + total model. Deliberately boring: ridge regression on pre-game,
season-to-date EPA/play (offense and defense) for both teams, with a prior
shrink early in the season, plus home field. Probabilities come from an
empirical residual distribution so key numbers are respected.

Everything here is testable against nflverse closing lines going back years.
If it does not beat the close in backtest, it does not get live weight.
"""

PRIOR_WEIGHT_GAMES = 4  # early-season shrink toward league average
# Market-anchored: the closing spread/total are features. The model learns how much
# to trust public EPA *beyond* what the market already priced. Disagreement with the
# market is then a genuine residual signal, and probabilities come out calibrated.
FEATURES = ["h_off", "h_def", "a_off", "a_def", "h_rest_edge", "spread_line", "total_line"]


def _season_to_date(stats: pd.DataFrame) -> pd.DataFrame:
    """For each team-week, offense/defense EPA/play using only games BEFORE that week."""
    stats = stats.sort_values(["season", "team", "week"]).copy()
    for col in ["off_epa_play", "def_epa_play"]:
        grp = stats.groupby(["season", "team"])[col]
        csum = grp.cumsum() - stats[col]
        cnt = grp.cumcount()
        # shrink toward 0 (league average EPA/play is ~0)
        stats[f"{col}_std"] = csum / (cnt + PRIOR_WEIGHT_GAMES)
    return stats[["season", "week", "team", "off_epa_play_std", "def_epa_play_std"]]


def build_frame(games: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    g = games[(games["game_type"] == "REG") & games["spread_line"].notna()].copy()
    std = _season_to_date(stats)
    h = std.rename(columns={"team": "home_team", "off_epa_play_std": "h_off", "def_epa_play_std": "h_def"})
    a = std.rename(columns={"team": "away_team", "off_epa_play_std": "a_off", "def_epa_play_std": "a_def"})
    g = g.merge(h, on=["season", "week", "home_team"], how="left").merge(a, on=["season", "week", "away_team"], how="left")
    g["h_rest_edge"] = (g["home_rest"] - g["away_rest"]).clip(-7, 7).fillna(0)
    g["total_actual"] = g["home_score"] + g["away_score"]
    return g.dropna(subset=FEATURES)


class NFLModel:
    def __init__(self, alpha: float = 1.0):
        self.margin_reg = Ridge(alpha=alpha)
        self.total_reg = Ridge(alpha=alpha)
        self.margin_dist: EmpiricalMargin | None = None
        self.total_sigma: float = 13.0

    def fit(self, train: pd.DataFrame):
        X = train[FEATURES].values
        self.margin_reg.fit(X, train["result"].values)
        self.total_reg.fit(X, train["total_actual"].values)
        res = train["result"].values - self.margin_reg.predict(X)
        self.margin_dist = EmpiricalMargin(res)
        self.total_sigma = float((train["total_actual"].values - self.total_reg.predict(X)).std())
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        X = rows[FEATURES].values
        out = rows[["game_id", "season", "week", "home_team", "away_team", "spread_line", "total_line"]].copy()
        out["pred_margin"] = self.margin_reg.predict(X)
        out["pred_total"] = self.total_reg.predict(X)
        cov, push = zip(*[self.margin_dist.p_home_cover(m, s) for m, s in zip(out["pred_margin"], out["spread_line"])])
        out["p_home_cover"] = cov
        out["p_push"] = push
        out["p_home_ml"] = [self.margin_dist.p_home_win(m) for m in out["pred_margin"]]
        from scipy.stats import norm
        out["p_under"] = norm.cdf(out["total_line"], loc=out["pred_total"], scale=self.total_sigma)
        km = pd.DataFrame([self.margin_dist.key_mass(m) for m in out["pred_margin"]], index=out.index)
        out = pd.concat([out, km], axis=1)
        q = [self.margin_dist.iqr(m) for m in out["pred_margin"]]
        out["margin_q25"] = [x[0] for x in q]
        out["margin_q75"] = [x[1] for x in q]
        # projected final that matches the side (your rule, kept): split pred_total around pred_margin
        out["proj_home"] = ((out["pred_total"] + out["pred_margin"]) / 2).round(0)
        out["proj_away"] = ((out["pred_total"] - out["pred_margin"]) / 2).round(0)
        return out


# ======================================================================
# jerry/models/nhl.py
# ======================================================================
"""NHL goals model: independent Poisson with a Dixon-Coles low-score correction,
driven by 5v5 xG rates and a confirmed-goalie GSAx term. Outputs the whole
distribution, so puck line, one-goal share, regulation-win share and under
probability are all read off the same object — no separate rules.

Confirmed starter is an INPUT. If unconfirmed, pass goalie_gsax=None and the
model uses the team average; the card must then say 'goalie unconfirmed'.
"""

LEAGUE_XG_60 = 2.6       # 5v5 xG/60 league-average ballpark; recomputed from data in fit_league()
HOME_EDGE = 1.05         # multiplicative on home expected goals (~55% home win historically ~ this)
PP_PK_SHARE = 0.30       # share of all-situation goals NOT scored at 5v5 (league avg)
FIVE_V_FIVE_MIN = 48     # minutes per game at 5v5, league avg
OT_SHARE_OF_TIES = 1.0   # all regulation ties go to OT/SO
REG_MINUTES = 60
RHO = -0.10              # Dixon-Coles correction for 0-0/1-0/0-1/1-1


def fit_league(team_df) -> float:
    return float(team_df["xgf_60"].mean())


def expected_goals(home, away, league_xg60, home_goalie_gsax60=0.0, away_goalie_gsax60=0.0,
                   home_b2b=False, away_b2b=False) -> tuple[float, float]:
    """home/away are rows from teams(). Returns (lambda_home, lambda_away) for 60 min."""
    # Attack strength vs opponent defensive weakness, relative to league
    lam_h = league_xg60 * (home["xgf_60"] / league_xg60) * (away["xga_60"] / league_xg60)
    lam_a = league_xg60 * (away["xgf_60"] / league_xg60) * (home["xga_60"] / league_xg60)
    # 5v5 per-60 rate -> 5v5 goals per game -> all situations per game
    scale = (FIVE_V_FIVE_MIN / 60) / (1 - PP_PK_SHARE)
    lam_h *= scale; lam_a *= scale
    # Goalie: GSAx per 60 (all situations) above average lowers opponent goals
    lam_h -= away_goalie_gsax60 or 0.0
    lam_a -= home_goalie_gsax60 or 0.0
    lam_h *= HOME_EDGE
    if home_b2b: lam_h *= 0.96; lam_a *= 1.03
    if away_b2b: lam_a *= 0.96; lam_h *= 1.03
    return max(lam_h, 0.8), max(lam_a, 0.8)


def score_matrix(lam_h, lam_a, max_goals=10) -> np.ndarray:
    h = poisson.pmf(np.arange(max_goals + 1), lam_h)
    a = poisson.pmf(np.arange(max_goals + 1), lam_a)
    M = np.outer(h, a)
    # Dixon-Coles tau adjustment on low scores
    tau = {(0, 0): 1 - lam_h * lam_a * RHO, (0, 1): 1 + lam_h * RHO, (1, 0): 1 + lam_a * RHO, (1, 1): 1 - RHO}
    for (i, j), t in tau.items():
        M[i, j] *= t
    return M / M.sum()


def nhl_summarize(lam_h, lam_a, total_line: float, pl_line: float = 1.5) -> dict:
    M = score_matrix(lam_h, lam_a)
    n = M.shape[0]
    idx = np.arange(n)
    diff = idx[:, None] - idx[None, :]
    tot = idx[:, None] + idx[None, :]
    p_home_reg = M[diff > 0].sum(); p_away_reg = M[diff < 0].sum(); p_tie = M[diff == 0].sum()
    # OT/SO is close to a coin flip, slightly favouring the better team
    edge = (lam_h - lam_a) * 0.05
    p_home_ot = p_tie * min(max(0.5 + edge, 0.4), 0.6)
    out = {
        "lambda_home": lam_h, "lambda_away": lam_a, "exp_total": lam_h + lam_a,
        "p_home_ml": p_home_reg + p_home_ot, "p_away_ml": p_away_reg + (p_tie - p_home_ot),
        "p_regulation_decided": 1 - p_tie, "p_ot_so": p_tie,
        "one_goal_share": M[np.abs(diff) == 1].sum() + p_tie,   # games decided by exactly one goal incl. OT/SO
        "p_home_cover_pl": M[diff >= pl_line + 0.5].sum(),       # home -1.5
        "p_away_cover_pl": 1 - M[diff >= pl_line + 0.5].sum() + 0.0,  # away +1.5 == home wins by <2
        "p_under": M[tot < total_line].sum() + (0.5 * M[tot == total_line].sum() if float(total_line).is_integer() else 0),
        "p_3plus_home_loss": M[diff <= -3].sum(),
        "p_3plus_away_loss": M[diff >= 3].sum(),
        "p_home_by_2plus": M[diff >= 2].sum(),
        "p_away_by_2plus": M[diff <= -2].sum(),
    }
    # projected final that matches the side
    fav_home = out["p_home_ml"] >= 0.5
    ph, pa = round(lam_h), round(lam_a)
    if fav_home and ph <= pa: ph = pa + 1
    if not fav_home and pa <= ph: pa = ph + 1
    out["proj_home"], out["proj_away"] = int(ph), int(pa)
    return out


# ======================================================================
# jerry/models/mlb.py
# ======================================================================
"""MLB runs model for a short series (World Series). Team runs scored/allowed,
adjusted for the named starting pitcher's FIP vs league, with a bullpen-neutral
assumption. Runs are over-dispersed, so use a negative binomial, not Poisson.
Honest caveat: single-game MLB is close to a coin flip; the model's job is
calibrated probabilities to compare against the price, not conviction."""

LEAGUE_RUNS_G = 4.4
LEAGUE_FIP = 4.1
STARTER_SHARE = 0.60   # share of a game's runs attributable to the starter's innings
NB_DISPERSION = 7.0    # size parameter; fit from data later


def expected_runs(team_rs_g, opp_ra_g, opp_starter_fip=None, home=False, park_factor=1.0) -> float:
    lam = team_rs_g * (opp_ra_g / LEAGUE_RUNS_G)
    if opp_starter_fip is not None:
        adj = (opp_starter_fip / LEAGUE_FIP)
        lam = lam * (STARTER_SHARE * adj + (1 - STARTER_SHARE))
    lam *= park_factor
    lam *= 1.04 if home else 1.0
    return max(lam, 1.5)


def _pmf(lam, n=25):
    p = NB_DISPERSION / (NB_DISPERSION + lam)
    return nbinom.pmf(np.arange(n), NB_DISPERSION, p)


def mlb_summarize(lam_h, lam_a, total_line: float, run_line: float = 1.5) -> dict:
    M = np.outer(_pmf(lam_h), _pmf(lam_a)); M /= M.sum()
    idx = np.arange(M.shape[0]); diff = idx[:, None] - idx[None, :]; tot = idx[:, None] + idx[None, :]
    p_tie = M[diff == 0].sum()  # extra innings — split by strength
    edge = (lam_h - lam_a) * 0.05
    p_home = M[diff > 0].sum() + p_tie * (0.5 + edge)
    return {
        "lambda_home": lam_h, "lambda_away": lam_a, "exp_total": lam_h + lam_a,
        "p_home_ml": p_home, "p_away_ml": 1 - p_home,
        "p_home_cover_rl": M[diff >= run_line + 0.5].sum(),
        "p_under": M[tot < total_line].sum() + (0.5 * M[tot == total_line].sum() if float(total_line).is_integer() else 0),
        "one_run_share": M[np.abs(diff) == 1].sum() + p_tie,
        "proj_home": int(round(lam_h)), "proj_away": int(round(lam_a)),
    }


# ======================================================================
# jerry/eval/backtest_nfl.py
# ======================================================================
"""Walk-forward backtest of the NFL model against nflverse closing lines.

For each season S in the test range: train on all seasons < S (plus weeks < W of S),
predict week W. No leakage. Reports:
  - hit rate when model disagrees with the close by >= threshold points
  - Brier score / log loss of cover probabilities vs. 50% baseline
  - calibration table
The honest bar: ~52.4% on spreads at -110 to break even. Expect ~50-53% from a
public-EPA model; the edge, if any, is in selective betting and line timing, not volume.
"""


def run(train_from=2015, test_seasons=(2022, 2023, 2024, 2025), edge_pts=(0.0, 1.5, 2.5, 3.5)) -> dict:
    g = games()
    seasons = list(range(train_from, max(test_seasons) + 1))
    stats = team_week_stats(seasons)
    frame = build_frame(g, stats)
    frame = frame[frame["result"].notna()]
    preds = []
    for s in test_seasons:
        for w in sorted(frame.loc[frame.season == s, "week"].unique()):
            train = frame[(frame.season < s) | ((frame.season == s) & (frame.week < w))]
            test = frame[(frame.season == s) & (frame.week == w)]
            if len(train) < 200 or test.empty:
                continue
            m = NFLModel().fit(train)
            p = m.predict(test)
            p["result"] = test["result"].values
            p["total_actual"] = test["total_actual"].values
            preds.append(p)
    P = pd.concat(preds, ignore_index=True)
    P["home_covered"] = (P["result"] > P["spread_line"]).astype(int)
    P["pushed"] = P["result"] == P["spread_line"]
    P["model_edge_pts"] = P["pred_margin"] - P["spread_line"]  # >0 => model likes home vs close
    P["went_under"] = (P["total_actual"] < P["total_line"]).astype(int)
    P["total_edge_pts"] = P["total_line"] - P["pred_total"]      # >0 => model likes under

    out = {"n_games": len(P)}
    np_ = P[~P["pushed"]]
    out["spread_brier"] = brier(np_["p_home_cover"], np_["home_covered"])
    out["spread_brier_baseline_50"] = brier(np.full(len(np_), 0.5), np_["home_covered"])
    out["spread_logloss"] = log_loss(np_["p_home_cover"], np_["home_covered"])
    out["total_brier"] = brier(P["p_under"], P["went_under"])
    out["calibration_spread"] = calibration_table(np_["p_home_cover"], np_["home_covered"])

    rows = []
    for e in edge_pts:
        sel = np_[np_["model_edge_pts"].abs() >= e]
        pick_home = sel["model_edge_pts"] > 0
        hits = (pick_home & (sel["home_covered"] == 1)) | (~pick_home & (sel["home_covered"] == 0))
        rows.append({"market": "spread", "min_edge_pts": e, "n": len(sel), "hit_rate": float(hits.mean()) if len(sel) else np.nan})
        selt = P[P["total_edge_pts"].abs() >= e]
        pick_under = selt["total_edge_pts"] > 0
        hits_t = (pick_under & (selt["went_under"] == 1)) | (~pick_under & (selt["went_under"] == 0))
        rows.append({"market": "total", "min_edge_pts": e, "n": len(selt), "hit_rate": float(hits_t.mean()) if len(selt) else np.nan})
    out["selectivity"] = pd.DataFrame(rows)
    out["predictions"] = P
    return out




# ======================================================================
# jerry/eval/ledger.py
# ======================================================================
"""Bet ledger: one row per BET (not per game). This replaces jerry-accuracy-tracker.csv.

Grading commands compute CLV, Brier, calibration, ROI with confidence intervals,
and — most importantly — whether Jerry's overrides of the model helped or hurt.
"""

COLUMNS = [
    "bet_id", "placed_ts", "game_date", "sport", "event_id", "game", "market",  # market: spread|total|ml|puck_line|run_line|prop_<type>
    "selection", "line_at_bet", "decimal_odds_at_bet", "book", "stake_units",
    "model_prob", "jerry_prob", "override_reason",
    "closing_line", "closing_decimal_odds", "closing_book", "clv_cents",
    "result",  # W | L | P | pending
    "pnl_units", "notes",
]


STR_COLS = {"bet_id", "placed_ts", "game_date", "sport", "event_id", "game", "market", "selection", "book",
            "override_reason", "closing_book", "result", "notes"}


def load() -> pd.DataFrame:
    if LEDGER_PATH.exists():
        df = pd.read_csv(LEDGER_PATH, dtype={c: "string" for c in STR_COLS})
    else:
        df = pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c in STR_COLS:
            df[c] = df[c].astype("string")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


def save(df: pd.DataFrame):
    df[COLUMNS].to_csv(LEDGER_PATH, index=False)


def add_bet(**row) -> str:
    df = load()
    row.setdefault("bet_id", f"B{len(df)+1:05d}")
    row.setdefault("result", "pending")
    row.setdefault("stake_units", 1.0)
    for c in COLUMNS:
        row.setdefault(c, np.nan)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save(df)
    return row["bet_id"]


def grade(bet_id: str, result: str, closing_line=None, closing_decimal_odds=None, closing_book=None):
    df = load()
    i = df.index[df.bet_id == bet_id]
    if len(i) != 1:
        raise ValueError(f"bet {bet_id} not found")
    i = i[0]
    df.loc[i, "result"] = result
    odds = float(df.loc[i, "decimal_odds_at_bet"]); st = float(df.loc[i, "stake_units"])
    df.loc[i, "pnl_units"] = {"W": st * (odds - 1), "L": -st, "P": 0.0}[result]
    if closing_decimal_odds is not None:
        df.loc[i, "closing_line"] = closing_line
        df.loc[i, "closing_decimal_odds"] = closing_decimal_odds
        df.loc[i, "closing_book"] = closing_book
        df.loc[i, "clv_cents"] = clv_cents(odds, float(closing_decimal_odds))
    save(df)


def report() -> str:
    df = load()
    g = df[df.result.isin(["W", "L", "P"])].copy()
    if g.empty:
        return "No graded bets yet."
    lines = [f"Graded bets: {len(g)}  (recalibration threshold per market: {RECALIBRATION_MIN_BETS})"]
    g["y"] = (g.result == "W").astype(float)
    np_ = g[g.result != "P"]
    roi, lo, hi = roi_ci(np_.pnl_units.values, np_.stake_units.values)
    lines.append(f"ROI {roi:+.1%}  90% CI [{lo:+.1%}, {hi:+.1%}]  <- if the CI spans zero, you have learned nothing yet")
    if np_.clv_cents.notna().any():
        c = np_.clv_cents.dropna()
        lines.append(f"CLV: mean {c.mean():+.1f} cents over {len(c)} bets, {(c>0).mean():.0%} beat the close")
    for mkt, sub in np_.groupby("market"):
        r, l, h = roi_ci(sub.pnl_units.values, sub.stake_units.values)
        clv = sub.clv_cents.dropna()
        lines.append(f"  {mkt:<14} n={len(sub):<4} ROI {r:+.1%} [{l:+.1%},{h:+.1%}]  CLV {clv.mean():+.1f}c" if len(clv) else
                     f"  {mkt:<14} n={len(sub):<4} ROI {r:+.1%} [{l:+.1%},{h:+.1%}]")
    # Does Jerry (LLM) add value over the model?
    both = np_.dropna(subset=["model_prob", "jerry_prob"])
    if len(both) >= 10:
        bm, bj = brier(both.model_prob, both.y), brier(both.jerry_prob, both.y)
        mkt_b = brier(both.decimal_odds_at_bet.map(implied_prob), both.y)
        lines.append(f"Brier — model {bm:.4f} | Jerry {bj:.4f} | market price {mkt_b:.4f}  (lower is better)")
        lines.append("  Jerry's overrides are " + ("HELPING" if bj < bm else "HURTING") + " vs. the raw model.")
        lines.append(calibration_table(both.jerry_prob, both.y).to_string(index=False))
    return "\n".join(lines)


# ======================================================================
# jerry/report/card.py
# ======================================================================
"""Builds the JSON that Claude ('Jerry') receives. Jerry does not compute probabilities;
it reads these, checks them against news it can verify, writes the card, and may
shade by at most LLM_OVERRIDE_CAP with a logged reason."""


def edge_vs_price(model_prob: float, decimal_odds: float, opp_decimal_odds: float | None = None) -> dict:
    fair = devig_two_way(decimal_odds, opp_decimal_odds)[0] if opp_decimal_odds else 1 / decimal_odds
    ev = model_prob * (decimal_odds - 1) - (1 - model_prob)
    return {"implied_fair_prob": round(fair, 4), "model_minus_market": round(model_prob - fair, 4), "ev_per_unit": round(ev, 4)}


def build(sport: str, game: str, lines: dict, model_out: dict, context: dict) -> dict:
    """lines: posted numbers + decimal odds from the latest snapshot.
    model_out: output of the sport model (NFL predict row / nhl.summarize / mlb.summarize).
    context: things only a reader can supply (goalie confirmed?, injuries, weather, b2b)."""
    card = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sport": sport, "game": game,
        "posted_lines": lines,
        "model": {k: (round(float(v), 4) if isinstance(v, (int, float)) else v) for k, v in model_out.items()},
        "context_for_jerry": context,
        "rules": {
            "override_cap_abs": LLM_OVERRIDE_CAP,
            "jerry_may": ["verify roster/goalie/weather against named public sources",
                          "shade model_prob within cap with a one-line reason",
                          "veto a bet (log reason)",
                          "write the card and projected final (must match the side the probability favours)"],
            "jerry_may_not": ["invent probabilities", "change the posted line", "add rules to the model"],
        },
    }
    for mkt in ("spread", "total", "ml", "puck_line", "run_line"):
        if mkt in lines and "decimal_odds" in lines[mkt]:
            p_key = {"spread": "p_home_cover", "total": "p_under", "ml": "p_home_ml",
                     "puck_line": "p_home_cover_pl", "run_line": "p_home_cover_rl"}[mkt]
            if p_key in model_out:
                card.setdefault("edges", {})[mkt] = edge_vs_price(float(model_out[p_key]), lines[mkt]["decimal_odds"], lines[mkt].get("opp_decimal_odds"))
    return card


def dumps(card: dict) -> str:
    return json.dumps(card, indent=2, default=float)


# ======================================================================
# UFC module — data: UFCStats (via Greco1899/scrape_ufc_stats, updated daily on GitHub)
#              odds:  shortlikeafox/ultimate_ufc_dataset (moneyline + method odds 2010→)
# ======================================================================

UFC_STATS_BASE = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main"
UFC_ODDS_URL = "https://raw.githubusercontent.com/shortlikeafox/ultimate_ufc_dataset/master/ufc-master.csv"
UFC_CACHE = CACHE


def _ufc_csv(url, name, refresh=False):
    import requests
    p = UFC_CACHE / name
    if refresh or not p.exists():
        r = requests.get(url, timeout=120); r.raise_for_status(); p.write_bytes(r.content)
    return pd.read_csv(p, low_memory=False)


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower().replace("-", " "))
    return re.sub(r"\s+", " ", s).strip()


def _of(x):
    """'12 of 30' -> (12, 30)"""
    try:
        a, b = str(x).split(" of "); return int(a), int(b)
    except Exception:
        return 0, 0


def _secs(x):
    try:
        m, s = str(x).split(":"); return int(m) * 60 + int(s)
    except Exception:
        return 0


def ufc_load(refresh=False) -> pd.DataFrame:
    """One row per completed fight with per-fighter totals, chronological."""
    ev = _ufc_csv(f"{UFC_STATS_BASE}/ufc_event_details.csv", "ufc_events.csv", refresh)
    res = _ufc_csv(f"{UFC_STATS_BASE}/ufc_fight_results.csv", "ufc_results.csv", refresh)
    st = _ufc_csv(f"{UFC_STATS_BASE}/ufc_fight_stats.csv", "ufc_stats.csv", refresh)
    tott = _ufc_csv(f"{UFC_STATS_BASE}/ufc_fighter_tott.csv", "ufc_tott.csv", refresh)

    ev["date"] = pd.to_datetime(ev["DATE"], errors="coerce")
    for d in (ev, res, st):
        d["EVENT"] = d["EVENT"].str.strip()
    res["BOUT"] = res["BOUT"].str.replace(r"\s+", " ", regex=True).str.strip()
    st["BOUT"] = st["BOUT"].str.replace(r"\s+", " ", regex=True).str.strip()
    st["FIGHTER"] = st["FIGHTER"].str.strip()

    res = res.merge(ev[["EVENT", "date"]], on="EVENT", how="left")
    ab = res["BOUT"].str.split(" vs. ", n=1, expand=True)
    res["a"], res["b"] = ab[0].str.strip(), ab[1].str.strip()
    res = res[res["OUTCOME"].isin(["W/L", "L/W"])].copy()
    res["a_won"] = (res["OUTCOME"] == "W/L").astype(int)
    m = res["METHOD"].fillna("")
    res["method"] = np.select([m.str.startswith("Decision"), m.str.contains("Submission"),
                               m.str.contains("KO|Could Not Continue|Doctor")], ["DEC", "SUB", "KO"], "OTHER")
    res = res[res["method"] != "OTHER"]
    res["minutes"] = (res["ROUND"].fillna(1) - 1) * 5 + res["TIME"].map(_secs) / 60
    res["five_rd"] = res["TIME FORMAT"].fillna("").str.startswith("5 Rnd").astype(int)
    res["title"] = res["WEIGHTCLASS"].fillna("").str.contains("Title").astype(int)
    res["women"] = res["WEIGHTCLASS"].fillna("").str.contains("Women").astype(int)

    # per-fighter fight totals
    for c in ["SIG.STR.", "TD", "HEAD"]:
        st[[c + "_l", c + "_a"]] = st[c].apply(lambda x: pd.Series(_of(x)))
    st["ctrl_s"] = st["CTRL"].map(_secs)
    agg = st.groupby(["EVENT", "BOUT", "FIGHTER"], as_index=False).agg(
        sig_l=("SIG.STR._l", "sum"), sig_a=("SIG.STR._a", "sum"), td_l=("TD_l", "sum"), td_a=("TD_a", "sum"),
        kd=("KD", "sum"), sub=("SUB.ATT", "sum"), ctrl=("ctrl_s", "sum"))
    cols = ["sig_l", "sig_a", "td_l", "td_a", "kd", "sub", "ctrl"]
    a = agg.rename(columns={"FIGHTER": "a", **{c: "a_" + c for c in cols}})
    b = agg.rename(columns={"FIGHTER": "b", **{c: "b_" + c for c in cols}})
    res = res.merge(a, on=["EVENT", "BOUT", "a"], how="left").merge(b, on=["EVENT", "BOUT", "b"], how="left")

    # physicals (by name; duplicates resolved by first)
    tott = tott.drop_duplicates("FIGHTER")
    def inches(h):
        mm = re.match(r"(\d+)' (\d+)", str(h)); return int(mm[1]) * 12 + int(mm[2]) if mm else np.nan
    phys = pd.DataFrame({"name": tott["FIGHTER"].str.strip(), "height": tott["HEIGHT"].map(inches),
                         "reach": pd.to_numeric(tott["REACH"].astype(str).str.replace('"', ""), errors="coerce"),
                         "dob": pd.to_datetime(tott["DOB"], errors="coerce"),
                         "southpaw": (tott["STANCE"] == "Southpaw").astype(int)}).set_index("name")
    res = res.dropna(subset=["date"]).sort_values(["date", "URL"]).reset_index(drop=True)
    return res, phys


# ---------- pre-fight state ----------
PRIOR = dict(slpm=3.5, sapm=3.5, acc=0.45, sdef=0.55, td15=1.3, tdacc=0.38, tddef=0.62, sub15=0.45, kd15=0.25, ctrl=0.25)
K_MIN, K_ATT = 20.0, 40.0
ELO_K, ELO_FINISH = 32.0, 1.25
UFC_LEAN_EDGE, UFC_PLAY_EDGE = 0.05, 0.08   # locked until 150 graded UFC bets


class FighterState:
    __slots__ = ("n", "mins", "sl", "sa", "sabs", "satt_opp", "tdl", "tda", "tdopp_l", "tdopp_a", "kd", "kdabs",
                 "sub", "ctrl", "octrl", "wins", "ko_losses", "fin_wins", "last", "elo", "form")

    def __init__(self):
        self.n = 0; self.mins = 0.0; self.sl = self.sa = self.sabs = self.satt_opp = 0.0
        self.tdl = self.tda = self.tdopp_l = self.tdopp_a = 0.0; self.kd = self.kdabs = self.sub = 0.0
        self.ctrl = self.octrl = 0.0; self.wins = self.ko_losses = self.fin_wins = 0; self.last = None; self.elo = 1500.0
        self.form = []

    def feats(self, date, phys_row):
        m = self.mins
        f = {
            "elo": self.elo,
            "slpm": (self.sl + PRIOR["slpm"] * K_MIN) / (m + K_MIN),
            "sapm": (self.sabs + PRIOR["sapm"] * K_MIN) / (m + K_MIN),
            "acc": (self.sl + PRIOR["acc"] * K_ATT) / (self.sa + K_ATT),
            "sdef": 1 - (self.sabs + (1 - PRIOR["sdef"]) * K_ATT) / (self.satt_opp + K_ATT),
            "td15": (self.tdl + PRIOR["td15"] / 15 * K_MIN) / (m + K_MIN) * 15,
            "tdacc": (self.tdl + PRIOR["tdacc"] * 10) / (self.tda + 10),
            "tddef": 1 - (self.tdopp_l + (1 - PRIOR["tddef"]) * 10) / (self.tdopp_a + 10),
            "sub15": (self.sub + PRIOR["sub15"] / 15 * K_MIN) / (m + K_MIN) * 15,
            "kd15": (self.kd + PRIOR["kd15"] / 15 * K_MIN) / (m + K_MIN) * 15,
            "kdabs15": (self.kdabs + PRIOR["kd15"] / 15 * K_MIN) / (m + K_MIN) * 15,
            "ctrl": (self.ctrl + PRIOR["ctrl"] * K_MIN * 60) / (m * 60 + K_MIN * 60),   # share of cage time in control
            "ko_loss_rate": (self.ko_losses + 0.1 * 3) / (self.n + 3),
            "fin_rate": (self.fin_wins + 0.5 * 3) / (self.wins + 3),
            "log_n": np.log1p(self.n),
            "form3": np.mean(self.form[-3:]) if self.form else 0.5,
            "layoff": min((date - self.last).days, 730) / 365 if self.last is not None else 1.0,
        }
        if phys_row is not None:
            dob = phys_row["dob"]
            f["age"] = (date - dob).days / 365.25 if pd.notna(dob) else np.nan
            f["reach"] = phys_row["reach"]; f["height"] = phys_row["height"]
        else:
            f["age"] = f["reach"] = f["height"] = np.nan
        return f


FEATS = ["elo", "slpm", "sapm", "acc", "sdef", "td15", "tdacc", "tddef", "sub15", "kd15", "kdabs15", "ctrl",
         "ko_loss_rate", "fin_rate", "log_n", "form3", "layoff", "age", "reach", "height"]


def _update(sa, sb, row):
    a_won = row["a_won"]
    for me, op, pre_me, pre_op, won in ((sa, sb, "a_", "b_", a_won), (sb, sa, "b_", "a_", 1 - a_won)):
        g = lambda k, p=pre_me: float(row.get(p + k) or 0) if pd.notna(row.get(p + k)) else 0.0
        me.n += 1; me.mins += row["minutes"]
        me.sl += g("sig_l"); me.sa += g("sig_a"); me.sabs += g("sig_l", pre_op); me.satt_opp += g("sig_a", pre_op)
        me.tdl += g("td_l"); me.tda += g("td_a"); me.tdopp_l += g("td_l", pre_op); me.tdopp_a += g("td_a", pre_op)
        me.kd += g("kd"); me.kdabs += g("kd", pre_op); me.sub += g("sub"); me.ctrl += g("ctrl"); me.octrl += g("ctrl", pre_op)
        me.wins += won; me.fin_wins += int(won and row["method"] != "DEC")
        me.ko_losses += int((not won) and row["method"] == "KO"); me.last = row["date"]; me.form.append(won)
    # Elo
    ea = 1 / (1 + 10 ** ((sb.elo - sa.elo) / 400))
    k = ELO_K * (ELO_FINISH if row["method"] != "DEC" else 1.0)
    sa.elo += k * (a_won - ea); sb.elo -= k * (a_won - ea)


def ufc_build_features(res, phys):
    """Walk chronologically; record each fight's PRE-fight features, then update. Returns (frame, states)."""
    states, rows = {}, []
    ph = phys.to_dict("index")
    for _, r in res.iterrows():
        sa = states.setdefault(r["a"], FighterState()); sb = states.setdefault(r["b"], FighterState())
        fa = sa.feats(r["date"], ph.get(r["a"])); fb = sb.feats(r["date"], ph.get(r["b"]))
        row = {"date": r["date"], "a": r["a"], "b": r["b"], "a_won": r["a_won"], "method": r["method"],
               "five_rd": r["five_rd"], "title": r["title"], "women": r["women"], "a_n": sa.n, "b_n": sb.n}
        for k in FEATS:
            row["d_" + k] = fa[k] - fb[k]
            row["s_" + k] = fa[k] + fb[k]
        rows.append(row)
        _update(sa, sb, r)
    return pd.DataFrame(rows), states


def ufc_attach_odds(df, refresh=False):
    o = _ufc_csv(UFC_ODDS_URL, "ufc_odds.csv", refresh)
    o = o[["R_fighter", "B_fighter", "R_odds", "B_odds", "date"]].copy()
    o["date"] = pd.to_datetime(o["date"])
    def dec(am): return np.where(am > 0, 1 + am / 100, 1 + 100 / np.abs(am))
    o["r_dec"], o["b_dec"] = dec(o["R_odds"]), dec(o["B_odds"])
    o["key"] = [frozenset((norm_name(x), norm_name(y))) for x, y in zip(o["R_fighter"], o["B_fighter"])]
    o["rn"] = o["R_fighter"].map(norm_name)
    look = {}
    for _, r in o.dropna(subset=["R_odds", "B_odds"]).iterrows():
        look[(r["key"], r["date"])] = r
    out = []
    for _, r in df.iterrows():
        key = frozenset((norm_name(r["a"]), norm_name(r["b"])))
        hit = None
        for dd in (0, -1, 1):
            hit = look.get((key, r["date"] + pd.Timedelta(days=dd)))
            if hit is not None: break
        if hit is None:
            out.append((np.nan, np.nan)); continue
        a_is_red = hit["rn"] == norm_name(r["a"])
        out.append((hit["r_dec"], hit["b_dec"]) if a_is_red else (hit["b_dec"], hit["r_dec"]))
    df = df.copy()
    df["a_dec"], df["b_dec"] = zip(*out)
    ia, ib = 1 / df["a_dec"], 1 / df["b_dec"]
    df["p_mkt_a"] = ia / (ia + ib)
    return df


def _sym(X, y=None):
    """Symmetrize: each fight both ways, diffs negated, sums unchanged."""
    Xf = X.copy()
    for c in Xf.columns:
        if c.startswith("d_") or c == "mkt_logit": Xf[c] = -Xf[c]
    Xs = pd.concat([X, Xf], ignore_index=True)
    return (Xs, np.concatenate([y, 1 - y])) if y is not None else Xs


class UFCModel:
    """Logistic regression on fighter-difference features. market=True adds the de-vigged
    market log-odds as a feature (the model then learns only what the market misses)."""
    WIN_COLS = ["d_" + k for k in FEATS]

    def __init__(self, market=False, C=0.3):
        self.market = market; self.C = C

    def _X(self, df):
        X = df[self.WIN_COLS].copy()
        if self.market:
            p = df["p_mkt_a"].clip(0.02, 0.98); X["mkt_logit"] = np.log(p / (1 - p))
        return X

    def fit(self, df):
        X, y = _sym(self._X(df), df["a_won"].values)
        self.med = X.median(); X = X.fillna(self.med)
        self.sc = StandardScaler(with_mean=False).fit(X)
        self.lr = LogisticRegression(C=self.C, fit_intercept=False, max_iter=2000).fit(self.sc.transform(X), y)
        # method model: P(decision) from sums (pace/finishing of both) + format
        mcols = ["s_" + k for k in FEATS] + ["five_rd", "women"]
        Xm = df[mcols].fillna(df[mcols].median())
        self.mcols, self.mmed = mcols, df[mcols].median()
        self.msc = StandardScaler().fit(Xm)
        self.mlr = LogisticRegression(C=0.3, max_iter=2000).fit(self.msc.transform(Xm), (df["method"] == "DEC").astype(int))
        return self

    def predict(self, df):
        X = self._X(df).fillna(self.med)
        p = self.lr.predict_proba(self.sc.transform(X))[:, 1]
        Xm = df[self.mcols].fillna(self.mmed)
        p_dec = self.mlr.predict_proba(self.msc.transform(Xm))[:, 1]
        return p, p_dec

    def coefs(self):
        names = self.WIN_COLS + (["mkt_logit"] if self.market else [])
        return pd.Series(self.lr.coef_[0], index=names).sort_values(key=np.abs, ascending=False)


def ufc_backtest(test_years=range(2016, 2027), min_fights=1, refresh=False):
    res, phys = ufc_load(refresh)
    df, _ = ufc_build_features(res, phys)
    df = ufc_attach_odds(df, refresh)
    df["year"] = df["date"].dt.year
    out = []
    for yr in test_years:
        train = df[(df.year < yr) & (df.year >= 2005)]
        test = df[(df.year == yr) & df.p_mkt_a.notna() & (df.a_n >= min_fights) & (df.b_n >= min_fights)]
        if test.empty: continue
        trm = train.dropna(subset=["p_mkt_a"])
        m0 = UFCModel(market=False).fit(train)
        m1 = UFCModel(market=True).fit(trm)
        t = test.copy()
        t["p_model"], t["p_dec"] = m0.predict(t)
        t["p_anch"], _ = m1.predict(t)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def ufc_report(P, edges=(0.0, 0.03, 0.05, 0.08)):
    from math import isnan
    y = P["a_won"].values
    B = lambda p: float(np.mean((np.asarray(p) - y) ** 2))
    lines = [f"fights: {len(P)}  ({P.date.min().date()} → {P.date.max().date()})",
             f"Brier  market {B(P.p_mkt_a):.4f} | stats model {B(P.p_model):.4f} | market-anchored {B(P.p_anch):.4f} | coin 0.2500"]
    dec = (P["method"] == "DEC").astype(int)
    lines.append(f"P(decision) Brier {np.mean((P.p_dec - dec) ** 2):.4f} vs base-rate {np.mean((dec.mean() - dec) ** 2):.4f}")
    rows = []
    for col in ("p_model", "p_anch"):
        for e in edges:
            ea = P[col] - P["p_mkt_a"]
            bet_a = ea >= e; bet_b = -ea >= e
            pnl = np.where(bet_a, np.where(P.a_won == 1, P.a_dec - 1, -1), 0) + np.where(bet_b, np.where(P.a_won == 0, P.b_dec - 1, -1), 0)
            n = int(bet_a.sum() + bet_b.sum())
            if n == 0: continue
            pn = pnl[(bet_a | bet_b)]
            boots = np.random.default_rng(0).choice(pn, (3000, len(pn))).mean(1)
            rows.append({"model": col, "min_edge": e, "bets": n, "roi": pn.mean(), "roi_p05": np.percentile(boots, 5), "roi_p95": np.percentile(boots, 95)})
    return "\n".join(lines), pd.DataFrame(rows)


def ufc_card(fights, refresh=False):
    """fights: list of dicts {a, b, a_odds, b_odds, five_rd, title, women} with American odds (posted numbers).
    Returns model table: stats-only prob, market-anchored prob, fair market prob, edge, P(decision)."""
    res, phys = ufc_load(refresh)
    df, states = ufc_build_features(res, phys)
    df = ufc_attach_odds(df, refresh)
    m0 = UFCModel(market=False).fit(df)
    m1 = UFCModel(market=True).fit(df.dropna(subset=["p_mkt_a"]))
    ph = phys.to_dict("index"); today = pd.Timestamp.today().normalize()
    def find(name):
        if name in states: return name
        n = norm_name(name); c = [k for k in states if norm_name(k) == n]
        if c: return c[0]
        raise KeyError(f"{name} not found in UFCStats history (debut? spelling?)")
    rows = []
    for f in fights:
        A, Bn = find(f["a"]), find(f["b"])
        fa = states[A].feats(today, ph.get(A)); fb = states[Bn].feats(today, ph.get(Bn))
        r = {"a": A, "b": Bn, "five_rd": f.get("five_rd", 0), "title": f.get("title", 0), "women": f.get("women", 0),
             "a_n": states[A].n, "b_n": states[Bn].n}
        for k in FEATS:
            r["d_" + k] = fa[k] - fb[k]; r["s_" + k] = fa[k] + fb[k]
        dec = lambda am: 1 + am / 100 if am > 0 else 1 + 100 / abs(am)
        r["a_dec"], r["b_dec"] = dec(f["a_odds"]), dec(f["b_odds"])
        ia, ib = 1 / r["a_dec"], 1 / r["b_dec"]; r["p_mkt_a"] = ia / (ia + ib)
        rows.append(r)
    T = pd.DataFrame(rows)
    T["p_stats"], T["p_dec"] = m0.predict(T)
    T["p_model"], _ = m1.predict(T)
    T["edge_a"] = T["p_model"] - T["p_mkt_a"]
    T["ev_a"] = T["p_model"] * (T["a_dec"] - 1) - (1 - T["p_model"])
    T["ev_b"] = (1 - T["p_model"]) * (T["b_dec"] - 1) - T["p_model"]
    def call(r):
        e = abs(r.edge_a); side = r.a if r.edge_a > 0 else r.b
        thin = min(r.a_n, r.b_n) < 4
        # thresholds from the 2016-2026 walk-forward: 3-5% edges lost money, 5%+ made money
        tag = "Play" if e >= UFC_PLAY_EDGE else "Lean" if e >= UFC_LEAN_EDGE else "Sit"
        if thin and tag == "Play": tag = "Lean"   # <4 UFC fights: stats too thin to trust a full Play
        return f"{tag} {side}" if tag != "Sit" else "Sit"
    T["call"] = T.apply(call, axis=1)
    return T[["a", "b", "a_n", "b_n", "p_mkt_a", "p_stats", "p_model", "edge_a", "ev_a", "ev_b", "p_dec", "call"]]

# ======================================================================
# cli.py
# ======================================================================
"""Jerry command line.

  python cli.py snapshot                       # pull current lines for NFL/NHL/MLB -> data/odds_snapshots.csv
  python cli.py close                          # same, tagged as closing snapshot (run just before start)
  python cli.py backtest-nfl                   # walk-forward vs nflverse closing lines
  python cli.py nfl-week 2026 7                # model output for every week-7 game (needs lines in nflverse or snapshot)
  python cli.py nhl HOME AWAY TOTAL [--hg GSAX60] [--ag GSAX60] [--home-b2b] [--away-b2b]
  python cli.py mlb "Home Team" "Away Team" TOTAL [--home-fip X] [--away-fip Y]
  python cli.py bet ...                        # add a bet to the ledger (see --help)
  python cli.py grade BET_ID W|L|P [--close-odds 1.91 --close-line -7 --close-book pinnacle]
  python cli.py report                         # ROI/CLV/Brier + whether Jerry's overrides help
"""


def cmd_nfl_week(season: int, week: int):
    g = games(refresh=True)
    seasons = list(range(2015, season + 1))
    stats = team_week_stats(seasons, refresh=True)
    frame = build_frame(g, stats)
    train = frame[frame.result.notna() & ((frame.season < season) | ((frame.season == season) & (frame.week < week)))]
    test = frame[(frame.season == season) & (frame.week == week)]
    if test.empty:
        print("No games with lines for that week yet in  Lines usually appear a few days out; "
              "or pass posted lines from your own snapshot."); return
    m = NFLModel().fit(train)
    out = m.predict(test)
    cols = ["away_team", "home_team", "spread_line", "pred_margin", "p_home_cover", "p_home_ml", "abs_margin_3", "abs_margin_7",
            "total_line", "pred_total", "p_under", "proj_away", "proj_home"]
    print(out[cols].round(3).to_string(index=False))
    out.to_json(f"data/nfl_{season}_w{week}.json", orient="records", indent=2)
    print(f"\nwritten data/nfl_{season}_w{week}.json  (feed this to Jerry)")


def cmd_nhl(a):
    season = date.today().year if date.today().month >= 9 else date.today().year - 1
    tdf = teams(season).set_index("team")
    league = fit_league(tdf.reset_index())
    lh, la = expected_goals(tdf.loc[a.home], tdf.loc[a.away], league, a.hg or 0.0, a.ag or 0.0, a.home_b2b, a.away_b2b)
    out = nhl_summarize(lh, la, a.total)
    ctx = {"home_goalie_confirmed": a.hg is not None, "away_goalie_confirmed": a.ag is not None,
           "home_b2b": a.home_b2b, "away_b2b": a.away_b2b, "note": "goalie unconfirmed -> total lean defaults to Sit" if a.hg is None or a.ag is None else ""}
    c = build("NHL", f"{a.away}@{a.home}", {"total": {"line": a.total}}, out, ctx)
    print(dumps(c))


def cmd_mlb(a):
    season = date.today().year
    tr = team_runs(season).set_index("team")
    h, v = tr.loc[a.home], tr.loc[a.away]
    lh = expected_runs(h.rs_g, v.ra_g, a.away_fip, home=True)
    la = expected_runs(v.rs_g, h.ra_g, a.home_fip, home=False)
    out = mlb_summarize(lh, la, a.total)
    c = build("MLB", f"{a.away}@{a.home}", {"total": {"line": a.total}}, out,
                    {"home_starter_fip": a.home_fip, "away_starter_fip": a.away_fip, "note": "bullpen/park not modelled yet"})
    print(dumps(c))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sn = sub.add_parser("snapshot"); sn.add_argument("sports", nargs="*", default=["NFL", "NHL", "MLB", "WS", "UFC"])
    sn.add_argument("--force", action="store_true", help="pull even with no events in the next 48h")
    sub.add_parser("close"); sub.add_parser("close-watch"); sub.add_parser("usage"); sub.add_parser("budget")
    bf = sub.add_parser("backfill"); bf.add_argument("sport"); bf.add_argument("start"); bf.add_argument("end")
    bf.add_argument("--markets", default="h2h,totals"); bf.add_argument("--hour", type=int, default=23); bf.add_argument("--dry-run", action="store_true")
    sub.add_parser("backtest-nfl"); sub.add_parser("report")
    sub.add_parser("backtest-ufc")
    u = sub.add_parser("ufc-card"); u.add_argument("card_json", help="JSON list of {a, b, a_odds, b_odds, five_rd, title, women}")
    u.add_argument("--refresh", action="store_true", help="re-download UFCStats + odds history first")
    w = sub.add_parser("nfl-week"); w.add_argument("season", type=int); w.add_argument("week", type=int)
    n = sub.add_parser("nhl"); n.add_argument("home"); n.add_argument("away"); n.add_argument("total", type=float)
    n.add_argument("--hg", type=float); n.add_argument("--ag", type=float); n.add_argument("--home-b2b", action="store_true"); n.add_argument("--away-b2b", action="store_true")
    m = sub.add_parser("mlb"); m.add_argument("home"); m.add_argument("away"); m.add_argument("total", type=float)
    m.add_argument("--home-fip", type=float); m.add_argument("--away-fip", type=float)
    b = sub.add_parser("bet")
    for f in ["game_date", "sport", "game", "market", "selection", "book"]: b.add_argument(f"--{f}", required=True)
    b.add_argument("--line", type=float); b.add_argument("--odds", type=float, required=True); b.add_argument("--stake", type=float, default=1.0)
    b.add_argument("--model-prob", type=float); b.add_argument("--jerry-prob", type=float); b.add_argument("--override-reason", default=""); b.add_argument("--event-id", default="")
    gr = sub.add_parser("grade"); gr.add_argument("bet_id"); gr.add_argument("result", choices=["W", "L", "P"])
    gr.add_argument("--close-odds", type=float); gr.add_argument("--close-line", type=float); gr.add_argument("--close-book")
    a = p.parse_args()

    if a.cmd in ("snapshot", "close", "close-watch", "backfill") and (ROOT / "desk.db").exists() \
            and not os.environ.get("JERRY_ALLOW_PULLS"):
        print("jerry_desk.py owns the Odds API feed (desk.db found) — not spending credits here.\n"
              "Its closes land in data/odds_snapshots.csv, so grade/report/CLV work unchanged.\n"
              "Override with JERRY_ALLOW_PULLS=1 if you really mean it.")
        return
    if a.cmd == "snapshot": snapshot(tuple(x.upper() for x in a.sports), force=a.force)
    elif a.cmd == "close": snapshot(is_close=True, force=True)
    elif a.cmd == "close-watch": close_watch()
    elif a.cmd == "usage": print(json.dumps(usage(), indent=2, default=str))
    elif a.cmd == "budget":
        b = budget(); print(b.to_string(index=False))
        tot = b.credits.sum(); print(f"\nlive: {tot} credits/month of {ODDS_MONTHLY_CREDITS}  (reserve {ODDS_RESERVE_FOR_CLOSES})")
        print("\none-time backfill for NHL/MLB backtests:")
        bft = 0
        for sp, n, mk, w in BACKFILL_PLAN:
            c = backfill_cost(n, mk); bft += c; print(f"  {sp}: {n} days x {mk} = {c} credits  ({w})")
        print(f"\nmonth 1: {tot + bft} of {ODDS_MONTHLY_CREDITS}  ->  {ODDS_MONTHLY_CREDITS - tot - bft} spare;  after: {ODDS_MONTHLY_CREDITS - tot} spare/month")
    elif a.cmd == "backfill": backfill(a.sport.upper(), a.start, a.end, a.hour, a.markets, a.dry_run)
    elif a.cmd == "backtest-nfl": r = run(); print(r["selectivity"].to_string(index=False)); print(r["calibration_spread"].to_string(index=False)); print(f"spread Brier {r['spread_brier']:.4f} vs coin {r['spread_brier_baseline_50']:.4f}")
    elif a.cmd == "nfl-week": cmd_nfl_week(a.season, a.week)
    elif a.cmd == "backtest-ufc":
        P = ufc_backtest(refresh=True); txt, tab = ufc_report(P); print(txt); print(tab.round(3).to_string(index=False))
        R = P[P.date >= "2023-01-01"]; txt2, tab2 = ufc_report(R); print("\n-- 2023 onward --"); print(txt2); print(tab2.round(3).to_string(index=False))
    elif a.cmd == "ufc-card":
        T = ufc_card(json.load(open(a.card_json)), refresh=a.refresh)
        pd.set_option("display.width", 250); print(T.round(3).to_string(index=False))
        T.to_json(DATA / "ufc_card.json", orient="records", indent=2); print(f"\nwritten {DATA / 'ufc_card.json'}  (feed this to Jerry)")
    elif a.cmd == "nhl": cmd_nhl(a)
    elif a.cmd == "mlb": cmd_mlb(a)
    elif a.cmd == "bet":
        bid = add_bet(placed_ts=pd.Timestamp.now('UTC').isoformat(timespec="seconds"), game_date=a.game_date, sport=a.sport, event_id=a.event_id,
                             game=a.game, market=a.market, selection=a.selection, line_at_bet=a.line, decimal_odds_at_bet=a.odds, book=a.book,
                             stake_units=a.stake, model_prob=a.model_prob, jerry_prob=a.jerry_prob, override_reason=a.override_reason)
        print(f"logged {bid}")
    elif a.cmd == "grade":
        grade(a.bet_id, a.result, a.close_line, a.close_odds, a.close_book); print("graded")
    elif a.cmd == "report": print(report())




if __name__ == "__main__":
    main()
