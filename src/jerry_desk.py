#!/usr/bin/env python3
"""Jerry desk v3 — the ONLY program that spends Odds API credits.

Covers NFL, NBA, NHL, MLB (incl. World Series games + Series price) and numbered UFC cards.
Standard library only. Run:  python jerry_desk.py        (opens http://127.0.0.1:8765)

How it spends the month (20K plan):
  * A token bucket refills at  (credits left - reserve) / days left  per day, so the month's
    credits are spread evenly and spent down to the reserve by the reset — nothing expires unused.
  * Priority when tokens are short:  closes > props > sharp-window tape > baseline tape > openers.
  * Whenever the bucket is nearly full (you are under-spending), surplus goes to HISTORICAL
    backfill (closing lines for past NBA/NHL/MLB/NFL seasons -> backtests). Surplus never expires.
  * Closes always run, even from the reserve.

Data written:
  desk.db                      every price CHANGE (not every pull), opens, alerts, history, usage
  data/odds_snapshots.csv      close pulls in jerry.py format -> `jerry.py grade/report` CLV works
  data/odds_usage.csv          every paid call -> `jerry.py usage` works
UFC: only fights listed in cards/*.json are tracked (numbered events). Fight Nights are ignored.
Bets: log them on the dashboard. Moneyline/spread/total bets grade themselves from final scores
(2 credits per sport, only when a bet is due) and get closing-line value from the desk's own closes.
Props and UFC: tap W/L/P. The home page embeds a plain-text summary so Claude can read it by fetching the URL.
"""
import csv
import glob
import json
import jerry_reason as jr
import jerry_grade as jg
import math
import os
import re
import sqlite3
import threading
import time
import unicodedata
import urllib.parse
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.abspath(__file__))
# On Railway, mount a Volume and set JERRY_DATA_DIR to its path (e.g. /data) or every redeploy wipes history.
STORE = os.environ.get("JERRY_DATA_DIR", ROOT)
DB = os.path.join(STORE, "desk.db")   # new name: the old Grok desk used jerry.db with a different schema
DATA = os.path.join(STORE, "data")
CARDS = os.path.join(ROOT, "cards")
KEY_FILE = os.path.join(ROOT, "jerry.key")
os.makedirs(DATA, exist_ok=True)   # creates STORE too
os.makedirs(CARDS, exist_ok=True)
ET = ZoneInfo("America/New_York")
UTC = timezone.utc
PORT = 8765
API = "https://api.the-odds-api.com/v4"

# ---------------------------------------------------------------- config
MONTHLY_CREDITS = int(os.environ.get("ODDS_MONTHLY_CREDITS", "20000"))
RESERVE = 400            # held for closes; tape/props/backfill never spend below it
HARD_FLOOR = 25          # even closes stop here
# 10 named books = 1 region on the quota. Three sharp books + seven retail.
# 20 books, standard across every sport (2026-10-29). The feed bills per batch of 10, so
# 20 costs exactly 2x of 10 — never pull 11-19. Exchanges (Betfair, Matchbook, Smarkets) are
# the sharpest soccer referees; DraftKings, FanDuel and Fanatics are the only CT-bettable books.
BOOKS_CORE = "pinnacle,lowvig,betonlineag,draftkings,fanduel,betmgm,williamhill_us,espnbet,fanatics,betrivers"
BOOKS_EXTRA = "betfair_ex_eu,matchbook,smarkets,marathonbet,williamhill,hardrockbet,ballybet,bovada,betus,mybookieag"
BOOKS = os.environ.get("BOOKS", BOOKS_CORE + "," + BOOKS_EXTRA)
BOOK_UNITS = max(1, -(-len([b for b in BOOKS.split(",") if b.strip()]) // 10))   # credits multiplier per market
SHARP = {"pinnacle", "lowvig", "betonlineag"}
FAMILY = {"lowvig": "betonline", "betonlineag": "betonline"}   # LowVig is BetOnline's sister book

SPORTS = {
    # prop pulls are limited to these markets only (owner's spec, 2026-10-29)
    "NFL": dict(key="americanfootball_nfl", markets="h2h,spreads,totals",
                props=["player_anytime_td", "player_rush_yds", "player_reception_yds", "player_receptions",
                       "player_pass_yds", "player_pass_tds"]),
    "EPL": dict(key="soccer_epl", markets="h2h,spreads,totals",
                props=["player_goal_scorer_anytime", "player_assists"]),
    # The Odds API carries no Champions League player props (EPL, La Liga, Bundesliga, Serie A,
    # Ligue 1 and MLS only), so UCL is game lines only — prop pulls would cost credits for nothing.
    "UCL": dict(key="soccer_uefa_champs_league", markets="h2h,spreads,totals", props=[]),
    "NHL": dict(key="icehockey_nhl", markets="h2h,spreads,totals",
                props=["player_goals", "player_shots_on_goal"]),
    "NBA": dict(key="basketball_nba", markets="h2h,spreads,totals",
                props=["player_points", "player_assists", "player_blocks", "player_rebounds"]),
    # UFC: moneyline only. The Odds API does not carry method of victory for MMA (checked
    # 2026-10-29), so that needs a different source if it's ever wanted.
    "UFC": dict(key="mma_mixed_martial_arts", markets="h2h", props=[]),
}
SOCCER = {"EPL", "UCL"}
# NBA game lines are pulled less often (tape intervals x3); props follow the same 6h/2h/30m schedule as every sport
SPORT_CADENCE = {"NBA": 3}
PROP_SLOTS_BY_SPORT = {"NBA": (360, 120, 60, 30)}   # NBA gets an extra 1-hour prop pull: its prop edge is the proven one
OUTRIGHTS = {"WS": "baseball_mlb_world_series_winner"}   # pulled every 6h while it returns data

# cadence (minutes) — the bucket stretches these automatically when credits are tight
SHARP_WINDOW_MIN = 90      # last 90 minutes before a start cluster
SHARP_EVERY = 120          # 2026-10-29: no 5-minute tape in the last 90 min; closes still pulled below
BASELINE_HOURS = 36        # games inside 36h get baseline tape
BASELINE_EVERY = 120       # games inside 36h: game lines every 2 hours (owner's spec, 2026-10-29)
OPENER_DAYS = 7            # games 36h-7d out get a pull every 12h (openers + early moves)
OPENER_EVERY = 720         # every 12 hours (was 6): early lines matter least for price edges
PROP_SLOTS_MIN = (360, 120, 30)      # owner's spec 2026-10-29: props at 6h, 2h and 30m; the 30m pull is the prop close
CLOSE_SLOTS_MIN = ((12, 8), (4, 0))  # two close pulls per start cluster: T-12..T-8 and T-4..T-0
CLUSTER_MIN = 30
UFC_LOOKAHEAD_DAYS = 30   # numbered cards graded from a month out (2026-10-29)
PROP_FLOOR_HOURS = 8       # props get first claim: credits for every prop pull due in the next 8h are held back from tape

# historical backfill queue (featured markets, one snapshot per game day). Cost: 10 x 3 = 30/day.
# Two snapshots per game day: ~8h before (15 UTC) = "opener-ish", and 23 UTC = near close, so backtests can
# use line movement as a feature. NFL: 17 UTC Sunday close + 05 UTC (Saturday night) opener. Cost 60/day.
BACKFILL = [
    ("NHL", "2024-10-04", "2025-04-17", (15, 23)), ("NFL", "2024-09-05", "2025-01-05", (5, 17)),
    ("NBA", "2024-10-22", "2025-04-13", (15, 23)), ("MLB", "2025-03-27", "2025-09-28", (15, 23)),
    ("NHL", "2023-10-10", "2024-04-18", (15, 23)), ("NFL", "2023-09-07", "2024-01-07", (5, 17)),
    ("NBA", "2023-10-24", "2024-04-14", (15, 23)), ("MLB", "2024-03-28", "2024-09-29", (15, 23)),
    ("NBA", "2022-10-18", "2023-04-09", (15, 23)), ("NHL", "2022-10-07", "2023-04-14", (15, 23)),
]
NFL_GAME_WEEKDAYS = {0, 3, 6}   # Mon, Thu, Sun — skip other NFL days in backfill
BACKFILL_WHEN_FULL = 0.85       # spend on backfill whenever bucket >= 85% full (under-spending)
BACKFILL_SHARE = 0.20           # ...and always give backfill 20% of each day's allowance until the queue is done


# ---------------------------------------------------------------- utils
def load_key():
    if os.environ.get("ODDS_API_KEY"):
        return os.environ["ODDS_API_KEY"].strip()
    if os.path.exists(KEY_FILE):
        return open(KEY_FILE).read().strip()
    return ""


def now_utc():
    return datetime.now(UTC)


def iso(dt):
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def parse(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", s.lower().replace("-", " "))).strip()


def family(book):
    return FAMILY.get(book, book)


def implied(dec):
    return 1.0 / dec if dec else None


def db():
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    con = db()
    con.executescript("""
    create table if not exists events(event_id text primary key, sport text, commence text, home text, away text,
        card text, seen text);
    create table if not exists last_quote(event_id text, book text, market text, outcome text, player text,
        line real, price real, ts text, primary key(event_id, book, market, outcome, player));
    create table if not exists ticks(id integer primary key, ts text, sport text, event_id text, book text,
        market text, outcome text, player text, line real, price real, is_close int);
    create index if not exists ticks_ev on ticks(event_id, market, outcome, player, ts);
    create table if not exists opens(event_id text, book text, market text, outcome text, player text,
        line real, price real, ts text, primary key(event_id, book, market, outcome, player));
    create table if not exists alerts(id integer primary key, ts text, sport text, event_id text, book text,
        family text, sharp int, market text, outcome text, player text, line_from real, line_to real,
        price_from real, price_to real, pp real, dir int, kind text);
    create index if not exists alerts_ev on alerts(event_id, market, outcome, player, ts);
    create index if not exists alerts_ts on alerts(ts);
    create table if not exists usage(id integer primary key, ts text, sport text, endpoint text, cost int, remaining int);
    create index if not exists usage_ts on usage(ts);
    create table if not exists hist(snap_ts text, sport text, event_id text, commence text, home text, away text,
        book text, market text, outcome text, line real, price real);
    create table if not exists backfill_done(sport text, day text, primary key(sport, day));
    create table if not exists slots(k text primary key, ts text);
    create table if not exists model_probs(event_id text, market text, outcome text, player text, line real,
        model_prob real, fair_prob real, edge real, call text, note text, ts text,
        primary key(event_id, market, outcome, player));
    create table if not exists meta(k text primary key, v text);
    create table if not exists bets(id integer primary key, placed_ts text, sport text, event_id text, game text,
        commence text, market text, selection text, player text, line real, price real, book text, stake real,
        model_prob real, note text, result text default 'pending', pnl real, home_score real, away_score real,
        close_book text, close_line real, close_price real, clv_pp real, graded_ts text);
    create index if not exists bets_res on bets(result);
    create table if not exists value_edges(event_id text, market text, outcome text, player text, line real,
        book text, price real, fair real, ev real, ts text, primary key(event_id, market, outcome, player, book));
    """)
    _betcols = {r[1] for r in con.execute("pragma table_info(bets)")}
    if "paper" not in _betcols:
        con.execute("alter table bets add column paper int default 0")
    # reasoning stored as fields, so a loss can be attributed instead of just counted
    _mpcols = {r[1] for r in con.execute("pragma table_info(model_probs)")}
    if "reasoning" not in _mpcols:
        con.execute("alter table model_probs add column reasoning text")
    if "flag" not in _mpcols:
        con.execute("alter table model_probs add column flag text")
    if "plain" not in _mpcols:
        con.execute("alter table model_probs add column plain text")
    _vecols = {r[1] for r in con.execute("pragma table_info(value_edges)")}
    if "flag" not in _vecols:
        con.execute("alter table value_edges add column flag text")
    if "activity" not in _vecols:
        con.execute("alter table value_edges add column activity text")
    for _c, _t in (("comp_json", "text"), ("ratio", "real"), ("model_number", "real"),
                   ("market_number", "real"), ("evidence_kind", "text"),
                   ("corr_group", "text"), ("falsifier", "text"),
                   ("falsifier_fired", "int"), ("steelman", "text"),
                   ("stake_pre_corr", "real"), ("logged_call", "text"), ("sharp_flag", "text"),
                   ("dk_bets_pct", "real"), ("dk_handle_pct", "real"), ("card", "int"), ("card_rank", "int"),
                   ("card_score", "real"), ("card_ts", "text"), ("card_tags", "text")):
        if _c not in _betcols:
            con.execute(f"alter table bets add column {_c} {_t}")
    con.execute("create index if not exists bets_paper on bets(paper, event_id, market, selection, player)")
    con.commit()
    con.close()


def meta_get(con, k, default=None):
    r = con.execute("select v from meta where k=?", (k,)).fetchone()
    return r["v"] if r else default


def meta_set(con, k, v):
    con.execute("insert into meta(k,v) values(?,?) on conflict(k) do update set v=excluded.v", (k, str(v)))


# ---------------------------------------------------------------- HTTP (overridable for tests)
BOOKS_FALLBACK = {"active": False}


def http_get(path, params):
    """One request to the odds feed. If the feed rejects a bookmaker key in the 20-book list,
    retry with the core 10 so KRONOS keeps pricing, and remember the fallback for the summary."""
    p = dict(params)
    if BOOKS_FALLBACK["active"] and p.get("bookmakers") == BOOKS:
        p["bookmakers"] = BOOKS_CORE
    q = urllib.parse.urlencode({"apiKey": load_key(), **p})
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{API}{path}?{q}"), timeout=40) as r:
            return json.loads(r.read().decode()), {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:300]
        except Exception:
            pass
        if e.code in (400, 422) and "bookmaker" in body.lower() and p.get("bookmakers") == BOOKS:
            BOOKS_FALLBACK.update(active=True, reason=f"{e.code} {body[:160]}")
            return http_get(path, params)
        raise



def _clv_for(b, c, con=None):
    """Closing line value in probability points, translated back to the line bet.

    The previous rule recorded CLV only when the closing line exactly matched the
    bet line, which discarded every bet whose number moved — the bets that carry
    the most information. Props were hit hardest: 157 closed rows, none usable.
    """
    # c arrives as a sqlite3.Row, which has no .get(); 2026-10-27 that crashed auto_grade
    # (and so stopped ALL grading) the first time props had a closing price to measure.
    if not c:
        return None
    try:
        c = dict(c)
        if not c.get("price"):
            return None
        comp = {}
        try:
            comp = json.loads(b["comp_json"] or "{}") if "comp_json" in b.keys() else {}
        except Exception:
            comp = {}
        return jr.clv_pp(b["line"], b["price"], c["line"], c["price"],
                         b["market"], b["selection"],
                         sd=comp.get("sd"), mu=comp.get("mu"),
                         close_fair=_close_fair(con, b, c) if con is not None else None)
    except Exception:
        return None      # one bad close must never stop the grading pass


def _close_fair(con, b, c):
    """De-vig the close with BOTH sides of the closing book (power method).
    Falls back to None (single-side approximation) when the other side is missing."""
    try:
        line = c["line"]
        other_line = (-line if (b["market"] == "spreads" and line is not None) else line)
        r = con.execute("""select price from ticks where event_id=? and market=? and player=? and book=?
                           and outcome<>? and line is ? and ts<=? order by ts desc limit 1""",
                        (b["event_id"], b["market"], b["player"] or "", c["book"], b["selection"],
                         other_line, b["commence"])).fetchone()
        if not r or not r["price"]:
            return None
        return jr.fair_two_way_power(float(c["price"]), float(r["price"]))
    except Exception:
        return None

# sports with a player-level results feed wired in; everything else cannot be
# graded yet and is reported as such rather than counted as a backlog
GRADEABLE_PROP_SPORTS = {"NFL", "NBA", "NHL"}   # NBA/NHL via jerry_feeds box scores


class Desk:
    """All scheduling state. tick(now) is the whole brain; tests drive it with a fake clock."""

    def __init__(self, get=http_get):
        self.get = get
        self.lock = threading.Lock()
        self.remaining = None
        self.events_at = {}
        self.cards = {}

    # ---------- credits
    def _record(self, con, sport, endpoint, hdr, now):
        cost = int(hdr.get("x-requests-last", 0) or 0)
        rem = hdr.get("x-requests-remaining")
        if rem not in (None, ""):
            new = int(float(rem))
            if self.remaining is not None and new > self.remaining + 1000:   # monthly reset
                meta_set(con, "tokens", "")
            self.remaining = new
            meta_set(con, "remaining", new)
        if cost:
            con.execute("insert into usage(ts,sport,endpoint,cost,remaining) values(?,?,?,?,?)",
                        (iso(now), sport, endpoint, cost, self.remaining))
            path = os.path.join(DATA, "odds_usage.csv")
            new_file = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["ts", "sport", "endpoint", "cost", "used", "remaining"])
                w.writerow([iso(now), sport, endpoint, cost, hdr.get("x-requests-used", ""), self.remaining])
            self._spend(con, cost, now)
        return cost

    def days_left(self, now):
        nxt = (now.replace(day=1, hour=0, minute=0, second=0, microsecond=0) + timedelta(days=32)).replace(day=1)
        return max((nxt - now).total_seconds() / 86400, 0.05)

    def reserve(self, now):
        """Full reserve most of the month; tapers over the final day so it isn't left to expire."""
        d = self.days_left(now)
        return RESERVE if d >= 1 else max(HARD_FLOOR * 2, RESERVE * d)

    def allowance(self, now):
        """Credits per day that spends the month down to the reserve by the reset."""
        if self.remaining is None:
            return 0.0
        return max(self.remaining - self.reserve(now), 0) / self.days_left(now)

    def _bucket(self, con, now):
        cap = max(self.allowance(now), 1.0)
        raw = meta_get(con, "tokens", "")
        last = meta_get(con, "tokens_ts", "")
        tokens = float(raw) if raw not in ("", None) else cap * 0.5
        if last:
            mins = max((now - parse(last)).total_seconds() / 60, 0)
            tokens = min(cap, tokens + cap / 1440 * mins)
        meta_set(con, "tokens", tokens)
        meta_set(con, "tokens_ts", iso(now))
        return tokens, cap

    def _spend(self, con, cost, now):
        tokens, _ = self._bucket(con, now)
        meta_set(con, "tokens", tokens - cost)

    def can_spend(self, con, cost, now, floor=0.0, priority="tape"):
        if self.remaining is None:
            return False
        if priority == "close":
            return self.remaining - cost >= HARD_FLOOR
        if self.remaining - cost < self.reserve(now):
            return False
        tokens, _ = self._bucket(con, now)
        return tokens - cost >= floor

    # ---------- schedule (free endpoints)
    def discover_ufc_cards(self, con, now):
        """Numbered UFC cards (UFC 3xx) from UFCStats' upcoming-events page, within
        UFC_LOOKAHEAD_DAYS. Writes cards_auto.json so load_cards picks them up; hand-made files in
        cards/ still work and win on conflicts. Runs at most every 12 hours."""
        last = meta_get(con, "ufc_cards_auto_ts")
        if last and (now - parse(last)).total_seconds() < 12 * 3600:
            return
        meta_set(con, "ufc_cards_auto_ts", iso(now))
        ua = {"User-Agent": "Mozilla/5.0 (KRONOS desk)"}
        def fetch(u):
            with urllib.request.urlopen(urllib.request.Request(u, headers=ua), timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        try:
            page = fetch("http://ufcstats.com/statistics/events/upcoming")
        except Exception as ex:
            meta_set(con, "ufc_cards_auto", f"{iso(now)} upcoming page failed: {ex}"[:200]); return
        found = []
        for href, name in re.findall(r'<a[^>]+href="(http[s]?://ufcstats\.com/event-details/[0-9a-f]+)"[^>]*>\s*([^<]+?)\s*</a>', page):
            if not re.match(r"^UFC\s+\d+", name.strip()):
                continue                                   # numbered cards only, no Fight Nights
            tail = page[page.find(href):page.find(href) + 900]
            dm = re.search(r"([A-Z][a-z]+ \d{1,2}, \d{4})", tail)
            try:
                when = datetime.strptime(dm.group(1), "%B %d, %Y").replace(tzinfo=UTC) if dm else None
            except ValueError:
                when = None
            if when and (when - now).days > UFC_LOOKAHEAD_DAYS:
                continue
            try:
                ev = fetch(href)
            except Exception:
                continue
            names = re.findall(r'href="http[s]?://ufcstats\.com/fighter-details/[0-9a-f]+"[^>]*>\s*([^<]+?)\s*</a>', ev)
            fights = [{"a": names[i], "b": names[i + 1]} for i in range(0, len(names) - 1, 2)]
            if fights:
                found.append({"event": name.strip(), "date": dm.group(1) if dm else None, "fights": fights})
        try:
            with open(os.path.join(STORE, "cards_auto.json"), "w") as fh:
                json.dump(found, fh)
            meta_set(con, "ufc_cards_auto", f"{iso(now)} {len(found)} numbered cards: " +
                     ", ".join(f"{c['event']} ({len(c['fights'])} fights)" for c in found)[:300])
        except Exception as ex:
            meta_set(con, "ufc_cards_auto", f"{iso(now)} write failed: {ex}"[:200])

    def load_cards(self):
        cards = {}
        try:
            for c in json.load(open(os.path.join(STORE, "cards_auto.json"))):
                for f in c.get("fights", []):
                    cards[frozenset((norm_name(f["a"]), norm_name(f["b"])))] = c["event"]
                    cards[self.last_names(f["a"], f["b"])] = c["event"]
        except Exception:
            pass
        for path in glob.glob(os.path.join(CARDS, "*.json")):
            try:
                raw = json.load(open(path))
            except Exception:
                continue
            name = raw.get("event") if isinstance(raw, dict) else os.path.splitext(os.path.basename(path))[0]
            fights = raw.get("fights", []) if isinstance(raw, dict) else raw
            for f in fights:
                cards[frozenset((norm_name(f["a"]), norm_name(f["b"])))] = name
                cards[self.last_names(f["a"], f["b"])] = name   # "Alex"/"Alexander", "Ramazan"/"Ramazonbek" still match
        self.cards = cards

    @staticmethod
    def last_names(a, b):
        return frozenset(("last", norm_name(a).split(" ")[-1], norm_name(b).split(" ")[-1]))

    def refresh_events(self, con, sport, now, force=False):
        if not force and sport in self.events_at and now - self.events_at[sport] < timedelta(minutes=30):
            return
        data, hdr = self.get(f"/sports/{SPORTS[sport]['key']}/events", {"dateFormat": "iso"})
        self._record(con, sport, "events", hdr, now)
        self.events_at[sport] = now
        if sport == "UFC":
            try:
                self.discover_ufc_cards(con, now)
            except Exception as ex:
                meta_set(con, "ufc_cards_auto", f"{iso(now)} error: {ex}"[:200])
            self.load_cards()
        for e in data:
            card = None
            if sport == "UFC":
                card = (self.cards.get(frozenset((norm_name(e.get("home_team")), norm_name(e.get("away_team")))))
                        or self.cards.get(self.last_names(e.get("home_team", ""), e.get("away_team", ""))))
                if not card:
                    continue   # not on a numbered card -> ignored
            con.execute("""insert into events(event_id,sport,commence,home,away,card,seen) values(?,?,?,?,?,?,?)
                           on conflict(event_id) do update set commence=excluded.commence, seen=excluded.seen,
                           card=coalesce(excluded.card, events.card)""",
                        (e["id"], sport, e["commence_time"], e.get("home_team", ""), e.get("away_team", ""), card, iso(now)))

    def upcoming(self, con, sport, now, within):
        rows = con.execute("select * from events where sport=? and commence>? and commence<=?",
                           (sport, iso(now), iso(now + within))).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def clusters(events):
        starts = sorted(parse(e["commence"]) for e in events)
        out = []
        for s in starts:
            if not out or (s - out[-1]).total_seconds() > CLUSTER_MIN * 60:
                out.append(s)
        return out

    # ---------- paid pulls
    def pull_featured(self, con, sport, now, is_close=False, close_ids=None):
        cfg = SPORTS[sport]
        data, hdr = self.get(f"/sports/{cfg['key']}/odds", {"bookmakers": BOOKS, "markets": cfg["markets"],
                                                              "oddsFormat": "decimal", "dateFormat": "iso"})
        self._record(con, sport, "close" if is_close else "odds", hdr, now)
        keep = {r["event_id"] for r in con.execute("select event_id from events where sport=?", (sport,))}
        if is_close and close_ids is not None:
            keep &= set(close_ids)     # a close pull only records the games about to start
        rows = []
        for g in data:
            if g["id"] not in keep:
                continue
            for b in g.get("bookmakers", []):
                for m in b.get("markets", []):
                    for o in m.get("outcomes", []):
                        rows.append((g, b["key"], m["key"], o["name"], "", o.get("point"), o["price"]))
        self.store(con, sport, rows, now, is_close)
        meta_set(con, f"last_{sport}", iso(now))
        if not is_close:
            self._scan_price(con, now)
        return len(data)

    def pull_props(self, con, sport, event_id, now, is_close=False):
        cfg = SPORTS[sport]
        data, hdr = self.get(f"/sports/{cfg['key']}/events/{event_id}/odds",
                             {"bookmakers": BOOKS, "markets": ",".join(cfg["props"]), "oddsFormat": "decimal"})
        self._record(con, sport, "props", hdr, now)
        rows = []
        for b in data.get("bookmakers", []):
            for m in b.get("markets", []):
                for o in m.get("outcomes", []):
                    rows.append((data, b["key"], m["key"], o["name"], o.get("description", ""), o.get("point"), o["price"]))
        self.store(con, sport, rows, now, is_close)
        self._scan_price(con, now)

    def pull_outright(self, con, label, now):
        data, hdr = self.get(f"/sports/{OUTRIGHTS[label]}/odds", {"bookmakers": BOOKS, "markets": "outrights",
                                                                 "oddsFormat": "decimal"})
        self._record(con, label, "outright", hdr, now)
        if not data:
            meta_set(con, f"outright_off_{label}", iso(now + timedelta(days=7)))
            return
        rows = []
        for g in data:
            con.execute("insert or ignore into events(event_id,sport,commence,home,away,seen) values(?,?,?,?,?,?)",
                        (g["id"], label, g["commence_time"], g.get("sport_title", label), "", iso(now)))
            for b in g.get("bookmakers", []):
                for m in b.get("markets", []):
                    for o in m.get("outcomes", []):
                        rows.append((g, b["key"], m["key"], o["name"], "", None, o["price"]))
        self.store(con, label, rows, now, False)

    def pull_history(self, con, sport, day, hour, now):
        snap = datetime.fromisoformat(day[:10]).replace(hour=hour, tzinfo=UTC)
        data, hdr = self.get(f"/historical/sports/{SPORTS[sport]['key']}/odds",
                             {"bookmakers": BOOKS, "markets": "h2h,spreads,totals", "oddsFormat": "decimal",
                              "date": snap.strftime("%Y-%m-%dT%H:%M:%SZ")})
        self._record(con, sport, "historical", hdr, now)
        for g in (data or {}).get("data", []):
            if parse(g["commence_time"]) < snap:
                continue
            for b in g.get("bookmakers", []):
                for m in b.get("markets", []):
                    for o in m.get("outcomes", []):
                        con.execute("insert into hist values(?,?,?,?,?,?,?,?,?,?,?)",
                                    ((data or {}).get("timestamp", iso(snap)), sport, g["id"], g["commence_time"],
                                     g.get("home_team", ""), g.get("away_team", ""), b["key"], m["key"], o["name"],
                                     o.get("point"), o["price"]))
        con.execute("insert or ignore into backfill_done values(?,?)", (sport, day))

    # ---------- storage + move detection
    def store(self, con, sport, rows, now, is_close):
        ts = iso(now)
        close_rows = []
        for g, book, market, outcome, player, line, price in rows:
            eid = g["id"]
            key = (eid, book, market, outcome, player)
            prev = con.execute("""select line, price from last_quote where event_id=? and book=? and market=?
                                  and outcome=? and player=?""", key).fetchone()
            changed = prev is None or prev["price"] != price or prev["line"] != line
            if changed or is_close:
                con.execute("insert into ticks(ts,sport,event_id,book,market,outcome,player,line,price,is_close) "
                            "values(?,?,?,?,?,?,?,?,?,?)", (ts, sport, eid, book, market, outcome, player, line, price, int(is_close)))
            if changed:
                con.execute("""insert into last_quote values(?,?,?,?,?,?,?,?) on conflict(event_id,book,market,outcome,player)
                               do update set line=excluded.line, price=excluded.price, ts=excluded.ts""",
                            (*key, line, price, ts))
                con.execute("insert or ignore into opens values(?,?,?,?,?,?,?,?)", (*key, line, price, ts))
            if prev is not None and changed:
                self.alert(con, sport, eid, book, market, outcome, player, prev["line"], line, prev["price"], price, now)
            if is_close and market in ("h2h", "spreads", "totals"):
                close_rows.append([ts, 1, sport, eid, g.get("commence_time", ""), g.get("home_team", ""),
                                   g.get("away_team", ""), book, market, outcome, line, price])
        if close_rows:
            path = os.path.join(DATA, "odds_snapshots.csv")
            new_file = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["snapshot_ts", "is_close", "sport", "event_id", "commence_time", "home_team",
                                "away_team", "book", "market", "outcome", "line", "decimal_odds"])
                w.writerows(close_rows)

    @staticmethod
    def direction(market, outcome, lf, lt, pf, pt):
        """+1 = this outcome got more expensive / more support, -1 = less."""
        if lf is not None and lt is not None and lf != lt:
            if market == "totals" or outcome in ("Over", "Under"):
                d = 1 if lt > lf else -1
                return d if outcome == "Over" else -d
            return 1 if lt < lf else -1          # spread: -3 -> -3.5 = more support
        return 1 if (implied(pt) or 0) > (implied(pf) or 0) else -1

    def alert(self, con, sport, eid, book, market, outcome, player, lf, lt, pf, pt, now):
        pp = round(((implied(pt) or 0) - (implied(pf) or 0)) * 100, 2)
        if lf == lt and abs(pp) < 1.0:
            return
        d = self.direction(market, outcome, lf, lt, pf, pt)
        con.execute("""insert into alerts(ts,sport,event_id,book,family,sharp,market,outcome,player,line_from,line_to,
                       price_from,price_to,pp,dir,kind) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (iso(now), sport, eid, book, family(book), int(book in SHARP), market, outcome, player, lf, lt, pf, pt, pp, d, "Seen"))
        lo = iso(now - timedelta(minutes=15))
        win = con.execute("""select id, family, sharp, dir from alerts where event_id=? and market=? and outcome=?
                             and player=? and ts>=?""", (eid, market, outcome, player, lo)).fetchall()
        same = [w for w in win if w["dir"] == d]
        fams = {w["family"] for w in same}
        sharp = any(w["sharp"] for w in same)
        kind = "Steam" if sharp and len(fams) >= 3 else "Sharp" if sharp else "Move" if len(fams) >= 2 else "Seen"
        if kind != "Seen":
            con.executemany("update alerts set kind=? where id=?", [(kind, w["id"]) for w in same])
        if kind == "Steam" and os.environ.get("TELEGRAM_BOT_TOKEN"):
            k = f"alert:steam:{eid}:{market}:{outcome}:{player}:{now.strftime('%Y%m%d%H')}"
            if not self._done(con, k):
                ev = con.execute("select home, away from events where event_id=?", (eid,)).fetchone()
                try:
                    import jerry_live
                    jerry_live.telegram(f"STEAM {sport} {ev['away'] if ev else ''} @ {ev['home'] if ev else ''}: {player} {outcome} {market} "
                                        f"{lf}->{lt} {pf}->{pt} led by {book}")
                except Exception:
                    pass
                self._mark(con, k, now)

    # ---------- the brain
    def tick(self, now=None):
        now = now or now_utc()
        with self.lock:
            con = db()
            try:
                if self.remaining is None:
                    _, hdr = self.get("/sports", {})
                    self._record(con, "-", "sports", hdr, now)
                self._tick(con, now)
                con.commit()
            finally:
                con.close()

    def _done(self, con, k):
        return con.execute("select 1 from slots where k=?", (k,)).fetchone() is not None

    def _scan_price(self, con, now):
        """Price edges need no model and no API spend, so scan right after every odds pull
        instead of once an hour (a mispriced DK prop rarely lasts 59 minutes). New paper
        rows are pushed to Telegram the moment they appear."""
        import sys as _sys
        L = _sys.modules.get("jerry_live")
        if L is None or not hasattr(L, "price_edges"):
            return
        try:
            before = con.execute("select coalesce(max(id),0) from bets").fetchone()[0]
            L.price_edges(con, now); con.commit()
            new = con.execute("""select sport, game, market, player, selection, line, book, price, note, sharp_flag
                                 from bets where id>? and paper=1 and note like 'price edge%'""", (before,)).fetchall()
            if new and os.environ.get("TELEGRAM_BOT_TOKEN") and hasattr(L, "telegram"):
                msg = "\n".join(f"PRICE {r['sport']} {r['game']} | {r['market']} {r['player'] or ''} {r['selection']} "
                                f"{'' if r['line'] is None else r['line']} @ {r['book']} {r['price']} | {r['note'][11:60]} [{r['sharp_flag'] or 'NO READ'}]"
                                for r in new[:8])
                L.telegram(msg)
        except Exception as ex:   # noqa: BLE001 - a scan must never block a pull
            try:
                con.rollback()
            except Exception:
                pass
            meta_set(con, "error_price_scan", f"{iso(now)} {ex}"[:200])

    def _mark(self, con, k, now):
        con.execute("insert or ignore into slots values(?,?)", (k, iso(now)))

    def _due(self, con, k, every_min, now):
        last = meta_get(con, f"due_{k}")
        return last is None or now - parse(last) >= timedelta(minutes=every_min)

    def _tick(self, con, now):
        plan = {}
        for sport in SPORTS:
            try:
                self.refresh_events(con, sport, now)
            except Exception as e:
                meta_set(con, f"error_{sport}", f"{iso(now)} {e}"[:200])
                continue
            look = timedelta(days=UFC_LOOKAHEAD_DAYS if sport == "UFC" else OPENER_DAYS)
            plan[sport] = self.upcoming(con, sport, now, look)

        # 1) closes — always
        for sport, evs in plan.items():
            for c in self.clusters(evs):
                mins = (c - now).total_seconds() / 60
                for i, (hi, lo) in enumerate(CLOSE_SLOTS_MIN):
                    k = f"close:{sport}:{iso(c)}:{i}"
                    if lo <= mins < hi and not self._done(con, k):
                        if sport == "UFC" and i == 0:
                            self._mark(con, k, now); continue      # one close per UFC fight cluster
                        if self.can_spend(con, len(SPORTS[sport]["markets"].split(",")) * BOOK_UNITS, now, priority="close"):
                            ids = [e["event_id"] for e in evs
                                   if abs((parse(e["commence"]) - c).total_seconds()) <= CLUSTER_MIN * 60]
                            self.pull_featured(con, sport, now, is_close=True, close_ids=ids)
                            meta_set(con, f"due_tape_{sport}", iso(now))   # a close also counts as tape
                            self._mark(con, k, now)

        # 2) props — reserve tokens for props due in the next 2h so tape can't starve them
        prop_floor = 0.0
        for sport, evs in plan.items():
            props = SPORTS[sport]["props"]
            for e in evs:
                start = parse(e["commence"])
                for s in PROP_SLOTS_BY_SPORT.get(sport, PROP_SLOTS_MIN):
                    k = f"prop:{e['event_id']}:{s}"
                    mins = (start - now).total_seconds() / 60
                    if self._done(con, k) or not props:
                        continue
                    is_last = (s == min(PROP_SLOTS_BY_SPORT.get(sport, PROP_SLOTS_MIN)))
                    window = 15 if s <= 30 else 30
                    if s - window < mins <= s:
                        if self.can_spend(con, len(props) * BOOK_UNITS, now, priority="close" if is_last else "props"):
                            try:
                                self.pull_props(con, sport, e["event_id"], now, is_close=is_last)
                            except Exception as ex:
                                meta_set(con, f"error_props_{sport}", f"{iso(now)} {ex}"[:200])
                        self._mark(con, k, now)   # missed slots are skipped, not retried
                    elif 0 < mins - s <= PROP_FLOOR_HOURS * 60:
                        prop_floor += len(props) * BOOK_UNITS   # tokens held back so tape can't starve upcoming props

        # 3) tape: sharp window > baseline > openers. Each sport pull covers every game in it.
        tokens, cap = self._bucket(con, now)
        day0 = iso(now.astimezone(ET).replace(hour=0, minute=0, second=0, microsecond=0))
        bf_done = con.execute("select coalesce(sum(cost),0) c from usage where endpoint='historical' and ts>=?",
                              (day0,)).fetchone()["c"]
        hold_bf = max(cap * BACKFILL_SHARE - bf_done, 0) if self.next_backfill(con) else 0
        # Walk sports most-starved first. Dict order (NFL, NBA, NHL, MLB) meant the
        # prop floor was always consumed by the first two, and NHL/MLB never got a
        # game-line pull at all — their boards showed empty moneylines for hours.
        def _staleness(sp):
            last = meta_get(con, f"due_tape_{sp}")
            return (now - parse(last)).total_seconds() if last else 1e9
        for sport in sorted(plan, key=_staleness, reverse=True):
            evs = plan[sport]
            if not evs:
                continue
            cost = len(SPORTS[sport]["markets"].split(",")) * BOOK_UNITS
            soonest = min((parse(e["commence"]) - now).total_seconds() / 60 for e in evs)
            in_sharp = any(0 < (c - now).total_seconds() / 60 <= SHARP_WINDOW_MIN for c in self.clusters(evs))
            if in_sharp:
                every, floor = SHARP_EVERY, prop_floor
            elif soonest <= BASELINE_HOURS * 60:
                every, floor = BASELINE_EVERY, prop_floor + hold_bf * 0.5 + cost * 6
            else:
                every, floor = OPENER_EVERY, prop_floor + hold_bf + cost * 12
            every = every * SPORT_CADENCE.get(sport, 1)
            # starvation guard: a sport with games inside the baseline window and no
            # game-line quotes at all gets one pull regardless of the prop floor
            if soonest <= BASELINE_HOURS * 60:
                ids = [e["event_id"] for e in evs if (parse(e["commence"]) - now).total_seconds() <= BASELINE_HOURS * 3600]
                # count GAMES with a fresh (last 12h) game line, not quote rows: on 2026-10-28 one NBA game
                # still had yesterday's quotes, the guard saw "covered", and 11 games went unpriced all day
                fresh = iso(now - timedelta(hours=12))
                covered = con.execute(f"""select count(distinct event_id) from last_quote where player='' and market='h2h'
                                          and ts>? and event_id in ({",".join("?" * len(ids))})""", (fresh, *ids)).fetchone()[0] if ids else 0
                if ids and covered < 0.5 * len(ids):
                    floor = 0
                    meta_set(con, f"starved_{sport}", f"{iso(now)} {covered}/{len(ids)} games priced")
            if self._due(con, f"tape_{sport}", every, now) and self.can_spend(con, cost, now, floor=floor):
                try:
                    self.pull_featured(con, sport, now)
                except Exception as ex:
                    meta_set(con, f"error_{sport}", f"{iso(now)} {ex}"[:200])
                meta_set(con, f"due_tape_{sport}", iso(now))

        # 4) outrights (World Series price) every 6h while live
        for label in OUTRIGHTS:
            off = meta_get(con, f"outright_off_{label}")
            if off and parse(off) > now:
                continue
            if self._due(con, f"out_{label}", 360, now) and self.can_spend(con, 1, now, floor=prop_floor + 10):
                try:
                    self.pull_outright(con, label, now)
                except Exception as ex:
                    meta_set(con, f"error_{label}", f"{iso(now)} {ex}"[:200])
                meta_set(con, f"due_out_{label}", iso(now))

        # 4b) auto-grade pending bets (scores endpoint, 2 credits per sport, only when something is due)
        try:
            self.auto_grade(con, now)
            # a clean pass retires the old error, so the summary never shows a fixed problem as live
            con.execute("delete from meta where k='error_grade'")
        except Exception as ex:
            meta_set(con, "error_grade", f"{iso(now)} {ex}"[:200])

        # 5) historical backfill: a guaranteed daily share, plus any surplus when the bucket is nearly full
        tokens, cap = self._bucket(con, now)
        day0 = iso(now.astimezone(ET).replace(hour=0, minute=0, second=0, microsecond=0))
        bf_today = con.execute("select coalesce(sum(cost),0) c from usage where endpoint='historical' and ts>=?",
                               (day0,)).fetchone()["c"]
        owed = bf_today + 30 <= cap * BACKFILL_SHARE
        if (owed and self.can_spend(con, 30, now, floor=prop_floor)) or \
           (tokens >= cap * BACKFILL_WHEN_FULL and self.can_spend(con, 30, now, floor=cap * 0.5)):
            job = self.next_backfill(con)
            if job:
                try:
                    self.pull_history(con, *job, now)
                except Exception as ex:
                    meta_set(con, "error_backfill", f"{iso(now)} {ex}"[:200])
                    con.execute("insert or ignore into backfill_done values(?,?)", (job[0], job[1]))

    def next_backfill(self, con):
        for sport, start, end, hours in BACKFILL:
            d = datetime.fromisoformat(start)
            stop = datetime.fromisoformat(end)
            while d <= stop:
                if not (sport == "NFL" and d.weekday() not in NFL_GAME_WEEKDAYS):
                    for hour in hours:
                        day = f"{d.date().isoformat()}T{hour:02d}"
                        if not con.execute("select 1 from backfill_done where sport=? and day=?", (sport, day)).fetchone():
                            return sport, day, hour
                d += timedelta(days=1)
        return None

    # ---------- dashboard data
    def status(self):
        con = db()
        now = now_utc()
        tokens, cap = self._bucket(con, now) if self.remaining is not None else (0, 0)
        con.commit()
        day0 = iso(now.astimezone(ET).replace(hour=0, minute=0, second=0, microsecond=0))
        month0 = iso(now.replace(day=1, hour=0, minute=0, second=0, microsecond=0))
        today = con.execute("select sport, endpoint, sum(cost) c from usage where ts>=? group by sport, endpoint", (day0,)).fetchall()
        month = con.execute("select coalesce(sum(cost),0) c from usage where ts>=?", (month0,)).fetchone()["c"]
        bf = con.execute("select sport, count(*) n from backfill_done group by sport").fetchall()
        errs = {r["k"]: r["v"] for r in con.execute("select k, v from meta where k like 'error%'")}
        con.close()
        return {"remaining": self.remaining, "allowance_per_day": round(self.allowance(now)),
                "tokens": round(tokens), "bucket_cap": round(cap), "days_to_reset": round(self.days_left(now), 2),
                "spent_today": sum(r["c"] for r in today), "spent_month_by_desk": month,
                "today": [dict(r) for r in today], "backfill_days": {r["sport"]: r["n"] for r in bf},
                "reserve": RESERVE, "errors": errs}

    def board(self, sport):
        con = db()
        now = now_utc()
        evs = con.execute("select * from events where sport=? and commence>? order by commence limit 60",
                          (sport, iso(now - timedelta(hours=4)))).fetchall()
        out = []
        for e in evs:
            q = con.execute("select * from last_quote where event_id=? and player=''", (e["event_id"],)).fetchall()
            o = {(r["book"], r["market"], r["outcome"]): r for r in con.execute(
                "select * from opens where event_id=? and player=''", (e["event_id"],))}
            def summary(market):
                rows = [r for r in q if r["market"] == market]
                if not rows:
                    return None
                ref = [r for r in rows if r["book"] == "pinnacle"] or [r for r in rows if r["book"] in SHARP] or rows
                sides = {}
                for r in ref:
                    op = o.get((r["book"], r["market"], r["outcome"]))
                    sides[r["outcome"]] = {"book": r["book"], "line": r["line"], "price": r["price"],
                                           "open_line": op["line"] if op else None, "open_price": op["price"] if op else None,
                                           "best": max(x["price"] for x in rows if x["outcome"] == r["outcome"] and x["line"] == r["line"])}
                return sides
            kinds = {r["kind"]: r["n"] for r in con.execute(
                "select kind, count(*) n from alerts where event_id=? and kind!='Seen' group by kind", (e["event_id"],))}
            last = con.execute("select ts, outcome, market, kind from alerts where event_id=? and kind in ('Steam','Sharp') "
                               "order by id desc limit 1", (e["event_id"],)).fetchone()
            mp = {(r["market"], r["outcome"]): dict(r) for r in con.execute(
                "select * from model_probs where event_id=? and player=''", (e["event_id"],))}
            out.append({"event_id": e["event_id"], "start": e["commence"], "home": e["home"], "away": e["away"],
                        "card": e["card"], "h2h": summary("h2h"), "spreads": summary("spreads"), "totals": summary("totals"),
                        "moves": kinds, "last_signal": dict(last) if last else None,
                        "value": [dict(v) for v in con.execute("select * from value_edges where event_id=? order by ev desc", (e["event_id"],))],
                        "model": {f"{k[0]}|{k[1]}": {"p": v["model_prob"], "fair": v["fair_prob"], "edge": v["edge"], "call": v["call"], "note": v["note"],
                                                           "plain": v["plain"] if "plain" in v.keys() else None} for k, v in mp.items()}})
        con.close()
        return out


    # ---------- models (jerry_live) — runs in its own thread, never blocks the odds loop
    def refresh_models(self):
        try:
            import jerry_live
            res = jerry_live.refresh_all(DB)
            con = db(); meta_set(con, "models_last", iso(now_utc()) + " " + json.dumps(res)); con.commit(); con.close()
            print(f"[desk] models refreshed: {res}", flush=True)
        except Exception as e:
            print(f"[desk] model refresh failed: {e}", flush=True)
            con = db(); meta_set(con, "error_models", f"{iso(now_utc())} {e}"[:200]); con.commit(); con.close()

    def set_goalie(self, team_abbr, name, hours=12):
        con = db()
        meta_set(con, f"goalie_{team_abbr.upper()}", json.dumps({"name": name, "until": (now_utc() + timedelta(hours=hours)).isoformat()}))
        con.commit(); con.close()

    def model_rows(self, event_id):
        con = db()
        rows = [dict(r) for r in con.execute("select * from model_probs where event_id=? order by market, player, outcome", (event_id,))]
        con.close(); return rows

    def edges(self, min_abs=0.015, limit=60):
        con = db()
        rows = [dict(r) for r in con.execute("""select m.*, e.sport, e.home, e.away, e.commence from model_probs m join events e using(event_id)
                 where e.commence>? and m.call not like 'see %' and m.call not like 'Sit%' and abs(coalesce(m.edge,0))>=?
                 order by abs(m.edge) desc limit ?""", (iso(now_utc()), min_abs, limit))]
        con.close(); return rows

    # ---------- bets: log, close, grade, report
    def add_bet(self, f, now):
        con = db()
        ev = con.execute("select * from events where event_id=?", (f.get("event_id", ""),)).fetchone()
        if f.get("market") == "parlay":   # legs can span games; graded by hand
            ev = {"sport": ev["sport"] if ev else f.get("sport", "MIX"), "event_id": f"parlay-{int(now.timestamp())}",
                  "away": "", "home": "Parlay", "commence": iso(now)}
        if not ev:
            con.close()
            raise ValueError("unknown event_id — pick the game from the board")
        if f.get("market") != "parlay" and parse(ev["commence"]) <= now:
            con.close()   # keeps the public picks page honest: nothing logged after the start
            raise ValueError("that event has already started — picks must be logged before the start")
        am = float(f["odds"])
        dec = 1 + am / 100 if am > 0 else 1 + 100 / abs(am)
        line = float(f["line"]) if f.get("line") not in (None, "") else None
        con.execute("""insert into bets(placed_ts,sport,event_id,game,commence,market,selection,player,line,price,book,
                       stake,model_prob,note) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (iso(now), ev["sport"], ev["event_id"], f"{ev['away']} @ {ev['home']}" if ev["away"] else ev["home"], ev["commence"], f["market"],
                     f.get("selection", ""), f.get("player", ""), line, round(dec, 4), f.get("book", ""),
                     float(f.get("stake") or 1), float(f["model_prob"]) if f.get("model_prob") else None, f.get("note", "")))
        con.commit()
        con.close()

    @staticmethod
    def settle(market, selection, line, home, away, hs, as_):
        """Returns W / L / P from final scores, or None if this market can't be graded from scores."""
        if hs is None or as_ is None:
            return None
        if selection == home:
            mine, theirs = hs, as_
        elif selection == away:
            mine, theirs = as_, hs
        else:
            mine = theirs = None
        if market == "h2h" and selection == "Draw":
            return "W" if hs == as_ else "L"                     # three-way soccer moneyline
        if market == "h2h" and mine is not None:
            return "W" if mine > theirs else "L" if mine < theirs else "P"
        if market == "spreads" and mine is not None and line is not None:
            m = mine + line - theirs
            return "W" if m > 0 else "L" if m < 0 else "P"
        if market == "totals" and line is not None and selection in ("Over", "Under"):
            t = hs + as_
            if t == line:
                return "P"
            return "W" if (t > line) == (selection == "Over") else "L"
        return None   # player props are settled by jerry_grade, not from team scores

    def closing(self, con, b):
        """Closing number for the bet: Pinnacle close at the same line if possible, else any close, else last tick."""
        q = """select book, line, price from ticks where event_id=? and market=? and outcome=? and player=?
               and ts<=? {extra} order by (book='pinnacle') desc, ts desc limit 1"""
        args = (b["event_id"], b["market"], b["selection"], b["player"] or "", b["commence"])
        for extra in ("and is_close=1 and line is ?", "and is_close=1", "and line is ?", ""):
            a = args + ((b["line"],) if "line is ?" in extra else ())
            r = con.execute(q.format(extra=extra), a).fetchone()
            if r:
                return r
        return None

    def auto_grade(self, con, now):
        pend = con.execute("select * from bets where result='pending' and commence<=?",
                           (iso(now - timedelta(hours=3)),)).fetchall()
        for b in pend:   # closing line value can be filled as soon as the game starts, even for props we can't score
            if b["close_price"] is None:
                c = self.closing(con, b)
                if c and c["price"]:
                    clv = _clv_for(b, c, con)
                    con.execute("update bets set close_book=?, close_line=?, close_price=?, clv_pp=? where id=?",
                                (c["book"], c["line"], c["price"], clv, b["id"]))
        if not pend:
            return
        sports = {b["sport"] for b in pend if b["sport"] in SPORTS and b["market"] in ("h2h", "spreads", "totals")}
        try:
            jr._ensure_numbers(con)
            sports |= {r[0] for r in con.execute("""select distinct sport from jerry_numbers where commence<? and
                       event_id not in (select event_id from final_scores)""", (iso(now),)) if r[0] in SPORTS}
        except sqlite3.Error:
            pass
        self._grade_props(con, pend, now)
        scores = {}
        for sp in sports:
            if not self._due(con, f"scores_{sp}", 30, now) or not self.can_spend(con, 2, now, floor=0):
                continue
            data, hdr = self.get(f"/sports/{SPORTS[sp]['key']}/scores", {"daysFrom": 3, "dateFormat": "iso"})
            self._record(con, sp, "scores", hdr, now)
            meta_set(con, f"due_scores_{sp}", iso(now))
            for g in data:
                if g.get("completed") and g.get("scores"):
                    sc = {x["name"]: float(x["score"]) for x in g["scores"]}
                    scores[g["id"]] = (sc.get(g.get("home_team")), sc.get(g.get("away_team")))
                    try:
                        jr._ensure_numbers(con)
                        hs_, as__ = scores[g["id"]]
                        if hs_ is not None and as__ is not None:
                            con.execute("insert or replace into final_scores values(?,?,?,?,?)", (g["id"], sp, hs_, as__, iso(now)))
                    except sqlite3.Error:
                        pass
        for b in pend:
            ev = con.execute("select home, away from events where event_id=?", (b["event_id"],)).fetchone()
            if b["event_id"] not in scores or not ev:
                continue
            hs, as_ = scores[b["event_id"]]
            res = self.settle(b["market"], b["selection"], b["line"], ev["home"], ev["away"], hs, as_)
            if res:
                self.finish(con, b, res, now, hs, as_)


    def _grade_props(self, con, pend, now):
        """Settle player props from a player-stats feed. Scratches void, exact lines push."""
        nfl = [b for b in pend if b["sport"] == "NFL" and (b["market"] or "").startswith("player_")]
        if not nfl:
            return
        try:
            season, week = self._nfl_season_week(now)
            rep = jg.grade_nfl_props(con, season, week, iso(now))
            # a quarantine row is a live failure only while its bet is still
            # pending; once the bet grades the row is history and must go, or the
            # count never falls and real failures stay hidden behind stale ones
            try:
                con.execute("""delete from grade_quarantine where bet_id in
                               (select id from bets where result!='pending')""")
                con.commit()
            except sqlite3.Error:
                pass
            if rep.get("quarantined"):
                meta_set(con, "grade_quarantine_n", str(rep["quarantined"]))
        except Exception as e:   # noqa: BLE001 - grading must never take the desk down
            meta_set(con, "grade_props_error", f"{iso(now)} {e}")

    @staticmethod
    def _nfl_season_week(now):
        """NFL week from the date: season starts the Thursday after Labor Day."""
        y = now.year if now.month >= 3 else now.year - 1
        sept = datetime(y, 9, 1, tzinfo=UTC)
        labor = sept + timedelta(days=(7 - sept.weekday()) % 7)      # first Monday
        kickoff = labor + timedelta(days=3)                          # Thursday
        wk = max(1, min(22, ((now - kickoff).days // 7) + 1))
        return y, wk

    def finish(self, con, b, res, now, hs=None, as_=None):
        pnl = {"W": b["stake"] * (b["price"] - 1), "L": -b["stake"], "P": 0.0}[res]
        c = self.closing(con, b)
        clv = _clv_for(b, c, con) if c else None   # +pp = the market moved toward you
        con.execute("""update bets set result=?, pnl=?, home_score=?, away_score=?, close_book=?, close_line=?,
                       close_price=?, clv_pp=?, graded_ts=? where id=?""",
                    (res, round(pnl, 4), hs, as_, c["book"] if c else None, c["line"] if c else None,
                     c["price"] if c else None, clv, iso(now), b["id"]))

    def grade_manual(self, bet_id, res):
        con = db()
        b = con.execute("select * from bets where id=?", (bet_id,)).fetchone()
        if b and res in ("W", "L", "P"):
            self.finish(con, b, res, now_utc())
            con.commit()
        con.close()

    def report(self):
        import random
        con = db()
        rows = [dict(r) for r in con.execute("select * from bets where coalesce(paper,0)=0 order by id")]
        con.close()
        done = [r for r in rows if r["result"] in ("W", "L", "P")]
        out = {"bets_total": len(rows), "pending": sum(r["result"] == "pending" for r in rows), "graded": len(done)}
        def block(rs):
            if not rs:
                return None
            st = sum(r["stake"] for r in rs); pnl = sum(r["pnl"] for r in rs)
            rng = random.Random(0); boots = []
            for _ in range(2000):
                smp = [rs[rng.randrange(len(rs))] for _ in rs]
                boots.append(sum(x["pnl"] for x in smp) / max(sum(x["stake"] for x in smp), 1e-9))
            boots.sort()
            clv = [r["clv_pp"] for r in rs if r["clv_pp"] is not None]
            probs = [(r["model_prob"], 1.0 if r["result"] == "W" else 0.0) for r in rs if r["model_prob"] and r["result"] != "P"]
            return {"n": len(rs), "record": f"{sum(r['result']=='W' for r in rs)}-{sum(r['result']=='L' for r in rs)}-{sum(r['result']=='P' for r in rs)}",
                    "units": round(pnl, 2), "roi": round(pnl / st, 4) if st else None,
                    "roi_90ci": [round(boots[100], 4), round(boots[1899], 4)],
                    "clv_pp_avg": round(sum(clv) / len(clv), 2) if clv else None,
                    "beat_close_pct": round(sum(c > 0 for c in clv) / len(clv), 3) if clv else None,
                    "brier": round(sum((p - y) ** 2 for p, y in probs) / len(probs), 4) if probs else None}
        out["all"] = block(done)
        out["by_sport"] = {sp: block([r for r in done if r["sport"] == sp]) for sp in sorted({r["sport"] for r in done})}
        out["by_market"] = {m: block([r for r in done if r["market"] == m]) for m in sorted({r["market"] for r in done})}
        out["by_type"] = {t: block([r for r in done if bet_type(r) == t]) for t in sorted({bet_type(r) for r in done})}
        fights = {}
        for r in done:
            if r["sport"] == "UFC" and r["market"] != "parlay" and r["selection"] not in ("Over", "Under"):
                fights.setdefault(r["selection"], []).append(r)
        out["by_fighter"] = {k: block(v) for k, v in sorted(fights.items(), key=lambda kv: -len(kv[1]))}
        against = {}
        for r in done:
            if r["sport"] == "UFC" and r["market"] in ("h2h", "method") and " @ " in (r["game"] or ""):
                a, h = r["game"].split(" @ ", 1)
                opp = h if r["selection"] == a else a if r["selection"] == h else None
                if opp:
                    against.setdefault(opp, []).append(r)
        out["against_fighter"] = {k: block(v) for k, v in against.items()}
        out["recent"] = rows[-25:]
        return out

    def paper_report(self):
        """Every Lean/Play and every price edge, paper-traded at 1u: the evidence for which markets have a real edge."""
        con = db()
        rows = [dict(r) for r in con.execute("select * from bets where paper=1 order by id")]
        con.close()
        def kind(r):
            n = r["note"] or ""
            return "price edge, line-adj" if n.startswith("price edge (line-adj") else "price edge" if n.startswith("price edge") else "model"
        def fam(r):
            m = r["market"]
            return "props" if m.startswith("player_") else {"h2h": "moneyline", "spreads": "spread", "totals": "total"}.get(m, m)
        def block(rs):
            clv = [r["clv_pp"] for r in rs if r["clv_pp"] is not None]
            done = [r for r in rs if r["result"] in ("W", "L", "P")]   # V = void, excluded
            st = sum(r["stake"] for r in done); pnl = sum(r["pnl"] or 0 for r in done)
            return {"n": len(rs), "closed": len(clv), "clv_pp_avg": round(sum(clv) / len(clv), 2) if clv else None,
                    "beat_close_pct": round(sum(c > 0 for c in clv) / len(clv), 3) if clv else None,
                    "graded": len(done), "record": f"{sum(r['result']=='W' for r in done)}-{sum(r['result']=='L' for r in done)}-{sum(r['result']=='P' for r in done)}",
                    "units": round(pnl, 2), "roi": round(pnl / st, 4) if st else None}
        def game_ci(rs, n_boot=2000):
            """90% ROI interval resampling GAMES, not bets: six props in one game are one bet."""
            done = [r for r in rs if r["result"] in ("W", "L", "P") and (r["stake"] or 0) > 0]
            by_g = {}
            for r in done:
                g = by_g.setdefault(r["event_id"], [0.0, 0.0]); g[0] += r["pnl"] or 0; g[1] += r["stake"]
            gs = list(by_g.values())
            if len(gs) < 8:
                return None
            import random
            rnd = random.Random(7); out_ = []
            for _ in range(n_boot):
                smp = [gs[rnd.randrange(len(gs))] for _ in gs]
                st = sum(x[1] for x in smp)
                out_.append(sum(x[0] for x in smp) / st if st else 0.0)
            out_.sort()
            return [round(out_[int(0.05 * n_boot)], 4), round(out_[int(0.95 * n_boot) - 1], 4), len(gs)]
        def block2(rs):
            b = block(rs); b["roi_90ci_by_game"] = game_ci(rs); return b
        groups, versions, ebuckets, hbuckets = {}, {}, {}, {}
        for r in rows:
            groups.setdefault(f"{r['sport']} {fam(r)} ({kind(r)})", []).append(r)
            versions.setdefault(r.get("model_version") or "pre-version (before 2026-10-26)", []).append(r)
            e = r.get("edge")
            if e is not None and (r["stake"] or 0) > 0:
                e = abs(float(e)) * (100 if abs(float(e)) < 1 else 1)
                bk = "0-2%" if e < 2 else "2-4%" if e < 4 else "4-6%" if e < 6 else "6%+"
                ebuckets.setdefault(f"{kind(r).split(',')[0]} {bk}", []).append(r)
            h = r.get("hours_to_start")
            if h is not None:
                hb = "<1h" if h < 1 else "1-6h" if h < 6 else "6-24h" if h < 24 else "24h+"
                hbuckets.setdefault(hb, []).append(r)
        out = {"total": len(rows), "all": block2(rows) if rows else None,
               "by_group": {k: block2(v) for k, v in sorted(groups.items())},
               "by_version": {k: block(v) for k, v in sorted(versions.items())},
               # the calibration question: do bigger edges win more? A flat curve means the edge is noise.
               "by_edge_bucket": {k: block(v) for k, v in sorted(ebuckets.items())},
               "by_hours_to_start": {k: block(v) for k, v in sorted(hbuckets.items())},
               "recent": rows[-40:]}
        return out


    def _grading_status_line(self):
        """Prop grading health, in the summary so it can be read without SQL."""
        con = db()
        try:
            q = con.execute("select count(*) from grade_quarantine").fetchone()[0]
        except sqlite3.Error:
            q = 0
        try:
            done = con.execute("""select count(*) from bets where result in ('W','L','P','V')
                                  and market like 'player_%'""").fetchone()[0]
            void = con.execute("select count(*) from bets where result='V'").fetchone()[0]
            # props only grade for sports with a player results feed; counting the
            # rest as "pending" makes a known gap look like a stuck queue forever
            rows = con.execute("""select sport, count(*) n from bets where result='pending'
                                  and market like 'player_%' group by sport""").fetchall()
            pend = sum(r["n"] for r in rows if r["sport"] in GRADEABLE_PROP_SPORTS)
            nofeed = {r["sport"]: r["n"] for r in rows if r["sport"] not in GRADEABLE_PROP_SPORTS}
        except sqlite3.Error:
            done = void = pend = 0
            nofeed = {}
        err = con.execute("select v from meta where k='grade_props_error'").fetchone()
        feed_err = {r[0][11:]: r[1] for r in con.execute("select k, v from meta where k like 'feed_error_%'")}
        try:
            nres = {r[0]: r[1] for r in con.execute("select sport, count(*) from game_results group by sport")}
            rejects = con.execute("select count(*) from meta where k like 'label_reject_%'").fetchone()[0]
        except sqlite3.Error:
            nres, rejects = {}, 0
        con.close()
        bits = [f"PROPS GRADED: {done} settled ({void} void), {pend} pending"]
        if nres:
            bits.append("| results on file: " + ", ".join(f"{k} {v}" for k, v in sorted(nres.items())))
        if rejects:
            bits.append(f"| {rejects} props rejected on roster mismatch")
        for sp, msg in feed_err.items():
            bits.append(f"| {sp} feed error: {msg[:90]}")
        if nofeed:
            detail = ", ".join(f"{k} {v}" for k, v in sorted(nofeed.items()))
            bits.append(f"| {sum(nofeed.values())} awaiting a results feed ({detail}) — not gradeable yet")
        if q:
            bits.append(f"| {q} quarantined (unresolved names — see /api/quarantine)")
        if err:
            bits.append(f"| last grading error: {err[0][:120]}")
        return " ".join(bits)


    def _spin_lines(self):
        """Jerry's own number for every upcoming game, the market's, the gap, and the
        verdict — plus picks the sharp gate blocked, and how Jerry's spins have graded."""
        con = db(); out = []
        try:
            jr._ensure_numbers(con)
            rows = con.execute("""select * from jerry_numbers where commence>? order by commence limit 60""",
                               (iso(now_utc()),)).fetchall()
            out += ["", "KRONOS SPIN (KRONOS's own projection vs the market; EXPLOIT = gap >= 0.75 error bands; "
                        "[flag] shows the sharp/public side, warnings show failed blind-spot checks):"]
            for r in rows:
                sp_ = jr.spin(con, r)
                j = f"{r['away']} {r['away_pts']:.1f} @ {r['home']} {r['home_pts']:.1f}"
                m = (f"market {sp_['market'][1]:.1f}-{sp_['market'][0]:.1f}" if sp_["market"]
                     else (f"market total {r['mkt_total']}" if r["mkt_total"] is not None else "market n/a"))
                gaps = []
                if sp_["gap_margin"] is not None:
                    gaps.append(f"margin {sp_['gap_margin']:+.1f} ({sp_['r_margin']:.2f})")
                if sp_["gap_total"] is not None:
                    gaps.append(f"total {sp_['gap_total']:+.1f} ({sp_['r_total']:.2f})")
                if r["p_home"] is not None and r["mkt_p_home"] is not None:
                    gaps.append(f"home win {r['p_home']:.0%} vs {r['mkt_p_home']:.0%}")
                out.append(f"{r['sport']} {str(r['commence'])[:16]}Z | KRONOS {j} | {m} | gap {', '.join(gaps) or 'n/a'}")
                out.append(f"      {sp_['why']} | {r['detail'][:160]}")
            # flag scorecard: does being on the sharp side actually pay? measured on paper bets
            fr = con.execute("""select coalesce(sharp_flag,'NO READ') f, count(*) n,
                                       sum(case when clv_pp is not null then 1 else 0 end) nc,
                                       sum(case when clv_pp>0 then 1 else 0 end) bc, avg(clv_pp) clv,
                                       sum(case when result='W' then 1 else 0 end) w, sum(case when result='L' then 1 else 0 end) l,
                                       sum(case when result in ('W','L','P') then pnl else 0 end) u,
                                       sum(case when result in ('W','L','P') then stake else 0 end) st
                                from bets where paper=1 and sharp_flag is not null group by f order by n desc""").fetchall()
            if fr:
                out += ["", "FLAG SCORECARD (paper bets by money flag — is the sharp side worth following?):"]
                for f in fr:
                    bc = f"{f['bc'] / f['nc']:.0%}" if f["nc"] else "n/a"
                    roi = f"{f['u'] / f['st']:+.1%}" if f["st"] else "n/a"
                    clv = f"{f['clv']:+.2f}pp" if f["clv"] is not None else "n/a"
                    out.append(f"[{f['f']}] {f['n']} bets | beat close {bc} | CLV {clv} | {f['w']}-{f['l']} | ROI {roi}")
            try:
                sm = {k: v for k, v in con.execute("select k, v from meta where k in ('splits_articles_source','splits_table')")}
                nsp = con.execute("select count(*) from splits").fetchone()[0]
                bysrc = ", ".join(f"{r[0]} {r[1]}" for r in con.execute("select source, count(*) from splits group by source"))
                out += ["", f"DK SPLITS: {nsp} rows on file ({bysrc or 'none'})",
                        f"   table: {sm.get('splits_table', 'not run')}",
                        f"   articles: {sm.get('splits_articles_source', 'not run')}"]
            except sqlite3.Error:
                pass
            # ---- KRONOS shadow card: rule-built, locked 90 min before start, flat 1u ----
            try:
                from zoneinfo import ZoneInfo as _Z
                today = datetime.now(_Z("America/New_York")).date().isoformat()
                card = con.execute("""select b.*, e.home, e.away from bets b join events e using(event_id)
                                      where b.paper=1 and b.card=1 order by b.commence, b.card_rank""").fetchall()
                tonight = [c for c in card if str(c["card_ts"] or "")[:10] >= (now_utc() - timedelta(hours=30)).date().isoformat()]
                if tonight:
                    out += ["", "KRONOS CARD (shadow; chosen by rule: shrunk EV, consensus price, projection agrees, verified minutes, "
                                "<=2 per game, <=60% one direction; locks 90 min before start):"]
                    live = {(v["event_id"], v["market"], v["outcome"], v["player"]) for v in con.execute(
                        "select event_id, market, outcome, player from value_edges")}
                    for c in tonight:
                        still = "edge still live" if (c["event_id"], c["market"], c["selection"], c["player"]) in live else "edge gone at last refresh"
                        out.append(f"#{c['card_rank']} {c['sport']} {c['away']} @ {c['home']} | {c['player'] or ''} {c['selection']} "
                                   f"{'' if c['line'] is None else c['line']} {c['market']} | {c['book']} {c['price']} | score {c['card_score']} "
                                   f"| {c['result']} | {still} | {c['card_tags'] or ''}")
                sc = con.execute("""select
                        sum(case when card=1 and result in ('W','L','P') then 1 else 0 end) cn,
                        sum(case when card=1 and result='W' then 1 else 0 end) cw,
                        sum(case when card=1 and result in ('W','L','P') then pnl else 0 end) cu,
                        avg(case when card=1 then clv_pp end) cclv,
                        sum(case when card=1 and clv_pp>0 then 1 else 0 end) cbc, sum(case when card=1 and clv_pp is not null then 1 else 0 end) cbn,
                        sum(case when result in ('W','L','P') then 1 else 0 end) an,
                        sum(case when result='W' then 1 else 0 end) aw,
                        sum(case when result in ('W','L','P') then pnl else 0 end) au,
                        avg(clv_pp) aclv,
                        sum(case when clv_pp>0 then 1 else 0 end) abc, sum(case when clv_pp is not null then 1 else 0 end) abn
                        from bets where paper=1 and evidence_kind in ('price','price_adj') and model_version like 'price-v4%'""").fetchone()
                if sc and (sc["an"] or 0):
                    def _f(n, w, u, clv, bc, bn):
                        return (f"{w}-{(n or 0) - (w or 0)} | {(u or 0):+.2f}u ({(u or 0) / n:+.1%}) | CLV "
                                f"{'n/a' if clv is None else f'{clv:+.2f}pp'} | beat close {'n/a' if not bn else f'{bc / bn:.0%}'}") if n else "no graded bets yet"
                    out += ["", "SHADOW SCORECARD (price-v4 rules, flat 1u):",
                            f"  KRONOS card : {_f(sc['cn'], sc['cw'], sc['cu'], sc['cclv'], sc['cbc'], sc['cbn'])}",
                            f"  all edges   : {_f(sc['an'], sc['aw'], sc['au'], sc['aclv'], sc['abc'], sc['abn'])}"]
            except Exception as _e:   # noqa: BLE001
                out.append(f"(shadow card unavailable: {str(_e)[:100]})")
            # ---- learning layer (jerry_signals): measured weights, calibration, autopsy, timing ----
            try:
                def _mj(k):
                    r_ = con.execute("select v from meta where k=?", (k,)).fetchone()
                    try:
                        return json.loads(r_[0]) if r_ else None
                    except Exception:
                        return r_[0] if r_ else None
                sw, lr = _mj("signal_weights"), _mj("learn_report")
                out += ["", "LEARNING (every signal starts at weight 0; it moves a price only after 20 graded spots and a measured effect):"]
                if sw:
                    grad = [k for k, v in sw["status"].items() if v == "graduated"]
                    live_w = {k: v for k, v in sw["weights"].items() if v}
                    out.append(f"  graded rows with signals: {sw['n_rows']} | graduated: {', '.join(grad) or 'none yet'}"
                               + (f" | weights: {', '.join(f'{k} {v:+.3f}' for k, v in live_w.items())}" if live_w else ""))
                    prob = [f"{k} {v.split(' ')[1]}" for k, v in sw["status"].items() if v.startswith("probation")]
                    out.append("  on probation: " + ", ".join(prob))
                if lr:
                    cal = lr.get("calibration") or {}
                    if cal.get("buckets"):
                        out.append("  calibration (price says -> actually hit): " + " | ".join(
                            f"{b['bucket']} n{b['n']} hit {b['hit']:.0%}" for b in cal["buckets"]) + f" | Brier {cal.get('brier')}")
                    if lr.get("autopsy"):
                        out.append("  results by cause, 14 days: " + ", ".join(f"{k} {v}" for k, v in
                                   sorted(lr["autopsy"].items(), key=lambda x: -x[1])))
                    if lr.get("timing"):
                        out.append("  timing: " + " | ".join(f"{k} n{v['n']} ROI {v['roi']:+.1%} CLV {v['clv'] if v['clv'] is not None else 'n/a'}"
                                                         for k, v in sorted(lr["timing"].items())))
                    sc = lr.get("scorecard") or {}
                    best = [(k, v) for k, v in sc.items() if v.get("n", 0) >= 20 and v.get("high") and v.get("low")]
                    for k, v in best[:8]:
                        out.append(f"  {k}: high n{v['high']['n']} ROI {v['high']['roi']:+.1%} vs low n{v['low']['n']} ROI {v['low']['roi']:+.1%}")
                cx = _mj("contradictions") or []
                if cx:
                    out.append("  CONTRADICTIONS: " + " || ".join(cx[:6]))
                feeds = [(_mj("trends_status"), "DK trends"), (_mj("nfl_usage_status"), "nflverse"),
                         (_mj("signals_error_last"), "signals error")]
                out.append("  feeds: " + " | ".join(f"{lab}: {str(v)[:140]}" for v, lab in feeds if v))
            except Exception as _e:   # noqa: BLE001
                out.append(f"(learning section unavailable: {str(_e)[:100]})")
            g = jr.grade_spins(con)
            if g:
                out += ["", "SPIN SCORECARD (mean absolute error on completed games; lower is better):"]
                for sp, v in sorted(g.items()):
                    out.append(f"{sp}: {v['games']} games | margin KRONOS {v['margin_mae_jerry']} vs market {v['margin_mae_market']} "
                               f"| total KRONOS {v['total_mae_jerry']} vs market {v['total_mae_market']}")
        except Exception as e:   # noqa: BLE001
            out.append(f"(spin section unavailable: {str(e)[:120]})")
        finally:
            con.close()
        return out


    def _credit_forecast_line(self, st):
        """Projected spend to the reset date, by sport, so a squeeze shows up before it slows the data."""
        con = db()
        try:
            since = iso(now_utc() - timedelta(days=3))
            rows = con.execute("""select sport, coalesce(sum(cost),0) c from usage where ts>? group by sport""", (since,)).fetchall()
            per_day = {r["sport"] or "?": r["c"] / 3.0 for r in rows}
            tot = sum(per_day.values())
            days = float(st.get("days_to_reset") or 0)
            left = float(st.get("remaining") or 0)
            proj = tot * days
            verdict = "on track" if proj <= left else f"SHORT by ~{proj - left:.0f} — KRONOS will stretch pull intervals"
            by = ", ".join(f"{k} {v:.0f}" for k, v in sorted(per_day.items(), key=lambda x: -x[1]))
            fb = f" | BOOKS FALLBACK to core 10: {BOOKS_FALLBACK.get('reason')}" if BOOKS_FALLBACK.get("active") else f" | {len(BOOKS.split(','))} books"
            ufc = meta_get(con, "ufc_cards_auto")
            return (f"CREDIT FORECAST: ~{tot:.0f}/day over the last 3 days ({by}) -> ~{proj:.0f} to reset vs {left:.0f} left: {verdict}{fb}"
                    + (f"\nUFC CARDS: {ufc}" if ufc else ""))
        except Exception as e:   # noqa: BLE001
            return f"CREDIT FORECAST unavailable ({str(e)[:80]})"
        finally:
            con.close()

    def summary_text(self):
        """Plain-text digest embedded in the home page so a web fetch (no JavaScript) can read everything."""
        st = self.status(); rp = self.report()
        lines = ["KRONOS DESK SUMMARY  (generated " + iso(now_utc()) + ")",
                 f"credits left {st['remaining']} | budget/day {st['allowance_per_day']} | spent today {st['spent_today']} | "
                 f"reset in {st['days_to_reset']}d | backfill days {st['backfill_days']}",
                 self._credit_forecast_line(st),
                 f"errors: {st['errors'] or 'none'}", "",
                 f"BETS: {rp['bets_total']} total, {rp['pending']} pending, {rp['graded']} graded",
                 self._grading_status_line(),
                 f"overall: {json.dumps(rp['all'])}",
                 f"by sport: {json.dumps(rp['by_sport'])}",
                 f"by bet type: {json.dumps(rp['by_type'])}",
                 f"UFC by fighter backed: {json.dumps(rp['by_fighter'])}",
                 f"UFC by fighter faded: {json.dumps(rp['against_fighter'])}", "", "RECENT BETS:"]
        for b in rp["recent"]:
            lines.append(f"#{b['id']} {b['sport']} {b['game']} | {b['market']} {b['player'] or ''} {b['selection']} "
                         f"{'' if b['line'] is None else b['line']} @ {b['price']} ({b['book']}) x{b['stake']} | "
                         f"model {b['model_prob']} | {b['result']} pnl {b['pnl']} | close {b['close_book']} "
                         f"{b['close_line']} @ {b['close_price']} | CLV {b['clv_pp']}pp | score {b['away_score']}-{b['home_score']}")
        pr = self.paper_report()
        lines += ["", f"PAPER LEDGER (every call auto-logged at 1u; the evidence for which markets have a real edge): {pr['total']} rows",
                  f"all: {json.dumps(pr['all'])}"]
        for k, v in pr["by_group"].items():
            lines.append(f"  {k}: {json.dumps(v)}")
        for title, key in (("by model version (the 150-bet clock runs per version)", "by_version"),
                           ("by edge size at call (bigger edges should win more; flat = noise)", "by_edge_bucket"),
                           ("by hours before start when logged", "by_hours_to_start")):
            if pr.get(key):
                lines.append(f" {title}:")
                for k, v in pr[key].items():
                    lines.append(f"  {k}: n {v['n']} | CLV {v['clv_pp_avg']} | beat close {v['beat_close_pct']} | {v['record']} | {v['units']}u | ROI {v['roi']}")
        lines += ["", "PRICE EDGES (soft-book price beats Pinnacle fair; CT books):"]
        c3 = db()
        for v in c3.execute("""select v.*, e.sport, e.home, e.away, e.commence from value_edges v join events e using(event_id)
                               where e.commence>? order by v.ev desc limit 40""", (iso(now_utc()),)):
            fl = v["flag"] if "flag" in v.keys() and v["flag"] else "NO READ"
            act = v["activity"] if "activity" in v.keys() and v["activity"] else ""
            lines.append(f"{v['sport']} {v['commence'][:16]}Z {v['away']} @ {v['home']} | {v['market']} {v['player'] or ''} {v['outcome']} "
                         f"{'' if v['line'] is None else v['line']} | {v['book']} {v['price']} vs fair {v['fair']:.1%} | EV {v['ev']:+.1%} | [{fl}]"
                         + (f" [{act}]" if act else ""))
        c3.close()
        lines += ["", "MODEL EDGES (KRONOS models vs Pinnacle fair price; [flag] = where the sharp and public money is):"]
        for r in self.edges():
            if r.get("plain"):
                lines.append(f"{r['sport']} {r['commence'][:16]}Z {r['away']} @ {r['home']} | >> {r['plain']}")
            lines.append(f"{r['sport']} {r['commence'][:16]}Z {r['away']} @ {r['home']} | {r['market']} {r['player'] or ''} {r['outcome']} "
                         f"{'' if r['line'] is None else r['line']} | model {r['model_prob']:.1%} fair {r['fair_prob']:.1%} edge {r['edge']*100:+.1f} | {r['call']} "
                         f"[{r.get('flag') or 'NO READ'}] | {r['note'] or ''}")
            rz = r["reasoning"] if "reasoning" in r.keys() else None
            if rz:
                lines.append(f"      why: {rz}")
        mlast = None
        try:
            c2 = db(); mlast = meta_get(c2, "models_last"); c2.close()
        except Exception:
            pass
        lines.append(f"(models last refreshed: {mlast})")
        lines += self._spin_lines()
        lines += ["", "RECENT SIGNALS (Steam/Sharp/Move):"]
        for a in self.feed(40):
            lines.append(f"{a['ts']} {a['kind']} {a['sport']} {a.get('away') or ''} @ {a.get('home') or ''} | {a['market']} "
                         f"{a['player'] or ''} {a['outcome']} {a['line_from']}->{a['line_to']} {a['price_from']}->{a['price_to']} ({a['book']})")
        lines += ["", "UPCOMING BOARD (Pinnacle or sharpest; decimal odds):"]
        for sp in SPORTS:
            for e in self.board(sp)[:20]:
                def fmt(m):
                    d = e.get(m) or {}
                    return " / ".join(f"{k} {'' if v['line'] is None else v['line']} {v['price']} (open "
                                      f"{'' if v['open_line'] is None else str(v['open_line']) + ' '}{v['open_price']}, best {v['best']})"
                                      for k, v in d.items())
                lines.append(f"{sp} {e['start']} {e['away']} @ {e['home']} {('['+e['card']+']') if e['card'] else ''} id={e['event_id']}")
                lines.append(f"   ML: {fmt('h2h')}")
                if e.get("spreads"): lines.append(f"   SPREAD: {fmt('spreads')}")
                if e.get("totals"): lines.append(f"   TOTAL: {fmt('totals')}")
                if e["moves"]: lines.append(f"   moves: {e['moves']}")
        return "\n".join(lines)

    def feed(self, limit=80):
        con = db()
        rows = con.execute("""select a.*, e.home, e.away from alerts a left join events e using(event_id)
                              where kind in ('Steam','Sharp','Move') order by a.id desc limit ?""", (limit,)).fetchall()
        con.close()
        return [dict(r) for r in rows]


BET_TYPES = {"h2h": "Moneyline", "spreads": "Spread", "totals": "Total", "method": "Method of victory",
             "rounds": "Rounds", "prop": "Player prop", "parlay": "Parlay"}


def bet_type(r):
    return BET_TYPES.get(r["market"], r["market"])


# ---------------------------------------------------------------- web

# ------------------------------------------------------------------ odds board (/odds)
# Every book side by side for each game, each price colored by its edge over a fair price.
BOOK_SHORT = {"draftkings": "DK", "fanduel": "FD", "fanatics": "FAN", "pinnacle": "PIN", "lowvig": "LV",
              "betonlineag": "BOL", "betfair_ex_eu": "BFX", "matchbook": "MB", "smarkets": "SMK", "marathonbet": "MAR",
              "betmgm": "MGM", "williamhill_us": "CZR", "espnbet": "ESPN", "betrivers": "BR", "hardrockbet": "HR",
              "ballybet": "BAL", "williamhill": "WH", "bovada": "BOV", "betus": "BUS", "mybookieag": "MYB"}
BETTABLE = ("draftkings", "fanduel", "fanatics")            # the only legal books in Connecticut
ODDS_SHARP_W = {"pinnacle": 3.0, "betfair_ex_eu": 3.0, "lowvig": 2.0, "betonlineag": 2.0, "matchbook": 2.0,
                "smarkets": 2.0, "marathonbet": 1.5}
MKT_LABEL = {"h2h": "Moneyline", "spreads": "Spread", "totals": "Total", "player_points": "Points",
             "player_rebounds": "Rebounds", "player_assists": "Assists", "player_blocks": "Blocks",
             "player_anytime_td": "Anytime TD", "player_rush_yds": "Rushing yards", "player_reception_yds": "Receiving yards",
             "player_receptions": "Receptions", "player_pass_yds": "Passing yards", "player_pass_tds": "Passing TDs",
             "player_goals": "Goals", "player_shots_on_goal": "Shots on goal", "player_goal_scorer_anytime": "Anytime goalscorer"}


def _dec(dec):
    """Decimal odds only (owner's rule, 2026-10-29): American odds are never shown."""
    try:
        dec = float(dec)
    except (TypeError, ValueError):
        return "—"
    return "—" if dec <= 1.0 else f"{dec:.2f}"


def _book_order(books):
    rank = {b: i for i, b in enumerate(list(BETTABLE) + list(ODDS_SHARP_W) + list(BOOK_SHORT))}
    return sorted(books, key=lambda b: rank.get(b, 99))


def odds_grid(con, event_id):
    """Rows of {section, label, prices{book: dec}, fair, ev{book: ev}, best, best_bet} for one game.

    Fair price: each book's own prices are de-vigged (power method), then averaged with sharp books
    weighted up (Pinnacle and Betfair x3). A book's edge is measured against the fair price built from
    every OTHER book, so a book can't make its own price look good."""
    q = [dict(r) for r in con.execute("select * from last_quote where event_id=?", (event_id,))]
    groups = {}
    for r in q:
        if not r["price"]:
            continue
        m, pl = r["market"], r["player"] or ""
        if m == "h2h":
            key = None
        elif m == "spreads":
            key = None if r["line"] is None else round(abs(float(r["line"])), 2)
        else:
            key = None if r["line"] is None else round(float(r["line"]), 2)
        groups.setdefault((m, pl, key), []).append(r)
    rows = []
    for (m, pl, key), rs in groups.items():
        sides = {}
        for r in rs:
            sides.setdefault((r["outcome"], r["line"]), {})[r["book"]] = float(r["price"])
        # each book's de-vigged view of this market (needs every side from that book)
        side_ids = list(sides)
        book_fair = {}
        for bk in {r["book"] for r in rs}:
            if m == "spreads":
                # pair complementary sides: same |line|, opposite sign, different team
                pairs = [(a, b) for i, a in enumerate(side_ids) for b in side_ids[i + 1:]
                         if a[0] != b[0] and a[1] is not None and b[1] is not None and abs(float(a[1]) + float(b[1])) < 1e-9]
                for a, b in pairs:
                    if bk in sides[a] and bk in sides[b]:
                        f = jr.power_devig([sides[a][bk], sides[b][bk]])
                        if f:
                            book_fair.setdefault(bk, {})[a] = f[0]; book_fair[bk][b] = f[1]
            else:
                if all(bk in sides[s] for s in side_ids) and len(side_ids) >= 2:
                    f = jr.power_devig([sides[s][bk] for s in side_ids])
                    if f:
                        book_fair[bk] = {s: f[i] for i, s in enumerate(side_ids)}
        for s in side_ids:
            prices = sides[s]
            def fair_excluding(ex):
                num = den = 0.0
                for bk, fv in book_fair.items():
                    if bk == ex or s not in fv:
                        continue
                    w = ODDS_SHARP_W.get(bk, 1.0)
                    num += fv[s] * w; den += w
                return (num / den) if den else None
            fair_all = fair_excluding(None)
            ev = {}
            vs_avg = False
            for bk, p in prices.items():
                fx = fair_excluding(bk)
                if fx:
                    ev[bk] = p * fx - 1
            if fair_all is None and len(prices) >= 3:
                # one-sided market (anytime scorer / TD: books post "Yes" only) — no second side to de-vig,
                # so compare each price with the other books' average implied probability instead
                vs_avg = True
                for bk, p in prices.items():
                    others = [1 / v for b2, v in prices.items() if b2 != bk]
                    ev[bk] = p * (sum(others) / len(others)) - 1
            best = max(prices, key=lambda b: prices[b]) if prices else None
            bet_books = [b for b in prices if b in BETTABLE]
            best_bet = max(bet_books, key=lambda b: prices[b]) if bet_books else None
            out, line = s
            if m == "h2h":
                label = out
            elif m == "spreads":
                label = f"{out} {float(line):+g}"
            elif m in ("player_anytime_td", "player_goal_scorer_anytime"):
                label = f"{pl} {out}"
            else:
                label = f"{(pl + ' ') if pl else ''}{out} {float(line):g}" if line is not None else f"{pl} {out}"
            section = MKT_LABEL.get(m, m.replace("player_", "").replace("_", " ").title())
            rows.append(dict(section=section, market=m, player=pl, label=label, prices=prices, fair=fair_all,
                             ev=ev, best=best, best_bet=best_bet, vs_avg=vs_avg))
    order = {"Moneyline": 0, "Spread": 1, "Total": 2}
    rows.sort(key=lambda r: (order.get(r["section"], 3), r["section"], r["player"], r["label"]))
    return rows, (max((r["ts"] for r in q), default=None))


ODDS_CSS = """
:root{box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px);
 --bg:#f4f1ea;--ink:#1c1a17;--ink2:#6d675e;--card:#fffdf8;--line:#e2dcd0;--head:#ece6da;
 --g3:#17663a;--g3b:#cfeed9;--g2:#2b7a4b;--g2b:#e3f4e8;--neg:#9a948a;--gold:#b8860b;--bet:#1f4fa0;--betb:#e5edfa}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
 --bg:#14171b;--ink:#ebe6da;--ink2:#a39e93;--card:#1d2127;--line:#2e343c;--head:#252a31;
 --g3:#7ee0a2;--g3b:#173a26;--g2:#8fd4a8;--g2b:#1a2c22;--neg:#6f7680;--gold:#e8b84a;--bet:#9ab8f2;--betb:#1c2740}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#14171b;--ink:#ebe6da;--ink2:#a39e93;--card:#1d2127;--line:#2e343c;--head:#252a31;
 --g3:#7ee0a2;--g3b:#173a26;--g2:#8fd4a8;--g2b:#1a2c22;--neg:#6f7680;--gold:#e8b84a;--bet:#9ab8f2;--betb:#1c2740}
*,*::before,*::after{box-sizing:inherit}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 -apple-system,system-ui,"Segoe UI",Roboto,sans-serif;font-variant-numeric:tabular-nums}
.wrap{max-width:1500px;margin:0 auto;padding:14px 12px 60px}
a{color:var(--bet)} h1{font:600 26px/1.1 Georgia,serif;margin:4px 0 2px} h2{font:600 17px Georgia,serif;margin:18px 0 6px}
.sub{color:var(--ink2);font-size:12.5px}
.legend{display:flex;flex-wrap:wrap;gap:6px 12px;margin:10px 0 4px;font-size:12px;color:var(--ink2)}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:4px}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:separate;border-spacing:0;font-size:12.5px;min-width:100%}
th,td{padding:5px 6px;text-align:center;border-bottom:1px solid var(--line);white-space:nowrap}
th{background:var(--head);font-weight:600;font-size:11.5px;position:sticky;top:0}
th.bet{color:var(--bet);background:var(--betb)}
td.sel,th.sel{position:sticky;left:0;background:var(--card);text-align:left;z-index:2;max-width:190px;overflow:hidden;text-overflow:ellipsis;border-right:1px solid var(--line)}
th.sel{background:var(--head);z-index:3}
td.fair{color:var(--ink2);border-right:1px solid var(--line)}
td small{display:block;font-size:10px;line-height:1.1;color:var(--ink2)}
td.e3{background:var(--g3b);color:var(--g3);font-weight:700} td.e3 small{color:var(--g3)}
td.e2{background:var(--g2b);color:var(--g2)} td.e2 small{color:var(--g2)}
td.neg{color:var(--neg)}
td.best{box-shadow:inset 0 0 0 2px var(--gold)}
td.bb{font-weight:700} td.bb::after{content:"\u2605";color:var(--bet);font-size:10px;margin-left:2px}
td.na{color:var(--line)}
.games a{display:block;padding:8px 10px;border-bottom:1px solid var(--line);text-decoration:none;color:var(--ink)}
.games a span{color:var(--ink2);font-size:12px} .games{background:var(--card);border:1px solid var(--line);border-radius:8px}
details{margin-top:8px} summary{cursor:pointer;font:600 15px Georgia,serif;padding:6px 0}
"""


def odds_page(q):
    import html as _h
    con = db()
    try:
        eid = q.get("event")
        head = (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content=\"width=device-width,"
                f" initial-scale=1, viewport-fit=cover\"><title>KRONOS odds board</title><style>{ODDS_CSS}</style></head><body><div class=wrap>")
        if not eid:
            now = now_utc()
            evs = con.execute("""select e.*, (select count(distinct book) from last_quote l where l.event_id=e.event_id) nb
                                 from events e where commence>? and commence<? order by sport, commence""",
                              (iso(now - timedelta(hours=3)), iso(now + timedelta(days=UFC_LOOKAHEAD_DAYS)))).fetchall()
            out = [head, "<div class=sub>KRONOS</div><h1>Odds board</h1><div class=sub>Every book for every game, priced against a fair line. "
                   "<a href=\"/\">Back to the desk</a></div>"]
            cur = None
            for e in evs:
                if e["nb"] == 0:
                    continue
                if e["sport"] != cur:
                    if cur is not None:
                        out.append("</div>")
                    cur = e["sport"]; out.append(f"<h2>{_h.escape(cur)}</h2><div class=games>")
                t = parse(e["commence"]).astimezone(ET).strftime("%a %b %d, %-I:%M %p ET")
                out.append(f'<a href="/odds?event={_h.escape(e["event_id"])}">{_h.escape(e["away"])} @ {_h.escape(e["home"])}'
                           f'<br><span>{t} · {e["nb"]} books</span></a>')
            if cur is not None:
                out.append("</div>")
            if cur is None:
                out.append("<p class=sub>No priced games right now.</p>")
            return "".join(out) + "</div></body></html>"
        e = con.execute("select * from events where event_id=?", (eid,)).fetchone()
        if not e:
            return head + "<p>Game not found. <a href=\"/odds\">All games</a></p></div></body></html>"
        rows, ts = odds_grid(con, eid)
        books = _book_order({b for r in rows for b in r["prices"]})
        t = parse(e["commence"]).astimezone(ET).strftime("%a %b %d, %-I:%M %p ET")
        age = ""
        if ts:
            mins = (now_utc() - parse(ts)).total_seconds() / 60
            age = f" · prices updated {int(mins)} min ago" if mins < 180 else f" · prices updated {mins / 60:.1f} h ago"
        out = [head, f'<div class=sub><a href="/odds">All games</a> · {_h.escape(e["sport"])}</div>',
               f"<h1>{_h.escape(e['away'])} @ {_h.escape(e['home'])}</h1><div class=sub>{t}{age} · {len(books)} books</div>",
               "<div class=legend><span><i class=sw style=\"background:var(--g3b);outline:1px solid var(--g3)\"></i>edge +3% or more</span>"
               "<span><i class=sw style=\"background:var(--g2b)\"></i>edge +1% to +3%</span>"
               "<span><i class=sw style=\"box-shadow:inset 0 0 0 2px var(--gold)\"></i>best price anywhere</span>"
               "<span><b style=\"color:var(--bet)\">\u2605</b> best price you can bet (DK / FD / Fanatics)</span>"
               "<span>Fair = all other books de-vigged, Pinnacle and Betfair weighted x3. Exchange prices exclude commission. "
               "Rows marked <i>vs avg</i> (anytime scorer / TD) have no second side to de-vig, so % is against the other books' average.</span></div>"]
        if not rows:
            out.append("<p class=sub>No prices on file for this game yet.</p>")
        sec = None
        for r in rows:
            if r["section"] != sec:
                if sec is not None:
                    out.append("</table></div>" + ("</details>" if sec not in ("Moneyline", "Spread", "Total") else ""))
                sec = r["section"]
                is_prop = r["market"].startswith("player_")
                hdr = "".join(f'<th class="{"bet" if b in BETTABLE else ""}">{BOOK_SHORT.get(b, b[:4].upper())}</th>' for b in books)
                out.append((f"<details><summary>{_h.escape(sec)}</summary>" if is_prop else f"<h2>{_h.escape(sec)}</h2>")
                           + f"<div class=scroll><table><tr><th class=sel>Selection</th><th>Fair</th>{hdr}</tr>")
            fair = r["fair"]
            cells = [f'<td class=sel title="{_h.escape(r["label"])}">{_h.escape(r["label"])}</td>',
                     f"<td class=fair>{_dec(1 / fair) if fair else ('vs avg' if r.get('vs_avg') else '—')}<small>{f'{fair:.1%}' if fair else ''}</small></td>"]
            for b in books:
                p = r["prices"].get(b)
                if p is None:
                    cells.append("<td class=na>·</td>"); continue
                ev = r["ev"].get(b)
                cls = []
                if ev is not None:
                    cls.append("e3" if ev >= 0.03 else "e2" if ev >= 0.01 else "neg" if ev < -0.02 else "")
                if b == r["best"]:
                    cls.append("best")
                if b == r["best_bet"]:
                    cls.append("bb")
                sub = f"{ev:+.1%}" if ev is not None else ""
                cells.append(f'<td class="{" ".join(c for c in cls if c)}">{_dec(p)}<small>{sub}</small></td>')
            out.append("<tr>" + "".join(cells) + "</tr>")
        if sec is not None:
            out.append("</table></div>" + ("</details>" if sec not in ("Moneyline", "Spread", "Total") else ""))
        return "".join(out) + "</div></body></html>"
    finally:
        con.close()

PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>KRONOS desk</title>
<style>
body{margin:0;font:15px Georgia,serif;background:#f4f1ea;color:#1c1a17}
header{padding:16px 16px 4px}h1{font-weight:500;font-size:32px;margin:0}.sub{color:#6d675e;font-size:13px;letter-spacing:.05em}
.bar{padding:4px 16px 10px;color:#6d675e;font-size:13px}
nav{padding:0 16px;position:sticky;top:0;background:#f4f1ea;z-index:5;border-bottom:1px solid #e2dcd0}
nav button{border:0;background:none;font:16px Georgia,serif;margin-right:14px;cursor:pointer;padding:10px 0;color:#2a5db0}
nav button.on{font-weight:700;color:#1c1a17;border-bottom:2px solid #1c1a17}
main{display:grid;grid-template-columns:1fr 320px;gap:18px;padding:12px 16px}
@media(max-width:900px){main{grid-template-columns:1fr}}
.slate{color:#6d675e;font-size:14px;margin:4px 0 12px}
.card{background:#fffdf8;border:1px solid #e2dcd0;border-radius:10px;padding:12px 12px 8px;margin-bottom:12px}
.card.hot{border-color:#b9a77a}
.top{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.when{font-size:13px;color:#6d675e}.title{font-size:18px;margin:2px 0 8px}
.grid{display:grid;grid-template-columns:1.1fr 1fr 1fr 1fr;gap:6px 8px;font-size:15px;align-items:start}
.grid .h{font-size:12px;color:#8a8378;text-transform:uppercase;letter-spacing:.04em}
.grid .team{font-weight:600}
.was{font-size:11px;color:#8a8378}.best{font-size:11px;color:#1d5a3a}
.up{background:#e7f0e4;padding:0 3px;border-radius:3px}.down{background:#f6e4df;padding:0 3px;border-radius:3px}
.call{margin-top:8px;padding:8px 10px;border-radius:8px;background:#eef4ec;font-size:14px}
.call.play{background:#f6ead9}.call b{font-size:15px}.call .note{font-size:12px;color:#6d675e;margin-top:2px}
.call button,.prow button{font:12px Georgia,serif;margin-left:6px;padding:2px 8px;border:1px solid #b9b2a5;background:#fff;border-radius:6px}
details.props{margin-top:8px;border-top:1px solid #eee7da;padding-top:6px}
details.props summary{cursor:pointer;color:#2a5db0;font-size:14px}
.prow{display:flex;justify-content:space-between;gap:8px;padding:6px 0;border-bottom:1px solid #f0ebe1;font-size:14px}
.prow .pl{font-weight:600}.prow .sit{color:#8a8378}
.badge{font-size:12px;padding:1px 6px;border-radius:10px;margin-left:4px}
.value{margin-top:6px;padding:6px 10px;border-radius:8px;background:#fdf6e3;font-size:13px}.value b{color:#8a3b12}
.Steam{color:#8a3b12;font-weight:700}.Sharp{color:#1d5a3a;font-weight:700}.Move{color:#555}
.mute{color:#8a8378}
table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:6px 8px;border-bottom:1px solid #e2dcd0;font-size:14px;vertical-align:top}
form input,form select{font:14px Georgia,serif;margin:2px 0}
</style>
<header><div class="sub">THE SPORTS DESK</div><h1>KRONOS</h1><div class="sub"><a href="/odds">Odds board: every book, every game &rarr;</a></div></header>
<div class="bar" id="bar">loading…</div>
<nav id="nav"></nav>
<main><section>
<div class="slate" id="slate"></div>
<div id="board"></div>
<h3 id="logbet">Log a bet</h3>
<form id="betform" method="post" action="/api/bet">
<select name="event_id" id="ev"></select> <select name="market" id="mk"><option value="h2h">moneyline</option><option value="spreads">spread</option><option value="totals">total</option><option value="method">method of victory (UFC)</option><option value="rounds">rounds over/under (UFC)</option><option value="prop">player prop</option><option value="parlay">parlay</option></select>
<input name="selection" id="sel" list="sellist" placeholder="side / fighter" size=16><datalist id="sellist"></datalist>
<input name="player" id="det" placeholder="detail" size=16><input type=hidden name="sport" id="sp">
<input name="line" id="ln" placeholder="line" size=5> <input name="odds" id="od" placeholder="odds e.g. -110" size=7 required>
<input name="book" placeholder="book" size=9> <input name="stake" value="1" size=3> <input name="model_prob" id="mp" placeholder="model prob" size=8>
<button>Log</button></form>
<form method="post" action="/api/goalie" class=mute>Confirm NHL goalie by hand: <input name="team" placeholder="BOS" size=4> <input name="name" placeholder="Jeremy Swayman" size=16> <button>Set</button> <button formaction="/api/refresh-models" formmethod="post">Re-run models now</button></form>
<h3>Bets</h3><div id="bets"></div>
<h3>Paper ledger <span class=mute style="font-size:13px">every call, auto-logged at 1u, graded vs Pinnacle close</span></h3><div id="paper"></div>
<h3>By bet type</h3><div id="bytype"></div>
<h3>UFC fighters</h3><div id="byfighter"></div>
<p class=mute><a href="/picks">Public picks page →</a></p></section>
<aside><h3>Signals</h3><div id="feed"></div></aside></main>
<details><summary class=mute>Plain-text summary (what Claude reads)</summary><pre id="summary" style="white-space:pre-wrap;font-size:12px">__SUMMARY__</pre></details>
<details><summary class=mute>Fresh-read links for Claude (one per hour, next 14 days)</summary><p class=mute style="font-size:11px">__SLOTS__</p></details>
<script>
const SPORTS=["NFL","EPL","UCL","NBA","NHL","UFC"];let sport="NFL";let BOARD=[],EDGES=[];
const am=d=>d==null?"—":(d>=2?"+"+Math.round((d-1)*100):Math.round(-100/(d-1)));
const t=s=>new Date(s).toLocaleString("en-US", {weekday:"short",month:"short",day:"numeric",hour:"numeric",minute:"2-digit",timeZone:"America/New_York"})+" ET";
const pct=x=>(x*100).toFixed(0)+"%";
const sgn=x=>(x>0?"+":"")+(x*100).toFixed(1)+"%";
const TWO=["Maple Leafs","Golden Knights","Blue Jackets","Red Wings","Red Sox","White Sox","Blue Jays","Trail Blazers"];
function short(n){if(!n)return"";if(sport=="UFC")return n.split(" ").slice(-1)[0];for(const w of TWO){if(n.endsWith(w))return w}return n.split(" ").slice(-1)[0]}
const MK={player_pass_yds:"pass yds",player_rush_yds:"rush yds",player_reception_yds:"rec yds",player_receptions:"receptions",player_anytime_td:"anytime TD",player_pass_tds:"pass TDs"};
const isCall=c=>c&&(c.startsWith("Lean")||c.startsWith("Play"));
function px(x,kind,pre){if(!x)return'<span class=mute>—</span>';
 let l="";if(kind=="s"&&x.line!=null)l=(x.line>0?"+":"")+x.line+" ";if(kind=="t"&&x.line!=null)l=pre+x.line+" ";
 const ch=x.open_price&&x.price!==x.open_price,lch=x.open_line!=null&&x.open_line!==x.line;
 const cls=ch?(1/x.price>1/x.open_price?"up":"down"):"";
 const was=(ch||lch)?`<div class="was">was ${lch?x.open_line+" ":""}${am(x.open_price)}</div>`:"";
 const best=x.best&&x.best>x.price+0.001?`<div class="best">best ${am(x.best)}</div>`:"";
 return `<span class="${cls}">${l}${am(x.price)}</span>${was}${best}`}
function callSel(c,market){const s=c.split(" ").slice(1).join(" ");return market=="totals"?s.split(" ")[0]:s}
function callBox(r,k,v){const market=k.split("|")[0],out=k.split("|").slice(1).join("|");const sel=callSel(v.call,market);
 const mine=sel==out;const p=mine?v.p:1-v.p,f=mine?v.fair:1-v.fair;
 return `<div class="call ${v.call.startsWith("Play")?"play":""}"><b>${v.plain||v.call}</b> <span class=mute>${sgn(p-f)} · model ${pct(p)} vs market ${pct(f)}</span>
 <button onclick="logCall('${r.event_id}','${market}','${sel.replace(/'/g,"")}',${p.toFixed(3)})">Log</button><div class="note">${v.note||""}</div></div>`}
function card(r){const ml=r.h2h||{},sp=r.spreads||{},to=r.totals||{};
 const mv=Object.entries(r.moves||{}).map(([k,n])=>`<span class="badge ${k}">${k} ${n}</span>`).join("");
 const calls=Object.entries(r.model||{}).filter(([k,v])=>isCall(v.call));
 const nprops=EDGES.filter(e=>e.event_id==r.event_id&&e.player&&isCall(e.call)).length;
 let body="";
 if(r.away){const hasS=Object.keys(sp).length>0,hasT=Object.keys(to).length>0;
  body=`<div class="grid"><div class="h"></div><div class="h">Money</div><div class="h">${hasS?"Spread":""}</div><div class="h">${hasT?"Total":""}</div>`;
  [[r.away,"Over","o"],[r.home,"Under","u"]].forEach(([n,tk,pre])=>{body+=`<div class="team">${short(n)}</div><div>${px(ml[n])}</div><div>${hasS?px(sp[n],"s"):""}</div><div>${hasT?px(to[tk],"t",pre):""}</div>`});
  body+="</div>"}
 else{body="<div>"+Object.entries(ml).sort((a,b)=>a[1].price-b[1].price).slice(0,20).map(([n,x])=>`<div class="prow"><span>${n}</span><span>${px(x)}</span></div>`).join("")+"</div>"}
 const title=r.away?`${short(r.away)} @ ${short(r.home)}`:r.home;
 const props=sport=="NFL"&&r.away?`<details class="props" ontoggle="props(this,'${r.event_id}')"><summary>Props${nprops?" · "+nprops+" KRONOS call"+(nprops>1?"s":""):""}</summary><div class=mute>loading…</div></details>`:"";
 const BK={draftkings:"DraftKings",fanduel:"FanDuel",fanatics:"Fanatics",betmgm:"BetMGM",espnbet:"ESPN Bet",betrivers:"BetRivers",williamhill_us:"Caesars"};
 const vals=(r.value||[]).slice(0,4).map(v=>`<div class="value"><b>${BK[v.book]||v.book}</b> ${v.player?v.player+" ":""}${v.market.startsWith("player_")?(MK[v.market]||v.market)+" ":""}${v.market=="spreads"||v.market=="totals"||v.market.startsWith("player_")?short(v.outcome)+" "+(v.line??""):short(v.outcome)} ${am(v.price)} <span class=mute>vs fair ${am(1/v.fair)} · EV ${sgn(v.ev)}</span></div>`).join("");
 return `<div class="card ${calls.length||vals?"hot":""}"><div class="top"><span class="when">${t(r.start)}${r.card?" · "+r.card:""}</span><span>${mv}</span></div>
 <div class="title">${title}</div>${body}${calls.map(([k,v])=>callBox(r,k,v)).join("")}${vals}${props}</div>`}
async function props(el,eid){if(!el.open||el.dataset.done)return;el.dataset.done=1;
 const rows=(await (await fetch("/api/model?event_id="+eid)).json()).filter(x=>x.player&&!String(x.call).startsWith("see"));
 const live=rows.filter(x=>isCall(x.call)).sort((a,b)=>Math.abs(b.edge)-Math.abs(a.edge));
 const rest=rows.filter(x=>!isCall(x.call)).sort((a,b)=>a.player.localeCompare(b.player));
 const line=x=>{const c=isCall(x.call);const sel=c?x.call.split(" ")[1]:"";const mine=sel==x.outcome||!c;const p=mine?x.model_prob:1-x.model_prob,f=mine?x.fair_prob:1-x.fair_prob;
  return `<div class="prow"><span><span class="pl">${x.player}</span> · ${MK[x.market]||x.market}${x.line!=null?" "+x.line:""}<div class="was">${x.note||""}</div></span>
  <span class="${c?"":"sit"}" style="text-align:right;white-space:nowrap">${c?"<b>"+(x.plain||x.call.split(" ").slice(0,2).join(" "))+"</b><br>"+sgn(p-f):(x.plain||x.call)}${c?`<button onclick="logProp('${eid}','${x.player.replace(/'/g,"")}','${sel}',${x.line??"null"},${p.toFixed(3)},'${x.market}')">Log</button>`:""}</span></div>`};
 el.querySelector("div").innerHTML=(live.length?live.map(line).join(""):"<p class=mute>No prop calls for this game.</p>")+
  (rest.length?`<details><summary class=mute>All other props (${rest.length})</summary>${rest.map(line).join("")}</details>`:"")}
function pick(s){sport=s;load()}
function nav(){document.getElementById("nav").innerHTML=SPORTS.map(s=>`<button class="${s==sport?"on":""}" onclick="pick('${s}')">${s}</button>`).join("")}
async function load(){nav();
 const [rows,e]=await Promise.all([fetch("/api/board?sport="+sport).then(x=>x.json()),fetch("/api/edges?min=0&limit=1000").then(x=>x.json())]);
 BOARD=rows.sort((a,b)=>a.start<b.start?-1:1);EDGES=e.filter(x=>x.sport==sport);
 const ncalls=BOARD.reduce((n,r)=>n+Object.values(r.model||{}).filter(v=>isCall(v.call)).length,0);
 const np=EDGES.filter(x=>x.player).length;
 document.getElementById("slate").innerHTML=BOARD.length?`${BOARD.length} ${sport=="UFC"?"fights":"games"} · ${ncalls} KRONOS call${ncalls==1?"":"s"} on sides/totals${np?" · "+np+" on props":""}`:"";
 const open=[...document.querySelectorAll("details.props[open]")].map(d=>d.getAttribute("ontoggle"));
 document.getElementById("board").innerHTML=BOARD.length?BOARD.map(card).join("")+"<p class=mute>Prices are Pinnacle (else sharpest). Green = got more expensive since open, red = cheaper. “was” = opening number.</p>":"<p class=mute>Nothing on the board for "+sport+".</p>";
 document.querySelectorAll("details.props").forEach(d=>{if(open.includes(d.getAttribute("ontoggle")))d.open=true});
 fillForm()}
async function status(){const s=await (await fetch("/api/status")).json();
 document.getElementById("bar").innerHTML=`${s.remaining??"—"} credits left · ${s.allowance_per_day}/day · spent today ${s.spent_today} · reset in ${s.days_to_reset}d`;
 const f=await (await fetch("/api/feed")).json();
 document.getElementById("feed").innerHTML=f.slice(0,40).map(a=>`<div style="padding:4px 0;border-bottom:1px solid #eee7da"><span class="${a.kind}">${a.kind}</span> ${a.sport} ${a.player?a.player+" ":""}${a.outcome} ${a.market}
  ${a.line_from!==a.line_to?(a.line_from+"→"+a.line_to):""} ${am(a.price_from)}→${am(a.price_to)} <span class=mute>${a.book} · ${t(a.ts)}</span></div>`).join("")||"<p class=mute>No signals yet.</p>"}
function fillForm(){const ev=document.getElementById("ev"),mk=document.getElementById("mk"),sel=document.getElementById("sel");
 ev.innerHTML=BOARD.map(r=>`<option value="${r.event_id}">${t(r.start)} ${r.away?short(r.away)+" @ "+short(r.home):r.home}</option>`).join("");
 const upd=()=>{const r=BOARD.find(x=>x.event_id==ev.value)||{};const m=mk.value;
  const opts=(m=="totals"||m=="rounds"||m=="prop")?["Over","Under","Yes"]:m=="parlay"?[]:[r.away,r.home].filter(Boolean);
  document.getElementById("sellist").innerHTML=opts.map(o=>`<option value="${o}">`).join("");
  sel.value=opts[0]||""; document.getElementById("sp").value=sport;
  document.getElementById("det").placeholder={method:"KO/TKO, SUB or DEC",prop:"player name",parlay:"legs, e.g. Evloev ML + Merab ML",rounds:"",h2h:"",spreads:"",totals:""}[m]||"detail"};
 ev.onchange=upd;mk.onchange=upd;upd()}
function prefill(eid,market,sel,line,price,p,player){const ev=document.getElementById("ev"),mk=document.getElementById("mk");
 ev.value=eid;mk.value=market;ev.onchange();document.getElementById("sel").value=sel;document.getElementById("det").value=player||"";
 document.getElementById("ln").value=line??"";document.getElementById("od").value=price?am(price):"";document.getElementById("mp").value=p;
 document.getElementById("logbet").scrollIntoView({behavior:"smooth"})}
function logCall(eid,market,sel,p){const r=BOARD.find(x=>x.event_id==eid)||{};const x=(r[market]||{})[sel]||{};prefill(eid,market,sel,x.line,x.price,p)}
function logProp(eid,player,sel,line,p,market){prefill(eid,"prop",sel,line,null,p,player+" "+(MK[market]||market))}
async function bets(){const r=await (await fetch("/api/report")).json();
 let h=`<p>${r.graded} graded · ${r.pending} pending · ${r.all?("record "+r.all.record+" · "+r.all.units+"u · ROI "+(r.all.roi*100).toFixed(1)+"% (90% CI "+(r.all.roi_90ci[0]*100).toFixed(0)+"% to "+(r.all.roi_90ci[1]*100).toFixed(0)+"%) · CLV "+r.all.clv_pp_avg+"pp"):""}</p><table><tr><th>#</th><th>Game</th><th>Bet</th><th>Price</th><th>Result</th><th>Close / CLV</th></tr>`;
 r.recent.slice().reverse().forEach(b=>{h+=`<tr><td>${b.id}</td><td>${b.sport} ${b.game}</td><td>${b.market} ${b.player||""} ${b.selection} ${b.line??""}</td><td>${am(b.price)}</td>
  <td>${b.result=="pending"?`pending <form method=post action="/api/grade" style="display:inline"><input type=hidden name=id value=${b.id}><button name=result value=W>W</button><button name=result value=L>L</button><button name=result value=P>P</button></form>`:b.result+" "+(b.pnl>0?"+":"")+b.pnl+"u"}</td>
  <td>${b.close_price?am(b.close_price)+" ("+b.close_book+") "+(b.clv_pp??"")+"pp":"—"}</td></tr>`});
 document.getElementById("bets").innerHTML=h+"</table>";
 const bt=(o)=>{const e=Object.entries(o||{});if(!e.length)return"<p class=mute>Nothing graded yet.</p>";
  return "<table><tr><th></th><th>Bets</th><th>Record</th><th>Units</th><th>ROI</th><th>CLV</th></tr>"+e.map(([k,v])=>`<tr><td>${k}</td><td>${v.n}</td><td>${v.record}</td><td>${v.units>0?"+":""}${v.units}</td><td>${(v.roi*100).toFixed(1)}%</td><td>${v.clv_pp_avg??"—"}</td></tr>`).join("")+"</table>"};
 document.getElementById("bytype").innerHTML=bt(r.by_type);
 document.getElementById("byfighter").innerHTML="<p class=mute>Backing</p>"+bt(r.by_fighter)+"<p class=mute>Betting against</p>"+bt(r.against_fighter)}
async function paper(){const r=await (await fetch("/api/paper")).json();const e=Object.entries(r.by_group||{});
 document.getElementById("paper").innerHTML=(r.all?`<p>${r.total} paper bets · ${r.all.closed} with a closing line · CLV ${r.all.clv_pp_avg??"—"}pp · beat close ${r.all.beat_close_pct!=null?(r.all.beat_close_pct*100).toFixed(0)+"%":"—"} · ${r.all.graded} graded ${r.all.record} (${r.all.units>0?"+":""}${r.all.units}u)</p>`:"<p class=mute>No paper bets yet. They appear as soon as the models make a call.</p>")+
 (e.length?"<table><tr><th>Group</th><th>Bets</th><th>CLV</th><th>Beat close</th><th>Record</th><th>Units</th></tr>"+e.map(([k,v])=>`<tr><td>${k}</td><td>${v.n}</td><td>${v.clv_pp_avg??"—"}</td><td>${v.beat_close_pct!=null?(v.beat_close_pct*100).toFixed(0)+"%":"—"}</td><td>${v.graded?v.record:"—"}</td><td>${v.graded?(v.units>0?"+":"")+v.units:"—"}</td></tr>`).join("")+"</table>":"")}
load();status();bets();paper();setInterval(status,30000);setInterval(load,90000);setInterval(bets,60000);setInterval(paper,300000);
</script>"""


def picks_page(rp, desk):
    import html as h
    con = db()
    rows = [dict(r) for r in con.execute("select * from bets where coalesce(paper,0)=0 order by commence desc, id desc limit 200")]
    con.close()
    a = rp.get("all") or {}
    def amer(d):
        return "—" if not d else (f"+{round((d-1)*100)}" if d >= 2 else f"{round(-100/(d-1))}")
    def line(b):
        typ = bet_type(b)
        if b["market"] == "parlay":
            return f"Parlay: {h.escape(b['player'] or b['selection'] or '')}"
        if b["market"] == "method":
            return f"{h.escape(b['selection'])} by {h.escape(b['player'] or '?')}"
        if b["market"] in ("totals", "rounds"):
            return f"{h.escape(b['selection'])} {b['line'] if b['line'] is not None else ''} {'rounds' if b['market']=='rounds' else ''}"
        if b["market"] == "prop":
            return f"{h.escape(b['player'] or '')} {h.escape(b['selection'])} {b['line'] if b['line'] is not None else ''}"
        sp = f" {b['line']:+g}" if b["market"] == "spreads" and b["line"] is not None else ""
        return f"{h.escape(b['selection'])}{sp} ({typ.lower()})"
    open_ = [b for b in rows if b["result"] == "pending"]
    done = [b for b in rows if b["result"] != "pending"]
    def tr(b):
        res = {"W": "<b class=w>WIN</b>", "L": "<b class=l>LOSS</b>", "P": "PUSH"}.get(b["result"], "pending")
        u = "" if b["pnl"] is None else f"{b['pnl']:+.2f}u"
        return (f"<tr><td>{h.escape(b['sport'])}</td><td>{h.escape(b['game'] or '')}</td><td>{line(b)}</td>"
                f"<td>{amer(b['price'])}</td><td>{b['stake']:g}u</td><td>{res} {u}</td></tr>")
    head = "<tr><th>Sport</th><th>Event</th><th>Pick</th><th>Odds</th><th>Risk</th><th>Result</th></tr>"
    rec = (f"{a.get('record','0-0-0')} &middot; {a.get('units',0):+}u &middot; ROI {((a.get('roi') or 0)*100):.1f}%"
           if a else "No graded picks yet")
    return f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1"><title>Jerry picks</title>
<style>body{{margin:0;font:16px Georgia,serif;background:#f4f1ea;color:#1c1a17}}main{{max-width:860px;margin:auto;padding:24px}}
h1{{font-weight:500;font-size:40px;margin:0}}.sub{{color:#6d675e}}table{{width:100%;border-collapse:collapse;margin:8px 0 24px}}
td,th{{text-align:left;padding:8px;border-bottom:1px solid #e2dcd0;font-size:15px}}.w{{color:#1d5a3a}}.l{{color:#8a3b12}}
.big{{font-size:22px;margin:12px 0 18px}}</style>
<main><div class=sub>THE SPORTS DESK</div><h1>KRONOS picks</h1><div class=big>{rec}</div>
<h3>Open picks</h3><table>{head}{''.join(tr(b) for b in open_) or '<tr><td colspan=6 class=sub>No open picks.</td></tr>'}</table>
<h3>Results</h3><table>{head}{''.join(tr(b) for b in done) or '<tr><td colspan=6 class=sub>Nothing graded yet.</td></tr>'}</table>
<p class=sub>Every pick is logged before the event starts and graded automatically or by result. Units are risk amounts. For entertainment; bet responsibly.</p></main>"""

DESK = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        if not self.authed():
            return
        n = int(self.headers.get("Content-Length", 0) or 0)
        f = dict(urllib.parse.parse_qsl(self.rfile.read(n).decode()))
        u = urllib.parse.urlparse(self.path)
        try:
            if u.path == "/api/bet":
                DESK.add_bet(f, now_utc())
            elif u.path == "/api/grade":
                DESK.grade_manual(int(f["id"]), f.get("result"))
            elif u.path == "/api/goalie":
                DESK.set_goalie(f["team"], f["name"])
            elif u.path == "/api/refresh-models":
                threading.Thread(target=DESK.refresh_models, daemon=True).start()
        except Exception as e:
            return self.send(f"error: {e}", "text/plain")
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def send(self, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def authed(self):
        pw = os.environ.get("DESK_PASSWORD")
        if not pw:
            return True
        import base64
        got = self.headers.get("Authorization", "")
        if got.startswith("Basic "):
            try:
                if base64.b64decode(got[6:]).decode().split(":", 1)[1] == pw:
                    return True
            except Exception:
                pass
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="KRONOS"')
        self.end_headers()
        return False

    def do_GET(self):   # reads are public so Claude can fetch the page; writes (POST) need DESK_PASSWORD
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        if u.path in ("/export/odds_snapshots.csv", "/export/odds_usage.csv"):
            p = os.path.join(DATA, os.path.basename(u.path))
            return self.send(open(p, "rb").read() if os.path.exists(p) else b"", "text/csv")
        if u.path == "/health":
            return self.send("ok", "text/plain")
        if u.path == "/":
            import html as _h
            try:
                summ = DESK.summary_text()
            except Exception as e:
                summ = f"summary error: {e}"
            base = f"https://{self.headers.get('Host', 'localhost')}"
            t0 = now_utc().replace(minute=0, second=0, microsecond=0)
            slots = " ".join(f'<a href="{base}/s/{(t0 + timedelta(hours=i)).strftime("%Y%m%d%H")}">{(t0 + timedelta(hours=i)).strftime("%Y%m%d%H")}</a>'
                             for i in range(24 * 14))
            return self.send(PAGE.replace("__SUMMARY__", _h.escape(summ)).replace("__SLOTS__", slots), "text/html; charset=utf-8")
        if u.path == "/picks":
            return self.send(picks_page(DESK.report(), DESK), "text/html; charset=utf-8")
        if u.path == "/odds":
            return self.send(odds_page(q), "text/html; charset=utf-8")
        if u.path.startswith("/s/"):   # hourly fresh-read slots (each address is new, so no stale cache)
            return self.send(DESK.summary_text(), "text/plain; charset=utf-8")
        if u.path == "/summary":
            return self.send(DESK.summary_text(), "text/plain; charset=utf-8")
        if u.path == "/api/report":
            return self.send(json.dumps(DESK.report()))
        if u.path == "/api/paper":
            return self.send(json.dumps(DESK.paper_report()))
        if u.path == "/api/quarantine":
            con = db()
            try:
                rows = [dict(r) for r in con.execute(
                    "select * from grade_quarantine order by seen_ts desc limit 200")]
            except sqlite3.Error:
                rows = []
            con.close()
            return self.send(json.dumps({"quarantined": rows}))
        if u.path == "/api/edges":
            return self.send(json.dumps(DESK.edges(float(q.get("min", 0.015)), int(q.get("limit", 60)))))
        if u.path == "/api/model":
            return self.send(json.dumps(DESK.model_rows(q.get("event_id", ""))))
        if u.path == "/api/status":
            return self.send(json.dumps(DESK.status()))
        if u.path == "/api/board":
            return self.send(json.dumps(DESK.board(q.get("sport", "NFL"))))
        if u.path == "/api/feed":
            return self.send(json.dumps(DESK.feed()))
        self.send_response(404)
        self.end_headers()


def run_loop(desk):
    while True:
        try:
            desk.tick()
        except Exception as e:
            print(f"[desk] tick error: {e}", flush=True)
            con = db()
            meta_set(con, "error_loop", f"{iso(now_utc())} {e}"[:200])
            con.commit()
            con.close()
        time.sleep(60)


def model_loop(desk):
    time.sleep(90)   # let the first odds pulls land
    while True:
        desk.refresh_models()
        time.sleep(3600)


def main():
    global DESK
    init_db()
    DESK = Desk()
    if load_key():
        threading.Thread(target=run_loop, args=(DESK,), daemon=True).start()
        threading.Thread(target=model_loop, args=(DESK,), daemon=True).start()
    else:   # keep serving so the dashboard/logs explain the problem instead of a crash loop
        print("ODDS_API_KEY is not set — dashboard is up but NOTHING is being pulled. Add the variable and redeploy.")
        con = db(); meta_set(con, "error_key", "ODDS_API_KEY not set"); con.commit(); con.close()
    cloud = bool(os.environ.get("PORT"))            # Railway / Render set PORT
    port = int(os.environ.get("PORT", PORT))
    host = "0.0.0.0" if cloud else "127.0.0.1"
    server = ThreadingHTTPServer((host, port), Handler)
    if not cloud and "--no-browser" not in os.sys.argv:
        webbrowser.open(f"http://127.0.0.1:{port}")
    if cloud and not os.environ.get("DESK_PASSWORD"):
        print("WARNING: no DESK_PASSWORD — anyone with the link can log or grade bets. Viewing is always public.")
    print(f"KRONOS desk on {host}:{port}, data in {STORE} — the only program that should use the Odds API key.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
