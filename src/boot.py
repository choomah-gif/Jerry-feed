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
        for market in ("h2h", "spreads", "totals"):
            outs = [(o, v) for (m, o), v in q.items() if m == market]
            if len(outs) != 2: continue
            inv = sum(1 / v[1] for _, v in outs)
            calc = []
            for o, (line, price) in outs:
                fair = (1 / price) / inv
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
                if o == best[0]:
                    call = f"Lean {o}" if edge >= 0.02 else "Sit"
                else:
                    call = "see other row"
                note = base + (f"; total {mu_h+mu_a:.1f} vs {line}" if market == "totals" else "")
                rows.append((eid, market, o, "", None if line is None else float(line), float(mp), float(fair), float(edge), call, note, now.isoformat()))
        con.executemany("insert or replace into model_probs values(?,?,?,?,?,?,?,?,?,?,?)", rows)
        con.commit(); n += len(rows)
    con.close()
    return n

# ---------- hook into the desk and start it
import os
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
    try:  # NHL quarantine: model output failed sanity checks on 2026-10-24
        c = sqlite3.connect(db_path, timeout=60)
        c.execute("""update model_probs set call='Sit (NHL under repair)'
                     where event_id in (select event_id from events where sport='NHL') and call not like 'see %'""")
        c.commit(); c.close()
    except Exception:
        traceback.print_exc()
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
    return res
if L:
    L.refresh_all = refresh_all
    print("[boot] fixes active: NHL icetime alias, MLB model", flush=True)
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
    for f in ("jerry.py", "jerry_live.py", "jerry_desk.py", "boot.py"):
        fp = os.path.join(here, f)
        if os.path.exists(fp):
            _put(f"src/{f}", open(fp, encoding="utf-8").read(), token, f"source {f}")
            done.append(f)
    return f"source published: {done}"

def push_feed():
    token = os.environ.get("FEED_TOKEN")
    if not token:
        return "no FEED_TOKEN set"
    txt = _summary_text()
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
