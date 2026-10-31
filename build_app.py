#!/usr/bin/env python3
"""
Rebuilds the Premier League predictor page from the latest football-data.co.uk
results and fixtures. Run it on a schedule (see .github/workflows/update.yml).

    python build_app.py --out site/index.html

New seasons are handled automatically: the season code is worked out from today's
date, promoted clubs start at league-average ratings, and the page never shows a
half-built result (a failed download leaves the previous page untouched).
"""
import argparse, datetime as dt, html, json, os, sys, time, urllib.error, urllib.request
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

BASE = "https://www.football-data.co.uk"
DISPLAY = {"Man City": "Manchester City", "Man United": "Manchester United", "Nott'm Forest": "Nottingham Forest"}
K_GOALS, K_CORNERS, N_SEASONS, R_DEFAULT = 0.02, 0.005, 5, 8.8


def disp(name):
    return DISPLAY.get(name, name).replace("'", "\u2019").strip()


class Ratings:
    """Attack/defence multipliers per team, updated after every match (log-likelihood step)."""
    def __init__(self, k):
        self.k, self.att, self.dfn = k, {}, {}
        self.n, self.sh, self.sa = 0, 0.0, 0.0

    @property
    def avg_h(self): return self.sh / self.n if self.n else 1.5
    @property
    def avg_a(self): return self.sa / self.n if self.n else 1.2

    def rates(self, h, a):
        g = lambda d, t: d.get(t, 1.0)
        return self.avg_h * g(self.att, h) * g(self.dfn, a), self.avg_a * g(self.att, a) * g(self.dfn, h)

    def update(self, h, a, hg, ag):
        lh, la = self.rates(h, a)
        eh, ea = self.k * (hg - lh), self.k * (ag - la)
        self.att[h] = self.att.get(h, 1.0) * np.exp(eh); self.dfn[a] = self.dfn.get(a, 1.0) * np.exp(eh)
        self.att[a] = self.att.get(a, 1.0) * np.exp(ea); self.dfn[h] = self.dfn.get(h, 1.0) * np.exp(ea)
        self.n += 1; self.sh += hg; self.sa += ag


def season_codes(today):
    y = today.year if today.month >= 7 else today.year - 1
    return [f"{(y - i) % 100:02d}{(y - i + 1) % 100:02d}" for i in range(N_SEASONS - 1, -1, -1)]


def fetch(url, dest, retries=3):
    """Download a CSV. Returns 'ok' or 'missing' (HTTP 404). Raises on any other failure."""
    err = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (football-predictor build)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
            if "HomeTeam" not in raw.decode("latin-1").split("\n")[0]:
                raise ValueError("does not look like a results CSV")
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            open(dest, "wb").write(raw)
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404: return "missing"
            err = e
        except Exception as e:
            err = e
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"could not download {url}: {err}")


def read_results(path, code):
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = [c.strip().lstrip("\ufeff").replace("ï»¿", "") for c in df.columns]
    need = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    if df.empty or any(c not in df.columns for c in need):
        return pd.DataFrame(columns=need + ["HC", "AC", "season"])
    df = df.dropna(subset=need).copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    for c in ("HC", "AC"):
        if c not in df.columns: df[c] = np.nan
    df["season"] = code
    return df[need + ["HC", "AC", "season"]]


def r3(a, b, c):
    v = [a * 100, b * 100, c * 100]; f = [int(x) for x in v]
    for i in sorted(range(3), key=lambda i: -(v[i] - f[i]))[:100 - sum(f)]: f[i] += 1
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/index.html")
    ap.add_argument("--template", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html"))
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--base", default=BASE, help="override the data site (used for testing)")
    ap.add_argument("--today", help="YYYY-MM-DD, pretend it is this date (used for testing)")
    a = ap.parse_args()
    today = dt.date.fromisoformat(a.today) if a.today else dt.date.today()
    codes = season_codes(today); cur = codes[-1]

    # ---- 1. download (any failure other than "new season file not published yet" stops the build)
    frames = {}
    for code in codes:
        dest = os.path.join(a.data_dir, f"E0_{code}.csv")
        try:
            status = fetch(f"{a.base}/mmz4281/{code}/E0.csv", dest)
        except RuntimeError as e:
            if code == cur or not os.path.exists(dest): raise
            print("WARNING:", e, "- using the saved copy"); status = "ok"
        if status == "ok": frames[code] = read_results(dest, code)
        else: print(f"note: season {code} not published yet"); frames[code] = pd.DataFrame()
    fx_path = os.path.join(a.data_dir, "fixtures.csv")
    if fetch(f"{a.base}/fixtures.csv", fx_path) != "ok":
        sys.exit("fixtures.csv was not found, keeping the previous page")

    res = pd.concat([f for f in frames.values() if len(f)], ignore_index=True)
    if len(res) < 200: sys.exit("not enough results downloaded")
    res = res.sort_values("Date", kind="stable").reset_index(drop=True)
    cur_rows = res[res.season == cur]
    prev_codes = [c for c in codes if c != cur and len(frames.get(c, []))]

    # ---- 2. fit the goal and corner models
    gm, cm = Ratings(K_GOALS), Ratings(K_CORNERS); mu_c, y_c = [], []
    for r in res.itertuples():
        gm.update(r.HomeTeam, r.AwayTeam, r.FTHG, r.FTAG)
        if not (np.isnan(r.HC) or np.isnan(r.AC)):
            mu_c.append(cm.rates(r.HomeTeam, r.AwayTeam)); y_c.append((r.HC, r.AC))
            cm.update(r.HomeTeam, r.AwayTeam, r.HC, r.AC)
    R = R_DEFAULT
    w = min(380, len(mu_c) // 3)
    if w >= 50:
        mu_w, y_w = np.array(mu_c[:w]), np.array(y_c[:w], float)
        v = ((y_w - mu_w) ** 2).mean() - mu_w.mean()
        if v > 0: R = float(np.clip(round((mu_w ** 2).mean() / v, 1), 2.0, 50.0))
    asof = res.Date.max()

    # ---- 3. which clubs are in the league now, and the upcoming gameweek
    fx = pd.read_csv(fx_path, encoding="latin-1")
    fx.columns = [c.strip().lstrip("\ufeff").replace("ï»¿", "") for c in fx.columns]
    fx = fx[fx["Div"] == "E0"].dropna(subset=["HomeTeam", "AwayTeam"]).copy() if "Div" in fx.columns else fx.iloc[0:0]
    if len(fx):
        fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, format="mixed")
        fx = fx[fx.Date.dt.date >= today]
        if "Time" not in fx.columns: fx["Time"] = ""
        fx = fx.sort_values(["Date", "Time"], kind="stable")
    played = set(zip(cur_rows.HomeTeam, cur_rows.AwayTeam))
    fx = fx[[(h, t) not in played for h, t in zip(fx.HomeTeam, fx.AwayTeam)]] if len(fx) else fx
    picked, used = [], set()
    if len(fx):
        d0 = fx.Date.iloc[0]
        for r in fx.itertuples():
            if r.Date > d0 + pd.Timedelta(days=3) or len(picked) >= 10: break
            if r.HomeTeam in used or r.AwayTeam in used: continue
            picked.append(r); used |= {r.HomeTeam, r.AwayTeam}
    fx_teams = set(fx.HomeTeam) | set(fx.AwayTeam) if len(fx) else set()
    if len(cur_rows) >= 20: teams = set(cur_rows.HomeTeam) | set(cur_rows.AwayTeam) | fx_teams
    elif len(fx_teams) >= 18: teams = fx_teams                      # new season, before enough matches are played
    else:
        last = res[res.season == prev_codes[-1]]; teams = set(last.HomeTeam) | set(last.AwayTeam) | fx_teams
    if len(teams) < 18: sys.exit(f"only {len(teams)} clubs found, refusing to publish")

    # ---- 4. league table (this season; before matches are played, last season's)
    basis = cur_rows if len(cur_rows) else res[res.season == prev_codes[-1]]
    tab = {}
    for r in basis.itertuples():
        for t, gf, ga in ((r.HomeTeam, r.FTHG, r.FTAG), (r.AwayTeam, r.FTAG, r.FTHG)):
            x = tab.setdefault(t, [0, 0, 0, 0, 0, 0]); x[0] += 1; x[4] += int(gf); x[5] += int(ga)
            x[1 if gf > ga else 2 if gf == ga else 3] += 1
    rows = {t: tab[t] for t in teams if t in tab}
    newcomers = [t for t in teams if t not in rows]
    for t in newcomers: rows[t] = [1, 0, 1, 0, 1, 1]               # neutral placeholder so the club can be picked
    snap_rows = sorted(((3 * v[1] + v[2], v[4] - v[5], v[4], disp(t), v) for t, v in rows.items()), reverse=True)
    snap = ";".join(f"{n} " + " ".join(map(str, v)) for _, _, _, n, v in snap_rows)
    zone = lambda i, n: "zb" if i < 4 else "zo" if i == 4 else "zr" if i >= n - 3 else ""
    trs = '<div class="tr th"><span></span><span>#</span><span class="tn">Team</span><span>P</span><span>W</span><span>D</span><span>L</span><span>F</span><span>A</span><span>GD</span><span>PTS</span></div>'
    for i, (pts, gd, _, n, v) in enumerate(snap_rows):
        trs += f'<div class="tr"><span class="zbar {zone(i, len(snap_rows))}"></span><span>{i+1}</span><span class="tn">{html.escape(n)}</span>' + "".join(f"<span>{x}</span>" for x in v[:6]) + f"<span>{gd}</span><b>{pts}</b></div>"
    static_table = ('<details class="tw" open><summary>League table</summary><div class="tbl" role="table" aria-label="League table">' + trs +
                    '</div>\n  <p class="hint">Blue: Champions League places. Orange: Europa League place. Red: relegation places. Zones are approximate.</p></details>')
    played_counts = [v[0] for t, v in rows.items() if t not in newcomers]
    mw = (min(played_counts) + 1) if (len(cur_rows) and played_counts) else 1

    # ---- 5. model data for the page, and the pre-computed fixture cards
    K = {"L": float(np.sqrt(cm.avg_h * cm.avg_a)), "HA": float(np.sqrt(cm.avg_h / cm.avg_a)), "r": R}
    gt = lambda t: [round(gm.att.get(t, 1.0), 6), round(gm.dfn.get(t, 1.0), 6), round(cm.att.get(t, 1.0), 6), round(cm.dfn.get(t, 1.0), 6)]
    fixtures = [[disp(r.HomeTeam), disp(r.AwayTeam)] for r in picked if r.HomeTeam in teams and r.AwayTeam in teams]
    model = {"pl": {"L": float(np.sqrt(gm.avg_h * gm.avg_a)), "HA": float(np.sqrt(gm.avg_h / gm.avg_a)), "n": len(res),
                    "asOf": asof.strftime("%d %b %Y"), "teams": {disp(t): gt(t) for t in sorted(teams)}, "fixtures": fixtures, "K": K}}
    raw = {disp(t): t for t in teams}

    def card(hn, an):
        h, t = raw[hn], raw[an]
        lh, la = (max(x, .05) for x in gm.rates(h, t)); g = np.arange(11)
        m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la)); m /= m.sum()
        p = r3(float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum()))
        i, j = np.unravel_index(m.argmax(), m.shape)
        over = round(float(sum(m[q, s] for q in range(11) for s in range(11) if q + s > 2)) * 100); btts = round(float(m[1:, 1:].sum()) * 100)
        tot = lh + la; none = float(np.exp(-tot)); F = r3(lh / tot * (1 - none), none, la / tot * (1 - none))
        ch, ca = (max(x, .5) for x in cm.rates(h, t)); n = np.arange(61)
        tt = np.convolve(nbinom.pmf(n, R, R / (R + ch)), nbinom.pmf(n, R, R / (R + ca)))[:61]
        e = html.escape
        return (f'<div class="card sc"><p class="fixture">{e(hn)} <span>vs</span> {e(an)}</p>'
                f'<div class="bar"><div class="sh" style="width:{p[0]}%">{p[0]}%</div><div class="sd" style="width:{p[1]}%">{p[1]}%</div><div class="sa" style="width:{p[2]}%">{p[2]}%</div></div>'
                f'<div class="legend"><span>{e(hn)} win</span><span>Draw</span><span>{e(an)} win</span></div>'
                f'<p class="sm">Most likely score {i}\u2013{j} &middot; Expected goals {lh:.2f}&ndash;{la:.2f} &middot; Over 2.5: {over}% &middot; Both score: {btts}%</p>'
                f'<p class="sm">Scores first: {e(hn)} {F[0]}% &middot; none {F[1]}% &middot; {e(an)} {F[2]}%</p>'
                f'<p class="sm">Expected corners {ch:.1f}&ndash;{ca:.1f} (total {ch+ca:.1f}) &middot; Over 9.5 corners: {round(float(tt[10:].sum())*100)}%</p></div>')
    cards = "".join(card(h, t) for h, t in fixtures) or '<p class="hint">No upcoming fixtures are listed yet.</p>'
    if picked:
        d1, d2 = picked[0].Date, picked[-1].Date
        span = d1.strftime("%d %b") if d1 == d2 else f"{d1.strftime('%d %b')}\u2013{d2.strftime('%d %b')}"
        title = f"Upcoming fixtures (Matchweek {mw}, {span})"
    else: title = "Upcoming fixtures"

    out = (open(a.template, encoding="utf-8").read()
           .replace("__MODEL__", json.dumps(model, ensure_ascii=False)).replace("__SNAP__", json.dumps(snap, ensure_ascii=False))
           .replace("__FIXTURE_TITLE__", title).replace("__FIXTURE_CARDS__", cards).replace("__STATIC_TABLE__", static_table)
           .replace("__N__", str(len(res))).replace("__ASOF__", model["pl"]["asOf"]))
    left = [t for t in ("__MODEL__", "__SNAP__", "__FIXTURE", "__STATIC_TABLE__", "__N__", "__ASOF__") if t in out]
    if left: sys.exit(f"template placeholders left unfilled: {left}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"; open(tmp, "w", encoding="utf-8").write(out); os.replace(tmp, a.out)
    print(f"built {a.out}: {len(res)} matches to {model['pl']['asOf']} | season {cur} | {len(teams)} clubs "
          f"({len(newcomers)} without history) | matchweek {mw} | {len(fixtures)} fixtures | corner dispersion r={R}")


if __name__ == "__main__":
    main()
