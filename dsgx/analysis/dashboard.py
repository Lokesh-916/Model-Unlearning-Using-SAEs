"""Panel dashboard: ONE self-contained HTML page with the live status of both machines, the claims table and the
key figures, regenerated from summary.json by one command.

    python -m dsgx.analysis.dashboard                  # refresh summaries + write results/dashboard.html
    python -m dsgx.analysis.dashboard --no-refresh     # only re-render from the existing summary.json files
    python -m dsgx.analysis.dashboard --every 15       # regenerate every 15 min (the page reloads itself)

Sources (all read-only; nothing is written to $DSG_RESULTS or the server):
  * lab PC queue: dsgx.queue.status (job states, progress, ETA; MOVED-TO-SERVER jobs listed separately);
  * gpuws: `squeue`, GPU and disk via `cluster/server.sh status` (read-only ssh; "unreachable" if it fails),
    plus the fetched job summaries in dsg_results_cluster/jobs/*/summary.json;
  * claims and figures: dsgx.analysis.final_report --interim, one report per hardware label (lab PC runs and
    gpuws runs are never mixed), written to results/dashboard_data/<hardware>/ (summary.json, figures/*.png);
    the Q2 attribution panel (TOFU, fictitious) if fetched.
Only ids, statuses and aggregate metrics are shown; no prompt, question or generation is read ($DSG_PRIVATE is
never opened). Output: <repo>/results/dashboard.html (figures embedded as data URIs; < 16 MB).
"""
import argparse
import base64
import html
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from dsgx import paths

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
CLUSTER = PROJECT / "dsg_results_cluster"
OUT = REPO / "results"
KEY_FIGURES = 10        # per hardware
MAX_FIG_BYTES = 900_000


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


# ----------------------------------------------------------------------------- data
def lab_status() -> dict:
    try:
        from dsgx.queue import common as q
        from dsgx.queue.status import collect, overall
    except Exception as e:  # pragma: no cover - harness missing
        return {"error": f"{type(e).__name__}: {e}"}
    try:
        jobs, states, rows, now = collect()
    except Exception as e:
        return {"error": f"queue not readable ({type(e).__name__})"}
    counts, pct, remaining = overall(rows)
    running, failed, moved, waves = [], [], [], Counter()
    for r in rows:
        s, j = r["st"]["status"], r["job"]
        if s == q.RUNNING:
            p = r["prog"] or {}
            frac = (p.get("items_done") or 0) / p["items_total"] if p.get("items_total") else None
            running.append({"id": r["id"], "exp": j.get("exp_id"), "frac": frac, "metric": p.get("current_metric")})
        elif s in (q.FAILED, q.BLOCKED):
            failed.append({"id": r["id"], "exp": j.get("exp_id"), "status": s})
        elif s == q.MOVED:
            moved.append(j.get("exp_id"))
        if s not in (q.DONE, q.MOVED):
            waves[j.get("wave", 0)] += 1
    by_exp = {}
    for r in rows:
        e = by_exp.setdefault(r["job"].get("exp_id"), Counter())
        e[r["st"]["status"]] += 1
    return {"counts": dict(counts), "pct": pct, "remaining_min": remaining, "running": running, "failed": failed,
            "moved": dict(Counter(moved)), "open_by_wave": dict(sorted(waves.items())), "n_jobs": len(rows),
            "by_exp": {k: dict(v) for k, v in sorted(by_exp.items(), key=lambda kv: str(kv[0]))}}


def server_status(timeout=40) -> dict:
    try:
        r = subprocess.run([str(REPO / "cluster/server.sh"), "status"], capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        return {"error": f"unreachable ({type(e).__name__})"}
    if r.returncode != 0:
        return {"error": "unreachable (ssh failed)"}
    jobs, gpu, disk, ours, sec = [], None, None, None, None
    for line in r.stdout.splitlines():
        if line.startswith("---"):
            sec = line.strip("- ").strip()
            continue
        f = line.split()
        if sec == "our jobs" and f and f[0].isdigit():
            jobs.append({"id": f[0], "name": f[1], "state": f[2], "time": f[3], "limit": f[4], "reason": " ".join(f[5:])})
        elif sec == "GPU" and line.strip():
            gpu = line.strip()
        elif sec == "disk" and f:
            if line.startswith("/"):
                disk = {"size": f[1], "used": f[2], "free": f[3], "pct": f[4]}
            else:
                ours = f[0]
    return {"jobs": jobs, "gpu": gpu, "disk": disk, "ours": ours}


def cluster_jobs() -> list[dict]:
    out = []
    for p in sorted((CLUSTER / "jobs").glob("*/summary.json")):
        try:
            s = json.loads(p.read_text())
        except ValueError:
            continue
        head = {}
        sel = s.get("selected") or s.get("selection")
        if isinstance(sel, dict):
            head["selected"] = sel.get("cfg", sel.get("selected_alpha"))
            if sel.get("at_grid_edge"):
                head["grid edge"] = ", ".join(sel["at_grid_edge"])
        for k in ("finished", "complete", "n_items", "corpora"):
            if k in s:
                head[k] = s[k]
        out.append({"job": p.parent.name, "time": s.get("time"), "hardware": (s.get("hardware") or {}).get("label")
                    if isinstance(s.get("hardware"), dict) else s.get("hardware"), "head": head,
                    "fetched": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))})
    return out


def refresh_reports(n_boot: int) -> dict:
    """final_report --interim per hardware into results/dashboard_data/<hw> (read-only on runs)."""
    done = {}
    for hw, runs in (("labpc", None), ("gpuws", CLUSTER / "runs")):
        out = OUT / "dashboard_data" / hw
        args = [sys.executable, "-m", "dsgx.analysis.final_report", "--interim", "--hardware", hw, "--out", str(out),
                "--n-boot", str(n_boot)] + (["--runs", str(runs)] if runs else [])
        if runs is not None and not runs.exists():
            done[hw] = "no runs"
            continue
        r = subprocess.run(args, capture_output=True, text=True, cwd=REPO)
        done[hw] = "ok" if r.returncode == 0 else f"failed: {(r.stderr or r.stdout).strip().splitlines()[-1:]}"
    return done


def load_summary(hw) -> dict | None:
    p = OUT / "dashboard_data" / hw / "summary.json"
    try:
        s = json.loads(p.read_text())
        s["_mtime"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
        return s
    except (OSError, ValueError):
        return None


def figures_for(hw) -> list[tuple[str, str]]:
    d = OUT / "dashboard_data" / hw / "figures"
    figs = []
    for p in sorted(d.glob("*.png")) if d.exists() else []:
        if p.stat().st_size <= MAX_FIG_BYTES:
            figs.append((p.stem.replace("_", " "), p))
    extra = CLUSTER / "runs" / "Q2-graphs" / "tofu" / "figures" / "panel_fact0.png"
    if hw == "gpuws" and extra.exists():
        figs.insert(0, ("Q2 attribution graphs (TOFU fact): base, DSG, D2, French attack", extra))
    return [(t, "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()) for t, p in figs[:KEY_FIGURES]]


def fixed_claims() -> list[dict]:
    from dsgx.analysis import claims

    return [{"id": k, "claim": v, "criterion": claims.CRITERIA.get(k), "verdict": "Pending"} for k, v in claims.CLAIMS.items()]


# ----------------------------------------------------------------------------- render
CSS = """
/* Layout: an instrument status board - summary strip, two machine columns, then claims and figures. */
:root {
  --bg: #f3f5f4; --panel: #ffffff; --ink: #18211e; --muted: #5c6b66; --line: #d6dedb; --accent: #0f6e63;
  --ok: #1d7a46; --warn: #a3620a; --bad: #b42d2d; --idle: #6b7a86;
  --ok-bg: #e2f2e8; --warn-bg: #fbefdc; --bad-bg: #f8e1e1; --idle-bg: #e7ecef;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111816; --panel: #18211e; --ink: #e3ebe8; --muted: #93a39e; --line: #2b3834; --accent: #4fbfae;
  --ok: #5fcf8b; --warn: #e3a54a; --bad: #ef7d7d; --idle: #9aa9b4;
  --ok-bg: #173325; --warn-bg: #3a2b14; --bad-bg: #3d1d1d; --idle-bg: #232d33; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #111816; --panel: #18211e; --ink: #e3ebe8; --muted: #93a39e; --line: #2b3834; --accent: #4fbfae;
  --ok: #5fcf8b; --warn: #e3a54a; --bad: #ef7d7d; --idle: #9aa9b4;
  --ok-bg: #173325; --warn-bg: #3a2b14; --bad-bg: #3d1d1d; --idle-bg: #232d33; color-scheme: dark }
* { box-sizing: border-box }
body { background: var(--bg); color: var(--ink); font: 14px/1.5 var(--body); margin: 0 }
.wrap { max-width: 1180px; margin: 0 auto; padding-inline: 16px; padding-block: 20px 48px; display: grid; gap: 22px }
header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px 24px;
  border-bottom: 2px solid var(--ink); padding-bottom: 10px }
h1 { font: 600 28px/1.1 var(--display); margin: 0; letter-spacing: .01em; text-wrap: balance }
h2 { font: 600 17px/1.2 var(--display); margin: 0 0 10px; letter-spacing: .02em; text-transform: uppercase; color: var(--accent) }
h3 { font: 600 14px/1.3 var(--body); margin: 14px 0 6px }
.meta { color: var(--muted); font: 12px/1.4 var(--mono) }
.strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px }
.tile { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; min-width: 0 }
.tile .k { color: var(--muted); font: 11px/1.3 var(--body); text-transform: uppercase; letter-spacing: .06em }
.tile .v { font: 600 22px/1.2 var(--mono); font-variant-numeric: tabular-nums }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px }
.cols > section { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px; min-width: 0 }
.scroll { overflow-x: auto }
table { border-collapse: collapse; width: 100%; font-size: 13px }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--line); vertical-align: top }
th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .05em }
td.num, th.num { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap }
.pill { display: inline-block; padding: 1px 8px; border-radius: 10px; font: 600 11px/1.6 var(--mono); white-space: nowrap }
.ok { color: var(--ok); background: var(--ok-bg) } .warn { color: var(--warn); background: var(--warn-bg) }
.bad { color: var(--bad); background: var(--bad-bg) } .idle { color: var(--idle); background: var(--idle-bg) }
.bar { height: 6px; background: var(--idle-bg); border-radius: 3px; overflow: hidden; min-width: 60px }
.bar > i { display: block; height: 100%; background: var(--accent) }
.note { color: var(--muted); font-size: 12px; max-width: 70ch }
.figs { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px }
figure { margin: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 8px; min-width: 0 }
figure img { width: 100%; height: auto; display: block; background: #fff; border-radius: 3px }
figcaption { font-size: 12px; color: var(--muted); padding-top: 6px }
code { font-family: var(--mono); font-size: 12px }
"""


def pill(state: str) -> str:
    s = str(state).upper()
    cls = {"ON GPUWS": "idle", "DONE": "ok", "SUPPORTED": "ok", "R": "ok", "RUNNING": "ok", "CD": "ok",
           "FAILED": "bad", "BLOCKED": "bad", "NOT SUPPORTED": "bad", "F": "bad", "CA": "bad",
           "PD": "warn", "WAITING": "warn", "PAUSED": "warn", "INCONCLUSIVE": "warn"}.get(s, "idle")
    return f'<span class="pill {cls}">{esc(state)}</span>'


def render(lab, srv, cjobs, sums, figs, refreshed, every) -> str:
    now = time.strftime("%Y-%m-%d %H:%M")
    L = ['<title>DSG Project Board</title>',
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600&display=swap">']
    if every:
        L.append(f'<meta http-equiv="refresh" content="{int(every * 60)}">')
    L += [f"<style>{CSS}</style>", '<div class="wrap">',
          '<header><div><h1>DSG Project Board</h1><div class="meta">SAE guardrails: break, explain, fix &middot; '
          'gemma-2-2b-it, Gemma Scope layer 3</div></div>'
          f'<div class="meta">generated {esc(now)} &middot; reports refreshed: {esc(refreshed)}</div></header>']
    # summary strip
    c = lab.get("counts", {}) if "error" not in lab else {}
    sj = srv.get("jobs", []) if "error" not in srv else []
    tiles = [("lab jobs done", f"{c.get('DONE', 0)}/{lab.get('n_jobs', '?')}"),
             ("lab progress", f"{lab.get('pct', 0):.0f}%" if "pct" in lab else "?"),
             ("lab ETA", f"{lab['remaining_min'] / 60:.1f} h" if lab.get("remaining_min") is not None else "?"),
             ("lab failed/blocked", str(c.get("FAILED", 0) + c.get("BLOCKED", 0))),
             ("gpuws queued", str(len(sj)) if "error" not in srv else "?"),
             ("gpuws running", next((j["name"].replace("dsg-", "") for j in sj if j["state"] == "R"), "none") if sj else "none"),
             ("gpuws disk ours", srv.get("ours") or "?")]
    L.append('<div class="strip">' + "".join(f'<div class="tile"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>' for k, v in tiles) + "</div>")
    # machines
    L.append('<div class="cols">')
    L.append('<section><h2>Lab PC &middot; RTX 2000 Ada 16 GB</h2>')
    if "error" in lab:
        L.append(f'<p class="note">Queue status unavailable: {esc(lab["error"])}</p>')
    else:
        L.append('<h3>Running</h3><div class="scroll"><table><tr><th>job</th><th>experiment</th><th>progress</th><th>metric</th></tr>')
        for r in lab["running"] or [{"id": "none", "exp": "", "frac": None, "metric": ""}]:
            bar = f'<div class="bar"><i style="width:{100 * r["frac"]:.0f}%"></i></div>' if r["frac"] is not None else ""
            L.append(f'<tr><td><code>{esc(r["id"])}</code></td><td>{esc(r["exp"])}</td><td>{bar}</td><td class="meta">{esc(r["metric"])}</td></tr>')
        L.append("</table></div><h3>Job states</h3><p>" + " ".join(f'{pill(k)} {v}' for k, v in sorted(c.items())) + "</p>")
        if lab["failed"]:
            L.append('<h3>Needs attention</h3><p class="note">Triage with <code>python -m dsgx.queue.doctor</code>.</p><div class="scroll"><table>'
                     + "".join(f'<tr><td><code>{esc(f["id"])}</code></td><td>{esc(f["exp"])}</td><td>{pill(f["status"])}</td></tr>' for f in lab["failed"]) + "</table></div>")
        if lab["open_by_wave"]:
            L.append('<h3>Open jobs by wave</h3><p class="meta">' + " &middot; ".join(f"wave {w}: {n}" for w, n in lab["open_by_wave"].items()) + "</p>")
        if lab["moved"]:
            L.append('<p class="note">Moved to gpuws: ' + ", ".join(f"{esc(k)} ({v})" for k, v in lab["moved"].items()) + "</p>")
        L.append('<h3>By experiment</h3><div class="scroll"><table><tr><th>experiment</th><th class="num">done or moved</th><th class="num">open</th><th>state</th></tr>')
        for e, cnt in lab["by_exp"].items():
            tot = sum(cnt.values())
            mv = cnt.get("MOVED-TO-SERVER", 0)
            dn = cnt.get("DONE", 0) + mv
            worst = "FAILED" if cnt.get("FAILED") else "BLOCKED" if cnt.get("BLOCKED") else "RUNNING" if cnt.get("RUNNING") else \
                    "ON GPUWS" if mv and dn == tot else "DONE" if dn == tot else "WAITING"
            L.append(f'<tr><td>{esc(e)}</td><td class="num">{dn}/{tot}</td><td class="num">{tot - dn}</td><td>{pill(worst)}</td></tr>')
        L.append("</table></div>")
    L.append("</section>")
    L.append('<section><h2>gpuws &middot; RTX 6000 Ada 48 GB</h2>')
    if "error" in srv:
        L.append(f'<p class="note">Server {esc(srv["error"])}. The last fetched job summaries are shown below.</p>')
    else:
        d = srv.get("disk") or {}
        L.append(f'<p class="meta">GPU (used, total, util): {esc(srv.get("gpu"))}<br>disk: {esc(d.get("free"))} free of {esc(d.get("size"))}; ours {esc(srv.get("ours"))} (limit 100 GB)</p>')
        L.append('<h3>Our Slurm queue</h3><div class="scroll"><table><tr><th>id</th><th>job</th><th>state</th><th class="num">time</th><th class="num">limit</th><th>reason</th></tr>')
        for j in sorted(sj, key=lambda j: int(j["id"])) or []:
            L.append(f'<tr><td class="num">{esc(j["id"])}</td><td>{esc(j["name"].replace("dsg-", ""))}</td><td>{pill(j["state"])}</td>'
                     f'<td class="num">{esc(j["time"])}</td><td class="num">{esc(j["limit"])}</td><td class="meta">{esc(j["reason"])}</td></tr>')
        if not sj:
            L.append('<tr><td colspan="6" class="note">No jobs of ours in the queue.</td></tr>')
        L.append("</table></div>")
    L.append('<h3>Fetched results</h3><div class="scroll"><table><tr><th>job</th><th>finished</th><th>headline</th></tr>')
    for j in cjobs:
        h = "; ".join(f"{k}: {v}" for k, v in j["head"].items())
        L.append(f'<tr><td>{esc(j["job"])}</td><td class="meta">{esc((j["time"] or "")[:16])}</td><td class="meta">{esc(h)}</td></tr>')
    L.append("</table></div></section></div>")
    # claims
    L.append('<section class="cols"><section style="grid-column: 1 / -1"><h2>Claims C-H1 to C-H7</h2>'
             '<p class="note">Rules are fixed in <code>dsgx/analysis/claims.py</code> and applied to TEST results. Each machine is its own '
             'hardware baseline, so lab PC and gpuws verdicts are computed separately. Interim: verdicts can change until all runs finish.</p>'
             '<div class="scroll"><table><tr><th>id</th><th>claim</th><th>lab PC</th><th>gpuws</th><th>rule</th></tr>')
    base = fixed_claims()
    per = {hw: {c["id"]: c for c in ((s or {}).get("claims") or [])} for hw, s in sums.items()}
    for cl in base:
        cells = []
        for hw in ("labpc", "gpuws"):
            v = per.get(hw, {}).get(cl["id"])
            cells.append(pill(v["verdict"]) + (f'<div class="meta">{esc(v.get("evidence") or "")[:160]}</div>' if v and v.get("evidence") else "") if v else pill("no report"))
        L.append(f'<tr><td><code>{esc(cl["id"])}</code></td><td>{esc(cl["claim"])}</td><td>{cells[0]}</td><td>{cells[1]}</td><td class="note">{esc(cl["criterion"])}</td></tr>')
    L.append("</table></div>")
    for hw, s in sums.items():
        if s and s.get("completeness"):
            comp = s["completeness"]
            st = Counter(v.get("status") for v in comp.values() if isinstance(v, dict)) if isinstance(comp, dict) else {}
            txt = "experiments: " + ", ".join(f"{n} {k}" for k, n in st.most_common())
            L.append(f'<p class="meta">{esc(hw)} report {esc(s.get("_mtime"))} &middot; {esc(txt)[:300]}</p>')
    L.append("</section></section>")
    # figures
    for hw, title in (("labpc", "Key figures, lab PC"), ("gpuws", "Key figures, gpuws")):
        fl = figs.get(hw) or []
        L.append(f'<section><h2>{esc(title)}</h2>')
        if fl:
            L.append('<div class="figs">' + "".join(f'<figure><img src="{u}" alt="{esc(t)}" loading="lazy"><figcaption>{esc(t)}</figcaption></figure>' for t, u in fl) + "</div>")
        else:
            L.append('<p class="note">No figures yet. They appear after <code>python -m dsgx.analysis.dashboard</code> runs the interim report on finished runs.</p>')
        L.append("</section>")
    L.append('<p class="note">Aggregate numbers only (ids, states, metrics). Regenerate: <code>python -m dsgx.analysis.dashboard</code>.</p></div>')
    return "\n".join(L) + "\n"


def build(refresh=True, n_boot=2000, server=True) -> Path:
    from dsgx.util import atomic_write_text

    refreshed = refresh_reports(n_boot) if refresh else "not refreshed"
    if isinstance(refreshed, dict):
        refreshed = ", ".join(f"{k} {v}" for k, v in refreshed.items())
    lab = lab_status()
    srv = server_status() if server else {"error": "not queried (--no-server)"}
    sums = {hw: load_summary(hw) for hw in ("labpc", "gpuws")}
    figs = {hw: figures_for(hw) for hw in ("labpc", "gpuws")}
    page = render(lab, srv, cluster_jobs(), sums, figs, refreshed, build.every)
    out = OUT / "dashboard.html"
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, page)
    return out


build.every = None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-refresh", action="store_true", help="do not re-run the interim reports")
    ap.add_argument("--no-server", action="store_true", help="do not query gpuws")
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples for the interim reports")
    ap.add_argument("--every", type=float, default=None, help="regenerate every N minutes (page reloads itself)")
    a = ap.parse_args(argv)
    build.every = a.every
    while True:
        out = build(not a.no_refresh, a.n_boot, not a.no_server)
        print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
        if not a.every:
            return 0
        time.sleep(a.every * 60)


if __name__ == "__main__":
    raise SystemExit(main())
