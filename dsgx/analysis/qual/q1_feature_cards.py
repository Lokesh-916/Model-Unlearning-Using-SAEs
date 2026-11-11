"""Q1 feature cards: one card per DSG-selected feature (layer-3 Gemma Scope 16k, average_l0_142).

    python -m dsgx.analysis.qual.q1_feature_cards --probe        # 1 request: is l0_142 hosted on Neuronpedia?
    python -m dsgx.analysis.qual.q1_feature_cards --fetch        # fetch explanations (<= --max features, 0.5 s apart)
    python -m dsgx.analysis.qual.q1_feature_cards                # build cards from logs + the fetched cache

Card content: feature id; which runs selected it (A1-test seeds, C1 attribution / chi2 / DSG sets, X1); forget
and retain fire rates and their ratio from the bio activation cache (seed 0); N9 splitting indicator if logged;
Neuronpedia explanation(s) and density.

IMPORTANT: Neuronpedia's default `gemma-2-2b/3-gemmascope-res-16k` is the CANONICAL SAE (average_l0_59), not
DSG's average_l0_142, so its feature indices do NOT match. This script only requests the l0_142 source
(NP_SOURCE) and records "not hosted" on 404; it never falls back to the canonical set.
Only explanation strings and density numbers are cached; top-activating examples (which can contain
hazardous text) are never requested into the cache or printed.
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from dsgx import paths
from dsgx.analysis.collect import load_all
from dsgx.analysis.qual import out_dir
from dsgx.util import atomic_write_json, atomic_write_text

NP_MODEL = "gemma-2-2b"
NP_SOURCE = "3-gemmascope-res-16k__l0-142"
API = "https://www.neuronpedia.org/api/feature/{model}/{source}/{idx}"


def selected_features(runs) -> dict[int, list[str]]:
    sel = defaultdict(list)
    for r in runs:
        if not r.is_mcq or r.case != "bio" or r.is_base:
            continue
        mi = r.config.get("method_info") or {}
        feats = mi.get("features")
        if not feats:
            continue
        g, mdl = r.cfg.get("method", {}).get("gate") or {}, r.cfg.get("model") or {}
        rel = g.get("sae_release") or mdl.get("sae_release") or "gemma-scope-2b-pt-res"
        sid = g.get("sae_id") or mdl.get("sae_id") or "layer_3/width_16k/average_l0_142"
        if (rel, sid) != ("gemma-scope-2b-pt-res", "layer_3/width_16k/average_l0_142") or (mdl.get("name") or "gemma-2-2b-it") != "gemma-2-2b-it":
            continue  # other SAEs (C3 canonical l0_59, other layers, Gemma 3) index different features
        tag = f"{r.base_exp}:{r.method}" + (f":s{r.seed}" if r.base_exp == "A1-test" else "")
        if r.base_exp == "C1":
            tag = "C1:" + Path((r.cfg["method"].get("gate") or {}).get("features_file", "?")).stem
        for f in feats:
            if tag not in sel[int(f)]:
                sel[int(f)].append(tag)
    for f in sorted((paths.cache_dir() / "features").glob("C1_*_bio.json")):
        for x in json.loads(f.read_text()).get("features", [])[:20]:
            t = f"C1-file:{f.stem}"
            if t not in sel[int(x)]:
                sel[int(x)].append(t)
    return dict(sel)


def cache_stats(feats):
    try:
        from dsgx.data import activation_cache as ac

        c = ac.open_cache("gemma-2-2b-it", "gemma-scope-2b-pt-res", "layer_3/width_16k/average_l0_142",
                                             "bio-forget-corpus", "wikitext", 0)
        fr, rr = c.stats("forget")["fire_rate"], c.stats("retain")["fire_rate"]
        return {f: {"forget_fire_rate": float(fr[f]), "retain_fire_rate": float(rr[f]),
                    "ratio": float((fr[f] + 1e-6) / (rr[f] + 1e-6))} for f in feats}
    except Exception as e:  # cache not present (e.g. tests): cards still build
        return {"_error": f"{type(e).__name__}: {e}"}


def np_cache_path(root, idx):
    return root / "neuronpedia" / f"{NP_MODEL}__{NP_SOURCE}__{idx}.json"


def fetch_one(idx, timeout=20):
    url = API.format(model=NP_MODEL, source=NP_SOURCE, idx=idx)
    req = urllib.request.Request(url, headers={"User-Agent": "dsg-capstone-feature-cards"})
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        return {"status": "not hosted" if e.code == 404 else f"http {e.code}", "url": url}
    except Exception as e:
        return {"status": f"error {type(e).__name__}", "url": url}
    # keep ONLY explanation strings and density; never activation examples
    expl = [x.get("description") for x in (d.get("explanations") or []) if x.get("description")]
    return {"status": "ok", "url": url, "explanations": expl[:3], "frac_nonzero": d.get("frac_nonzero"),
            "max_act": d.get("maxActApprox"), "source_check": d.get("layer")}  # must equal NP_SOURCE


def build(out: Path, feats_by: dict, stats: dict) -> list[dict]:
    cards = []
    for f in sorted(feats_by, key=lambda x: (-len(feats_by[x]), x)):
        npd = json.loads(np_cache_path(out, f).read_text()) if np_cache_path(out, f).exists() else {"status": "not fetched"}
        s = stats.get(f, {})
        cards.append({"feature": f, "selected_by": feats_by[f], "n_selections": len(feats_by[f]), **s,
                      "neuronpedia": npd, "neuronpedia_page": f"https://www.neuronpedia.org/{NP_MODEL}/{NP_SOURCE}/{f}"})
    L = ["# Q1 feature cards — DSG layer-3 features (Gemma Scope 16k, average_l0_142)", "",
         f"{len(cards)} features. Neuronpedia source `{NP_SOURCE}` (NOT the canonical l0_59 set). "
         "Fire rates: bio activation cache, seed 0. No activation examples are stored.", ""]
    if "_error" in stats:
        L += [f"_activation-cache stats unavailable: {stats['_error']}_", ""]
    for c in cards:
        np_ = c["neuronpedia"]
        L += [f"## Feature {c['feature']}", "",
              f"- selected by ({c['n_selections']}): {', '.join(c['selected_by'])}",
              f"- forget fire rate {c.get('forget_fire_rate', 'n/a')}, retain {c.get('retain_fire_rate', 'n/a')}, ratio {c.get('ratio', 'n/a')}",
              f"- Neuronpedia ({np_.get('status')}): " + ("; ".join(np_.get("explanations") or []) or "—")
              + (f" (density {np_['frac_nonzero']:.2e})" if isinstance(np_.get("frac_nonzero"), (int, float)) else ""),
              f"- page: {c['neuronpedia_page']}", ""]
    atomic_write_text(out / "Q1_FEATURE_CARDS.md", "\n".join(L))
    atomic_write_json(out / "q1_feature_cards.json", cards)
    return cards


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="fetch Neuronpedia explanations (network)")
    ap.add_argument("--probe", action="store_true", help="one request to check NP_SOURCE is hosted")
    ap.add_argument("--max", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = out_dir(a.out)
    runs = load_all(smoke=a.smoke)
    feats_by = selected_features(runs)
    if a.probe:
        f = next(iter(feats_by), 0)
        print(json.dumps({k: v for k, v in fetch_one(f).items() if k != "explanations"}))
        return 0
    if a.fetch:
        (out / "neuronpedia").mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(feats_by, key=lambda x: -len(feats_by[x]))[: a.max]:
            p = np_cache_path(out, f)
            if p.exists():
                continue
            rec = fetch_one(f)
            atomic_write_json(p, rec)
            n += 1
            if rec["status"] == "not hosted" and n == 1:
                print(f"{NP_SOURCE} is not hosted on Neuronpedia (404); stopping. Cards use cache stats only.")
                break
            time.sleep(0.5)
        print(f"fetched {n} feature(s)")
    cards = build(out, feats_by, cache_stats(list(feats_by)))
    print(f"wrote {out / 'Q1_FEATURE_CARDS.md'} ({len(cards)} features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
