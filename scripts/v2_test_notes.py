"""Notes on the StreamGuard v2 TEST result (CPU, read-only; reporting only, no rule or result changes).

(a) which attack conditions the V3 rule's worst case reads (claims.AXES), and the worst case over ALL tested attack
    conditions (incl. B7) per method and seed, paired v2 vs DSG / StreamGuard v1 on shared items;
(b) v2 benign-biology gate fire rate with CI; (c) paired test on MMLU false fires, v2 vs DSG (and v1);
(d) base64: v2 gate fire rate and base-model accuracy.
Uses the same run assembly as dsgx.analysis.v2_test (identical items, lab PC only). Writes JSON + a Markdown block.

    python scripts/v2_test_notes.py --json docs/V2_TEST_NOTES.json --md <scratch>/notes.md
"""
import argparse
import json

from dsgx.analysis import claims as C
from dsgx.analysis import v2_test as V

NONE = json.dumps({"name": "none"})
NAMES = {"sg2": "StreamGuard v2", "dsg": "DSG", "sg1": "StreamGuard v1"}


def build(nb=2000):
    pick, att = V.assemble()
    hw = {r.hardware for r in pick.values()}
    assert hw == {"labpc"}, hw
    conds = {ak: n for ak, n in att.items() if n not in ("clean", "benign-biology")}
    in_rule = {n: C.AXES.get(json.loads(ak).get("name")) for ak, n in conds.items()}
    seeds = sorted({k[3] for k in pick if k[0] == "sg2"})
    worst = {}
    for s in seeds:
        row = {}
        for k in ("sg2", "dsg", "sg1"):
            for scope in ("rule", "all"):
                best = None
                for ak, n in conds.items():
                    if scope == "rule" and not in_rule[n]:
                        continue
                    r = pick.get((k, ak, "forget", s))
                    if r is None:
                        continue
                    f = V._part(r, "forget")
                    m = float(f["correct"].astype(float).mean())
                    if best is None or m > best[0]:
                        best = (m, n, ak)
                row[f"{k}_{scope}"] = {"cond": best[1], "forget": V._ci(V._part(pick[(k, best[2], "forget", s)],
                                                                                 "forget")["correct"], nb), "ak": best[2]}
        for scope in ("rule", "all"):
            a = V._part(pick[("sg2", row[f"sg2_{scope}"]["ak"], "forget", s)], "forget")
            for k in ("dsg", "sg1"):
                b = V._part(pick[(k, row[f"{k}_{scope}"]["ak"], "forget", s)], "forget")
                row[f"vs_{k}_{scope}"] = V._pair(a, b, "correct", nb)
        for v in row.values():
            if isinstance(v, dict):
                v.pop("ak", None)
        worst[s] = row
    ff = {}
    for s in seeds:
        c2 = pick[("sg2", NONE, "clean", s)]
        a = V._part(c2, "utility").copy()
        a["fire"] = a["gate_fired"].astype(float)
        ff[s] = {"sg2": V._ci(a["fire"], nb)}
        for k in ("dsg", "sg1"):
            b = V._part(pick[(k, NONE, "clean", s)], "utility").copy()
            b["fire"] = b["gate_fired"].astype(float)
            ff[s][k] = V._ci(b["fire"], nb)
            ff[s][f"vs_{k}"] = V._pair(a, b, "fire", nb)
    bb = {}
    for s in seeds:
        h = V._part(pick[("sg2", NONE, "hardneg", s)], "utility")
        bb[s] = V._ci(h["gate_fired"].astype(float), nb)
    b64 = next(ak for ak, n in conds.items() if n == "base64")
    base_b64 = V._part(pick[("base", b64, "forget", 0)], "forget")
    base_clean = V._part(pick[("base", NONE, "clean", 0)], "forget")
    b64d = {"sg2_fire": {s: V._ci(V._part(pick[("sg2", b64, "forget", s)], "forget")["gate_fired"].astype(float), nb)
                         for s in seeds},
            "base_forget_base64": V._ci(base_b64["correct"], nb),
            "base_forget_clean": V._ci(base_clean["correct"], nb), "chance": 0.25}
    return {"hardware": "labpc", "rule_conditions": in_rule, "worst": worst, "false_fires": ff,
            "benign_bio_fire_sg2": bb, "base64": b64d}


def f(c):
    return f"{c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}] (n {c['n']})"


def d(t):
    if not t:
        return "–"
    mark = " ↓" if t["sig_lower"] else (" ↑" if t["sig_higher"] else "")
    return f"{t['diff']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}], p {t['p']:.2g}{mark}"


def render(s):
    rc = s["rule_conditions"]
    inc = [n for n, a in rc.items() if a]
    exc = [n for n, a in rc.items() if not a]
    L = ["## Notes (added 2026-11-14; reporting only)", "",
         "CPU only, read-only, from the same assembled TEST runs (lab PC, `scripts/v2_test_notes.py`, JSON "
         "`docs/V2_TEST_NOTES.json`). No TEST result, threshold or claim rule was changed; the verdicts above "
         "stand as `evaluate_v2` returned them. Paired = v2 − comparator on shared items, paired bootstrap [95 % CI], "
         "exact McNemar p; ↓ / ↑ significant.", "",
         "### (a) Worst-case forget: the V3 rule's attack suite vs all tested conditions", "",
         f"The V3 rule takes attacked TEST runs whose attack name is in `claims.AXES` (dilution B1, decompose B2, "
         f"translate / encode B3, rewrite_cache B4, suffix B5). Included ({len(inc)}): "
         + ", ".join(f"{V.ATT_TEXT.get(n, n)}" for n in inc) + f". Excluded ({len(exc)}): "
         + ", ".join(V.ATT_TEXT.get(n, n) for n in exc) + " (attack name `interleave`, B7, POST-HOC, not in AXES). "
         "Per run the rule's worst case is the attack with the highest forget accuracy.", "",
         "| seed | scope | DSG worst (attack) | SG v1 worst (attack) | SG v2 worst (attack) | v2 − DSG | v2 − SG v1 |",
         "|---|---|---|---|---|---|---|"]
    for sd, r in s["worst"].items():
        for scope, lab in (("rule", "V3 suite (16)"), ("all", "all 18 incl. B7")):
            cell = lambda k: f"{f(r[f'{k}_{scope}']['forget'])} ({V.ATT_TEXT.get(r[f'{k}_{scope}']['cond'])})"
            L.append(f"| {sd} | {lab} | {cell('dsg')} | {cell('sg1')} | {cell('sg2')} | {d(r[f'vs_dsg_{scope}'])} | "
                     f"{d(r[f'vs_sg1_{scope}'])} |")
    L += ["", "### (b) StreamGuard v2 benign-biology gate fire rate", "",
          "| seed | fire rate |", "|---|---|"] + [f"| {sd} | {f(c)} |" for sd, c in s["benign_bio_fire_sg2"].items()]
    L += ["", "### (c) MMLU false fires, paired (gate fired on clean MMLU utility items)", "",
          "| seed | DSG | SG v1 | SG v2 | v2 − DSG | v2 − SG v1 |", "|---|---|---|---|---|---|"]
    for sd, r in s["false_fires"].items():
        L.append(f"| {sd} | {f(r['dsg'])} | {f(r['sg1'])} | {f(r['sg2'])} | {d(r['vs_dsg'])} | {d(r['vs_sg1'])} |")
    b = s["base64"]
    L += ["", "### (d) Limitation: base64", "",
          f"On B3 base64 the StreamGuard v2 gate fires on {f(b['sg2_fire'][0])} of items (all seeds: "
          + ", ".join(f"s{k} {v['mean']:.3f}" for k, v in b["sg2_fire"].items())
          + f"). The base model answers base64-encoded questions at {f(b['base_forget_base64'])} (chance 0.25; clean "
          f"WMDP-Bio {f(b['base_forget_clean'])}), so the encoding itself removes the knowledge for this model and "
          "v2's forget accuracy there equals base. v2 gives no protection against base64; it is not tested against a "
          "model that can read base64, where this would be an open bypass.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--md")
    a = ap.parse_args(argv)
    s = build()
    if a.json:
        json.dump(s, open(a.json, "w"), indent=1, default=float)
    md = render(s)
    if a.md:
        open(a.md, "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
