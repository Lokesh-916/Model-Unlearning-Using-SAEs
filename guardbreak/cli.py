"""guardbreak run --gate dsg --attack dilution [--pad 400] [--position before] [--n 50] [--split test]

Builds a dsgx run config (gate -> method, attack -> transformation) and runs it; prints the run dir
and the headline (forget accuracy with CI). Gates: dsg (DSG-faithful), window, cusum, probe, base.
"""
import argparse
import json

GATES = {"dsg": {"name": "dsg-faithful", "n_features": 20, "retain_pct": 95, "multiplier": 500},
         "window": {"name": "gated", "gate": {"type": "window", "w": 16}},
         "cusum": {"name": "gated", "gate": {"type": "cusum"}},
         "probe": {"name": "gated", "gate": {"type": "probe_resid", "probe_train": "mcq-dev"}},
         "base": {"name": "base"}}


def build_config(a) -> dict:
    attack = {"name": a.attack}
    for k in ("pad", "position", "source", "variant", "k", "lang", "encoding"):
        v = getattr(a, k, None)
        if v is not None:
            attack[k] = v
    return {"exp_id": a.exp_id, "case": a.case, "split": a.split, "purpose": "report" if a.split == "test" else "select",
            "view": "both", "batch_size": 1, "datasets": ["@forget"], "dataset_label": "forget",
            "limit": a.n, "method": dict(GATES[a.gate]), "attack": attack}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="guardbreak")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--gate", choices=sorted(GATES), default="dsg")
    r.add_argument("--attack", default="dilution")
    r.add_argument("--pad", type=int); r.add_argument("--position"); r.add_argument("--source")
    r.add_argument("--variant"); r.add_argument("--k", type=int); r.add_argument("--lang"); r.add_argument("--encoding")
    r.add_argument("--case", default="bio"); r.add_argument("--split", default="test")
    r.add_argument("--n", type=int, default=None); r.add_argument("--exp-id", dest="exp_id", default="guardbreak")
    sub.add_parser("list")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        import guardbreak

        print(json.dumps({"attacks": sorted(guardbreak.ATTACKS), "gates": sorted(GATES)}))
        return None
    from dsgx.run import run

    rd = run(build_config(a))
    head = json.loads((rd / "DONE").read_text())["headline"]
    print(json.dumps({"run_dir": str(rd), "headline": head}, default=str))
    return rd
