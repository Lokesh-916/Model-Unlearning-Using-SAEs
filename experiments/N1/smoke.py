"""N1 task: run the GuardBreak CLI end to end for each gate x a small attack set (smoke-sized by args)."""
import json


def task(ctx):
    import guardbreak  # noqa: F401
    from guardbreak.cli import main

    a = ctx.args
    out = {}
    combos = a.get("combos", [["dsg", "dilution", ["--pad", "400"]], ["window", "dilution", ["--pad", "400"]],
                              ["dsg", "encode", ["--encoding", "base64"]], ["base", "dilution", ["--pad", "400"]]])
    ctx.progress.update(items_total=len(combos), items_done=0, force=True)
    for gate, att, extra in combos:
        rd = main(["run", "--gate", gate, "--attack", att, *extra, "--split", a.get("split", "test"),
                   "--n", str(a.get("n", 50)), "--exp-id", ctx.exp_id])
        out[f"{gate}:{att}:{' '.join(extra)}"] = json.loads((rd / "DONE").read_text())["headline"]["forget"]
        ctx.progress.advance(1)
    ctx.write_metrics({"results": out})
    ctx.finish({"view": "guardbreak-cli", "forget": None})
