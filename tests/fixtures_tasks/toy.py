"""A tiny task entry used by test_harness_p1b (runs on CPU through the real worker)."""


def main(ctx):
    ctx.write_config(note="toy")
    n = int(ctx.args.get("n", 3))
    ctx.progress.update(items_total=n, force=True)
    for _ in range(n):
        ctx.progress.advance(1)
    (ctx.private_dir() / "secret.txt").write_text("private only")
    ctx.write_metrics({"n": n, "smoke": ctx.smoke})
    ctx.finish({"forget": {"mean": 0.5, "lo": 0.4, "hi": 0.6, "n": n}, "view": "toy"})
    return {"n": n}
