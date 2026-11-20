"""Queue task that runs one external command as one GPU job (e.g. a table script of the separate clean repository),
so it is scheduled one job at a time with the lab queue's watchdog, logs and status like any other job.

args:
  cmd  list[str]   the command (run without a shell)
  cwd  str         working directory
  env  dict        environment overrides (e.g. DSG_RESULTS / DSG_CACHE / DSG_PRIVATE of an isolated tree, so the
                   command never writes into this harness's results or caches); "" removes a variable
Writes external.log (the command's stdout + stderr) and metrics.json {rc, wall_seconds} in the task run dir; raises
on a non-zero exit (the job is then FAILED and can be re-queued with dsgx.queue.doctor). Heartbeat every 60 s so the
watchdog does not take a long, quiet command for a hang. The command must itself keep hazardous text out of its
stdout (the clean repository's scripts print ids, paths and metrics only).
"""
import os
import subprocess
import threading
import time


def task(ctx):
    a = ctx.args
    env = dict(os.environ)
    for k, v in (a.get("env") or {}).items():
        if v == "":
            env.pop(k, None)
        else:
            env[k] = str(v)
    ctx.write_config(cmd=a["cmd"], cwd=a["cwd"])
    log = ctx.run_dir() / "external.log"
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"=== {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(a['cmd'])} (cwd {a['cwd']})\n")
        f.flush()
        p = subprocess.Popen(a["cmd"], cwd=a["cwd"], env=env, stdout=f, stderr=subprocess.STDOUT)
        stop = threading.Event()

        def beat():
            while not stop.wait(60):
                ctx.progress.update(force=True, phase=f"external {int(time.time() - t0) // 60} min")

        th = threading.Thread(target=beat, daemon=True)
        th.start()
        rc = p.wait()
        stop.set()
        f.write(f"=== exit {rc} after {time.time() - t0:.0f} s\n")
    wall = time.time() - t0
    ctx.write_metrics({"rc": rc, "wall_seconds": wall})
    if rc != 0:
        raise RuntimeError(f"external command exited {rc} after {wall:.0f} s; see {log}")
    ctx.finish({"view": "external", "forget": None, "rc": rc, "wall_seconds": round(wall)})
    return {"rc": rc, "wall_seconds": round(wall)}
