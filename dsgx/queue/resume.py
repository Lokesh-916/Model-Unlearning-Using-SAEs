"""Resume the queue: remove the PAUSED marker (and acknowledge a wave pause point).

    python -m dsgx.queue.resume
"""
from dsgx.queue import common as q


def main() -> int:
    p = q.qdir() / "PAUSED"
    reason = q.paused()
    ctl = q.control()
    w = ctl.get("pause_after_wave")
    if w is not None and w not in ctl.get("resumed_waves", []) and reason and f"wave {w}" in reason:
        ctl["resumed_waves"] = ctl.get("resumed_waves", []) + [w]
        q.save_control(ctl)
    if p.exists():
        p.unlink()
        print(f"resumed (was: {reason})")
    else:
        print("queue was not paused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
