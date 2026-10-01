"""Pause the queue: running jobs finish, no new jobs start.

    python -m dsgx.queue.pause "reason"
"""
import sys

from dsgx.queue import common as q


def main() -> int:
    q.ensure_dirs()
    reason = " ".join(sys.argv[1:]) or "paused by user"
    q.set_paused(reason)
    print(f"paused: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
