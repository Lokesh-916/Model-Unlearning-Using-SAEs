"""GPU / RAM / disk probes used by the scheduler, the status command and preflight."""
import shutil
import subprocess

from dsgx import paths


def gpu() -> dict:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        used, total, free, util, temp = [float(x) for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"used_gb": used / 1024, "total_gb": total / 1024, "free_gb": free / 1024,
                "util_pct": util, "temp_c": temp, "ok": True}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def gpu_processes() -> list[dict]:
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        out = []
        for line in r.stdout.strip().splitlines():
            pid, mem, name = [x.strip() for x in line.split(",", 2)]
            out.append({"pid": int(pid), "mem_mb": float(mem), "name": name})
        return out
    except Exception:  # noqa: BLE001
        return []


def ram() -> dict:
    import psutil

    vm = psutil.virtual_memory()
    return {"used_gb": (vm.total - vm.available) / 1e9, "total_gb": vm.total / 1e9, "percent": vm.percent}


def disk() -> dict:
    p = paths.results_dir()
    p.mkdir(parents=True, exist_ok=True)
    du = shutil.disk_usage(p)
    return {"free_gb": du.free / 1e9, "total_gb": du.total / 1e9}


def snapshot() -> dict:
    return {"gpu": gpu(), "ram": ram(), "disk": disk()}
