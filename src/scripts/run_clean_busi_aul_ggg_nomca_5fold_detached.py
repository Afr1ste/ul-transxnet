from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(r"<LOCAL_THYROID_ROOT>")
PYTHON_EXE = Path(r"<LOCAL_CONDA_ROOT>\envs\Thyroid\python.exe")
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
MASTER_LOG_DIR = PROJECT_ROOT / "clean_dataset_ggg_nomca_run_logs"
MASTER_LOG = MASTER_LOG_DIR / f"clean_busi_aul_ggg_nomca_5fold_detached_{RUN_TS}.log"
STATUS_JSON = MASTER_LOG_DIR / f"clean_busi_aul_ggg_nomca_5fold_detached_{RUN_TS}.status.json"
LATEST_STATUS = MASTER_LOG_DIR / "clean_busi_aul_ggg_nomca_5fold_detached_latest.status.json"


COMMANDS = [
    {
        "name": "BUSI_GGG_withoutMCA_clean_5fold",
        "args": [
            "run_busi_ggg_nomca_clean_5fold.py",
            "--output-root",
            "busi_roi_runs_ggg_nomca_clean_5fold_safe",
            "--log-root",
            "busi_ggg_nomca_clean_5fold_safe_logs",
            "--skip-if-complete",
            "0",
            "--continue-on-error",
            "0",
        ],
    },
    {
        "name": "AUL_GGG_withoutMCA_clean_5fold",
        "args": [
            "run_aul_ggg_nomca_clean_5fold.py",
            "--output-root",
            "aul_roi_runs_ggg_nomca_clean_5fold_safe",
            "--log-root",
            "aul_ggg_nomca_clean_5fold_safe_logs",
            "--skip-if-complete",
            "0",
            "--continue-on-error",
            "0",
        ],
    },
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_status(state: dict) -> None:
    MASTER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["updated_at"] = now()
    state["master_log"] = str(MASTER_LOG)
    state["status_json"] = str(STATUS_JSON)
    STATUS_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    LATEST_STATUS.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def log(message: str) -> None:
    MASTER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with MASTER_LOG.open("a", encoding="utf-8", newline="") as f:
        f.write(f"[{now()}] {message}\n")
        f.flush()


def run_command(name: str, args: list[str]) -> int:
    cmd = [str(PYTHON_EXE), *args]
    log(f"START {name}")
    log("COMMAND " + " ".join(cmd))
    write_status({"state": "running", "current": name, "pid": os.getpid(), "command": cmd})

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    creationflags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags |= subprocess.CREATE_NO_WINDOW

    with MASTER_LOG.open("a", encoding="utf-8", newline="") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=creationflags,
        )
        write_status(
            {
                "state": "running",
                "current": name,
                "pid": os.getpid(),
                "child_pid": proc.pid,
                "command": cmd,
            }
        )
        code = proc.wait()

    log(f"END {name} exit_code={code}")
    return int(code)


def main() -> int:
    MASTER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"ProjectRoot={PROJECT_ROOT}")
    log(f"PythonExe={PYTHON_EXE}")
    log(f"SupervisorPid={os.getpid()}")
    log(f"MasterLog={MASTER_LOG}")
    write_status({"state": "started", "pid": os.getpid(), "run_ts": RUN_TS})

    if not PYTHON_EXE.exists():
        error = f"missing python exe: {PYTHON_EXE}"
        log(f"ERROR {error}")
        write_status({"state": "failed", "pid": os.getpid(), "error": error})
        return 2

    for item in COMMANDS:
        code = run_command(item["name"], item["args"])
        if code != 0:
            write_status(
                {
                    "state": "failed",
                    "pid": os.getpid(),
                    "failed_command": item["name"],
                    "exit_code": code,
                }
            )
            return code

    log("ALL_DONE")
    write_status({"state": "completed", "pid": os.getpid(), "run_ts": RUN_TS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
