from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(r"<LOCAL_THYROID_ROOT>")
PYTHON_EXE = Path(r"<LOCAL_CONDA_ROOT>\envs\Thyroid\python.exe")

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
MASTER_LOG_DIR = PROJECT_ROOT / "withmca_full_three_dataset_queue_logs"
MASTER_LOG = MASTER_LOG_DIR / f"withmca_full_three_dataset_{RUN_TS}.log"
STATUS_JSON = MASTER_LOG_DIR / f"withmca_full_three_dataset_{RUN_TS}.status.json"
LATEST_STATUS = MASTER_LOG_DIR / "withmca_full_three_dataset_latest.status.json"


COMMANDS = [
    {
        "name": "TN5000_GGG_withMCA_3seed",
        "args": ["run_tn5000_ggg_mca_enabled_3seed.py"],
        "log_root": "tn5000_ggg_mca_enabled_3seed_logs",
        "run_root": "tn5000_roi_runs_ggg_mca_enabled_3seed",
    },
    {
        "name": "BUSI_GGG_withMCA_clean_5fold",
        "args": [
            "run_busi_structure_ablation_5fold.py",
            "--only-backbones",
            "transxnetggg",
            "--output-root",
            "busi_roi_runs_ggg_mca_clean_5fold_safe",
            "--log-root",
            "busi_ggg_mca_clean_5fold_safe_logs",
            "--skip-if-complete",
            "0",
            "--continue-on-error",
            "0",
        ],
        "log_root": "busi_ggg_mca_clean_5fold_safe_logs",
        "run_root": "busi_roi_runs_ggg_mca_clean_5fold_safe",
    },
    {
        "name": "AUL_GGG_withMCA_clean_5fold",
        "args": [
            "run_aul_structure_confirm_5fold.py",
            "--only-backbones",
            "transxnetggg",
            "--output-root",
            "aul_roi_runs_ggg_mca_clean_5fold_safe",
            "--log-root",
            "aul_ggg_mca_clean_5fold_safe_logs",
            "--skip-if-complete",
            "0",
            "--continue-on-error",
            "0",
        ],
        "log_root": "aul_ggg_mca_clean_5fold_safe_logs",
        "run_root": "aul_roi_runs_ggg_mca_clean_5fold_safe",
    },
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def latest_subdir(root_name: str) -> str | None:
    root = PROJECT_ROOT / root_name
    if not root.exists():
        return None
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    return str(max(subdirs, key=lambda p: p.stat().st_mtime))


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


def run_command(item: dict, completed: list[dict]) -> int:
    name = item["name"]
    cmd = [str(PYTHON_EXE), *item["args"]]
    log(f"START {name}")
    log("COMMAND " + " ".join(cmd))

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

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
                "run_ts": RUN_TS,
                "pid": os.getpid(),
                "child_pid": proc.pid,
                "current": name,
                "completed": completed,
                "command": cmd,
            }
        )
        code = proc.wait()

    log_dir = latest_subdir(item["log_root"])
    run_dir = latest_subdir(item["run_root"])
    result = {
        "name": name,
        "exit_code": int(code),
        "log_dir": log_dir,
        "run_dir": run_dir,
    }
    completed.append(result)
    log(f"END {name} exit_code={code} log_dir={log_dir} run_dir={run_dir}")
    return int(code)


def main() -> int:
    MASTER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"ProjectRoot={PROJECT_ROOT}")
    log(f"PythonExe={PYTHON_EXE}")
    log(f"SupervisorPid={os.getpid()}")
    log(f"MasterLog={MASTER_LOG}")
    write_status({"state": "started", "run_ts": RUN_TS, "pid": os.getpid(), "completed": []})

    if not PYTHON_EXE.exists():
        message = f"missing python exe: {PYTHON_EXE}"
        log(f"ERROR {message}")
        write_status({"state": "failed", "run_ts": RUN_TS, "pid": os.getpid(), "error": message})
        return 2

    completed: list[dict] = []
    for item in COMMANDS:
        code = run_command(item, completed)
        if code != 0:
            write_status(
                {
                    "state": "failed",
                    "run_ts": RUN_TS,
                    "pid": os.getpid(),
                    "failed_command": item["name"],
                    "exit_code": code,
                    "completed": completed,
                }
            )
            return code

    log("ALL_DONE")
    write_status({"state": "completed", "run_ts": RUN_TS, "pid": os.getpid(), "completed": completed})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
