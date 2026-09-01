from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

WORKSPACE = Path("/workspace")
CONTROL = WORKSPACE / ".deepfind"
REQUEST = Path(os.environ.get("DEEPFIND_REQUEST_PATH", "/run/deepfind-request.json"))
RESULT = CONTROL / "result.json"

ALLOWED_COMMAND_BINS = frozenset(
    {
        "python",
        "python3",
        "ls",
        "cat",
        "head",
        "tail",
        "wc",
        "find",
        "stat",
        "file",
        "pwd",
        "echo",
        "tree",
        "grep",
        "rg",
        "sed",
        "awk",
        "sort",
        "uniq",
        "cut",
        "tr",
        "diff",
        "jq",
    }
)


def main() -> int:
    CONTROL.mkdir(mode=0o700, exist_ok=True)
    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    task_id = str(request["task_id"])
    plan = request["plan"]
    limits = request["limits"]
    timeout = int(limits["timeout_seconds"])
    started = time.monotonic()
    commands: list[dict[str, Any]] = []
    artifacts = [str(item["path"]) for item in plan["files"]]
    try:
        for item in plan["files"]:
            path = safe_path(str(item["path"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(item["content"]), encoding="utf-8")
        for command in plan["commands"]:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError
            commands.append(run_command(command, remaining, limits))
            if commands[-1]["exit_code"] != 0:
                write_result(
                    {
                        "ok": False,
                        "task_id": task_id,
                        "answer": "A validation command failed.",
                        "commands": commands,
                        "artifacts": artifacts,
                        "error": {
                            "code": "execution_failed",
                            "message": "A validation command failed.",
                        },
                    }
                )
                return 1
        write_result(
            {
                "ok": True,
                "task_id": task_id,
                "answer": str(plan["answer"]),
                "commands": commands,
                "artifacts": artifacts,
                "error": None,
            }
        )
        return 0
    except TimeoutError:
        write_result(
            {
                "ok": False,
                "task_id": task_id,
                "answer": "The coding task exceeded its time limit.",
                "commands": commands,
                "artifacts": artifacts,
                "error": {"code": "timed_out", "message": "The coding task exceeded its time limit."},
            }
        )
        return 124
    except Exception as exc:
        print(f"runner error: {type(exc).__name__}", file=sys.stderr)
        write_result(
            {
                "ok": False,
                "task_id": task_id,
                "answer": "The coding task failed.",
                "commands": commands,
                "artifacts": artifacts,
                "error": {"code": "execution_failed", "message": "The coding task failed."},
            }
        )
        return 1


def safe_path(relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("invalid path")
    path = (WORKSPACE / candidate).resolve()
    path.relative_to(WORKSPACE)
    if path == CONTROL or CONTROL in path.parents or path.name == ".venv":
        raise ValueError("reserved path")
    return path


def run_command(command: list[str], timeout: float, limits: dict[str, Any]) -> dict[str, Any]:
    args = [str(item) for item in command]
    if not args or args[0] not in ALLOWED_COMMAND_BINS:
        raise ValueError("unsupported command")
    if "/" in args[0] or "\\" in args[0]:
        raise ValueError("unsupported command")
    env = {
        "HOME": "/tmp/home",
        "TMPDIR": "/tmp",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_CONFIG_FILE": "/dev/null",
        "PIP_CACHE_DIR": "/tmp/pip-cache",
    }
    started = time.monotonic()
    try:
        proc = subprocess.run(
            args,
            cwd=WORKSPACE,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            preexec_fn=set_limits,
        )
        exit_code = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        stdout = decode_timeout(exc.stdout)
        stderr = decode_timeout(exc.stderr)
    return {
        "command": args,
        "exit_code": exit_code,
        "duration_ms": round((time.monotonic() - started) * 1000),
        "stdout": sanitize(stdout, int(limits["stdout_chars"])),
        "stderr": sanitize(stderr, int(limits["stderr_chars"])),
    }


def set_limits() -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))


def sanitize(value: str, limit: int) -> str:
    cleaned = "".join(char if char in "\n\r\t" or ord(char) >= 32 else "\ufffd" for char in value)
    return cleaned[:limit]


def decode_timeout(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def write_result(value: dict[str, Any]) -> None:
    temporary = CONTROL / f".result-{os.getpid()}.tmp"
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(RESULT)


if __name__ == "__main__":
    raise SystemExit(main())
