from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import stat
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any
from uuid import uuid4

from .coding_runtime import CodingRuntime, CodingRuntimeError
from .config import Settings
from .json_utils import try_load_json
from .llm_transport import complete_text

MAX_QUERY_CHARS = 20_000
MAX_CONTEXT_CHARS = 100_000
MAX_RESULT_BYTES = 4 * 1024 * 1024
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_CHARS = 1024 * 1024

_ALLOWED_COMMAND_BINS: frozenset[str] = frozenset(
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

_PLANNER_PROMPT = """You are a coding planner. Return one JSON object only.
Create a complete, minimal solution for the requested task. Python code must target Python 3.12 standard library only.
Schema:
{
  "answer": "short final explanation",
  "files": [{"path": "relative/path", "content": "complete UTF-8 file content"}],
  "commands": [["python", "relative/script.py"], ["ls", "-la"], ["cat", "solution.py"]]
}
Rules:
- Paths must be relative, use forward slashes, and must not contain '..'.
- Do not create .venv or request/result control files.
- Commands are argument arrays, never shell strings. No pipes, redirects, or metacharacters.
- Allowed command binaries: python, python3, ls, cat, head, tail, wc, find, stat, file, pwd,
  echo, tree, grep, rg, sed, awk, sort, uniq, cut, tr, diff, jq.
- Shells (sh/bash/zsh), network tools, package managers, and mutation commands (rm/mv/chmod/...)
  are NOT allowed.
- Include commands that validate or demonstrate the result.
"""


class CodingPathError(ValueError):
    pass


@dataclass(frozen=True)
class CodingError:
    code: str
    message: str


@dataclass(frozen=True)
class Artifact:
    path: str
    size: int
    media_type: str
    sha256: str


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    exit_code: int
    duration_ms: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class CodingResult:
    ok: bool
    task_id: str
    status: str
    answer: str
    artifacts: list[Artifact] = field(default_factory=list)
    commands: list[CommandResult] = field(default_factory=list)
    error: CodingError | None = None
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CodingConfig:
    runtime: str
    image: str
    root: Path
    timeout: int
    max_concurrent: int
    retention: int


class CodingService:
    def __init__(
        self,
        settings: Settings,
        config: CodingConfig,
        *,
        runtime: Any | None = None,
    ) -> None:
        self.settings = settings
        self.config = config
        self._slots = BoundedSemaphore(config.max_concurrent)
        self.runtime = runtime or CodingRuntime(
            runtime=config.runtime,
            image=config.image,
            seccomp_profile=Path(__file__).with_name("coding_seccomp.json"),
            timeout=config.timeout,
        )
        self.runtime_info = self.runtime.probe()

    def coding(self, query: str, context: str | None = None) -> CodingResult:
        started = time.monotonic()
        task_id = f"task_{uuid4().hex}"
        error = _validate_input(query, context)
        if error:
            return _result_error(task_id, started, "failed", error)
        if not self._slots.acquire(blocking=False):
            return _result_error(
                task_id,
                started,
                "failed",
                CodingError("resource_limit", "The coding task concurrency limit was reached"),
            )

        task_root: Path | None = None
        try:
            task_root = self._create_task_root(task_id)
            workspace = task_root / "workspace"
            control = workspace / ".deepfind"
            logs = task_root / "logs"
            workspace.mkdir(mode=0o700)
            control.mkdir()
            logs.mkdir()
            plan = self._plan(query.strip(), context)
            remaining_timeout = self.config.timeout - int(time.monotonic() - started)
            if remaining_timeout <= 0:
                return _result_error(
                    task_id,
                    started,
                    "timed_out",
                    CodingError("timed_out", "The coding task exceeded its time limit"),
                )
            request = {
                "version": "coding.v1",
                "task_id": task_id,
                "query": query.strip(),
                "context": context,
                "plan": plan,
                "limits": {
                    "timeout_seconds": remaining_timeout,
                    "stdout_chars": MAX_OUTPUT_CHARS,
                    "stderr_chars": MAX_OUTPUT_CHARS,
                },
            }
            request_path = task_root / "request.json"
            _write_json(request_path, request)
            request_path.chmod(stat.S_IREAD)
            _write_json(control / "request.json", request)

            execution = self.runtime.run(
                workspace,
                task_id,
                timeout=remaining_timeout,
            )
            _write_limited_text(logs / "runtime.stdout.log", execution.stdout)
            _write_limited_text(logs / "runtime.stderr.log", execution.stderr)
            if execution.cleanup_failed:
                return _result_error(
                    task_id,
                    started,
                    "failed",
                    CodingError("cleanup_failed", "The coding container could not be cleaned up"),
                )
            if execution.timed_out:
                return _result_error(
                    task_id,
                    started,
                    "timed_out",
                    CodingError("timed_out", "The coding task exceeded its time limit"),
                )
            if execution.exit_code == 137:
                return _result_error(
                    task_id,
                    started,
                    "failed",
                    CodingError("resource_limit", "The coding task exceeded a resource limit"),
                )
            if execution.exit_code == 124:
                return _result_error(
                    task_id,
                    started,
                    "timed_out",
                    CodingError("timed_out", "The coding task exceeded its time limit"),
                )
            result = self._load_result(control / "result.json", workspace, task_id)
            _write_json(task_root / "result.json", result.to_dict())
            if execution.exit_code != 0 and result.ok:
                return _result_error(
                    task_id,
                    started,
                    "failed",
                    CodingError("execution_failed", "The coding container exited unsuccessfully"),
                    commands=result.commands,
                )
            return CodingResult(
                ok=result.ok,
                task_id=task_id,
                status=result.status,
                answer=result.answer,
                artifacts=result.artifacts,
                commands=result.commands,
                error=result.error,
                duration_ms=_duration_ms(started),
            )
        except CodingRuntimeError as exc:
            return _result_error(
                task_id,
                started,
                "failed",
                CodingError(exc.code, exc.message),
            )
        except CodingPathError as exc:
            _write_diagnostic(task_root, exc)
            return _result_error(
                task_id,
                started,
                "failed",
                CodingError("invalid_path", "The coding task produced an unsafe path"),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            _write_diagnostic(task_root, exc)
            return _result_error(
                task_id,
                started,
                "failed",
                CodingError("invalid_result", "The coding task produced an invalid result"),
            )
        except Exception as exc:
            _write_diagnostic(task_root, exc)
            return _result_error(
                task_id,
                started,
                "failed",
                CodingError("execution_failed", "The coding task could not be completed"),
            )
        finally:
            self._slots.release()
            if task_root is not None and self.config.retention == 0:
                try:
                    _safe_remove_task_root(self.config.root, task_root)
                except OSError:
                    return _result_error(
                        task_id,
                        started,
                        "failed",
                        CodingError("cleanup_failed", "The coding task directory could not be cleaned up"),
                    )

    def _create_task_root(self, task_id: str) -> Path:
        root = self.config.root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        _reject_link(root)
        task_root = root / task_id
        _ensure_within(root, task_root)
        task_root.mkdir(mode=0o700)
        return task_root

    def _plan(self, query: str, context: str | None) -> dict[str, Any]:
        if not self.settings.api_key:
            raise RuntimeError("A configured LLM API key is required for coding")
        user_input = query
        if context:
            user_input += f"\n\nContext:\n{context}"
        text = complete_text(
            self.settings.new_client(),
            api_mode=self.settings.api_mode,
            model=self.settings.model,
            instructions=_PLANNER_PROMPT,
            user_input=user_input,
            max_output_tokens=8000,
            timeout=self.config.timeout,
        )
        plan = try_load_json(text)
        return _validate_plan(plan)

    def _load_result(self, path: Path, workspace: Path, expected_task_id: str) -> CodingResult:
        _ensure_safe_file(workspace, path, max_bytes=MAX_RESULT_BYTES)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("result must be an object")
        if raw.get("task_id") != expected_task_id:
            raise ValueError("result task id does not match")
        commands_raw = raw.get("commands")
        if not isinstance(commands_raw, list):
            raise ValueError("commands must be a list")
        commands = [_parse_command_result(item) for item in commands_raw]
        ok = bool(raw.get("ok"))
        answer = str(raw.get("answer", ""))[:20_000]
        error = None
        if not ok:
            error_raw = raw.get("error")
            if not isinstance(error_raw, dict):
                raise ValueError("failed result requires error")
            code = str(error_raw.get("code", "execution_failed"))
            if code not in {
                "dependency_denied",
                "resource_limit",
                "timed_out",
                "execution_failed",
            }:
                code = "execution_failed"
            error = CodingError(code, str(error_raw.get("message", "Coding execution failed"))[:1000])
        artifacts = _collect_artifacts(workspace)
        return CodingResult(
            ok=ok,
            task_id=str(raw.get("task_id", "")),
            status=(
                "completed"
                if ok
                else "timed_out"
                if error and error.code == "timed_out"
                else "failed"
            ),
            answer=answer,
            artifacts=artifacts,
            commands=commands,
            error=error,
        )


def coding_config(settings: Settings) -> CodingConfig:
    return CodingConfig(
        runtime=settings.coding_runtime,
        image=settings.coding_image,
        root=Path(settings.coding_root),
        timeout=settings.coding_timeout,
        max_concurrent=settings.coding_max_concurrent,
        retention=settings.coding_retention,
    )


@lru_cache(maxsize=8)
def get_coding_service(settings: Settings) -> CodingService:
    return CodingService(settings, coding_config(settings))


def _validate_input(query: Any, context: Any) -> CodingError | None:
    if not isinstance(query, str) or not query.strip():
        return CodingError("invalid_input", "query must not be empty")
    if len(query) > MAX_QUERY_CHARS:
        return CodingError("invalid_input", f"query must not exceed {MAX_QUERY_CHARS} characters")
    if context is not None and not isinstance(context, str):
        return CodingError("invalid_input", "context must be a string or null")
    if isinstance(context, str) and len(context) > MAX_CONTEXT_CHARS:
        return CodingError("invalid_input", f"context must not exceed {MAX_CONTEXT_CHARS} characters")
    return None


def _validate_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("planner result must be an object")
    answer = plan.get("answer")
    files = plan.get("files")
    commands = plan.get("commands")
    if not isinstance(answer, str) or not isinstance(files, list) or not isinstance(commands, list):
        raise ValueError("planner result has an invalid schema")
    if len(files) > 100 or len(commands) > 20:
        raise ValueError("planner result exceeds action limits")
    normalized_files: list[dict[str, str]] = []
    total_content = 0
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("file action must be an object")
        path = _safe_relative_path(item.get("path"))
        content = item.get("content")
        if not isinstance(content, str):
            raise ValueError("file content must be text")
        total_content += len(content.encode("utf-8"))
        if total_content > MAX_ARTIFACT_BYTES:
            raise ValueError("planned files exceed the artifact limit")
        normalized_files.append({"path": path, "content": content})
    normalized_commands: list[list[str]] = []
    for command in commands:
        if not isinstance(command, list) or not command:
            raise ValueError("command must be a non-empty argument array")
        args = [str(arg) for arg in command]
        if any("\x00" in arg or len(arg) > 10_000 for arg in args):
            raise ValueError("command argument is invalid")
        if args[0] not in _ALLOWED_COMMAND_BINS:
            raise ValueError(f"command binary is not allowed: {args[0]}")
        if "/" in args[0] or "\\" in args[0]:
            raise ValueError("command binary must be a bare name")
        normalized_commands.append(args)
    return {"answer": answer[:20_000], "files": normalized_files, "commands": normalized_commands}


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CodingPathError("path must be non-empty")
    path = Path(value.replace("/", os.sep))
    if path.is_absolute() or path.drive or ".." in path.parts:
        raise CodingPathError("path must stay inside the workspace")
    normalized = path.as_posix()
    if normalized.startswith(".deepfind/") or normalized in {".deepfind", ".venv"}:
        raise CodingPathError("reserved path")
    return normalized


def _collect_artifacts(workspace: Path) -> list[Artifact]:
    artifacts: list[Artifact] = []
    total = 0
    for path in sorted(workspace.rglob("*")):
        relative = path.relative_to(workspace)
        if relative.parts and relative.parts[0] in {".deepfind", ".venv"}:
            continue
        if path.is_dir():
            _reject_link(path)
            continue
        _ensure_safe_file(workspace, path, max_bytes=MAX_FILE_BYTES)
        size = path.stat().st_size
        total += size
        if total > MAX_ARTIFACT_BYTES:
            raise ValueError("artifact size limit exceeded")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        artifacts.append(
            Artifact(
                path=relative.as_posix(),
                size=size,
                media_type=media_type,
                sha256=_sha256_file(path),
            )
        )
    return artifacts


def _parse_command_result(value: Any) -> CommandResult:
    if not isinstance(value, dict) or not isinstance(value.get("command"), list):
        raise ValueError("invalid command result")
    return CommandResult(
        command=[str(item)[:10_000] for item in value["command"]],
        exit_code=int(value.get("exit_code", 1)),
        duration_ms=max(0, int(value.get("duration_ms", 0))),
        stdout=_sanitize_output(str(value.get("stdout", ""))),
        stderr=_sanitize_output(str(value.get("stderr", ""))),
    )


def _sanitize_output(value: str) -> str:
    safe = "".join(char if char in "\n\r\t" or ord(char) >= 32 else "\ufffd" for char in value)
    return safe[:MAX_OUTPUT_CHARS]


def _ensure_safe_file(root: Path, path: Path, *, max_bytes: int) -> None:
    _ensure_within(root.resolve(), path.resolve())
    _reject_link(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > max_bytes:
        raise ValueError("unsafe file")


def _reject_link(path: Path) -> None:
    if path.is_symlink():
        raise CodingPathError("symbolic links are not allowed")
    info = path.lstat()
    attributes = getattr(info, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if attributes & reparse:
        raise CodingPathError("reparse points are not allowed")


def _ensure_within(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CodingPathError("path escapes the task root") from exc


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_limited_text(path: Path, value: str) -> None:
    path.write_text(_sanitize_output(value), encoding="utf-8")


def _write_diagnostic(task_root: Path | None, exc: Exception) -> None:
    if task_root is None:
        return
    try:
        logs = task_root / "logs"
        logs.mkdir(exist_ok=True)
        (logs / "host-error.log").write_text(
            f"{type(exc).__name__}: {exc}"[:MAX_OUTPUT_CHARS],
            encoding="utf-8",
        )
    except OSError:
        return


def _safe_remove_task_root(root: Path, task_root: Path) -> None:
    resolved_root = root.resolve()
    resolved_task = task_root.resolve()
    _ensure_within(resolved_root, resolved_task)
    if resolved_task.parent != resolved_root or not resolved_task.name.startswith("task_"):
        raise OSError("unsafe cleanup path")
    shutil.rmtree(resolved_task, onerror=_make_writable_and_retry)


def _make_writable_and_retry(function, path: str, _error_info) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duration_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))


def _result_error(
    task_id: str,
    started: float,
    status: str,
    error: CodingError,
    *,
    commands: list[CommandResult] | None = None,
) -> CodingResult:
    return CodingResult(
        ok=False,
        task_id=task_id,
        status=status,
        answer=error.message,
        commands=commands or [],
        error=error,
        duration_ms=_duration_ms(started),
    )
