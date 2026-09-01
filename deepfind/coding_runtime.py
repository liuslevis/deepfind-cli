from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

_MAX_FILE_BYTES = 32 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 128 * 1024 * 1024


class CodingRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class RuntimeInfo:
    runtime: str
    version: str
    image: str
    image_digest: str


@dataclass(frozen=True)
class RuntimeExecution:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    cleanup_failed: bool = False


class CodingRuntime:
    def __init__(
        self,
        *,
        runtime: str,
        image: str,
        seccomp_profile: Path,
        timeout: int,
    ) -> None:
        self.runtime = runtime
        self.image = image
        self.seccomp_profile = seccomp_profile.resolve()
        self.timeout = timeout
        self._probe_lock = Lock()
        self._runtime_info: RuntimeInfo | None = None

    def probe(self) -> RuntimeInfo:
        with self._probe_lock:
            version = self._runtime_version()
            digest = self._image_digest()
            cached = self._runtime_info
            if cached and cached.version == version and cached.image_digest == digest:
                return cached
            self._run_security_probe()
            info = RuntimeInfo(
                runtime=self.runtime,
                version=version,
                image=self.image,
                image_digest=digest,
            )
            self._runtime_info = info
            return info

    def run(
        self,
        workspace: Path,
        task_id: str,
        *,
        timeout: int | None = None,
    ) -> RuntimeExecution:
        self.probe()
        execution_timeout = timeout or self.timeout
        container_name = f"deepfind-{task_id}"
        request_path = (workspace / ".deepfind" / "request.json").resolve()
        if "," in str(request_path):
            raise CodingRuntimeError(
                "invalid_path",
                "The coding request path contains unsupported characters",
            )
        runner_path = (Path(__file__).parent / "coding_runner.py").resolve()
        if "," in str(runner_path) or not runner_path.is_file():
            raise CodingRuntimeError(
                "invalid_path",
                "The coding runner path is invalid",
            )
        create_command = [self.runtime, "create"]
        create_command.extend(self._container_options(container_name))
        create_command.extend(
            [
                "--mount",
                f"type=bind,source={request_path},target=/run/deepfind-request.json,readonly",
                "--mount",
                f"type=bind,source={runner_path},target=/opt/deepfind/runner.py,readonly",
                "--tmpfs",
                "/workspace:rw,nosuid,nodev,noexec,size=128m,uid=65532,gid=65532,mode=0700",
                "--entrypoint",
                "sleep",
                self.image,
                "infinity",
            ]
        )
        timed_out = False
        cleanup_failed = False
        exit_code = 1
        stdout = ""
        stderr = ""
        try:
            created = subprocess.run(
                create_command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
            if created.returncode != 0:
                raise CodingRuntimeError(
                    "sandbox_unavailable",
                    "The isolated coding container could not be created",
                )
            started = subprocess.run(
                [self.runtime, "start", container_name],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
            if started.returncode != 0:
                raise CodingRuntimeError(
                    "sandbox_unavailable",
                    "The isolated coding container could not be started",
                )
            executed = subprocess.run(
                [
                    self.runtime,
                    "exec",
                    container_name,
                    "python",
                    "/opt/deepfind/runner.py",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=execution_timeout,
                check=False,
            )
            stdout = executed.stdout or ""
            stderr = executed.stderr or ""
            exit_code = executed.returncode
            archived = subprocess.run(
                [
                    self.runtime,
                    "exec",
                    container_name,
                    "python",
                    "-c",
                    (
                        "import sys,tarfile;"
                        "t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|');"
                        "t.add('/workspace',arcname='.',recursive=True);"
                        "t.close()"
                    ),
                ],
                capture_output=True,
                timeout=30,
                check=False,
            )
            if archived.returncode != 0:
                raise CodingRuntimeError(
                    "invalid_result",
                    "The coding result could not be exported from the sandbox",
                )
            _extract_workspace_archive(archived.stdout or b"", workspace)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124
            stdout = _timeout_text(exc.stdout)
            stderr = _timeout_text(exc.stderr)
        except FileNotFoundError as exc:
            raise CodingRuntimeError(
                "runtime_unavailable",
                f"{self.runtime} container runtime is unavailable",
            ) from exc
        finally:
            try:
                cleanup = subprocess.run(
                    [self.runtime, "rm", "-f", container_name],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=15,
                    check=False,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                cleanup_failed = True
            else:
                cleanup_text = f"{cleanup.stdout}\n{cleanup.stderr}".lower()
                cleanup_failed = cleanup.returncode != 0 and "no such container" not in cleanup_text
        return RuntimeExecution(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            cleanup_failed=cleanup_failed,
        )

    def _runtime_version(self) -> str:
        try:
            proc = subprocess.run(
                [self.runtime, "version", "--format", "{{json .}}"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
        except FileNotFoundError as exc:
            raise CodingRuntimeError(
                "runtime_unavailable",
                f"{self.runtime} container runtime is unavailable",
            ) from exc
        if proc.returncode != 0:
            raise CodingRuntimeError(
                "runtime_unavailable",
                f"{self.runtime} container runtime is unavailable",
            )
        return (proc.stdout or "").strip()

    def _image_digest(self) -> str:
        try:
            proc = subprocess.run(
                [
                    self.runtime,
                    "image",
                    "inspect",
                    self.image,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                check=False,
            )
        except FileNotFoundError as exc:
            raise CodingRuntimeError(
                "runtime_unavailable",
                f"{self.runtime} container runtime is unavailable",
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise CodingRuntimeError(
                "image_unavailable",
                "The configured coding image could not be inspected",
            ) from exc
        if proc.returncode != 0:
            raise CodingRuntimeError(
                "image_unavailable",
                "The configured coding image is unavailable",
            )
        try:
            inspected = json.loads(proc.stdout or "[]")
        except json.JSONDecodeError as exc:
            raise CodingRuntimeError(
                "image_unavailable",
                "The configured coding image digest could not be verified",
            ) from exc
        if not isinstance(inspected, list) or not inspected or not isinstance(inspected[0], dict):
            raise CodingRuntimeError(
                "image_unavailable",
                "The configured coding image digest could not be verified",
            )
        image_data = inspected[0]
        image_id = str(image_data.get("Id", ""))
        repo_digests = image_data.get("RepoDigests")
        if self.image.startswith("sha256:"):
            if image_id != self.image:
                raise CodingRuntimeError(
                    "image_unavailable",
                    "The configured coding image ID could not be verified",
                )
            return image_id
        expected = self.image.rsplit("@", 1)[-1]
        matched = next(
            (
                str(item).rsplit("@", 1)[-1]
                for item in repo_digests or []
                if isinstance(item, str) and item.rsplit("@", 1)[-1] == expected
            ),
            None,
        )
        if matched is None:
            raise CodingRuntimeError(
                "image_unavailable",
                "The configured coding image digest could not be verified",
            )
        return matched

    def _run_security_probe(self) -> None:
        if not self.seccomp_profile.is_file():
            raise CodingRuntimeError(
                "sandbox_unavailable",
                "The coding seccomp profile is unavailable",
            )
        probe = (
            "import os,pathlib;"
            "s=open('/proc/self/status',encoding='utf-8').read();"
            "assert os.getuid()!=0;"
            "assert 'NoNewPrivs:\\t1' in s;"
            "assert 'Seccomp:\\t2' in s;"
            "assert int(next(x.split()[1] for x in s.splitlines() if x.startswith('CapEff:')),16)==0;"
            "p=pathlib.Path('/rootfs-write-test');"
            "\ntry:p.write_text('x')\nexcept OSError:pass\nelse:raise AssertionError('rootfs writable')"
        )
        probe_name = f"deepfind-coding-probe-{os.getpid()}"
        command = [self.runtime, "run", "--rm"]
        command.extend(self._container_options(probe_name))
        command.extend(
            [
                "--tmpfs",
                "/workspace:rw,nosuid,nodev,noexec,size=1m",
                "--entrypoint",
                "python",
                self.image,
                "-c",
                probe,
            ]
        )
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        subprocess.run(
            [self.runtime, "rm", "-f", probe_name],
            capture_output=True,
            timeout=10,
            check=False,
        )
        if proc.returncode != 0:
            raise CodingRuntimeError(
                "sandbox_unavailable",
                "Required container isolation controls are unavailable",
            )

    def _container_options(self, container_name: str) -> list[str]:
        return [
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--user",
            "65532:65532",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--security-opt",
            f"seccomp={self.seccomp_profile}",
            "--pids-limit",
            "64",
            "--cpus",
            "1",
            "--memory",
            "512m",
            "--memory-swap",
            "512m",
            "--ulimit",
            "nofile=256:256",
            "--ulimit",
            "nproc=64:64",
            "--ulimit",
            "fsize=33554432:33554432",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=64m",
            "--env",
            "HOME=/tmp/home",
            "--env",
            "TMPDIR=/tmp",
            "--env",
            "PIP_CONFIG_FILE=/dev/null",
            "--env",
            "PIP_CACHE_DIR=/tmp/pip-cache",
        ]


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _extract_workspace_archive(archive: bytes, workspace: Path) -> None:
    if len(archive) > _MAX_ARCHIVE_BYTES + 1024 * 1024:
        raise CodingRuntimeError("resource_limit", "The coding artifacts exceeded the size limit")
    root = workspace.resolve()
    total = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as bundle:
            for member in bundle:
                relative = Path(member.name.replace("/", os.sep))
                if relative.is_absolute() or ".." in relative.parts:
                    raise CodingRuntimeError(
                        "invalid_path",
                        "The coding result contained an unsafe path",
                    )
                target = (root / relative).resolve()
                try:
                    target.relative_to(root)
                except ValueError as exc:
                    raise CodingRuntimeError(
                        "invalid_path",
                        "The coding result contained an unsafe path",
                    ) from exc
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile() or member.size > _MAX_FILE_BYTES:
                    raise CodingRuntimeError(
                        "invalid_result",
                        "The coding result contained an unsupported file",
                    )
                total += member.size
                if total > _MAX_ARCHIVE_BYTES:
                    raise CodingRuntimeError(
                        "resource_limit",
                        "The coding artifacts exceeded the size limit",
                    )
                source = bundle.extractfile(member)
                if source is None:
                    raise CodingRuntimeError(
                        "invalid_result",
                        "The coding result contained an unreadable file",
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
    except (tarfile.TarError, OSError) as exc:
        if isinstance(exc, CodingRuntimeError):
            raise
        raise CodingRuntimeError(
            "invalid_result",
            "The coding result archive was invalid",
        ) from exc
