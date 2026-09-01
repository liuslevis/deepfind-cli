from __future__ import annotations

import base64
import fcntl
import json
import os
import pty
import select
import signal
import struct
import sys
import termios


def emit(message: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def resize(fd: int, cols: int, rows: int) -> None:
    size = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, size)


def main() -> int:
    pid, master = pty.fork()
    if pid == 0:
        os.chdir("/workspace")
        env = {
            "HOME": "/workspace",
            "LANG": "C.UTF-8",
            "PATH": "/opt/venv-template/bin:/usr/local/bin:/usr/bin:/bin",
            "TERM": "xterm-256color",
        }
        os.execvpe("/bin/bash", ["/bin/bash", "--noprofile", "--norc"], env)

    flags = fcntl.fcntl(master, fcntl.F_GETFL)
    fcntl.fcntl(master, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    emit({"type": "ready", "cwd": ".", "pid": pid})

    input_fd = sys.stdin.fileno()
    buffer = b""
    while True:
        readable, _, _ = select.select([master, input_fd], [], [], 0.25)
        if master in readable:
            try:
                data = os.read(master, 65536)
            except OSError:
                data = b""
            if data:
                emit({"type": "output", "data": base64.b64encode(data).decode("ascii")})
            else:
                break

        if input_fd in readable:
            chunk = os.read(input_fd, 65536)
            if not chunk:
                os.kill(pid, signal.SIGHUP)
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line:
                    continue
                message = json.loads(line.decode("utf-8"))
                message_type = message.get("type")
                if message_type == "input":
                    os.write(master, base64.b64decode(str(message.get("data", ""))))
                elif message_type == "resize":
                    resize(master, int(message["cols"]), int(message["rows"]))
                elif message_type == "signal":
                    signal_name = str(message.get("signal", "SIGINT"))
                    os.killpg(pid, getattr(signal, signal_name))
                elif message_type == "close":
                    os.killpg(pid, signal.SIGHUP)

        exited, status = os.waitpid(pid, os.WNOHANG)
        if exited:
            if os.WIFEXITED(status):
                emit({"type": "exit", "exit_code": os.WEXITSTATUS(status), "signal": None})
            else:
                emit({"type": "exit", "exit_code": None, "signal": os.WTERMSIG(status)})
            return 0

    try:
        _, status = os.waitpid(pid, 0)
    except ChildProcessError:
        return 0
    if os.WIFEXITED(status):
        emit({"type": "exit", "exit_code": os.WEXITSTATUS(status), "signal": None})
    else:
        emit({"type": "exit", "exit_code": None, "signal": os.WTERMSIG(status)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
