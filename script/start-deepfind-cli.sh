#!/usr/bin/env bash
set -euo pipefail

restart=false
if [[ "${1:-}" == "--restart" ]]; then
  restart=true
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--restart]" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
python="$repo_root/.venv/bin/python"
host="${DEEPFIND_HOST:-0.0.0.0}"
port="${DEEPFIND_PORT:-62000}"
state_dir="$HOME/Library/Application Support/DeepFind"
log_dir="$HOME/Library/Logs/DeepFind"
pid_file="$state_dir/deepfind-web.pid"
log_file="$log_dir/backend.log"

if [[ ! -x "$python" ]]; then
  echo "DeepFind virtual environment not found. Run 'uv sync' in $repo_root first." >&2
  exit 1
fi

mkdir -p "$state_dir" "$log_dir"

if [[ -f "$pid_file" ]]; then
  existing_pid="$(cat "$pid_file")"
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    if [[ "$restart" != true ]]; then
      echo "DeepFind backend is already running with PID $existing_pid."
      exit 0
    fi
    kill "$existing_pid"
    for _ in {1..100}; do
      if ! kill -0 "$existing_pid" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    if kill -0 "$existing_pid" 2>/dev/null; then
      echo "DeepFind backend PID $existing_pid did not stop." >&2
      exit 1
    fi
  fi
  rm -f "$pid_file"
fi

if health_response="$(curl --silent --fail --max-time 2 "http://127.0.0.1:$port/api/health" 2>/dev/null)" \
  && grep -q '"service":"deepfind-web"' <<<"$health_response"; then
  if [[ "$restart" == true ]]; then
    echo "DeepFind is responding on port $port, but its PID is not tracked in $pid_file." >&2
    echo "Stop that process explicitly before restarting." >&2
    exit 1
  fi
  echo "DeepFind backend is already responding at http://127.0.0.1:$port."
  exit 0
fi

cd "$repo_root"
nohup "$python" -m deepfind.web_api \
  --host "$host" \
  --port "$port" \
  </dev/null >>"$log_file" 2>&1 &
backend_pid=$!
echo "$backend_pid" > "$pid_file"

for _ in {1..30}; do
  if health_response="$(curl --silent --fail --max-time 2 "http://127.0.0.1:$port/api/health" 2>/dev/null)" \
    && grep -q '"service":"deepfind-web"' <<<"$health_response"; then
    echo "DeepFind backend started with PID $backend_pid at http://$host:$port."
    echo "Log: $log_file"
    exit 0
  fi
  if ! kill -0 "$backend_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done

rm -f "$pid_file"
echo "DeepFind backend failed to start. Check $log_file." >&2
exit 1
