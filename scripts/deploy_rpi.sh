#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RPI_HOST="${RPI_HOST:-192.168.0.205}"
RPI_USER="${RPI_USER:-david}"
RPI_SSH_PORT="${RPI_SSH_PORT:-22}"
RPI_APP_DIR="${RPI_APP_DIR:-/home/david/apps/deepfind-cli}"
RPI_PORT="${RPI_PORT:-8000}"
SERVICE_NAME="${SERVICE_NAME:-deepfind-web}"
LOCAL_ENV_FILE="${LOCAL_ENV_FILE:-${REPO_ROOT}/.env}"
NODE_CHANNEL="${NODE_CHANNEL:-latest-v20.x}"
TARGET="${RPI_USER}@${RPI_HOST}"
SSH_OPTS=(
  -p "${RPI_SSH_PORT}"
  -o BatchMode=yes
  -o ConnectTimeout=5
  -o StrictHostKeyChecking=accept-new
)
SCP_OPTS=(
  -P "${RPI_SSH_PORT}"
  -o BatchMode=yes
  -o ConnectTimeout=5
  -o StrictHostKeyChecking=accept-new
)

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd ssh
require_cmd scp
require_cmd tar

if [[ ! -f "${LOCAL_ENV_FILE}" ]]; then
  echo "Expected env file not found: ${LOCAL_ENV_FILE}" >&2
  exit 1
fi

echo "Checking SSH key access to ${TARGET}:${RPI_SSH_PORT}..."
if ! ssh "${SSH_OPTS[@]}" "${TARGET}" "exit 0"; then
  echo "Key-based SSH failed for ${TARGET}. Configure passwordless SSH first." >&2
  echo "Hint: ./scripts/bootstrap_rpi_ssh_key.sh" >&2
  exit 1
fi

echo "Preparing ${RPI_APP_DIR} on the Raspberry Pi..."
ssh "${SSH_OPTS[@]}" "${TARGET}" "
  set -euo pipefail
  mkdir -p '${RPI_APP_DIR}'
  find '${RPI_APP_DIR}' -mindepth 1 -maxdepth 1 -exec rm -rf {} +
"

echo "Uploading repository archive..."
COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 tar \
  --no-xattrs \
  --no-mac-metadata \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  --exclude='.DS_Store' \
  --exclude='*.pyc' \
  --exclude='.env' \
  --exclude='node_modules' \
  --exclude='web/node_modules' \
  --exclude='web/dist' \
  --exclude='tmp' \
  --exclude='audio' \
  -C "${REPO_ROOT}" \
  -czf - . | ssh "${SSH_OPTS[@]}" "${TARGET}" "tar -xzf - -C '${RPI_APP_DIR}'"

echo "Copying ${LOCAL_ENV_FILE}..."
scp "${SCP_OPTS[@]}" "${LOCAL_ENV_FILE}" "${TARGET}:${RPI_APP_DIR}/.env"

echo "Building the app and configuring a user service..."
ssh "${SSH_OPTS[@]}" "${TARGET}" \
  "APP_DIR=$(printf '%q' "${RPI_APP_DIR}") SERVICE_NAME=$(printf '%q' "${SERVICE_NAME}") APP_PORT=$(printf '%q' "${RPI_PORT}") NODE_CHANNEL=$(printf '%q' "${NODE_CHANNEL}") bash -se" <<'REMOTE'
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export NPM_CONFIG_PREFIX="${HOME}/.local/npm-global"
export PATH="${NPM_CONFIG_PREFIX}/bin:${PATH}"

require_remote_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required remote command: $1" >&2
    exit 1
  fi
}

require_remote_cmd python3
require_remote_cmd curl
require_remote_cmd tar
require_remote_cmd systemctl

if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
  echo "Python 3.11 or newer is required on the Raspberry Pi." >&2
  exit 1
fi

ensure_node() {
  local current_major=""
  local arch=""
  local shasums=""
  local tarball=""
  local node_dir=""
  local node_root="${HOME}/.local/node"

  if command -v node >/dev/null 2>&1; then
    current_major="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || true)"
  fi

  if [[ -n "${current_major}" ]] && (( current_major >= 18 )) && command -v npm >/dev/null 2>&1; then
    return
  fi

  case "$(uname -m)" in
    aarch64|arm64)
      arch="arm64"
      ;;
    x86_64|amd64)
      arch="x64"
      ;;
    armv7l)
      arch="armv7l"
      ;;
    *)
      echo "Unsupported Raspberry Pi architecture: $(uname -m)" >&2
      exit 1
      ;;
  esac

  echo "Installing Node.js ${NODE_CHANNEL} for ${arch} under ${node_root}..."
  mkdir -p "${node_root}" "${HOME}/.local/bin"
  shasums="$(curl -fsSL "https://nodejs.org/dist/${NODE_CHANNEL}/SHASUMS256.txt")"
  tarball="$(printf '%s\n' "${shasums}" | awk '/linux-'"${arch}"'\.tar\.gz$/ {print $2; exit}')"

  if [[ -z "${tarball}" ]]; then
    echo "Could not find a Node.js tarball for architecture ${arch} in ${NODE_CHANNEL}." >&2
    exit 1
  fi

  node_dir="${tarball%.tar.gz}"
  rm -rf "${node_root:?}/${node_dir}"
  curl -fsSL "https://nodejs.org/dist/${NODE_CHANNEL}/${tarball}" | tar -xzf - -C "${node_root}"

  ln -sfn "${node_root}/${node_dir}/bin/node" "${HOME}/.local/bin/node"
  ln -sfn "${node_root}/${node_dir}/bin/npm" "${HOME}/.local/bin/npm"
  ln -sfn "${node_root}/${node_dir}/bin/npx" "${HOME}/.local/bin/npx"
  if [[ -x "${node_root}/${node_dir}/bin/corepack" ]]; then
    ln -sfn "${node_root}/${node_dir}/bin/corepack" "${HOME}/.local/bin/corepack"
  fi
  hash -r
}

ensure_node
require_remote_cmd node
require_remote_cmd npm

if ! node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 18 ? 0 : 1)' >/dev/null 2>&1; then
  echo "Node.js 18 or newer is required on the Raspberry Pi." >&2
  exit 1
fi

ensure_opencli() {
  if command -v opencli >/dev/null 2>&1; then
    return
  fi

  echo "Installing @jackwener/opencli under ${NPM_CONFIG_PREFIX}..."
  mkdir -p "${NPM_CONFIG_PREFIX}"
  npm install -g @jackwener/opencli
  hash -r

  if ! command -v opencli >/dev/null 2>&1; then
    echo "opencli was not found after npm installation." >&2
    exit 1
  fi
}

ensure_opencli

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

UV_BIN="$(command -v uv)"
if [[ -z "${UV_BIN}" ]]; then
  echo "uv was not found after installation." >&2
  exit 1
fi

cd "${APP_DIR}"
"${UV_BIN}" sync --frozen

cd "${APP_DIR}/web"
npm ci
npm run build

mkdir -p "${HOME}/.config/systemd/user"
cat > "${HOME}/.config/systemd/user/${SERVICE_NAME}.service" <<SERVICE
[Unit]
Description=DeepFind Web
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${APP_DIR}
Environment=PATH=${NPM_CONFIG_PREFIX}/bin:${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile=${APP_DIR}/.env
ExecStart=${UV_BIN} run deepfind-web --host 0.0.0.0 --port ${APP_PORT}
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
SERVICE

if ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "systemctl --user is not available for this session." >&2
  echo "Log into the Raspberry Pi once with a normal shell and rerun deploy." >&2
  exit 1
fi

systemctl --user daemon-reload
systemctl --user enable "${SERVICE_NAME}.service"
systemctl --user restart "${SERVICE_NAME}.service"
systemctl --user --no-pager --full status "${SERVICE_NAME}.service"

for attempt in $(seq 1 20); do
  if curl -fsS "http://127.0.0.1:${APP_PORT}/api/health" >/dev/null; then
    break
  fi
  sleep 1
done

curl -fsS "http://127.0.0.1:${APP_PORT}/api/health"

if command -v loginctl >/dev/null 2>&1; then
  linger_status="$(loginctl show-user "${USER}" -p Linger --value 2>/dev/null || true)"
  if [[ "${linger_status}" != "yes" ]]; then
    echo
    echo "Note: user lingering is not enabled for ${USER}."
    echo "The service is running now, but boot persistence needs a one-time admin command:"
    echo "  sudo loginctl enable-linger ${USER}"
  fi
fi
REMOTE

echo "Deployment complete. Open http://${RPI_HOST}:${RPI_PORT} on your LAN."
