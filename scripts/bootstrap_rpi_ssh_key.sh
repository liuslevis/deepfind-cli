#!/usr/bin/env bash
set -euo pipefail

RPI_HOST="${RPI_HOST:-192.168.0.205}"
RPI_USER="${RPI_USER:-david}"
RPI_SSH_PORT="${RPI_SSH_PORT:-22}"
PUBKEY_PATH="${PUBKEY_PATH:-}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

detect_pubkey() {
  if [[ -n "${PUBKEY_PATH}" ]]; then
    printf "%s\n" "${PUBKEY_PATH}"
    return
  fi

  for candidate in "${HOME}/.ssh/id_ed25519.pub" "${HOME}/.ssh/id_rsa.pub"; do
    if [[ -f "${candidate}" ]]; then
      printf "%s\n" "${candidate}"
      return
    fi
  done

  echo "No SSH public key found. Create one with:" >&2
  echo "  ssh-keygen -t ed25519 -C \"${USER}@$(hostname)\"" >&2
  exit 1
}

require_cmd ssh

PUBKEY_PATH="$(detect_pubkey)"
TARGET="${RPI_USER}@${RPI_HOST}"
SSH_OPTS=(
  -p "${RPI_SSH_PORT}"
  -o BatchMode=yes
  -o ConnectTimeout=5
  -o StrictHostKeyChecking=accept-new
)

if [[ ! -f "${PUBKEY_PATH}" ]]; then
  echo "Public key not found: ${PUBKEY_PATH}" >&2
  exit 1
fi

echo "Checking passwordless SSH access to ${TARGET}:${RPI_SSH_PORT}..."
if ssh "${SSH_OPTS[@]}" "${TARGET}" "exit 0"; then
  echo "Passwordless SSH is ready for ${TARGET}."
  exit 0
fi

echo "Passwordless SSH is not configured for ${TARGET}." >&2
echo "Install this public key on the Raspberry Pi once, then rerun the script:" >&2
echo >&2
echo "  ${PUBKEY_PATH}" >&2
echo >&2
echo "If ssh-copy-id is available, the usual command is:" >&2
echo "  ssh-copy-id -i ${PUBKEY_PATH} -p ${RPI_SSH_PORT} ${TARGET}" >&2
echo >&2
echo "After that, rerun ./scripts/bootstrap_rpi_ssh_key.sh and then ./scripts/deploy_rpi.sh." >&2
exit 1
