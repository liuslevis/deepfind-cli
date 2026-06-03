#!/usr/bin/env bash
set -euo pipefail

# Parse command-line arguments
SKIP_SETUP=false
SKIP_XHS_LOGIN=false
USE_TMUX=true

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-setup)
      SKIP_SETUP=true
      shift
      ;;
    --skip-xhs-login)
      SKIP_XHS_LOGIN=true
      shift
      ;;
    --no-tmux)
      USE_TMUX=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--skip-setup] [--skip-xhs-login] [--no-tmux]"
      exit 1
      ;;
  esac
done

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
WEB_ROOT="$REPO_ROOT/web"
WEB_NODE_MODULES="$WEB_ROOT/node_modules"
WEB_PACKAGE_LOCK="$WEB_ROOT/package-lock.json"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info() {
  echo -e "${CYAN}$1${NC}"
}

error() {
  echo -e "${RED}ERROR: $1${NC}" >&2
}

warn() {
  echo -e "${YELLOW}$1${NC}"
}

success() {
  echo -e "${GREEN}$1${NC}"
}

# Check if command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Verify web root exists
if [[ ! -d "$WEB_ROOT" ]]; then
  error "Web app directory not found: $WEB_ROOT"
  exit 1
fi

# Check required dependencies
if ! command_exists uv; then
  error "uv is not available in PATH. Install uv first:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if ! command_exists npm; then
  error "npm is not available in PATH. Install Node.js first:"
  echo "  brew install node"
  exit 1
fi

if [[ "$SKIP_XHS_LOGIN" == "false" ]] && ! command_exists xhs; then
  error "xhs is not available in PATH. Install xiaohongshu-cli first:"
  echo "  uv tool install xiaohongshu-cli"
  exit 1
fi

# Setup step
if [[ "$SKIP_SETUP" == "false" ]]; then
  info "Syncing Python dependencies with uv..."
  cd "$REPO_ROOT"
  uv sync --extra media --extra local-llm --extra browser
  info "Installing Playwright browsers..."
  # Install Playwright browsers if playwright is available
  if uv run python -c "import playwright" 2>/dev/null; then
    if ! uv run playwright install --help &>/dev/null; then
      warn "Playwright not yet installed, installing browsers..."
      uv run playwright install
    elif ! ls ~/.cache/ms-playwright/chromium-* &>/dev/null 2>&1; then
      info "Playwright browsers not found, installing..."
      uv run playwright install
    else
      success "Playwright browsers already installed."
    fi
  else
    warn "Playwright not available in dependencies, skipping browser installation."
  fi

  info "Installing web dependencies..."
  cd "$WEB_ROOT"
  if [[ -f "$WEB_PACKAGE_LOCK" ]] && [[ ! -d "$WEB_NODE_MODULES" ]]; then
    npm ci
  else
    npm install
  fi
  cd "$REPO_ROOT"
else
  warn "Skipping setup step."
fi

# Xiaohongshu login
if [[ "$SKIP_XHS_LOGIN" == "false" ]]; then
  info "Checking Xiaohongshu authentication..."

  # Check if opencli is installed
  if ! command -v opencli &> /dev/null; then
    warn "opencli not found. XHS features will be disabled."
    echo "Install with: npm install -g @opencli/cli"
  else
    # Check if logged in by trying to get creator profile
    if opencli xiaohongshu creator-profile -f json &>/dev/null; then
      success "✅ XHS logged in"
      echo ""
      info "Your XHS Profile:"
      opencli xiaohongshu creator-profile -f yaml | head -20
      echo ""
    else
      warn "⚠️  XHS not logged in"
      echo ""
      echo "To log in to Xiaohongshu:"
      echo "  Run: opencli xiaohongshu creator-profile"
      echo "  This will open a browser window for you to log in."
      echo ""
      echo "Alternatively, skip XHS features for now with:"
      echo "  ./scripts/start-mac-dev.sh --skip-xhs-login"
      echo ""

      # Ask user if they want to continue anyway
      read -p "Continue without XHS login? [Y/n] " -n 1 -r
      echo
      if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
        error "Aborting. Please log in to Xiaohongshu first."
        exit 1
      fi
      warn "Continuing without XHS authentication. XHS features will not work."
    fi
  fi
else
  warn "Skipping Xiaohongshu login."
fi

# Start backend and frontend
BACKEND_CMD="cd '$REPO_ROOT' && uv run --no-sync deepfind-web --reload"
FRONTEND_CMD="cd '$WEB_ROOT' && npm run dev -- --host"

if [[ "$USE_TMUX" == "true" ]] && command_exists tmux; then
  SESSION_NAME="deepfind-dev"

  # Check if session already exists
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    warn "tmux session '$SESSION_NAME' already exists."
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2. Kill and recreate: tmux kill-session -t $SESSION_NAME && $0"
    exit 1
  fi

  info "Starting backend and frontend in tmux session: $SESSION_NAME"

  # Create new tmux session with backend in first pane
  tmux new-session -d -s "$SESSION_NAME" -n "deepfind" -c "$REPO_ROOT"
  tmux send-keys -t "$SESSION_NAME:0.0" "$BACKEND_CMD" C-m

  # Split window vertically and start frontend
  tmux split-window -v -t "$SESSION_NAME:0" -c "$WEB_ROOT"
  tmux send-keys -t "$SESSION_NAME:0.1" "$FRONTEND_CMD" C-m

  # Adjust pane sizes (backend gets 60%, frontend gets 40%)
  tmux resize-pane -t "$SESSION_NAME:0.0" -y 60%

  success "Development environment started in tmux session: $SESSION_NAME"
  echo ""
  echo "To attach to the session:"
  echo "  tmux attach -t $SESSION_NAME"
  echo ""
  echo "To detach from the session:"
  echo "  Press: Ctrl+B, then D"
  echo ""
  echo "To kill the session:"
  echo "  tmux kill-session -t $SESSION_NAME"

  # Auto-attach to the session
  tmux attach -t "$SESSION_NAME"

elif command_exists osascript; then
  # Use AppleScript to open separate Terminal windows (macOS fallback)
  info "Opening backend in new Terminal window..."
  osascript <<EOF
tell application "Terminal"
    do script "cd '$REPO_ROOT' && uv run deepfind-web --reload"
    activate
end tell
EOF

  sleep 1

  info "Opening frontend in new Terminal window..."
  osascript <<EOF
tell application "Terminal"
    do script "cd '$WEB_ROOT' && npm run dev -- --host"
    activate
end tell
EOF

  success "Backend and frontend started in separate Terminal windows."

else
  # Fallback: print manual instructions
  warn "Neither tmux nor osascript is available."
  echo ""
  echo "Please run the following commands in separate terminal windows:"
  echo ""
  echo "Terminal 1 (Backend):"
  echo "  cd '$REPO_ROOT'"
  echo "  uv run deepfind-web --reload"
  echo ""
  echo "Terminal 2 (Frontend):"
  echo "  cd '$WEB_ROOT'"
  echo "  npm run dev -- --host"
fi
