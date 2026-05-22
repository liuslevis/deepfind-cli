#!/usr/bin/env bash
set -euo pipefail

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

info "Xiaohongshu Login Helper"
echo ""

# Check if xhs command exists
if ! command -v xhs &>/dev/null; then
  error "xhs command not found. Install it with:"
  echo "  uv tool install xiaohongshu-cli"
  exit 1
fi

# Check current login status
info "Checking current login status..."
if xhs whoami &>/dev/null; then
  success "Already logged in to Xiaohongshu!"
  xhs whoami
  exit 0
fi

warn "Not currently logged in to Xiaohongshu."
echo ""

# Provide instructions
info "Login Instructions:"
echo ""
echo "Method 1: Browser Cookie Extraction (Recommended)"
echo "  1. Make sure you're logged into Xiaohongshu in Chrome, Safari, or Firefox"
echo "  2. Visit: https://www.xiaohongshu.com"
echo "  3. Log in if not already logged in"
echo "  4. Run: xhs login"
echo ""
echo "Method 2: Manual Cookie Entry"
echo "  1. Log into https://www.xiaohongshu.com in your browser"
echo "  2. Open Developer Tools (F12 or Cmd+Opt+I)"
echo "  3. Go to Application > Cookies > https://www.xiaohongshu.com"
echo "  4. Copy the cookie values"
echo ""

# Ask user what they want to do
echo "What would you like to do?"
echo "  1. Open Xiaohongshu in browser and try cookie extraction"
echo "  2. Try cookie extraction now (if already logged in browser)"
echo "  3. Exit and login manually later"
echo ""

read -p "Enter choice [1-3]: " -n 1 -r choice
echo ""

case $choice in
  1)
    info "Opening Xiaohongshu in your default browser..."
    open "https://www.xiaohongshu.com"
    echo ""
    warn "Please log in to Xiaohongshu in the browser, then press Enter to continue..."
    read
    info "Attempting cookie extraction..."
    if xhs login; then
      success "Login successful!"
      xhs whoami
    else
      error "Login failed. Please try again or login manually with: xhs login"
      exit 1
    fi
    ;;
  2)
    info "Attempting cookie extraction..."
    if xhs login; then
      success "Login successful!"
      xhs whoami
    else
      error "Login failed."
      echo ""
      echo "Possible solutions:"
      echo "  - Make sure you're logged into Xiaohongshu in Chrome/Safari/Firefox"
      echo "  - Try clearing browser cache and logging in again"
      echo "  - Restart your browser"
      exit 1
    fi
    ;;
  3)
    info "Exiting. You can login later with:"
    echo "  xhs login"
    echo ""
    echo "Or skip XHS features with:"
    echo "  ./scripts/start-mac-dev.sh --skip-xhs-login"
    exit 0
    ;;
  *)
    error "Invalid choice."
    exit 1
    ;;
esac

success "All done! You can now use XHS features."
