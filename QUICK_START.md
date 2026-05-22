# Quick Start Guide - Development Environment

## ✅ Status

Your development environment is now **RUNNING**!

- **Backend**: http://localhost:8000 ✅
- **Frontend**: http://localhost:5173 ✅

## Next Steps

### 1. Access the Web UI

Open your browser to:
```
http://localhost:5173
```

### 2. Fix Xiaohongshu Login (Optional)

The XHS QR code login is currently broken. Use browser cookie extraction instead:

**Option A: Using the helper script**
```bash
./scripts/fix-xhs-login.sh
```

**Option B: Manual steps**
1. Make sure you're logged into https://www.xiaohongshu.com in Chrome/Safari/Firefox
2. Run: `xhs login`
3. Verify: `xhs whoami`

See [XIAOHONGSHU_LOGIN_FIX.md](./XIAOHONGSHU_LOGIN_FIX.md) for detailed instructions.

### 3. Start Using the CLI

```bash
# Test CLI (no XHS needed)
uv run -m deepfind.cli "What's new in AI?" --num-agent 2

# Test with video (no XHS needed)
uv run -m deepfind.cli "Summarize this video https://www.youtube.com/watch?v=dQw4w9WgXcQ" --num-agent 1

# Test with XHS (requires login)
uv run -m deepfind.cli "What do people think about AI in Xiaohongshu?" --num-agent 2
```

## Viewing Logs

The backend and frontend are running in separate Terminal windows. Check those windows for logs.

## Stopping the Environment

Close the Terminal windows running:
- Backend: `uv run deepfind-web --reload`
- Frontend: `npm run dev -- --host`

Or press `Ctrl+C` in each window.

## Restarting

```bash
# Full restart with setup
./scripts/start-mac-dev.sh

# Quick restart (skip setup and XHS login)
./scripts/start-mac-dev.sh --skip-setup --skip-xhs-login

# With tmux (recommended)
# First install: brew install tmux
./scripts/start-mac-dev.sh --skip-xhs-login
# Then attach with: tmux attach -t deepfind-dev
```

## Troubleshooting

### Ports Already in Use

```bash
# Find what's using the ports
lsof -i :8000
lsof -i :5173

# Kill the processes
kill -9 <PID>
```

### Backend Not Responding

```bash
# Check backend health
curl http://localhost:8000/api/health

# Restart backend only
cd /Users/xiaojiliu/dev/deepfind-cli
uv run deepfind-web --reload
```

### Frontend Not Loading

```bash
# Restart frontend only
cd /Users/xiaojiliu/dev/deepfind-cli/web
npm run dev -- --host
```

## Development Tips

1. **Backend changes**: Auto-reload enabled (uvicorn --reload)
2. **Frontend changes**: Hot module replacement enabled (Vite HMR)
3. **Python dependencies**: Run `uv sync` after changing pyproject.toml
4. **Web dependencies**: Run `npm install` after changing package.json

## Resources

- [README.md](./README.md) - Full project documentation
- [scripts/start-mac-dev.md](./scripts/start-mac-dev.md) - Detailed setup guide
- [XIAOHONGSHU_LOGIN_FIX.md](./XIAOHONGSHU_LOGIN_FIX.md) - XHS login troubleshooting

---

**Current Session:**
- Backend: Running on http://localhost:8000
- Frontend: Running on http://localhost:5173
- XHS Login: ⚠️ Pending (see XIAOHONGSHU_LOGIN_FIX.md)
