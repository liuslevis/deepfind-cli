# Xiaohongshu Login Fix Guide

## Problem

The `xhs login --qrcode` command fails with a Camoufox addon error:
```
camoufox.exceptions.InvalidAddonPath: manifest.json is missing
```

This is a known issue with the Camoufox browser automation library used by xiaohongshu-cli v0.6.4.

## Solution: Use Browser Cookie Extraction

Instead of QR code login, use browser cookie extraction which is more reliable.

### Step-by-Step Fix

1. **Log into Xiaohongshu in your browser:**
   ```bash
   open https://www.xiaohongshu.com
   ```
   
   - Use Chrome, Safari, or Firefox
   - Complete the login process
   - Make sure you're fully logged in (can see your profile)

2. **Extract cookies automatically:**
   ```bash
   xhs login
   ```
   
   This will automatically find and extract cookies from your browser.

3. **Verify login:**
   ```bash
   xhs whoami
   ```

### Using the Helper Script

I've created a helper script for you:

```bash
./scripts/fix-xhs-login.sh
```

This script will:
- Check your current login status
- Open Xiaohongshu in your browser
- Guide you through the cookie extraction process
- Verify the login was successful

## Alternative: Skip XHS Login

If you don't need Xiaohongshu features right now, you can skip the login:

```bash
./scripts/start-mac-dev.sh --skip-xhs-login
```

## Troubleshooting

### Cookie Extraction Fails

If `xhs login` doesn't work:

1. **Make sure you're logged in** to https://www.xiaohongshu.com in your browser
2. **Close and reopen** your browser
3. **Try a different browser** (Chrome is most reliable)
4. **Check browser permissions** - the CLI needs to read browser cookies

### Session Expired Error

If you see "Session expired", your browser cookies are outdated:

1. Go to https://www.xiaohongshu.com
2. Log out completely
3. Log in again
4. Run `xhs login` immediately after logging in

### Still Not Working?

Temporarily skip XHS features:
```bash
# Development without XHS
./scripts/start-mac-dev.sh --skip-xhs-login

# CLI usage without XHS
uv run -m deepfind.cli "your query" --num-agent 2
```

XHS-specific tools (`xhs_search_user`, `xhs_user`, `xhs_user_posts`, `xhs_read`) won't work, but all other features will function normally.

## Status Check Commands

```bash
# Check if logged in
xhs whoami

# View profile
xhs whoami --json

# Test search (requires login)
xhs search --keyword "test" --limit 1
```

## When QR Code Login Works Again

The upstream xiaohongshu-cli project may fix the Camoufox issue in a future release. To update:

```bash
uv tool upgrade xiaohongshu-cli
```

Then you can try:
```bash
xhs login --qrcode
```

## Summary

- ✅ **Recommended**: Use browser cookie extraction (`xhs login`)
- ⚠️ **Broken**: QR code login (Camoufox issue)
- ✅ **Workaround**: Skip XHS login (`--skip-xhs-login`)

For most development work, browser cookie extraction is faster and more reliable than QR code login anyway.
