from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Semaphore

from .web_fetch import (
    WEB_FETCH_MAX_MARKDOWN_CHARS,
    EmptyWebContentError,
    HttpWebFetchError,
    PreparedWebDocument,
    WebFetchBlockedError,
    WebFetchError,
    WebFetchTimeoutError,
    _detect_bot_challenge,
    _html_to_markdown,
    _normalized_content_type,
    _truncate_markdown,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE_DIR = REPO_ROOT / "tmp" / "browser_profile"
_BROWSER_FETCH_SEMAPHORE = Semaphore(1)


class MissingBrowserDependencyError(WebFetchError):
    error_code = "missing_dependency"


def _interactive_wait_seconds(timeout: int) -> int:
    raw = os.environ.get("DEEPFIND_BROWSER_INTERACTIVE_WAIT", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return max(15, min(int(timeout), 120))


def _page_snapshot(page, response) -> tuple[str, str, str, int, dict]:
    html = page.content()
    title = page.title()
    final_url = page.url
    status_code = response.status if response is not None else 0
    headers = response.headers if response is not None else {}
    return html, title, final_url, status_code, headers


def _wait_for_manual_clearance(page, timeout_seconds: int) -> tuple[str, str, str, int, dict]:
    deadline = time.time() + max(timeout_seconds, 0)
    last_response = None
    last_snapshot = _page_snapshot(page, last_response)
    while time.time() < deadline:
        html, title, final_url, status_code, headers = _page_snapshot(page, last_response)
        block_reason = _detect_bot_challenge(status_code, html)
        if not block_reason:
            return html, title, final_url, status_code, headers
        time.sleep(2)
        try:
            last_response = page.reload(wait_until="domcontentloaded", timeout=15_000)
            try:
                page.wait_for_load_state("networkidle", timeout=8_000)
            except Exception:
                pass
        except Exception:
            last_response = None
        last_snapshot = _page_snapshot(page, last_response)
    return last_snapshot


def fetch_web_document_browser(
    url: str,
    timeout: int,
    *,
    headless: bool = True,
    profile_dir: str | None = None,
) -> PreparedWebDocument:
    """Fetch a page by rendering it in a real browser (Playwright).

    Useful for JS-heavy pages or sites that require cookies that normal http fetches
    don't have. The browser session is persisted under tmp/ by default.
    """
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise MissingBrowserDependencyError(
            "Playwright is not installed. Install with: pip install 'deepfind-cli[browser]' "
            "then run: playwright install (or use a system Chrome channel)."
        ) from exc

    profile_root = (
        Path(profile_dir).expanduser()
        if profile_dir
        else Path(os.environ.get("DEEPFIND_BROWSER_PROFILE_DIR", "")).expanduser()
        if os.environ.get("DEEPFIND_BROWSER_PROFILE_DIR")
        else DEFAULT_PROFILE_DIR
    )
    if not profile_root.is_absolute():
        profile_root = REPO_ROOT / profile_root
    profile_root.mkdir(parents=True, exist_ok=True)

    timeout_ms = max(int(timeout), 1) * 1000

    _BROWSER_FETCH_SEMAPHORE.acquire()
    try:
        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(
                str(profile_root),
                headless=headless,
                channel="chrome" if os.name == "nt" else None,
            )
            try:
                page = context.new_page()
                response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                try:
                    page.wait_for_load_state("networkidle", timeout=min(10_000, timeout_ms))
                except PlaywrightTimeoutError:
                    pass
                try:
                    page.wait_for_load_state("load", timeout=min(10_000, timeout_ms))
                except PlaywrightTimeoutError:
                    pass
                html, title, final_url, status_code, headers = _page_snapshot(page, response)
                if not headless and _detect_bot_challenge(status_code, html):
                    html, title, final_url, status_code, headers = _wait_for_manual_clearance(
                        page,
                        _interactive_wait_seconds(timeout),
                    )
            finally:
                try:
                    context.close()
                except Exception:
                    pass
    except PlaywrightTimeoutError as exc:
        raise WebFetchTimeoutError(f"browser request timed out for {url}") from exc
    except Exception as exc:
        raise WebFetchError(str(exc) or f"browser fetch failed for {url}") from exc
    finally:
        _BROWSER_FETCH_SEMAPHORE.release()

    content_type = _normalized_content_type(headers.get("content-type") if isinstance(headers, dict) else None)
    block_reason = _detect_bot_challenge(status_code, html)
    if block_reason:
        raise WebFetchBlockedError(
            f"blocked by anti-bot challenge at {url} (status {status_code or 'unknown'}): {block_reason}"
        )
    if status_code >= 400:
        raise HttpWebFetchError(f"HTTP {status_code} while fetching {url}")

    extracted_title, markdown = _html_to_markdown(html)
    final_title = (title or extracted_title).strip()
    if not markdown:
        raise EmptyWebContentError(
            f"no readable text found at {url} (status {status_code or 'unknown'})"
        )

    full_length = len(markdown)
    truncated_markdown, truncated = _truncate_markdown(markdown, WEB_FETCH_MAX_MARKDOWN_CHARS)
    normalized_type = content_type or "text/html"
    return PreparedWebDocument(
        url=url,
        final_url=final_url or url,
        title=final_title,
        content_type=normalized_type,
        markdown=truncated_markdown,
        truncated=truncated,
        markdown_chars=full_length,
    )


__all__ = ["fetch_web_document_browser", "MissingBrowserDependencyError"]
