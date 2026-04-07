from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - covered indirectly when dependency is missing
    PdfReader = None

WEB_FETCH_MODEL = "qwen-plus"
WEB_FETCH_MAX_MARKDOWN_CHARS = 12_000
_DEFAULT_USER_AGENT = "deepfind-cli/0.1 (+https://github.com/openai/codex)"
_HTMLISH_MARKERS = ("text/html", "application/xhtml+xml")
_MARKDOWNISH_MARKERS = ("text/markdown", "text/x-markdown")
_PDFISH_MARKERS = ("application/pdf", "application/x-pdf")
_TEXTISH_MARKERS = ("text/plain",)
_NOISE_SELECTORS = (
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "aside",
    "form",
    "iframe",
    "svg",
    "canvas",
    "button",
    "input",
    "select",
    "textarea",
)


class WebFetchError(Exception):
    error_code = "request_failed"


class WebFetchTimeoutError(WebFetchError):
    error_code = "timeout"


class UnsupportedContentTypeError(WebFetchError):
    error_code = "unsupported_content_type"


class MissingPdfDependencyError(WebFetchError):
    error_code = "missing_dependency"


class EmptyWebContentError(WebFetchError):
    error_code = "empty_content"


class HttpWebFetchError(WebFetchError):
    error_code = "http_error"


class WebSummaryError(WebFetchError):
    error_code = "summary_failed"


@dataclass(frozen=True)
class PreparedWebDocument:
    url: str
    final_url: str
    title: str
    content_type: str
    markdown: str
    truncated: bool
    markdown_chars: int


def fetch_web_document(url: str, timeout: int) -> PreparedWebDocument:
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers={
                "User-Agent": _DEFAULT_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/pdf,text/markdown,text/plain;q=0.9,*/*;q=0.1",
            },
        ) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise WebFetchTimeoutError(f"request timed out for {url}") from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise HttpWebFetchError(f"HTTP {status} while fetching {url}") from exc
    except httpx.HTTPError as exc:
        raise WebFetchError(str(exc) or f"request failed for {url}") from exc

    content_type = _normalized_content_type(response.headers.get("content-type"))
    body_bytes = response.content
    is_pdf = _looks_like_pdf(content_type, str(response.url), body_bytes)
    title = ""

    if is_pdf:
        title, markdown = _pdf_to_markdown(body_bytes)
    else:
        body_text = response.text
        looks_like_html = _looks_like_html(content_type, body_text)

        if looks_like_html:
            title, markdown = _html_to_markdown(body_text)
        elif _is_text_like(content_type):
            markdown = _normalize_text(body_text)
        else:
            raise UnsupportedContentTypeError(
                f"unsupported content type: {content_type or 'unknown'}"
            )

    if not markdown:
        raise EmptyWebContentError(f"no readable text found at {url}")

    normalized_type = content_type
    if is_pdf and content_type in {"", "application/octet-stream"}:
        normalized_type = "application/pdf"

    full_length = len(markdown)
    truncated_markdown, truncated = _truncate_markdown(markdown, WEB_FETCH_MAX_MARKDOWN_CHARS)

    return PreparedWebDocument(
        url=url,
        final_url=str(response.url),
        title=title,
        content_type=normalized_type or "text/plain",
        markdown=truncated_markdown,
        truncated=truncated,
        markdown_chars=full_length,
    )


def _normalized_content_type(value: str | None) -> str:
    if not value:
        return ""
    return value.split(";", 1)[0].strip().lower()


def _looks_like_pdf(content_type: str, url: str, body_bytes: bytes) -> bool:
    if any(marker in content_type for marker in _PDFISH_MARKERS):
        return True
    if body_bytes.lstrip().startswith(b"%PDF-"):
        return True
    lower_url = url.lower().split("#", 1)[0].split("?", 1)[0]
    return content_type in {"", "application/octet-stream"} and lower_url.endswith(".pdf")


def _pdf_to_markdown(pdf_bytes: bytes) -> tuple[str, str]:
    if PdfReader is None:
        raise MissingPdfDependencyError("PDF support requires the 'pypdf' package")

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as exc:
        raise WebFetchError("failed to parse PDF content") from exc

    metadata = getattr(reader, "metadata", None)
    raw_title = getattr(metadata, "title", "") if metadata is not None else ""
    if not raw_title and hasattr(metadata, "get"):
        raw_title = metadata.get("/Title", "")
    title = _normalize_text(raw_title or "")

    sections: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        normalized_page = _normalize_text(page_text)
        if normalized_page:
            sections.append(f"## Page {index}\n\n{normalized_page}")

    markdown = "\n\n".join(sections)
    if title and markdown:
        markdown = f"# {title}\n\n{markdown}"
    return title, markdown


def _looks_like_html(content_type: str, body_text: str) -> bool:
    if any(marker in content_type for marker in _HTMLISH_MARKERS):
        return True
    snippet = body_text[:512].lower()
    return "<html" in snippet or "<body" in snippet or "<article" in snippet


def _is_text_like(content_type: str) -> bool:
    if not content_type:
        return True
    if any(marker in content_type for marker in _MARKDOWNISH_MARKERS):
        return True
    if any(marker in content_type for marker in _TEXTISH_MARKERS):
        return True
    return content_type.startswith("text/")


def _html_to_markdown(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = _normalize_text(soup.title.get_text(" ", strip=True) if soup.title else "")

    for selector in _NOISE_SELECTORS:
        for node in soup.select(selector):
            node.decompose()

    root = _best_root(soup)
    markdown = markdownify(
        str(root),
        heading_style="ATX",
        bullets="-",
        strip=["img"],
    )
    return title, _normalize_text(markdown)


def _best_root(soup: BeautifulSoup):
    for selector in ("article", "main", "body"):
        node = soup.select_one(selector)
        if node and node.get_text(" ", strip=True):
            return node
    return soup


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.rstrip() for line in text.split("\n")]
    normalized = "\n".join(lines).strip()
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    return normalized


def _truncate_markdown(markdown: str, limit: int) -> tuple[str, bool]:
    if len(markdown) <= limit:
        return markdown, False

    truncated = markdown[:limit]
    split_index = max(truncated.rfind("\n\n"), truncated.rfind("\n"))
    if split_index >= limit // 2:
        truncated = truncated[:split_index]
    return truncated.rstrip(), True


def _truncate_plain_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    shortened = text[:limit].rsplit(" ", 1)[0].strip()
    return shortened or text[:limit].strip()


def summarize_web_document(client, *, prompt: str, document: PreparedWebDocument) -> str:
    system_prompt = (
        "You summarize fetched web pages for a research agent. "
        "Answer the user's prompt using only the provided page content. "
        "Be concise, factual, and explicit about uncertainty. "
        "Do not mention that you are an AI or quote the full page."
    )
    user_prompt = (
        f"User prompt:\n{prompt.strip() or 'Summarize the page.'}\n\n"
        f"URL: {document.final_url}\n"
        f"Title: {document.title or '(untitled)'}\n"
        f"Content-Type: {document.content_type}\n"
        f"Truncated: {'yes' if document.truncated else 'no'}\n\n"
        "Page content in Markdown:\n"
        f"{document.markdown}"
    )
    try:
        response = client.chat.completions.create(
            model=WEB_FETCH_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=500,
        )
    except Exception as exc:  # pragma: no cover - API failures are mocked in tests
        raise WebSummaryError(str(exc) or "summary generation failed") from exc

    summary = (response.choices[0].message.content or "").strip()
    if summary:
        return summary

    fallback = _truncate_plain_text(document.markdown, 600)
    if fallback:
        return fallback
    raise WebSummaryError("summary generation returned empty output")
