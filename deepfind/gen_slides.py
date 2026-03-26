from __future__ import annotations

import hashlib
import html
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from .config import DEFAULT_BASE_URL, DEFAULT_MODEL
from .json_utils import try_load_json


DEFAULT_SLIDE_DIR = "tmp"
DEFAULT_SLIDE_COUNT = 1
MAX_SLIDE_COUNT = 12
REPO_ROOT = Path(__file__).resolve().parent.parent

SLIDE_SYSTEM_PROMPT = (
    "You create concise slide deck outlines for HTML presentations. "
    'Return JSON only using this exact schema: {"title":"string","slides":[{"title":"string","bullets":["string"]}]}. '
    "Create exactly the requested number of slides. Do not wrap the JSON in markdown."
)


class SlideGenerationError(RuntimeError):
    """Raised when slide generation fails."""


def resolve_slide_root(slide_dir: str | None) -> Path:
    raw = (slide_dir or DEFAULT_SLIDE_DIR).strip() or DEFAULT_SLIDE_DIR
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _normalize_prompt(prompt: str) -> str:
    text = prompt.strip()
    if not text:
        raise SlideGenerationError("prompt cannot be empty.")
    return text


def _normalize_slide_count(value: int | None) -> int:
    slide_count = value if value is not None else DEFAULT_SLIDE_COUNT
    if not isinstance(slide_count, int) or not 1 <= slide_count <= MAX_SLIDE_COUNT:
        raise SlideGenerationError(f"slide_count must be between 1 and {MAX_SLIDE_COUNT}.")
    return slide_count


def _file_name(title: str, prompt: str) -> str:
    base = title.strip() or prompt.strip()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", base).strip("-").lower()[:24] or "slides"
    digest = hashlib.sha1(f"{title}\n{prompt}".encode("utf-8")).hexdigest()[:8]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{slug}-{digest}.html"


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SlideGenerationError(f"{field} must be a non-empty string.")
    return value.strip()


def _normalize_outline(payload: Any, slide_count: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SlideGenerationError("Slide model returned invalid JSON.")

    title = _non_empty_string(payload.get("title"), "title")
    raw_slides = payload.get("slides")
    if not isinstance(raw_slides, list):
        raise SlideGenerationError("slides must be a list.")
    if len(raw_slides) != slide_count:
        raise SlideGenerationError(f"Expected {slide_count} slides, got {len(raw_slides)}.")

    slides: list[dict[str, Any]] = []
    for index, raw_slide in enumerate(raw_slides, 1):
        if not isinstance(raw_slide, dict):
            raise SlideGenerationError(f"slides[{index}] must be an object.")
        slide_title = _non_empty_string(raw_slide.get("title"), f"slides[{index}].title")
        raw_bullets = raw_slide.get("bullets")
        if not isinstance(raw_bullets, list):
            raise SlideGenerationError(f"slides[{index}].bullets must be a list.")
        bullets = [str(item).strip() for item in raw_bullets if str(item).strip()]
        if not bullets:
            raise SlideGenerationError(f"slides[{index}] must contain at least one bullet.")
        slides.append({"title": slide_title, "bullets": bullets[:6]})

    return {"title": title, "slides": slides}


def _slide_markup(deck_title: str, slide: dict[str, Any], index: int, total: int) -> str:
    bullets_html = "\n".join(
        f"            <li>{html.escape(bullet)}</li>"
        for bullet in slide["bullets"]
    )
    return f"""      <section class="slide" data-slide="{index}">
        <div class="slide-panel">
          <div class="slide-kicker">{html.escape(deck_title)}</div>
          <div class="slide-counter">{index + 1:02d} / {total:02d}</div>
          <h2>{html.escape(slide["title"])}</h2>
          <ul>
{bullets_html}
          </ul>
        </div>
      </section>"""


def render_slides_html(title: str, slides: list[dict[str, Any]]) -> str:
    slide_sections = "\n".join(
        _slide_markup(title, slide, index, len(slides))
        for index, slide in enumerate(slides)
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{html.escape(title)}</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: radial-gradient(circle at top, #1d4ed8 0%, #0f172a 45%, #020617 100%);
        --panel: rgba(15, 23, 42, 0.82);
        --panel-border: rgba(148, 163, 184, 0.18);
        --text: #e2e8f0;
        --muted: #bfdbfe;
        --accent: #f59e0b;
        --accent-2: #38bdf8;
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        min-height: 100vh;
        font-family: "Avenir Next", "Segoe UI", sans-serif;
        background: var(--bg);
        color: var(--text);
      }}

      .app {{
        min-height: 100vh;
        display: grid;
        place-items: center;
        padding: 24px;
      }}

      .deck {{
        width: min(100%, 1200px);
      }}

      .topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 16px;
      }}

      .brand {{
        font-size: 0.88rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--muted);
      }}

      .title {{
        font-size: clamp(1.2rem, 2vw, 1.6rem);
        font-weight: 700;
      }}

      .frame {{
        position: relative;
        width: 100%;
        aspect-ratio: 16 / 9;
        border-radius: 28px;
        overflow: hidden;
        background: linear-gradient(160deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.88));
        border: 1px solid var(--panel-border);
        box-shadow: 0 26px 70px rgba(2, 6, 23, 0.45);
      }}

      .slide {{
        position: absolute;
        inset: 0;
        display: none;
        padding: clamp(28px, 4vw, 52px);
      }}

      .slide.is-active {{
        display: block;
      }}

      .slide-panel {{
        position: relative;
        height: 100%;
        border-radius: 24px;
        padding: clamp(28px, 4vw, 48px);
        background:
          linear-gradient(135deg, rgba(56, 189, 248, 0.1), transparent 36%),
          linear-gradient(220deg, rgba(245, 158, 11, 0.13), transparent 28%),
          var(--panel);
        border: 1px solid var(--panel-border);
      }}

      .slide-panel::after {{
        content: "";
        position: absolute;
        inset: auto 28px 24px auto;
        width: 140px;
        height: 140px;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(245, 158, 11, 0.32), rgba(245, 158, 11, 0));
        pointer-events: none;
      }}

      .slide-kicker {{
        font-size: 0.82rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 18px;
      }}

      .slide-counter {{
        position: absolute;
        top: 28px;
        right: 28px;
        font-weight: 700;
        color: var(--accent-2);
      }}

      h2 {{
        margin: 0;
        max-width: 80%;
        font-size: clamp(2rem, 4vw, 3.5rem);
        line-height: 1.05;
      }}

      ul {{
        margin: 28px 0 0;
        padding-left: 1.2em;
        max-width: 72%;
        font-size: clamp(1rem, 1.8vw, 1.5rem);
        line-height: 1.5;
      }}

      li + li {{
        margin-top: 0.7em;
      }}

      .controls {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-top: 16px;
      }}

      .hint {{
        color: var(--muted);
        font-size: 0.95rem;
      }}

      .buttons {{
        display: flex;
        gap: 10px;
      }}

      button {{
        appearance: none;
        border: 0;
        border-radius: 999px;
        padding: 12px 18px;
        font: inherit;
        font-weight: 700;
        cursor: pointer;
        color: #0f172a;
        background: linear-gradient(135deg, #f8fafc, #bae6fd);
      }}

      button:disabled {{
        cursor: default;
        opacity: 0.45;
      }}

      .pager {{
        min-width: 5ch;
        text-align: center;
        font-weight: 700;
        color: var(--accent);
      }}

      @media (max-width: 860px) {{
        .topbar,
        .controls {{
          flex-direction: column;
          align-items: flex-start;
        }}

        .frame {{
          aspect-ratio: auto;
          min-height: 70vh;
        }}

        .slide {{
          position: static;
          min-height: 70vh;
        }}

        .slide-panel {{
          min-height: calc(70vh - 48px);
        }}

        h2,
        ul {{
          max-width: 100%;
        }}
      }}
    </style>
  </head>
  <body>
    <div class="app">
      <div class="deck">
        <div class="topbar">
          <div>
            <div class="brand">deepfind slides</div>
            <div class="title">{html.escape(title)}</div>
          </div>
          <div class="pager"><span id="current">01</span> / <span id="total">{len(slides):02d}</span></div>
        </div>
        <div class="frame">
{slide_sections}
        </div>
        <div class="controls">
          <div class="hint">Use arrow keys, Page Up/Page Down, Home, or End to navigate.</div>
          <div class="buttons">
            <button id="prev" type="button">Previous</button>
            <button id="next" type="button">Next</button>
          </div>
        </div>
      </div>
    </div>
    <script>
      const slides = Array.from(document.querySelectorAll(".slide"));
      const prevButton = document.getElementById("prev");
      const nextButton = document.getElementById("next");
      const current = document.getElementById("current");
      const total = document.getElementById("total");
      let index = 0;

      function clamp(value) {{
        return Math.max(0, Math.min(value, slides.length - 1));
      }}

      function indexFromHash() {{
        const match = window.location.hash.match(/^#slide-(\\d+)$/);
        if (!match) {{
          return 0;
        }}
        return clamp(Number(match[1]) - 1);
      }}

      function show(nextIndex) {{
        index = clamp(nextIndex);
        slides.forEach((slide, slideIndex) => {{
          slide.classList.toggle("is-active", slideIndex === index);
        }});
        current.textContent = String(index + 1).padStart(2, "0");
        total.textContent = String(slides.length).padStart(2, "0");
        prevButton.disabled = index === 0;
        nextButton.disabled = index === slides.length - 1;
        const targetHash = `#slide-${{index + 1}}`;
        if (window.location.hash !== targetHash) {{
          window.history.replaceState(null, "", targetHash);
        }}
      }}

      prevButton.addEventListener("click", () => show(index - 1));
      nextButton.addEventListener("click", () => show(index + 1));
      document.addEventListener("keydown", (event) => {{
        if (["ArrowRight", "PageDown", " "].includes(event.key)) {{
          event.preventDefault();
          show(index + 1);
        }}
        if (["ArrowLeft", "PageUp"].includes(event.key)) {{
          event.preventDefault();
          show(index - 1);
        }}
        if (event.key === "Home") {{
          event.preventDefault();
          show(0);
        }}
        if (event.key === "End") {{
          event.preventDefault();
          show(slides.length - 1);
        }}
      }});
      window.addEventListener("hashchange", () => show(indexFromHash()));
      show(indexFromHash());
    </script>
  </body>
</html>
"""


def generate_slides(
    prompt: str,
    *,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    slide_dir: str | None = None,
    slide_count: int = DEFAULT_SLIDE_COUNT,
    timeout: int = 90,
) -> dict[str, Any]:
    prompt = _normalize_prompt(prompt)
    slide_count = _normalize_slide_count(slide_count)
    resolved_key = api_key.strip()
    if not resolved_key:
        raise SlideGenerationError("api_key cannot be empty.")

    client = OpenAI(api_key=resolved_key, base_url=base_url)
    user_prompt = (
        f"Create a standalone slide deck outline with exactly {slide_count} slides.\n"
        f"Keep each slide concise and presentation-ready.\n"
        f"Request:\n{prompt}"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SLIDE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max(600, slide_count * 220),
            timeout=max(timeout, 1),
        )
    except Exception as exc:
        raise SlideGenerationError(f"Slide outline request failed: {exc}") from exc

    content = response.choices[0].message.content or ""
    outline = _normalize_outline(try_load_json(content), slide_count)
    html_text = render_slides_html(outline["title"], outline["slides"])

    slide_root = resolve_slide_root(slide_dir)
    slide_root.mkdir(parents=True, exist_ok=True)
    html_path = slide_root / _file_name(outline["title"], prompt)
    html_path.write_text(html_text, encoding="utf-8")

    return {
        "title": outline["title"],
        "slide_count": slide_count,
        "html_path": str(html_path),
    }


__all__ = [
    "DEFAULT_SLIDE_COUNT",
    "DEFAULT_SLIDE_DIR",
    "MAX_SLIDE_COUNT",
    "SlideGenerationError",
    "generate_slides",
    "render_slides_html",
    "resolve_slide_root",
]
