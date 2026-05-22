from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from .config import DEFAULT_BASE_URL, DEFAULT_MODEL


DEFAULT_SLIDE_DIR = "slide"
DEFAULT_SLIDE_COUNT = 1
MAX_SLIDE_COUNT = 12
DEFAULT_TEMPLATE_NAME = "monochrome"
REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "beautiful-html-templates" / "templates"


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


def _validate_template_name(template_name: str | None) -> str:
    name = (template_name or "").strip() or DEFAULT_TEMPLATE_NAME
    template_dir = TEMPLATES_DIR / name
    if not template_dir.is_dir() or not (template_dir / "template.html").exists():
        available = sorted(
            d.name
            for d in TEMPLATES_DIR.iterdir()
            if d.is_dir() and (d / "template.html").exists()
        )
        raise SlideGenerationError(
            f"Template '{name}' not found. Available: {', '.join(available)}"
        )
    return name


def _load_template(template_name: str) -> dict[str, Any]:
    """Load a template and split it into prefix, example slides, and suffix."""
    template_dir = TEMPLATES_DIR / template_name
    html_text = (template_dir / "template.html").read_text(encoding="utf-8")

    meta: dict[str, Any] = {}
    meta_path = template_dir / "template.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    prefix, slides_html, suffix = _split_template(html_text)

    return {
        "prefix": prefix,
        "slides_html": slides_html,
        "suffix": suffix,
        "full_html": html_text,
        "meta": meta,
    }


def _split_template(html_text: str) -> tuple[str, str, str]:
    """Split template HTML into (prefix, slides, suffix).

    Finds the first ``<section`` with a ``slide`` class and the last
    ``</section>`` to delineate the slide region.
    """
    first_match = re.search(r"<section\b[^>]*class=\"[^\"]*\bslide\b", html_text)
    if not first_match:
        raise SlideGenerationError("Template has no slide sections.")

    prefix_end = first_match.start()

    last_close = html_text.rfind("</section>")
    if last_close < prefix_end:
        raise SlideGenerationError("Template has no closing </section> tags after slides.")
    suffix_start = last_close + len("</section>")

    return html_text[:prefix_end], html_text[prefix_end:suffix_start], html_text[suffix_start:]


def _build_system_prompt(template: dict[str, Any]) -> str:
    meta_json = json.dumps(template["meta"], ensure_ascii=False, indent=2) if template["meta"] else "{}"
    return (
        "You are a professional slide deck designer. You create HTML slide <section> elements "
        "for presentations using the exact design system from the template below.\n\n"
        "RULES:\n"
        "1. Output ONLY the <section> elements — no <!doctype>, <html>, <head>, <body>, <style>, or <script> tags.\n"
        "2. Use the EXACT same CSS classes, HTML structure, and data-anim/data-delay attributes "
        "from the template's slide sections.\n"
        "3. The first slide MUST be a cover/title slide.\n"
        "4. The last slide MUST be a closing/end slide.\n"
        "5. Middle slides should use varied layout types from the template "
        "(e.g. statement, list, stats, split, compare, quote, etc.).\n"
        "6. Replace ALL placeholder content with content matching the user's request.\n"
        "7. Generate EXACTLY the requested number of slides.\n"
        "8. Write content in the same language as the user's prompt.\n"
        "9. Do NOT wrap the output in markdown code blocks.\n"
        "10. Do NOT output any text before or after the <section> elements.\n\n"
        f"TEMPLATE METADATA:\n{meta_json}\n\n"
        f"FULL TEMPLATE HTML (study the CSS classes and all slide structures):\n"
        f"{template['full_html']}\n"
    )


def _extract_title_from_html(html_text: str) -> str:
    """Extract deck title from the first h1 or h2 element."""
    match = re.search(r"<h[12][^>]*>(.*?)</h[12]>", html_text, re.DOTALL)
    if match:
        title = re.sub(r"<[^>]+>", " ", match.group(1))
        return re.sub(r"\s+", " ", title).strip()
    return ""


def _title_slug(title: str, prompt: str) -> str:
    base = title or prompt
    slug = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "-", base).strip("-")[:40] or "slides"
    return slug


def _fix_slide_count_in_suffix(suffix: str, slide_count: int) -> str:
    """Replace hardcoded ``totalSlides`` constants in JS with the actual count."""
    return re.sub(
        r"(const\s+totalSlides\s*=\s*)\d+",
        rf"\g<1>{slide_count}",
        suffix,
    )


def _fix_slide_count_in_prefix(prefix: str, slide_count: int) -> str:
    """Replace hardcoded slide counter text like ``01 / 10`` in the prefix."""
    return re.sub(
        r">\s*\d{1,2}\s*/\s*\d{1,2}\s*<",
        f">01 / {slide_count:02d}<",
        prefix,
        count=1,
    )


def _extract_slides_from_html(html_text: str) -> str:
    """Extract slide ``<section>`` elements from a complete HTML file."""
    _, slides_html, _ = _split_template(html_text)
    return slides_html


def _clean_llm_output(content: str) -> str:
    """Strip markdown code fences and leading/trailing whitespace."""
    content = content.strip()
    if content.startswith("```"):
        first_newline = content.find("\n")
        content = content[first_newline + 1 :] if first_newline != -1 else content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def generate_slides(
    prompt: str,
    *,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    slide_dir: str | None = None,
    slide_count: int = DEFAULT_SLIDE_COUNT,
    template_name: str | None = None,
    timeout: int = 90,
    html_path: str | None = None,
) -> dict[str, Any]:
    prompt = _normalize_prompt(prompt)
    slide_count = _normalize_slide_count(slide_count)
    validated_template = _validate_template_name(template_name)

    resolved_key = api_key.strip()
    if not resolved_key:
        raise SlideGenerationError("api_key cannot be empty.")

    template = _load_template(validated_template)
    system_prompt = _build_system_prompt(template)

    if html_path:
        existing_path = Path(html_path)
        if not existing_path.is_absolute():
            existing_path = REPO_ROOT / existing_path
        if not existing_path.exists():
            raise SlideGenerationError(f"Existing HTML not found: {html_path}")
        existing_slides = _extract_slides_from_html(
            existing_path.read_text(encoding="utf-8")
        )
        user_prompt = (
            f"Here are the current slide sections of an existing deck:\n\n"
            f"{existing_slides}\n\n"
            f"Modification request: {prompt}\n\n"
            f"Output the complete updated set of <section> elements, keeping unchanged slides "
            f"as-is and applying the requested modifications."
        )
    else:
        user_prompt = (
            f"Create a slide deck with exactly {slide_count} slides.\n"
            f"Topic/Request: {prompt}\n\n"
            f"Output only the <section> elements."
        )

    client = OpenAI(api_key=resolved_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max(4000, slide_count * 800),
            timeout=max(timeout, 1),
        )
    except Exception as exc:
        raise SlideGenerationError(f"Slide generation request failed: {exc}") from exc

    content = response.choices[0].message.content or ""
    slides_html = _clean_llm_output(content)

    if not slides_html or "<section" not in slides_html:
        raise SlideGenerationError("LLM did not generate valid slide HTML sections.")

    actual_count = len(re.findall(r"<section\b", slides_html))
    title = _extract_title_from_html(slides_html) or prompt[:40]

    prefix = _fix_slide_count_in_prefix(template["prefix"], actual_count)
    suffix = _fix_slide_count_in_suffix(template["suffix"], actual_count)
    final_html = prefix + slides_html + suffix

    if html_path:
        output_path = Path(html_path)
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
    else:
        slug = _title_slug(title, prompt)
        slide_root = resolve_slide_root(slide_dir)
        output_dir = slide_root / slug
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "index.html"

    output_path.write_text(final_html, encoding="utf-8")

    try:
        relative_path = output_path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        relative_path = output_path.resolve()

    return {
        "title": title,
        "slide_count": actual_count,
        "html_path": str(relative_path),
        "template_name": validated_template,
    }


__all__ = [
    "DEFAULT_SLIDE_COUNT",
    "DEFAULT_SLIDE_DIR",
    "DEFAULT_TEMPLATE_NAME",
    "MAX_SLIDE_COUNT",
    "SlideGenerationError",
    "generate_slides",
    "resolve_slide_root",
]
