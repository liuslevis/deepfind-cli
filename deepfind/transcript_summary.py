from __future__ import annotations

from .llm_transport import CHAT_COMPLETIONS_API, TruncatedCompletionError, complete_text

BILI_TRANSCRIPT_SUMMARY_MODEL = "qwen-plus"
BILI_TRANSCRIPT_SUMMARY_CHUNK_CHARS = 12_000
# Reasoning models spend part of this budget on hidden thinking tokens, so the
# ceilings must stay well above the size of the visible answer.
QUERY_CHUNK_SUMMARY_MAX_TOKENS = 8_000
QUERY_MERGE_SUMMARY_MAX_TOKENS = 16_000
DOCUMENT_CHUNK_NOTES_MAX_TOKENS = 8_000
DOCUMENT_COMPOSE_MAX_TOKENS = 32_000


class TranscriptSummaryError(RuntimeError):
    """Raised when transcript summarization fails."""


def summarize_transcript_for_query(
    client,
    *,
    transcript: str,
    query: str,
    transcript_path: str = "",
    model: str = BILI_TRANSCRIPT_SUMMARY_MODEL,
    api_mode: str = CHAT_COMPLETIONS_API,
) -> tuple[str, int]:
    normalized_query = query.strip()
    normalized_transcript = transcript.strip()
    if not normalized_query:
        raise TranscriptSummaryError("query cannot be empty")
    if not normalized_transcript:
        raise TranscriptSummaryError("transcript cannot be empty")

    chunks = _chunk_text(normalized_transcript, BILI_TRANSCRIPT_SUMMARY_CHUNK_CHARS)
    chunk_summaries: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_summaries.append(
            _summarize_chunk(
                client,
                model=model,
                query=normalized_query,
                chunk=chunk,
                index=index,
                total=len(chunks),
                transcript_path=transcript_path,
                api_mode=api_mode,
            )
        )

    if len(chunk_summaries) == 1:
        return chunk_summaries[0], 1
    return (
        _merge_chunk_summaries(
            client,
            model=model,
            query=normalized_query,
            chunk_summaries=chunk_summaries,
            api_mode=api_mode,
        ),
        len(chunks),
    )


def summarize_transcript_to_markdown(
    client,
    *,
    transcript: str,
    title: str = "",
    focus: str = "",
    source_url: str = "",
    model: str = BILI_TRANSCRIPT_SUMMARY_MODEL,
    api_mode: str = CHAT_COMPLETIONS_API,
) -> tuple[str, int]:
    """Summarize a transcript into a standalone Markdown document."""

    normalized_transcript = transcript.strip()
    if not normalized_transcript:
        raise TranscriptSummaryError("transcript cannot be empty")

    normalized_title = title.strip()
    normalized_focus = focus.strip()
    chunks = _chunk_text(normalized_transcript, BILI_TRANSCRIPT_SUMMARY_CHUNK_CHARS)
    notes: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        notes.append(
            _summarize_document_chunk(
                client,
                model=model,
                chunk=chunk,
                index=index,
                total=len(chunks),
                title=normalized_title,
                focus=normalized_focus,
                api_mode=api_mode,
            )
        )

    document = _compose_markdown_document(
        client,
        model=model,
        notes=notes,
        title=normalized_title,
        focus=normalized_focus,
        source_url=source_url.strip(),
        api_mode=api_mode,
    )
    return document, len(chunks)


def _summarize_document_chunk(
    client,
    *,
    model: str,
    chunk: str,
    index: int,
    total: int,
    title: str,
    focus: str,
    api_mode: str,
) -> str:
    system_prompt = (
        "You take notes on one chunk of a long video transcript so a writer can assemble a "
        "faithful summary document. Keep every concrete fact: names, numbers, percentages, "
        "dates, quotes, and causal explanations. Drop greetings, ads, and filler. "
        "Never invent facts that are absent from the chunk. "
        "Write the notes in the same language as the summary focus when one is given, "
        "otherwise in Chinese."
    )
    user_prompt = (
        f"Video title: {title or '(unknown)'}\n"
        f"Summary focus: {focus or '(general summary of the whole video)'}\n"
        f"Chunk: {index}/{total}\n\n"
        "Return concise bullet notes grouped by topic. No preamble.\n\n"
        "Transcript chunk:\n"
        f"{chunk}"
    )
    notes = _chat_complete(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=DOCUMENT_CHUNK_NOTES_MAX_TOKENS,
        api_mode=api_mode,
    )
    if notes:
        return notes
    raise TranscriptSummaryError(f"chunk notes {index}/{total} returned empty output")


def _compose_markdown_document(
    client,
    *,
    model: str,
    notes: list[str],
    title: str,
    focus: str,
    source_url: str,
    api_mode: str,
) -> str:
    system_prompt = (
        "You write a standalone Markdown summary document from transcript notes. "
        "The document is emailed to a Kindle, so it must read well on its own. "
        "Use only the supplied notes, deduplicate repeated points, keep all concrete numbers, "
        "and never invent facts. "
        "Write in the same language as the summary focus when one is given, otherwise in Chinese."
    )
    notes_blob = "\n\n".join(f"[Chunk {index}]\n{item}" for index, item in enumerate(notes, start=1))
    user_prompt = (
        f"Video title: {title or '(unknown)'}\n"
        f"Source: {source_url or '(unknown)'}\n"
        f"Summary focus: {focus or '(general summary of the whole video)'}\n\n"
        "Write the document with this structure:\n"
        f"1. A level-1 heading using the video title ('# {title}' when a title is given).\n"
        "2. A short '## 摘要' section of 3-6 sentences covering the whole video.\n"
        "3. '## 要点' sections with level-2 or level-3 headings per topic, using bullets and "
        "keeping concrete numbers inline.\n"
        "4. A final '## 结论' section with the main takeaways.\n\n"
        "Return Markdown only, with no code fences around the whole document.\n\n"
        "Transcript notes:\n"
        f"{notes_blob}"
    )
    document = _chat_complete(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=DOCUMENT_COMPOSE_MAX_TOKENS,
        api_mode=api_mode,
    )
    if document:
        return document
    raise TranscriptSummaryError("markdown summary returned empty output")


def _summarize_chunk(
    client,
    *,
    model: str,
    query: str,
    chunk: str,
    index: int,
    total: int,
    transcript_path: str,
    api_mode: str,
) -> str:
    system_prompt = (
        "You compress long Bilibili transcript chunks for a research agent. "
        "Focus only on material that helps answer the research query. "
        "Preserve concrete numbers, percentages, counts, costs, gross margin, traffic estimates, "
        "coverage, demographics, and profitability judgments. "
        "Drop greetings, repetition, and filler. "
        "Reply in the same language as the research query when possible."
    )
    user_prompt = (
        f"Research query:\n{query}\n\n"
        f"Transcript path: {transcript_path or '(unknown)'}\n"
        f"Chunk: {index}/{total}\n\n"
        "Return a compact research note with these sections:\n"
        "Relevant points:\n"
        "Key numbers:\n"
        "Profitability judgment:\n"
        "Uncertainty:\n\n"
        "If a section has no relevant information in this chunk, write 'none'.\n\n"
        "Transcript chunk:\n"
        f"{chunk}"
    )
    summary = _chat_complete(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=QUERY_CHUNK_SUMMARY_MAX_TOKENS,
        api_mode=api_mode,
    )
    if summary:
        return summary
    raise TranscriptSummaryError(f"chunk summary {index}/{total} returned empty output")


def _merge_chunk_summaries(
    client,
    *,
    model: str,
    query: str,
    chunk_summaries: list[str],
    api_mode: str,
) -> str:
    system_prompt = (
        "You merge partial transcript summaries for a research agent. "
        "Answer the research query using only the provided chunk summaries. "
        "Deduplicate repeated points, keep all concrete numbers and caveats, and stay concise. "
        "Reply in the same language as the research query when possible."
    )
    summary_blob = "\n\n".join(
        f"[Chunk {index}]\n{item}" for index, item in enumerate(chunk_summaries, start=1)
    )
    user_prompt = (
        f"Research query:\n{query}\n\n"
        "Produce one compact final note with these sections:\n"
        "Overview:\n"
        "Key numbers:\n"
        "Profitability judgment:\n"
        "Open questions:\n\n"
        "Chunk summaries:\n"
        f"{summary_blob}"
    )
    summary = _chat_complete(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=QUERY_MERGE_SUMMARY_MAX_TOKENS,
        api_mode=api_mode,
    )
    if summary:
        return summary
    raise TranscriptSummaryError("merged transcript summary returned empty output")


def _chat_complete(
    client,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    api_mode: str,
) -> str:
    budget = max_tokens
    for attempt in range(2):
        try:
            return complete_text(
                client,
                api_mode=api_mode,
                model=model,
                instructions=system_prompt,
                user_input=user_prompt,
                max_output_tokens=budget,
            )
        except TruncatedCompletionError as exc:
            if attempt == 0:
                budget *= 2
                continue
            raise TranscriptSummaryError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - API failures are mocked in tests
            raise TranscriptSummaryError(str(exc) or "transcript summarization failed") from exc
    raise TranscriptSummaryError("transcript summarization failed")



def _chunk_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            split_index = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end))
            if split_index > start + limit // 2:
                end = split_index
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks or [text[:limit].strip()]
