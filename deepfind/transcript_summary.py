from __future__ import annotations

BILI_TRANSCRIPT_SUMMARY_MODEL = "qwen-plus"
BILI_TRANSCRIPT_SUMMARY_CHUNK_CHARS = 12_000


class TranscriptSummaryError(RuntimeError):
    """Raised when transcript summarization fails."""


def summarize_transcript_for_query(
    client,
    *,
    transcript: str,
    query: str,
    transcript_path: str = "",
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
                query=normalized_query,
                chunk=chunk,
                index=index,
                total=len(chunks),
                transcript_path=transcript_path,
            )
        )

    if len(chunk_summaries) == 1:
        return chunk_summaries[0], 1
    return _merge_chunk_summaries(client, query=normalized_query, chunk_summaries=chunk_summaries), len(chunks)


def _summarize_chunk(
    client,
    *,
    query: str,
    chunk: str,
    index: int,
    total: int,
    transcript_path: str,
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
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=900,
    )
    if summary:
        return summary
    raise TranscriptSummaryError(f"chunk summary {index}/{total} returned empty output")


def _merge_chunk_summaries(client, *, query: str, chunk_summaries: list[str]) -> str:
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
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=1_200,
    )
    if summary:
        return summary
    raise TranscriptSummaryError("merged transcript summary returned empty output")


def _chat_complete(client, *, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    try:
        response = client.chat.completions.create(
            model=BILI_TRANSCRIPT_SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
    except Exception as exc:  # pragma: no cover - API failures are mocked in tests
        raise TranscriptSummaryError(str(exc) or "transcript summarization failed") from exc

    return (response.choices[0].message.content or "").strip()


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
