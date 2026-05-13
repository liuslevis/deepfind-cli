import type { ChatListResponse, ChatMode, ModelTarget, ProgressEvent, WebChatDetail } from "./types";

async function readErrorMessage(response: Response): Promise<string> {
  const text = await response.text();
  if (!text) {
    return response.statusText || `Request failed (${response.status})`;
  }
  try {
    const parsed = JSON.parse(text) as { detail?: unknown; message?: unknown };
    if (typeof parsed.detail === "string" && parsed.detail.trim()) {
      return parsed.detail;
    }
    if (typeof parsed.message === "string" && parsed.message.trim()) {
      return parsed.message;
    }
  } catch {
    // Non-JSON error bodies should fall back to the raw response text.
  }
  return text;
}

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }
  return (await response.json()) as T;
}

function parseEventBlock(block: string): ProgressEvent | null {
  const lines = block
    .split("\n")
    .map((line) => line.trimEnd())
    .filter(Boolean);
  if (lines.length === 0) {
    return null;
  }

  let type = "message";
  const dataLines: string[] = [];
  for (const line of lines) {
    if (line.startsWith("event:")) {
      type = line.slice("event:".length).trim();
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trim());
    }
  }

  const payload = dataLines.join("\n");
  if (!payload) {
    return null;
  }
  const parsed = JSON.parse(payload) as { timestamp?: string; data?: Record<string, unknown> };
  return {
    type,
    timestamp: parsed.timestamp ?? new Date().toISOString(),
    data: parsed.data ?? {},
  };
}

export async function listChats(): Promise<ChatListResponse> {
  const response = await fetch("/api/chats");
  return readJson<ChatListResponse>(response);
}

export async function createChat(title?: string): Promise<WebChatDetail> {
  const response = await fetch("/api/chats", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(title ? { title } : {}),
  });
  const payload = await readJson<{ chat: WebChatDetail }>(response);
  return payload.chat;
}

export async function getChat(chatId: string): Promise<WebChatDetail> {
  const response = await fetch(`/api/chats/${chatId}`);
  const payload = await readJson<{ chat: WebChatDetail }>(response);
  return payload.chat;
}

export async function deleteChat(chatId: string): Promise<void> {
  const response = await fetch(`/api/chats/${chatId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }
}

export async function streamChatMessage(
  chatId: string,
  payload: { content: string; mode: ChatMode; model_target: ModelTarget },
  onEvent: (event: ProgressEvent) => void,
  options?: { signal?: AbortSignal },
): Promise<void> {
  const response = await fetch(`/api/chats/${chatId}/messages/stream`, {
    method: "POST",
    headers: {
      Accept: "text/event-stream",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal: options?.signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(await readErrorMessage(response));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

    while (buffer.includes("\n\n")) {
      const separatorIndex = buffer.indexOf("\n\n");
      const block = buffer.slice(0, separatorIndex);
      buffer = buffer.slice(separatorIndex + 2);
      const event = parseEventBlock(block);
      if (event) {
        onEvent(event);
      }
    }
  }

  const leftover = buffer.trim();
  if (leftover) {
    const event = parseEventBlock(leftover);
    if (event) {
      onEvent(event);
    }
  }
}
