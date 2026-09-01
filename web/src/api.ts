import type {
  ChatListResponse,
  ChatMode,
  ModelTarget,
  ProgressEvent,
  ResearchMode,
  SheetRange,
  WebChatDetail,
  WordDocument,
  WorkbookMetadata,
  WorkspaceListing,
  WorkspaceStatus,
} from "./types";

const TOKEN_KEY = "deepfind_auth_token";

export function getAuthToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setAuthToken(token: string | null): void {
  try {
    if (token) {
      localStorage.setItem(TOKEN_KEY, token);
    } else {
      localStorage.removeItem(TOKEN_KEY);
    }
  } catch {
    /* ignore */
  }
}

function authHeaders(): Record<string, string> {
  const token = getAuthToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function readErrorMessage(response: Response): Promise<string> {
  const text = await response.text();
  if (!text) {
    return response.statusText || `Request failed (${response.status})`;
  }
  try {
    const parsed = JSON.parse(text) as {
      detail?: unknown;
      message?: unknown;
    };
    if (typeof parsed.detail === "string" && parsed.detail.trim()) {
      return parsed.detail;
    }
    if (
      parsed.detail &&
      typeof parsed.detail === "object" &&
      "message" in parsed.detail &&
      typeof parsed.detail.message === "string"
    ) {
      return parsed.detail.message;
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

export interface HealthResponse {
  status: string;
  requires_token: boolean;
}

export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch("/api/health");
  return readJson<HealthResponse>(response);
}

export async function listChats(): Promise<ChatListResponse> {
  const response = await fetch("/api/chats", { headers: authHeaders() });
  return readJson<ChatListResponse>(response);
}

export async function createChat(title?: string): Promise<WebChatDetail> {
  const response = await fetch("/api/chats", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
    },
    body: JSON.stringify(title ? { title } : {}),
  });
  const payload = await readJson<{ chat: WebChatDetail }>(response);
  return payload.chat;
}

export async function getChat(chatId: string): Promise<WebChatDetail> {
  const response = await fetch(`/api/chats/${chatId}`, { headers: authHeaders() });
  const payload = await readJson<{ chat: WebChatDetail }>(response);
  return payload.chat;
}

export async function deleteChat(chatId: string): Promise<void> {
  const response = await fetch(`/api/chats/${chatId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }
}

export async function getWorkspaceStatus(chatId: string): Promise<WorkspaceStatus> {
  return readJson<WorkspaceStatus>(
    await fetch(`/api/chats/${chatId}/workspace`, { headers: authHeaders() }),
  );
}

export async function listWorkspaceFiles(chatId: string, path = "."): Promise<WorkspaceListing> {
  const query = new URLSearchParams({ path });
  return readJson<WorkspaceListing>(
    await fetch(`/api/chats/${chatId}/workspace/files?${query}`, { headers: authHeaders() }),
  );
}

export function workspaceContentUrl(chatId: string, path: string): string {
  return `/api/chats/${chatId}/workspace/content?${new URLSearchParams({ path })}`;
}

export function workspacePdfUrl(chatId: string, path: string): string {
  return `/api/chats/${chatId}/workspace/documents/pdf?${new URLSearchParams({ path })}`;
}

export async function getWorkbook(chatId: string, path: string): Promise<WorkbookMetadata> {
  const query = new URLSearchParams({ path });
  return readJson<WorkbookMetadata>(
    await fetch(`/api/chats/${chatId}/workspace/documents/workbook?${query}`, {
      headers: authHeaders(),
    }),
  );
}

export async function getSheetRange(
  chatId: string,
  path: string,
  sheetId: string,
  cellRange: string,
): Promise<SheetRange> {
  const query = new URLSearchParams({ path, sheet_id: sheetId, cell_range: cellRange });
  return readJson<SheetRange>(
    await fetch(`/api/chats/${chatId}/workspace/documents/sheet?${query}`, {
      headers: authHeaders(),
    }),
  );
}

export async function getWordDocument(chatId: string, path: string): Promise<WordDocument> {
  const query = new URLSearchParams({ path });
  return readJson<WordDocument>(
    await fetch(`/api/chats/${chatId}/workspace/documents/word?${query}`, {
      headers: authHeaders(),
    }),
  );
}

export async function createWorkspaceTerminal(chatId: string): Promise<{
  terminal_id: string;
  status: string;
}> {
  return readJson(
    await fetch(`/api/chats/${chatId}/workspace/terminals`, {
      method: "POST",
      headers: authHeaders(),
    }),
  );
}

export async function listWorkspaceTerminals(chatId: string): Promise<Array<{
  terminal_id: string;
  status: string;
  sequence: number;
}>> {
  const payload = await readJson<{ terminals: Array<{ terminal_id: string; status: string; sequence: number }> }>(
    await fetch(`/api/chats/${chatId}/workspace/terminals`, { headers: authHeaders() }),
  );
  return payload.terminals;
}

export async function closeWorkspaceTerminal(chatId: string, terminalId: string): Promise<void> {
  const response = await fetch(`/api/chats/${chatId}/workspace/terminals/${terminalId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }
}

export function workspaceTerminalUrl(chatId: string, terminalId: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const token = getAuthToken();
  const query = new URLSearchParams();
  if (token) {
    query.set("token", token);
  }
  return `${protocol}//${window.location.host}/api/chats/${chatId}/workspace/terminals/${terminalId}/stream${
    query.size ? `?${query}` : ""
  }`;
}

export async function streamChatMessage(
  chatId: string,
  payload: {
    content: string;
    mode: ChatMode;
    model_target: ModelTarget;
    deep_mode?: boolean;
    research_mode?: ResearchMode;
    selected_tools?: string[];
  },
  onEvent: (event: ProgressEvent) => void,
  options?: { signal?: AbortSignal },
): Promise<void> {
  const response = await fetch(`/api/chats/${chatId}/messages/stream`, {
    method: "POST",
    headers: {
      Accept: "text/event-stream",
      "Content-Type": "application/json",
      ...authHeaders(),
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
