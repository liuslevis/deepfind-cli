import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mermaidMock = vi.hoisted(() => ({
  initialize: vi.fn(),
  render: vi.fn(async () => ({
    svg: '<svg data-testid="mermaid-svg" viewBox="0 0 10 10"><text x="1" y="9">diagram</text></svg>',
    bindFunctions: vi.fn(),
  })),
}));

vi.mock("mermaid", () => ({
  default: mermaidMock,
}));

import App from "./App";

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "Content-Type": "application/json",
    },
  });
}

function streamResponse(events: Array<{ type: string; data: Record<string, unknown> }>): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      for (const event of events) {
        controller.enqueue(
          encoder.encode(
            `event: ${event.type}\ndata: ${JSON.stringify({
              timestamp: "2026-03-22T12:00:00Z",
              data: event.data,
            })}\n\n`,
          ),
        );
      }
      controller.close();
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream",
    },
  });
}

function createControlledStreamResponse() {
  const encoder = new TextEncoder();
  let controllerRef: ReadableStreamDefaultController<Uint8Array> | null = null;

  const response = new Response(
    new ReadableStream({
      start(controller) {
        controllerRef = controller;
      },
    }),
    {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
      },
    },
  );

  return {
    response,
    push(event: { type: string; data: Record<string, unknown> }) {
      if (!controllerRef) {
        throw new Error("Stream controller is not ready");
      }
      controllerRef.enqueue(
        encoder.encode(
          `event: ${event.type}\ndata: ${JSON.stringify({
            timestamp: "2026-03-22T12:00:00Z",
            data: event.data,
          })}\n\n`,
        ),
      );
    },
    close() {
      controllerRef?.close();
    },
  };
}

describe("App", () => {
  beforeEach(() => {
    localStorage.clear();
    mermaidMock.initialize.mockClear();
    mermaidMock.render.mockClear();
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("sends the selected mode in the stream request", async () => {
    let capturedBody = "";
    let chatsRequestCount = 0;
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      const method = (init?.method ?? "GET").toUpperCase();

      if (url === "/api/chats" && method === "GET") {
        chatsRequestCount += 1;
        if (chatsRequestCount === 1) {
          return jsonResponse({ chats: [] });
        }
        return jsonResponse({
          chats: [
            {
              id: "chat_1",
              title: "Expert mode answer",
              created_at: "2026-03-22T00:00:00Z",
              updated_at: "2026-03-22T00:05:00Z",
              preview: "Expert mode answer",
            },
          ],
        });
      }
      if (url === "/api/chats" && method === "POST") {
        return jsonResponse({
          chat: {
            id: "chat_1",
            title: "New chat",
            created_at: "2026-03-22T00:00:00Z",
            updated_at: "2026-03-22T00:00:00Z",
            messages: [],
          },
        });
      }
      if (url === "/api/chats/chat_1/messages/stream") {
        capturedBody = String(init?.body ?? "");
        return streamResponse([
          { type: "run_started", data: { num_agent: 4 } },
          {
            type: "answer_final",
            data: {
              answer_markdown: "Expert mode answer",
              sources: [],
              artifacts: [],
              mode: "expert",
            },
          },
          { type: "done", data: { chat_id: "chat_1" } },
        ]);
      }
      throw new Error(`Unhandled request: ${method} ${url}`);
    });

    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await userEvent.click(screen.getByRole("button", { name: "Expert (4 agents)" }));
    await userEvent.type(screen.getByLabelText("Ask DeepFind"), "Explain the latest AI launches");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    await screen.findAllByText("Expert mode answer");
    expect(JSON.parse(capturedBody)).toMatchObject({ mode: "expert" });
  });

  it("restores the selected chat from local storage", async () => {
    localStorage.setItem("deepfind.web.selected-chat", "chat_b");

    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      const method = (init?.method ?? "GET").toUpperCase();

      if (url === "/api/chats" && method === "GET") {
        return jsonResponse({
          chats: [
            {
              id: "chat_a",
              title: "Alpha",
              created_at: "2026-03-22T00:00:00Z",
              updated_at: "2026-03-22T00:01:00Z",
              preview: "Alpha preview",
            },
            {
              id: "chat_b",
              title: "Bravo",
              created_at: "2026-03-22T00:00:00Z",
              updated_at: "2026-03-22T00:02:00Z",
              preview: "Bravo preview",
            },
          ],
        });
      }
      if (url === "/api/chats/chat_b" && method === "GET") {
        return jsonResponse({
          chat: {
            id: "chat_b",
            title: "Bravo",
            created_at: "2026-03-22T00:00:00Z",
            updated_at: "2026-03-22T00:02:00Z",
            messages: [
              {
                id: "msg_1",
                role: "assistant",
                content: "Loaded from the selected chat",
                created_at: "2026-03-22T00:02:00Z",
                mode: "fast",
                sources: [],
                artifacts: [],
              },
            ],
          },
        });
      }
      throw new Error(`Unhandled request: ${method} ${url}`);
    });

    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await screen.findByText("Loaded from the selected chat");
    expect(screen.getByRole("heading", { name: "Bravo" })).toBeInTheDocument();
    expect(screen.queryByText("View detailed trace")).not.toBeInTheDocument();
  });

  it("renders mermaid markdown blocks as diagrams", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      const method = (init?.method ?? "GET").toUpperCase();

      if (url === "/api/chats" && method === "GET") {
        return jsonResponse({
          chats: [
            {
              id: "chat_mermaid",
              title: "Mermaid",
              created_at: "2026-03-22T00:00:00Z",
              updated_at: "2026-03-22T00:02:00Z",
              preview: "Diagram answer",
            },
          ],
        });
      }
      if (url === "/api/chats/chat_mermaid" && method === "GET") {
        return jsonResponse({
          chat: {
            id: "chat_mermaid",
            title: "Mermaid",
            created_at: "2026-03-22T00:00:00Z",
            updated_at: "2026-03-22T00:02:00Z",
            messages: [
              {
                id: "msg_1",
                role: "assistant",
                content: "Here is the graph:\n\nflowchart TD\nA --> B\nstyle A fill:#e6f3ff,stroke:#333\n\nAfter the graph.",
                created_at: "2026-03-22T00:02:00Z",
                mode: "fast",
                sources: [],
                artifacts: [],
              },
            ],
          },
        });
      }
      throw new Error(`Unhandled request: ${method} ${url}`);
    });

    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await screen.findByLabelText("Mermaid diagram");
    await waitFor(() => expect(mermaidMock.render).toHaveBeenCalled());
    expect(mermaidMock.initialize).toHaveBeenCalled();
    expect(mermaidMock.render).toHaveBeenCalledWith(
      expect.stringMatching(/^mermaid_/),
      "flowchart TD\nA --> B\nstyle A fill:#e6f3ff,stroke:#333",
    );
  });

  it("toggles the mobile chat drawer and closes it after selecting a chat", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      const method = (init?.method ?? "GET").toUpperCase();

      if (url === "/api/chats" && method === "GET") {
        return jsonResponse({
          chats: [
            {
              id: "chat_a",
              title: "Alpha",
              created_at: "2026-03-22T00:00:00Z",
              updated_at: "2026-03-22T00:01:00Z",
              preview: "Alpha preview",
            },
            {
              id: "chat_b",
              title: "Bravo",
              created_at: "2026-03-22T00:00:00Z",
              updated_at: "2026-03-22T00:02:00Z",
              preview: "Bravo preview",
            },
          ],
        });
      }
      if (url === "/api/chats/chat_a" && method === "GET") {
        return jsonResponse({
          chat: {
            id: "chat_a",
            title: "Alpha",
            created_at: "2026-03-22T00:00:00Z",
            updated_at: "2026-03-22T00:01:00Z",
            messages: [],
          },
        });
      }
      if (url === "/api/chats/chat_b" && method === "GET") {
        return jsonResponse({
          chat: {
            id: "chat_b",
            title: "Bravo",
            created_at: "2026-03-22T00:00:00Z",
            updated_at: "2026-03-22T00:02:00Z",
            messages: [],
          },
        });
      }
      throw new Error(`Unhandled request: ${method} ${url}`);
    });

    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await screen.findByRole("heading", { name: "Alpha" });

    const openChatsButton = screen.getByRole("button", { name: "Open chats" });
    expect(openChatsButton).toHaveAttribute("aria-expanded", "false");

    await userEvent.click(openChatsButton);
    expect(openChatsButton).toHaveAttribute("aria-expanded", "true");

    await userEvent.click(screen.getByRole("button", { name: /bravo/i }));

    await screen.findByRole("heading", { name: "Bravo" });
    expect(openChatsButton).toHaveAttribute("aria-expanded", "false");
  });

  it("scrolls to the latest assistant answer when opening chat history", async () => {
    const scrollTargets: Element[] = [];
    vi.spyOn(window.HTMLElement.prototype, "scrollIntoView").mockImplementation(function (this: Element) {
      scrollTargets.push(this);
    });

    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      const method = (init?.method ?? "GET").toUpperCase();

      if (url === "/api/chats" && method === "GET") {
        return jsonResponse({
          chats: [
            {
              id: "chat_a",
              title: "Alpha",
              created_at: "2026-03-22T00:00:00Z",
              updated_at: "2026-03-22T00:01:00Z",
              preview: "Alpha preview",
            },
            {
              id: "chat_b",
              title: "Bravo",
              created_at: "2026-03-22T00:00:00Z",
              updated_at: "2026-03-22T00:02:00Z",
              preview: "Bravo preview",
            },
          ],
        });
      }
      if (url === "/api/chats/chat_a" && method === "GET") {
        return jsonResponse({
          chat: {
            id: "chat_a",
            title: "Alpha",
            created_at: "2026-03-22T00:00:00Z",
            updated_at: "2026-03-22T00:01:00Z",
            messages: [],
          },
        });
      }
      if (url === "/api/chats/chat_b" && method === "GET") {
        return jsonResponse({
          chat: {
            id: "chat_b",
            title: "Bravo",
            created_at: "2026-03-22T00:00:00Z",
            updated_at: "2026-03-22T00:02:00Z",
            messages: [
              {
                id: "msg_1",
                role: "user",
                content: "Question one",
                created_at: "2026-03-22T00:00:00Z",
                mode: "fast",
                sources: [],
                artifacts: [],
              },
              {
                id: "msg_2",
                role: "assistant",
                content: "Earlier answer",
                created_at: "2026-03-22T00:01:00Z",
                mode: "fast",
                sources: [],
                artifacts: [],
              },
              {
                id: "msg_3",
                role: "user",
                content: "Question two",
                created_at: "2026-03-22T00:02:00Z",
                mode: "fast",
                sources: [],
                artifacts: [],
              },
              {
                id: "msg_4",
                role: "assistant",
                content: "Latest answer starts here",
                created_at: "2026-03-22T00:03:00Z",
                mode: "fast",
                sources: [],
                artifacts: [],
              },
            ],
          },
        });
      }
      throw new Error(`Unhandled request: ${method} ${url}`);
    });

    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await screen.findByRole("heading", { name: "Alpha" });
    scrollTargets.length = 0;

    await userEvent.click(screen.getByRole("button", { name: /bravo/i }));

    await screen.findByText("Latest answer starts here");

    const assistantMessages = document.querySelectorAll('article[data-message-role="assistant"]');
    expect(assistantMessages).toHaveLength(2);
    expect(scrollTargets.at(-1)).toBe(assistantMessages[assistantMessages.length - 1]);
  });

  it("shows compact milestones, grouped sources, and a collapsed detailed trace by default", async () => {
    let chatsRequestCount = 0;
    const stream = createControlledStreamResponse();
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      const method = (init?.method ?? "GET").toUpperCase();

      if (url === "/api/chats" && method === "GET") {
        chatsRequestCount += 1;
        if (chatsRequestCount === 1) {
          return jsonResponse({ chats: [] });
        }
        return jsonResponse({
          chats: [
            {
              id: "chat_2",
              title: "Final streamed answer",
              created_at: "2026-03-22T00:00:00Z",
              updated_at: "2026-03-22T00:05:00Z",
              preview: "Final streamed answer",
            },
          ],
        });
      }
      if (url === "/api/chats" && method === "POST") {
        return jsonResponse({
          chat: {
            id: "chat_2",
            title: "New chat",
            created_at: "2026-03-22T00:00:00Z",
            updated_at: "2026-03-22T00:00:00Z",
            messages: [],
          },
        });
      }
      if (url === "/api/chats/chat_2/messages/stream") {
        return stream.response;
      }
      throw new Error(`Unhandled request: ${method} ${url}`);
    });

    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await userEvent.type(screen.getByLabelText("Ask DeepFind"), "Give me a quick brief");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    stream.push({ type: "run_started", data: { num_agent: 1 } });
    await screen.findByText("Planning");

    stream.push({ type: "plan_ready", data: { tasks: ["core facts", "sources"] } });
    await screen.findByText("2 tasks outlined for 1 agent");
    expect(screen.getByText("Planner split the work into 2 tasks")).not.toBeVisible();

    stream.push({ type: "worker_started", data: { name: "researcher", task: "core facts" } });
    stream.push({ type: "tool_call", data: { name: "researcher", tool_name: "web_search" } });
    await screen.findByText("Researching");

    stream.push({ type: "synthesize_started", data: { report_count: 2 } });
    await screen.findByText("Synthesizing");
    expect(screen.getByText("Merging 2 reports into the final answer")).toBeInTheDocument();

    stream.push({ type: "answer_delta", data: { delta: "Final streamed " } });
    stream.push({
      type: "answer_final",
      data: {
        answer_markdown: "Final streamed answer",
        sources: [
          "https://zhihu.com/question/1",
          "https://zhihu.com/question/2",
          "https://baidu.com/s?wd=atlas",
        ],
        artifacts: [
          {
            kind: "slides",
            label: "deck.html",
            path: "C:/tmp/deck.html",
            url: "/api/files?path=deck",
          },
        ],
        mode: "fast",
      },
    });
    stream.push({ type: "done", data: { chat_id: "chat_2" } });
    stream.close();

    await screen.findAllByText("Final streamed answer");
    expect(screen.getByText("Complete")).toBeInTheDocument();
    expect(screen.getByText("7 updates hidden")).toBeInTheDocument();
    expect(screen.getByText("researcher called web_search")).not.toBeVisible();
    expect(screen.getAllByText("zhihu.com")).toHaveLength(1);
    expect(screen.getByText("baidu.com")).toBeInTheDocument();

    await userEvent.click(screen.getByText("View detailed trace"));

    expect(screen.getByText("Planner split the work into 2 tasks")).toBeVisible();
    expect(screen.getByText("researcher called web_search")).toBeVisible();
    expect(screen.getByRole("link", { name: "zhihu.com source 1" })).toHaveAttribute("href", "https://zhihu.com/question/1");
    expect(screen.getByRole("link", { name: "zhihu.com source 2" })).toHaveAttribute("href", "https://zhihu.com/question/2");
    expect(screen.getByRole("link", { name: "baidu.com source 1" })).toHaveAttribute("href", "https://baidu.com/s?wd=atlas");
    expect(screen.getByRole("link", { name: /deck.html/i })).toHaveAttribute("href", "/api/files?path=deck");
  });

  it("recovers after a streamed error and allows the next turn", async () => {
    let streamCount = 0;
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      const method = (init?.method ?? "GET").toUpperCase();

      if (url === "/api/chats" && method === "GET") {
        return jsonResponse(
          streamCount === 0
            ? { chats: [] }
            : {
                chats: [
                  {
                    id: "chat_3",
                    title: "Recovery chat",
                    created_at: "2026-03-22T00:00:00Z",
                    updated_at: "2026-03-22T00:05:00Z",
                    preview: "Recovered answer",
                  },
                ],
              },
        );
      }
      if (url === "/api/chats" && method === "POST") {
        return jsonResponse({
          chat: {
            id: "chat_3",
            title: "New chat",
            created_at: "2026-03-22T00:00:00Z",
            updated_at: "2026-03-22T00:00:00Z",
            messages: [],
          },
        });
      }
      if (url === "/api/chats/chat_3/messages/stream") {
        streamCount += 1;
        if (streamCount === 1) {
          return streamResponse([
            { type: "run_started", data: { num_agent: 1 } },
            { type: "error", data: { message: "Network hiccup" } },
            { type: "done", data: { chat_id: "chat_3" } },
          ]);
        }
        return streamResponse([
          { type: "run_started", data: { num_agent: 1 } },
          {
            type: "answer_final",
            data: {
              answer_markdown: "Recovered answer",
              sources: [],
              artifacts: [],
              mode: "fast",
            },
          },
          { type: "done", data: { chat_id: "chat_3" } },
        ]);
      }
      throw new Error(`Unhandled request: ${method} ${url}`);
    });

    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    await userEvent.type(screen.getByLabelText("Ask DeepFind"), "First attempt");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    await screen.findAllByText("Network hiccup");
    expect(screen.getByText("Error")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: "Send" })).toBeEnabled());

    await userEvent.type(screen.getByLabelText("Ask DeepFind"), "Second attempt");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    await screen.findAllByText("Recovered answer");
  });
});
