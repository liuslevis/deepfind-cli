import type { FormEvent } from "react";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { createChat, deleteChat, getChat, listChats, streamChatMessage } from "./api";
import type {
  ActivityPhase,
  ActivitySummary,
  ChatMode,
  ProgressEvent,
  TurnResult,
  WebChatDetail,
  WebChatSummary,
  WebMessage,
} from "./types";

const STORAGE_KEY = "deepfind.web.selected-chat";

interface ClientMessage extends WebMessage {
  pending?: boolean;
  error?: string | null;
  activity?: ProgressEvent[];
}

interface SourceGroup {
  label: string;
  urls: string[];
}

const RESEARCH_EVENT_TYPES = ["worker_started", "iteration", "tool_call", "tool_result"] as const;

function modeLabel(mode: ChatMode | null): string {
  if (mode === "expert") {
    return "Expert (4 agents)";
  }
  return "Fast (1 agent)";
}

function summarize(text: string, width = 56): string {
  const clean = text.trim().replace(/\s+/g, " ");
  if (clean.length <= width) {
    return clean;
  }
  return `${clean.slice(0, width - 1).trimEnd()}...`;
}

function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

function messageFromServer(message: WebMessage): ClientMessage {
  return {
    ...message,
    activity: [],
    error: null,
    pending: false,
  };
}

function newClientMessage(role: "user" | "assistant", content: string, mode: ChatMode): ClientMessage {
  return {
    id: `local_${crypto.randomUUID()}`,
    role,
    content,
    created_at: new Date().toISOString(),
    mode,
    sources: [],
    artifacts: [],
    pending: role === "assistant",
    error: null,
    activity: [],
  };
}

function describeEvent(event: ProgressEvent): string {
  const { data } = event;
  switch (event.type) {
    case "run_started":
      return `Run started with ${String(data.num_agent ?? "?")} agents`;
    case "plan_ready":
      return `Planner split the work into ${Array.isArray(data.tasks) ? data.tasks.length : 0} tasks`;
    case "worker_started":
      return `${String(data.name ?? "worker")} started ${String(data.task ?? "research")}`;
    case "iteration":
      if (data.status === "done") {
        return `${String(data.name ?? "agent")} finished round ${String(data.iteration ?? "?")}`;
      }
      return `${String(data.name ?? "agent")} entered round ${String(data.iteration ?? "?")}`;
    case "tool_call":
      return `${String(data.name ?? "agent")} called ${String(data.tool_name ?? "tool")}`;
    case "tool_result":
      return `${String(data.tool_name ?? "tool")} ${String(data.status ?? "done")}${data.summary ? `: ${String(data.summary)}` : ""}`;
    case "synthesize_started":
      return `Lead agent is merging ${String(data.report_count ?? "?")} reports`;
    case "answer_delta":
      return "Answer is streaming in";
    case "answer_final":
      return "Answer complete";
    case "error":
      return String(data.message ?? "Something went wrong");
    case "done":
      return "Run finished";
    default:
      return event.type;
  }
}

function phaseLabel(phase: ActivityPhase): string {
  switch (phase) {
    case "planning":
      return "Planning";
    case "researching":
      return "Researching";
    case "synthesizing":
      return "Synthesizing";
    case "complete":
      return "Complete";
    case "error":
      return "Error";
    default:
      return phase;
  }
}

function toCount(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
}

function latestEventOfTypes(activity: ProgressEvent[], types: readonly string[]): ProgressEvent | null {
  for (let index = activity.length - 1; index >= 0; index -= 1) {
    if (types.includes(activity[index].type)) {
      return activity[index];
    }
  }
  return null;
}

function hiddenCountLabel(count: number): string {
  return `${pluralize(count, "update")} hidden`;
}

function buildActivitySummary(activity: ProgressEvent[]): ActivitySummary | null {
  if (activity.length === 0) {
    return null;
  }

  let numAgent = 0;
  let taskCount = 0;
  let workerCount = 0;
  let iterationCount = 0;
  let toolCallCount = 0;
  let toolResultCount = 0;
  let reportCount = 0;

  for (const event of activity) {
    if (event.type === "run_started") {
      numAgent = toCount(event.data.num_agent) ?? numAgent;
      continue;
    }
    if (event.type === "plan_ready") {
      taskCount = Array.isArray(event.data.tasks) ? event.data.tasks.length : taskCount;
      continue;
    }
    if (event.type === "worker_started") {
      workerCount += 1;
      continue;
    }
    if (event.type === "iteration") {
      iterationCount += 1;
      continue;
    }
    if (event.type === "tool_call") {
      toolCallCount += 1;
      continue;
    }
    if (event.type === "tool_result") {
      toolResultCount += 1;
      continue;
    }
    if (event.type === "synthesize_started") {
      reportCount = toCount(event.data.report_count) ?? reportCount;
    }
  }

  const errorEvent = latestEventOfTypes(activity, ["error"]);
  if (errorEvent) {
    return {
      phase: "error",
      label: phaseLabel("error"),
      text: summarize(String(errorEvent.data.message ?? describeEvent(errorEvent)), 96),
      hiddenCount: activity.length,
    };
  }

  const answerFinalEvent = latestEventOfTypes(activity, ["answer_final"]);
  const doneEvent = latestEventOfTypes(activity, ["done"]);
  if (answerFinalEvent || doneEvent) {
    const usage: string[] = [];
    if (workerCount > 0) {
      usage.push(pluralize(workerCount, "worker"));
    }
    if (toolCallCount > 0) {
      usage.push(pluralize(toolCallCount, "tool call"));
    }
    const latest = summarize(describeEvent(answerFinalEvent ?? doneEvent ?? activity[activity.length - 1]), 84);
    return {
      phase: "complete",
      label: phaseLabel("complete"),
      text: usage.length > 0 ? `${usage.join(" / ")} used. Latest: ${latest}` : latest,
      hiddenCount: activity.length,
    };
  }

  const synthesizeEvent = latestEventOfTypes(activity, ["synthesize_started"]);
  if (synthesizeEvent) {
    const text =
      reportCount > 0
        ? `Merging ${pluralize(reportCount, "report")} into the final answer`
        : summarize(describeEvent(synthesizeEvent), 96);
    return {
      phase: "synthesizing",
      label: phaseLabel("synthesizing"),
      text,
      hiddenCount: activity.length,
    };
  }

  const researchEvent = latestEventOfTypes(activity, RESEARCH_EVENT_TYPES);
  if (researchEvent) {
    const counters: string[] = [];
    if (workerCount > 0) {
      counters.push(pluralize(workerCount, "worker"));
    }
    if (toolCallCount > 0) {
      counters.push(pluralize(toolCallCount, "tool call"));
    } else if (toolResultCount > 0) {
      counters.push(pluralize(toolResultCount, "tool result"));
    } else if (iterationCount > 0) {
      counters.push(pluralize(iterationCount, "research update"));
    }
    const lead = counters.length > 0 ? `${counters.join(" / ")} in flight.` : "Collecting evidence.";
    return {
      phase: "researching",
      label: phaseLabel("researching"),
      text: `${lead} Latest: ${summarize(describeEvent(researchEvent), 84)}`,
      hiddenCount: activity.length,
    };
  }

  const planningEvent = latestEventOfTypes(activity, ["plan_ready", "run_started"]) ?? activity[activity.length - 1];
  const planningLead =
    taskCount > 0
      ? `${pluralize(taskCount, "task")} outlined${numAgent > 0 ? ` for ${pluralize(numAgent, "agent")}` : ""}`
      : numAgent > 0
        ? `${pluralize(numAgent, "agent")} online`
        : "Preparing the run";
  return {
    phase: "planning",
    label: phaseLabel("planning"),
    text: planningEvent.type === "plan_ready" && taskCount > 0 ? planningLead : `${planningLead}. Latest: ${summarize(describeEvent(planningEvent), 84)}`,
    hiddenCount: activity.length,
  };
}

function formatTime(value: string): string {
  return new Date(value).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function sourceLabel(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

function groupSources(sources: string[]): SourceGroup[] {
  const grouped = new Map<string, string[]>();
  for (const source of sources) {
    const label = sourceLabel(source);
    const existing = grouped.get(label);
    if (existing) {
      existing.push(source);
      continue;
    }
    grouped.set(label, [source]);
  }
  return Array.from(grouped, ([label, urls]) => ({ label, urls }));
}

function summaryFromChat(chat: WebChatDetail): WebChatSummary {
  const preview = chat.messages.length > 0 ? summarize(chat.messages[chat.messages.length - 1].content, 72) : "";
  return {
    id: chat.id,
    title: chat.title,
    created_at: chat.created_at,
    updated_at: chat.updated_at,
    preview,
  };
}

function upsertSummary(list: WebChatSummary[], next: WebChatSummary): WebChatSummary[] {
  const merged = [next, ...list.filter((item) => item.id !== next.id)];
  return merged.sort((left, right) => right.updated_at.localeCompare(left.updated_at));
}

function ActivityPanel({ activity, pending }: { activity: ProgressEvent[]; pending: boolean }) {
  const summary = buildActivitySummary(activity);
  if (!summary) {
    return null;
  }

  return (
    <div className={`activity-panel activity-panel--${summary.phase}`}>
      <div className="activity-summary" aria-live={pending ? "polite" : "off"}>
        <span className={`activity-summary__phase activity-summary__phase--${summary.phase}`}>{summary.label}</span>
        <span className="activity-summary__text">{summary.text}</span>
        <span className="activity-summary__count">{hiddenCountLabel(summary.hiddenCount)}</span>
      </div>

      <details className="activity-trace">
        <summary>View detailed trace</summary>
        <ul className="activity-list">
          {activity.map((event, index) => (
            <li key={`${event.timestamp}-${event.type}-${index}`} className="activity-list__item">
              <span>{describeEvent(event)}</span>
              <time>{formatTime(event.timestamp)}</time>
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}

function MessageCard({ message }: { message: ClientMessage }) {
  const body = message.content || (message.pending ? "Thinking through the web..." : "");
  const sourceGroups = groupSources(message.sources);

  return (
    <article className={`message message--${message.role}${message.pending ? " message--pending" : ""}`}>
      <div className="message__meta">
        <span className="message__author">{message.role === "assistant" ? "DeepFind" : "You"}</span>
        {message.mode ? <span className="message__mode">{modeLabel(message.mode)}</span> : null}
        <span className="message__time">{formatTime(message.created_at)}</span>
      </div>

      {message.role === "assistant" ? (
        <div className="message__body markdown">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              a: ({ ...props }) => <a {...props} target="_blank" rel="noreferrer" />,
            }}
          >
            {body}
          </ReactMarkdown>
        </div>
      ) : (
        <p className="message__body message__body--plain">{body}</p>
      )}

      {message.error ? <p className="message__error">{message.error}</p> : null}

      {sourceGroups.length > 0 ? (
        <div className="message__sources">
          {sourceGroups.map((group) => (
            <div key={group.label} className="source-group">
              <span className="source-group__label">{group.label}</span>
              <div className="source-group__links">
                {group.urls.map((source, index) => (
                  <a
                    key={source}
                    className="source-group__link"
                    href={source}
                    target="_blank"
                    rel="noreferrer"
                    aria-label={`${group.label} source ${index + 1}`}
                    title={source}
                  >
                    {index + 1}
                  </a>
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : null}

      {message.artifacts.length > 0 ? (
        <div className="artifact-grid">
          {message.artifacts.map((artifact) => (
            <a key={artifact.url} className="artifact-card" href={artifact.url} target="_blank" rel="noreferrer">
              {artifact.kind === "image" ? (
                <img className="artifact-card__image" src={artifact.url} alt={artifact.label} />
              ) : null}
              <span className="artifact-card__kind">{artifact.kind}</span>
              <strong>{artifact.label}</strong>
              <span className="artifact-card__path">{artifact.path}</span>
            </a>
          ))}
        </div>
      ) : null}

      {message.activity && message.activity.length > 0 ? (
        <ActivityPanel activity={message.activity} pending={Boolean(message.pending)} />
      ) : null}
    </article>
  );
}

export default function App() {
  const [mode, setMode] = useState<ChatMode>("fast");
  const [composerValue, setComposerValue] = useState("");
  const [chats, setChats] = useState<WebChatSummary[]>([]);
  const [currentChat, setCurrentChat] = useState<WebChatDetail | null>(null);
  const [messages, setMessages] = useState<ClientMessage[]>([]);
  const [selectedChatId, setSelectedChatId] = useState<string | null>(() => localStorage.getItem(STORAGE_KEY));
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [pageError, setPageError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  async function hydrateChats(preferredChatId?: string | null): Promise<void> {
    const nextChats = await listChats();
    setChats(nextChats);
    const targetId =
      preferredChatId && nextChats.some((chat) => chat.id === preferredChatId)
        ? preferredChatId
        : selectedChatId && nextChats.some((chat) => chat.id === selectedChatId)
          ? selectedChatId
          : nextChats[0]?.id ?? null;

    if (!targetId) {
      setSelectedChatId(null);
      setCurrentChat(null);
      setMessages([]);
      localStorage.removeItem(STORAGE_KEY);
      return;
    }

    const chat = await getChat(targetId);
    setCurrentChat(chat);
    setMessages(chat.messages.map(messageFromServer));
    setSelectedChatId(targetId);
    localStorage.setItem(STORAGE_KEY, targetId);
  }

  useEffect(() => {
    let cancelled = false;

    async function boot() {
      setLoading(true);
      setPageError(null);
      try {
        const nextChats = await listChats();
        if (cancelled) {
          return;
        }
        setChats(nextChats);
        const storedId = localStorage.getItem(STORAGE_KEY);
        const targetId =
          storedId && nextChats.some((chat) => chat.id === storedId) ? storedId : nextChats[0]?.id ?? null;
        if (targetId) {
          const chat = await getChat(targetId);
          if (cancelled) {
            return;
          }
          setCurrentChat(chat);
          setMessages(chat.messages.map(messageFromServer));
          setSelectedChatId(targetId);
          localStorage.setItem(STORAGE_KEY, targetId);
        }
      } catch (error) {
        if (!cancelled) {
          setPageError(error instanceof Error ? error.message : "Failed to load chats");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void boot();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  async function handleOpenChat(chatId: string) {
    setLoading(true);
    setPageError(null);
    try {
      const chat = await getChat(chatId);
      setCurrentChat(chat);
      setMessages(chat.messages.map(messageFromServer));
      setSelectedChatId(chatId);
      localStorage.setItem(STORAGE_KEY, chatId);
    } catch (error) {
      setPageError(error instanceof Error ? error.message : "Failed to open chat");
    } finally {
      setLoading(false);
    }
  }

  async function handleCreateChat() {
    setPageError(null);
    try {
      const chat = await createChat();
      setCurrentChat(chat);
      setMessages([]);
      setSelectedChatId(chat.id);
      setChats((current) => upsertSummary(current, summaryFromChat(chat)));
      localStorage.setItem(STORAGE_KEY, chat.id);
    } catch (error) {
      setPageError(error instanceof Error ? error.message : "Failed to create chat");
    }
  }

  async function handleDeleteCurrentChat() {
    if (!currentChat || !window.confirm(`Delete "${currentChat.title}"?`)) {
      return;
    }
    try {
      await deleteChat(currentChat.id);
      await hydrateChats(null);
    } catch (error) {
      setPageError(error instanceof Error ? error.message : "Failed to delete chat");
    }
  }

  async function ensureActiveChat(content: string): Promise<WebChatDetail> {
    if (currentChat) {
      return currentChat;
    }
    const chat = await createChat();
    const titledChat = {
      ...chat,
      title: summarize(content, 48) || chat.title,
    };
    setCurrentChat(titledChat);
    setSelectedChatId(chat.id);
    setChats((current) => upsertSummary(current, summaryFromChat(titledChat)));
    localStorage.setItem(STORAGE_KEY, chat.id);
    return titledChat;
  }

  function appendActivity(messageId: string, event: ProgressEvent) {
    if (event.type === "answer_delta") {
      return;
    }
    setMessages((current) =>
      current.map((message) =>
        message.id === messageId
          ? {
              ...message,
              activity: [...(message.activity ?? []), event],
            }
          : message,
      ),
    );
  }

  function applyFinalTurn(messageId: string, turnResult: TurnResult) {
    setMessages((current) =>
      current.map((message) =>
        message.id === messageId
          ? {
              ...message,
              content: turnResult.answer_markdown,
              mode: turnResult.mode,
              sources: turnResult.sources,
              artifacts: turnResult.artifacts,
              pending: false,
              error: null,
            }
          : message,
      ),
    );
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (sending || !composerValue.trim()) {
      return;
    }

    const content = composerValue.trim();
    setComposerValue("");
    setPageError(null);
    setSending(true);

    try {
      const chat = await ensureActiveChat(content);
      const userMessage = newClientMessage("user", content, mode);
      const assistantMessage = newClientMessage("assistant", "", mode);
      const nextTitle = currentChat?.title && currentChat.title !== "New chat" ? currentChat.title : summarize(content, 48);
      const chatSnapshot = {
        ...chat,
        title: nextTitle || chat.title,
        updated_at: assistantMessage.created_at,
      };

      setCurrentChat(chatSnapshot);
      setMessages((current) => [...current, userMessage, assistantMessage]);
      setChats((current) => upsertSummary(current, summaryFromChat({ ...chatSnapshot, messages: [...messages, userMessage] })));

      await streamChatMessage(chat.id, { content, mode }, (progressEvent) => {
        appendActivity(assistantMessage.id, progressEvent);
        if (progressEvent.type === "answer_delta") {
          const delta = String(progressEvent.data.delta ?? "");
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantMessage.id
                ? {
                    ...message,
                    content: `${message.content}${delta}`,
                  }
                : message,
            ),
          );
        }
        if (progressEvent.type === "answer_final") {
          applyFinalTurn(assistantMessage.id, progressEvent.data as unknown as TurnResult);
        }
        if (progressEvent.type === "error") {
          const text = String(progressEvent.data.message ?? "Something went wrong");
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantMessage.id
                ? {
                    ...message,
                    pending: false,
                    error: text,
                    content: message.content || "The run ended before a final answer was produced.",
                  }
                : message,
            ),
          );
          setPageError(text);
        }
      });

      const nextChats = await listChats();
      setChats(nextChats);
      const refreshedSummary = nextChats.find((item) => item.id === chat.id);
      if (refreshedSummary) {
        setCurrentChat((current) =>
          current
            ? {
                ...current,
                title: refreshedSummary.title,
                updated_at: refreshedSummary.updated_at,
              }
            : current,
        );
      }
    } catch (error) {
      setPageError(error instanceof Error ? error.message : "Failed to send message");
    } finally {
      setSending(false);
    }
  }

  const selectedTitle = currentChat?.title ?? "New chat";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar__header">
          <div>
            <p className="eyebrow">DeepFind</p>
            <h1>Web Research Cockpit</h1>
          </div>
          <button className="ghost-button" type="button" onClick={handleCreateChat}>
            New chat
          </button>
        </div>

        <div className="sidebar__list">
          {chats.length === 0 ? <p className="sidebar__empty">No saved chats yet.</p> : null}
          {chats.map((chat) => (
            <button
              key={chat.id}
              className={`chat-tile${chat.id === selectedChatId ? " chat-tile--active" : ""}`}
              type="button"
              onClick={() => void handleOpenChat(chat.id)}
            >
              <strong>{chat.title}</strong>
              <span>{chat.preview || "Ready for a new question"}</span>
              <time>{formatTime(chat.updated_at)}</time>
            </button>
          ))}
        </div>
      </aside>

      <main className="workspace">
        <header className="workspace__header">
          <div>
            <p className="eyebrow">Local-first</p>
            <h2>{selectedTitle}</h2>
          </div>
          {currentChat ? (
            <button className="ghost-button ghost-button--danger" type="button" onClick={() => void handleDeleteCurrentChat()}>
              Delete chat
            </button>
          ) : null}
        </header>

        <section className="transcript">
          {loading ? <p className="state-text">Loading chats...</p> : null}
          {!loading && messages.length === 0 ? (
            <div className="hero-empty">
              <p className="eyebrow">Parallel web research</p>
              <h3>Ask for live research, then decide how much horsepower you want.</h3>
              <p>
                Fast keeps it lean with one agent. Expert fans out to four agents and returns with a denser brief,
                sources, and any generated assets.
              </p>
            </div>
          ) : null}

          {messages.map((message) => (
            <MessageCard key={message.id} message={message} />
          ))}

          <div ref={bottomRef} />
        </section>

        <footer className="composer-wrap">
          <div className="mode-toggle" aria-label="Chat mode">
            {(["fast", "expert"] as const).map((option) => (
              <button
                key={option}
                type="button"
                className={`mode-toggle__button${mode === option ? " mode-toggle__button--active" : ""}`}
                onClick={() => setMode(option)}
              >
                {modeLabel(option)}
              </button>
            ))}
          </div>

          <form className="composer" onSubmit={handleSubmit}>
            <label className="sr-only" htmlFor="chat-input">
              Ask DeepFind
            </label>
            <textarea
              id="chat-input"
              className="composer__input"
              rows={3}
              placeholder="Ask DeepFind to search, summarize, compare, or generate artifacts..."
              value={composerValue}
              onChange={(changeEvent) => setComposerValue(changeEvent.target.value)}
              onKeyDown={(keyEvent) => {
                if (keyEvent.key === "Enter" && !keyEvent.shiftKey) {
                  keyEvent.preventDefault();
                  keyEvent.currentTarget.form?.requestSubmit();
                }
              }}
            />
            <button className="send-button" type="submit" disabled={sending}>
              {sending ? "Running..." : "Send"}
            </button>
          </form>

          {pageError ? <p className="composer__error">{pageError}</p> : null}
        </footer>
      </main>
    </div>
  );
}
