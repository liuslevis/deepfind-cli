import type { FormEvent, KeyboardEvent as ReactKeyboardEvent, ReactNode } from "react";
import { Children, isValidElement, memo, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { createChat, deleteChat, getChat, listChats, streamChatMessage } from "./api";
import type {
  ActivityPhase,
  ActivitySummary,
  CitationLink,
  ChatMode,
  ProgressEvent,
  TurnResult,
  WebChatDetail,
  WebChatSummary,
  WebMessage,
} from "./types";

const STORAGE_KEY = "deepfind.web.selected-chat";
const MERMAID_BLOCK_START =
  /^(flowchart|graph|sequenceDiagram|classDiagram|stateDiagram(?:-v2)?|erDiagram|journey|gantt|pie|mindmap|timeline|quadrantChart|requirementDiagram|gitGraph|C4Context|C4Container|C4Component|C4Dynamic|C4Deployment|xychart-beta|sankey-beta|block-beta)\b/;
const MERMAID_STRUCTURAL_LINE =
  /^(subgraph|end|style|classDef|class|linkStyle|click|section|accTitle|accDescr|title|todayMarker)\b/;
let mermaidPromise: Promise<typeof import("mermaid").default> | null = null;

interface ClientMessage extends WebMessage {
  pending?: boolean;
  error?: string | null;
  activity?: ProgressEvent[];
}

interface ChatRuntime {
  messages: ClientMessage[];
  pending: boolean;
  error: string | null;
  abortController?: AbortController;
}

interface SourceGroup {
  label: string;
  urls: string[];
}

type TranscriptScrollTarget = "bottom" | "last-assistant-head";

const RESEARCH_EVENT_TYPES = ["worker_started", "iteration", "tool_call", "tool_result"] as const;

interface SlashCommandOption {
  command: string;
  description: string;
}

const SLASH_COMMANDS: SlashCommandOption[] = [
  {
    command: "/list-tool",
    description: "List all available tools and their descriptions.",
  },
];

function normalizeSlashValue(value: string): string {
  return value.trimStart().toLowerCase();
}

function matchSlashCommands(value: string): SlashCommandOption[] {
  const normalized = normalizeSlashValue(value);
  if (!normalized.startsWith("/")) {
    return [];
  }
  return SLASH_COMMANDS.filter((item) => item.command.startsWith(normalized));
}

function resolveSlashCommand(value: string, matches: SlashCommandOption[], selectedIndex: number): string {
  const trimmed = value.trim();
  const normalized = trimmed.toLowerCase();
  const exactMatch = matches.find((item) => item.command === normalized);
  if (exactMatch) {
    return exactMatch.command;
  }
  return matches[Math.min(selectedIndex, matches.length - 1)]?.command ?? trimmed;
}

function storageGetItem(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function storageSetItem(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Ignore storage failures so the chat UI still works in constrained browsers/tests.
  }
}

function storageRemoveItem(key: string): void {
  try {
    window.localStorage.removeItem(key);
  } catch {
    // Ignore storage failures so the chat UI still works in constrained browsers/tests.
  }
}

function isStandalonePwa(): boolean {
  if (typeof window === "undefined") {
    return false;
  }

  const navigatorWithStandalone = window.navigator as Navigator & { standalone?: boolean };
  if (navigatorWithStandalone.standalone === true) {
    return true;
  }

  return typeof window.matchMedia === "function" && window.matchMedia("(display-mode: standalone)").matches;
}

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

function readString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function describeToolCall(data: Record<string, unknown>): string {
  const agentName = String(data.name ?? "agent");
  const toolName = String(data.tool_name ?? "tool");
  const argumentsValue =
    typeof data.arguments === "object" && data.arguments !== null
      ? (data.arguments as Record<string, unknown>)
      : null;

  if (toolName === "web_search") {
    const query = readString(argumentsValue?.query);
    if (query) {
      return `${agentName} called web_search for "${summarize(query, 72)}"`;
    }
  }

  return `${agentName} called ${toolName}`;
}

function displayAgentName(value: unknown): string {
  const name = String(value ?? "agent");
  switch (name) {
    case "lead-plan":
      return "Lead planner";
    case "lead-synthesis":
      return "Lead synthesis";
    case "lead-final":
      return "Lead agent";
    default:
      return name;
  }
}

function shouldHideActivityEvent(event: ProgressEvent): boolean {
  if (event.type !== "tool_result") {
    return false;
  }

  const toolName = String(event.data.tool_name ?? "");
  const status = String(event.data.status ?? "");
  return toolName === "web_search" && status === "ok";
}

function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

interface AnsiStyleState {
  color: string | null;
  bold: boolean;
  dim: boolean;
}

const ANSI_COLOR_STYLES: Record<number, string> = {
  31: "#ff7b7b",
  32: "#8ad86c",
  33: "#ffd479",
  34: "#7fb7ff",
  35: "#d7a6ff",
  36: "#7ce8ff",
};

function escapeHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function styleToCss(state: AnsiStyleState): string {
  const parts: string[] = [];
  if (state.color) {
    parts.push(`color: ${state.color}`);
  }
  if (state.bold) {
    parts.push("font-weight: 700");
  }
  if (state.dim) {
    parts.push("opacity: 0.72");
  }
  return parts.join("; ");
}

function updateAnsiStyle(state: AnsiStyleState, code: number): AnsiStyleState {
  if (code === 0) {
    return { color: null, bold: false, dim: false };
  }
  if (code === 1) {
    return { ...state, bold: true };
  }
  if (code === 2) {
    return { ...state, dim: true };
  }
  if (code === 22) {
    return { ...state, bold: false, dim: false };
  }
  if (code === 39) {
    return { ...state, color: null };
  }
  if (code in ANSI_COLOR_STYLES) {
    return { ...state, color: ANSI_COLOR_STYLES[code] };
  }
  return state;
}

function ansiToHtml(text: string): string {
  const pattern = /\x1b\[([0-9;]*)m/g;
  let cursor = 0;
  let html = "";
  let state: AnsiStyleState = { color: null, bold: false, dim: false };

  function appendChunk(chunk: string): void {
    if (!chunk) {
      return;
    }
    const escaped = escapeHtml(chunk);
    const css = styleToCss(state);
    html += css ? `<span style="${css}">${escaped}</span>` : escaped;
  }

  for (const match of text.matchAll(pattern)) {
    const index = match.index ?? 0;
    appendChunk(text.slice(cursor, index));
    const codes = (match[1] || "0")
      .split(";")
      .map((value) => Number.parseInt(value || "0", 10))
      .filter((value) => Number.isFinite(value));
    for (const code of codes) {
      state = updateAnsiStyle(state, code);
    }
    cursor = index + match[0].length;
  }

  appendChunk(text.slice(cursor));
  return html;
}

function messageFromServer(message: WebMessage): ClientMessage {
  return {
    ...message,
    key_points: message.key_points ?? [],
    citations: message.citations ?? [],
    activity: [],
    error: null,
    pending: false,
  };
}

function createClientMessageId(): string {
  const cryptoApi = globalThis.crypto;
  if (cryptoApi && typeof cryptoApi.randomUUID === "function") {
    return cryptoApi.randomUUID();
  }
  return `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

function getMermaid() {
  if (!mermaidPromise) {
    mermaidPromise = import("mermaid")
      .then(({ default: loadedMermaid }) => {
        loadedMermaid.initialize({
          startOnLoad: false,
          theme: "dark",
          securityLevel: "strict",
          fontFamily: '"IBM Plex Mono", monospace',
        });
        return loadedMermaid;
      })
      .catch((error) => {
        mermaidPromise = null;
        throw error;
      });
  }
  return mermaidPromise;
}

function flattenNodeText(node: ReactNode): string {
  return Children.toArray(node)
    .map((child) => {
      if (typeof child === "string" || typeof child === "number") {
        return String(child);
      }
      if (isValidElement<{ children?: ReactNode }>(child)) {
        return flattenNodeText(child.props.children ?? "");
      }
      return "";
    })
    .join("");
}

function extractMermaidChart(children: ReactNode): string | null {
  const child = Array.isArray(children) ? children[0] : children;
  if (!isValidElement<{ className?: string; children?: ReactNode }>(child)) {
    return null;
  }

  const className = typeof child.props.className === "string" ? child.props.className : "";
  const chart = flattenNodeText(child.props.children ?? "").replace(/\n$/, "");
  if (!chart.trim()) {
    return null;
  }

  if (/\blanguage-mermaid\b/.test(className) || MERMAID_BLOCK_START.test(chart.trimStart())) {
    return chart;
  }
  return null;
}

function looksLikeMermaidContinuation(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed) {
    return false;
  }
  return (
    MERMAID_BLOCK_START.test(trimmed) ||
    MERMAID_STRUCTURAL_LINE.test(trimmed) ||
    /^\d{4}-\d{2}-\d{2}\s*:/.test(trimmed) ||
    /-->|---|==>|-.->|:::/.test(trimmed)
  );
}

function normalizeMermaidMarkdown(body: string): string {
  if (!body.trim()) {
    return body;
  }

  const lines = body.split("\n");
  const normalized: string[] = [];
  let insideFence = false;

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();

    if (/^```/.test(trimmed)) {
      insideFence = !insideFence;
      normalized.push(line);
      continue;
    }

    if (!insideFence && MERMAID_BLOCK_START.test(trimmed)) {
      const block = [line];
      while (index + 1 < lines.length && looksLikeMermaidContinuation(lines[index + 1])) {
        index += 1;
        block.push(lines[index]);
      }
      normalized.push("```mermaid", ...block, "```");
      continue;
    }

    normalized.push(line);
  }

  return normalized.join("\n");
}

function MermaidBlock({ chart }: { chart: string }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const renderIdRef = useRef(`mermaid_${createClientMessageId().replace(/[^a-zA-Z0-9_-]/g, "_")}`);
  const [svg, setSvg] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function renderChart() {
      try {
        const mermaid = await getMermaid();
        const { svg: nextSvg, bindFunctions } = await mermaid.render(renderIdRef.current, chart);
        if (cancelled) {
          return;
        }
        setSvg(nextSvg);
        setError(null);
        requestAnimationFrame(() => {
          if (cancelled) {
            return;
          }
          const container = containerRef.current;
          if (container) {
            bindFunctions?.(container);
          }
        });
      } catch (cause) {
        if (cancelled) {
          return;
        }
        setSvg("");
        setError(cause instanceof Error ? cause.message : "Unable to render Mermaid diagram");
      }
    }

    void renderChart();
    return () => {
      cancelled = true;
    };
  }, [chart]);

  return (
    <figure className="mermaid-block" aria-label="Mermaid diagram">
      {svg ? (
        <div
          ref={containerRef}
          className="mermaid-block__diagram"
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      ) : null}
      {!svg && !error ? (
        <div className="mermaid-block__status" aria-busy="true">
          Rendering diagram...
        </div>
      ) : null}
      {error ? (
        <div className="mermaid-block__fallback">
          <p>Mermaid render failed: {error}</p>
          <pre>
            <code>{chart}</code>
          </pre>
        </div>
      ) : null}
    </figure>
  );
}

function newClientMessage(role: "user" | "assistant", content: string, mode: ChatMode): ClientMessage {
  return {
    id: `local_${createClientMessageId()}`,
    role,
    content,
    created_at: new Date().toISOString(),
    mode,
    sources: [],
    artifacts: [],
    key_points: [],
    citations: [],
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
      return `${displayAgentName(data.name ?? "worker")} started ${String(data.task ?? "research")}`;
    case "iteration":
      if (data.status === "done") {
        return `${displayAgentName(data.name ?? "agent")} finished round ${String(data.iteration ?? "?")}`;
      }
      return `${displayAgentName(data.name ?? "agent")} entered round ${String(data.iteration ?? "?")}`;
    case "tool_call":
      return describeToolCall(data);
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

function hasReferenceSection(markdown: string): boolean {
  return /^\s{0,3}#{1,6}\s*Reference(?:s)?\s*$/im.test(markdown);
}

function buildCitationLookup(citations: CitationLink[]): Map<string, CitationLink> {
  return new Map(citations.map((citation) => [citation.id, citation]));
}

function citationOrdinal(citationId: string, fallbackIndex: number): string {
  const match = citationId.match(/\d+/);
  if (!match) {
    return String(fallbackIndex);
  }
  const parsed = Number.parseInt(match[0], 10);
  return Number.isFinite(parsed) ? String(parsed) : String(fallbackIndex);
}

function citationDisplayText(citation: CitationLink): string {
  const hostname = sourceLabel(citation.canonical_url || citation.url);
  if (citation.title) {
    return `${hostname}: ${citation.title}`;
  }
  return hostname;
}

function pointConfidenceLabel(confidence: string): string {
  const normalized = confidence.trim().toLowerCase();
  if (normalized === "high") {
    return "High confidence";
  }
  if (normalized === "low") {
    return "Low confidence";
  }
  return "Medium confidence";
}

type ChatLike = Pick<WebChatDetail, "id" | "title" | "created_at" | "updated_at"> & {
  messages: Array<{ content: string }>;
};

function summaryFromChat(chat: ChatLike): WebChatSummary {
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

function sidebarTitle(title: string): string {
  return summarize(title, 32);
}

const ActivityPanel = memo(function ActivityPanel({
  activity,
  pending,
}: {
  activity: ProgressEvent[];
  pending: boolean;
}) {
  const summaryActivity = activity.filter(
    (event) => event.type !== "console_line" && !shouldHideActivityEvent(event),
  );
  const consoleLines = activity.flatMap((event) => {
    if (event.type !== "console_line" || typeof event.data.text !== "string") {
      return [];
    }
    return [event.data.text];
  });
  const summary = buildActivitySummary(summaryActivity);
  if (!summary) {
    return null;
  }
  const hiddenCount = consoleLines.length > 0 ? consoleLines.length : summary.hiddenCount;
  const consoleHtml = ansiToHtml(consoleLines.join("\n"));

  return (
    <div className={`activity-panel activity-panel--${summary.phase}`}>
      <div className="activity-summary" aria-live={pending ? "polite" : "off"}>
        <span className={`activity-summary__phase activity-summary__phase--${summary.phase}`}>{summary.label}</span>
        <span className="activity-summary__text">{summary.text}</span>
        <span className="activity-summary__count">{hiddenCountLabel(hiddenCount)}</span>
      </div>

      {consoleLines.length > 0 ? (
        <details className="activity-trace">
          <summary>View detailed trace</summary>
          <pre className="activity-terminal" data-testid="activity-terminal">
            <code
              className="activity-terminal__content"
              dangerouslySetInnerHTML={{ __html: consoleHtml }}
            />
          </pre>
        </details>
      ) : null}
    </div>
  );
});

ActivityPanel.displayName = "ActivityPanel";

const MessageCard = memo(function MessageCard({ message }: { message: ClientMessage }) {
  const body = message.content || (message.pending ? "Thinking through the web..." : "");
  const markdownBody = normalizeMermaidMarkdown(body);
  const citations = message.citations ?? [];
  const citationLookup = buildCitationLookup(citations);
  const citationOrdinals = new Map(citations.map((citation, index) => [citation.id, citationOrdinal(citation.id, index + 1)]));
  const hasInlineReferences = hasReferenceSection(markdownBody);
  const sourceGroups = citations.length === 0 ? groupSources(message.sources) : [];
  const keyPoints = message.key_points ?? [];

  return (
    <article
      className={`message message--${message.role}${message.pending ? " message--pending" : ""}`}
      data-message-id={message.id}
      data-message-role={message.role}
    >
      <div className="message__meta">
        <span className="message__author">{message.role === "assistant" ? "DeepFind" : "You"}</span>
        {message.role === "assistant" && message.mode ? (
          <span className="message__mode">{modeLabel(message.mode)}</span>
        ) : null}
      </div>

      {message.role === "assistant" ? (
        <div className="message__body markdown">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              a: ({ ...props }) => <a {...props} target="_blank" rel="noreferrer" />,
              table: ({ children, ...props }) => (
                <div className="markdown-table-scroll">
                  <table {...props}>{children}</table>
                </div>
              ),
              pre: ({ children, ...props }) => {
                const chart = extractMermaidChart(children);
                if (chart) {
                  return <MermaidBlock chart={chart} />;
                }
                return <pre {...props}>{children}</pre>;
              },
            }}
          >
            {markdownBody}
          </ReactMarkdown>
        </div>
      ) : (
        <p className="message__body message__body--plain">{body}</p>
      )}

      {message.role === "assistant" && keyPoints.length > 0 ? (
        <section className="message__structured">
          <h3 className="message__section-title">Key Points</h3>
          <ol className="key-point-list">
            {keyPoints.map((point, index) => (
              <li key={`${point.text}_${index}`} className="key-point-card">
                <p className="key-point-card__text">{point.text}</p>
                <div className="key-point-card__meta">
                  <span className="key-point-card__confidence">{pointConfidenceLabel(point.confidence)}</span>
                  {point.citation_ids.map((citationId, citationIndex) => {
                    const citation = citationLookup.get(citationId);
                    if (!citation) {
                      return null;
                    }
                    const ordinal = citationOrdinals.get(citationId) ?? citationOrdinal(citationId, citationIndex + 1);
                    return (
                      <a
                        key={citationId}
                        className="citation-chip"
                        href={citation.url}
                        target="_blank"
                        rel="noreferrer"
                        title={citation.url}
                      >
                        [{ordinal}] {citationDisplayText(citation)}
                      </a>
                    );
                  })}
                </div>
              </li>
            ))}
          </ol>
        </section>
      ) : null}

      {message.error ? <p className="message__error">{message.error}</p> : null}

      {message.role === "assistant" && citations.length > 0 && !hasInlineReferences ? (
        <section className="message__structured">
          <h3 className="message__section-title">References</h3>
          <ol className="reference-list">
            {citations.map((citation, index) => {
              const ordinal = citationOrdinals.get(citation.id) ?? citationOrdinal(citation.id, index + 1);
              return (
                <li key={citation.id} className="reference-card">
                  <a href={citation.url} target="_blank" rel="noreferrer">
                    [{ordinal}] {citationDisplayText(citation)}
                  </a>
                  <span className="reference-card__url">{citation.canonical_url}</span>
                </li>
              );
            })}
          </ol>
        </section>
      ) : null}

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
});

MessageCard.displayName = "MessageCard";

export default function App() {
  const [mode, setMode] = useState<ChatMode>("fast");
  const [composerValue, setComposerValue] = useState("");
  const [chats, setChats] = useState<WebChatSummary[]>([]);
  const [currentChat, setCurrentChat] = useState<WebChatDetail | null>(null);
  const [chatRuntimeById, setChatRuntimeById] = useState<Record<string, ChatRuntime>>({});
  const [selectedChatId, setSelectedChatId] = useState<string | null>(() => storageGetItem(STORAGE_KEY));
  const [loading, setLoading] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [modeMenuOpen, setModeMenuOpen] = useState(false);
  const [pageError, setPageError] = useState<string | null>(null);
  const transcriptRef = useRef<HTMLElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const composerInputRef = useRef<HTMLInputElement | null>(null);
  const modeSelectRef = useRef<HTMLDivElement | null>(null);
  const pendingScrollTargetRef = useRef<TranscriptScrollTarget | null>(null);
  const selectedChatIdRef = useRef<string | null>(selectedChatId);
  const [slashCommandIndex, setSlashCommandIndex] = useState(0);
  const activeRuntime = selectedChatId ? chatRuntimeById[selectedChatId] : null;
  const activeMessages = activeRuntime?.messages ?? [];
  const sending = activeRuntime?.pending ?? false;
  const slashMatches = matchSlashCommands(composerValue);
  const showSlashAutocomplete = slashMatches.length > 0;

  function queueTranscriptScroll(target: TranscriptScrollTarget) {
    pendingScrollTargetRef.current = target;
  }

  function createRuntime(messages: ClientMessage[] = []): ChatRuntime {
    return {
      messages,
      pending: false,
      error: null,
    };
  }

  function ensureChatRuntime(chatId: string, messages: ClientMessage[] = []) {
    setChatRuntimeById((current) => {
      if (current[chatId]) {
        return current;
      }
      return {
        ...current,
        [chatId]: createRuntime(messages),
      };
    });
  }

  function updateChatRuntime(chatId: string, updater: (runtime: ChatRuntime) => ChatRuntime) {
    setChatRuntimeById((current) => {
      const existing = current[chatId] ?? createRuntime();
      return {
        ...current,
        [chatId]: updater(existing),
      };
    });
  }

  function updateExistingChatRuntime(chatId: string, updater: (runtime: ChatRuntime) => ChatRuntime) {
    setChatRuntimeById((current) => {
      const existing = current[chatId];
      if (!existing) {
        return current;
      }
      return {
        ...current,
        [chatId]: updater(existing),
      };
    });
  }

  function pruneChatRuntimes(validIds: Set<string>) {
    setChatRuntimeById((current) => {
      let changed = false;
      const next: Record<string, ChatRuntime> = {};
      for (const [chatId, runtime] of Object.entries(current)) {
        if (validIds.has(chatId)) {
          next[chatId] = runtime;
          continue;
        }
        changed = true;
      }
      return changed ? next : current;
    });
  }

  function summaryForChat(chatId: string): WebChatSummary | null {
    return chats.find((chat) => chat.id === chatId) ?? null;
  }

  function setCurrentChatFromSummary(chatId: string) {
    const summary = summaryForChat(chatId);
    if (!summary) {
      return;
    }
    setCurrentChat({
      id: summary.id,
      title: summary.title,
      created_at: summary.created_at,
      updated_at: summary.updated_at,
      messages: [],
    });
  }

  function updateCurrentChatIfActive(chatId: string, updater: (chat: WebChatDetail) => WebChatDetail) {
    setCurrentChat((current) => {
      if (!current || current.id !== chatId) {
        return current;
      }
      return updater(current);
    });
  }

  async function hydrateChats(preferredChatId?: string | null): Promise<void> {
    const nextChats = await listChats();
    setChats(nextChats);
    pruneChatRuntimes(new Set(nextChats.map((chat) => chat.id)));
    const targetId =
      preferredChatId && nextChats.some((chat) => chat.id === preferredChatId)
        ? preferredChatId
        : selectedChatId && nextChats.some((chat) => chat.id === selectedChatId)
          ? selectedChatId
          : nextChats[0]?.id ?? null;

    if (!targetId) {
      setSelectedChatId(null);
      setCurrentChat(null);
      storageRemoveItem(STORAGE_KEY);
      return;
    }

    const chat = await getChat(targetId);
    setCurrentChat(chat);
    ensureChatRuntime(chat.id, chat.messages.map(messageFromServer));
    setSelectedChatId(targetId);
    storageSetItem(STORAGE_KEY, targetId);
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
        pruneChatRuntimes(new Set(nextChats.map((chat) => chat.id)));
        const storedId = storageGetItem(STORAGE_KEY);
        const targetId =
          storedId && nextChats.some((chat) => chat.id === storedId) ? storedId : nextChats[0]?.id ?? null;
        if (targetId) {
          const chat = await getChat(targetId);
          if (cancelled) {
            return;
          }
          setCurrentChat(chat);
          ensureChatRuntime(chat.id, chat.messages.map(messageFromServer));
          setSelectedChatId(targetId);
          storageSetItem(STORAGE_KEY, targetId);
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
    const pendingTarget = pendingScrollTargetRef.current;
    if (pendingTarget === "last-assistant-head") {
      pendingScrollTargetRef.current = null;
      const transcript = transcriptRef.current;
      if (!transcript) {
        return;
      }
      const assistantCards = transcript.querySelectorAll<HTMLElement>('[data-message-role="assistant"]');
      const lastAssistant = assistantCards.item(assistantCards.length - 1);
      if (lastAssistant) {
        lastAssistant.scrollIntoView({ behavior: "auto", block: "start" });
        return;
      }
      transcript.scrollTop = 0;
      return;
    }

    if (pendingTarget === "bottom" || sending) {
      bottomRef.current?.scrollIntoView({ behavior: sending ? "auto" : "smooth", block: "end" });
      if (!sending) {
        pendingScrollTargetRef.current = null;
      }
    }
  }, [activeMessages, sending]);

  useEffect(() => {
    const className = "deepfind-drawer-open";
    document.body.classList.toggle(className, sidebarOpen);
    return () => {
      document.body.classList.remove(className);
    };
  }, [sidebarOpen]);

  useEffect(() => {
    const className = "deepfind-standalone";
    document.body.classList.toggle(className, isStandalonePwa());
    return () => {
      document.body.classList.remove(className);
    };
  }, []);

  useEffect(() => {
    selectedChatIdRef.current = selectedChatId;
  }, [selectedChatId]);

  useEffect(() => {
    if (!showSlashAutocomplete) {
      setSlashCommandIndex(0);
      return;
    }
    setSlashCommandIndex((current) => Math.min(current, slashMatches.length - 1));
  }, [showSlashAutocomplete, slashMatches.length]);

  useEffect(() => {
    if (!modeMenuOpen) {
      return;
    }

    function handleOutsideClick(event: MouseEvent) {
      const target = event.target as Node | null;
      if (target && modeSelectRef.current?.contains(target)) {
        return;
      }
      setModeMenuOpen(false);
    }

    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setModeMenuOpen(false);
      }
    }

    document.addEventListener("mousedown", handleOutsideClick);
    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("mousedown", handleOutsideClick);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [modeMenuOpen]);

  useEffect(() => {
    if (!sidebarOpen) {
      return;
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setSidebarOpen(false);
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [sidebarOpen]);

  async function handleOpenChat(chatId: string) {
    setLoading(true);
    setPageError(null);
    try {
      queueTranscriptScroll("last-assistant-head");
      if (!chatRuntimeById[chatId]) {
        const chat = await getChat(chatId);
        setCurrentChat(chat);
        ensureChatRuntime(chat.id, chat.messages.map(messageFromServer));
      } else {
        setCurrentChatFromSummary(chatId);
      }
      setSelectedChatId(chatId);
      storageSetItem(STORAGE_KEY, chatId);
      setSidebarOpen(false);
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
      queueTranscriptScroll("bottom");
      setCurrentChat(chat);
      ensureChatRuntime(chat.id, []);
      setSelectedChatId(chat.id);
      setChats((current) => upsertSummary(current, summaryFromChat(chat)));
      storageSetItem(STORAGE_KEY, chat.id);
      setSidebarOpen(false);
    } catch (error) {
      setPageError(error instanceof Error ? error.message : "Failed to create chat");
    }
  }

  async function handleDeleteChat(chat: WebChatSummary) {
    setPageError(null);
    try {
      await deleteChat(chat.id);
      setChatRuntimeById((current) => {
        const runtime = current[chat.id];
        if (!runtime) {
          return current;
        }
        runtime.abortController?.abort();
        const next = { ...current };
        delete next[chat.id];
        return next;
      });
      if (chat.id === currentChat?.id || chat.id === selectedChatId) {
        await hydrateChats(null);
        return;
      }
      setChats((current) => current.filter((item) => item.id !== chat.id));
    } catch (error) {
      setPageError(error instanceof Error ? error.message : "Failed to delete chat");
    }
  }

  async function ensureActiveChat(content: string): Promise<WebChatDetail> {
    if (currentChat) {
      if (!chatRuntimeById[currentChat.id]) {
        ensureChatRuntime(currentChat.id, []);
      }
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
    storageSetItem(STORAGE_KEY, chat.id);
    ensureChatRuntime(chat.id, []);
    return titledChat;
  }

  function appendActivity(chatId: string, messageId: string, event: ProgressEvent) {
    if (event.type === "answer_delta") {
      return;
    }
    updateExistingChatRuntime(chatId, (runtime) => ({
      ...runtime,
      messages: runtime.messages.map((message) =>
        message.id === messageId
          ? {
              ...message,
              activity: [...(message.activity ?? []), event],
            }
          : message,
      ),
    }));
  }

  function applyFinalTurn(chatId: string, messageId: string, turnResult: TurnResult) {
    updateExistingChatRuntime(chatId, (runtime) => ({
      ...runtime,
      messages: runtime.messages.map((message) =>
        message.id === messageId
          ? {
              ...message,
              content: turnResult.answer_markdown,
              mode: turnResult.mode,
              sources: turnResult.sources,
              artifacts: turnResult.artifacts,
              key_points: turnResult.key_points ?? [],
              citations: turnResult.citations ?? [],
              pending: false,
              error: null,
            }
          : message,
      ),
    }));
  }

  function selectSlashCommand(command: string) {
    setComposerValue(command);
    setSlashCommandIndex(0);
    composerInputRef.current?.focus();
  }

  function handleComposerKeyDown(keyEvent: ReactKeyboardEvent<HTMLInputElement>) {
    if (showSlashAutocomplete) {
      if (keyEvent.key === "ArrowDown") {
        keyEvent.preventDefault();
        setSlashCommandIndex((current) => (current + 1) % slashMatches.length);
        return;
      }
      if (keyEvent.key === "ArrowUp") {
        keyEvent.preventDefault();
        setSlashCommandIndex((current) => (current - 1 + slashMatches.length) % slashMatches.length);
        return;
      }
      if (keyEvent.key === "Tab") {
        const selected = slashMatches[slashCommandIndex];
        if (selected) {
          keyEvent.preventDefault();
          selectSlashCommand(selected.command);
        }
        return;
      }
    }

    if (keyEvent.key === "Enter" && !keyEvent.shiftKey) {
      keyEvent.preventDefault();
      keyEvent.currentTarget.form?.requestSubmit();
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    let content = composerValue.trim();
    if (slashMatches.length > 0) {
      content = resolveSlashCommand(content, slashMatches, slashCommandIndex);
    }

    if (sending || !content) {
      return;
    }

    setComposerValue("");
    setPageError(null);

    const controller = new AbortController();
    let activeChat: WebChatDetail | null = null;

    try {
      const chat = await ensureActiveChat(content);
      activeChat = chat;
      const userMessage = newClientMessage("user", content, mode);
      const assistantMessage = newClientMessage("assistant", "", mode);
      const nextTitle = currentChat?.title && currentChat.title !== "New chat" ? currentChat.title : summarize(content, 48);
      const chatSnapshot = {
        ...chat,
        title: nextTitle || chat.title,
        updated_at: assistantMessage.created_at,
      };
      const priorMessages = chatRuntimeById[chat.id]?.messages ?? [];

      queueTranscriptScroll("bottom");
      setCurrentChat(chatSnapshot);
      updateChatRuntime(chat.id, (runtime) => ({
        ...runtime,
        messages: [...runtime.messages, userMessage, assistantMessage],
        pending: true,
        error: null,
        abortController: controller,
      }));
      setChats((current) =>
        upsertSummary(current, summaryFromChat({ ...chatSnapshot, messages: [...priorMessages, userMessage] })),
      );

      await streamChatMessage(
        chat.id,
        { content, mode },
        (progressEvent) => {
          appendActivity(chat.id, assistantMessage.id, progressEvent);
          if (progressEvent.type === "answer_delta") {
            const delta = String(progressEvent.data.delta ?? "");
            updateExistingChatRuntime(chat.id, (runtime) => ({
              ...runtime,
              messages: runtime.messages.map((message) =>
                message.id === assistantMessage.id
                  ? {
                      ...message,
                      content: `${message.content}${delta}`,
                    }
                  : message,
              ),
            }));
          }
          if (progressEvent.type === "answer_final") {
            applyFinalTurn(chat.id, assistantMessage.id, progressEvent.data as unknown as TurnResult);
          }
          if (progressEvent.type === "error") {
            const text = String(progressEvent.data.message ?? "Something went wrong");
            updateExistingChatRuntime(chat.id, (runtime) => ({
              ...runtime,
              pending: false,
              error: text,
              abortController: undefined,
              messages: runtime.messages.map((message) =>
                message.id === assistantMessage.id
                  ? {
                      ...message,
                      pending: false,
                      error: text,
                      content: message.content || "The run ended before a final answer was produced.",
                    }
                  : message,
              ),
            }));
            if (selectedChatIdRef.current === chat.id) {
              setPageError(text);
            }
          }
        },
        { signal: controller.signal },
      );

      const nextChats = await listChats();
      setChats(nextChats);
      pruneChatRuntimes(new Set(nextChats.map((chat) => chat.id)));
      const refreshedSummary = nextChats.find((item) => item.id === chat.id);
      if (refreshedSummary) {
        updateCurrentChatIfActive(chat.id, (current) => ({
          ...current,
          title: refreshedSummary.title,
          updated_at: refreshedSummary.updated_at,
        }));
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return;
      }
      const message = error instanceof Error ? error.message : "Failed to send message";
      if (activeChat && selectedChatIdRef.current === activeChat.id) {
        setPageError(message);
      }
      if (activeChat) {
        updateExistingChatRuntime(activeChat.id, (runtime) => ({
          ...runtime,
          pending: false,
          error: message,
          abortController: undefined,
        }));
      }
    } finally {
      if (activeChat) {
        updateExistingChatRuntime(activeChat.id, (runtime) => ({
          ...runtime,
          pending: false,
          abortController: undefined,
        }));
      }
    }
  }

  const selectedSummary = selectedChatId ? summaryForChat(selectedChatId) : null;
  const selectedTitle =
    currentChat?.id === selectedChatId ? currentChat.title : selectedSummary?.title ?? "New chat";

  return (
    <div className={`app-shell${sidebarOpen ? " app-shell--sidebar-open" : ""}`}>
      <button
        className="sidebar-backdrop"
        type="button"
        aria-label="Close chats"
        aria-hidden={!sidebarOpen}
        onClick={() => setSidebarOpen(false)}
      />

      <aside id="chat-sidebar" className="sidebar">
        <div className="sidebar__header">
          <div>
            <h1>DeepFind-CLI</h1>
          </div>
          <div className="sidebar__actions">
            <button className="ghost-button" type="button" onClick={handleCreateChat}>
              New chat
            </button>
          </div>
        </div>

        <div className="sidebar__list">
          {chats.length === 0 ? <p className="sidebar__empty">No saved chats yet.</p> : null}
          {chats.map((chat) => {
            const isRunning = Boolean(chatRuntimeById[chat.id]?.pending);
            return (
              <div
                key={chat.id}
                className={`chat-tile${chat.id === selectedChatId ? " chat-tile--active" : ""}`}
              >
                {isRunning ? (
                  <span className="chat-tile__badge" aria-hidden="true">
                    Running
                  </span>
                ) : null}
                <button
                  className="chat-tile__open"
                  type="button"
                  aria-label={chat.title}
                  onClick={() => void handleOpenChat(chat.id)}
                  title={chat.title}
                >
                  <strong className="chat-tile__title">{sidebarTitle(chat.title)}</strong>
                </button>
                <button
                  className="chat-tile__delete"
                  type="button"
                  aria-label={`Delete ${chat.title}`}
                  title={`Delete ${chat.title}`}
                  onClick={() => void handleDeleteChat(chat)}
                >
                  x
                </button>
              </div>
            );
          })}
        </div>
      </aside>

      <main className="workspace">
        <header className="workspace__header">
          <div className="workspace__title">
            <h2>{selectedTitle}</h2>
          </div>
          <div className="workspace__actions">
            <button
              className="ghost-button workspace__menu-button mobile-only"
              type="button"
              aria-controls="chat-sidebar"
              aria-expanded={sidebarOpen}
              aria-label="Open chats"
              onClick={() => setSidebarOpen((current) => !current)}
            >
              <span aria-hidden="true">☰</span>
            </button>
            <span className="workspace__mobile-title mobile-only">{selectedTitle}</span>
          </div>
        </header>

        <section ref={transcriptRef} className="transcript">
          {loading ? <p className="state-text">Loading chats...</p> : null}
          {!loading && activeMessages.length === 0 ? (
            <div className="hero-empty">
              <p className="eyebrow">Parallel web research</p>
              <h3>Ask for live research, then decide how much horsepower you want.</h3>
              <p>
                Fast keeps it lean with one agent. Expert fans out to four agents and returns with a denser brief,
                sources, and any generated assets.
              </p>
            </div>
          ) : null}

          {activeMessages.map((message) => (
            <MessageCard key={message.id} message={message} />
          ))}

          <div ref={bottomRef} />
        </section>

        <footer className="composer-wrap">
          {showSlashAutocomplete ? (
            <div id="slash-command-menu" className="composer-autocomplete" role="listbox" aria-label="Slash commands">
              <div className="composer-autocomplete__label">Slash commands</div>
              <div className="composer-autocomplete__list">
                {slashMatches.map((command, index) => (
                  <button
                    key={command.command}
                    type="button"
                    role="option"
                    className={`composer-autocomplete__option${
                      index === slashCommandIndex ? " composer-autocomplete__option--active" : ""
                    }`}
                    aria-selected={index === slashCommandIndex}
                    onMouseEnter={() => setSlashCommandIndex(index)}
                    onMouseDown={(mouseEvent) => {
                      mouseEvent.preventDefault();
                      selectSlashCommand(command.command);
                    }}
                  >
                    <span className="composer-autocomplete__command">{command.command}</span>
                    <span className="composer-autocomplete__description">{command.description}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          <form className="composer" onSubmit={handleSubmit}>
            <label className="sr-only" htmlFor="chat-input">
              Ask DeepFind
            </label>
            <input
              ref={composerInputRef}
              id="chat-input"
              className="composer__input"
              type="text"
              placeholder="Ask DeepFind to search, summarize, compare, or generate artifacts..."
              aria-autocomplete="list"
              aria-controls={showSlashAutocomplete ? "slash-command-menu" : undefined}
              aria-expanded={showSlashAutocomplete}
              value={composerValue}
              onChange={(changeEvent) => setComposerValue(changeEvent.target.value)}
              onKeyDown={handleComposerKeyDown}
            />
            <div className="mode-select" ref={modeSelectRef}>
              <span className="sr-only" id="mode-select-label">
                Mode
              </span>
              <button
                type="button"
                className="mode-select__button"
                aria-label="Mode"
                aria-haspopup="listbox"
                aria-expanded={modeMenuOpen}
                aria-controls="mode-select-menu"
                onClick={() => setModeMenuOpen((current) => !current)}
              >
                <span className="mode-select__value">{modeLabel(mode)}</span>
              </button>
              {modeMenuOpen ? (
                <div id="mode-select-menu" className="mode-select__menu" role="listbox" aria-labelledby="mode-select-label">
                  {(["fast", "expert"] as const).map((option) => (
                    <button
                      key={option}
                      type="button"
                      role="option"
                      className={`mode-select__option${mode === option ? " mode-select__option--active" : ""}`}
                      aria-selected={mode === option}
                      onClick={() => {
                        setMode(option);
                        setModeMenuOpen(false);
                      }}
                    >
                      {modeLabel(option)}
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
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
