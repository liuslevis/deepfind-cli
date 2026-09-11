import { useEffect, useMemo, useRef, useState } from "react";
import DOMPurify from "dompurify";
import "@xterm/xterm/css/xterm.css";
import type { PDFDocumentProxy } from "pdfjs-dist";

import {
  closeWorkspaceTerminal,
  createWorkspaceTerminal,
  getAuthToken,
  getSheetRange,
  getWordDocument,
  getWorkbook,
  getWorkspaceStatus,
  listWorkspaceTerminals,
  listWorkspaceFiles,
  workspaceContentUrl,
  workspacePdfUrl,
  workspaceTerminalUrl,
} from "./api";
import type {
  SheetRange,
  WordDocument,
  WorkbookMetadata,
  WorkspaceFile,
  WorkspaceTab,
} from "./types";

const PANEL_WIDTH_KEY = "deepfind.workspace.width";
const MIN_PANEL_WIDTH = 360;
const DEFAULT_PANEL_WIDTH = 640;

export interface WorkspaceOpenRequest {
  id: number;
  path: string;
  title?: string;
}

export interface WorkspaceCommandRequest {
  id: number;
  command: string;
}

interface WorkspacePanelProps {
  chatId: string | null;
  visible: boolean;
  onVisibleChange: (visible: boolean) => void;
  openRequest?: WorkspaceOpenRequest | null;
  commandRequest?: WorkspaceCommandRequest | null;
}

function storageKey(chatId: string): string {
  return `deepfind.workspace.tabs.${chatId}`;
}

function loadWidth(): number {
  const value = Number(localStorage.getItem(PANEL_WIDTH_KEY));
  return Number.isFinite(value) && value >= MIN_PANEL_WIDTH ? value : DEFAULT_PANEL_WIDTH;
}

function loadWorkspaceState(chatId: string): { tabs: WorkspaceTab[]; selectedTabId: string | null } {
  try {
    const parsed = JSON.parse(localStorage.getItem(storageKey(chatId)) ?? "[]") as
      | WorkspaceTab[]
      | { tabs?: WorkspaceTab[]; selectedTabId?: string | null };
    const tabs = Array.isArray(parsed) ? parsed : parsed.tabs ?? [];
    const restored = tabs.map((tab) => ({
      ...tab,
      status: tab.app === "terminal" ? "disconnected" as const : "loading" as const,
    }));
    const selectedTabId = Array.isArray(parsed) ? restored[0]?.id ?? null : parsed.selectedTabId ?? restored[0]?.id ?? null;
    return { tabs: restored, selectedTabId };
  } catch {
    return { tabs: [], selectedTabId: null };
  }
}

function appForPath(path: string): WorkspaceTab["app"] {
  const extension = path.split(".").pop()?.toLowerCase();
  if (extension === "pdf") return "pdf";
  if (extension && ["xlsx", "xls", "csv", "tsv"].includes(extension)) return "excel";
  if (extension === "docx") return "word";
  return "files";
}

function tabId(app: WorkspaceTab["app"], path?: string): string {
  return `${app}_${path ?? crypto.randomUUID()}`;
}

function authHeaders(): HeadersInit {
  const token = getAuthToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function FileBrowser({
  chatId,
  onOpen,
}: {
  chatId: string;
  onOpen: (file: WorkspaceFile) => void;
}) {
  const [path, setPath] = useState(".");
  const [files, setFiles] = useState<WorkspaceFile[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");

  useEffect(() => {
    let active = true;
    setLoading(true);
    listWorkspaceFiles(chatId, path)
      .then((listing) => {
        if (active) {
          setFiles(listing.entries);
          setError(null);
        }
      })
      .catch((reason: unknown) => active && setError(reason instanceof Error ? reason.message : "Unable to list files"))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [chatId, path]);

  const visibleFiles = files.filter((file) => file.name.toLowerCase().includes(query.trim().toLowerCase()));
  const parent = path === "." ? null : path.split("/").slice(0, -1).join("/") || ".";

  return (
    <div className="workspace-files">
      <div className="workspace-toolbar">
        <button type="button" disabled={!parent} onClick={() => parent && setPath(parent)}>
          Up
        </button>
        <span className="workspace-path">/{path === "." ? "" : path}</span>
        <input
          aria-label="Search files"
          placeholder="Search this folder"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </div>
      {loading ? <p className="state-text">Loading workspace files...</p> : null}
      {error ? <p className="workspace-error">{error}</p> : null}
      <div className="workspace-file-list" role="list">
        {visibleFiles.map((file) => (
          <button
            type="button"
            role="listitem"
            key={file.path}
            className="workspace-file"
            onClick={() => (file.is_directory ? setPath(file.path) : onOpen(file))}
          >
            <span aria-hidden="true">{file.is_directory ? "DIR" : "FILE"}</span>
            <strong>{file.name}</strong>
            <small>{file.is_directory ? "Folder" : `${Math.ceil(file.size / 1024)} KB`}</small>
          </button>
        ))}
      </div>
    </div>
  );
}

function FilePreview({ chatId, path }: { chatId: string; path: string }) {
  const [content, setContent] = useState<string>("");
  const [imageUrl, setImageUrl] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const image = /\.(png|jpe?g|gif|webp|svg)$/i.test(path);

  useEffect(() => {
    const controller = new AbortController();
    let objectUrl = "";
    fetch(workspaceContentUrl(chatId, path), { headers: authHeaders(), signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(await response.text());
        return image ? response.blob() : response.text();
      })
      .then((value) => {
        if (typeof value === "string") {
          setContent(value);
        } else {
          objectUrl = URL.createObjectURL(value);
          setImageUrl(objectUrl);
        }
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Unable to preview file");
      });
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [chatId, image, path]);

  if (image) {
    return imageUrl
      ? <img className="workspace-image" src={imageUrl} alt={path.split("/").pop() ?? path} />
      : <p className="state-text">Loading image...</p>;
  }
  if (error) return <p className="workspace-error">{error}</p>;
  return <pre className="workspace-text-preview">{content || "Loading preview..."}</pre>;
}

function PdfViewer({ chatId, path }: { chatId: string; path: string }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [documentProxy, setDocumentProxy] = useState<PDFDocumentProxy | null>(null);
  const [page, setPage] = useState(1);
  const [scale, setScale] = useState(1.2);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let task: ReturnType<(typeof import("pdfjs-dist"))["getDocument"]> | null = null;
    let active = true;
    void import("pdfjs-dist").then((pdfjs) => {
      if (!active) return;
      pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();
      task = pdfjs.getDocument({
        url: workspacePdfUrl(chatId, path),
        httpHeaders: authHeaders() as Record<string, string>,
      });
      return task.promise;
    }).then((document) => {
      if (active && document) setDocumentProxy(document);
    }).catch((reason: unknown) => {
      if (active) setError(reason instanceof Error ? reason.message : "Unable to load PDF");
    });
    return () => {
      active = false;
      if (task) void task.destroy();
    };
  }, [chatId, path]);

  useEffect(() => {
    if (!documentProxy || !canvasRef.current) return;
    let cancelled = false;
    documentProxy.getPage(page).then((pdfPage) => {
      if (cancelled || !canvasRef.current) return;
      const viewport = pdfPage.getViewport({ scale });
      const canvas = canvasRef.current;
      const context = canvas.getContext("2d");
      if (!context) return;
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      void pdfPage.render({ canvas, canvasContext: context, viewport }).promise;
    });
    return () => {
      cancelled = true;
    };
  }, [documentProxy, page, scale]);

  if (error) return <p className="workspace-error">{error}</p>;
  return (
    <div className="workspace-document">
      <div className="workspace-toolbar">
        <button type="button" disabled={page <= 1} onClick={() => setPage((value) => value - 1)}>Previous</button>
        <span>Page {page} / {documentProxy?.numPages ?? "..."}</span>
        <button
          type="button"
          disabled={!documentProxy || page >= documentProxy.numPages}
          onClick={() => setPage((value) => value + 1)}
        >
          Next
        </button>
        <button type="button" onClick={() => setScale((value) => Math.max(0.5, value - 0.2))}>-</button>
        <button type="button" onClick={() => setScale((value) => Math.min(3, value + 0.2))}>+</button>
      </div>
      <div className="workspace-pdf-canvas"><canvas ref={canvasRef} /></div>
    </div>
  );
}

function ExcelViewer({ chatId, path }: { chatId: string; path: string }) {
  const [workbook, setWorkbook] = useState<WorkbookMetadata | null>(null);
  const [sheet, setSheet] = useState<SheetRange | null>(null);
  const [selectedSheet, setSelectedSheet] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getWorkbook(chatId, path)
      .then((value) => {
        setWorkbook(value);
        setSelectedSheet(value.sheets[0]?.id ?? "");
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "Unable to parse workbook"));
  }, [chatId, path]);

  useEffect(() => {
    if (!selectedSheet) return;
    getSheetRange(chatId, path, selectedSheet, "A1:Z100")
      .then(setSheet)
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "Unable to load worksheet"));
  }, [chatId, path, selectedSheet]);

  if (error) return <p className="workspace-error">{error}</p>;
  return (
    <div className="workspace-document">
      <div className="workspace-toolbar">
        <select aria-label="Worksheet" value={selectedSheet} onChange={(event) => setSelectedSheet(event.target.value)}>
          {workbook?.sheets.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
        </select>
        {workbook?.warnings.map((warning) => <span className="workspace-warning" key={warning}>{warning}</span>)}
      </div>
      <div className="workspace-sheet-scroll">
        <table className="workspace-sheet">
          <tbody>
            {sheet?.cells.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((cell, columnIndex) => (
                  <td key={columnIndex} title={cell.formula ?? cell.number_format ?? ""}>
                    {cell.display}
                    {cell.formula ? <small>{cell.formula}</small> : null}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function WordViewer({ chatId, path }: { chatId: string; path: string }) {
  const [document, setDocument] = useState<WordDocument | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    getWordDocument(chatId, path)
      .then(setDocument)
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "Unable to parse document"));
  }, [chatId, path]);
  if (error) return <p className="workspace-error">{error}</p>;
  if (!document) return <p className="state-text">Loading document...</p>;
  return (
    <div className="workspace-word">
      <aside aria-label="Document outline">
        {document.outline.map((heading) => <a key={heading.id} href={`#${heading.id}`}>{heading.text}</a>)}
      </aside>
      <article dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(document.html) }} />
    </div>
  );
}

function TerminalView({
  chatId,
  terminalId,
  onExit,
  initialCommand,
  onCommandSent,
}: {
  chatId: string;
  terminalId?: string;
  onExit: () => void;
  initialCommand?: string;
  onCommandSent?: () => void;
}) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const initialCommandRef = useRef(initialCommand);
  const onExitRef = useRef(onExit);
  const onCommandSentRef = useRef(onCommandSent);
  onExitRef.current = onExit;
  onCommandSentRef.current = onCommandSent;
  useEffect(() => {
    if (!terminalId || !hostRef.current) return;
    let active = true;
    let cleanup = () => {};
    void Promise.all([import("@xterm/xterm"), import("@xterm/addon-fit")]).then(([xterm, addon]) => {
      if (!active || !hostRef.current) return;
      const terminal = new xterm.Terminal({ convertEol: true, cursorBlink: true, scrollback: 10_000 });
      const fit = new addon.FitAddon();
      terminal.loadAddon(fit);
      terminal.open(hostRef.current);
      fit.fit();
      terminal.focus();
      terminal.writeln("\x1b[33mDocker shell - commands may modify this chat workspace.\x1b[0m");
      const socket = new WebSocket(workspaceTerminalUrl(chatId, terminalId));
      socket.addEventListener("open", () => {
        socket.send(JSON.stringify({ type: "resize", cols: terminal.cols, rows: terminal.rows }));
        if (initialCommandRef.current) {
          socket.send(JSON.stringify({ type: "input", data: `${initialCommandRef.current}\n` }));
          initialCommandRef.current = undefined;
          onCommandSentRef.current?.();
        }
      });
      socket.addEventListener("message", (event) => {
        const message = JSON.parse(String(event.data)) as { type: string; data?: string; exit_code?: number };
        if (message.type === "output") terminal.write(message.data ?? "");
        if (message.type === "exit") {
          terminal.writeln(`\r\n[process exited: ${message.exit_code ?? "signal"}]`);
          onExitRef.current();
        }
      });
      socket.addEventListener("close", (event) => {
        if (active && event.code !== 1000) {
          terminal.writeln(`\r\n\x1b[31mTerminal connection closed (${event.code || "network error"}). Refresh or open a new terminal.\x1b[0m`);
        }
      });
      const input = terminal.onData((data) => socket.readyState === WebSocket.OPEN && socket.send(JSON.stringify({ type: "input", data })));
      const resize = terminal.onResize(({ cols, rows }) => socket.readyState === WebSocket.OPEN && socket.send(JSON.stringify({ type: "resize", cols, rows })));
      const observer = new ResizeObserver(() => fit.fit());
      observer.observe(hostRef.current);
      cleanup = () => {
        observer.disconnect();
        input.dispose();
        resize.dispose();
        socket.close();
        terminal.dispose();
      };
    });
    return () => {
      active = false;
      cleanup();
    };
  }, [chatId, terminalId]);

  if (!terminalId) {
    return <p className="workspace-warning">This terminal process was not restored after refresh. Open a new terminal.</p>;
  }
  return <div ref={hostRef} className="workspace-terminal" aria-label="Docker terminal output" />;
}

export default function WorkspacePanel({
  chatId,
  visible,
  onVisibleChange,
  openRequest,
  commandRequest,
}: WorkspacePanelProps) {
  const [tabs, setTabs] = useState<WorkspaceTab[]>([]);
  const [tabsChatId, setTabsChatId] = useState<string | null>(null);
  const [selectedTabId, setSelectedTabId] = useState<string | null>(null);
  const [width, setWidth] = useState(loadWidth);
  const [available, setAvailable] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const dragRef = useRef<{ startX: number; startWidth: number } | null>(null);
  const handledCommandRequestIdRef = useRef<number | null>(null);
  const [initialCommands, setInitialCommands] = useState<Record<string, string>>({});

  useEffect(() => {
    let active = true;
    if (!chatId) {
      setTabs([]);
      setTabsChatId(null);
      setSelectedTabId(null);
      return () => {
        active = false;
      };
    }
    const restored = loadWorkspaceState(chatId);
    setAvailable(null);
    setError(null);
    setTabs(restored.tabs);
    setTabsChatId(chatId);
    setSelectedTabId(restored.selectedTabId);
    getWorkspaceStatus(chatId)
      .then(async (status) => {
        if (!active) return;
        setAvailable(status.available);
        if (!status.available) return;
        const liveTerminals = await listWorkspaceTerminals(chatId);
        if (!active) return;
        const liveIds = new Set(liveTerminals.map((terminal) => terminal.terminal_id));
        setTabs((current) => current.map((tab) =>
          tab.app === "terminal" && tab.terminalId && liveIds.has(tab.terminalId)
            ? { ...tab, status: "ready" }
            : tab,
        ));
      })
      .catch((reason: unknown) => {
        if (!active) return;
        setAvailable(false);
        setError(reason instanceof Error ? reason.message : "Container workspace unavailable");
      });
    return () => {
      active = false;
    };
  }, [chatId]);

  useEffect(() => {
    if (!chatId || tabsChatId !== chatId) return;
    localStorage.setItem(storageKey(chatId), JSON.stringify({ tabs, selectedTabId }));
  }, [chatId, selectedTabId, tabs, tabsChatId]);

  const selectedTab = useMemo(() => tabs.find((tab) => tab.id === selectedTabId) ?? null, [selectedTabId, tabs]);

  function openPath(path: string, title = path.split("/").pop() ?? path) {
    if (!chatId) return;
    const app = appForPath(path);
    const id = tabId(app, path);
    setTabs((current) => current.some((tab) => tab.id === id)
      ? current
      : [...current, { id, app, title, chatId, relativePath: path, status: "loading" }]);
    setSelectedTabId(id);
    onVisibleChange(true);
  }

  useEffect(() => {
    if (openRequest) openPath(openRequest.path, openRequest.title);
    // The request id intentionally controls one-time external opens.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openRequest?.id]);

  async function openTerminal(initialCommand?: string) {
    if (!chatId) return;
    try {
      const terminal = await createWorkspaceTerminal(chatId);
      const id = tabId("terminal");
      setTabs((current) => [...current, {
        id,
        app: "terminal",
        title: `Terminal ${current.filter((tab) => tab.app === "terminal").length + 1}`,
        chatId,
        terminalId: terminal.terminal_id,
        status: "ready",
      }]);
      if (initialCommand) {
        setInitialCommands((current) => ({ ...current, [terminal.terminal_id]: initialCommand }));
      }
      setSelectedTabId(id);
      onVisibleChange(true);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to create terminal");
    }
  }

  useEffect(() => {
    if (!commandRequest || handledCommandRequestIdRef.current === commandRequest.id) return;
    handledCommandRequestIdRef.current = commandRequest.id;
    void openTerminal(commandRequest.command);
    // The request id intentionally controls one approved command.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [commandRequest?.id]);

  function ensureFilesTab() {
    if (!chatId) return;
    const id = `files_${chatId}`;
    setTabs((current) => current.some((tab) => tab.id === id)
      ? current
      : [...current, { id, app: "files", title: "Files", chatId, status: "ready" }]);
    setSelectedTabId(id);
    onVisibleChange(true);
  }

  async function closeTab(tab: WorkspaceTab) {
    if (tab.app === "terminal" && tab.terminalId) {
      if (!window.confirm("Close this running terminal and terminate its process?")) return;
      try {
        await closeWorkspaceTerminal(tab.chatId, tab.terminalId);
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "Unable to close terminal");
        return;
      }
    }
    setTabs((current) => {
      const index = current.findIndex((item) => item.id === tab.id);
      const next = current.filter((item) => item.id !== tab.id);
      if (selectedTabId === tab.id) setSelectedTabId(next[Math.max(0, index - 1)]?.id ?? next[0]?.id ?? null);
      return next;
    });
  }

  function moveTab(tabId: string, direction: -1 | 1) {
    setTabs((current) => {
      const index = current.findIndex((tab) => tab.id === tabId);
      const destination = index + direction;
      if (index < 0 || destination < 0 || destination >= current.length) return current;
      const next = [...current];
      [next[index], next[destination]] = [next[destination], next[index]];
      return next;
    });
  }

  useEffect(() => {
    function move(event: MouseEvent) {
      if (!dragRef.current) return;
      setWidth(Math.max(MIN_PANEL_WIDTH, dragRef.current.startWidth + dragRef.current.startX - event.clientX));
    }
    function stop() {
      if (!dragRef.current) return;
      dragRef.current = null;
      localStorage.setItem(PANEL_WIDTH_KEY, String(width));
    }
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", stop);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", stop);
    };
  }, [width]);

  if (!visible) return null;
  return (
    <aside className="manus-workspace" style={{ width }} aria-label="Chat workspace">
      <button
        type="button"
        className="workspace-resizer"
        aria-label="Resize workspace panel"
        onMouseDown={(event) => {
          dragRef.current = { startX: event.clientX, startWidth: width };
        }}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") setWidth((value) => value + 24);
          if (event.key === "ArrowRight") setWidth((value) => Math.max(MIN_PANEL_WIDTH, value - 24));
        }}
      />
      <div className="workspace-panel-header">
        <div>
          <strong>Workspace</strong>
          <span>{available ? "Docker connected" : available === false ? "Unavailable" : "Connecting..."}</span>
        </div>
        <div>
          <button type="button" onClick={ensureFilesTab} disabled={!chatId || !available}>Files</button>
          <button type="button" onClick={() => void openTerminal()} disabled={!chatId || !available}>Terminal</button>
          <button type="button" onClick={() => onVisibleChange(false)} aria-label="Collapse workspace">Close</button>
        </div>
      </div>
      {error ? <p className="workspace-error">{error}</p> : null}
      <div className="workspace-tabs" role="tablist" aria-label="Open workspace items">
        {tabs.map((tab) => (
          <div className="workspace-tab-wrap" key={tab.id}>
            <button
              type="button"
              className="workspace-tab-move"
              aria-label={`Move ${tab.title} left`}
              onClick={() => moveTab(tab.id, -1)}
            >
              &lt;
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab.id === selectedTabId}
              className={tab.id === selectedTabId ? "workspace-tab workspace-tab--active" : "workspace-tab"}
              onClick={() => setSelectedTabId(tab.id)}
            >
              {tab.title}
            </button>
            <button
              type="button"
              className="workspace-tab-move"
              aria-label={`Move ${tab.title} right`}
              onClick={() => moveTab(tab.id, 1)}
            >
              &gt;
            </button>
            <button type="button" className="workspace-tab-close" aria-label={`Close ${tab.title}`} onClick={() => void closeTab(tab)}>x</button>
          </div>
        ))}
      </div>
      <div className="workspace-tab-content" role="tabpanel">
        {!chatId ? <p className="state-text">Select a chat to open its container workspace.</p> : null}
        {chatId && available === false ? <p className="workspace-error">The chat container is unavailable. Check Docker and the configured workspace image.</p> : null}
        {chatId && available && !selectedTab ? (
          <div className="workspace-launcher">
            <button type="button" onClick={ensureFilesTab}>Browse files</button>
            <button type="button" onClick={() => void openTerminal()}>Open terminal</button>
          </div>
        ) : null}
        {chatId && selectedTab?.app === "files" && !selectedTab.relativePath ? (
          <FileBrowser chatId={chatId} onOpen={(file) => openPath(file.path, file.name)} />
        ) : null}
        {chatId && selectedTab?.app === "files" && selectedTab.relativePath ? <FilePreview chatId={chatId} path={selectedTab.relativePath} /> : null}
        {chatId && selectedTab?.app === "pdf" && selectedTab.relativePath ? <PdfViewer chatId={chatId} path={selectedTab.relativePath} /> : null}
        {chatId && selectedTab?.app === "excel" && selectedTab.relativePath ? <ExcelViewer chatId={chatId} path={selectedTab.relativePath} /> : null}
        {chatId && selectedTab?.app === "word" && selectedTab.relativePath ? <WordViewer chatId={chatId} path={selectedTab.relativePath} /> : null}
        {chatId && selectedTab?.app === "terminal" ? (
          <TerminalView
            chatId={chatId}
            terminalId={selectedTab.terminalId}
            initialCommand={selectedTab.terminalId ? initialCommands[selectedTab.terminalId] : undefined}
            onCommandSent={() => {
              if (!selectedTab.terminalId) return;
              setInitialCommands((current) => {
                const next = { ...current };
                delete next[selectedTab.terminalId!];
                return next;
              });
            }}
            onExit={() => setTabs((current) => current.map((tab) => tab.id === selectedTab.id ? { ...tab, status: "disconnected" } : tab))}
          />
        ) : null}
      </div>
    </aside>
  );
}
