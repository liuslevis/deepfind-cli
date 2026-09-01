export type ChatMode = "fast" | "expert";
export type ResearchMode = "deep_research" | "research" | "chat";
export type ArtifactKind = "image" | "slides" | "file";
export type MessageRole = "user" | "assistant";
export type ModelTarget = "qwen" | "mimo" | "minimax" | "glm" | "deepseek" | "gpu";
export type ServerModelTarget = ModelTarget | "cloud";

export interface ArtifactLink {
  kind: ArtifactKind;
  label: string;
  path: string;
  url: string;
}

export interface CitationLink {
  id: string;
  canonical_url: string;
  url: string;
  title: string;
  publisher: string;
}

export interface KeyPoint {
  text: string;
  citation_ids: string[];
  confidence: string;
}

export interface GpuInfo {
  available: boolean;
  name: string;
  memory_total_mb: number | null;
}

export interface LocalModelInfo {
  available: boolean;
  backend: string;
  model: string;
  base_url: string;
  reason: string;
  gpu: GpuInfo;
}

export interface ToolOption {
  name: string;
  description: string;
  parameters: string[];
}

export interface WebMessage {
  id: string;
  role: MessageRole;
  content: string;
  created_at: string;
  mode: ChatMode | null;
  research_mode?: ResearchMode;
  sources: string[];
  artifacts: ArtifactLink[];
  key_points?: KeyPoint[];
  citations?: CitationLink[];
  model_target?: ServerModelTarget;
  model_label?: string;
}

export interface WebChatSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  preview: string;
}

export interface WebChatDetail {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: WebMessage[];
}

export interface ChatListResponse {
  chats: WebChatSummary[];
  local_model?: LocalModelInfo | null;
  tools?: ToolOption[];
}

export interface TurnResult {
  answer_markdown: string;
  sources: string[];
  artifacts: ArtifactLink[];
  key_points?: KeyPoint[];
  citations?: CitationLink[];
  mode: ChatMode;
  research_mode?: ResearchMode;
  model_target?: ServerModelTarget;
  model_label?: string;
}

export interface ProgressEvent {
  type: string;
  timestamp: string;
  data: Record<string, unknown>;
}

export type WorkspaceApp = "terminal" | "pdf" | "excel" | "word" | "files";

export interface WorkspaceStatus {
  available: boolean;
  status: "running" | "stopped" | "missing";
  chat_id: string;
  container_id?: string;
  workspace_root?: string;
}

export interface WorkspaceFile {
  name: string;
  path: string;
  size: number;
  modified_at: string;
  is_directory: boolean;
  mime_type: string;
}

export interface WorkspaceListing {
  path: string;
  entries: WorkspaceFile[];
  truncated: boolean;
}

export interface WorkspaceTab {
  id: string;
  app: WorkspaceApp;
  title: string;
  chatId: string;
  relativePath?: string;
  terminalId?: string;
  status: "loading" | "ready" | "running" | "disconnected" | "error";
}

export interface WorkbookMetadata {
  file: WorkspaceFile;
  sheets: Array<{
    id: string;
    name: string;
    rows: number;
    columns: number;
    hidden: boolean;
    freeze_panes?: string;
  }>;
  warnings: string[];
}

export interface SheetRange {
  sheet_id: string;
  range: string;
  cells: Array<Array<{ value: unknown; display: string; formula: string | null; number_format?: string }>>;
  truncated: boolean;
}

export interface WordDocument {
  title: string;
  outline: Array<{ id: string; level: number; text: string }>;
  html: string;
  warnings: string[];
}

export type ActivityPhase = "planning" | "researching" | "synthesizing" | "complete" | "error";

export interface ActivitySummary {
  phase: ActivityPhase;
  label: string;
  text: string;
  hiddenCount: number;
}
