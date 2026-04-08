export type ChatMode = "fast" | "expert";
export type ArtifactKind = "image" | "slides" | "file";
export type MessageRole = "user" | "assistant";
export type ModelTarget = "cloud" | "gpu";

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

export interface WebMessage {
  id: string;
  role: MessageRole;
  content: string;
  created_at: string;
  mode: ChatMode | null;
  sources: string[];
  artifacts: ArtifactLink[];
  key_points?: KeyPoint[];
  citations?: CitationLink[];
  model_target?: ModelTarget;
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
}

export interface TurnResult {
  answer_markdown: string;
  sources: string[];
  artifacts: ArtifactLink[];
  key_points?: KeyPoint[];
  citations?: CitationLink[];
  mode: ChatMode;
  model_target?: ModelTarget;
  model_label?: string;
}

export interface ProgressEvent {
  type: string;
  timestamp: string;
  data: Record<string, unknown>;
}

export type ActivityPhase = "planning" | "researching" | "synthesizing" | "complete" | "error";

export interface ActivitySummary {
  phase: ActivityPhase;
  label: string;
  text: string;
  hiddenCount: number;
}
