export type ChatMode = "fast" | "expert";
export type ArtifactKind = "image" | "slides" | "file";
export type MessageRole = "user" | "assistant";

export interface ArtifactLink {
  kind: ArtifactKind;
  label: string;
  path: string;
  url: string;
}

export interface WebMessage {
  id: string;
  role: MessageRole;
  content: string;
  created_at: string;
  mode: ChatMode | null;
  sources: string[];
  artifacts: ArtifactLink[];
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

export interface TurnResult {
  answer_markdown: string;
  sources: string[];
  artifacts: ArtifactLink[];
  mode: ChatMode;
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
