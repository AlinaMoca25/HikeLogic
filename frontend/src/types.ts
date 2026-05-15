export type AskMode = "full" | "retrieve";

export interface SourceItem {
  index: number;
  text: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface AskResponse {
  query: string;
  answer: string;
  sources: SourceItem[];
  abstained: boolean;
  mode: AskMode;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  abstained?: boolean;
  sources?: SourceItem[];
  mode?: AskMode;
}
