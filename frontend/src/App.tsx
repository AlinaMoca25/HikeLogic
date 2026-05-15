import { useCallback, useEffect, useRef, useState } from "react";
import { askQuestion, checkHealth } from "./api";
import SourcePanel from "./components/SourcePanel";
import type { AskMode, ChatMessage, SourceItem } from "./types";
import "./App.css";

const EXAMPLE_QUERIES = [
  "Cum ajung la Cabana Bâlea?",
  "Recomandă-mi un traseu ușor lângă un lac.",
  "Există Salvamont la Cheile Turzii?",
  "Care este cel mai dificil traseu din Făgăraș?",
  "Unde găsesc apă la Fântâna Bâlea?",
];

function newId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sources, setSources] = useState<SourceItem[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<AskMode>("full");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    checkHealth().then(setApiOnline);
    const interval = setInterval(() => {
      checkHealth().then(setApiOnline);
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const submit = useCallback(
    async (query: string) => {
      const trimmed = query.trim();
      if (!trimmed || loading) return;

      setError(null);
      setLoading(true);
      setInput("");

      const userMsg: ChatMessage = {
        id: newId(),
        role: "user",
        content: trimmed,
      };
      setMessages((prev) => [...prev, userMsg]);

      try {
        const res = await askQuestion(trimmed, mode);
        setSources(res.sources);

        const assistantMsg: ChatMessage = {
          id: newId(),
          role: "assistant",
          content: res.answer,
          abstained: res.abstained,
          sources: res.sources,
          mode: res.mode,
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Eroare necunoscută.";
        setError(message);
        setMessages((prev) => [
          ...prev,
          {
            id: newId(),
            role: "assistant",
            content:
              "Nu am putut procesa întrebarea. Verifică dacă serverul API rulează " +
              "(uvicorn) și conexiunea la Qdrant.",
            abstained: true,
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [loading, mode],
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    void submit(input);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void submit(input);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="brand-icon" aria-hidden>
            ⛰
          </div>
          <div>
            <h1>HikeLogic</h1>
            <p>Asistent trasee montane · România · RAG + SLM</p>
          </div>
        </div>
        <div
          className="status-pill"
          title={
            apiOnline === null
              ? "Se verifică conexiunea…"
              : apiOnline
                ? "API conectat"
                : "API indisponibil — pornește uvicorn"
          }
        >
          <span
            className={`status-dot ${apiOnline === true ? "online" : apiOnline === false ? "offline" : ""}`}
          />
          {apiOnline === null
            ? "Conectare…"
            : apiOnline
              ? "API online"
              : "API offline"}
        </div>
      </header>

      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}

      <main className="main">
        <section className="chat-column" aria-label="Conversație">
          <div className="messages">
            {messages.length === 0 && !loading && (
              <div className="welcome">
                <h2>Unde vrei să mergi la munte?</h2>
                <p>
                  Întreabă despre trasee, cabane, izvoare, dificultate sau
                  puncte Salvamont. Răspunsurile folosesc doar date din
                  OpenStreetMap recuperate prin căutare hibridă.
                </p>
                <div className="examples">
                  {EXAMPLE_QUERIES.map((q) => (
                    <button
                      key={q}
                      type="button"
                      className="example-chip"
                      onClick={() => void submit(q)}
                      disabled={loading}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`message ${msg.role}${msg.abstained ? " abstain" : ""}`}
              >
                <div className="bubble">{msg.content}</div>
                {msg.role === "assistant" && (
                  <span className="message-meta">
                    {msg.abstained
                      ? "Abținere — context insuficient"
                      : msg.mode === "retrieve"
                        ? "Doar recuperare (fără SLM)"
                        : "Răspuns generat · surse în panoul din dreapta"}
                  </span>
                )}
              </div>
            ))}

            {loading && (
              <div className="loading-row" aria-live="polite">
                <span className="spinner" />
                {mode === "full"
                  ? "Se caută trasee și se generează răspunsul (poate dura la prima rulare)…"
                  : "Se caută trasee relevante…"}
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form className="composer" onSubmit={handleSubmit}>
            <div className="mode-toggle" role="group" aria-label="Mod răspuns">
              <button
                type="button"
                className={`mode-btn ${mode === "full" ? "active" : ""}`}
                onClick={() => setMode("full")}
                disabled={loading}
              >
                Răspuns complet (RAG + SLM)
              </button>
              <button
                type="button"
                className={`mode-btn ${mode === "retrieve" ? "active" : ""}`}
                onClick={() => setMode("retrieve")}
                disabled={loading}
              >
                Doar căutare (mai rapid)
              </button>
            </div>
            <div className="composer-row">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ex: Cum ajung la Cabana Bâlea?"
                rows={2}
                disabled={loading}
                aria-label="Întrebarea ta"
              />
              <button type="submit" className="send-btn" disabled={loading || !input.trim()}>
                Trimite
              </button>
            </div>
            <p className="composer-hint">
              Enter trimite · Shift+Enter linie nouă · Răspunsurile pot include citări [1], [2]
            </p>
          </form>
        </section>

        <SourcePanel sources={sources} />
      </main>
    </div>
  );
}
