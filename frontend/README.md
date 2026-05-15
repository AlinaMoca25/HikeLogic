# HikeLogic Demo UI

React chat interface for the HikeLogic RAG + SLM pipeline.

## Run locally

**Terminal 1 — API** (from project root):

```bash
pip install -r backend/requirements.txt
uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — Frontend**:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Modes

- **Răspuns complet** — hybrid retrieval + rerank + fine-tuned Qwen SLM
- **Doar căutare** — retrieval only (faster, no GPU generation)

Ensure `backend/.env` has valid Qdrant credentials and the collection is ingested.
