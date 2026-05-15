"""HikeLogic REST API for the demo frontend."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.api.schemas import AskRequest, AskResponse, HealthResponse, SourceItem
from backend.rag.config import MIN_RERANK_SCORE_FOR_GENERATION

ABSTENTION_MARKER = "nu am găsit surse relevante"

app = FastAPI(
    title="HikeLogic API",
    description="Romanian hiking assistant — RAG + fine-tuned SLM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _hits_to_sources(hits) -> list[SourceItem]:
    return [
        SourceItem(
            index=i,
            text=hit.text,
            score=hit.score,
            metadata=hit.metadata or {},
        )
        for i, hit in enumerate(hits, 1)
    ]


def _is_abstention(text: str, hits) -> bool:
    if ABSTENTION_MARKER in text.casefold():
        return True
    if not hits:
        return True
    return max(hit.score for hit in hits) < MIN_RERANK_SCORE_FOR_GENERATION


@app.get("/")
def root() -> dict:
    return {
        "service": "hikelogic",
        "message": "API is running. Use the frontend at http://localhost:5173",
        "health": "/api/health",
        "ask": "POST /api/ask",
        "docs": "/docs",
    }


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="hikelogic")


@app.post("/api/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        if body.mode == "retrieve":
            from backend.rag.search import search

            hits = search(query)
            return AskResponse(
                query=query,
                answer=(
                    "Mod doar căutare: sursele relevante sunt afișate în panoul "
                    "din dreapta. Activează „Răspuns complet” pentru generare SLM."
                ),
                sources=_hits_to_sources(hits),
                abstained=_is_abstention("", hits),
                mode="retrieve",
            )

        from backend.rag.pipeline import answer

        result = answer(query)
        return AskResponse(
            query=result.query,
            answer=result.text,
            sources=_hits_to_sources(result.sources),
            abstained=_is_abstention(result.text, result.sources),
            mode="full",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline error: {exc}",
        ) from exc
