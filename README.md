# HikeLogic

A hiking assistant for Romanian trails. Fine-tuned 7B SLM + hybrid RAG over OpenStreetMap data + a small agent layer for live weather and distance.

**Team Kaleidoscope:** Moldovan Anita · Motofelea Viorelia · Moca Alina · Jităreanu Eduard-David · Oană Rareș

## Pipeline

```
query → hybrid search (BGE-M3 dense + sparse, RRF) → top-20
      → cross-encoder rerank (BGE-reranker-v2-m3) → top-5
      → entity-grounding guard (abstain on missed entities)
      → agent: weather / distance tools (when intent matches)
      → fine-tuned Qwen2.5-7B with citation discipline
```

## Components

| Layer | Choice |
|---|---|
| Data | OpenStreetMap via Overpass API — trails + named POIs (cabins, peaks, springs, viewpoints, saddles, caves, rescue bases, via ferrata) |
| Chunking | One entity per chunk — each route relation and named POI becomes its own markdown doc with YAML frontmatter |
| Vector DB | Qdrant Cloud, named multi-vector collection (dense 1024-d cosine + sparse), server-side RRF |
| Embeddings | `BAAI/bge-m3` (dense + sparse in one pass), `BAAI/bge-reranker-v2-m3` for rerank |
| Generator | `edededi/hikelogic-qwen2.5-7b` (4-bit local via transformers, or HF Inference API) |
| Agent tools | Open-Meteo for weather, haversine for distance |

## Setup

```bash
pip install -r backend/requirements.txt
```

Create `backend/.env`:

```
QDRANT_URL=<cluster endpoint>
QDRANT_API_KEY=<key>
COLLECTION_NAME=hike_logic_romania
HF_TOKEN=<hf read token>
GENERATION_MODEL=edededi/hikelogic-qwen2.5-7b
GENERATION_BACKEND=local
GENERATION_LOAD_4BIT=1
```

Build the index:

```bash
cd chunking_setup && python create_hiking_docs.py    # ~3 min, ~18K chunks
cd ../backend && python setup_qdrant.py              # drop + recreate collection
python ingest_all.py                                 # batched embed + upsert, ~2 min on GPU
```

## Usage

```python
from backend.rag.pipeline import answer        # plain RAG
from backend.rag.agent import answer_with_agent  # RAG + weather/distance tools

r = answer_with_agent("Cât de înalt este Vârful Negoiu?")
print(r.text)                                  # cited answer
for h in r.sources:
    print(h.metadata["name"], h.score)
```

End-to-end smoke test (3 RAG + 1 agent query):

```bash
python -m backend.test_e2e
```

For a one-shot Colab run (sets everything up and drops into a chat REPL), open `chat.ipynb`.

## Repository layout

```
HikeLogic/
├── chunking_setup/
│   ├── overpass_query              # Overpass-Turbo query (route relations + POI tags)
│   ├── romania_hiking.json         # raw OSM extract
│   ├── create_hiking_docs.py       # JSON → markdown chunks with YAML frontmatter
│   └── hiking_docs/                # generated chunks (gitignored)
├── backend/
│   ├── setup_qdrant.py             # create the collection
│   ├── ingest_all.py               # batched embed + upsert (uses tqdm)
│   ├── test_e2e.py                 # end-to-end smoke test
│   ├── evaluate_ragas.py           # RAGAS faithfulness / answer_relevancy / context_precision
│   ├── eval/ragas_queries.json     # eval set
│   └── rag/
│       ├── config.py               # env vars, model names, top-k
│       ├── qdrant_client.py        # client + create_collection
│       ├── embeddings.py           # BGE-M3 wrapper (dense + sparse, batched)
│       ├── embedder.py             # batched upsert
│       ├── retriever.py            # hybrid search via Qdrant Query API + RRF
│       ├── reranker.py             # cross-encoder rerank
│       ├── search.py               # search(query) → list[Hit]
│       ├── prompt.py               # system prompt + context formatting
│       ├── generator.py            # Qwen2.5-7B (local 4-bit or hf_api)
│       ├── pipeline.py             # answer(query) + entity-grounding guard
│       ├── tools.py                # weather (Open-Meteo) + haversine
│       └── agent.py                # answer_with_agent(query) — RAG + tools
└── finetune/
    ├── build_dpo_candidates.py     # sample answer pairs for human ranking → DPO
    ├── dpo_prompts.txt
    └── HikeLogic_dpo.ipynb         # DPO training notebook
```

## Hallucination mitigation

Three layers, each catches a different failure mode:

1. **System prompt** requires `[N]` citations and explicit abstention when context doesn't support an answer.
2. **POI chunk design** repeats the name through the body (so short POI docs win retrieval against long trail docs) and labels the "nearby trails" section with `NU descriu X` (so the model doesn't conflate a POI with adjacent ones).
3. **Entity-grounding guard** (`pipeline.check_entity_grounding`) extracts capitalized multi-word phrases from the query and abstains pre-generation if no top-N hit's name contains them. Catches near-miss conflations (e.g. `Cabana Bâlea Lac` → `Curmătura Bâlei`).

## Status

| Component | State |
|---|---|
| OSM extract + chunking | ✅ trails + named POIs (~18K chunks) |
| Qdrant hybrid + rerank | ✅ |
| Fine-tuned generator | ✅ Qwen2.5-7B |
| Agent tools | ✅ weather + distance |
| Hallucination guard | ✅ system prompt + POI design + entity grounding |
| RAGAS evaluation | wired (`evaluate_ragas.py`), pending judge-LLM key |
| DPO / RLHF | pair generation wired; needs human labels + training run |

## License / data

- OSM data © OpenStreetMap contributors, [ODbL](https://www.openstreetmap.org/copyright)
- Code: not yet licensed
