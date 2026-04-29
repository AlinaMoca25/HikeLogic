# HikeLogic


## Setup
1. Install dependencies
pip install -r backend/requirements.txt

2. Create backend/.env (credentials from team chat)
QDRANT_URL=...
QDRANT_API_KEY=...
COLLECTION_NAME=hikelogic_docs

3. Transform JSON data in .md files + yaml header
cd chunking_setup
python create_hiking_docs.py

4. Setup Qdrant
cd backend
python setup_qdrant.py

5. Embed and upsert files in the vector DB
cd backend
python ingest_all.py

6. Sanity check the retriever (from project root)
python -m backend.test_retrieval


---
**Kaleidoscope Team**
- Moldovan Anita
- Motofelea Viorelia
- Moca Alina
- Jitareanu Eduard-David
- Oana Rares

---

## Dataset Selection

### RAG Dataset

**OpenStreetMap (OSM)**
- Open-source
- https://www.openstreetmap.org
- https://wiki.openstreetmap.org/wiki/Overpass_API
- *Overpass Turbo tool* - run a simple query to export every hiking trail, spring, and cabin in Romania as a JSON file
- Database of facts (coordinates, trail names, difficulty tags)

### Fine-tuning Dataset

**Hugging Face Datasets**
- https://huggingface.co/datasets/JasleenSingh91/travel-QA
- https://huggingface.co/datasets/soniawmeyer/travel-conversations-finetuning

**Synthetic Data**
- Transform OSM Data using an LLM (prompt it to write blog posts based on the technical data)

**Scrape local website**
- Extract trail descriptions, difficulty levels, estimated times, and "points of interest" (cabins, springs, peaks)

---

## Chosen Chunking Strategy

### Entity-based chunking for OSM and route facts

OpenStreetMap and Overpass exports are not long narratives; they are collections of factual records. Because of that, the system should not cut them into generic token windows.

- Each **chunk** should represent **one meaningful object**: a trail, cabin, spring, peak access point, or route segment
- This keeps the numerical facts **together**, such as coordinates, route name, distance, elevation, and difficulty tags
- It also makes retrieval **easier** to explain: the user asks about a trail => the retriever returns the trail record
- One of the **highest-value, lowest-complexity** choices for HikeLogic because the RAG data is already naturally organized as **real-world objects**

---

## Database Selection

### Unified Hybrid Engine

Qdrant supports Multi-Vector Collections, allowing us to store both BGE-M3 dense embeddings and sparse (BM25-style) vectors in the same record.

- **Benefit:** No need for a separate database for trail names. Qdrant performs the keyword and semantic search simultaneously, reducing latency and infrastructure complexity.

### Native RRF (Reciprocal Rank Fusion)

The fusion of "Easy walk" (semantic) and "Cabana Bâlea" (keyword) results is handled server-side within Qdrant.

- **Benefit:** Eliminates manual "glue code" in Python. We send one query; Qdrant returns a single, mathematically re-ranked list of candidates ready for the cross-encoder.

---

## Model Choices

### 01. Chosen Model — Mistral-7B-Instruct

- Open-source (Hugging Face)
- ~7B parameters (efficient)
- Strong reasoning for a small model
- Supports domain-specific fine-tuning

### 02. Why This Model

- Optimized for RAG-based architecture
- Low cost (no API required)
- Aligns with SLM design goal
- Adaptable to hiking & safety data
- Can be aligned for safety-critical decisions

**Backup Models:** Llama 3 (8B) - balanced performance

https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

---

## RAG Strategies

### 1. Hybrid Search

Dense retrieval (BGE-M3) matches semantic meaning — "easy walk near a lake" finds "gentle lakeside path." Sparse retrieval (BGE-M3 lexical weights, BM25-style) catches exact names — "Retezat", "Cabana Bâlea." RRF fuses both. Neither alone handles HikeLogic's mix of vague queries and specific trail names.

### 2. Re-ranking

A cross-encoder (BGE-reranker-v2-m3) re-scores the top-20 retrieved chunks and keeps only the top-5. Ensures safety-critical content — "trail closed, avalanche risk" — outranks generic descriptions regardless of embedding similarity. Improves RAGAS faithfulness directly.

**Pipeline:**
1. User query
2. Hybrid search — Dense (BGE-M3) + sparse (BGE-M3 lexical weights) fused with RRF — top-20 chunks
3. Re-ranking — Cross-encoder re-scores top-20, keeps top-5 for generation
4. LLM generation — Fine-tuned SLM + safety prompt

---

## Public API

The retriever is exposed as a single function. From the project root:

from backend.rag.search import search
hits = search("Cabana Bâlea")

`search()` returns `list[Hit]`. Each `Hit` has `.text`, `.score`, and `.metadata` (name, region, difficulty, marking).

---

## Status

- [x] Dataset + chunking
- [x] Qdrant collection (named dense+sparse)
- [x] Ingest (1589 docs)
- [x] Retriever + reranker
- [ ] Generator
- [ ] Fine-tuning
- [ ] RLHF + demo

---

## Timeline Proposal

| Phase | Tasks |
|---|---|
| **Weeks 5–6** | Choose architecture |
| **Weeks 7–8** | Prepare dataset; Integrate the model with a RAG pipeline prototype |
| **Weeks 9–10** | Fine-tune LLM; Tool integration |
| **Weeks 11–12** | Refinement; RLHF; Demo |
