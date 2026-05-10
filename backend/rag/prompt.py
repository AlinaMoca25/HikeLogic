SYSTEM_PROMPT = """You are HikeLogic, an assistant for Romanian hiking trails.
Answer the user's question using ONLY the trail context provided below.
Every factual claim about a trail must be supported by one or more source ids in square brackets, such as [1] or [2].
If the context does not contain the answer, say so plainly and do not invent details.
When the context contains safety information (closures, avalanche risk, exposure, difficulty), surface it explicitly.
Prefer concise, factual answers. Cite trail names when referring to specific trails.
Do not recommend a route as safe unless the provided context explicitly supports that."""


def format_context(hits) -> str:
    if not hits:
        return "(no trails matched the query)"

    blocks = []
    for i, h in enumerate(hits, 1):
        meta = h.metadata or {}
        name = meta.get("name") or "?"
        difficulty = meta.get("difficulty") or "?"
        marking = meta.get("marking") or "?"
        region = meta.get("region") or "?"
        osm_url = meta.get("osm_url")
        header = (
            f"[{i}] {name}  "
            f"(difficulty: {difficulty}, marking: {marking}, region: {region})"
        )
        if osm_url:
            header = f"{header} source: {osm_url}"
        blocks.append(f"{header}\n{(h.text or '').strip()}")
    return "\n\n---\n\n".join(blocks)


def build_user_message(query: str, hits) -> str:
    context = format_context(hits)
    return f"Trail context:\n{context}\n\nQuestion: {query}"
