import uuid
from datetime import datetime, timezone
from config import client, embed_fn

memory_collection = client.get_or_create_collection(
    name="agent_memory", embedding_function=embed_fn     # type: ignore
)


def save_memory(summary: str) -> str:
    """Embed a one-sentence memory and store it in ChromaDB."""
    summary = summary.strip()
    if not summary:
        return "Nothing to save."
    memory_collection.add(
        documents=[summary],
        ids=[f"mem_{uuid.uuid4().hex}"],
        metadatas=[{"created_at": datetime.now(timezone.utc).isoformat()}],
    )
    return f"Saved memory: {summary}"


def retrieve_memories(query: str, n: int = 4) -> list[str]:
    """Semantic search: the most relevant past memories for a query."""
    count = memory_collection.count()
    if count == 0:
        return []
    result = memory_collection.query(query_texts=[query], n_results=min(n, count))
    return result["documents"][0]                  # type: ignore
