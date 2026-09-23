import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

embed_func = OllamaEmbeddingFunction(
    url = "http://localhost:11434/api/embeddings",
    model_name = 'nomic-embed-text:latest'
)


client = chromadb.PersistentClient('./chroma_db')
memory_collection = client.get_or_create_collection(name="agent_memory", embedding_function=embed_func)



def save_memory(summary: str) -> str:
    """Embed a memory summary and store it in ChromaDB."""
    mem_id = f'mem_{memory_collection.count()}'
    memory = memory_collection.add(documents=[summary], ids=[mem_id])
    return f'Saved memory {summary}'


def retrieve_memories(query: str, n: int = 4) ->list[str]:
    """Semantic search — find the most relevant past memories for a query."""
    if memory_collection.count() == 0:
        return []
    memory_list = memory_collection.query(
        query_texts=[query],
        n_results= min(n, memory_collection.count())
    )
    docs = memory_list['documents'][0]
    metas = memory_list['metadatas'][0]

    return [f"{m.get('source', '?')}] {d}" for d,m in zip(docs, metas)]


