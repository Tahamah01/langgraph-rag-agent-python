from langchain_core.tools import tool
from memory import save_memory, retrieve_memories
from rag import retrieve_docs


@tool
def memorize(summary: str) -> str:
    """
    Save an important fact to long-term memory.

    Before calling this tool, YOU must first condense the information
    into one concise sentence capturing only the essential fact.
    Do not pass raw conversation text — pass your own summary.

    When to use:
    - User shares their name, preferences, background, or goals
    - A key decision or conclusion was reached
    - User corrects something important
    """
    return save_memory(summary)

@tool
def recall(query: str) -> str:
    """
    Search long-term memory for facts relevant to the current query.
    Use when the user references something from a past session,
    or when you sense you might have stored relevant context before.
    """
    memories = retrieve_memories(query)
    if not memories:
        return f'No memories found for {query}'
    return 'Relevant memories:\n' + '\n'.join(f'- {m}' for m in memories)


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the document knowledge base for information.
    Use this before answering any factual question that might
    be covered by ingested reference documents.
    """
    results = retrieve_docs(query)
    if not results:
        return 'No relevant documents found!'
    return f'Relevant excerpts:\n\n' + '\n\n'.join(results)
