from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from config import LLM_MODEL
from memory import retrieve_memories
from rag import retrieve_docs
from tools import memorize, recall, search_knowledge_base

BASE_PROMPT = """You are a helpful assistant with long-term memory and a document knowledge base.

Tools available:
- memorize: YOU summarize the fact first into one sentence, then call this to persist it
- recall: semantic search over past memories when you need older context
- search_knowledge_base: search ingested reference documents before answering factual questions

Always use search_knowledge_base before answering factual questions.
Proactively memorize important user facts, but always summarize them yourself first."""


def build_system_prompt(user_query: str) -> str:
    """Pre-load memories and document excerpts relevant to the first message."""
    memories = retrieve_memories(user_query, n=5)
    docs = retrieve_docs(user_query, n=5)

    prompt = BASE_PROMPT
    if docs:
        prompt += "\n\nRelevant excerpts from the ingested documents:\n"
        prompt += "\n".join(f"- {d}" for d in docs)
    if memories:
        prompt += "\n\nWhat you already know about this user:\n"
        prompt += "\n".join(f"- {m}" for m in memories)
    return prompt


def build_agent(first_message: str):
    """Build the agent once per session. The checkpointer keeps the conversation
    history between turns (keyed by thread_id)."""
    llm = ChatOllama(model=LLM_MODEL)
    tools = [search_knowledge_base, memorize, recall]
    return create_agent(
        llm,
        tools,
        system_prompt=build_system_prompt(first_message),
        checkpointer=InMemorySaver(),
    )
