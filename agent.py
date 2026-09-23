from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from memory import save_memory, retrieve_memories
from tools import search_knowledge_base, memorize, retrieve_docs



def build_systemPrompt(user_query: str) -> str:
    """
    Called once at agent startup with the first user message.
    Pre-loads semantically relevant memories into the system prompt
    so the agent doesn't need to call 'recall' for basic context.
    """
    memories = retrieve_memories(user_query, n=5)
    docs_content = retrieve_docs(user_query, n=5)
    prompt = """You are a helpful assistant with long-term memory and a document knowledge base.

    Tools available:
    - memorize: YOU summarize the fact first into one sentence, then call this to persist it
    - recall: semantic search over past memories when you need older context
    - search_knowledge_base: search ingested reference documents before answering factual questions

    Always use search_knowledge_base before answering factual questions.
    Proactively memorize important user facts — but always summarize them yourself first."""

    if docs_content:
        block = '\n'.join(f' - {d}' for d in docs_content)
        prompt += f'\n\n Relevant info for the user prompt from the documents saved:\n{block}'
    if memories:
        block = '\n'.join(f' - {m}' for m in memories)
        prompt += f'\n\nWhat you already know about this user: \n{block}'
    return prompt



def build_agent(first_message: str):
    """
    Builds the agent once per session.
    first_message seeds the memory injection — relevant past memories
    are retrieved and embedded in the system prompt from the start.
    """
    llm = ChatOllama(model='gemma:latest')
    tools = [search_knowledge_base, memorize, retrieve_docs]
    system_prompt = build_systemPrompt(first_message)
    #with SqliteSaver.from_conn_string('./checkpoints.db') as checkpointer:
    return create_react_agent(
            llm,
            tools,
            prompt=system_prompt,
        )
