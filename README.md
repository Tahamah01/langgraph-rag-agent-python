# Local RAG Agent with Long-Term Memory

A fully local conversational agent built with **LangGraph**, **Ollama** and **ChromaDB**. It answers questions from your own documents (RAG) and remembers facts about the user across sessions.

## Features
- **Agent** (LangChain) with three tools: `search_knowledge_base`, `memorize`, `recall`
- **Document RAG:** ingest PDF / DOCX / TXT, chunk (500 chars, 50 overlap), embed with `nomic-embed-text`, store in ChromaDB
- **Long-term memory:** the agent summarizes facts into one sentence and stores them as embeddings; relevant memories are recalled semantically
- **Conversation state** kept between turns with a LangGraph checkpointer

## Architecture
```
main.py (CLI loop) -> agent.py (LangGraph ReAct agent + Ollama LLM)
                         |-- tools.py -> memory.py (agent_memory collection)
                         |            -> rag.py    (documents collection)
                         config.py (models, ChromaDB client)
```
At session start, memories and document excerpts relevant to the first message are injected into the system prompt.

## Setup
```bash
ollama pull llama3.1:8b          # any Ollama model with tool-calling support
ollama pull nomic-embed-text
pip install -r requirements.txt

python rag.py ./your_docs/       # ingest documents (re-running replaces old chunks)
python main.py                   # chat, type 'quit' to exit
```
Settings can be changed with env vars: `LLM_MODEL`, `EMBED_MODEL`, `OLLAMA_URL`, `CHROMA_PATH`.

## Transcript
You: Hi, my name is Taha
Agent loading...

Agent: Hi Taha! It's nice to meet you. How can I help you today?

You: 
Goodbye...

-- a new session --
You: what's my name?
Agent loading...

Agent: Your name is Taha.

You: what grade did the report card say i got on my AI class?
Agent loading...

Agent: According to the report card, your final grade for the Introduction à l'Intelligence Artificielle module (Module 33, ######### ###### Taha) was **18,48**, and you were validated.

## Known limitations
- Memories and documents are injected into the system prompt only from the **first message** of a session; later turns rely on the `recall` / `search_knowledge_base` tools.
- Conversation history is in memory only (lost on exit); long-term memory persists in `chroma_db/`.
- Memories are never updated or deduplicated. No evaluation of retrieval quality yet.
