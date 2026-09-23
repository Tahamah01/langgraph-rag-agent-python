import os
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from dotenv import load_dotenv


load_dotenv()

LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/embeddings")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

embed_fn = OllamaEmbeddingFunction(url=OLLAMA_URL, model_name=EMBED_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)
