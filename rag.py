import sys
from pathlib import Path
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import client, embed_fn

docs_collection = client.get_or_create_collection(
    name="documents", embedding_function=embed_fn          # type: ignore
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""],
)

LOADERS = {
    ".pdf": lambda p: PyPDFLoader(p),
    ".docx": lambda p: Docx2txtLoader(p),
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
}


def ingest_doc(filepath: Path) -> str:
    """Load a document, split it into chunks, embed and store them in ChromaDB."""
    make_loader = LOADERS.get(filepath.suffix.lower())
    if make_loader is None:
        return f"Skipped unsupported format: {filepath.name}"
    try:
        pages = make_loader(str(filepath)).load()
        chunks = splitter.split_documents(pages)
        if not chunks:
            return f"No text found in {filepath.name}"

        docs_collection.delete(where={"source": filepath.name})
        docs_collection.upsert(
            documents=[c.page_content for c in chunks],
            metadatas=[{"source": filepath.name, "chunk": i} for i in range(len(chunks))],
            ids=[f"{filepath.name}_{i}" for i in range(len(chunks))],
        )
        return f"Ingested {len(chunks)} chunks from {filepath.name}"
    except Exception as e:
        return f"Failed to process {filepath.name}: {e}"


def retrieve_docs(query: str, n: int = 3) -> list[str]:
    """Semantic search over the document knowledge base."""
    count = docs_collection.count()
    if count == 0:
        return []
    res = docs_collection.query(query_texts=[query], n_results=min(n, count))
    return [
        f"[{(meta or {}).get('source', '?')}] {doc}"
        for doc, meta in zip(res["documents"][0], res["metadatas"][0])       # type: ignore
    ]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag.py <file_or_folder>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Path not found: {target}")
        sys.exit(1)

    files = sorted(f for f in target.iterdir() if f.is_file()) if target.is_dir() else [target]
    if not files:
        print("No files found.")
        sys.exit(0)
    for f in files:
        print(ingest_doc(f))
