import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import sys
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

embed_function = OllamaEmbeddingFunction(
    url = "http://localhost:11434/api/embeddings",
    model_name= 'nomic-embed-text:latest'
)

client = chromadb.PersistentClient(path ="./chroma_db")

docs_collection = client.get_or_create_collection(name ='documents',
                                                  embedding_function = embed_function)



splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
    separators=["\n\n", "\n", ".", " ", ""]
)

def ingest_doc(filepath: Path) -> str:
    """ takes the path of a document and converts it to embeddings and stores it in chromadb """
    ext = filepath.suffix.lower()

    try:
        if ext == '.pdf':
            loader = PyPDFLoader(str(filepath))
        elif ext == '.docx':
            loader = Docx2txtLoader(str(filepath))
        elif ext == '.txt':
            loader = TextLoader(str(filepath))
        else:
            print(f'Skipping unsupported format: {ext}')
            return

        text = loader.load()
        chunks = splitter.split_documents(text)

        ids = [f"{filepath.stem}_{i}" for i in range(len(chunks))]
        metadata = [{'Source' : filepath.name, "chunk" : i} for i in range(len(chunks))]

        docs_collection.add(documents=[c.page_content for c in chunks], metadatas= metadata, ids=ids)
        return f'Ingested {len(chunks)} chunks from {filepath.name}'
    except Exception as e:
        print(f"Failed to process {filepath.name}: {e}")



def retrieve_docs(query: str, n: int = 3) -> list[str]:
    """Semantic search over the document knowledge base."""
    if docs_collection.count() == 0:
        return []
    mem = docs_collection.query(
        query_texts=[query],
        n_results= min(n, docs_collection.count())
    )

    docs = mem['documents'][0]
    meta = mem['metadatas'][0]

    return [f"[{m.get('source', '?')}] {d}" for d, m in zip(docs, meta)]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file_or_folder>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        files = [f for f in target.iterdir() if f.is_file()]
        if not files:
            print('No files were found!')
        elif files:
            for f in files:
                ingest_doc(f)
        else:
            print(f'Path not found: {target}')
            sys.exit(1)
    elif target.is_file():
        ingest_doc(target)
        print(f"Successfully ingested {target}!")
    else:
        print('Something went wrong...')