import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def get_base_dir():
    """Return project root directory (works from any working directory)."""
    # src/rag/ingest.py -> src/rag -> src -> project root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_documents():
    """Load .txt and .pdf documents from data/raw."""
    base_dir = get_base_dir()
    raw_dir = os.path.join(base_dir, "data", "raw")

    print(f"📁 Raw data path: {raw_dir}")

    if not os.path.exists(raw_dir):
        print("❌ data/raw folder not found!")
        return []

    all_documents = []

    # ---- Load .txt files ----
    print("⏳ Loading .txt files...")
    txt_loader = DirectoryLoader(
        raw_dir,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    txt_docs = txt_loader.load()
    print(f"   Found {len(txt_docs)} text file(s).")
    all_documents.extend(txt_docs)

    # ---- Load .pdf files ----
    print("⏳ Loading .pdf files...")
    pdf_count = 0
    for filename in os.listdir(raw_dir):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(raw_dir, filename)
            try:
                pdf_loader = PyPDFLoader(filepath)
                pdf_docs = pdf_loader.load()
                all_documents.extend(pdf_docs)
                pdf_count += 1
                print(f"   Loaded: {filename} ({len(pdf_docs)} pages)")
            except Exception as e:
                print(f"   ⚠️ Failed to load {filename}: {e}")

    if pdf_count == 0:
        print("   No PDF files found.")

    print(f"\n📄 Total documents loaded: {len(all_documents)}")
    return all_documents


def chunk_documents(documents):
    """Split documents into smaller chunks."""
    print("\n⏳ Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents.")

    if len(chunks) > 0:
        print("\n📌 Sample chunk (first one):")
        print("-" * 40)
        print(chunks[0].page_content[:200])
        print("-" * 40)

    return chunks


def create_vector_store(chunks):
    """Create embeddings and persist ChromaDB."""
    base_dir = get_base_dir()
    vector_dir = os.path.join(base_dir, "data", "vectorstore")

    print(f"\n📁 Vector store path: {vector_dir}")
    print("⏳ Creating embeddings and saving to ChromaDB...")
    print("   (First run may take 1-2 minutes to download the model)")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=vector_dir
    )

    print(f"🎉 Vector store saved to {vector_dir}")
    print(f"   Total vectors stored: {len(chunks)}")
    return vector_store


def run_ingestion():
    """Full pipeline: load -> chunk -> embed -> store."""
    docs = load_documents()

    if len(docs) == 0:
        print("\n⚠️ No documents found. Add files to data/raw/")
        return None

    chunks = chunk_documents(docs)

    if len(chunks) == 0:
        print("\n⚠️ No chunks created.")
        return None

    vector_store = create_vector_store(chunks)
    print("\n✅ RAG Ingestion Pipeline Complete!")
    return vector_store


if __name__ == "__main__":
    run_ingestion()