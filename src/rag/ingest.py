import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load .env
load_dotenv()

def load_documents():
    raw_dir = "data/raw"
    
    if not os.path.exists(raw_dir):
        print("❌ data/raw folder not found!")
        return []
    
    all_documents = []
    
    # Load .txt files
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
    
    # Load .pdf files
    print("⏳ Loading .pdf files...")
    pdf_count = 0
    for filename in os.listdir(raw_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(raw_dir, filename)
            pdf_loader = PyPDFLoader(filepath)
            pdf_docs = pdf_loader.load()
            all_documents.extend(pdf_docs)
            pdf_count += 1
            print(f"   Loaded: {filename} ({len(pdf_docs)} pages)")
    
    if pdf_count == 0:
        print("   No PDF files found.")
    
    print(f"\n📄 Total documents loaded: {len(all_documents)}")
    return all_documents

def chunk_documents(documents):
    print("\n⏳ Splitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents.")
    return chunks

def create_vector_store(chunks):
    vector_dir = "data/vectorstore"
    
    print("\n⏳ Creating embeddings and saving to ChromaDB...")
    print("   (First time may take 1-2 minutes to download model)")
    
    # FREE local embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create and save vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=vector_dir
    )
    
    print(f"🎉 Vector store saved to {vector_dir}/")
    print(f"   Total vectors stored: {len(chunks)}")
    return vector_store

# Run full pipeline
if __name__ == "__main__":
    # Step 1: Load
    docs = load_documents()
    
    if len(docs) > 0:
        # Step 2: Chunk
        chunks = chunk_documents(docs)
        
        # Step 3: Embed + Save
        if len(chunks) > 0:
            vector_store = create_vector_store(chunks)
            print("\n✅ RAG Ingestion Pipeline Complete!")