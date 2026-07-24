import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import PyPDFLoader

def load_documents():
    raw_dir = "data/raw"
    
    # Check folder exists
    if not os.path.exists(raw_dir):
        print("❌ data/raw folder not found!")
        return []
    
    all_documents = []
    
    # ---- Load .txt files (with UTF-8 encoding) ----
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
        if filename.endswith(".pdf"):
            filepath = os.path.join(raw_dir, filename)
            pdf_loader = PyPDFLoader(filepath)
            pdf_docs = pdf_loader.load()
            all_documents.extend(pdf_docs)
            pdf_count += 1
            print(f"   Loaded: {filename} ({len(pdf_docs)} pages)")
    
    if pdf_count == 0:
        print("   No PDF files found.")
    
    # ---- Final count ----
    print("=" * 40)
    if len(all_documents) == 0:
        print("⚠️ No documents found! Add files to data/raw/")
    else:
        print(f"🎉 Total loaded: {len(all_documents)} document(s)")
    
    return all_documents

# Run
if __name__ == "__main__":
    docs = load_documents()