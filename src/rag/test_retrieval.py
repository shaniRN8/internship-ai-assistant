from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def test_retrieval():
    vector_dir = "data/vectorstore"
    
    # Same embedding model used in ingestion
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Load existing vector store
    print("⏳ Loading vector store...")
    vector_store = Chroma(
        persist_directory=vector_dir,
        embedding_function=embeddings
    )
    
    # 5 test queries
    queries = [
        "How do I write a CV for an IT internship?",
        "What are common internship interview questions?",
        "How should I apply for internships?",
        "What should I include in a cover letter?",
        "What documents are needed for internship applications?"
    ]
    
    print("=" * 50)
    print("RETRIEVAL TEST — 5 Sample Queries")
    print("=" * 50)
    
    for i, query in enumerate(queries, 1):
        print(f"\n📌 Query {i}: {query}")
        print("-" * 40)
        
        # Retrieve top 3 chunks
        results = vector_store.similarity_search(query, k=3)
        
        if len(results) == 0:
            print("   ❌ No results found.")
        else:
            for j, doc in enumerate(results, 1):
                source = doc.metadata.get("source", "unknown")
                preview = doc.page_content[:150].replace("\n", " ")
                print(f"   Result {j}:")
                print(f"   Source: {source}")
                print(f"   Preview: {preview}...")
                print()

if __name__ == "__main__":
    test_retrieval()