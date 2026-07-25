import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def create_worker_agent():
    """Create RAG Worker using stronger Groq model"""
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )
    return llm


def load_vector_store():
    """Load existing ChromaDB using absolute project path"""
    # Project root = src/agents/../../
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    vector_dir = os.path.join(base_dir, "data", "vectorstore")

    print(f"   Vector DB path: {vector_dir}")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=vector_dir,
        embedding_function=embeddings
    )
    return vector_store


def process_query(structured_message):
    """
    RAG Worker Agent:
    1. Receive structured message from Router
    2. Retrieve relevant chunks from ChromaDB
    3. Generate answer using retrieved context
    """

    query = structured_message.get("query", "")
    intent = structured_message.get("intent", "general")

    print(f"\n📚 RAG Worker Agent Processing...")
    print(f"   Intent received: {intent}")

    # Step 1: Retrieve from vector store
    vector_store = load_vector_store()
    retrieved_docs = vector_store.similarity_search(query, k=3)

    # Build context from retrieved chunks
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    sources = [doc.metadata.get("source", "unknown") for doc in retrieved_docs]

    print(f"   Retrieved {len(retrieved_docs)} relevant chunks.")

    # Step 2: Generate answer
    llm = create_worker_agent()

    answer_prompt = f"""You are an internship assistant helping university students.
Use ONLY the following context to answer the question.
If the context doesn't contain enough information, say so honestly.

Context:
{context}

Question: {query}

Provide a helpful, clear answer:"""

    response = llm.invoke(answer_prompt)

    # Create structured output
    worker_output = {
        "intent": intent,
        "query": query,
        "draft_answer": response.content,
        "sources": list(set(sources)),
        "chunks_used": len(retrieved_docs)
    }

    print(f"   ✅ Draft answer generated.")
    return worker_output


# Test
if __name__ == "__main__":
    test_message = {
        "intent": "cv_help",
        "query": "How do I write a good CV for an IT internship?",
        "needs_rag": True
    }

    result = process_query(test_message)
    print(f"\n📄 Draft Answer:\n{result['draft_answer'][:300]}...")
    print(f"\n📎 Sources: {result['sources']}")