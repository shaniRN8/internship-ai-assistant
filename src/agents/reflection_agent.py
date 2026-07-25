import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def create_reflection_agent():
    """Create Reflection Agent using Groq"""
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )
    return llm

def reflect_and_improve(worker_output):
    """
    Reflection Agent:
    1. Receive draft answer from Worker
    2. Check quality, clarity, completeness
    3. Return improved final answer
    """
    
    draft = worker_output.get("draft_answer", "")
    query = worker_output.get("query", "")
    intent = worker_output.get("intent", "")
    
    print(f"\n🪞 Reflection Agent Reviewing...")
    
    llm = create_reflection_agent()
    
    reflection_prompt = f"""You are a quality reviewer for an internship assistant.

Original Question: {query}
Category: {intent}
Draft Answer: {draft}

Review the draft answer and improve it:
1. Is it clear and easy to understand?
2. Is it complete? Any missing important points?
3. Is the tone professional but friendly?
4. Remove any irrelevant information.

Provide the improved final answer:"""
    
    response = llm.invoke(reflection_prompt)
    
    # Create final output
    final_output = {
        "intent": intent,
        "query": query,
        "draft_answer": draft,
        "final_answer": response.content,
        "sources": worker_output.get("sources", []),
        "status": "reviewed"
    }
    
    print(f"   ✅ Answer reviewed and improved.")
    return final_output

# Test
if __name__ == "__main__":
    test_worker_output = {
        "intent": "cv_help",
        "query": "How do I write a good CV?",
        "draft_answer": "A CV should include your name, education, skills, and experience. Keep it to one page.",
        "sources": ["cv_guide.pdf"],
        "chunks_used": 3
    }
    
    result = reflect_and_improve(test_worker_output)
    print(f"\n✨ Final Answer:\n{result['final_answer']}")