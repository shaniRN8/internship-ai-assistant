import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def create_router_agent():
    """Create Router Agent using fast Groq model"""
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    return llm

def route_query(user_query):
    """
    Classify user intent and create structured message.
    
    Intents:
    - cv_help: CV writing, resume tips
    - interview_help: Interview questions, preparation
    - internship_info: How to apply, internship process
    - cover_letter: Cover letter writing tips
    - general: Other questions
    """
    
    llm = create_router_agent()
    
    routing_prompt = f"""You are a routing agent for an internship assistant.
Classify the following user question into ONE of these categories:
- cv_help
- interview_help
- internship_info
- cover_letter
- general

Respond with ONLY a JSON object, nothing else:
{{"intent": "<category>", "query": "<original question>", "needs_rag": true}}

User question: {user_query}"""
    
    response = llm.invoke(routing_prompt)
    
    # Parse response
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        result = {
            "intent": "general",
            "query": user_query,
            "needs_rag": True
        }
    
    print(f"🔀 Router Agent Decision:")
    print(f"   Intent: {result.get('intent')}")
    print(f"   Needs RAG: {result.get('needs_rag')}")
    
    return result

# Test
if __name__ == "__main__":
    test_queries = [
        "How do I write a good CV?",
        "What questions are asked in interviews?",
        "How to apply for internships?",
        "Help me write a cover letter",
        "What is machine learning?"
    ]
    
    for q in test_queries:
        print(f"\n📝 Query: {q}")
        result = route_query(q)
        print()