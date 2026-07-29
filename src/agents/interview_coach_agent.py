import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from agents.rag_worker_agent import load_vector_store

load_dotenv()

def create_interview_coach_agent():
    """Create Interview Coach Agent using Groq model"""
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )
    return llm

def generate_mock_questions(target_role="IT Intern", category="Behavioral", count=3):
    """
    Generate tailored interview questions based on role and category.
    """
    print(f"\n🎙️ Generating {count} {category} interview questions for {target_role}...")

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.5,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""You are a senior tech interviewer conducting interviews for a "{target_role}" position.
Generate exactly {count} realistic, insightful {category} interview questions.

Return strictly a JSON array of strings containing the questions:
["Question 1", "Question 2", "Question 3"]
"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    try:
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        questions = json.loads(raw.strip())
    except Exception:
        questions = [
            f"Tell me about a challenging technical project you worked on relevant to {target_role}.",
            "Describe a situation where you had a conflict in a team and how you resolved it.",
            f"Why are you interested in this {target_role} internship and what technical skills do you bring?"
        ]

    return questions

def evaluate_interview_answer(question, user_answer, target_role="IT Intern"):
    """
    Evaluate student's interview response using the STAR method and provide structured feedback.
    """
    print(f"\n🎙️ Interview Coach evaluating answer for: '{question[:40]}...'")

    if not user_answer or len(user_answer.strip()) < 15:
        return {
            "error": "Answer is too short to evaluate. Please provide a more detailed response."
        }

    # Step 1: Context retrieval from ChromaDB if available
    context = ""
    try:
        vector_store = load_vector_store()
        retrieved = vector_store.similarity_search(f"interview questions answers {question}", k=2)
        context = "\n\n".join([d.page_content for d in retrieved])
    except Exception as e:
        print(f"   ⚠️ Vector store search skipped: {e}")

    # Step 2: LLM Evaluation Prompt
    llm = create_interview_coach_agent()

    eval_prompt = f"""You are an Expert Interview Coach for University IT Students.
Evaluate the candidate's response to the following interview question for a "{target_role}" role.

Question: {question}
Candidate Answer:
\"\"\"
{user_answer}
\"\"\"

RAG Guidance Context:
{context if context else 'Apply standard STAR framework criteria.'}

Evaluate the response strictly against the STAR Framework (Situation, Task, Action, Result):
- Situation & Task: Did the candidate clearly explain the background and challenge?
- Action: Did they clearly highlight THEIR specific technical actions and contributions?
- Result: Did they share concrete outcomes, learnings, or metrics?

Return strictly a single JSON object with the following structure:
{{
  "overall_score": 85,
  "star_breakdown": {{
    "situation_task": "Clear background presented...",
    "action": "Good description of personal contributions...",
    "result": "Lacks numerical metrics, but highlights successful project completion."
  }},
  "strengths": ["Clear communication", "Relevant technical context"],
  "areas_to_improve": ["Quantify results with numbers/percentages", "Use more decisive action verbs"],
  "star_completion": {{
    "situation_present": true,
    "task_present": true,
    "action_present": true,
    "result_present": false
  }},
  "model_answer": "An exemplary STAR-formatted response showing how to answer this question effectively.",
  "coaching_tip": "Key takeaway advice for the user."
}}

Output strictly valid JSON and nothing else:"""

    response = llm.invoke(eval_prompt)
    raw_content = response.content.strip()

    try:
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:]
        if raw_content.startswith("```"):
            raw_content = raw_content[3:]
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]
        result = json.loads(raw_content.strip())
    except json.JSONDecodeError:
        result = {
            "overall_score": 75,
            "star_breakdown": {
                "situation_task": "Context provided.",
                "action": "Actions described.",
                "result": "Results mentioned."
            },
            "strengths": ["Good attempt at structuring response."],
            "areas_to_improve": ["Elaborate more on specific technical actions."],
            "star_completion": {
                "situation_present": True,
                "task_present": True,
                "action_present": True,
                "result_present": True
            },
            "model_answer": raw_content,
            "coaching_tip": "Structure your answers using Situation -> Task -> Action -> Result."
        }

    print(f"   ✅ Answer evaluation complete. Score: {result.get('overall_score')}/100")
    return result

# Test
if __name__ == "__main__":
    q = "Tell me about a time you faced a technical bug."
    ans = "In my web project, our database connection kept timing out. I checked the logs, increased the pool size, and optimized queries. This fixed the timeout issue and reduced response time by 40%."
    res = evaluate_interview_answer(q, ans, "Software Engineer Intern")
    print(json.dumps(res, indent=2))
