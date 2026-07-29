import os
from agents.router_agent import route_query
from agents.rag_worker_agent import process_query
from agents.reflection_agent import reflect_and_improve
from agents.cv_analysis_agent import analyze_cv, extract_text_from_pdf
from agents.interview_coach_agent import generate_mock_questions, evaluate_interview_answer

def run_pipeline(user_question):
    """
    Full RAG Agentic Pipeline:
    User → Router → RAG Worker → Reflection → Final Answer
    """
    
    print("=" * 50)
    print(f"🎯 User Question: {user_question}")
    print("=" * 50)
    
    # Step 1: Router Agent
    print("\n[STEP 1] Router Agent...")
    router_message = route_query(user_question)
    
    # Step 2: RAG Worker Agent
    print("\n[STEP 2] RAG Worker Agent...")
    worker_output = process_query(router_message)
    
    # Step 3: Reflection Agent
    print("\n[STEP 3] Reflection Agent...")
    final_output = reflect_and_improve(worker_output)
    
    print("\n" + "=" * 50)
    print("✅ PIPELINE COMPLETE")
    print("=" * 50)
    print(f"\n🎯 Question: {final_output['query']}")
    print(f"📂 Intent: {final_output['intent']}")
    print(f"📎 Sources: {final_output['sources']}")
    print(f"\n✨ Final Answer:\n{final_output['final_answer']}")
    
    return final_output

def run_cv_analysis(cv_input, target_role="IT Intern"):
    """
    Run dedicated CV Analysis Agent workflow.
    cv_input can be a text string or a PDF file object/path.
    """
    if isinstance(cv_input, str):
        if cv_input.lower().endswith(".pdf") and os.path.exists(cv_input):
            cv_text = extract_text_from_pdf(cv_input)
        else:
            cv_text = cv_input
    else:
        cv_text = extract_text_from_pdf(cv_input)

    return analyze_cv(cv_text, target_role=target_role)

def run_interview_coach(question, user_answer, target_role="IT Intern"):
    """
    Run dedicated Interview Coach Agent workflow.
    """
    return evaluate_interview_answer(question, user_answer, target_role=target_role)

# Test
if __name__ == "__main__":
    question = "How do I write a good CV for a software internship?"
    result = run_pipeline(question)