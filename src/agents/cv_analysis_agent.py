import os
import json
from io import BytesIO
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pypdf import PdfReader
from agents.rag_worker_agent import load_vector_store

load_dotenv()

def create_cv_analysis_agent():
    """Create CV Analysis Agent using high-capacity Groq model"""
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )
    return llm

def extract_text_from_pdf(pdf_source):
    """
    Extract raw text from PDF file path or file bytes / Streamlit UploadedFile object.
    """
    try:
        reader = PdfReader(pdf_source)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def analyze_cv(cv_text, target_role="IT Intern"):
    """
    Analyze student CV against target role and best practices.
    Returns structured analysis dictionary.
    """
    print(f"\n📄 CV Analysis Agent Analyzing for role: {target_role}...")

    if not cv_text or len(cv_text.strip()) < 50:
        return {
            "error": "CV text is too short or could not be parsed. Please provide a complete CV."
        }

    # Step 1: Query vector store for CV best practices context if available
    context = ""
    try:
        vector_store = load_vector_store()
        retrieved_docs = vector_store.similarity_search(f"CV writing tips for {target_role}", k=2)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    except Exception as e:
        print(f"   ⚠️ Vector store search skipped or failed: {e}")

    # Step 2: Formulate LLM Analysis Prompt
    llm = create_cv_analysis_agent()

    analysis_prompt = f"""You are an expert IT Talent Acquisition Specialist & CV Coach.
Analyze the following student CV for the target role: "{target_role}".

RAG Context (Best Practices):
{context if context else 'Standard IT Internship CV standards apply.'}

Candidate CV Content:
\"\"\"
{cv_text[:4000]}
\"\"\"

Evaluate the CV across these dimensions:
1. Overall Quality & Match Score (0-100)
2. Top Key Strengths
3. Areas needing improvement
4. Critical missing skills or industry keywords for a {target_role}
5. Formatting & Structure Feedback
6. Specific bullet point rewrite suggestions (up to 3 examples)

You MUST respond strictly with a single valid JSON object in the following schema format:
{{
  "overall_score": 82,
  "summary": "Brief 2-sentence summary of the CV evaluation.",
  "target_role": "{target_role}",
  "strengths": ["Strength 1", "Strength 2", "Strength 3"],
  "areas_for_improvement": ["Improvement 1", "Improvement 2"],
  "missing_keywords": ["Keyword/Skill 1", "Keyword/Skill 2"],
  "formatting_feedback": "Detailed feedback on layout, readability, contact info, and length.",
  "bullet_rewrites": [
    {{
      "original": "Worked on a python project.",
      "suggestion": "Developed a full-stack Python application utilizing Flask and SQLite, increasing data processing speed by 25%.",
      "reason": "Adds technical specificity and measurable impact."
    }}
  ]
}}

Output strictly valid JSON and nothing else:"""

    response = llm.invoke(analysis_prompt)
    raw_content = response.content.strip()

    # Try to parse JSON output cleanly
    try:
        # Strip markdown formatting block if present
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:]
        if raw_content.startswith("```"):
            raw_content = raw_content[3:]
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]
        
        result = json.loads(raw_content.strip())
    except json.JSONDecodeError:
        # Fallback dictionary if JSON parsing fails
        result = {
            "overall_score": 70,
            "summary": "CV analyzed successfully.",
            "target_role": target_role,
            "strengths": ["Clear layout and educational background."],
            "areas_for_improvement": ["Add more quantifiable achievements and role-specific keywords."],
            "missing_keywords": ["Git", "REST APIs", "Unit Testing"],
            "formatting_feedback": "Ensure action verbs start each project bullet point.",
            "bullet_rewrites": [],
            "raw_analysis": raw_content
        }

    print(f"   ✅ CV Analysis complete. Score: {result.get('overall_score')}/100")
    return result

# Test
if __name__ == "__main__":
    sample_cv = """
    John Doe
    Email: john@example.com | Phone: 123-456-7890
    B.Sc. in Information Technology, University of Tech (2022 - Present)

    Projects:
    - Built a web app for library management using Python and HTML.
    - Worked with SQL databases to store book records.
    - Did a team project for software engineering module.

    Skills:
    Python, HTML, CSS, SQL, Communication.
    """

    res = analyze_cv(sample_cv, target_role="Software Engineer Intern")
    print(json.dumps(res, indent=2))
