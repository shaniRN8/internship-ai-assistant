import streamlit as st
import sys
import os
import json

# ---- Project paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---- Page config (must be first Streamlit command) ----
st.set_page_config(
    page_title="Internship AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Auto-build vector store (for cloud deployment) ----
@st.cache_resource
def ensure_vector_store():
    """Build ChromaDB if it does not exist (first run on Streamlit Cloud)."""
    vector_dir = os.path.join(BASE_DIR, "data", "vectorstore")

    needs_build = (
        not os.path.exists(vector_dir)
        or len(os.listdir(vector_dir)) == 0
    )

    if needs_build:
        with st.spinner("🔨 Building knowledge base (first time only, ~2 minutes)..."):
            from rag.ingest import run_ingestion
            run_ingestion()
        st.success("✅ Knowledge base built successfully!")

    return True

# ---- Load orchestrator pipeline ----
@st.cache_resource
def get_pipeline():
    from agents.orchestrator import run_pipeline
    return run_pipeline

# ---- Custom CSS ----
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subheader {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .agent-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .score-badge {
        font-size: 2rem;
        font-weight: bold;
        color: #2e7d32;
    }
    .star-chip {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-right: 0.5rem;
    }
    .star-present {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #a5d6a7;
    }
    .star-missing {
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #ef9a9a;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown('<div class="main-header">🎓 Internship AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Your Autonomous Multi-Agent Suite for Internship Success</div>', unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    An intelligent multi-agent platform supporting university IT students with:
    - 💬 **Instant RAG Q&A**
    - 📄 **Automated CV Analysis & Scoring**
    - 🎙️ **Mock Interview Coaching & STAR Evaluation**
    """)

    st.divider()

    st.header("🤖 5-Agent Architecture")
    st.write("""
    1. **Router Agent** — Classifies query intent
    2. **RAG Worker Agent** — Retrieves context & drafts answers
    3. **Reflection Agent** — Reviews & polishes output
    4. **CV Analysis Agent** — Scores CVs & suggests bullet revisions
    5. **Interview Coach Agent** — Evaluates answers using STAR framework
    """)

    st.divider()

    st.header("🔧 Tech Stack")
    st.write("""
    - LangChain (Orchestration)
    - Groq (Llama 3.3 70B & 3.1 8B)
    - ChromaDB (Vector Store)
    - PyPDF (Document Parsing)
    - HuggingFace (Embeddings)
    - Streamlit (UI)
    """)

    st.divider()
    st.caption("Built for IT41043 — Intelligent Systems")

# ---- Ensure knowledge base exists ----
try:
    ensure_vector_store()
except Exception as e:
    st.error(f"❌ Failed to build knowledge base: {e}")
    st.stop()


# ---- Main Navigation Tabs ----
tab1, tab2, tab3 = st.tabs([
    "💬 Ask AI Assistant",
    "📄 CV Analysis Agent",
    "🎙️ Interview Coach Agent"
])


# ==============================================================================
# TAB 1: ASK AI ASSISTANT (RAG Pipeline)
# ==============================================================================
with tab1:
    st.subheader("Ask Your Internship Question")
    
    input_method = st.radio(
        "How would you like to ask?",
        ["Type your question", "Select from examples"],
        horizontal=True,
        key="rag_input_method"
    )

    example_questions = {
        "CV Writing": "How do I write a good CV for an IT internship?",
        "Interview Prep": "What are common internship interview questions?",
        "How to Apply": "What is the process for applying to internships?",
        "Cover Letter": "What should I include in a cover letter?",
        "Required Documents": "What documents are needed for internship applications?"
    }

    if input_method == "Type your question":
        user_question = st.text_area(
            "Your question:",
            placeholder="e.g., How do I write a good CV for a software internship?",
            height=100,
            key="user_rag_question"
        )
    else:
        selected = st.selectbox("Choose an example question:", list(example_questions.keys()), key="example_rag_select")
        user_question = example_questions[selected]
        st.info(user_question)

    st.divider()

    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        submit = st.button("🚀 Get Answer", use_container_width=True, type="primary", key="submit_rag")

    if submit:
        if not user_question or not user_question.strip():
            st.warning("⚠️ Please enter a question first.")
        else:
            try:
                run_pipeline = get_pipeline()

                with st.spinner("🤔 Agents are working on your question..."):
                    result = run_pipeline(user_question)

                st.success("✅ Answer generated successfully!")

                # Question
                st.subheader("📝 Your Question")
                st.info(user_question)

                # Intent
                st.subheader("🎯 Detected Category")
                intent_labels = {
                    "cv_help": "🔵 CV Help",
                    "cv_analysis": "📄 CV Analysis",
                    "interview_help": "🟢 Interview Help",
                    "interview_coaching": "🎙️ Interview Coaching",
                    "internship_info": "🟠 Internship Information",
                    "cover_letter": "🔴 Cover Letter",
                    "general": "⚫ General"
                }
                intent = result.get("intent", "general")
                st.write(intent_labels.get(intent, f"⚫ {intent}"))

                # Final answer
                st.subheader("✨ Final Answer")
                with st.container(border=True):
                    st.markdown(result.get("final_answer", "No answer generated."))

                # Sources
                sources = result.get("sources", [])
                if sources:
                    st.subheader("📎 Sources Used")
                    with st.container(border=True):
                        for src in sources:
                            st.write(f"📄 `{os.path.basename(str(src))}`")
                else:
                    st.warning("⚠️ No sources retrieved for this question.")

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Chunks Retrieved", result.get("chunks_used", 0))
                m2.metric("Sources Used", len(sources))
                m3.metric("Status", result.get("status", "complete"))

                # Agent pipeline details
                with st.expander("🔍 View agent pipeline details"):
                    st.markdown("**Router Agent output (intent):**")
                    st.code(intent)

                    st.markdown("**RAG Worker Agent draft answer:**")
                    st.text(result.get("draft_answer", "N/A")[:1000])

                    st.markdown("**Reflection Agent final answer:**")
                    st.text(result.get("final_answer", "N/A")[:1000])

            except Exception as e:
                st.error(f"❌ Error processing your question: {e}")
                st.info("Check that your GROQ_API_KEY is configured correctly.")


# ==============================================================================
# TAB 2: CV ANALYSIS AGENT
# ==============================================================================
with tab2:
    st.subheader("📄 Automated CV Analysis & Feedback Agent")
    st.write("Upload your CV or paste your CV text to receive a detailed match evaluation, missing keywords audit, and action-oriented rewrite suggestions.")

    col_role, col_mode = st.columns([1, 1])
    with col_role:
        target_role = st.selectbox(
            "Target Internship Role:",
            [
                "Software Engineer Intern",
                "Data Analyst Intern",
                "DevOps Intern",
                "Cybersecurity Intern",
                "QA / Test Engineer Intern",
                "Full Stack Developer Intern",
                "IT Support Specialist Intern"
            ],
            key="cv_target_role"
        )
    with col_mode:
        cv_source_type = st.radio(
            "CV Source:",
            ["Upload PDF Document", "Paste Text Content"],
            horizontal=True,
            key="cv_source_type"
        )

    cv_text_to_analyze = ""

    if cv_source_type == "Upload PDF Document":
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], key="cv_pdf_uploader")
        if uploaded_file is not None:
            from agents.cv_analysis_agent import extract_text_from_pdf
            cv_text_to_analyze = extract_text_from_pdf(uploaded_file)
            st.success(f"✅ Extracted {len(cv_text_to_analyze)} characters from uploaded PDF `{uploaded_file.name}`.")
            with st.expander("👁️ View Extracted CV Text"):
                st.text(cv_text_to_analyze[:1500] + ("..." if len(cv_text_to_analyze) > 1500 else ""))
    else:
        cv_text_to_analyze = st.text_area(
            "Paste your CV text here:",
            placeholder="Paste your resume/CV text including education, projects, skills, and experience...",
            height=250,
            key="cv_text_area"
        )

    st.divider()

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        analyze_btn = st.button("🔍 Analyze CV", type="primary", use_container_width=True, key="analyze_cv_btn")

    if analyze_btn:
        if not cv_text_to_analyze or len(cv_text_to_analyze.strip()) < 50:
            st.warning("⚠️ Please upload a valid PDF or paste substantial CV text (at least 50 characters).")
        else:
            from agents.orchestrator import run_cv_analysis
            with st.spinner("🤖 CV Analysis Agent is evaluating your resume against industry standards..."):
                analysis_res = run_cv_analysis(cv_text_to_analyze, target_role=target_role)

            if "error" in analysis_res:
                st.error(f"❌ {analysis_res['error']}")
            else:
                st.success("✅ CV Evaluation Complete!")

                # Score & Summary Header
                score = analysis_res.get("overall_score", 0)
                m_col1, m_col2 = st.columns([1, 3])
                with m_col1:
                    st.metric(label=f"Overall Match Score for {target_role}", value=f"{score} / 100")
                with m_col2:
                    st.subheader("📊 Executive Summary")
                    st.write(analysis_res.get("summary", "Analysis completed successfully."))

                st.divider()

                # Strengths & Improvements
                s_col, i_col = st.columns(2)
                with s_col:
                    st.markdown("### 🟢 Top Strengths")
                    for strength in analysis_res.get("strengths", []):
                        st.markdown(f"- ✅ {strength}")
                with i_col:
                    st.markdown("### 🟠 Key Areas to Improve")
                    for item in analysis_res.get("areas_for_improvement", []):
                        st.markdown(f"- ⚠️ {item}")

                st.divider()

                # Missing Keywords
                st.markdown("### 🔑 Missing Skills & Industry Keywords")
                missing_kw = analysis_res.get("missing_keywords", [])
                if missing_kw:
                    st.write("Consider adding these relevant keywords to your skills or project descriptions:")
                    kw_cols = st.columns(min(len(missing_kw), 5))
                    for idx, kw in enumerate(missing_kw):
                        with kw_cols[idx % len(kw_cols)]:
                            st.info(f"💡 `{kw}`")
                else:
                    st.success("Great job! No major critical keywords appear missing.")

                # Formatting feedback
                st.markdown("### 🎨 Layout & Formatting Feedback")
                st.info(analysis_res.get("formatting_feedback", "No specific formatting issues found."))

                # Actionable Bullet Rewrites
                st.markdown("### ✍️ Actionable Bullet Point Improvements")
                rewrites = analysis_res.get("bullet_rewrites", [])
                if rewrites:
                    for idx, item in enumerate(rewrites, 1):
                        with st.expander(f"Recommendation #{idx}: {item.get('original', 'Bullet Point')[:50]}..."):
                            st.markdown(f"**Original Bullet:** `{item.get('original')}`")
                            st.markdown(f"**Suggested Rewrite:** `{item.get('suggestion')}`")
                            st.caption(f"💡 **Reason:** {item.get('reason')}")
                else:
                    st.write("No specific bullet point rewrites recommended.")


# ==============================================================================
# TAB 3: INTERVIEW COACH AGENT
# ==============================================================================
with tab3:
    st.subheader("🎙️ Mock Interview Coach & STAR Answer Evaluator")
    st.write("Practice interview questions tailored to your target role and get real-time feedback using the STAR framework.")

    ic_tab1, ic_tab2 = st.tabs(["🎲 Practice Question Generator", "⚡ Evaluate My Answer"])

    # --- Sub-tab 1: Question Generator ---
    with ic_tab1:
        st.markdown("### Generate Tailored Interview Questions")
        q_col1, q_col2, q_col3 = st.columns([1, 1, 1])
        with q_col1:
            q_role = st.selectbox(
                "Target Role:",
                ["Software Engineer Intern", "Data Analyst Intern", "DevOps Intern", "Cybersecurity Intern", "Full Stack Developer Intern"],
                key="gen_q_role"
            )
        with q_col2:
            q_cat = st.selectbox(
                "Question Category:",
                ["Behavioral (STAR Method)", "Technical & Coding Concept", "System Design & Architecture", "Internship Fit & Motivation"],
                key="gen_q_cat"
            )
        with q_col3:
            q_count = st.slider("Number of Questions:", min_value=2, max_value=5, value=3, key="gen_q_count")

        if st.button("✨ Generate Questions", type="primary", key="btn_gen_q"):
            from agents.interview_coach_agent import generate_mock_questions
            with st.spinner("🤖 Interview Coach is formulating realistic questions..."):
                questions = generate_mock_questions(target_role=q_role, category=q_cat, count=q_count)

            st.markdown("### 📋 Generated Interview Questions:")
            for idx, q in enumerate(questions, 1):
                st.info(f"**Question {idx}:** {q}")

    # --- Sub-tab 2: STAR Answer Evaluator ---
    with ic_tab2:
        st.markdown("### Evaluate Your Answer with the STAR Framework")
        st.caption("The STAR framework stands for Situation, Task, Action, and Result.")

        eval_role = st.selectbox(
            "Role for Answer Evaluation:",
            ["Software Engineer Intern", "Data Analyst Intern", "DevOps Intern", "Cybersecurity Intern", "Full Stack Developer Intern"],
            key="eval_role"
        )

        eval_question = st.text_input(
            "Interview Question:",
            value="Tell me about a time you faced a challenging technical bug or problem.",
            key="eval_question_input"
        )

        user_answer = st.text_area(
            "Your Response:",
            placeholder="Structure your answer with what happened, what your goal was, what specific action you took, and what the final outcome was...",
            height=180,
            key="eval_answer_input"
        )

        if st.button("⚡ Evaluate Answer", type="primary", key="btn_eval_ans"):
            if not user_answer or len(user_answer.strip()) < 20:
                st.warning("⚠️ Please enter a detailed answer (at least 20 characters) for evaluation.")
            else:
                from agents.orchestrator import run_interview_coach
                with st.spinner("🎙️ Interview Coach Agent is analyzing your STAR structure and technical depth..."):
                    feedback = run_interview_coach(eval_question, user_answer, target_role=eval_role)

                if "error" in feedback:
                    st.error(f"❌ {feedback['error']}")
                else:
                    st.success("✅ Evaluation Complete!")

                    # Score Header
                    e_score = feedback.get("overall_score", 0)
                    st.metric(label="STAR Communication Score", value=f"{e_score} / 100")

                    # STAR Badge Checklist
                    st.markdown("#### STAR Framework Checklist")
                    star_comp = feedback.get("star_completion", {})

                    c_sit, c_task, c_act, c_res = st.columns(4)
                    c_sit.markdown(f"**Situation:** {'✅ Present' if star_comp.get('situation_present') else '❌ Missing'}")
                    c_task.markdown(f"**Task:** {'✅ Present' if star_comp.get('task_present') else '❌ Missing'}")
                    c_act.markdown(f"**Action:** {'✅ Present' if star_comp.get('action_present') else '❌ Missing'}")
                    c_res.markdown(f"**Result:** {'✅ Present' if star_comp.get('result_present') else '❌ Missing'}")

                    st.divider()

                    # Breakdown
                    st.markdown("#### 🔍 STAR Element Breakdown")
                    breakdown = feedback.get("star_breakdown", {})
                    st.write(f"**Situation & Task:** {breakdown.get('situation_task', 'N/A')}")
                    st.write(f"**Action Taken:** {breakdown.get('action', 'N/A')}")
                    st.write(f"**Result / Outcome:** {breakdown.get('result', 'N/A')}")

                    st.divider()

                    # Strengths & Improvement
                    e_col1, e_col2 = st.columns(2)
                    with e_col1:
                        st.markdown("#### 🟢 Strengths")
                        for s in feedback.get("strengths", []):
                            st.markdown(f"- ✅ {s}")
                    with e_col2:
                        st.markdown("#### 🟠 Areas to Refine")
                        for tip in feedback.get("areas_to_improve", []):
                            st.markdown(f"- 💡 {tip}")

                    # Model Answer & Tip
                    st.markdown("#### 🏆 Exemplary STAR Model Answer")
                    st.success(feedback.get("model_answer", "N/A"))

                    st.markdown("#### 💡 Coach's Pro Tip")
                    st.info(feedback.get("coaching_tip", "Keep practicing using clear quantifiable metrics in your results!"))


# ---- Footer ----
st.divider()
st.caption("🎓 Internship AI Assistant | LangChain · Groq · ChromaDB · Streamlit")