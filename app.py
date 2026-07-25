import streamlit as st
import sys
import os

# Base Directory & Source Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from agents.orchestrator import run_pipeline

# Page config
st.set_page_config(
    page_title="🎓 Internship AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with explicit Dark/Light contrast fixes
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #3b82f6;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subheader {
        font-size: 1.1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #1e293b;
        color: #f8fafc !important;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .source-box {
        background-color: #0f172a;
        color: #cbd5e1 !important;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border-left: 4px solid #10b981;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">🎓 Internship AI Assistant</div>
    <div class="subheader">Your Personal Guide to Internship Success</div>
""", unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This AI assistant helps university students with:
    - 📝 CV writing guidance
    - 🎤 Interview preparation
    - 📋 Internship application process
    - 💌 Cover letter tips
    """)
    
    st.divider()
    
    st.header("🔧 System Info")
    st.write("""
    **Technology Stack:**
    - LangChain (Agent orchestration)
    - Groq (LLM - Llama 3.1 / 3.3)
    - ChromaDB (Vector store)
    - Streamlit (UI)
    """)

# Main input layout
st.subheader("Ask Your Question")

user_question = st.text_area(
    label="Your question:",
    placeholder="e.g., How do I write a good CV for a software internship?",
    height=100,
    key="user_input"
)

# Process button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    submit_button = st.button("🚀 Get Answer", use_container_width=True, type="primary")

# Process query
if submit_button and user_question:
    with st.spinner("🤔 Processing your question with Multi-Agent System..."):
        try:
            # Run agent pipeline
            result = run_pipeline(user_question)
            
            st.success("✅ Answer generated successfully!")
            
            # Question Category
            st.subheader("🎯 Question Category")
            intent_map = {
                "cv_help": "🔵 CV Writing Help",
                "interview_help": "🟢 Interview Preparation",
                "internship_info": "🟠 Internship Guidance",
                "cover_letter": "🔴 Cover Letter Assistance",
                "general": "⚪ General Query"
            }
            intent_display = intent_map.get(result.get("intent", "general"), "⚪ General Query")
            st.markdown(f"### {intent_display}")
            
            # Final Answer Display
            st.subheader("✨ Final Answer")
            answer_text = result.get('final_answer', 'No answer generated.')
            
            # Formatted text display inside clear container
            st.markdown(f'<div class="answer-box">{answer_text}</div>', unsafe_allow_html=True)
            
            # Sources Display
            sources = result.get("sources", [])
            if sources:
                st.subheader("📎 Sources Used")
                source_list_html = "<br>".join([f"📄 {s}" for s in sources])
                st.markdown(f'<div class="source-box">{source_list_html}</div>', unsafe_allow_html=True)
            
            # Metrics
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Chunks Retrieved from Vector DB", result.get("chunks_used", 0))
            with col_info2:
                st.metric("Pipeline Status", "✅ Fully Reviewed")
        
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")
            st.info("Please check your Groq API key in .env file.")

elif submit_button and not user_question:
    st.warning("⚠️ Please enter a question first!")

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.9rem;">
    🎓 Internship AI Assistant | Multi-Agent RAG System
    </div>
""", unsafe_allow_html=True)