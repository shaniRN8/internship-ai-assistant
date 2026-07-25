import streamlit as st
import sys
import os



# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from agents.orchestrator import run_pipeline
# Add src to path


# Page config
st.set_page_config(
    page_title="🎓 Internship AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-top: 1rem;
    }
    .source-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
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
    - Groq (LLM - Llama 3.1/3.3)
    - ChromaDB (Vector store)
    - Streamlit (UI)
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ask Your Question")
    
    # Input options
    input_method = st.radio(
        "How would you like to ask?",
        ["Type your question", "Select from examples"],
        horizontal=True
    )
    
    if input_method == "Type your question":
        user_question = st.text_area(
            label="Your question:",
            placeholder="e.g., How do I write a good CV for a software internship?",
            height=100,
            key="user_input"
        )
    else:
        example_questions = {
            "CV Writing": "How do I write a good CV for an IT internship?",
            "Interview Prep": "What are common internship interview questions?",
            "How to Apply": "What's the process for applying to internships?",
            "Cover Letter": "How should I write a cover letter for an internship?",
            "Documents": "What documents are needed for internship applications?"
        }
        selected_example = st.selectbox(
            "Choose an example question:",
            list(example_questions.keys())
        )
        user_question = example_questions[selected_example]
        st.info(f"**Selected:** {user_question}")

with col2:
    st.subheader("Quick Actions")
    
    # Example buttons
    if st.button("❓ Ask Example Question", use_container_width=True):
        user_question = "How do I write a good CV for an IT internship?"
    
    if st.button("🔄 Clear All", use_container_width=True):
        st.rerun()

# Process button
st.divider()

col_submit = st.columns([1, 1, 1])
with col_submit[1]:
    submit_button = st.button(
        "🚀 Get Answer",
        use_container_width=True,
        type="primary"
    )

# Process query
if submit_button and user_question:
    with st.spinner("🤔 Processing your question..."):
        try:
            # Run agent pipeline
            result = run_pipeline(user_question)
            
            # Display results
            st.success("✅ Answer generated successfully!")
            
            # Question echo
            st.subheader("📝 Your Question")
            st.write(f"**{user_question}**")
            
            # Intent detected
            st.subheader("🎯 Question Category")
            intent_colors = {
                "cv_help": "🔵",
                "interview_help": "🟢",
                "internship_info": "🟠",
                "cover_letter": "🔴",
                "general": "⚫"
            }
            intent_emoji = intent_colors.get(result.get("intent", "general"), "⚫")
            st.write(f"{intent_emoji} **{result.get('intent', 'general').replace('_', ' ').title()}**")
            
            # Final answer
            st.subheader("✨ Final Answer")
            with st.container():
                st.markdown(f"""
                <div class="answer-box">
                {result.get('final_answer', 'No answer generated')}
                </div>
                """, unsafe_allow_html=True)
            
            # Sources
            sources = result.get("sources", [])
            if sources:
                st.subheader("📎 Sources Used")
                with st.container():
                    st.markdown("""
                    <div class="source-box">
                    """, unsafe_allow_html=True)
                    for source in sources:
                        st.write(f"📄 {source}")
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Additional info
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Chunks Retrieved", result.get("chunks_used", 0))
            with col_info2:
                st.metric("Processing Status", "✅ Complete")
        
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")
            st.info("Please check your API keys and try again.")

elif submit_button and not user_question:
    st.warning("⚠️ Please enter a question first!")

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
    🎓 Internship AI Assistant | Built with LangChain, Groq & Streamlit
    </div>
""", unsafe_allow_html=True)