# Internship AI Assistant

## 1. Project Description
An Agentic AI system that helps university IT students with internship preparation,
CV guidance, and interview readiness using multi-agent collaboration and RAG.

## 2. Features
- Router Agent for intent detection
- RAG Worker Agent for document-based answers
- Reflection Agent for answer improvement
- Streamlit web interface

## 3. Architecture Diagram
      
      

```text
Student Question
      |
      v
[Router Agent]  (Fast/Cheap Model)
      |
      |  structured message
      |  {intent, query, needs_rag}
      v
[RAG Worker Agent]  (Strong Reasoning Model)
      |
      |  1) retrieve top document chunks
      |  2) generate draft answer
      v
[Reflection Agent]
      |
      |  improve clarity + completeness
      v
Final Answer
      |
      v
Streamlit UI

## 4. Agent Communication Flow


1. User enters a question in Streamlit.
2. Router Agent classifies intent (`cv_help`, `interview_help`, `internship_info`, `general_question`).
3. Router sends a structured message to RAG Worker Agent.
4. RAG Worker Agent retrieves relevant chunks from the vector database.
5. RAG Worker Agent generates a draft answer using retrieved context.
6. Reflection Agent reviews and improves the draft.
7. Final answer is shown in the Streamlit UI.

### Structured message example
```json
{
  "intent": "cv_help",
  "query": "How do I write a CV for a software internship?",
  "needs_rag": true
}

## 5. Model Selection Strategy
| Sub-task | Model | Why chosen |
|----------|-------|------------|
| Intent routing | TBD | TBD |
| Final answer generation | TBD | TBD |

## 6. RAG Pipeline
- Corpus:
- Chunking strategy:
- Embedding model:
- Vector store:
- Retrieval evaluation:

## 7. Setup Instructions
(to be added)

## 8. Live Demo
Streamlit URL: (to be added)

## 9. Design Patterns Used
- **Router pattern** → Router Agent decides which path to use
- **Tool-use pattern** → RAG Worker uses vector DB retrieval as a tool
- **ReAct pattern** → retrieve → reason → answer
- **Reflection pattern** → Reflection Agent improves final answer

## 10. Known Limitations
(to be added)