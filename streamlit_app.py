# streamlit_app.py
import streamlit as st
from datetime import datetime
import google.generativeai as genai
import json

from config import pc, GOOGLE_API_KEY, PINECONE_INDEX
from rag import (
    load_and_process_all_pdfs,
    embed_chunks,
    upsert_to_pinecone,
    calculate_metrics,
    extract_json
)

# ---- Setup Page ----
st.set_page_config(page_title="AI Resume Matcher", layout="wide")
st.title("🤖 RAG Based LLM with GEMINI Vision Model")

# ---- Initial Load + Indexing ----
with st.spinner("📥 Loading and embedding all resumes from Azure Blob..."):
    all_chunks = load_and_process_all_pdfs()
    embedded_chunks = embed_chunks(all_chunks)
    index = upsert_to_pinecone(embedded_chunks)
    st.success(f"✅ {len(embedded_chunks)} chunks embedded and indexed from Azure Blob PDFs.")

# ---- Sidebar Query Input ----
st.sidebar.header("🔍 Ask a Question")
query = st.sidebar.text_input("Type your question", " ")
search_btn = st.sidebar.button("Search")

# ---- Query Logic ----
if search_btn and query:
    st.info("🔍 Running hybrid search...")

    response = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = response["embedding"]

    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    context = "\n\n".join([match["metadata"]["text"] for match in results["matches"]])

    prompt = f"""
You are a helpful assistant that returns answers in raw JSON format without any preamble or explanation.

ONLY return a valid JSON object with the following structure:
{{
  "top_candidate": "string",
  "experience_years": number,
  "skills": ["string", ...],
  "matched_chunks": ["string", ...]
}}

Using the context below, identify the best matching candidate.

Context:
{context}

Question:
{query}

IMPORTANT: Do not include any commentary, explanation, or formatting outside the JSON object.
"""
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    rag_response = model.generate_content(prompt)
    answer_json = extract_json(rag_response.text)

    if answer_json:
        st.subheader("🧠 Gemini JSON Answer")
        st.json(answer_json)

        correct_answer = answer_json.get("top_candidate", "").strip()
        if not correct_answer:
            st.warning("⚠️ No `top_candidate` found in Gemini response. Evaluation may be invalid.")

        eval_metrics = calculate_metrics(
            [match["metadata"]["text"] for match in results["matches"]],
            correct_answer,
            correct_answer
        )

        evaluation_result = {
            "query": query,
            "generated_answer": answer_json,
            "evaluation_metrics": eval_metrics,
            "timestamp": datetime.now().isoformat()
        }

        with open('rag_evaluations.json', 'a') as f:
            json.dump(evaluation_result, f, indent=4)

        st.subheader("📊 Evaluation Metrics")
        st.write(f"**Answer Accuracy**: {eval_metrics['answer_accuracy']}")
        st.write(f"**Precision@5**: {eval_metrics['precision_at_k']}")
        st.write(f"**Recall@5**: {eval_metrics['recall_at_k']}")
        st.write(f"**MRR**: {eval_metrics['mrr']}")
    else:
        st.error("⚠️ Gemini response could not be parsed as JSON.")
        st.subheader("🔎 Raw Gemini Response")
        st.code(rag_response.text)
