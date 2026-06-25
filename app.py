"""Streamlit UI — shows the answer AND the retrieved evidence + citations.
Run: streamlit run app.py
"""
import streamlit as st

from src.config import settings
from src.generate import answer

st.set_page_config(page_title="Enterprise RAG + Eval", page_icon="📄", layout="wide")
st.title("📄 Grounded Document Q&A over SEC 10-K Filings")
st.caption("RAG with hybrid retrieval + reranking, citations, and guardrails.")

with st.sidebar:
    st.header("Pipeline")
    st.write(f"**Embeddings:** {settings.embed_model}")
    st.write(f"**Reranker:** {settings.reranker_model if settings.use_reranker else 'off'}")
    st.write(f"**Hybrid (BM25+dense):** {'on' if settings.use_hybrid else 'off'}")
    st.write(f"**LLM backend:** {settings.llm_backend}")
    st.write(f"**top_k → LLM:** {settings.top_k_final}")
    st.info("Toggle components via env vars and compare in eval/EVALUATION_REPORT.md")

q = st.text_input("Ask a question about the filings:",
                  "What are the company's main risk factors?")
if st.button("Ask", type="primary") and q:
    with st.spinner("Retrieving + generating..."):
        a = answer(q)
    if a.refused:
        st.warning(a.text)
    else:
        st.subheader("Answer")
        st.write(a.text)
    if a.sources:
        st.subheader("Retrieved evidence")
        for h in a.sources:
            with st.expander(f"[{h.source}:{h.chunk}]  ·  score={h.score:.3f}"):
                st.write(h.text)
