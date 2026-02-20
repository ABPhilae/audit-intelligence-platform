"""Evaluation page — RAGAS scores with category breakdown."""
import streamlit as st
import requests

API_URL = "http://localhost:8000"
ROLE = st.session_state.get("user_role", "admin")
HEADERS = {"X-User-Role": ROLE}

st.title("🎯 RAG Quality Evaluation")
st.markdown("""
Evaluate the RAG system against 32 test questions across 6 categories:
**factual**, **comparative**, **multi-hop**, **structural**, **synthesis**,
and **meta**.
""")

# -----------------------------------------------------------
# Evaluation buttons
# -----------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Run Full Evaluation", type="primary"):
        with st.spinner(
            "Running RAGAS evaluation (32 questions)... ~5-10 minutes"
        ):
            try:
                resp = requests.post(
                    f"{API_URL}/evaluate",
                    headers=HEADERS,
                    timeout=900,
                )
                if resp.status_code == 200:
                    r = resp.json()
                    if "error" in r:
                        st.error(f"Error: {r['error']}")
                    else:
                        # Overall score with color coding
                        score = r["overall_score"]
                        color = (
                            "🏆" if score >= 0.9 else
                            "✅" if score >= 0.7 else
                            "⚠️" if score >= 0.5 else "❌"
                        )
                        st.success(
                            f"{color} Overall Score: {score:.2%}"
                        )

                        # Individual metric scores
                        m1, m2 = st.columns(2)
                        m1.metric(
                            "Faithfulness",
                            f"{r['faithfulness']:.2%}",
                        )
                        m1.metric(
                            "Context Precision",
                            f"{r['context_precision']:.2%}",
                        )
                        m2.metric(
                            "Answer Relevancy",
                            f"{r['answer_relevancy']:.2%}",
                        )
                        m2.metric(
                            "Context Recall",
                            f"{r['context_recall']:.2%}",
                        )
                        st.caption(
                            f"Evaluated {r['questions_evaluated']} "
                            f"questions at {r['timestamp']}"
                        )
            except requests.Timeout:
                st.error("Evaluation timed out.")
            except requests.ConnectionError:
                st.error("Cannot connect to the API.")

with col2:
    if st.button("📊 Run By Category"):
        with st.spinner("Running category-level evaluation..."):
            try:
                resp = requests.post(
                    f"{API_URL}/evaluate/by-category",
                    headers=HEADERS,
                    timeout=900,
                )
                if resp.status_code == 200:
                    r = resp.json()
                    for cat, scores in r.get("category_scores", {}).items():
                        icon = {
                            "factual": "📌", "comparative": "⚖️",
                            "multi_hop": "🔗", "structural": "🏗️",
                            "synthesis": "🧠",
                        }.get(cat, "📋")
                        overall = scores.get("overall_score")
                        count = scores.get("question_count", 0)
                        if overall is not None:
                            st.markdown(
                                f"{icon} **{cat.replace('_', ' ').title()}** "
                                f"({count}q): **{overall:.2%}**"
                            )
                        else:
                            st.markdown(
                                f"{icon} **{cat.replace('_', ' ').title()}** "
                                f"({count}q): Error"
                            )
            except Exception as e:
                st.error(f"Error: {e}")

# -----------------------------------------------------------
# Evaluation history (shows previous runs for trend tracking)
# -----------------------------------------------------------
st.markdown("---")
st.subheader("📈 Evaluation History")
try:
    resp = requests.get(f"{API_URL}/evaluate/history", headers=HEADERS)
    if resp.status_code == 200:
        history = resp.json()
        if history:
            for entry in reversed(history):
                st.markdown(
                    f"**{entry['timestamp'][:16]}** — "
                    f"Overall: {entry['overall_score']:.2%} "
                    f"(F:{entry['faithfulness']:.2f} "
                    f"AR:{entry['answer_relevancy']:.2f} "
                    f"CP:{entry['context_precision']:.2f} "
                    f"CR:{entry['context_recall']:.2f})"
                )
        else:
            st.info("No evaluations run yet.")
except Exception:
    pass

# -----------------------------------------------------------
# Reference table for interpreting scores
# -----------------------------------------------------------
st.markdown("---")
st.markdown("""
### Score Interpretation Guide

| Low Score | Likely Problem | Fix |
|-----------|---------------|-----|
| Low Faithfulness | LLM hallucinating | Improve prompt, lower temperature |
| Low Answer Relevancy | Not answering the question | Review prompt, check retrieval |
| Low Context Precision | Too much noise in retrieval | Add re-ranking, raise threshold |
| Low Context Recall | Missing relevant chunks | Improve chunking, increase top_k |
""")
