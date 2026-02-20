"""
Audit Intelligence Platform — Streamlit Frontend.

5-page app:
- Chat: Conversational Q&A with engine selection
- Upload: Document upload with category and access group tagging
- Dashboard: Stats, cache, and system health
- Evaluation: RAGAS scores with category breakdown
- Documents: Document manager with details and deletion
"""
import streamlit as st

st.set_page_config(
    page_title="Audit Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------
# Sidebar — role selector (simulates authentication)
# -----------------------------------------------------------
st.sidebar.title("🏛️ Audit Intelligence")
st.sidebar.markdown("Enterprise Document Intelligence")
st.sidebar.markdown("---")

role = st.sidebar.selectbox(
    "👤 Current Role",
    ["admin", "apac_auditor", "emea_auditor", "viewer"],
    index=0,
    help="Simulates role-based access control. Different roles see different documents.",
)
st.session_state["user_role"] = role

role_names = {
    "admin": "🔑 Admin (full access)",
    "apac_auditor": "🌏 APAC Auditor",
    "emea_auditor": "🌍 EMEA Auditor",
    "viewer": "👁️ Viewer (read-only)",
}
st.sidebar.info(f"Logged in as: {role_names.get(role, role)}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Powered by:**")
st.sidebar.markdown("LangChain • LlamaIndex • RAGAS")
st.sidebar.markdown("Qdrant • Redis • LangSmith")

# -----------------------------------------------------------
# Main page content
# -----------------------------------------------------------
st.title("🏛️ Audit Intelligence Platform")
st.markdown("""
Enterprise-grade audit document intelligence with multi-format ingestion,
advanced RAG, and measurable quality scores.

**Features:**
- **Multi-format**: PDF, Word, Excel, PowerPoint, plain text
- **Three query engines**: Standard (fast), Router (auto-select), Sub-Question (multi-doc)
- **Role-based access**: Documents filtered by your permissions
- **Quality measured**: RAGAS evaluation across 32 test questions
- **Full audit trail**: Every query traced via LangSmith

Navigate using the pages in the sidebar.
""")
