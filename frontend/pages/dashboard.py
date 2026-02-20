"""Dashboard page — statistics, cache info, and system health."""
import streamlit as st
import requests

API_URL = "http://localhost:8000"
ROLE = st.session_state.get("user_role", "admin")
HEADERS = {"X-User-Role": ROLE}

st.title("📊 Dashboard")

try:
    # Fetch data from API
    stats = requests.get(f"{API_URL}/stats", headers=HEADERS).json()
    health = requests.get(f"{API_URL}/health", headers=HEADERS).json()

    # -----------------------------------------------------------
    # Key metrics row
    # -----------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Documents", stats["total_documents"])
    col2.metric("🧩 Chunks", stats["total_chunks"])
    col3.metric("🗂️ Collections", len(stats.get("collection_names", [])))
    cache = stats.get("cache_stats", {})
    col4.metric("⚡ Cached", cache.get("cached_entries", 0))

    # -----------------------------------------------------------
    # Documents by file type
    # -----------------------------------------------------------
    st.subheader("Documents by File Type")
    if stats["documents_by_type"]:
        icons = {
            "pdf": "📕", "docx": "📘", "xlsx": "📗",
            "pptx": "📙", "txt": "📄",
        }
        for ft, count in stats["documents_by_type"].items():
            st.markdown(f"{icons.get(ft, '📄')} **{ft.upper()}**: {count}")
    else:
        st.info("No documents uploaded yet.")

    # -----------------------------------------------------------
    # Documents by category
    # -----------------------------------------------------------
    st.subheader("Documents by Category")
    if stats["documents_by_category"]:
        cat_icons = {"audit": "📋", "policy": "📜", "financial": "💰"}
        for cat, count in stats["documents_by_category"].items():
            st.markdown(
                f"{cat_icons.get(cat, '📄')} **{cat.title()}**: {count}"
            )

    # -----------------------------------------------------------
    # Cache status and management
    # -----------------------------------------------------------
    st.subheader("Cache Status")
    if cache.get("enabled"):
        st.markdown(
            f"✅ Redis connected — {cache['cached_entries']} cached entries "
            f"— TTL: {cache.get('ttl_seconds', 3600)}s"
        )
        if ROLE == "admin":
            if st.button("🗑️ Clear Cache"):
                requests.post(f"{API_URL}/cache/clear", headers=HEADERS)
                st.success("Cache cleared!")
                st.rerun()
    else:
        st.warning("❌ Redis not connected — caching disabled")

    # -----------------------------------------------------------
    # System health indicators
    # -----------------------------------------------------------
    st.subheader("System Health")
    for service, ok in health.get("services", {}).items():
        status_icon = "✅" if ok else "❌"
        status_text = "Running" if ok else "Down"
        st.markdown(f"{status_icon} **{service.title()}**: {status_text}")

except requests.ConnectionError:
    st.warning("Cannot connect to the API.")
