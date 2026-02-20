"""Documents page — document manager with details, access groups, and deletion."""
import streamlit as st
import requests

API_URL = "http://localhost:8000"
ROLE = st.session_state.get("user_role", "admin")
HEADERS = {"X-User-Role": ROLE}

st.title("📁 Document Manager")

try:
    resp = requests.get(f"{API_URL}/documents", headers=HEADERS)
    if resp.status_code == 200:
        docs = resp.json()

        if not docs:
            st.info(
                "No documents uploaded yet. "
                "Go to the Upload page to add documents."
            )
        else:
            st.markdown(
                f"**{len(docs)} document(s)** visible with your "
                f"current role ({ROLE})"
            )
            st.markdown("---")

            for doc in docs:
                # File type icon
                icon = {
                    "pdf": "📕", "docx": "📘", "xlsx": "📗",
                    "pptx": "📙", "txt": "📄",
                }.get(doc.get("file_type", ""), "📄")

                # Category icon
                cat_icon = {
                    "audit": "📋", "policy": "📜", "financial": "💰",
                }.get(doc.get("category", ""), "📄")

                with st.container():
                    col1, col2, col3, col4, col5 = st.columns(
                        [3, 2, 2, 1, 1]
                    )

                    with col1:
                        st.markdown(f"{icon} **{doc['filename']}**")

                    with col2:
                        st.caption(
                            f"{cat_icon} {doc.get('category', 'N/A').title()}"
                        )

                    with col3:
                        st.caption(f"🔒 {doc.get('access_group', 'N/A')}")

                    with col4:
                        st.caption(
                            f"🧩 {doc.get('chunk_count', 0)} chunks"
                        )

                    with col5:
                        if ROLE == "admin":
                            if st.button(
                                "🗑️",
                                key=f"del_{doc['document_id']}",
                                help="Delete this document",
                            ):
                                del_resp = requests.delete(
                                    f"{API_URL}/documents/{doc['document_id']}",
                                    headers=HEADERS,
                                )
                                if del_resp.status_code == 200:
                                    st.success(f"Deleted {doc['filename']}")
                                    st.rerun()
                                else:
                                    st.error(
                                        f"Delete failed: {del_resp.text}"
                                    )

                    # Expandable details panel
                    with st.expander("Details"):
                        st.markdown(
                            f"**Document ID:** `{doc['document_id']}`"
                        )
                        st.markdown(
                            f"**File Type:** {doc['file_type'].upper()}"
                        )
                        st.markdown(
                            f"**Category:** {doc.get('category', 'N/A')}"
                        )
                        st.markdown(
                            f"**Access Group:** "
                            f"{doc.get('access_group', 'N/A')}"
                        )
                        st.markdown(
                            f"**Chunks:** {doc.get('chunk_count', 0)}"
                        )
                        st.markdown(
                            f"**Status:** {doc.get('status', 'N/A')}"
                        )
                        st.markdown(
                            f"**Uploaded:** {doc.get('uploaded_at', 'N/A')}"
                        )

                st.markdown("---")

    elif resp.status_code == 403:
        st.error("🔒 Access denied for your role.")
    else:
        st.error(f"Error loading documents: {resp.text}")

except requests.ConnectionError:
    st.warning("Cannot connect to the API.")
