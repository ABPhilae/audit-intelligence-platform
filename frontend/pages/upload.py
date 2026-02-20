"""Upload page — document upload with category and access group tagging."""
import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"
ROLE = st.session_state.get("user_role", "admin")
HEADERS = {"X-User-Role": ROLE}

st.title("📤 Upload Documents")

# -----------------------------------------------------------
# Upload options (category + access group)
# -----------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    category = st.selectbox(
        "Document Category",
        ["audit", "policy", "financial"],
        format_func=lambda x: {
            "audit": "📋 Audit Reports",
            "policy": "📜 Policies & Compliance",
            "financial": "💰 Financial Data",
        }.get(x, x),
        help="Determines which LlamaIndex index the document is routed to.",
    )
with col2:
    access_group = st.selectbox(
        "Access Group",
        ["GLOBAL_AUDIT", "APAC_AUDIT", "EMEA_AUDIT"],
        help="Controls which roles can see this document.",
    )

# -----------------------------------------------------------
# File uploader (multi-file)
# -----------------------------------------------------------
uploaded_files = st.file_uploader(
    "Drag and drop files here",
    type=["pdf", "docx", "xlsx", "pptx", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.spinner(f"Uploading {uploaded_file.name}..."):
            response = requests.post(
                f"{API_URL}/documents/upload",
                files={
                    "file": (uploaded_file.name, uploaded_file.getvalue())
                },
                data={
                    "category": category,
                    "access_group": access_group,
                },
                headers=HEADERS,
            )

            if response.status_code == 200:
                result = response.json()
                job_id = result["job_id"]
                st.info(
                    f"📄 {uploaded_file.name} → {category} / {access_group} "
                    f"(Job: {job_id[:8]}...)"
                )

                # Poll for completion with progress bar
                progress = st.progress(0)
                status = "pending"
                attempts = 0
                while status not in ("completed", "failed") and attempts < 60:
                    time.sleep(2)
                    attempts += 1
                    progress.progress(min(attempts * 5, 95))
                    resp = requests.get(
                        f"{API_URL}/documents/{job_id}/status",
                        headers=HEADERS,
                    )
                    if resp.status_code == 200:
                        status = resp.json()["status"]

                progress.progress(100)
                if status == "completed":
                    st.success(
                        f"✅ {uploaded_file.name} — processed successfully!"
                    )
                else:
                    st.error(
                        f"❌ {uploaded_file.name} — processing failed"
                    )
            else:
                st.error(f"Upload failed: {response.text}")
