"""Chat page — conversational Q&A with engine selection and streaming."""
import streamlit as st
import requests

API_URL = "http://localhost:8000"
ROLE = st.session_state.get("user_role", "admin")
HEADERS = {"X-User-Role": ROLE}

st.title("💬 Chat with Audit Documents")

# -----------------------------------------------------------
# Engine selector — lets you pick which query engine to use
# -----------------------------------------------------------
col1, col2 = st.columns([3, 1])
with col2:
    engine = st.selectbox(
        "Query Engine",
        ["standard", "router", "sub_question"],
        format_func=lambda x: {
            "standard": "⚡ Standard (fast)",
            "router": "🔀 Router (auto-select)",
            "sub_question": "🔍 Sub-Question (multi-doc)",
        }.get(x, x),
        help=(
            "Standard: LangChain LCEL chain (fast, good for direct questions)\n"
            "Router: automatically picks the right document collection\n"
            "Sub-Question: breaks complex questions into parts "
            "(best for comparisons)"
        ),
    )

# -----------------------------------------------------------
# Session state for chat history
# -----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

# Controls
col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("🔄 New Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()
with col_b:
    use_reranking = st.checkbox(
        "Re-ranking", value=True, help="Cohere re-ranking for better quality"
    )

# -----------------------------------------------------------
# Display chat history
# -----------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander(f"📎 Sources ({len(message['sources'])})"):
                for source in message["sources"]:
                    icon = {
                        "pdf": "📕", "docx": "📘",
                        "xlsx": "📗", "pptx": "📙"
                    }.get(source.get("file_type", ""), "📄")
                    st.markdown(
                        f"{icon} **{source.get('source', 'Unknown')}**"
                    )
                    st.caption(source.get("content", "")[:200])
        if "meta" in message:
            st.caption(message["meta"])

# -----------------------------------------------------------
# Chat input and response
# -----------------------------------------------------------
if prompt := st.chat_input("Ask about your audit documents..."):
    # Show the user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if engine == "standard":
                    # Use /chat for conversation memory
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "message": prompt,
                            "conversation_id": st.session_state.conversation_id,
                            "engine": engine,
                        },
                        headers=HEADERS,
                        timeout=120,
                    )
                else:
                    # Use /ask for router/sub-question engines
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={
                            "question": prompt,
                            "engine": engine,
                            "use_reranking": use_reranking,
                            "use_parent_child": True,
                        },
                        headers=HEADERS,
                        timeout=120,
                    )

                if response.status_code == 200:
                    data = response.json()
                    st.markdown(data["answer"])

                    # Save conversation ID for multi-turn
                    if data.get("conversation_id"):
                        st.session_state.conversation_id = data[
                            "conversation_id"
                        ]

                    # Display sources
                    sources = data.get("sources", [])
                    if sources:
                        with st.expander(
                            f"📎 Sources ({len(sources)})"
                        ):
                            for source in sources:
                                icon = {
                                    "pdf": "📕", "docx": "📘",
                                    "xlsx": "📗", "pptx": "📙",
                                }.get(source.get("file_type", ""), "📄")
                                st.markdown(
                                    f"{icon} **{source.get('source', 'Unknown')}**"
                                )
                                st.caption(source.get("content", "")[:200])

                    # Display metadata bar
                    meta_parts = []
                    if data.get("engine_used"):
                        meta_parts.append(f"Engine: {data['engine_used']}")
                    if data.get("processing_time_ms"):
                        meta_parts.append(
                            f"Time: {data['processing_time_ms']:.0f}ms"
                        )
                    if data.get("from_cache"):
                        meta_parts.append("⚡ From cache")
                    meta = " | ".join(meta_parts)
                    if meta:
                        st.caption(meta)

                    # Save to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": sources,
                        "meta": meta,
                    })

                elif response.status_code == 403:
                    st.error(
                        "🔒 Access denied. Your role does not have permission."
                    )
                else:
                    st.error(f"Error: {response.text}")

            except requests.ConnectionError:
                st.error(
                    "Cannot connect to the API. Is the backend running?"
                )
            except requests.Timeout:
                st.error(
                    "Request timed out. Try a simpler question "
                    "or the standard engine."
                )
