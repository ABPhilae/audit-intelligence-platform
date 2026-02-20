"""
LangChain Orchestration Service.

The central coordinator that ties together:
- LCEL chain for standard RAG
- Conversation memory for multi-turn chat
- Streaming for real-time responses
- Parent-child retrieval integration

This is the "consulting firm" — it manages the full workflow
from question to answer, delegating to specialists as needed.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from src.config import settings
from src.services.parent_child_retriever import ParentChildRetriever
import uuid
import time
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert audit intelligence assistant for a financial institution.
Answer questions based ONLY on the provided document context.
If the context does not contain enough information, say so clearly.
Always cite the specific document and section your answer comes from.
For comparative questions, structure your answer with clear sections per entity being compared.

Context from documents:
{context}"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

MAX_HISTORY = 5


class LangChainOrchestrator:
    """
    Orchestrates the full RAG pipeline with memory and streaming.
    """

    def __init__(self, retriever, parent_child_retriever: ParentChildRetriever = None):
        self.retriever = retriever
        self.parent_child = parent_child_retriever

        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            openai_api_key=settings.openai_api_key,
            max_tokens=settings.max_tokens,
            streaming=True,
        )

        # Conversation store
        self.conversations: dict[str, list] = {}

    def _format_docs(self, docs: list[Document]) -> str:
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", doc.metadata.get("source", "Unknown"))
            label = f"[Source {i}: {source}]"
            formatted.append(f"{label}\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def _get_history(self, conversation_id: str) -> list:
        history = self.conversations.get(conversation_id, [])
        messages = []
        for entry in history[-MAX_HISTORY:]:
            messages.append(HumanMessage(content=entry["question"]))
            messages.append(AIMessage(content=entry["answer"]))
        return messages

    def ask(self, question: str, conversation_id: str = None) -> dict:
        """
        Ask a question with conversation memory.

        Args:
            question: The user's question
            conversation_id: Optional conversation ID for follow-ups

        Returns:
            Dict with answer, sources, conversation_id, processing_time_ms
        """
        start_time = time.time()

        # Manage conversation
        if not conversation_id or conversation_id not in self.conversations:
            conversation_id = str(uuid.uuid4())
            self.conversations[conversation_id] = []

        # Retrieve documents
        retrieved_docs = self.retriever.invoke(question)

        # If parent-child is available, swap children for parents
        if self.parent_child:
            context_docs = self.parent_child.get_parents_for_children(retrieved_docs)
            if not context_docs:  # Fallback if no parents found
                context_docs = retrieved_docs
        else:
            context_docs = retrieved_docs

        # Build and run the chain
        context = self._format_docs(context_docs)
        history = self._get_history(conversation_id)

        chain = RAG_PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "chat_history": history,
            "question": question,
        })

        # Save exchange
        self.conversations[conversation_id].append({
            "question": question,
            "answer": answer,
        })
        if len(self.conversations[conversation_id]) > MAX_HISTORY:
            self.conversations[conversation_id] = self.conversations[conversation_id][-MAX_HISTORY:]

        processing_time = (time.time() - start_time) * 1000

        sources = [
            {
                "content": doc.page_content[:300],
                "source": doc.metadata.get("filename", "Unknown"),
                "file_type": doc.metadata.get("file_type", "unknown"),
                "page": doc.metadata.get("page"),
                "sheet_name": doc.metadata.get("sheet_name"),
                "slide_number": doc.metadata.get("slide_number"),
            }
            for doc in retrieved_docs[:5]
        ]

        return {
            "answer": answer,
            "sources": sources,
            "conversation_id": conversation_id,
            "processing_time_ms": round(processing_time, 2),
            "model_used": settings.openai_model,
        }
