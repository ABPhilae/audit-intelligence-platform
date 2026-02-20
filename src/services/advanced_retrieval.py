"""
Advanced Retrieval Service for the Audit Intelligence Platform.

Extends Project 1's retrieval with:
1. Parent-child retrieval integration (search children, return parents)
2. Retry logic with exponential backoff on external API calls
3. Graceful fallback: if re-ranker fails, use original order
"""
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential
from src.services.parent_child_retriever import ParentChildRetriever
from src.config import settings
import logging

logger = logging.getLogger(__name__)


class AdvancedRetrievalService:
    def __init__(self, base_retriever, parent_child_retriever: ParentChildRetriever = None):
        self.base_retriever = base_retriever
        self.parent_child = parent_child_retriever

        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=ChatOpenAI(model=settings.openai_model, openai_api_key=settings.openai_api_key, temperature=0.3),
        )

        self.reranker = None
        if settings.cohere_api_key:
            try:
                self.reranker = CohereRerank(
                    model="rerank-english-v3.0", top_n=settings.rerank_top_n,
                    cohere_api_key=settings.cohere_api_key,
                )
                logger.info("Cohere re-ranker initialized")
            except Exception as e:
                logger.warning(f"Cohere re-ranker not available: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _retrieve_with_retry(self, question: str, use_multi_query: bool) -> list[Document]:
        if use_multi_query:
            return self.multi_query_retriever.invoke(question)
        return self.base_retriever.invoke(question)

    def retrieve(self, question: str, use_reranking: bool = True,
                 use_multi_query: bool = True, use_parent_child: bool = True) -> list[Document]:
        # Step 1: Retrieve candidates (with retry)
        try:
            docs = self._retrieve_with_retry(question, use_multi_query)
            logger.info(f"Retrieved {len(docs)} candidates")
        except Exception as e:
            logger.error(f"Retrieval failed after retries: {e}")
            try:
                docs = self.base_retriever.invoke(question)
            except Exception:
                return []

        # Step 2: Re-rank (with fallback)
        if use_reranking and self.reranker and len(docs) > 0:
            try:
                rerank_retriever = ContextualCompressionRetriever(
                    base_compressor=self.reranker, base_retriever=self.base_retriever,
                )
                docs = rerank_retriever.invoke(question)
                logger.info(f"Re-ranked to {len(docs)} documents")
            except Exception as e:
                logger.warning(f"Re-ranking failed, using original order: {e}")
                docs = docs[:settings.rerank_top_n]

        # Step 3: Parent-child expansion
        if use_parent_child and self.parent_child:
            try:
                parent_docs = self.parent_child.get_parents_for_children(docs)
                if parent_docs:
                    docs = parent_docs
                    logger.info(f"Expanded to {len(docs)} parent chunks")
            except Exception as e:
                logger.warning(f"Parent-child expansion failed: {e}")

        return docs
