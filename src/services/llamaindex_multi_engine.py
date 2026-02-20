"""
LlamaIndex Multi-Index Architecture.

Three specialised indexes connected by a Router:
1. Audit Reports Index — for finding-specific queries
2. Policy Index — for compliance and regulation queries
3. Financial Data Index — for budget and financial queries

Plus two advanced query engines:
- SubQuestionQueryEngine: Breaks complex questions into sub-questions
- RouterQueryEngine: Automatically picks the right index

Analogy: Instead of one big library with every book on one shelf,
you now have three specialised sections (audit, compliance, finance).
The librarian (Router) knows which section to check based on your
question. For complex questions ("compare audit findings with budget
allocations"), the research assistant (Sub-Question Engine) visits
multiple sections and combines what they find.
"""
from llama_index.core import VectorStoreIndex, Settings as LlamaSettings
from llama_index.core.query_engine import SubQuestionQueryEngine, RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from src.config import settings
import logging

logger = logging.getLogger(__name__)


class MultiIndexEngine:
    """
    Multi-index LlamaIndex architecture with Router and Sub-Question engines.
    """

    def __init__(self):
        LlamaSettings.llm = LlamaOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        LlamaSettings.embed_model = OpenAIEmbedding(
            model_name=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

        # Three specialised indexes
        self.indexes = {
            "audit": None,
            "policy": None,
            "financial": None,
        }
        self.query_engines = {}
        self.router_engine = None
        self.sub_question_engine = None

    def add_documents(self, documents: list[dict], index_name: str):
        """
        Add documents to a specific index.

        Args:
            documents: List of dicts with 'content' and 'metadata'
            index_name: One of 'audit', 'policy', 'financial'
        """
        if index_name not in self.indexes:
            raise ValueError(f"Unknown index: {index_name}. Use: {list(self.indexes.keys())}")

        nodes = [
            TextNode(text=doc["content"], metadata=doc.get("metadata", {}))
            for doc in documents
        ]

        self.indexes[index_name] = VectorStoreIndex(nodes)
        self.query_engines[index_name] = self.indexes[index_name].as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
        )
        logger.info(f"Built LlamaIndex '{index_name}' index with {len(nodes)} nodes")

        # Rebuild router and sub-question engines
        self._rebuild_engines()

    def _rebuild_engines(self):
        """Rebuild the Router and Sub-Question engines after index changes."""
        tools = []
        descriptions = {
            "audit": "Audit reports, findings, observations, risk assessments, and remediation status",
            "policy": "Compliance policies, regulations, procedures, and governance frameworks",
            "financial": "Financial data, budgets, cost allocations, and financial tracking spreadsheets",
        }

        for name, engine in self.query_engines.items():
            tools.append(QueryEngineTool.from_defaults(
                query_engine=engine,
                description=descriptions.get(name, f"Documents in the {name} collection"),
            ))

        if len(tools) >= 2:
            # Router: automatically picks the right index
            self.router_engine = RouterQueryEngine(
                selector=PydanticSingleSelector.from_defaults(),
                query_engine_tools=tools,
            )

            # Sub-Question: breaks complex queries into sub-questions
            self.sub_question_engine = SubQuestionQueryEngine.from_defaults(
                query_engine_tools=tools,
            )
            logger.info(f"Rebuilt Router and Sub-Question engines with {len(tools)} tools")

    def query(self, question: str, engine_type: str = "router") -> dict:
        """
        Query across indexes.

        Args:
            question: Natural language question
            engine_type: 'router' (auto-select index) or 'sub_question' (multi-index)

        Returns:
            Dict with answer and sources
        """
        engine = None
        if engine_type == "sub_question" and self.sub_question_engine:
            engine = self.sub_question_engine
        elif self.router_engine:
            engine = self.router_engine
        elif self.query_engines:
            # Fallback: use the first available engine
            engine = list(self.query_engines.values())[0]

        if engine is None:
            return {"answer": "No documents indexed yet.", "sources": []}

        response = engine.query(question)

        sources = []
        for node in getattr(response, "source_nodes", []):
            sources.append({
                "content": node.text[:300] + "..." if len(node.text) > 300 else node.text,
                "source": node.metadata.get("filename", "Unknown"),
                "file_type": node.metadata.get("file_type", "unknown"),
                "relevance_score": round(node.score, 4) if node.score else None,
            })

        return {"answer": str(response), "sources": sources}
