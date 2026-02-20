"""
Document Processor Service.

Core ingestion and retrieval service:
- Loads documents via the multi-format loader router
- Splits into chunks with RecursiveCharacterTextSplitter
- Embeds with OpenAI text-embedding-3-small
- Stores vectors in Qdrant
- Exposes a LangChain retriever for the orchestrator
"""
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from src.config import settings
from src.loaders import load_document
import logging

logger = logging.getLogger(__name__)

# text-embedding-3-small produces 1536-dimensional vectors
EMBEDDING_DIM = 1536


class DocumentProcessor:
    """
    Handles document ingestion into Qdrant and provides a retriever
    for the LangChain orchestrator.
    """

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )

        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        self.collection_name = settings.audit_collection
        self._ensure_collection()

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _ensure_collection(self):
        """Create the Qdrant collection if it does not exist yet."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")

    def get_retriever(self):
        """Return a LangChain retriever over the main vector store."""
        return self.vector_store.as_retriever(
            search_kwargs={"k": settings.retrieval_top_k}
        )

    def ingest_file(self, file_path: str) -> dict:
        """
        Load, split, embed and store a document file.

        Returns:
            {"chunk_count": int}
        """
        documents = load_document(file_path)
        chunks = self.splitter.split_documents(documents)

        if chunks:
            self.vector_store.add_documents(chunks)

        logger.info(f"Ingested {file_path}: {len(chunks)} chunks")
        return {"chunk_count": len(chunks)}

    def get_stats(self) -> dict:
        """Return basic statistics about the vector store."""
        try:
            info = self.client.get_collection(self.collection_name)
            total_chunks = info.points_count or 0
        except Exception:
            total_chunks = 0
        return {"total_chunks": total_chunks}
