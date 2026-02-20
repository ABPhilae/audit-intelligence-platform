"""
Parent-Child Retrieval Service.

The most effective advanced RAG technique for long documents.

The problem: You need SMALL chunks for precise retrieval (focused
vectors that match specific questions), but LARGE chunks for the
LLM to generate a coherent answer (more context = better generation).

The solution: Store chunks at two sizes:
- Child chunks (200 chars): Used for SEARCH — high precision
- Parent chunks (1500 chars): Sent to the LLM — rich context

When a child chunk is retrieved, the system automatically fetches
its parent chunk to give the LLM the full surrounding context.

Analogy: Imagine looking up a word in a dictionary index (small,
precise entries). Once you find the right entry, you read the FULL
dictionary page around it (large, context-rich) to understand the
word in context. The index helps you find the right place; the page
gives you the understanding.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import settings
import logging
import uuid

logger = logging.getLogger(__name__)


class ParentChildRetriever:
    """
    Implements parent-child retrieval.

    Stores both parent and child chunks. Searches children,
    returns parents for the LLM.
    """

    def __init__(self):
        # Parent splitter: large chunks for LLM context
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.parent_chunk_size,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Child splitter: small chunks for precise retrieval
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.child_chunk_size,
            chunk_overlap=20,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Store parent chunks: {parent_id: Document}
        self.parent_store: dict[str, Document] = {}

    def create_parent_child_chunks(self, documents: list[Document]) -> tuple[list[Document], list[Document]]:
        """
        Split documents into parent and child chunks.

        Returns:
            Tuple of (child_chunks_for_indexing, parent_chunks_for_context)
        """
        all_children = []

        for doc in documents:
            # Create parent chunks
            parents = self.parent_splitter.split_documents([doc])

            for parent in parents:
                parent_id = str(uuid.uuid4())
                parent.metadata["parent_id"] = parent_id
                self.parent_store[parent_id] = parent

                # Create child chunks from this parent
                children = self.child_splitter.split_documents([parent])
                for child in children:
                    child.metadata["parent_id"] = parent_id

                all_children.extend(children)

        logger.info(
            f"Created {len(self.parent_store)} parents, "
            f"{len(all_children)} children"
        )

        return all_children, list(self.parent_store.values())

    def get_parents_for_children(self, child_docs: list[Document]) -> list[Document]:
        """
        Given retrieved child chunks, return their parent chunks.

        This is the key step: search found the children (precise),
        but we send the parents (context-rich) to the LLM.
        """
        seen_parent_ids = set()
        parents = []

        for child in child_docs:
            parent_id = child.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parent_ids:
                parent = self.parent_store.get(parent_id)
                if parent:
                    parents.append(parent)
                    seen_parent_ids.add(parent_id)

        logger.info(f"Retrieved {len(parents)} parent chunks for {len(child_docs)} children")
        return parents
