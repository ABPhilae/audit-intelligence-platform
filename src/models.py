"""
Data models for the Audit Intelligence Platform.

Extends Project 1's models with:
- Access control fields (user role, access groups)
- Multi-engine query options (router, sub-question, standard)
- Streaming request model
- Category-level evaluation results
- Cache statistics
"""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ================================================================
# ENUMS
# ================================================================

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryEngine(str, Enum):
    """Which query engine to use."""
    STANDARD = "standard"          # LangChain LCEL chain
    ROUTER = "router"              # LlamaIndex auto-routing
    SUB_QUESTION = "sub_question"  # LlamaIndex multi-document


class DocumentCategory(str, Enum):
    """Document categories for multi-index routing."""
    AUDIT = "audit"
    POLICY = "policy"
    FINANCIAL = "financial"


# ================================================================
# REQUEST MODELS
# ================================================================

class QuestionRequest(BaseModel):
    """A question with advanced query options."""
    question: str = Field(..., min_length=5, max_length=2000)
    engine: QueryEngine = Field(
        default=QueryEngine.STANDARD,
        description="Query engine: standard (fast), router (auto-select), sub_question (multi-doc)"
    )
    use_reranking: bool = Field(default=True)
    use_parent_child: bool = Field(
        default=True,
        description="Use parent-child retrieval for richer context"
    )
    top_k: int = Field(default=5, ge=1, le=20)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "question": "Compare the audit findings between Hong Kong and Singapore",
                "engine": "sub_question",
                "use_reranking": True,
                "use_parent_child": True,
            }]
        }
    }


class ChatRequest(BaseModel):
    """Conversational chat request."""
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    engine: QueryEngine = Field(default=QueryEngine.STANDARD)


class DocumentUploadMeta(BaseModel):
    """Optional metadata for uploaded documents."""
    category: DocumentCategory = Field(default=DocumentCategory.AUDIT)
    access_group: str = Field(
        default="GLOBAL_AUDIT",
        description="Access group for role-based filtering"
    )
    description: Optional[str] = None


# ================================================================
# RESPONSE MODELS
# ================================================================

class SourceChunk(BaseModel):
    content: str
    source: str
    file_type: str
    page: Optional[int] = None
    sheet_name: Optional[str] = None
    slide_number: Optional[int] = None
    relevance_score: Optional[float] = None


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model_used: str
    engine_used: str
    processing_time_ms: float
    from_cache: bool = False
    trace_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    conversation_id: str
    engine_used: str


class IngestionResponse(BaseModel):
    job_id: str
    filename: str
    category: str
    access_group: str
    status: ProcessingStatus
    message: str


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_type: str
    category: str
    access_group: str
    chunk_count: int
    status: ProcessingStatus
    uploaded_at: str


class DashboardStats(BaseModel):
    total_documents: int
    total_chunks: int
    documents_by_type: dict[str, int]
    documents_by_category: dict[str, int]
    cache_stats: dict
    collection_names: list[str]


class CategoryEvaluationResult(BaseModel):
    """Per-category RAGAS scores."""
    category: str
    question_count: int
    overall_score: Optional[float]
    faithfulness: Optional[float]
    answer_relevancy: Optional[float]
    context_precision: Optional[float]
    context_recall: Optional[float]


class EvaluationResponse(BaseModel):
    """Full evaluation response with category breakdown."""
    overall_score: float
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    questions_evaluated: int
    category_breakdown: Optional[list[CategoryEvaluationResult]] = None
    timestamp: str
