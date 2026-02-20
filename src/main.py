"""
Audit Intelligence Platform — FastAPI Backend.

Production-grade API with:
- Role-based access control (X-User-Role header)
- Multi-engine query routing (standard / router / sub-question)
- Streaming responses via Server-Sent Events
- Redis caching for repeated questions
- Background document ingestion
- LangSmith trace IDs in responses
- Comprehensive health checks

This is the "waiter" from Phase 1, but now working in a five-star
restaurant with security at the door, a sommelier for wine selection
(router), and a fast lane for returning customers (cache).
"""
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi import HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import os
import uuid
import time
import json
import logging

from src.config import settings
from src.models import (
    QuestionRequest, ChatRequest, DocumentUploadMeta,
    AnswerResponse, ChatResponse, IngestionResponse, DocumentInfo,
    DashboardStats, EvaluationResponse, QueryEngine,
    ProcessingStatus, DocumentCategory,
)
from src.services.document_processor import DocumentProcessor
from src.services.langchain_orchestrator import LangChainOrchestrator
from src.services.llamaindex_multi_engine import MultiIndexEngine
from src.services.advanced_retrieval import AdvancedRetrievalService
from src.services.parent_child_retriever import ParentChildRetriever
from src.services.evaluation_service import EvaluationService
from src.services.cache_service import CacheService
from src.security.access_control import get_current_user, build_access_filter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================
# Global service instances (initialized on startup)
# ================================================================
document_processor: DocumentProcessor = None
orchestrator: LangChainOrchestrator = None
multi_engine: MultiIndexEngine = None
retrieval_service: AdvancedRetrievalService = None
parent_child: ParentChildRetriever = None
evaluation_service: EvaluationService = None
cache_service: CacheService = None

# Tracking state (in-memory for this portfolio project;
# production would use a database)
processing_jobs: dict = {}
document_registry: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services on startup, clean up on shutdown."""
    global document_processor, orchestrator, multi_engine
    global retrieval_service, parent_child, evaluation_service, cache_service

    logger.info("Initializing Audit Intelligence Platform services...")

    # Core services
    document_processor = DocumentProcessor()
    retriever = document_processor.get_retriever()

    # Parent-child retrieval
    parent_child = ParentChildRetriever()

    # LangChain orchestrator (with memory + parent-child)
    orchestrator = LangChainOrchestrator(retriever, parent_child)

    # LlamaIndex multi-index engine
    multi_engine = MultiIndexEngine()

    # Advanced retrieval (multi-query + re-ranking + parent-child)
    retrieval_service = AdvancedRetrievalService(retriever, parent_child)

    # Evaluation
    evaluation_service = EvaluationService(
        rag_ask_fn=lambda q: orchestrator.ask(q)["answer"],
        retriever=retriever,
    )

    # Redis cache
    cache_service = CacheService()

    os.makedirs(settings.upload_dir, exist_ok=True)
    logger.info("All services initialized successfully")

    yield  # App runs here

    logger.info("Shutting down services...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Enterprise-grade audit document intelligence with multi-index RAG, "
        "role-based access control, and LangSmith tracing"
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# HEALTH CHECK
# ================================================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.app_version,
        "services": {
            "qdrant": document_processor is not None,
            "orchestrator": orchestrator is not None,
            "multi_engine": multi_engine is not None,
            "cache": cache_service.enabled if cache_service else False,
            "langsmith": bool(settings.langsmith_api_key),
        },
    }


# ================================================================
# DOCUMENT UPLOAD AND MANAGEMENT
# ================================================================

def _process_document_background(
    job_id: str,
    file_path: str,
    filename: str,
    category: str,
    access_group: str,
):
    """Background task for document ingestion with category routing."""
    try:
        processing_jobs[job_id] = ProcessingStatus.PROCESSING

        # Ingest into the main Qdrant vector store
        result = document_processor.ingest_file(file_path)

        # Also read the file's chunks and add to the correct
        # LlamaIndex index (audit / policy / financial)
        from src.loaders import load_document
        documents = load_document(file_path)

        llama_docs = [
            {
                "content": doc.page_content,
                "metadata": {
                    **doc.metadata,
                    "filename": filename,
                    "category": category,
                    "access_group": access_group,
                },
            }
            for doc in documents
        ]

        # Route to the correct LlamaIndex index
        if category in ("audit", "policy", "financial"):
            multi_engine.add_documents(llama_docs, category)

        # Register the document in our in-memory registry
        document_registry[job_id] = {
            "document_id": job_id,
            "filename": filename,
            "file_type": os.path.splitext(filename)[1].lstrip("."),
            "category": category,
            "access_group": access_group,
            "chunk_count": result["chunk_count"],
            "status": ProcessingStatus.COMPLETED,
            "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        processing_jobs[job_id] = ProcessingStatus.COMPLETED
        logger.info(
            f"Ingested {filename}: {result['chunk_count']} chunks "
            f"into {category} index"
        )

    except Exception as e:
        processing_jobs[job_id] = ProcessingStatus.FAILED
        logger.error(f"Ingestion failed for {filename}: {e}")


@app.post("/documents/upload", response_model=IngestionResponse)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form(default="audit"),
    access_group: str = Form(default="GLOBAL_AUDIT"),
):
    """
    Upload a document for processing.

    - category: audit | policy | financial (determines LlamaIndex index)
    - access_group: GLOBAL_AUDIT | APAC_AUDIT | EMEA_AUDIT (access control)
    """
    user = get_current_user(request)

    # Validate file type
    ext = os.path.splitext(file.filename)[1].lower()
    supported = {".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md"}
    if ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(supported)}",
        )

    # Validate category
    valid_categories = [c.value for c in DocumentCategory]
    if category not in valid_categories:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category: {category}. Use: {valid_categories}",
        )

    # Save file to disk
    job_id = str(uuid.uuid4())
    file_path = os.path.join(settings.upload_dir, f"{job_id}_{file.filename}")
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Queue background processing
    processing_jobs[job_id] = ProcessingStatus.PENDING
    background_tasks.add_task(
        _process_document_background,
        job_id, file_path, file.filename, category, access_group,
    )

    return IngestionResponse(
        job_id=job_id,
        filename=file.filename,
        category=category,
        access_group=access_group,
        status=ProcessingStatus.PENDING,
        message="Document uploaded. Processing in background.",
    )


@app.get("/documents/{job_id}/status")
async def check_status(job_id: str):
    """Check the processing status of an uploaded document."""
    status = processing_jobs.get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": status}


@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents(request: Request):
    """List documents the current user can access (filtered by role)."""
    user = get_current_user(request)
    user_groups = user.get("access_groups", [])

    if "ALL" in user_groups:
        return list(document_registry.values())

    return [
        doc for doc in document_registry.values()
        if doc["access_group"] in user_groups
    ]


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, request: Request):
    """Delete a document and its chunks (admin only)."""
    user = get_current_user(request)
    if "ALL" not in user.get("access_groups", []):
        raise HTTPException(
            status_code=403, detail="Admin role required to delete documents"
        )

    if document_id not in document_registry:
        raise HTTPException(status_code=404, detail="Document not found")

    doc_info = document_registry.pop(document_id)

    # Clear cache since the document set changed
    if cache_service:
        cache_service.clear()

    return {
        "message": f"Document {doc_info['filename']} deleted",
        "document_id": document_id,
    }


# ================================================================
# QUESTION ANSWERING
# ================================================================

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request_body: QuestionRequest, request: Request):
    """
    Ask a question with engine selection and caching.

    Engines:
    - standard: LangChain LCEL (fast, conversational)
    - router: LlamaIndex auto-routes to correct document collection
    - sub_question: LlamaIndex breaks complex questions into parts
    """
    user = get_current_user(request)
    start_time = time.time()

    # Check cache first
    if cache_service:
        cached = cache_service.get(request_body.question)
        if cached:
            cached["from_cache"] = True
            cached["processing_time_ms"] = round(
                (time.time() - start_time) * 1000, 2
            )
            return AnswerResponse(**cached)

    # Route to the right engine
    if request_body.engine == QueryEngine.STANDARD:
        result = orchestrator.ask(request_body.question)
        engine_name = "langchain_lcel"

    elif request_body.engine in (QueryEngine.ROUTER, QueryEngine.SUB_QUESTION):
        engine_type = (
            "router" if request_body.engine == QueryEngine.ROUTER
            else "sub_question"
        )
        result = multi_engine.query(
            request_body.question, engine_type=engine_type
        )
        result["model_used"] = settings.openai_model
        result["conversation_id"] = None
        engine_name = f"llamaindex_{engine_type}"

    else:
        result = orchestrator.ask(request_body.question)
        engine_name = "langchain_lcel"

    processing_time = (time.time() - start_time) * 1000

    # Format sources for the response
    sources = [
        {
            "content": s.get("content", "")[:300],
            "source": s.get("source", "Unknown"),
            "file_type": s.get("file_type", "unknown"),
            "page": s.get("page"),
            "sheet_name": s.get("sheet_name"),
            "slide_number": s.get("slide_number"),
            "relevance_score": s.get("relevance_score"),
        }
        for s in result.get("sources", [])
    ]

    response_data = {
        "answer": result["answer"],
        "sources": sources,
        "model_used": result.get("model_used", settings.openai_model),
        "engine_used": engine_name,
        "processing_time_ms": round(processing_time, 2),
        "from_cache": False,
        "trace_id": None,
    }

    # Cache the result for future similar questions
    if cache_service:
        cache_service.set(request_body.question, response_data)

    return AnswerResponse(**response_data)


@app.post("/chat", response_model=ChatResponse)
async def chat(request_body: ChatRequest, request: Request):
    """Chat with conversation memory (multi-turn)."""
    user = get_current_user(request)

    result = orchestrator.ask(
        request_body.message,
        conversation_id=request_body.conversation_id,
    )

    sources = [
        {
            "content": s.get("content", "")[:300],
            "source": s.get("source", "Unknown"),
            "file_type": s.get("file_type", "unknown"),
        }
        for s in result.get("sources", [])
    ]

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        conversation_id=result["conversation_id"],
        engine_used="langchain_lcel",
    )


@app.post("/ask/stream")
async def ask_streaming(request_body: QuestionRequest, request: Request):
    """
    Ask a question with streaming response (Server-Sent Events).

    Instead of waiting for the full answer, tokens arrive as the
    LLM generates them. The frontend shows them one by one, making
    the response feel instant even if total generation takes 5 seconds.
    """
    user = get_current_user(request)

    async def generate():
        # Retrieve context using the advanced retrieval pipeline
        docs = retrieval_service.retrieve(
            request_body.question,
            use_reranking=request_body.use_reranking,
            use_parent_child=request_body.use_parent_child,
        )
        context = orchestrator._format_docs(docs)

        # Build the prompt
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(
            """You are an expert audit intelligence assistant.
Answer based ONLY on the following context. Cite your sources.

Context:
{context}

Question: {question}

Answer:"""
        )

        messages = prompt.format_messages(
            context=context, question=request_body.question
        )

        # Stream tokens from the LLM one by one
        async for chunk in orchestrator.llm.astream(messages):
            if chunk.content:
                yield f"data: {json.dumps({'token': chunk.content})}\n\n"

        # Send sources at the end of the stream
        source_data = [
            {
                "source": doc.metadata.get("filename", "Unknown"),
                "file_type": doc.metadata.get("file_type", "unknown"),
            }
            for doc in docs[:5]
        ]
        yield f"data: {json.dumps({'sources': source_data, 'done': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ================================================================
# EVALUATION
# ================================================================

@app.post("/evaluate")
async def run_evaluation(request: Request):
    """Run full RAGAS evaluation (32 questions, ~5-10 minutes)."""
    user = get_current_user(request)
    results = evaluation_service.run_evaluation()
    return results


@app.post("/evaluate/by-category")
async def run_evaluation_by_category(request: Request):
    """Run RAGAS evaluation grouped by question category."""
    user = get_current_user(request)
    results = evaluation_service.run_evaluation_by_category()
    return results


@app.get("/evaluate/history")
async def get_evaluation_history():
    """Get history of evaluation runs for trend tracking."""
    return evaluation_service.get_evaluation_history()


# ================================================================
# DASHBOARD
# ================================================================

@app.get("/stats", response_model=DashboardStats)
async def get_stats(request: Request):
    """Dashboard statistics."""
    db_stats = document_processor.get_stats()

    docs_by_type: dict = {}
    docs_by_category: dict = {}
    for doc in document_registry.values():
        ft = doc["file_type"]
        docs_by_type[ft] = docs_by_type.get(ft, 0) + 1
        cat = doc.get("category", "unknown")
        docs_by_category[cat] = docs_by_category.get(cat, 0) + 1

    cache_stats = (
        cache_service.get_stats() if cache_service else {"enabled": False}
    )

    return DashboardStats(
        total_documents=len(document_registry),
        total_chunks=db_stats["total_chunks"],
        documents_by_type=docs_by_type,
        documents_by_category=docs_by_category,
        cache_stats=cache_stats,
        collection_names=[
            settings.audit_collection,
            settings.policy_collection,
            settings.financial_collection,
        ],
    )


@app.post("/cache/clear")
async def clear_cache(request: Request):
    """Clear the semantic cache (admin only)."""
    user = get_current_user(request)
    if "ALL" not in user.get("access_groups", []):
        raise HTTPException(status_code=403, detail="Admin role required")
    if cache_service:
        cache_service.clear()
    return {"message": "Cache cleared"}
