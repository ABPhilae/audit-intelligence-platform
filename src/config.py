"""
Configuration for the Audit Intelligence Platform.

Extends Project 1's configuration with:
- LangSmith tracing settings (observability)
- Role-based access control settings
- Redis caching configuration
- Parent-child retrieval parameters
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Audit Intelligence Platform"
    app_version: str = "2.0.0"
    debug: bool = False

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    max_tokens: int = 2500

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    # Multiple collections for the router
    audit_collection: str = "audit_reports"
    policy_collection: str = "policies"
    financial_collection: str = "financial_data"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_ttl_seconds: int = 3600  # 1 hour cache

    # Cohere
    cohere_api_key: str = ""

    # LangSmith (observability)
    langsmith_api_key: str = ""
    langsmith_project: str = "audit-intelligence-platform"
    langchain_tracing_v2: str = "true"

    # Retrieval settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    # Parent-child: small chunks for search, large for LLM
    parent_chunk_size: int = 1500
    child_chunk_size: int = 200
    retrieval_top_k: int = 20
    rerank_top_n: int = 5

    # Security
    enable_access_control: bool = True

    # File upload
    max_upload_size_mb: int = 100
    upload_dir: str = "data/uploads"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Enable LangSmith tracing if API key is provided
if settings.langsmith_api_key:
    import os
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
