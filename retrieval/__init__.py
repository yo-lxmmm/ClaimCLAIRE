"""Retrieval package public interface.

This package provides:

Classes:
    Block: Base retrieval/indexing unit.

Functions:
    search_web: Web search via Serper API with optional Gemini 2.5 Flash reranking
    llm_rerank: LLM-based listwise reranking for search results

Only symbols listed in __all__ are considered stable. Internal modules should not be imported
from outside the package; prefer importing from `retrieval` directly:

    from retrieval import Block, search_web, llm_rerank

"""

from .document_block import Block
from .web_search import search_web
from .listwise_llm_reranker import llm_rerank


__all__ = [
    "Block",
    "search_web",
    "llm_rerank",
]
