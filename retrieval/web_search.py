"""Web search with LLM-based reranking for fact-checking.

Performs web search via Serper API with optional Gemini 2.5 Flash reranking
to improve relevance for fact-checking claims.
"""

from typing import Optional

from retrieval.document_block import Block
from retrieval.retriever_client import retrieve_via_api
from retrieval.listwise_llm_reranker import llm_rerank
from utils.logger import logger


async def search_web(
    search_query: str,
    num_results: int,
    do_client_side_reranking: bool = False,
    num_results_before_reranking: Optional[int] = None,
    article_to_exclude: Optional[str] = None,
    reranking_engine: Optional[str] = None,
    model_provider: str = "google_genai",
) -> list[Block]:
    """Search the web using Serper API with optional LLM reranking.

    Args:
        search_query: The search query
        num_results: Number of final results to return
        do_client_side_reranking: If True, apply Gemini 2.5 Flash reranking
        num_results_before_reranking: Number of results to retrieve before reranking
        article_to_exclude: Optional article title to filter out
        reranking_engine: Legacy parameter (ignored, Gemini is always used)
        model_provider: Legacy parameter (ignored, Gemini is always used)

    Returns:
        list[Block]: Search results with trust ratings, optionally reranked
    """
    if num_results_before_reranking is None:
        num_results_before_reranking = num_results

    # If reranking requested, get more results to rerank
    if do_client_side_reranking:
        num_results_before_reranking = max(num_results_before_reranking, num_results * 2)

    search_results: list[Block] = await retrieve_via_api(
        query=search_query,
        retriever_endpoint=None,  # Uses Serper.dev API
        num_results=num_results_before_reranking,
    )

    # Apply LLM reranking if requested
    if do_client_side_reranking and len(search_results) > 0:
        logger.info(f"🔄 Reranking {len(search_results)} results using Gemini 2.5 Flash...")
        try:
            search_results = await llm_rerank(
                query=search_query,
                query_result=search_results,
                query_context=None,  # Could pass claim context if needed
            )
            logger.info(f"✓ Reranking complete. Kept {len(search_results)} results.")
        except Exception as e:
            logger.warning(f"⚠ Reranking failed: {e}. Using Serper's original ranking.")

    # Exclude article if specified
    if article_to_exclude:
        search_results = [
            result for result in search_results
            if result.document_title != article_to_exclude
        ]

    # Return top N results after reranking/filtering
    if len(search_results) > num_results:
        search_results = search_results[:num_results]

    return search_results
