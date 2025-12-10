import asyncio
import os
import re
import weakref
import pickle
from datetime import datetime, timedelta
from contextvars import ContextVar

import httpx
from langchain.chat_models import init_chat_model

from retrieval.document_block import Block
from retrieval.source_trust_rating import get_source_trust_rating
from utils.disk_cache import cache
import utils.logger as logger


logger = logger.logger

# Prompt for rephrasing sensitive queries
QUERY_REPHRASE_PROMPT = """You are helping to rephrase a search query that was blocked by a content filter.

The original query was rejected, likely because it contains sensitive terms, profanity, slurs, or other content that violates search API policies.

Your task: Rephrase the query to search for the same information while avoiding sensitive terms.

CRITICAL: You MUST preserve ALL specific details from the original query:
- Keep all names, dates, locations, and specific facts EXACTLY as they appear
- Keep all quoted phrases (except the offensive term itself)
- Maintain the exact context and claim being investigated
- Only replace the offensive/blocked terms with neutral descriptive alternatives

Guidelines:
1. Replace ONLY the explicit/offensive terms with neutral, descriptive alternatives
2. Keep ALL other details, names, dates, and context identical
3. Use professional, academic language for replacements
4. Maintain the exact factual claim and search intent
5. Do NOT summarize, generalize, or lose any specificity

Examples:
- "Judge Amy Barrett said N-word not hostile environment" → "Judge Amy Barrett said racial slur not hostile environment"
- "Biden f*** you statement 2020" → "Biden profane insult statement 2020"
- "Photo shows [explicit violence] at protest" → "Photo shows graphic violence at protest"

Original query: {query}

Respond with ONLY the rephrased query, no explanation or quotes."""


async def _rephrase_sensitive_query(query: str) -> str | None:
    """Use LLM to rephrase a query that was blocked by content filters.
    
    Args:
        query: The original query that was blocked
        
    Returns:
        Rephrased query string, or None if rephrasing failed
    """
    try:
        # Use Gemini 2.5 Flash for query rephrasing (needs to handle sensitive content accurately)
        model = init_chat_model(
            model="gemini-2.5-flash",
            model_provider="google_genai",
            temperature=0.3,  # Low temperature for consistent rephrasing
        )
        
        response = await model.ainvoke([
            {"role": "user", "content": QUERY_REPHRASE_PROMPT.format(query=query)}
        ])
        
        rephrased = response.content.strip()
        # Remove quotes if LLM added them
        rephrased = rephrased.strip('"').strip("'")
        
        logger.info(f"Rephrased query from '{query}' to '{rephrased}'")
        return rephrased
        
    except Exception as e:
        logger.error(f"Failed to rephrase query: {e}")
        return None


async def _retrieve_with_rephrased_query(
    rephrased_query: str,
    num_results: int,
    cutoff_date_str: str | None,
    original_cache_key: bytes,
) -> list[Block]:
    """Helper to retry retrieval with a rephrased query.
    
    This is separated to avoid infinite recursion - we only retry once.
    """
    cutoff_date = _parse_cutoff_date(cutoff_date_str)
    requested_results = num_results
    
    # Get Serper API key
    serper_api_key = os.getenv("SERPER_API_KEY")
    if not serper_api_key:
        raise ValueError("SERPER_API_KEY environment variable is not set.")
    
    serper_api_url = "https://google.serper.dev/search"
    
    # Build request body
    request_body = {
        "q": rephrased_query,
        "num": requested_results,
    }
    
    # Add date filtering if needed
    if cutoff_date:
        cd_max = f"{cutoff_date.month}/{cutoff_date.day}/{cutoff_date.year}"
        tbs_param = f"cdr:1,cd_max:{cd_max}"
        request_body["tbs"] = tbs_param
    
    try:
        client = await _get_http_client()
        response = await client.post(
            serper_api_url,
            json=request_body,
            headers={
                "X-API-KEY": serper_api_key,
                "Content-Type": "application/json"
            },
        )
    except httpx.RequestError as exc:
        logger.error(f"Error with rephrased query: {exc}")
        raise
    
    # If rephrased query also fails, don't retry again - just raise
    if response.status_code != 200:
        logger.error(f"Rephrased query also failed: {response.text}")
        raise Exception(f"Serper API returned status {response.status_code} for rephrased query: {response.text}")
    
    results_data = response.json()
    serper_results = results_data.get("organic", [])
    
    search_results = []
    for item in serper_results:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        url = item.get("link", "")
        
        # Extract date if available
        date_str = item.get("date", None)
        
        # Get trust rating (async call)
        trust_rating = await get_source_trust_rating(url, title)
        
        block = Block(
            title=title,
            text=snippet,
            url=url,
            published_date=date_str,
            trust_rating=trust_rating,
        )
        search_results.append(block)
    
    # Cache the rephrased results under the ORIGINAL cache key
    # so future calls with the same original query will use rephrased version
    try:
        cache[original_cache_key] = pickle.dumps(search_results)
    except Exception as e:
        logger.warning(f"Failed to cache rephrased results: {e}")
    
    return search_results[:num_results]


_CLIENT_TIMEOUT = httpx.Timeout(30.0)
_CLIENT_LIMITS = httpx.Limits(max_connections=50, max_keepalive_connections=20)
_loop_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = (
    weakref.WeakKeyDictionary()
)
_loop_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
    weakref.WeakKeyDictionary()
)


# Context variable to optionally constrain search results to a cutoff date (inclusive).
# When set (e.g., "2021-01-31"), only results with published_date/last_edit_date <= cutoff are returned.
search_before_date_var: ContextVar[str | None] = ContextVar("search_before_date", default=None)

def set_search_before_date(date_string: str | None) -> None:
    """Set an optional cutoff date (YYYY-MM-DD) for retrieval filtering.

    Callers may set this per-request before invoking analysis. Pass None to clear.
    """
    search_before_date_var.set(date_string)

def _parse_cutoff_date(date_string: str | None) -> datetime | None:
    """Parse a flexible date string into a datetime (00:00:00 of that day).

    Accepts multiple formats, e.g. '2018-11-01', '1 Nov 2018', 'Nov 1, 2018',
    '01/11/2018', '11/01/2018'. Returns None if parsing fails.
    """
    if not date_string:
        return None
    s = date_string.strip().replace(",", "").strip()
    candidates = [
        "%Y-%m-%d",  # 2018-11-01
        "%d-%m-%Y",  # 01-11-2018
        "%d/%m/%Y",  # 01/11/2018
        "%m/%d/%Y",  # 11/01/2018
        "%d %b %Y",  # 1 Nov 2018
        "%d %B %Y",  # 1 November 2018
        "%b %d %Y",  # Nov 1 2018
        "%B %d %Y",  # November 1 2018
    ]
    # Try title-case for month names to satisfy strptime
    s_title = s.title()
    for fmt in candidates:
        for candidate in (s, s_title):
            try:
                dt = datetime.strptime(candidate, fmt)
                # Normalize to date-only (midnight)
                return datetime(dt.year, dt.month, dt.day)
            except ValueError:
                continue
    return None


async def _get_http_client() -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    lock = _loop_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _loop_locks[loop] = lock

    async with lock:
        client = _loop_clients.get(loop)
        if client is None or getattr(client, "is_closed", False):
            _loop_clients[loop] = httpx.AsyncClient(
                timeout=_CLIENT_TIMEOUT,
                limits=_CLIENT_LIMITS,
            )
            client = _loop_clients[loop]
        return client


async def close_retriever_http_client() -> None:
    """Gracefully close the shared HTTP client for the current loop."""
    loop = asyncio.get_running_loop()
    lock = _loop_locks.get(loop)
    if lock is None:
        return

    async with lock:
        client = _loop_clients.pop(loop, None)
        if client is not None and not getattr(client, "is_closed", False):
            await client.aclose()


async def retrieve_via_api(
    query: str,
    retriever_endpoint: str | None = None,  # Kept for backward compatibility but not used
    num_results: int = 10,
) -> list[Block]:
    """Internal helper: retrieve search results from Serper.dev API for a single query.

    Args:
        query (str): The query string to be sent to Serper.
        retriever_endpoint (str | None): Unused, kept for backward compatibility.
        num_results (int): Number of blocks to retrieve.

    Returns:
    list[Block]: List of retrieved blocks for the query.

    Raises:
        Exception: If the API key is missing, rate limit is reached, or if there is an error with the Serper API request.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a str")

    # Check for date filter in context and include it in cache key
    cutoff_date_str = search_before_date_var.get()
    cutoff_date: datetime | None = _parse_cutoff_date(cutoff_date_str)

    # With server-side filtering via tbs parameter, we don't need to request extra results
    # The API will return only results within the date range
    requested_results = num_results
    
    # Create cache key that includes the date filter
    cache_key = pickle.dumps((
        "retrieve_via_api",
        query,
        num_results,  # Use original num_results for cache key
        cutoff_date_str,  # Include date filter in cache key
    ))
    
    # Check cache
    if cache_key in cache:
        try:
            cached_bytes = cache[cache_key]
            cached_results = pickle.loads(cached_bytes)
            # If we have enough results after filtering, return them
            if len(cached_results) >= num_results:
                return cached_results[:num_results]
            # Otherwise, we'll need to fetch more
        except Exception as e:
            logger.warning(f"Error loading from cache: {e}")
            del cache[cache_key]

    # Get Serper API key from environment
    serper_api_key = os.getenv("SERPER_API_KEY")
    if not serper_api_key:
        raise ValueError(
            "SERPER_API_KEY environment variable is not set. "
            "Please set it to your Serper API key from https://serper.dev/"
        )

    serper_api_url = "https://google.serper.dev/search"

    # Build request body with optional date filtering via tbs parameter
    request_body = {
        "q": query,
        "num": requested_results,
    }

    # Add server-side date filtering if cutoff_date is set
    if cutoff_date:
        # Format: M/D/YYYY for Google's tbs parameter
        # Only set maximum date (before the cutoff date)
        cd_max = f"{cutoff_date.month}/{cutoff_date.day}/{cutoff_date.year}"
        tbs_param = f"cdr:1,cd_max:{cd_max}"
        request_body["tbs"] = tbs_param
        logger.info(f"Server-side date filter active: tbs={tbs_param} (results before {cutoff_date.date()})")

    try:
        client = await _get_http_client()
        response = await client.post(
            serper_api_url,
            json=request_body,
            headers={
                "X-API-KEY": serper_api_key,
                "Content-Type": "application/json"
            },
        )
    except httpx.RequestError as exc:
        logger.error(
            "Error encountered when sending this request to Serper API error={}",
            exc,
        )
        raise

    if response.status_code == 429:
        raise Exception(
            "You have reached your rate limit for the Serper API. Please wait and try later."
        )
    if response.status_code == 401:
        raise Exception(
            "Invalid Serper API key. Please check your SERPER_API_KEY environment variable."
        )
    if response.status_code == 400:
        # Check if this is a "Query not allowed" error (sensitive content)
        try:
            error_data = response.json()
            if "Query not allowed" in error_data.get("message", ""):
                logger.warning(f"Query blocked by content filter. Attempting to rephrase: {query}")
                # Attempt to rephrase the query using LLM
                rephrased_query = await _rephrase_sensitive_query(query)
                if rephrased_query and rephrased_query != query:
                    logger.info(f"Retrying with rephrased query: {rephrased_query}")
                    # Retry with rephrased query (recursive call, but only once)
                    return await _retrieve_with_rephrased_query(
                        rephrased_query, 
                        num_results, 
                        cutoff_date_str,
                        cache_key
                    )
        except Exception as e:
            logger.error(f"Error handling blocked query: {e}")
        
        # If rephrasing failed or wasn't applicable, raise the original error
        logger.error(f"Error encountered when sending this request to Serper API: {response.text}")
        raise Exception(f"Serper API returned status {response.status_code}: {response.text}")
    if response.status_code != 200:
        logger.error(f"Error encountered when sending this request to Serper API: {response.text}")
        raise Exception(f"Serper API returned status {response.status_code}: {response.text}")

    results_data = response.json()

    # Serper response structure: {"organic": [...], "knowledgeGraph": {...}, ...}
    serper_results = results_data.get("organic", [])
    
    # Debug: Log first result structure to see what fields Serper returns
    if serper_results:
        logger.debug(f"Sample Serper result fields: {list(serper_results[0].keys())}")
        if "date" in serper_results[0]:
            logger.debug(f"Sample date value: {serper_results[0].get('date')}")

    search_results = []

    # Track statistics for logging
    results_with_dates = 0
    results_without_dates = 0
    results_after_cutoff = 0

    # Server-side filtering via tbs should handle most filtering
    # We still parse dates for metadata and as a sanity check
    
    for result in serper_results:
        # Parse date if available (Serper may include this)
        # Serper typically includes 'date' field in search results
        last_edit_date = None
        date_str = None
        
        # Try different field names that Serper might use
        for field_name in ["date", "published", "publishedDate", "datePublished"]:
            if field_name in result and result[field_name]:
                date_str = str(result[field_name])
                break
        
        if date_str:
            try:
                date_str_lower = date_str.lower().strip()
                
                # Handle relative dates like "11 hours ago", "3 days ago", "2 weeks ago"
                now = datetime.now()
                
                if "hour" in date_str_lower or "hours ago" in date_str_lower:
                    # Extract number of hours
                    hours_match = re.search(r'(\d+)\s*hours?', date_str_lower)
                    if hours_match:
                        hours = int(hours_match.group(1))
                        last_edit_date = now - timedelta(hours=hours)
                    else:
                        # If we can't parse, assume it's very recent (today)
                        last_edit_date = now
                elif "day" in date_str_lower and "ago" in date_str_lower:
                    # Extract number of days
                    days_match = re.search(r'(\d+)\s*days?', date_str_lower)
                    if days_match:
                        days = int(days_match.group(1))
                        last_edit_date = now - timedelta(days=days)
                    else:
                        # "yesterday" or "today"
                        if "yesterday" in date_str_lower:
                            last_edit_date = now - timedelta(days=1)
                        else:
                            last_edit_date = now
                elif "week" in date_str_lower and "ago" in date_str_lower:
                    weeks_match = re.search(r'(\d+)\s*weeks?', date_str_lower)
                    if weeks_match:
                        weeks = int(weeks_match.group(1))
                        last_edit_date = now - timedelta(weeks=weeks)
                    else:
                        last_edit_date = now - timedelta(weeks=1)
                elif "month" in date_str_lower and "ago" in date_str_lower:
                    months_match = re.search(r'(\d+)\s*months?', date_str_lower)
                    if months_match:
                        months = int(months_match.group(1))
                        # Approximate months as 30 days
                        last_edit_date = now - timedelta(days=months * 30)
                    else:
                        last_edit_date = now - timedelta(days=30)
                elif "year" in date_str_lower and "ago" in date_str_lower:
                    years_match = re.search(r'(\d+)\s*years?', date_str_lower)
                    if years_match:
                        years = int(years_match.group(1))
                        last_edit_date = now - timedelta(days=years * 365)
                    else:
                        last_edit_date = now - timedelta(days=365)
                else:
                    # Try parsing as absolute date using the existing parser
                    # Handle formats like "Nov 10, 2024", "2018-11-01", etc.
                    last_edit_date = _parse_cutoff_date(date_str)
                    
                    if last_edit_date is None:
                        # Try one more time with common Serper formats
                        # Serper often returns "Nov 10, 2024" format
                        try:
                            # Remove commas and try parsing
                            cleaned = date_str.replace(",", "").strip()
                            # Try "Nov 10 2024" format
                            for fmt in ["%b %d %Y", "%B %d %Y", "%d %b %Y", "%d %B %Y"]:
                                try:
                                    last_edit_date = datetime.strptime(cleaned, fmt)
                                    break
                                except ValueError:
                                    continue
                        except Exception:
                            pass
                
                if last_edit_date is None:
                    logger.debug(f"Could not parse date from Serper result: {date_str}")
                else:
                    # Normalize to date-only (midnight) for comparison
                    last_edit_date = datetime(last_edit_date.year, last_edit_date.month, last_edit_date.day)
                    
            except Exception as e:
                logger.debug(f"Error parsing date from Serper result: {date_str}, error: {e}")
                last_edit_date = None

        # Track date statistics
        if last_edit_date is not None:
            results_with_dates += 1
            if cutoff_date and last_edit_date > cutoff_date:
                results_after_cutoff += 1
        else:
            results_without_dates += 1
        
        # Map Serper result to Block format
        # Serper organic results have: title, link, snippet, position, etc.
        block = Block(
            document_title=result.get("title", "Untitled"),
            section_title="",  # Serper doesn't provide section titles
            content=result.get("snippet", ""),  # Serper uses 'snippet' instead of 'content'
            last_edit_date=last_edit_date,
            url=result.get("link", ""),  # Serper uses 'link' instead of 'url'
        )
        
        # Apply client-side validation if cutoff_date is set (as sanity check)
        # Server-side filtering via tbs should already handle this, but we double-check
        if cutoff_date is not None:
            if last_edit_date is not None and last_edit_date > cutoff_date:
                # Date is after cutoff - this shouldn't happen with tbs, but exclude as sanity check
                logger.warning(f"Server returned result after cutoff: '{block.document_title}' - date {last_edit_date.date()} > {cutoff_date.date()}")
                continue
            elif last_edit_date is not None:
                # Date is on/before cutoff - include it
                logger.debug(f"Including result '{block.document_title}' - date {last_edit_date.date()}")
                search_results.append(block)
            else:
                # No date available - EXCLUDE undated sources (strict date filtering)
                # Undated sources are excluded when date filter is active to prevent evidence leakage
                logger.debug(f"Excluding undated result '{block.document_title}' (strict date filtering excludes undated sources)")
                continue
        else:
            # No date filter - include all results
            search_results.append(block)
    
    # Log detailed statistics
    if cutoff_date is not None:
        logger.info(
            f"Server-side date filter: Retrieved {len(search_results)} results from Serper (requested {num_results}). "
            f"({results_with_dates} had dates and passed, {results_without_dates} undated excluded, {results_after_cutoff} failed validation)"
        )
    else:
        logger.debug(f"Retrieved {len(search_results)} results (no date filter)")

    # Note: When date filter is active, we exclude undated sources (strict date filtering)
    # Dated sources are filtered to be on/before cutoff_date, undated sources are excluded

    # Limit to requested num_results (though server should already do this)
    if len(search_results) > num_results:
        search_results = search_results[:num_results]
    
    # Compute trust ratings for all blocks in parallel
    if search_results:
        logger.debug(f"Computing trust ratings for {len(search_results)} sources...")
        try:
            # Create tasks for parallel trust rating computation
            trust_tasks = [
                get_source_trust_rating(block.url or "", block.document_title)
                for block in search_results
            ]
            trust_ratings = await asyncio.gather(*trust_tasks, return_exceptions=True)
            
            # Update blocks with trust ratings
            for block, trust_info in zip(search_results, trust_ratings):
                if isinstance(trust_info, Exception):
                    logger.warning(f"Error computing trust rating for {block.url}: {trust_info}")
                    # Default to mixed if computation fails
                    block.trust_rating = "mixed"
                    block.trust_source = "llm"
                    block.trust_reason = "Error computing trust rating"
                    block.trust_confidence = 0.3  # Low confidence for errors
                elif isinstance(trust_info, dict):
                    block.trust_rating = trust_info.get("rating", "mixed")
                    block.trust_source = trust_info.get("source", "list")
                    block.trust_reason = trust_info.get("reason", "")
                    block.trust_confidence = trust_info.get("confidence", 0.75)
                else:
                    # Fallback for unexpected format
                    block.trust_rating = "mixed"
                    block.trust_source = "list"
                    block.trust_confidence = 0.5
            
            logger.debug(f"Successfully computed trust ratings for {len(search_results)} sources")
        except Exception as e:
            logger.warning(f"Error computing trust ratings: {e}, continuing without trust ratings")
            # Continue without trust ratings if computation fails
    
    # Cache the results (including the date filter in the cache key)
    try:
        cache[cache_key] = pickle.dumps(search_results)
    except Exception as e:
        logger.warning(f"Error caching results: {e}")

    return search_results
