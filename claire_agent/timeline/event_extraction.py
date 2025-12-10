"""LLM-powered event extraction helpers for timeline analysis."""

from typing import List, Dict, Any
from langchain.chat_models import init_chat_model
from utils.logger import logger

from .models import (
    ExtractedEvent,
    EventExtraction,
    KeyEvent,
    FilteredTimeline,
    TimelineSummary
)


def format_search_results(all_results: List[List[Any]]) -> str:
    """Format search results for LLM consumption."""
    formatted = []
    result_num = 1

    for results_batch in all_results:
        for result in results_batch:
            title = result.document_title if hasattr(result, 'document_title') else str(result)
            preview = result.preview if hasattr(result, 'preview') else ''
            url = result.url if hasattr(result, 'url') else ''

            formatted.append(
                f"[{result_num}] {title}\n"
                f"    URL: {url}\n"
                f"    Preview: {preview[:200]}..."
            )
            result_num += 1

            # Limit to first 40 results for token efficiency
            if result_num > 40:
                break

        if result_num > 40:
            break

    return "\n\n".join(formatted)


def format_events(events: List[ExtractedEvent]) -> str:
    """Format extracted events for filtering prompt."""
    formatted = []
    for i, event in enumerate(events, 1):
        formatted.append(
            f"{i}. Date: {event.date}\n"
            f"   Type: {event.event_type}\n"
            f"   Description: {event.description}\n"
            f"   Source: {event.source_title}\n"
            f"   Confidence: {event.confidence:.2f}"
        )
    return "\n\n".join(formatted)


def format_sources(sources: List[Dict[str, Any]]) -> str:
    """Format sources for explanation prompt."""
    if not sources:
        return "No sources available"

    formatted = []
    for i, source in enumerate(sources, 1):
        title = source.get('title', 'Unknown')
        url = source.get('url', '')
        trust = source.get('trust_rating', 'mixed')

        formatted.append(f"[{i}] {title} ({trust}) - {url}")

    return "\n".join(formatted)


def format_component_evals(component_evaluations: List[Any]) -> str:
    """Format component evaluations for explanation prompt."""
    if not component_evaluations:
        return "No component evaluations available"

    formatted = []
    for i, eval in enumerate(component_evaluations, 1):
        verdict = eval.verdict if hasattr(eval, 'verdict') else 'unknown'
        reasoning = eval.reasoning if hasattr(eval, 'reasoning') else 'No reasoning provided'

        formatted.append(f"{i}. {verdict.upper()}: {reasoning[:150]}...")

    return "\n".join(formatted)


def format_timeline_events_with_explanations(timeline_events: List[Dict[str, Any]]) -> str:
    """Format timeline events with explanations for summary prompt."""
    formatted = []
    for event in timeline_events:
        formatted.append(
            f"📅 {event['date']} | Verdict: {event['verdict'].upper()}\n"
            f"   Event: {event['event_description']}\n"
            f"   Evidence: {event['num_sources']} sources ({event['reliable_count']} reliable)\n"
            f"   Explanation: {event.get('explanation', 'No explanation')}"
        )

    return "\n\n".join(formatted)


async def extract_events_from_search_results(
    claim: str,
    start_date: str,
    end_date: str,
    all_results: List[List[Any]]
) -> EventExtraction:
    """
    Step 2: Extract timeline events from search results using LLM.

    Args:
        claim: The claim being analyzed
        start_date: Start of time range
        end_date: End of time range
        all_results: List of search result batches

    Returns:
        EventExtraction with list of extracted events
    """
    logger.info("Step 2: Extracting events from search results...")

    extraction_prompt = f"""You are analyzing search results to extract timeline events for this claim:

Claim: "{claim}"
Time range: {start_date} to {end_date}

Search results:
{format_search_results(all_results)}

Extract ALL events that might affect the claim's verification status:
- When the claim first appeared or was first made
- Fact-check publications (Snopes, PolitiFact, FactCheck.org, etc.)
- New studies, research, or data releases
- Official updates or corrections to numbers/facts
- Retractions, clarifications, or walk-backs
- Major news events related to the claim

For each event:
1. Extract the date as precisely as possible (YYYY-MM-DD format preferred, but "August 2023" is ok if that's all that's mentioned)
2. Classify the event type: "claim_origin", "fact_check", "new_evidence", "official_update", or "retraction"
3. Write a brief but clear description (1 sentence)
4. Note which source mentioned it (use the source title from above)
5. Provide confidence (0.0-1.0) based on how clearly the date was stated

Rules:
- Only include events within the time range {start_date} to {end_date}
- If a date is mentioned as "early 2023" or "mid-June", make your best estimate
- Include the event even if you're not 100% certain of the exact date (but mark confidence lower)
- Focus on events that could change whether the claim is true/false
- Return events in chronological order (earliest first)

Return a structured list of events."""

    model = init_chat_model(
        model="gemini-2.5-flash-lite",
        model_provider="google_genai"
    )

    extraction_model = model.with_structured_output(EventExtraction)

    extracted = await extraction_model.ainvoke([
        {"role": "system", "content": "You are an expert at extracting timeline events from search results. Be thorough but precise."},
        {"role": "user", "content": extraction_prompt}
    ])

    # Handle case where LLM returns dict instead of dataclass
    if isinstance(extracted, dict):
        logger.info(f"LLM returned dict, converting to EventExtraction")
        events_list = []
        for event_dict in extracted.get('events', []):
            events_list.append(ExtractedEvent(**event_dict))
        extracted = EventExtraction(events=events_list)

    logger.info(f"Extracted {len(extracted.events)} events from search results")

    return extracted


async def filter_and_prioritize_events(
    claim: str,
    start_date: str,
    end_date: str,
    extracted_events: EventExtraction
) -> FilteredTimeline:
    """
    Step 3: Filter, deduplicate, and prioritize events using LLM.

    Args:
        claim: The claim being analyzed
        start_date: Start of time range
        end_date: End of time range
        extracted_events: Raw extracted events from Step 2

    Returns:
        FilteredTimeline with 4-6 key events
    """
    logger.info("Step 3: Filtering and prioritizing events...")

    filtering_prompt = f"""You extracted {len(extracted_events.events)} events for this claim. Now filter and prioritize them AGGRESSIVELY.

Original claim: "{claim}"
Time range: {start_date} to {end_date}

Extracted events:
{format_events(extracted_events.events)}

CRITICAL RULES:
1. **ONE EVENT PER DAY (if consistent)**: If multiple events happened on the same date AND they represent the same verdict/status, pick the most significant one and discard the rest. However, if events on the same date represent DIFFERENT verdicts or show the claim's status changing within that day, you MAY include multiple events for that date.
2. **SPACING REQUIREMENT**: Events should be spread out - prefer events at least 5-7 days apart unless the verdict clearly changed.
3. **2-4 EVENTS**: Return between 2-4 events. Quality over quantity - fewer meaningful events is better than many redundant ones.
4. **Remove duplicates**: If multiple events describe similar occurrences with the same verdict, keep only the most significant one.

Your tasks:
1. **Smart deduplication**:
   - If you see multiple events on the same date with the SAME verdict/status, KEEP ONLY ONE (the most significant)
   - If events on the same date show DIFFERENT verdicts or a change in status, you may include multiple events for that date
   - If you see similar events on nearby dates, combine them or keep the most significant
2. **Normalize dates**: Convert all dates to YYYY-MM-DD format
   - "August 2023" → "2023-08-15" (middle of month)
   - "early 2023" → "2023-01-15"
   - "mid-June 2023" → "2023-06-15"
3. **Select 3-4 KEY events that show:**
   - The claim's origin or first appearance
   - Major verdict-changing moments (e.g., official announcement, major fact-check)
   - Final status (if different from origin)

Priority (select from these ONLY):
- Priority 1 (MUST include): Events where verdict definitely changed
- Priority 2: Claim origin OR final confirmation
- Priority 3: Major fact-checks from authoritative sources

REJECT:
- Multiple events on the same date with the SAME verdict (keep only one)
- Minor updates or rumors
- Events too close together (less than 5 days apart) unless verdict changed
- Similar events that don't show meaningful change

Return EXACTLY 3-4 events in chronological order.
Explain your reasoning for why you selected these specific events and why you rejected others."""

    model = init_chat_model(
        model="gemini-2.5-flash-lite",
        model_provider="google_genai"
    )

    filtering_model = model.with_structured_output(FilteredTimeline)

    filtered = await filtering_model.ainvoke([
        {"role": "system", "content": "You are an expert at identifying the most important events in a timeline. Be selective and focus on verdict-changing events."},
        {"role": "user", "content": filtering_prompt}
    ])

    # Handle case where LLM returns dict instead of dataclass
    if isinstance(filtered, dict):
        logger.info(f"LLM returned dict, converting to FilteredTimeline")
        events_list = []
        for event_dict in filtered.get('key_events', []):
            events_list.append(KeyEvent(**event_dict))
        filtered = FilteredTimeline(
            key_events=events_list,
            reasoning=filtered.get('reasoning', '')
        )

    logger.info(f"Filtered down to {len(filtered.key_events)} key events")
    logger.info(f"Filtering reasoning: {filtered.reasoning}")

    return filtered


async def generate_event_explanation(
    claim: str,
    event: Dict[str, Any],
    all_timeline_events: List[Dict[str, Any]]
) -> str:
    """
    Step 5: Generate contextual explanation for a single event.

    Args:
        claim: The claim being analyzed
        event: The specific event to explain
        all_timeline_events: All events for context

    Returns:
        Explanation string (2-3 sentences)
    """
    explanation_prompt = f"""Generate a concise explanation (2-3 sentences) for this timeline event.

Claim: "{claim}"
Date: {event['date']}
Event: {event['event_description']}
Verdict at this time: {event['verdict']}

Evidence available:
- {event['num_sources']} sources total
- {event['reliable_count']} from reliable sources
- {event.get('mixed_count', 0)} from mixed-reliability sources

Top sources:
{format_sources(event.get('sources', []))}

Component evaluations:
{format_component_evals(event.get('component_evaluations', []))}

Context - Other events in timeline:
{format_timeline_events_with_explanations([e for e in all_timeline_events if e['date'] != event['date']])}

Your explanation should:
1. Explain WHY the verdict was "{event['verdict']}" at this specific time
2. Describe what evidence was (or wasn't) available at this point
3. Connect this event to the claim's evolution (how it fits in the bigger picture)

Write in past tense. Be factual and concise. Focus on the evidence and reasoning, not speculation.
Example good explanation: "At this time, official records confirmed 83 deaths, making the claim accurate based on available information. However, search efforts were still ongoing, and officials noted the toll might rise as more victims were identified."

Example bad explanation: "The claim was true because sources said so."

Return only the explanation text (2-3 sentences), nothing else."""

    model = init_chat_model(
        model="gemini-2.5-flash-lite",
        model_provider="google_genai"
    )

    response = await model.ainvoke([
        {"role": "system", "content": "You are an expert at explaining fact-checking verdicts with proper context. Be clear and concise."},
        {"role": "user", "content": explanation_prompt}
    ])

    return response.content.strip()


async def generate_timeline_summary(
    claim: str,
    start_date: str,
    end_date: str,
    timeline_events: List[Dict[str, Any]]
) -> TimelineSummary:
    """
    Step 6: Generate overall narrative summary.

    Args:
        claim: The claim being analyzed
        start_date: Start of time range
        end_date: End of time range
        timeline_events: All events with explanations

    Returns:
        TimelineSummary with category, summary, status, and insight
    """
    logger.info("Step 6: Generating narrative summary...")

    summary_prompt = f"""Synthesize this timeline into a coherent narrative.

Claim: "{claim}"
Time range: {start_date} to {end_date}

Timeline events with explanations:
{format_timeline_events_with_explanations(timeline_events)}

Your tasks:

1. **Categorize this timeline** (choose ONE):
   - "evolving_fact": Numbers/facts changed as a situation developed (not misinformation, just updates)
   - "debunking_journey": A false claim that got fact-checked and debunked over time
   - "stable_truth": Claim remained consistent/true throughout the period
   - "stable_falsehood": Claim remained inconsistent/false throughout the period
   - "narrative_mutation": The claim evolved or changed its wording to evade fact-checking

2. **Write an overall summary** (3-4 sentences):
   - Describe how the claim's verification status evolved over time
   - Explain what drove any changes (new evidence, corrections, fact-checks, etc.)
   - Clarify whether this represents misinformation, legitimate factual evolution, or something else

3. **Current status** (as of {end_date}) (2 sentences):
   - What's the final verdict?
   - Brief explanation of why

4. **Key insight** (one clear sentence):
   - The most important takeaway someone should understand from this timeline
   - Make it actionable or educational

Be balanced and factual. Distinguish between:
- False claims (misinformation)
- Outdated but once-accurate claims (evolving facts)
- Claims that were always true/false (stable)

Example good summary: "The claim that '83 people died' was accurate when first reported on August 10, 2023, based on official county records. As search and recovery efforts continued over the following weeks, the death toll was revised upward multiple times, reaching a final count of 115 by September 1. This represents an evolving fact rather than misinformation—the original number was correct but incomplete as the situation developed."

Return structured output."""

    model = init_chat_model(
        model="gemini-2.5-flash-lite",
        model_provider="google_genai"
    )

    summary_model = model.with_structured_output(TimelineSummary)

    summary = await summary_model.ainvoke([
        {"role": "system", "content": "You are an expert at synthesizing fact-checking timelines into clear narratives. Be balanced and educational."},
        {"role": "user", "content": summary_prompt}
    ])

    # Handle case where LLM returns dict instead of dataclass
    if isinstance(summary, dict):
        logger.info(f"LLM returned dict, converting to TimelineSummary")
        summary = TimelineSummary(**summary)

    logger.info(f"Generated summary with category: {summary.category}")

    return summary
