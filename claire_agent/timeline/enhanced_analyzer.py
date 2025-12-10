"""Enhanced event-driven timeline analyzer using LLM-powered event extraction."""

import asyncio
from datetime import datetime
from typing import Any, List, Dict
from utils.logger import logger

from retrieval.retriever_client import retrieve_via_api, set_search_before_date
from .models import TimelineEvent, TimelineNarrative
from .event_extraction import (
    extract_events_from_search_results,
    filter_and_prioritize_events,
    generate_event_explanation,
    generate_timeline_summary
)


async def build_enhanced_timeline(
    claim: str,
    start_date: str,
    end_date: str,
    agent: Any  # InconsistencyAgent instance
) -> TimelineNarrative:
    """
    Build an event-driven timeline narrative using LLM-powered analysis.

    This is a sequential pipeline (NOT a ReAct agent):
    1. Event Discovery: Execute targeted searches
    2. Event Extraction: LLM extracts events from results
    3. Event Filtering: LLM prioritizes 4-6 key events
    4. Verdict Verification: Run main pipeline at each key date
    5. Event Explanation: LLM generates contextual explanations
    6. Narrative Summary: LLM synthesizes overall story

    Args:
        claim: The claim to analyze
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        agent: The InconsistencyAgent instance (main pipeline)

    Returns:
        TimelineNarrative with events, explanations, and summary
    """
    logger.info(f"Building enhanced timeline for: {claim}")
    logger.info(f"Time range: {start_date} to {end_date}")

    total_searches = 0
    total_pipeline_runs = 0
    start_time = datetime.now()

    try:
        # ============================================
        # STEP 1: Event Discovery (Search Phase)
        # ============================================
        logger.info("Step 1: Discovering events via targeted searches...")

        # Define 6 targeted search queries
        search_queries = [
            f'"{claim}" first claimed OR originated OR first reported OR first appeared',
            f'"{claim}" fact-check Snopes PolitiFact FactCheck.org verified debunked',
            f'"{claim}" study published OR research released OR data shows OR evidence',
            f'"{claim}" official update OR confirmed OR revised OR corrected numbers',
            f'"{claim}" debunked OR false OR incorrect OR retracted OR walked back',
            f'"{claim}" timeline events history changes updates between {start_date} {end_date}'
        ]

        # Log the queries being used
        logger.info(f"Executing {len(search_queries)} targeted searches in parallel...")
        for i, query in enumerate(search_queries, 1):
            logger.debug(f"  Query {i}: {query}")

        # Execute all searches in parallel for speed
        search_results = await asyncio.gather(*[
            retrieve_via_api(query, num_results=10) for query in search_queries
        ])
        total_searches = len(search_queries)

        # Count total results retrieved
        total_results = sum(len(results) for results in search_results)
        logger.info(f"Retrieved {total_results} total search results from {len(search_queries)} queries")
        
        # Log per-query results for debugging
        for i, (query, results) in enumerate(zip(search_queries, search_results), 1):
            logger.debug(f"  Query {i} returned {len(results)} results: {query[:80]}...")

        # If all targeted searches return 0 results, try a fallback simpler search
        if total_results == 0:
            logger.warning("All targeted searches returned 0 results. Trying fallback simpler search...")
            fallback_query = claim  # Just search for the claim itself
            logger.info(f"Fallback query: {fallback_query}")
            fallback_results = await retrieve_via_api(fallback_query, num_results=20)
            if fallback_results:
                logger.info(f"Fallback search found {len(fallback_results)} results")
                search_results = [fallback_results]  # Wrap in list to match expected format
                total_results = len(fallback_results)
                total_searches += 1
            else:
                logger.warning("Fallback search also returned 0 results")

        # ============================================
        # STEP 2: Event Extraction (LLM Phase 1)
        # ============================================
        try:
            extracted_events = await extract_events_from_search_results(
                claim=claim,
                start_date=start_date,
                end_date=end_date,
                all_results=search_results
            )

            # Handle case where LLM returns dict instead of EventExtraction object
            logger.info(f"Extracted events type: {type(extracted_events)}")
            if isinstance(extracted_events, dict):
                logger.info(f"Converting dict to EventExtraction. Dict keys: {extracted_events.keys()}")
                from .models import EventExtraction, ExtractedEvent
                events_list = []
                for event_dict in extracted_events.get('events', []):
                    events_list.append(ExtractedEvent(**event_dict))
                extracted_events = EventExtraction(events=events_list)
        except Exception as e:
            logger.error(f"Error in event extraction: {e}", exc_info=True)
            raise

        if not extracted_events.events:
            logger.warning("No events extracted from search results")
            # Return empty timeline
            return TimelineNarrative(
                claim=claim,
                time_range=(start_date, end_date),
                category="insufficient_data",
                events=[],
                summary="Insufficient data available to construct a timeline for this claim.",
                key_insight="No significant events or evidence found in the specified time range.",
                current_status="Unable to determine due to lack of evidence",
                metadata={
                    'total_searches': total_searches,
                    'total_results': total_results,
                    'events_extracted': 0,
                    'events_verified': 0,
                    'pipeline_runs': 0,
                    'duration_seconds': (datetime.now() - start_time).total_seconds()
                }
            )

        # ============================================
        # STEP 3: Event Filtering (LLM Phase 2)
        # ============================================
        filtered_timeline = await filter_and_prioritize_events(
            claim=claim,
            start_date=start_date,
            end_date=end_date,
            extracted_events=extracted_events
        )

        # Handle case where LLM returns dict instead of FilteredTimeline object
        if isinstance(filtered_timeline, dict):
            from .models import FilteredTimeline, KeyEvent
            events_list = []
            for event_dict in filtered_timeline.get('key_events', []):
                events_list.append(KeyEvent(**event_dict))
            filtered_timeline = FilteredTimeline(
                key_events=events_list,
                reasoning=filtered_timeline.get('reasoning', '')
            )

        if not filtered_timeline.key_events:
            logger.warning("No key events after filtering")
            # Return minimal timeline
            return TimelineNarrative(
                claim=claim,
                time_range=(start_date, end_date),
                category="insufficient_data",
                events=[],
                summary="No significant events identified that would affect the claim's verification status.",
                key_insight="The claim's truth value likely remained stable throughout this period.",
                current_status="Unable to determine due to lack of key events",
                metadata={
                    'total_searches': total_searches,
                    'total_results': total_results,
                    'events_extracted': len(extracted_events.events),
                    'events_verified': 0,
                    'pipeline_runs': 0,
                    'duration_seconds': (datetime.now() - start_time).total_seconds()
                }
            )

        logger.info(f"Proceeding with {len(filtered_timeline.key_events)} key events")

        # ============================================
        # STEP 4: Verdict Verification (Main Pipeline) - PARALLEL
        # ============================================
        logger.info("Step 4: Verifying verdicts at key event dates IN PARALLEL...")

        async def verify_single_event(key_event, index):
            """Verify a single event date (runs in parallel with others)."""
            logger.info(f"Verifying event {index}/{len(filtered_timeline.key_events)}: {key_event.date} - {key_event.description}")

            # Set temporal filter for this date
            set_search_before_date(key_event.date)

            try:
                # Call main pipeline (no modifications!)
                # This runs the full 5-stage pipeline
                reports = await agent.analyze_passage_for_inconsistencies(
                    passage=claim
                )

                if not reports or len(reports) == 0:
                    logger.warning(f"No report generated for {key_event.date}")
                    return None

                report = reports[0]  # Single claim

                # Extract verdict and evidence
                verdict = report.verdict if hasattr(report, 'verdict') else 'unknown'
                if isinstance(verdict, str):
                    verdict = verdict.lower()

                # Get sources from report
                sources = report.search_results if hasattr(report, 'search_results') else []

                # Count by trust rating
                reliable_count = sum(1 for s in sources if getattr(s, 'trust_rating', 'mixed') == 'reliable')
                mixed_count = sum(1 for s in sources if getattr(s, 'trust_rating', 'mixed') == 'mixed')
                unreliable_count = sum(1 for s in sources if getattr(s, 'trust_rating', 'mixed') == 'unreliable')

                # Format top 3 sources for display
                key_sources = []
                for source in sources[:3]:
                    key_sources.append({
                        'title': source.document_title if hasattr(source, 'document_title') else 'Unknown',
                        'url': source.url if hasattr(source, 'url') else '',
                        'trust_rating': source.trust_rating if hasattr(source, 'trust_rating') else 'mixed',
                        'preview': source.preview if hasattr(source, 'preview') else ''
                    })

                # Create timeline event
                timeline_event = TimelineEvent(
                    date=key_event.date,
                    event_description=key_event.description,
                    event_type=key_event.event_type,
                    verdict=verdict,
                    num_sources=len(sources),
                    reliable_count=reliable_count,
                    mixed_count=mixed_count,
                    unreliable_count=unreliable_count,
                    key_sources=key_sources,
                    component_evaluations=report.component_evaluations if hasattr(report, 'component_evaluations') else [],
                    explanation=None  # Will generate in next step
                )

                logger.info(f"  → Verdict: {verdict} ({len(sources)} sources, {reliable_count} reliable)")
                return timeline_event

            except Exception as e:
                logger.error(f"Error analyzing at {key_event.date}: {e}", exc_info=True)
                return None

            finally:
                # Always clear temporal filter
                set_search_before_date(None)

        # Run all event verifications in parallel
        verification_tasks = [
            verify_single_event(key_event, i+1)
            for i, key_event in enumerate(filtered_timeline.key_events)
        ]

        timeline_events_raw = await asyncio.gather(*verification_tasks)
        total_pipeline_runs = len(filtered_timeline.key_events)

        # Filter out None results (failed verifications)
        timeline_events = [event for event in timeline_events_raw if event is not None]

        if not timeline_events:
            logger.error("No timeline events successfully verified")
            return TimelineNarrative(
                claim=claim,
                time_range=(start_date, end_date),
                category="analysis_failed",
                events=[],
                summary="Timeline analysis failed to verify events.",
                key_insight="Unable to complete analysis due to errors.",
                current_status="Analysis incomplete",
                metadata={
                    'total_searches': total_searches,
                    'total_results': total_results,
                    'events_extracted': len(extracted_events.events),
                    'events_verified': 0,
                    'pipeline_runs': total_pipeline_runs,
                    'duration_seconds': (datetime.now() - start_time).total_seconds()
                }
            )

        logger.info(f"Successfully verified {len(timeline_events)} events")

        # Deduplicate by date - but only if verdict is consistent
        # If multiple events on same date have different verdicts, keep them all
        # If multiple events on same date have same verdict, keep only the first one
        date_verdicts_seen = {}  # date -> set of verdicts we've already kept for this date
        deduped_events = []
        for event in timeline_events:
            if event.date not in date_verdicts_seen:
                # First event for this date - always keep it
                date_verdicts_seen[event.date] = {event.verdict}
                deduped_events.append(event)
            else:
                # We've seen this date before - check if verdict is different
                if event.verdict not in date_verdicts_seen[event.date]:
                    # Different verdict on same date - keep it (event changed within the day)
                    date_verdicts_seen[event.date].add(event.verdict)
                    deduped_events.append(event)
                    logger.info(f"Keeping multiple events on {event.date} due to verdict change: {event.verdict} (previously seen: {date_verdicts_seen[event.date] - {event.verdict}})")
                else:
                    # Same verdict on same date - skip this duplicate
                    logger.info(f"Removing duplicate event on {event.date} with same verdict ({event.verdict}): {event.event_description}")

        timeline_events = deduped_events
        logger.info(f"After deduplication: {len(timeline_events)} events (kept multiple events per day only when verdict changed)")

        # Sort events chronologically by date
        timeline_events.sort(key=lambda e: e.date)
        logger.info("Events sorted chronologically")

        # ============================================
        # STEP 5: Event Explanation (LLM Phase 3)
        # ============================================
        logger.info("Step 5: Generating event explanations...")

        # Convert timeline_events to dicts for formatting
        timeline_events_dicts = []
        for event in timeline_events:
            event_dict = event.to_dict()
            event_dict['sources'] = event.key_sources  # Include sources
            event_dict['component_evaluations'] = event.component_evaluations
            timeline_events_dicts.append(event_dict)

        # Generate explanation for each event
        for i, (event, event_dict) in enumerate(zip(timeline_events, timeline_events_dicts), 1):
            logger.info(f"Generating explanation {i}/{len(timeline_events)} for {event.date}...")

            try:
                explanation = await generate_event_explanation(
                    claim=claim,
                    event=event_dict,
                    all_timeline_events=timeline_events_dicts
                )
                event.explanation = explanation
                logger.info(f"  → Generated: {explanation[:100]}...")
            except Exception as e:
                logger.error(f"Error generating explanation for {event.date}: {e}")
                event.explanation = f"At this time, the claim was {event.verdict} based on {event.num_sources} sources."

        # ============================================
        # STEP 6: Narrative Summary (LLM Phase 4)
        # ============================================
        # Update dicts with explanations
        for event, event_dict in zip(timeline_events, timeline_events_dicts):
            event_dict['explanation'] = event.explanation

        summary = await generate_timeline_summary(
            claim=claim,
            start_date=start_date,
            end_date=end_date,
            timeline_events=timeline_events_dicts
        )

        # Handle case where LLM returns dict instead of TimelineSummary object
        if isinstance(summary, dict):
            from .models import TimelineSummary
            summary = TimelineSummary(**summary)

        # ============================================
        # Build Final Result
        # ============================================
        duration = (datetime.now() - start_time).total_seconds()

        result = TimelineNarrative(
            claim=claim,
            time_range=(start_date, end_date),
            category=summary.category,
            events=timeline_events,
            summary=summary.overall_summary,
            key_insight=summary.key_insight,
            current_status=summary.current_status,
            metadata={
                'total_searches': total_searches,
                'total_results': total_results,
                'events_extracted': len(extracted_events.events),
                'events_filtered': len(filtered_timeline.key_events),
                'events_verified': len(timeline_events),
                'pipeline_runs': total_pipeline_runs,
                'duration_seconds': duration,
                'filtering_reasoning': filtered_timeline.reasoning
            }
        )

        logger.info(f"Timeline complete! Category: {summary.category}, {len(timeline_events)} events, {duration:.1f}s")

        return result

    except Exception as e:
        logger.error(f"Enhanced timeline analysis failed: {e}", exc_info=True)
        raise

