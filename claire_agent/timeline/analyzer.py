"""Timeline analyzer using binary search to find verdict transitions."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from utils.logger import logger

from retrieval.retriever_client import set_search_before_date
from .models import Transition, TimelinePeriod, TimelineAnalysis

# Configuration
MAX_TRANSITIONS = 5
MAX_DEPTH = 8


async def find_exact_transitions(
    claim: str,
    start_date: str,
    end_date: str,
    agent: Any,  # InconsistencyAgent instance
    max_transitions: int = MAX_TRANSITIONS,
    precision: str = "month"
) -> TimelineAnalysis:
    """
    Find exact dates when claim verdict changed using binary search.

    Args:
        claim: The claim to analyze
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        agent: The InconsistencyAgent instance to use for analysis
        max_transitions: Maximum number of transitions to find (default 5)
        precision: Precision level - "month", "week", or "day"

    Returns:
        TimelineAnalysis object with transitions and periods
    """

    logger.info(f"Starting timeline analysis: {claim} ({start_date} to {end_date})")

    total_checks = 0

    try:
        # Step 1: Check endpoints
        logger.info("Checking start endpoint...")
        start_result = await analyze_at_date(claim, start_date, agent)
        total_checks += 1

        logger.info("Checking end endpoint...")
        end_result = await analyze_at_date(claim, end_date, agent)
        total_checks += 1

        start_verdict = start_result['verdict']
        end_verdict = end_result['verdict']

        logger.info(f"Endpoints: {start_verdict} ({start_date}) → {end_verdict} ({end_date})")

        # Step 2: Check if transition exists
        if start_verdict == end_verdict:
            # No transition - stable throughout
            logger.info(f"No transitions found. Verdict remained '{start_verdict}' throughout.")

            period = TimelinePeriod(
                start_date=start_date,
                end_date=end_date,
                verdict=start_verdict,
                num_sources=end_result['num_sources'],
                reliable_sources=end_result['reliable_count'],
                mixed_sources=end_result['mixed_count'],
                unreliable_sources=end_result['unreliable_count'],
                representative_sources=end_result.get('sources', [])
            )

            return TimelineAnalysis(
                claim=claim,
                time_range=(start_date, end_date),
                transitions=[],
                periods=[period],
                metadata={
                    'total_checks': total_checks,
                    'transitions_found': 0,
                    'max_transitions_reached': False,
                    'message': f'Verdict remained "{start_verdict}" throughout entire period'
                }
            )

        # Step 3: Binary search for all transitions
        logger.info("Transitions detected. Starting binary search...")

        transitions_found = []
        transitions = await find_all_transitions_recursive(
            claim=claim,
            start_date=start_date,
            end_date=end_date,
            start_verdict=start_verdict,
            end_verdict=end_verdict,
            agent=agent,
            depth=0,
            transitions_found=transitions_found,
            precision=precision,
            max_transitions=max_transitions
        )

        # Count total checks (rough estimate based on transitions found)
        total_checks = 2 + len(transitions) * 3  # 2 endpoints + ~3 checks per transition

        logger.info(f"Found {len(transitions)} transition(s)")

        # Step 4: Generate periods from transitions
        periods = generate_timeline_periods(
            transitions=transitions,
            start_date=start_date,
            end_date=end_date,
            start_result=start_result,
            end_result=end_result
        )

        # Step 5: Build transition objects
        transition_objects = []
        for i, trans in enumerate(transitions):
            # Get details from the transition dict
            transition_obj = Transition(
                from_date=trans['from_date'],
                to_date=trans['to_date'],
                from_verdict=trans['from_verdict'],
                to_verdict=trans['to_verdict'],
                from_sources_count=trans.get('from_sources_count', 0),
                to_sources_count=trans.get('to_sources_count', 0),
                from_reliable_count=trans.get('from_reliable_count', 0),
                to_reliable_count=trans.get('to_reliable_count', 0),
                confidence=trans.get('confidence', 1.0),
                explanation=None  # Could generate later with LLM
            )
            transition_objects.append(transition_obj)

        max_reached = len(transitions) >= max_transitions

        return TimelineAnalysis(
            claim=claim,
            time_range=(start_date, end_date),
            transitions=transition_objects,
            periods=periods,
            metadata={
                'total_checks': total_checks,
                'transitions_found': len(transitions),
                'max_transitions_reached': max_reached,
                'message': f'Found {len(transitions)} transition(s)' + (
                    '. Stopped at maximum.' if max_reached else ''
                ),
                'estimated_cost': total_checks * 0.0005
            }
        )

    except Exception as e:
        logger.error(f"Timeline analysis failed: {e}", exc_info=True)
        raise


async def find_all_transitions_recursive(
    claim: str,
    start_date: str,
    end_date: str,
    start_verdict: str,
    end_verdict: str,
    agent: Any,
    depth: int = 0,
    transitions_found: List = None,
    precision: str = "month",
    max_transitions: int = MAX_TRANSITIONS
) -> List[Dict]:
    """
    Recursively find all transitions using binary search.

    Returns:
        List of transition dicts with from_date, to_date, from_verdict, to_verdict
    """

    if transitions_found is None:
        transitions_found = []

    # Base case 1: Max depth reached
    if depth > MAX_DEPTH:
        logger.warning(f"Max depth {MAX_DEPTH} reached at depth {depth}")
        return []

    # Base case 2: Max transitions reached
    if len(transitions_found) >= max_transitions:
        logger.info(f"Max transitions ({max_transitions}) reached")
        return []

    # Base case 3: Precision reached
    if dates_within_precision(start_date, end_date, precision):
        logger.info(f"Precision reached: {start_date} to {end_date} (within {precision})")

        # Get source counts for this transition
        start_result = await analyze_at_date(claim, start_date, agent)
        end_result = await analyze_at_date(claim, end_date, agent)

        return [{
            'from_date': start_date,
            'to_date': end_date,
            'from_verdict': start_verdict,
            'to_verdict': end_verdict,
            'from_sources_count': start_result['num_sources'],
            'to_sources_count': end_result['num_sources'],
            'from_reliable_count': start_result['reliable_count'],
            'to_reliable_count': end_result['reliable_count'],
            'confidence': 1.0
        }]

    # Calculate midpoint
    mid_date = calculate_midpoint_date(start_date, end_date)
    logger.info(f"Checking midpoint: {mid_date} (depth {depth})")

    # Analyze at midpoint
    mid_result = await analyze_at_date(claim, mid_date, agent)
    mid_verdict = mid_result['verdict']

    logger.info(f"Midpoint verdict: {mid_verdict}")

    transitions = []

    # Search left half: start → mid
    if start_verdict != mid_verdict:
        logger.info(f"Transition detected in left half ({start_verdict} → {mid_verdict})")
        left_transitions = await find_all_transitions_recursive(
            claim=claim,
            start_date=start_date,
            end_date=mid_date,
            start_verdict=start_verdict,
            end_verdict=mid_verdict,
            agent=agent,
            depth=depth + 1,
            transitions_found=transitions_found,
            precision=precision,
            max_transitions=max_transitions
        )
        transitions.extend(left_transitions)
        transitions_found.extend(left_transitions)

    # Search right half: mid → end (if not at max)
    if len(transitions_found) < max_transitions and mid_verdict != end_verdict:
        logger.info(f"Transition detected in right half ({mid_verdict} → {end_verdict})")
        right_transitions = await find_all_transitions_recursive(
            claim=claim,
            start_date=mid_date,
            end_date=end_date,
            start_verdict=mid_verdict,
            end_verdict=end_verdict,
            agent=agent,
            depth=depth + 1,
            transitions_found=transitions_found,
            precision=precision,
            max_transitions=max_transitions
        )
        transitions.extend(right_transitions)
        transitions_found.extend(right_transitions)

    return transitions


async def analyze_at_date(claim: str, before_date: str, agent: Any) -> Dict:
    """
    Analyze claim using sources published before specified date.

    Args:
        claim: The claim to analyze
        before_date: Only use sources from before this date (YYYY-MM-DD)
        agent: The InconsistencyAgent instance

    Returns:
        Dict with verdict, num_sources, counts by trust rating, and sources
    """

    # Set date context for this analysis
    set_search_before_date(before_date)

    try:
        logger.debug(f"Analyzing claim at date: {before_date}")

        # Run standard analysis - date filter applied via context
        reports = await agent.analyze_passage_for_inconsistencies(
            passage=claim
        )

        if not reports or len(reports) == 0:
            return {
                'verdict': 'unknown',
                'num_sources': 0,
                'reliable_count': 0,
                'mixed_count': 0,
                'unreliable_count': 0,
                'sources': []
            }

        # Get verdict from first report (single claim)
        report = reports[0]

        # Extract sources from report - InconsistencyReport has search_results attribute
        sources = report.search_results if hasattr(report, 'search_results') else []

        # Count sources by trust rating
        reliable_count = sum(1 for s in sources if getattr(s, 'trust_rating', 'mixed') == 'reliable')
        mixed_count = sum(1 for s in sources if getattr(s, 'trust_rating', 'mixed') == 'mixed')
        unreliable_count = sum(1 for s in sources if getattr(s, 'trust_rating', 'mixed') == 'unreliable')

        # Determine verdict - InconsistencyReport has verdict attribute
        verdict = report.verdict if hasattr(report, 'verdict') else 'unknown'
        if isinstance(verdict, str):
            verdict = verdict.lower()

        # If no sources, verdict is unknown
        if len(sources) == 0:
            verdict = 'unknown'

        logger.debug(f"Date {before_date}: {verdict} ({len(sources)} sources, {reliable_count} reliable)")

        return {
            'verdict': verdict,
            'num_sources': len(sources),
            'reliable_count': reliable_count,
            'mixed_count': mixed_count,
            'unreliable_count': unreliable_count,
            'sources': sources[:5]  # Keep top 5 for display
        }

    finally:
        # Always clear the date context
        set_search_before_date(None)


def calculate_midpoint_date(start: str, end: str) -> str:
    """Calculate temporal midpoint between two dates."""
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    delta = (end_dt - start_dt) / 2
    mid_dt = start_dt + delta

    return mid_dt.strftime("%Y-%m-%d")


def dates_within_precision(start: str, end: str, precision: str) -> bool:
    """Check if dates are within desired precision threshold."""
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    delta_days = (end_dt - start_dt).days

    thresholds = {
        "month": 30,
        "week": 7,
        "day": 1
    }

    return delta_days <= thresholds.get(precision, 30)


def generate_timeline_periods(
    transitions: List[Dict],
    start_date: str,
    end_date: str,
    start_result: Dict,
    end_result: Dict
) -> List[TimelinePeriod]:
    """
    Generate stable periods from transitions.

    Args:
        transitions: List of transition dicts
        start_date: Overall start date
        end_date: Overall end date
        start_result: Analysis result at start
        end_result: Analysis result at end

    Returns:
        List of TimelinePeriod objects
    """

    if len(transitions) == 0:
        # No transitions - single period
        return [
            TimelinePeriod(
                start_date=start_date,
                end_date=end_date,
                verdict=start_result['verdict'],
                num_sources=end_result['num_sources'],
                reliable_sources=end_result['reliable_count'],
                mixed_sources=end_result['mixed_count'],
                unreliable_sources=end_result['unreliable_count'],
                representative_sources=end_result.get('sources', [])
            )
        ]

    # Sort transitions by date
    sorted_transitions = sorted(transitions, key=lambda t: t['from_date'])

    periods = []

    # First period: start to first transition
    first_trans = sorted_transitions[0]
    periods.append(
        TimelinePeriod(
            start_date=start_date,
            end_date=first_trans['from_date'],
            verdict=first_trans['from_verdict'],
            num_sources=first_trans['from_sources_count'],
            reliable_sources=first_trans['from_reliable_count'],
            mixed_sources=0,  # Simplified
            unreliable_sources=0,
            representative_sources=[]
        )
    )

    # Middle periods: between transitions
    for i in range(len(sorted_transitions) - 1):
        curr_trans = sorted_transitions[i]
        next_trans = sorted_transitions[i + 1]

        periods.append(
            TimelinePeriod(
                start_date=curr_trans['to_date'],
                end_date=next_trans['from_date'],
                verdict=curr_trans['to_verdict'],
                num_sources=curr_trans['to_sources_count'],
                reliable_sources=curr_trans['to_reliable_count'],
                mixed_sources=0,
                unreliable_sources=0,
                representative_sources=[]
            )
        )

    # Last period: last transition to end
    last_trans = sorted_transitions[-1]
    periods.append(
        TimelinePeriod(
            start_date=last_trans['to_date'],
            end_date=end_date,
            verdict=last_trans['to_verdict'],
            num_sources=last_trans['to_sources_count'],
            reliable_sources=last_trans['to_reliable_count'],
            mixed_sources=end_result['mixed_count'],
            unreliable_sources=end_result['unreliable_count'],
            representative_sources=end_result.get('sources', [])
        )
    )

    return periods
