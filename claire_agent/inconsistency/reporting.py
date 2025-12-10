"""Utilities for summarising the agent run into a final inconsistency report."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import BaseMessage

from retrieval.document_block import Block
from retrieval.web_search import search_web as sw
from utils.logger import logger

from .models import (
    InconsistencyAgentState,
    InconsistencyReportResponse,
    ExplanationResponse,
    ClaimDecomposition,
    ComponentEvaluation,
)
from .prompts import (
    REPORT_SYSTEM_PROMPT,
    EVALUATE_COMPONENT_PROMPT,
)


def _generate_reliability_summary(blocks: list[Block]) -> str:
    """Generate a summary of source reliability distribution for the agent's context."""
    if not blocks:
        return "No sources found."
    
    reliable_count = sum(1 for b in blocks if b.trust_rating == "reliable")
    mixed_count = sum(1 for b in blocks if b.trust_rating == "mixed")
    unreliable_count = sum(1 for b in blocks if b.trust_rating == "unreliable")
    unrated_count = sum(1 for b in blocks if not b.trust_rating)
    
    summary_parts = []
    if reliable_count > 0:
        summary_parts.append(f"{reliable_count} reliable source(s)")
    if mixed_count > 0:
        summary_parts.append(f"{mixed_count} mixed reliability source(s)")
    if unreliable_count > 0:
        summary_parts.append(f"{unreliable_count} unreliable source(s)")
    if unrated_count > 0:
        summary_parts.append(f"{unrated_count} unrated source(s)")
    
    if summary_parts:
        return f"Total sources: {len(blocks)}. Reliability breakdown: {', '.join(summary_parts)}."
    return f"Total sources: {len(blocks)}. No reliability ratings available."


def _summarize_effective_weights(blocks: list[Block]) -> str:
    """Summarize effective weights for each block."""
    summaries: list[str] = []
    for idx, block in enumerate(blocks, start=1):
        effective_weight = block.effective_weight
        if effective_weight is None:
            continue

        rating = block.trust_rating or "unknown"
        summaries.append(
            f"[{idx}] {block.document_title} — {rating} (Effective Weight: {effective_weight:.2f})"
        )

    if not summaries:
        return "Effective weights unavailable for current sources."

    return "\n".join(summaries)


async def _evaluate_component_initial(
    component: str,
    component_index: int,
    all_search_results: list[Block],
    final_message_text: str,
    model: Any,
    enable_trust_weighting: bool = True,
) -> ComponentEvaluation:
    """Perform initial evaluation of a component (parallelizable).

    This function does NOT modify shared state and can run in parallel.

    Args:
        enable_trust_weighting: If False, skip trust/weight summaries
    """
    if enable_trust_weighting:
        reliability_summary = _generate_reliability_summary(all_search_results)
        weight_summary = _summarize_effective_weights(all_search_results)

        component_prompt = (
            f"Component to verify: {component}\n\n"
            f"Source Reliability Summary:\n{reliability_summary}\n\n"
            f"Effective Weight Summary:\n{weight_summary}\n\n"
            f"Search results:\n{Block.block_list_to_string(all_search_results)}\n\n"
            f"Agent's Investigation Summary: {final_message_text}\n\n"
            f"Determine if this component is verified, refuted, or unverified based on the evidence."
        )
    else:
        # Simple evaluation without trust weighting
        component_prompt = (
            f"Component to verify: {component}\n\n"
            f"Search results:\n{Block.block_list_to_string(all_search_results)}\n\n"
            f"Agent's Investigation Summary: {final_message_text}\n\n"
            f"Determine if this component is verified, refuted, or unverified based on the evidence."
        )

    evaluate_model = model.with_structured_output(ComponentEvaluation)
    evaluation: ComponentEvaluation = await evaluate_model.ainvoke(
        [
            {"role": "system", "content": EVALUATE_COMPONENT_PROMPT},
            {"role": "user", "content": component_prompt},
        ]
    )

    logger.info(f"Component {component_index + 1} initial evaluation: {evaluation.verdict}")
    return evaluation


def _build_support_verification_query(component: str, claim_text: str) -> str:
    """Build a query optimized for finding supporting evidence for a component.
    
    Strategy: Use component + full claim for maximum context.
    Adds a separator to distinguish component from claim context.
    
    Args:
        component: The component text to verify
        claim_text: Full claim text for context
        
    Returns:
        Search query optimized for finding supporting evidence
    """
    component = component.strip()
    claim_text = claim_text.strip()
    
    # Use component + full claim for context
    # Search engines (Google/Serper) treat space as AND by default, so this works well
    # The full claim provides additional context that helps with disambiguation
    return f"{component} {claim_text}"


def _build_gap_fill_query(component: str, claim_text: str, all_search_results: list[Block], use_full_claim: bool = False) -> str:
    """Build an optimized gap-filling query for an unverified component.
    
    Strategy: Always use component + full claim for maximum context.
    Adds a separator to distinguish component from claim context.
    The use_full_claim parameter is kept for API compatibility but both modes now use full claim.
    
    Args:
        component: The component text to verify
        claim_text: Full claim text for context
        all_search_results: Existing search results (unused, kept for API compatibility)
        use_full_claim: Unused, kept for API compatibility
        
    Returns:
        Search query with component + full claim
    """
    component = component.strip()
    claim_text = claim_text.strip()
    
    # Always use component + full claim with separator for clarity
    # Search engines handle this well, and the separator helps distinguish parts
    return f"{component} {claim_text}"


async def _gap_fill_and_reevaluate(
    component: str,
    component_index: int,
    initial_verdict: str,
    all_search_results: list[Block],
    final_message_text: str,
    claim_text: str,
    model: Any,
    state: InconsistencyAgentState,
) -> ComponentEvaluation:
    """Perform ONE targeted search for an unverified component and re-evaluate.

    This function DOES modify shared state (all_search_results) and must run sequentially.
    """
    logger.info(f"🔍 GAP-FILLING: Component {component_index + 1} '{component[:50]}...' is unverified")
    logger.info(f"   Initial verdict: {initial_verdict}")

    # Build optimized targeted query using improved strategy
    targeted_query = _build_gap_fill_query(component, claim_text, all_search_results)
    logger.info(f"   Targeted query: {targeted_query[:150]}")

    # Execute ONE targeted search with reranking
    additional_results = await sw(
        search_query=targeted_query,
        num_results=5,
        do_client_side_reranking=True,
        num_results_before_reranking=15,
    )

    logger.info(f"   Found {len(additional_results)} additional sources")

    # Add to shared evidence pool
    all_search_results.extend(additional_results)

    # Track gap-filling
    if "_gap_filled_components" not in state:
        state["_gap_filled_components"] = set()
    state["_gap_filled_components"].add(component_index)

    # Re-evaluate ONCE with expanded evidence
    expanded_reliability_summary = _generate_reliability_summary(all_search_results)
    expanded_weight_summary = _summarize_effective_weights(all_search_results)
    re_evaluation_prompt = (
        f"Component to verify: {component}\n\n"
        f"Source Reliability Summary:\n{expanded_reliability_summary}\n\n"
        f"Effective Weight Summary:\n{expanded_weight_summary}\n\n"
        f"Search results (including gap-filled sources):\n{Block.block_list_to_string(all_search_results)}\n\n"
        f"Agent's Investigation Summary: {final_message_text}\n\n"
        f"Determine if this component is verified, refuted, or unverified based on ALL evidence."
    )

    evaluate_model = model.with_structured_output(ComponentEvaluation)
    evaluation = await evaluate_model.ainvoke(
        [
            {"role": "system", "content": EVALUATE_COMPONENT_PROMPT},
            {"role": "user", "content": re_evaluation_prompt},
        ]
    )

    # Set metadata
    evaluation.initial_verdict = initial_verdict
    evaluation.gap_filled = True
    evaluation.verdict_changed = (evaluation.verdict != initial_verdict)

    logger.info(f"   Re-evaluation verdict: {evaluation.verdict}")
    if evaluation.verdict_changed:
        logger.info(f"   ✓ Verdict changed: {initial_verdict} → {evaluation.verdict}")
    else:
        logger.info(f"   ⏭️ Verdict unchanged: still {evaluation.verdict}")
    
    return evaluation


def _apply_logic_rules(component_evaluations: list[ComponentEvaluation]) -> str:
    """Apply deterministic logic rules to determine final verdict.

    Simplified: ALL components must be verified for consistency (AND logic).
    """
    # ALL components must be verified
    if all(e.verdict == "verified" for e in component_evaluations):
        return "consistent"
    else:
        return "inconsistent"


async def generate_report(
    *,
    model: Any,
    state: InconsistencyAgentState,
    enable_trust_weighting: bool = True,
    enable_gap_filling: bool = True,
    enable_support_verification: bool = True,
    progress_callback=None,
) -> None:
    """Populate the agent state with a structured inconsistency report.

    New simplified flow with gap-filling:
    1. Decomposition already done in agent.py (BEFORE ReAct)
    2. Evaluate each component against evidence (with gap-filling if unverified)
    3. Apply deterministic logic rules for verdict
    4. Generate explanation and wording feedback

    Args:
        model: LLM model for evaluation
        state: Agent state containing evidence and decomposition
        enable_trust_weighting: If False, skip trust rating summaries in evaluation
        enable_gap_filling: If False, skip gap-filling for unverified components
        enable_support_verification: If False, skip support verification searches
        progress_callback: Optional callback for progress updates
    """
    messages = list(state.get("messages", []))
    if not messages:
        logger.warning("No messages available for explanation generation.")
        return

    final_message = next(
        (msg for msg in reversed(messages) if isinstance(msg, BaseMessage)),
        None,
    )
    if final_message is None:
        logger.warning("Unable to locate final message for explanation generation.")
        return

    final_message_text = final_message.text()
    all_search_results = list(state.get("all_search_results", []))

    # Deduplicate search results
    unique_results: dict[str, Block] = {}
    for block in all_search_results:
        if block.combined_text not in unique_results:
            unique_results[block.combined_text] = block
    all_search_results = list(unique_results.values())

    passage = state.get("passage", "")
    claim_text = state.get("claim", "")

    # Get decomposition (already done in agent.py)
    decomposition = state.get("claim_decomposition")
    if not decomposition:
        logger.error("No claim decomposition found in state!")
        return

    logger.info("STAGE 3: Evaluating components with hybrid parallel evaluation...")

    # STEP 3a: PARALLEL initial evaluation of ALL components
    logger.info("STEP 3a: Parallel initial evaluation of all components...")

    initial_eval_tasks = [
        _evaluate_component_initial(
            component=component,
            component_index=idx,
            all_search_results=all_search_results,
            final_message_text=final_message_text,
            model=model,
            enable_trust_weighting=enable_trust_weighting,
        )
        for idx, component in enumerate(decomposition.components)
    ]

    # Use return_exceptions=True to handle individual failures gracefully
    initial_eval_results = await asyncio.gather(*initial_eval_tasks, return_exceptions=True)

    # Process results and handle exceptions
    initial_evaluations: list[ComponentEvaluation] = []
    for idx, result in enumerate(initial_eval_results):
        if isinstance(result, Exception):
            # Create fallback evaluation for failed components
            logger.error(
                f"Component {idx + 1} evaluation failed: {result}. "
                f"Creating fallback 'unverified' evaluation."
            )
            fallback_eval = ComponentEvaluation(
                verdict="unverified",
                reasoning=f"Evaluation failed due to error: {str(result)[:200]}",
                initial_verdict=None,
                gap_filled=False,
                verdict_changed=False,
            )
            initial_evaluations.append(fallback_eval)
        elif isinstance(result, ComponentEvaluation):
            initial_evaluations.append(result)
        else:
            # Unexpected result type
            logger.error(
                f"Component {idx + 1} evaluation returned unexpected type: {type(result)}. "
                f"Creating fallback 'unverified' evaluation."
            )
            fallback_eval = ComponentEvaluation(
                verdict="unverified",
                reasoning="Evaluation returned unexpected result type",
                initial_verdict=None,
                gap_filled=False,
                verdict_changed=False,
            )
            initial_evaluations.append(fallback_eval)

    # Validate that we have the correct number of evaluations
    if len(initial_evaluations) != len(decomposition.components):
        logger.error(
            f"Component evaluation count mismatch: {len(initial_evaluations)} != "
            f"{len(decomposition.components)}. This should not happen."
        )
        # Try to pad with fallback evaluations if somehow we're short
        while len(initial_evaluations) < len(decomposition.components):
            fallback_eval = ComponentEvaluation(
                verdict="unverified",
                reasoning="Missing evaluation - component was not evaluated",
                initial_verdict=None,
                gap_filled=False,
                verdict_changed=False,
            )
            initial_evaluations.append(fallback_eval)

    if progress_callback:
        await progress_callback({
            'stage': 3,
            'stage_name': 'Verification',
            'status': 'update',
            'message': f'Evaluated {len(initial_evaluations)} components'
        })

    # STEP 3b: Support verification for potentially "Supported" claims
    # If no components are "refuted" (only "verified" or "unverified"),
    # perform targeted searches for supporting evidence
    has_refuted = any(eval.verdict == "refuted" for eval in initial_evaluations)
    has_unverified = any(eval.verdict == "unverified" for eval in initial_evaluations)

    if enable_support_verification and not has_refuted and has_unverified:
        # No contradictions found, but some components unverified
        # Perform targeted searches for supporting evidence
        logger.info("STEP 3b.1: No contradictions found. Performing support verification searches...")
        support_verification_count = 0
        
        for idx, (component, initial_eval) in enumerate(zip(decomposition.components, initial_evaluations)):
            if initial_eval.verdict == "unverified":
                # Build query optimized for finding supporting evidence
                # Use component + key verification terms
                support_query = _build_support_verification_query(component, claim_text)
                logger.info(f"   Support verification for component {idx + 1}: {support_query[:100]}")
                
                try:
                    support_results = await sw(
                        search_query=support_query,
                        num_results=5,
                        do_client_side_reranking=True,
                        num_results_before_reranking=15,
                    )
                    
                    if support_results:
                        all_search_results.extend(support_results)
                        support_verification_count += 1
                        logger.info(f"   Found {len(support_results)} supporting sources for component {idx + 1}")
                except Exception as e:
                    logger.warning(f"   Support verification search failed for component {idx + 1}: {e}")
        
        if support_verification_count > 0:
            logger.info(f"   Added supporting evidence for {support_verification_count} components")
            # Re-evaluate unverified components with new supporting evidence
            logger.info("   Re-evaluating unverified components with supporting evidence...")
            updated_evaluations = []
            for idx, (component, initial_eval) in enumerate(zip(decomposition.components, initial_evaluations)):
                if initial_eval.verdict == "unverified":
                    # Quick re-evaluation with expanded evidence
                    reliability_summary = _generate_reliability_summary(all_search_results)
                    weight_summary = _summarize_effective_weights(all_search_results)
                    re_eval_prompt = (
                        f"Component to verify: {component}\n\n"
                        f"Source Reliability Summary:\n{reliability_summary}\n\n"
                        f"Effective Weight Summary:\n{weight_summary}\n\n"
                        f"Search results (including support verification):\n{Block.block_list_to_string(all_search_results)}\n\n"
                        f"Agent's Investigation Summary: {final_message_text}\n\n"
                        f"Determine if this component is verified, refuted, or unverified based on ALL evidence."
                    )
                    
                    evaluate_model = model.with_structured_output(ComponentEvaluation)
                    try:
                        re_eval = await evaluate_model.ainvoke(
                            [
                                {"role": "system", "content": EVALUATE_COMPONENT_PROMPT},
                                {"role": "user", "content": re_eval_prompt},
                            ]
                        )
                        if re_eval.verdict != initial_eval.verdict:
                            logger.info(f"   Component {idx + 1} verdict changed after support verification: {initial_eval.verdict} → {re_eval.verdict}")
                            re_eval.initial_verdict = "unverified"
                            re_eval.gap_filled = False  # Not gap-filled yet, just support verification
                            re_eval.verdict_changed = True
                            updated_evaluations.append(re_eval)
                        else:
                            updated_evaluations.append(initial_eval)
                    except Exception as e:
                        logger.warning(f"   Re-evaluation failed for component {idx + 1}: {e}")
                        updated_evaluations.append(initial_eval)
                else:
                    updated_evaluations.append(initial_eval)
            
            # Update initial_evaluations with re-evaluated results
            initial_evaluations = updated_evaluations

    # STEP 3c: BATCH gap-filling for remaining unverified components
    # Use different query strategies based on whether support verification ran
    has_unverified_remaining = any(eval.verdict == "unverified" for eval in initial_evaluations)

    component_evaluations: list[ComponentEvaluation] = []
    gap_fill_count = 0

    if enable_gap_filling and has_unverified_remaining:
        # Determine query strategy based on whether support verification ran
        # If support verification ran (not has_refuted), continue using snippet queries
        # If support verification didn't run (has_refuted), use full claim or component-only
        use_full_claim_for_gap_fill = has_refuted  # Use full claim when there are refuted components
        
        logger.info(f"STEP 3c: Batch gap-filling for remaining unverified components...")
        if use_full_claim_for_gap_fill:
            logger.info("   Using regular gap-filling strategy (component-only or component + full claim)")
        else:
            logger.info("   Using support-focused strategy (component + claim snippet)")
        
        # Phase 1: Batch search for all unverified components
        search_count = 0
        for idx, (component, initial_eval) in enumerate(zip(decomposition.components, initial_evaluations)):
            if initial_eval.verdict == "unverified":
                # Build query based on context
                # If support verification ran (not has_refuted), use snippet queries
                # If support verification didn't run (has_refuted), use full claim queries
                gap_query = _build_gap_fill_query(component, claim_text, all_search_results, use_full_claim=use_full_claim_for_gap_fill)
                
                logger.info(f"   Searching for component {idx + 1}: {gap_query[:100]}")
                
                try:
                    search_results = await sw(
                        search_query=gap_query,
                        num_results=5,
                        do_client_side_reranking=True,
                        num_results_before_reranking=15,
                    )
                    
                    if search_results:
                        all_search_results.extend(search_results)
                        search_count += 1
                        logger.info(f"   Found {len(search_results)} sources for component {idx + 1}")
                except Exception as e:
                    logger.warning(f"   Gap-filling search failed for component {idx + 1}: {e}")
        
        if search_count > 0:
            logger.info(f"   Collected evidence for {search_count} components")
            
            # Phase 2: Batch re-evaluate ALL unverified components with expanded evidence pool
            logger.info("   Re-evaluating unverified components with expanded evidence...")
            reliability_summary = _generate_reliability_summary(all_search_results)
            weight_summary = _summarize_effective_weights(all_search_results)
            
            for idx, (component, initial_eval) in enumerate(zip(decomposition.components, initial_evaluations)):
                if initial_eval.verdict == "unverified":
                    # Re-evaluate with all collected evidence
                    re_eval_prompt = (
                        f"Component to verify: {component}\n\n"
                        f"Source Reliability Summary:\n{reliability_summary}\n\n"
                        f"Effective Weight Summary:\n{weight_summary}\n\n"
                        f"Search results (including gap-filled sources):\n{Block.block_list_to_string(all_search_results)}\n\n"
                        f"Agent's Investigation Summary: {final_message_text}\n\n"
                        f"Determine if this component is verified, refuted, or unverified based on ALL evidence."
                    )
                    
                    evaluate_model = model.with_structured_output(ComponentEvaluation)
                    try:
                        re_eval = await evaluate_model.ainvoke(
                            [
                                {"role": "system", "content": EVALUATE_COMPONENT_PROMPT},
                                {"role": "user", "content": re_eval_prompt},
                            ]
                        )
                        
                        # Set metadata
                        re_eval.initial_verdict = "unverified"
                        re_eval.gap_filled = True
                        re_eval.verdict_changed = (re_eval.verdict != initial_eval.verdict)
                        
                        if re_eval.verdict_changed:
                            logger.info(f"   Component {idx + 1} verdict changed: {initial_eval.verdict} → {re_eval.verdict}")
                            gap_fill_count += 1
                        
                        component_evaluations.append(re_eval)
                    except Exception as e:
                        logger.warning(f"   Re-evaluation failed for component {idx + 1}: {e}")
                        # Keep original evaluation on failure
                        initial_eval.initial_verdict = initial_eval.verdict
                        initial_eval.gap_filled = False
                        initial_eval.verdict_changed = False
                        component_evaluations.append(initial_eval)
                else:
                    # Already verified/refuted, no gap-filling needed
                    initial_eval.initial_verdict = initial_eval.verdict
                    initial_eval.gap_filled = False
                    initial_eval.verdict_changed = False
                    component_evaluations.append(initial_eval)
        else:
            # No searches succeeded, use initial evaluations with metadata
            logger.warning("   No additional evidence collected")
            for initial_eval in initial_evaluations:
                initial_eval.initial_verdict = initial_eval.verdict
                initial_eval.gap_filled = False
                initial_eval.verdict_changed = False
                component_evaluations.append(initial_eval)
    else:
        # No unverified components remaining, use initial evaluations with metadata
        logger.info("STEP 3c: All components already verified or refuted, skipping gap-filling")
        for initial_eval in initial_evaluations:
            initial_eval.initial_verdict = initial_eval.verdict
            initial_eval.gap_filled = False
            initial_eval.verdict_changed = False
            component_evaluations.append(initial_eval)
    
    # Track gap-filling statistics
    state["gap_fill_triggered"] = gap_fill_count > 0
    state["num_gap_fills"] = gap_fill_count

    state["component_evaluations"] = component_evaluations
    state["all_search_results"] = all_search_results  # Update with gap-filled results

    if progress_callback:
        if gap_fill_count > 0:
            await progress_callback({
                'stage': 4,
                'stage_name': 'Gap Filling',
                'status': 'complete',
                'message': f'Gap-filled {gap_fill_count} components'
            })
        await progress_callback({
            'stage': 3,
            'stage_name': 'Verification',
            'status': 'complete',
            'message': 'All components evaluated'
        })

    # STAGE 4: Apply deterministic logic rules
    logger.info("STAGE 4: Applying verdict logic rules...")
    
    # Validate that we have component evaluations before applying logic
    if not component_evaluations:
        logger.error(
            "No component evaluations available! Cannot determine verdict. "
            "This indicates a critical failure in the evaluation pipeline."
        )
        # Raise an error instead of defaulting to a verdict
        raise RuntimeError(
            "All component evaluations failed. Cannot determine claim verdict. "
            "This indicates a critical failure in the evaluation pipeline."
        )
    
    verdict = _apply_logic_rules(component_evaluations)
    logger.info(f"Deterministic verdict: {verdict}")
    
    # Mark stage 4 as complete
    if progress_callback:
        await progress_callback({
            'stage': 4,
            'stage_name': 'Verdict Synthesis',
            'status': 'complete',
            'message': f'Verdict determined: {verdict}'
        })

    # STAGE 5: Generate explanation and wording feedback
    logger.info("STAGE 5: Generating explanation...")
    if progress_callback:
        await progress_callback({
            'stage': 5,
            'stage_name': 'Final Report',
            'status': 'start',
            'message': 'Generating final explanation...'
        })

    # Format component evaluations for the prompt
    component_summary = "\n".join([
        f"- {comp}\n  → {eval.verdict.upper()}: {eval.reasoning}"
        for comp, eval in zip(decomposition.components, component_evaluations)
    ])

    reliability_summary = _generate_reliability_summary(all_search_results)

    # Organize blocks by unique URL to ensure consistent numbering
    # First, get unique URLs in order of first appearance
    url_to_blocks: dict[str, list[Block]] = {}
    url_order: list[str] = []
    for block in all_search_results:
        url = block.url or ""
        if url not in url_to_blocks:
            url_to_blocks[url] = []
            url_order.append(url)
        url_to_blocks[url].append(block)
    
    # Reorder blocks: first occurrence of each URL, then subsequent occurrences
    ordered_blocks: list[Block] = []
    for url in url_order:
        ordered_blocks.extend(url_to_blocks[url])

    explanation_prompt = (
        f"Original Passage:\n{passage}\n\n"
        f"Investigated Claim:\n{claim_text}\n\n"
        f"Component Evaluations:\n{component_summary}\n\n"
        f"Verdict (already determined): {verdict}\n\n"
        f"Source Reliability Summary:\n{reliability_summary}\n\n"
        f"Search results:\n{Block.block_list_to_string(ordered_blocks)}\n\n"
        f"Agent's Investigation Summary: {final_message_text}\n\n"
        f"Write a coherent explanation (1-2 paragraphs) with [n] citations. "
        f"Citations [n] refer to unique source URLs numbered in order. "
        f"Also provide wording feedback for the claim."
    )

    explanation_model = model.with_structured_output(ExplanationResponse)
    explanation_response: ExplanationResponse = await explanation_model.ainvoke(
        [
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": explanation_prompt},
        ]
    )

    # Combine into final report
    inconsistency_report_response = InconsistencyReportResponse(
        verdict=verdict,
        explanation=explanation_response.explanation,
        wording_feedback=explanation_response.wording_feedback,
    )

    state["inconsistency_report"] = inconsistency_report_response

    if progress_callback:
        await progress_callback({
            'stage': 5,
            'stage_name': 'Final Report',
            'status': 'complete',
            'message': 'Report generation complete'
        })


__all__ = ["generate_report"]
