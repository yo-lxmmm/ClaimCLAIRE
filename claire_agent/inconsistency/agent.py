"""High-level wrapper around the Claire inconsistency detection agent."""

from __future__ import annotations

from typing import Any, Callable, Awaitable

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from utils.logger import logger

from .models import InconsistencyAgentState, InconsistencyReport, ClaimDecomposition, DecompositionValidation
from .prompts import AGENT_SYSTEM_PROMPT, build_agent_system_prompt, DECOMPOSE_CLAIM_PROMPT, VALIDATE_DECOMPOSITION_PROMPT
from .reporting import generate_report
from .tools import InconsistencyToolset


class InconsistencyAgent:
    """High-level wrapper around the inconsistency detection agent."""

    def __init__(
        self,
        engine: str,
        model_provider: str,
        num_results_per_query: int,
        reasoning_effort: str | None = None,
        # Ablation feature flags
        enable_react_agent: bool = True,
        enable_iterative_decomposition: bool = True,
        enable_trust_weighting: bool = True,
        enable_gap_filling: bool = True,
        enable_support_verification: bool = True,
    ) -> None:
        if not isinstance(engine, str) or not engine:
            raise ValueError("engine must be a non-empty string")
        if not isinstance(model_provider, str) or not model_provider:
            raise ValueError("model_provider must be a non-empty string")
        if not isinstance(num_results_per_query, int) or num_results_per_query <= 0:
            raise ValueError("num_results_per_query must be a positive integer")

        # Store ablation flags
        self._enable_react_agent = enable_react_agent
        self._enable_iterative_decomposition = enable_iterative_decomposition
        self._enable_trust_weighting = enable_trust_weighting
        self._enable_gap_filling = enable_gap_filling
        self._enable_support_verification = enable_support_verification
        self._engine = engine
        self._model_provider = model_provider
        self._num_results_per_query = num_results_per_query

        init_kwargs: dict[str, Any] = {
            "model": engine,
            "model_provider": model_provider,
        }
        # Only add reasoning parameter if not using Google GenAI
        if reasoning_effort is not None and model_provider != "google_genai":
            init_kwargs["reasoning"] = {"effort": reasoning_effort}

        self._model = init_chat_model(**init_kwargs)

        # Only create tools and agent if ReAct agent is enabled
        if enable_react_agent:
            toolset = InconsistencyToolset(
                model=self._model,
                num_results_per_query=num_results_per_query,
                reranking_engine=engine,
                model_provider=model_provider,
            )
            self._tools = toolset.build_tools()

            self._react_agent = create_agent(
                model=self._model,
                state_schema=InconsistencyAgentState,
                tools=self._tools,
                prompt=AGENT_SYSTEM_PROMPT,
            )
        else:
            self._tools = None
            self._react_agent = None

    async def analyze_claim(
        self,
        claim_text: str,
        passage: str,
        progress_callback: Callable[[dict], Awaitable[None]] | None = None,
    ) -> InconsistencyReport:
        if not isinstance(passage, str) or not passage.strip():
            raise ValueError("passage must be a non-empty string")

        # STAGE 1: Iterative Decomposition (if enabled)
        if self._enable_iterative_decomposition:
            logger.info("STAGE 1: Decomposing claim into atomic components with iterative validation...")
            if progress_callback:
                await progress_callback({
                    'stage': 1,
                    'stage_name': 'Claim Decomposition',
                    'status': 'start',
                    'message': 'Decomposing claim into atomic components...'
                })

            # Use Claude Sonnet 4 for decomposition
            pro_model = init_chat_model(
                model="claude-sonnet-4-20250514",
                model_provider="anthropic",
            )
            decompose_model = pro_model.with_structured_output(ClaimDecomposition)
            validate_model = pro_model.with_structured_output(DecompositionValidation)

            max_iterations = 3
            for iteration in range(max_iterations):
                decomposition: ClaimDecomposition = await decompose_model.ainvoke(
                    [
                        {"role": "system", "content": DECOMPOSE_CLAIM_PROMPT},
                        {"role": "user", "content": f"Claim to decompose:\n{claim_text}"},
                    ]
                )
                logger.info(f"Iteration {iteration + 1}: Claim decomposed into {len(decomposition.components)} component(s)")
                for i, comp in enumerate(decomposition.components):
                    logger.info(f"  Component {i+1}: {comp}")

                # Validate if decomposition is exhaustive
                validation: DecompositionValidation = await validate_model.ainvoke(
                    [
                        {"role": "system", "content": VALIDATE_DECOMPOSITION_PROMPT},
                        {"role": "user", "content": (
                            f"Original claim:\n{claim_text}\n\n"
                            f"Proposed components:\n" + "\n".join(f"- {c}" for c in decomposition.components)
                        )},
                    ]
                )

                logger.info(f"Validation: is_exhaustive={validation.is_exhaustive}")

                if validation.is_exhaustive:
                    logger.info("Decomposition validated as exhaustive. Proceeding to investigation.")
                    break
                else:
                    logger.warning(f"Decomposition incomplete. Missing: {validation.missing_components}")
                    if iteration < max_iterations - 1:
                        logger.info("Re-decomposing with feedback...")
                        decomposition.components.extend(validation.missing_components)
                    else:
                        logger.warning("Max iterations reached. Using current decomposition with added missing components.")
                        decomposition.components.extend(validation.missing_components)

            logger.info(f"Final decomposition has {len(decomposition.components)} component(s)")

            if progress_callback:
                await progress_callback({
                    'stage': 1,
                    'stage_name': 'Claim Decomposition',
                    'status': 'complete',
                    'message': f'Decomposed into {len(decomposition.components)} components',
                    'data': {
                        'components': decomposition.components,
                        'count': len(decomposition.components)
                    }
                })
        else:
            # No decomposition - treat full claim as single component (A0, A1)
            logger.info("STAGE 1: SKIPPING decomposition (treating claim as single component)")
            decomposition = ClaimDecomposition(components=[claim_text])

        # STAGE 2: Evidence Gathering
        state = self._build_initial_state(claim_text, passage)
        state["claim_decomposition"] = decomposition

        if self._enable_react_agent:
            # Use ReAct agent for adaptive evidence gathering (A1+)
            logger.info("STAGE 2: Running ReAct agent for adaptive evidence gathering...")
            if progress_callback:
                await progress_callback({
                    'stage': 2,
                    'stage_name': 'Evidence Gathering',
                    'status': 'start',
                    'message': 'Gathering evidence with ReAct agent...'
                })

            # Build component-aware agent with dynamic prompt
            component_aware_prompt = build_agent_system_prompt(decomposition.components)
            component_aware_agent = create_agent(
                model=self._model,
                state_schema=InconsistencyAgentState,
                tools=self._tools,
                prompt=component_aware_prompt,
            )

            await component_aware_agent.ainvoke(state, config={"recursion_limit": 100})
        else:
            # Simple procedural search (A0 only)
            logger.info("STAGE 2: Running SIMPLE search (no ReAct agent)...")
            if progress_callback:
                await progress_callback({
                    'stage': 2,
                    'stage_name': 'Evidence Gathering',
                    'status': 'start',
                    'message': 'Gathering evidence with simple search...'
                })

            # Import here to avoid circular dependency
            from retrieval.web_search import search_web

            # Simple search on full claim
            search_results = await search_web(
                search_query=claim_text,
                num_results=self._num_results_per_query,
                do_client_side_reranking=False,  # No reranking for baseline
            )
            state["all_search_results"] = search_results
            logger.info(f"Simple search found {len(search_results)} results")

        if progress_callback:
            await progress_callback({
                'stage': 2,
                'stage_name': 'Evidence Gathering',
                'status': 'complete',
                'message': f'Collected {len(state.get("all_search_results", []))} sources',
                'data': {
                    'results_found': len(state.get("all_search_results", []))
                }
            })

        # STAGES 3-5: Component evaluation, gap-filling, verdict synthesis, and report generation
        if progress_callback:
            await progress_callback({
                'stage': 3,
                'stage_name': 'Verification',
                'status': 'start',
                'message': 'Evaluating each component...'
            })
        await generate_report(
            model=self._model,
            state=state,
            enable_trust_weighting=self._enable_trust_weighting,
            enable_gap_filling=self._enable_gap_filling,
            enable_support_verification=self._enable_support_verification,
            progress_callback=progress_callback,
        )

        report = state.get("inconsistency_report")
        assert report is not None
        all_search_results = state.get("all_search_results", [])
        component_evaluations = state.get("component_evaluations", [])

        # Get gap-filling statistics from state (set in reporting.py)
        gap_fill_triggered = state.get("gap_fill_triggered", False)
        num_gap_fills = state.get("num_gap_fills", 0)

        return InconsistencyReport(
            claim_text=claim_text,
            claim_passage=passage,
            verdict=report.verdict,
            wording_feedback=report.wording_feedback,
            explanation=report.explanation,
            search_results=all_search_results,
            components=decomposition.components,
            component_evaluations=component_evaluations,
            gap_fill_triggered=gap_fill_triggered,
            num_gap_fills=num_gap_fills,
        )

    def _build_initial_state(
        self,
        claim_text: str,
        passage: str,
    ) -> InconsistencyAgentState:
        # Check if a date filter is active (from context variable)
        from retrieval.retriever_client import search_before_date_var
        cutoff_date_str = search_before_date_var.get()

        # Build the initial message with optional temporal context
        content_parts = ["You are investigating the following claim extracted from a passage."]

        if cutoff_date_str:
            content_parts.append(
                f"\nIMPORTANT: Evaluate this claim AS OF {cutoff_date_str}. "
                f"The claim should be verified based on information available on or before {cutoff_date_str}. "
                f"Ignore any events or information from after this date."
            )

        # Check if claim mentions media (photo/video/image) and add special instruction
        claim_lower = claim_text.lower()
        media_keywords = ["photo", "photograph", "image", "video", "picture"]
        has_media = any(keyword in claim_lower for keyword in media_keywords)
        
        if has_media:
            content_parts.append(
                "\n\nCRITICAL: This claim mentions a photo, photograph, image, or video. "
                "Your PRIMARY task is to fact-check whether the CLAIM about what the media shows is TRUE or FALSE (real or fake). "
                "You are NOT trying to verify whether the media file exists. "
                "You are NOT fact-checking tangential details like who was president at a given time. "
                "Focus EXCLUSIVELY on whether the description of what is shown in the media (the action, event, or scene depicted) is accurate and truthful. "
                "For example, if the claim says 'a photo shows X doing Y', verify whether X actually did Y, not whether X held a particular title or position."
            )

        content_parts.append(f"\n\nPassage:\n{passage}\n\nClaim:\n{claim_text}")

        initial_message = HumanMessage(content="".join(content_parts))

        return {
            "messages": [initial_message],
            "all_search_results": [],
            "inconsistency_report": None,
            "passage": passage,
            "claim": claim_text,
            "claim_decomposition": None,
            "component_evaluations": [],
        }


__all__ = ["InconsistencyAgent"]
