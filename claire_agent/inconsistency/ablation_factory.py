"""Factory for creating ablation agents with proper feature flags.

This module provides a factory function to create InconsistencyAgent instances
configured for different ablation studies (A0-A4).

Ablation Study Design:
- A0: Baseline RAG (simple search, no decomposition, no trust weighting, no gap-filling, no ReAct)
- A1: A0 + ReAct Agent (adaptive evidence gathering)
- A2: A1 + Iterative Decomposition (with validation for exhaustive extraction)
- A3: A2 + Trust Weighting (source credibility scoring)
- A4: A3 + Gap-Filling + Support Verification (adaptive re-querying) [Full System]

This design allows testing each contribution:
- A1 vs A0: Value of adaptive evidence gathering (ReAct)
- A2 vs A1: Value of iterative decomposition (Contribution A: Iterative Decomposition)
- A3 vs A2: Value of source credibility weighting
- A4 vs A3: Value of adaptive gap-filling (Contribution B: Adaptive Gap-Filling)
"""

from __future__ import annotations

from typing import Literal

from .agent import InconsistencyAgent


AblationID = Literal["A0", "A1", "A2", "A3", "A4"]


# Ablation configuration mapping
ABLATION_CONFIGS: dict[AblationID, dict[str, bool]] = {
    "A0": {
        # Baseline RAG: Simple search, no features
        "enable_react_agent": False,
        "enable_iterative_decomposition": False,
        "enable_trust_weighting": False,
        "enable_gap_filling": False,
        "enable_support_verification": False,
    },
    "A1": {
        # A0 + ReAct Agent for adaptive evidence gathering
        "enable_react_agent": True,
        "enable_iterative_decomposition": False,
        "enable_trust_weighting": False,
        "enable_gap_filling": False,
        "enable_support_verification": False,
    },
    "A2": {
        # A1 + Iterative Decomposition (with validation)
        # Tests Contribution A: Iterative Decomposition
        "enable_react_agent": True,
        "enable_iterative_decomposition": True,
        "enable_trust_weighting": False,
        "enable_gap_filling": False,
        "enable_support_verification": False,
    },
    "A3": {
        # A2 + Trust Weighting (source credibility scoring)
        "enable_react_agent": True,
        "enable_iterative_decomposition": True,
        "enable_trust_weighting": True,
        "enable_gap_filling": False,
        "enable_support_verification": False,
    },
    "A4": {
        # A3 + Gap-Filling + Support Verification (adaptive re-querying)
        # Tests Contribution B: Adaptive Gap-Filling
        # This is the full system
        "enable_react_agent": True,
        "enable_iterative_decomposition": True,
        "enable_trust_weighting": True,
        "enable_gap_filling": True,
        "enable_support_verification": True,
    },
}


def create_ablation_agent(
    ablation_id: AblationID,
    engine: str,
    model_provider: str,
    num_results_per_query: int,
    reasoning_effort: str | None = None,
) -> InconsistencyAgent:
    """Factory for creating ablation agents with proper feature flags.

    Args:
        ablation_id: Ablation variant ("A0", "A1", "A2", "A3", or "A4")
        engine: LLM engine to use
        model_provider: Model provider (e.g., "google_genai", "azure_openai")
        num_results_per_query: Number of search results per query
        reasoning_effort: Optional reasoning effort parameter

    Returns:
        InconsistencyAgent configured with appropriate feature flags

    Raises:
        ValueError: If ablation_id is not recognized

    Example:
        >>> agent = create_ablation_agent("A2", "gemini-2.5-flash", "google_genai", 10)
        >>> # Creates agent with ReAct + iterative decomposition, no trust weighting/gap-filling
    """
    if ablation_id not in ABLATION_CONFIGS:
        raise ValueError(
            f"Unknown ablation_id: {ablation_id}. "
            f"Must be one of: {', '.join(ABLATION_CONFIGS.keys())}"
        )

    config = ABLATION_CONFIGS[ablation_id]

    return InconsistencyAgent(
        engine=engine,
        model_provider=model_provider,
        num_results_per_query=num_results_per_query,
        reasoning_effort=reasoning_effort,
        **config,
    )


def get_ablation_description(ablation_id: AblationID) -> str:
    """Get a human-readable description of an ablation variant.

    Args:
        ablation_id: Ablation variant ("A0" through "A4")

    Returns:
        Description string explaining what features are enabled

    Example:
        >>> print(get_ablation_description("A2"))
        A2: ReAct + Iterative Decomposition [Contribution A]
    """
    descriptions = {
        "A0": "Baseline RAG (simple search, no features)",
        "A1": "A0 + ReAct Agent (adaptive evidence gathering)",
        "A2": "A1 + Iterative Decomposition",
        "A3": "A2 + Trust Weighting (source credibility scoring)",
        "A4": "A3 + Gap-Filling + Support Verification (full system)",
    }
    return descriptions.get(ablation_id, "Unknown ablation")


__all__ = ["create_ablation_agent", "get_ablation_description", "AblationID", "ABLATION_CONFIGS"]
