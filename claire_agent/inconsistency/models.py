"""Data models and agent state definitions for inconsistency detection."""

from __future__ import annotations

from typing import Annotated, Literal, Sequence

from langchain.agents import AgentState
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from retrieval.document_block import Block


class ClaimDecomposition(BaseModel):
    """Decomposition of a complex claim into atomic verifiable components (simplified)."""

    components: list[str] = Field(
        description=(
            "List of atomic components. Each should be a complete, standalone statement "
            "that can be verified independently. All components are treated equally and all must "
            "be verified for the claim to be consistent (AND logic)."
        ),
    )


class DecompositionValidation(BaseModel):
    """Schema for validating if a decomposition is exhaustive."""
    is_exhaustive: bool = Field(description="Whether the decomposition captures ALL factual assertions")
    missing_components: list[str] = Field(
        default_factory=list,
        description="List of missing factual assertions (empty if exhaustive)"
    )
    explanation: str = Field(description="Brief explanation of what's missing or why it's complete")


class ComponentEvaluation(BaseModel):
    """Evaluation result for a single claim component."""

    verdict: Literal["verified", "refuted", "unverified"] = Field(
        description=(
            "'verified' if evidence supports it, 'refuted' if evidence contradicts it, "
            "'unverified' if insufficient evidence exists."
        ),
    )
    reasoning: str = Field(
        description="Brief explanation of the verdict with source citations.",
    )
    # Metadata for tracking gap-filling
    initial_verdict: str | None = Field(
        default=None,
        description="Verdict from initial evaluation (before gap-filling)",
    )
    gap_filled: bool = Field(
        default=False,
        description="Whether this component triggered gap-filling",
    )
    verdict_changed: bool = Field(
        default=False,
        description="Whether gap-filling changed the verdict",
    )


class ExplanationResponse(BaseModel):
    """Intermediate response containing explanation and wording feedback (Step 1)."""

    explanation: str = Field(
        description="Detailed narrative analyzing the claim against search results with citations in [n] format.",
    )
    wording_feedback: str = Field(
        description="Guidance on whether the claims' wording in the original passage should be revised and why.",
    )


class VerdictResponse(BaseModel):
    """Final verdict response based on the explanation (Step 2)."""
    
    verdict: Literal["consistent", "inconsistent"] = Field(
        description=(
            "Overall determination based on the explanation. Use 'consistent' if the explanation shows "
            "the claim is supported by evidence, 'inconsistent' if the explanation shows evidence contradicts it."
        ),
    )


class InconsistencyReportResponse(BaseModel):
    """Structured response returned by the report generation model."""

    verdict: Literal["consistent", "inconsistent"] = Field(
        description=(
            "Overall determination of the investigated claim. Use 'consistent' if the claim is supported, "
            "'inconsistent' if at least one piece of evidence contradicts it."
        ),
    )
    wording_feedback: str = Field(
        description="Guidance on whether the claims' wording in the original passage should be revised and why.",
    )
    explanation: str = Field(
        description="Detailed narrative explaining the verdict with citations in [n] format.",
    )


class InconsistencyReport(InconsistencyReportResponse):
    """Used for returning the final report along with all search results."""

    claim_text: str = Field(
        description="The claim text that was investigated.",
    )
    claim_passage: str = Field(
        description="The passage from which the claim was extracted.",
    )
    search_results: list[Block] = Field(
        description="List of all search results retrieved during the investigation.",
    )
    components: list[str] = Field(
        default_factory=list,
        description="List of atomic components the claim was decomposed into.",
    )
    component_evaluations: list[ComponentEvaluation] = Field(
        default_factory=list,
        description="Evaluation results for each component.",
    )
    gap_fill_triggered: bool = Field(
        default=False,
        description="Whether gap-filling was triggered for any component.",
    )
    num_gap_fills: int = Field(
        default=0,
        description="Number of components that required gap-filling.",
    )


class InconsistencyAgentState(AgentState):
    """Runtime state for the inconsistency detection agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    passage: str
    claim: str
    all_search_results: list[Block]
    inconsistency_report: InconsistencyReportResponse | None
    claim_decomposition: ClaimDecomposition | None
    component_evaluations: list[ComponentEvaluation]


__all__ = [
    "InconsistencyAgentState",
    "InconsistencyReport",
    "InconsistencyReportResponse",
]
