"""Data models for timeline analysis."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field as PydanticField


@dataclass
class Transition:
    """Represents a detected verdict change."""
    from_date: str              # "2022-05-15"
    to_date: str                # "2022-06-20"
    from_verdict: str           # "consistent", "inconsistent", "unknown"
    to_verdict: str             # "consistent", "inconsistent", "unknown"
    from_sources_count: int
    to_sources_count: int
    from_reliable_count: int
    to_reliable_count: int
    confidence: float = 1.0     # Confidence in this transition (0-1)
    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TimelinePeriod:
    """Represents a stable period between transitions."""
    start_date: str
    end_date: str
    verdict: str                # "consistent", "inconsistent", "unknown"
    num_sources: int
    reliable_sources: int
    mixed_sources: int = 0
    unreliable_sources: int = 0
    representative_sources: List[Any] = None

    def __post_init__(self):
        if self.representative_sources is None:
            self.representative_sources = []

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Don't include full source objects (too large)
        data['representative_sources'] = [
            {
                'title': s.document_title if hasattr(s, 'document_title') else str(s),
                'url': s.url if hasattr(s, 'url') else '',
                'trust_rating': s.trust_rating if hasattr(s, 'trust_rating') else 'mixed'
            }
            for s in self.representative_sources[:3]  # Only top 3
        ]
        return data


@dataclass
class TimelineAnalysis:
    """Complete timeline analysis result."""
    claim: str
    time_range: tuple
    transitions: List[Transition]
    periods: List[TimelinePeriod]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'claim': self.claim,
            'time_range': {
                'start': self.time_range[0],
                'end': self.time_range[1]
            },
            'transitions': [t.to_dict() for t in self.transitions],
            'periods': [p.to_dict() for p in self.periods],
            'metadata': self.metadata
        }


# ============================================
# Enhanced Timeline Models (Event-Driven)
# ============================================

class ExtractedEvent(BaseModel):
    """Event extracted from search results by LLM."""
    date: str = PydanticField(description="Event date in YYYY-MM-DD or fuzzy format like 'August 2023'")
    event_type: str = PydanticField(description="Type: claim_origin, fact_check, new_evidence, official_update, or retraction")
    description: str = PydanticField(description="Brief description of the event")
    source_title: str = PydanticField(description="Title of the source")
    source_url: str = PydanticField(description="URL of the source")
    confidence: float = PydanticField(description="Confidence in the date (0-1)")


class EventExtraction(BaseModel):
    """Result of event extraction from search results."""
    events: List[ExtractedEvent] = PydanticField(default_factory=list, description="List of extracted events")


class KeyEvent(BaseModel):
    """Filtered and prioritized key event."""
    date: str = PydanticField(description="Normalized date in YYYY-MM-DD format")
    event_type: str = PydanticField(description="Event type")
    description: str = PydanticField(description="Event description")
    likely_changes_verdict: bool = PydanticField(description="Whether this event likely changes the verdict")
    priority: int = PydanticField(description="Priority from 1 (must check) to 5 (optional)")


class FilteredTimeline(BaseModel):
    """Result of event filtering and prioritization."""
    key_events: List[KeyEvent] = PydanticField(default_factory=list, description="List of key events")
    reasoning: str = PydanticField(description="Why these events were selected")


@dataclass
class TimelineEvent:
    """A single event in the enhanced timeline with verdict and explanation."""
    date: str
    event_description: str
    event_type: str
    verdict: str  # "consistent" | "inconsistent" | "unverified"
    num_sources: int
    reliable_count: int
    mixed_count: int
    unreliable_count: int
    key_sources: List[Dict[str, Any]] = field(default_factory=list)  # Top 3 sources
    component_evaluations: List[Any] = field(default_factory=list)
    explanation: Optional[str] = None  # LLM-generated contextual explanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date,
            'event_description': self.event_description,
            'event_type': self.event_type,
            'verdict': self.verdict,
            'num_sources': self.num_sources,
            'reliable_count': self.reliable_count,
            'mixed_count': self.mixed_count,
            'unreliable_count': self.unreliable_count,
            'key_sources': self.key_sources,
            'explanation': self.explanation,
            'component_count': len(self.component_evaluations)
        }


class TimelineSummary(BaseModel):
    """Overall narrative summary of the timeline."""
    category: str = PydanticField(description="Category: evolving_fact, debunking_journey, stable_truth, stable_falsehood, or narrative_mutation")
    overall_summary: str = PydanticField(description="3-4 sentence narrative summary")
    current_status: str = PydanticField(description="Current verdict with brief explanation")
    key_insight: str = PydanticField(description="One-line takeaway")


@dataclass
class TimelineNarrative:
    """Complete event-driven timeline narrative."""
    claim: str
    time_range: tuple  # (start_date, end_date)
    category: str
    events: List[TimelineEvent]
    summary: str
    key_insight: str
    current_status: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'claim': self.claim,
            'time_range': {
                'start': self.time_range[0],
                'end': self.time_range[1]
            },
            'category': self.category,
            'events': [e.to_dict() for e in self.events],
            'summary': self.summary,
            'key_insight': self.key_insight,
            'current_status': self.current_status,
            'metadata': self.metadata
        }
