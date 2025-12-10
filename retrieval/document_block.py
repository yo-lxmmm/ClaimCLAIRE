from datetime import datetime
import re
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)


class Block(BaseModel):
    """An indexing/retrieval unit. Can be a paragraph, list, linearized table, or linearized Infobox."""

    document_title: str = Field(..., description="The title of the document")
    section_title: str = Field(
        ...,
        description="The hierarchical section title of the block excluding `document_title`, e.g. 'Land > Central Campus'. Section title can be empty for web articles.",
    )
    content: str = Field(..., description="The content of the block, usually in Markdown format")
    last_edit_date: Optional[datetime] = Field(
        None,
        description="The last edit date of the block in the format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS",
    )
    url: Optional[str] = Field(None, description="The URL of the block")
    trust_rating: Optional[str] = Field(
        None, 
        description="Source reliability rating: 'reliable', 'mixed', or 'unreliable'"
    )
    trust_source: Optional[str] = Field(
        None,
        description="How trust rating was determined: 'list' or 'llm'"
    )
    trust_reason: Optional[str] = Field(
        None,
        description="Explanation for trust rating (only if from LLM)"
    )
    trust_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score for trust rating (0.0 to 1.0)"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",  # Disallow extra fields
    )

    @field_serializer("last_edit_date")
    def serialize_datetime(self, v: datetime):
        return v.strftime("%Y-%m-%d") if v else None

    @property
    def effective_weight(self) -> Optional[float]:
        """Compute trust_weight × confidence, or None if not available."""
        if not self.trust_rating:
            return None

        # Local import to avoid circular dependency at import time
        from claire_agent.inconsistency.prompts import TRUST_WEIGHTS

        trust_weight = TRUST_WEIGHTS.get(self.trust_rating, 0.5)
        confidence = self.trust_confidence if self.trust_confidence is not None else 0.75
        value = trust_weight * confidence

        # Clamp to [0, 1] for safety
        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return value

    @property
    def date_human_readable(self) -> Optional[str]:
        return self.last_edit_date.strftime("%B %d, %Y") if self.last_edit_date else None

    @property
    def full_title(self) -> str:
        if not self.section_title:
            return self.document_title
        return self.document_title + " > " + self.section_title

    @property
    def combined_text(self) -> str:
        return self.full_title + " " + self.content

    @property
    def id(self) -> int:
        return abs(hash(self.combined_text))

    @field_validator("last_edit_date", mode="before")
    def parse_last_edit_date(cls, last_edit_date):
        if isinstance(last_edit_date, str):
            try:
                return Block.convert_string_to_datetime(last_edit_date)
            except ValueError:
                raise ValueError(
                    f"Invalid date format for `last_edit_date`: '{last_edit_date}'. It should be in the format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ"
                )
        return last_edit_date

    @staticmethod
    def convert_string_to_datetime(date_string: str) -> datetime:
        """Convert a string to a datetime object.

        The string can be in the following formats:
        - YYYY-MM-DDTHH:MM:SSZ
        - YYYY-MM-DDTHH:MM:SS
        - YYYY-MM-DD HH:MM:SS
        - YYYY-MM-DD
        This conversion will ignore the timezone information.
        """
        date_formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
        ]
        for date_format in date_formats:
            try:
                return datetime.strptime(date_string, date_format)
            except ValueError:
                continue
        raise ValueError(
            f"Invalid date format for `last_edit_date`: '{date_string}'. It should be in the format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD HH:MM:SS"
        )

    @staticmethod
    def block_list_to_string(
        blocks: list["Block"], start_index: int = 1, truncate_text: bool = False
    ) -> str:
        def maybe_truncate(text: str) -> str:
            max_length = 200
            if not truncate_text or len(text) <= max_length:
                return text

            prefix = text[:max_length]
            cutoff = None
            for match in re.finditer(r"\s", prefix):
                cutoff = match.start()

            if cutoff is not None and cutoff > 0 and cutoff >= max_length // 2:
                truncated = text[:cutoff]
            else:
                truncated = prefix

            truncated = truncated.rstrip()
            if not truncated:
                truncated = prefix.rstrip() or prefix

            remaining_characters = len(text) - len(truncated)
            return f"{truncated} ... [{remaining_characters} more characters]"
        
        def format_trust_badge(block: "Block") -> str:
            """Format trust rating badge for display in search results with weight and confidence."""
            if not block.trust_rating:
                return ""
            
            rating_labels = {
                "reliable": "✓ Reliable Source",
                "mixed": "⚠ Mixed Reliability",
                "unreliable": "✗ Unreliable Source"
            }
            label = rating_labels.get(block.trust_rating, block.trust_rating.title())
            source_info = f" ({block.trust_source})" if block.trust_source else ""
            
            # Calculate and show weight
            from claire_agent.inconsistency.prompts import TRUST_WEIGHTS
            trust_weight = TRUST_WEIGHTS.get(block.trust_rating, 0.5)
            confidence = block.trust_confidence or 0.75
            combined_weight = trust_weight * confidence
            
            confidence_pct = int(confidence * 100)
            return f" [Trust: {label}{source_info}, Weight: {combined_weight:.2f} (confidence: {confidence_pct}%)]"

        # Group blocks by URL and assign numbers based on unique URLs
        url_to_number: dict[str, int] = {}
        url_order: list[str] = []
        current_number = start_index
        
        # First pass: assign numbers to unique URLs (non-empty URLs only)
        for b in blocks:
            url = b.url or ""
            if url and url not in url_to_number:
                url_to_number[url] = current_number
                url_order.append(url)
                current_number += 1
        
        # Track numbers for blocks without URLs
        no_url_counter = current_number
        
        # Second pass: format blocks with URL-based numbering
        result_lines = []
        for b in blocks:
            url = b.url or ""
            if url and url in url_to_number:
                # Block has a URL - use the URL's assigned number
                block_number = url_to_number[url]
            else:
                # Block has no URL - assign sequential number
                block_number = no_url_counter
                no_url_counter += 1
            
            trust_badge = format_trust_badge(b)
            result_lines.append(
                f"[{block_number}]{trust_badge} {maybe_truncate(b.combined_text)}"
            )
        
        return "\n\n".join(result_lines)
