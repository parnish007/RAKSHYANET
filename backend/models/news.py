"""
NewsEvent model — incoming disaster reports with severity and confidence scoring.
"""
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# Trusted sources trigger automatic re-optimization (Tier 1 whitelist)
TRUSTED_SOURCES: frozenset = frozenset({
    "@nepalpolice",
    "@nepalarmyofficial",
    "nepal army",
    "nepal police",
    "red cross",
    "icrc",
    "government sms",
    "ndrrma",                  # National Disaster Risk Reduction and Management Authority
    "kathmandu post",
    "my republica",
    "the himalayan times",
    "ekantipur",
})

# Keyword → severity weight (blueprint §2)
SEVERITY_KEYWORDS: Dict[str, float] = {
    "buried": 0.25,
    "collapse": 0.20,
    "collapsed": 0.20,
    "medical": 0.15,
    "landslide": 0.15,
    "critical": 0.20,
    "casualties": 0.25,
    "dead": 0.25,
    "deaths": 0.25,
    "injured": 0.15,
    "trapped": 0.20,
    "flood": 0.15,
    "evacuation": 0.10,
    "emergency": 0.10,
    "sos": 0.20,
    "rescue": 0.10,
}


class NewsEvent(BaseModel):
    id: str = Field(..., description="Unique event identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = Field(..., description="Source handle or publication name")
    text: str = Field(..., min_length=5, description="Raw news text")

    # Computed / set by RAG pipeline
    severity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    village_id: Optional[str] = Field(default=None, description="Linked Nepal village ID")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Verification tier result
    verification_tier: int = Field(default=0, ge=0, le=3, description="0=unverified, 1=trusted, 2=multi-source, 3=HITL")
    processed: bool = False

    model_config = {"arbitrary_types_allowed": True}

    # ---------------------------------------------------------------- #
    #  Methods                                                          #
    # ---------------------------------------------------------------- #

    def extract_severity_keywords(self) -> Dict[str, float]:
        """
        Scan text for severity keywords and return matched {keyword: weight} pairs.
        Severity score = sum of weights, capped at 1.0.
        """
        text_lower = self.text.lower()
        matched: Dict[str, float] = {}
        for keyword, weight in SEVERITY_KEYWORDS.items():
            if keyword in text_lower:
                matched[keyword] = weight
        return matched

    def computed_severity(self) -> float:
        """Re-derive severity from text keywords (for validation / re-scoring)."""
        return min(1.0, sum(self.extract_severity_keywords().values()))

    def is_trusted_source(self) -> bool:
        """Return True if source is on Tier-1 whitelist."""
        return self.source.lower().strip() in TRUSTED_SOURCES

    def compute_confidence(
        self,
        multi_source_confirmed: bool = False,
        keyword_count: Optional[int] = None,
        hours_since_event: float = 0.0,
    ) -> float:
        """
        Confidence formula (blueprint §1):
          0.4 * source_tier + 0.3 * multi_source + 0.2 * keywords + 0.1 * temporal

        Returns value in [0, 1].
        """
        source_tier = 1.0 if self.is_trusted_source() else 0.3
        multi = 1.0 if multi_source_confirmed else 0.0
        kw_count = keyword_count if keyword_count is not None else len(self.extract_severity_keywords())
        keyword_score = min(1.0, kw_count / 3.0)  # normalise: 3+ keywords → full score
        # temporal: freshness decays after 1 hour
        import math
        temporal = math.exp(-hours_since_event)

        return round(
            0.4 * source_tier + 0.3 * multi + 0.2 * keyword_score + 0.1 * temporal,
            4
        )

    @property
    def auto_optimize(self) -> bool:
        """True if confidence is high enough to trigger automatic re-optimization."""
        return self.confidence_score >= 0.8

    @property
    def requires_hitl(self) -> bool:
        """True if event needs Human-in-the-Loop approval."""
        return 0.5 <= self.confidence_score < 0.8

    def __repr__(self) -> str:
        return (
            f"NewsEvent(source={self.source!r}, village={self.village_id}, "
            f"severity={self.severity_score:.2f}, confidence={self.confidence_score:.2f})"
        )
