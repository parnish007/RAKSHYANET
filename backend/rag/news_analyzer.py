"""
News Analyzer -- Prompt 3.1

Rule-based RAG pipeline for analyzing disaster news.
No LLM API calls in this demo implementation.
In production: replace extract_structured_data() with a GPT-4
structured-output call for higher extraction accuracy.

Pipeline:
  1. extract_structured_data  -- regex + keyword parse
  2. calculate_confidence     -- weighted confidence score
  3. identify_affected_villages -- fuzzy village matching
  4. assess_severity          -- integer severity 0-10
  5. determine_action         -- AUTO_OPTIMIZE | HITL_REQUIRED | IGNORE
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from backend.models.village import Village


# ------------------------------------------------------------------ #
#  Action constants                                                    #
# ------------------------------------------------------------------ #

ACTION_AUTO_OPTIMIZE = "AUTO_OPTIMIZE"
ACTION_HITL_REQUIRED = "HITL_REQUIRED"
ACTION_IGNORE        = "IGNORE"


# ------------------------------------------------------------------ #
#  Source reliability lookup                                           #
# ------------------------------------------------------------------ #

# Substring tokens that identify official / NGO / news sources
_OFFICIAL_TOKENS = frozenset({
    "police", "army", "government", "ndrrma", "ministry",
    "municipality", "district administration", "official",
    "nepal army", "nepal police",
})
_NGO_TOKENS = frozenset({
    "redcross", "red cross", "icrc", "oxfam", "unicef", "who",
    "save the children", "action aid", "care nepal",
})
_NEWS_TOKENS = frozenset({
    "kathmandu post", "my republica", "himalayan times", "ekantipur",
    "republica", "setopati", "onlinekhabar", "ratopati",
})


# ------------------------------------------------------------------ #
#  Event-type keyword patterns                                         #
# ------------------------------------------------------------------ #

_EVENT_PATTERNS: Dict[str, List[str]] = {
    "landslide":       ["landslide", "mudslide", "debris flow", "rockslide", "soil erosion"],
    "earthquake":      ["earthquake", "tremor", "seismic", "quake", "aftershock"],
    "flood":           ["flash flood", "flood", "inundation", "submerged", "overflowing"],
    "bridge_collapse": ["bridge collapse", "bridge collapsed", "bridge failure"],
    "fire":            ["wildfire", "forest fire", "blaze"],
}


# ------------------------------------------------------------------ #
#  Resource keyword mapping                                            #
# ------------------------------------------------------------------ #

_RESOURCE_KEYWORDS: Dict[str, List[str]] = {
    "medical_kit":      ["medical", "medicine", "hospital", "doctor", "nurse",
                         "injured", "casualt", "health", "wounded", "airlift"],
    "food":             ["food", "hunger", "starvation", "supplies", "relief", "ration"],
    "water":            ["water", "drinking", "sanitation", "dehydrat"],
    "blanket":          ["shelter", "homeless", "displaced", "blanket", "cold"],
    "rescue_equipment": ["trapped", "buried", "rescue", "debris", "collapsed"],
    "tarpaulin":        ["tarpaulin", "tent", "shelter", "displaced"],
    "first_aid":        ["first aid", "wound", "bleed", "fracture"],
}


# ------------------------------------------------------------------ #
#  District → village-ID mapping (Nepal demo data)                    #
# ------------------------------------------------------------------ #

_DISTRICT_MAP: Dict[str, List[str]] = {
    "kavre":          ["dhulikhel", "panauti", "banepa", "namobuddha",
                       "panchkhal", "temal", "bethanchowk", "khopasi"],
    "kavrepalanchok": ["dhulikhel", "panauti", "banepa", "namobuddha",
                       "panchkhal", "temal", "bethanchowk", "khopasi"],
    "sindhupalchok":  [],
    "dolakha":        [],
    "ramechhap":      [],
}


# ------------------------------------------------------------------ #
#  Urgency / severity vocabulary                                       #
# ------------------------------------------------------------------ #

_IMMEDIATE_WORDS = frozenset({
    "immediate", "urgent", "emergency", "critical", "sos", "right away",
    "life-threatening", "life threatening",
})
_INFRA_DAMAGE_PHRASES = frozenset({
    "road blocked", "road closed", "road cut", "bridge collapsed",
    "bridge collapse", "bridge failure", "road inaccessible",
})
_HOSPITAL_DAMAGE_PHRASES = frozenset({
    "hospital damaged", "clinic buried", "medical facility",
    "health post destroyed", "hospital overwhelmed",
})


# ------------------------------------------------------------------ #
#  Location extraction stop-words (common non-place capitalised words) #
# ------------------------------------------------------------------ #

_LOCATION_STOP: frozenset = frozenset({
    "Breaking", "Ndrrma", "Major", "Flash", "Source", "Multiple",
    "Search", "Critical", "Nepal", "District", "Area", "Alert",
    "Communities", "Rescue", "Teams", "Hospital", "Medical",
    "Road", "Bridge", "Warning", "Advisory", "Issued", "Confirmed",
    "Need", "Help", "Urgent", "Emergency", "People", "Families",
    "Official", "Twitter", "Report", "News", "Media", "Police",
    "Army", "Government", "Ministry", "Local", "National",
    "Relief", "Operation", "Support", "Access", "Food", "Water",
    "Supply", "Transport", "Affected", "Isolated", "Displaced",
    "Unconfirmed", "Landslide", "Earthquake", "Flood", "Fire",
    "Mudslide", "Heard", "Sure", "Not", "Their", "This", "That",
    "With", "From", "Into", "Over", "Upon", "Near", "Also", "Then",
    "Both", "While", "After", "Before", "Since", "Under", "Above",
    "Flash", "Airlift", "Personnel", "Team", "Crew", "Unit",
    "Health", "Care", "Post", "Home", "House", "Village", "Town",
    "City", "State", "Country", "Region", "Zone", "Ward",
    "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "Monday", "January", "February", "March", "April",
    "May", "June", "July", "August", "September", "October",
    "November", "December",
})


# ------------------------------------------------------------------ #
#  Resource implications per event type                               #
# ------------------------------------------------------------------ #

_EVENT_RESOURCE_BASE: Dict[str, Dict[str, float]] = {
    "landslide":       {"medical_kit": 30.0, "rescue_equipment": 20.0, "food": 10.0},
    "earthquake":      {"medical_kit": 40.0, "food": 30.0, "blanket": 20.0, "rescue_equipment": 30.0},
    "flood":           {"water": 25.0, "food": 25.0, "blanket": 20.0, "tarpaulin": 15.0},
    "bridge_collapse": {},   # Routing impact only
    "fire":            {"medical_kit": 15.0, "blanket": 10.0, "tarpaulin": 10.0},
}
_RESOURCE_EXTRA: Dict[str, float] = {
    "medical_kit": 15.0, "food": 10.0, "water": 10.0,
    "blanket": 8.0, "rescue_equipment": 12.0, "tarpaulin": 8.0, "first_aid": 10.0,
}


# ================================================================== #
#  Output models                                                       #
# ================================================================== #

class NewsEvent(BaseModel):
    """Structured representation of a single analyzed news event."""
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}")
    timestamp: str = ""
    source: str = ""
    raw_text: str
    location: List[str] = Field(default_factory=list)
    severity: int = Field(default=0, ge=0, le=10)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    affected_villages: List[str] = Field(default_factory=list)
    resource_implications: Dict[str, float] = Field(default_factory=dict)
    requires_hitl: bool = False


class IntelligenceReport(BaseModel):
    """Full output of one news-analysis run."""
    event: NewsEvent
    analysis_summary: str = ""
    urgency_change: Dict[str, float] = Field(default_factory=dict)
    recommended_action: str = ACTION_IGNORE
    confidence_reasoning: str = ""


# ================================================================== #
#  NewsAnalyzer                                                        #
# ================================================================== #

class NewsAnalyzer:
    """
    Rule-based RAG news analyzer (hackathon demo — no LLM API calls).

    In production, replace extract_structured_data() with a GPT-4
    structured-output call for higher accuracy.

    Args:
        model:                 LLM model name (kept for production parity).
        confidence_thresholds: Dict with 'high' (default 0.8) and
                               'medium' (default 0.5) boundaries.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        confidence_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.model = model  # In production: call GPT-4 here
        self.confidence_thresholds = confidence_thresholds or {
            "high":   0.80,
            "medium": 0.50,
        }

    # -------------------------------------------------------------- #
    #  Step 1: Extract structured data                                #
    # -------------------------------------------------------------- #

    def extract_structured_data(self, raw_text: str) -> Dict:
        """
        Parse raw news text using regex + keyword matching.

        In production: call GPT-4 with structured-output schema here.

        Returns dict with keys:
            event_type, location, casualties, infrastructure_damage,
            hospital_damage, resources_needed, time_constraint,
            source_raw, source_reliability, location_specificity,
            detail_completeness, cross_validation.
        """
        text_lower = raw_text.lower()

        # ---- event type ------------------------------------------ #
        event_type = "unknown"
        for etype, keywords in _EVENT_PATTERNS.items():
            if any(kw in text_lower for kw in keywords):
                event_type = etype
                break

        # ---- location extraction ---------------------------------- #
        # Capture capitalised phrases (single or multi-word)
        raw_mentions: List[str] = re.findall(
            r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b", raw_text
        )
        location_mentions = [m for m in raw_mentions if m not in _LOCATION_STOP]

        # ---- casualty extraction ---------------------------------- #
        casualties = 0
        # Pattern: "15 people injured", "50 dead", "12 families isolated"
        for pattern in (
            r"(\d+)\s+(?:people?|persons?|individuals?)\s*(?:injured|killed|dead|trapped|missing)",
            r"(\d+)\s+(?:injured|killed|dead|missing|trapped|casualties|deaths)",
            r"(\d+)\s+(?:families|households|residents|villagers)\s+(?:isolated|affected|displaced)",
        ):
            m = re.search(pattern, text_lower)
            if m:
                val = int(m.group(1))
                # families → approximate 10 persons each
                if "famil" in pattern or "household" in pattern:
                    val = val // 10
                casualties = max(casualties, val)

        # ---- infrastructure damage -------------------------------- #
        infra_damage = ""
        for phrase in _INFRA_DAMAGE_PHRASES:
            if phrase in text_lower:
                infra_damage = phrase
                break
        hospital_damage = any(phrase in text_lower for phrase in _HOSPITAL_DAMAGE_PHRASES)

        # ---- resources needed ------------------------------------ #
        resources_needed: List[str] = []
        for resource, keywords in _RESOURCE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                resources_needed.append(resource)

        # ---- time constraint ------------------------------------- #
        time_constraint = "72h"
        if any(w in text_lower for w in _IMMEDIATE_WORDS):
            time_constraint = "immediate"
        elif any(p in text_lower for p in ("24 hour", "24h", "by tomorrow", "tonight")):
            time_constraint = "24h"

        # ---- source identification ------------------------------- #
        source_raw = ""
        src_m = re.search(r"source[:\s]+([^\.\n]{4,60})", raw_text, re.IGNORECASE)
        if src_m:
            source_raw = src_m.group(1).strip()
        handle_m = re.search(r"@(\w+)", raw_text)
        if handle_m and not source_raw:
            source_raw = f"@{handle_m.group(1)}"

        source_reliability   = self._classify_source(source_raw, raw_text)
        location_specificity = self._assess_location_specificity(location_mentions)

        # ---- detail completeness --------------------------------- #
        fields_filled = sum([
            event_type != "unknown",
            bool(location_mentions),
            casualties > 0,
            bool(resources_needed),
            time_constraint != "72h",
        ])
        detail_completeness = fields_filled / 5.0

        return {
            "event_type":            event_type,
            "location":              location_mentions,
            "casualties":            casualties,
            "infrastructure_damage": infra_damage,
            "hospital_damage":       hospital_damage,
            "resources_needed":      resources_needed,
            "time_constraint":       time_constraint,
            "source_raw":            source_raw,
            "source_reliability":    source_reliability,
            "location_specificity":  location_specificity,
            "detail_completeness":   detail_completeness,
            "cross_validation":      0.0,   # caller overrides for multi-source
        }

    # -------------------------------------------------------------- #
    #  Source and location helpers                                    #
    # -------------------------------------------------------------- #

    def _classify_source(self, source_raw: str, full_text: str) -> float:
        """Reliability score 0-1 based on source classification."""
        combined = (source_raw + " " + full_text).lower()
        if any(tok in combined for tok in _OFFICIAL_TOKENS):
            return 1.0
        if any(tok in combined for tok in _NGO_TOKENS):
            return 0.9
        if any(tok in combined for tok in _NEWS_TOKENS):
            return 0.8
        if "@" in source_raw or "twitter" in combined or "facebook" in combined:
            return 0.4
        if not source_raw:
            return 0.2
        return 0.5   # named but unrecognised

    def _assess_location_specificity(self, location_mentions: List[str]) -> float:
        """
        Return how specific the location information is:
          1.0 – a named non-district location (e.g. specific village / town)
          0.5 – only district-level mention (e.g. Kavre)
          0.2 – vague / no useful location
          0.1 – no mentions at all
        """
        if not location_mentions:
            return 0.1
        mentions_lower = [m.lower() for m in location_mentions]
        # Check if any mention is NOT a district and has ≥ 4 chars
        for m in mentions_lower:
            if m not in _DISTRICT_MAP and len(m) >= 4:
                return 1.0
        # Only district-level mentions
        if any(m in _DISTRICT_MAP for m in mentions_lower):
            return 0.5
        return 0.2

    # -------------------------------------------------------------- #
    #  Step 2: Calculate confidence                                   #
    # -------------------------------------------------------------- #

    def calculate_confidence(self, extracted_data: Dict) -> float:
        """
        confidence = loc*0.30 + src*0.25 + detail*0.25 + cross*0.20

        All inputs are normalised [0, 1].
        Result is rounded to 4 decimal places and clamped to [0, 1].
        """
        loc   = extracted_data.get("location_specificity", 0.0)
        src   = extracted_data.get("source_reliability",   0.0)
        det   = extracted_data.get("detail_completeness",  0.0)
        cross = extracted_data.get("cross_validation",     0.0)

        conf = loc * 0.30 + src * 0.25 + det * 0.25 + cross * 0.20
        return round(min(1.0, max(0.0, conf)), 4)

    # -------------------------------------------------------------- #
    #  Step 3: Identify affected villages                             #
    # -------------------------------------------------------------- #

    def identify_affected_villages(
        self,
        location_mentions: List[str],
        villages: List[Village],
        fuzzy_threshold: float = 0.80,
    ) -> List[str]:
        """
        Return village IDs whose names match any extracted location mention.

        Matching order:
          1. District-level: Kavre → all Kavre villages in the provided list
          2. Exact case-insensitive name or ID match
          3. Fuzzy match via SequenceMatcher (threshold 0.80)
          4. Containment match (one is substring of the other, length ≥ 4)
        """
        village_id_set = {v.id for v in villages}
        affected: List[str] = []

        for mention in location_mentions:
            mention_lower = mention.lower().strip()

            # 1. District-level match
            for district, district_vids in _DISTRICT_MAP.items():
                if mention_lower == district:
                    for vid in district_vids:
                        if vid in village_id_set and vid not in affected:
                            affected.append(vid)

            # 2-4. Per-village matching
            for village in villages:
                vid        = village.id
                vname_low  = village.name.lower()

                if vid in affected:
                    continue

                # Exact match (name or id)
                if mention_lower in (vname_low, vid):
                    affected.append(vid)
                    continue

                # Fuzzy match
                ratio = SequenceMatcher(None, mention_lower, vname_low).ratio()
                if ratio >= fuzzy_threshold:
                    affected.append(vid)
                    continue

                # Containment match (at least 4 chars to reduce false positives)
                if len(mention_lower) >= 4 and (
                    mention_lower in vname_low or vname_low in mention_lower
                ):
                    affected.append(vid)

        return affected

    # -------------------------------------------------------------- #
    #  Step 4: Assess severity                                        #
    # -------------------------------------------------------------- #

    def assess_severity(self, extracted_data: Dict) -> int:
        """
        Compute integer severity 0-10:

          Casualties component  (max 4)
            > 50 → +4 | > 10 → +2 | > 0 → +1
          Infrastructure damage (max 3)
            road/bridge blocked → +3
          Hospital damage       (max 2)
            clinic/hospital hit → +2
          Time constraint       (max 2)
            immediate → +2 | 24h → +1
          Resources needed      (+1 if any detected)

        Result capped at 10.
        """
        score = 0

        casualties = extracted_data.get("casualties", 0)
        if casualties > 50:
            score += 4
        elif casualties > 10:
            score += 2
        elif casualties > 0:
            score += 1

        if extracted_data.get("infrastructure_damage"):
            score += 3
        if extracted_data.get("hospital_damage"):
            score += 2

        tc = extracted_data.get("time_constraint", "72h")
        if tc == "immediate":
            score += 2
        elif tc == "24h":
            score += 1

        if extracted_data.get("resources_needed"):
            score += 1

        return min(10, score)

    # -------------------------------------------------------------- #
    #  Step 5: Determine action                                       #
    # -------------------------------------------------------------- #

    def determine_action(self, confidence: float, severity: int) -> str:
        """
        confidence >= 0.8        → AUTO_OPTIMIZE
        0.5 <= confidence < 0.8  → HITL_REQUIRED
        confidence < 0.5         → IGNORE
        """
        high   = self.confidence_thresholds.get("high",   0.80)
        medium = self.confidence_thresholds.get("medium", 0.50)

        if confidence >= high:
            return ACTION_AUTO_OPTIMIZE
        elif confidence >= medium:
            return ACTION_HITL_REQUIRED
        else:
            return ACTION_IGNORE

    # -------------------------------------------------------------- #
    #  Helpers for report building                                    #
    # -------------------------------------------------------------- #

    def _build_resource_implications(
        self, extracted_data: Dict, event_type: str
    ) -> Dict[str, float]:
        """Map event type + detected resource needs to quantity deltas."""
        implications: Dict[str, float] = {}
        implications.update(_EVENT_RESOURCE_BASE.get(event_type, {}))
        for res in extracted_data.get("resources_needed", []):
            if res not in implications:
                implications[res] = _RESOURCE_EXTRA.get(res, 5.0)
        return implications

    def _build_summary(
        self,
        event_type: str,
        locations: List[str],
        confidence: float,
        severity: int,
        action: str,
        casualties: int,
    ) -> str:
        loc_str  = ", ".join(locations[:3]) if locations else "unknown location"
        cas_str  = f"{casualties} casualties." if casualties else "No confirmed casualty count."
        tiers    = {ACTION_AUTO_OPTIMIZE: "High", ACTION_HITL_REQUIRED: "Medium", ACTION_IGNORE: "Low"}
        tier_str = tiers.get(action, "Low")
        return (
            f"{tier_str}-confidence {event_type} event detected in {loc_str}. "
            f"{cas_str} Severity {severity}/10. Action: {action}."
        )

    # -------------------------------------------------------------- #
    #  Public API                                                     #
    # -------------------------------------------------------------- #

    def analyze_news(
        self,
        raw_text: str,
        villages: List[Village],
        source: str = "",
        multi_source_confirmed: bool = False,
    ) -> IntelligenceReport:
        """
        Full 5-step RAG pipeline for a single news item.

        Args:
            raw_text:              Raw news text to analyze.
            villages:              Known villages for location matching.
            source:                Optional explicit source identifier.
                                   Overrides source extracted from text.
            multi_source_confirmed: Set True if multiple independent sources
                                    confirmed this event (adds 0.20 to confidence).

        Returns:
            IntelligenceReport with recommended action and full detail.
        """
        # Step 1
        extracted = self.extract_structured_data(raw_text)
        if source:
            extracted["source_raw"]         = source
            extracted["source_reliability"] = self._classify_source(source, raw_text)
        if multi_source_confirmed:
            extracted["cross_validation"] = 1.0

        # Step 2
        confidence = self.calculate_confidence(extracted)

        # Step 3
        affected = self.identify_affected_villages(extracted["location"], villages)

        # Step 4
        severity = self.assess_severity(extracted)

        # Step 5
        action = self.determine_action(confidence, severity)

        # Build output
        implications = self._build_resource_implications(extracted, extracted["event_type"])

        event = NewsEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=extracted.get("source_raw") or source,
            raw_text=raw_text,
            location=extracted["location"],
            severity=severity,
            confidence=confidence,
            affected_villages=affected,
            resource_implications=implications,
            requires_hitl=(action == ACTION_HITL_REQUIRED),
        )

        urgency_change = {vid: float(severity) / 10.0 for vid in affected}

        reasoning = (
            f"loc_specificity={extracted['location_specificity']:.2f}*0.30 + "
            f"src_reliability={extracted['source_reliability']:.2f}*0.25 + "
            f"detail={extracted['detail_completeness']:.2f}*0.25 + "
            f"cross_val={extracted['cross_validation']:.2f}*0.20 "
            f"=> conf={confidence:.4f}"
        )

        summary = self._build_summary(
            event_type=extracted["event_type"],
            locations=extracted["location"],
            confidence=confidence,
            severity=severity,
            action=action,
            casualties=extracted["casualties"],
        )

        return IntelligenceReport(
            event=event,
            analysis_summary=summary,
            urgency_change=urgency_change,
            recommended_action=action,
            confidence_reasoning=reasoning,
        )
