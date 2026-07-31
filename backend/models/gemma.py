"""Strict contracts for evidence-grounded, non-authoritative Gemma analysis."""
from __future__ import annotations

from typing import List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.models.optimization import utc_now


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EvidenceRecord(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9._:/-]+$")
    source_category: str = Field(min_length=1, max_length=100)
    source_name: str = Field(min_length=1, max_length=200)
    source_identifier: str = Field(min_length=1, max_length=500)
    retrieved_at: str
    freshness_minutes: int = Field(ge=0)
    reliability: float = Field(ge=0.0, le=1.0)
    text: str = Field(min_length=1, max_length=20_000)
    provider: str = "deterministic_demo_fixture"
    cache_status: str = "fixture"
    error_status: Optional[str] = None
    simulated: bool = True
    operator_context: Optional[str] = Field(default=None, max_length=2000)
    gap_target: Optional[str] = Field(default=None, max_length=200)
    reported_latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    reported_longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)

    @model_validator(mode="after")
    def validate_reported_location(self) -> "EvidenceRecord":
        if (self.reported_latitude is None) != (self.reported_longitude is None):
            raise ValueError("Reported latitude and longitude must be supplied together")
        return self


class GroundedValue(StrictModel):
    value: Optional[str] = Field(default=None, max_length=100)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: List[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_unknown_contract(self) -> "GroundedValue":
        if self.value is None and (self.confidence != 0 or self.evidence_ids):
            raise ValueError("UNKNOWN values require confidence=0 and no evidence IDs")
        if self.value is not None and not self.evidence_ids:
            raise ValueError("Inferred values require at least one evidence ID")
        return self


class GroundedScore(StrictModel):
    value: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: List[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_unknown_contract(self) -> "GroundedScore":
        if self.value is None and (self.confidence != 0 or self.evidence_ids):
            raise ValueError("UNKNOWN scores require confidence=0 and no evidence IDs")
        if self.value is not None and not self.evidence_ids:
            raise ValueError("Inferred scores require at least one evidence ID")
        return self


class GroundedRange(StrictModel):
    min: Optional[float] = None
    expected: Optional[float] = None
    max: Optional[float] = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: List[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_range_contract(self) -> "GroundedRange":
        values = (self.min, self.expected, self.max)
        if all(value is None for value in values):
            if self.confidence != 0 or self.evidence_ids:
                raise ValueError("UNKNOWN ranges require confidence=0 and no evidence IDs")
            return self
        if any(value is None for value in values):
            raise ValueError("Ranges must provide min, expected, and max together")
        if not self.evidence_ids:
            raise ValueError("Inferred ranges require at least one evidence ID")
        if not self.min <= self.expected <= self.max:
            raise ValueError("Range must satisfy min <= expected <= max")
        return self


class Contradiction(StrictModel):
    claim_a: str = Field(min_length=1, max_length=500)
    claim_b: str = Field(min_length=1, max_length=500)
    evidence_ids: List[str] = Field(min_length=1, max_length=20)


class GemmaStructuredOutput(StrictModel):
    incident_type: GroundedValue
    severity: GroundedRange
    affected_population: GroundedRange
    medical_urgency: GroundedScore
    accessibility_risk: GroundedScore
    contradictions: List[Contradiction] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list, max_length=10)
    needs_more_evidence: bool
    needs_human_review: bool
    requested_tools: List[str] = Field(default_factory=list, max_length=10)
    summary: str = Field(min_length=1, max_length=1_200)

    @field_validator("missing_information")
    @classmethod
    def validate_missing_information(cls, values: List[str]) -> List[str]:
        if any(not value.strip() or len(value) > 300 for value in values):
            raise ValueError("Missing-information entries must be concise")
        return list(dict.fromkeys(values))

    @field_validator("requested_tools")
    @classmethod
    def unique_requested_tools(cls, values: List[str]) -> List[str]:
        return list(dict.fromkeys(values))

    @field_validator("follow_up_questions")
    @classmethod
    def validate_follow_up_questions(cls, values: List[str]) -> List[str]:
        if any(not value.strip() or len(value) > 300 for value in values):
            raise ValueError("Follow-up questions must be concise")
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def validate_domain_ranges(self) -> "GemmaStructuredOutput":
        severity_values = (self.severity.min, self.severity.expected, self.severity.max)
        if any(value is not None and not 0 <= value <= 1 for value in severity_values):
            raise ValueError("Severity range values must be normalized to [0, 1]")
        population_values = (
            self.affected_population.min,
            self.affected_population.expected,
            self.affected_population.max,
        )
        if any(value is not None and (value < 0 or not float(value).is_integer())
               for value in population_values):
            raise ValueError("Affected-population values must be non-negative whole numbers")
        grounded_fields = (
            self.incident_type,
            self.severity,
            self.affected_population,
            self.medical_urgency,
            self.accessibility_risk,
        )
        has_unknown = any(
            getattr(field, "value", getattr(field, "expected", None)) is None
            for field in grounded_fields
        )
        has_unresolved_evidence = bool(
            has_unknown
            or self.contradictions
            or self.missing_information
            or self.requested_tools
        )
        if self.missing_information and not self.follow_up_questions:
            self.follow_up_questions = [
                f"Can you provide or verify: {item}?"
                for item in self.missing_information
            ]
        if has_unresolved_evidence and not self.needs_more_evidence:
            raise ValueError(
                "Unknowns, contradictions, gaps, or tool requests require more evidence"
            )
        if (has_unknown or self.contradictions) and not self.needs_human_review:
            raise ValueError("Unknown or contradictory material requires human review")
        return self


class DecisionTraceStep(StrictModel):
    step_id: str
    title: str
    status: str = "completed"
    input_ids: List[str] = Field(default_factory=list)
    output_summary: str
    duration_ms: float = Field(ge=0.0)
    warnings: List[str] = Field(default_factory=list)


class EvidenceQuestionDisposition(StrictModel):
    question_id: str = Field(min_length=1, max_length=120, pattern=r"^question-\d+$")
    question: str = Field(min_length=1, max_length=1000)
    status: Literal["assigned", "unavailable"]
    owner: str = Field(min_length=2, max_length=120)
    reason: str = Field(min_length=8, max_length=1000)
    recorded_at: str = Field(default_factory=utc_now)


class EvidenceQuestionDispositionRequest(StrictModel):
    status: Literal["assigned", "unavailable"]
    owner: str = Field(min_length=2, max_length=120)
    reason: str = Field(min_length=8, max_length=1000)


class GemmaAnalysisRecord(StrictModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        protected_namespaces=(),
    )

    analysis_id: str = Field(default_factory=lambda: f"gemma_{uuid4().hex[:12]}")
    scenario_id: str
    correlation_id: Optional[str] = None
    created_at: str = Field(default_factory=utc_now)
    provider: str = "mock_deterministic"
    model: str = "gemma-demo-fallback"
    model_version: str = "1.0"
    prompt_version: str = "nepal-grounded-extraction-v3"
    temperature: float = 0.0
    fixture_notice: str
    evidence: List[EvidenceRecord]
    output: GemmaStructuredOutput
    model_confidence: float = Field(ge=0.0, le=1.0)
    system_confidence: float = Field(ge=0.0, le=1.0)
    trace_steps: List[DecisionTraceStep]
    requested_tools: List[str] = Field(default_factory=list)
    question_dispositions: List[EvidenceQuestionDisposition] = Field(default_factory=list)
    termination_reason: str

    # -- Raw model exchange, unabstracted ---------------------------------
    #
    # Everything above is validated, bounded, and safe to act on. The three
    # fields below are the opposite: they are the untouched wire content of the
    # exchange with the model, exposed so a reader can audit what was actually
    # sent and returned rather than trusting this record's summary of it.
    #
    # None of it is validated and none of it may influence any decision. In
    # particular `model_reasoning` holds the provider's own `thought` parts,
    # which are the model's deliberation, NOT a grounded claim: it is not
    # citation-checked, so treating it as a finding would bypass every validator
    # in the pipeline. It is shown for transparency and nothing else.
    prompt_sent: Optional[str] = Field(default=None, max_length=60000)
    model_reasoning: List[str] = Field(default_factory=list)
    raw_response_text: Optional[str] = Field(default=None, max_length=60000)

    # Whether the provider signalled that it produced internal reasoning, and
    # how many tokens it spent on it. Measured against gemma-4-26b-a4b-it, the
    # response marks a thought part but returns an EMPTY body for it: the model
    # reports that it deliberated without exposing the deliberation. So
    # `thinking_reported` can be true while `model_reasoning` is empty, and that
    # is an honest description of the provider rather than a bug. The interface
    # must say so instead of inventing reasoning text.
    thinking_reported: bool = False
    thinking_token_count: Optional[int] = Field(default=None, ge=0)


class GemmaAnalysisRequest(StrictModel):
    scenario_id: str = Field(default="nepal-national-demo", min_length=1, max_length=100)


class GemmaEvidenceInput(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9._:/-]+$")
    source_category: str = Field(default="operator_report", min_length=1, max_length=100)
    source_name: str = Field(min_length=1, max_length=200)
    source_identifier: str = Field(default="operator://submitted", min_length=1, max_length=500)
    text: str = Field(min_length=10, max_length=20_000)
    reliability: float = Field(default=0.7, ge=0.0, le=1.0)
    freshness_minutes: int = Field(default=0, ge=0)
    operator_context: Optional[str] = Field(default=None, max_length=2000)
    gap_target: Optional[str] = Field(default=None, max_length=200)
    reported_latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    reported_longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)

    @model_validator(mode="after")
    def validate_reported_location(self) -> "GemmaEvidenceInput":
        if (self.reported_latitude is None) != (self.reported_longitude is None):
            raise ValueError("Reported latitude and longitude must be supplied together")
        return self


class GemmaCustomAnalysisRequest(StrictModel):
    scenario_id: str = Field(default="operator-submitted", min_length=1, max_length=100)
    evidence: List[GemmaEvidenceInput] = Field(min_length=1, max_length=10)
