"""Records of Gemma's native function-calling turns over the route engine.

These are kept as first-class, persisted records rather than log lines because
the audit question a reviewer asks is not "did the model help" but "what exactly
did the model ask for, what did we check, and what did we let through".
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OperatorDirective(BaseModel):
    """A human instruction carried into a function-calling turn.

    The operator does not call the tool; they ask the model to. Recording the
    directive separately from the resulting call is what lets the audit trail
    distinguish "the model decided to check" from "a person told it to".
    """

    model_config = ConfigDict(extra="forbid")

    corridor_id: str = Field(min_length=1, max_length=120)
    incident_type: Literal["flood", "landslide"]
    evidence_id: str = Field(min_length=1, max_length=120)


class ToolCallRecord(BaseModel):
    """One function call the model emitted, and what happened to it."""

    # `model_complied` is a contract-fixed field name that collides with
    # pydantic's reserved `model_` prefix. The name is what the frontend and the
    # audit trail read, so the namespace guard is released rather than the name
    # renamed.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str = Field(min_length=1, max_length=120)
    raw_arguments: Dict[str, Any] = Field(default_factory=dict)
    validated_arguments: Optional[Dict[str, Any]] = None
    accepted: bool = False
    rejection_reason: Optional[str] = Field(default=None, max_length=600)
    result_summary: Optional[str] = Field(default=None, max_length=600)

    # Who caused this call to exist. "model" is the default because it is the
    # normal case; "operator" marks a call the model made because a human
    # directed it to in this turn, which is a materially weaker claim about the
    # model's autonomy and must not be presented as the same thing.
    initiated_by: str = Field(default="model", max_length=40)
    # Only meaningful on a forced turn. False records that the model ignored an
    # operator directive and the backend performed the check deterministically
    # instead — the button always does something, and the record says how.
    model_complied: Optional[bool] = None


class OrchestrationRecord(BaseModel):
    """A complete function-calling turn: what was declared, called, and accepted."""

    model_config = ConfigDict(extra="forbid")

    orchestration_id: str = Field(
        default_factory=lambda: f"orc_{uuid4().hex[:12]}"
    )
    analysis_id: str
    provider: str
    model: str
    prompt_version: str
    declared_functions: List[str] = Field(default_factory=list)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    # The model's own deliberation, verbatim. Unlike the extraction path — where
    # forcing a JSON response mime type leaves the thought body empty — the
    # function-calling turn returns full reasoning text. It is captured for
    # transparency only: it is not citation-validated and must never be read as
    # a finding.
    reasoning: List[str] = Field(default_factory=list)
    chosen_arguments: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    created_at: str = Field(default_factory=utc_now)
