"""Evidence-grounded Gemma analysis endpoints and typed events."""
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from backend.api.websocket_manager import WSMessage, ws_manager
from backend.models.gemma import (
    EvidenceQuestionDispositionRequest,
    EvidenceRecord,
    GemmaAnalysisRecord,
    GemmaAnalysisRequest,
    GemmaCustomAnalysisRequest,
)
from backend.services.gemma_service import GemmaInputPolicyError, gemma_service


router = APIRouter(tags=["Gemma analysis"])


async def broadcast_gemma_analysis(record: GemmaAnalysisRecord) -> None:
    common = {
        "scenario_id": record.scenario_id,
        "correlation_id": record.correlation_id,
    }
    await ws_manager.broadcast(WSMessage(
        type="evidence_retrieved",
        payload={
            "analysis_id": record.analysis_id,
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "source_category": item.source_category,
                    "source_name": item.source_name,
                    "reliability": item.reliability,
                    "simulated": item.simulated,
                }
                for item in record.evidence
            ],
        },
        **common,
    ))
    await ws_manager.broadcast(WSMessage(
        type="gemma_analysis_started",
        payload={
            "analysis_id": record.analysis_id,
            "provider": record.provider,
            "model": record.model,
            "prompt_version": record.prompt_version,
        },
        **common,
    ))
    await ws_manager.broadcast(WSMessage(
        type="gemma_analysis_completed",
        payload=record.model_dump(mode="json"),
        **common,
    ))


@router.get("/api/gemma/status")
async def gemma_status():
    return gemma_service.status()


@router.post("/api/gemma/analyze", response_model=GemmaAnalysisRecord)
async def analyze_reports(request: GemmaAnalysisRequest):
    try:
        record = await run_in_threadpool(
            gemma_service.analyze,
            request.scenario_id,
        )
    except GemmaInputPolicyError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    await broadcast_gemma_analysis(record)
    return record


@router.post("/api/gemma/analyze-submitted", response_model=GemmaAnalysisRecord)
async def analyze_submitted_reports(request: GemmaCustomAnalysisRequest):
    evidence = [EvidenceRecord(
        evidence_id=item.evidence_id,
        source_category=item.source_category,
        source_name=item.source_name,
        source_identifier=item.source_identifier,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        freshness_minutes=item.freshness_minutes,
        reliability=item.reliability,
        text=item.text,
        provider="operator_submission",
        cache_status="submitted",
        simulated=False,
        operator_context=item.operator_context,
        gap_target=item.gap_target,
        reported_latitude=item.reported_latitude,
        reported_longitude=item.reported_longitude,
    ) for item in request.evidence]
    try:
        record = await run_in_threadpool(
            gemma_service.analyze_submitted,
            request.scenario_id,
            evidence,
        )
    except GemmaInputPolicyError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    await broadcast_gemma_analysis(record)
    return record


@router.get("/api/gemma/analyses", response_model=list[GemmaAnalysisRecord])
async def list_analyses():
    return gemma_service.list_analyses()


@router.get("/api/gemma/analyses/latest", response_model=GemmaAnalysisRecord)
async def latest_analysis():
    record = gemma_service.latest()
    if record is None:
        raise HTTPException(status_code=404, detail="No Gemma analysis is available")
    return record


@router.get("/api/gemma/analyses/{analysis_id}", response_model=GemmaAnalysisRecord)
async def get_analysis(analysis_id: str):
    """Fetch one analysis by id.

    The interface needs the analysis a specific run consumed, not the newest one
    that happens to exist. Reading "latest" alongside a run is how the evidence
    queue ends up asking for a field the displayed plan already has a value for.
    """
    record = gemma_service.get(analysis_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Gemma analysis '{analysis_id}' not found",
        )
    return record


@router.post(
    "/api/gemma/analyses/{analysis_id}/questions/{question_id}",
    response_model=GemmaAnalysisRecord,
)
async def record_question_disposition(
    analysis_id: str,
    question_id: str,
    request: EvidenceQuestionDispositionRequest,
):
    try:
        record = gemma_service.record_question_disposition(
            analysis_id,
            question_id,
            request,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Gemma analysis '{analysis_id}' not found")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    disposition = next(
        item
        for item in record.question_dispositions
        if item.question_id == question_id
    )
    await ws_manager.broadcast(WSMessage(
        type="evidence_question_disposition_recorded",
        scenario_id=record.scenario_id,
        correlation_id=record.correlation_id,
        payload={
            "analysis_id": record.analysis_id,
            "disposition": disposition.model_dump(mode="json"),
        },
    ))
    return record
