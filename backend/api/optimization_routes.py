"""REST and WebSocket integration for deterministic optimization runs."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, status
from starlette.concurrency import run_in_threadpool

from backend.api.gemma_routes import broadcast_gemma_analysis
from backend.api.websocket_manager import WSMessage, ws_manager
from backend.models.optimization import (
    OptimizationDecisionRequest,
    OptimizationRunRecord,
    OptimizationRunRequest,
)
from backend.models.orchestration import OperatorDirective
from backend.services.optimization_service import optimization_service
from backend.services.gemma_service import GemmaProviderError, gemma_service
from backend.services.gemma_orchestrator import (
    _load_corridors,
    function_declarations,
    gemma_orchestrator,
)
from backend.services.imagery_verifier import satellite_tool_enabled
from backend.services.baseline_service import DEFAULT_CLOSURE_EDGE, baseline_service


router = APIRouter(tags=["Optimization"])


class OrchestrationRunRequest(OptimizationRunRequest):
    """An optimization request that may also carry a human instruction.

    Declared as a subclass rather than a second body parameter so the wire
    format stays flat: every existing client posting an `OptimizationRunRequest`
    to this endpoint is unaffected, and the directive is purely additive.
    """

    operator_directive: Optional[OperatorDirective] = None


async def broadcast_run_started(record: OptimizationRunRecord) -> None:
    await ws_manager.broadcast(WSMessage(
        type="optimization_started",
        scenario_id=record.scenario_id,
        correlation_id=record.correlation_id,
        payload={"run_id": record.run_id, "requested_by": record.requested_by},
    ))


async def broadcast_run(
    record: OptimizationRunRecord,
    include_started: bool = True,
) -> None:
    common = {
        "scenario_id": record.scenario_id,
        "correlation_id": record.correlation_id,
    }
    if include_started:
        await ws_manager.broadcast(WSMessage(
            type="optimization_started",
            payload={"run_id": record.run_id, "requested_by": record.requested_by},
            **common,
        ))

    if record.result is None:
        await ws_manager.broadcast(WSMessage(
            type="optimization_failed",
            payload={"run_id": record.run_id, "error": record.error},
            **common,
        ))
        return

    result = record.result
    await ws_manager.broadcast(WSMessage(
        type="urgency_updated",
        payload={
            "run_id": record.run_id,
            "model": "legacy_unmet_need",
            "scores": [score.model_dump(mode="json") for score in result.urgency_scores],
        },
        **common,
    ))

    if result.vrp_solution is not None:
        await ws_manager.broadcast(WSMessage(
            type="route_generated",
            payload={
                "run_id": record.run_id,
                "routing_method": record.routing_method,
                "solution": result.vrp_solution.model_dump(mode="json"),
            },
            **common,
        ))

    if result.nash_equilibrium is not None:
        allocation = result.nash_equilibrium
        for point in allocation.convergence_history:
            await ws_manager.broadcast(WSMessage(
                type="proportional_iteration",
                payload={
                    "run_id": record.run_id,
                    "iteration": point.iteration,
                    "max_strategy_change": point.max_strategy_change,
                    "max_normalized_change": point.max_normalized_change,
                    "total_utility": point.total_utility,
                    "converged": (
                        point.max_normalized_change
                        < allocation.convergence_threshold
                    ),
                },
                **common,
            ))
        await ws_manager.broadcast(WSMessage(
            type="allocation_generated",
            payload={
                "run_id": record.run_id,
                "allocation_method": record.allocation_method,
                "allocation": allocation.model_dump(mode="json"),
                "social_welfare_candidate": (
                    result.social_welfare_allocation.model_dump(mode="json")
                    if result.social_welfare_allocation is not None
                    else None
                ),
                "method_comparison": (
                    result.allocation_comparison.model_dump(mode="json")
                    if result.allocation_comparison is not None
                    else None
                ),
            },
            **common,
        ))

    if result.kkt_verification is not None:
        await ws_manager.broadcast(WSMessage(
            type="validation_completed",
            payload={
                "run_id": record.run_id,
                "diagnostic_scope": record.diagnostic_scope,
                "validation": result.kkt_verification.model_dump(mode="json"),
            },
            **common,
        ))

    await ws_manager.broadcast(WSMessage(
        type="optimization_completed",
        payload=record.model_dump(mode="json"),
        **common,
    ))
    await ws_manager.broadcast(WSMessage(
        type="hitl_review_required",
        payload={
            "run_id": record.run_id,
            "status": record.status.value,
            "reason": "Every dispatch plan requires operator approval.",
        },
        **common,
    ))


@router.post("/api/optimization/run", response_model=OptimizationRunRecord)
@router.post("/api/optimize", response_model=OptimizationRunRecord, include_in_schema=False)
@router.post("/api/optimize/start", response_model=OptimizationRunRecord, include_in_schema=False)
async def run_optimization(
    request: OptimizationRunRequest = OptimizationRunRequest(),
):
    analysis = None
    if request.analysis_id:
        analysis = gemma_service.get(request.analysis_id)
        if analysis is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Gemma analysis '{request.analysis_id}' not found",
            )
    if analysis is None:
        analysis = await run_in_threadpool(
            gemma_service.analyze,
            request.scenario_id,
        )
    request.analysis_id = analysis.analysis_id
    record = await run_in_threadpool(
        optimization_service.run,
        request,
        analysis,
    )
    record.analysis_id = analysis.analysis_id
    analysis.correlation_id = record.correlation_id
    await broadcast_run_started(record)
    await broadcast_gemma_analysis(analysis)
    await broadcast_run(record, include_started=False)
    return record


@router.get("/api/optimization/baseline")
async def compare_against_naive_baseline(
    closure_edge_id: str = DEFAULT_CLOSURE_EDGE,
    time_elapsed_hours: float = 2.0,
):
    """Head-to-head against the documented naive planner.

    The baseline is the same engine with terrain weighting and closure filtering
    switched off, so the difference is attributable to those two behaviours and
    nothing else.
    """
    # A corridor id that does not exist yields a technically-true but useless
    # comparison — nothing traverses a road that is not in the graph, so both
    # planners look identical. A typo during a demo would silently read as
    # "terrain reasoning makes no difference", so reject it instead.
    known = {edge["id"] for edge in _load_corridors()}
    if closure_edge_id not in known:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown corridor '{closure_edge_id}'. "
                f"Known corridors: {', '.join(sorted(known))}"
            ),
        )
    if not 0.0 <= time_elapsed_hours <= 72.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="time_elapsed_hours must be between 0 and 72",
        )
    # Compare against the plan the operator is actually looking at, which means
    # the same Gemma analysis the latest run consumed.
    latest = gemma_service.latest()
    return await run_in_threadpool(
        baseline_service.compare,
        closure_edge_id=closure_edge_id,
        time_elapsed_hours=time_elapsed_hours,
        gemma_analysis=latest,
    )


@router.get("/api/optimization/tools")
async def list_declared_functions():
    """The function schemas Gemma is given.

    Exposed so the declared contract can be inspected directly rather than taken
    on trust from the write-up.
    """
    validation = [
        "analysis_id must match the analysis opened for this turn",
        "every blocked corridor id must exist in the terrain graph",
        "time_elapsed_hours must fall within 0..72",
        "rationale is screened for allocation, dispatch, and approval language",
    ]
    if satellite_tool_enabled():
        validation.extend([
            "imagery corridor_id must exist in the terrain graph",
            "imagery evidence_id must belong to the analysis opened for this turn",
            "imagery incident_type and trigger_reason must match their enums",
            (
                "a corridor whose only supporting evidence is an overhead-imagery "
                "record is rejected from blocked_edge_ids"
            ),
        ])
    return {
        "declared_functions": function_declarations(),
        "validation": validation,
        "authority": (
            "Gemma selects the computation and its inputs. It cannot allocate, "
            "route, approve, or dispatch; the returned plan awaits human approval."
        ),
    }


@router.post("/api/optimization/orchestrate", response_model=OptimizationRunRecord)
async def orchestrate_optimization(
    request: OrchestrationRunRequest = OrchestrationRunRequest(),
):
    """Let Gemma drive the engine through native function calling.

    Gemma receives the validated analysis and the declared function schemas,
    calls `list_corridor_status` to ground itself in the real road graph, then
    calls `run_optimization`. Its arguments are validated against the graph
    before the engine executes them, and the resulting plan still requires human
    approval.
    """
    analysis = None
    if request.analysis_id:
        analysis = gemma_service.get(request.analysis_id)
        if analysis is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Gemma analysis '{request.analysis_id}' not found",
            )
    if analysis is None:
        analysis = await run_in_threadpool(gemma_service.analyze, request.scenario_id)

    if request.operator_directive is not None:
        known_ids = {edge["id"] for edge in _load_corridors()}
        if request.operator_directive.corridor_id not in known_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unknown corridor '{request.operator_directive.corridor_id}' "
                    "in operator_directive"
                ),
            )

    try:
        orchestration = await run_in_threadpool(
            gemma_orchestrator.plan,
            analysis,
            default_elapsed_hours=request.time_elapsed_hours,
            operator_directive=request.operator_directive,
        )
    except GemmaProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemma did not produce an executable optimization call: {exc}",
        ) from exc

    chosen = orchestration.chosen_arguments
    request.analysis_id = analysis.analysis_id
    request.blocked_edge_ids = list(chosen.get("blocked_edge_ids", []))
    request.time_elapsed_hours = float(chosen.get("time_elapsed_hours", 0.0))
    request.trigger = "gemma_function_call"
    request.disruption_reason = chosen.get("rationale")

    record = await run_in_threadpool(optimization_service.run, request, analysis)
    record.analysis_id = analysis.analysis_id
    record.orchestration = orchestration
    analysis.correlation_id = record.correlation_id
    await broadcast_run_started(record)
    await broadcast_gemma_analysis(analysis)
    await broadcast_run(record, include_started=False)
    return record


@router.get("/api/optimization/runs", response_model=list[OptimizationRunRecord])
@router.get("/api/optimize/history", response_model=list[OptimizationRunRecord], include_in_schema=False)
async def list_optimization_runs():
    return optimization_service.list_runs()


@router.get("/api/optimization/runs/{run_id}", response_model=OptimizationRunRecord)
async def get_optimization_run(run_id: str):
    record = optimization_service.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Optimization run '{run_id}' not found")
    return record


async def _review_run(
    run_id: str,
    decision: OptimizationDecisionRequest,
    approved: bool,
) -> OptimizationRunRecord:
    try:
        record = (
            optimization_service.approve(run_id, decision)
            if approved
            else optimization_service.reject(run_id, decision)
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Optimization run '{run_id}' not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    await ws_manager.broadcast(WSMessage(
        type="hitl_approved" if approved else "hitl_rejected",
        scenario_id=record.scenario_id,
        correlation_id=record.correlation_id,
        payload={
            "run_id": record.run_id,
            "reviewer": record.reviewed_by,
            "notes": record.review_notes,
            "status": record.status.value,
        },
    ))
    return record


@router.post("/api/optimization/runs/{run_id}/approve", response_model=OptimizationRunRecord)
async def approve_optimization_run(run_id: str, decision: OptimizationDecisionRequest):
    return await _review_run(run_id, decision, approved=True)


@router.post("/api/optimization/runs/{run_id}/reject", response_model=OptimizationRunRecord)
async def reject_optimization_run(run_id: str, decision: OptimizationDecisionRequest):
    return await _review_run(run_id, decision, approved=False)


def _latest_result():
    record = optimization_service.latest()
    if record is None or record.result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No completed optimization result is available",
        )
    return record.result


@router.get("/api/vrp/solution")
async def get_latest_vrp_solution():
    return _latest_result().vrp_solution


@router.get("/api/nash/equilibrium")
@router.get("/api/allocation/proportional")
async def get_latest_proportional_allocation():
    """Return capped proportional allocation (legacy Nash URL is an alias)."""
    return _latest_result().nash_equilibrium


@router.get("/api/allocation/social-welfare")
async def get_latest_social_welfare_allocation():
    """Return the continuous Nash-social-welfare comparison candidate."""
    return _latest_result().social_welfare_allocation


@router.get("/api/allocation/compare")
async def get_latest_allocation_comparison():
    return _latest_result().allocation_comparison


@router.get("/api/kkt/verify")
async def get_latest_kkt_diagnostics():
    return _latest_result().kkt_verification
