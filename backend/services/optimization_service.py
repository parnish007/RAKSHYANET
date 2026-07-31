"""Deterministic optimization service backed by the existing StateManager."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

from backend.algorithms.state_manager import OptimizationState, StateManager
from backend.algorithms.urgency_calculator import UrgencyCalculator
from backend.models.optimization import (
    OptimizationDecisionRequest,
    OptimizationRunRecord,
    OptimizationRunRequest,
    OptimizationRunStatus,
    utc_now,
)
from backend.models.resource import ResourceType
from backend.models.vehicle import (
    TerrainCapability,
    Vehicle,
    VehicleCategory,
    VehicleType,
)
from backend.models.village import Village
from backend.models.gemma import GemmaAnalysisRecord


class OptimizationService:
    """Runs and retains deterministic recommendation plans for one process."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = data_dir or Path(__file__).resolve().parents[1] / "data"
        self.runs: Dict[str, OptimizationRunRecord] = {}
        self.run_order: List[str] = []
        self._json_cache: Dict[str, tuple] = {}

    def _read_json(self, name: str) -> dict:
        """Cache bundled scenario inputs, keyed on file mtime.

        Optimization does not mutate them, so caching is safe. Keying on mtime
        means a fixture edited between runs is picked up without restarting the
        server, which matters while rehearsing against the demo data.
        """
        path = self.data_dir / name
        stamp = path.stat().st_mtime_ns
        cached = self._json_cache.get(name)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._json_cache[name] = (stamp, payload)
        return payload

    @staticmethod
    def _asset_location(item: dict, vehicle_positions) -> tuple:
        """Where this asset starts the plan.

        Re-planning used to reload every asset at its configured depot position,
        which silently teleported a fleet that was already in flight. When the
        caller reports actual positions, a re-plan starts from them instead.
        """
        reported = (vehicle_positions or {}).get(item["id"])
        if reported is not None:
            lat = getattr(reported, "lat", None)
            lng = getattr(reported, "lng", None)
            if lat is None and isinstance(reported, dict):
                lat, lng = reported.get("lat"), reported.get("lng")
            if lat is not None and lng is not None:
                return (float(lat), float(lng))
        location = item["current_location"]
        return (location["lat"], location["lng"])

    def load_demo_inputs(self, vehicle_positions=None):
        config = self._read_json("config.json")
        fleet_data = self._read_json("fleet_config.json")
        terrain_graph = self._read_json("terrain_graph.json")
        village_data = self._read_json("nepal_villages.json")

        resource_types = {
            resource_id: ResourceType(**definition)
            for resource_id, definition in config["resource_types"].items()
        }
        villages = [
            Village(
                id=item["id"],
                name=item["name"],
                lat=item["lat"],
                lng=item["lng"],
                population=item["population"],
                terrain_difficulty=item["terrain_difficulty"],
                urgency_score=item["initial_urgency"],
                previous_urgency=item["initial_urgency"],
                disaster_impact=item["disaster_impact"],
                has_medical_facility=item["has_medical_facility"],
                accessibility=item["accessibility"],
                resource_needs=item["resource_needs"],
            )
            for item in village_data["villages"]
        ]

        vehicles: List[Vehicle] = []
        for item in fleet_data["helicopters"]:
            vehicle_type = VehicleType(
                type_id="helicopter",
                name="Relief Helicopter",
                category=VehicleCategory.AIRCRAFT,
                capacity_kg=item["capacity_kg"],
                speed_kmh=item["speed_kmh"],
                fuel_hours=item["fuel_hours"],
                terrain_capability=TerrainCapability.ANY,
                cost_per_km=3.5,
                preferred_resources=["medical_kit", "first_aid"],
            )
            location = self._asset_location(item, vehicle_positions)
            vehicles.append(Vehicle(
                id=item["id"],
                name=item["name"],
                vehicle_type=vehicle_type,
                current_location=location,
            ))

        for item in fleet_data["trucks"]:
            vehicle_type = VehicleType(
                type_id="truck",
                name="Relief Truck",
                category=VehicleCategory.GROUND_HEAVY,
                capacity_kg=item["capacity_kg"],
                speed_kmh=item["speed_kmh"],
                fuel_hours=item["fuel_hours"],
                terrain_capability=TerrainCapability.PAVED_ROADS,
                cost_per_km=1.5,
                preferred_resources=["food", "water", "tarpaulin", "blanket"],
            )
            location = self._asset_location(item, vehicle_positions)
            vehicles.append(Vehicle(
                id=item["id"],
                name=item["name"],
                vehicle_type=vehicle_type,
                current_location=location,
            ))

        depot = village_data["depot"]
        state_manager = StateManager(
            depot_location=(depot["lat"], depot["lng"]),
            depot_resources=depot["available_resources"],
            terrain_graph=terrain_graph,
            resource_types=resource_types,
            config=config,
        )
        return state_manager, villages, vehicles

    def run(
        self,
        request: OptimizationRunRequest,
        gemma_analysis: Optional[GemmaAnalysisRecord] = None,
    ) -> OptimizationRunRecord:
        record = OptimizationRunRecord(
            scenario_id=request.scenario_id,
            analysis_id=request.analysis_id,
            requested_by=request.requested_by,
            blocked_edge_ids=list(request.blocked_edge_ids),
            parent_run_id=request.parent_run_id,
            trigger=request.trigger,
            disruption_reason=request.disruption_reason,
        )
        self.runs[record.run_id] = record
        self.run_order.append(record.run_id)

        try:
            manager, villages, vehicles = self.load_demo_inputs(
                request.vehicle_positions
            )
            elapsed = timedelta(hours=request.time_elapsed_hours)
            baseline_scores = UrgencyCalculator(
                manager.resource_types
            ).rank_villages(villages, elapsed)
            signal_summary = self._apply_gemma_signal(villages, gemma_analysis)
            result = manager.run_full_optimization(
                villages=villages,
                vehicles=vehicles,
                time_elapsed=elapsed,
                blocked_edge_ids=request.blocked_edge_ids,
            )
            baseline_by_village = {
                score.village_id: score for score in baseline_scores
            }
            signal_summary["effects"] = [
                {
                    "village_id": score.village_id,
                    "baseline_urgency": baseline_by_village[
                        score.village_id
                    ].total_urgency,
                    "final_urgency": score.total_urgency,
                    "urgency_delta": round(
                        score.total_urgency
                        - baseline_by_village[score.village_id].total_urgency,
                        4,
                    ),
                    "baseline_rank": baseline_by_village[
                        score.village_id
                    ].ranking,
                    "final_rank": score.ranking,
                }
                for score in result.urgency_scores
                if score.village_id in signal_summary.get("matched_villages", [])
            ]
            result.gemma_signal = signal_summary
            record.result = result
            # A failed stage leaves vrp_solution unset. Reading through it here
            # would raise and overwrite result.error_message with an
            # AttributeError, destroying the only description of the real cause.
            routes = (
                list(result.vrp_solution.routes)
                if result.vrp_solution is not None
                else []
            )
            record.route_feasible = bool(routes) and all(
                route.feasible for route in routes
            )
            if result.vrp_solution is None:
                record.approval_blockers.append(
                    "Routing produced no solution: "
                    + (result.error_message or "solver returned no result.")
                )
            elif not routes:
                record.approval_blockers.append(
                    "No assigned routes were generated."
                )
            elif not record.route_feasible:
                infeasible_assets = ", ".join(
                    route.vehicle_id
                    for route in routes
                    if not route.feasible
                )
                record.approval_blockers.append(
                    "Infeasible assigned route set"
                    + (
                        f" for assets: {infeasible_assets}."
                        if infeasible_assets
                        else "."
                    )
                )
            if result.state == OptimizationState.COMPLETE:
                record.status = OptimizationRunStatus.AWAITING_APPROVAL
            else:
                record.status = OptimizationRunStatus.FAILED
                record.error = result.error_message or "Optimization did not complete"
        except Exception as exc:  # noqa: BLE001
            record.status = OptimizationRunStatus.FAILED
            record.error = str(exc)
        record.updated_at = utc_now()
        return record

    def _apply_gemma_signal(
        self,
        villages: List[Village],
        analysis: Optional[GemmaAnalysisRecord],
    ) -> dict:
        if analysis is None:
            return {
                "applied": False,
                "reason": "No Gemma analysis attached",
                "input_scores": [],
                "calculation": None,
            }
        output = analysis.output
        input_scores = [
            {
                "field": "severity",
                "label": "Incident severity",
                "value": output.severity.expected,
                "confidence": output.severity.confidence,
                "evidence_ids": list(output.severity.evidence_ids),
                "status": "supported" if output.severity.expected is not None else "unknown",
            },
            {
                "field": "medical_urgency",
                "label": "Medical urgency",
                "value": output.medical_urgency.value,
                "confidence": output.medical_urgency.confidence,
                "evidence_ids": list(output.medical_urgency.evidence_ids),
                "status": "supported" if output.medical_urgency.value is not None else "unknown",
            },
            {
                "field": "accessibility_risk",
                "label": "Accessibility risk",
                "value": output.accessibility_risk.value,
                "confidence": output.accessibility_risk.confidence,
                "evidence_ids": list(output.accessibility_risk.evidence_ids),
                "status": "supported" if output.accessibility_risk.value is not None else "unknown",
            },
        ]
        scores = [
            item["value"]
            for item in input_scores
            if item["value"] is not None
        ]
        signal = max(scores, default=0.0)
        for item in input_scores:
            item["selected_for_max"] = (
                item["value"] is not None
                and abs(float(item["value"]) - signal) < 1e-12
            )
        boost = round(signal * analysis.system_confidence, 4)
        evidence_text = " ".join(item.text.lower() for item in analysis.evidence)
        matched = []
        for village in villages:
            aliases = {village.id.lower(), village.name.lower()}
            if any(alias in evidence_text for alias in aliases):
                village.external_urgency_boost = boost
                matched.append(village.id)
        return {
            "applied": bool(matched),
            "analysis_id": analysis.analysis_id,
            "signal": round(signal, 4),
            "system_confidence": analysis.system_confidence,
            "boost": boost,
            "matched_villages": matched,
            "source_evidence_ids": [item.evidence_id for item in analysis.evidence],
            "input_scores": input_scores,
            "calculation": {
                "maximum_supported_score": round(signal, 4),
                "system_confidence": analysis.system_confidence,
                "resulting_boost": boost,
                "unknown_fields_ignored": [
                    item["field"]
                    for item in input_scores
                    if item["value"] is None
                ],
            },
            "method": "max(severity, medical_urgency, accessibility_risk) × system_confidence",
        }

    def list_runs(self) -> List[OptimizationRunRecord]:
        return [self.runs[run_id] for run_id in reversed(self.run_order)]

    def latest(self) -> Optional[OptimizationRunRecord]:
        return self.runs[self.run_order[-1]] if self.run_order else None

    def latest_superseding(self) -> Optional[OptimizationRunRecord]:
        """Newest run that represents a real plan.

        A FAILED run never produced a plan, so it cannot supersede the good run
        before it. Treating it as newer would deadlock approval: the good run is
        refused as stale, and the failed run is not awaiting approval either.
        """
        for run_id in reversed(self.run_order):
            record = self.runs[run_id]
            if record.status != OptimizationRunStatus.FAILED:
                return record
        return None

    def get(self, run_id: str) -> Optional[OptimizationRunRecord]:
        return self.runs.get(run_id)

    def approve(
        self,
        run_id: str,
        decision: OptimizationDecisionRequest,
    ) -> OptimizationRunRecord:
        return self._decide(run_id, decision, OptimizationRunStatus.APPROVED)

    def reject(
        self,
        run_id: str,
        decision: OptimizationDecisionRequest,
    ) -> OptimizationRunRecord:
        return self._decide(run_id, decision, OptimizationRunStatus.REJECTED)

    def _decide(
        self,
        run_id: str,
        decision: OptimizationDecisionRequest,
        status: OptimizationRunStatus,
    ) -> OptimizationRunRecord:
        record = self.runs.get(run_id)
        if record is None:
            raise KeyError(run_id)
        if record.status != OptimizationRunStatus.AWAITING_APPROVAL:
            raise ValueError(
                f"Run '{run_id}' is {record.status.value}; only awaiting-approval runs can be reviewed"
            )
        if decision.expected_updated_at != record.updated_at:
            raise ValueError(
                f"Run '{run_id}' changed after the review snapshot was opened"
            )
        if decision.expected_analysis_id != record.analysis_id:
            raise ValueError(
                f"Run '{run_id}' no longer references the reviewed Gemma analysis"
            )
        latest = self.latest_superseding()
        if latest is not None and latest.run_id != run_id:
            # A REJECTED newer run still supersedes: it was computed from a later
            # world state, so the run before it is genuinely out of date and must
            # not become approvable again just because the newer plan was refused.
            # Skipping REJECTED here would let an operator authorise a plan built
            # on facts that have since changed — the exact thing this rule exists
            # to prevent. What was missing was telling the operator how to recover.
            recovery = (
                " Re-run the pipeline to produce a current plan; rejecting a plan "
                "does not reinstate the one before it."
                if latest.status is OptimizationRunStatus.REJECTED
                else ""
            )
            raise ValueError(
                f"Run '{run_id}' is stale because newer run '{latest.run_id}' "
                f"({latest.status.value}) exists.{recovery}"
            )
        if (
            status == OptimizationRunStatus.APPROVED
            and record.route_feasible is not True
        ):
            detail = "; ".join(record.approval_blockers) or (
                "The assigned route set is infeasible or empty."
            )
            raise ValueError(
                f"Run '{run_id}' cannot be approved: {detail}"
            )
        record.status = status
        record.reviewed_by = decision.reviewer
        record.review_notes = decision.notes or (
            "Operator approved the versioned coordination snapshot."
            if status == OptimizationRunStatus.APPROVED
            else "Operator requested changes to the versioned coordination snapshot."
        )
        record.reviewed_at = utc_now()
        record.updated_at = record.reviewed_at
        return record


optimization_service = OptimizationService()
