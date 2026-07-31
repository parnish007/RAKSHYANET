"""Replay every active-pipeline mock scenario and print a compact audit."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.demo.scenario_replay import (  # noqa: E402
    ScenarioReplayEngine,
    load_scenarios,
)


def main() -> None:
    scenarios = load_scenarios(ROOT / "backend" / "demo" / "scenarios")
    engine = ScenarioReplayEngine()

    print(
        "scenario".ljust(42),
        "decision".ljust(10),
        "routes",
        "blocked edge",
    )
    print("-" * 96)
    for scenario in scenarios:
        result = engine.replay(scenario)
        child = next(
            entry
            for entry in result.entries
            if entry.event_type == "road_block_report"
        )
        print(
            scenario.scenario_id.ljust(42),
            result.final_status.ljust(10),
            str(child.route_count).rjust(6),
            ", ".join(result.blocked_edge_ids),
        )

    print(f"\n{len(scenarios)} scenarios replayed successfully.")


if __name__ == "__main__":
    main()
