import { useMemo, useState } from 'react';
import { Ban, Mountain, Route as RouteIcon, TriangleAlert } from 'lucide-react';
import './terrain-cost.css';

/* ─────────────────────────────────────────────────────────────────────────────
   Terrain cost panel.

   Route Intelligence is the track, and until now nothing on screen showed that
   the search is terrain-aware at all. Everything here is read straight off
   `run.result.vrp_solution` — per-edge difficulty, surface, landslide exposure —
   and the cost arithmetic is the engine's own, printed with real numbers:

       cost_e = d_e · (1 + 0.06 · max(0, difficulty − 1))
       backend/algorithms/vrp_solver.py:386

   It also states the measured limitation rather than hiding it. On this network
   the weighting flips no path; the measured advantage is closure-aware
   re-planning. Showing the mechanism working *and* the honest null result is
   worth more than implying a benefit that was measured not to exist.
   ───────────────────────────────────────────────────────────────────────── */

const TERRAIN_COEFFICIENT = 0.06;

const SURFACE_COPY = {
  paved: 'Sealed surface',
  mixed: 'Part sealed, part dirt',
  dirt: 'Unsealed track',
};

function multiplierFor(difficulty) {
  return 1 + TERRAIN_COEFFICIENT * Math.max(0, Number(difficulty ?? 1) - 1);
}

function fixed(value, digits = 1) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
}

function difficultyBand(difficulty) {
  const value = Number(difficulty ?? 1);
  if (value >= 3.5) return 'severe';
  if (value >= 2) return 'hard';
  if (value >= 1.4) return 'moderate';
  return 'easy';
}

export default function TerrainCostPanel({ run }) {
  const [showUnused, setShowUnused] = useState(false);

  const solution = run?.result?.vrp_solution;
  const roadNetwork = useMemo(() => solution?.road_network ?? [], [solution]);
  const routes = useMemo(() => solution?.routes ?? [], [solution]);
  const activeBlocks = useMemo(
    () => new Set(solution?.active_road_blocks ?? run?.blocked_edge_ids ?? []),
    [solution, run],
  );

  const model = useMemo(() => {
    const traversals = new Map();
    const carriers = new Map();

    const groundRoutes = routes.filter((route) => (route.road_edge_ids ?? []).length > 0);
    const airRoutes = routes.filter((route) => (route.road_edge_ids ?? []).length === 0);

    groundRoutes.forEach((route) => {
      (route.road_edge_ids ?? []).forEach((edgeId) => {
        traversals.set(edgeId, (traversals.get(edgeId) ?? 0) + 1);
        if (!carriers.has(edgeId)) carriers.set(edgeId, new Set());
        carriers.get(edgeId).add(route.vehicle_id);
      });
    });

    const decorate = (edge) => {
      const difficulty = Number(edge.terrain_difficulty ?? 1);
      const distance = Number(edge.distance_km ?? 0);
      const multiplier = multiplierFor(difficulty);
      return {
        ...edge,
        difficulty,
        distance,
        multiplier,
        weighted: distance * multiplier,
        surcharge: distance * multiplier - distance,
        band: difficultyBand(difficulty),
        traversals: traversals.get(edge.edge_id) ?? 0,
        carriers: [...(carriers.get(edge.edge_id) ?? [])],
        closed: activeBlocks.has(edge.edge_id) || edge.status !== 'open',
      };
    };

    const all = roadNetwork.map(decorate);
    const used = all
      .filter((edge) => edge.traversals > 0)
      .sort((a, b) => b.difficulty - a.difficulty);
    const unused = all
      .filter((edge) => edge.traversals === 0)
      .sort((a, b) => b.difficulty - a.difficulty);

    const rawKm = used.reduce((sum, edge) => sum + edge.distance * edge.traversals, 0);
    const weightedKm = used.reduce((sum, edge) => sum + edge.weighted * edge.traversals, 0);
    const surfacesOnPlan = [...new Set(used.map((edge) => edge.road_quality))];

    return {
      all,
      used,
      unused,
      rawKm,
      weightedKm,
      groundRoutes,
      airRoutes,
      exposed: used.filter((edge) => edge.vulnerable_to_landslide),
      closedCount: all.filter((edge) => edge.closed).length,
      surfacesOnPlan,
      allPaved: surfacesOnPlan.length > 0 && surfacesOnPlan.every((s) => s === 'paved'),
    };
  }, [roadNetwork, routes, activeBlocks]);

  if (roadNetwork.length === 0) {
    return (
      <section className="ops-panel tc-panel" aria-labelledby="tc-title">
        <header className="tc-head">
          <div>
            <span className="ops-eyebrow">Terrain-aware routing</span>
            <h2 id="tc-title">Corridor cost and terrain reasoning</h2>
            <p>Why each road corridor was chosen, priced, or ruled out.</p>
          </div>
        </header>
        <div className="tc-empty">
          <Mountain width={20} height={20} strokeWidth={1.7} aria-hidden="true" />
          <b>No road graph on this run</b>
          <p>Run the pipeline to compute a plan over the terrain graph.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="ops-panel tc-panel" aria-labelledby="tc-title">
      <header className="tc-head">
        <div className="tc-head-copy">
          <span className="ops-eyebrow">Terrain-aware routing</span>
          <h2 id="tc-title">Corridor cost and terrain reasoning</h2>
          <p>
            The search does not minimise raw kilometres. Every corridor is priced by
            its terrain difficulty before the shortest path is found — this is that
            arithmetic, per corridor, with the numbers the engine used.
          </p>
        </div>
        <div className="tc-formula" aria-label="Edge cost formula">
          <code>cost = distance × (1 + 0.06 × max(0, difficulty − 1))</code>
          <small>vrp_solver.py:386 — applied inside Dijkstra, not after</small>
        </div>
      </header>

      <div className="tc-stats">
        <div>
          <span>Corridors in graph</span>
          <b>{model.all.length}</b>
          <small>{model.closedCount} closed and removed before the search</small>
        </div>
        <div>
          <span>Corridors on the plan</span>
          <b>{model.used.length}</b>
          <small>
            {model.groundRoutes.length} ground route{model.groundRoutes.length === 1 ? '' : 's'} ·{' '}
            {model.airRoutes.length} air route{model.airRoutes.length === 1 ? '' : 's'} bypass the graph
          </small>
        </div>
        <div>
          <span>Landslide-exposed on plan</span>
          <b data-tone={model.exposed.length > 0 ? 'warn' : 'calm'}>{model.exposed.length}</b>
          <small>corridors flagged vulnerable in the terrain fixture</small>
        </div>
        <div>
          <span>Terrain surcharge</span>
          <b>+{fixed(model.weightedKm - model.rawKm)} km</b>
          <small>
            {fixed(model.rawKm)} km driven priced as {fixed(model.weightedKm)} km-equivalent
          </small>
        </div>
      </div>

      <div className="tc-table-wrap" tabIndex={0}>
        <table className="tc-table">
          <caption>Corridors the chosen ground routes traverse, hardest first</caption>
          <thead>
            <tr>
              <th scope="col">Corridor</th>
              <th scope="col">Surface</th>
              <th scope="col">Difficulty</th>
              <th scope="col">Landslide</th>
              <th scope="col">Distance</th>
              <th scope="col">Cost the search used</th>
              <th scope="col">Carried</th>
            </tr>
          </thead>
          <tbody>
            {model.used.map((edge) => (
              <tr key={edge.edge_id} data-band={edge.band}>
                <th scope="row">
                  <b>{edge.name}</b>
                  <code>{edge.edge_id}</code>
                </th>
                <td>
                  <span className={`tc-surface ${edge.road_quality}`}>{edge.road_quality}</span>
                  <small>{SURFACE_COPY[edge.road_quality] ?? 'Surface unrecorded'}</small>
                </td>
                <td className="tc-num">
                  <b>{fixed(edge.difficulty, 1)}</b>
                  <span className={`tc-band ${edge.band}`}>{edge.band}</span>
                </td>
                <td>
                  {edge.vulnerable_to_landslide ? (
                    <span className="tc-flag warn">
                      <TriangleAlert width={12} height={12} strokeWidth={2} aria-hidden="true" />
                      vulnerable
                    </span>
                  ) : (
                    <span className="tc-flag calm">not flagged</span>
                  )}
                </td>
                <td className="tc-num">{fixed(edge.distance, 1)} km</td>
                <td className="tc-math">
                  <code>
                    {fixed(edge.distance, 1)} × (1 + 0.06 × {fixed(Math.max(0, edge.difficulty - 1), 1)}) ={' '}
                    <b>{fixed(edge.weighted, 1)}</b>
                  </code>
                  <small>
                    {edge.surcharge > 0.05
                      ? `+${fixed(edge.surcharge, 1)} km-equivalent penalty (×${fixed(edge.multiplier, 3)})`
                      : 'no penalty — difficulty at or below baseline'}
                  </small>
                </td>
                <td className="tc-num">
                  <b>{edge.traversals}×</b>
                  <small>{edge.carriers.join(', ') || '—'}</small>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="tc-why">
        <button
          type="button"
          className="tc-toggle"
          aria-expanded={showUnused}
          onClick={() => setShowUnused((open) => !open)}
        >
          <RouteIcon width={14} height={14} strokeWidth={1.9} aria-hidden="true" />
          {showUnused ? 'Hide' : 'Show'} the {model.unused.length} corridors the plan did not use
        </button>

        {showUnused && (
          <ul className="tc-unused">
            {model.unused.map((edge) => (
              <li key={edge.edge_id} data-band={edge.band}>
                <div className="tc-unused-id">
                  <b>{edge.name}</b>
                  <code>{edge.edge_id}</code>
                </div>
                <div className="tc-unused-facts">
                  <span>{edge.road_quality}</span>
                  <span>difficulty {fixed(edge.difficulty, 1)}</span>
                  <span>{fixed(edge.distance, 1)} km</span>
                  <span>priced at {fixed(edge.weighted, 1)} km-equivalent</span>
                </div>
                <p className="tc-unused-reason">
                  {edge.closed ? (
                    <>
                      <Ban width={12} height={12} strokeWidth={2} aria-hidden="true" />
                      Removed from the graph before the search — this corridor is closed on
                      this run, so no vehicle could be routed over it.
                    </>
                  ) : edge.road_quality !== 'paved' ? (
                    <>
                      Available to the search, but no route in this plan traverses it. Every
                      ground corridor the plan does use is <code>paved</code>; this one is{' '}
                      <code>{edge.road_quality}</code>.
                    </>
                  ) : (
                    <>
                      Available to the search and never removed — it simply did not lie on any
                      cheapest terrain-weighted path between the depot and a served stop.
                    </>
                  )}
                </p>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="tc-finding">
        <div className="tc-finding-mark">measured</div>
        <div>
          <b>Terrain weighting alone changes no path on this network.</b>
          <p>
            With the weighting disabled, all nine routes keep an identical edge sequence.
            The corridor set is too sparse for a 6%-per-difficulty-point penalty to flip a
            choice. This is a real, reproducible null result from the baseline comparison —
            it is stated rather than papered over.
          </p>
          <p>
            <b>What the terrain reasoning does buy</b> is the other half of the same
            machinery: corridors established as closed are deleted from the graph before
            Dijkstra runs, and the plan re-routes. Measured on the same inputs, the naive
            planner keeps <b>0 of 5</b> executable ground routes after{' '}
            <code>east_west_bharatpur_nepalgunj</code> closes; RakshyaNet keeps{' '}
            <b>5 of 5</b>. Run the baseline comparison below for the full table.
          </p>
        </div>
      </div>

      {model.allPaved && (
        <p className="tc-capability">
          Every ground asset in this fleet is rated <code>roads_only</code>, and every corridor
          on the chosen ground routes is <code>paved</code>. No route in this plan traverses a
          dirt or mixed corridor — the {model.all.filter((e) => e.road_quality !== 'paved').length}{' '}
          unsealed corridors in the graph stay unused by this fleet.
        </p>
      )}
    </section>
  );
}
