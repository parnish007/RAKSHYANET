const API_BASE = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL.replace(/\/$/, '')}/api`
  : 'http://localhost:8000/api';

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`API ${res.status} ${path}: ${text}`);
  }
  return res.json();
}

/* Capability discovery.
 *
 * Some imagery action endpoints are optional: a backend built without the
 * satellite tool may not route them. The client used to find that out by
 * calling an action and catching the 404 — which works, but every such probe
 * leaves a red "Failed to load resource: 404" line in the browser console, and
 * an operations console that logs errors during normal operation has taught its
 * users to ignore errors.
 *
 * The server's own OpenAPI document answers the question without a failing
 * request. It is fetched once, cached, and any failure is treated as "assume
 * the capability is present and let the individual call decide" so a backend
 * that does not serve the schema is not silently stripped of features.
 */
let capabilityProbe = null;

function serverPaths() {
  if (capabilityProbe) return capabilityProbe;
  const root = API_BASE.replace(/\/api$/, '');
  capabilityProbe = fetch(`${root}/openapi.json`)
    .then((res) => (res.ok ? res.json() : null))
    .then((doc) => (doc?.paths ? Object.keys(doc.paths) : null))
    .catch(() => null);
  return capabilityProbe;
}

export async function hasCapability(prefix) {
  const paths = await serverPaths();
  if (paths === null) return true;
  return paths.some((path) => path.startsWith(prefix));
}

// The tile is an image, not JSON, so it is fetched by the browser as an <img>
// source rather than through request(). Absolute, because the API lives on a
// different origin from the dev server.
export function imageryTileUrl(tileId) {
  return `${API_BASE}/imagery/tile/${encodeURIComponent(tileId ?? '')}`;
}

export const api = {
  // ── HITL ──────────────────────────────────────────────────────
  getPendingRequests: ()        => request('/hitl/pending'),
  getApprovalQueue:  ()        => request('/hitl/queue'),
  getQueueStats:     ()        => request('/hitl/queue/stats'),
  getRequestDetails: (id)      => request(`/hitl/request/${id}`),
  approveRequest:    (id, reviewer = 'operator') =>
    request(`/hitl/approve/${id}`, {
      method: 'POST',
      body:   JSON.stringify({ reviewer }),
    }),
  rejectRequest: (id, reviewer = 'operator', reason = '') =>
    request(`/hitl/reject/${id}`, {
      method: 'POST',
      body:   JSON.stringify({ reviewer, reason }),
    }),

  // ── Simulation ────────────────────────────────────────────────
  getSimulationStatus: ()       => request('/simulation/status'),
  startSimulation:     (cfg={}) => request('/simulation/start', {
    method: 'POST',
    body:   JSON.stringify(cfg),
  }),
  stopSimulation: () => request('/simulation/stop', { method: 'POST' }),

  // ── Optimization pipeline ─────────────────────────────────────
  runOptimization: (input = {}) => request('/optimization/run', {
    method: 'POST',
    body: JSON.stringify(input),
  }),
  // Gemma drives this one: it calls list_corridor_status, then run_optimization,
  // and the engine executes only after its arguments pass validation.
  orchestrateOptimization: (input = {}) => request('/optimization/orchestrate', {
    method: 'POST',
    body: JSON.stringify(input),
  }),
  getDeclaredFunctions: () => request('/optimization/tools'),
  getBaselineComparison: () => request('/optimization/baseline'),
  getOptimizationHistory: ()  => request('/optimize/history'),
  getOptimizationRun:     (id) => request(`/optimization/runs/${id}`),
  approveOptimizationRun: (id, reviewer = 'operator', notes = '', expectedUpdatedAt = null, expectedAnalysisId = null) =>
    request(`/optimization/runs/${id}/approve`, {
      method: 'POST',
      body: JSON.stringify({
        reviewer,
        notes,
        expected_updated_at: expectedUpdatedAt,
        expected_analysis_id: expectedAnalysisId,
      }),
    }),
  rejectOptimizationRun: (id, reviewer = 'operator', notes = '', expectedUpdatedAt = null, expectedAnalysisId = null) =>
    request(`/optimization/runs/${id}/reject`, {
      method: 'POST',
      body: JSON.stringify({
        reviewer,
        notes,
        expected_updated_at: expectedUpdatedAt,
        expected_analysis_id: expectedAnalysisId,
      }),
    }),
  getVRPSolution:         ()  => request('/vrp/solution'),
  getNashEquilibrium:     ()  => request('/nash/equilibrium'),
  getKKTVerification:     ()  => request('/kkt/verify'),

  // Explicitly simulated product stories used by the operator-facing scenario
  // switcher. Activating one creates a fresh reviewable run.
  getDemoScenarios: () => request('/demo/scenarios'),
  activateDemoScenario: (scenarioId, stage = 'baseline') =>
    request(`/demo/scenarios/${scenarioId}/activate`, {
      method: 'POST',
      body: JSON.stringify({
        stage,
        requested_by: 'mission-control-scenario-switcher',
      }),
    }),

  // Hosted Gemma analysis is the default. The backend records and exposes
  // deterministic fallback use when API credentials or connectivity fail.
  getGemmaStatus:         () => request('/gemma/status'),
  getLatestGemmaAnalysis: () => request('/gemma/analyses/latest'),
  // A run must be shown next to the analysis it actually consumed. Reading
  // "latest" instead is what desynchronises the evidence queue from the plan.
  getGemmaAnalysis: (id) => request(`/gemma/analyses/${id}`),
  runGemmaAnalysis: (scenario_id = 'nepal-national-demo') =>
    request('/gemma/analyze', {
      method: 'POST',
      body: JSON.stringify({ scenario_id }),
    }),
  analyzeSubmittedEvidence: (evidence, scenario_id = 'operator-submitted') =>
    request('/gemma/analyze-submitted', {
      method: 'POST',
      body: JSON.stringify({ scenario_id, evidence }),
    }),
  recordQuestionDisposition: (analysisId, questionId, disposition) =>
    request(`/gemma/analyses/${analysisId}/questions/${questionId}`, {
      method: 'POST',
      body: JSON.stringify(disposition),
    }),

  // ── Overhead imagery verification ─────────────────────────────
  // Status is served even when the GPU-backed tool is disabled, so callers can
  // distinguish an intentionally unavailable feature from a failed action.
  getImageryStatus: (options = {}) => request('/imagery/status', options),
  // The direct escape hatch: no Gemma round trip, one classifier call, the
  // resulting record appended to the latest analysis.
  verifyCorridorImagery: (corridorId, incidentType, evidenceId = null) =>
    request('/imagery/verify', {
      method: 'POST',
      body: JSON.stringify({
        corridor_id: corridorId,
        incident_type: incidentType,
        evidence_id: evidenceId,
      }),
    }),

  // ── P2P ───────────────────────────────────────────────────────
  getTopology:   () => request('/p2p/topology'),
  getGossipStats:() => request('/p2p/stats'),

  // ── Data ──────────────────────────────────────────────────────
  getVillages: () => request('/villages'),
  getVehicles: () => request('/vehicles'),

  // ── Health ────────────────────────────────────────────────────
  getHealth: () => request('/health'),
};
