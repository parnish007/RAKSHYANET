import { lazy, memo, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  BadgeCheck,
  Ban,
  Boxes,
  Calculator,
  Check,
  CircleAlert,
  CircleCheck,
  Clock3,
  FileText,
  FlaskConical,
  FolderOpen,
  GitBranch,
  Globe2,
  Info,
  Landmark,
  Layers3,
  Link2,
  ListPlus,
  LoaderCircle,
  LockKeyhole,
  MapPinPlus,
  Maximize2,
  Mountain,
  Navigation,
  PackageCheck,
  RadioTower,
  RefreshCw,
  RotateCcw,
  Route,
  ScrollText,
  ShieldCheck,
  Sparkles,
  TriangleAlert,
  Undo2,
  Wrench,
  X,
  Activity,
  Database,
  SlidersHorizontal,
  TableProperties,
  Play,
  Pause,
  ArrowLeftRight,
  Braces,
} from 'lucide-react';
import { useWebSocket } from './hooks/useWebSocket';
import { api, imageryTileUrl } from './services/api';
import AgentConsole from './components/AgentConsole';
import TerrainCostPanel from './components/TerrainCostPanel';

const TerrainMissionMap = lazy(() => import('./components/TerrainMissionMap'));

// Playback rate for the operator-controlled mission clock.
// Playback speed, in mission minutes per real second. 12x completed a 141-minute
// helicopter route in twelve seconds, which is too fast to read: the operator
// cannot see a stop being served. 2x is the default and the operator can change it.
const MISSION_SPEEDS = [1, 2, 6, 12];
const DEFAULT_MISSION_SPEED = 2;

const ICONS = {
  account_tree: GitBranch,
  add_link: Link2,
  add_location_alt: MapPinPlus,
  auto_awesome: Sparkles,
  block: Ban,
  compare_arrows: ArrowLeftRight,
  function: Braces,
  boxes: Boxes,
  build: Wrench,
  calculate: Calculator,
  check: Check,
  check_circle: CircleCheck,
  close: X,
  description: FileText,
  error: CircleAlert,
  folder_open: FolderOpen,
  info: Info,
  landscape: Mountain,
  layers: Layers3,
  link: Link2,
  lock: LockKeyhole,
  open_in_full: Maximize2,
  package_check: PackageCheck,
  playlist_add: ListPlus,
  policy: Landmark,
  progress_activity: LoaderCircle,
  public: Globe2,
  refresh: RefreshCw,
  restart_alt: RotateCcw,
  route: Route,
  navigation: Navigation,
  radio: RadioTower,
  schedule: Clock3,
  scroll_text: ScrollText,
  science: FlaskConical,
  shield_lock: ShieldCheck,
  undo: Undo2,
  verified: BadgeCheck,
  verified_user: ShieldCheck,
  warning: TriangleAlert,
  fact_check: BadgeCheck,
  help_outline: Info,
  activity: Activity,
  database: Database,
  tune: SlidersHorizontal,
  table: TableProperties,
  play_arrow: Play,
  pause: Pause,
};

function Icon({ name, size = 18 }) {
  const Glyph = ICONS[name] ?? CircleAlert;
  return (
    <Glyph
      className={`ops-icon ${name === 'progress_activity' ? 'spin' : ''}`}
      width={size}
      height={size}
      strokeWidth={1.8}
      aria-hidden="true"
    />
  );
}

function number(value, digits = 0) {
  return Number.isFinite(Number(value))
    ? Number(value).toLocaleString(undefined, {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })
    : '—';
}

function percent(value) {
  return Number.isFinite(Number(value)) ? `${Math.round(Number(value) * 100)}%` : '—';
}

function isFeasibleRoute(route) {
  return route?.feasible !== false;
}

// ── Overhead imagery verification ───────────────────────────────────────────
// The imagery tool depends on a local GPU sidecar that the hosted deployment
// cannot run. Historical records remain visible, but actions require a positive
// status response and every unavailable surface explains the local-only limit.
const IMAGERY_CATEGORY = 'overhead_imagery_analysis';
const IMAGERY_STATUS_TIMEOUT_MS = 5000;
const IMAGERY_HOSTED_UNAVAILABLE =
  'Overhead-imagery verification is unavailable in the hosted demo because it needs a local GPU sidecar. It runs in the local build.';
const IMAGERY_STATUS_UNAVAILABLE = Object.freeze({
  enabled: false,
  sidecar_reachable: false,
  tile_count: 0,
});

const IMAGERY_TIERS = {
  local_model_inference: {
    label: 'live model',
    tone: 'live',
    note: 'Classified during this turn by a local model.',
  },
  bundled_imagery_fixture: {
    label: 'precomputed',
    tone: 'precomputed',
    note: 'A cached result from a real earlier run; the live classifier was not reachable.',
  },
  imagery_check_unavailable: {
    label: 'unavailable',
    tone: 'unavailable',
    note: 'The check could not be completed. Absence of information, not evidence of absence.',
  },
};

const IMAGERY_CHIP_DOT = {
  live: 'nominal',
  precomputed: 'attention',
  unavailable: 'critical',
  off: 'attention',
};

function isImageryRecord(record) {
  return record?.source_category === IMAGERY_CATEGORY;
}

// `eurosat://nepal-corridor-07` → `nepal-corridor-07`.
function imageryTileId(record) {
  const match = /^eurosat:\/\/(.+)$/.exec(String(record?.source_identifier ?? ''));
  return match ? match[1] : null;
}

// The classifier facts live inside the record text, which is the only place the
// backend guarantees them. Parsing beats inventing fields the contract does not
// promise; anything that does not match is simply not shown.
function imageryReadout(text = '') {
  const source = String(text);
  const classified = /returned\s+"([^"]+)"\s+at\s+([\d.]+)\s*%\s*confidence\s+against\s+a\s+reference\s+of\s+"([^"]+)"/.exec(source);
  const live = /Classified live by\s+(.+?)\s+on\s+([^.\s]+)\./.exec(source);
  const cached = /precomputed by\s+(.+?);/.exec(source);
  return {
    label: classified?.[1] ?? null,
    confidence: classified?.[2] ?? null,
    reference: classified?.[3] ?? null,
    modelId: live?.[1] ?? cached?.[1] ?? null,
    device: live?.[2] ?? null,
  };
}

// Corridor ids are terrain-graph edge ids, and village ids are graph node ids,
// so the corridor to check for a selected incident is an edge that touches it.
// A landslide-vulnerable corridor is preferred because that is the one an
// operator is asking about.
function corridorForVillage(roadNetwork = [], villageId) {
  if (!villageId) return null;
  const touching = roadNetwork.filter(
    (edge) => edge?.to_node_id === villageId || edge?.from_node_id === villageId,
  );
  return touching.find((edge) => edge.vulnerable_to_landslide) ?? touching[0] ?? null;
}

// The tool only accepts flood or landslide. Read what Gemma classified; fall
// back to what the corridor is known to be vulnerable to.
function imageryIncidentType(analysis, corridor) {
  const classified = String(analysis?.output?.incident_type?.value ?? '').toLowerCase();
  if (classified.includes('flood')) return 'flood';
  if (classified.includes('landslide')) return 'landslide';
  return corridor?.vulnerable_to_landslide ? 'landslide' : 'flood';
}

// The claim being tested. Prefer a record that actually mentions the incident,
// so the check is bound to the report it is checking rather than to a stranger.
function imageryEvidenceId(analysis, incidentType) {
  const records = Array.isArray(analysis?.evidence) ? analysis.evidence : [];
  const matched = records.find((record) =>
    String(record?.text ?? '').toLowerCase().includes(incidentType));
  return (matched ?? records[0])?.evidence_id ?? null;
}

function imageryActionsAvailable(status) {
  if (!status || typeof status !== 'object') return false;
  const enabled = status.enabled ?? status.satellite_tool_enabled ?? status.tool_enabled;
  if (enabled !== true) return false;

  const sidecar = status.sidecar ?? {};
  const tiles = status.tiles ?? status.tile_count ?? status.precomputed_tiles;
  const tileCount = Array.isArray(tiles) ? tiles.length : Number(tiles);
  return sidecar.status === 'ok'
    || status.sidecar_reachable === true
    || status.live_available === true
    || status.tier === 'live'
    || status.precomputed_available === true
    || status.tier === 'precomputed'
    || tileCount > 0;
}

// `/imagery/status` is the backend's shape, not ours. Unknown or malformed
// responses fail closed, because enabling a dead verification control is worse
// than withholding an optional tool.
function imageryStatusChip(status) {
  if (!status || typeof status !== 'object') {
    return {
      tone: 'off',
      label: 'Checking imagery availability',
      detail: 'Verification controls stay disabled until the status check completes.',
    };
  }
  const enabled = status.enabled ?? status.satellite_tool_enabled ?? status.tool_enabled;
  if (enabled !== true) {
    return {
      tone: 'off',
      label: 'Overhead imagery unavailable',
      detail: IMAGERY_HOSTED_UNAVAILABLE,
    };
  }
  const sidecar = status.sidecar ?? {};
  if (
    sidecar.status === 'ok'
    || status.sidecar_reachable === true
    || status.live_available === true
    || status.tier === 'live'
  ) {
    const model = sidecar.model_id ?? status.model_id;
    const device = sidecar.device ?? status.device;
    return {
      tone: 'live',
      label: 'Imagery classifier live',
      detail: [model, device].filter(Boolean).join(' · ')
        || 'The local land-cover classifier answered a health check.',
    };
  }
  const tiles = status.tiles ?? status.tile_count ?? status.precomputed_tiles;
  const tileCount = Array.isArray(tiles) ? tiles.length : Number(tiles);
  if (status.precomputed_available === true || status.tier === 'precomputed' || tileCount > 0) {
    return {
      tone: 'precomputed',
      label: 'Imagery precomputed only',
      detail: 'The live classifier is not reachable; checks return cached results and say so.',
    };
  }
  if (enabled === true || status.tier === 'unavailable') {
    return {
      tone: 'unavailable',
      label: 'Imagery check unavailable',
      detail: 'No classifier and no cached tile. A check would return no corroboration.',
    };
  }
  return null;
}

// Gemma has three distinct runtime states and collapsing them is a truthfulness
// bug, not a cosmetic one: "no analysis has run yet" is not the same claim as
// "the deterministic fallback produced this". Reading `provider` with a plain
// ternary reported the fallback before Gemma had been invoked at all, which is
// the first thing a reader sees on a cold load.
const GEMMA_NOT_RUN = 'not-run';
const GEMMA_HOSTED = 'hosted';
const GEMMA_FALLBACK = 'fallback';

function gemmaRuntimeState(analysis) {
  if (!analysis) return GEMMA_NOT_RUN;
  return analysis.provider === 'gemini_api' ? GEMMA_HOSTED : GEMMA_FALLBACK;
}

const GEMMA_STATE_LABEL = {
  [GEMMA_NOT_RUN]: 'Gemma not yet run',
  [GEMMA_HOSTED]: 'Hosted Gemma',
  [GEMMA_FALLBACK]: 'Declared fallback',
};

const GEMMA_STAGE_LABEL = {
  [GEMMA_NOT_RUN]: 'not yet run',
  [GEMMA_HOSTED]: 'hosted',
  [GEMMA_FALLBACK]: 'bounded fallback',
};

// Sending the operator to the override form scrolled an inner container and
// "3 Gemma fields unknown" tells an operator that something is wrong but not
// what, so it cannot be acted on without leaving the screen. This turns a list
// of field names into the sentence a person would say out loud.
function namePhrase(names, limit = 3) {
  const list = names.filter(Boolean);
  if (!list.length) return '';
  const head = list.slice(0, limit);
  const rest = list.length - head.length;
  const joined = head.length > 1
    ? `${head.slice(0, -1).join(', ')} and ${head.at(-1)}`
    : head[0];
  return rest > 0 ? `${joined}, and ${rest} more` : joined;
}

// focused a checkbox, which is invisible feedback: the button looked inert and
// read as broken. Flash the target so it is obvious where the interface moved to.
function focusApprovalForm() {
  const target = document.getElementById('approval-override');
  if (!target) return false;
  target.scrollIntoView({ block: 'center', behavior: 'smooth' });
  target.classList.add('targeted');
  window.setTimeout(() => target.classList.remove('targeted'), 1800);
  const field = target.querySelector('input[type="checkbox"], textarea');
  if (field) window.setTimeout(() => field.focus(), 320);
  return true;
}

// Where each asset is at the operator's current mission time, derived from the
// solver's own ETAs. Sent on re-optimization so a plan recomputed mid-mission
// departs from actual positions instead of teleporting the fleet to the depot.
function fleetPositionsAt(routes, elapsedMinutes) {
  const positions = {};
  if (!(elapsedMinutes > 0)) return positions;
  for (const route of routes ?? []) {
    const details = route.stop_details ?? [];
    if (!details.length) continue;
    const served = details.filter((stop) => (stop.eta_minutes ?? 0) <= elapsedMinutes);
    if (!served.length) continue;
    const last = served[served.length - 1];
    positions[route.vehicle_id] = {
      lat: last.lat,
      lng: last.lng,
      served_stops: served.map((stop) => stop.village_id),
    };
  }
  return positions;
}

function formatDecisionTime(value) {
  if (!value) return 'time not recorded';
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toUTCString();
}

function gemmaModelLabel(analysis) {
  const state = gemmaRuntimeState(analysis);
  if (state === GEMMA_NOT_RUN) return 'Gemma not yet run';
  return analysis.model?.replace(/^models\//, '') ?? GEMMA_STATE_LABEL[state];
}

function resourceLabel(resourceId, resourceTypes = {}) {
  return resourceTypes?.[resourceId]?.name ?? resourceId?.replaceAll('_', ' ') ?? 'Unknown resource';
}

function resourceUnit(resourceId, resourceTypes = {}) {
  return resourceTypes?.[resourceId]?.unit ?? 'unit';
}

// A gap the operator has already answered must not be offered again — neither in
// the review queue nor in the intake drawer. Evidence submitted against a gap
// carries `gap_target`, and anything sitting in the unsent queue carries
// `gapTarget`, so both are matched here. A gap that is still UNKNOWN after
// evidence was supplied is reported as "evidence supplied, still unsupported"
// rather than silently re-prompting, because the honest state is not "missing".
function evidenceGaps(analysis, queuedGapTargets = []) {
  const answered = new Set([
    ...(analysis?.evidence ?? [])
      .map((item) => item.gap_target)
      .filter(Boolean)
      .map((value) => String(value).toLowerCase()),
    ...queuedGapTargets.filter(Boolean).map((value) => String(value).toLowerCase()),
  ]);
  const output = analysis?.output;
  if (!output) {
    return [{
      id: 'analysis-not-run',
      label: 'Evidence has not been analyzed',
      detail: 'Run Gemma before treating any required field as supported.',
      field: 'analysis',
      tone: 'critical',
    }];
  }
  const gaps = [];
  const seen = new Set();
  const addGap = (gap) => {
    const signature = `${gap.field}:${gap.detail}`.toLowerCase();
    if (seen.has(signature)) return;
    seen.add(signature);
    if (
      answered.has(String(gap.label).toLowerCase())
      || answered.has(String(gap.detail).toLowerCase())
    ) {
      gaps.push({ ...gap, supplied: true, tone: 'attention' });
      return;
    }
    gaps.push(gap);
  };
  [
    ['incident_type', 'Incident classification', output.incident_type?.value],
    ['severity', 'Incident severity', output.severity?.expected],
    ['affected_population', 'Affected population', output.affected_population?.expected],
    ['medical_urgency', 'Medical urgency', output.medical_urgency?.value],
    ['accessibility_risk', 'Road accessibility', output.accessibility_risk?.value],
  ].forEach(([id, label, value]) => {
    if (value == null) {
      addGap({
        id: `unknown-${id}`,
        label,
        detail: `Gemma returned UNKNOWN for ${label.toLowerCase()}; provide a source that states or supports it.`,
        field: id,
        tone: 'critical',
      });
    }
  });
  (output.missing_information ?? []).forEach((detail, index) => {
    addGap({
      id: `missing-${index}-${detail}`,
      label: 'Missing information',
      detail,
      field: 'missing_information',
      tone: 'attention',
    });
  });
  (output.contradictions ?? []).forEach((contradiction, index) => {
    addGap({
      id: `contradiction-${index}`,
      label: 'Contradiction requires corroboration',
      detail: `${contradiction.claim_a} / ${contradiction.claim_b}`,
      field: 'contradiction',
      tone: 'critical',
    });
  });
  return gaps;
}

const GENERAL_EVIDENCE_PROMPTS = [
  {
    id: 'observation_time',
    question: 'When was this observed or last verified?',
    placeholder: 'Example: 14:20 NPT on 29 July…',
  },
  {
    id: 'observation_place',
    question: 'Where exactly does this apply?',
    placeholder: 'Village, ward, facility, road segment, or coordinates…',
  },
  {
    id: 'remaining_unknowns',
    question: 'What is still unknown or unverified?',
    placeholder: 'State any limits, uncertainty, or conflicting reports…',
  },
];

function evidenceIntakeConfig(gap) {
  if (!gap) {
    return {
      answerLabel: 'Report text',
      answerQuestion: 'What happened?',
      answerHint: 'Gemma receives this exact text. Unsupported facts remain UNKNOWN.',
      answerPlaceholder: 'Type or paste what happened… Include the place, observation time, injuries, access problems, damage, and requested supplies when known.',
      prompts: GENERAL_EVIDENCE_PROMPTS,
    };
  }

  const target = `${gap.field ?? ''} ${gap.label ?? ''} ${gap.detail ?? ''}`.toLowerCase();
  const base = {
    answerLabel: gap.field === 'follow_up_question' ? 'Direct answer' : 'Evidence answer',
    answerQuestion: gap.field === 'follow_up_question'
      ? gap.detail
      : `What did the source report that resolves “${gap.label}”?`,
    answerHint: 'Answer only from the named source. Say “unknown” for anything the source did not verify.',
    answerPlaceholder: 'Enter the source-backed answer… Include concrete values and units when available.',
    prompts: GENERAL_EVIDENCE_PROMPTS,
  };

  if (/medical|injur|casualt|fatal|patient|hospital|clinic|treatment/.test(target)) {
    return {
      ...base,
      answerPlaceholder: 'Example: 8 injured, 2 critical, no confirmed fatalities as of 14:20 NPT…',
      prompts: [
        {
          id: 'casualty_status',
          question: 'How many are injured, critical, missing, or confirmed dead?',
          placeholder: 'Use separate confirmed and unverified counts…',
        },
        {
          id: 'care_capacity',
          question: 'What care is needed, and what local capacity is available?',
          placeholder: 'Treatment need, facility status, beds, staff, evacuation need…',
        },
        GENERAL_EVIDENCE_PROMPTS[0],
      ],
    };
  }

  if (/location|located|where|coordinate|village|ward|place/.test(target)) {
    return {
      ...base,
      answerPlaceholder: 'Example: The event is verified at Sindhupalchok ward 6, 1.2 km east of the ward office…',
      prompts: [
        {
          id: 'exact_location',
          question: 'What exact village, ward, landmark, or coordinates did the source verify?',
          placeholder: 'Administrative area, landmark, latitude/longitude, or mapped point…',
        },
        {
          id: 'location_method',
          question: 'How did the source establish that location?',
          placeholder: 'Direct observation, GPS, mapped radio report, official bulletin…',
        },
        GENERAL_EVIDENCE_PROMPTS[0],
      ],
    };
  }

  if (/road|access|bridge|route|corridor|blocked|vehicle|airstrip/.test(target)) {
    return {
      ...base,
      answerPlaceholder: 'Example: Mechi corridor is blocked at chainage 42 km; motorcycles can pass, trucks cannot…',
      prompts: [
        {
          id: 'corridor_status',
          question: 'Which exact corridor or segment is open, restricted, or blocked?',
          placeholder: 'Road name, bridge, chainage, landmark, or edge…',
        },
        {
          id: 'vehicle_access',
          question: 'Which vehicle types can safely pass, and is an alternate route known?',
          placeholder: 'Truck, light vehicle, motorcycle, foot, helicopter only…',
        },
        GENERAL_EVIDENCE_PROMPTS[0],
      ],
    };
  }

  if (/population|people|household|displaced|stranded|affected/.test(target)) {
    return {
      ...base,
      answerPlaceholder: 'Example: Ward register confirms 620 affected people across 114 households…',
      prompts: [
        {
          id: 'population_breakdown',
          question: 'How many people or households are affected, displaced, or stranded?',
          placeholder: 'Provide counts and distinguish estimates from confirmed figures…',
        },
        {
          id: 'count_method',
          question: 'How was that count obtained?',
          placeholder: 'Ward register, field enumeration, facility list, estimate…',
        },
        GENERAL_EVIDENCE_PROMPTS[0],
      ],
    };
  }

  if (/resource|supply|stock|food|water|medicine|shelter|kit|payload/.test(target)) {
    return {
      ...base,
      answerPlaceholder: 'Example: 40 trauma kits and 1,200 L of water are needed at the ward office within 3 hours…',
      prompts: [
        {
          id: 'resource_quantity',
          question: 'Which resources are needed, in what quantity and unit?',
          placeholder: 'Resource, quantity, unit, and critical minimum…',
        },
        {
          id: 'delivery_requirement',
          question: 'Where and by when are they needed?',
          placeholder: 'Destination, deadline, handling constraint, receiving contact…',
        },
        {
          id: 'local_shortfall',
          question: 'What usable stock is already available locally?',
          placeholder: 'Available stock, reserved stock, and remaining shortfall…',
        },
      ],
    };
  }

  if (/severity|damage|destroyed|impact|essential service/.test(target)) {
    return {
      ...base,
      answerPlaceholder: 'Describe observed damage, service disruption, geographic extent, and whether conditions are worsening…',
      prompts: [
        {
          id: 'observed_damage',
          question: 'What physical damage or essential-service disruption was directly observed?',
          placeholder: 'Buildings, roads, power, water, communications, health services…',
        },
        {
          id: 'impact_extent',
          question: 'How widespread is the impact, and is it stable or worsening?',
          placeholder: 'Affected area, trend, continuing hazard, secondary risks…',
        },
        GENERAL_EVIDENCE_PROMPTS[0],
      ],
    };
  }

  if (/contradiction|corroborat|independent|conflict/.test(target)) {
    return {
      ...base,
      answerPlaceholder: 'State which claim the independent source supports, contradicts, or cannot verify…',
      prompts: [
        {
          id: 'verified_claim',
          question: 'Which conflicting claim did this source independently verify?',
          placeholder: 'Quote or summarize the verified claim and value…',
        },
        {
          id: 'verification_method',
          question: 'How did the source verify it independently?',
          placeholder: 'Direct observation, official register, sensor, separate witness…',
        },
        GENERAL_EVIDENCE_PROMPTS[0],
      ],
    };
  }

  if (/incident|classification|event type|hazard/.test(target)) {
    return {
      ...base,
      answerPlaceholder: 'Name the observed event or hazard without inferring an unsupported classification…',
      prompts: [
        {
          id: 'observed_event',
          question: 'What event or hazard was directly observed?',
          placeholder: 'Flood, landslide, bridge failure, fire, earthquake impact…',
        },
        {
          id: 'event_extent',
          question: 'Where and when did it begin, and is it still active?',
          placeholder: 'Origin, affected boundary, onset time, current state…',
        },
        GENERAL_EVIDENCE_PROMPTS[2],
      ],
    };
  }

  return base;
}

function StatusDot({ tone = 'nominal' }) {
  return <span className={`ops-status-dot ${tone}`} aria-hidden="true" />;
}

function useDialogFocus(open, onClose) {
  const dialogRef = useRef(null);
  const onCloseRef = useRef(onClose);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    if (!open) return undefined;

    const previouslyFocused = document.activeElement;
    const previousOverflow = document.body.style.overflow;
    const dialog = dialogRef.current;
    const focusableSelector = [
      'button:not([disabled])',
      'a[href]',
      'input:not([disabled])',
      'textarea:not([disabled])',
      'select:not([disabled])',
      '[tabindex]:not([tabindex="-1"])',
    ].join(',');

    document.body.style.overflow = 'hidden';
    const focusable = [...(dialog?.querySelectorAll(focusableSelector) ?? [])];
    const preferredFocus = dialog?.querySelector('[data-dialog-autofocus]');
    (preferredFocus ?? focusable[0] ?? dialog)?.focus();

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onCloseRef.current();
        return;
      }
      if (event.key !== 'Tab' || !dialog) return;

      const currentFocusable = [...dialog.querySelectorAll(focusableSelector)];
      if (!currentFocusable.length) {
        event.preventDefault();
        dialog.focus();
        return;
      }
      const first = currentFocusable[0];
      const last = currentFocusable.at(-1);
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = previousOverflow;
      previouslyFocused?.focus?.();
    };
  }, [open]);

  return dialogRef;
}

function Header({
  connected,
  transport = 'connecting',
  loading,
  run,
  analysis,
  onRun,
  onAddEvidence,
  onReset,
  resetBusy,
  activeWorkspace,
  onWorkspaceChange,
}) {
  const workspaces = [
    ['operations', 'Operations', 'public'],
    ['evidence', 'Gemma evidence', 'auto_awesome'],
    ['math', 'Math lab', 'calculate'],
    ['review', 'Review & authorize', 'fact_check'],
  ];
  return (
    <header className="ops-header">
      <button
        className="ops-brand"
        type="button"
        onClick={() => onWorkspaceChange('operations')}
        aria-label="Open RakshyaNet operations workspace"
      >
        <span className="ops-brand-mark"><Icon name="shield_lock" size={20} /></span>
        <span>
          <strong>RakshyaNet</strong>
          <small>Route Intelligence</small>
        </span>
      </button>

      <nav className="ops-nav" aria-label="Mission workspace">
        {workspaces.map(([id, label, icon]) => (
          <button
            key={id}
            type="button"
            className={activeWorkspace === id ? 'active' : ''}
            onClick={() => onWorkspaceChange(id)}
            aria-pressed={activeWorkspace === id}
          >
            <Icon name={icon} size={16} /> {label}
          </button>
        ))}
      </nav>

      <div className="ops-header-state" aria-label="System status">
        <span>
          <StatusDot tone={connected ? 'nominal' : transport === 'unavailable' ? 'attention' : 'critical'} />
          {connected
            ? 'Event stream connected'
            : transport === 'unavailable'
              ? 'Hosted event stream unavailable'
              : 'Event stream reconnecting'}
        </span>
        <span
          className={`ops-header-model ${gemmaRuntimeState(analysis)}`}
          title={analysis?.model ?? 'Gemma runtime not yet selected'}
        >
          <Icon name="auto_awesome" size={14} />
          {gemmaModelLabel(analysis)}
        </span>
      </div>

      <div className="ops-header-actions">
        {/* Deliberately labelled "Start fresh", not "Reset": it starts a new run
            rather than undoing anything, and the dialog states that scope. */}
        <button
          className="ops-button ghost ops-reset-button"
          type="button"
          onClick={onReset}
          disabled={resetBusy || loading}
          title="Discard this session's operator work and rebuild a clean baseline plan"
        >
          <Icon name={resetBusy ? 'progress_activity' : 'restart_alt'} size={17} />
          <span>{resetBusy ? 'Starting fresh…' : 'Start fresh'}</span>
        </button>
        <button className="ops-button ghost" type="button" onClick={onAddEvidence}>
          <Icon name="add_link" size={17} />
          <span>Add evidence</span>
        </button>
        <button className={`ops-button pipeline ${loading ? 'running' : ''}`} type="button" onClick={onRun} disabled={loading}>
          <Icon name={loading ? 'progress_activity' : 'refresh'} size={17} />
          <span>{loading ? 'Running pipeline…' : run?.run_id ? 'Run full pipeline' : 'Start analysis'}</span>
        </button>
      </div>
      {loading && <span className="ops-header-progress" aria-label="Analysis in progress" />}
    </header>
  );
}

function MissionBrief({ run, analysis, loading }) {
  const evidenceCount = analysis?.evidence?.length ?? 0;
  const checks = run?.result?.kkt_verification?.conditions ?? [];
  const passedChecks = checks.filter((item) => item.satisfied).length;
  const submittedReports = (analysis?.evidence ?? []).filter((item) => !item.simulated).length;
  return (
    <section className="ops-mission-brief" aria-labelledby="mission-title">
      <div>
        <span className="ops-eyebrow">
          Nepal national response / {run?.run_id ?? 'new mission'}
        </span>
        <h1 id="mission-title">Disaster route intelligence</h1>
        <p>
          Gemma converts uncertain reports into cited signals. Deterministic models
          recalculate the route plan. A human authorizes the result.
        </p>
      </div>

      <div className="ops-causal-path" aria-label="Decision pipeline">
        <div><span>01</span><b>Evidence</b><small>{evidenceCount} reports</small></div>
        <i aria-hidden="true">→</i>
        <div className="gemma"><span>02</span><b>Gemma</b><small>{GEMMA_STAGE_LABEL[gemmaRuntimeState(analysis)]}</small></div>
        <i aria-hidden="true">→</i>
        <div><span>03</span><b>Math</b><small>{loading ? 'recomputing' : checks.length ? `${passedChecks}/${checks.length} checks` : 'not run'}</small></div>
        <i aria-hidden="true">→</i>
        <div className="human"><span>04</span><b>Human</b><small>{run?.status?.replaceAll('_', ' ') ?? 'required'}</small></div>
      </div>

      <div className="ops-readiness" aria-label="Mission readiness">
        <span><StatusDot tone={run?.result ? 'nominal' : 'attention'} /> {run?.result ? 'Solver snapshot ready' : 'Solver awaiting input'}</span>
        <span><StatusDot tone={submittedReports ? 'nominal' : 'attention'} /> {submittedReports ? `${submittedReports} submitted reports` : `${evidenceCount} simulated reports`}</span>
        <span><StatusDot tone={checks.length && passedChecks === checks.length ? 'nominal' : 'attention'} /> {checks.length ? `${passedChecks}/${checks.length} checks passed` : 'Checks pending'}</span>
        <span><StatusDot tone={run?.status === 'approved' ? 'nominal' : 'attention'} /> {run?.status?.replaceAll('_', ' ') ?? 'assembling plan'}</span>
      </div>
    </section>
  );
}

const SCENARIO_STAGES = [
  ['baseline', 'Initial report'],
  ['disrupted', 'After road block'],
];

const TIMELINE_LABELS = {
  evidence_report: 'Report',
  optimization_requested: 'Baseline',
  road_block_report: 'Road block',
  evidence_disposition: 'Evidence owner',
  review_decision: 'Review',
};

const ScenarioSwitcher = memo(function ScenarioSwitcher({
  scenarios,
  selectedId,
  stage,
  activeScenarioId,
  busy,
  onSelect,
  onStageChange,
  onActivate,
}) {
  const selected = scenarios.find((item) => item.scenario_id === selectedId);
  // A bare spinner for what can be a 20-second wait reads as a hang. Activation
  // is one blocking backend call, so there is no honest per-stage percentage to
  // show — but elapsed time and what the system is currently waiting on are both
  // real, and they tell the operator whether to keep waiting.
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!busy) {
      setElapsed(0);
      return undefined;
    }
    const startedAt = Date.now();
    const id = window.setInterval(
      () => setElapsed((Date.now() - startedAt) / 1000),
      250,
    );
    return () => window.clearInterval(id);
  }, [busy]);

  const phase = elapsed < 1.5
    ? 'Sending the scenario reports to Gemma…'
    : elapsed < 20
      ? 'Gemma is extracting bounded values from the reports…'
      : elapsed < 45
        ? 'The hosted model is slow to answer. Still waiting — the declared fallback takes over if it times out.'
        : 'Hosted Gemma did not answer in time. Falling back to the deterministic screening path; the run will be labelled accordingly.';

  return (
    <section className="ops-scenario-switcher" aria-labelledby="scenario-switcher-title">
      <div className="ops-scenario-intro">
        <span className="ops-scenario-icon"><Icon name="science" size={18} /></span>
        <span>
          <span className="ops-eyebrow">Road closures · path 2 of 2 · {scenarios.length} timelines</span>
          <b id="scenario-switcher-title">Replay a scripted incident timeline</b>
          <small>
            Loads a bundled situation with its own reports and a closure written
            into the timeline, replacing the current picture. To close a road on
            the plan you already have, use the disruption lab in Math lab →
            Full diagnostics → Routes &amp; closures instead.
          </small>
        </span>
      </div>

      <label className="ops-scenario-select">
        <span>Mock scenario</span>
        <select
          name="mock_scenario"
          value={selectedId}
          onChange={(event) => onSelect(event.target.value)}
          disabled={!scenarios.length || busy}
        >
          {scenarios.map((scenario) => (
            <option key={scenario.scenario_id} value={scenario.scenario_id}>
              {scenario.title}
            </option>
          ))}
        </select>
      </label>

      <div className="ops-scenario-stage" role="group" aria-label="Scenario timeline stage">
        {SCENARIO_STAGES.map(([id, label]) => (
          <button
            key={id}
            type="button"
            className={stage === id ? 'active' : ''}
            aria-pressed={stage === id}
            onClick={() => onStageChange(id)}
            disabled={busy}
          >
            {label}
          </button>
        ))}
      </div>

      <button
        className="ops-button scenario"
        type="button"
        onClick={onActivate}
        disabled={!selected || busy}
      >
        <Icon name={busy ? 'progress_activity' : 'refresh'} size={17} />
        {busy ? `Loading scenario… ${elapsed.toFixed(0)}s` : 'Load this scenario'}
      </button>

      {busy && (
        <div className="ops-scenario-progress" role="status" aria-live="polite">
          <span className="ops-scenario-progress-bar" aria-hidden="true"><i /></span>
          <small>{phase}</small>
        </div>
      )}

      {selected && (
        <div className="ops-scenario-detail">
          <div>
            <span className="ops-simulated-label">SIMULATED</span>
            <b>{selected.title}</b>
            <small>{selected.description}</small>
          </div>
          <ol aria-label={`${selected.title} timeline`}>
            {(selected.timeline ?? []).map((item) => (
              <li
                key={item.step_id}
                className={
                  item.event_type === 'road_block_report' && stage === 'disrupted'
                    ? 'active'
                    : ''
                }
              >
                <span>t+{number(item.t_seconds)}s</span>
                <b>{TIMELINE_LABELS[item.event_type] ?? item.label}</b>
              </li>
            ))}
          </ol>
          <span className="ops-scenario-closure">
            <Icon name="block" size={14} />
            {stage === 'disrupted'
              ? `${selected.closure?.blocked_edge_ids?.join(', ')} closed`
              : `Closure arrives at t+${number(selected.closure?.t_seconds)}s`}
          </span>
          {activeScenarioId === selected.scenario_id && (
            <span className="ops-scenario-active"><StatusDot tone="nominal" /> Active runtime</span>
          )}
        </div>
      )}
    </section>
  );
});

function IncidentRail({ villages, urgency, selectedId, onSelect }) {
  const urgencyById = Object.fromEntries(
    urgency.map((item) => [item.village_id, item]),
  );
  const ranked = [...villages].sort(
    (a, b) =>
      (urgencyById[a.id]?.ranking ?? 999) -
      (urgencyById[b.id]?.ranking ?? 999),
  );
  const critical = urgency.filter((item) => item.has_critical_shortage).length;

  return (
    <aside className="ops-panel ops-incidents" aria-labelledby="incident-list-title">
      <div className="ops-panel-heading">
        <div>
          <span className="ops-eyebrow">Operational picture</span>
          <h2 id="incident-list-title">Active incidents</h2>
        </div>
        <span className="ops-count-badge">{villages.length}</span>
      </div>
      <div className="ops-incident-summary">
        <div><b>{villages.length}</b><span>reported</span></div>
        <div><b>{critical}</b><span>critical gaps</span></div>
      </div>
      <div className="ops-incident-list">
        {ranked.map((village) => {
          const score = urgencyById[village.id];
          return (
            <button
              type="button"
              className={`incident-row ops-incident-row ${selectedId === village.id ? 'selected' : ''}`}
              key={village.id}
              onClick={() => onSelect(village.id)}
              aria-pressed={selectedId === village.id}
            >
              <span className="rank">{score?.ranking ?? '—'}</span>
              <span>
                <b>{village.name}</b>
                <small>{village.accessibility?.replaceAll('_', ' ')} · {number(village.population)} people</small>
              </span>
              <strong>{number(score?.total_urgency, 2)}</strong>
            </button>
          );
        })}
      </div>
      <div className="ops-panel-note">
        <Icon name="info" size={16} />
        <span>Ranking is deterministic. Gemma contributes cited evidence signals but never dispatch authority.</span>
      </div>
    </aside>
  );
}

function formatClock(minutes) {
  const total = Math.max(0, Math.round(Number(minutes) || 0));
  return `${String(Math.floor(total / 60)).padStart(2, '0')}h ${String(total % 60).padStart(2, '0')}m`;
}

function MissionClock({
  elapsed,
  horizon,
  playing,
  speed,
  onSpeedChange,
  dispatchActive,
  runStatus,
  stopStatus,
  onScrub,
  onTogglePlay,
  onReset,
}) {
  if (!dispatchActive) {
    // The control used to be omitted entirely until a plan was approved, so
    // there was no way to discover that a mission clock exists at all. Render it
    // disabled instead, with the reason and the action that unlocks it.
    return (
      <div className="ops-mission-clock held" role="status">
        <Icon name="lock" size={16} />
        <div className="ops-clock-held-copy">
          <b>Mission clock locked · fleet held at depot</b>
          <small>
            {runStatus === 'awaiting_approval'
              ? 'Authorize this plan in Review & authorize to unlock the time slider. No asset moves before then.'
              : `Plan status: ${String(runStatus ?? 'none').replaceAll('_', ' ')}. The time slider unlocks once a plan is approved.`}
          </small>
        </div>
        <label className="ops-clock-slider" aria-hidden="true">
          <input type="range" min={0} max={100} value={0} disabled readOnly />
        </label>
        <span className="ops-clock-readout">
          <b>T+00:00</b>
          <small>locked</small>
        </span>
      </div>
    );
  }
  return (
    <div className="ops-mission-clock" role="group" aria-label="Mission clock">
      <button
        type="button"
        className={`ops-button compact ${playing ? 'active' : ''}`}
        onClick={onTogglePlay}
        aria-pressed={playing}
      >
        <Icon name={playing ? 'pause' : 'play_arrow'} size={16} />
        {playing ? 'Pause' : 'Play'}
      </button>
      <label className="ops-clock-slider">
        <span className="ops-sr-only">Mission elapsed time in minutes</span>
        <input
          type="range"
          min={0}
          max={Math.max(1, Math.ceil(horizon))}
          step={1}
          value={Math.min(elapsed, horizon)}
          onChange={(event) => onScrub(Number(event.target.value))}
          aria-valuetext={`${Math.round(elapsed)} of ${Math.ceil(horizon)} minutes`}
        />
      </label>
      <span className="ops-clock-readout">
        <b>T+{formatClock(elapsed)}</b>
        <small>of {formatClock(horizon)} plan horizon</small>
      </span>
      <span className="ops-clock-stops">
        <b>{stopStatus.served}/{stopStatus.total}</b>
        <small>stops served{stopStatus.nextEta != null ? ` · next T+${formatClock(stopStatus.nextEta)}` : ' · all complete'}</small>
      </span>
      <label className="ops-clock-speed">
        <span className="ops-sr-only">Playback speed</span>
        <select
          aria-label="Playback speed, mission minutes per real second"
          value={speed}
          onChange={(event) => onSpeedChange(Number(event.target.value))}
        >
          {MISSION_SPEEDS.map((option) => (
            <option key={option} value={option}>{option}× speed</option>
          ))}
        </select>
      </label>
      <button type="button" className="ops-button compact ghost" onClick={onReset}>
        <Icon name="undo" size={15} /> Reset
      </button>
    </div>
  );
}

const MapPanel = memo(function MapPanel({
  villages,
  depot,
  routes,
  selectedId,
  onSelect,
  addMode,
  draftIncident,
  onToggleAdd,
  onDraftIncident,
  roadNetwork,
  onOpenDisruption,
  clock,
}) {
  const feasibleRoutes = routes.filter(isFeasibleRoute);
  const infeasibleRouteCount = routes.length - feasibleRoutes.length;
  const selectedRouteCount = feasibleRoutes.filter((route) =>
    (route.stops ?? []).includes(selectedId)).length;
  return (
    <section className="ops-panel ops-map-panel" aria-labelledby="map-title">
      <div className="ops-panel-heading map-heading">
        <div>
          {/* The road graph is a bundled fixture, not a live feed. Labelling it
              "live" is exactly the claim this product exists to refuse. */}
          <span className="ops-eyebrow">Geospatial twin · fixture road graph</span>
          <h2 id="map-title">Nepal response grid</h2>
        </div>
        <div className="ops-map-heading-meta">
          <span><Icon name="route" size={15} /> {selectedRouteCount} of {feasibleRoutes.length} feasible routes reach selection</span>
          {infeasibleRouteCount > 0 && (
            <span><Icon name="warning" size={15} /> {infeasibleRouteCount} route exception excluded</span>
          )}
          <button
            className={`ops-button compact ${addMode ? 'active' : 'ghost'}`}
            type="button"
            onClick={onToggleAdd}
          >
            <Icon name="add_location_alt" size={16} />
            {addMode ? 'Cancel placement' : 'Report map evidence'}
          </button>
          {/* Placement is two steps — arm the button, then click the map — and
              nothing said so, which read as the button doing nothing. */}
          {addMode && (
            <span className="ops-place-hint" role="status">
              <Icon name="add_location_alt" size={14} />
              Now click anywhere on the map to place the report
            </span>
          )}
        </div>
      </div>
      {addMode && (
        <div className="ops-placement-banner" role="status">
          <Icon name="add_location_alt" size={18} />
          <span><b>Choose the reported location on the map</b><small>The next step collects source details and evidence. No incident or route is created from coordinates alone.</small></span>
          <button type="button" onClick={onToggleAdd}>Cancel</button>
        </div>
      )}
      <div className="ops-map-frame">
        <Suspense fallback={<div className="ops-map-loading"><Icon name="landscape" size={28} /> Loading terrain…</div>}>
          <TerrainMissionMap
            villages={villages}
            depot={depot}
            routes={routes}
            selectedId={selectedId}
            onSelect={onSelect}
            addMode={addMode}
            draftIncident={draftIncident}
            onDraftIncident={onDraftIncident}
            roadNetwork={roadNetwork}
            onOpenDisruption={onOpenDisruption}
            elapsedMinutes={clock.elapsed}
            dispatchActive={clock.dispatchActive}
          />
        </Suspense>
      </div>
      <MissionClock {...clock} />
      <div className="ops-map-footer">
        <span><Icon name="schedule" size={15} /> Operator-controlled mission clock</span>
        <span><Icon name="layers" size={15} /> Deterministic route geometry</span>
        <span><Icon name="navigation" size={15} /> Ground fleet follows mocked road graph</span>
        <span><StatusDot /> Telemetry synchronized</span>
      </div>
    </section>
  );
});

function IncidentInspector({
  selectedVillage,
  urgency,
  routes,
  onVerify,
  onOpenRoutes,
  imageryTarget = null,
  imageryBusy = false,
  imageryResult = null,
  imageryAvailable = false,
  imageryAvailabilityNotice = '',
  onAskGemmaImagery,
  onCheckImagery,
}) {
  const score = urgency.find((item) => item.village_id === selectedVillage?.id);
  const inbound = routes.filter((route) =>
    isFeasibleRoute(route) && (route.stops ?? []).includes(selectedVillage?.id));
  const routeExceptions = routes.filter((route) =>
    !isFeasibleRoute(route) && (route.stops ?? []).includes(selectedVillage?.id));
  const earliest = [...inbound].sort(
    (a, b) => Number(a.total_time_minutes ?? Infinity) - Number(b.total_time_minutes ?? Infinity),
  )[0];

  return (
    <aside className="ops-panel ops-inspector" aria-labelledby="incident-inspector-title">
      <div className="ops-panel-heading">
        <div>
          <span className="ops-eyebrow">Selected incident</span>
          <h2 id="incident-inspector-title">{selectedVillage?.name ?? 'Choose an incident'}</h2>
        </div>
        <span className="ops-count-badge">#{score?.ranking ?? '—'}</span>
      </div>
      <div className="ops-inspector-hero">
        <span>Priority index</span>
        <b>{number(score?.total_urgency, 2)}</b>
        <small>Dimensionless ranking index—not a probability.</small>
      </div>
      <dl className="ops-inspector-facts">
        <div><dt>Population exposed</dt><dd>{number(selectedVillage?.population)}</dd></div>
        <div><dt>Accessibility</dt><dd>{selectedVillage?.accessibility?.replaceAll('_', ' ') ?? 'Unknown'}</dd></div>
        <div><dt>Inbound assets</dt><dd>{inbound.length}</dd></div>
        <div><dt>Earliest arrival</dt><dd>{earliest ? `${number(earliest.total_time_minutes)} min` : 'Unassigned'}</dd></div>
        {routeExceptions.length > 0 && (
          <div><dt>Route exceptions</dt><dd>{routeExceptions.length} not dispatched</dd></div>
        )}
      </dl>
      <div className="ops-equation-compact">
        <span>Why this rank</span>
        <b>
          {number(score?.base_resource_urgency, 2)} need
          {' + '}{number(score?.critical_penalty, 2)} survival
          {' + '}{number(score?.external_signal, 3)} Gemma
        </b>
      </div>
      <div className="ops-inspector-actions">
        <button className="ops-button primary" type="button" onClick={onVerify}>
          <Icon name="add_link" size={17} /> Verify with evidence
        </button>
        <button className="ops-button ghost" type="button" onClick={onOpenRoutes}>
          <Icon name="route" size={17} /> Inspect routes
        </button>
      </div>
      {/* Overhead verification of the corridor serving this incident. Rendered
          only when a corridor can actually be named, because a button that
          cannot say what it would check is not an offer. Both paths append a
          citable record; neither can close a road on its own. */}
      {imageryTarget && (
        <div className="ops-imagery-actions">
          <span className="ops-eyebrow">Overhead check · {imageryTarget.corridorName}</span>
          {imageryAvailabilityNotice && (
            <p className="ops-imagery-unavailable" id="imagery-action-availability" role="status">
              {imageryAvailabilityNotice}
            </p>
          )}
          <div className="ops-imagery-buttons">
            <button
              className="ops-button ghost"
              type="button"
              onClick={() => onAskGemmaImagery?.(imageryTarget)}
              disabled={imageryBusy || !imageryAvailable}
              aria-describedby={imageryAvailabilityNotice ? 'imagery-action-availability' : undefined}
              title={imageryAvailable
                ? 'Gemma emits the tool call itself; the record shows the operator asked.'
                : imageryAvailabilityNotice}
            >
              <Icon name={imageryBusy ? 'progress_activity' : 'auto_awesome'} size={16} />
              Ask Gemma to verify
            </button>
            <button
              className="ops-button ghost"
              type="button"
              onClick={() => onCheckImagery?.(imageryTarget)}
              disabled={imageryBusy || !imageryAvailable}
              aria-describedby={imageryAvailabilityNotice ? 'imagery-action-availability' : undefined}
              title={imageryAvailable
                ? 'Skips the model: one classifier call, record appended directly.'
                : imageryAvailabilityNotice}
            >
              <Icon name={imageryBusy ? 'progress_activity' : 'public'} size={16} />
              Check imagery now
            </button>
          </div>
          {imageryResult && (
            <p className={`ops-imagery-result ${imageryResult.tone}`}>{imageryResult.text}</p>
          )}
          <small>
            Imagery corroborates a report. It never closes a corridor — validation
            rejects a closure whose only support is an imagery record.
          </small>
        </div>
      )}
      <p className="ops-panel-note">
        Map selection changes only this inspector. Authorization always applies to the full versioned run.
      </p>
    </aside>
  );
}

function DecisionPanel({
  run,
  analysis,
  selectedVillage,
  urgency,
  onApprove,
  onReject,
  reviewBusy,
  pipelineBusy,
  onOpenDiagnostics,
  onReviewIssues,
  onSupplyGap,
}) {
  const result = run?.result;
  const score = urgency.find((item) => item.village_id === selectedVillage?.id);
  const routes = result?.vrp_solution?.routes ?? [];
  const inbound = routes.filter((item) =>
    isFeasibleRoute(item) && (item.stops ?? []).includes(selectedVillage?.id),
  );
  const checks = result?.kkt_verification?.conditions ?? [];
  const failedChecks = checks.filter((item) => !item.satisfied);
  const passed = checks.filter((item) => item.satisfied).length;
  const evidence = analysis?.evidence ?? [];
  const simulatedCount = evidence.filter((item) => item.simulated).length;
  const submittedCount = evidence.length - simulatedCount;
  const simulatedOnly = evidence.length > 0 && submittedCount === 0;
  const signal = result?.gemma_signal;
  const effect = signal?.effects?.find(
    (item) => item.village_id === selectedVillage?.id,
  );
  const activeRoadBlocks = result?.vrp_solution?.active_road_blocks ?? [];
  const unassignedCritical = urgency.filter(
    (item) =>
      item.has_critical_shortage &&
      !routes.some((route) =>
        isFeasibleRoute(route) && (route.stops ?? []).includes(item.village_id)),
  );
  // The old count looked at three of the five extraction fields directly and
  // ignored incident classification, affected population and every flagged
  // contradiction. A run whose only defects were in the two uncounted fields
  // therefore scored zero warnings, rendered no override form, and approved on
  // the first click with required evidence still UNKNOWN. The gap machinery the
  // rest of the product already uses covers all five plus contradictions, so
  // authorization now reads from that and nothing else.
  const unresolvedGaps = evidenceGaps(analysis).filter((gap) => !gap.supplied);
  const requiredGaps = unresolvedGaps.filter((gap) => gap.tone === 'critical');
  const advisoryGaps = unresolvedGaps.filter((gap) => gap.tone !== 'critical');
  const missingFieldPhrase = namePhrase(
    requiredGaps
      .filter((gap) => gap.field !== 'contradiction' && gap.field !== 'analysis')
      .map((gap) => gap.label.toLowerCase()),
  );
  const followUpQuestions = analysis?.output?.follow_up_questions ?? [];
  const questionDispositions = analysis?.question_dispositions ?? [];
  const ownedQuestions = new Set(
    questionDispositions.map((item) => item.question_id),
  ).size;
  const unownedQuestions = Math.max(0, followUpQuestions.length - ownedQuestions);
  const analysisMatches =
    Boolean(run?.analysis_id) &&
    run.analysis_id === analysis?.analysis_id &&
    (!signal?.analysis_id || signal.analysis_id === analysis?.analysis_id);
  const routesFeasible =
    run?.route_feasible !== false &&
    routes.length > 0 &&
    routes.every((route) => route.feasible !== false);
  const hardBlocked = !analysisMatches || failedChecks.length > 0 || !routesFeasible;
  const warningCount = (
    unassignedCritical.length
    + unresolvedGaps.length
    + followUpQuestions.length
    + (simulatedOnly ? 1 : 0)
    + activeRoadBlocks.length
  );
  // The server computes the authoritative refusal reasons and returns them on
  // the run record. The interface previously re-derived its own and never
  // showed these, so an operator could satisfy every on-screen condition and
  // still be refused with no explanation.
  const backendBlockers = run?.approval_blockers ?? [];
  const decided = Boolean(run?.reviewed_at);
  const [acknowledged, setAcknowledged] = useState(false);
  const [rationale, setRationale] = useState('');

  useEffect(() => {
    setAcknowledged(false);
    setRationale('');
  }, [run?.run_id]);
  const canReview =
    run?.status === 'awaiting_approval' &&
    !pipelineBusy &&
    !hardBlocked &&
    (warningCount === 0 || (acknowledged && rationale.trim().length >= 12));
  const needsIssueReview = hardBlocked || (warningCount > 0 && !canReview);
  // Warnings that an operator can clear by acknowledging, as opposed to a hard
  // block that no acknowledgement can override.
  const needsApproval = (
    run?.status === 'awaiting_approval' && warningCount > 0 && !hardBlocked
  );
  const reviewIssueCount = (
    warningCount
    + failedChecks.length
    + (analysisMatches ? 0 : 1)
    + (routesFeasible ? 0 : 1)
  );

  return (
    <aside className="ops-panel ops-decision" aria-labelledby="decision-title">
      <div className="ops-decision-top">
        <div>
          <span className="ops-eyebrow">Operator decision brief</span>
          <h2 id="decision-title">Run authorization</h2>
          <small>{run?.run_id ?? 'No versioned run'}</small>
        </div>
        <span className={`ops-plan-state ${run?.status ?? 'pending'}`}>
          {run?.status?.replaceAll('_', ' ') ?? 'pending'}
        </span>
      </div>

      <div className="ops-decision-scroll">
      {decided && (
        <div className={`ops-decision-receipt ${run.status}`}>
          <Icon name={run.status === 'approved' ? 'verified' : 'undo'} size={18} />
          <div>
            <b>
              {run.status === 'approved' ? 'Authorized' : 'Changes requested'} by{' '}
              {run.reviewed_by ?? 'unknown operator'}
            </b>
            <small>{formatDecisionTime(run.reviewed_at)} · run {run.run_id}</small>
            {run.review_notes && <p>“{run.review_notes}”</p>}
          </div>
        </div>
      )}

      {backendBlockers.length > 0 && (
        <div className="ops-backend-blockers" role="status">
          <Icon name="block" size={17} />
          <div>
            <b>Server refuses approval</b>
            <ul>
              {backendBlockers.map((blocker) => (
                <li key={blocker}>{blocker}</li>
              ))}
            </ul>
            <small>
              Reported by the run record itself. The server re-checks these on submit,
              so clearing them in the interface is not sufficient.
            </small>
          </div>
        </div>
      )}

      {simulatedOnly && (
        <div className="ops-demo-data">
          <Icon name="science" size={17} />
          <div>
            <b>Demo data — {simulatedCount}/{evidence.length} reports simulated</b>
            <small>This approval records a demonstration plan, not an operational dispatch.</small>
          </div>
        </div>
      )}

      {/* The urgency figure used to be repeated here in a red tile directly
          above the arithmetic that derives it. One number, one place: the
          plinth below owns it, and the rank moved into it. */}
      <div className="ops-priority-block">
        <div className="ops-priority-copy">
          <span>Selected incident context only · {selectedVillage?.name ?? 'none'}</span>
          <strong>{number(selectedVillage?.population)} people exposed</strong>
          <small>{selectedVillage?.accessibility?.replaceAll('_', ' ') ?? 'access unknown'} · {inbound.length} inbound routes</small>
        </div>
      </div>
      {/* The signature readout. This is the one line in the product where the
          model's contribution to a ranking decision is visible, bounded, and
          checkable in a single glance — two deterministic terms computed from
          stock and population, one bounded term from Gemma, and the sum that
          orders the incident. Every term is monospace because every term can be
          verified against /api/optimization/runs/{id}. Only the Gemma term
          carries the model hue. */}
      <div className="ops-urgency-eq">
        <div className="ueq-head">
          <span className="ops-eyebrow">How this incident got its rank</span>
          <button
            className="ueq-explain"
            type="button"
            onClick={() => onOpenDiagnostics('urgency')}
          >
            Explain score
          </button>
        </div>
        <p className="ueq-line">
          <span className="ueq-term">
            <b>{number(score?.base_resource_urgency, 2)}</b>
            <small>need</small>
          </span>
          <span className="ueq-op" aria-hidden="true">+</span>
          <span className="ueq-term">
            <b>{number(score?.critical_penalty, 2)}</b>
            <small>survival</small>
          </span>
          <span className="ueq-op" aria-hidden="true">+</span>
          <span className="ueq-term" data-actor="model">
            <b>{number(score?.external_signal, 3)}</b>
            <small>Gemma</small>
          </span>
          <span className="ueq-op" aria-hidden="true">=</span>
          <span className="ueq-term ueq-total">
            <b>{number(score?.total_urgency, 2)}</b>
            <small>urgency · rank {score?.ranking ?? '—'}/{urgency.length || '—'}</small>
          </span>
        </p>
        {/* The bound sits inside the same plinth on purpose: the ratio between
            the model's ceiling and the survival penalty IS the authority
            argument, and splitting it into a footnote loses it. */}
        <p className="ueq-bound">
          <span><b>Gemma ≤ 1.00</b></span>
          <span>·</span>
          <span>survival penalty = {number(score?.critical_penalty, 2)}</span>
        </p>
        <p className="ueq-note">
          Need and survival are computed from stock levels and population. The Gemma
          term is the model&rsquo;s only numeric influence on this ranking — signal ×
          system confidence, capped at 1.00. It can move an incident up the queue; it
          cannot outweigh a measured shortage.
        </p>
      </div>

      <div className="ops-decision-metrics">
        <div><span>Evidence</span><b>{evidence.length}</b><small>{submittedCount} submitted · {simulatedCount} simulated</small></div>
        <div><span>System confidence</span><b>{percent(analysis?.system_confidence)}</b><small>calibrated</small></div>
        <div>
          <span>Gemma score impact</span>
          <b>{effect ? `${number(effect.baseline_urgency, 2)} → ${number(effect.final_urgency, 2)}` : 'No change'}</b>
          <small>{effect ? `+${number(effect.urgency_delta, 3)} · rank ${effect.baseline_rank} → ${effect.final_rank}` : 'not applied here'}</small>
        </div>
      </div>

      <div className={`ops-exception-banner ${hardBlocked ? 'blocked' : warningCount ? 'warning' : 'ready'}`}>
        <Icon name={hardBlocked ? 'block' : warningCount ? 'warning' : 'verified'} size={18} />
        <div>
          <b>{hardBlocked ? 'Approval blocked' : warningCount ? 'Override acknowledgement required' : 'Ready for operator review'}</b>
          <small>
            {!analysisMatches
              ? 'Displayed Gemma analysis does not match this solver snapshot.'
              : failedChecks.length
                ? `${failedChecks.length} solver checks failed.`
                : !routesFeasible
                  ? 'Route plan is infeasible or empty. Approval is blocked until the solver produces a feasible assigned route set.'
                : warningCount
                  ? [
                      missingFieldPhrase && `Required evidence is still UNKNOWN: ${missingFieldPhrase}.`,
                      unassignedCritical.length && `${unassignedCritical.length} critical locations unassigned.`,
                      followUpQuestions.length && `${ownedQuestions}/${followUpQuestions.length} follow-up questions owned${unownedQuestions ? `, ${unownedQuestions} unowned` : ''}.`,
                      simulatedOnly && 'Simulated evidence only.',
                      activeRoadBlocks.length && `${activeRoadBlocks.length} active road closure${activeRoadBlocks.length === 1 ? '' : 's'}.`,
                    ].filter(Boolean).join(' ')
                  : 'Evidence, model output, and solver snapshot are aligned.'}
          </small>
        </div>
      </div>

      <div className="ops-causal-proof">
        <div><span>01</span><div><b>Gemma extracted</b><small>{analysis?.output?.incident_type?.value ?? 'incident type unknown'} · {evidence.length} cited reports</small></div></div>
        <div><span>02</span><div><b>Math recalculated</b><small>{routes.length} routes · {number(result?.vrp_solution?.total_distance_km, 0)} km evaluated</small></div></div>
        <div><span>03</span><div><b>Guardrails checked</b><small>{passed} / {checks.length || '—'} diagnostics passed</small></div></div>
      </div>

      <div className="ops-approval-scope">
        <span className="ops-eyebrow">Run approval scope</span>
        <p>
          This {simulatedOnly ? 'records a demo review' : 'authorizes coordination'} for run <b>{run?.run_id ?? '—'}</b>:
          its ranking, allocation matrix, assigned vehicle routes, and validation
          snapshot across all {urgency.length} incidents. It does not dispatch vehicles.
        </p>
        <div>
          <span><Icon name="description" size={14} /> {evidence.length} evidence records</span>
          <span><Icon name="route" size={14} /> {routes.length} assigned routes</span>
          <span><Icon name="verified" size={14} /> {passed} checks passed</span>
          <span><Icon name="help_outline" size={14} /> {ownedQuestions}/{followUpQuestions.length} evidence questions owned</span>
        </div>
      </div>

      {/* An operator cannot authorize an integer. This is the plan itself —
          every asset, where it goes, how far, and whether it is feasible —
          rendered in the same panel as the button that approves it. */}
      {routes.length > 0 && (
        <div className="ops-route-manifest">
          <span className="ops-eyebrow">Route manifest under authorization</span>
          <div className="ops-route-manifest-scroll">
            <table>
              <caption className="ops-sr-only">
                Assigned routes in run {run?.run_id ?? 'unknown'}
              </caption>
              <thead>
                {/* State sits second, not last. The table scrolls horizontally
                    in a narrow panel, and feasibility is the one column an
                    operator must never have to scroll to find. */}
                <tr>
                  <th scope="col">Asset</th>
                  <th scope="col">State</th>
                  <th scope="col">Mode</th>
                  <th scope="col">Stops</th>
                  <th scope="col">Distance</th>
                  <th scope="col">Time</th>
                </tr>
              </thead>
              <tbody>
                {routes.map((route) => {
                  const feasible = isFeasibleRoute(route);
                  return (
                    <tr key={route.vehicle_id} className={feasible ? '' : 'infeasible'}>
                      <th scope="row">{route.vehicle_id}</th>
                      <td>
                        <span className={`ops-route-state ${feasible ? 'ok' : 'bad'}`}>
                          {feasible ? 'feasible' : 'excluded'}
                        </span>
                      </td>
                      <td>{route.transport_mode ?? '—'}</td>
                      <td className="ops-route-stops">
                        {(route.stops ?? []).join(', ') || 'none'}
                      </td>
                      <td>{number(route.total_distance_km, 0)} km</td>
                      <td>{number(route.total_time_minutes, 0)} min</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <small>
            Distances are the raw sum of edge lengths, not the terrain-weighted
            search cost.
          </small>
        </div>
      )}

      {warningCount > 0 && !hardBlocked && (
        <div className="ops-override" id="approval-override">
          {/* The gaps are listed HERE, at the button, rather than only on the
              evidence workspace two clicks away. An operator who forgot to fill
              a field finds out at the moment they try to authorize, sees which
              field it is, and can resolve it without losing the review. */}
          {requiredGaps.length > 0 && (
            <div className="ops-override-gaps">
              <div className="ops-override-gaps-head">
                <Icon name="error" size={17} />
                <div>
                  <b>
                    {requiredGaps.length} required{' '}
                    {requiredGaps.length === 1 ? 'field is' : 'fields are'} unresolved
                  </b>
                  <small>
                    Supply a source for each, or acknowledge below and record why
                    coordination may proceed without it.
                  </small>
                </div>
              </div>
              <ul>
                {requiredGaps.map((gap) => (
                  <li key={gap.id}>
                    <div>
                      <b>{gap.label} is UNKNOWN</b>
                      <small>{gap.detail}</small>
                    </div>
                    <button
                      type="button"
                      className="ops-button ghost compact"
                      onClick={() => onSupplyGap?.(gap)}
                    >
                      <Icon name="add_link" size={15} /> Supply now
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {advisoryGaps.length > 0 && (
            <p className="ops-override-advisory">
              <Icon name="info" size={15} />
              <span>
                {advisoryGaps.length} further item{advisoryGaps.length === 1 ? '' : 's'} the
                model flagged as missing or unverified — these do not name a required
                field, but they are part of what you are acknowledging.
              </span>
            </p>
          )}
          <label>
            <input
              type="checkbox"
              name="acknowledge_exceptions"
              checked={acknowledged}
              onChange={(event) => setAcknowledged(event.target.checked)}
            />
            {missingFieldPhrase
              ? `I am authorizing with ${missingFieldPhrase} still UNKNOWN, and I acknowledge every other unresolved exception in this snapshot.`
              : 'I acknowledge the unresolved exceptions in this snapshot.'}
          </label>
          <label>
            {missingFieldPhrase ? 'Justification for authorizing without this evidence' : 'Override rationale'}
            <textarea
              name="override_rationale"
              value={rationale}
              onChange={(event) => setRationale(event.target.value)}
              rows="2"
              minLength="12"
              aria-describedby="override-rationale-help"
              placeholder={missingFieldPhrase
                ? `Why can coordination proceed with ${missingFieldPhrase} unknown, and who owns closing it?`
                : 'Why may coordination proceed, and who owns mitigation?'}
            />
            <small id="override-rationale-help">
              Minimum 12 characters · {rationale.trim().length}/12 entered ·
              recorded verbatim on the run
            </small>
          </label>
        </div>
      )}
      </div>

      {/* The primary button used to read "Review N issues" and route to a
          diagnostics dialog, while the two controls that actually unblock
          approval sat below a scroll. That reads as "approve is broken". State
          the requirement, and show how far the operator has got. */}
      {needsApproval && (
        <div className="ops-approval-requirement">
          <Icon name="fact_check" size={17} />
          <div>
            <b>
              {missingFieldPhrase
                ? `Authorization is held: ${missingFieldPhrase} ${requiredGaps.length === 1 ? 'is' : 'are'} UNKNOWN`
                : 'Two steps unlock authorization'}
            </b>
            {missingFieldPhrase && (
              <small className="ops-approval-requirement-lede">
                Supply the evidence below, or take the override on the record.
              </small>
            )}
            <ol>
              <li className={acknowledged ? 'met' : ''}>
                {acknowledged ? '✓' : '1'} Acknowledge the unresolved exceptions
              </li>
              <li className={rationale.trim().length >= 12 ? 'met' : ''}>
                {rationale.trim().length >= 12 ? '✓' : '2'} Write an override
                rationale ({rationale.trim().length}/12 characters)
              </li>
            </ol>
          </div>
          <div className="ops-approval-requirement-actions">
            {/* Inspecting the exceptions must stay possible before acknowledging
                them — the primary button no longer opens diagnostics. */}
            <button type="button" className="ops-text-button" onClick={onReviewIssues}>
              Review {reviewIssueCount} issue{reviewIssueCount === 1 ? '' : 's'}
            </button>
            <button
              type="button"
              className="ops-button compact"
              onClick={() => {
                const target = document.getElementById('approval-override');
                target?.scrollIntoView({ block: 'center', behavior: 'smooth' });
                target?.querySelector('input,textarea')?.focus();
              }}
            >
              Go to the form
            </button>
          </div>
        </div>
      )}

      <div className="ops-decision-actions">
        <button
          className="ops-button reject"
          type="button"
          onClick={() => onReject()}
          disabled={run?.status !== 'awaiting_approval' || reviewBusy || pipelineBusy}
        >
          <Icon name="undo" size={17} /> Request changes
        </button>
        <button
          className={`ops-button ${needsIssueReview ? 'issue-review' : 'approve'}`}
          type="button"
          onClick={() => {
            if (needsApproval && !canReview) {
              focusApprovalForm();
              return;
            }
            return needsIssueReview ? onReviewIssues() : onApprove(rationale);
          }}
          disabled={run?.status !== 'awaiting_approval' || reviewBusy || pipelineBusy}
        >
          <Icon name={reviewBusy ? 'progress_activity' : needsIssueReview ? 'fact_check' : 'check'} size={18} />
          {reviewBusy
            ? 'Recording…'
            : needsApproval && !canReview
              ? (missingFieldPhrase
                  ? 'Resolve or override the missing evidence'
                  : 'Acknowledge below to authorize')
              : needsIssueReview
                ? `Review ${reviewIssueCount} issue${reviewIssueCount === 1 ? '' : 's'}`
                : simulatedOnly ? 'Approve demo plan' : 'Approve for coordination'}
        </button>
      </div>
      <div className="ops-human-gate">
        <Icon name="lock" size={15} />
        Gemma cannot approve, allocate, route, or dispatch.
      </div>
    </aside>
  );
}

// The track requires Gemma to orchestrate the engine through native function
// calling. A reader should not have to take that on trust from a document, so
// this renders the actual call: what the model asked for, what was checked, and
// what the engine was allowed to execute.
// Wayfinding. Three separate complaints — "I can't find where to initiate",
// "where to supply evidence", "where to approve" — all came from the same gap:
// the workspace tabs sit in the header competing with the brand and the run
// button, and nothing anywhere says what to do next. This strip is always
// visible, shows which stages are done, and its right-hand button performs the
// single next action, navigating to the workspace that owns it.
function StageStepper({
  activeWorkspace,
  onWorkspaceChange,
  run,
  analysis,
  loading,
  onRun,
}) {
  const hasAnalysis = Boolean(analysis?.analysis_id);
  const hasPlan = Boolean(run?.result?.vrp_solution?.routes?.length);
  const decided = Boolean(run?.reviewed_at);
  const awaiting = run?.status === 'awaiting_approval';

  const stages = [
    {
      id: 'operations',
      n: 1,
      label: 'Operations',
      done: hasPlan,
      hint: 'Build a plan',
    },
    {
      id: 'evidence',
      n: 2,
      label: 'Gemma evidence',
      done: hasAnalysis,
      hint: 'Check what Gemma extracted, or add a report',
    },
    {
      id: 'math',
      n: 3,
      label: 'Math lab',
      done: hasPlan,
      hint: 'Inspect the maths and beat the baseline',
    },
    {
      id: 'review',
      n: 4,
      label: 'Review & authorize',
      done: decided,
      hint: 'Authorize the plan',
    },
  ];

  let next;
  if (loading) {
    next = { label: 'Working…', target: null };
  } else if (!hasPlan) {
    next = { label: 'Build the first plan', target: 'operations', act: onRun };
  } else if (awaiting) {
    next = { label: 'Authorize this plan', target: 'review' };
  } else if (decided) {
    next = { label: 'Inspect the authorized plan', target: 'math' };
  } else {
    next = { label: 'Inspect the extraction', target: 'evidence' };
  }

  return (
    <nav className="ops-stepper" aria-label="Mission stages">
      <ol>
        {stages.map((stage) => {
          const current = activeWorkspace === stage.id;
          return (
            <li key={stage.id}>
              {/* The accessible name deliberately omits the workspace label:
                  the header already exposes a tab with that exact name, and two
                  controls sharing one accessible name is ambiguous for assistive
                  tech and for anything selecting by name. */}
              <button
                type="button"
                className={`${current ? 'current' : ''} ${stage.done ? 'done' : ''}`}
                aria-current={current ? 'step' : undefined}
                aria-label={`Go to stage ${stage.n} of 4: ${stage.hint}`}
                onClick={() => onWorkspaceChange(stage.id)}
                title={stage.hint}
              >
                <span className="ops-stepper-n" aria-hidden="true">
                  {stage.done ? '✓' : stage.n}
                </span>
                <span className="ops-stepper-label">{stage.label}</span>
              </button>
            </li>
          );
        })}
      </ol>

      <div className="ops-stepper-next">
        <span>Next</span>
        <button
          type="button"
          className="ops-button primary compact"
          disabled={loading || !next.target}
          onClick={() => {
            if (next.act) next.act();
            if (next.target && next.target !== activeWorkspace) {
              onWorkspaceChange(next.target);
              if (next.target === 'review') {
                window.setTimeout(focusApprovalForm, 500);
              }
              return;
            }
            // Already on the target workspace: perform the action instead of
            // navigating nowhere.
            if (next.target === 'review') focusApprovalForm();
          }}
        >
          <Icon name={loading ? 'progress_activity' : 'navigation'} size={16} />
          {next.label}
        </button>
      </div>
    </nav>
  );
}

// The primary action was only reachable from a header button competing with the
// workspace tabs, so on a first visit there was no obvious way to start anything.
// This states the three ways to begin, in plain language, at the top of the first
// workspace, and calls out which one the Route Intelligence track cares about.
function MissionLauncher({
  run,
  analysis,
  loading,
  orchestrateBusy,
  onRun,
  onOrchestrate,
  onOpenEvidence,
}) {
  const started = Boolean(run?.run_id);
  const busy = loading || orchestrateBusy;
  const submittedCount = (analysis?.evidence ?? []).filter(
    (item) => !item.simulated,
  ).length;

  return (
    <section
      className={`ops-launcher ${started ? '' : 'cold'}`}
      aria-labelledby="launcher-title"
    >
      <div className="ops-launcher-copy">
        <span className="ops-eyebrow">
          {started ? 'Current plan' : 'Start here'}
        </span>
        <h2 id="launcher-title">
          {started
            ? `Plan ${run.run_id} · ${run.status?.replaceAll('_', ' ')}`
            : 'No plan has been computed yet'}
        </h2>
        <p>
          {started
            ? `Built from Gemma analysis ${run.analysis_id ?? 'none'}. Re-run either way below to replace it with a new versioned plan.`
            : 'Choose how to build the first plan. Both paths run the same deterministic engine; they differ in who decides what to compute.'}
        </p>
      </div>

      {/* Submitted reports were carried into the analysis but never shown back
          here, so after analysing it looked as though the evidence had been lost
          and had to be entered again. */}
      {submittedCount > 0 && (
        <div className="ops-launcher-evidence">
          <Icon name="description" size={16} />
          <span>
            <b>
              {submittedCount} operator report{submittedCount === 1 ? '' : 's'} already
              analysed
            </b>{' '}
            — carried into every re-run. You do not need to re-enter them.
          </span>
          <button type="button" className="ops-text-button" onClick={onOpenEvidence}>
            View in Gemma evidence
          </button>
        </div>
      )}

      <div className="ops-launcher-actions">
        <button
          type="button"
          className="ops-launch-card"
          onClick={onRun}
          disabled={busy}
        >
          <span className="ops-launch-head">
            <Icon name={loading ? 'progress_activity' : 'refresh'} size={18} />
            <b>{loading ? 'Running…' : started ? 'Re-run pipeline' : 'Run full pipeline'}</b>
          </span>
          <small>
            You choose the inputs. Gemma reads the field reports, then the engine
            computes urgency, routes, and allocation.
          </small>
        </button>

        <button
          type="button"
          className="ops-launch-card primary"
          onClick={onOrchestrate}
          disabled={busy}
        >
          <span className="ops-launch-head">
            <Icon name={orchestrateBusy ? 'progress_activity' : 'auto_awesome'} size={18} />
            <b>{orchestrateBusy ? 'Gemma is calling…' : 'Let Gemma run the engine'}</b>
          </span>
          <small>
            Gemma decides. It calls <code>list_corridor_status</code>, then{' '}
            <code>run_optimization</code> through native function calling. Takes
            around a minute.
          </small>
        </button>
      </div>

      <p className="ops-launcher-note">
        <Icon name="lock" size={14} />
        Neither path dispatches anything. Both produce a plan that a human must
        authorize in <b>Stage 4 · Review &amp; authorize</b>.
        {analysis ? '' : ' No Gemma analysis exists yet.'}
      </p>
    </section>
  );
}

// The raw model exchange, unabstracted: the exact prompt sent, the exact JSON
// returned, and whatever reasoning the provider chose to expose. Nothing here is
// validated, and it is labelled that way, because the whole point of the panel is
// to let a reader check the pipeline's own account of itself against the wire.
function RawExchangePanel({ analysis }) {
  const [open, setOpen] = useState('response');

  if (!analysis) {
    return (
      <section className="ops-panel ops-raw-exchange" aria-labelledby="raw-title">
        <div className="ops-panel-heading">
          <div>
            <span className="ops-eyebrow">Unabstracted model exchange</span>
            <h2 id="raw-title">Raw prompt, reasoning, and response</h2>
          </div>
        </div>
        <div className="ops-empty-state">
          <Icon name="function" size={22} />
          <b>No analysis has run yet</b>
          <p>Run the pipeline to capture the exchange.</p>
        </div>
      </section>
    );
  }

  const reasoning = analysis.model_reasoning ?? [];
  const views = [
    ['prompt', `Prompt sent (${(analysis.prompt_sent ?? '').length} chars)`],
    ['reasoning', `Model reasoning (${reasoning.length})`],
    ['response', `Raw response (${(analysis.raw_response_text ?? '').length} chars)`],
  ];

  return (
    <section className="ops-panel ops-raw-exchange" aria-labelledby="raw-title">
      <div className="ops-panel-heading">
        <div>
          <span className="ops-eyebrow">Unabstracted model exchange</span>
          <h2 id="raw-title">Raw prompt, reasoning, and response</h2>
          <p>
            Exactly what was sent to {analysis.model} and exactly what came back,
            before any validation. Shown so the pipeline&apos;s account of itself
            can be checked against the wire.
          </p>
        </div>
      </div>

      <div className="ops-subnav" role="tablist" aria-label="Raw exchange view">
        {views.map(([id, label]) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={open === id}
            className={open === id ? 'active' : ''}
            onClick={() => setOpen(id)}
          >
            {label}
          </button>
        ))}
      </div>

      {open === 'prompt' && (
        <pre className="ops-raw-block" tabIndex={0}>
          {analysis.prompt_sent ?? 'The offline provider does not issue a prompt.'}
        </pre>
      )}

      {open === 'reasoning' && (
        reasoning.length > 0 ? (
          <div className="ops-raw-reasoning">
            {reasoning.map((entry, index) => (
              <pre key={index} className="ops-raw-block" tabIndex={0}>{entry}</pre>
            ))}
          </div>
        ) : (
          <div className="ops-solver-caveat">
            <Icon name="info" size={17} />
            <span>
              {analysis.thinking_reported ? (
                <>
                  <b>The model reported reasoning but returned none.</b>{' '}
                  {analysis.model} marks a thought segment in its response and
                  {analysis.thinking_token_count
                    ? ` spent ${analysis.thinking_token_count} tokens on it,`
                    : ''}{' '}
                  but the segment body is empty: the provider signals that it
                  deliberated without exposing the deliberation. No reasoning text
                  is shown here because none exists — it is not summarised,
                  reconstructed, or invented.
                </>
              ) : (
                <>
                  <b>This provider exposes no reasoning channel.</b> The response
                  contained no thought segment, so there is nothing to display.
                </>
              )}
            </span>
          </div>
        )
      )}

      {open === 'response' && (
        <pre className="ops-raw-block" tabIndex={0}>
          {analysis.raw_response_text ?? 'No raw response was captured for this provider.'}
        </pre>
      )}

      <div className="ops-human-gate">
        <Icon name="lock" size={15} />
        None of this is validated. Only the schema-checked fields drive any decision.
      </div>
    </section>
  );
}

// The track requires a documented naive baseline and a stated metric it is beaten
// on. This renders the measured comparison, including the place where terrain
// weighting alone turned out to change nothing.
function BaselinePanel({ report, onLoad, busy, error }) {
  if (!report) {
    return (
      <section className="ops-panel ops-baseline" aria-labelledby="baseline-title">
        <div className="ops-panel-heading">
          <div>
            <span className="ops-eyebrow">Documented naive baseline</span>
            <h2 id="baseline-title">Shortest-path-only comparison</h2>
            <p>
              The same engine with terrain weighting and closure filtering
              switched off, run over identical inputs.
            </p>
          </div>
          <button className="ops-button pipeline" type="button" onClick={onLoad} disabled={busy}>
            <Icon name={busy ? 'progress_activity' : 'compare_arrows'} size={17} />
            {busy ? 'Running both planners…' : 'Run comparison'}
          </button>
        </div>
        {error && <div className="ops-empty-state"><Icon name="warning" size={22} /><b>{error}</b></div>}
      </section>
    );
  }

  const naive = report.after_closure?.naive ?? {};
  const ours = report.after_closure?.rakshyanet ?? {};
  const openNaive = report.undisrupted?.naive ?? {};
  const openOurs = report.undisrupted?.rakshyanet ?? {};

  return (
    <section className="ops-panel ops-baseline" aria-labelledby="baseline-title">
      <div className="ops-panel-heading">
        <div>
          <span className="ops-eyebrow">Documented naive baseline</span>
          <h2 id="baseline-title">{report.baseline_definition?.name}</h2>
          <p>{report.headline?.statement}</p>
        </div>
        <button className="ops-button ghost" type="button" onClick={onLoad} disabled={busy}>
          <Icon name={busy ? 'progress_activity' : 'refresh'} size={17} /> Recompute
        </button>
      </div>

      <div className="ops-baseline-headline">
        <div className="bad">
          <span>Naive planner</span>
          <b>{report.headline?.naive}</b>
          <small>executable routes after closure</small>
        </div>
        <i aria-hidden="true">vs</i>
        <div className="good">
          <span>RakshyaNet</span>
          <b>{report.headline?.rakshyanet}</b>
          <small>executable routes after closure</small>
        </div>
      </div>

      <div className="ops-table-wrap">
        <table className="ops-baseline-table">
          <caption>
            Closure of {report.scenario?.closure_edge_id}, identical inputs to both planners
          </caption>
          <thead>
            <tr>
              <th scope="col">Measure</th>
              <th scope="col">Naive</th>
              <th scope="col">RakshyaNet</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row">Routes through the closed corridor</th>
              <td className="bad">{naive.routes_traversing_closed_corridor}</td>
              <td className="good">{ours.routes_traversing_closed_corridor}</td>
            </tr>
            <tr>
              <th scope="row">Executable routes</th>
              <td className="bad">{naive.executable_routes} / {naive.routes}</td>
              <td className="good">{ours.executable_routes} / {ours.routes}</td>
            </tr>
            <tr>
              <th scope="row">Fleet distance</th>
              <td>{number(naive.total_distance_km, 0)} km</td>
              <td>{number(ours.total_distance_km, 0)} km</td>
            </tr>
            <tr>
              <th scope="row">Fleet time</th>
              <td>{number(naive.total_time_minutes, 0)} min</td>
              <td>{number(ours.total_time_minutes, 0)} min</td>
            </tr>
            <tr>
              <th scope="row">Distance before any closure</th>
              <td>{number(openNaive.total_distance_km, 0)} km</td>
              <td>{number(openOurs.total_distance_km, 0)} km</td>
            </tr>
          </tbody>
        </table>
      </div>

      {report.baseline_definition?.measured_limitation && (
        <div className="ops-solver-caveat">
          <Icon name="info" size={17} />
          <span>
            <b>Measured limitation.</b>{' '}
            {report.baseline_definition.measured_limitation}
          </span>
        </div>
      )}

      <div className="ops-baseline-definition">
        <span className="ops-eyebrow">What the baseline shares with production</span>
        <ul>
          {(report.baseline_definition?.shared_with_production ?? []).map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
        <span className="ops-eyebrow">What was removed</span>
        <ul>
          {(report.baseline_definition?.removed_from_production ?? []).map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}

// There are two ways to close a road and both must stay: the scenario deck
// replays a scripted timeline, the diagnostics lab closes a corridor on the plan
// you are already looking at. What was missing was the answer to "re-planned from
// what?" — so this states the basis before the operator commits to it: the
// mission time, where each asset actually is, what it has already delivered, and
// which corridors are removed from the graph.
function ReplanBasis({ routes, elapsedMinutes, blockedEdgeIds = [], roadNetwork = [], dispatchActive }) {
  const positions = fleetPositionsAt(routes, elapsedMinutes);
  const moved = Object.keys(positions);
  const served = [...new Set(
    Object.values(positions).flatMap((item) => item.served_stops ?? []),
  )];
  const groundAssets = (routes ?? []).filter((route) => route.transport_mode !== 'air');

  return (
    <div className="ops-replan-basis">
      <div className="ops-replan-head">
        <Icon name="schedule" size={17} />
        <span>
          <b>A re-plan now would start from this state</b>
          <small>
            {dispatchActive
              ? 'The mission clock is live, so re-planning uses current asset positions rather than the depot.'
              : 'No plan is authorized yet, so every asset is still at the depot and re-planning starts from there.'}
          </small>
        </span>
      </div>
      <dl>
        <div>
          <dt>Mission time</dt>
          <dd>T+{formatClock(elapsedMinutes)}</dd>
        </div>
        <div>
          <dt>Assets re-planned from their current position</dt>
          <dd>{moved.length} of {groundAssets.length} ground {groundAssets.length === 1 ? 'asset' : 'assets'}</dd>
        </div>
        <div>
          <dt>Stops already served, excluded from re-planning</dt>
          <dd>{served.length ? served.join(', ') : 'none yet'}</dd>
        </div>
        <div>
          <dt>Corridors removed from the graph</dt>
          <dd>
            {blockedEdgeIds.length
              ? blockedEdgeIds
                  .map((id) => roadNetwork.find((edge) => edge.edge_id === id)?.name ?? id)
                  .join(', ')
              : 'none'}
          </dd>
        </div>
      </dl>
      <small className="ops-replan-note">
        Aircraft are unaffected by road closures and are always re-planned from
        their depot origin.
      </small>
    </div>
  );
}

// "On what data is this based?" is the first question a reviewer asks, and until
// now the answer was a bare list of ids like `report-municipality-002`. These two
// pieces turn every cited id into the actual report text behind it.
function EvidenceCite({ evidenceIds = [], onCite, emptyLabel = 'no cited source' }) {
  if (!evidenceIds.length) {
    return <span className="ops-cite-empty">{emptyLabel}</span>;
  }
  return (
    <span className="ops-cite-row">
      {evidenceIds.map((id) => (
        <button
          key={id}
          type="button"
          className="ops-cite"
          onClick={() => onCite?.(evidenceIds, id)}
          title={`Open the source report ${id}`}
        >
          <Icon name="description" size={12} /> {id}
        </button>
      ))}
    </span>
  );
}

// An imagery citation is the one source in this system with a picture behind it,
// and the picture is the least trustworthy part of it: a real Sentinel-2 patch
// that is not imagery of the named corridor, classified by a model that cannot
// see water depth. The tile is shown because a judge should see what the model
// saw — directly above the record's own caveats, never instead of them.
function ImageryEvidencePanel({ record, available = false, availabilityNotice = '' }) {
  const [tileHidden, setTileHidden] = useState(false);
  const tileId = imageryTileId(record);
  const tier = IMAGERY_TIERS[record?.provider] ?? null;
  const readout = imageryReadout(record?.text);
  const unavailable = record?.provider === 'imagery_check_unavailable';
  // No tile is fetched for an unavailable check: there is nothing behind it, and
  // a request that is known to fail is not worth a broken frame.
  const showTile = available && Boolean(tileId) && !tileHidden && !unavailable;

  return (
    <div className="ops-imagery">
      {availabilityNotice && (
        <p className="ops-imagery-unavailable" role="status">
          {availabilityNotice}
        </p>
      )}
      {showTile && (
        <figure className="ops-imagery-tile">
          <img
            src={imageryTileUrl(tileId)}
            alt={`Satellite tile ${tileId} used for this land-cover classification`}
            loading="lazy"
            onError={() => setTileHidden(true)}
          />
          <figcaption>
            <code>{tileId}</code> · EuroSAT Sentinel-2 RGB patch bound to this
            corridor for demonstration. Not imagery of the named corridor.
          </figcaption>
        </figure>
      )}
      <dl className="ops-imagery-facts">
        {readout.label && (
          <div>
            <dt>Classified</dt>
            <dd>
              {readout.label}
              {readout.confidence ? ` · ${readout.confidence}% confidence` : ''}
            </dd>
          </div>
        )}
        {readout.reference && (
          <div><dt>Reference</dt><dd>{readout.reference}</dd></div>
        )}
        {readout.modelId && (
          <div><dt>Model</dt><dd><code>{readout.modelId}</code></dd></div>
        )}
        {readout.device && (
          <div><dt>Device</dt><dd>{readout.device}</dd></div>
        )}
      </dl>
      {tier && <p className="ops-imagery-note">{tier.note}</p>}
    </div>
  );
}

function SourceReportDialog({
  request,
  analysis,
  imageryAvailable = false,
  imageryAvailabilityNotice = '',
  onClose,
}) {
  const open = Boolean(request);
  const dialogRef = useDialogFocus(open, onClose);
  if (!open) return null;

  const records = Array.isArray(analysis?.evidence) ? analysis.evidence : [];
  const evidenceIds = Array.isArray(request?.evidenceIds) ? request.evidenceIds : [];
  const cited = evidenceIds
    .map((id) => records.find((item) => item.evidence_id === id) ?? { evidence_id: id, missing: true });
  const focusId = request.focusId ?? cited[0]?.evidence_id;

  return (
    <div className="ops-overlay centered" role="presentation" onMouseDown={onClose}>
      <section
        className="ops-source-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="source-report-title"
        ref={dialogRef}
        tabIndex="-1"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="ops-drawer-heading">
          <div>
            <span className="ops-eyebrow">Provenance · what this value is based on</span>
            <h2 id="source-report-title">{request.title}</h2>
          </div>
          <button className="ops-icon-button" type="button" onClick={onClose} aria-label="Close source report">
            <Icon name="close" />
          </button>
        </div>

        <div className="ops-source-value">
          <span>{request.valueLabel ?? 'Extracted value'}</span>
          <b>{request.valueText ?? '—'}</b>
          {request.note && <small>{request.note}</small>}
        </div>

        <div className="ops-source-list">
          {cited.map((record) => {
            const imagery = isImageryRecord(record) && !record.missing;
            const tier = imagery ? IMAGERY_TIERS[record?.provider] : null;
            return (
            <article
              key={record.evidence_id}
              className={record.evidence_id === focusId ? 'focused' : ''}
            >
              <header>
                <div>
                  <b>{record.source_name ?? record.source_identifier ?? record.evidence_id}</b>
                  <small>{record.source_category ?? 'Uncategorised source'} · {record.evidence_id}</small>
                </div>
                <span className="ops-source-tags">
                  {tier && <em className={`tier-${tier.tone}`}>{tier.label}</em>}
                  <em className={record.simulated === false ? 'submitted' : 'simulated'}>
                    {record.simulated === false ? 'operator submitted' : 'simulated fixture'}
                  </em>
                </span>
              </header>
              {record.missing ? (
                <p className="ops-source-missing">
                  This id was cited by the model but is not present in the analysis
                  it was attached to. Nothing downstream consumed it.
                </p>
              ) : (
                <>
                  {imagery && (
                    <ImageryEvidencePanel
                      record={record}
                      available={imageryAvailable}
                      availabilityNotice={imageryAvailabilityNotice}
                    />
                  )}
                  <blockquote>{record.text}</blockquote>
                  <dl>
                    <div><dt>Reliability</dt><dd>{percent(record.reliability)}</dd></div>
                    <div><dt>Freshness</dt><dd>{number(record.freshness_minutes)} min old</dd></div>
                    <div><dt>Identifier</dt><dd><code>{record.source_identifier ?? '—'}</code></dd></div>
                    <div><dt>Retrieved</dt><dd>{formatDecisionTime(record.retrieved_at)}</dd></div>
                  </dl>
                  {record.operator_context && (
                    <p className="ops-source-context">
                      <b>Operator context:</b> {record.operator_context}
                    </p>
                  )}
                  {(record.reported_latitude != null || record.reported_longitude != null) && (
                    <p className="ops-source-context">
                      <b>Reported location:</b> {number(record.reported_latitude, 4)}, {number(record.reported_longitude, 4)}
                    </p>
                  )}
                </>
              )}
            </article>
            );
          })}
          {!cited.length && (
            <p className="ops-empty">
              No source was cited for this value. Nothing was inferred from it — an
              uncited field is reported as UNKNOWN rather than estimated.
            </p>
          )}
        </div>

        <footer className="ops-source-footer">
          <Icon name="science" size={16} />
          <span>
            {analysis?.fixture_notice
              ?? 'Evidence provenance unavailable for this analysis.'}
            {' '}Extraction by {gemmaModelLabel(analysis)} under prompt{' '}
            <code>{analysis?.prompt_version ?? 'unknown'}</code>, analysis{' '}
            <code>{analysis?.analysis_id ?? 'unknown'}</code>.
          </span>
        </footer>
      </section>
    </div>
  );
}

function ScoreBar({ label, field, tone = 'cyan', onSupplyEvidence, onCite }) {
  const value = field?.value ?? field?.expected;
  const bounded = value == null ? null : Math.max(0, Math.min(1, Number(value)));
  return (
    <div className="ops-score">
      <div>
        <span>{label}</span>
        <b className={bounded == null ? 'unknown' : tone}>
          {bounded == null ? 'UNKNOWN' : `${Math.round(bounded * 100)} / 100`}
        </b>
      </div>
      <div className={`ops-score-track ${bounded == null ? 'unknown' : ''}`}>
        <span className={tone} style={{ transform: `scaleX(${bounded ?? 0})` }} />
      </div>
      <small>
        confidence {percent(field?.confidence)} ·{' '}
        <EvidenceCite
          evidenceIds={field?.evidence_ids}
          onCite={(ids, focusId) => onCite?.({
            title: `${label} — cited sources`,
            valueLabel: label,
            valueText: bounded == null ? 'UNKNOWN' : `${Math.round(bounded * 100)} / 100`,
            note: `Model confidence ${percent(field?.confidence)}. Open each report to read the text the value was drawn from.`,
            evidenceIds: ids,
            focusId,
          })}
          emptyLabel="no cited evidence"
        />
      </small>
      {bounded == null && onSupplyEvidence && (
        <button type="button" className="ops-score-evidence" onClick={onSupplyEvidence}>
          <Icon name="add_link" size={14} /> Supply evidence
        </button>
      )}
    </div>
  );
}

// Kept temporarily as a migration reference while the new task-focused workbench is validated.
// eslint-disable-next-line no-unused-vars
function GemmaWorkbench({ analysis, onRun, loading, onSupplyGap }) {
  const output = analysis?.output;
  const evidence = analysis?.evidence ?? [];
  const gaps = evidenceGaps(analysis).filter((gap) => !gap.supplied);
  return (
    <section className="ops-panel ops-gemma" id="gemma-signals" aria-labelledby="gemma-title">
      <div className="ops-panel-heading">
        <div>
          <span className="ops-eyebrow">Gemma → math boundary</span>
          <h2 id="gemma-title">Gemma evidence analysis</h2>
        </div>
        <span className={`ops-provider ${analysis?.provider === 'gemini_api' ? 'hosted' : 'fallback'}`}>
          <StatusDot tone={analysis?.provider === 'gemini_api' ? 'nominal' : 'attention'} />
          {analysis?.provider === 'gemini_api' ? 'Hosted' : 'Fallback'}
        </span>
      </div>

      <div className="ops-gemma-body">
        <div className="ops-gemma-summary">
          <span className="ops-eyebrow">Validated model output</span>
          <p>{output?.summary ?? 'Run Gemma against the current evidence to generate a bounded, cited extraction.'}</p>
          <div className="ops-confidence-row">
            <span>Model <b>{percent(analysis?.model_confidence)}</b></span>
            <span>System <b>{percent(analysis?.system_confidence)}</b></span>
            <span>Review <b>{output?.needs_human_review ? 'required' : 'recommended'}</b></span>
          </div>
        </div>
        <div className="ops-score-stack">
          <ScoreBar label="Incident severity" field={output?.severity} tone="amber" />
          <ScoreBar label="Medical urgency" field={output?.medical_urgency} tone="critical" />
          <ScoreBar label="Accessibility risk" field={output?.accessibility_risk} tone="amber" />
        </div>
      </div>

      <section className={`ops-gap-portal ${gaps.length ? 'has-gaps' : 'complete'}`} aria-labelledby="evidence-gap-title">
        <div className="ops-gap-heading">
          <div>
            <span className="ops-eyebrow">Evidence gap portal</span>
            <b id="evidence-gap-title">
              {gaps.length ? `${gaps.length} unresolved evidence need${gaps.length === 1 ? '' : 's'}` : 'Required fields supported'}
            </b>
          </div>
          <span><Icon name={gaps.length ? 'warning' : 'verified'} size={17} /> {gaps.length ? 'operator input needed' : 'no open gaps'}</span>
        </div>
        {gaps.length > 0 && (
          <div className="ops-gap-list">
            {gaps.slice(0, 5).map((gap) => (
              <div key={gap.id}>
                <Icon name={gap.tone === 'critical' ? 'error' : 'info'} size={17} />
                <span><b>{gap.label}</b><small>{gap.detail}</small></span>
                <button type="button" onClick={() => onSupplyGap(gap)}>Supply evidence</button>
              </div>
            ))}
          </div>
        )}
        {(output?.follow_up_questions ?? []).length > 0 && (
          <div className="ops-follow-up-questions">
            <span className="ops-eyebrow">Gemma asks before it will claim certainty</span>
            {(output.follow_up_questions ?? []).map((question) => (
              <div key={question}>
                <Icon name="help_outline" size={16} />
                <span>{question}</span>
              </div>
            ))}
          </div>
        )}
      </section>

      <div className="ops-evidence-ledger">
        <div className="ops-ledger-heading">
          <div><span className="ops-eyebrow">Evidence ledger</span><b>{evidence.length} records consulted</b></div>
          <button className="ops-text-button" type="button" onClick={onRun} disabled={loading}>
            <Icon name={loading ? 'progress_activity' : 'refresh'} size={16} />
            {loading ? 'Gemma running…' : 'Re-run Gemma'}
          </button>
        </div>
        <div className="ops-evidence-list">
          {evidence.slice(0, 4).map((item) => (
            <div key={item.evidence_id}>
              <Icon name={item.simulated ? 'science' : 'description'} size={17} />
              <span><b>{item.source_name ?? item.source_identifier ?? item.evidence_id}</b><small>{item.evidence_id} · reliability {percent(item.reliability)}</small></span>
              <em>{item.simulated ? 'simulated' : 'submitted'}</em>
            </div>
          ))}
          {!evidence.length && <p className="ops-empty">No evidence loaded.</p>}
        </div>
      </div>

      <div className="ops-tool-boundary">
        <span><Icon name="verified_user" size={15} /> Schema validated</span>
        {(analysis?.requested_tools ?? []).map((tool) => (
          <span key={tool}><Icon name="build" size={14} /> {tool.replaceAll('_', ' ')}</span>
        ))}
      </div>
    </section>
  );
}

function GemmaWorkbenchV2({
  analysis,
  signal,
  onRun,
  loading,
  onSupplyGap,
  onDisposition,
  dispositionBusy,
  onCite,
}) {
  const [activeTab, setActiveTab] = useState('summary');
  const [editingDisposition, setEditingDisposition] = useState(null);
  const [dispositionOwner, setDispositionOwner] = useState('');
  const [dispositionReason, setDispositionReason] = useState('');
  const [dispositionError, setDispositionError] = useState('');
  const [dispositionErrorField, setDispositionErrorField] = useState('');
  const output = analysis?.output;
  const evidence = analysis?.evidence ?? [];
  const dispositions = Object.fromEntries(
    (analysis?.question_dispositions ?? []).map((item) => [item.question_id, item]),
  );
  const gaps = evidenceGaps(analysis).filter((gap) => !gap.supplied);
  const fallbackInputs = [
    ['severity', 'Incident severity', output?.severity?.expected, output?.severity?.confidence, output?.severity?.evidence_ids],
    ['medical_urgency', 'Medical urgency', output?.medical_urgency?.value, output?.medical_urgency?.confidence, output?.medical_urgency?.evidence_ids],
    ['accessibility_risk', 'Accessibility risk', output?.accessibility_risk?.value, output?.accessibility_risk?.confidence, output?.accessibility_risk?.evidence_ids],
  ].map(([field, label, value, confidence, evidenceIds]) => ({
    field,
    label,
    value,
    confidence,
    evidence_ids: evidenceIds ?? [],
    selected_for_max: value != null && Number(value) === Number(signal?.signal),
    status: value == null ? 'unknown' : 'supported',
  }));
  const inputs = signal?.input_scores?.length ? signal.input_scores : fallbackInputs;
  const tabs = [
    ['summary', 'What Gemma found'],
    ['handoff', 'Values sent to math'],
    ['gaps', `Evidence needs (${gaps.length})`],
    ['ledger', `Sources (${evidence.length})`],
  ];

  useEffect(() => {
    setEditingDisposition(null);
    setDispositionOwner('');
    setDispositionReason('');
    setDispositionError('');
    setDispositionErrorField('');
  }, [analysis?.analysis_id]);

  const beginDisposition = (questionId, status) => {
    setEditingDisposition({ questionId, status });
    setDispositionOwner(status === 'assigned' ? 'Field coordination desk' : 'Mission control');
    setDispositionReason('');
    setDispositionError('');
    setDispositionErrorField('');
  };

  const saveDisposition = async (event) => {
    event.preventDefault();
    if (!editingDisposition || dispositionBusy) return;
    const owner = dispositionOwner.trim();
    const reason = dispositionReason.trim();
    if (owner.length < 2) {
      setDispositionError('Enter the team or person responsible for this evidence need.');
      setDispositionErrorField('owner');
      event.currentTarget.elements.disposition_owner?.focus();
      return;
    }
    if (reason.length < 8) {
      setDispositionError(
        editingDisposition.status === 'assigned'
          ? 'Add a short collection plan: what the owner will verify, where, or by when (at least 8 characters).'
          : 'Explain why this answer cannot be obtained right now (at least 8 characters).',
      );
      setDispositionErrorField('reason');
      event.currentTarget.elements.disposition_reason?.focus();
      return;
    }
    setDispositionError('');
    setDispositionErrorField('');
    const saved = await onDisposition(
      editingDisposition.questionId,
      {
        status: editingDisposition.status,
        owner,
        reason,
      },
    );
    if (saved) {
      setEditingDisposition(null);
      setDispositionOwner('');
      setDispositionReason('');
      setDispositionError('');
      setDispositionErrorField('');
    } else {
      setDispositionError('The assignment was not recorded. Review the system alert, then try again.');
      setDispositionErrorField('form');
    }
  };

  return (
    <section className="ops-panel ops-gemma ops-gemma-v2" id="gemma-signals" aria-labelledby="gemma-title-v2">
      <div className="ops-panel-heading">
        <div>
          <span className="ops-eyebrow">Gemma evidence boundary</span>
          <h2 id="gemma-title-v2">Grounded report analysis</h2>
          <p>Inspect extraction, uncertainty, lineage, and the exact bounded values handed to math.</p>
        </div>
        <div className="ops-heading-actions">
          <span className={`ops-provider ${gemmaRuntimeState(analysis)}`}>
            <StatusDot tone={gemmaRuntimeState(analysis) === GEMMA_HOSTED ? 'nominal' : 'attention'} />
            {GEMMA_STATE_LABEL[gemmaRuntimeState(analysis)]}
          </span>
          <button className="ops-text-button" type="button" onClick={onRun} disabled={loading}>
            <Icon name={loading ? 'progress_activity' : 'refresh'} size={16} />
            {loading ? 'Recomputing…' : 'Analyze & rebuild plan'}
          </button>
        </div>
      </div>

      <div className="ops-subnav" role="tablist" aria-label="Gemma inspection mode">
        {tabs.map(([id, label]) => (
          <button
            key={id}
            id={`gemma-tab-${id}`}
            type="button"
            role="tab"
            aria-selected={activeTab === id}
            aria-controls={`gemma-panel-${id}`}
            className={activeTab === id ? 'active' : ''}
            onClick={() => setActiveTab(id)}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="ops-tab-panel" id={`gemma-panel-${activeTab}`} role="tabpanel" aria-labelledby={`gemma-tab-${activeTab}`}>
        {activeTab === 'summary' && (
          <div className="ops-gemma-summary-view">
            <div className="ops-gemma-overview">
              <div className="ops-gemma-summary">
                <span className="ops-eyebrow">Validated model output</span>
                <p>{output?.summary ?? 'No Gemma analysis exists for this evidence set yet.'}</p>
                <div className="ops-confidence-row">
                  <span>Model confidence <b>{percent(analysis?.model_confidence)}</b><small>response quality</small></span>
                  <span>System confidence <b>{percent(analysis?.system_confidence)}</b><small>grounding quality</small></span>
                  <span>Human review <b>{output ? (output.needs_human_review ? 'required' : 'recommended') : 'blocked'}</b><small>Gemma never authorizes</small></span>
                </div>
              </div>
              {/* Classification and the population range were validated by the
                  backend and then rendered nowhere, so the two facts a reviewer
                  asks for first — what happened, and to how many — were invisible.
                  The range is shown as a range because the source stated bounds;
                  collapsing it to the midpoint would present arithmetic as a
                  reported figure. */}
              <div className="ops-fact-stack">
                <article>
                  <span>Incident classification</span>
                  <b>{output?.incident_type?.value ?? 'UNKNOWN'}</b>
                  <small>
                    confidence {percent(output?.incident_type?.confidence)} ·{' '}
                    <EvidenceCite
                      evidenceIds={output?.incident_type?.evidence_ids}
                      onCite={(ids, focusId) => onCite?.({
                        title: 'Incident classification — cited sources',
                        valueLabel: 'Incident type',
                        valueText: output?.incident_type?.value ?? 'UNKNOWN',
                        note: 'The classification must appear as a substring of a cited report; it cannot be invented.',
                        evidenceIds: ids,
                        focusId,
                      })}
                    />
                  </small>
                </article>
                <article>
                  <span>Affected population</span>
                  <b>
                    {output?.affected_population?.expected == null
                      ? 'UNKNOWN'
                      : `${number(output.affected_population.min)}–${number(output.affected_population.max)}`}
                  </b>
                  <small>
                    {output?.affected_population?.expected == null
                      ? 'No source stated a count; nothing was estimated.'
                      : `midpoint ${number(output.affected_population.expected)} · confidence ${percent(output.affected_population.confidence)}`}
                    {' · '}
                    <EvidenceCite
                      evidenceIds={output?.affected_population?.evidence_ids}
                      onCite={(ids, focusId) => onCite?.({
                        title: 'Affected population — cited sources',
                        valueLabel: 'Affected population',
                        valueText: output?.affected_population?.expected == null
                          ? 'UNKNOWN'
                          : `${number(output.affected_population.min)}–${number(output.affected_population.max)} (midpoint ${number(output.affected_population.expected)})`,
                        note: 'Each bound must appear literally in a cited report. The midpoint is arithmetic on those bounds, not a claim any source made.',
                        evidenceIds: ids,
                        focusId,
                      })}
                    />
                  </small>
                </article>
              </div>
              <div className="ops-score-stack">
                <ScoreBar
                  label="Incident severity"
                  field={output?.severity}
                  tone="amber"
                  onCite={onCite}
                  onSupplyEvidence={() => onSupplyGap({
                    id: 'severity',
                    label: 'Incident severity',
                    detail: 'Provide a source that explicitly supports current incident severity.',
                    field: 'severity',
                    tone: 'critical',
                  })}
                />
                <ScoreBar
                  label="Medical urgency"
                  field={output?.medical_urgency}
                  tone="critical"
                  onCite={onCite}
                  onSupplyEvidence={() => onSupplyGap({
                    id: 'medical-urgency',
                    label: 'Medical urgency',
                    detail: 'Provide a field or medical report that supports current injury and care urgency.',
                    field: 'medical_urgency',
                    tone: 'critical',
                  })}
                />
                <ScoreBar
                  label="Accessibility risk"
                  field={output?.accessibility_risk}
                  tone="amber"
                  onCite={onCite}
                  onSupplyEvidence={() => onSupplyGap({
                    id: 'accessibility-risk',
                    label: 'Accessibility risk',
                    detail: 'Provide a current source describing road, air, or foot access constraints.',
                    field: 'accessibility_risk',
                    tone: 'attention',
                  })}
                />
              </div>
            </div>
            <div className="ops-evidence-posture">
              <article><Icon name="folder_open" size={17} /><span><b>Source posture</b><small>{evidence.filter((item) => !item.simulated).length} submitted · {evidence.filter((item) => item.simulated).length} simulated</small></span></article>
              <article><Icon name={(output?.contradictions ?? []).length ? 'warning' : 'verified'} size={17} /><span><b>Contradictions</b><small>{(output?.contradictions ?? []).length || 'None recorded'} in this extraction</small></span></article>
              <article><Icon name="help_outline" size={17} /><span><b>Evidence ownership</b><small>{analysis?.question_dispositions?.length ?? 0}/{output?.follow_up_questions?.length ?? 0} follow-up questions assigned or marked unavailable</small></span></article>
              <article><Icon name="build" size={17} /><span><b>Requested retrieval</b><small>{analysis?.requested_tools?.length ?? 0} allowlisted tool request{analysis?.requested_tools?.length === 1 ? '' : 's'}</small></span></article>
            </div>
          </div>
        )}

        {activeTab === 'handoff' && (
          <div className="ops-handoff">
            <div className="ops-handoff-formula" aria-label="Gemma to math formula">
              <span>Highest supported field</span>
              <b>{number(signal?.calculation?.maximum_supported_score ?? signal?.signal, 4)}</b>
              <i>×</i>
              <span>System confidence</span>
              <b>{number(signal?.system_confidence ?? analysis?.system_confidence, 4)}</b>
              <i>=</i>
              <span>Bounded urgency delta</span>
              <b className="result">+{number(signal?.calculation?.resulting_boost ?? signal?.boost, 4)}</b>
            </div>
            <div className="ops-handoff-grid">
              {inputs.map((input) => (
                <article className={input.selected_for_max ? 'selected' : ''} key={input.field}>
                  <header><span>{input.label}</span><em>{input.selected_for_max ? `${input.status} · selected maximum` : input.status}</em></header>
                  <b>{input.value == null ? 'UNKNOWN' : number(input.value, 4)}</b>
                  <small>Field confidence {percent(input.confidence)}</small>
                  <small>
                    <EvidenceCite
                      evidenceIds={input.evidence_ids}
                      emptyLabel="No supporting evidence IDs"
                      onCite={(ids, focusId) => onCite?.({
                        title: `${input.label} — the reports behind the handed-off value`,
                        valueLabel: `${input.label} sent to the engine`,
                        valueText: input.value == null ? 'UNKNOWN (excluded from the handoff)' : number(input.value, 4),
                        note: 'This is the value the deterministic engine received. Unknown fields are excluded from max(), never treated as zero.',
                        evidenceIds: ids,
                        focusId,
                      })}
                    />
                  </small>
                </article>
              ))}
            </div>
            <div className="ops-authority-boundary">
              <Icon name="shield_lock" size={18} />
              <p>Gemma affects ranking only through this bounded delta. It does not allocate stock, choose vehicles, approve routes, or dispatch assets. Unknown fields are excluded—not guessed.</p>
            </div>
          </div>
        )}

        {activeTab === 'gaps' && (
          <section className={`ops-gap-portal ${gaps.length ? 'has-gaps' : 'complete'}`} aria-labelledby="evidence-gap-title-v2">
            <div className="ops-gap-heading">
              <div><span className="ops-eyebrow">Operator evidence queue</span><b id="evidence-gap-title-v2">{gaps.length ? `${gaps.length} unresolved evidence need${gaps.length === 1 ? '' : 's'}` : 'Required fields supported'}</b></div>
              <span><Icon name={gaps.length ? 'warning' : 'verified'} size={17} /> {gaps.length ? 'input required' : 'no open gaps'}</span>
            </div>
            <div className="ops-gap-list">
              {gaps.map((gap) => (
                <div key={gap.id}>
                  <Icon name={gap.tone === 'critical' ? 'error' : 'info'} size={17} />
                  <span><b>{gap.label}</b><small>{gap.detail}</small></span>
                  <button type="button" onClick={() => onSupplyGap(gap)}>Answer with evidence</button>
                </div>
              ))}
              {!gaps.length && <p className="ops-empty">No missing required fields or contradictions were reported.</p>}
            </div>
            {(output?.follow_up_questions ?? []).length > 0 && (
              <div className="ops-follow-up-questions">
                <span className="ops-eyebrow">Questions Gemma will not guess</span>
                {(output.follow_up_questions ?? []).map((question, index) => {
                  const id = `question-${index}`;
                  const disposition = dispositions[id];
                  const gap = { id, label: 'Gemma follow-up', detail: question, field: 'follow_up_question', tone: 'attention' };
                  return (
                    <div key={id}>
                      <Icon name="help_outline" size={16} />
                      <span>
                        <b>{question}</b>
                        <small>
                          {disposition
                            ? `${disposition.status.toUpperCase()} · ${disposition.owner} · ${disposition.reason}`
                            : 'Unresolved—no answer is inferred.'}
                        </small>
                      </span>
                      <div>
                        <button
                          type="button"
                          onClick={() => onSupplyGap(gap)}
                          aria-label={`Answer with evidence: ${question}`}
                        >
                          Answer with evidence
                        </button>
                        <button type="button" onClick={() => beginDisposition(id, 'assigned')}>Assign</button>
                        <button type="button" onClick={() => beginDisposition(id, 'unavailable')}>Unavailable</button>
                      </div>
                      {editingDisposition?.questionId === id && (
                        <form
                          className="ops-disposition-form"
                          onSubmit={saveDisposition}
                          aria-label={editingDisposition.status === 'assigned' ? 'Assign evidence question' : 'Mark evidence unavailable'}
                          aria-busy={dispositionBusy}
                          noValidate
                        >
                          <div className="ops-disposition-intro">
                            <Icon name={editingDisposition.status === 'assigned' ? 'fact_check' : 'block'} size={17} />
                            <span>
                              <b>{editingDisposition.status === 'assigned' ? 'Assign evidence collection' : 'Record evidence as unavailable'}</b>
                              <small>
                                {editingDisposition.status === 'assigned'
                                  ? 'Name who owns the question and what they will do to answer it. This records accountability; it does not invent an answer.'
                                  : 'Name who confirmed the limitation and why the answer cannot be obtained.'}
                              </small>
                            </span>
                          </div>
                          <label>
                            {editingDisposition.status === 'assigned' ? 'Assigned team or person' : 'Recorded by'}
                            <input
                              name="disposition_owner"
                              value={dispositionOwner}
                              onChange={(event) => {
                                setDispositionOwner(event.target.value);
                                if (dispositionError) {
                                  setDispositionError('');
                                  setDispositionErrorField('');
                                }
                              }}
                              minLength="2"
                              autoComplete="off"
                              aria-describedby={`${id}-owner-help${dispositionErrorField === 'owner' ? ` ${id}-owner-error` : ''}`}
                              aria-invalid={dispositionErrorField === 'owner'}
                              required
                            />
                            <small id={`${id}-owner-help`}>The accountable owner shown in the review ledger.</small>
                            {dispositionErrorField === 'owner' && (
                              <small className="ops-field-error" id={`${id}-owner-error`} role="alert">{dispositionError}</small>
                            )}
                          </label>
                          <label>
                            {editingDisposition.status === 'assigned' ? 'Collection plan' : 'Why unavailable'}
                            <input
                              name="disposition_reason"
                              value={dispositionReason}
                              onChange={(event) => {
                                setDispositionReason(event.target.value);
                                if (dispositionError) {
                                  setDispositionError('');
                                  setDispositionErrorField('');
                                }
                              }}
                              onBlur={() => {
                                if (dispositionReason.trim() && dispositionReason.trim().length < 8) {
                                  setDispositionError('Add at least 8 characters so the operational record is actionable.');
                                  setDispositionErrorField('reason');
                                }
                              }}
                              minLength="8"
                              autoComplete="off"
                              aria-describedby={`${id}-reason-help${dispositionErrorField === 'reason' ? ` ${id}-reason-error` : ''}`}
                              aria-invalid={dispositionErrorField === 'reason'}
                              placeholder={editingDisposition.status === 'assigned'
                                ? 'Example: Verify bridge status with the ward desk by 16:00…'
                                : 'Example: Communications are unavailable until morning…'}
                              required
                            />
                            <small id={`${id}-reason-help`}>
                              {editingDisposition.status === 'assigned'
                                ? 'State what will be checked, where, or by when.'
                                : 'State the constraint and when it may change.'}
                            </small>
                            {dispositionErrorField === 'reason' && (
                              <small className="ops-field-error" id={`${id}-reason-error`} role="alert">{dispositionError}</small>
                            )}
                          </label>
                          <p
                            className={`ops-disposition-feedback ${dispositionErrorField === 'form' ? 'error' : ''}`}
                            id={`${id}-disposition-error`}
                            role={dispositionErrorField === 'form' ? 'alert' : 'status'}
                          >
                            {dispositionErrorField === 'form'
                              ? dispositionError
                              : 'Both fields are required. The action remains available and will explain anything missing.'}
                          </p>
                          <div className="ops-disposition-actions">
                            <button
                              type="button"
                              onClick={() => {
                                setEditingDisposition(null);
                                setDispositionError('');
                                setDispositionErrorField('');
                              }}
                            >
                              Cancel
                            </button>
                            <button type="submit" disabled={dispositionBusy}>
                              <Icon name={dispositionBusy ? 'progress_activity' : 'check'} size={15} />
                              {dispositionBusy
                                ? 'Recording…'
                                : editingDisposition.status === 'assigned'
                                  ? 'Record assignment'
                                  : 'Record unavailable status'}
                            </button>
                          </div>
                        </form>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        )}

        {activeTab === 'ledger' && (
          <div className="ops-evidence-ledger">
            <div className="ops-ledger-heading">
              <div><span className="ops-eyebrow">Evidence ledger</span><b>{evidence.length} records consulted</b></div>
              <span>{evidence.filter((item) => !item.simulated).length} submitted · {evidence.filter((item) => item.simulated).length} simulated</span>
            </div>
            <div className="ops-evidence-list">
              {evidence.map((item) => (
                <div key={item.evidence_id}>
                  <Icon name={item.simulated ? 'science' : 'description'} size={17} />
                  <span><b>{item.source_name ?? item.source_identifier ?? item.evidence_id}</b><small>{item.evidence_id} · reliability {percent(item.reliability)} · freshness {number(item.freshness_minutes)} min</small></span>
                  <button
                    type="button"
                    className="ops-text-button"
                    onClick={() => onCite?.({
                      title: item.source_name ?? item.evidence_id,
                      valueLabel: 'Source record',
                      valueText: item.evidence_id,
                      note: 'The full text below is exactly what Gemma received for this record.',
                      evidenceIds: [item.evidence_id],
                      focusId: item.evidence_id,
                    })}
                  >
                    Read report
                  </button>
                  <em>{item.simulated ? 'simulated' : 'submitted'}</em>
                </div>
              ))}
              {!evidence.length && <p className="ops-empty">No evidence loaded. Add a report to begin grounded analysis.</p>}
            </div>
            <div className="ops-tool-boundary">
              <span><Icon name="verified_user" size={15} /> Schema validated</span>
              {(analysis?.requested_tools ?? []).map((tool) => <span key={tool}><Icon name="build" size={14} /> {tool.replaceAll('_', ' ')}</span>)}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

function TraceCard({ analysis, run, messages, onOpen }) {
  const events = analysis?.trace_steps ?? [];
  const dispositions = analysis?.question_dispositions ?? [];
  return (
    <section className="ops-panel ops-trace-card" id="decision-trace" aria-labelledby="trace-title">
      <div className="ops-panel-heading">
        <div>
          <span className="ops-eyebrow">Bounded decision trace</span>
          <h2 id="trace-title">How the run was assembled</h2>
        </div>
        <span className="ops-count-badge">{events.length}</span>
      </div>
      <div className="ops-trace-preview">
        {events.slice(0, 5).map((step, index) => (
          <div key={step.step_id}>
            <span>{String(index + 1).padStart(2, '0')}</span>
            <div><b>{step.title}</b><small>{step.output_summary}</small></div>
            <em>{number(step.duration_ms, 1)} ms</em>
          </div>
        ))}
      </div>
      <div className="ops-run-integrity">
        <span className="ops-eyebrow">Snapshot integrity</span>
        <div><span>Run</span><b>{run?.run_id ?? 'not available'}</b></div>
        <div><span>Gemma analysis</span><b>{run?.analysis_id ?? 'not linked'}</b></div>
        <div><span>Evidence ownership</span><b>{dispositions.length} follow-up disposition{dispositions.length === 1 ? '' : 's'} recorded</b></div>
        <div><span>Decision state</span><b>{run?.status?.replaceAll('_', ' ') ?? 'pending'}</b></div>
      </div>
      <div className="ops-trace-footer">
        <span><StatusDot /> {messages?.length ?? 0} live events · hidden reasoning withheld</span>
        <button className="ops-button ghost" type="button" onClick={onOpen}>
          <Icon name="account_tree" size={17} /> Inspect complete decision trace
        </button>
      </div>
    </section>
  );
}

// Kept temporarily as a migration reference while the new task-focused workbench is validated.
// eslint-disable-next-line no-unused-vars
function MathEngine({ run, onOpenDiagnostics }) {
  const result = run?.result;
  const allocation = result?.nash_equilibrium;
  const validation = result?.kkt_verification;
  const urgency = result?.urgency_scores ?? [];
  const routes = result?.vrp_solution?.routes ?? [];
  const strategies = allocation?.strategies ?? [];
  const signal = result?.gemma_signal;
  const allocated = strategies.reduce(
    (sum, item) =>
      sum +
      Object.values(item.allocated_resources ?? {}).reduce(
        (subtotal, value) => subtotal + Number(value || 0),
        0,
      ),
    0,
  );
  const needed = strategies.reduce(
    (sum, item) =>
      sum +
      Object.values(item.demanded_resources ?? {}).reduce(
        (subtotal, value) => subtotal + Number(value || 0),
        0,
      ),
    0,
  );

  return (
    <section className="ops-panel ops-math" id="math-engine" aria-labelledby="math-title">
      <div className="ops-panel-heading">
        <div>
          <span className="ops-eyebrow">Deterministic optimization</span>
          <h2 id="math-title">Proposed plan, with the math visible</h2>
          <p>Backend values only. KKT remains a diagnostic, not proof of global optimality.</p>
        </div>
        <button
          type="button"
          className="ops-button ghost"
          onClick={() => onOpenDiagnostics('overview')}
          disabled={!result}
          aria-describedby={!result ? 'diagnostics-readiness' : undefined}
        >
          <Icon name="open_in_full" size={17} /> Open full diagnostics
        </button>
      </div>
      {!result && (
        <p className="ops-diagnostics-readiness" id="diagnostics-readiness" role="status">
          Diagnostics unlock when the first versioned optimization run completes.
        </p>
      )}

      <div className="ops-math-pipeline">
        <div><span>01</span><b>Gemma observed</b><small>{signal?.source_evidence_ids?.length ?? 0} cited records</small></div>
        <div><span>02</span><b>Urgency updated</b><small>{signal?.applied ? `+${number(signal.boost, 3)} confidence-weighted boost` : 'no boost applied'}</small></div>
        <div><span>03</span><b>Routes solved</b><small>{routes.length} assigned vehicle routes</small></div>
        <div><span>04</span><b>Human decides</b><small>{run?.status?.replaceAll('_', ' ') ?? 'review required'}</small></div>
      </div>

      <div className="ops-math-metrics">
        <div><span>Urgency model</span><b>Need + criticality</b><small>{urgency.length} ranked villages</small></div>
        <div><span>Route model</span><b>Terrain-aware VRP</b><small>{number(result?.vrp_solution?.total_distance_km, 0)} km evaluated</small></div>
        <div><span>Allocation</span><b>Urgency weighted</b><small>{allocation?.iterations ?? '—'} iterations</small></div>
        <div><span>Coverage</span><b>{needed ? `${Math.round((allocated / needed) * 100)}%` : '—'}</b><small>proposed demand coverage</small></div>
      </div>

      <div className="ops-math-body">
        <div className="ops-table-wrap">
          <div className="ops-table-title"><b>Decision matrix</b><span>Exact run snapshot</span></div>
          <div className="ops-table">
            <table>
              <thead>
                <tr><th>Village</th><th>Urgency</th><th>Coverage</th><th>Vehicle</th><th>ETA</th></tr>
              </thead>
              <tbody>
                {urgency.slice(0, 8).map((score) => {
                  const route = routes.find((item) =>
                    (item.stops ?? []).includes(score.village_id),
                  );
                  const strategy = strategies.find(
                    (item) => item.village_id === score.village_id,
                  );
                  const allocationTotal = Object.values(
                    strategy?.allocated_resources ?? {},
                  ).reduce((sum, value) => sum + Number(value || 0), 0);
                  const demandTotal = Object.values(
                    strategy?.demanded_resources ?? {},
                  ).reduce((sum, value) => sum + Number(value || 0), 0);
                  return (
                    <tr key={score.village_id}>
                      <th scope="row">{score.village_id}</th>
                      <td><strong className={score.has_critical_shortage ? 'critical' : 'amber'}>{number(score.total_urgency, 3)}</strong></td>
                      <td>{demandTotal ? `${Math.round((allocationTotal / demandTotal) * 100)}%` : '—'}</td>
                      <td>{route?.vehicle_id ?? 'Unassigned'}</td>
                      <td>{route ? `${number(route.total_time_minutes)} min` : '—'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        <div className="ops-proof">
          <div className="ops-table-title"><b>Constraint checks</b><span>Auditable output</span></div>
          {(validation?.conditions ?? []).map((condition) => (
            <div className="ops-proof-row" key={condition.condition_name}>
              <Icon name={condition.satisfied ? 'check_circle' : 'error'} size={17} />
              <span><b>{condition.condition_name}</b><small>{condition.description}</small></span>
              <strong>{condition.satisfied ? 'PASS' : 'FAIL'}</strong>
            </div>
          ))}
          <div className="ops-signal-proof">
            <span>Attached Gemma signal</span>
            <b>{signal?.applied ? `+${number(signal.boost, 3)}` : '0.000'}</b>
            <small>{signal?.matched_villages?.join(', ') || 'No matched village'}</small>
          </div>
        </div>
      </div>
    </section>
  );
}

function ConvergenceInspector({ allocation, welfare }) {
  const history = allocation?.convergence_history ?? [];
  const threshold = Number(allocation?.convergence_threshold ?? 0.01);
  const width = 760;
  const height = 246;
  const padding = { top: 28, right: 70, bottom: 48, left: 72 };
  // This axis is a dimensionless normalized residual. `max_strategy_change` is
  // in each resource's own unit — litres, kits, sheets — which MATH.md §5.1
  // forbids comparing across resource types, and the trailing `?? 0` additionally
  // rendered a *missing* measurement as a converged one. A point without a
  // normalized residual is not plotted and is reported as unavailable.
  const residual = (point) => {
    const value = point?.max_normalized_change;
    return Number.isFinite(Number(value)) ? Number(value) : null;
  };
  const plottable = history
    .map((point, index) => ({ index, value: residual(point) }))
    .filter((item) => item.value !== null);
  const unplottableCount = history.length - plottable.length;
  const values = plottable.map((item) => Math.max(item.value, threshold / 10));
  const maxValue = Math.max(...values, threshold * 10, 1);
  const minValue = Math.min(...values, threshold / 10);
  const logMax = Math.log10(maxValue);
  const logMin = Math.log10(minValue);
  const x = (index) => (
    padding.left
    + (history.length <= 1 ? 0 : index * ((width - padding.left - padding.right) / (history.length - 1)))
  );
  const y = (value) => {
    const logValue = Math.log10(Math.max(Number(value || 0), threshold / 10));
    const ratio = logMax === logMin ? 0.5 : (logMax - logValue) / (logMax - logMin);
    return padding.top + ratio * (height - padding.top - padding.bottom);
  };
  const path = plottable
    .map((item, order) => `${order ? 'L' : 'M'} ${x(item.index)} ${y(item.value)}`)
    .join(' ');
  const finalResidual = Number(
    allocation?.normalized_epsilon_convergence
    ?? allocation?.epsilon_convergence
    ?? plottable.at(-1)?.value,
  );
  const hasDisplayFloor = plottable.some((item) => item.value <= 0);

  if (!history.length || !plottable.length) {
    return (
      <div className="ops-empty-state">
        <Icon name="activity" size={22} />
        <b>No normalized fixed-point history was recorded</b>
        <p>The interface will not fabricate a convergence curve.</p>
      </div>
    );
  }

  return (
    <div className="ops-convergence">
      <div className="ops-chart-title">
        <div>
          <span className="ops-eyebrow">Recorded backend telemetry</span>
          <h3>Capped proportional allocation · fixed-point stabilization</h3>
          <p>Dimensionless maximum change normalized by village demand, depot stock, or 1 for each resource. This proves numerical stability only—not a strategic Nash equilibrium or global optimum.</p>
        </div>
        <span className={allocation?.converged ? 'pass' : 'fail'}>
          {allocation?.converged ? 'Converged' : 'Not converged'} · residual {number(finalResidual, 4)} {allocation?.converged ? '≤' : '>'} tolerance {number(threshold, 4)}
        </span>
      </div>
      <div className="ops-chart-wrap">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby="convergence-title convergence-desc">
          <title id="convergence-title">Allocation fixed-point convergence</title>
          <desc id="convergence-desc">{history.length} recorded iterations. Final normalized allocation residual is {number(finalResidual, 4)} against a {number(threshold, 4)} tolerance.</desc>
          <line className="axis" x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} />
          <line className="axis" x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} />
          <line className="threshold" x1={padding.left} y1={y(threshold)} x2={width - padding.right} y2={y(threshold)} />
          <text className="threshold-label" x={width - padding.right + 8} y={y(threshold) + 4}>ε {number(threshold, 2)}</text>
          <text className="axis-label" x={(padding.left + width - padding.right) / 2} y={height - 12}>Iteration</text>
          <text className="axis-label vertical" transform={`translate(18 ${(padding.top + height - padding.bottom) / 2}) rotate(-90)`}>Normalized residual · log scale</text>
          <path className="residual-line" d={path} />
          {plottable.map((item) => (
            <g key={history[item.index].iteration} className="chart-point">
              <circle cx={x(item.index)} cy={y(item.value)} r="6" />
              <text x={x(item.index)} y={height - padding.bottom + 24} textAnchor="middle">{history[item.index].iteration}</text>
              <text x={x(item.index)} y={Math.max(18, y(item.value) - 12)} textAnchor="middle">{number(item.value, 4)}</text>
            </g>
          ))}
        </svg>
      </div>
      {unplottableCount > 0 && (
        <p className="ops-chart-footnote">
          {unplottableCount} of {history.length} iterations recorded no normalized
          residual and are omitted from the curve rather than plotted as zero.
        </p>
      )}
      {hasDisplayFloor && (
        <p className="ops-chart-footnote">
          Exact zero is rendered at the 0.001 display floor because a logarithmic axis cannot plot zero. The table remains exact.
        </p>
      )}
      <div className="ops-table-wrap ops-convergence-table">
        <table>
          <caption>Exact convergence values and utility</caption>
          <thead><tr><th>Iteration</th><th>Normalized residual</th><th>Largest native-unit change</th><th>Total utility</th><th>Status</th></tr></thead>
          <tbody>
            {history.map((point) => (
              <tr key={point.iteration}>
                <th scope="row">{point.iteration}</th>
                <td>{residual(point) === null ? 'not recorded' : number(residual(point), 6)}</td>
                <td>{number(point.max_strategy_change, 6)} in its resource&apos;s native unit</td>
                <td>{number(point.total_utility, 6)}</td>
                <td>
                  {residual(point) === null
                    ? 'unknown'
                    : residual(point) < threshold ? 'Within tolerance' : 'Continue'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="ops-solver-caveat">
        <Icon name="info" size={17} />
        <span><b>Social-welfare solver history is unavailable.</b> The backend records {welfare?.iterations ?? 'only the final'} SLSQP iterations, objective, and feasibility—not per-iteration points—so no false curve is drawn.</span>
      </div>
    </div>
  );
}

function MathEngineV2({ run, analysis, onOpenDiagnostics, onSupplyGap, onCite }) {
  const [activeTab, setActiveTab] = useState('plan');
  const result = run?.result;
  const allocation = result?.nash_equilibrium;
  const welfare = result?.social_welfare_allocation;
  const comparison = result?.allocation_comparison;
  const validation = result?.kkt_verification;
  const urgency = result?.urgency_scores ?? [];
  const routes = result?.vrp_solution?.routes ?? [];
  const routeAllocations = result?.vrp_solution?.allocations ?? [];
  const signal = result?.gemma_signal;
  const resourceTypes = result?.resource_snapshot?.resource_types ?? {};
  const allRoutesFeasible = routes.length > 0 && routes.every((route) => route.feasible !== false);
  const residual = allocation?.normalized_epsilon_convergence ?? allocation?.epsilon_convergence;
  const gaps = evidenceGaps(analysis).filter((gap) => !gap.supplied);
  const unassignedCritical = urgency.filter(
    (score) => score.has_critical_shortage && !routes.some((route) =>
      isFeasibleRoute(route) && (route.stops ?? []).includes(score.village_id)),
  );
  const failedChecks = (validation?.conditions ?? []).filter((condition) => !condition.satisfied);
  const tabs = [
    ['plan', 'Decision plan'],
    ['inputs', 'Gemma inputs'],
    ['convergence', 'Convergence'],
    ['comparison', 'Allocation comparison'],
    ['validation', 'Validation & improvements'],
  ];

  return (
    <section className="ops-panel ops-math ops-math-v2" id="math-engine" aria-labelledby="math-title-v2">
      <div className="ops-panel-heading">
        <div>
          <span className="ops-eyebrow">Deterministic optimization workbench</span>
          <h2 id="math-title-v2">Plan mathematics and solver evidence</h2>
          <p>Every value below comes from the current versioned run. Candidate scopes are kept separate.</p>
        </div>
        <button className="ops-button ghost" type="button" onClick={() => onOpenDiagnostics('overview')} disabled={!result}>
          <Icon name="open_in_full" size={17} /> Full diagnostics
        </button>
      </div>

      <div className="ops-plan-strip">
        <div><span>Approvable plan</span><b>{allRoutesFeasible ? 'Route-feasible VRP snapshot' : 'VRP snapshot requires route repair'}</b><small>{routes.length} routes · {routeAllocations.length} allocation records</small></div>
        <div><span>Priority model</span><b>Need + survival + bounded AI delta</b><small>Asset fit then scores ETA, payload, impact, specialty & fuel</small></div>
        <div><span>Allocator stability</span><b>{allocation?.converged ? 'Fixed point reached' : 'Not converged'}</b><small>{allocation?.iterations ?? '—'} recorded iterations · residual {number(residual, 4)} / tolerance {number(allocation?.convergence_threshold, 4)}</small></div>
        <div><span>Continuous comparison</span><b>{welfare?.solver_success ? 'SLSQP feasible' : 'Unavailable'}</b><small>Not route-feasible and not directly approvable</small></div>
      </div>

      <div className="ops-subnav" role="tablist" aria-label="Math engine inspection mode">
        {tabs.map(([id, label]) => (
          <button key={id} id={`math-tab-${id}`} type="button" role="tab" aria-selected={activeTab === id} aria-controls={`math-panel-${id}`} className={activeTab === id ? 'active' : ''} onClick={() => setActiveTab(id)}>
            {label}
          </button>
        ))}
      </div>

      <div className="ops-tab-panel" id={`math-panel-${activeTab}`} role="tabpanel" aria-labelledby={`math-tab-${activeTab}`}>
        {activeTab === 'plan' && (
          <div className="ops-plan-view">
            <div className="ops-plan-explainer">
              <Icon name="route" size={20} />
              <span><b>This is the reviewable coordination plan.</b><small>Routes, vehicles, ETAs and payloads are from the same VRP output. Asset selection weighs response time against payload fit and complete-tour fuel feasibility.</small></span>
            </div>
            {/* The provenance question lands here, on the allocations themselves —
                so the answer has to be one click from the table, and it has to be
                precise about the split: reports move ranking, fixtures set stock. */}
            <div className="ops-plan-provenance">
              <Icon name="fact_check" size={19} />
              <span>
                <b>What this plan is based on</b>
                <small>
                  {(signal?.source_evidence_ids ?? []).length} source report
                  {(signal?.source_evidence_ids ?? []).length === 1 ? '' : 's'} produced a bounded
                  urgency delta of +{number(signal?.calculation?.resulting_boost ?? signal?.boost, 4)},
                  applied to {(signal?.matched_villages ?? []).length
                    ? (signal.matched_villages ?? []).join(', ')
                    : 'no village (no name matched the evidence text)'}.
                  Stock levels, vehicle capacities and the road graph come from the
                  bundled fixture data, not from these reports.
                </small>
              </span>
              <button
                type="button"
                className="ops-button ghost"
                disabled={!(signal?.source_evidence_ids ?? []).length}
                onClick={() => onCite?.({
                  title: 'Every report behind this plan',
                  valueLabel: 'Bounded urgency delta applied to ranking',
                  valueText: `+${number(signal?.calculation?.resulting_boost ?? signal?.boost, 4)}`,
                  note: `max(supported severity, medical urgency, access risk) = ${number(signal?.calculation?.maximum_supported_score ?? signal?.signal, 4)} × system confidence ${number(signal?.system_confidence, 4)}. This delta changes who is served first. It never changes how much stock exists, which vehicles are available, or which roads are passable.`,
                  evidenceIds: signal?.source_evidence_ids ?? [],
                })}
              >
                <Icon name="description" size={16} /> Read the source reports
              </button>
            </div>
            <div className="ops-table-wrap">
              <table>
                <caption>{allRoutesFeasible ? 'Route-feasible vehicle plan' : 'Vehicle plan with route exceptions'}</caption>
                <thead><tr><th>Vehicle</th><th>Stops</th><th>Distance</th><th>Route time</th><th>Payload</th><th>Status</th></tr></thead>
                <tbody>
                  {routes.map((route) => (
                    <tr key={route.vehicle_id}>
                      <th scope="row">{route.vehicle_id}</th>
                      <td>{(route.stops ?? []).join(' → ') || 'No assigned stop'}</td>
                      <td>{number(route.total_distance_km, 1)} km</td>
                      <td>{number(route.total_time_minutes)} min</td>
                      <td>{Object.entries(route.cargo_manifest ?? {}).map(([resource, quantity]) => `${number(quantity)} ${resourceUnit(resource, resourceTypes)} ${resourceLabel(resource, resourceTypes)}`).join(' · ') || 'No payload'}</td>
                      <td>{route.feasible === false ? route.infeasibility_reason ?? 'Infeasible' : 'Feasible'}</td>
                    </tr>
                  ))}
                  {!routes.length && <tr><td colSpan="6">No route-feasible plan has been generated.</td></tr>}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'inputs' && (
          <div className="ops-math-inputs">
            <div className="ops-handoff-formula">
              <span>max(supported severity, medical urgency, access risk)</span>
              <b>{number(signal?.calculation?.maximum_supported_score ?? signal?.signal, 4)}</b><i>×</i>
              <span>system confidence</span><b>{number(signal?.system_confidence, 4)}</b><i>=</i>
              <span>urgency delta</span><b className="result">+{number(signal?.calculation?.resulting_boost ?? signal?.boost, 4)}</b>
            </div>
            <div className="ops-handoff-grid">
              {(signal?.input_scores ?? []).map((input) => (
                <article className={input.selected_for_max ? 'selected' : ''} key={input.field}>
                  <header><span>{input.label}</span><em>{input.selected_for_max ? `${input.status} · selected maximum` : input.status}</em></header>
                  <b>{input.value == null ? 'UNKNOWN' : number(input.value, 4)}</b>
                  <small>Confidence {percent(input.confidence)}</small>
                  <small>
                    <EvidenceCite
                      evidenceIds={input.evidence_ids}
                      emptyLabel="No evidence IDs"
                      onCite={(ids, focusId) => onCite?.({
                        title: `${input.label} — source reports`,
                        valueLabel: `${input.label} consumed by this run`,
                        valueText: input.value == null ? 'UNKNOWN (excluded)' : number(input.value, 4),
                        note: `Run ${run?.run_id} computed its urgency delta from these records.`,
                        evidenceIds: ids,
                        focusId,
                      })}
                    />
                  </small>
                </article>
              ))}
            </div>
            {!signal?.input_scores?.length && <p className="ops-empty">This run predates the explicit Gemma handoff contract. Re-run the pipeline to expose every input.</p>}
            <div className="ops-urgency-list">
              {urgency.slice(0, 8).map((score) => (
                <div key={score.village_id}>
                  <span><b>{score.village_id}</b><small>dimensionless priority index · rank {score.ranking}</small></span>
                  <code>{number(score.base_resource_urgency, 3)} + {number(score.critical_penalty, 3)} + {number(score.external_signal, 3)} = {number(score.total_urgency, 3)}</code>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'convergence' && <ConvergenceInspector allocation={allocation} welfare={welfare} />}

        {activeTab === 'comparison' && (
          <div className="ops-comparison">
            <div className="ops-scope-warning">
              <Icon name="info" size={18} />
              <span><b>Comparison candidate only.</b><small>The social-welfare model respects continuous stock and unmet-need constraints, but excludes vehicle and route feasibility.</small></span>
            </div>
            <div className="ops-comparison-grid">
              <article><span>Capped proportional mean coverage</span><b>{percent(comparison?.proportional_mean_coverage)}</b><small>Normalized per-resource utility; units are never summed.</small></article>
              <article><span>Social-welfare mean coverage</span><b>{percent(comparison?.optimized_mean_coverage)}</b><small>Continuous candidate · not reviewable as a route plan.</small></article>
              <article><span>Minimum coverage trade-off</span><b>{percent(comparison?.optimized_minimum_coverage)}</b><small>vs {percent(comparison?.proportional_minimum_coverage)} proportional.</small></article>
              <article><span>Objective improvement</span><b>{number(comparison?.objective_improvement, 4)}</b><small>Within the continuous comparison scope only.</small></article>
            </div>
            <div className="ops-table-wrap">
              <table>
                <caption>Normalized social-welfare coverage by village</caption>
                <thead><tr><th>Village</th><th>Normalized coverage</th><th>Weight</th><th>Scope</th></tr></thead>
                <tbody>
                  {Object.entries(welfare?.village_coverage ?? {}).map(([villageId, coverage]) => (
                    <tr key={villageId}><th scope="row">{villageId}</th><td>{percent(coverage)}</td><td>{number(welfare?.village_weights?.[villageId], 3)}</td><td>Continuous candidate</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'validation' && (
          <div className="ops-validation-grid">
            <div className="ops-proof">
              <div className="ops-table-title"><b>Validation checks</b><span>Scope and status</span></div>
              {(validation?.conditions ?? []).map((condition) => (
                <div className="ops-proof-row" key={condition.condition_name}>
                  <Icon name={condition.satisfied ? 'check_circle' : 'error'} size={17} />
                  <span><b>{condition.condition_name}</b><small>{condition.description}</small></span>
                  <strong>{condition.satisfied ? 'PASS' : 'FAIL'}</strong>
                </div>
              ))}
            </div>
            <div className="ops-improvement-queue">
              <div className="ops-table-title"><b>What needs improvement</b><span>{gaps.length + unassignedCritical.length + failedChecks.length} open</span></div>
              {failedChecks.map((condition) => <div key={condition.condition_name}><Icon name="error" size={17} /><span><b>Failed validation</b><small>{condition.condition_name}: {condition.description}</small></span></div>)}
              {unassignedCritical.map((score) => <div key={score.village_id}><Icon name="warning" size={17} /><span><b>Critical location unassigned</b><small>{score.village_id} has no route-feasible inbound asset.</small></span></div>)}
              {gaps.map((gap) => <div key={gap.id}><Icon name="info" size={17} /><span><b>{gap.label}</b><small>{gap.detail}</small></span><button type="button" onClick={() => onSupplyGap(gap)}>Add evidence</button></div>)}
              {!gaps.length && !unassignedCritical.length && !failedChecks.length && <p className="ops-empty">No blocking validation, route, or evidence issue is open.</p>}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

function EvidenceDrawer({
  open,
  sources,
  setSources,
  onClose,
  onAnalyze,
  loading,
  gap,
}) {
  const [kind, setKind] = useState('field_report');
  const [label, setLabel] = useState('');
  const [text, setText] = useState('');
  const [note, setNote] = useState('');
  const [reliability, setReliability] = useState('0.75');
  const [reportedPlace, setReportedPlace] = useState('');
  const [casualtySummary, setCasualtySummary] = useState('');
  const [damageLevel, setDamageLevel] = useState('unknown');
  const [roadAccess, setRoadAccess] = useState('unknown');
  const [resourceNeeds, setResourceNeeds] = useState('');
  const [contextAnswers, setContextAnswers] = useState({});
  const [formError, setFormError] = useState('');
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  const intakeConfig = useMemo(
    () => evidenceIntakeConfig(gap),
    [gap],
  );
  const resetDraft = useCallback(() => {
    setKind('field_report');
    setLabel('');
    setText('');
    setNote('');
    setReliability('0.75');
    setReportedPlace('');
    setCasualtySummary('');
    setDamageLevel('unknown');
    setRoadAccess('unknown');
    setResourceNeeds('');
    setContextAnswers({});
    setFormError('');
  }, []);
  const hasUnqueuedDraft = Boolean(
    label.trim()
    || text.trim()
    || note.trim()
    || reportedPlace.trim()
    || casualtySummary.trim()
    || damageLevel !== 'unknown'
    || roadAccess !== 'unknown'
    || resourceNeeds.trim()
    || Object.values(contextAnswers).some((value) => value.trim()),
  );
  const requestClose = useCallback(() => {
    if (hasUnqueuedDraft) {
      setConfirmDiscard(true);
      return;
    }
    onClose();
  }, [hasUnqueuedDraft, onClose]);
  const dialogRef = useDialogFocus(open, requestClose);

  useEffect(() => {
    if (open) {
      setConfirmDiscard(false);
      setFormError('');
    }
  }, [open]);

  if (!open) return null;

  const queueSource = (event) => {
    event.preventDefault();
    if (text.trim().length < 10) {
      setFormError('Add a source-backed answer of at least 10 characters so Gemma has enough evidence to evaluate.');
      event.currentTarget.elements.report_text?.focus();
      return;
    }
    if (!label.trim()) {
      setFormError('Name the person, field desk, bulletin, or publication that supplied this answer.');
      event.currentTarget.elements.source_name?.focus();
      return;
    }
    const contextualFacts = intakeConfig.prompts
      .map((prompt) => {
        const answer = contextAnswers[prompt.id]?.trim();
        return answer ? `${prompt.question} Answer: ${answer}` : '';
      })
      .filter(Boolean);
    const structuredFacts = [
      ...contextualFacts,
      reportedPlace.trim() && `Reported place: ${reportedPlace.trim()}.`,
      casualtySummary.trim() && `Casualty or injury report: ${casualtySummary.trim()}.`,
      damageLevel !== 'unknown' && `Reported damage level: ${damageLevel}.`,
      roadAccess !== 'unknown' && `Reported road access: ${roadAccess.replaceAll('_', ' ')}.`,
      resourceNeeds.trim() && `Reported resource needs: ${resourceNeeds.trim()}.`,
    ].filter(Boolean);
    const evidenceText = [
      text.trim(),
      structuredFacts.length
        ? `Operator-supplied structured facts:\n- ${structuredFacts.join('\n- ')}`
        : '',
    ].filter(Boolean).join('\n\n');
    setSources((current) => [
      ...current,
      {
        id: `source-${Date.now()}`,
        kind,
        label: label.trim(),
        text: evidenceText,
        note: note.trim() || (gap ? `Evidence target: ${gap.label}. Verify: ${gap.detail}` : ''),
        reliability: Number(reliability),
        structuredFacts,
        gapTarget: gap?.label ?? null,
        location: gap?.location ?? null,
      },
    ]);
    resetDraft();
  };

  return (
    <div className="ops-overlay" role="presentation" onMouseDown={requestClose}>
      <aside
        ref={dialogRef}
        className="ops-evidence-drawer"
        role="dialog"
        aria-modal="true"
        aria-label="Add evidence"
        tabIndex="-1"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="ops-drawer-heading">
          <div>
            <span className="ops-eyebrow">Evidence intake</span>
            <h2>{gap ? `Close ${gap.label.toLowerCase()}` : 'Give Gemma a real report'}</h2>
          </div>
          <button className="ops-icon-button" type="button" onClick={requestClose} aria-label="Close evidence intake"><Icon name="close" /></button>
        </div>
        {confirmDiscard && (
          <div className="ops-discard-confirm" role="alert">
            <span><b>Discard the unqueued draft?</b><small>Queued evidence is safe; only the fields currently being edited will be cleared.</small></span>
            <button type="button" onClick={() => setConfirmDiscard(false)}>Keep editing</button>
            <button
              type="button"
              onClick={() => {
                resetDraft();
                setConfirmDiscard(false);
                onClose();
              }}
            >
              Discard draft
            </button>
          </div>
        )}
        <div className="ops-security-notice">
          <Icon name="verified_user" size={17} />
          Report text is untrusted data. Embedded instructions are screened before model use.
        </div>
        {gap && (
          <div className="ops-gap-target">
            <Icon name="radio" size={17} />
            <span><b>Closing: {gap.label}</b><small>{gap.detail}</small></span>
          </div>
        )}
        <form className="ops-source-form" onSubmit={queueSource}>
          <div className="ops-required-intro">
            <span>2 required fields</span>
            <small>{gap ? 'Answer Gemma’s exact question, then name the source. Add only facts the source actually provides.' : 'Start with the report, then name its source. Add structured facts only when provided.'}</small>
          </div>
          {gap && (
            <div className="ops-evidence-question" aria-labelledby="evidence-question-title">
              <span>Question to answer</span>
              <b id="evidence-question-title">{intakeConfig.answerQuestion}</b>
              <small>Do not estimate missing details. “Unknown” is safer than an unsupported value.</small>
            </div>
          )}
          <label>
            {intakeConfig.answerLabel}
            <textarea
              data-dialog-autofocus
              aria-label={gap ? 'Evidence answer' : 'Report text'}
              name="report_text"
              autoComplete="off"
              value={text}
              onChange={(event) => {
                setText(event.target.value);
                if (formError) setFormError('');
              }}
              rows="6"
              minLength="10"
              placeholder={intakeConfig.answerPlaceholder}
              aria-describedby={`evidence-answer-help${formError ? ' evidence-form-error' : ''}`}
              aria-invalid={Boolean(formError && text.trim().length < 10)}
              required
            />
            <small id="evidence-answer-help">{intakeConfig.answerHint}</small>
          </label>
          <label>
            Source name
            <input
              aria-label="Source name"
              name="source_name"
              autoComplete="organization"
              value={label}
              onChange={(event) => {
                setLabel(event.target.value);
                if (formError) setFormError('');
              }}
              placeholder="Example: Taplejung field desk…"
              aria-describedby={`evidence-source-help${formError ? ' evidence-form-error' : ''}`}
              aria-invalid={Boolean(formError && !label.trim())}
              required
            />
            <small id="evidence-source-help">Name the person, desk, bulletin, or publication that supplied the report.</small>
          </label>
          <fieldset className="ops-context-questions">
            <legend>{gap ? 'Supporting questions' : 'Useful details to confirm'} <span>optional</span></legend>
            <p>Answer only what this source knows. Blank answers remain unknown.</p>
            <div>
              {intakeConfig.prompts.map((prompt, index) => (
                <label key={prompt.id}>
                  <span aria-hidden="true">{String(index + 1).padStart(2, '0')}</span>
                  <b>{prompt.question}</b>
                  <input
                    name={`context_${prompt.id}`}
                    autoComplete="off"
                    value={contextAnswers[prompt.id] ?? ''}
                    onChange={(event) => setContextAnswers((current) => ({
                      ...current,
                      [prompt.id]: event.target.value,
                    }))}
                    placeholder={prompt.placeholder}
                  />
                </label>
              ))}
            </div>
          </fieldset>
          {formError && <p className="ops-inline-error" id="evidence-form-error" role="alert">{formError}</p>}

          <details className="ops-evidence-options">
            <summary>
              <span><Icon name="tune" size={16} /> Add optional facts & provenance</span>
              <small>Location, casualties, damage, access, resources, file import, source type, reliability</small>
            </summary>
            <div className="ops-evidence-options-body">
              <div className="ops-optional-grid">
                <label>
                  Reported place <span>optional</span>
                  <input name="reported_place" autoComplete="off" value={reportedPlace} onChange={(event) => setReportedPlace(event.target.value)} placeholder="Example: Taplejung ward 4…" />
                </label>
                <label>
                  Casualties or injuries <span>optional</span>
                  <input name="casualty_summary" autoComplete="off" value={casualtySummary} onChange={(event) => setCasualtySummary(event.target.value)} placeholder="Example: 8 injured, fatalities unverified…" />
                </label>
                <label>
                  Damage level <span>optional</span>
                  <select name="damage_level" value={damageLevel} onChange={(event) => setDamageLevel(event.target.value)}>
                    <option value="unknown">Not provided</option>
                    <option value="minor">Minor</option>
                    <option value="major">Major</option>
                    <option value="severe">Severe</option>
                    <option value="catastrophic">Catastrophic</option>
                  </select>
                </label>
                <label>
                  Road access <span>optional</span>
                  <select name="road_access" value={roadAccess} onChange={(event) => setRoadAccess(event.target.value)}>
                    <option value="unknown">Not provided</option>
                    <option value="open">Open</option>
                    <option value="restricted">Restricted</option>
                    <option value="blocked">Blocked</option>
                    <option value="helicopter_only">Helicopter only</option>
                  </select>
                </label>
              </div>
              <label>
                Requested resources <span>optional</span>
                <input name="resource_needs" autoComplete="off" value={resourceNeeds} onChange={(event) => setResourceNeeds(event.target.value)} placeholder="Example: medical kits, water, tarpaulins…" />
              </label>
              <label>
                Load a plain-text export <span>optional</span>
                <input
                  type="file"
                  name="evidence_file"
                  accept=".txt,.md"
                  onChange={async (event) => {
                    const file = event.target.files?.[0];
                    if (!file) return;
                    setLabel(file.name);
                    setText(await file.text());
                  }}
                />
              </label>
              <fieldset>
                <legend>Source type</legend>
                <div className="ops-segmented">
                  {[
                    ['field_report', 'Field report', 'description'],
                    ['article', 'Article', 'link'],
                    ['official_bulletin', 'Official bulletin', 'policy'],
                  ].map(([value, copy, icon]) => (
                    <button
                      type="button"
                      key={value}
                      className={kind === value ? 'active' : ''}
                      onClick={() => setKind(value)}
                      aria-pressed={kind === value}
                    >
                      <Icon name={icon} size={16} /> {copy}
                    </button>
                  ))}
                </div>
              </fieldset>
              <label>
                Source reliability
                <select name="source_reliability" value={reliability} onChange={(event) => setReliability(event.target.value)}>
                  <option value="0.95">0.95 · authenticated authority</option>
                  <option value="0.85">0.85 · verified field desk</option>
                  <option value="0.75">0.75 · corroborated report</option>
                  <option value="0.55">0.55 · uncorroborated report</option>
                </select>
                <small>This is an explicit operator input; it is never inferred from the source name.</small>
              </label>
              <label>
                Operator context <span>optional</span>
                <textarea name="operator_context" autoComplete="off" value={note} onChange={(event) => setNote(event.target.value)} rows="3" placeholder="What should the reviewer verify?…" />
              </label>
            </div>
          </details>
          <button className="ops-button primary full" type="submit"><Icon name="playlist_add" size={17} /> Queue source</button>
        </form>
        <div className="ops-queued">
          <div><span className="ops-eyebrow">Queued evidence</span><b>{sources.length}</b></div>
          {sources.map((source) => (
            <div className="ops-queued-row" key={source.id}>
              <Icon name="description" size={17} />
              <span>
                <b>{source.label}</b>
                <small>
                  {source.text.length} characters · reliability {source.reliability.toFixed(2)}
                  {source.gapTarget ? ` · closes ${source.gapTarget}` : ''}
                  {source.location ? ` · map ${source.location.latitude.toFixed(4)}, ${source.location.longitude.toFixed(4)}` : ''}
                </small>
              </span>
              <button type="button" onClick={() => setSources((current) => current.filter((item) => item.id !== source.id))} aria-label={`Remove ${source.label}`}><Icon name="close" size={16} /></button>
            </div>
          ))}
          <button className="ops-button approve full" type="button" onClick={onAnalyze} disabled={!sources.length || loading}>
            <Icon name={loading ? 'progress_activity' : 'auto_awesome'} size={17} />
            {loading ? 'Analyzing…' : 'Analyze and recalculate plan'}
          </button>
        </div>
      </aside>
    </div>
  );
}

const DIAGNOSTIC_TABS = [
  ['issues', 'Review issues', 'warning'],
  ['overview', 'Overview', 'calculate'],
  ['inventory', 'Resources', 'boxes'],
  ['urgency', 'Urgency', 'radio'],
  ['routing', 'Routes & closures', 'navigation'],
  ['validation', 'Validation', 'verified'],
  ['audit', 'Audit trace', 'scroll_text'],
];

function DiagnosticsDialog({
  open,
  initialTab,
  run,
  analysis,
  villages,
  onClose,
  onReoptimize,
  onSupplyGap,
  loading,
  elapsedMinutes = 0,
  dispatchActive = false,
  onOpenScenarios,
}) {
  const dialogRef = useDialogFocus(open, onClose);
  const [activeTab, setActiveTab] = useState(initialTab ?? 'overview');
  const [villageId, setVillageId] = useState('');
  const [closureEdgeId, setClosureEdgeId] = useState('');

  useEffect(() => {
    if (open) setActiveTab(initialTab ?? 'overview');
  }, [open, initialTab]);

  const result = run?.result;
  const solution = result?.vrp_solution;
  const resourceSnapshot = result?.resource_snapshot ?? {};
  const resourceTypes = resourceSnapshot.resource_types ?? {};
  const allocations = solution?.allocations ?? [];
  const routes = solution?.routes ?? [];
  const scores = result?.urgency_scores ?? [];
  const roadNetwork = solution?.road_network ?? [];
  const activeBlocks = solution?.active_road_blocks ?? [];
  const checks = result?.kkt_verification?.conditions ?? [];
  const failedChecks = checks.filter((item) => !item.satisfied);
  const fleet = result?.fleet_snapshot ?? [];
  const defaultVillageId = villageId || scores[0]?.village_id || villages[0]?.id || '';
  const selectedVillage = villages.find((item) => item.id === defaultVillageId);
  const selectedAllocation = allocations.find((item) => item.village_id === defaultVillageId);
  const selectedScore = scores.find((item) => item.village_id === defaultVillageId);
  const openRoadEdges = roadNetwork.filter((edge) => edge.status !== 'blocked');
  const proposedByResource = allocations.reduce((totals, allocation) => {
    Object.entries(allocation.allocated_resources ?? {}).forEach(([resourceId, quantity]) => {
      totals[resourceId] = (totals[resourceId] ?? 0) + Number(quantity || 0);
    });
    return totals;
  }, {});
  const allResourceIds = Object.keys(resourceTypes);
  const passedChecks = checks.filter((item) => item.satisfied).length;
  const assignedFleet = fleet.filter((item) => item.status === 'assigned').length;
  const reroutedFleet = routes.filter((route) => route.rerouted_due_to?.length).length;
  const gaps = evidenceGaps(analysis).filter((gap) => !gap.supplied);
  const unassignedCritical = scores.filter(
    (score) => score.has_critical_shortage && !routes.some((route) =>
      isFeasibleRoute(route) && (route.stops ?? []).includes(score.village_id)),
  );

  if (!open) return null;

  const injectClosure = async () => {
    if (!closureEdgeId || loading) return;
    const edge = roadNetwork.find((item) => item.edge_id === closureEdgeId);
    await onReoptimize(
      [...new Set([...activeBlocks, closureEdgeId])],
      `Simulated sudden closure: ${edge?.name ?? closureEdgeId}`,
    );
    setClosureEdgeId('');
  };

  const reopenEdge = async (edgeId) => {
    await onReoptimize(
      activeBlocks.filter((item) => item !== edgeId),
      `Operator reopened corridor: ${edgeId}`,
    );
  };

  return (
    <div className="ops-overlay centered diagnostics" role="presentation" onMouseDown={onClose}>
      <section
        ref={dialogRef}
        className="ops-diagnostics-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Math engine full diagnostics"
        tabIndex="-1"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="ops-diagnostics-hero">
          <div>
            <span className="ops-eyebrow">Decision assurance suite / {run?.run_id ?? 'no run'}</span>
            <h2>Full math and route diagnostics</h2>
            <p>Inventory, allocation reasons, urgency terms, graph routes, disruptions, and validation from one versioned snapshot.</p>
          </div>
          <div className="ops-diagnostics-hero-state">
            <span><StatusDot tone={run?.status === 'awaiting_approval' ? 'attention' : 'nominal'} /> {run?.status?.replaceAll('_', ' ')}</span>
            <small>Decorative relief study · operational values are tabular</small>
          </div>
          <button className="ops-icon-button" type="button" onClick={onClose} aria-label="Close full diagnostics"><Icon name="close" /></button>
        </header>

        <div className="ops-diagnostics-shell">
          <nav className="ops-diagnostics-tabs" aria-label="Diagnostic sections">
            {DIAGNOSTIC_TABS.map(([id, label, icon]) => (
              <button
                type="button"
                key={id}
                className={activeTab === id ? 'active' : ''}
                aria-pressed={activeTab === id}
                onClick={() => setActiveTab(id)}
              >
                <Icon name={icon} size={17} />
                <span>{label}</span>
                {id === 'routing' && activeBlocks.length > 0 && <em>{activeBlocks.length}</em>}
              </button>
            ))}
          </nav>

          <div className="ops-diagnostics-content">
            {activeTab === 'issues' && (
              <section className="ops-diagnostic-section">
                <div className="ops-diagnostic-heading">
                  <div><span className="ops-eyebrow">Human review queue</span><h3>Issues that must be resolved or explicitly owned</h3></div>
                  <span>{gaps.length + failedChecks.length + unassignedCritical.length + activeBlocks.length}</span>
                </div>
                <div className="ops-issue-review-list">
                  {gaps.map((gap) => (
                    <article key={gap.id} className="critical">
                      <Icon name="help_outline" size={19} />
                      <div><b>{gap.label}</b><p>{gap.detail}</p><small>Gemma will keep this field UNKNOWN until evidence supports it.</small></div>
                      <button type="button" className="ops-button ghost" onClick={() => onSupplyGap(gap)}>Supply evidence</button>
                    </article>
                  ))}
                  {failedChecks.map((condition) => (
                    <article key={condition.condition_name}>
                      <Icon name="error" size={19} />
                      <div><b>{condition.condition_name}</b><p>{condition.description}</p><small>Approval remains blocked until the solver check passes.</small></div>
                    </article>
                  ))}
                  {unassignedCritical.map((score) => (
                    <article key={score.village_id}>
                      <Icon name="route" size={19} />
                      <div><b>{score.village_id} has a critical shortage</b><p>No assigned route currently reaches this location.</p><small>Review resource capacity and route feasibility in Resources and Routes & closures.</small></div>
                    </article>
                  ))}
                  {activeBlocks.map((edgeId) => (
                    <article key={edgeId}>
                      <Icon name="block" size={19} />
                      <div><b>Road corridor blocked: {edgeId}</b><p>Ground assets were recomputed around the closure.</p><small>Open Routes & closures to inspect the child run and reopen the corridor.</small></div>
                    </article>
                  ))}
                  {!gaps.length && !failedChecks.length && !unassignedCritical.length && !activeBlocks.length && (
                    <div className="ops-empty-state"><Icon name="verified" size={20} /><b>No unresolved issues in this snapshot.</b></div>
                  )}
                </div>
              </section>
            )}
            {activeTab === 'overview' && (
              <section className="ops-diagnostic-section">
                <div className="ops-data-provenance">
                  <Icon name="science" size={19} />
                  <div>
                    <b>{resourceSnapshot.source_label ?? 'Data provenance unavailable'}</b>
                    <small>
                      Inventory: {resourceSnapshot.source_file ?? 'unknown'} · routing: {solution?.routing_source_label ?? 'unknown'}.
                      These are mocked hackathon fixtures, not live government databases.
                    </small>
                  </div>
                </div>
                <div className="ops-diagnostic-metrics">
                  <div><span>Depot resource types</span><b>{allResourceIds.length}</b><small>explicit units and mass conversion</small></div>
                  <div><span>Fleet disposition</span><b>{assignedFleet}/{fleet.length}</b><small>assigned / available assets</small></div>
                  <div><span>Ground graph</span><b>{roadNetwork.length}</b><small>{activeBlocks.length} blocked corridors</small></div>
                  <div><span>Validation</span><b>{passedChecks}/{checks.length || '—'}</b><small>scoped checks passed</small></div>
                </div>
                <div className="ops-diagnostic-callouts">
                  <article>
                    <Icon name="package_check" size={19} />
                    <div>
                      <b>What was allocated</b>
                      <p>
                        {allocations.length} village allocation records across {Object.keys(proposedByResource).length} resource types.
                        Quantities remain separated in their native units; inspect the ledger for each stock, allocation, and remainder.
                      </p>
                    </div>
                    <button type="button" onClick={() => setActiveTab('inventory')}>Inspect ledger</button>
                  </article>
                  <article>
                    <Icon name="radio" size={19} />
                    <div><b>Why urgency changed</b><p>Every score exposes resource contributions, elapsed-time factor, survival penalty, and the bounded Gemma signal separately.</p></div>
                    <button type="button" onClick={() => setActiveTab('urgency')}>Explain urgency</button>
                  </article>
                  <article>
                    <Icon name="navigation" size={19} />
                    <div><b>What if a road closes</b><p>{reroutedFleet ? `${reroutedFleet} ground routes currently avoid a blocked corridor.` : 'Inject a corridor closure to create a fresh solver run and reset human approval.'}</p></div>
                    <button type="button" onClick={() => setActiveTab('routing')}>Open disruption lab</button>
                  </article>
                </div>
              </section>
            )}

            {activeTab === 'inventory' && (
              <section className="ops-diagnostic-section">
                <div className="ops-diagnostic-heading">
                  <div><span className="ops-eyebrow">Supply → demand → proposal</span><h3>Resource ledger</h3></div>
                  <span className="ops-source-chip"><Icon name="science" size={14} /> mocked backend fixture</span>
                </div>
                <div className="ops-diagnostic-table">
                  <table>
                    <thead><tr><th>Resource</th><th>Unit</th><th>Depot available</th><th>Already in field</th><th>Reported demand</th><th>Proposed now</th><th>Depot after plan</th></tr></thead>
                    <tbody>
                      {allResourceIds.map((resourceId) => {
                        const available = Number(resourceSnapshot.depot_available?.[resourceId] ?? 0);
                        const proposed = Number(proposedByResource[resourceId] ?? 0);
                        return (
                          <tr key={resourceId}>
                            <th scope="row">{resourceLabel(resourceId, resourceTypes)}</th>
                            <td>{resourceUnit(resourceId, resourceTypes)}</td>
                            <td>{number(available, 1)}</td>
                            <td>{number(resourceSnapshot.existing_field_allocations?.[resourceId], 1)}</td>
                            <td>{number(resourceSnapshot.reported_demand?.[resourceId], 1)}</td>
                            <td><strong>{number(proposed, 1)}</strong></td>
                            <td>{number(Math.max(0, available - proposed), 1)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div className="ops-allocation-inspector">
                  <div className="ops-diagnostic-heading compact">
                    <div><span className="ops-eyebrow">Per-incident allocation reason</span><h3>{selectedVillage?.name ?? defaultVillageId}</h3></div>
                    <select name="diagnostic_incident_allocation" value={defaultVillageId} onChange={(event) => setVillageId(event.target.value)} aria-label="Select incident allocation">
                      {scores.map((score) => {
                        const village = villages.find((item) => item.id === score.village_id);
                        return <option key={score.village_id} value={score.village_id}>#{score.ranking} {village?.name ?? score.village_id}</option>;
                      })}
                    </select>
                  </div>
                  <p className="ops-allocation-summary">{selectedAllocation?.allocation_explanation ?? 'No allocation explanation is available for this run.'}</p>
                  <div className="ops-resource-decisions">
                    {(selectedAllocation?.resource_decisions ?? []).map((decision) => (
                      <article key={decision.resource_type}>
                        <header>
                          <span><b>{resourceLabel(decision.resource_type, resourceTypes)}</b><small>{decision.unit}</small></span>
                          <strong>{number(decision.proposed_now, 1)} proposed</strong>
                        </header>
                        <div>
                          <span>Need <b>{number(decision.current_need, 1)}</b></span>
                          <span>Existing <b>{number(decision.existing_allocated, 1)}</b></span>
                          <span>Survival floor <b>{number(decision.survival_threshold, 1)}</b></span>
                          <span>Gap after <b>{number(decision.unmet_after_plan, 1)}</b></span>
                        </div>
                        <p>{decision.explanation}</p>
                        {(decision.asset_selection ?? []).length > 0 && (
                          <div className="ops-asset-selection-list" aria-label={`${decision.resource_type} asset selection rationale`}>
                            {decision.asset_selection.map((asset, index) => (
                              <div key={`${asset.vehicle_id}-${index}`}>
                                <Icon name={asset.transport_mode === 'air' ? 'radio' : 'navigation'} size={15} />
                                <span>
                                  <b>{asset.vehicle_id} · {asset.transport_mode}</b>
                                  <small>
                                    {number(asset.quantity, 1)} {decision.unit} / {number(asset.payload_kg, 1)} kg · direct ETA {number(asset.estimated_one_way_minutes)} min · projected tour {number(asset.projected_route_minutes)} min
                                  </small>
                                </span>
                                <em>{number(asset.selection_score, 2)}</em>
                                <small>
                                  time {number(asset.time_pressure, 2)} · ETA {number(asset.eta_score, 2)} · payload {number(asset.payload_fit_score, 2)} · impact {number(asset.incident_impact, 2)}
                                </small>
                              </div>
                            ))}
                          </div>
                        )}
                        <footer>
                          {(decision.reason_codes ?? []).map((code) => <span key={code}>{code.replaceAll('_', ' ')}</span>)}
                        </footer>
                      </article>
                    ))}
                  </div>
                </div>

                <div className="ops-diagnostic-heading compact"><div><span className="ops-eyebrow">Fleet resource</span><h3>Available and assigned assets</h3></div></div>
                <div className="ops-fleet-ledger">
                  {fleet.map((vehicle) => (
                    <div key={vehicle.vehicle_id}>
                      <Icon name={vehicle.category === 'aircraft' ? 'radio' : 'navigation'} size={17} />
                      <span><b>{vehicle.name}</b><small>{vehicle.terrain_capability.replaceAll('_', ' ')} · {number(vehicle.capacity_kg)} kg capacity</small></span>
                      <em className={vehicle.status}>{vehicle.status}</em>
                      <strong>{vehicle.assigned_route ? `${number(vehicle.assigned_route.total_cargo_kg)} kg loaded` : 'reserve'}</strong>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {activeTab === 'urgency' && (
              <section className="ops-diagnostic-section">
                <div className="ops-diagnostic-heading">
                  <div><span className="ops-eyebrow">No hidden score</span><h3>Urgency derivation</h3></div>
                  <select name="diagnostic_urgency_score" value={defaultVillageId} onChange={(event) => setVillageId(event.target.value)} aria-label="Select urgency score">
                    {scores.map((score) => {
                      const village = villages.find((item) => item.id === score.village_id);
                      return <option key={score.village_id} value={score.village_id}>#{score.ranking} {village?.name ?? score.village_id}</option>;
                    })}
                  </select>
                </div>
                <div className="ops-formula-card">
                  <span>Total urgency</span>
                  <b>{number(selectedScore?.base_resource_urgency, 4)} <i>resource need</i> + {number(selectedScore?.critical_penalty, 2)} <i>survival penalty</i> + {number(selectedScore?.external_signal, 4)} <i>Gemma signal</i> = {number(selectedScore?.total_urgency, 4)}</b>
                  <small>Time factor {number(selectedScore?.time_factor, 4)} at {number(selectedScore?.time_elapsed_hours, 1)} elapsed hours. A +10 penalty is applied once if any resource is below its survival threshold.</small>
                </div>
                <div className="ops-diagnostic-table">
                  <table>
                    <thead><tr><th>Resource term</th><th>Unmet / need</th><th>Multiplier</th><th>Time factor</th><th>Contribution</th><th>Threshold</th></tr></thead>
                    <tbody>
                      {Object.values(selectedScore?.components ?? {}).map((component) => (
                        <tr key={component.resource_type}>
                          <th scope="row">{resourceLabel(component.resource_type, resourceTypes)}</th>
                          <td>{number(component.unmet_need, 1)} / {number(component.current_need, 1)} ({percent(component.unmet_ratio)})</td>
                          <td>{number(component.urgency_multiplier, 2)}×</td>
                          <td>{number(component.time_factor, 3)}×</td>
                          <td><strong>{number(component.contribution, 4)}</strong></td>
                          <td>{component.below_survival_threshold ? 'Below survival floor' : 'Above survival floor'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="ops-gemma-score-boundary">
                  <Icon name="auto_awesome" size={18} />
                  <div>
                    <b>Gemma contribution is bounded and separately visible</b>
                    <p>{result?.gemma_signal?.method ?? 'No Gemma signal method attached.'} The model never writes the base need score or the +10 threshold penalty.</p>
                  </div>
                  <strong>{number(selectedScore?.external_signal, 4)}</strong>
                </div>
              </section>
            )}

            {activeTab === 'routing' && (
              <section className="ops-diagnostic-section">
                <div className="ops-data-provenance road">
                  <Icon name="navigation" size={19} />
                  <div><b>{solution?.routing_source_label ?? 'Routing source unavailable'}</b><small>Illustrative corridor graph for the hackathon. Aircraft use direct geodesic legs; ground assets must traverse compatible open edges.</small></div>
                </div>
                {/* Two entry points exist and both are legitimate. Naming the
                    other one here stops the operator hunting for it, and stops a
                    reviewer assuming the product has one path implemented twice. */}
                <div className="ops-closure-paths">
                  <article className="active">
                    <span className="ops-eyebrow">You are here · path 1 of 2</span>
                    <b>Manual closure on the current plan</b>
                    <small>
                      Keeps the evidence and plan you are looking at, removes the
                      corridor you pick, and re-plans from where the fleet is right
                      now. Use this to answer “what if this road went, today”.
                    </small>
                  </article>
                  <article>
                    <span className="ops-eyebrow">Path 2 of 2</span>
                    <b>Scripted scenario timeline</b>
                    <small>
                      The scenario deck in Operations replays a bundled incident
                      with its own reports and a closure written into the timeline.
                      It starts a fresh situation rather than modifying this one.
                    </small>
                    {onOpenScenarios && (
                      <button type="button" className="ops-text-button" onClick={onOpenScenarios}>
                        <Icon name="science" size={15} /> Go to the scenario deck
                      </button>
                    )}
                  </article>
                </div>
                <ReplanBasis
                  routes={routes}
                  elapsedMinutes={elapsedMinutes}
                  blockedEdgeIds={activeBlocks}
                  roadNetwork={roadNetwork}
                  dispatchActive={dispatchActive}
                />
                <div className="ops-disruption-lab">
                  <div>
                    <span className="ops-eyebrow">Sudden road-event simulation</span>
                    <h3>Close a corridor and force a new plan</h3>
                    <p>The backend removes this edge, recomputes truck paths and ETAs, versions a child run, and resets approval. Helicopter paths remain unchanged.</p>
                  </div>
                  <div className="ops-disruption-control">
                    <label>
                      Corridor to block
                      <select name="road_closure_edge" value={closureEdgeId} onChange={(event) => setClosureEdgeId(event.target.value)}>
                        <option value="">Select an open road edge</option>
                        {openRoadEdges.map((edge) => <option key={edge.edge_id} value={edge.edge_id}>{edge.name} · {number(edge.distance_km)} km</option>)}
                      </select>
                    </label>
                    <button className="ops-button reject" type="button" disabled={!closureEdgeId || loading} onClick={injectClosure}>
                      <Icon name={loading ? 'progress_activity' : 'warning'} size={17} />
                      {loading ? 'Re-optimizing…' : 'Inject closure & re-optimize'}
                    </button>
                  </div>
                </div>
                {activeBlocks.length > 0 && (
                  <div className="ops-active-closures">
                    <span className="ops-eyebrow">Active blocked corridors</span>
                    {activeBlocks.map((edgeId) => {
                      const edge = roadNetwork.find((item) => item.edge_id === edgeId);
                      return (
                        <div key={edgeId}>
                          <Icon name="block" size={17} />
                          <span><b>{edge?.name ?? edgeId}</b><small>{reroutedFleet} routes report closure avoidance in this run</small></span>
                          <button type="button" onClick={() => reopenEdge(edgeId)} disabled={loading}>Reopen & recompute</button>
                        </div>
                      );
                    })}
                  </div>
                )}
                <div className="ops-diagnostic-table">
                  <table>
                    <thead><tr><th>Asset</th><th>Mode</th><th>Stops</th><th>Distance</th><th>Graph edges</th><th>Closure response</th></tr></thead>
                    <tbody>
                      {routes.map((route) => (
                        <tr key={route.vehicle_id}>
                          <th scope="row">{route.vehicle_id}</th>
                          <td>{route.transport_mode}</td>
                          <td>{route.stops.join(' → ')}</td>
                          <td>{number(route.total_distance_km, 1)} km</td>
                          <td>{route.road_edge_ids?.length ?? 0}</td>
                          <td>{route.rerouted_due_to?.length ? `Avoids ${route.rerouted_due_to.join(', ')}` : route.transport_mode === 'air' ? 'Road independent' : 'No detour'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            )}

            {activeTab === 'validation' && (
              <section className="ops-diagnostic-section">
                <div className="ops-diagnostic-heading"><div><span className="ops-eyebrow">Scoped, not overclaimed</span><h3>Constraint and method diagnostics</h3></div><span>{passedChecks}/{checks.length || '—'} passed</span></div>
                <div className="ops-validation-grid">
                  {checks.map((condition) => (
                    <article key={condition.condition_name} className={condition.satisfied ? 'pass' : 'fail'}>
                      <Icon name={condition.satisfied ? 'check_circle' : 'error'} size={19} />
                      <div><b>{condition.condition_name}</b><p>{condition.description}</p></div>
                      <strong>{condition.satisfied ? 'PASS' : 'FAIL'}</strong>
                    </article>
                  ))}
                </div>
                <div className="ops-method-register">
                  <div><span>Allocation</span><b>{run?.allocation_method}</b><small>Compared with {run?.comparison_allocation_method}</small></div>
                  <div><span>Routing</span><b>{run?.routing_engine ?? run?.routing_method}</b><small>graph-constrained for ground, direct for air</small></div>
                  <div><span>Diagnostic scope</span><b>KKT consistency</b><small>{run?.diagnostic_scope}</small></div>
                  <div><span>Runtime</span><b>{number(result?.execution_time_seconds, 3)} sec</b><small>deterministic backend execution</small></div>
                </div>
              </section>
            )}

            {activeTab === 'audit' && (
              <section className="ops-diagnostic-section">
                <div className="ops-diagnostic-heading"><div><span className="ops-eyebrow">Evidence-linked summaries</span><h3>Run audit trace</h3></div><span>{analysis?.trace_steps?.length ?? 0} Gemma steps</span></div>
                <div className="ops-audit-metadata">
                  <span>Run <b>{run?.run_id}</b></span>
                  <span>Parent <b>{run?.parent_run_id ?? 'none'}</b></span>
                  <span>Trigger <b>{run?.trigger ?? 'manual'}</b></span>
                  <span>Analysis <b>{run?.analysis_id}</b></span>
                </div>
                <div className="ops-full-trace diagnostic">
                  {(analysis?.trace_steps ?? []).map((step, index) => (
                    <div key={step.step_id}>
                      <span>{String(index + 1).padStart(2, '0')}</span>
                      <div><b>{step.title}</b><p>{step.output_summary}</p><small>{number(step.duration_ms, 1)} ms · inputs: {step.input_ids.join(', ')}</small></div>
                    </div>
                  ))}
                </div>
                <div className="ops-security-notice">
                  <Icon name="lock" size={17} />
                  This is an auditable execution summary, not hidden chain-of-thought. It includes inputs, outputs, methods, warnings, and version IDs.
                </div>
              </section>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

function TraceDialog({ open, analysis, onClose }) {
  const dialogRef = useDialogFocus(open, onClose);

  if (!open) return null;
  const events = analysis?.trace_steps ?? [];
  const evidence = analysis?.evidence ?? [];
  return (
    <div className="ops-overlay centered" role="presentation" onMouseDown={onClose}>
      <section
        ref={dialogRef}
        className="ops-trace-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Complete decision trace"
        tabIndex="-1"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="ops-drawer-heading">
          <div><span className="ops-eyebrow">Decision trace</span><h2>How this run was assembled</h2></div>
          <button className="ops-icon-button" type="button" onClick={onClose} aria-label="Close decision trace"><Icon name="close" /></button>
        </div>
        <div className="ops-trace-dialog-body">
          <section className="ops-consulted">
            <div><Icon name="folder_open" size={18} /><span><b>Evidence consulted</b><small>{evidence.length} provenance-tagged records</small></span></div>
            <div className="ops-evidence-chips">
              {evidence.map((item) => <span key={item.evidence_id}>{item.evidence_id}</span>)}
            </div>
          </section>
          <div className="ops-full-trace">
            {events.map((step, index) => (
              <div key={step.step_id}>
                <span>{String(index + 1).padStart(2, '0')}</span>
                <div><b>{step.title}</b><p>{step.output_summary}</p><small>{number(step.duration_ms, 1)} ms · inputs: {step.input_ids.join(', ')}</small></div>
              </div>
            ))}
          </div>
          <div className="ops-security-notice">
            <Icon name="lock" size={17} />
            Hidden chain-of-thought is never displayed. This trace contains only evidence-linked, schema-validated summaries.
          </div>
        </div>
      </section>
    </div>
  );
}

function DecisionReviewDialog({ action, run, onClose, onConfirm, busy }) {
  const dialogRef = useDialogFocus(Boolean(action), onClose);
  const [notes, setNotes] = useState('');
  useEffect(() => setNotes(''), [action?.kind, run?.run_id]);
  if (!action) return null;
  const isApproval = action.kind === 'approve';
  const valid = isApproval || notes.trim().length >= 12;
  return (
    <div className="ops-overlay centered ops-review-overlay" role="presentation" onMouseDown={onClose}>
      <section
        ref={dialogRef}
        className="ops-review-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="review-dialog-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="ops-review-dialog-heading">
          <span className="ops-eyebrow">Human authorization checkpoint</span>
          <button className="ops-icon-button" type="button" onClick={onClose} aria-label="Close review confirmation"><Icon name="close" /></button>
        </div>
        <h2 id="review-dialog-title">{isApproval ? 'Confirm this plan review' : 'Request changes to this plan'}</h2>
        <p>
          {isApproval
            ? `You are approving run ${run?.run_id ?? '—'} for the recorded coordination snapshot. This does not dispatch vehicles.`
            : 'This sends the run back for correction and records why the current snapshot cannot be accepted.'}
        </p>
        <div className="ops-review-snapshot">
          <span><b>Pinned run</b><code>{run?.run_id ?? '—'}</code></span>
          <span><b>Gemma analysis</b><code>{run?.analysis_id ?? '—'}</code></span>
          <span><b>Snapshot timestamp</b><code>{run?.updated_at ?? '—'}</code></span>
        </div>
        {!isApproval && (
          <label>
            Reason for requesting changes
            <textarea name="review_change_reason" autoComplete="off" value={notes} onChange={(event) => setNotes(event.target.value)} minLength="12" rows="4" autoFocus placeholder="State the issue, missing evidence, or correction required." />
            <small>Minimum 12 characters · {notes.trim().length}/12 entered</small>
          </label>
        )}
        {isApproval && <div className="ops-review-summary"><Icon name="verified_user" size={19} /><span><b>Human-only gate</b><small>Gemma supplied bounded evidence signals; deterministic math produced the plan; you own this decision.</small></span></div>}
        <div className="ops-review-actions">
          <button className="ops-button ghost" type="button" onClick={onClose} disabled={busy}>Cancel</button>
          <button className={`ops-button ${isApproval ? 'approve' : 'reject'}`} type="button" onClick={() => onConfirm(isApproval, isApproval ? (action.rationale ?? '') : notes.trim())} disabled={!valid || busy}>
            <Icon name={busy ? 'progress_activity' : isApproval ? 'check' : 'undo'} size={17} />
            {busy ? 'Recording…' : isApproval ? 'Confirm approval' : 'Submit request for changes'}
          </button>
        </div>
      </section>
    </div>
  );
}

// A demo accumulates state that has no other exit: a road closure written into
// the plan stays in every subsequent re-plan, queued evidence sits in the
// drawer, the mission clock keeps its position. There was no way back to the
// opening picture short of a page reload — and a reload does not clear a
// closure either, because the closure lives on the server's newest run and
// bootstrap adopts exactly that run.
//
// This is the way back. It is deliberately honest about its scope: it starts a
// NEW run from the stock scenario, it does not delete anything the server has
// already recorded.
function ResetDemoDialog({ open, onClose, onConfirm, busy, closureCount }) {
  const dialogRef = useDialogFocus(open, busy ? () => {} : onClose);
  if (!open) return null;
  return (
    <div
      className="ops-overlay centered ops-review-overlay"
      role="presentation"
      onMouseDown={busy ? undefined : onClose}
    >
      <section
        ref={dialogRef}
        className="ops-review-dialog ops-reset-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="reset-dialog-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="ops-review-dialog-heading">
          <span className="ops-eyebrow">Start over</span>
          <button
            className="ops-icon-button"
            type="button"
            onClick={onClose}
            disabled={busy}
            aria-label="Close reset confirmation"
          >
            <Icon name="close" />
          </button>
        </div>
        <h2 id="reset-dialog-title">Reset the demo to its opening state?</h2>
        <p>
          This discards the operator work in this session and builds a fresh
          baseline plan from the stock national scenario.
        </p>
        <div className="ops-reset-scope">
          <div>
            <span className="ops-eyebrow">This clears</span>
            <ul>
              <li>
                {closureCount > 0
                  ? `${closureCount} road closure${closureCount === 1 ? '' : 's'} written into the current plan`
                  : 'any road closure written into the current plan'}
              </li>
              <li>Evidence queued in the drawer and evidence submitted this session</li>
              <li>The draft incident, map add-mode, and the map selection</li>
              <li>The mission clock — back to T+00:00 and re-locked</li>
              <li>Open dialogs, the baseline comparison, and any error banner</li>
            </ul>
          </div>
          <div>
            <span className="ops-eyebrow">This does not</span>
            <ul>
              <li>
                Delete anything on the server. Every earlier run, analysis and
                approval stays in the audit history — this starts a new run
                alongside them.
              </li>
              <li>
                Erase the recorded Gemma function-calling turn. The agent console
                keeps it and labels which run it came from.
              </li>
            </ul>
          </div>
        </div>
        <div className="ops-review-actions">
          <button className="ops-button ghost" type="button" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button
            className="ops-button primary"
            type="button"
            onClick={onConfirm}
            disabled={busy}
            data-dialog-autofocus
          >
            <Icon name={busy ? 'progress_activity' : 'refresh'} size={17} />
            {busy ? 'Rebuilding a clean plan…' : 'Reset and re-plan'}
          </button>
        </div>
        {busy && (
          <p className="ops-reset-progress" role="status">
            <span className="ops-reset-bar" aria-hidden="true" />
            Re-reading the stock reports with Gemma, then recomputing urgency,
            routes and allocation with no closures. This takes a few seconds.
          </p>
        )}
      </section>
    </div>
  );
}

const WORKSPACE_IDS = ['operations', 'evidence', 'math', 'review'];

function workspaceFromHash() {
  const candidate = window.location.hash.replace(/^#/, '');
  return WORKSPACE_IDS.includes(candidate) ? candidate : 'operations';
}

function timeValue(value) {
  const parsed = Date.parse(value ?? '');
  return Number.isFinite(parsed) ? parsed : 0;
}

function newerRun(current, incoming) {
  if (!incoming) return current;
  if (!current) return incoming;
  if (current.run_id === incoming.run_id) {
    return timeValue(incoming.updated_at) >= timeValue(current.updated_at)
      ? incoming
      : current;
  }
  return timeValue(incoming.created_at) >= timeValue(current.created_at)
    ? incoming
    : current;
}

function newerAnalysis(current, incoming) {
  if (!incoming) return current;
  if (!current) return incoming;
  if (current.analysis_id === incoming.analysis_id) return incoming;
  return timeValue(incoming.created_at) >= timeValue(current.created_at)
    ? incoming
    : current;
}

/* What this deployment cannot do, said on arrival.
 *
 * The system was built to run locally, where a persistent server holds a
 * telemetry socket open and a second process runs the imagery classifier on a
 * GPU. Serverless hosting provides neither. Both absences are already labelled
 * at the point of use, but a visitor meets those points several clicks in and
 * reads an unexplained absence as a broken feature. Naming them once on arrival
 * costs a dismissible banner and removes that reading.
 *
 * Session-scoped rather than permanent: a judge who reloads is still a first
 * time reader, while someone working through the interface is not told twice.
 */
function HostedBuildNotice() {
  const [dismissed, setDismissed] = useState(() => {
    try {
      return sessionStorage.getItem('rakshyanet.hosted-notice') === 'dismissed';
    } catch {
      return false;
    }
  });

  // Only the hosted build is missing anything. Running the same bundle locally
  // against a local backend, both capabilities are present and the notice would
  // be false.
  if (!import.meta.env.PROD || dismissed) return null;

  const dismiss = () => {
    try {
      sessionStorage.setItem('rakshyanet.hosted-notice', 'dismissed');
    } catch {
      /* a browser refusing storage is not a reason to keep the banner up */
    }
    setDismissed(true);
  };

  return (
    <aside className="ops-hosted-notice" role="status">
      <Icon name="info" size={18} />
      <div className="ops-hosted-notice-body">
        <b>You are looking at the hosted build.</b>
        <p>
          RakshyaNet was developed to run locally, where a persistent server keeps a
          live telemetry connection open and a second local process runs the
          overhead-imagery classifier on a GPU. Serverless hosting provides neither,
          so the same code runs here with two capabilities switched off:{' '}
          <b>the satellite / overhead-imagery check</b>, whose controls are disabled
          rather than broken, and <b>the live event stream</b>, which is why the agent
          console shows a recorded turn instead of claiming live frames.
        </p>
        <p>
          Everything else is real: Gemma&rsquo;s grounded extraction with citations and{' '}
          <code>UNKNOWN</code>s, native function calling, the routing engine, the naive
          baseline, and the human approval gate. Where the two builds differ, this
          interface says so on screen rather than simulating the difference away.
        </p>
      </div>
      <button type="button" className="ops-text-button" onClick={dismiss}>
        Dismiss
      </button>
    </aside>
  );
}

export default function PremiumApp() {
  const [run, setRun] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [villagesData, setVillagesData] = useState({ depot: null, villages: [] });
  const [selectedVillageId, setSelectedVillageId] = useState(null);
  const [fleet, setFleet] = useState([]);
  const [scenarios, setScenarios] = useState([]);
  const [declaredFunctions, setDeclaredFunctions] = useState([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState('');
  const [scenarioStage, setScenarioStage] = useState('baseline');
  const [scenarioBusy, setScenarioBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sources, setSources] = useState([]);
  const [showIntake, setShowIntake] = useState(false);
  const [evidenceGap, setEvidenceGap] = useState(null);
  const [traceOpen, setTraceOpen] = useState(false);
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  const [diagnosticsTab, setDiagnosticsTab] = useState('overview');
  const [addMode, setAddMode] = useState(false);
  const [draftIncident, setDraftIncident] = useState(null);
  const [reviewBusy, setReviewBusy] = useState(false);
  const [dispositionBusy, setDispositionBusy] = useState(false);
  const [orchestrateBusy, setOrchestrateBusy] = useState(false);
  // Null is the short, bounded status-checking state. Any failed or malformed
  // response is stored as disabled so controls never drift into a dead action.
  const [imageryBusy, setImageryBusy] = useState(false);
  const [imageryResult, setImageryResult] = useState(null);
  const [imageryStatus, setImageryStatus] = useState(null);
  const [baseline, setBaseline] = useState(null);
  const [baselineBusy, setBaselineBusy] = useState(false);
  const [baselineError, setBaselineError] = useState(null);
  const [reviewDialog, setReviewDialog] = useState(null);
  const [sourceRequest, setSourceRequest] = useState(null);
  // Survives runs that carry no orchestration of their own. See AgentConsole.
  const [lastOrchestration, setLastOrchestration] = useState(null);
  const [activeWorkspace, setActiveWorkspace] = useState(workspaceFromHash);
  const [missionElapsed, setMissionElapsed] = useState(0);
  const [missionPlaying, setMissionPlaying] = useState(false);
  const [missionSpeed, setMissionSpeed] = useState(DEFAULT_MISSION_SPEED);
  const [resetOpen, setResetOpen] = useState(false);
  const [resetBusy, setResetBusy] = useState(false);
  const bootstrapped = useRef(false);
  const focusedGemmaLocation = useRef(false);
  const runRef = useRef(null);
  // Read inside async handlers so a re-plan uses the operator's mission time at
  // the moment they triggered it, not the value captured when the closure formed.
  const missionElapsedRef = useRef(0);
  const analysisRef = useRef(null);
  const failedRetryRef = useRef(null);
  const { messages, isConnected, lastMessage, transport } = useWebSocket();

  const mergeRun = useCallback((incoming) => {
    setRun((current) => {
      const selected = newerRun(current, incoming);
      runRef.current = selected;
      return selected;
    });
  }, []);

  const mergeAnalysis = useCallback((incoming) => {
    setAnalysis((current) => {
      const selected = newerAnalysis(current, incoming);
      analysisRef.current = selected;
      return selected;
    });
  }, []);

  const mergeQuestionDisposition = useCallback((analysisId, disposition) => {
    if (!analysisId || !disposition) return;
    setAnalysis((current) => {
      if (!current || current.analysis_id !== analysisId) return current;
      const next = {
        ...current,
        question_dispositions: [
          ...(current.question_dispositions ?? []).filter(
            (item) => item.question_id !== disposition.question_id,
          ),
          disposition,
        ],
      };
      analysisRef.current = next;
      return next;
    });
  }, []);

  const clearFailure = useCallback(() => {
    failedRetryRef.current = null;
    setError(null);
  }, []);

  const reportFailure = useCallback((requestError, retryLabel, retry) => {
    failedRetryRef.current = retry ? { label: retryLabel, run: retry } : null;
    setError({
      message: requestError?.message ?? String(requestError),
      retryLabel,
    });
  }, []);

  const changeWorkspace = useCallback((workspace) => {
    if (!WORKSPACE_IDS.includes(workspace)) return;
    setActiveWorkspace(workspace);
    window.history.replaceState(null, '', `#${workspace}`);
    window.requestAnimationFrame(() => {
      document.querySelector('[data-workspace-heading]')?.focus?.();
    });
  }, []);

  const openEvidence = useCallback((gap = null) => {
    setEvidenceGap(gap);
    setShowIntake(true);
  }, []);
  const closeEvidence = useCallback(() => {
    setShowIntake(false);
    setEvidenceGap(null);
    setDraftIncident(null);
  }, []);

  useEffect(() => {
    const syncWorkspace = () => setActiveWorkspace(workspaceFromHash());
    window.addEventListener('hashchange', syncWorkspace);
    return () => window.removeEventListener('hashchange', syncWorkspace);
  }, []);

  const loadRun = useCallback(async (forceNew = false) => {
    setLoading(true);
    clearFailure();
    try {
      const history = forceNew ? [] : await api.getOptimizationHistory();
      // History is newest-first. Recover the most recent function-calling record
      // so a page reload does not erase the track's mandatory evidence just
      // because the newest run happened to be operator-started.
      const orchestrated = (history ?? []).find((item) => item.orchestration);
      if (orchestrated) {
        setLastOrchestration({
          record: orchestrated.orchestration,
          runId: orchestrated.run_id,
        });
      }
      const nextRun =
        history?.[0] ??
        (await api.runOptimization({
          scenario_id: 'nepal-national-demo',
          requested_by: 'mission-control',
        }));
      mergeRun(nextRun);
      // Load the analysis this run consumed, not whatever analysis is newest.
      // Those differ constantly — the backend keeps every analysis, and any
      // scenario activation mints a new one. Pairing a run with a stranger's
      // analysis is what made the evidence queue demand medical evidence while
      // the math engine displayed a medical value from a different generation.
      const paired = nextRun?.analysis_id
        ? await api.getGemmaAnalysis(nextRun.analysis_id).catch(() => null)
        : null;
      const attached = paired ?? await api.getLatestGemmaAnalysis().catch(() => null);
      if (attached) mergeAnalysis(attached);
    } catch (requestError) {
      reportFailure(requestError, 'run snapshot', () => loadRun(forceNew));
    } finally {
      setLoading(false);
    }
  }, [clearFailure, mergeAnalysis, mergeRun, reportFailure]);

  const rerunGemma = async () => {
    if (loading) return;
    setLoading(true);
    clearFailure();
    try {
      const nextAnalysis = await api.runGemmaAnalysis('nepal-national-demo');
      const currentAnalysis = analysisRef.current;
      if (
        currentAnalysis
        && currentAnalysis.analysis_id !== nextAnalysis.analysis_id
        && timeValue(currentAnalysis.created_at) > timeValue(nextAnalysis.created_at)
      ) return;
      const nextRun = await api.runOptimization({
        scenario_id: 'nepal-national-demo',
        requested_by: 'mission-control',
        analysis_id: nextAnalysis.analysis_id,
        vehicle_positions: fleetPositionsAt(
          runRef.current?.result?.vrp_solution?.routes,
          missionElapsedRef.current,
        ),
      });
      mergeAnalysis(nextAnalysis);
      mergeRun(nextRun);
    } catch (requestError) {
      reportFailure(requestError, 'Gemma pipeline', rerunGemma);
    } finally {
      setLoading(false);
    }
  };

  // Gemma drives the engine here rather than the operator. The run that comes
  // back carries the function-call record, so the interface can show the exact
  // arguments the model produced next to the plan they produced.
  const orchestrateWithGemma = async () => {
    if (orchestrateBusy || loading) return;
    setOrchestrateBusy(true);
    clearFailure();
    try {
      const nextRun = await api.orchestrateOptimization({
        scenario_id: 'nepal-national-demo',
        requested_by: 'gemma-function-call',
      });
      mergeRun(nextRun);
      const nextAnalysis = await api.getLatestGemmaAnalysis().catch(() => null);
      if (nextAnalysis?.analysis_id === nextRun.analysis_id) {
        mergeAnalysis(nextAnalysis);
      }
    } catch (requestError) {
      reportFailure(requestError, 'Gemma function calling', orchestrateWithGemma);
    } finally {
      setOrchestrateBusy(false);
    }
  };

  // ARCH §4 · the two operator paths to the same evidence record. B1 makes Gemma
  // emit the call itself, so the audit shows a real model-emitted tool call that
  // is nonetheless marked operator-directed. B2 skips the model entirely, for
  // when a forty-second round trip is not available. Both handlers repeat the
  // status guard so a disabled control cannot be bypassed programmatically.
  const askGemmaForImagery = async (target) => {
    if (
      !target?.corridorId
      || !imageryActionsAvailable(imageryStatus)
      || imageryBusy
      || orchestrateBusy
      || loading
    ) return;
    setImageryBusy(true);
    setImageryResult(null);
    clearFailure();
    try {
      const nextRun = await api.orchestrateOptimization({
        scenario_id: runRef.current?.scenario_id ?? 'nepal-national-demo',
        requested_by: 'operator-imagery-directive',
        operator_directive: {
          corridor_id: target.corridorId,
          incident_type: target.incidentType,
          evidence_id: target.evidenceId,
        },
      });
      mergeRun(nextRun);
      const nextAnalysis = await api.getLatestGemmaAnalysis().catch(() => null);
      if (nextAnalysis?.analysis_id === nextRun?.analysis_id) {
        mergeAnalysis(nextAnalysis);
      }
      setImageryResult({
        tone: 'info',
        text: `Directive sent for ${target.corridorId}. The emitted call is recorded in the function-call panel under Gemma evidence.`,
      });
    } catch (requestError) {
      reportFailure(
        requestError,
        'imagery directive',
        () => askGemmaForImagery(target),
      );
    } finally {
      setImageryBusy(false);
    }
  };

  const checkImageryNow = async (target) => {
    if (
      !target?.corridorId
      || !imageryActionsAvailable(imageryStatus)
      || imageryBusy
      || loading
    ) return;
    setImageryBusy(true);
    setImageryResult(null);
    clearFailure();
    try {
      const result = await api.verifyCorridorImagery(
        target.corridorId,
        target.incidentType,
        target.evidenceId,
      );
      // The endpoint returns the whole analysis with the record appended, so the
      // citation is live in the evidence ledger without another fetch.
      if (result?.analysis?.analysis_id) mergeAnalysis(result.analysis);
      const record = result?.record ?? null;
      const tier = IMAGERY_TIERS[record?.provider] ?? null;
      const readout = imageryReadout(record?.text);
      setImageryResult({
        tone: tier?.tone ?? 'info',
        text: [
          tier?.label ?? 'imagery check recorded',
          readout.label
            && `${readout.label}${readout.confidence ? ` ${readout.confidence}%` : ''}`,
          record?.evidence_id && `cited as ${record.evidence_id}`,
        ].filter(Boolean).join(' · '),
      });
    } catch (requestError) {
      reportFailure(
        requestError,
        'imagery check',
        () => checkImageryNow(target),
      );
    } finally {
      setImageryBusy(false);
    }
  };

  const loadBaseline = async () => {
    if (baselineBusy) return;
    setBaselineBusy(true);
    setBaselineError(null);
    try {
      setBaseline(await api.getBaselineComparison());
    } catch (requestError) {
      setBaselineError(String(requestError.message ?? requestError));
    } finally {
      setBaselineBusy(false);
    }
  };

  const activateScenario = async () => {
    if (!selectedScenarioId || scenarioBusy || loading) return;
    setScenarioBusy(true);
    setLoading(true);
    clearFailure();
    try {
      const activated = await api.activateDemoScenario(
        selectedScenarioId,
        scenarioStage,
      );
      focusedGemmaLocation.current = false;
      mergeAnalysis(activated.analysis);
      mergeRun(activated.run);
      const matchedVillage = activated.run?.result?.gemma_signal?.matched_villages?.[0];
      if (
        matchedVillage
        && villagesData.villages.some((item) => item.id === matchedVillage)
      ) {
        setSelectedVillageId(matchedVillage);
        focusedGemmaLocation.current = true;
      }
      setSources([]);
      setEvidenceGap(null);
      setShowIntake(false);
      setDraftIncident(null);
      changeWorkspace('operations');
    } catch (requestError) {
      reportFailure(requestError, 'mock scenario', activateScenario);
    } finally {
      setScenarioBusy(false);
      setLoading(false);
    }
  };

  const analyzeSubmitted = async () => {
    if (!sources.length) return;
    setLoading(true);
    clearFailure();
    try {
      const existingEvidence = (analysis?.evidence ?? []).map((item) => ({
        evidence_id: item.evidence_id,
        source_category: item.source_category,
        source_name: item.source_name,
        source_identifier: item.source_identifier,
        text: item.text,
        reliability: item.reliability,
        freshness_minutes: item.freshness_minutes,
        operator_context: item.operator_context,
        gap_target: item.gap_target,
        reported_latitude: item.reported_latitude,
        reported_longitude: item.reported_longitude,
      }));
      const submittedEvidence = sources.map((source) => ({
          evidence_id: source.id,
          source_category: source.kind,
          source_name: source.label,
          source_identifier: source.location
            ? `map://${source.location.latitude.toFixed(6)},${source.location.longitude.toFixed(6)}`
            : source.label,
          text: source.text,
          reliability: source.reliability,
          freshness_minutes: 0,
          operator_context: source.note || null,
          gap_target: source.gapTarget,
          reported_latitude: source.location?.latitude ?? null,
        reported_longitude: source.location?.longitude ?? null,
      }));
      const evidenceScenarioId = (
        analysis?.scenario_id
        ?? run?.scenario_id
        ?? 'operator-submitted'
      );
      const nextAnalysis = await api.analyzeSubmittedEvidence(
        [...existingEvidence, ...submittedEvidence].slice(-10),
        evidenceScenarioId,
      );
      const nextRun = await api.runOptimization({
        scenario_id: evidenceScenarioId,
        requested_by: 'mission-control',
        analysis_id: nextAnalysis.analysis_id,
      });
      mergeAnalysis(nextAnalysis);
      mergeRun(nextRun);
      setSources([]);
      setShowIntake(false);
      setEvidenceGap(null);
      setDraftIncident(null);
    } catch (requestError) {
      reportFailure(requestError, 'submitted evidence', analyzeSubmitted);
    } finally {
      setLoading(false);
    }
  };

  // Back to the opening picture. Two halves:
  //
  //  (a) Client state is cleared synchronously, before the network work starts,
  //      so nothing stale is on screen while the new plan is computed.
  //  (b) A road closure is NOT client state — it lives on the server run as
  //      `result.vrp_solution.active_road_blocks`, and every re-plan inherits it
  //      through `parent_run_id`. The only honest way to clear it without
  //      inventing an endpoint is to compute a genuinely new run that has no
  //      parent and no blocked edges, which is what this does.
  const resetDemo = async () => {
    if (resetBusy) return;
    setResetBusy(true);
    setLoading(true);

    setSources([]);
    setShowIntake(false);
    setEvidenceGap(null);
    setDraftIncident(null);
    setAddMode(false);
    setSourceRequest(null);
    setDiagnosticsOpen(false);
    setDiagnosticsTab('overview');
    setTraceOpen(false);
    setReviewDialog(null);
    setImageryResult(null);
    setBaseline(null);
    setBaselineError(null);
    setSelectedScenarioId('');
    setScenarioStage('baseline');
    setSelectedVillageId(null);
    setMissionPlaying(false);
    setMissionElapsed(0);
    missionElapsedRef.current = 0;
    setMissionSpeed(DEFAULT_MISSION_SPEED);
    focusedGemmaLocation.current = false;
    clearFailure();

    try {
      // A fresh analysis drops operator-submitted evidence back to the stock
      // fixture set. No analysis_id is reused, so nothing carries over.
      const nextAnalysis = await api.runGemmaAnalysis('nepal-national-demo');
      // No parent_run_id, no blocked_edge_ids, no vehicle_positions: an
      // unconstrained plan with the fleet at the depot at T+00:00.
      const nextRun = await api.runOptimization({
        scenario_id: 'nepal-national-demo',
        requested_by: 'mission-control',
        analysis_id: nextAnalysis.analysis_id,
      });
      setAnalysis(nextAnalysis);
      analysisRef.current = nextAnalysis;
      setRun(nextRun);
      runRef.current = nextRun;
      setResetOpen(false);
      changeWorkspace('operations');
      return true;
    } catch (requestError) {
      // Leave the dialog open on failure — silently closing it would read as
      // "reset done" when the plan on screen is still the old one.
      reportFailure(requestError, 'demo reset', resetDemo);
      return false;
    } finally {
      setResetBusy(false);
      setLoading(false);
    }
  };

  const recordQuestionDisposition = async (questionId, disposition) => {
    const analysisId = analysisRef.current?.analysis_id;
    if (!analysisId || dispositionBusy) return false;
    setDispositionBusy(true);
    clearFailure();
    try {
      const updated = await api.recordQuestionDisposition(
        analysisId,
        questionId,
        disposition,
      );
      if (analysisRef.current?.analysis_id !== analysisId) {
        reportFailure(
          new Error('The Gemma analysis changed before the evidence disposition was recorded.'),
          null,
          null,
        );
        return false;
      }
      mergeAnalysis(updated);
      return true;
    } catch (requestError) {
      reportFailure(
        requestError,
        'evidence disposition',
        () => recordQuestionDisposition(questionId, disposition),
      );
      return false;
    } finally {
      setDispositionBusy(false);
    }
  };

  const openDiagnostics = useCallback((tab = 'overview') => {
    setDiagnosticsTab(tab);
    setDiagnosticsOpen(true);
  }, []);

  const reoptimizeForRoadBlocks = async (blockedEdgeIds, disruptionReason) => {
    if (loading || !analysis?.analysis_id) return;
    setLoading(true);
    clearFailure();
    try {
      // A corridor closes at a moment in the mission, not before it starts. This
      // path used to omit vehicle_positions entirely, so closing a road silently
      // teleported the whole fleet back to the depot and re-planned as if nothing
      // had been delivered yet — the plan looked plausible and was wrong about
      // every asset already on the road. Read the clock through the refs so the
      // basis is the operator's mission time at the moment they pressed it.
      const elapsedMinutes = missionElapsedRef.current;
      const positions = fleetPositionsAt(
        runRef.current?.result?.vrp_solution?.routes,
        elapsedMinutes,
      );
      const nextRun = await api.runOptimization({
        scenario_id: run?.scenario_id ?? 'nepal-national-demo',
        requested_by: 'mission-control',
        analysis_id: analysis.analysis_id,
        blocked_edge_ids: blockedEdgeIds,
        parent_run_id: run?.run_id,
        trigger: blockedEdgeIds.length ? 'road_closure' : 'road_reopened',
        disruption_reason: disruptionReason,
        vehicle_positions: positions,
        time_elapsed_hours: Math.min(168, Math.max(0, elapsedMinutes / 60)),
      });
      mergeRun(nextRun);
    } catch (requestError) {
      reportFailure(
        requestError,
        'road-disruption plan',
        () => reoptimizeForRoadBlocks(blockedEdgeIds, disruptionReason),
      );
    } finally {
      setLoading(false);
    }
  };

  const reviewRun = async (approved, rationale = '', snapshot = null) => {
    const pinned = snapshot ?? {
      run_id: runRef.current?.run_id,
      analysis_id: runRef.current?.analysis_id,
      updated_at: runRef.current?.updated_at,
    };
    const current = runRef.current;
    if (
      !pinned?.run_id
      || !current
      || current.run_id !== pinned.run_id
      || current.analysis_id !== pinned.analysis_id
      || current.updated_at !== pinned.updated_at
      || reviewBusy
      || loading
    ) {
      reportFailure(
        new Error('The reviewed snapshot changed. Re-open authorization on the current run.'),
        null,
        null,
      );
      return false;
    }
    setReviewBusy(true);
    clearFailure();
    try {
      const next = approved
        ? await api.approveOptimizationRun(
            pinned.run_id,
            'mission-control',
            `Operator approved versioned coordination snapshot ${pinned.run_id}.${rationale ? ` Override rationale: ${rationale}` : ''}`,
            pinned.updated_at,
            pinned.analysis_id,
          )
        : await api.rejectOptimizationRun(
            pinned.run_id,
            'mission-control',
            rationale || 'Operator rejected the proposed response plan.',
            pinned.updated_at,
            pinned.analysis_id,
          );
      mergeRun(next);
      return true;
    } catch (requestError) {
      reportFailure(
        requestError,
        'review decision',
        () => reviewRun(approved, rationale, pinned),
      );
      return false;
    } finally {
      setReviewBusy(false);
    }
  };

  useEffect(() => {
    if (bootstrapped.current) return;
    bootstrapped.current = true;
    const loadReferenceData = () => {
      clearFailure();
      return Promise.all([
        api.getVillages(),
        api.getVehicles(),
        api.getDemoScenarios(),
      ])
        .then(([villages, vehicles, demoScenarios]) => {
          setVillagesData(villages);
          setFleet(vehicles.vehicles ?? []);
          const availableScenarios = demoScenarios.scenarios ?? [];
          setScenarios(availableScenarios);
          // Default to a scenario whose closure actually changes the plan.
          // Fixtures load in filename order, so the first one was Taplejung —
          // whose corridor carries no routes, making the rerouting moment a
          // measured no-op (0.00 km change). Nepalgunj's closure moves all five
          // trucks; Janakpur's moves two. Prefer those, in that order.
          setSelectedScenarioId((current) => {
            if (current) return current;
            const byImpact = [
              'nepalgunj-hospital-eastwest-closure',
              'janakpur-flood-bp-closure',
            ];
            const preferred = byImpact.find((id) =>
              availableScenarios.some((item) => item.scenario_id === id));
            return preferred || availableScenarios[0]?.scenario_id || '';
          });
        })
        .catch((requestError) => reportFailure(requestError, 'map resources', loadReferenceData));
    };
    loadReferenceData();
    loadRun();
    // The declared schemas are what the agent console prints as the contract
    // handed to the model. A failure here degrades the console to bare function
    // names rather than surfacing an error, so it is deliberately not reported.
    api.getDeclaredFunctions()
      .then((payload) => setDeclaredFunctions(payload?.declared_functions ?? []))
      .catch(() => setDeclaredFunctions([]));
  }, [clearFailure, loadRun, reportFailure]);

  useEffect(() => {
    if (!selectedVillageId && villagesData.villages.length) {
      setSelectedVillageId(villagesData.villages[0].id);
    }
  }, [selectedVillageId, villagesData.villages]);

  useEffect(() => {
    const matchedVillage = run?.result?.gemma_signal?.matched_villages?.[0];
    if (
      !focusedGemmaLocation.current &&
      matchedVillage &&
      villagesData.villages.some((item) => item.id === matchedVillage)
    ) {
      focusedGemmaLocation.current = true;
      setSelectedVillageId(matchedVillage);
    }
  }, [run, villagesData.villages]);

  useEffect(() => {
    const type = lastMessage?.event_type ?? lastMessage?.type;
    if (type === 'optimization_completed' && lastMessage.payload?.run_id) {
      mergeRun(lastMessage.payload);
    }
    if (type === 'gemma_analysis_completed' && lastMessage.payload?.analysis_id) {
      mergeAnalysis(lastMessage.payload);
    }
    if (type === 'evidence_question_disposition_recorded') {
      mergeQuestionDisposition(
        lastMessage.payload?.analysis_id,
        lastMessage.payload?.disposition,
      );
    }
    if (
      ['hitl_review_required', 'hitl_approved', 'hitl_rejected'].includes(type)
      && lastMessage.payload?.run_id
    ) {
      api.getOptimizationRun(lastMessage.payload.run_id)
        .then(mergeRun)
        .catch((requestError) => reportFailure(
          requestError,
          'event run snapshot',
          () => api.getOptimizationRun(lastMessage.payload.run_id).then(mergeRun),
        ));
    }
  }, [
    lastMessage,
    mergeAnalysis,
    mergeQuestionDisposition,
    mergeRun,
    reportFailure,
  ]);

  useEffect(() => {
    if (!run?.orchestration) return;
    setLastOrchestration({ record: run.orchestration, runId: run.run_id });
  }, [run?.orchestration, run?.run_id]);

  // Runs and analyses arrive on independent channels: a run can come from a
  // websocket event, a re-plan, or a scenario activation, while the analysis on
  // screen is whatever was fetched last. Whenever they drift apart, pull the
  // analysis the displayed run actually consumed. Without this the evidence
  // queue reasons about one extraction while the plan was built from another —
  // which reads to an operator as the system contradicting itself.
  const pairedAnalysisId = run?.analysis_id ?? null;
  const analysisMismatch = Boolean(
    pairedAnalysisId && analysis && analysis.analysis_id !== pairedAnalysisId,
  );
  useEffect(() => {
    if (!pairedAnalysisId) return;
    if (analysisRef.current?.analysis_id === pairedAnalysisId) return;
    let cancelled = false;
    api.getGemmaAnalysis(pairedAnalysisId)
      .then((record) => {
        if (cancelled) return;
        // Bypass newerAnalysis: the run's own analysis wins even when it is
        // older by timestamp, because it is the one the plan was computed from.
        analysisRef.current = record;
        setAnalysis(record);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [pairedAnalysisId]);

  useEffect(() => {
    const snapshot = reviewDialog?.runSnapshot;
    if (
      !snapshot
      || !run
      || (
        run.run_id === snapshot.run_id
        && run.analysis_id === snapshot.analysis_id
        && run.updated_at === snapshot.updated_at
      )
    ) return;
    setReviewDialog(null);
    reportFailure(
      new Error('Authorization closed because a newer versioned run replaced the reviewed snapshot.'),
      null,
      null,
    );
  }, [reportFailure, reviewDialog, run]);

  // Stable identities: a fresh `?? []` on every render defeats MapPanel's memo
  // and re-triggers the mission-clock derivations.
  const urgency = useMemo(
    () => run?.result?.urgency_scores ?? [],
    [run?.result?.urgency_scores],
  );
  const routes = useMemo(
    () => run?.result?.vrp_solution?.routes ?? [],
    [run?.result?.vrp_solution?.routes],
  );

  // ---- Mission clock -------------------------------------------------------
  // Movement is gated on human authorization: an unapproved plan never animates.
  const dispatchActive = run?.status === 'approved';
  const missionHorizon = useMemo(() => {
    const durations = routes
      .filter(isFeasibleRoute)
      .map((route) => Number(route.total_time_minutes))
      .filter((value) => Number.isFinite(value) && value > 0);
    return durations.length ? Math.max(...durations) : 0;
  }, [routes]);

  // Stop-level state at the current mission time: what has already been served
  // is history and cannot be re-planned; only pending stops remain actionable.
  const stopStatus = useMemo(() => {
    const stops = routes
      .filter(isFeasibleRoute)
      .flatMap((route) => (route.stop_details ?? []).map((stop) => ({
        vehicleId: route.vehicle_id,
        villageId: stop.village_id,
        eta: Number(stop.eta_minutes),
      })))
      .filter((stop) => Number.isFinite(stop.eta));
    const served = stops.filter((stop) => stop.eta <= missionElapsed);
    const pending = stops.filter((stop) => stop.eta > missionElapsed);
    const nextEta = pending.length
      ? Math.min(...pending.map((stop) => stop.eta))
      : null;
    return {
      total: stops.length,
      served: served.length,
      pending: pending.length,
      servedStops: served,
      pendingStops: pending,
      nextEta,
    };
  }, [routes, missionElapsed]);

  // A new plan, or the loss of approval, always parks the fleet again.
  useEffect(() => {
    missionElapsedRef.current = missionElapsed;
  }, [missionElapsed]);

  useEffect(() => {
    setMissionElapsed(0);
    setMissionPlaying(false);
  }, [run?.run_id, run?.status]);

  useEffect(() => {
    if (!missionPlaying || !dispatchActive) return undefined;
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches) return undefined;
    let frame = 0;
    let last = performance.now();
    const step = (now) => {
      const deltaMinutes = ((now - last) / 1000) * missionSpeed;
      last = now;
      setMissionElapsed((current) => {
        const next = current + deltaMinutes;
        if (next >= missionHorizon) {
          setMissionPlaying(false);
          return missionHorizon;
        }
        return next;
      });
      frame = window.requestAnimationFrame(step);
    };
    frame = window.requestAnimationFrame(step);
    return () => window.cancelAnimationFrame(frame);
  }, [missionPlaying, dispatchActive, missionHorizon, missionSpeed]);

  const missionClock = useMemo(() => ({
    elapsed: missionElapsed,
    horizon: missionHorizon,
    playing: missionPlaying,
    speed: missionSpeed,
    onSpeedChange: setMissionSpeed,
    dispatchActive,
    runStatus: run?.status,
    stopStatus,
    onScrub: (value) => {
      setMissionPlaying(false);
      setMissionElapsed(value);
    },
    onTogglePlay: () => setMissionPlaying((current) => !current),
    onReset: () => {
      setMissionPlaying(false);
      setMissionElapsed(0);
    },
  }), [
    missionElapsed, missionHorizon, missionPlaying, missionSpeed,
    dispatchActive, run?.status, stopStatus,
  ]);

  const selectedVillage = useMemo(
    () =>
      villagesData.villages.find((item) => item.id === selectedVillageId),
    [villagesData.villages, selectedVillageId],
  );

  // Corridor ids are terrain-graph edge ids and village ids are node ids, so the
  // corridor an operator would check for this incident is one that touches it.
  const imageryTarget = useMemo(() => {
    const corridor = corridorForVillage(
      run?.result?.vrp_solution?.road_network ?? [],
      selectedVillageId,
    );
    if (!corridor?.edge_id) return null;
    const incidentType = imageryIncidentType(analysis, corridor);
    return {
      corridorId: corridor.edge_id,
      corridorName: corridor.name ?? corridor.edge_id,
      incidentType,
      evidenceId: imageryEvidenceId(analysis, incidentType),
    };
  }, [run?.result?.vrp_solution?.road_network, selectedVillageId, analysis]);

  // Selecting a different incident invalidates the last check: the result named
  // a corridor that is no longer the one on screen.
  useEffect(() => setImageryResult(null), [selectedVillageId]);

  // The operations workspace exposes imagery controls before the evidence view
  // opens, so availability is resolved at mount. A timeout makes a missing
  // backend a terminal disabled state instead of an endless pending affordance.
  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), IMAGERY_STATUS_TIMEOUT_MS);

    api.getImageryStatus({ signal: controller.signal })
      .then((status) => {
        if (!active) return;
        setImageryStatus(status && typeof status === 'object'
          ? status
          : IMAGERY_STATUS_UNAVAILABLE);
      })
      .catch(() => {
        if (active) setImageryStatus(IMAGERY_STATUS_UNAVAILABLE);
      })
      .finally(() => clearTimeout(timeout));

    return () => {
      active = false;
      clearTimeout(timeout);
      controller.abort();
    };
  }, []);

  const imageryChip = useMemo(() => imageryStatusChip(imageryStatus), [imageryStatus]);
  const imageryAvailable = useMemo(
    () => imageryActionsAvailable(imageryStatus),
    [imageryStatus],
  );
  const imageryAvailabilityNotice = imageryAvailable ? '' : imageryChip.detail;
  const toggleAddMode = useCallback(
    () => setAddMode((current) => !current),
    [],
  );
  const placeDraftIncident = useCallback(({ lat, lng }) => {
    setDraftIncident({ lat, lng });
    setAddMode(false);
    openEvidence({
      id: `map-report-${lat.toFixed(4)}-${lng.toFixed(4)}`,
      label: 'Map event report',
      detail: `Verify the reported event at ${lat.toFixed(4)}, ${lng.toFixed(4)} with a named source. This marker is evidence context, not a routable incident until validated.`,
      field: 'map_location',
      tone: 'attention',
      location: { latitude: lat, longitude: lng },
    });
  }, [openEvidence]);
  const openRoadDiagnostics = useCallback(
    () => openDiagnostics('routing'),
    [openDiagnostics],
  );
  const openSourceReport = useCallback((request) => setSourceRequest(request), []);
  const closeSourceReport = useCallback(() => setSourceRequest(null), []);

  return (
    <div className="ops-app" aria-busy={loading}>
      <a className="ops-skip-link" href="#mission-workspace">Skip to operations</a>
      <HostedBuildNotice />
      <Header
        connected={isConnected}
        transport={transport}
        loading={loading}
        run={run}
        analysis={analysis}
        onRun={rerunGemma}
        onAddEvidence={() => openEvidence(null)}
        onReset={() => setResetOpen(true)}
        resetBusy={resetBusy}
        activeWorkspace={activeWorkspace}
        onWorkspaceChange={changeWorkspace}
      />
      <StageStepper
        activeWorkspace={activeWorkspace}
        onWorkspaceChange={changeWorkspace}
        run={run}
        analysis={analysis}
        loading={loading}
        onRun={rerunGemma}
      />
      <span className="ops-live-region" role="status" aria-live="polite">
        {loading
          ? 'Gemma and deterministic optimization pipeline running.'
          : run?.status
            ? `Pipeline complete. Run status: ${run.status.replaceAll('_', ' ')}.`
            : 'Mission workspace ready.'}
      </span>
      {error && (
        <div className="ops-error" role="alert">
          <Icon name="error" size={18} />
          <span>{error.message}</span>
          {failedRetryRef.current && (
            <button type="button" onClick={() => failedRetryRef.current?.run()}>
              Retry {error.retryLabel}
            </button>
          )}
        </div>
      )}
      <main className="ops-shell" id="mission-workspace">
        {activeWorkspace === 'operations' && (
          <div className="ops-workspace" data-workspace="operations">
            <div className="ops-workspace-heading" data-workspace-heading tabIndex="-1">
              <span className="ops-eyebrow">Stage 1 · Operations</span>
              <h1>See what happened, and where help is going</h1>
              <p>Pick an incident, load a scenario, or add a field report. Anything you change here produces a new plan &mdash; and every plan still needs a human signature.</p>
            </div>
            <MissionLauncher
              run={run}
              analysis={analysis}
              loading={loading}
              orchestrateBusy={orchestrateBusy}
              onRun={rerunGemma}
              onOrchestrate={orchestrateWithGemma}
              onOpenEvidence={() => changeWorkspace('evidence')}
            />
            <ScenarioSwitcher
              scenarios={scenarios}
              selectedId={selectedScenarioId}
              stage={scenarioStage}
              activeScenarioId={run?.scenario_id}
              busy={scenarioBusy}
              onSelect={setSelectedScenarioId}
              onStageChange={setScenarioStage}
              onActivate={activateScenario}
            />
            <MissionBrief run={run} analysis={analysis} loading={loading} />
            <section className="ops-deck" id="overview" aria-label="Mission operations">
              <IncidentRail
                villages={villagesData.villages}
                urgency={urgency}
                selectedId={selectedVillageId}
                onSelect={setSelectedVillageId}
              />
              <MapPanel
                villages={villagesData.villages}
                depot={villagesData.depot}
                routes={routes}
                selectedId={selectedVillageId}
                onSelect={setSelectedVillageId}
                addMode={addMode}
                draftIncident={draftIncident}
                onToggleAdd={toggleAddMode}
                onDraftIncident={placeDraftIncident}
                roadNetwork={run?.result?.vrp_solution?.road_network ?? []}
                onOpenDisruption={openRoadDiagnostics}
                clock={missionClock}
              />
              <IncidentInspector
                selectedVillage={selectedVillage}
                urgency={urgency}
                routes={routes}
                onVerify={() => openEvidence({
                  id: `verify-${selectedVillage?.id ?? 'incident'}`,
                  label: `${selectedVillage?.name ?? 'Incident'} verification`,
                  detail: 'Provide a named report that verifies current impact, access, injuries, or unmet resources.',
                  field: 'incident_verification',
                  tone: 'attention',
                })}
                onOpenRoutes={() => openDiagnostics('routing')}
                imageryTarget={imageryTarget}
                imageryBusy={imageryBusy || orchestrateBusy || loading}
                imageryResult={imageryResult}
                imageryAvailable={imageryAvailable}
                imageryAvailabilityNotice={imageryAvailabilityNotice}
                onAskGemmaImagery={askGemmaForImagery}
                onCheckImagery={checkImageryNow}
              />
            </section>
          </div>
        )}

        {activeWorkspace === 'evidence' && (
          <div className="ops-workspace" data-workspace="evidence">
            <div className="ops-workspace-heading" data-workspace-heading tabIndex="-1">
              <span className="ops-eyebrow">Stage 2 · Gemma evidence</span>
              <h1>Watch the model work, then check what it was allowed to change</h1>
              <p>The console below replays Gemma&rsquo;s function-calling turn line by line. Under it: what the extraction found, what it refused to guess, and the exact numbers handed to the maths.</p>
            </div>
            {imageryChip && (
              <div className={`ops-imagery-status ${imageryChip.tone}`} role="status">
                <StatusDot tone={IMAGERY_CHIP_DOT[imageryChip.tone] ?? 'attention'} />
                <b>{imageryChip.label}</b>
                <small>{imageryChip.detail}</small>
              </div>
            )}
            {analysisMismatch && (
              <div className="ops-pairing-notice" role="status">
                <Icon name="sync" size={17} />
                <span>
                  <b>Re-pairing evidence with the current plan.</b>
                  <small>
                    Run {run?.run_id} was computed from analysis {pairedAnalysisId}; the
                    panel below is still showing {analysis?.analysis_id}. Loading the
                    matching extraction now — values are never mixed across analyses.
                  </small>
                </span>
              </div>
            )}
            <AgentConsole
              orchestration={run?.orchestration ?? lastOrchestration?.record}
              sourceRunId={run?.orchestration ? run.run_id : lastOrchestration?.runId}
              currentRunId={run?.run_id}
              run={run}
              messages={messages}
              transport={transport}
              declaredFunctions={declaredFunctions}
              onOrchestrate={orchestrateWithGemma}
              busy={orchestrateBusy || loading}
            />
            <GemmaWorkbenchV2
              analysis={analysis}
              signal={run?.result?.gemma_signal}
              onRun={rerunGemma}
              loading={loading}
              onSupplyGap={openEvidence}
              onDisposition={recordQuestionDisposition}
              dispositionBusy={dispositionBusy}
              onCite={openSourceReport}
            />
            <RawExchangePanel analysis={analysis} />
          </div>
        )}

        {activeWorkspace === 'math' && (
          <div className="ops-workspace" data-workspace="math">
            <div className="ops-workspace-heading" data-workspace-heading tabIndex="-1">
              <span className="ops-eyebrow">Stage 3 · Math lab</span>
              <h1>Check the maths: routes, terrain cost, and the baseline</h1>
              <p>Every number here comes from the run on screen. The terrain panel shows why each corridor was priced as it was; the baseline shows what the same engine does with terrain reasoning switched off.</p>
            </div>
            <MathEngineV2
              run={run}
              analysis={analysis}
              onOpenDiagnostics={openDiagnostics}
              onSupplyGap={openEvidence}
              onCite={openSourceReport}
            />
            <TerrainCostPanel run={run} />
            <BaselinePanel
              report={baseline}
              onLoad={loadBaseline}
              busy={baselineBusy}
              error={baselineError}
            />
          </div>
        )}

        {activeWorkspace === 'review' && (
          <div className="ops-workspace" data-workspace="review">
            <div className="ops-workspace-heading" data-workspace-heading tabIndex="-1">
              <span className="ops-eyebrow">Stage 4 · Review & authorize</span>
              <h1>Read the whole run, then approve or reject it</h1>
              <p>Approval covers the entire national plan, not the incident you happen to have selected. Nothing dispatches until a person signs for it.</p>
            </div>
            <section className="ops-review-workspace" aria-label="Run review and authorization">
              <DecisionPanel
                run={run}
                analysis={analysis}
                selectedVillage={selectedVillage}
                urgency={urgency}
                onApprove={(rationale) => setReviewDialog({
                  kind: 'approve',
                  rationale,
                  runSnapshot: {
                    run_id: run?.run_id,
                    analysis_id: run?.analysis_id,
                    updated_at: run?.updated_at,
                    status: run?.status,
                  },
                })}
                onReject={() => setReviewDialog({
                  kind: 'reject',
                  runSnapshot: {
                    run_id: run?.run_id,
                    analysis_id: run?.analysis_id,
                    updated_at: run?.updated_at,
                    status: run?.status,
                  },
                })}
                reviewBusy={reviewBusy}
                pipelineBusy={loading}
                onOpenDiagnostics={openDiagnostics}
                onReviewIssues={() => openDiagnostics('issues')}
                onSupplyGap={openEvidence}
              />
              <TraceCard analysis={analysis} run={run} messages={messages} onOpen={() => setTraceOpen(true)} />
            </section>
          </div>
        )}

        <footer className="ops-footer">
          <span>Gemma {analysis?.prompt_version ?? 'nepal-grounded-extraction-v3'}</span>
          <span>{fleet.length} fleet assets</span>
          <span>{analysis?.fixture_notice ?? 'Evidence provenance unavailable'}</span>
        </footer>
      </main>

      <EvidenceDrawer
        open={showIntake}
        sources={sources}
        setSources={setSources}
        onClose={closeEvidence}
        onAnalyze={analyzeSubmitted}
        loading={loading}
        gap={evidenceGap}
      />
      <DiagnosticsDialog
        open={diagnosticsOpen}
        initialTab={diagnosticsTab}
        run={run}
        analysis={analysis}
        villages={villagesData.villages}
            onClose={() => setDiagnosticsOpen(false)}
            onReoptimize={reoptimizeForRoadBlocks}
            elapsedMinutes={missionElapsed}
            dispatchActive={dispatchActive}
            onOpenScenarios={() => {
              setDiagnosticsOpen(false);
              changeWorkspace('operations');
              window.requestAnimationFrame(() => {
                document
                  .querySelector('.ops-scenario-switcher')
                  ?.scrollIntoView({ behavior: 'smooth', block: 'center' });
              });
            }}
            onSupplyGap={(gap) => {
              setDiagnosticsOpen(false);
              setEvidenceGap(gap);
              setShowIntake(true);
            }}
            loading={loading}
          />
      <TraceDialog
        open={traceOpen}
        analysis={analysis}
        onClose={() => setTraceOpen(false)}
      />
      <SourceReportDialog
        request={sourceRequest}
        analysis={analysis}
        imageryAvailable={imageryAvailable}
        imageryAvailabilityNotice={imageryAvailabilityNotice}
        onClose={closeSourceReport}
      />
      <ResetDemoDialog
        open={resetOpen}
        onClose={() => setResetOpen(false)}
        onConfirm={resetDemo}
        busy={resetBusy}
        closureCount={(run?.result?.vrp_solution?.active_road_blocks ?? []).length}
      />
      <DecisionReviewDialog
        action={reviewDialog}
        run={reviewDialog?.runSnapshot ?? run}
        onClose={() => setReviewDialog(null)}
        onConfirm={async (approved, notes) => {
          const recorded = await reviewRun(approved, notes, reviewDialog?.runSnapshot);
          if (recorded) setReviewDialog(null);
        }}
        busy={reviewBusy}
      />
    </div>
  );
}
