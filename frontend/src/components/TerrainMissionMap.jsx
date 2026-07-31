import { useEffect, useRef, useState } from 'react';
import { Map, Maximize2, Mountain, TriangleAlert, Truck } from 'lucide-react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

// Legacy model builders remain below for migration history, but the active
// renderer is the generated WebP symbol layer. Keeping Three.js out of the
// module graph removes its startup and per-frame cost.
const THREE = null;

const ROUTE_COLORS = [
  '#ce5948', // copper
  '#7490b1', // slate blue
  '#dc9b52', // brass
  '#d1832b', // clay
  '#72a856', // sage
  '#7591b2', // mauve
  '#8ea171', // teal
  '#dea15d', // tan
  '#89a0bd', // periwinkle
  '#7aae61', // olive
];

// Colour is an identity channel on this map: if two assets share one, the map
// cannot answer "which vehicle is that". Hashing nine fleet ids into a
// five-colour palette produced four collisions on the standard fleet. Fleet ids
// are structured (`family_N`), so derive a stable slot from family and index and
// fall back to the hash only for ids that do not follow the convention.
const VEHICLE_FAMILY_SLOT = { heli: 0, truck: 4, boat: 8 };
const EMPTY_COLLECTION = { type: 'FeatureCollection', features: [] };
const ENABLE_LEGACY_THREE_LAYER = false;
const THREE_FRAME_INTERVAL_MS = 50;
const TERRAIN_SAMPLE_INTERVAL_MS = 500;
const SIMULATION_STEP_MS = 100;
const SIMULATION_MINUTES_PER_REAL_SECOND = 12;

function colorForVehicle(vehicleId = '') {
  const structured = /^([a-z]+)_(\d+)$/i.exec(vehicleId);
  if (structured) {
    const slot = VEHICLE_FAMILY_SLOT[structured[1].toLowerCase()];
    if (slot !== undefined) {
      const index = (slot + Number(structured[2]) - 1) % ROUTE_COLORS.length;
      return ROUTE_COLORS[index];
    }
  }
  const hash = [...vehicleId].reduce(
    (value, character) => ((value * 31) + character.charCodeAt(0)) >>> 0,
    0,
  );
  return ROUTE_COLORS[hash % ROUTE_COLORS.length];
}

function isFeasibleRoute(route) {
  return route?.feasible !== false;
}

function prefersReducedMotion() {
  return window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
}

function motionDuration(duration) {
  return prefersReducedMotion() ? 0 : duration;
}

function frameIncidentCorridor(map, depot, village, schematic = false) {
  if (!map || !village) return;
  if (!depot) {
    map.flyTo({
      center: [village.lng, village.lat],
      zoom: Math.max(map.getZoom(), 8.2),
      pitch: schematic ? 0 : 62,
      bearing: schematic ? 0 : -12,
      duration: motionDuration(850),
      essential: false,
    });
    return;
  }

  const mapWidth = map.getContainer()?.clientWidth ?? window.innerWidth;
  const compact = mapWidth <= 980;
  const bounds = new maplibregl.LngLatBounds()
    .extend([depot.lng, depot.lat])
    .extend([village.lng, village.lat]);
  map.fitBounds(bounds, {
    padding: compact
      ? {
          top: 96,
          right: Math.min(270, Math.round(mapWidth * 0.38)),
          bottom: 96,
          left: 28,
        }
      : {
          top: 96,
          right: Math.min(340, Math.round(mapWidth * 0.42)),
          bottom: 104,
          left: Math.min(120, Math.round(mapWidth * 0.12)),
        },
    maxZoom: 8.2,
    pitch: schematic ? 0 : 56,
    bearing: schematic ? 0 : -8,
    duration: motionDuration(420),
    essential: false,
  });
}

function routeCoordinates(route, depot, villagesById) {
  if (Array.isArray(route.path_coordinates) && route.path_coordinates.length > 1) {
    return route.path_coordinates;
  }
  const coordinates = depot ? [[depot.lng, depot.lat]] : [];
  const details = Array.isArray(route.stop_details) ? route.stop_details : [];

  if (details.length && typeof details[0] === 'object') {
    details.forEach((stop) => {
      if (Number.isFinite(stop.lat) && Number.isFinite(stop.lng)) {
        coordinates.push([stop.lng, stop.lat]);
      }
    });
    const finalStopDistance = Number(details.at(-1)?.cumulative_distance_km) || 0;
    const includesReturnLeg = (
      depot
      && Number(route.total_distance_km) > finalStopDistance + 0.01
    );
    if (includesReturnLeg) coordinates.push([depot.lng, depot.lat]);
  } else {
    (route.stops ?? []).forEach((villageId) => {
      const village = villagesById[villageId];
      if (village) coordinates.push([village.lng, village.lat]);
    });
  }

  return coordinates;
}

function circlePolygon(lng, lat, radiusMeters = 9000, steps = 28) {
  const latitudeDelta = radiusMeters / 111320;
  const longitudeDelta = radiusMeters / (111320 * Math.cos((lat * Math.PI) / 180));
  const ring = [];

  for (let index = 0; index <= steps; index += 1) {
    const angle = (index / steps) * Math.PI * 2;
    ring.push([
      lng + Math.cos(angle) * longitudeDelta,
      lat + Math.sin(angle) * latitudeDelta,
    ]);
  }

  return [ring];
}

function interpolateCoordinates(coordinates, progress) {
  if (coordinates.length < 2) return coordinates[0] ?? [85.324, 27.7172];

  const lengths = [];
  let total = 0;
  for (let index = 1; index < coordinates.length; index += 1) {
    const [previousLng, previousLat] = coordinates[index - 1];
    const [lng, lat] = coordinates[index];
    const length = Math.hypot(
      (lng - previousLng) * Math.cos((lat * Math.PI) / 180),
      lat - previousLat,
    );
    lengths.push(length);
    total += length;
  }

  let target = total * progress;
  for (let index = 0; index < lengths.length; index += 1) {
    if (target <= lengths[index]) {
      const ratio = lengths[index] === 0 ? 0 : target / lengths[index];
      const start = coordinates[index];
      const end = coordinates[index + 1];
      return [
        start[0] + (end[0] - start[0]) * ratio,
        start[1] + (end[1] - start[1]) * ratio,
      ];
    }
    target -= lengths[index];
  }

  return coordinates.at(-1);
}

function routeEntriesFromData(data) {
  const villagesById = Object.fromEntries(
    data.villages.map((village) => [village.id, village]),
  );
  return data.routes
    .filter(isFeasibleRoute)
    .map((route) => ({
      route,
      vehicleId: route.vehicle_id,
      color: colorForVehicle(route.vehicle_id),
      coordinates: routeCoordinates(route, data.depot, villagesById),
      relevant: !data.focusActive || route.stops?.includes(data.selectedId),
    }))
    .filter((entry) => entry.coordinates.length > 1);
}

function formatMinutes(value) {
  const minutes = Number(value);
  if (!Number.isFinite(minutes)) return 'ETA unavailable';
  if (minutes < 60) return `${Math.round(minutes)} min`;
  const hours = Math.floor(minutes / 60);
  const remainder = Math.round(minutes % 60);
  return `${hours}h ${remainder}m`;
}

function formatPayload(manifest = {}) {
  const entries = Object.entries(manifest)
    .filter(([, quantity]) => Number(quantity) > 0)
    .sort(([, a], [, b]) => Number(b) - Number(a));
  if (!entries.length) return 'No recorded payload';
  return entries
    .slice(0, 3)
    .map(([resource, quantity]) => (
      `${resource.replaceAll('_', ' ')} ${Number(quantity).toLocaleString()}`
    ))
    .join(' · ');
}

function isRelevantProperty(value) {
  return value === true || value === 1 || value === 'true' || value === '1';
}

function routeDestinationNames(route, villagesById) {
  return (route.stops ?? [])
    .map((id) => villagesById[id]?.name ?? id)
    .join(' → ');
}

function stopEta(route, villageId) {
  const stop = (route.stop_details ?? []).find((detail) => detail.village_id === villageId);
  return stop?.eta_minutes ?? route.total_time_minutes;
}

function createRoutePopupNode(properties) {
  const root = document.createElement('div');
  root.style.cssText = [
    'min-width:220px',
    'max-width:300px',
    'padding:4px',
    'color:#2e241b',
    'font:12px/1.45 "Public Sans Variable",system-ui,sans-serif',
  ].join(';');
  const title = document.createElement('strong');
  title.style.cssText = 'display:block;margin-bottom:7px;font-size:13px;color:#201913';
  title.textContent = properties.vehicle_id;
  root.append(title);
  [
    ['Origin', properties.origin],
    ['Destinations', properties.destinations],
    ['ETA', properties.eta],
    ['Distance', properties.distance],
    ['Payload', properties.payload],
  ].forEach(([label, value]) => {
    const row = document.createElement('div');
    row.style.cssText = 'display:grid;grid-template-columns:76px 1fr;gap:8px;margin-top:4px';
    const key = document.createElement('span');
    key.style.color = '#8a6b51';
    key.textContent = label;
    const content = document.createElement('b');
    content.style.fontWeight = '650';
    content.textContent = value;
    row.append(key, content);
    root.append(row);
  });
  return root;
}

function labelsOverlap(a, b, padding = 5) {
  return !(
    a.right + padding < b.left
    || a.left - padding > b.right
    || a.bottom + padding < b.top
    || a.top - padding > b.bottom
  );
}

function syncIncidentLabelVisibility(entries, selectedId, focusActive) {
  const ordered = [...entries].sort((a, b) => {
    if (a.village.id === selectedId) return -1;
    if (b.village.id === selectedId) return 1;
    return b.village.disaster_impact - a.village.disaster_impact;
  });

  ordered.forEach(({ village, element }) => {
    const selected = village.id === selectedId;
    element.classList.toggle('selected', selected);
    element.style.display = 'flex';
    element.style.visibility = 'visible';
    element.style.opacity = focusActive && !selected ? '0.14' : '1';
    element.style.pointerEvents = focusActive && !selected ? 'none' : 'auto';
  });

  window.requestAnimationFrame(() => {
    const accepted = [];
    ordered.forEach(({ village, element }) => {
      const selected = village.id === selectedId;
      if (focusActive && !selected) {
        element.style.visibility = 'hidden';
        return;
      }
      const bounds = element.getBoundingClientRect();
      if (!selected && accepted.some((acceptedBounds) => labelsOverlap(bounds, acceptedBounds))) {
        element.style.visibility = 'hidden';
        return;
      }
      accepted.push(bounds);
    });
  });
}

function routeSetSignature(entries) {
  return entries.map(({ route, vehicleId, coordinates }) => (
    `${vehicleId}:${Number(route.total_time_minutes).toFixed(3)}:`
    + `${(route.stop_details ?? []).map((stop) => (
      `${stop.village_id}@${Number(stop.eta_minutes).toFixed(3)}`
    )).join(',')}:`
    + coordinates.map(([lng, lat]) => `${lng.toFixed(4)},${lat.toFixed(4)}`).join(';')
  )).join('|');
}

function syncSimulationClock(clockRef, entries, timestamp) {
  const signature = routeSetSignature(entries);
  if (signature !== clockRef.current.signature) {
    clockRef.current = {
      signature,
      startedAtMs: timestamp,
      timelineAudit: auditRouteTimelines(entries),
    };
  }
  return clockRef.current;
}

function routeElapsedMinutes(clock, timestamp) {
  const elapsedMs = Math.max(0, timestamp - clock.startedAtMs);
  const fixedElapsedMs = Math.floor(elapsedMs / SIMULATION_STEP_MS) * SIMULATION_STEP_MS;
  return (fixedElapsedMs / 1000) * SIMULATION_MINUTES_PER_REAL_SECOND;
}

function routeProgress(route, clock, timestamp) {
  const durationMinutes = Number(route.total_time_minutes);
  if (!Number.isFinite(durationMinutes) || durationMinutes <= 0) return 1;
  return Math.min(routeElapsedMinutes(clock, timestamp) / durationMinutes, 1);
}

function routePositionAtElapsed(route, fallbackCoordinates, elapsedMinutes) {
  const durationMinutes = Number(route.total_time_minutes);
  const elapsed = Math.max(
    0,
    Math.min(
      Number.isFinite(Number(elapsedMinutes)) ? Number(elapsedMinutes) : 0,
      Number.isFinite(durationMinutes) ? durationMinutes : 0,
    ),
  );
  const legs = Array.isArray(route.legs)
    ? route.legs.filter((leg) => Array.isArray(leg.geometry) && leg.geometry.length > 1)
    : [];
  const stopDetails = Array.isArray(route.stop_details) ? route.stop_details : [];

  if (!legs.length || !Number.isFinite(durationMinutes) || durationMinutes <= 0) {
    const fallbackProgress = durationMinutes > 0 ? elapsed / durationMinutes : 1;
    return interpolateCoordinates(fallbackCoordinates, fallbackProgress);
  }

  let startsAt = 0;
  for (let index = 0; index < legs.length; index += 1) {
    const stopEta = Number(stopDetails[index]?.eta_minutes);
    const endsAt = Number.isFinite(stopEta) && stopEta >= startsAt
      ? Math.min(stopEta, durationMinutes)
      : index === legs.length - 1
        ? durationMinutes
        : startsAt + (
            (durationMinutes - startsAt)
            / Math.max(1, legs.length - index)
          );
    if (elapsed <= endsAt || index === legs.length - 1) {
      const legProgress = endsAt <= startsAt
        ? 1
        : Math.max(0, Math.min((elapsed - startsAt) / (endsAt - startsAt), 1));
      return interpolateCoordinates(legs[index].geometry, legProgress);
    }
    startsAt = endsAt;
  }

  return fallbackCoordinates.at(-1);
}

function routeHeadingAtElapsed(route, fallbackCoordinates, elapsedMinutes) {
  const durationMinutes = Math.max(Number(route.total_time_minutes) || 0, 0);
  const current = routePositionAtElapsed(route, fallbackCoordinates, elapsedMinutes);
  const sampleStep = Math.max(durationMinutes * 0.002, 0.02);
  const compareAt = elapsedMinutes >= durationMinutes - sampleStep
    ? Math.max(elapsedMinutes - sampleStep, 0)
    : Math.min(elapsedMinutes + sampleStep, durationMinutes);
  const comparison = routePositionAtElapsed(route, fallbackCoordinates, compareAt);
  const direction = compareAt < elapsedMinutes ? 1 : -1;
  return Math.atan2(
    (current[1] - comparison[1]) * direction,
    (current[0] - comparison[0]) * direction,
  );
}

function auditRouteTimelines(entries) {
  let maximumError = 0;
  let checkpointCount = 0;
  entries.forEach(({ route, coordinates }) => {
    (route.stop_details ?? []).forEach((stop) => {
      const expected = [Number(stop.lng), Number(stop.lat)];
      const eta = Number(stop.eta_minutes);
      if (!expected.every(Number.isFinite) || !Number.isFinite(eta)) return;
      const actual = routePositionAtElapsed(route, coordinates, eta);
      const latitudeScale = Math.cos((expected[1] * Math.PI) / 180);
      maximumError = Math.max(
        maximumError,
        Math.hypot(
          (actual[0] - expected[0]) * latitudeScale,
          actual[1] - expected[1],
        ),
      );
      checkpointCount += 1;
    });
  });
  return { maximumError, checkpointCount };
}

function createHelicopterModel(color) {
  const group = new THREE.Group();
  const bodyMaterial = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.36,
    metalness: 0.42,
    depthTest: false,
    depthWrite: false,
  });
  const darkMaterial = new THREE.MeshStandardMaterial({
    color: 0x242824,
    roughness: 0.3,
    metalness: 0.65,
    depthTest: false,
    depthWrite: false,
  });
  const glassMaterial = new THREE.MeshStandardMaterial({
    color: 0x8d9992,
    emissive: 0x202622,
    emissiveIntensity: 0.3,
    roughness: 0.15,
    metalness: 0.2,
    depthTest: false,
    depthWrite: false,
  });
  const signalMaterial = new THREE.MeshBasicMaterial({
    color: 0xd66d5e,
    depthTest: false,
    depthWrite: false,
  });

  const body = new THREE.Mesh(new THREE.CapsuleGeometry(2.15, 4.8, 6, 12), bodyMaterial);
  body.rotation.z = Math.PI / 2;
  body.scale.set(1.25, 1, 0.92);
  body.position.set(0.5, 0, 3.5);
  const cockpit = new THREE.Mesh(new THREE.SphereGeometry(2.05, 16, 10), glassMaterial);
  cockpit.scale.set(1.15, 0.96, 0.82);
  cockpit.position.set(3.65, 0, 3.7);
  const tail = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.72, 8.5, 8), bodyMaterial);
  tail.rotation.z = Math.PI / 2;
  tail.position.set(-6.8, 0, 3.75);
  const tailFin = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.35, 3.8), bodyMaterial);
  tailFin.position.set(-11.1, 0, 5.15);
  tailFin.rotation.y = -0.15;
  const horizontalStabilizer = new THREE.Mesh(
    new THREE.BoxGeometry(1.7, 4.4, 0.25),
    bodyMaterial,
  );
  horizontalStabilizer.position.set(-9.6, 0, 3.9);

  const skidLeft = new THREE.Mesh(new THREE.BoxGeometry(7.5, 0.25, 0.25), darkMaterial);
  skidLeft.position.set(0.5, -2.25, 0.75);
  const skidRight = skidLeft.clone();
  skidRight.position.y = 2.25;
  const skidBraceA = new THREE.Mesh(new THREE.BoxGeometry(0.25, 4.7, 0.25), darkMaterial);
  skidBraceA.position.set(-1.5, 0, 1.65);
  skidBraceA.rotation.x = 0.32;
  const skidBraceB = skidBraceA.clone();
  skidBraceB.position.x = 2.6;
  skidBraceB.rotation.x = -0.32;

  const rotor = new THREE.Group();
  const rotorMast = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.28, 1.6, 8), darkMaterial);
  rotorMast.rotation.x = Math.PI / 2;
  rotorMast.position.z = -0.8;
  const rotorHub = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.55, 0.35, 12), darkMaterial);
  rotorHub.rotation.x = Math.PI / 2;
  const bladeA = new THREE.Mesh(new THREE.BoxGeometry(18, 0.38, 0.12), darkMaterial);
  const bladeB = bladeA.clone();
  bladeB.rotation.z = Math.PI / 2;
  rotor.add(rotorMast, rotorHub, bladeA, bladeB);
  rotor.position.set(0, 0, 7.1);

  const tailRotor = new THREE.Group();
  const tailBladeA = new THREE.Mesh(new THREE.BoxGeometry(0.2, 3.6, 0.16), darkMaterial);
  const tailBladeB = tailBladeA.clone();
  tailBladeB.rotation.z = Math.PI / 2;
  tailRotor.add(tailBladeA, tailBladeB);
  tailRotor.position.set(-11.35, -0.36, 5.1);
  tailRotor.rotation.x = Math.PI / 2;

  const navigationLight = new THREE.Mesh(
    new THREE.SphereGeometry(0.3, 8, 6),
    signalMaterial,
  );
  navigationLight.position.set(-11.25, 0, 7.2);

  group.add(
    body,
    cockpit,
    tail,
    tailFin,
    horizontalStabilizer,
    skidLeft,
    skidRight,
    skidBraceA,
    skidBraceB,
    rotor,
    tailRotor,
    navigationLight,
  );
  group.userData = { rotor, tailRotor, vehicleType: 'helicopter' };
  return group;
}

function createTruckModel(color) {
  const group = new THREE.Group();
  const bodyMaterial = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.42,
    metalness: 0.4,
    depthTest: false,
    depthWrite: false,
  });
  const cargoMaterial = new THREE.MeshStandardMaterial({
    color: 0xd8d2c5,
    roughness: 0.68,
    metalness: 0.12,
    depthTest: false,
    depthWrite: false,
  });
  const glassMaterial = new THREE.MeshStandardMaterial({
    color: 0x84918a,
    emissive: 0x202622,
    roughness: 0.18,
    depthTest: false,
    depthWrite: false,
  });
  const wheelMaterial = new THREE.MeshStandardMaterial({
    color: 0x20231f,
    roughness: 0.9,
    depthTest: false,
    depthWrite: false,
  });
  const detailMaterial = new THREE.MeshStandardMaterial({
    color: 0x30342f,
    roughness: 0.7,
    metalness: 0.25,
    depthTest: false,
    depthWrite: false,
  });
  const lightMaterial = new THREE.MeshBasicMaterial({
    color: 0xe3c98c,
    depthTest: false,
    depthWrite: false,
  });

  const chassis = new THREE.Mesh(new THREE.BoxGeometry(17, 5.2, 0.65), detailMaterial);
  chassis.position.set(-0.5, 0, 2.05);
  const cargo = new THREE.Mesh(new THREE.BoxGeometry(10.8, 5.4, 5.2), cargoMaterial);
  cargo.position.set(-3.2, 0, 5);
  const cargoRoof = new THREE.Mesh(new THREE.BoxGeometry(11.1, 5.65, 0.32), bodyMaterial);
  cargoRoof.position.set(-3.2, 0, 7.72);
  const cab = new THREE.Mesh(new THREE.BoxGeometry(5.2, 5.25, 4.7), bodyMaterial);
  cab.position.set(5.1, 0, 4.75);
  const hood = new THREE.Mesh(new THREE.BoxGeometry(2.3, 5.1, 2.5), bodyMaterial);
  hood.position.set(8.6, 0, 3.65);
  const windshield = new THREE.Mesh(new THREE.BoxGeometry(0.2, 4.1, 1.8), glassMaterial);
  windshield.position.set(7.72, 0, 5.65);
  const grille = new THREE.Mesh(new THREE.BoxGeometry(0.24, 3.25, 1.15), detailMaterial);
  grille.position.set(9.78, 0, 3.65);
  const bumper = new THREE.Mesh(new THREE.BoxGeometry(0.45, 5.6, 0.55), detailMaterial);
  bumper.position.set(9.9, 0, 2.25);
  const headlightLeft = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.65, 0.55), lightMaterial);
  headlightLeft.position.set(9.94, -1.75, 4.25);
  const headlightRight = headlightLeft.clone();
  headlightRight.position.y = 1.75;
  group.add(
    chassis,
    cargo,
    cargoRoof,
    cab,
    hood,
    windshield,
    grille,
    bumper,
    headlightLeft,
    headlightRight,
  );

  const wheels = [];
  [-5.7, -1.8, 6.3].forEach((x) => {
    [-2.95, 2.95].forEach((y) => {
      const wheel = new THREE.Mesh(
        new THREE.CylinderGeometry(1.25, 1.25, 0.75, 14),
        wheelMaterial,
      );
      wheel.position.set(x, y, 1.75);
      wheel.rotation.x = Math.PI / 2;
      wheels.push(wheel);
      group.add(wheel);
    });
  });
  group.userData = { wheels, vehicleType: 'truck' };
  return group;
}

function createVehicleModel(transportMode, color) {
  const model = transportMode === 'air'
    ? createHelicopterModel(color)
    : createTruckModel(color);
  model.matrixAutoUpdate = false;
  return model;
}

function routeHeading(coordinates, progress) {
  const current = interpolateCoordinates(coordinates, progress);
  const comparisonProgress = progress >= 0.998
    ? Math.max(progress - 0.002, 0)
    : Math.min(progress + 0.002, 1);
  const comparison = interpolateCoordinates(coordinates, comparisonProgress);
  const direction = progress >= 0.998 ? 1 : -1;
  return Math.atan2(
    (current[1] - comparison[1]) * direction,
    (current[0] - comparison[0]) * direction,
  );
}

function mapBearingFromHeading(headingRadians) {
  return (90 - (headingRadians * 180) / Math.PI + 360) % 360;
}

function createVehicleSprite(type) {
  const canvas = document.createElement('canvas');
  canvas.width = 96;
  canvas.height = 96;
  const context = canvas.getContext('2d');
  context.scale(2, 2);
  context.lineCap = 'round';
  context.lineJoin = 'round';
  context.shadowColor = 'rgba(0, 0, 0, 0.42)';
  context.shadowBlur = 5;
  context.shadowOffsetY = 3;

  if (type === 'helicopter') {
    context.fillStyle = '#ddd0c5';
    context.beginPath();
    context.ellipse(24, 23, 6, 13, 0, 0, Math.PI * 2);
    context.fill();
    context.fillRect(22, 29, 4, 12);
    context.fillStyle = '#c84935';
    context.beginPath();
    context.ellipse(24, 16, 4, 5, 0, 0, Math.PI * 2);
    context.fill();
    context.strokeStyle = '#eae2dc';
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(5, 23);
    context.lineTo(43, 23);
    context.moveTo(24, 5);
    context.lineTo(24, 42);
    context.stroke();
    context.strokeStyle = '#463629';
    context.lineWidth = 2.5;
    context.beginPath();
    context.moveTo(17, 39);
    context.lineTo(31, 39);
    context.stroke();
  } else {
    context.fillStyle = '#ddd0c5';
    context.fillRect(14, 17, 20, 25);
    context.fillStyle = '#c84935';
    context.fillRect(13, 7, 22, 13);
    context.fillStyle = '#3e3025';
    context.fillRect(16, 10, 16, 5);
    context.fillStyle = '#2a2018';
    context.fillRect(10, 13, 4, 10);
    context.fillRect(34, 13, 4, 10);
    context.fillRect(10, 33, 4, 9);
    context.fillRect(34, 33, 4, 9);
  }
  return context.getImageData(0, 0, canvas.width, canvas.height);
}

function loadGeneratedFleetSprite(map, name, url) {
  const image = new Image();
  image.decoding = 'async';
  image.onload = () => {
    if (!map.getStyle() || !map.hasImage(name)) return;
    const canvas = document.createElement('canvas');
    canvas.width = 96;
    canvas.height = 96;
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, 96, 96);
    const scale = Math.min(90 / image.naturalWidth, 90 / image.naturalHeight);
    const width = image.naturalWidth * scale;
    const height = image.naturalHeight * scale;
    context.drawImage(
      image,
      (96 - width) / 2,
      (96 - height) / 2,
      width,
      height,
    );
    map.updateImage(name, context.getImageData(0, 0, 96, 96));
  };
  image.src = url;
}

function disposeObject(object) {
  object.traverse((child) => {
    child.geometry?.dispose();
    if (Array.isArray(child.material)) {
      child.material.forEach((material) => material.dispose());
    } else {
      child.material?.dispose();
    }
  });
}

function createMissionThreeLayer(villages, dataRef, simulationClockRef) {
  let beacons = [];
  let vehicles = [];
  const terrainElevations = new globalThis.Map();
  let vehicleSignature = '';
  let lastTerrainSampleMs = 0;
  let repaintTimer = null;

  return {
    id: 'three-mission-assets',
    type: 'custom',
    renderingMode: '3d',
    onAdd(map, gl) {
      this.map = map;
      this.sceneElement = map.getContainer().closest('.terrain-scene');
      this.camera = new THREE.Camera();
      this.scene = new THREE.Scene();
      this.scene.add(new THREE.AmbientLight(0xf1ece2, 2.8));
      const directional = new THREE.DirectionalLight(0xd7d0c2, 2.4);
      directional.position.set(0, -1, 1);
      this.scene.add(directional);
      this.renderer = new THREE.WebGLRenderer({
        canvas: map.getCanvas(),
        context: gl,
      });
      this.renderer.autoClear = false;
      vehicleSignature = '';
      vehicles = [];
      terrainElevations.clear();
      lastTerrainSampleMs = 0;
      beacons = villages.map((village, index) => {
        const group = new THREE.Group();
        group.matrixAutoUpdate = false;
        const color = village.disaster_impact >= 0.7 ? 0xd66d5e : 0xc8a45c;
        const core = new THREE.Mesh(
          new THREE.OctahedronGeometry(1.05, 0),
          new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: 0.9,
            depthTest: false,
            depthWrite: false,
          }),
        );
        core.position.z = 1.45;

        const ring = new THREE.Mesh(
          new THREE.TorusGeometry(1.6, 0.12, 8, 24),
          new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: 0.58,
            depthTest: false,
            depthWrite: false,
          }),
        );
        const stem = new THREE.Mesh(
          new THREE.CylinderGeometry(0.08, 0.14, 1.25, 8),
          new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: 0.72,
            depthTest: false,
            depthWrite: false,
          }),
        );
        stem.rotation.x = Math.PI / 2;
        stem.position.z = 0.65;
        group.add(core, ring, stem);
        this.scene.add(group);
        return {
          village,
          group,
          core,
          ring,
          stem,
          phase: index * 0.55,
        };
      });
      if (!prefersReducedMotion()) {
        repaintTimer = window.setInterval(
          () => this.map?.triggerRepaint(),
          THREE_FRAME_INTERVAL_MS,
        );
      }
    },
    render(_gl, args) {
      const projection = new THREE.Matrix4().fromArray(args.defaultProjectionData.mainMatrix);
      const timestamp = performance.now();
      const motionEnabled = !prefersReducedMotion();
      const now = motionEnabled ? timestamp / 1000 : 0;
      const currentData = dataRef.current;
      const routeEntries = routeEntriesFromData(currentData);
      const signature = routeSetSignature(routeEntries);
      const clock = syncSimulationClock(simulationClockRef, routeEntries, timestamp);
      if (signature !== vehicleSignature) {
        vehicles.forEach(({ model }) => {
          this.scene.remove(model);
          disposeObject(model);
        });
        vehicles = routeEntries.map((entry) => {
          const model = createVehicleModel(entry.route.transport_mode, entry.color);
          this.scene.add(model);
          return { ...entry, model };
        });
        vehicleSignature = signature;
      } else {
        routeEntries.forEach((entry, index) => {
          vehicles[index].route = entry.route;
          vehicles[index].coordinates = entry.coordinates;
          vehicles[index].relevant = entry.relevant;
        });
      }
      if (this.sceneElement) {
        this.sceneElement.dataset.threeAssetCount = String(vehicles.length);
      }

      const shouldSampleTerrain = (
        timestamp - lastTerrainSampleMs >= TERRAIN_SAMPLE_INTERVAL_MS
      );
      if (shouldSampleTerrain) lastTerrainSampleMs = timestamp;

      beacons.forEach(({ village, group, core, ring, stem, phase }) => {
        const elevationKey = `incident:${village.id}`;
        if (shouldSampleTerrain || !terrainElevations.has(elevationKey)) {
          terrainElevations.set(
            elevationKey,
            this.map.queryTerrainElevation([village.lng, village.lat]) ?? 0,
          );
        }
        const elevation = terrainElevations.get(elevationKey) ?? 0;
        const coordinate = maplibregl.MercatorCoordinate.fromLngLat(
          [village.lng, village.lat],
          elevation + 20,
        );
        const scale = coordinate.meterInMercatorCoordinateUnits();
        const selected = currentData.selectedId === village.id;
        const dimmed = currentData.focusActive && !selected;
        const pulse = 1 + Math.sin(now * 2 + phase) * (selected ? 0.1 : 0.05);
        const zoomScale = Math.min(1500, Math.max(
          240,
          (selected ? 620 : 430) * (2 ** (6.3 - this.map.getZoom())),
        ));
        group.matrix.copy(new THREE.Matrix4()
          .makeTranslation(coordinate.x, coordinate.y, coordinate.z)
          .scale(new THREE.Vector3(
            scale * pulse * zoomScale,
            -scale * pulse * zoomScale,
            scale * pulse * zoomScale,
          )));
        group.matrixWorldNeedsUpdate = true;
        core.material.opacity = dimmed ? 0.12 : selected ? 1 : 0.76;
        stem.material.opacity = dimmed ? 0.08 : 0.65;
        ring.material.opacity = dimmed
          ? 0.05
          : (selected ? 0.72 : 0.38) + Math.sin(now * 2 + phase) * 0.08;
      });

      vehicles.forEach((vehicle) => {
        vehicle.model.visible = vehicle.relevant && this.map.getPitch() >= 30;
        if (!vehicle.relevant) return;
        const progress = routeProgress(
          vehicle.route,
          clock,
          motionEnabled ? timestamp : clock.startedAtMs,
        );
        const lngLat = interpolateCoordinates(vehicle.coordinates, progress);
        const heading = routeHeading(vehicle.coordinates, progress);
        const elevationKey = `vehicle:${vehicle.vehicleId}`;
        if (shouldSampleTerrain || !terrainElevations.has(elevationKey)) {
          terrainElevations.set(
            elevationKey,
            this.map.queryTerrainElevation(lngLat) ?? 0,
          );
        }
        const terrainElevation = terrainElevations.get(elevationKey) ?? 0;
        const isHelicopter = vehicle.model.userData.vehicleType === 'helicopter';
        const coordinate = maplibregl.MercatorCoordinate.fromLngLat(
          lngLat,
          terrainElevation + (isHelicopter ? 240 : 16),
        );
        const unit = coordinate.meterInMercatorCoordinateUnits();
        // Models are operational symbols, so keep a stable on-screen footprint
        // across national and local zoom levels without adding more render passes.
        const zoomScale = 2 ** (6.3 - this.map.getZoom());
        const displayScale = Math.min(
          720,
          Math.max(90, (isHelicopter ? 360 : 310) * zoomScale),
        );
        vehicle.model.matrix.copy(new THREE.Matrix4()
          .makeTranslation(coordinate.x, coordinate.y, coordinate.z)
          .scale(new THREE.Vector3(
            unit * displayScale,
            -unit * displayScale,
            unit * displayScale,
          ))
          .multiply(new THREE.Matrix4().makeRotationZ(heading)));
        vehicle.model.matrixWorldNeedsUpdate = true;

        if (isHelicopter) {
          vehicle.model.userData.rotor.rotation.z = now * 8.5;
          vehicle.model.userData.tailRotor.rotation.z = now * 13;
        } else {
          vehicle.model.userData.wheels.forEach((wheel) => {
            wheel.rotation.y = -now * 4.5;
          });
        }
      });

      this.camera.projectionMatrix.copy(projection);
      this.renderer.resetState();
      this.renderer.render(this.scene, this.camera);
    },
    onRemove() {
      if (repaintTimer) window.clearInterval(repaintTimer);
      if (this.sceneElement) delete this.sceneElement.dataset.threeAssetCount;
      disposeObject(this.scene);
      this.renderer?.dispose();
    },
  };
}

function buildIncidentData(villages, selectedId, focusActive = false) {
  return {
    type: 'FeatureCollection',
    features: villages.map((village) => ({
      type: 'Feature',
      properties: {
        id: village.id,
        name: village.name,
        selected: village.id === selectedId ? 1 : 0,
        dimmed: focusActive && village.id !== selectedId ? 1 : 0,
        color: village.id === selectedId
          ? '#d06150'
          : village.disaster_impact >= 0.7 ? '#d46f5f' : '#dc9d56',
        height: village.id === selectedId
          ? 1650
          : 520 + village.disaster_impact * 540,
      },
      geometry: {
        type: 'Polygon',
        coordinates: circlePolygon(
          village.lng,
          village.lat,
          village.id === selectedId
            ? 2400
            : 1100 + village.disaster_impact * 900,
        ),
      },
    })),
  };
}

function buildIncidentPointData(villages, selectedId, focusActive = false) {
  return {
    type: 'FeatureCollection',
    features: villages.map((village) => ({
      type: 'Feature',
      properties: {
        id: village.id,
        name: village.name,
        impact: Math.round(village.disaster_impact * 100),
        selected: village.id === selectedId ? 1 : 0,
        dimmed: focusActive && village.id !== selectedId ? 1 : 0,
      },
      geometry: {
        type: 'Point',
        coordinates: [village.lng, village.lat],
      },
    })),
  };
}

function buildMissionGridData() {
  const features = [];
  features.push({
    type: 'Feature',
    properties: { boundary: 1 },
    geometry: {
      type: 'LineString',
      coordinates: [
        [79.25, 26.25],
        [88.5, 26.25],
        [88.5, 30.5],
        [79.25, 30.5],
        [79.25, 26.25],
      ],
    },
  });
  for (let lng = 79.5; lng <= 88.5; lng += 0.5) {
    features.push({
      type: 'Feature',
      properties: {},
      geometry: {
        type: 'LineString',
        coordinates: [[lng, 26.25], [lng, 30.5]],
      },
    });
  }
  for (let lat = 26.5; lat <= 30.5; lat += 0.5) {
    features.push({
      type: 'Feature',
      properties: {},
      geometry: {
        type: 'LineString',
        coordinates: [[79.25, lat], [88.5, lat]],
      },
    });
  }
  return { type: 'FeatureCollection', features };
}

function buildRouteData(routes, depot, villages, selectedId, focusActive = false) {
  const villagesById = Object.fromEntries(villages.map((village) => [village.id, village]));
  return {
    type: 'FeatureCollection',
    features: routes.filter(isFeasibleRoute).map((route) => ({
      type: 'Feature',
      properties: {
        route_id: route.vehicle_id,
        vehicle_id: route.vehicle_id,
        color: colorForVehicle(route.vehicle_id),
        relevant: !focusActive || route.stops?.includes(selectedId) ? 1 : 0,
        origin: depot?.name ?? 'Dispatch depot',
        destinations: routeDestinationNames(route, villagesById),
        eta: formatMinutes(
          focusActive && route.stops?.includes(selectedId)
            ? stopEta(route, selectedId)
            : route.total_time_minutes,
        ),
        distance: `${Number(route.total_distance_km ?? 0).toFixed(1)} km`,
        payload: formatPayload(route.cargo_manifest),
      },
      geometry: {
        type: 'LineString',
        coordinates: routeCoordinates(route, depot, villagesById),
      },
    })).filter((feature) => feature.geometry.coordinates.length > 1),
  };
}

function buildRouteExceptionData(routes, depot, villages, selectedId, focusActive = false) {
  const villagesById = Object.fromEntries(villages.map((village) => [village.id, village]));
  return {
    type: 'FeatureCollection',
    features: routes.filter((route) => !isFeasibleRoute(route)).map((route) => ({
      type: 'Feature',
      properties: {
        route_id: route.vehicle_id,
        vehicle_id: route.vehicle_id,
        relevant: !focusActive || route.stops?.includes(selectedId) ? 1 : 0,
        destinations: routeDestinationNames(route, villagesById),
        reason: route.infeasibility_reason ?? 'Route failed feasibility validation.',
      },
      geometry: {
        type: 'LineString',
        coordinates: routeCoordinates(route, depot, villagesById),
      },
    })).filter((feature) => feature.geometry.coordinates.length > 1),
  };
}

function buildRoadNetworkData(roadNetwork = [], blockedOnly = false, routes = []) {
  return {
    type: 'FeatureCollection',
    features: roadNetwork
      .filter((edge) => !blockedOnly || edge.status === 'blocked')
      .map((edge) => {
        const affectedRoutes = routes.filter((route) =>
          (route.rerouted_due_to ?? []).includes(edge.edge_id));
        return {
          type: 'Feature',
          properties: {
            edge_id: edge.edge_id,
            name: edge.name,
            status: edge.status,
            quality: edge.road_quality,
            distance: edge.distance_km,
            affected_assets: affectedRoutes.length,
            affected_vehicle_ids: affectedRoutes.map((route) => route.vehicle_id).join(', '),
          },
          geometry: {
            type: 'LineString',
            coordinates: edge.geometry ?? [],
          },
        };
      })
      .filter((feature) => feature.geometry.coordinates.length > 1),
  };
}

export default function TerrainMissionMap({
  villages,
  depot,
  routes,
  selectedId,
  onSelect,
  addMode,
  draftIncident,
  onDraftIncident,
  roadNetwork = [],
  onOpenDisruption,
  elapsedMinutes = 0,
  dispatchActive = false,
}) {
  // Shown on the terrain badge. Mission time is the operator's, and it is pinned
  // at zero while the fleet is held, so the badge can never imply movement on an
  // unauthorized plan.
  const missionElapsedSafe = dispatchActive
    ? Math.max(0, Number(elapsedMinutes) || 0)
    : 0;
  const containerRef = useRef(null);
  const updatePositionsRef = useRef(null);
  const mapRef = useRef(null);
  const markersRef = useRef([]);
  const incidentLabelsRef = useRef([]);
  const animationRef = useRef(null);
  const draftMarkerRef = useRef(null);
  const routePopupRef = useRef(null);
  const simulationClockRef = useRef({ signature: '', startedAtMs: 0 });
  const visibilityHandlerRef = useRef(null);
  const hoverKeyRef = useRef('');
  const terrainFallbackRef = useRef(false);
  const previousSelectedIdRef = useRef(selectedId);
  const [ready, setReady] = useState(false);
  const [terrainStatus, setTerrainStatus] = useState('loading');
  const [focusActive, setFocusActive] = useState(false);
  const [cameraMode, setCameraMode] = useState('terrain');
  const [cameraPitch, setCameraPitch] = useState(56);
  const [hoverDetail, setHoverDetail] = useState(null);
  const dataRef = useRef({
    villages,
    depot,
    routes,
    selectedId,
    focusActive,
    roadNetwork,
  });
  const interactionRef = useRef({ addMode, onDraftIncident });

  dataRef.current = {
    villages,
    depot,
    routes,
    selectedId,
    focusActive,
    roadNetwork,
    elapsedMinutes,
    dispatchActive,
  };
  interactionRef.current = { addMode, onDraftIncident };

  // Recompute fleet positions whenever the operator's mission clock moves or
  // the authorization state changes. This is the only driver of vehicle motion.
  useEffect(() => {
    updatePositionsRef.current?.();
  }, [elapsedMinutes, dispatchActive, routes]);

  const villagesById = Object.fromEntries(villages.map((village) => [village.id, village]));
  const selectedVillage = villagesById[selectedId];
  const selectedRoutes = focusActive
    ? routes
      .filter((route) =>
        isFeasibleRoute(route) && route.stops?.includes(selectedId))
      .sort((a, b) => stopEta(a, selectedId) - stopEta(b, selectedId))
    : [];
  const primaryRoute = selectedRoutes[0];
  const blockedRoads = roadNetwork.filter((edge) => edge.status === 'blocked');

  useEffect(() => {
    if (!ready || !containerRef.current || !mapRef.current) return undefined;
    const map = mapRef.current;
    const resize = () => map.resize();
    const frame = window.requestAnimationFrame(resize);
    const observer = new ResizeObserver(resize);
    observer.observe(containerRef.current);
    return () => {
      window.cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [ready]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current || !depot || !villages.length) return undefined;

    const map = new maplibregl.Map({
      container: containerRef.current,
      center: [84.05, 28.15],
      zoom: 6.3,
      pitch: 56,
      bearing: -8,
      maxPitch: 82,
      cooperativeGestures: true,
      pixelRatio: Math.min(window.devicePixelRatio || 1, 1.25),
      canvasContextAttributes: {
        antialias: false,
      },
      style: {
        version: 8,
        sources: {
          offlineRelief: {
            type: 'image',
            url: '/assets/nepal-relief-fallback.webp',
            coordinates: [
              [79.25, 30.5],
              [88.5, 30.5],
              [88.5, 26.25],
              [79.25, 26.25],
            ],
          },
          osm: {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution: '&copy; OpenStreetMap contributors',
          },
          terrainSource: {
            type: 'raster-dem',
            tiles: ['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'],
            tileSize: 256,
            encoding: 'terrarium',
            maxzoom: 15,
          },
        },
        terrain: {
          source: 'terrainSource',
          exaggeration: 1.65,
        },
        layers: [
          {
            id: 'background',
            type: 'background',
            paint: { 'background-color': '#18130e' },
          },
          {
            id: 'offline-relief',
            type: 'raster',
            source: 'offlineRelief',
            paint: {
              'raster-opacity': 0.52,
              'raster-saturation': -0.35,
              'raster-contrast': 0.08,
            },
          },
          {
            id: 'osm-base',
            type: 'raster',
            source: 'osm',
            paint: {
              'raster-saturation': -0.82,
              'raster-contrast': 0.16,
              'raster-brightness-min': 0.12,
              'raster-brightness-max': 0.58,
            },
          },
          {
            id: 'terrain-shade',
            type: 'hillshade',
            source: 'terrainSource',
            paint: {
              'hillshade-shadow-color': '#0f0c09',
              'hillshade-highlight-color': '#d7c9bc',
              'hillshade-accent-color': '#7b6049',
              'hillshade-exaggeration': 0.48,
            },
          },
        ],
      },
    });
    mapRef.current = map;
    const activateSchematicFallback = () => {
      terrainFallbackRef.current = true;
      setTerrainStatus('fallback');
      if (!map.isStyleLoaded()) return;
      try {
        map.setTerrain(null);
        if (map.getLayer('terrain-shade')) {
          map.setLayoutProperty('terrain-shade', 'visibility', 'none');
        }
        if (map.getLayer('osm-base')) {
          map.setPaintProperty('osm-base', 'raster-opacity', 0.48);
          map.setPaintProperty('osm-base', 'raster-saturation', -0.78);
        }
        if (map.getLayer('offline-relief')) {
          map.setPaintProperty('offline-relief', 'raster-opacity', 0.9);
        }
        if (map.getLayer('mission-grid-lines')) {
          map.setPaintProperty('mission-grid-lines', 'line-opacity', 0.42);
        }
        map.fitBounds(
          [[79.25, 26.25], [88.5, 30.5]],
          {
            padding: { top: 26, right: 26, bottom: 26, left: 26 },
            pitch: 0,
            bearing: 0,
            duration: motionDuration(420),
            essential: false,
          },
        );
      } catch {
        // The status label remains authoritative if a provider fails mid-load.
      }
    };
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right');
    map.addControl(new maplibregl.ScaleControl({ unit: 'metric', maxWidth: 110 }), 'bottom-left');
    map.addControl(
      new maplibregl.TerrainControl({
        source: 'terrainSource',
        exaggeration: 1.65,
      }),
      'top-right',
    );
    map.on('movestart', (event) => {
      if (event.originalEvent) setCameraMode('custom');
    });

    map.on('click', (event) => {
      if (interactionRef.current.addMode) {
        interactionRef.current.onDraftIncident({
          lat: event.lngLat.lat,
          lng: event.lngLat.lng,
        });
      }
    });

    map.on('error', (event) => {
      const message = String(event?.error?.message ?? '').toLowerCase();
      if (
        message.includes('terrain')
        || event?.sourceId === 'terrainSource'
        || event?.source?.id === 'terrainSource'
      ) {
        activateSchematicFallback();
      }
    });

    map.once('style.load', () => {
      const current = dataRef.current;
      const publishHover = (detail) => {
        const key = detail ? `${detail.kind}:${detail.title}:${detail.detail}` : '';
        if (hoverKeyRef.current === key) return;
        hoverKeyRef.current = key;
        setHoverDetail(detail);
      };
      const routeData = buildRouteData(
        current.routes,
        current.depot,
        current.villages,
        current.selectedId,
        current.focusActive,
      );
      map.addSource('incidents-3d', {
        type: 'geojson',
        data: buildIncidentData(
          current.villages,
          current.selectedId,
          current.focusActive,
        ),
      });
      map.addSource('incident-points', {
        type: 'geojson',
        data: buildIncidentPointData(
          current.villages,
          current.selectedId,
          current.focusActive,
        ),
      });
      map.addSource('mission-grid', {
        type: 'geojson',
        data: buildMissionGridData(),
      });
      map.addSource('road-network', {
        type: 'geojson',
        data: buildRoadNetworkData(current.roadNetwork, false, current.routes),
      });
      map.addSource('road-blocks', {
        type: 'geojson',
        data: buildRoadNetworkData(current.roadNetwork, true, current.routes),
      });
      map.addSource('routes', {
        type: 'geojson',
        data: routeData,
        lineMetrics: true,
      });
      map.addSource('route-exceptions', {
        type: 'geojson',
        data: buildRouteExceptionData(
          current.routes,
          current.depot,
          current.villages,
          current.selectedId,
          current.focusActive,
        ),
      });
      map.addSource('focus-routes', {
        type: 'geojson',
        data: EMPTY_COLLECTION,
      });
      map.addSource('moving-vehicles', {
        type: 'geojson',
        data: EMPTY_COLLECTION,
      });
      map.addImage('fleet-helicopter', createVehicleSprite('helicopter'), {
        pixelRatio: 2,
      });
      map.addImage('fleet-truck', createVehicleSprite('truck'), {
        pixelRatio: 2,
      });
      loadGeneratedFleetSprite(
        map,
        'fleet-helicopter',
        '/assets/fleet/helicopter-topdown-v2.webp',
      );
      loadGeneratedFleetSprite(
        map,
        'fleet-truck',
        '/assets/fleet/truck-topdown-v2.webp',
      );

      map.addLayer({
        id: 'mission-grid-lines',
        type: 'line',
        source: 'mission-grid',
        paint: {
          'line-color': '#977659',
          'line-width': 1,
          'line-opacity': 0.2,
          'line-dasharray': [2, 5],
        },
      });
      map.addLayer({
        id: 'incident-columns',
        type: 'fill-extrusion',
        source: 'incidents-3d',
        paint: {
          'fill-extrusion-color': ['get', 'color'],
          'fill-extrusion-height': ['get', 'height'],
          'fill-extrusion-base': 0,
          'fill-extrusion-opacity': [
            'case',
            ['==', ['get', 'dimmed'], 1],
            0.08,
            ['==', ['get', 'selected'], 1],
            0.8,
            0.46,
          ],
          'fill-extrusion-vertical-gradient': true,
        },
      });
      map.addLayer({
        id: 'incident-halo',
        type: 'circle',
        source: 'incident-points',
        paint: {
          'circle-radius': [
            'case',
            ['==', ['get', 'selected'], 1],
            18,
            10,
          ],
          'circle-color': [
            'case',
            ['>=', ['get', 'impact'], 70],
            '#d57061',
            '#dc9b52',
          ],
          'circle-opacity': [
            'case',
            ['==', ['get', 'dimmed'], 1],
            0.08,
            0.2,
          ],
          'circle-blur': 0.3,
          'circle-stroke-width': [
            'case',
            ['==', ['get', 'selected'], 1],
            2,
            1,
          ],
          'circle-stroke-color': '#eee7e2',
          'circle-stroke-opacity': [
            'case',
            ['==', ['get', 'dimmed'], 1],
            0.1,
            0.9,
          ],
        },
      });
      map.addLayer({
        id: 'incident-hit-area',
        type: 'circle',
        source: 'incident-points',
        paint: {
          'circle-radius': 24,
          'circle-color': '#000103',
          'circle-opacity': 0,
        },
      });
      map.addLayer({
        id: 'road-network-context',
        type: 'line',
        source: 'road-network',
        layout: {
          'line-cap': 'round',
          'line-join': 'round',
        },
        paint: {
          'line-color': '#99785a',
          'line-width': 1.35,
          'line-opacity': 0.34,
          'line-dasharray': [1, 2.5],
        },
      });
      map.addLayer({
        id: 'active-road-blocks',
        type: 'line',
        source: 'road-blocks',
        layout: {
          'line-cap': 'round',
          'line-join': 'round',
        },
        paint: {
          'line-color': '#d67464',
          'line-width': 7,
          'line-opacity': 0.9,
          'line-dasharray': [0.7, 1.1],
        },
      });
      map.addLayer({
        id: 'route-shadow',
        type: 'line',
        source: 'routes',
        layout: {
          'line-cap': 'round',
          'line-join': 'round',
        },
        paint: {
          'line-color': ['get', 'color'],
          'line-width': 5,
          'line-opacity': 0.1,
          'line-blur': 4,
        },
      });
      map.addLayer({
        id: 'active-routes',
        type: 'line',
        source: 'routes',
        layout: {
          'line-cap': 'round',
          'line-join': 'round',
        },
        paint: {
          'line-color': ['get', 'color'],
          'line-width': 2.5,
          'line-opacity': 0.78,
        },
      });
      map.addLayer({
        id: 'focus-route-shadow',
        type: 'line',
        source: 'focus-routes',
        layout: {
          'line-cap': 'round',
          'line-join': 'round',
        },
        paint: {
          'line-color': ['get', 'color'],
          'line-width': 10,
          'line-opacity': 0.22,
          'line-blur': 4,
        },
      });
      map.addLayer({
        id: 'focus-routes',
        type: 'line',
        source: 'focus-routes',
        layout: {
          'line-cap': 'round',
          'line-join': 'round',
        },
        paint: {
          'line-color': ['get', 'color'],
          'line-width': 4.5,
          'line-opacity': 0.96,
        },
      });
      // Route exceptions must paint ABOVE the assigned routes. Adding them
      // before the route layers hid them exactly where they overlap, which is
      // the only place the distinction between "assigned" and "excluded from
      // dispatch" carries any information.
      map.addLayer({
        id: 'route-exceptions',
        type: 'line',
        source: 'route-exceptions',
        layout: {
          'line-cap': 'butt',
          'line-join': 'round',
        },
        paint: {
          'line-color': '#d57060',
          'line-width': 3,
          'line-opacity': [
            'case',
            ['==', ['get', 'relevant'], 1],
            0.9,
            0.18,
          ],
          'line-dasharray': [1, 1.5],
        },
      });
      map.addLayer({
        id: 'vehicle-halo',
        type: 'circle',
        source: 'moving-vehicles',
        paint: {
          'circle-radius': 11,
          'circle-color': ['get', 'color'],
          'circle-opacity': [
            'case',
            ['==', ['get', 'relevant'], true],
            0.24,
            0.01,
          ],
          'circle-blur': 0.5,
        },
      });
      map.addLayer({
        id: 'vehicle-core',
        type: 'circle',
        source: 'moving-vehicles',
        paint: {
          'circle-radius': 5,
          'circle-color': '#eae2dc',
          'circle-stroke-color': ['get', 'color'],
          'circle-stroke-width': 3,
          'circle-opacity': [
            'case',
            ['==', ['get', 'relevant'], true],
            1,
            0.02,
          ],
        },
      });
      map.addLayer({
        id: 'vehicle-symbols',
        type: 'symbol',
        source: 'moving-vehicles',
        layout: {
          'icon-image': ['get', 'icon'],
          'icon-size': 0.92,
          'icon-rotate': ['get', 'heading'],
          'icon-rotation-alignment': 'map',
          'icon-pitch-alignment': 'viewport',
          'icon-allow-overlap': true,
          'icon-ignore-placement': true,
        },
        paint: {
          'icon-opacity': [
            'case',
            ['==', ['get', 'relevant'], true],
            1,
            0.015,
          ],
        },
      });
      if (ENABLE_LEGACY_THREE_LAYER) {
        map.addLayer(createMissionThreeLayer(
          current.villages,
          dataRef,
          simulationClockRef,
        ));
      }
      map.on('mouseenter', 'incident-hit-area', () => {
        map.getCanvas().style.cursor = 'pointer';
      });
      map.on('mousemove', 'incident-hit-area', (event) => {
        const villageId = event.features?.[0]?.properties?.id;
        const village = current.villages.find((item) => item.id === villageId);
        if (!village) return;
        publishHover({
          kind: 'Incident',
          title: village.name,
          detail: `${Math.round(Number(village.disaster_impact ?? 0) * 100)}% reported impact · ${Number(village.population ?? 0).toLocaleString()} people`,
        });
      });
      map.on('mouseleave', 'incident-hit-area', () => {
        map.getCanvas().style.cursor = interactionRef.current.addMode ? 'crosshair' : '';
        publishHover(null);
      });
      map.on('click', 'incident-hit-area', (event) => {
        if (interactionRef.current.addMode) return;
        const villageId = event.features?.[0]?.properties?.id;
        if (!villageId) return;
        setFocusActive(true);
        setCameraMode('incident');
        onSelect(villageId);
        const village = current.villages.find((item) => item.id === villageId);
        frameIncidentCorridor(
          map,
          current.depot,
          village,
          terrainFallbackRef.current,
        );
      });

      ['active-routes', 'focus-routes'].forEach((layerId) => {
        map.on('mouseenter', layerId, () => {
          map.getCanvas().style.cursor = 'pointer';
        });
        map.on('mousemove', layerId, (event) => {
          const properties = event.features?.[0]?.properties;
          if (!properties || !isRelevantProperty(properties.relevant)) return;
          publishHover({
            kind: 'Assigned route',
            title: properties.vehicle_id ?? 'Route',
            detail: `${properties.destinations ?? 'Assigned corridor'} · ${properties.eta ?? 'ETA unavailable'} · ${properties.distance ?? 'distance unavailable'}`,
          });
        });
        map.on('mouseleave', layerId, () => {
          map.getCanvas().style.cursor = interactionRef.current.addMode ? 'crosshair' : '';
          publishHover(null);
        });
        map.on('click', layerId, (event) => {
          if (interactionRef.current.addMode) return;
          const feature = event.features?.[0];
          if (!feature?.properties || !isRelevantProperty(feature.properties.relevant)) return;
          routePopupRef.current?.remove();
          routePopupRef.current = new maplibregl.Popup({
            closeButton: true,
            closeOnClick: true,
            maxWidth: '320px',
            offset: 12,
          })
            .setLngLat(event.lngLat)
            .setDOMContent(createRoutePopupNode(feature.properties))
            .addTo(map);
        });
      });
      map.on('mouseenter', 'route-exceptions', () => {
        map.getCanvas().style.cursor = 'help';
      });
      map.on('mousemove', 'route-exceptions', (event) => {
        const properties = event.features?.[0]?.properties;
        if (!properties || !isRelevantProperty(properties.relevant)) return;
        publishHover({
          kind: 'Route exception · not dispatched',
          title: properties.vehicle_id ?? 'Infeasible route',
          detail: `${properties.destinations ?? 'Unresolved corridor'} · ${properties.reason ?? 'Feasibility validation failed'}`,
        });
      });
      map.on('mouseleave', 'route-exceptions', () => {
        map.getCanvas().style.cursor = interactionRef.current.addMode ? 'crosshair' : '';
        publishHover(null);
      });
      map.on('mouseenter', 'vehicle-symbols', () => {
        map.getCanvas().style.cursor = 'pointer';
      });
      map.on('mousemove', 'vehicle-symbols', (event) => {
        const properties = event.features?.[0]?.properties;
        if (!properties || !isRelevantProperty(properties.relevant)) return;
        publishHover({
          kind: 'Fleet asset',
          title: properties.vehicle_id ?? 'Vehicle',
          detail: `${properties.destinations ?? 'Assigned route'} · ${formatMinutes(properties.eta_minutes)} remaining · ${properties.payload ?? 'manifest unavailable'}`,
        });
      });
      map.on('mouseleave', 'vehicle-symbols', () => {
        map.getCanvas().style.cursor = interactionRef.current.addMode ? 'crosshair' : '';
        publishHover(null);
      });
      map.on('click', 'vehicle-symbols', (event) => {
        if (interactionRef.current.addMode) return;
        const properties = event.features?.[0]?.properties;
        if (!properties || !isRelevantProperty(properties.relevant)) return;
        routePopupRef.current?.remove();
        routePopupRef.current = new maplibregl.Popup({
          closeButton: true,
          closeOnClick: true,
          maxWidth: '320px',
          offset: 18,
        })
          .setLngLat(event.lngLat)
          .setDOMContent(createRoutePopupNode({
            ...properties,
            eta: formatMinutes(properties.eta_minutes),
            distance: 'Moving on assigned route',
          }))
          .addTo(map);
      });
      map.on('mouseenter', 'active-road-blocks', () => {
        map.getCanvas().style.cursor = 'pointer';
      });
      map.on('mousemove', 'active-road-blocks', (event) => {
        const properties = event.features?.[0]?.properties;
        if (!properties) return;
        publishHover({
          kind: 'Blocked corridor',
          title: properties.name ?? properties.id ?? 'Road closure',
          detail: `${properties.quality ?? 'road'} · ${Number(properties.distance ?? 0).toFixed(1)} km · ${Number(properties.affected_assets ?? 0)} affected asset${Number(properties.affected_assets ?? 0) === 1 ? '' : 's'}${properties.affected_vehicle_ids ? ` (${properties.affected_vehicle_ids})` : ''} · inspect the road event workspace for the child run`,
        });
      });
      map.on('mouseleave', 'active-road-blocks', () => {
        map.getCanvas().style.cursor = interactionRef.current.addMode ? 'crosshair' : '';
        publishHover(null);
      });

      const depotElement = document.createElement('div');
      depotElement.className = 'terrain-depot-marker';
      depotElement.title = current.depot.name;
      depotElement.setAttribute('role', 'img');
      depotElement.setAttribute('aria-label', `Central depot, ${current.depot.name}`);
      depotElement.innerHTML = '<span class="terrain-marker-glyph" aria-hidden="true">D</span><b>Central depot</b>';
      markersRef.current.push(
        new maplibregl.Marker({ element: depotElement, anchor: 'bottom' })
          .setLngLat([current.depot.lng, current.depot.lat])
          .addTo(map),
      );
      current.villages.forEach((village) => {
        const markerElement = document.createElement('button');
        markerElement.type = 'button';
        markerElement.className = 'terrain-incident-label';
        markerElement.dataset.villageId = village.id;
        markerElement.title = `Focus ${village.name}`;
        markerElement.setAttribute(
          'aria-label',
          `Focus ${village.name}, ${Math.round(village.disaster_impact * 100)} percent impact`,
        );
        markerElement.style.cssText = [
          'min-width:0',
          'max-width:118px',
          'padding:4px 7px',
          'white-space:nowrap',
          'transition:opacity 160ms ease-out,transform 160ms ease-out',
        ].join(';');
        const name = document.createElement('b');
        name.textContent = village.name;
        const impact = document.createElement('span');
        impact.textContent = `· ${Math.round(village.disaster_impact * 100)}% impact`;
        markerElement.append(name, impact);
        markerElement.addEventListener('click', () => {
          setFocusActive(true);
          setCameraMode('incident');
          onSelect(village.id);
          frameIncidentCorridor(
            map,
            current.depot,
            village,
            terrainFallbackRef.current,
          );
        });
        const marker = new maplibregl.Marker({
          element: markerElement,
          anchor: 'bottom',
          offset: [0, -18],
        })
          .setLngLat([village.lng, village.lat])
          .addTo(map);
        markersRef.current.push(marker);
        incidentLabelsRef.current.push({
          village,
          element: markerElement,
        });
      });
      const syncLabels = () => {
        const latest = dataRef.current;
        setCameraPitch(Math.round(map.getPitch()));
        syncIncidentLabelVisibility(
          incidentLabelsRef.current,
          latest.selectedId,
          latest.focusActive,
        );
      };
      map.on('moveend', syncLabels);
      map.on('zoomend', syncLabels);
      syncLabels();

      if (terrainFallbackRef.current) {
        activateSchematicFallback();
      } else {
        setTerrainStatus('active');
      }
      setReady(true);

      const updateVehiclePositions = () => {
        const timestamp = performance.now();
        const latest = dataRef.current;
        const entries = routeEntriesFromData(latest);
        const clock = syncSimulationClock(simulationClockRef, entries, timestamp);
        // Mission time is controlled by the operator, not by wall-clock playback.
        // Until a plan is authorized, the fleet is held at the depot at t=0.
        const missionElapsed = latest.dispatchActive
          ? Math.max(0, Number(latest.elapsedMinutes) || 0)
          : 0;
        const vehicleFeatures = entries.map((entry) => {
          const elapsedMinutes = missionElapsed;
          const durationMinutes = Number(entry.route.total_time_minutes);
          const progress = Number.isFinite(durationMinutes) && durationMinutes > 0
            ? Math.min(elapsedMinutes / durationMinutes, 1)
            : 1;
          return {
          type: 'Feature',
          properties: {
            vehicle_id: entry.vehicleId,
            color: entry.color,
            progress,
            relevant: entry.relevant,
            transport_mode: entry.route.transport_mode,
            held: !latest.dispatchActive,
            eta_minutes: Number.isFinite(durationMinutes)
              ? Math.max(0, durationMinutes - elapsedMinutes)
              : 0,
            origin: latest.depot?.name ?? 'Dispatch depot',
            destinations: routeDestinationNames(
              entry.route,
              Object.fromEntries(latest.villages.map((village) => [village.id, village])),
            ),
            payload: formatPayload(entry.route.cargo_manifest),
            icon: entry.route.transport_mode === 'air' ? 'fleet-helicopter' : 'fleet-truck',
            heading: mapBearingFromHeading(
              routeHeadingAtElapsed(entry.route, entry.coordinates, elapsedMinutes),
            ),
          },
          geometry: {
            type: 'Point',
            coordinates: routePositionAtElapsed(
              entry.route,
              entry.coordinates,
              elapsedMinutes,
            ),
          },
        };
        });
        map.getSource('moving-vehicles')?.setData({
          type: 'FeatureCollection',
          features: vehicleFeatures,
        });
        const scene = containerRef.current?.closest('.terrain-scene');
        if (scene) {
          scene.dataset.vehicleCount = String(vehicleFeatures.length);
          scene.dataset.airVehicleCount = String(
            vehicleFeatures.filter((feature) => feature.properties.transport_mode === 'air').length,
          );
          scene.dataset.roadVehicleCount = String(
            vehicleFeatures.filter((feature) => feature.properties.transport_mode === 'road').length,
          );
          scene.dataset.motionTick = String(Date.now());
          scene.dataset.motionTimeline = 'solver-stop-eta';
          scene.dataset.missionElapsedMinutes = missionElapsed.toFixed(2);
          scene.dataset.dispatchActive = String(Boolean(latest.dispatchActive));
          scene.dataset.timelineCheckpointCount = String(
            clock.timelineAudit?.checkpointCount ?? 0,
          );
          scene.dataset.timelineCheckpointMaxError = Number(
            clock.timelineAudit?.maximumError ?? Number.POSITIVE_INFINITY,
          ).toFixed(6);
          scene.dataset.leadVehiclePosition = vehicleFeatures[0]?.geometry?.coordinates?.join(',') ?? '';
        }
      };
      // The operator's mission clock is the only time source. Positions are
      // recomputed when that clock changes, never on a free-running interval,
      // so scrubbing and authorization gating are exact rather than approximate.
      updatePositionsRef.current = updateVehiclePositions;
      updateVehiclePositions();
    });

    return () => {
      if (animationRef.current) window.clearInterval(animationRef.current);
      animationRef.current = null;
      if (visibilityHandlerRef.current) {
        document.removeEventListener('visibilitychange', visibilityHandlerRef.current);
        visibilityHandlerRef.current = null;
      }
      markersRef.current.forEach((marker) => marker.remove());
      draftMarkerRef.current?.remove();
      routePopupRef.current?.remove();
      routePopupRef.current = null;
      draftMarkerRef.current = null;
      markersRef.current = [];
      incidentLabelsRef.current = [];
      map.remove();
      mapRef.current = null;
    };
  }, [depot, villages, onSelect]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map?.getSource('routes')) return;
    const incidentSource = map.getSource('incidents-3d');
    incidentSource?.setData(buildIncidentData(villages, selectedId, focusActive));
    map.getSource('incident-points')?.setData(
      buildIncidentPointData(villages, selectedId, focusActive),
    );
    const visibleRoutes = focusActive
      ? routes.filter((route) =>
        isFeasibleRoute(route) && route.stops?.includes(selectedId))
      : routes.filter(isFeasibleRoute);
    const visibleExceptions = focusActive
      ? routes.filter((route) =>
        !isFeasibleRoute(route) && route.stops?.includes(selectedId))
      : routes.filter((route) => !isFeasibleRoute(route));
    map.getSource('routes')?.setData(
      buildRouteData(visibleRoutes, depot, villages, selectedId, focusActive),
    );
    map.getSource('focus-routes')?.setData(
      focusActive
        ? buildRouteData(visibleRoutes, depot, villages, selectedId, false)
        : EMPTY_COLLECTION,
    );
    map.getSource('route-exceptions')?.setData(
      buildRouteExceptionData(
        visibleExceptions,
        depot,
        villages,
        selectedId,
        focusActive,
      ),
    );
    map.getSource('road-network')?.setData(buildRoadNetworkData(roadNetwork, false, routes));
    map.getSource('road-blocks')?.setData(buildRoadNetworkData(roadNetwork, true, routes));
    if (map.getLayer('route-shadow')) {
      map.setLayoutProperty('route-shadow', 'visibility', focusActive ? 'none' : 'visible');
    }
    if (map.getLayer('active-routes')) {
      map.setLayoutProperty('active-routes', 'visibility', focusActive ? 'none' : 'visible');
    }
    const relevantOnly = focusActive
      ? ['==', ['get', 'relevant'], true]
      : null;
    ['vehicle-halo', 'vehicle-core', 'vehicle-symbols'].forEach(
      (layerId) => {
        if (map.getLayer(layerId)) map.setFilter(layerId, relevantOnly);
      },
    );
    syncIncidentLabelVisibility(
      incidentLabelsRef.current,
      selectedId,
      focusActive,
    );
  }, [selectedId, villages, routes, depot, focusActive, roadNetwork]);

  useEffect(() => {
    if (!ready || !selectedId || previousSelectedIdRef.current === selectedId) return;
    previousSelectedIdRef.current = selectedId;
    const village = villages.find((item) => item.id === selectedId);
    if (!village) return;
    setFocusActive(true);
    setCameraMode('incident');
    frameIncidentCorridor(
      mapRef.current,
      depot,
      village,
      terrainFallbackRef.current,
    );
  }, [ready, selectedId, villages, depot]);

  useEffect(() => {
    if (!mapRef.current) return;
    simulationClockRef.current = { signature: '', startedAtMs: performance.now() };
  }, [routes, depot, villages, ready]);

  useEffect(() => {
    const map = mapRef.current;
    draftMarkerRef.current?.remove();
    draftMarkerRef.current = null;
    if (!map || !draftIncident) return;

    const marker = document.createElement('div');
    marker.className = 'terrain-draft-marker';
    marker.innerHTML = '<span class="terrain-marker-glyph" aria-hidden="true">+</span><b>Draft incident</b>';
    draftMarkerRef.current = new maplibregl.Marker({ element: marker, anchor: 'bottom' })
      .setLngLat([draftIncident.lng, draftIncident.lat])
      .addTo(map);
  }, [draftIncident, ready]);

  const setCamera = (mode) => {
    const map = mapRef.current;
    if (!map) return;
    setCameraMode(mode);
    if (terrainFallbackRef.current) {
      if (mode === 'fleet') {
        map.easeTo({ center: [84.2, 28.1], zoom: 7.35, pitch: 0, bearing: 0, duration: motionDuration(420) });
      } else {
        map.fitBounds(
          [[79.25, 26.25], [88.5, 30.5]],
          {
            padding: { top: 26, right: 26, bottom: 26, left: 26 },
            pitch: 0,
            bearing: 0,
            duration: motionDuration(420),
            essential: false,
          },
        );
      }
      return;
    }
    if (mode === 'terrain') {
      map.easeTo({ center: [84.05, 28.15], zoom: 6.3, pitch: 56, bearing: -8, duration: motionDuration(420) });
    } else if (mode === 'fleet') {
      map.easeTo({ center: [84.2, 28.1], zoom: 7.35, pitch: 67, bearing: -18, duration: motionDuration(420) });
    } else {
      map.easeTo({ center: [84.05, 28.15], zoom: 6.25, pitch: 0, bearing: 0, duration: motionDuration(420) });
    }
  };

  const resetOverview = () => {
    setFocusActive(false);
    routePopupRef.current?.remove();
    routePopupRef.current = null;
    setCamera('terrain');
  };

  return (
    <div
      className={`terrain-scene ${addMode ? 'placing-incident' : ''}`}
      data-route-count={routes.filter(isFeasibleRoute).length}
      data-infeasible-route-count={routes.filter((route) => !isFeasibleRoute(route)).length}
      data-visible-route-count={focusActive
        ? selectedRoutes.length
        : routes.filter(isFeasibleRoute).length}
      data-mission-driven="operator-clock"
      data-focus-active={focusActive ? 'true' : 'false'}
      data-camera-mode={cameraMode}
      data-camera-pitch={cameraPitch}
      data-terrain-status={terrainStatus}
      data-fleet-renderer="webp-symbol"
    >
      <div
        ref={containerRef}
        className="terrain-map-canvas"
        aria-label="Interactive Nepal incident, route, fleet, and road-closure map"
      />
      {hoverDetail && (
        <div className="terrain-hover-card" role="status">
          <small>{hoverDetail.kind}</small>
          <b>{hoverDetail.title}</b>
          <span>{hoverDetail.detail}</span>
        </div>
      )}
      {!ready && (
        <div className="terrain-loading">
          <Mountain aria-hidden="true" />
          <b>Building terrain model</b>
          <small>Loading elevation and operational layers</small>
        </div>
      )}
      <div className="terrain-status">
        <span className={`terrain-status-dot ${terrainStatus}`} />
        <div>
          <b>{terrainStatus === 'active' ? '3D terrain active' : terrainStatus === 'fallback' ? '3D fallback active' : 'Terrain loading'}</b>
          <small>
            {terrainStatus === 'fallback'
              ? 'Schematic grid / flat relief / ETA-synced fleet'
              : `Elevation x1.65 / ${blockedRoads.length} road closure${blockedRoads.length === 1 ? '' : 's'} / ${
                  dispatchActive
                    ? `T+${String(Math.floor(missionElapsedSafe / 60)).padStart(2, '0')}:${String(Math.round(missionElapsedSafe % 60)).padStart(2, '0')} mission time`
                    : 'not dispatched'
                }`}
          </small>
        </div>
      </div>
      {focusActive && selectedVillage && (
        <section
          className="terrain-focus-card"
          aria-label={`${selectedVillage.name} inbound response`}
          aria-live="polite"
        >
          <header>
            <div>
              <small className="terrain-focus-kicker">
                INCIDENT FOCUS · {Math.round(selectedVillage.disaster_impact * 100)}% IMPACT
              </small>
              <strong>
                {selectedVillage.name} · {selectedRoutes.length} inbound
              </strong>
            </div>
            <button
              className="terrain-overview-button"
              type="button"
              onClick={resetOverview}
              title="Return to national overview"
            >
              <Maximize2 size={16} aria-hidden="true" />
              Overview
            </button>
          </header>
          {primaryRoute ? (
            <p className="terrain-focus-compact-route">
              {primaryRoute.vehicle_id} · {formatMinutes(stopEta(primaryRoute, selectedId))} · {Number(primaryRoute.total_distance_km ?? 0).toFixed(1)} km
            </p>
          ) : (
            <p className="terrain-focus-empty">No feasible inbound route in this snapshot.</p>
          )}
        </section>
      )}
      <div className="camera-presets">
        <button type="button" className={cameraMode === 'terrain' ? 'active' : ''} aria-pressed={cameraMode === 'terrain'} onClick={() => setCamera('terrain')} aria-label="Show terrain perspective">
          <Mountain size={16} aria-hidden="true" />
          <span>Perspective</span>
        </button>
        <button type="button" className={cameraMode === 'fleet' ? 'active' : ''} aria-pressed={cameraMode === 'fleet'} onClick={() => setCamera('fleet')} aria-label="Focus the moving fleet">
          <Truck size={16} aria-hidden="true" />
          <span>Fleet view</span>
        </button>
        <button type="button" className={cameraMode === 'top' ? 'active' : ''} aria-pressed={cameraMode === 'top'} onClick={() => setCamera('top')} aria-label="Show top-down map">
          <Map size={16} aria-hidden="true" />
          <span>Top down</span>
        </button>
        <button type="button" onClick={() => onOpenDisruption?.()} aria-label="Inspect or report a road disruption">
          <TriangleAlert size={16} aria-hidden="true" />
          <span>Road event</span>
        </button>
      </div>
      <div className="terrain-legend">
        <span><i className="beacon-critical" /> Critical incident</span>
        <span><i className="beacon-route" /> Active route</span>
        <span><i className="beacon-vehicle" /> Vehicle in transit</span>
        {routes.some((route) => !isFeasibleRoute(route)) && (
          <span><i className="beacon-route-exception" /> Route exception · not dispatched</span>
        )}
        {blockedRoads.length > 0 && <span><i className="beacon-closure" /> Blocked corridor</span>}
      </div>
    </div>
  );
}
