/* RakshyaNet deck runtime.
   Three jobs: fit the 1600x900 stage to any viewport, drive slide navigation,
   and build the three data-driven figures from real project values so nothing
   in this deck is a hand-drawn approximation. */

const P = {
  ivory: '#efeae0', sand: '#e5ded0', clay: '#d8cfbc', ink: '#2a1e17',
  muted: '#5f4d40', subtle: '#8a7867', terra: '#c2673b', ember: '#d07f3c',
  rust: '#8f4426', sage: '#7d8a6a', good: '#6e8c5a', bad: '#a8402c',
};

const svgNS = 'http://www.w3.org/2000/svg';
function el(name, attrs = {}, text) {
  const node = document.createElementNS(svgNS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  if (text != null) node.textContent = text;
  return node;
}

/* ── Figure: the branching tool-selection pipeline ────────────────────
   Not a straight line. Evidence reaches Gemma, Gemma reaches a decision
   point, and three declared functions hang off it — two always taken, the
   imagery check taken only on a condition Gemma judges for itself. The
   engine and the human sit past `run_optimization`, which is the only
   branch that leads to a plan, and the plan still stops at a person.
   viewBox is 1480 x 462.                                              */
function buildPipeline(g) {
  const defs = el('defs');
  [['tip', P.subtle], ['tipSage', P.sage], ['tipTerra', P.terra]].forEach(([id, colour]) => {
    const marker = el('marker', {
      id, viewBox: '0 0 10 10', refX: 8, refY: 5,
      markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse',
    });
    marker.appendChild(el('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: colour }));
    defs.appendChild(marker);
  });
  g.appendChild(defs);

  const at = (x, y, size, fill, txt, cls = '') => {
    const t = el('text', { x, y, 'font-size': size, fill }, txt);
    if (cls) t.setAttribute('class', cls);
    return t;
  };
  const box = (x, y, w, h, colour, sw, dashed) => {
    const r = el('rect', {
      x, y, width: w, height: h, rx: 13,
      fill: P.ivory, stroke: colour, 'stroke-width': sw,
    });
    if (dashed) r.setAttribute('stroke-dasharray', '9 7');
    return r;
  };
  const card = (delay) => {
    const node = el('g', { class: 'pop' });
    node.style.transitionDelay = `${delay}ms`;
    return node;
  };
  const link = (d, colour, delay, opts = {}) => {
    const p = el('path', {
      d, stroke: colour, 'stroke-width': opts.width || 1.9, fill: 'none',
      'marker-end': `url(#${opts.tip || 'tip'})`,
    });
    if (opts.back) p.setAttribute('marker-start', `url(#${opts.tip || 'tip'})`);
    if (opts.dash) {
      // A conditional edge is drawn, not wiped in, so the long dash pattern
      // survives — a dashed line that animates via stroke-dashoffset cannot
      // also carry its own dash rhythm.
      p.setAttribute('stroke-dasharray', opts.dash);
      p.setAttribute('class', 'fade');
    } else {
      p.setAttribute('class', 'draw');
      p.style.setProperty('--len', String(opts.len || 240));
    }
    p.style.transitionDelay = `${delay}ms`;
    return p;
  };
  const chip = (x, y, w, label, colour) => {
    const gg = el('g');
    gg.appendChild(el('rect', { x, y, width: w, height: 25, rx: 12.5, fill: colour }));
    gg.appendChild(el('text', {
      x: x + w / 2, y: y + 17.5, 'font-size': 12, fill: P.ivory,
      'text-anchor': 'middle', class: 'm', 'letter-spacing': '0.09em',
    }, label));
    return gg;
  };

  /* Framing copy, in the empty quadrants rather than under the diagram. */
  const intro = el('g', { class: 'fade' });
  intro.style.transitionDelay = '300ms';
  intro.appendChild(at(0, 22, 13, P.rust, 'NATIVE FUNCTION CALLING', 'm'));
  intro.appendChild(at(0, 60, 21, P.ink, 'Gemma chooses which of three'));
  intro.appendChild(at(0, 88, 21, P.ink, 'declared functions to call.'));
  g.appendChild(intro);

  const bound = el('g', { class: 'fade' });
  bound.style.transitionDelay = '2400ms';
  bound.appendChild(at(0, 372, 13, P.rust, 'THE BOUNDARY', 'm'));
  bound.appendChild(at(0, 408, 21, P.ink, 'It cannot allocate, route,'));
  bound.appendChild(at(0, 436, 21, P.ink, 'approve or dispatch.'));
  g.appendChild(bound);

  /* 1 · evidence */
  const ev = card(200);
  ev.appendChild(box(0, 170, 215, 120, P.subtle, 1.7));
  ev.appendChild(at(20, 200, 12, P.subtle, 'EVIDENCE', 'm'));
  ev.appendChild(at(20, 234, 17, P.ink, 'three field reports'));
  ev.appendChild(at(20, 258, 17, P.ink, 'that disagree'));
  g.appendChild(ev);
  g.appendChild(link('M 222 230 L 252 230', P.subtle, 380, { len: 32 }));

  /* 2 · the decision point */
  const gem = card(480);
  gem.appendChild(box(262, 140, 268, 180, P.terra, 2.4));
  gem.appendChild(at(288, 174, 12, P.terra, 'DECISION POINT', 'm'));
  gem.appendChild(at(288, 216, 34, P.ink, 'Gemma', 't'));
  gem.appendChild(at(288, 252, 17, P.muted, 'reads the evidence,'));
  gem.appendChild(at(288, 276, 17, P.muted, 'chooses the call,'));
  gem.appendChild(at(288, 300, 17, P.muted, 'cites its sources'));
  g.appendChild(gem);

  /* 3 · the three branches */
  g.appendChild(link('M 530 200 C 586 200, 586 61, 640 61', P.sage, 900,
    { len: 200, tip: 'tipSage', back: true }));
  const b1 = card(1050);
  b1.appendChild(box(640, 0, 410, 122, P.sage, 1.8));
  b1.appendChild(chip(664, 18, 96, 'ALWAYS', P.sage));
  b1.appendChild(at(664, 78, 22, P.ink, 'list_corridor_status()', 'm'));
  b1.appendChild(at(664, 104, 15, P.muted, 'grounds itself in the real road graph'));
  g.appendChild(b1);

  /* The conditional edge: long dashes, its own decision lozenge, and a
     card border that is dashed too, so "sometimes" is legible at 20 feet. */
  g.appendChild(link('M 530 231 L 566 231', P.terra, 1200,
    { dash: '11 8', width: 2.2, tip: 'tipTerra' }));
  const dia = el('g', { class: 'pop' });
  dia.style.transitionDelay = '1300ms';
  dia.appendChild(el('path', {
    d: 'M 601 210 L 622 231 L 601 252 L 580 231 Z',
    fill: P.ivory, stroke: P.terra, 'stroke-width': 2.2,
  }));
  dia.appendChild(el('text', {
    x: 601, y: 238, 'font-size': 19, fill: P.terra,
    'text-anchor': 'middle', class: 'm',
  }, '?'));
  g.appendChild(dia);
  g.appendChild(link('M 626 231 L 640 231', P.terra, 1400,
    { dash: '11 8', width: 2.2, tip: 'tipTerra' }));

  const b2 = card(1450);
  b2.appendChild(box(640, 152, 410, 158, P.terra, 2.2, true));
  b2.appendChild(chip(664, 170, 128, 'CONDITIONAL', P.terra));
  b2.appendChild(at(664, 230, 21, P.ink, 'verify_report_with_imagery()', 'm'));
  b2.appendChild(at(664, 258, 15, P.rust, 'only if evidence is uncorroborated,'));
  b2.appendChild(at(664, 280, 15, P.rust, 'sources contradict, or an advisory'));
  b2.appendChild(at(664, 302, 15, P.rust, 'implies risk before any report exists'));
  g.appendChild(b2);

  g.appendChild(link('M 530 262 C 586 262, 586 399, 640 399', P.sage, 1650,
    { len: 200, tip: 'tipSage' }));
  const b3 = card(1800);
  b3.appendChild(box(640, 338, 410, 122, P.sage, 1.8));
  b3.appendChild(chip(664, 356, 96, 'ALWAYS', P.sage));
  b3.appendChild(at(664, 416, 22, P.ink, 'run_optimization()', 'm'));
  b3.appendChild(at(664, 442, 15, P.muted, 'every argument checked against the graph'));
  g.appendChild(b3);

  /* 4 · past the only branch that produces a plan */
  g.appendChild(link('M 1056 399 L 1092 399', P.sage, 2000, { len: 38, tip: 'tipSage' }));
  const eng = card(2050);
  eng.appendChild(box(1100, 338, 180, 122, P.sage, 1.8));
  eng.appendChild(at(1122, 368, 11, P.sage, 'DETERMINISTIC', 'm'));
  eng.appendChild(at(1122, 404, 26, P.ink, 'Engine', 't'));
  eng.appendChild(at(1122, 430, 14, P.muted, 'urgency, Dijkstra,'));
  eng.appendChild(at(1122, 450, 14, P.muted, 'allocation'));
  g.appendChild(eng);

  g.appendChild(link('M 1286 399 L 1318 399', P.subtle, 2250, { len: 34 }));
  const hum = card(2300);
  hum.appendChild(box(1326, 338, 154, 122, P.rust, 2.2));
  hum.appendChild(at(1348, 368, 11, P.rust, 'AUTHORITY', 'm'));
  hum.appendChild(at(1348, 404, 26, P.ink, 'Human', 't'));
  hum.appendChild(at(1348, 430, 14, P.muted, 'authorises,'));
  hum.appendChild(at(1348, 450, 14, P.muted, 'or refuses'));
  g.appendChild(hum);
}

/* ── Figure: urgency time factor ─────────────────────────────────────── */
function buildUrgency(g) {
  const W = 760, H = 430, L = 58, R = 22, T = 26, B = 56;
  const xMax = 12, yMax = 19;
  const px = (t) => L + (t / xMax) * (W - L - R);
  const py = (v) => H - B - (v / yMax) * (H - T - B);
  const T_of = (t) => 1 + 0.5 * (Math.exp(0.3 * t) - 1);

  for (let v = 0; v <= yMax; v += 5) {
    g.appendChild(el('line', {
      x1: L, y1: py(v), x2: W - R, y2: py(v), stroke: P.clay, 'stroke-width': 1.2,
    }));
    g.appendChild(el('text', {
      x: L - 12, y: py(v) + 5, 'font-size': 14, fill: P.subtle,
      'text-anchor': 'end', class: 'm',
    }, String(v)));
  }
  for (let t = 0; t <= xMax; t += 4) {
    g.appendChild(el('text', {
      x: px(t), y: H - B + 26, 'font-size': 14, fill: P.subtle,
      'text-anchor': 'middle', class: 'm',
    }, String(t)));
  }
  g.appendChild(el('text', {
    x: (L + W - R) / 2, y: H - 8, 'font-size': 16, fill: P.muted,
    'text-anchor': 'middle',
  }, 'hours since the incident began'));

  const steps = [];
  for (let i = 0; i <= 240; i += 1) steps.push(i * (xMax / 240));

  const linear = steps.map((t) => `${px(t)},${py(1 + 0.2 * t)}`).join(' ');
  const dashed = el('polyline', {
    points: linear, fill: 'none', stroke: P.subtle,
    'stroke-width': 1.6, 'stroke-dasharray': '7 6',
  });
  dashed.setAttribute('class', 'fade f1');
  g.appendChild(dashed);

  const curvePts = steps.map((t) => `${px(t)},${py(Math.min(T_of(t), yMax))}`);
  const area = el('polygon', {
    points: `${px(0)},${py(0)} ${curvePts.join(' ')} ${px(xMax)},${py(0)}`,
    fill: P.terra, opacity: 0.11, class: 'area',
  });
  g.appendChild(area);

  const curve = el('polyline', {
    points: curvePts.join(' '), fill: 'none', stroke: P.terra,
    'stroke-width': 3.6, 'stroke-linecap': 'round', class: 'draw d1',
  });
  curve.style.setProperty('--len', '1800');
  g.appendChild(curve);

  g.appendChild(el('text', {
    x: px(11.6), y: py(1 + 0.2 * 11.6) + 30, 'font-size': 15, fill: P.subtle,
    'text-anchor': 'end', class: 'fade f2', 'font-style': 'italic',
  }, 'if urgency grew linearly'));

  [[2, 1, -14, -16], [4, 1, -14, -16], [8, 0, 16, 6]].forEach(([hour, right, dx, dy], i) => {
    const v = T_of(hour);
    const dot = el('circle', {
      cx: px(hour), cy: py(v), r: 6.5, fill: P.terra,
      stroke: P.ivory, 'stroke-width': 2.6, class: `fade f${i + 2}`,
    });
    g.appendChild(dot);
    g.appendChild(el('text', {
      x: px(hour) + dx, y: py(v) + dy, 'font-size': 15, fill: P.ink,
      'text-anchor': right ? 'end' : 'start', class: `m fade f${i + 2}`,
    }, `T(${hour}) = ${v.toFixed(3)}`));
  });

  const head = el('circle', {
    r: 5.5, fill: P.ember, opacity: 0.9, class: 'fade f4',
  });
  head.setAttribute('cx', String(px(0)));
  head.setAttribute('cy', String(py(1)));
  g.appendChild(head);
  let hk = 0;
  (function crawl() {
    hk = (hk + 0.0026) % 1;
    const tt = hk * xMax;
    head.setAttribute('cx', String(px(tt)));
    head.setAttribute('cy', String(py(Math.min(T_of(tt), yMax))));
    requestAnimationFrame(crawl);
  })();

  const label = el('text', {
    x: px(0.6), y: py(16.4), 'font-size': 17, fill: P.rust, class: 'fade f4',
  }, 'what the survival curve');
  g.appendChild(label);
  g.appendChild(el('text', {
    x: px(0.6), y: py(15.1), 'font-size': 17, fill: P.rust, class: 'fade f4',
  }, 'actually looks like'));
}

/* ── Figure: the real corridor graph, drawn from terrain values ───────
   Nodes and edges are transcribed verbatim from backend/data/terrain_graph.json
   — the same fixture `list_corridor_status()` returns to Gemma. Stroke weight
   is terrain difficulty, colour is the surface, and a dashed stroke is the
   file's own vulnerable_to_landslide flag. Nothing here is drawn by eye.  */
const NODES = {
  depot: [85.324, 27.7172, 'Kathmandu hub'],
  mahendranagar: [80.1772, 28.9639, 'Mahendranagar'],
  jumla: [82.1838, 29.2747, 'Jumla'],
  pokhara: [83.9856, 28.2096, 'Pokhara'],
  bharatpur: [84.3542, 27.5291, 'Bharatpur'],
  janakpur: [85.925, 26.7288, 'Janakpur'],
  dharan: [87.2846, 26.8065, 'Dharan'],
  taplejung: [87.668, 27.352, 'Taplejung'],
  nepalgunj: [81.6167, 28.05, 'Nepalgunj'],
};
// [from, to, distance_km, terrain_difficulty, road_quality, vulnerable_to_landslide]
const EDGES = [
  ['depot', 'bharatpur', 146.0, 1.4, 'paved', 1],
  ['bharatpur', 'pokhara', 145.0, 2.0, 'paved', 1],
  ['depot', 'pokhara', 202.0, 2.2, 'paved', 1],
  ['depot', 'janakpur', 225.0, 1.3, 'paved', 0],
  ['janakpur', 'dharan', 190.0, 1.4, 'paved', 0],
  ['depot', 'dharan', 356.0, 2.0, 'paved', 1],
  ['bharatpur', 'janakpur', 274.0, 1.5, 'paved', 0],
  ['bharatpur', 'nepalgunj', 335.0, 1.3, 'paved', 0],
  ['pokhara', 'nepalgunj', 346.0, 1.8, 'paved', 1],
  ['nepalgunj', 'mahendranagar', 230.0, 1.2, 'paved', 0],
  ['pokhara', 'jumla', 280.0, 3.8, 'dirt', 1],
  ['nepalgunj', 'jumla', 285.0, 3.6, 'mixed', 1],
  ['dharan', 'taplejung', 160.0, 4.2, 'dirt', 1],
];
const LABEL_DX = {
  nepalgunj: [-6, 28], bharatpur: [34, 26], janakpur: [4, 30],
  dharan: [22, 30], depot: [0, -18], taplejung: [0, -18],
  jumla: [0, -18], mahendranagar: [0, -18], pokhara: [50, -8],
};
// Only the corridors whose difficulty is the argument get an inline tau label;
// thirteen of them would be a wall of numbers rather than a map.
const TAU_LABEL = {
  'dharan|taplejung': [98, -4], 'pokhara|jumla': [58, -16],
  'nepalgunj|jumla': [0, 150], 'nepalgunj|mahendranagar': [-51, 48],
};

function buildCorridors(g) {
  // Asymmetric right margin: Taplejung is the easternmost node and its corridor
  // label needs somewhere to live that is not on top of the rail.
  const W = 900, H = 600, padL = 66, padR = 156, mapTop = 40, mapBottom = 430;
  const lngs = Object.values(NODES).map((n) => n[0]);
  const lats = Object.values(NODES).map((n) => n[1]);
  const [x0, x1] = [Math.min(...lngs), Math.max(...lngs)];
  const [y0, y1] = [Math.min(...lats), Math.max(...lats)];
  const sx = (lng) => padL + ((lng - x0) / (x1 - x0)) * (W - padL - padR);
  const sy = (lat) => mapBottom - ((lat - y0) / (y1 - y0)) * (mapBottom - mapTop);
  // Difficulty drives weight so the mountain corridors read as heavy; the
  // surface drives colour, so "dirt" is visible before anyone reads a number.
  const weight = (tau) => 1.7 + (tau - 1) * 1.25;
  const shade = { paved: P.sage, mixed: P.ember, dirt: P.rust };

  EDGES.forEach(([a, b, km, tau, quality, vuln], i) => {
    const A = NODES[a], B = NODES[b];
    // Deliberately not `.draw`: that class owns stroke-dasharray for its wipe,
    // and a CSS declaration beats the presentation attribute, so the
    // landslide dashes would silently disappear. These fade in instead.
    const line = el('line', {
      x1: sx(A[0]), y1: sy(A[1]), x2: sx(B[0]), y2: sy(B[1]),
      stroke: shade[quality] || P.sage,
      'stroke-width': weight(tau).toFixed(2),
      'stroke-linecap': 'round',
      opacity: quality === 'paved' ? 0.72 : 0.95,
      class: 'fade',
    });
    if (vuln) line.setAttribute('stroke-dasharray', '13 9');
    line.style.transitionDelay = `${140 + i * 60}ms`;
    g.appendChild(line);

    const tag = TAU_LABEL[`${a}|${b}`];
    if (tag) {
      const mx = (sx(A[0]) + sx(B[0])) / 2 + tag[0];
      const my = (sy(A[1]) + sy(B[1])) / 2 + tag[1];
      const t = el('text', {
        x: mx, y: my, 'font-size': 15, fill: shade[quality] || P.sage,
        'text-anchor': 'middle', class: 'm fade',
      }, `${km.toFixed(0)} km · τ ${tau.toFixed(1)}`);
      t.style.transitionDelay = `${1150 + i * 40}ms`;
      g.appendChild(t);
    }
  });

  Object.entries(NODES).forEach(([id, [lng, lat, name]], i) => {
    const hub = id === 'depot';
    const dot = el('circle', {
      cx: sx(lng), cy: sy(lat), r: hub ? 11 : 7,
      fill: hub ? P.rust : P.ink, stroke: P.ivory, 'stroke-width': 3,
      class: 'pop',
    });
    dot.style.transitionDelay = `${700 + i * 45}ms`;
    g.appendChild(dot);
    const [dx, dy] = LABEL_DX[id] || [0, -18];
    const t = el('text', {
      x: sx(lng) + dx, y: sy(lat) + dy, 'font-size': 15,
      fill: hub ? P.ink : P.muted, 'text-anchor': 'middle', class: 'fade f2',
    }, name);
    g.appendChild(t);
  });

  /* Legend — the encoding, stated, because a judge should not have to infer it. */
  const key = el('g', { class: 'fade' });
  key.style.transitionDelay = '1500ms';
  const ly = H - 88;
  key.appendChild(el('text', {
    x: 0, y: ly - 26, 'font-size': 13, fill: P.rust, class: 'm',
    'letter-spacing': '0.1em',
  }, 'HOW THIS MAP IS ENCODED'));
  const rows = [
    [P.sage, weight(1.2), null, 'paved · τ 1.2 lowest on the graph'],
    [P.rust, weight(4.2), null, 'dirt · τ 4.2 highest on the graph'],
    [P.muted, 2.4, '13 8', 'vulnerable_to_landslide · 8 of 13 corridors'],
  ];
  rows.forEach(([colour, sw, dash, label], i) => {
    const y = ly + i * 25;
    const seg = el('line', {
      x1: 0, y1: y, x2: 54, y2: y, stroke: colour,
      'stroke-width': sw, 'stroke-linecap': 'round', opacity: 0.9,
    });
    if (dash) seg.setAttribute('stroke-dasharray', dash);
    key.appendChild(seg);
    key.appendChild(el('text', { x: 68, y: y + 5, 'font-size': 15, fill: P.muted }, label));
  });
  g.appendChild(key);
}

/* ── Count-up on any [data-count] ───────────────────────────────────── */
function runCounters(slide) {
  slide.querySelectorAll('[data-count]').forEach((node) => {
    const target = Number(node.dataset.count);
    const suffix = node.dataset.suffix || '';
    const delay = node.classList.contains('p2') ? 1250 : 900;
    node.textContent = `0${suffix}`;
    window.setTimeout(() => {
      const started = performance.now();
      const dur = 620;
      const tick = (now) => {
        const k = Math.min(1, (now - started) / dur);
        const eased = 1 - Math.pow(1 - k, 3);
        node.textContent = `${Math.round(target * eased)}${suffix}`;
        if (k < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }, delay);
  });
}

/* ── Fit the fixed 1600x900 stage into the viewport ─────────────────── */
function fit() {
  const scale = Math.min(window.innerWidth / 1600, window.innerHeight / 900);
  document.querySelectorAll('.slide').forEach((s) => {
    s.style.transform = `scale(${scale})`;
  });
}

/* ── Navigation ─────────────────────────────────────────────────────── */
const slides = [...document.querySelectorAll('.slide')];
let index = 0;

function show(next, replay = true) {
  index = Math.max(0, Math.min(slides.length - 1, next));
  slides.forEach((s, i) => {
    if (i === index) return;
    s.classList.remove('is-active');
  });
  const slide = slides[index];
  if (replay) {
    // Re-arm the reveal choreography so a revisited slide animates again.
    slide.classList.remove('is-active');
    void slide.offsetWidth;
  }
  slide.classList.add('is-active');
  runCounters(slide);
  document.getElementById('counter').textContent = `${index + 1} / ${slides.length}`;
  document.getElementById('bar').style.width =
    `${((index + 1) / slides.length) * 100}%`;
  if (location.hash !== `#${index + 1}`) {
    history.replaceState(null, '', `#${index + 1}`);
  }
}

document.addEventListener('keydown', (event) => {
  const k = event.key;
  if (k === 'ArrowRight' || k === 'PageDown' || k === ' ' || k === 'Enter') {
    event.preventDefault(); show(index + 1);
  } else if (k === 'ArrowLeft' || k === 'PageUp' || k === 'Backspace') {
    event.preventDefault(); show(index - 1);
  } else if (k === 'Home') { show(0); }
  else if (k === 'End') { show(slides.length - 1); }
  else if (k === 'f') { document.documentElement.requestFullscreen?.(); }
  else if (/^[0-9]$/.test(k)) {
    // Jump targets. The script tells the presenter to "type 3" or "type 8"
    // mid-answer; 0 is the tenth slide, so a ten-slide deck stays reachable.
    event.preventDefault();
    show((k === '0' ? 10 : Number(k)) - 1);
  }
});
document.getElementById('next').addEventListener('click', () => show(index + 1));
document.getElementById('prev').addEventListener('click', () => show(index - 1));
document.getElementById('stage').addEventListener('click', (event) => {
  if (event.target.closest('#chrome')) return;
  show(index + (event.clientX > window.innerWidth * 0.35 ? 1 : -1));
});
window.addEventListener('resize', fit);

// Guarded so the same script drives both the full deck and the 3-minute pitch
// cut, which carries only some of these figures.
const pipelineHost = document.getElementById('pipeline');
const urgencyHost = document.getElementById('urgency');
const corridorHost = document.getElementById('corridors');
if (pipelineHost) buildPipeline(pipelineHost);
if (urgencyHost) buildUrgency(urgencyHost);
if (corridorHost) buildCorridors(corridorHost);
fit();
show(Math.max(0, (Number(location.hash.slice(1)) || 1) - 1), false);
