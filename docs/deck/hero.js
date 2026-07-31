/* Three.js hero: Gemma reading a terrain of evidence.
   A displaced ridge with signal pulses that converge on a single decision point.

   PERFORMANCE: the first version recomputed every vertex in JavaScript and
   re-uploaded the whole position buffer each frame, which dropped frames on the
   title slide. All displacement now happens in a vertex shader — the CPU uploads
   one float per frame (the clock) and nothing else. Each scene runs only while
   its own slide is active. */

(function () {
  if (typeof THREE === 'undefined') return;

  const RIDGE = `
    // Shared displacement: two crossed sine systems plus a central massif, so
    // the field reads as mountains rather than a wave.
    float ridge(vec2 p, float t) {
      float massif = exp(-((p.x * p.x) / 14.0 + (p.y * p.y) / 9.0)) * 2.8;
      return sin(p.x * 0.62 + t * 0.28) * 0.62
           + sin(p.y * 0.94 - t * 0.22) * 0.46
           + sin((p.x + p.y) * 0.40 + t * 0.16) * 0.50
           + massif;
    }
  `;

  const MESH_VERT = `
    uniform float uTime;
    varying float vH;
    varying float vD;
    ${RIDGE}
    void main() {
      vec3 p = position;
      p.y = ridge(vec2(p.x, p.z), uTime);
      vH = p.y;
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      vD = -mv.z;
      gl_Position = projectionMatrix * mv;
    }
  `;

  // Height and depth both drive opacity, giving the field aerial perspective:
  // ridges read as near, valleys recede.
  const MESH_FRAG = `
    uniform vec3 uColor;
    uniform float uOpacity;
    varying float vH;
    varying float vD;
    void main() {
      float byHeight = 0.34 + clamp(vH / 3.4, 0.0, 1.0) * 0.66;
      float byDepth  = 1.0 - clamp((vD - 5.0) / 12.0, 0.0, 0.72);
      gl_FragColor = vec4(uColor, uOpacity * byHeight * byDepth);
    }
  `;

  const PT_VERT = `
    uniform float uTime;
    attribute vec2 aOrigin;
    attribute float aSpeed;
    attribute float aPhase;
    varying float vFade;
    ${RIDGE}
    void main() {
      // Each pulse eases from its origin toward the hub, then loops.
      float k = fract(aPhase + uTime * aSpeed);
      float e = 1.0 - pow(1.0 - k, 2.4);
      vec2 xz = aOrigin * (1.0 - e);
      vec3 p = vec3(xz.x, ridge(xz, uTime) + 0.16, xz.y);
      vFade = smoothstep(0.0, 0.10, k) * (1.0 - smoothstep(0.86, 1.0, k));
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      gl_PointSize = (150.0 / max(-mv.z, 0.001)) * (0.7 + 0.5 * vFade);
      gl_Position = projectionMatrix * mv;
    }
  `;

  const PT_FRAG = `
    uniform vec3 uColor;
    varying float vFade;
    void main() {
      float d = length(gl_PointCoord - vec2(0.5));
      if (d > 0.5) discard;
      float soft = 1.0 - smoothstep(0.22, 0.5, d);
      gl_FragColor = vec4(uColor, soft * vFade * 0.95);
    }
  `;

  function build(holder, opts) {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
    camera.position.set(0, opts.camY, opts.camZ);
    camera.lookAt(0, opts.lookY, 0);

    const renderer = new THREE.WebGLRenderer({
      antialias: true, alpha: true, powerPreference: 'high-performance',
    });
    // Capping the ratio is the other half of the performance fix: a 2x buffer on
    // a 1600px stage is 4x the fragments for no visible gain on a projector.
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    holder.appendChild(renderer.domElement);

    const clock = { value: 0 };

    const COLS = 96, ROWS = 56, SPAN = 16, DEPTH = 10;
    const verts = [];
    const push = (c, r) => verts.push(
      (c / (COLS - 1) - 0.5) * SPAN, 0, (r / (ROWS - 1) - 0.5) * DEPTH,
    );
    for (let r = 0; r < ROWS; r += 1) {
      for (let c = 0; c < COLS - 1; c += 1) { push(c, r); push(c + 1, r); }
    }
    for (let c = 0; c < COLS; c += 3) {
      for (let r = 0; r < ROWS - 1; r += 1) { push(c, r); push(c, r + 1); }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));

    const mesh = new THREE.LineSegments(geo, new THREE.ShaderMaterial({
      uniforms: {
        uTime: clock,
        uColor: { value: new THREE.Color(0x2a1e17) },
        uOpacity: { value: opts.meshOpacity },
      },
      vertexShader: MESH_VERT,
      fragmentShader: MESH_FRAG,
      transparent: true,
      depthWrite: false,
    }));
    scene.add(mesh);

    const N = opts.pulses;
    const pg = new THREE.BufferGeometry();
    const origin = new Float32Array(N * 2);
    const speed = new Float32Array(N);
    const phase = new Float32Array(N);
    for (let i = 0; i < N; i += 1) {
      origin[i * 2] = (Math.random() - 0.5) * SPAN;
      origin[i * 2 + 1] = (Math.random() - 0.5) * DEPTH;
      speed[i] = 0.055 + Math.random() * 0.075;
      phase[i] = Math.random();
    }
    pg.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(N * 3), 3));
    pg.setAttribute('aOrigin', new THREE.Float32BufferAttribute(origin, 2));
    pg.setAttribute('aSpeed', new THREE.Float32BufferAttribute(speed, 1));
    pg.setAttribute('aPhase', new THREE.Float32BufferAttribute(phase, 1));
    scene.add(new THREE.Points(pg, new THREE.ShaderMaterial({
      uniforms: { uTime: clock, uColor: { value: new THREE.Color(0xc2673b) } },
      vertexShader: PT_VERT,
      fragmentShader: PT_FRAG,
      transparent: true,
      depthWrite: false,
    })));

    const hub = new THREE.Mesh(
      new THREE.SphereGeometry(0.13, 18, 18),
      new THREE.MeshBasicMaterial({ color: 0x8f4426 }),
    );
    scene.add(hub);
    const rings = [0.26, 0.40].map((r, i) => {
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(r, r + 0.022, 56),
        new THREE.MeshBasicMaterial({
          color: 0xc2673b, transparent: true, opacity: 0.4 - i * 0.14,
          side: THREE.DoubleSide, depthWrite: false,
        }),
      );
      ring.rotation.x = -Math.PI / 2;
      scene.add(ring);
      return ring;
    });

    // Hub height only: the one place the CPU still evaluates the ridge, at x=z=0.
    const hubY = (t) => Math.sin(t * 0.28) * 0.62 + Math.sin(-t * 0.22) * 0.46
      + Math.sin(t * 0.16) * 0.50 + 2.8;

    function resize() {
      const w = holder.clientWidth, h = holder.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, true);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }

    let raf = 0, running = false, last = 0;

    function frame(now) {
      if (!running) return;
      const dt = last ? Math.min((now - last) / 1000, 0.05) : 0.016;
      last = now;
      clock.value += dt;
      const t = clock.value;

      const hy = hubY(t) + 0.28;
      hub.position.y = hy;
      rings.forEach((ring, i) => {
        ring.position.y = hy - 0.24;
        const beat = 1 + Math.sin(t * 1.5 - i * 0.7) * 0.14;
        ring.scale.setScalar(beat);
        ring.material.opacity = (0.34 - i * 0.12) + Math.sin(t * 1.5 - i * 0.7) * 0.14;
      });
      mesh.rotation.y = Math.sin(t * 0.05) * 0.045;

      renderer.render(scene, camera);
      raf = requestAnimationFrame(frame);
    }

    return {
      start() {
        if (running) return;
        running = true; last = 0; resize();
        raf = requestAnimationFrame(frame);
      },
      stop() { running = false; cancelAnimationFrame(raf); },
      resize,
      once() { resize(); renderer.render(scene, camera); },
    };
  }

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  document.querySelectorAll('[data-hero]').forEach((holder) => {
    const scene = build(holder, holder.dataset.hero === 'close'
      ? { camY: 2.2, camZ: 10.5, lookY: 0.6, meshOpacity: 0.20, pulses: 22 }
      // lookY was 0.1, which aimed the camera far enough below the massif that
      // the hub — the "decision point" the legend names — sat on or above the
      // top edge for most of its vertical swing. The hub oscillates roughly
      // 1.5–4.7 in world units; aiming at 1.6 keeps every part of that swing
      // inside the 19-degree half-FOV, so the thing the legend points at is
      // actually on screen.
      : { camY: 4.6, camZ: 8.6, lookY: 1.6, meshOpacity: 0.32, pulses: 34 });

    window.addEventListener('resize', scene.resize);
    if (reduced) { scene.once(); return; }

    const slide = holder.closest('.slide');
    new MutationObserver(() => {
      slide.classList.contains('is-active') ? scene.start() : scene.stop();
    }).observe(slide, { attributes: true, attributeFilter: ['class'] });
    if (slide.classList.contains('is-active')) scene.start();
  });
})();
