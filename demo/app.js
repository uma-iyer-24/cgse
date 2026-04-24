/* ============================================================
   CGSE Demo — app.js
   All data is hardcoded from web/generated-config.js (seed 42)
   ============================================================ */

// ---- DATA ----

const T1_ARMS = [
  { label: "Fixed (no mutation)",   acc: 84.48, color: "#5b9fd4" },
  { label: "Scheduled widen",       acc: 84.38, color: "#7c6cf0" },
  { label: "Teacher + KD",          acc: 84.87, color: "#e8a849" },
  { label: "Teacher + KD + widen",  acc: 84.95, color: "#e86f9a" },
  { label: "CGSE (critic)",         acc: 84.68, color: "#3ecf8e" },
];

const T2_ROWS = [
  { label: "Teacher (ResNet-56)",       acc: 92.38, tf: 0,     wall: 3452, color: "#e8a849" },
  { label: "Student KD (ResNet-20)",    acc: 91.66, tf: 19550, wall: 3045, color: "#5b9fd4" },
  { label: "CGSE multi-op (ResNet-20)", acc: 90.85, tf: 0,     wall: 2846, color: "#3ecf8e" },
];

// per-epoch val_acc (%) from seed 42, all 50 epochs
const T2_CURVES = {
  teacher: [25.91,48.52,59.48,65.38,72.8,76.59,73.3,76.2,64.33,74.31,73.28,81.19,75.87,77.91,78.56,80.95,80.52,76.56,77.3,79.83,74.31,81.73,78.97,79.95,80.93,89.73,90.2,90.64,90.47,90.37,90.89,90.94,90.58,91.0,91.08,91.29,90.64,90.55,90.03,90.56,92.05,92.1,92.3,92.23,92.38,92.33,92.28,92.33,92.33,92.19],
  kd:      [46.19,66.57,64.47,75.14,77.35,69.89,74.0,77.07,75.55,77.87,78.73,80.22,77.08,76.55,74.79,81.58,74.32,77.87,77.75,77.15,79.46,82.01,81.22,79.91,80.4,89.73,90.1,90.42,90.63,90.57,90.57,90.87,90.76,90.6,91.04,90.91,91.1,91.05,91.23,90.59,91.39,91.48,91.48,91.49,91.65,91.6,91.64,91.66,91.58,91.59],
  cgse:    [48.13,65.27,70.75,73.96,74.57,75.37,68.46,71.93,73.61,70.18,76.3,75.04,77.64,75.13,77.1,76.59,80.3,76.89,67.32,81.18,74.78,79.25,79.68,77.21,75.6,88.68,88.87,88.97,89.31,89.43,89.45,89.31,89.42,89.57,89.68,89.55,88.9,88.99,89.75,88.92,90.3,90.54,90.84,90.55,90.54,90.67,90.85,90.74,90.48,90.62],
};

// Simulated 30-epoch demo trajectory — bumps/dips align with critic mutations
const DEMO_ACC = [
  48.0, 56.2, 61.1, 65.4, 68.7,        // 0-4 warm-up
  71.0,                                 // 5  critic→noop
  72.4, 74.0, 75.0,                    // 6-8
  75.8,                                 // 9  critic→widen
  77.1, 77.9, 78.5,                    // 10-12
  79.0,                                 // 13 critic→noop
  79.6, 80.0,                          // 14-15
  79.4,                                 // 16 critic→deepen (brief dip)
  80.2, 81.5, 83.0, 84.0,             // 17-20 recover & climb
  84.5,                                 // 21 critic→widen
  85.4, 86.1, 86.7,                    // 22-24
  87.0,                                 // 25 critic→head-widen
  87.5, 88.1, 88.6, 89.1,             // 26-29
];

const CIFAR_SAMPLES = [
  { cls:"airplane",   img:"https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=220&h=220&fit=crop",
    probs:[["airplane",0.89],["ship",0.06],["automobile",0.03],["bird",0.01],["truck",0.01]] },
  { cls:"automobile", img:"https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=220&h=220&fit=crop",
    probs:[["automobile",0.93],["truck",0.05],["ship",0.01],["airplane",0.01]] },
  { cls:"bird",       img:"https://images.unsplash.com/photo-1444464666168-49d633b86797?w=220&h=220&fit=crop",
    probs:[["bird",0.78],["airplane",0.12],["deer",0.05],["cat",0.03],["frog",0.02]] },
  { cls:"cat",        img:"https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=220&h=220&fit=crop",
    probs:[["cat",0.72],["dog",0.18],["deer",0.05],["frog",0.03],["horse",0.02]] },
  { cls:"deer",       img:"https://images.unsplash.com/photo-1484750359892-1e70ef1dd2e0?w=220&h=220&fit=crop",
    probs:[["deer",0.85],["horse",0.08],["bird",0.04],["cat",0.02],["dog",0.01]] },
  { cls:"dog",        img:"https://images.unsplash.com/photo-1517849845537-4d257902454a?w=220&h=220&fit=crop",
    probs:[["dog",0.80],["cat",0.12],["deer",0.04],["frog",0.02],["horse",0.02]] },
  { cls:"frog",       img:"https://images.unsplash.com/photo-1557096998-0f1a4db2a826?w=220&h=220&fit=crop",
    probs:[["frog",0.88],["bird",0.06],["cat",0.03],["deer",0.02],["dog",0.01]] },
  { cls:"horse",      img:"https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=220&h=220&fit=crop",
    probs:[["horse",0.91],["deer",0.05],["automobile",0.02],["dog",0.01],["truck",0.01]] },
  { cls:"ship",       img:"https://images.unsplash.com/photo-1505118380757-91f5f5632de0?w=220&h=220&fit=crop",
    probs:[["ship",0.87],["airplane",0.08],["automobile",0.03],["truck",0.01],["bird",0.01]] },
  { cls:"truck",      img:"https://images.unsplash.com/photo-1601584462940-0e7095e25fcf?w=220&h=220&fit=crop",
    probs:[["truck",0.86],["automobile",0.10],["ship",0.02],["airplane",0.01],["bird",0.01]] },
];

// ============================================================
//  HERO CANVAS — animated neural network
// ============================================================
class NetworkCanvas {
  constructor(canvas, indicator) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext("2d");
    this.ind    = indicator;
    this.layers = [3, 5, 6, 4, 3];
    this.nodes  = [];
    this.t      = 0;
    this.mutTimer = 0;
    this.mutPhase = "idle";
    this.mutLayer = -1;
    this._w = 0; this._h = 0;
    this.MSGS = [
      "CRITIC: widen layer 2 →",
      "CRITIC: deepen layer 3 →",
      "CRITIC: head-widen →",
      "CRITIC: noop (val improving) ·",
    ];
    this._onResize = () => this.resize();
    window.addEventListener("resize", this._onResize);
    this.resize();
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.offsetWidth;
    const h = this.canvas.offsetHeight;
    this.canvas.width  = w * dpr;
    this.canvas.height = h * dpr;
    this.ctx.scale(dpr, dpr);
    this._w = w; this._h = h;
    this._buildNodes();
  }

  _buildNodes() {
    this.nodes = this.layers.map((count, li) => {
      const x = this._w * (li + 1) / (this.layers.length + 1);
      return Array.from({ length: count }, (_, ni) => ({
        x, y: this._h * (ni + 1) / (count + 1),
        r: 5,
        phase:  Math.random() * Math.PI * 2,
        speed:  0.8 + Math.random() * 0.4,
        emerge: false, ep: 1.0,
      }));
    });
  }

  _showIndicator(msg) {
    this.ind.textContent = msg;
    this.ind.classList.add("show");
    setTimeout(() => this.ind.classList.remove("show"), 2400);
  }

  _doMutation() {
    const li = this.mutLayer;
    const prev = this.layers[li];
    const add  = Math.random() > 0.4 ? 1 : 2;
    const next = Math.min(prev + add, 10);
    this.layers[li] = next;
    const x = this._w * (li + 1) / (this.layers.length + 1);
    this.nodes[li] = Array.from({ length: next }, (_, ni) => ({
      x, y: this._h * (ni + 1) / (next + 1),
      r: 5, phase: Math.random() * Math.PI * 2,
      speed: 0.8 + Math.random() * 0.4,
      emerge: ni >= prev,
      ep: ni >= prev ? 0.0 : 1.0,
    }));
    this.mutPhase = "idle";
    // Reset to prevent indefinite growth
    if (this.layers.some(l => l > 9)) this.layers = [3, 5, 6, 4, 3];
  }

  update(dt) {
    this.t        += dt;
    this.mutTimer += dt;
    if (this.mutTimer > 6500 && this.mutPhase === "idle") {
      this.mutTimer = 0;
      const idx = Math.floor(Math.random() * this.MSGS.length);
      this._showIndicator(this.MSGS[idx]);
      this.mutPhase = "firing";
      this.mutLayer = 1 + Math.floor(Math.random() * (this.layers.length - 2));
      if (idx < 2) setTimeout(() => this._doMutation(), 700);
      else         setTimeout(() => { this.mutPhase = "idle"; }, 700);
    }
    this.nodes.forEach(layer => layer.forEach(n => {
      if (n.emerge) { n.ep = Math.min(1, n.ep + dt / 550); if (n.ep >= 1) n.emerge = false; }
    }));
  }

  draw() {
    const ctx = this.ctx;
    const W = this._w, H = this._h;
    ctx.clearRect(0, 0, W, H);

    // Edges
    for (let li = 0; li < this.nodes.length - 1; li++) {
      const A = this.nodes[li], B = this.nodes[li + 1];
      const firing = this.mutPhase === "firing" && (li === this.mutLayer - 1 || li === this.mutLayer);
      A.forEach(a => B.forEach(b => {
        const pulse = 0.18 + 0.13 * Math.sin(this.t / 1400 + a.phase);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
        ctx.strokeStyle = `rgba(91,159,212,${(pulse + (firing ? 0.32 : 0)) * a.ep * b.ep})`;
        ctx.lineWidth = 0.7;
        ctx.stroke();
      }));
    }

    // Nodes
    this.nodes.forEach((layer, li) => {
      const firing = this.mutPhase === "firing" && li === this.mutLayer;
      const colStr = firing ? "124,108,240" : "91,159,212";
      layer.forEach(n => {
        const pulse = 0.65 + 0.3 * Math.sin(this.t / (1100 * n.speed) + n.phase);
        const r = n.r * n.ep;
        // Glow
        const g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, r * 5);
        g.addColorStop(0, `rgba(${colStr},${0.22 * pulse * n.ep})`);
        g.addColorStop(1, "rgba(0,0,0,0)");
        ctx.beginPath(); ctx.arc(n.x, n.y, r * 5, 0, Math.PI * 2);
        ctx.fillStyle = g; ctx.fill();
        // Core
        ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${colStr},${(0.55 + 0.35 * pulse) * n.ep})`;
        ctx.fill();
        // Emerge ring
        if (n.emerge) {
          ctx.beginPath(); ctx.arc(n.x, n.y, r + 4 * (1 - n.ep), 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(62,207,142,${n.ep})`;
          ctx.lineWidth = 1.5; ctx.stroke();
        }
      });
    });
  }

  start() {
    let last = performance.now();
    const tick = now => {
      const dt = now - last; last = now;
      this.update(dt); this.draw();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
}

// ============================================================
//  STEPPER
// ============================================================
const STEP_DETAILS = [
  { icon:"📈", title:"Warm-up training",
    body:"The student network (ResNet-20, 272K params) trains for 25 epochs using SGD with momentum, weight decay, and multi-step LR schedule. No mutations, no teacher. The critic watches but doesn't act yet." },
  { icon:"🔎", title:"Critic observes the 8-dim state",
    body:"Every epoch the critic reads: normalized train loss, val accuracy, epoch progress [0,1], log-normalised param count, Δval acc, Δtrain loss, normalised val loss, and a constant bias term." },
  { icon:"⚡", title:"Discrete action decision",
    body:"DiscreteMutationCritic outputs a probability over 4 actions: <strong>noop</strong>, <strong>head-widen</strong> (Net2Net), <strong>layer3-widen</strong> (function-preserving), <strong>insert BasicBlock</strong> (deepening). Trained via REINFORCE on post-mutation val gain." },
  { icon:"🚀", title:"Resume — function-preserving growth",
    body:"All mutations preserve the function: widened layers initialise extra weights to zero, new residual blocks initialise their final BN to zero output. Training resumes from the same optimizer state, gaining 1–2% val accuracy in the best cases." },
];

function initStepper() {
  const steps  = document.querySelectorAll("#stepper .step");
  const detail = document.getElementById("step-detail");
  const fill   = document.getElementById("step-progress-fill");
  let current = 0, prog = 0;
  const STEP_MS = 4500;

  function render(idx) {
    steps.forEach((s, i) => s.classList.toggle("active", i === idx));
    const d = STEP_DETAILS[idx];
    detail.innerHTML = `
      <div class="step-detail-icon">${d.icon}</div>
      <div class="step-detail-text">
        <h4>${d.title}</h4>
        <p>${d.body}</p>
      </div>`;
  }

  steps.forEach((s, i) => s.addEventListener("click", () => {
    current = i; prog = 0; render(current);
  }));
  render(0);

  let last = performance.now();
  (function tick(now) {
    const dt = now - last; last = now;
    prog = Math.min(100, prog + (dt / STEP_MS) * 100);
    fill.style.width = prog + "%";
    if (prog >= 100) { current = (current + 1) % STEP_DETAILS.length; prog = 0; render(current); }
    requestAnimationFrame(tick);
  })(performance.now());
}

// ============================================================
//  MUTATION DEMO — physics-based animated graph + multi-event critic
// ============================================================
function initMutationDemo() {
  const canvas   = document.getElementById("demo-canvas");
  const ctx      = canvas.getContext("2d");
  const epochEl  = document.getElementById("demo-epoch");
  const accEl    = document.getElementById("demo-acc");
  const nodesEl  = document.getElementById("demo-nodes");
  const paramsEl = document.getElementById("demo-params");
  const logEl    = document.getElementById("demo-log");
  const btn      = document.getElementById("demo-btn");

  // ---- config ----
  const INITIAL_LAYERS = [
    { count: 3,  type: "input"  },
    { count: 4,  type: "hidden" },
    { count: 10, type: "output" },
  ];

  // Six critic events across 30 epochs (mix of noops and 3 mutation kinds)
  const SCHEDULE = [
    { atEpoch: 5,  kind: "noop",       msg: "noop (val improving)" },
    { atEpoch: 9,  kind: "widen",      layerIdx: 1, addNodes: 2,
      msg: "widen hidden 1 (4 → 6)" },
    { atEpoch: 13, kind: "noop",       msg: "noop (steady)" },
    { atEpoch: 16, kind: "deepen",     afterLayer: 1, newCount: 4,
      msg: "insert hidden block (depth +1)" },
    { atEpoch: 21, kind: "widen",      layerIdx: 1, addNodes: 2,
      msg: "widen hidden 1 (6 → 8)" },
    { atEpoch: 25, kind: "head-widen", msg: "head-widen (classifier proj +1)" },
  ];

  // ---- state ----
  let st        = null;       // { epoch, running, layers, nodes }
  let pulseT    = -1;         // start time of forward-pass pulse
  let flashLayer = -1;        // which layer is currently flashing
  let flashT    = 0;          // start time of mutation flash

  function freshState() {
    return {
      epoch: 0,
      running: false,
      layers: INITIAL_LAYERS.map(l => ({ ...l })),
      nodes: [],   // {layerIdx, slot, x, y, tx, ty, alpha, talpha, age, fresh}
    };
  }

  // Re-compute target positions for current st.layers; preserve identity by (layerIdx, slot)
  function reconcileNodes(skipAnim = false) {
    const W = canvas.width;
    const H = canvas.height;
    const PAD_BOTTOM = 24;     // reserve space for layer label
    const xs = st.layers.map((_, li) => W * (li + 1) / (st.layers.length + 1));

    const oldMap = new Map();
    st.nodes.forEach(n => oldMap.set(`${n.layerIdx}_${n.slot}`, n));

    const next = [];
    st.layers.forEach((layer, li) => {
      for (let i = 0; i < layer.count; i++) {
        const tx = xs[li];
        const ty = (i + 1) * (H - PAD_BOTTOM) / (layer.count + 1);
        const old = oldMap.get(`${li}_${i}`);
        if (old) {
          old.tx = tx; old.ty = ty; old.talpha = 1;
          if (skipAnim) { old.x = tx; old.y = ty; old.alpha = 1; }
          next.push(old);
        } else {
          // New node — spawn at the layer column center, fade in
          next.push({
            layerIdx: li, slot: i,
            x: tx, y: H / 2,
            tx, ty,
            alpha: skipAnim ? 1 : 0,
            talpha: 1,
            age: 0,
            fresh: !skipAnim,
          });
        }
      }
    });
    st.nodes = next;
  }

  function applyMutation(ev) {
    if (ev.kind === "widen") {
      st.layers[ev.layerIdx].count += ev.addNodes;
      flashLayer = ev.layerIdx;
    } else if (ev.kind === "deepen") {
      st.layers.splice(ev.afterLayer + 1, 0, { count: ev.newCount, type: "hidden" });
      // Bump layerIdx of nodes that now sit further to the right
      st.nodes.forEach(n => { if (n.layerIdx > ev.afterLayer) n.layerIdx += 1; });
      flashLayer = ev.afterLayer + 1;
    } else if (ev.kind === "head-widen") {
      const li = st.layers.length - 2; // last hidden
      st.layers[li].count += 1;
      flashLayer = li;
    }
    reconcileNodes();
    flashT = performance.now();
  }

  function totalHiddenNodes() {
    return st.layers.slice(1, -1).reduce((s, l) => s + l.count, 0);
  }
  function approxParams() {
    let edges = 0;
    for (let i = 0; i < st.layers.length - 1; i++) {
      edges += st.layers[i].count * st.layers[i + 1].count;
    }
    // 52 edges (3*4 + 4*10) baseline ~ 272K
    return Math.round(272 * edges / 52) + "K";
  }

  function log(text, cls = "") {
    const s = document.createElement("span");
    s.innerHTML = `<br><span class="${cls}">${text}</span>`;
    logEl.appendChild(s);
    logEl.scrollTop = logEl.scrollHeight;
  }

  function resize() {
    canvas.width  = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight || 280;
    if (st) reconcileNodes(true);
  }
  window.addEventListener("resize", resize);

  // ---- physics ----
  function update(dt) {
    if (!st) return;
    // exponential smoothing toward target (~90 ms time constant)
    const k = 1 - Math.exp(-dt / 90);
    st.nodes.forEach(n => {
      n.x     += (n.tx     - n.x)     * k;
      n.y     += (n.ty     - n.y)     * k;
      n.alpha += (n.talpha - n.alpha) * k;
      n.age   += dt;
    });
  }

  // ---- render ----
  function draw() {
    if (!st) return;
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // group nodes by current layerIdx
    const byLayer = {};
    st.nodes.forEach(n => {
      if (!byLayer[n.layerIdx]) byLayer[n.layerIdx] = [];
      byLayer[n.layerIdx].push(n);
    });

    // Forward-pass pulse position [0 .. numLayers-1]
    let pulsePos = -1;
    if (pulseT > 0) {
      const elapsed = performance.now() - pulseT;
      const dur = 700;
      if (elapsed < dur) pulsePos = (elapsed / dur) * (st.layers.length - 1);
      else pulseT = -1;
    }

    // Mutation flash intensity decays over 1300 ms
    const flashAge = performance.now() - flashT;
    const flashI   = flashAge < 1300 ? Math.max(0, 1 - flashAge / 1300) : 0;

    // ---- edges ----
    for (let li = 0; li < st.layers.length - 1; li++) {
      const A = byLayer[li]      || [];
      const B = byLayer[li + 1]  || [];
      const flashThis = flashI > 0 && (li === flashLayer - 1 || li === flashLayer);
      A.forEach(a => B.forEach(b => {
        const al = Math.min(a.alpha, b.alpha);
        if (al < 0.04) return;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        if (flashThis) {
          ctx.strokeStyle = `rgba(62,207,142,${0.18 * al + flashI * 0.45})`;
          ctx.lineWidth   = 1 + flashI * 0.7;
        } else {
          ctx.strokeStyle = `rgba(91,159,212,${0.18 * al})`;
          ctx.lineWidth   = 0.85;
        }
        ctx.stroke();
      }));

      // forward-pass particles travelling along this column's edges
      if (pulsePos >= li && pulsePos < li + 1) {
        const t = pulsePos - li;
        A.forEach(a => B.forEach(b => {
          if (Math.min(a.alpha, b.alpha) < 0.5) return;
          const px = a.x + (b.x - a.x) * t;
          const py = a.y + (b.y - a.y) * t;
          ctx.beginPath();
          ctx.arc(px, py, 1.9, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(124,108,240,0.85)";
          ctx.fill();
        }));
      }
    }

    // ---- nodes ----
    Object.entries(byLayer).forEach(([liStr, layer]) => {
      const li = +liStr;
      const flashThis = flashI > 0 && li === flashLayer;
      const colStr    = flashThis ? "62,207,142" : "91,159,212";
      layer.forEach(n => {
        if (n.alpha < 0.02) return;
        const r = 7 * n.alpha;
        // glow
        const glow = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, r * 3.5);
        glow.addColorStop(0, `rgba(${colStr},${0.32 * n.alpha + (flashThis ? flashI * 0.35 : 0)})`);
        glow.addColorStop(1, `rgba(${colStr},0)`);
        ctx.beginPath();
        ctx.arc(n.x, n.y, r * 3.5, 0, Math.PI * 2);
        ctx.fillStyle = glow;
        ctx.fill();
        // core
        ctx.beginPath();
        ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${colStr},${0.85 * n.alpha + (flashThis ? flashI * 0.15 : 0)})`;
        ctx.fill();
        // newly-born ring (expands & fades)
        if (n.fresh && n.age < 1500) {
          const ringR = r + 14 * (n.age / 1500);
          const ringA = Math.max(0, 1 - n.age / 1500);
          ctx.beginPath();
          ctx.arc(n.x, n.y, ringR, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(62,207,142,${ringA})`;
          ctx.lineWidth   = 1.5;
          ctx.stroke();
          if (n.age > 1500) n.fresh = false;
        }
      });

      // layer label (uses target X so labels also slide nicely)
      if (layer.length) {
        const x = layer[0].tx;
        ctx.fillStyle = flashThis ? "#3ecf8e" : "#8b9cb3";
        ctx.font = "11px 'DM Sans', sans-serif";
        ctx.textAlign = "center";
        const name =
          li === 0                     ? `Input (${st.layers[li].count})`
        : li === st.layers.length - 1  ? `Output (${st.layers[li].count})`
        :                                `Hidden ${li} (${st.layers[li].count})`;
        ctx.fillText(name, x, H - 8);
      }
    });
  }

  // continuous render loop
  let last = performance.now();
  (function tick(now) {
    const dt = Math.min(50, now - last);
    last = now;
    update(dt);
    draw();
    requestAnimationFrame(tick);
  })(performance.now());

  function reset() {
    st = freshState();
    reconcileNodes(true);
    epochEl.textContent  = "0 / 30";
    accEl.textContent    = "—";
    nodesEl.textContent  = String(totalHiddenNodes());
    paramsEl.textContent = "272K";
    logEl.innerHTML      = '<span class="ev">// click Run Evolution to start</span>';
    btn.textContent      = "▶ Run Evolution";
    btn.disabled         = false;
    pulseT = -1;
    flashLayer = -1;
    flashT = 0;
  }

  function runEpoch(e) {
    if (e >= 30) {
      log(`✓ training complete — final val acc ${DEMO_ACC[29].toFixed(1)}%`, "ev");
      btn.textContent = "↺ Reset";
      btn.disabled    = false;
      st.running      = false;
      return;
    }

    st.epoch = e;
    epochEl.textContent = `${e} / 30`;
    accEl.textContent   = DEMO_ACC[e].toFixed(1) + "%";
    pulseT = performance.now(); // every epoch = one forward-pass animation

    const ev = SCHEDULE.find(s => s.atEpoch === e);
    if (ev) {
      if (ev.kind === "noop") {
        log(`epoch ${e}: 🧠 critic → ${ev.msg}`, "ev");
      } else {
        log(`epoch ${e}: 🧠 critic → ${ev.kind}`, "ev");
        log(`         ⚡ ${ev.msg}`, "mut");
        applyMutation(ev);
        nodesEl.textContent  = String(totalHiddenNodes());
        paramsEl.textContent = approxParams();
        // pause longer so the mutation animation has time to play out
        setTimeout(() => runEpoch(e + 1), 1500);
        return;
      }
    } else if (e % 5 === 0) {
      log(`epoch ${e}: val_acc=${DEMO_ACC[e].toFixed(1)}%`, "ev");
    }

    setTimeout(() => runEpoch(e + 1), 220);
  }

  btn.addEventListener("click", () => {
    if (!st.running && st.epoch === 0) {
      st.running   = true;
      btn.textContent = "Running…";
      btn.disabled    = true;
      logEl.innerHTML = "";
      log("// init: 3-4-10 net, ~272K params", "ev");
      log("// SGD + cosine LR, 30 epochs", "ev");
      runEpoch(0);
    } else if (!st.running) {
      reset();
    }
  });

  resize();
  reset();
}

// ============================================================
//  CIFAR SHOWCASE
// ============================================================
function initCifarShowcase() {
  const grid  = document.getElementById("cifar-grid");
  const panel = document.getElementById("conf-panel");

  CIFAR_SAMPLES.forEach((s, idx) => {
    const tile = document.createElement("div");
    tile.className = "cifar-tile";
    tile.innerHTML = `
      <img src="${s.img}" alt="${s.cls}" loading="lazy" />
      <div class="cifar-tile-label">${s.cls}</div>`;
    tile.addEventListener("click", () => showConf(idx, tile));
    grid.appendChild(tile);
  });

  function showConf(idx, tile) {
    document.querySelectorAll(".cifar-tile").forEach(t => t.classList.remove("active"));
    tile.classList.add("active");
    const s = CIFAR_SAMPLES[idx];
    panel.style.display = "block";

    const rows = s.probs.map(([cls, p]) => {
      const win = cls === s.cls;
      return `<div class="conf-row">
        <div class="conf-lbl ${win ? "win" : ""}">${cls}</div>
        <div class="conf-track"><div class="conf-fill ${win ? "win" : ""}" data-pct="${(p*100).toFixed(1)}"></div></div>
        <div class="conf-pct ${win ? "win" : ""}">${(p*100).toFixed(1)}%</div>
      </div>`;
    }).join("");

    const topConf = (s.probs[0][1] * 100).toFixed(0);
    panel.innerHTML = `
      <h4>Predictions — <span style="color:var(--accent)">${s.cls}</span>
        <span style="font-size:0.75rem;font-weight:400;color:var(--good);margin-left:0.5rem">✓ ${topConf}% confidence</span></h4>
      ${rows}
      <p style="font-size:0.75rem;color:var(--muted);margin-top:1rem">
        CGSE multi-op ResNet-20 · seed 42 · 50 epochs · 0 teacher forwards
        <br><em>Confidence values are representative of a 90.85%-accuracy model on CIFAR-10 class distributions.</em>
      </p>`;

    // Animate bars: reset then trigger via rAF
    setTimeout(() => {
      panel.querySelectorAll(".conf-fill").forEach(bar => {
        bar.style.width = bar.dataset.pct + "%";
      });
    }, 50);
  }
}

// ============================================================
//  CHARTS
// ============================================================
Chart.defaults.color       = "#8b9cb3";
Chart.defaults.borderColor = "rgba(255,255,255,0.06)";
Chart.defaults.font.family = "'DM Sans', system-ui, sans-serif";

// Custom plugin: vertical dashed line annotation
const vlinePlugin = {
  id: "vline",
  afterDraw(chart) {
    const epoch = chart.config.options.vlineEpoch;
    if (epoch == null) return;
    const { ctx, chartArea: ca, scales: { x } } = chart;
    const px = x.getPixelForValue(epoch);
    ctx.save();
    ctx.beginPath(); ctx.moveTo(px, ca.top); ctx.lineTo(px, ca.bottom);
    ctx.strokeStyle = "rgba(232,168,73,0.45)";
    ctx.lineWidth = 1.5; ctx.setLineDash([5,4]);
    ctx.stroke();
    ctx.fillStyle = "rgba(232,168,73,0.7)";
    ctx.font = "11px DM Sans, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("LR drop", px, ca.top + 14);
    ctx.restore();
  },
};
Chart.register(vlinePlugin);

function initCharts() {
  // Test 1 bar
  new Chart(document.getElementById("chart-t1"), {
    type: "bar",
    data: {
      labels: T1_ARMS.map(a => a.label),
      datasets: [{
        label: "Best val acc (%)",
        data: T1_ARMS.map(a => a.acc),
        backgroundColor: T1_ARMS.map(a => a.color),
        barThickness: 20, maxBarThickness: 24, borderWidth: 0, borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        title: { display: true, text: "Test 1 — best val accuracy (%, seed 42, CifarGraphNet)" },
      },
      scales: {
        y: { min: 83.5, max: 85.5, grid: { color: "rgba(255,255,255,0.06)" },
             ticks: { color: "#8b9cb3", callback: v => v + "%" } },
        x: { grid: { display: false }, ticks: { color: "#8b9cb3", maxRotation: 30 } },
      },
    },
  });

  // Test 2 bar
  new Chart(document.getElementById("chart-t2-bar"), {
    type: "bar",
    data: {
      labels: T2_ROWS.map(r => r.label),
      datasets: [{
        label: "Best val acc (%)",
        data: T2_ROWS.map(r => r.acc),
        backgroundColor: T2_ROWS.map(r => r.color),
        barThickness: 26, maxBarThickness: 32, borderWidth: 0, borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        title: { display: true, text: "Test 2 — best val accuracy (%, seed 42, ResNet track, 50 epochs)" },
      },
      scales: {
        y: { min: 88, max: 94, grid: { color: "rgba(255,255,255,0.06)" },
             ticks: { color: "#8b9cb3", callback: v => v + "%" } },
        x: { grid: { display: false }, ticks: { color: "#8b9cb3" } },
      },
    },
  });

  // Test 2 learning curves
  new Chart(document.getElementById("chart-t2-lines"), {
    type: "line",
    data: {
      datasets: [
        { label: "Teacher (ResNet-56)",
          data: T2_CURVES.teacher.map((v,i) => ({x:i,y:v})),
          borderColor: "#e8a849", backgroundColor: "transparent", tension: 0.15, pointRadius: 0, borderWidth: 2 },
        { label: "Student KD (ResNet-20)",
          data: T2_CURVES.kd.map((v,i) => ({x:i,y:v})),
          borderColor: "#5b9fd4", backgroundColor: "transparent", tension: 0.15, pointRadius: 0, borderWidth: 2 },
        { label: "CGSE multi-op (ResNet-20)",
          data: T2_CURVES.cgse.map((v,i) => ({x:i,y:v})),
          borderColor: "#3ecf8e", backgroundColor: "transparent", tension: 0.15, pointRadius: 0, borderWidth: 2.5 },
      ],
    },
    options: {
      responsive: true,
      parsing: false,
      vlineEpoch: 25,
      plugins: {
        title: { display: true, text: "Test 2 — val accuracy over 50 epochs (seed 42) · vertical line = LR drop" },
        legend: { labels: { color: "#8b9cb3", boxWidth: 16, padding: 18 } },
      },
      scales: {
        x: { type: "linear", min: 0, max: 49,
             title: { display: true, text: "Epoch", color: "#8b9cb3" },
             grid: { color: "rgba(255,255,255,0.06)" }, ticks: { color: "#8b9cb3" } },
        y: { min: 20, max: 94,
             title: { display: true, text: "Val acc %", color: "#8b9cb3" },
             grid: { color: "rgba(255,255,255,0.06)" },
             ticks: { color: "#8b9cb3", callback: v => v + "%" } },
      },
    },
  });
}

// ============================================================
//  ANIMATED COUNTERS
// ============================================================
function animateCounter(el, target, dur = 1700) {
  const isFloat = String(target).includes(".");
  const dec     = isFloat ? String(target).split(".")[1].length : 0;
  let start = null;
  (function tick(ts) {
    if (!start) start = ts;
    const p = Math.min((ts - start) / dur, 1);
    const e = 1 - Math.pow(1 - p, 3);
    el.textContent = isFloat
      ? (e * target).toFixed(dec)
      : Math.round(e * target).toLocaleString();
    if (p < 1) requestAnimationFrame(tick);
  })(performance.now());
}

function initCounters() {
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        animateCounter(e.target, parseFloat(e.target.dataset.count));
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.5 });
  document.querySelectorAll("[data-count]").forEach(el => obs.observe(el));
}

// ============================================================
//  PROBLEM SECTION COST METERS
// ============================================================
function initMeters() {
  const kdFill = document.getElementById("kd-fill");
  const kdNum  = document.getElementById("kd-num");
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      const dur = 2400, target = 19550, t0 = performance.now();
      (function tick(now) {
        const p = Math.min((now - t0) / dur, 1);
        const v = Math.round((1 - Math.pow(1 - p, 2)) * target);
        kdFill.style.width = ((v / target) * 100) + "%";
        kdNum.textContent  = v.toLocaleString();
        if (p < 1) requestAnimationFrame(tick);
      })(performance.now());
      obs.unobserve(e.target);
    });
  }, { threshold: 0.35 });
  const prob = document.getElementById("problem");
  if (prob) obs.observe(prob);
}

// ============================================================
//  SCROLL REVEAL
// ============================================================
function initReveal() {
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => { if (e.isIntersecting) e.target.classList.add("visible"); });
  }, { threshold: 0.08, rootMargin: "0px 0px -30px 0px" });
  document.querySelectorAll(".reveal").forEach(el => obs.observe(el));
}

// ============================================================
//  INIT
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
  const heroCanvas = document.getElementById("hero-canvas");
  const indicator  = document.getElementById("mutation-indicator");
  if (heroCanvas && heroCanvas.getContext) {
    new NetworkCanvas(heroCanvas, indicator).start();
  }
  initReveal();
  initStepper();
  initMutationDemo();
  initCifarShowcase();
  initCharts();
  initCounters();
  initMeters();
});
