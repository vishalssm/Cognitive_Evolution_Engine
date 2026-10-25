/* ==========================================================================
   Cognitive Evolution Engine — dashboard client
   Server-Sent Events in, DOM out. No framework, no build step.
   ========================================================================== */

"use strict";

const $ = (id) => document.getElementById(id);

const AXIS_SHORT = {
  wheel_rpm_avg: "rpm",
  imu_accel_x: "accel_x",
  imu_pitch_deg: "pitch",
  ultrasonic_cm: "radar_cm",
  system_temp_c: "cpu_temp",
  kin_residual: "kin_resid",
  stream_ok: "stream_ok",
};

const STAGE_ORDER = ["sense", "detect", "verify", "cognition", "act"];
const WORKING_LABEL = { 1: "Working 1 · Induction", 2: "Working 2 · Recall" };
const PHASE_CLASS = { induce: "induce", recall: "recall", verify: "verify", warn: "warn", ok: "" };

const ui = {
  state: null,
  prevHistoryLen: 0,
  kbSort: { key: "confidence", dir: -1 },
  kbOpen: null,
  lastHazard: null,
  hazardIdx: 0,
  paused: false,
  hover: null,
  panelErrors: {},
};

// ---------------------------------------------------------------- utilities

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const esc = (s) =>
  String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
const num = (v, d = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d);
const clockOf = (t) => new Date((t || 0) * 1000).toLocaleTimeString([], { hour12: false });
const pct = (v) => `${Math.round((v || 0) * 100)}%`;

function fps(fpsValue) {
  const st = ui.state;
  if (!st) return;
  $("frame-text").textContent = st.counters.frames;
}

// ---------------------------------------------------------------- transport

async function post(body) {
  try {
    const res = await fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return await res.json();
  } catch (err) {
    console.error("control failed", err);
    return { ok: false, error: String(err) };
  }
}

function connect() {
  const src = new EventSource("/api/stream");
  src.onopen = () => {
    setLink(true);
  };
  src.onmessage = (ev) => {
    try {
      render(JSON.parse(ev.data));
    } catch (err) {
      console.error("bad frame", err);
    }
  };
  src.onerror = () => {
    setLink(false);
  };
}

function setLink(up) {
  const pill = $("pill-link");
  const dot = pill.querySelector("i");
  dot.className = "dot " + (up ? "live" : "crit");
  pill.querySelector("b").textContent = up ? "live" : "lost";
}

// ------------------------------------------------------------------ top bar

function renderChrome(s) {
  $("session-text").textContent = s.meta.session;
  $("frame-text").textContent = s.counters.frames;
  $("pill-status").querySelector("#status-text").textContent = s.status.label;
  $("pill-status").querySelector("#status-dot").className = "dot " + statusDot(s.status.level);

  const t = s.transport;
  $("link-text").textContent = t.mode === "mqtt" ? `MQTT ${t.connected ? "up" : "down"}` : "local";

  $("foot-session").textContent = `session ${s.meta.session} · v${s.meta.version} · up ${Math.floor(
    s.meta.uptime / 60
  )}m${String(Math.floor(s.meta.uptime % 60)).padStart(2, "0")}s`;
  $("foot-transport").textContent = `transport ${t.mode}${
    t.topic ? " · " + t.topic : ""
  } · ${t.published} tx / ${t.received} rx`;
  $("foot-memory").textContent = `memory ${s.knowledge.count} rules · ${s.meta.memory_file.split(/[\\/]/).pop()}`;

  $("btn-pause").textContent = s.scenario.paused ? "Resume" : "Pause";
  $("btn-pause").classList.toggle("on", !!s.scenario.paused);
  $("btn-auto").classList.toggle("on", !!s.scenario.auto);

  ui.lastHazard = s.scenario.last_hazard || ui.lastHazard;
}

function statusDot(level) {
  if (level === "nominal") return "live";
  if (level === "acting") return "warn";
  if (level === "critical" || level === "anomalous") return "crit";
  return "";
}

// --------------------------------------------------------------------- KPIs

function renderKpis(s) {
  const k = s.knowledge;
  const st = k.stats || {};

  $("k-rules").textContent = k.count;
  $("k-recalls").textContent = st.recalls || 0;
  $("k-conf").textContent = num(k.mean_confidence, 2);
  $("k-rate").textContent = pct(k.recall_rate);
  $("k-ok").textContent = k.helped;
  $("k-bad").textContent = Math.max(0, (k.judged || 0) - (k.helped || 0));

  $("k-rules-d").textContent =
    (st.inductions || 0) + " induction event(s) so far";
  $("k-recalls-d").textContent = (st.anomalies_seen || 0) + " anomalies met";
  $("k-conf-d").textContent =
    (st.novel_detections || 0) > 0
      ? (st.novel_detections || 0) + " needed a new rule"
      : "no unhandled anomaly yet";
  $("k-rate-d").textContent =
    (st.anomalies_seen || 0) > 0
      ? `${(st.recalls || 0)} solved from memory`
      : "no anomaly observed yet";
  $("k-ok-d").textContent = (k.judged || 0) + " verification(s)";
  $("k-bad-d").textContent = "actions demoted";
}

// ----------------------------------------------------------- cognitive loop

function renderPipeline(s) {
  const cycle = s.cycle || {};
  const stage = cycle.stage || "idle";
  const el = $("pipeline");

  const hasInd = !!s.induction;
  const hasRec = !!s.recall;

  let html = "";
  for (const key of STAGE_ORDER) {
    const meta = (s.pipeline || []).find((p) => p.key === key);
    if (!meta) continue;
    let cls = "";
    if (key === stage) cls = stage === "recall" ? "w2" : "active";
    // A revision is still Working 1: the engine is rewriting a rule rather
    // than matching one, so it lights the induction branch, not a new stage.
    if (key === "cognition" && (stage === "induce" || stage === "revise")) cls = "w1";
    if (key === "verify" && stage === "verify") cls = "active";
    html += `<div class="stage ${cls}">
      <div class="idx">${String(STAGE_ORDER.indexOf(key) + 1).padStart(2, "0")}</div>
      <div class="nm">${esc(meta.title)}</div>
      <div class="ds">${esc(meta.desc)}</div>
    </div>`;
  }

  const litW1 = stage === "induce" || stage === "revise" ? "lit w1" : "";
  const litW2 = stage === "recall" ? "lit w2" : "";
  const w1Label = stage === "revise" ? "Working 1 · Revision" : "Working 1 · Induction";
  const w1Text = !hasInd
    ? "No rule written yet"
    : stage === "revise"
    ? `Revised to ${esc(s.induction.rule.id)}`
    : `Last rule: ${esc(s.induction.rule.id)}`;
  html += `<div class="branch">
    <div class="leaf ${litW1}">
      <div class="nm">${w1Label}</div>
      <div class="ds">${w1Text}</div>
    </div>
    <div class="leaf ${litW2}">
      <div class="nm">Working 2 · Recall</div>
      <div class="ds">${hasRec ? `Last match: ${esc(s.recall.chosen.rule_id)}` : "No rule matched yet"}</div>
    </div>
  </div>`;

  el.innerHTML = html;

  const tag = $("loop-tag");
  tag.textContent = stage;
  tag.className =
    "tag " +
    (stage === "induce" || stage === "revise"
      ? "amber"
      : stage === "recall"
      ? "green"
      : stage === "verify"
      ? "cyan"
      : "");

  const nar = $("narration");
  nar.innerHTML = highlight(s.narration);
  const ncls =
    stage === "induce" || stage === "revise"
      ? "w1"
      : stage === "recall"
      ? "w2"
      : s.status.level === "anomalous"
      ? "warn"
      : "";
  nar.className = "narration " + ncls;
}

function highlight(text) {
  if (!text) return "Waiting for telemetry…";
  return esc(text).replace(/\b(RULE_\d+|Working [12])\b/g, "<b>$1</b>");
}

// ------------------------------------------------------------ live telemetry

function renderGauges(s) {
  const axes = s.axes || [];
  const disp = s.display || {};
  const telem = s.telemetry || {};

  $("telemetry-tag").textContent = `${axes.length} channels${
    telem.stream_ok ? "" : " · link down"
  }`;

  const html = axes
    .map((ax) => {
      const d = disp[ax.key] || {};
      const dev = d.deviation || 0;
      let cls = "ok";
      if (d.invalid) cls = "invalid";
      else if (d.masked) cls = "warn";
      else if (Math.abs(dev) >= 1) cls = "danger";
      else if (Math.abs(dev) > 0.3) cls = "warn";
      return `<div class="gauge ${cls}" data-axis="${esc(ax.key)}">
        <canvas width="132" height="86" data-gauge="${esc(ax.key)}"
          data-value="${d.value}" data-dev="${dev}"
          data-lo="${ax.range ? ax.range[0] : -1}" data-hi="${ax.range ? ax.range[1] : 1}"
          data-n0="${ax.nominal ? ax.nominal[0] : 0}" data-n1="${ax.nominal ? ax.nominal[1] : 0}"
          data-w0="${ax.warn ? ax.warn[0] : 0}" data-w1="${ax.warn ? ax.warn[1] : 0}"
          data-invalid="${d.invalid ? 1 : 0}" data-masked="${d.masked ? 1 : 0}"></canvas>
        <div class="gl">${esc(ax.label)}</div>
        <div class="gv">${d.value === null ? "—" : num(d.value, 2)}<span class="gu"> ${esc(ax.unit || "")}</span></div>
      </div>`;
    })
    .join("");

  $("gauges").innerHTML = html;
  for (const cv of $("gauges").querySelectorAll("canvas")) drawGauge(cv);

  renderDeviationRows(s, axes, disp);
}

function renderDeviationRows(s, axes, disp) {
  const rows = axes
    .map((ax) => {
      const d = disp[ax.key] || {};
      const dev = clamp(d.deviation || 0, -1, 1);
      const w = Math.abs(dev) * 50;
      const left = dev >= 0 ? 50 : 50 - w;
      const color =
        Math.abs(dev) >= 1 ? "var(--red)" : Math.abs(dev) > 0.3 ? "var(--amber)" : "var(--green)";
      const band = ax.nominal
        ? `[${num(ax.nominal[0], 0)} … ${num(ax.nominal[1], 0)}]`
        : "—";
      return `<div class="row">
        <span class="rl">${esc(ax.label)}</span>
        <span class="rgrow"><span class="bar">
          <i class="mid"></i>
          <span style="left:${left}%;width:${w}%;background:${color}"></span>
        </span></span>
        <span class="rv">${d.value === null ? "—" : num(d.value, 2)} ${esc(ax.unit || "")}</span>
        <span class="rv dim" style="width:96px">band ${band}</span>
      </div>`;
    })
    .join("");
  $("anomaly-rows").innerHTML = rows;
}

// ------------------------------------------------------------------- gauges

function drawGauge(cv) {
  const ctx = cv.getContext("2d");
  const W = cv.width;
  const H = cv.height;
  const cx = W / 2;
  const cy = H - 12;
  const R = 46;
  const lo = Number(cv.dataset.lo);
  const hi = Number(cv.dataset.hi);
  const value = Number(cv.dataset.value);
  const dev = Number(cv.dataset.dev);
  const invalid = cv.dataset.invalid === "1";
  const masked = cv.dataset.masked === "1";
  const n0 = Number(cv.dataset.n0);
  const n1 = Number(cv.dataset.n1);
  const w0 = Number(cv.dataset.w0);
  const w1 = Number(cv.dataset.w1);

  const frac = (v) => clamp((v - lo) / (hi - lo || 1), 0, 1);
  const ang = (f) => Math.PI * (1 - f); // 180deg sweep, left to right

  const colour = invalid
    ? "#a78bfa"
    : masked
    ? "#fbbf24"
    : Math.abs(dev) >= 1
    ? "#f87171"
    : Math.abs(dev) > 0.3
    ? "#fbbf24"
    : "#34d399";

  ctx.clearRect(0, 0, W, H);

  // track
  ctx.lineWidth = 7;
  ctx.lineCap = "butt";
  ctx.strokeStyle = "#141d27";
  ctx.beginPath();
  ctx.arc(cx, cy, R, Math.PI, 0);
  ctx.stroke();

  // warn band
  if (w1 > w0) {
    ctx.strokeStyle = "rgba(251,191,36,0.34)";
    ctx.lineWidth = 7;
    ctx.beginPath();
    ctx.arc(cx, cy, R, ang(frac(w1)), ang(frac(w0)));
    ctx.stroke();
  }

  // nominal band
  ctx.strokeStyle = "rgba(52,211,153,0.4)";
  ctx.lineWidth = 7;
  ctx.beginPath();
  ctx.arc(cx, cy, R, ang(frac(n1)), ang(frac(n0)));
  ctx.stroke();

  // ticks
  ctx.strokeStyle = "#1e2a36";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const a = Math.PI + (Math.PI * i) / 4;
    const r0 = R - 5;
    const r1 = R - 11;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
    ctx.lineTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
    ctx.stroke();
  }

  // needle
  const f = invalid ? 0.5 : clamp(frac(value), 0, 1);
  const a = ang(f);
  ctx.strokeStyle = colour;
  ctx.lineWidth = 2.4;
  ctx.shadowColor = colour;
  ctx.shadowBlur = 9;
  ctx.beginPath();
  ctx.moveTo(cx + Math.cos(a + Math.PI / 2) * 6, cy + Math.sin(a + Math.PI / 2) * 6);
  ctx.lineTo(cx + Math.cos(a) * (R - 6), cy + Math.sin(a) * (R - 6));
  ctx.stroke();
  ctx.shadowBlur = 0;

  // hub
  ctx.fillStyle = colour;
  ctx.beginPath();
  ctx.arc(cx, cy, 3.4, 0, Math.PI * 2);
  ctx.fill();

  // masked hatch
  if (masked || invalid) {
    ctx.strokeStyle = masked ? "rgba(251,191,36,0.5)" : "rgba(167,139,250,0.55)";
    ctx.lineWidth = 1;
    for (let i = -H; i < W; i += 6) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i + H, H);
      ctx.stroke();
    }
    ctx.fillStyle = "rgba(5,7,11,0.62)";
    ctx.fillRect(0, 0, W, H);
  }
}

// ------------------------------------------------------------- interventions

function renderInterventions(s) {
  const a = s.actuators || {};
  const list = [...(a.active || []), ...(a.history || []).slice(0, 4)];
  const box = $("interventions");
  if (!list.length) {
    box.innerHTML = `<div class="empty" style="padding:14px">No actuator command issued yet</div>`;
    return;
  }
  box.innerHTML = list
    .map((it) => {
      const life = it.hold_total ? clamp(it.frames_left / it.hold_total, 0, 1) : 0;
      const active = it.frames_left > 0;
      return `<div class="intv">
        <span class="tag ${active ? "cyan" : ""}">${active ? "ACTIVE" : "PAST"}</span>
        <span class="ib">
          <span class="inm">${esc(it.label)}</span>
          <span class="irule">${esc(it.rule_id || "—")}${
            it.masks && it.masks.length ? " · masked " + it.masks.join(", ") : ""
          }${it.boosts && Object.keys(it.boosts).length ? " · boost " + Object.keys(it.boosts).join(", ") : ""}</span>
        </span>
        <span class="ibar"><span class="bar"><span style="left:0;width:${
          life * 100
        }%"></span></span></span>
        <span class="mono fs11 dim nowrap">${it.frames_left} f</span>
      </div>`;
    })
    .join("");
}

// ------------------------------------------------------------------- chart

const CHART = { series: [] };

const CHART_SPEC = [
  { key: "wheel_rpm_avg", colour: "#38bdf8", label: "RPM", scale: 1 },
  { key: "imu_accel_x", colour: "#a78bfa", label: "Accel X", scale: 1 },
  { key: "ultrasonic_cm", colour: "#34d399", label: "Radar", scale: 1 },
  { key: "system_temp_c", colour: "#fbbf24", label: "CPU temp", scale: 1 },
];

function renderChart(s) {
  const cv = $("chart");
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth || 900;
  const cssH = 360;
  if (cv.width !== Math.round(cssW * dpr)) {
    cv.width = Math.round(cssW * dpr);
    cv.height = Math.round(cssH * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const hist = s.history || [];
  if (hist.length < 2) {
    ctx.fillStyle = "#5a6b7d";
    ctx.font = "12px var(--mono), monospace";
    ctx.fillText("collecting telemetry…", 16, cssH / 2);
    return;
  }

  const padL = 46;
  const padR = 14;
  const padT = 14;
  const padB = 22;
  const plotW = cssW - padL - padR;
  const plotH = cssH - padT - padB;
  const rows = 4;
  const rowH = plotH / rows;

  // Collect normalised value per cell: value mapped into its own axis range.
  const cells = [];
  for (let r = 0; r < rows; r++) {
    const spec = CHART_SPEC[r];
    const ax = (s.axes || []).find((a) => a.key === spec.key);
    const lo = ax && ax.range ? ax.range[0] : 0;
    const hi = ax && ax.range ? ax.range[1] : 1;
    const pts = hist.map((h) => {
      const raw = h.v[spec.key];
      const v = raw === null || raw === undefined ? null : clamp((raw - lo) / (hi - lo || 1), 0, 1);
      return v === null ? null : padT + (1 - v) * rowH + r * rowH;
    });
    cells.push({ spec, lo, hi, pts });
  }

  // grid + row labels
  ctx.font = "10px ui-monospace, monospace";
  ctx.textBaseline = "middle";
  for (let r = 0; r < rows; r++) {
    const y0 = padT + r * rowH;
    ctx.strokeStyle = "rgba(26,36,48,0.85)";
    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const x = padL + (plotW * g) / 4;
      ctx.beginPath();
      ctx.moveTo(x, y0);
      ctx.lineTo(x, y0 + rowH);
      ctx.stroke();
    }
    const c = cells[r];
    ctx.fillStyle = "#5a6b7d";
    ctx.textAlign = "right";
    ctx.fillText(`${c.spec.label} ${num(c.hi, 0)}`, padL - 6, y0 + 9);
    ctx.fillText(`${num(c.lo, 0)}`, padL - 6, y0 + rowH - 9);
    ctx.strokeStyle = "rgba(36,50,63,0.9)";
    ctx.beginPath();
    ctx.moveTo(padL, y0 + rowH);
    ctx.lineTo(padL + plotW, y0 + rowH);
    ctx.stroke();
  }

  const xAt = (i) => padL + (plotW * i) / (hist.length - 1);

  // anomaly shading
  hist.forEach((h, i) => {
    if (!h.anomalous) return;
    ctx.fillStyle = "rgba(248,113,113,0.10)";
    ctx.fillRect(xAt(i) - 1.5, padT, 3, plotH);
  });

  // rule dispatch ticks
  hist.forEach((h, i) => {
    if (!h.rule) return;
    const x = xAt(i);
    ctx.strokeStyle = "rgba(56,189,248,0.55)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, padT + plotH);
    ctx.lineTo(x, padT + plotH - 12);
    ctx.stroke();
  });

  // series
  for (const cell of cells) {
    ctx.strokeStyle = cell.spec.colour;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    ctx.beginPath();
    let started = false;
    cell.pts.forEach((y, i) => {
      if (y === null) return;
      const x = xAt(i);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // hover crosshair
  if (ui.hover !== null && ui.hover >= 0 && ui.hover < hist.length) {
    const i = ui.hover;
    const x = xAt(i);
    ctx.strokeStyle = "rgba(230,237,243,0.5)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, padT + plotH);
    ctx.stroke();
    ctx.setLineDash([]);
    for (const cell of cells) {
      const y = cell.pts[i];
      if (y === null) continue;
      ctx.fillStyle = cell.spec.colour;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
    const ro = $("chart-readout");
    const h = hist[i];
    ro.innerHTML =
      `t-${hist.length - 1 - i} · ${h.rule ? esc(h.rule) : h.anomalous ? "anomaly" : "nominal"}<br>` +
      CHART_SPEC.map((sp) => `${sp.label} ${num(h.v[sp.key], 1)}`).join("  ");
  }

  CHART.series = { cells, hist, padL, plotW };
  $("timeline-hint").textContent = `${hist.length} frames · hover to inspect`;
}

function chartHit(ev) {
  const cv = $("chart");
  const rect = cv.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const s = CHART.series;
  if (!s || !s.hist) return;
  const rel = (x - s.padL) / (s.plotW || 1);
  ui.hover = clamp(Math.round(rel * (s.hist.length - 1)), 0, s.hist.length - 1);
}

// -------------------------------------------------------------- working 1/2

function renderWorkingOne(s) {
  const box = $("w1-body");
  const tag = $("w1-tag");
  const ind = s.induction;
  if (!ind) {
    box.innerHTML = `<div class="empty">Waiting for the first anomaly</div>`;
    tag.textContent = "no event yet";
    return;
  }
  const r = ind.rule;
  const best = (ind.candidates || [])[0] || {};
  const won = best.key === r.action;

  tag.textContent = `${r.id} · ${clockOf(r.created_at)}`;
  tag.className = "tag amber";

  const devRows = (ind.deviation_table || [])
    .map(
      (d) => `<div class="row">
        <span class="rl">${esc(d.label)}</span>
        <span class="rgrow"><span class="bar"><i class="mid"></i><span style="left:${
          d.deviation >= 0 ? 50 : 50 + d.deviation * 50
        }%;width:${Math.abs(d.deviation) * 50}%;background:${
        Math.abs(d.deviation) >= 1 ? "var(--red)" : "var(--amber)"
      }"></span></span></span>
        <span class="rv">${num(d.value, 2)}</span>
        <span class="rv dim" style="width:92px">band ${num(d.nominal[0], 0)}…${num(
          d.nominal[1],
          0
        )}</span>
      </div>`
    )
    .join("");

  const obs = ind.observation || {};
  const chips = Object.keys(obs)
    .map(
      (k) =>
        `<span class="chip">${esc(AXIS_SHORT[k] || k)} <b>${num(obs[k], 2)}</b></span>`
    )
    .join("");

  const cands = (ind.candidates || [])
    .slice(0, 6)
    .map((c) => {
      const cls = c.key === r.action ? "win" : !c.feasible ? "dead" : "";
      return `<tr class="${cls}">
        <td class="nm">${c.key === r.action ? "▸ " : ""}${esc(c.key)}</td>
        <td class="sc dim">${num(c.current_residual, 2)} → ${num(c.projected_residual, 2)}</td>
        <td class="sc dim">${num(c.cost, 2)}</td>
        <td class="sc">${num(c.score, 2)}</td>
      </tr>`;
    })
    .join("");

  box.innerHTML = `
    <div class="narration w1" style="margin-top:0">${esc(ind.explanation || "")}</div>

    <div style="margin-top:14px">
      <div class="rl" style="font-family:var(--mono);font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--dim);margin-bottom:6px">1 · Channels that left their band</div>
      <div class="rows">${devRows || `<div class="empty">none</div>`}</div>
      <div class="chips" style="margin-top:8px">${chips}</div>
    </div>

    <div style="margin-top:14px">
      <div class="rl" style="font-family:var(--mono);font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--dim);margin-bottom:6px">2 · Symbol synthesised</div>
      <div class="chips">${(r.conditions || [])
        .map(
          (c) =>
            `<span class="chip">${esc(c.text)}<br><span class="dim" style="font-size:9.5px">${
              c.role || ""
            }</span></span>`
        )
        .join("")}</div>
    </div>

    <div style="margin-top:14px">
      <div class="rl" style="font-family:var(--mono);font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--dim);margin-bottom:6px">3 · Remedy chosen by forward simulation</div>
      <table class="cand">
        <thead><tr><th>Primitive</th><th class="sc">Residual</th><th class="sc">Cost</th><th class="sc">Score</th></tr></thead>
        <tbody>${cands}</tbody>
      </table>
      ${
        won
          ? `<div class="fs11 muted" style="margin-top:7px">Predicted to cut the residual from <b>${num(
              best.current_residual,
              2
            )}</b> to <b>${num(best.projected_residual, 2)}</b> at cost ${num(
              best.cost,
              2
            )} — the best trade-off among ${(ind.candidates || []).length} primitives.</div>`
          : `<div class="fs11 muted" style="margin-top:7px">Top-scoring candidate was <b>${esc(
              best.key
            )}</b>; the bound rule uses <b>${esc(r.action)}</b>.</div>`
      }
    </div>

    <div style="margin-top:14px">
      <div class="rl" style="font-family:var(--mono);font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--dim);margin-bottom:6px">4 · Rule written to memory</div>
      <div class="mono fs11" style="color:var(--muted);line-height:1.6;word-break:break-word">${esc(
        r.text
      )}</div>
      <div class="fs11 muted" style="margin-top:6px">${esc(r.rationale || "")}</div>
      <div class="fs11 dim" style="margin-top:6px">Initial confidence ${num(
        r.confidence,
        2
      )} · specificity ${num(r.specificity, 2)} · signature ${esc(r.signature)}</div>
    </div>`;
}

function renderWorkingTwo(s) {
  const box = $("w2-body");
  const tag = $("w2-tag");
  const rec = s.recall;
  if (!rec) {
    box.innerHTML = `<div class="empty">Waiting for the first anomaly</div>`;
    tag.textContent = "no event yet";
    return;
  }
  const ch = rec.chosen;
  const tagCls = "tag green";

  tag.textContent = `${ch.rule_id} · confidence ${num(ch.confidence, 2)}`;

  const conds = (rec.rule.conditions || [])
    .map((c, i) => {
      const ok = (ch.condition_results || [])[i];
      return `<span class="chip ${ok ? "hit" : "miss"}">${ok ? "✓" : "✕"} ${esc(c.text)}</span>`;
    })
    .join("");

  const others = (rec.considered || [])
    .filter((c) => c.rule_id !== ch.rule_id)
    .slice(0, 4)
    .map(
      (c) => `<div class="row">
        <span class="rl">${esc(c.rule_id)}</span>
        <span class="rgrow"><span class="bar"><span style="left:0;width:${
          (c.score || 0) * 100
        }%;background:var(--edge-hi)"></span></span></span>
        <span class="rv">${c.satisfied}/${c.of}</span>
        <span class="rv dim" style="width:56px">${num(c.confidence, 2)}</span>
      </div>`
    )
    .join("");

  const rf = rec.reinforcement || {};
  const trace = rf.trace || [];
  const maxT = Math.max(0.55, ...trace);

  box.innerHTML = `
    <div class="narration w2" style="margin-top:0">${esc(rec.why || "")}</div>

    <div style="margin-top:14px">
      <div class="rl" style="font-family:var(--mono);font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--dim);margin-bottom:6px">1 · Predicate matched against live telemetry</div>
      <div class="chips">${conds || "—"}</div>
    </div>

    <div style="margin-top:14px">
      <div class="rl" style="font-family:var(--mono);font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--dim);margin-bottom:6px">2 · Ranked by confidence × specificity</div>
      <div class="rows">
        <div class="row">
          <span class="rl" style="color:var(--green)">${esc(ch.rule_id)}</span>
          <span class="rgrow"><span class="bar"><span style="left:0;width:${
            (ch.score || 0) * 100
          }%;background:var(--green)"></span></span></span>
          <span class="rv">${ch.satisfied}/${ch.of}</span>
          <span class="rv" style="width:56px;color:var(--green)">${num(ch.confidence, 2)}</span>
        </div>
        ${others || ""}
      </div>
      <div class="fs11 dim" style="margin-top:6px">${rec.considered_count} rule(s) considered</div>
    </div>

    <div style="margin-top:14px">
      <div class="rl" style="font-family:var(--mono);font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--dim);margin-bottom:6px">3 · Action dispatched without asking</div>
      <div class="mono fs11" style="color:var(--green)">▸ ${esc(rec.primitive.title)}</div>
      <div class="fs11 muted" style="margin-top:4px">${esc(rec.primitive.consequence)}</div>
    </div>

    <div style="margin-top:14px">
      <div class="rl" style="font-family:var(--mono);font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--dim);margin-bottom:6px">4 · Confidence carried forward</div>
      <div class="chart"><canvas id="reinforce" height="70" data-trace="${esc(
        trace.join(",")
      )}" data-max="${maxT}"></canvas></div>
      <div class="fs11 dim" style="margin-top:5px">
        ${rf.judged || 0} verification(s) · ${rf.applied || 0} application(s) · success ${pct(
          rf.success_rate
        )}
      </div>
    </div>`;

  const cv = $("reinforce");
  if (cv) drawTrace(cv, trace, maxT);
}

function drawTrace(cv, trace, max) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 320;
  const h = 70;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = "rgba(36,50,63,0.9)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 2; i++) {
    const y = 6 + ((h - 14) * i) / 2;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  if (trace.length < 2) {
    ctx.fillStyle = "#5a6b7d";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText("awaiting second observation", 6, h / 2);
    return;
  }

  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(52,211,153,0.34)");
  grad.addColorStop(1, "rgba(52,211,153,0)");

  const pt = (i, v) => [
    (w * i) / (trace.length - 1),
    6 + (1 - clamp(v / max, 0, 1)) * (h - 14),
  ];

  ctx.beginPath();
  ctx.moveTo(0, h);
  trace.forEach((v, i) => ctx.lineTo(...pt(i, v)));
  ctx.lineTo(w, h);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.beginPath();
  trace.forEach((v, i) => (i ? ctx.lineTo(...pt(i, v)) : ctx.moveTo(...pt(i, v))));
  ctx.strokeStyle = "#34d399";
  ctx.lineWidth = 1.8;
  ctx.stroke();

  const last = pt(trace.length - 1, trace[trace.length - 1]);
  ctx.fillStyle = "#34d399";
  ctx.beginPath();
  ctx.arc(last[0], last[1], 3, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#8b9aab";
  ctx.font = "10px ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.fillText(num(trace[trace.length - 1], 2), w - 4, last[1] - 7);
}

// -------------------------------------------------------------- verification

function renderVerification(s) {
  const v = s.verification;
  const box = $("verify-body");
  if (!v) {
    box.innerHTML = `<div class="empty">No rule has been verified yet</div>`;
    return;
  }
  const helped = v.verdict === "helped";
  const accuracy = v.expected ? clamp(1 - Math.abs(v.residual_after - v.expected) / Math.max(v.expected, 0.1), 0, 1) : 0;

  box.innerHTML = `
    <div class="grid g-2e" style="margin-top:0">
      <div>
        <div class="rl" style="font-family:var(--mono);font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--dim);margin-bottom:8px">Measured against the rule's own prediction</div>
        <div class="rows">
          <div class="row"><span class="rl">Rule</span><span class="rgrow"></span><span class="rv" style="color:var(--cyan)">${esc(
            v.rule_id
          )}</span></div>
          <div class="row"><span class="rl">Action</span><span class="rgrow"></span><span class="rv">${esc(
            v.title
          )}</span></div>
          <div class="row"><span class="rl">Residual before</span><span class="rgrow"></span><span class="rv">${num(
            v.residual_before,
            2
          )}</span></div>
          <div class="row"><span class="rl">Predicted after</span><span class="rgrow"></span><span class="rv">${num(
            v.expected,
            2
          )}</span></div>
          <div class="row"><span class="rl">Actually after</span><span class="rgrow"></span><span class="rv" style="color:${
            helped ? "var(--green)" : "var(--amber)"
          }">${num(v.residual_after, 2)}</span></div>
        </div>
      </div>
      <div>
        <div class="rl" style="font-family:var(--mono);font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--dim);margin-bottom:8px">Reinforcement</div>
        <div class="outcome ${helped ? "ok" : "bad"}" style="margin-bottom:10px">${esc(
          v.detail.charAt(0).toUpperCase() + v.detail.slice(1)
        )}.</div>
        <div class="row">
          <span class="rl">Verdict</span>
          <span class="rgrow"></span>
          <span class="tag ${helped ? "green" : "amber"}">${esc(v.verdict)}</span>
        </div>
        <div class="row">
          <span class="rl">Confidence</span>
          <span class="rgrow"><span class="bar"><span style="left:0;width:${
            (v.confidence || 0) * 100
          }%;background:${helped ? "var(--green)" : "var(--amber)"}"></span></span></span>
          <span class="rv" style="width:88px;color:${helped ? "var(--green)" : "var(--amber)"}">${num(
            v.confidence,
            2
          )} (${v.delta >= 0 ? "+" : ""}${num(v.delta, 2)})</span>
        </div>
        <div class="row">
          <span class="rl">Model accuracy</span>
          <span class="rgrow"><span class="bar"><span style="left:0;width:${
            accuracy * 100
          }%;background:var(--cyan)"></span></span></span>
          <span class="rv" style="width:88px">${pct(accuracy)}</span>
        </div>
        <div class="fs11 dim" style="margin-top:8px">A prediction is treated as confirmed when the measured residual lands inside the rule's own tolerance band of ${num(
          v.budget,
          2
        )}.</div>
      </div>
    </div>`;
}

// ------------------------------------------------------------ knowledge graph

function sortRules(rules) {
  const { key, dir } = ui.kbSort;
  const val = (r) => {
    if (key === "title") return r.title || "";
    if (key === "id") return parseInt(String(r.id).replace(/\D/g, ""), 10) || 0;
    if (key === "success") return (r.times_helped || 0) - (r.times_failed || 0);
    return r[key] || 0;
  };
  return [...rules].sort((a, b) => {
    const x = val(a);
    const y = val(b);
    if (typeof x === "string") return dir * x.localeCompare(y);
    return dir * (x - y);
  });
}

function renderKnowledge(s) {
  const rules = s.knowledge.rules || [];
  $("kb-tag").textContent = `${rules.length} rule${rules.length === 1 ? "" : "s"} · mean conf ${num(
    s.knowledge.mean_confidence,
    2
  )}`;

  for (const th of $("kb").querySelectorAll("th[data-sort]")) {
    // Every sortable header must carry an `.arw` slot, but a missing one must not
    // be able to throw: this runs inside the shared render pass, so a null here
    // would blank every panel drawn after it.
    const arw = th.querySelector(".arw");
    if (arw) {
      arw.textContent = ui.kbSort.key === th.dataset.sort ? (ui.kbSort.dir < 0 ? "▼" : "▲") : "";
    }
  }

  const body = $("kb-body");
  if (!rules.length) {
    body.innerHTML = `<tr><td colspan="5" class="empty" style="padding:26px">Knowledge graph is empty — induce a hazard to grow it</td></tr>`;
    return;
  }

  body.innerHTML = sortRules(rules)
    .map((r) => {
      const judged = (r.times_helped || 0) + (r.times_failed || 0);
      const open = ui.kbOpen === r.id;
      const main = `<tr data-open="${esc(r.id)}" class="${open ? "open" : ""}">
        <td class="rid">${esc(r.id)}</td>
        <td class="rtxt">${esc(r.title)}<br><span class="dim">${esc(r.action_title)}</span></td>
        <td>
          <span class="conf">
            <span class="bar"><span style="left:0;width:${
              (r.confidence || 0) * 100
            }%"></span></span>
            <span class="n">${num(r.confidence, 2)}</span>
          </span>
        </td>
        <td class="num">${r.times_applied || 0}</td>
        <td class="num">${judged ? pct((r.times_helped || 0) / judged) : "—"}</td>
      </tr>`;
      if (!open) return main;

      const ct = (r.confidence_trace || []).slice(-14);
      return (
        main +
        `<tr class="detail"><td colspan="5"><div class="detail-in">
          <div>
            <h5>Predicate</h5>
            <div class="chips">${(r.conditions || [])
              .map((c) => `<span class="chip">${esc(c.text)}</span>`)
              .join("")}</div>
            <h5 style="margin-top:12px">Consequent</h5>
            <div class="mono fs11" style="color:var(--muted)">${esc(r.action_title)}</div>
            <div class="fs11 dim">${esc(r.text)}</div>
          </div>
          <div>
            <h5>Why it exists</h5>
            <div class="rationale">${esc(r.rationale || "")}</div>
            <h5 style="margin-top:12px">Reinforcement</h5>
            <div class="chart"><canvas id="kb-trace-${esc(
              r.id
            )}" height="64" data-trace="${esc(ct.join(","))}" data-max="${Math.max(
              0.6,
              ...ct
            )}"></canvas></div>
            <div class="kv" style="display:flex;justify-content:space-between"><span>last outcome</span><span class="${
              r.last_outcome === "helped" ? "ok" : ""
            }">${esc(r.last_outcome || "—")}</span></div>
          </div>
          <div>
            <h5>Ledger</h5>
            ${[
              ["induced", r.times_induced],
              ["applied", r.times_applied],
              ["verified", judged],
              ["helped", r.times_helped],
              ["failed", r.times_failed],
              ["success rate", pct(r.success_rate)],
              ["specificity", num(r.specificity, 2)],
              ["signature", r.signature],
            ]
              .map(([k, v]) => `<div class="kv"><span>${k}</span><span>${esc(String(v))}</span></div>`)
              .join("")}
          </div>
        </div></td></tr>`
      );
    })
    .join("");

  for (const cv of body.querySelectorAll("canvas[data-trace]")) {
    const tr = (cv.dataset.trace || "").split(",").filter(Boolean).map(Number);
    drawTrace(cv, tr, Number(cv.dataset.max) || 1);
  }
}

// ------------------------------------------------------------------- events

function renderEvents(s) {
  const events = s.events || [];
  const box = $("events");

  // The stream arrives several times a second, and every row carries an entry
  // animation.  Rebuilding the panel each time restarts that animation on all
  // of them, which reads as the whole list blinking.  The newest event's seq is
  // therefore the panel's identity: while it is unchanged the existing DOM is
  // already correct and is left alone.
  const newest = events.length ? events[0].seq : 0;
  if (box.dataset.seq === String(newest)) return;
  box.dataset.seq = String(newest);

  $("events-hint").textContent = `${events.length} event(s), newest first`;
  if (!events.length) {
    box.innerHTML = `<div class="empty">No cognitive events recorded</div>`;
    return;
  }
  box.innerHTML = events
    .slice(0, 40)
    .map(
      (e) => `<div class="evt ${PHASE_CLASS[e.phase] || ""}">
        <span class="tm">${clockOf(e.t)}</span>
        <span class="ph">${esc(e.phase || "—")}</span>
        <span><span class="tt">${esc(e.title || "")}</span><span class="dt">${esc(
          e.detail || ""
        )}</span></span>
      </div>`
    )
    .join("");
}

// ------------------------------------------------------------------ hazards

function renderHazards(s) {
  const lit = s.scenario.active_hazard;
  const box = $("hazards");
  if (box.dataset.built) {
    for (const b of box.querySelectorAll(".haz")) {
      b.classList.toggle("lit", b.dataset.key === lit);
    }
    return;
  }
  box.innerHTML = (s.hazards || [])
    .map(
      (h) => `<button class="haz" data-key="${esc(h.key)}" title="${esc(h.blurb)}">
        <span class="hl">${esc(h.label)}</span>
        <span class="hd">${esc(h.domain)} · ${h.burst} frames</span>
      </button>`
    )
    .join("");
  box.dataset.built = "1";
  box.addEventListener("click", (ev) => {
    const b = ev.target.closest(".haz");
    if (b) inject(b.dataset.key);
  });
  for (const b of box.querySelectorAll(".haz")) b.classList.toggle("lit", b.dataset.key === lit);
}

async function inject(key) {
  ui.lastHazard = key;
  await post({ action: "inject", hazard: key });
}

function nextHazard() {
  const hz = ui.state ? ui.state.hazards : [];
  if (!hz.length) return;
  const key = hz[ui.hazardIdx % hz.length].key;
  ui.hazardIdx++;
  inject(key);
}

// ------------------------------------------------------------------- render

function render(s) {
  ui.state = s;
  // Each panel is isolated: they read the same payload but do not depend on one
  // another, so a defect in one must not blank the rest.  Without this a single
  // null dereference silently emptied every panel drawn after it.
  const panels = [
    ["chrome", renderChrome],
    ["kpis", renderKpis],
    ["pipeline", renderPipeline],
    ["gauges", renderGauges],
    ["interventions", renderInterventions],
    ["working1", renderWorkingOne],
    ["working2", renderWorkingTwo],
    ["verification", renderVerification],
    ["knowledge", renderKnowledge],
    ["events", renderEvents],
    ["hazards", renderHazards],
    ["chart", renderChart],
  ];
  for (const [name, fn] of panels) {
    try {
      fn(s);
    } catch (err) {
      console.error(`[cee] panel "${name}" failed to render:`, err);
      ui.panelErrors[name] = String((err && err.message) || err);
    }
  }
}

// ------------------------------------------------------------------ controls

$("btn-pause").addEventListener("click", () => {
  ui.paused = !ui.paused;
  post({ action: ui.paused ? "pause" : "resume" });
});

$("btn-auto").addEventListener("click", () => post({ action: "auto" }));
$("btn-repeat").addEventListener("click", () => {
  if (ui.lastHazard) post({ action: "repeat", hazard: ui.lastHazard });
});

$("btn-reset").addEventListener("click", async () => {
  if (!confirm("Delete every learned rule and restart the knowledge graph?")) return;
  await post({ action: "reset_memory" });
  ui.kbOpen = null;
});

$("speed-seg").addEventListener("click", (ev) => {
  const b = ev.target.closest("button[data-speed]");
  if (!b) return;
  for (const x of $("speed-seg").children) x.classList.remove("on");
  b.classList.add("on");
  post({ action: "speed", value: Number(b.dataset.speed) });
});

$("kb").addEventListener("click", (ev) => {
  const th = ev.target.closest("th[data-sort]");
  if (th) {
    const key = th.dataset.sort;
    if (ui.kbSort.key === key) ui.kbSort.dir *= -1;
    else ui.kbSort = { key, dir: key === "title" || key === "id" ? 1 : -1 };
    renderKnowledge(ui.state);
    return;
  }
  const row = ev.target.closest("tr[data-open]");
  if (row) {
    ui.kbOpen = ui.kbOpen === row.dataset.open ? null : row.dataset.open;
    renderKnowledge(ui.state);
  }
});

const chart = $("chart");
chart.addEventListener("mousemove", (ev) => {
  chartHit(ev);
  renderChart(ui.state);
});
chart.addEventListener("mouseleave", () => {
  ui.hover = null;
  $("chart-readout").innerHTML = "—";
  renderChart(ui.state);
});

document.addEventListener("keydown", (ev) => {
  if (ev.target.matches("input, textarea")) return;
  if (ev.code === "Space") {
    ev.preventDefault();
    $("btn-pause").click();
  } else if (ev.key === "i") inject(ui.lastHazard || "slip");
  else if (ev.key === "r") $("btn-repeat").click();
  else if (ev.key === "n") nextHazard();
});

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (ui.state) renderChart(ui.state);
  }, 120);
});

connect();
