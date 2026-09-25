/**
 * Page wiring: the simulation, the stage, the scroll steps and the controls.
 *
 * The stage is one sticky WebGL canvas; each step of text that scrolls over it
 * asks for a shot, a basis and an overlay. The simulation keeps running
 * underneath all of them, so what a step shows is always the live network.
 */

import { STATIONS, EDGES, SIM_START } from "./network.js";
import { Network, Simulation, ROSTER, dijkstra, dynamicWeight, congestionState, viaOf } from "./sim.js";
import { buildBases } from "./geometry.js";

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const network = new Network(STATIONS, EDGES);
const sim = new Simulation(network, ROSTER);

// Start mid-morning rather than on an empty network: 80 simulated minutes in,
// every train is out on the line.
function warmUp() {
  for (let i = 0; i < 80 * 4; i += 1) sim.step(0.25);
  sim.log.length = 0;
  sim.automated = 0;
}
warmUp();

const DEMO_CLOSURE = "MB-RMU";
const state = {
  step: "hero",
  paused: reducedMotion,
  rate: 3, // simulated minutes per second
  demoClosed: false,
  wide: window.matchMedia("(min-width: 901px)"),
  lastUserClose: -Infinity,
  lastTraceAt: -Infinity,
};

/* -------------------------------------------------------------- helpers */

function clock(minutes) {
  const [h, m] = SIM_START.split(":").map(Number);
  const total = Math.floor(h * 60 + m + minutes) % (24 * 60);
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

function sectionName(edge) {
  return `${edge.from}–${edge.to}`;
}

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

/* --------------------------------------------------------------- stage */

let stage = null;

async function boot() {
  const canvas = $("#stage-canvas");
  let webgl = false;
  try {
    webgl = Boolean(document.createElement("canvas").getContext("webgl2"));
  } catch {
    webgl = false;
  }
  if (!webgl) return fallback();

  try {
    const { Stage } = await import("./stage.js");
    stage = new Stage(canvas, sim, { reducedMotion });
  } catch (error) {
    console.error("RailFlow stage failed to start", error);
    return fallback();
  }

  canvas.addEventListener("webglcontextlost", (event) => {
    event.preventDefault();
    stage?.stop();
    fallback("The graphics context was lost. Reload the page to bring the 3D network back.");
  });

  const stageEl = $("#stage");
  const resize = () => stage.resize(stageEl.clientWidth, stageEl.clientHeight);
  new ResizeObserver(resize).observe(stageEl);
  resize();

  stage.onBeforeFrame = (dt) => {
    if (!state.paused) sim.step(dt * state.rate);
  };
  stage.onAfterFrame = frameOverlays;
  stage.on("hover", showTooltip);
  stage.on("pick", onPick);

  // Only render while the stage is on screen and the tab is visible.
  let onScreen = true;
  const sync = () => (onScreen && !document.hidden ? stage.start() : stage.stop());
  new IntersectionObserver(([entry]) => {
    onScreen = entry.isIntersecting;
    sync();
  }).observe($("#network"));
  document.addEventListener("visibilitychange", sync);

  applyStep(state.step, true);
  // Open at rail level on the fastest train and pull back to the region, so
  // the first thing the page shows is the scale it covers.
  if (state.step === "hero") stage.introFrom(pickChaseTrain().id);
  sync();
  window.__railflow = { stage, sim, network }; // for debugging from the console
}

function fallback(message) {
  $("#fallback").hidden = false;
  $("#stage-canvas").hidden = true;
  $(".legend").hidden = true;
  $("#zoom").hidden = true;
  // The HUD and the drag hint describe the live 3D network, which isn't running.
  $(".hud").hidden = true;
  $$(".hint").forEach((hint) => (hint.hidden = true));
  if (message) $("#fallback p").textContent = message;
  $$(".segmented button, #costs-toggle, #section-select, #train-select, #replay, #reset").forEach((b) => (b.disabled = true));

  // A flat diagram from the same schematic coordinates.
  const { schematic } = buildBases(STATIONS);
  const svg = $("#fallback-svg");
  const xs = Object.values(schematic).map((p) => p.x);
  const zs = Object.values(schematic).map((p) => p.z);
  const pad = 12;
  const minX = Math.min(...xs) - pad;
  const minZ = Math.min(...zs) - pad;
  svg.setAttribute("viewBox", `${minX} ${minZ} ${Math.max(...xs) - minX + pad} ${Math.max(...zs) - minZ + pad}`);
  const ns = "http://www.w3.org/2000/svg";
  for (const e of EDGES) {
    const a = schematic[e.from];
    const b = schematic[e.to];
    const line = document.createElementNS(ns, "line");
    Object.entries({ x1: a.x, y1: a.z, x2: b.x, y2: b.z, stroke: "#5b6870", "stroke-width": 1.4 }).forEach(([k, v]) => line.setAttribute(k, v));
    svg.appendChild(line);
  }
  for (const s of STATIONS) {
    const p = schematic[s.id];
    const dot = document.createElementNS(ns, "circle");
    Object.entries({ cx: p.x, cy: p.z, r: s.type === "minor" ? 1.4 : 2.4, fill: "#c4ced4" }).forEach(([k, v]) => dot.setAttribute(k, v));
    svg.appendChild(dot);
    if (s.type !== "minor") {
      const t = document.createElementNS(ns, "text");
      Object.entries({ x: p.x, y: p.z - 4, fill: "#a3aeb5", "font-size": 5, "text-anchor": "middle", "font-family": "monospace" }).forEach(([k, v]) => t.setAttribute(k, v));
      t.textContent = s.id;
      svg.appendChild(t);
    }
  }
}

/* --------------------------------------------------------------- steps */

function lens(kind = "side") {
  if (!state.wide.matches) return { lensX: 0, lensY: -0.17 };
  return kind === "hero" ? { lensX: 0.12, lensY: 0.02 } : { lensX: 0.19, lensY: 0 };
}

function centreOf(ids) {
  const points = ids.map((id) => stage.stationPosition(id));
  return {
    x: points.reduce((s, p) => s + p.x, 0) / points.length,
    z: points.reduce((s, p) => s + p.z, 0) / points.length,
  };
}

function pickChaseTrain() {
  const running = sim.trains.filter((t) => t.status === "running" && t.speed > 30);
  running.sort((a, b) => b.priority - a.priority || b.speed - a.speed);
  return running[0] || sim.trains.find((t) => t.status !== "scheduled") || sim.trains[0];
}

function setDemoClosure(closed) {
  const edge = network.edges.get(DEMO_CLOSURE);
  if (closed && !edge.blocked) {
    network.setBlocked(DEMO_CLOSURE, true);
    state.demoClosed = true;
  } else if (!closed && state.demoClosed) {
    network.setBlocked(DEMO_CLOSURE, false);
    state.demoClosed = false;
  }
}

function runDemoTrace() {
  const result = dijkstra(network, "MB", "LKO", sim.occupancy());
  stage.playTrace(result);
  const readout = $("#trace-readout");
  readout.replaceChildren(
    document.createTextNode(`MB → LKO: ${result.trace.settled.length} stations settled, `),
    el("b", {}, `${Math.round(result.minutes)} min`),
    document.createTextNode(` via ${viaOf(result.path)}.`)
  );
}

function applyStep(name, immediate = false) {
  state.step = name;
  tooltip.hidden = true;
  document.body.dataset.step = name;
  if (!stage) return;
  const free = name === "free";
  stage.setInteractive(free ? "free" : "story");
  $("#zoom").hidden = !free;
  if (name !== "closure") setDemoClosure(false);
  if (name !== "free") stage.setHighlightTrain(null);

  switch (name) {
    case "hero":
      stage.setBasis(0);
      stage.showCosts(false);
      stage.setShot({ x: 30, z: 6, extent: 330, heading: -0.62, pitch: 0.42, ...lens("hero") });
      break;
    case "maps":
      stage.setBasis(1, 0);
      stage.showCosts(false);
      stage.setShot({ x: 0, z: 0, extent: 440, heading: -0.05, pitch: 1.12, ...lens() });
      break;
    case "costs":
      stage.showCosts(true);
      stage.setShot({ x: 6, z: 8, extent: 360, heading: -0.62, pitch: 0.5, ...lens() });
      break;
    case "closure": {
      stage.showCosts(false);
      setDemoClosure(true);
      const c = centreOf(["MB", "RMU", "BE", "CH", "AO"]);
      stage.setShot({ x: c.x, z: c.z + 6, extent: 150, heading: -0.35, pitch: 0.95, ...lens() });
      runDemoTrace();
      break;
    }
    case "rail": {
      stage.showCosts(false);
      stage.clearTrace();
      const train = pickChaseTrain();
      $("#chase-name").textContent = `${train.id} ${train.name}`;
      const narrow = !state.wide.matches;
      stage.setShot({ follow: train.id, followDistance: narrow ? 6.5 : 4.6, pitch: 0.3, extent: 30, ...(narrow ? { lensX: 0, lensY: -0.08 } : lens()) });
      stage.setHighlightTrain(train.id);
      state.chase = train.id;
      break;
    }
    case "free": {
      stage.clearTrace();
      stage.showCosts($("#costs-toggle").checked);
      const shot = { x: 0, z: 0, extent: 440, heading: -0.12, pitch: 0.95, ...lens() };
      stage.home = { ...shot };
      stage.setShot(shot);
      const follow = $("#train-select").value;
      if (follow) {
        stage.followTrain(follow);
        stage.setHighlightTrain(follow);
      }
      break;
    }
    default:
      break;
  }
  if (immediate) stage._snap();
  syncBasisButtons();
}

const stepObserver = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) applyStep(entry.target.dataset.step);
    }
  },
  { rootMargin: "-48% 0px -48% 0px" }
);
$$(".step").forEach((step) => stepObserver.observe(step));
state.wide.addEventListener("change", () => applyStep(state.step));

/* ------------------------------------------------------------ overlays */

const labelLayer = $("#labels");
const stationLabels = new Map();
for (const s of STATIONS) {
  const node = el("span", { class: s.type === "minor" ? "label minor" : "label" }, s.id);
  labelLayer.appendChild(node);
  stationLabels.set(s.id, node);
}
const minuteLabels = [];

let hudTimer = 0;
function frameOverlays(dt) {
  // Station codes: junctions and terminals from far away, everything close up.
  const far = stage.viewDistance > 260;
  const tracing = stage.traceFade > 0.05;
  for (const { station, x, y } of stage.stationLabels()) {
    const node = stationLabels.get(station.id);
    const show = !tracing && (!far || station.type !== "minor") && x > -20 && y > -20 && x < stage.size.width + 20 && y < stage.size.height + 20;
    node.style.display = show ? "" : "none";
    if (show) node.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) translate(-50%, -150%)`;
  }

  // Minutes Dijkstra assigned, as each station is settled.
  const labels = stage.traceLabelsNow();
  while (minuteLabels.length < labels.length) {
    const node = el("span", { class: "label minutes" });
    labelLayer.appendChild(node);
    minuteLabels.push(node);
  }
  minuteLabels.forEach((node, i) => {
    const data = labels[i];
    if (!data) {
      node.style.display = "none";
      return;
    }
    node.style.display = "";
    node.style.opacity = String(Math.min(1, stage.traceFade));
    node.className = data.onPath ? "label minutes path" : "label minutes";
    node.textContent = `${data.id} ${Math.round(data.minutes)}`;
    node.style.transform = `translate(${data.x.toFixed(1)}px, ${data.y.toFixed(1)}px) translate(-50%, 70%)`;
  });

  hudTimer -= dt;
  if (hudTimer <= 0) {
    hudTimer = 0.25;
    updateHud();
  }
}

function updateHud() {
  const counts = sim.counts();
  $("#hud-clock").textContent = clock(sim.time);
  $("#hud-running").textContent = counts.running + counts.dwell;
  const held = $("#hud-held");
  held.textContent = counts.held;
  held.className = counts.held ? "warn" : "";
  const closed = $("#hud-closed");
  closed.textContent = network.blockedCount();
  closed.className = network.blockedCount() ? "bad" : "";
  $("#hud-decisions").textContent = sim.automated;

  if (state.step === "rail" && state.chase) {
    const t = sim.trains.find((x) => x.id === state.chase);
    if (t) {
      const edge = sim.currentEdge(t);
      const where = edge && t.status === "running" ? `on ${sectionName(edge)}` : `at ${t.route[t.leg]}`;
      $("#chase-readout").textContent = `${t.id} · ${Math.round(t.speed)} km/h ${where} · ${t.delay >= 1 ? `+${Math.round(t.delay)} min` : "on time"}`;
    }
  }
}

const tooltip = $("#tooltip");
function showTooltip(hit) {
  if (!hit) {
    tooltip.hidden = true;
    return;
  }
  tooltip.replaceChildren();
  if (hit.kind === "station") {
    const s = hit.station;
    tooltip.append(el("strong", {}, `${s.id} · ${s.name}`), el("span", {}, `${s.type} · base dwell ${s.dwell_base} min`));
  } else if (hit.kind === "edge") {
    const edge = network.edges.get(hit.edge.id);
    const load = sim.occupancy().get(edge.id) || 0;
    const minutes = dynamicWeight(edge, load);
    tooltip.append(
      el("strong", {}, sectionName(edge)),
      el("span", {}, `${edge.distance_km} km · ${edge.avg_speed} km/h · capacity ${edge.capacity}`),
      el("br"),
      el("span", {}, edge.blocked ? "closed" : `${load} train${load === 1 ? "" : "s"} · ${congestionState(edge, load)} · ${minutes.toFixed(1)} min expected`)
    );
    if (state.step === "free") tooltip.append(el("br"), el("em", {}, edge.blocked ? "click to reopen" : "click to close"));
  } else if (hit.kind === "train") {
    const t = hit.train;
    tooltip.append(
      el("strong", {}, `${t.id} · ${t.name}`),
      el("span", {}, `${t.type} · ${Math.round(t.speed)} km/h · ${t.source} → ${t.destination}`),
      el("br"),
      el("span", {}, `${t.status === "held" ? "held" : t.delay >= 1 ? `+${Math.round(t.delay)} min` : "on time"}`)
    );
    if (state.step === "free") tooltip.append(el("br"), el("em", {}, "click to follow"));
  }
  tooltip.hidden = false;
  const w = tooltip.offsetWidth;
  const x = Math.min(hit.x + 14, stage.size.width - w - 10);
  tooltip.style.left = `${Math.max(10, x)}px`;
  tooltip.style.top = `${Math.max(60, hit.y + 14)}px`;
}

/* ------------------------------------------------------------ controls */

function toggleSection(edgeId) {
  const edge = network.edges.get(edgeId);
  if (!edge) return;
  const closing = !edge.blocked;
  network.setBlocked(edgeId, closing);
  if (edgeId === DEMO_CLOSURE) state.demoClosed = false;
  state.lastUserClose = closing ? performance.now() : state.lastUserClose;
  addLog({ kind: "user", time: sim.time, text: `You ${closing ? "closed" : "reopened"} ${sectionName(edge)}.` });
  refreshSectionSelect();
}

function onPick(hit) {
  tooltip.hidden = true;
  if (state.step !== "free" || !hit) return;
  if (hit.kind === "edge") toggleSection(hit.edge.id);
  else if (hit.kind === "train") setFollow(hit.train.id);
}

function setFollow(id) {
  $("#train-select").value = id || "";
  stage?.setHighlightTrain(id || null);
  if (id) stage?.followTrain(id);
  else stage?.recentre();
}

// Every reroute the controller makes is logged; ones caused by a closure the
// reader made also get their search drawn, at most one every few seconds.
sim.on((event) => {
  if (event.kind === "reroute" || event.kind === "hold" || event.kind === "release") addLog(event);
  if (!stage || event.kind !== "reroute" || state.step !== "free") return;
  const now = performance.now();
  const followed = $("#train-select").value === event.train;
  const recentUser = now - state.lastUserClose < 30000;
  if ((recentUser || followed) && now - state.lastTraceAt > 5000) {
    state.lastTraceAt = now;
    stage.playTrace({ path: event.newRoute, trace: event.trace }, { speed: 1.4 });
  }
});

const logList = $("#log");
function addLog(event) {
  const item = el("li", { "data-kind": event.kind });
  item.append(el("time", {}, clock(event.time)), el("span", {}, event.text));
  logList.prepend(item);
  while (logList.children.length > 30) logList.lastElementChild.remove();
}

function refreshSectionSelect() {
  const select = $("#section-select");
  select.replaceChildren(el("option", { value: "" }, "Choose a section…"));
  for (const e of [...network.edges.values()].sort((a, b) => a.id.localeCompare(b.id))) {
    select.append(el("option", { value: e.id }, `${e.blocked ? "Reopen" : "Close"} ${sectionName(e)} (${e.distance_km} km)`));
  }
  select.value = "";
}
refreshSectionSelect();
$("#section-select").addEventListener("change", (event) => {
  if (event.target.value) toggleSection(event.target.value);
  event.target.value = "";
});

const trainSelect = $("#train-select");
for (const t of sim.trains) trainSelect.append(el("option", { value: t.id }, `${t.id} ${t.name} (${t.type})`));
trainSelect.addEventListener("change", (event) => setFollow(event.target.value));

function syncBasisButtons() {
  const target = stage ? stage.basisTarget : 0;
  $$("[data-basis]").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.basis) === target)));
}
$$("[data-basis]").forEach((button) =>
  button.addEventListener("click", () => {
    stage?.setBasis(Number(button.dataset.basis));
    syncBasisButtons();
  })
);

function syncSpeed() {
  $$("[data-speed]").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.speed) === state.rate)));
}
$$("[data-speed]").forEach((button) =>
  button.addEventListener("click", () => {
    state.rate = Number(button.dataset.speed);
    syncSpeed();
  })
);
syncSpeed();

$("#costs-toggle").addEventListener("change", (event) => stage?.showCosts(event.target.checked));
$("#replay").addEventListener("click", () => stage && runDemoTrace());
$("#reset").addEventListener("click", () => {
  sim.reset();
  warmUp();
  state.demoClosed = false;
  logList.replaceChildren();
  refreshSectionSelect();
  setFollow("");
});
$$("#zoom [data-zoom]").forEach((b) => b.addEventListener("click", () => stage?.zoom(Number(b.dataset.zoom))));
$("#recentre").addEventListener("click", () => stage?.recentre());

const pauseButton = $("#pause");
function syncPause() {
  pauseButton.textContent = state.paused ? "Run" : "Pause";
  pauseButton.setAttribute("aria-pressed", String(state.paused));
}
pauseButton.addEventListener("click", () => {
  state.paused = !state.paused;
  syncPause();
});
syncPause();

// Keyboard on the stage in free play: arrows pan, +/- zoom, C recentres.
document.addEventListener("keydown", (event) => {
  if (!stage || state.step !== "free" || event.target.closest("input, select, textarea")) return;
  const pan = { ArrowLeft: [-40, 0], ArrowRight: [40, 0], ArrowUp: [0, -40], ArrowDown: [0, 40] }[event.key];
  if (pan && event.shiftKey) {
    event.preventDefault();
    stage.goal.x += (pan[0] * stage.goal.extent) / 900;
    stage.goal.z += (pan[1] * stage.goal.extent) / 900;
  } else if (event.key === "+" || event.key === "=") stage.zoom(0.8);
  else if (event.key === "-") stage.zoom(1.25);
  else if (event.key === "c" || event.key === "C") stage.recentre();
});

/* ------------------------------------------------------------ sections */

const revealObserver = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add("in");
        revealObserver.unobserve(entry.target);
      }
    }
  },
  { threshold: 0.3 }
);
$$(".chart").forEach((chart) => {
  chart.classList.add("reveal");
  revealObserver.observe(chart);
});

$$(".copy").forEach((button) =>
  button.addEventListener("click", async () => {
    const code = button.closest(".code").querySelector("code").innerText;
    try {
      await navigator.clipboard.writeText(code);
      button.textContent = "Copied";
    } catch {
      button.textContent = "Select and copy";
    }
    setTimeout(() => (button.textContent = "Copy"), 1800);
  })
);

boot();
