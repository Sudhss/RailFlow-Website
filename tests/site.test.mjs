/**
 * The page claims its routing is RailFlow's routing. These tests hold it to
 * that: expected routes and minutes below were produced by the backend itself
 * (backend/graph.py, RailwayGraph.dijkstra / total_route_minutes) on the same
 * data, and the simulation invariants are the ones the backend selftest checks.
 *
 *   npm test
 */

import test from "node:test";
import assert from "node:assert/strict";

import { STATIONS, EDGES } from "../js/network.js";
import {
  Network,
  Simulation,
  ROSTER,
  dijkstra,
  dynamicWeight,
  effectiveSpeed,
  routeMinutes,
  congestionState,
  signalAspect,
} from "../js/sim.js";
import { waveAt, WAVE_SPREAD, WAVE_GLSL, buildBases, buildAlignment, sampleAt } from "../js/geometry.js";

const occ = (entries) => new Map(Object.entries(entries));
const net = () => new Network(STATIONS, EDGES);

test("network data matches the backend's size", () => {
  assert.equal(STATIONS.length, 36);
  assert.equal(EDGES.length, 42);
  const ids = new Set(STATIONS.map((s) => s.id));
  for (const e of EDGES) assert.ok(ids.has(e.from) && ids.has(e.to), e.id);
});

test("cost function is backend/graph.py dynamic_weight", () => {
  const edge = { distance_km: 8, avg_speed: 45, capacity: 3, blocked: false };
  const base = (8 / 45) * 60;
  assert.equal(dynamicWeight(edge, 0), base);
  assert.equal(dynamicWeight(edge, 2), base); // load 0.667 <= 0.7 runs free
  assert.ok(Math.abs(dynamicWeight(edge, 3) - base * 1.42) < 1e-9); // load 1.0
  assert.ok(Math.abs(dynamicWeight(edge, 6) - base * 4) < 1e-9); // load 2.0 -> x4
  assert.equal(dynamicWeight({ ...edge, blocked: true }, 0), Infinity);
});

test("speed rule is backend/simulation.py _effective_speed", () => {
  const edge = { avg_speed: 80, capacity: 2 };
  assert.equal(effectiveSpeed(120, edge, 0), 80);
  assert.equal(effectiveSpeed(65, edge, 0), 65);
  assert.equal(effectiveSpeed(120, edge, 2), 80 * 0.85); // load 1.0
  assert.equal(effectiveSpeed(120, edge, 4), 40); // load 2.0 -> v / load
});

test("congestion bands match the backend's edge snapshot", () => {
  const edge = { capacity: 10, blocked: false };
  assert.equal(congestionState(edge, 6), "normal");
  assert.equal(congestionState(edge, 7), "busy");
  assert.equal(congestionState(edge, 10), "busy");
  assert.equal(congestionState(edge, 11), "congested");
  assert.equal(congestionState({ ...edge, blocked: true }, 0), "closed");
});

test("signals show what an entering train would find", () => {
  const edge = { capacity: 3, blocked: false };
  assert.equal(signalAspect(edge, 0), "green"); // 1/3 after entry
  assert.equal(signalAspect(edge, 1), "green"); // 2/3 = 0.67, still under the 0.7 busy band
  assert.equal(signalAspect(edge, 2), "yellow"); // entering makes it 3/3: busy
  assert.equal(signalAspect(edge, 3), "red");
  assert.equal(signalAspect({ capacity: 1, blocked: false }, 0), "yellow"); // one train fills it
  assert.equal(signalAspect({ ...edge, blocked: true }, 0), "red");
});

test("Dijkstra returns the backend's routes", () => {
  const n = net();
  const free = dijkstra(n, "NDLS", "LKO");
  assert.deepEqual(free.path, "NDLS DSA GZB PKW HPU GMS GJL AMRO MB RMU MIL BE PMR TLH SPN AJI HRI SAN AMG LKO".split(" "));
  assert.ok(Math.abs(routeMinutes(n, free.path) - 396.4967) < 1e-3);

  const loaded = occ({ "GZB-PKW": 4, "PKW-HPU": 3, "HPU-GMS": 2 });
  const busy = dijkstra(n, "NDLS", "LKO", loaded);
  assert.deepEqual(busy.path, "NDLS DSA GZB KRJ ALJN HRS TDL FZD SKB ETW PHD CNB ON LKO".split(" "));
  assert.ok(Math.abs(busy.minutes - 417.6946) < 1e-3);

  assert.deepEqual(dijkstra(n, "KSJ", "ON").path, "KSJ BEM BE PMR TLH SPN AJI HRI ON".split(" "));
  assert.deepEqual(
    dijkstra(n, "LKO", "ALJN", occ({ "ON-LKO": 9 })).path,
    "LKO AMG SAN HRI AJI SPN TLH PMR BE BEM KSJ ALJN".split(" ")
  );

  n.setBlocked("MB-RMU", true);
  assert.deepEqual(dijkstra(n, "MB", "LKO").path, "MB CH AO BE PMR TLH SPN AJI HRI SAN AMG LKO".split(" "));
});

test("the trace is Dijkstra's real settlement order", () => {
  const { trace, path } = dijkstra(net(), "MB", "LKO");
  const minutes = trace.settled.map((s) => s.minutes);
  for (let i = 1; i < minutes.length; i += 1) assert.ok(minutes[i] >= minutes[i - 1] - 1e-9, "settled out of order");
  assert.equal(trace.settled[0].node, "MB");
  assert.equal(trace.settled.at(-1).node, "LKO");
  assert.equal(new Set(trace.settled.map((s) => s.node)).size, trace.settled.length, "a station settled twice");
  for (const node of path) assert.ok(trace.settled.some((s) => s.node === node));
});

test("no route when the destination is cut off", () => {
  const n = net();
  n.setBlocked("AMG-LKO", true);
  n.setBlocked("ON-LKO", true);
  assert.equal(dijkstra(n, "NDLS", "LKO").path, null);
});

function runWithClosures(minutes, closures) {
  const n = net();
  const sim = new Simulation(n, ROSTER);
  const entered = [];
  for (let t = 0; t < minutes * 4; t += 1) {
    const at = t / 4;
    for (const c of closures) {
      if (at === c.at) n.setBlocked(c.edge, true);
      if (at === c.until) n.setBlocked(c.edge, false);
    }
    const before = new Map(sim.trains.map((tr) => [tr.id, `${tr.leg}:${tr.route.join()}`]));
    sim.step(0.25);
    for (const tr of sim.trains) {
      if (tr.status !== "running" || tr.progressKm <= 0) continue;
      const edge = sim.currentEdge(tr);
      // A train that is inside a section it was not in a moment ago has just
      // entered it; it must not have entered a closed one.
      if (edge?.blocked && before.get(tr.id) !== `${tr.leg}:${tr.route.join()}`) entered.push(`${tr.id} ${edge.id} @${at}`);
    }
    for (const tr of sim.trains) {
      const edge = sim.currentEdge(tr);
      if (edge) assert.ok(tr.progressKm >= 0 && tr.progressKm <= edge.distance_km + 1e-6, `${tr.id} off its section`);
      assert.ok(tr.delay >= 0 && Number.isFinite(tr.delay), `${tr.id} delay ${tr.delay}`);
      for (let i = 0; i < tr.route.length - 1; i += 1) assert.ok(n.edgeBetween(tr.route[i], tr.route[i + 1]), `${tr.id} route breaks`);
    }
  }
  return { sim, entered };
}

test("no train enters a closed section, and every route stays connected", () => {
  const { entered, sim } = runWithClosures(600, [
    { edge: "MB-RMU", at: 60, until: 240 },
    { edge: "GZB-KRJ", at: 90, until: 400 },
    { edge: "HRI-SAN", at: 150, until: 300 },
  ]);
  assert.deepEqual(entered, []);
  assert.ok(sim.automated > 0, "controller never acted");
});

test("the controller only ever reroutes onto a faster route", () => {
  const n = net();
  const sim = new Simulation(n, ROSTER);
  const decisions = [];
  sim.on((e) => e.kind === "reroute" && decisions.push(e));
  for (let t = 0; t < 400 * 4; t += 1) {
    if (t === 200) n.setBlocked("MB-RMU", true);
    sim.step(0.25);
  }
  assert.ok(decisions.length > 0);
  for (const d of decisions) {
    assert.ok(d.after < d.before, `${d.train}: ${d.after} !< ${d.before}`);
    assert.notDeepEqual(d.newRoute, d.oldRoute);
    assert.ok(!n.routeEdges(d.newRoute).some((e) => e.blocked), "rerouted through a closure");
  }
});

test("time is conserved across stations (no distance lost at nodes)", () => {
  // One superfast on an empty network: it must arrive when its free-running
  // time says, whatever step size drives it.
  const spec = [{ id: "T", name: "t", type: "superfast", source: "NDLS", destination: "MB", depart: 0 }];
  const arrival = (dt) => {
    const sim = new Simulation(net(), spec);
    let when = null;
    sim.on((e) => e.kind === "arrive" && when === null && (when = e.time));
    for (let t = 0; t < 400 / dt && when === null; t += 1) sim.step(dt);
    return when;
  };
  const coarse = arrival(1);
  const fine = arrival(0.05);
  assert.ok(coarse !== null && fine !== null);
  assert.ok(Math.abs(coarse - fine) < 0.05, `${coarse} vs ${fine}`);
});

test("reset clears closures and keeps listeners", () => {
  const n = net();
  const sim = new Simulation(n, ROSTER);
  let heard = 0;
  sim.on(() => (heard += 1));
  n.setBlocked("MB-RMU", true);
  sim.reset();
  assert.equal(n.blockedCount(), 0);
  for (let i = 0; i < 40; i += 1) sim.step(0.5);
  assert.ok(heard > 0, "listener dropped by reset");
});

test("the layout wave is the same function in JS and GLSL", () => {
  assert.equal(waveAt(0.3, 0.3, 0), 0);
  assert.equal(waveAt(0.9, 0.1, 1), 1);
  assert.ok(waveAt(0.1, 0.1, 0.5) > waveAt(0.9, 0.1, 0.5), "the wave must travel outward from its origin");
  assert.ok(WAVE_GLSL.includes(WAVE_SPREAD.toFixed(3)));
});

test("both bases fit the same world box and track is arc-length parameterised", () => {
  const { schematic, geographic } = buildBases(STATIONS);
  for (const basis of [schematic, geographic]) {
    const xs = Object.values(basis).map((p) => p.x);
    const zs = Object.values(basis).map((p) => p.z);
    const span = (v) => Math.max(...v) - Math.min(...v);
    // fitToWorld scales the dominant axis (z weighted 1.6x) to WORLD_WIDTH.
    assert.ok(Math.abs(Math.max(span(xs), span(zs) * 1.6) - 420) < 1e-6);
    const sections = buildAlignment(STATIONS, EDGES, basis);
    const sec = sections["HRI-ON"];
    const a = sampleAt(sec, 0.25).position;
    const b = sampleAt(sec, 0.5).position;
    const c = sampleAt(sec, 0.75).position;
    const d1 = Math.hypot(b.x - a.x, b.z - a.z);
    const d2 = Math.hypot(c.x - b.x, c.z - b.z);
    assert.ok(Math.abs(d1 - d2) / d1 < 0.02, "equal fractions must be equal distances");
  }
});
