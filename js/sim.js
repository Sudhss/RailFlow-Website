/**
 * A JavaScript port of RailFlow's routing rules, small enough to run in the
 * page.
 *
 * This is not the backend. It is the same cost function, the same speed rule,
 * the same Dijkstra and the same two-stage controller gate as backend/graph.py,
 * backend/simulation.py and backend/agent.py, so what a visitor sees the trains
 * do here is what the real simulation would decide. Differences, stated so
 * nobody has to discover them:
 *
 *   - time advances continuously instead of in whole-minute ticks (the
 *     controller still runs once per simulated minute);
 *   - a train already inside a section when it closes runs through to the next
 *     station; the closure applies to entries;
 *   - trains turn round at their destination and work back, so the network
 *     never empties.
 *
 * Pure: no DOM, no renderer. Covered by tests/site.test.mjs.
 */

/** backend/train.py TRAIN_TYPE_PROFILES, plus a formation length for drawing. */
export const TRAIN_TYPES = {
  superfast: { priority: 5, maxSpeed: 120, minorStop: false, dwellMultiplier: 0.8, coaches: 9 },
  express: { priority: 4, maxSpeed: 105, minorStop: false, dwellMultiplier: 1.0, coaches: 8 },
  passenger: { priority: 3, maxSpeed: 80, minorStop: true, dwellMultiplier: 1.3, coaches: 6 },
  freight: { priority: 2, maxSpeed: 65, minorStop: false, dwellMultiplier: 1.8, coaches: 11 },
};

/** backend/agent.py */
export const REROUTE_COOLDOWN_MINUTES = 8;
const MAX_SECTIONS_PER_STEP = 8;
const TURNAROUND_MINUTES = 24;
const LOG_LIMIT = 60;

/* ------------------------------------------------------------------ graph */

export class Network {
  constructor(stations, edges) {
    this.stations = new Map(stations.map((s) => [s.id, s]));
    this.edges = new Map(edges.map((e) => [e.id, { ...e, blocked: false }]));
    this.adjacency = new Map(stations.map((s) => [s.id, []]));
    for (const edge of this.edges.values()) {
      this.adjacency.get(edge.from)?.push({ to: edge.to, edge });
      this.adjacency.get(edge.to)?.push({ to: edge.from, edge });
    }
  }

  edgeBetween(a, b) {
    for (const link of this.adjacency.get(a) || []) if (link.to === b) return link.edge;
    return null;
  }

  routeEdges(route) {
    const edges = [];
    for (let i = 0; i < route.length - 1; i += 1) {
      const edge = this.edgeBetween(route[i], route[i + 1]);
      if (!edge) return null;
      edges.push(edge);
    }
    return edges;
  }

  setBlocked(edgeId, blocked) {
    const edge = this.edges.get(edgeId);
    if (edge) edge.blocked = Boolean(blocked);
    return edge;
  }

  blockedCount() {
    let n = 0;
    for (const edge of this.edges.values()) if (edge.blocked) n += 1;
    return n;
  }
}

function speedLimit(edge) {
  return edge.speed_limit ?? edge.avg_speed;
}

/** backend/graph.py RailwayGraph.dynamic_weight -- expected minutes, congestion included. */
export function dynamicWeight(edge, trainsOnEdge = 0) {
  if (edge.blocked) return Infinity;
  const baseMinutes = (edge.distance_km / speedLimit(edge)) * 60;
  const load = trainsOnEdge / Math.max(1, edge.capacity);
  let multiplier;
  if (load <= 0.7) multiplier = 1.0;
  else if (load <= 1.0) multiplier = 1.0 + (load - 0.7) * 1.4;
  else multiplier = load * 2.0;
  return baseMinutes * multiplier;
}

/** backend/simulation.py _effective_speed, in km/h. */
export function effectiveSpeed(maxSpeed, edge, trainsOnEdge = 0) {
  const load = trainsOnEdge / Math.max(1, edge.capacity);
  let speed = Math.min(maxSpeed, speedLimit(edge));
  if (load > 1) speed *= 1 / load;
  else if (load >= 0.7) speed *= 0.85;
  return Math.max(0, speed);
}

export function congestionState(edge, trainsOnEdge = 0) {
  if (edge.blocked) return "closed";
  const load = trainsOnEdge / Math.max(1, edge.capacity);
  if (load < 0.7) return "normal";
  if (load <= 1.0) return "busy";
  return "congested";
}

/**
 * The aspect a lineside signal at the entrance to a section would show a
 * train about to enter it: red if the section is closed or already at
 * capacity, yellow if one more train would push it into the busy band,
 * green otherwise. Same bands as congestionState.
 *
 * RailFlow models capacity as slowdown rather than as a hard stop, so a train
 * can run past a red into a full section. A real signal forbids exactly that,
 * which is one reason RailFlow is not a signalling system.
 */
export function signalAspect(edge, trainsOnEdge = 0) {
  if (edge.blocked) return "red";
  const capacity = Math.max(1, edge.capacity);
  if (trainsOnEdge >= capacity) return "red";
  if ((trainsOnEdge + 1) / capacity >= 0.7) return "yellow";
  return "green";
}

/** A binary min-heap on [priority, value] -- the heapq the backend uses. */
class MinHeap {
  constructor() {
    this.items = [];
  }
  get size() {
    return this.items.length;
  }
  push(item) {
    const a = this.items;
    a.push(item);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[p][0] <= a[i][0]) break;
      [a[p], a[i]] = [a[i], a[p]];
      i = p;
    }
  }
  pop() {
    const a = this.items;
    const top = a[0];
    const last = a.pop();
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < a.length && a[l][0] < a[m][0]) m = l;
        if (r < a.length && a[r][0] < a[m][0]) m = r;
        if (m === i) break;
        [a[m], a[i]] = [a[i], a[m]];
        i = m;
      }
    }
    return top;
  }
}

/**
 * backend/graph.py RailwayGraph.dijkstra, with an optional trace of the order
 * stations were settled and the relaxations that improved a label. The trace
 * is what the page draws: the frontier you watch spread is this order.
 */
export function dijkstra(network, start, destination, occupancy = new Map()) {
  const trace = { settled: [], relaxed: [] };
  if (!network.stations.has(start) || !network.stations.has(destination)) {
    return { path: null, minutes: Infinity, trace };
  }
  const distances = new Map();
  const previous = new Map();
  for (const id of network.stations.keys()) {
    distances.set(id, Infinity);
    previous.set(id, null);
  }
  distances.set(start, 0);
  const queue = new MinHeap();
  queue.push([0, start]);

  while (queue.size) {
    const [distance, node] = queue.pop();
    if (distance > distances.get(node)) continue;
    trace.settled.push({ node, minutes: distance });
    if (node === destination) break;
    for (const { to, edge } of network.adjacency.get(node)) {
      const weight = dynamicWeight(edge, occupancy.get(edge.id) || 0);
      if (!Number.isFinite(weight)) continue;
      const candidate = distance + weight;
      if (candidate < distances.get(to)) {
        distances.set(to, candidate);
        previous.set(to, node);
        trace.relaxed.push({ edge: edge.id, from: node, to, minutes: candidate });
        queue.push([candidate, to]);
      }
    }
  }

  const minutes = distances.get(destination);
  if (!Number.isFinite(minutes)) return { path: null, minutes: Infinity, trace };
  const path = [];
  for (let cursor = destination; cursor !== null; cursor = previous.get(cursor)) path.push(cursor);
  path.reverse();
  return { path, minutes, trace };
}

/** backend/graph.py total_route_minutes. */
export function routeMinutes(network, route, occupancy = new Map()) {
  const edges = network.routeEdges(route);
  if (!edges) return Infinity;
  let total = 0;
  for (const edge of edges) total += dynamicWeight(edge, occupancy.get(edge.id) || 0);
  return total;
}

/* ------------------------------------------------------------- simulation */

function dwellFor(network, train, stationId) {
  const station = network.stations.get(stationId);
  if (!station) return 0;
  if (station.type === "minor" && !train.profile.minorStop) return 0;
  return Math.max(1, Math.round(station.dwell_base * train.profile.dwellMultiplier));
}

/**
 * Free-running minutes from a point on a route to its end: what the timetable
 * would allow with no congestion. Delay is measured against this, so a reroute
 * onto a longer path, a hold and a congested section all show up as delay.
 */
function freeRunMinutes(network, train, route, leg, progressKm) {
  let minutes = 0;
  for (let i = leg; i < route.length - 1; i += 1) {
    const edge = network.edgeBetween(route[i], route[i + 1]);
    if (!edge) return Infinity;
    const remaining = i === leg ? edge.distance_km - progressKm : edge.distance_km;
    minutes += (Math.max(0, remaining) / Math.min(train.profile.maxSpeed, speedLimit(edge))) * 60;
    if (i + 1 < route.length - 1) minutes += dwellFor(network, train, route[i + 1]);
  }
  return minutes;
}

export class Simulation {
  constructor(network, roster) {
    this.network = network;
    this.roster = roster;
    this.listeners = new Set();
    this.reset();
  }

  reset() {
    for (const edge of this.network.edges.values()) edge.blocked = false;
    this.time = 0;
    this.nextControl = 1;
    this.trains = this.roster.map((spec) => this._createTrain(spec));
    this.log = [];
    this.automated = 0;
    this.lastDecision = null;
  }

  on(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  _emit(event) {
    this.log.unshift(event);
    if (this.log.length > LOG_LIMIT) this.log.length = LOG_LIMIT;
    for (const listener of this.listeners) listener(event);
  }

  _createTrain(spec) {
    const profile = TRAIN_TYPES[spec.type];
    const train = {
      id: spec.id,
      name: spec.name,
      type: spec.type,
      profile,
      priority: profile.priority,
      source: spec.source,
      destination: spec.destination,
      route: [spec.source],
      leg: 0,
      progressKm: 0,
      status: "scheduled",
      departAt: spec.depart ?? 0,
      dwellLeft: 0,
      turnaroundLeft: 0,
      plannedArrival: Infinity,
      delay: 0,
      speed: 0,
      lastReroute: -Infinity,
      holdReason: null,
      reroutedAt: -Infinity,
    };
    this._plan(train, new Map());
    return train;
  }

  /** Give a train its initial route and timetable. */
  _plan(train, occupancy) {
    const { path } = dijkstra(this.network, train.source, train.destination, occupancy);
    train.route = path || [train.source, train.destination];
    train.leg = 0;
    train.progressKm = 0;
    train.plannedArrival = train.departAt + freeRunMinutes(this.network, train, train.route, 0, 0);
  }

  occupancy() {
    const counts = new Map();
    for (const train of this.trains) {
      if (train.status !== "running" && train.status !== "held") continue;
      if (train.status === "held" && train.progressKm === 0) continue;
      const edge = this.currentEdge(train);
      if (edge) counts.set(edge.id, (counts.get(edge.id) || 0) + 1);
    }
    return counts;
  }

  currentEdge(train) {
    if (train.leg >= train.route.length - 1) return null;
    return this.network.edgeBetween(train.route[train.leg], train.route[train.leg + 1]);
  }

  /** The station a decision about this train takes effect from. */
  decisionNode(train) {
    const onEdge = train.status === "running" && train.progressKm > 0;
    return onEdge ? train.route[train.leg + 1] : train.route[train.leg];
  }

  step(dtMinutes) {
    const dt = Math.max(0, Math.min(dtMinutes, 30));
    const occupancy = this.occupancy();
    this.time += dt;
    for (const train of this.trains) this._advance(train, dt, occupancy);
    while (this.time >= this.nextControl) {
      this.nextControl += 1;
      this._releaseHolds();
      this._control();
    }
  }

  _advance(train, dt, occupancy) {
    // Time left in this step once any wait (departure, dwell) is over. The
    // remainder is spent moving, so the step size never changes arrival times.
    let remaining = dt;
    switch (train.status) {
      case "scheduled": {
        if (this.time < train.departAt) return;
        train.status = "running";
        remaining = Math.min(dt, this.time - train.departAt);
        this._emit({ kind: "depart", time: this.time, train: train.id, text: `${train.id} departs ${train.source} for ${train.destination}.` });
        break;
      }
      case "dwell":
        train.speed = 0;
        train.dwellLeft -= dt;
        if (train.dwellLeft > 0) {
          this._updateDelay(train);
          return;
        }
        train.status = "running";
        remaining = -train.dwellLeft;
        train.dwellLeft = 0;
        break;
      case "held":
        train.speed = 0;
        this._updateDelay(train);
        return;
      case "arrived":
        train.speed = 0;
        train.turnaroundLeft -= dt;
        if (train.turnaroundLeft <= 0) this._turnRound(train, occupancy);
        return;
      default:
        break;
    }

    // Consume time, not distance: leftover time after reaching a station is
    // spent on the next section at that section's speed (backend integrator).
    for (let hops = 0; remaining > 1e-9 && hops < MAX_SECTIONS_PER_STEP; hops += 1) {
      const edge = this.currentEdge(train);
      if (!edge) break;
      if (edge.blocked && train.progressKm === 0) {
        this._hold(train, "route_unavailable");
        break;
      }
      const speed = effectiveSpeed(train.profile.maxSpeed, edge, occupancy.get(edge.id) || 0);
      train.speed = speed;
      if (speed <= 0) break;
      const kmPerMinute = speed / 60;
      const needed = (edge.distance_km - train.progressKm) / kmPerMinute;
      if (needed > remaining) {
        train.progressKm += remaining * kmPerMinute;
        remaining = 0;
        break;
      }
      remaining -= needed;
      train.leg += 1;
      train.progressKm = 0;
      const station = train.route[train.leg];
      if (station === train.destination) {
        this._updateDelay(train);
        train.status = "arrived";
        train.speed = 0;
        train.turnaroundLeft = TURNAROUND_MINUTES;
        this._emit({
          kind: "arrive",
          // The moment it actually reached the platform, not the end of the step.
          time: this.time - remaining,
          train: train.id,
          text: `${train.id} arrives ${station}${train.delay >= 1 ? `, ${Math.round(train.delay)} min late` : ", on time"}.`,
        });
        return;
      }
      const dwell = dwellFor(this.network, train, station);
      if (dwell > 0) {
        if (remaining >= dwell) {
          // The whole dwell fits in this step: stand, then carry on.
          remaining -= dwell;
          continue;
        }
        train.status = "dwell";
        train.dwellLeft = dwell - remaining;
        train.speed = 0;
        break;
      }
    }
    this._updateDelay(train);
  }

  _updateDelay(train) {
    const onEdge = train.status === "running" || (train.status === "held" && train.progressKm > 0);
    const rest = freeRunMinutes(this.network, train, train.route, train.leg, onEdge ? train.progressKm : 0);
    const dwell = train.status === "dwell" ? Math.max(0, train.dwellLeft) : 0;
    const projected = this.time + rest + dwell;
    if (Number.isFinite(projected) && Number.isFinite(train.plannedArrival)) {
      train.delay = Math.max(0, projected - train.plannedArrival);
    }
  }

  _turnRound(train, occupancy) {
    [train.source, train.destination] = [train.destination, train.source];
    train.departAt = this.time;
    this._plan(train, occupancy);
    train.status = "running";
    train.delay = 0;
    train.holdReason = null;
    this._emit({ kind: "depart", time: this.time, train: train.id, text: `${train.id} works back to ${train.destination}.` });
  }

  _hold(train, reason) {
    if (train.status === "held") return;
    train.status = "held";
    train.holdReason = reason;
    train.speed = 0;
    this._emit({
      kind: "hold",
      time: this.time,
      train: train.id,
      text: `${train.id} held at ${train.route[train.leg]}: no open route to ${train.destination}.`,
    });
  }

  /** backend/simulation.py _release_system_holds: a hold ends when its cause does. */
  _releaseHolds() {
    const occupancy = this.occupancy();
    for (const train of this.trains) {
      if (train.status !== "held") continue;
      const node = train.route[train.leg];
      const result = dijkstra(this.network, node, train.destination, occupancy);
      if (!result.path) continue;
      const tail = train.route.slice(train.leg);
      train.status = "running";
      train.holdReason = null;
      if (result.path.join() === tail.join()) {
        this._emit({ kind: "release", time: this.time, train: train.id, text: `${train.id} released at ${node}; its route is open again.` });
        continue;
      }
      train.route = train.route.slice(0, train.leg).concat(result.path);
      train.lastReroute = this.time;
      train.reroutedAt = this.time;
      this.automated += 1;
      const decision = {
        kind: "reroute",
        time: this.time,
        train: train.id,
        from: node,
        oldRoute: tail,
        newRoute: result.path,
        before: Infinity,
        after: result.minutes,
        trace: result.trace,
        reason: "closure",
        text: `${train.id} released at ${node} and rerouted via ${viaOf(result.path)}.`,
      };
      this.lastDecision = decision;
      this._emit(decision);
    }
  }

  /**
   * backend/agent.py _heuristic_decision + the safety gate: at most one
   * decision per simulated minute, highest priority first, and only if it is
   * valid and projected to help.
   */
  _control() {
    const occupancy = this.occupancy();
    const candidates = [...this.trains].sort(
      (a, b) => b.priority - a.priority || b.delay - a.delay || a.id.localeCompare(b.id)
    );

    for (const train of candidates) {
      if (train.status === "scheduled" || train.status === "arrived" || train.status === "held") continue;
      if (this.time - train.lastReroute < REROUTE_COOLDOWN_MINUTES) continue;

      const start = this.decisionNode(train);
      const startIndex = train.route.indexOf(start, train.leg);
      if (startIndex < 0) continue;
      const tail = train.route.slice(startIndex);
      if (tail.length < 2) continue;

      const tailEdges = this.network.routeEdges(tail) || [];
      const risky = tailEdges.filter(
        (edge) => edge.blocked || (occupancy.get(edge.id) || 0) / Math.max(1, edge.capacity) > 1
      );
      if (!risky.length && train.delay < 8) continue;

      const result = dijkstra(this.network, start, train.destination, occupancy);
      if (!result.path) continue; // the hold happens when the train reaches the closure
      if (result.path.join() === tail.join()) continue;

      const before = routeMinutes(this.network, tail, occupancy);
      const after = result.minutes;
      if (after >= before && !risky.length) continue;
      // Gate: valid (a different route) and beneficial (projected to help).
      if (!(after < before)) continue;

      train.route = train.route.slice(0, startIndex).concat(result.path);
      train.lastReroute = this.time;
      train.reroutedAt = this.time;
      this.automated += 1;
      const saved = Number.isFinite(before) ? `${Math.round(before - after)} min faster` : "the old route is closed";
      const decision = {
        kind: "reroute",
        time: this.time,
        train: train.id,
        from: start,
        oldRoute: tail,
        newRoute: result.path,
        before,
        after,
        trace: result.trace,
        reason: risky.some((e) => e.blocked) ? "closure" : risky.length ? "congestion" : "delay",
        text: `${train.id} rerouted at ${start} via ${viaOf(result.path)}: ${saved}.`,
      };
      this.lastDecision = decision;
      this._emit(decision);
      return decision;
    }
    return null;
  }

  counts() {
    const out = { running: 0, held: 0, dwell: 0, arrived: 0, scheduled: 0 };
    for (const train of this.trains) out[train.status] = (out[train.status] || 0) + 1;
    return out;
  }

  totalDelay() {
    return this.trains.reduce((sum, t) => sum + t.delay, 0);
  }
}

/** A short description of the distinctive middle of a route. */
export function viaOf(route) {
  if (route.length <= 3) return route.join("–");
  const middle = route.slice(1, -1);
  const pick = middle.length <= 3 ? middle : [middle[0], middle[Math.floor(middle.length / 2)], middle[middle.length - 1]];
  return pick.join("–");
}

/**
 * The page's roster. The first five are the backend's own scenario trains
 * (SCENARIO_LIBRARY, mixed_peak / kanpur_pressure); the rest fill the network
 * out in the same naming scheme. None of them is a real timetabled service.
 */
export const ROSTER = [
  { id: "RF-101", name: "Gomti Priority", type: "superfast", source: "NDLS", destination: "LKO", depart: 0 },
  { id: "RF-202", name: "Bareilly Passenger", type: "passenger", source: "NDLS", destination: "LKO", depart: 6 },
  { id: "RF-303", name: "Kanpur Freight", type: "freight", source: "CNB", destination: "NDLS", depart: 2 },
  { id: "RF-404", name: "Hardoi Local", type: "passenger", source: "LKO", destination: "MB", depart: 4 },
  { id: "RF-301", name: "Kanpur Superfast", type: "superfast", source: "NDLS", destination: "CNB", depart: 12 },
  { id: "RF-505", name: "Awadh Express", type: "express", source: "LKO", destination: "NDLS", depart: 0 },
  { id: "RF-606", name: "Ramganga Freight", type: "freight", source: "MB", destination: "ALJN", depart: 8 },
  { id: "RF-707", name: "Rohilkhand Express", type: "express", source: "BE", destination: "NDLS", depart: 3 },
  { id: "RF-808", name: "Ganga Passenger", type: "passenger", source: "KJN", destination: "HRI", depart: 10 },
  { id: "RF-909", name: "Doab Freight", type: "freight", source: "TDL", destination: "LKO", depart: 5 },
  { id: "RF-111", name: "Kasganj Passenger", type: "passenger", source: "KSJ", destination: "GZB", depart: 14 },
  { id: "RF-212", name: "Lucknow Superfast", type: "superfast", source: "LKO", destination: "NDLS", depart: 20 },
];
