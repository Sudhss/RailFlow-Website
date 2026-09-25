/**
 * Geometry shared with the RailFlow console.
 *
 * These are the console's own corridor modules (frontend/src/corridor:
 * projection, alignment, wave), carried over unchanged so the page draws the
 * network with the same projection, the same arc-length track and the same
 * propagating morph the product uses. Pure math, no renderer types; covered by
 * tests/site.test.mjs.
 */

export const WORLD_WIDTH = 420;

/** Web Mercator, in radians-ish units. Valid for the corridor's latitudes. */
export function mercator(lat, lng) {
  const clampedLat = Math.max(-85.05112878, Math.min(85.05112878, lat));
  const phi = (clampedLat * Math.PI) / 180;
  return {
    mx: (lng * Math.PI) / 180,
    my: Math.log(Math.tan(Math.PI / 4 + phi / 2)),
  };
}

function extent(values) {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (value < min) min = value;
    if (value > max) max = value;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 1, span: 1 };
  const span = max - min || 1;
  return { min, max, span };
}

/**
 * Fit a set of 2D points into a centred world box of WORLD_WIDTH across,
 * preserving the aspect ratio of the source so neither basis is distorted.
 */
function fitToWorld(points) {
  const ex = extent(points.map((p) => p.u));
  const ey = extent(points.map((p) => p.v));
  const scale = WORLD_WIDTH / Math.max(ex.span, ey.span * 1.6);
  return points.map((p) => ({
    id: p.id,
    x: (p.u - (ex.min + ex.span / 2)) * scale,
    z: (p.v - (ey.min + ey.span / 2)) * scale,
  }));
}

/**
 * Build both bases for a station list.
 * Returns { geographic: {id: {x,z}}, schematic: {id: {x,z}} }.
 */
export function buildBases(stations) {
  if (!stations || stations.length === 0) {
    return { geographic: {}, schematic: {}, order: [] };
  }

  const geoPoints = stations.map((station) => {
    const { mx, my } = mercator(Number(station.lat), Number(station.lng));
    // Mercator y grows north; world Z grows south, hence the negation.
    return { id: station.id, u: mx, v: -my };
  });

  // The diagram's y axis already points down the screen, which matches world Z.
  const schemaPoints = stations.map((station) => ({
    id: station.id,
    u: Number(station.x),
    v: Number(station.y),
  }));

  const geographic = {};
  for (const p of fitToWorld(geoPoints)) geographic[p.id] = { x: p.x, z: p.z };

  const schematic = {};
  for (const p of fitToWorld(schemaPoints)) schematic[p.id] = { x: p.x, z: p.z };

  return { geographic, schematic, order: stations.map((s) => s.id) };
}

/**
 * Corridor coordinate: 0 at the western-most station, 1 at the eastern-most,
 * measured on the true geography. Used as the phase offset for the layout wave
 * so a change sweeps along the corridor instead of cross-fading everywhere at
 * once -- the way a state change actually travels through a rail network.
 */
export function corridorCoordinates(geographic, order) {
  const xs = order.map((id) => geographic[id]?.x ?? 0);
  const { min, span } = extent(xs);
  const result = {};
  for (const id of order) {
    result[id] = ((geographic[id]?.x ?? 0) - min) / span;
  }
  return result;
}

/**
 * Ground-plane movement for a drag of (dx, dy) screen pixels.
 *
 * Drag-to-pan means the ground follows the cursor: grab a station, move the
 * mouse, and that station stays under the pointer. So the focus travels
 * *against* the drag.
 *
 * `heading` is the azimuth from the focus out to the camera, which makes the
 * direction from camera into the scene -(sin h, cos h) and screen-right
 * (cos h, -sin h) on the ground.
 *
 * The vertical component is divided by sin(pitch): a nearly flat camera sees
 * the ground heavily foreshortened, so one pixel of vertical drag covers far
 * more ground than one pixel of horizontal drag. Without this, panning feels
 * sluggish at low altitude and the ground slips under the cursor.
 */
export function panDelta(dx, dy, heading, pitch, pixelSize) {
  const rightX = Math.cos(heading);
  const rightZ = -Math.sin(heading);
  const forwardX = -Math.sin(heading);
  const forwardZ = -Math.cos(heading);

  const foreshorten = Math.max(0.25, Math.sin(pitch));
  const alongScreenY = (dy / foreshorten) * pixelSize;
  const alongScreenX = dx * pixelSize;

  return {
    x: -rightX * alongScreenX + forwardX * alongScreenY,
    z: -rightZ * alongScreenX + forwardZ * alongScreenY,
  };
}

const SAMPLES_PER_SECTION = 24;

function sub(a, b) {
  return { x: a.x - b.x, z: a.z - b.z };
}

function norm(v) {
  const length = Math.hypot(v.x, v.z);
  if (length < 1e-9) return { x: 0, z: 0 };
  return { x: v.x / length, z: v.z / length };
}

function dot(a, b) {
  return a.x * b.x + a.z * b.z;
}

function mix(a, b, t) {
  return { x: a.x + (b.x - a.x) * t, z: a.z + (b.z - a.z) * t };
}

/**
 * The axis a station is aligned on: the pair of neighbours pointing most
 * directly away from each other. A through station gets the corridor axis, a
 * terminal gets the direction of its single neighbour.
 */
export function throughAxis(stationId, neighbourIds, positions) {
  const here = positions[stationId];
  if (!here) return { x: 1, z: 0 };
  const dirs = neighbourIds
    .map((id) => positions[id])
    .filter(Boolean)
    .map((p) => norm(sub(p, here)));

  if (dirs.length === 0) return { x: 1, z: 0 };
  if (dirs.length === 1) return dirs[0];

  let best = dirs[0];
  let bestDot = Infinity;
  for (let i = 0; i < dirs.length; i += 1) {
    for (let j = i + 1; j < dirs.length; j += 1) {
      const d = dot(dirs[i], dirs[j]);
      if (d < bestDot) {
        bestDot = d;
        best = norm(sub(dirs[i], dirs[j]));
      }
    }
  }
  return best;
}

/** Cubic Bezier point. */
function bezier(p0, c0, c1, p1, t) {
  const u = 1 - t;
  const a = u * u * u;
  const b = 3 * u * u * t;
  const c = 3 * u * t * t;
  const d = t * t * t;
  return {
    x: a * p0.x + b * c0.x + c * c1.x + d * p1.x,
    z: a * p0.z + b * c0.z + c * c1.z + d * p1.z,
  };
}

/**
 * Sample one section into an arc-length table.
 *
 * `curvature` is how far the tangent is allowed to swing toward the station's
 * through-axis. Fast sections stay closer to straight, matching the larger
 * radii a high line speed requires.
 */
export function sampleSection(from, to, axisFrom, axisTo, curvature) {
  const chord = Math.hypot(to.x - from.x, to.z - from.z) || 1;
  const dir = norm(sub(to, from));

  // Orient each station's axis so it points along the direction of travel.
  const orientedFrom = dot(axisFrom, dir) < 0 ? { x: -axisFrom.x, z: -axisFrom.z } : axisFrom;
  const orientedTo = dot(axisTo, dir) < 0 ? { x: -axisTo.x, z: -axisTo.z } : axisTo;

  const tangentFrom = norm(mix(dir, orientedFrom, curvature));
  const tangentTo = norm(mix(dir, orientedTo, curvature));
  const handle = chord / 3;

  const p0 = from;
  const p1 = to;
  const c0 = { x: from.x + tangentFrom.x * handle, z: from.z + tangentFrom.z * handle };
  const c1 = { x: to.x - tangentTo.x * handle, z: to.z - tangentTo.z * handle };

  const points = [];
  const cumulative = [0];
  for (let i = 0; i <= SAMPLES_PER_SECTION; i += 1) {
    const point = bezier(p0, c0, c1, p1, i / SAMPLES_PER_SECTION);
    points.push(point);
    if (i > 0) {
      const previous = points[i - 1];
      cumulative.push(cumulative[i - 1] + Math.hypot(point.x - previous.x, point.z - previous.z));
    }
  }

  const length = cumulative[cumulative.length - 1] || 1;
  return { points, cumulative, length };
}

/**
 * Position at a fraction of the section measured along the rail, plus the unit
 * tangent and the signed curvature there. Curvature drives cant: a train
 * leaning into a curve is leaning by v^2 / r, not by an eased constant.
 */
export function sampleAt(section, fraction) {
  const { points, cumulative, length } = section;
  const target = Math.max(0, Math.min(1, fraction)) * length;

  let lo = 0;
  let hi = cumulative.length - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (cumulative[mid] <= target) lo = mid;
    else hi = mid;
  }

  const segmentLength = cumulative[hi] - cumulative[lo] || 1;
  const t = (target - cumulative[lo]) / segmentLength;
  const a = points[lo];
  const b = points[hi];
  const position = { x: a.x + (b.x - a.x) * t, z: a.z + (b.z - a.z) * t };
  const tangent = norm(sub(b, a));

  // Discrete curvature from the turn between the neighbouring segments.
  const prev = points[Math.max(0, lo - 1)];
  const next = points[Math.min(points.length - 1, hi + 1)];
  const inDir = norm(sub(a, prev));
  const outDir = norm(sub(next, b));
  const cross = inDir.x * outDir.z - inDir.z * outDir.x;
  const turn = Math.asin(Math.max(-1, Math.min(1, cross)));
  const span = Math.hypot(next.x - prev.x, next.z - prev.z) || 1;

  return { position, tangent, curvature: turn / span };
}

/**
 * Build the alignment for every edge, in one coordinate basis.
 * Returns a map of edgeId -> section table.
 */
export function buildAlignment(stations, edges, positions) {
  const neighbours = {};
  for (const station of stations) neighbours[station.id] = [];
  for (const edge of edges) {
    if (neighbours[edge.from]) neighbours[edge.from].push(edge.to);
    if (neighbours[edge.to]) neighbours[edge.to].push(edge.from);
  }

  const axes = {};
  for (const station of stations) {
    axes[station.id] = throughAxis(station.id, neighbours[station.id] || [], positions);
  }

  const sections = {};
  for (const edge of edges) {
    const from = positions[edge.from];
    const to = positions[edge.to];
    if (!from || !to) continue;
    // Line speed sets the minimum radius a section can hold: a 60 km/h branch
    // is allowed a noticeable bend, a 120 km/h main line stays close to
    // straight. The range is deliberately narrow -- a real corridor is mostly
    // tangent track with curves at the junctions, not a continuous snake.
    const speed = Number(edge.avg_speed) || 60;
    const curvature = Math.max(0.04, Math.min(0.30, (95 - speed) / 150));
    sections[edge.id] = sampleSection(from, to, axes[edge.from], axes[edge.to], curvature);
  }
  return sections;
}

export const WAVE_SPREAD = 0.55;

/**
 * @param corridor 0..1 position of this point along the corridor
 * @param origin   0..1 position the change radiates from
 * @param progress 0..1 overall transition progress
 */
export function waveAt(corridor, origin, progress) {
  const distance = Math.abs(corridor - origin);
  const delay = distance * WAVE_SPREAD;
  const local = (progress - delay) / (1 - WAVE_SPREAD);
  const t = Math.max(0, Math.min(1, local));
  return t * t * (3 - 2 * t); // smoothstep
}

/**
 * The corridor coordinate of a point part-way along a section.
 *
 * The shader hands every track vertex its own corridor coordinate and the GPU
 * interpolates it across the section, so the morph phase varies *along* a
 * section while the wave is passing. Anything the CPU places on that track --
 * trains, sleepers, signals -- has to use the phase at its own point, computed
 * the same way, or it leaves the rails mid-transition.
 */
export function corridorAt(fromCorridor, toCorridor, fraction) {
  return fromCorridor + (toCorridor - fromCorridor) * fraction;
}

/** The same function as GLSL, injected into the track shader. */
export const WAVE_GLSL = /* glsl */ `
  uniform float uWaveOrigin;
  uniform float uWaveProgress;

  float waveAt(float corridor) {
    float distance = abs(corridor - uWaveOrigin);
    float delay = distance * ${WAVE_SPREAD.toFixed(3)};
    float local = (uWaveProgress - delay) / (1.0 - ${WAVE_SPREAD.toFixed(3)});
    return smoothstep(0.0, 1.0, clamp(local, 0.0, 1.0));
  }
`;
