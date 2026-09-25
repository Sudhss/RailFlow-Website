/**
 * The page's WebGL stage: the real RailFlow network, drawn in both of its
 * coordinate systems at once, with the trains of the in-page simulation
 * running on it.
 *
 * Everything that moves here is driven by something: trains by the
 * simulation, the morph by the basis the reader chose, the frontier rings by
 * the order Dijkstra actually settled stations in, the bars by the cost
 * function evaluated against live occupancy. Nothing is animated for its own
 * sake.
 *
 * Draw calls: ground, track, sleepers, structures (platforms, canopies,
 * signals), cost bars, frontier rings, trains, trails, light halos. Nine,
 * whatever the train count.
 */

import * as THREE from "three";
import { STATIONS, EDGES } from "./network.js";
import {
  buildBases,
  corridorCoordinates,
  buildAlignment,
  sampleAt,
  waveAt,
  corridorAt,
  panDelta,
  WAVE_GLSL,
} from "./geometry.js";
import { dynamicWeight, congestionState, signalAspect } from "./sim.js";

/* ------------------------------------------------------------ constants */

// World units are ~1.19 km (the 420-unit world box spans the corridor). Rail
// objects are drawn about twelve times life size so a train stays a readable
// object from region scale, and screen-space floors keep them visible beyond
// that. The ratios between them are kept true: a coach is ~1.9 gauges wide,
// sleepers ~1.7 gauges long, a platform about one train long.
const SCALE = 0.4;
const TRACK_HALF_WIDTH = 0.45 * SCALE;
const RAIL_OFFSET = 0.36; // fraction of the half width
const COACH_LENGTH = 1.5 * SCALE;
const COACH_GAP = 0.14 * SCALE;
const COACH_WIDTH = 0.5 * SCALE;
const COACH_HEIGHT = 0.46 * SCALE;
const SLEEPER_SPACING = 0.32 * SCALE;
const TRACK_Y = 0.03;
const PLATFORM_WIDTH = 0.34;
const PLATFORM_LENGTH = { terminal: 7.5, junction: 6.5, major: 6.5, minor: 4.5 };
const LAMP_REACH = 2.8;
const MAX_LAMPS = 16;
const WAVE_SECONDS = 2.8;
const MAX_CANT = 0.105; // ~6 degrees, the most real track uses

// Night: the stage is lit by what is actually on the railway -- headlamps,
// platform lighting and signals -- over a near-black ground.
const PALETTE = {
  ground: "#07090b",
  grid: "#11161a",
  gridMajor: "#1a2126",
  steel: "#5f6c74",
  platform: "#8e989e",
  canopy: "#4c555b",
  mast: "#3a4247",
  headlight: "#ffe7bf",
  sodium: "#ffae57",
  signalGreen: "#4fd691",
  signalYellow: "#f5c451",
  signalRed: "#ff5548",
  rail: "#aeb9c0",
  route: "#dce5ea",
  clear: "#3fb27f",
  caution: "#d9a441",
  danger: "#e0695e",
  station: "#c4ced4",
  stationMinor: "#76838b",
  coach: "#b9c3c9",
  wagon: "#7b6f63",
  lamp: "#fff1cf",
};

const STATE_CODE = { normal: 0, busy: 1, congested: 2, closed: 3 };

function color(hex) {
  return new THREE.Color(hex);
}

function dampFactor(rate, dt) {
  return 1 - Math.exp(-rate * dt);
}

function shortestAngle(from, to) {
  let d = (to - from) % (Math.PI * 2);
  if (d > Math.PI) d -= Math.PI * 2;
  if (d < -Math.PI) d += Math.PI * 2;
  return d;
}

/* -------------------------------------------------------------- shaders */

const FOG_GLSL = /* glsl */ `
  uniform vec3 uFogColor;
  uniform float uFogNear;
  uniform float uFogFar;
  vec3 applyFog(vec3 c, float depth) {
    return mix(c, uFogColor, smoothstep(uFogNear, uFogFar, depth));
  }
`;

// Night light: headlamp throws (a cone ahead of each train plus a little
// spill) and sodium pools at stations. Shared by ground and track.
const LIGHT_GLSL = /* glsl */ `
  uniform vec4 uLamps[${MAX_LAMPS}];
  uniform int uLampCount;
  uniform vec3 uPools[${STATIONS.length}];
  uniform int uPoolCount;
  uniform vec3 uLampColor;
  uniform vec3 uPoolColor;
  uniform float uLampReach;
  vec3 nightLight(vec2 p) {
    vec3 light = vec3(0.0);
    for (int i = 0; i < ${MAX_LAMPS}; i++) {
      if (i >= uLampCount) break;
      vec2 d = p - uLamps[i].xy;
      float dist = length(d);
      float along = dot(d, uLamps[i].zw);
      float cone = smoothstep(0.62, 0.97, along / max(dist, 1e-4));
      float fall = 1.0 - smoothstep(0.0, uLampReach, dist);
      float spill = 1.0 - smoothstep(0.0, uLampReach * 0.16, dist);
      light += uLampColor * (cone * fall * fall + spill * 0.3);
    }
    for (int i = 0; i < ${STATIONS.length}; i++) {
      if (i >= uPoolCount) break;
      float fall = 1.0 - smoothstep(0.0, uPools[i].z, length(p - uPools[i].xy));
      light += uPoolColor * fall * fall * 0.5;
    }
    return light;
  }
`;

const TRACK_VERTEX = /* glsl */ `
  ${WAVE_GLSL}
  attribute vec2 aPosA;
  attribute vec2 aPosB;
  attribute vec2 aNrmA;
  attribute vec2 aNrmB;
  attribute vec2 aLen;
  attribute float aAcross;
  attribute float aCorridor;
  attribute float aEdge;
  attribute float aAlong;

  uniform float uFrom;
  uniform float uTo;
  uniform float uHalfWidth;
  uniform float uPixelPerDepth;
  uniform sampler2D uState;
  uniform float uEdgeCount;

  varying float vAcross;
  varying float vAlong;
  varying float vLen;
  varying float vDepth;
  varying float vPixels;
  varying vec4 vState;
  varying vec4 vTrace;
  varying vec2 vWorld;

  void main() {
    float b = mix(uFrom, uTo, waveAt(aCorridor));
    vec2 p = mix(aPosA, aPosB, b);
    vec2 n = normalize(mix(aNrmA, aNrmB, b));

    // True width, with a floor of ~1.5 px so the network never breaks up
    // into dashes at region scale.
    vec4 centre = modelViewMatrix * vec4(p.x, ${TRACK_Y}, p.y, 1.0);
    float pixel = -centre.z * uPixelPerDepth;
    float halfWidth = max(uHalfWidth, pixel * 2.2);

    vec3 world = vec3(p.x + n.x * halfWidth * aAcross, ${TRACK_Y}, p.y + n.y * halfWidth * aAcross);
    vWorld = world.xz;
    vec4 mv = modelViewMatrix * vec4(world, 1.0);
    gl_Position = projectionMatrix * mv;

    float u = (aEdge + 0.5) / uEdgeCount;
    vState = texture2D(uState, vec2(u, 0.25));
    vTrace = texture2D(uState, vec2(u, 0.75));
    vAcross = aAcross;
    vAlong = aAlong;
    vLen = mix(aLen.x, aLen.y, b);
    vDepth = -mv.z;
    vPixels = halfWidth / max(pixel, 1e-4);
  }
`;

const TRACK_FRAGMENT = /* glsl */ `
  ${FOG_GLSL}
  ${LIGHT_GLSL}
  uniform float uTime;
  uniform vec3 uSteel;
  uniform vec3 uRail;
  uniform vec3 uRoute;
  uniform vec3 uClear;
  uniform vec3 uCaution;
  uniform vec3 uDanger;

  varying float vAcross;
  varying float vAlong;
  varying float vLen;
  varying float vDepth;
  varying float vPixels;
  varying vec4 vState;
  varying vec4 vTrace;
  varying vec2 vWorld;

  // How far along this section a sweep that started at t0 has travelled,
  // measured in the direction the route runs.
  float sweep(float t0, float reversed, float duration) {
    if (t0 < 0.0) return 0.0;
    float reach = (uTime - t0) / duration;
    float at = reversed > 0.5 ? 1.0 - vAlong : vAlong;
    return clamp((reach - at) * 6.0, 0.0, 1.0);
  }

  void main() {
    float code = vState.r;
    vec3 tone = uSteel;
    if (code > 0.5 && code < 1.5) tone = uCaution;
    else if (code > 1.5 && code < 2.5) tone = mix(uCaution, uDanger, 0.7);

    // A closed section is dashed red: the one state that must read even
    // without colour vision.
    if (code > 2.5) {
      float dash = step(0.5, fract(vAlong * vLen / 2.2));
      tone = mix(uDanger, uDanger * 0.28, dash);
    }

    float explored = sweep(vTrace.b, vTrace.a, 0.45);
    tone = mix(tone, mix(uSteel, uClear, 0.4), explored * 0.7);

    if (vState.b > 0.5 && code < 2.5) tone = mix(tone, uRoute, 0.8);

    float path = sweep(vTrace.r, vTrace.g, 0.3);
    tone = mix(tone, mix(uClear, uRoute, 0.25) * 1.3, path);

    // Far away the ribbon is a single line in the section's state colour.
    float across = abs(vAcross);
    float aa = 1.0 / max(vPixels, 1.0);
    float near = smoothstep(5.0, 14.0, vPixels);
    float rail = 1.0 - smoothstep(0.035, 0.035 + aa * 1.5, abs(across - ${RAIL_OFFSET.toFixed(2)}));
    // Up close only the rails are drawn here; the sleepers beneath carry the
    // rest of the formation, with a faint state wash between them.
    vec3 c = mix(tone, mix(uRail, tone, 0.5), near * rail);
    // Rails glint where a headlamp or platform light reaches them.
    c += nightLight(vWorld) * (0.25 + 0.9 * rail * near);

    float alpha = 1.0 - smoothstep(1.0 - aa * 2.0, 1.0, across);
    alpha *= mix(1.0, max(rail, 0.16), near);
    gl_FragColor = vec4(applyFog(c, vDepth), alpha);
    #include <colorspace_fragment>
  }
`;

const GROUND_VERTEX = /* glsl */ `
  varying vec2 vWorld;
  varying float vDepth;
  void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    vWorld = world.xz;
    vec4 mv = viewMatrix * world;
    vDepth = -mv.z;
    gl_Position = projectionMatrix * mv;
  }
`;

const GROUND_FRAGMENT = /* glsl */ `
  ${FOG_GLSL}
  ${LIGHT_GLSL}
  uniform vec3 uGround;
  uniform vec3 uGrid;
  uniform vec3 uGridMajor;
  uniform float uMinor;
  varying vec2 vWorld;
  varying float vDepth;

  float gridLine(vec2 p, float spacing) {
    vec2 g = abs(fract(p / spacing - 0.5) - 0.5) / fwidth(p / spacing);
    return 1.0 - min(min(g.x, g.y), 1.0);
  }

  void main() {
    float minor = gridLine(vWorld, uMinor) * (1.0 - smoothstep(60.0, 260.0, vDepth));
    float major = gridLine(vWorld, uMinor * 5.0);
    vec3 c = uGround;
    c = mix(c, uGrid, minor * 0.9);
    c = mix(c, uGridMajor, major);
    c += nightLight(vWorld) * (0.2 + 0.3 * max(minor, major));
    gl_FragColor = vec4(applyFog(c, vDepth), 1.0);
    #include <colorspace_fragment>
  }
`;

// Instanced solids: stations, cost bars, coaches. Hemispheric light plus one
// key light -- enough to read form, no shadows, no PBR.
const SOLID_VERTEX = /* glsl */ `
  varying vec3 vNormalW;
  varying vec3 vColor;
  varying float vDepth;
  void main() {
    mat4 m = modelMatrix * instanceMatrix;
    vec4 world = m * vec4(position, 1.0);
    vNormalW = normalize(mat3(m) * normal);
    #ifdef USE_INSTANCING_COLOR
      vColor = instanceColor;
    #else
      vColor = vec3(1.0);
    #endif
    vec4 mv = viewMatrix * world;
    vDepth = -mv.z;
    gl_Position = projectionMatrix * mv;
  }
`;

const SOLID_FRAGMENT = /* glsl */ `
  ${FOG_GLSL}
  varying vec3 vNormalW;
  varying vec3 vColor;
  varying float vDepth;
  void main() {
    vec3 n = normalize(vNormalW);
    float sky = 0.5 + 0.5 * n.y;
    float key = max(dot(n, normalize(vec3(-0.35, 0.8, 0.45))), 0.0);
    vec3 c = vColor * (0.34 + 0.34 * sky + 0.42 * key);
    gl_FragColor = vec4(applyFog(c, vDepth), 1.0);
    #include <colorspace_fragment>
  }
`;

// Vehicles: the same lighting, plus the detail that makes a box read as rolling
// stock up close -- lit windows on coaches, a cab on the locomotive, ribs on
// wagons -- faded out by screen-space derivative before it can alias.
const VEHICLE_VERTEX = /* glsl */ `
  attribute float aKind;
  varying vec3 vNormalW;
  varying vec3 vNormalL;
  varying vec3 vLocal;
  varying vec3 vColor;
  varying float vDepth;
  varying float vKind;
  void main() {
    mat4 m = modelMatrix * instanceMatrix;
    vec4 world = m * vec4(position, 1.0);
    vNormalW = normalize(mat3(m) * normal);
    vNormalL = normal;
    vLocal = position;
    vColor = instanceColor;
    vKind = aKind;
    vec4 mv = viewMatrix * world;
    vDepth = -mv.z;
    gl_Position = projectionMatrix * mv;
  }
`;

const VEHICLE_FRAGMENT = /* glsl */ `
  ${FOG_GLSL}
  uniform vec3 uGlass;
  uniform vec3 uInterior;
  varying vec3 vNormalW;
  varying vec3 vNormalL;
  varying vec3 vLocal;
  varying vec3 vColor;
  varying float vDepth;
  varying float vKind;
  void main() {
    if (vKind > 2.5) {
      gl_FragColor = vec4(applyFog(vColor * 1.6, vDepth * 0.7), 1.0);
      #include <colorspace_fragment>
      return;
    }
    vec3 n = normalize(vNormalW);
    float sky = 0.5 + 0.5 * n.y;
    float key = max(dot(n, normalize(vec3(-0.35, 0.8, 0.45))), 0.0);
    vec3 c = vColor * (0.34 + 0.34 * sky + 0.42 * key);

    float side = step(0.5, abs(vNormalL.z));
    float front = step(0.5, abs(vNormalL.x));
    float roof = step(0.5, vNormalL.y);
    float detail = 1.0 - smoothstep(0.25, 0.6, fwidth(vLocal.x * 9.0));

    if (vKind < 0.5) {
      // Coach: a band of windows, lit from inside.
      float band = step(0.5, vLocal.y) * step(vLocal.y, 0.78);
      float pane = step(fract(vLocal.x * 9.0 + 0.5), 0.64) * step(abs(vLocal.x), 0.44);
      c = mix(c, uInterior, side * band * pane * detail);
      c *= 1.0 - roof * 0.18;
    } else if (vKind < 1.5) {
      // Locomotive: cab glazing at the leading end and a darker roof.
      float cab = step(0.34, vLocal.x) * step(0.54, vLocal.y) * step(vLocal.y, 0.82);
      c = mix(c, uGlass, max(side * cab, front * step(0.0, vLocal.x) * step(0.54, vLocal.y) * step(vLocal.y, 0.82)) * detail);
      c *= 1.0 - roof * 0.3;
    } else {
      // Wagon: vertical ribs.
      float rib = step(fract(vLocal.x * 8.0), 0.1);
      c *= 1.0 - rib * side * 0.35 * detail;
    }
    // A dark underframe line along the bottom of every vehicle.
    c *= 1.0 - (1.0 - step(0.14, vLocal.y)) * 0.55 * detail;
    // From region scale a train is a few pixels: lift it so it reads as the
    // brightest thing on its section.
    c += vColor * 0.45 * (1.0 - detail);
    gl_FragColor = vec4(applyFog(c, vDepth * mix(1.0, 0.6, 1.0 - detail)), 1.0);
    #include <colorspace_fragment>
  }
`;

// Halos: camera-facing glows for every light source -- signal lamps and
// headlamps. They carry a pixel floor, so from region scale the network is
// dotted with its own signal aspects, and up close each lamp blooms.
const HALO_VERTEX = /* glsl */ `
  attribute vec3 aCenter;
  attribute vec3 aColor;
  attribute float aSize;
  attribute float aMinPx;
  uniform float uPixelPerDepth;
  varying vec2 vUv;
  varying vec3 vColor;
  varying float vDepth;
  void main() {
    vec4 mv = viewMatrix * vec4(aCenter, 1.0);
    float pixel = -mv.z * uPixelPerDepth;
    float r = max(aSize, pixel * aMinPx);
    mv.xy += position.xy * r;
    gl_Position = projectionMatrix * mv;
    vUv = position.xy;
    vColor = aColor;
    vDepth = -mv.z;
  }
`;

const HALO_FRAGMENT = /* glsl */ `
  ${FOG_GLSL}
  varying vec2 vUv;
  varying vec3 vColor;
  varying float vDepth;
  void main() {
    float d = length(vUv);
    if (d > 1.0) discard;
    float glow = exp(-d * d * 5.0) * (1.0 - d);
    float core = 1.0 - smoothstep(0.12, 0.28, d);
    vec3 c = vColor * glow + mix(vColor, vec3(1.0), 0.55) * core;
    float fog = smoothstep(uFogNear, uFogFar, vDepth) * 0.5;
    gl_FragColor = vec4(c * (1.0 - fog), 1.0);
    #include <colorspace_fragment>
  }
`;

// Sleepers are placed entirely on the GPU so the morph moves them for free:
// each instance carries its position and heading in both bases.
const SLEEPER_VERTEX = /* glsl */ `
  ${WAVE_GLSL}
  attribute vec2 aPosA;
  attribute vec2 aPosB;
  attribute vec2 aTanA;
  attribute vec2 aTanB;
  attribute float aCorridor;
  uniform float uFrom;
  uniform float uTo;
  uniform float uVisible;
  varying float vDepth;
  varying float vShade;
  void main() {
    float b = mix(uFrom, uTo, waveAt(aCorridor));
    vec2 p = mix(aPosA, aPosB, b);
    vec2 t = normalize(mix(aTanA, aTanB, b));
    vec2 n = vec2(-t.y, t.x);
    vec3 local = position;
    vec3 world = vec3(p.x + t.x * local.x + n.x * local.z, local.y, p.y + t.y * local.x + n.y * local.z);
    vec4 mv = viewMatrix * vec4(world, 1.0);
    // Collapse sleepers that are too far away to resolve; they would only
    // alias into moire.
    float keep = uVisible * (1.0 - step(30.0, -mv.z));
    gl_Position = keep > 0.5 ? projectionMatrix * mv : vec4(2.0, 2.0, 2.0, 1.0);
    vDepth = -mv.z;
    vShade = 0.7 + 0.3 * step(0.015, local.y);
  }
`;

const SLEEPER_FRAGMENT = /* glsl */ `
  ${FOG_GLSL}
  uniform vec3 uColor;
  varying float vDepth;
  varying float vShade;
  void main() {
    gl_FragColor = vec4(applyFog(uColor * vShade, vDepth), 1.0);
    #include <colorspace_fragment>
  }
`;

// Dijkstra's frontier: a ring appears at each station at the moment it was
// settled, in settlement order.
const RING_VERTEX = /* glsl */ `
  attribute float aStart;
  attribute float aOnPath;
  uniform float uTime;
  uniform float uFade;
  varying float vAlpha;
  varying float vDepth;
  varying float vOnPath;
  void main() {
    float t = uTime - aStart;
    float live = aStart < 0.0 ? 0.0 : step(0.0, t);
    float grow = clamp(t / 0.35, 0.0, 1.0);
    float scale = live * (0.35 + 0.65 * (1.0 - pow(1.0 - grow, 3.0)));
    vec4 world = modelMatrix * instanceMatrix * vec4(position * scale, 1.0);
    vec4 mv = viewMatrix * world;
    gl_Position = projectionMatrix * mv;
    vAlpha = live * uFade * (0.55 + 0.45 * (1.0 - grow));
    vDepth = -mv.z;
    vOnPath = aOnPath;
  }
`;

const RING_FRAGMENT = /* glsl */ `
  ${FOG_GLSL}
  uniform vec3 uClear;
  uniform vec3 uRoute;
  varying float vAlpha;
  varying float vDepth;
  varying float vOnPath;
  void main() {
    vec3 c = mix(uClear, uRoute, vOnPath * 0.35);
    gl_FragColor = vec4(applyFog(c, vDepth), vAlpha);
    #include <colorspace_fragment>
  }
`;

// Trails: where each train has actually been over the last TRAIL_MINUTES of
// simulated time. Length is therefore speed -- a superfast draws a long
// stroke, a freight a short one, a held train none at all.
const TRAIL_VERTEX = /* glsl */ `
  attribute float aAlpha;
  attribute vec3 aColor;
  attribute float aSide;
  varying float vSide;
  varying float vAlpha;
  varying vec3 vColor;
  varying float vDepth;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    vAlpha = aAlpha;
    vSide = aSide;
    vColor = aColor;
    vDepth = -mv.z;
  }
`;

const TRAIL_FRAGMENT = /* glsl */ `
  ${FOG_GLSL}
  varying float vAlpha;
  varying float vSide;
  varying vec3 vColor;
  varying float vDepth;
  void main() {
    float fog = smoothstep(uFogNear, uFogFar, vDepth) * 0.35;
    // A soft cross-section: bright core, feathered edges.
    float profile = 1.0 - vSide * vSide;
    gl_FragColor = vec4(vColor, vAlpha * profile * (1.0 - fog));
    #include <colorspace_fragment>
  }
`;

const TRAIL_MINUTES = 30;
const TRAIL_SAMPLE = 0.5;

/* ---------------------------------------------------------------- stage */

export class Stage {
  constructor(canvas, sim, { reducedMotion = false } = {}) {
    this.canvas = canvas;
    this.sim = sim;
    this.network = sim.network;
    this.reducedMotion = reducedMotion;
    this.clock = 0;
    this.listeners = { hover: new Set(), pick: new Set() };

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(color(PALETTE.ground), 1);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(36, 1, 0.1, 4000);

    this.fog = {
      uFogColor: { value: color(PALETTE.ground) },
      uFogNear: { value: 300 },
      uFogFar: { value: 900 },
    };

    this.lights = {
      uLamps: { value: Array.from({ length: MAX_LAMPS }, () => new THREE.Vector4()) },
      uLampCount: { value: 0 },
      uPools: { value: STATIONS.map(() => new THREE.Vector3()) },
      uPoolCount: { value: STATIONS.length },
      uLampColor: { value: color(PALETTE.headlight).multiplyScalar(0.9) },
      uPoolColor: { value: color(PALETTE.sodium).multiplyScalar(0.55) },
      uLampReach: { value: LAMP_REACH },
    };

    this._buildBases();
    this._buildGround();
    this._buildTrack();
    this._buildSleepers();
    this._buildStations();
    this._buildBars();
    this._buildRings();
    this._buildTrains();
    this._buildTrails();
    this._buildHalos();

    // Camera rig: the goal is where a step wants the camera; `view` chases it.
    this.goal = { x: 0, z: 0, extent: 470, heading: -0.2, pitch: 0.95, lensX: 0, lensY: 0 };
    this.view = { ...this.goal, distance: 600 };
    this.orbit = { heading: 0, pitch: 0 };
    this.follow = null;
    this.freeLook = false;
    this.barsTarget = 0;
    this.barsIn = 0;
    this.traceFade = 0;
    this.traceUntil = -1;
    this.highlightTrain = null;
    this.hovered = null;

    this.size = { width: 1, height: 1 };
    this.running = false;
    this._frame = this._frame.bind(this);
    this._bindPointer();
  }

  /* ------------------------------------------------------------- build */

  _buildBases() {
    const bases = buildBases(STATIONS);
    this.positions = [bases.schematic, bases.geographic];
    this.corridor = corridorCoordinates(bases.geographic, bases.order);
    this.alignments = [
      buildAlignment(STATIONS, EDGES, bases.schematic),
      buildAlignment(STATIONS, EDGES, bases.geographic),
    ];
    this.edgeIndex = new Map(EDGES.map((e, i) => [e.id, i]));
    this.basis = { from: 0, to: 0, origin: 0, progress: 1, started: 0 };
  }

  /** The basis blend at a corridor coordinate: the JS twin of waveAt in GLSL. */
  blendAt(corridor) {
    const { from, to, origin, progress } = this.basis;
    return from + (to - from) * waveAt(corridor, origin, progress);
  }

  /** Mixed position, tangent and curvature at a fraction along an edge. */
  pointOnEdge(edge, fraction) {
    const a = sampleAt(this.alignments[0][edge.id], fraction);
    const b = sampleAt(this.alignments[1][edge.id], fraction);
    const c = corridorAt(this.corridor[edge.from], this.corridor[edge.to], fraction);
    const t = this.blendAt(c);
    const tx = a.tangent.x + (b.tangent.x - a.tangent.x) * t;
    const tz = a.tangent.z + (b.tangent.z - a.tangent.z) * t;
    const tl = Math.hypot(tx, tz) || 1;
    return {
      x: a.position.x + (b.position.x - a.position.x) * t,
      z: a.position.z + (b.position.z - a.position.z) * t,
      tx: tx / tl,
      tz: tz / tl,
      curvature: a.curvature + (b.curvature - a.curvature) * t,
      length: this.alignments[0][edge.id].length * (1 - t) + this.alignments[1][edge.id].length * t,
    };
  }

  stationPosition(id) {
    const a = this.positions[0][id];
    const b = this.positions[1][id];
    const t = this.blendAt(this.corridor[id]);
    return { x: a.x + (b.x - a.x) * t, z: a.z + (b.z - a.z) * t };
  }

  _buildGround() {
    const geometry = new THREE.PlaneGeometry(6000, 6000);
    geometry.rotateX(-Math.PI / 2);
    this.groundMaterial = new THREE.ShaderMaterial({
      vertexShader: GROUND_VERTEX,
      fragmentShader: GROUND_FRAGMENT,
      uniforms: {
        ...this.fog,
        uGround: { value: color(PALETTE.ground) },
        uGrid: { value: color(PALETTE.grid) },
        uGridMajor: { value: color(PALETTE.gridMajor) },
        // 10 km minor grid: the world box is 420 units across ~500 km.
        uMinor: { value: 10 / 1.19 },
        ...this.lights,
      },
    });
    const ground = new THREE.Mesh(geometry, this.groundMaterial);
    ground.renderOrder = -2;
    this.scene.add(ground);
  }

  _buildTrack() {
    const edgeCount = EDGES.length;
    this.stateData = new Float32Array(edgeCount * 2 * 4);
    for (let i = 0; i < edgeCount; i += 1) {
      this.stateData[(edgeCount + i) * 4 + 0] = -1; // path sweep start
      this.stateData[(edgeCount + i) * 4 + 2] = -1; // explored sweep start
    }
    this.stateTexture = new THREE.DataTexture(this.stateData, edgeCount, 2, THREE.RGBAFormat, THREE.FloatType);
    this.stateTexture.minFilter = THREE.NearestFilter;
    this.stateTexture.magFilter = THREE.NearestFilter;
    this.stateTexture.needsUpdate = true;

    const posA = [];
    const posB = [];
    const nrmA = [];
    const nrmB = [];
    const lens = [];
    const across = [];
    const corridor = [];
    const edgeAttr = [];
    const along = [];
    const index = [];
    this.pickSamples = [];

    EDGES.forEach((edge, e) => {
      const secA = this.alignments[0][edge.id];
      const secB = this.alignments[1][edge.id];
      const samples = [];
      for (let i = 0; i < secA.points.length; i += 1) {
        // Both bases are sampled at the same arc-length fraction, so a vertex
        // and a train at that fraction share one formula.
        const f = secA.cumulative[i] / secA.length;
        const a = sampleAt(secA, f);
        const b = sampleAt(secB, f);
        const c = corridorAt(this.corridor[edge.from], this.corridor[edge.to], f);
        const base = posA.length / 2;
        for (const side of [-1, 1]) {
          posA.push(a.position.x, a.position.z);
          posB.push(b.position.x, b.position.z);
          nrmA.push(-a.tangent.z, a.tangent.x);
          nrmB.push(-b.tangent.z, b.tangent.x);
          lens.push(secA.length, secB.length);
          across.push(side);
          corridor.push(c);
          edgeAttr.push(e);
          along.push(f);
        }
        if (i > 0) index.push(base - 2, base - 1, base, base - 1, base + 1, base);
        samples.push(f);
      }
      this.pickSamples.push({ edge, samples });
    });

    const geometry = new THREE.BufferGeometry();
    // `position` is unused by the shader but three.js wants one for bounds.
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(across.length * 3), 3));
    geometry.setAttribute("aPosA", new THREE.Float32BufferAttribute(posA, 2));
    geometry.setAttribute("aPosB", new THREE.Float32BufferAttribute(posB, 2));
    geometry.setAttribute("aNrmA", new THREE.Float32BufferAttribute(nrmA, 2));
    geometry.setAttribute("aNrmB", new THREE.Float32BufferAttribute(nrmB, 2));
    geometry.setAttribute("aLen", new THREE.Float32BufferAttribute(lens, 2));
    geometry.setAttribute("aAcross", new THREE.Float32BufferAttribute(across, 1));
    geometry.setAttribute("aCorridor", new THREE.Float32BufferAttribute(corridor, 1));
    geometry.setAttribute("aEdge", new THREE.Float32BufferAttribute(edgeAttr, 1));
    geometry.setAttribute("aAlong", new THREE.Float32BufferAttribute(along, 1));
    geometry.setIndex(index);

    this.basisUniforms = {
      uFrom: { value: 0 },
      uTo: { value: 0 },
      uWaveOrigin: { value: 0 },
      uWaveProgress: { value: 1 },
    };
    this.timeUniform = { value: 0 };
    this.pixelUniform = { value: 0.001 };

    this.trackMaterial = new THREE.ShaderMaterial({
      vertexShader: TRACK_VERTEX,
      fragmentShader: TRACK_FRAGMENT,
      uniforms: {
        ...this.fog,
        ...this.basisUniforms,
        uTime: this.timeUniform,
        uPixelPerDepth: this.pixelUniform,
        uHalfWidth: { value: TRACK_HALF_WIDTH },
        ...this.lights,
        uState: { value: this.stateTexture },
        uEdgeCount: { value: edgeCount },
        uSteel: { value: color(PALETTE.steel) },
        uRail: { value: color(PALETTE.rail) },
        uRoute: { value: color(PALETTE.route) },
        uClear: { value: color(PALETTE.clear) },
        uCaution: { value: color(PALETTE.caution) },
        uDanger: { value: color(PALETTE.danger) },
      },
      transparent: true,
      depthWrite: false,
    });
    const track = new THREE.Mesh(geometry, this.trackMaterial);
    track.frustumCulled = false;
    track.renderOrder = -1;
    this.scene.add(track);
  }

  _buildSleepers() {
    const posA = [];
    const posB = [];
    const tanA = [];
    const tanB = [];
    const corridor = [];
    for (const edge of EDGES) {
      const secA = this.alignments[0][edge.id];
      const count = Math.max(2, Math.floor(secA.length / SLEEPER_SPACING));
      for (let i = 0; i < count; i += 1) {
        const f = (i + 0.5) / count;
        const a = sampleAt(secA, f);
        const b = sampleAt(this.alignments[1][edge.id], f);
        posA.push(a.position.x, a.position.z);
        posB.push(b.position.x, b.position.z);
        tanA.push(a.tangent.x, a.tangent.z);
        tanB.push(b.tangent.x, b.tangent.z);
        corridor.push(corridorAt(this.corridor[edge.from], this.corridor[edge.to], f));
      }
    }
    const box = new THREE.BoxGeometry(0.045, 0.02, TRACK_HALF_WIDTH * 1.25);
    box.translate(0, 0.01, 0);
    const geometry = new THREE.InstancedBufferGeometry().copy(box);
    geometry.instanceCount = corridor.length;
    geometry.setAttribute("aPosA", new THREE.InstancedBufferAttribute(new Float32Array(posA), 2));
    geometry.setAttribute("aPosB", new THREE.InstancedBufferAttribute(new Float32Array(posB), 2));
    geometry.setAttribute("aTanA", new THREE.InstancedBufferAttribute(new Float32Array(tanA), 2));
    geometry.setAttribute("aTanB", new THREE.InstancedBufferAttribute(new Float32Array(tanB), 2));
    geometry.setAttribute("aCorridor", new THREE.InstancedBufferAttribute(new Float32Array(corridor), 1));
    this.sleeperCount = corridor.length;
    this.sleeperVisible = { value: 1 };
    const material = new THREE.ShaderMaterial({
      vertexShader: SLEEPER_VERTEX,
      fragmentShader: SLEEPER_FRAGMENT,
      uniforms: {
        ...this.fog,
        ...this.basisUniforms,
        uVisible: this.sleeperVisible,
        uColor: { value: color("#4a4640") },
      },
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    this.scene.add(mesh);
  }

  _solidMaterial() {
    return new THREE.ShaderMaterial({
      vertexShader: SOLID_VERTEX,
      fragmentShader: SOLID_FRAGMENT,
      uniforms: { ...this.fog },
    });
  }

  /**
   * Everything built beside the track, in one instanced mesh: platforms
   * (both sides at terminals and junctions), canopies on posts where a real
   * station would have them, and a signal mast and head at the entrance to
   * every section, on the left of the line as Indian Railways places them.
   */
  _buildStations() {
    // Platform axis per basis: the direction of the station's through-line.
    this.stationAxes = [0, 1].map((basis) => {
      const out = {};
      for (const s of STATIONS) {
        let best = null;
        for (const edge of EDGES) {
          if (edge.from !== s.id && edge.to !== s.id) continue;
          const sec = this.alignments[basis][edge.id];
          const p = sampleAt(sec, edge.from === s.id ? 0 : 1);
          if (!best) best = p.tangent;
        }
        out[s.id] = Math.atan2(best?.z ?? 0, best?.x ?? 1);
      }
      return out;
    });

    this.structureParts = [];
    for (const station of STATIONS) {
      const sides = station.type === "minor" || station.type === "major" ? [1] : [1, -1];
      for (const side of sides) {
        this.structureParts.push({ kind: "platform", station, side });
        if (station.type !== "minor") {
          this.structureParts.push({ kind: "canopy", station, side });
          this.structureParts.push({ kind: "post", station, side, at: -0.22 });
          this.structureParts.push({ kind: "post", station, side, at: 0.22 });
        }
      }
    }

    // One signal per section end, protecting entry into the section.
    this.signals = [];
    for (const edge of EDGES) {
      const length = this.alignments[0][edge.id].length;
      const f = Math.min(0.25, 0.7 / length);
      this.signals.push({ edge, entry: edge.from, fraction: f, reversed: false });
      this.signals.push({ edge, entry: edge.to, fraction: 1 - f, reversed: true });
    }
    for (const signal of this.signals) {
      this.structureParts.push({ kind: "mast", signal });
      this.structureParts.push({ kind: "head", signal });
    }

    const geometry = new THREE.BoxGeometry(1, 1, 1);
    geometry.translate(0, 0.5, 0);
    this.stations = new THREE.InstancedMesh(geometry, this._solidMaterial(), this.structureParts.length);
    this.stations.frustumCulled = false;
    const tones = {
      platform: color(PALETTE.platform),
      canopy: color(PALETTE.canopy),
      post: color(PALETTE.mast),
      mast: color(PALETTE.mast),
      head: color("#1d2327"),
    };
    this.structureParts.forEach((part, i) => this.stations.setColorAt(i, tones[part.kind]));
    this.scene.add(this.stations);
  }

  /** Where a signal stands: position and the direction of the train it faces. */
  _signalPose(signal) {
    const pose = this.pointOnEdge(signal.edge, signal.fraction);
    // Direction of a train entering the section past this signal.
    const tx = signal.reversed ? -pose.tx : pose.tx;
    const tz = signal.reversed ? -pose.tz : pose.tz;
    // Left of the direction of travel: (tz, -tx) with y up.
    const offset = TRACK_HALF_WIDTH + 0.14;
    return { x: pose.x + tz * offset, z: pose.z - tx * offset, tx, tz };
  }

  _buildHalos() {
    const capacity = this.signals.length + this.sim.trains.length;
    const quad = new THREE.PlaneGeometry(2, 2);
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.index = quad.index;
    geometry.setAttribute("position", quad.attributes.position);
    this.haloCenter = new Float32Array(capacity * 3);
    this.haloColor = new Float32Array(capacity * 3);
    this.haloSize = new Float32Array(capacity);
    this.haloMinPx = new Float32Array(capacity);
    const dynamic = (array, size) => new THREE.InstancedBufferAttribute(array, size).setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute("aCenter", dynamic(this.haloCenter, 3));
    geometry.setAttribute("aColor", dynamic(this.haloColor, 3));
    geometry.setAttribute("aSize", dynamic(this.haloSize, 1));
    geometry.setAttribute("aMinPx", dynamic(this.haloMinPx, 1));
    geometry.instanceCount = 0;
    this.haloGeometry = geometry;
    const material = new THREE.ShaderMaterial({
      vertexShader: HALO_VERTEX,
      fragmentShader: HALO_FRAGMENT,
      uniforms: { ...this.fog, uPixelPerDepth: this.pixelUniform },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    mesh.renderOrder = 3;
    this.scene.add(mesh);
    this.aspectColors = {
      green: color(PALETTE.signalGreen),
      yellow: color(PALETTE.signalYellow),
      red: color(PALETTE.signalRed),
    };
  }

  _updateHalos(lamps) {
    let n = 0;
    const put = (x, y, z, c, size, minPx) => {
      this.haloCenter.set([x, y, z], n * 3);
      this.haloColor.set([c.r, c.g, c.b], n * 3);
      this.haloSize[n] = size;
      this.haloMinPx[n] = minPx;
      n += 1;
    };
    for (const signal of this.signals) {
      const edge = this.network.edges.get(signal.edge.id);
      const aspect = signalAspect(edge, this.occupancy?.get(edge.id) || 0);
      const pose = this._signalPose(signal);
      // Lamp on the face of the head, towards the approaching train.
      put(pose.x - pose.tx * 0.05, 0.46, pose.z - pose.tz * 0.05, this.aspectColors[aspect], 0.16, 2.6);
    }
    const headlight = color(PALETTE.headlight);
    for (const lamp of lamps) put(lamp.x, lamp.y, lamp.z, headlight, 0.22, 3.2);
    this.haloGeometry.instanceCount = n;
    for (const name of ["aCenter", "aColor", "aSize", "aMinPx"]) this.haloGeometry.attributes[name].needsUpdate = true;
  }

  _buildBars() {
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    geometry.translate(0, 0.5, 0);
    this.bars = new THREE.InstancedMesh(geometry, this._solidMaterial(), EDGES.length);
    this.bars.frustumCulled = false;
    const zero = new THREE.Matrix4().makeScale(0, 0, 0);
    for (let i = 0; i < EDGES.length; i += 1) {
      this.bars.setMatrixAt(i, zero);
      this.bars.setColorAt(i, color(PALETTE.steel));
    }
    this.scene.add(this.bars);
  }

  _buildRings() {
    const geometry = new THREE.RingGeometry(0.78, 1, 48);
    geometry.rotateX(-Math.PI / 2);
    this.ringStart = new Float32Array(STATIONS.length).fill(-1);
    this.ringOnPath = new Float32Array(STATIONS.length);
    geometry.setAttribute("aStart", new THREE.InstancedBufferAttribute(this.ringStart, 1));
    geometry.setAttribute("aOnPath", new THREE.InstancedBufferAttribute(this.ringOnPath, 1));
    this.ringFade = { value: 0 };
    const material = new THREE.ShaderMaterial({
      vertexShader: RING_VERTEX,
      fragmentShader: RING_FRAGMENT,
      uniforms: {
        ...this.fog,
        uTime: this.timeUniform,
        uFade: this.ringFade,
        uClear: { value: color(PALETTE.clear) },
        uRoute: { value: color(PALETTE.route) },
      },
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    this.rings = new THREE.InstancedMesh(geometry, material, STATIONS.length);
    this.rings.frustumCulled = false;
    this.rings.renderOrder = 2;
    this.scene.add(this.rings);
  }

  _buildTrains() {
    const coachCount = this.sim.trains.reduce((n, t) => n + t.profile.coaches, 0);
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    geometry.translate(0, 0.5, 0);
    // One instance per vehicle, plus one headlamp per train.
    this.vehicleCapacity = coachCount + this.sim.trains.length;
    this.vehicleKind = new Float32Array(this.vehicleCapacity);
    geometry.setAttribute("aKind", new THREE.InstancedBufferAttribute(this.vehicleKind, 1).setUsage(THREE.DynamicDrawUsage));
    const material = new THREE.ShaderMaterial({
      vertexShader: VEHICLE_VERTEX,
      fragmentShader: VEHICLE_FRAGMENT,
      uniforms: {
        ...this.fog,
        uGlass: { value: color("#1a2328") },
        uInterior: { value: color("#e8c98f") },
      },
    });
    this.vehicles = new THREE.InstancedMesh(geometry, material, this.vehicleCapacity);
    this.vehicles.frustumCulled = false;
    this.vehicles.setColorAt(0, color(PALETTE.coach));
    this.scene.add(this.vehicles);
    this.trainScreen = new Map();
  }

  _buildTrails() {
    const perTrain = Math.ceil(TRAIL_MINUTES / TRAIL_SAMPLE) + 3;
    const vertices = this.sim.trains.length * perTrain * 2;
    const geometry = new THREE.BufferGeometry();
    this.trailPositions = new Float32Array(vertices * 3);
    this.trailAlpha = new Float32Array(vertices);
    this.trailColor = new Float32Array(vertices * 3);
    geometry.setAttribute("position", new THREE.BufferAttribute(this.trailPositions, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute("aAlpha", new THREE.BufferAttribute(this.trailAlpha, 1).setUsage(THREE.DynamicDrawUsage));
    this.trailSide = new Float32Array(vertices);
    geometry.setAttribute("aSide", new THREE.BufferAttribute(this.trailSide, 1).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute("aColor", new THREE.BufferAttribute(this.trailColor, 3).setUsage(THREE.DynamicDrawUsage));
    this.trailIndex = new Uint32Array(this.sim.trains.length * perTrain * 6);
    geometry.setIndex(new THREE.BufferAttribute(this.trailIndex, 1).setUsage(THREE.DynamicDrawUsage));
    this.trailGeometry = geometry;
    const material = new THREE.ShaderMaterial({
      vertexShader: TRAIL_VERTEX,
      fragmentShader: TRAIL_FRAGMENT,
      uniforms: { ...this.fog },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    mesh.renderOrder = 1;
    this.scene.add(mesh);
    this.trailHistory = new Map();
    this.trailClock = -Infinity;
  }

  /** Where a train's head is, as (edge, section fraction): stable across the morph. */
  _headLocation(train) {
    let leg = train.leg;
    let fraction;
    if (leg >= train.route.length - 1) {
      leg = train.route.length - 2;
      fraction = 1;
    } else {
      const edge = this.network.edgeBetween(train.route[leg], train.route[leg + 1]);
      if (!edge) return null;
      fraction = train.progressKm / edge.distance_km;
    }
    if (leg < 0) return null;
    const edge = this.network.edgeBetween(train.route[leg], train.route[leg + 1]);
    if (!edge) return null;
    return { edge, f: edge.from === train.route[leg] ? fraction : 1 - fraction };
  }

  _updateTrails() {
    const now = this.sim.time;
    if (now < this.trailClock) this.trailHistory.clear(); // the simulation was reset
    if (now - this.trailClock >= TRAIL_SAMPLE || now < this.trailClock) {
      this.trailClock = now;
      for (const train of this.sim.trains) {
        if (train.status === "scheduled") continue;
        const at = this._headLocation(train);
        if (!at) continue;
        let history = this.trailHistory.get(train.id);
        if (!history) this.trailHistory.set(train.id, (history = []));
        history.push({ ...at, t: now });
        while (history.length && now - history[0].t > TRAIL_MINUTES) history.shift();
      }
    }

    const tint = {
      superfast: color("#fff6e4"),
      express: color("#f0ead8"),
      passenger: color("#d8dfe4"),
      freight: color("#c9a77e"),
    };
    let v = 0;
    let i = 0;
    for (const train of this.sim.trains) {
      const history = this.trailHistory.get(train.id);
      if (!history || !history.length) continue;
      const head = this._headLocation(train);
      const points = head ? [...history, { ...head, t: now }] : history;
      const c = tint[train.type];
      let prev = null;
      for (let k = 0; k < points.length; k += 1) {
        const point = points[k];
        const p = this.pointOnEdge(point.edge, point.f);
        // Skip jumps (turnrounds, reroutes back through a node): a trail is
        // continuous motion or nothing.
        if (prev && Math.hypot(p.x - prev.x, p.z - prev.z) > 12) {
          prev = null;
        }
        let dx = prev ? p.x - prev.x : p.tx;
        let dz = prev ? p.z - prev.z : p.tz;
        const len = Math.hypot(dx, dz) || 1;
        dx /= len;
        dz /= len;
        const width = Math.max(0.1, this._pixelAt(p.x, p.z) * 3.6);
        const age = (now - point.t) / TRAIL_MINUTES;
        const alpha = Math.pow(Math.max(0, 1 - age), 1.1);
        for (const side of [-1, 1]) {
          this.trailPositions.set([p.x - dz * width * side, TRACK_Y + 0.006, p.z + dx * width * side], v * 3);
          this.trailSide[v] = side;
          this.trailAlpha[v] = alpha;
          this.trailColor.set([c.r, c.g, c.b], v * 3);
          v += 1;
        }
        if (prev) {
          const b = v - 4;
          this.trailIndex.set([b, b + 1, b + 2, b + 1, b + 3, b + 2], i);
          i += 6;
        }
        prev = p;
      }
    }
    this.trailGeometry.setDrawRange(0, i);
    this.trailGeometry.attributes.position.needsUpdate = true;
    this.trailGeometry.attributes.aAlpha.needsUpdate = true;
    this.trailGeometry.attributes.aSide.needsUpdate = true;
    this.trailGeometry.attributes.aColor.needsUpdate = true;
    this.trailGeometry.index.needsUpdate = true;
  }

  /** Open on one train at rail level and pull back to the shot. */
  introFrom(id) {
    const pose = this.trainPose(id);
    if (!pose || this.reducedMotion) return;
    this._snapNext = false;
    Object.assign(this.view, {
      x: pose.x,
      z: pose.z,
      distance: 4,
      heading: Math.atan2(-pose.tx, -pose.tz) + 0.6,
      pitch: 0.32,
    });
  }

  /* ----------------------------------------------------------- control */

  setBasis(target, origin = null) {
    const now = this._currentBlendEstimate();
    if (target === this.basis.to && this.basis.progress >= 1) return;
    this.basis = {
      from: now,
      to: target,
      origin: origin ?? this._focusCorridor(),
      progress: this.reducedMotion ? 1 : 0,
      started: this.clock,
    };
  }

  get basisTarget() {
    return this.basis.to;
  }

  _currentBlendEstimate() {
    // Snapshot the transition as a whole: finish where it is headed if it is
    // mostly done, otherwise restart from where it started.
    return this.basis.progress >= 0.5 ? this.basis.to : this.basis.from;
  }

  _focusCorridor() {
    // The corridor coordinate nearest the camera focus, so a change starts
    // where the reader is looking.
    let best = 0;
    let bestD = Infinity;
    for (const s of STATIONS) {
      const p = this.stationPosition(s.id);
      const d = Math.hypot(p.x - this.view.x, p.z - this.view.z);
      if (d < bestD) {
        bestD = d;
        best = this.corridor[s.id];
      }
    }
    return best;
  }

  setShot(shot) {
    Object.assign(this.goal, shot);
    this.follow = shot.follow ?? null;
    this.orbit.heading = 0;
    this.orbit.pitch = 0;
    this.freeLook = false;
    if (this.reducedMotion) this._snap();
  }

  followTrain(id) {
    this.follow = id;
    this.freeLook = false;
    if (this.reducedMotion) this._snap();
  }

  setInteractive(mode) {
    this.mode = mode; // "story" | "free"
    this.canvas.style.cursor = mode === "free" ? "grab" : "";
  }

  showCosts(on) {
    this.barsTarget = on ? 1 : 0;
  }

  setHighlightTrain(id) {
    this.highlightTrain = id;
  }

  zoom(factor) {
    this.goal.extent = Math.max(6, Math.min(900, this.goal.extent * factor));
    if (this.follow) this.goal.followDistance = Math.max(1.5, Math.min(120, (this.goal.followDistance || 5) * factor));
  }

  recentre() {
    this.freeLook = false;
    this.orbit.heading = 0;
    this.orbit.pitch = 0;
    if (this.home) Object.assign(this.goal, this.home);
  }

  /**
   * Draw a Dijkstra run: rings at stations in the order they were settled,
   * explored sections sweeping in relaxation order, then the chosen path.
   */
  playTrace(result, { speed = 1 } = {}) {
    const edgeCount = EDGES.length;
    for (let i = 0; i < edgeCount; i += 1) {
      this.stateData[(edgeCount + i) * 4 + 0] = -1;
      this.stateData[(edgeCount + i) * 4 + 2] = -1;
    }
    this.ringStart.fill(-1);
    this.ringOnPath.fill(0);
    const t0 = this.clock + 0.15;
    const stepSeconds = (this.reducedMotion ? 0 : 0.085) / speed;
    const settledAt = new Map();
    result.trace.settled.forEach((entry, i) => {
      const at = t0 + i * stepSeconds;
      settledAt.set(entry.node, at);
      const index = STATIONS.findIndex((s) => s.id === entry.node);
      if (index >= 0) this.ringStart[index] = at;
    });
    for (const relax of result.trace.relaxed) {
      const e = this.edgeIndex.get(relax.edge);
      const edge = EDGES[e];
      const at = settledAt.get(relax.from) ?? t0;
      const slot = (edgeCount + e) * 4;
      if (this.stateData[slot + 2] < 0) {
        this.stateData[slot + 2] = at;
        this.stateData[slot + 3] = edge.from === relax.from ? 0 : 1;
      }
    }
    const pathStart = t0 + result.trace.settled.length * stepSeconds + 0.2;
    const path = result.path || [];
    for (let i = 0; i < path.length - 1; i += 1) {
      const edge = this.network.edgeBetween(path[i], path[i + 1]);
      if (!edge) continue;
      const e = this.edgeIndex.get(edge.id);
      const slot = (edgeCount + e) * 4;
      this.stateData[slot] = pathStart + i * (this.reducedMotion ? 0 : 0.3 / speed);
      this.stateData[slot + 1] = edge.from === path[i] ? 0 : 1;
      const si = STATIONS.findIndex((s) => s.id === path[i]);
      if (si >= 0) this.ringOnPath[si] = 1;
    }
    this.rings.geometry.attributes.aStart.needsUpdate = true;
    this.rings.geometry.attributes.aOnPath.needsUpdate = true;
    this.stateTexture.needsUpdate = true;
    this.traceFade = 1;
    this.traceUntil = pathStart + path.length * 0.3 + 7;
    this.traceLabels = { settled: result.trace.settled, t0, stepSeconds };
    return { duration: pathStart - this.clock + path.length * 0.3 };
  }

  clearTrace() {
    this.traceUntil = this.clock;
  }

  on(kind, listener) {
    this.listeners[kind].add(listener);
    return () => this.listeners[kind].delete(listener);
  }

  /* ----------------------------------------------------------- pointer */

  _bindPointer() {
    const canvas = this.canvas;
    let drag = null;

    canvas.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 && event.button !== 1) return;
      drag = {
        id: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        moved: 0,
        rotate: this.mode !== "free" || event.shiftKey || event.button === 1 || event.pointerType === "touch",
      };
      canvas.setPointerCapture(event.pointerId);
      if (this.mode === "free") canvas.style.cursor = "grabbing";
    });

    canvas.addEventListener("pointermove", (event) => {
      if (!drag || drag.id !== event.pointerId) {
        if (event.pointerType !== "touch") this._hover(event);
        return;
      }
      const dx = event.clientX - drag.x;
      const dy = event.clientY - drag.y;
      drag.x = event.clientX;
      drag.y = event.clientY;
      drag.moved += Math.abs(dx) + Math.abs(dy);
      if (drag.moved < 4) return;
      if (drag.rotate) {
        this.orbit.heading -= dx * 0.005;
        this.orbit.pitch = Math.max(-0.6, Math.min(0.5, this.orbit.pitch + dy * 0.004));
      } else {
        // Grab-and-drag: the ground stays under the cursor.
        const pixel = this.view.distance * this.pixelUniform.value;
        const d = panDelta(dx, dy, this.view.heading + this.orbit.heading, this.view.pitch + this.orbit.pitch, pixel);
        this.goal.x += d.x;
        this.goal.z += d.z;
        this.view.x += d.x;
        this.view.z += d.z;
        this.follow = null;
        this.freeLook = true;
      }
    });

    const end = (event) => {
      if (!drag || drag.id !== event.pointerId) return;
      const click = drag.moved < 5;
      drag = null;
      if (this.mode === "free") canvas.style.cursor = "grab";
      if (click) {
        const hit = this._pick(event);
        for (const listener of this.listeners.pick) listener(hit);
      }
    };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", () => {
      drag = null;
    });
    canvas.addEventListener("pointerleave", () => {
      if (this.hovered) {
        this.hovered = null;
        for (const listener of this.listeners.hover) listener(null);
      }
    });
  }

  _screen(x, z, y = 0) {
    const v = new THREE.Vector3(x, y, z).project(this.camera);
    if (v.z > 1) return null;
    return { x: (v.x * 0.5 + 0.5) * this.size.width, y: (-v.y * 0.5 + 0.5) * this.size.height };
  }

  _pick(event) {
    const rect = this.canvas.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;

    let best = null;
    let bestD = 16;
    for (const s of STATIONS) {
      const p = this.stationPosition(s.id);
      const q = this._screen(p.x, p.z, 0.3);
      if (!q) continue;
      const d = Math.hypot(q.x - px, q.y - py);
      if (d < bestD) {
        bestD = d;
        best = { kind: "station", station: s, x: q.x, y: q.y };
      }
    }
    if (best && bestD < 11) return best;

    for (const [id, pos] of this.trainScreen) {
      const d = Math.hypot(pos.x - px, pos.y - py);
      if (d < 14 && d < bestD) {
        bestD = d;
        best = { kind: "train", train: this.sim.trains.find((t) => t.id === id), x: pos.x, y: pos.y };
      }
    }
    if (best && best.kind === "train") return best;

    let edgeBest = null;
    let edgeD = 12;
    for (const { edge, samples } of this.pickSamples) {
      let prev = null;
      for (const f of samples) {
        const p = this.pointOnEdge(edge, f);
        const q = this._screen(p.x, p.z);
        if (q && prev) {
          const d = segmentDistance(px, py, prev, q);
          if (d < edgeD) {
            edgeD = d;
            edgeBest = { kind: "edge", edge, x: px, y: py };
          }
        }
        prev = q;
      }
    }
    return edgeBest || best;
  }

  _hover(event) {
    const hit = this._pick(event);
    const key = hit ? `${hit.kind}:${hit.station?.id || hit.edge?.id || hit.train?.id}` : null;
    const prev = this.hovered ? this.hovered.key : null;
    this.hovered = hit ? { ...hit, key } : null;
    if (this.mode === "free") this.canvas.style.cursor = hit && hit.kind === "edge" ? "pointer" : "grab";
    for (const listener of this.listeners.hover) listener(this.hovered, key !== prev);
  }

  /* ------------------------------------------------------------- frame */

  resize(width, height) {
    this.size = { width: Math.max(1, width), height: Math.max(1, height) };
    this.renderer.setSize(this.size.width, this.size.height, false);
    this.camera.aspect = this.size.width / this.size.height;
    this.camera.updateProjectionMatrix();
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.last = performance.now();
    this.raf = requestAnimationFrame(this._frame);
  }

  stop() {
    this.running = false;
    cancelAnimationFrame(this.raf);
  }

  /** Advance everything by dt seconds and draw. Exposed so tests can drive it. */
  tick(dt) {
    this.clock += dt;
    this.timeUniform.value = this.clock;

    if (this.basis.progress < 1) {
      this.basis.progress = Math.min(1, (this.clock - this.basis.started) / WAVE_SECONDS);
    }
    this.basisUniforms.uFrom.value = this.basis.from;
    this.basisUniforms.uTo.value = this.basis.to;
    this.basisUniforms.uWaveOrigin.value = this.basis.origin;
    this.basisUniforms.uWaveProgress.value = this.basis.progress;

    this._updateCamera(dt);
    this._updateEdgeState();
    this._updateStations();
    this._updateBars(dt);
    this._updateTrains();
    this._updateTrails();
    this._updateTrace(dt);
    this.renderer.render(this.scene, this.camera);
  }

  _frame(now) {
    if (!this.running) return;
    const dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    if (this.onBeforeFrame) this.onBeforeFrame(dt);
    this.tick(dt);
    if (this.onAfterFrame) this.onAfterFrame(dt);
    this.raf = requestAnimationFrame(this._frame);
  }

  _snap() {
    this._snapNext = true;
  }

  _fitDistance(extent) {
    const vfov = (this.camera.fov * Math.PI) / 180;
    const aspect = this.camera.aspect;
    const byWidth = extent / 2 / (Math.tan(vfov / 2) * aspect);
    const byHeight = (extent * 0.34) / 2 / Math.tan(vfov / 2);
    return Math.max(byWidth, byHeight);
  }

  _updateCamera(dt) {
    const goal = { ...this.goal };
    // The lens shift pushes the frame sideways, so fit the extent into what
    // remains visible on the far side of it.
    let distance = this._fitDistance(goal.extent / Math.max(0.5, 1 - Math.abs(goal.lensX)));

    if (this.follow) {
      const pose = this.trainPose(this.follow);
      if (pose) {
        goal.x = pose.x + pose.tx * 1.0;
        goal.z = pose.z + pose.tz * 1.0;
        goal.heading = Math.atan2(-pose.tx, -pose.tz) + (this.camera.aspect < 1 ? 0.25 : 0.5);
        distance = goal.followDistance || 5;
      }
    }

    const snap = this._snapNext || this.reducedMotion;
    this._snapNext = false;
    const k = snap ? 1 : dampFactor(this.follow ? 3.2 : 1.8, dt);
    const v = this.view;
    v.x += (goal.x - v.x) * k;
    v.z += (goal.z - v.z) * k;
    v.heading += shortestAngle(v.heading, goal.heading) * k;
    v.pitch += (goal.pitch - v.pitch) * k;
    v.distance *= Math.exp(Math.log(distance / v.distance) * k);
    v.lensX += (goal.lensX - v.lensX) * k;
    v.lensY += (goal.lensY - v.lensY) * k;

    const heading = v.heading + this.orbit.heading;
    const pitch = Math.max(0.12, Math.min(1.45, v.pitch + this.orbit.pitch));
    const d = v.distance;
    this.camera.position.set(
      v.x + Math.sin(heading) * Math.cos(pitch) * d,
      Math.sin(pitch) * d,
      v.z + Math.cos(heading) * Math.cos(pitch) * d
    );
    this.camera.lookAt(v.x, 0, v.z);
    this.camera.near = Math.max(0.01, d * 0.004);
    this.camera.far = d * 8 + 400;
    // Lens shift keeps the network clear of the text column without moving
    // the camera: the focus is simply drawn off-centre.
    const { width, height } = this.size;
    this.camera.setViewOffset(width, height, -v.lensX * width, -v.lensY * height, width, height);
    this.camera.updateProjectionMatrix();

    this.fog.uFogNear.value = d * 1.35;
    this.fog.uFogFar.value = d * 3.6 + 60;
    this.pixelUniform.value = (2 * Math.tan((this.camera.fov * Math.PI) / 360)) / height;
    this.sleeperVisible.value = d < 40 ? 1 : 0;
  }

  _updateEdgeState() {
    const occupancy = this.sim.occupancy();
    this.occupancy = occupancy;
    const highlight = new Set();
    const train = this.highlightTrain && this.sim.trains.find((t) => t.id === this.highlightTrain);
    if (train && train.status !== "arrived") {
      for (let i = train.leg; i < train.route.length - 1; i += 1) {
        const edge = this.network.edgeBetween(train.route[i], train.route[i + 1]);
        if (edge) highlight.add(edge.id);
      }
    }
    EDGES.forEach((spec, i) => {
      const edge = this.network.edges.get(spec.id);
      const load = occupancy.get(edge.id) || 0;
      const slot = i * 4;
      this.stateData[slot] = STATE_CODE[congestionState(edge, load)];
      this.stateData[slot + 1] = load / Math.max(1, edge.capacity);
      this.stateData[slot + 2] = highlight.has(edge.id) ? 1 : 0;
    });
    this.stateTexture.needsUpdate = true;
  }

  _pixelAt(x, z) {
    const dx = this.camera.position.x - x;
    const dy = this.camera.position.y;
    const dz = this.camera.position.z - z;
    return Math.hypot(dx, dy, dz) * this.pixelUniform.value;
  }

  _updateStations() {
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3();
    const p = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);
    const identity = new THREE.Quaternion();
    const hidden = new THREE.Matrix4().makeScale(0, 0, 0);

    const frames = new Map();
    STATIONS.forEach((station, i) => {
      const pos = this.stationPosition(station.id);
      const t = this.blendAt(this.corridor[station.id]);
      const a0 = this.stationAxes[0][station.id];
      const a1 = this.stationAxes[1][station.id];
      const angle = a0 + shortestAngle(a0, a1) * t;
      const pixel = this._pixelAt(pos.x, pos.z);
      frames.set(station.id, { pos, angle, pixel, dx: Math.cos(angle), dz: Math.sin(angle) });

      // Platform lighting: a sodium pool sized to the station.
      const pool = station.type === "terminal" ? 4.5 : station.type === "minor" ? 1.8 : 3.2;
      this.lights.uPools.value[i].set(pos.x, pos.z, pool);

      s.setScalar(Math.max(2.2, pixel * 11));
      p.set(pos.x, 0.04, pos.z);
      m.compose(p, identity, s);
      this.rings.setMatrixAt(i, m);
    });

    this.structureParts.forEach((part, i) => {
      if (part.kind === "mast" || part.kind === "head") {
        const pose = this._signalPose(part.signal);
        // Masts are drawn only once they resolve; further out the lamp alone
        // (a halo) stands for the signal.
        if (this._pixelAt(pose.x, pose.z) > 0.05) {
          this.stations.setMatrixAt(i, hidden);
          return;
        }
        q.setFromAxisAngle(up, -Math.atan2(pose.tz, pose.tx));
        if (part.kind === "mast") {
          s.set(0.03, 0.42, 0.03);
          p.set(pose.x, 0, pose.z);
        } else {
          s.set(0.05, 0.13, 0.08);
          p.set(pose.x, 0.4, pose.z);
        }
        m.compose(p, q, s);
        this.stations.setMatrixAt(i, m);
        return;
      }

      const f = frames.get(part.station.id);
      const length = PLATFORM_LENGTH[part.station.type];
      const nx = -f.dz;
      const nz = f.dx;
      const width = Math.max(PLATFORM_WIDTH, f.pixel * 2.5);
      const across = (TRACK_HALF_WIDTH + width / 2 + 0.03) * part.side;
      q.setFromAxisAngle(up, -f.angle);
      if (part.kind === "platform") {
        s.set(Math.max(length, f.pixel * 9), Math.max(0.05, f.pixel * 1.2), width);
        p.set(f.pos.x + nx * across, 0, f.pos.z + nz * across);
      } else if (f.pixel > 0.05) {
        this.stations.setMatrixAt(i, hidden);
        return;
      } else if (part.kind === "canopy") {
        s.set(length * 0.6, 0.025, PLATFORM_WIDTH * 1.2);
        p.set(f.pos.x + nx * across, 0.34, f.pos.z + nz * across);
      } else {
        const along = length * part.at;
        s.set(0.025, 0.34, 0.025);
        p.set(f.pos.x + nx * across + f.dx * along, 0, f.pos.z + nz * across + f.dz * along);
      }
      m.compose(p, q, s);
      this.stations.setMatrixAt(i, m);
    });
    this.stations.instanceMatrix.needsUpdate = true;
    this.rings.instanceMatrix.needsUpdate = true;
  }

  _updateBars(dt) {
    const k = this.reducedMotion ? 1 : dampFactor(2.4, dt);
    this.barsIn += (this.barsTarget - this.barsIn) * k;
    this.bars.visible = this.barsIn > 0.002;
    if (!this.bars.visible) return;
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3();
    const p = new THREE.Vector3();
    const tone = {
      normal: color("#9aa7af"),
      busy: color(PALETTE.caution),
      congested: color(PALETTE.danger),
      closed: color(PALETTE.danger),
    };
    EDGES.forEach((spec, i) => {
      const edge = this.network.edges.get(spec.id);
      const load = this.occupancy?.get(edge.id) || 0;
      const state = congestionState(edge, load);
      const minutes = dynamicWeight(edge, load);
      const mid = this.pointOnEdge(edge, 0.5);
      const height = Number.isFinite(minutes) ? minutes * 0.22 : 0.4;
      const width = Math.max(0.9, this._pixelAt(mid.x, mid.z) * 7);
      // Stagger the rise along the corridor so the bars read as a sweep of
      // the cost function rather than a single pop.
      const local = Math.max(0, Math.min(1, this.barsIn * 1.6 - this.corridor[edge.from] * 0.6));
      s.set(width, Math.max(0.001, height * local), width);
      p.set(mid.x, 0, mid.z);
      m.compose(p, q, s);
      this.bars.setMatrixAt(i, m);
      this.bars.setColorAt(i, tone[state]);
    });
    this.bars.instanceMatrix.needsUpdate = true;
    this.bars.instanceColor.needsUpdate = true;
  }

  _updateTrace(dt) {
    const target = this.clock < this.traceUntil ? 1 : 0;
    this.traceFade += (target - this.traceFade) * (this.reducedMotion ? 1 : dampFactor(1.6, dt));
    this.ringFade.value = this.traceFade;
    if (this.traceFade < 0.01 && target === 0 && this._traceActive) {
      const edgeCount = EDGES.length;
      for (let i = 0; i < edgeCount; i += 1) {
        this.stateData[(edgeCount + i) * 4 + 0] = -1;
        this.stateData[(edgeCount + i) * 4 + 2] = -1;
      }
      this.stateTexture.needsUpdate = true;
      this._traceActive = false;
    }
    if (target) this._traceActive = true;
  }

  /** Label data for the page: minutes Dijkstra assigned to each settled station so far. */
  traceLabelsNow() {
    if (!this.traceLabels || this.traceFade < 0.05) return [];
    const { settled, t0, stepSeconds } = this.traceLabels;
    const out = [];
    settled.forEach((entry, i) => {
      if (this.clock < t0 + i * stepSeconds) return;
      const p = this.stationPosition(entry.node);
      const q = this._screen(p.x, p.z, 0.4);
      if (q) out.push({ id: entry.node, minutes: entry.minutes, x: q.x, y: q.y, onPath: this.ringOnPath[STATIONS.findIndex((s) => s.id === entry.node)] > 0 });
    });
    return out;
  }

  /**
   * A train's route laid out as one arc-length line: each leg's length in
   * the current (possibly mid-morph) basis, and where the train is on it.
   *
   * The simulation's position is the train's *midpoint*: a train standing at
   * a station is centred on the platform, as a real one stops, and the
   * formation extends evenly ahead and behind. Past either end of the route
   * the line continues straight, so a train at a terminal overruns into the
   * stub rather than folding up.
   */
  _routeFrame(train) {
    const legs = [];
    let total = 0;
    for (let i = 0; i < train.route.length - 1; i += 1) {
      const edge = this.network.edgeBetween(train.route[i], train.route[i + 1]);
      if (!edge) return null;
      const length = this.pointOnEdge(edge, 0.5).length;
      legs.push({ edge, forward: edge.from === train.route[i], length, start: total });
      total += length;
    }
    if (!legs.length) return null;
    let head = total;
    if (train.leg < legs.length) {
      const leg = legs[train.leg];
      head = leg.start + (train.progressKm / leg.edge.distance_km) * leg.length;
    }
    return { legs, total, head };
  }

  /** Position, travel direction and curvature at arc length `at` on a route frame. */
  _poseAt(frame, at) {
    const { legs, total } = frame;
    const clamped = Math.max(0, Math.min(total, at));
    let leg = legs[legs.length - 1];
    for (const candidate of legs) {
      if (clamped < candidate.start + candidate.length) {
        leg = candidate;
        break;
      }
    }
    const f = Math.max(0, Math.min(1, (clamped - leg.start) / leg.length));
    const pose = this.pointOnEdge(leg.edge, leg.forward ? f : 1 - f);
    if (!leg.forward) {
      pose.tx = -pose.tx;
      pose.tz = -pose.tz;
      pose.curvature = -pose.curvature;
    }
    const beyond = at - clamped;
    if (beyond !== 0) {
      pose.x += pose.tx * beyond;
      pose.z += pose.tz * beyond;
      pose.curvature = 0;
    }
    return pose;
  }

  _trainLength(train) {
    return train.profile.coaches * (COACH_LENGTH + COACH_GAP) - COACH_GAP;
  }

  trainPose(id) {
    const train = this.sim.trains.find((t) => t.id === id);
    if (!train || train.route.length < 2) return null;
    const frame = this._routeFrame(train);
    return frame ? this._poseAt(frame, frame.head) : null;
  }

  _updateTrains() {
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const roll = new THREE.Quaternion();
    const s = new THREE.Vector3();
    const p = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);
    const forwardAxis = new THREE.Vector3(1, 0, 0);
    const coach = color(PALETTE.coach);
    const wagon = color(PALETTE.wagon);
    const lamp = color(PALETTE.lamp);
    const loco = {
      running: color(PALETTE.route),
      dwell: color(PALETTE.route),
      held: color(PALETTE.caution),
      rerouted: color(PALETTE.clear),
      arrived: color("#8c989f"),
      scheduled: color("#8c989f"),
    };
    const pitch = COACH_LENGTH + COACH_GAP;
    const lamps = [];

    let n = 0;
    this.trainScreen.clear();
    for (const train of this.sim.trains) {
      if (train.status === "scheduled" || train.route.length < 2) continue;
      const frame = this._routeFrame(train);
      if (!frame) continue;
      const middle = this._poseAt(frame, frame.head);
      const front = frame.head + this._trainLength(train) / 2;
      const floor = this._pixelAt(middle.x, middle.z);
      const girth = Math.max(1, (floor * 4.2) / COACH_WIDTH);
      const recent = this.sim.time - train.reroutedAt < 20;
      const state = recent && train.status === "running" ? "rerouted" : train.status;
      const bodyColor = train.type === "freight" ? wagon : coach;

      // Cant: lean into the curve by v^2/r, capped where real track stops.
      const v = train.speed / 120;
      for (let c = 0; c < train.profile.coaches && n < this.vehicleCapacity - 1; c += 1) {
        const pose = this._poseAt(frame, front - c * pitch - COACH_LENGTH / 2);
        const angle = Math.atan2(pose.tz, pose.tx);
        const cant = Math.max(-MAX_CANT, Math.min(MAX_CANT, v * v * pose.curvature * 24));
        q.setFromAxisAngle(up, -angle);
        roll.setFromAxisAngle(forwardAxis, cant);
        q.multiply(roll);
        const isLoco = c === 0;
        const height = (isLoco ? COACH_HEIGHT * 1.12 : COACH_HEIGHT) * (train.type === "freight" && !isLoco ? 0.8 : 1);
        s.set(COACH_LENGTH, height * girth, COACH_WIDTH * girth);
        p.set(pose.x, TRACK_Y, pose.z);
        m.compose(p, q, s);
        this.vehicles.setMatrixAt(n, m);
        this.vehicles.setColorAt(n, isLoco ? loco[state] || coach : bodyColor);
        this.vehicleKind[n] = isLoco ? 1 : train.type === "freight" ? 2 : 0;
        n += 1;
      }

      // Headlamp: a small bright block on the leading face, which also
      // throws light ahead onto the rails and ground.
      const lampPose = this._poseAt(frame, front - 0.004);
      const lampY = TRACK_Y + COACH_HEIGHT * 0.62 * girth;
      if (n < this.vehicleCapacity) {
        q.setFromAxisAngle(up, -Math.atan2(lampPose.tz, lampPose.tx));
        s.set(0.02, 0.05 * girth, 0.09 * girth);
        p.set(lampPose.x, lampY, lampPose.z);
        m.compose(p, q, s);
        this.vehicles.setMatrixAt(n, m);
        this.vehicles.setColorAt(n, lamp);
        this.vehicleKind[n] = 3;
        n += 1;
      }
      // A parked train at its terminal has its lamps off.
      if (train.status !== "arrived") {
        lamps.push({ x: lampPose.x + lampPose.tx * 0.02, y: lampY, z: lampPose.z + lampPose.tz * 0.02, tx: lampPose.tx, tz: lampPose.tz });
      }

      const screen = this._screen(middle.x, middle.z, 0.2);
      if (screen) this.trainScreen.set(train.id, screen);
    }
    this.vehicles.count = n;
    this.vehicles.geometry.attributes.aKind.needsUpdate = true;
    this.vehicles.instanceMatrix.needsUpdate = true;
    if (this.vehicles.instanceColor) this.vehicles.instanceColor.needsUpdate = true;

    lamps.slice(0, MAX_LAMPS).forEach((l, i) => this.lights.uLamps.value[i].set(l.x, l.z, l.tx, l.tz));
    this.lights.uLampCount.value = Math.min(lamps.length, MAX_LAMPS);
    this._updateHalos(lamps);
  }

  /** Screen positions for the page's HTML labels. */
  stationLabels() {
    const out = [];
    for (const s of STATIONS) {
      const p = this.stationPosition(s.id);
      const q = this._screen(p.x, p.z, 0.4);
      if (q) out.push({ station: s, x: q.x, y: q.y });
    }
    return out;
  }

  get viewDistance() {
    return this.view.distance;
  }

  get drawCalls() {
    return this.renderer.info.render.calls;
  }

  dispose() {
    this.stop();
    this.scene.traverse((object) => {
      object.geometry?.dispose();
      object.material?.dispose();
    });
    this.stateTexture.dispose();
    this.renderer.dispose();
  }
}

function segmentDistance(px, py, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((px - a.x) * dx + (py - a.y) * dy) / len));
  return Math.hypot(px - (a.x + dx * t), py - (a.y + dy * t));
}

export function webglAvailable() {
  try {
    const canvas = document.createElement("canvas");
    // three.js r180 renders through WebGL 2 only.
    return Boolean(canvas.getContext("webgl2"));
  } catch {
    return false;
  }
}
