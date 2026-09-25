# RailFlow Website

The landing page for [RailFlow](https://github.com/Sudhss/RailFlow), a railway traffic-control simulator for the Delhi–Lucknow corridor.

This page doesn't use a screenshot or a canned animation as its hero. It runs the real network: all 36 stations and 42 sections from `backend/data/railway_graph.json`, drawn in 3D with Three.js, with trains driven by a JavaScript port of RailFlow's routing rules. Scrolling takes that network apart one idea at a time:

| Step | What the stage does |
|---|---|
| Hero | Opens at rail level on the fastest train, pulls back to the whole region |
| 01 Two maps | Morphs the network from the control-room diagram to real geography; the change travels outward from Delhi |
| 02 The price of a section | Raises a bar on every section: its expected minutes under live occupancy |
| 03 A closure | Closes Moradabad–Rampur and draws Dijkstra's search: a ring per station, in the order it was settled, with the minutes assigned |
| 04 Down to the rail | Chases a train: sleepers, rails, lit coach windows, cant on curves |
| 05 Your turn | Click any section to close or reopen it; the controller reroutes trains live and logs every decision |

Everything that moves is driven by something. Trains move because the simulation moves them. Trail length is the distance each train covered in the last 30 simulated minutes. The rings appear in the order Dijkstra actually settled the stations.

## How faithful is the in-page simulation?

`js/sim.js` uses the same logic as the backend: the same cost function (`graph.py dynamic_weight`), the same speed rule (`simulation.py _effective_speed`), the same Dijkstra, and the same controller gate (`agent.py`). The tests check its routes and minutes against output produced by the Python backend. It differs from the backend in three stated ways:

- time runs continuously rather than in whole-minute ticks;
- a closure applies to trains entering a section, so a train already inside runs through;
- trains turn round at their destination so the network never empties.

The benchmark figures on the page (59.7%, 100% vs 0%, 44 / 246, 20 min) are copied from `backend/benchmark_report.txt`. None of them was invented for the page.

## Structure

```
index.html        page and copy
css/site.css      design tokens and layout; same control-room language as the console
js/network.js     the network, generated from backend/data/railway_graph.json
js/geometry.js    projection, arc-length track, layout wave (shared with the console)
js/sim.js         routing rules + controller, pure, no DOM
js/stage.js       the Three.js stage: 8 draw calls, custom shaders, instancing
js/main.js        scroll steps, HUD, controls, overlays
tests/            node:test suite for sim.js and geometry.js
assets/           console screenshots
```

There's no build step and there are no dependencies. Three.js 0.180 loads from jsDelivr through an import map.

## Run and test

```bash
python -m http.server 5500      # then open http://localhost:5500
npm test                        # 13 tests, Node 18+
```

Because the page uses ES modules, it has to be served over HTTP. Opening `index.html` from disk won't work.

## Behaviour

- **Reduced motion:** the simulation starts paused, the camera cuts between steps instead of flying, and the morph applies instantly.
- **No WebGL 2:** the stage switches to a flat SVG diagram of the same network, and the text still covers everything.
- **Pause:** the HUD's Pause button stops all movement at any time.
- **Performance:** rendering stops when the stage is off screen or the tab is hidden.
- **Keyboard:** in free play, sections and trains can be chosen from the lists, `+`/`-` zooms, shift+arrows pans, and `C` recentres.

## Deploy

Serve the directory as static files: GitHub Pages, Netlify, Vercel or any static host.
