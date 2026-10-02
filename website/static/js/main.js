/* TutorMoments-Preview site
   Renders the leaderboard table and the findings charts from static/data/*.json.
   Charts are hand-rolled SVG; per-model colors and marker shapes match the paper's
   figures (Okabe-Ito palette; see analysis/working-paper-20260630 in allenai/tutormoments)
   so a model reads the same on the site and in the paper. */

(function () {
  "use strict";

  var MODEL_STYLE = {
    "claude-opus-4-8":             { provider: "anthropic", color: "#D55E00", marker: "triangle-down" },
    "claude-sonnet-4-6":           { provider: "anthropic", color: "#E69F00", marker: "square" },
    "deepseek-ai_DeepSeek-V4-Pro": { provider: "open_weight", color: "#CC79A7", marker: "triangle-up" },
    "gemini-2.5-pro":              { provider: "google", color: "#0072B2", marker: "diamond" },
    "gemini-3.5-flash":            { provider: "google", color: "#56B4E9", marker: "pentagon" },
    "gpt-5.5-2026-04-23":          { provider: "openai", color: "#009E73", marker: "plus" },
    "gpt-5.4-mini-2026-03-17":     { provider: "openai", color: "#F0529C", marker: "cross" },
    // Runs since the paper: the same hue per provider family, and a marker
    // no other model of that provider uses (the action chart shows one
    // provider at a time) or, where possible, any other on the cost chart.
    "claude-opus-5-5":             { provider: "anthropic", color: "#D55E00", marker: "circle" },
    "claude-fable-5-1":            { provider: "anthropic", color: "#D55E00", marker: "diamond" },
    "claude-sonnet-5-5":           { provider: "anthropic", color: "#E69F00", marker: "hexagon" },
    "deepseek-v4-pro-0813":        { provider: "open_weight", color: "#CC79A7", marker: "square" },
    "gemini-3.8-flash":            { provider: "google", color: "#56B4E9", marker: "triangle-up" },
    "gemini-3.6-flash":            { provider: "google", color: "#0072B2", marker: "triangle-down" },
    "gpt-5.5-2026-04-23-none":     { provider: "openai", color: "#009E73", marker: "circle" },
    "gpt-6-astra":                 { provider: "openai", color: "#F0529C", marker: "star" },
    "gpt-6-sol-none":              { provider: "openai", color: "#F0529C", marker: "pentagon" },
    "gpt-6-luna-none":             { provider: "openai", color: "#F0529C", marker: "square" }
  };

  // Shared y-axis for the two score scatters, so a model sits at the same
  // height on both.
  var SCORE_MIN = 0.5, SCORE_MAX = 1.0;

  var INK = "#0A3235";
  var INK_MUTED = "rgba(10, 50, 53, 0.48)";
  var GRID = "rgba(10, 50, 53, 0.08)";
  var SVG_NS = "http://www.w3.org/2000/svg";

  function el(name, attrs, parent) {
    var node = document.createElementNS(SVG_NS, name);
    for (var k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }

  // Data files keep their URLs across refreshes and are served without
  // cache headers, so a browser may reuse a stale copy heuristically.
  // "no-cache" revalidates every load: a 304 when unchanged, cheap.
  function fetchJSON(url) {
    return fetch(url, { cache: "no-cache" }).then(function (r) {
      if (!r.ok) throw new Error(url + " -> " + r.status);
      return r.json();
    });
  }

  /* ---------- markers (match matplotlib shapes used in the paper) ---------- */

  function polygonPoints(cx, cy, r, sides, rotationDeg) {
    var pts = [];
    for (var i = 0; i < sides; i++) {
      var a = (Math.PI * 2 * i) / sides + (rotationDeg * Math.PI) / 180;
      pts.push((cx + r * Math.sin(a)).toFixed(2) + "," + (cy - r * Math.cos(a)).toFixed(2));
    }
    return pts.join(" ");
  }

  function crossPoints(cx, cy, r, rotationDeg) {
    var w = r * 0.38; // half arm-width
    var base = [
      [-w, -r], [w, -r], [w, -w], [r, -w], [r, w], [w, w],
      [w, r], [-w, r], [-w, w], [-r, w], [-r, -w], [-w, -w]
    ];
    var rad = (rotationDeg * Math.PI) / 180;
    return base
      .map(function (p) {
        var x = p[0] * Math.cos(rad) - p[1] * Math.sin(rad);
        var y = p[0] * Math.sin(rad) + p[1] * Math.cos(rad);
        return (cx + x).toFixed(2) + "," + (cy + y).toFixed(2);
      })
      .join(" ");
  }

  function starPoints(cx, cy, r) {
    var pts = [];
    for (var i = 0; i < 10; i++) {
      var a = (Math.PI * i) / 5, rr = i % 2 ? r * 0.45 : r;
      pts.push((cx + rr * Math.sin(a)).toFixed(2) + "," + (cy - rr * Math.cos(a)).toFixed(2));
    }
    return pts.join(" ");
  }

  function markerNode(shape, cx, cy, r, color, parent) {
    if (shape === "circle") {
      return el("circle", { cx: cx, cy: cy, r: r * 0.95, fill: color, stroke: "#fff", "stroke-width": 1.2 }, parent);
    }
    var points;
    switch (shape) {
      case "triangle-down": points = polygonPoints(cx, cy, r * 1.1, 3, 180); break;
      case "triangle-up":   points = polygonPoints(cx, cy, r * 1.1, 3, 0); break;
      case "square":        points = polygonPoints(cx, cy, r, 4, 45); break;
      case "diamond":       points = polygonPoints(cx, cy, r * 1.1, 4, 0); break;
      case "pentagon":      points = polygonPoints(cx, cy, r * 1.1, 5, 0); break;
      case "plus":          points = crossPoints(cx, cy, r * 1.15, 0); break;
      case "cross":         points = crossPoints(cx, cy, r * 1.15, 45); break;
      case "star":          points = starPoints(cx, cy, r * 1.35); break;
      default:              points = polygonPoints(cx, cy, r, 6, 0);
    }
    return el("polygon", {
      points: points, fill: color, stroke: "#fff", "stroke-width": 1.2
    }, parent);
  }

  function legendSwatch(modelId) {
    var s = MODEL_STYLE[modelId];
    var svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 16 16");
    markerNode(s.marker, 8, 8, 6, s.color, svg);
    return svg;
  }

  /* ---------- tooltip ---------- */

  var tooltip = document.getElementById("chart-tooltip");

  function showTooltip(evt, html) {
    tooltip.innerHTML = html;
    tooltip.classList.add("visible");
    moveTooltip(evt);
  }

  function moveTooltip(evt) {
    var pad = 14;
    var w = tooltip.offsetWidth, h = tooltip.offsetHeight;
    var x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
    tooltip.style.left = x + "px";
    tooltip.style.top = y + "px";
  }

  function hideTooltip() { tooltip.classList.remove("visible"); }

  function attachHover(node, htmlFn) {
    node.addEventListener("mouseenter", function (e) { showTooltip(e, htmlFn()); });
    node.addEventListener("mousemove", moveTooltip);
    node.addEventListener("mouseleave", hideTooltip);
  }

  function ttRow(label, value) {
    return '<div class="tt-row"><span>' + label + '</span><span class="val">' + value + "</span></div>";
  }

  // Each model's configured reasoning parameters, from leaderboard.json
  // (the one place refresh-data.py writes them); filled before any chart
  // renders, so every tooltip can state what the model ran with.
  var REASONING = {};
  // Model ids in leaderboard rank order, and their display names; the
  // explorer lists models the same way.
  var RANKED = [];
  var NAMES = {};

  function reasoningRow(id) {
    return REASONING[id] ? ttRow("Reasoning", REASONING[id]) : "";
  }

  /* ---------- leaderboard ---------- */

  // The score the latency and cost charts plot: mean of evaluation-aware
  // appropriate scaffolding and appropriate rigor (latency.json's `score`).
  function overallScore(d) {
    return (d.eval_aware.scaffolding + d.eval_aware.rigor) / 2;
  }

  function renderLeaderboard(data) {
    var table = document.getElementById("leaderboard-table");
    // Rows ranked by the charts' overall score; ties alphabetical.
    var rows = data.models.slice().sort(function (a, b) {
      return overallScore(b) - overallScore(a) || a.name.localeCompare(b.name);
    });
    var metrics = ["scaffolding", "rigor", "avoids_over"];
    var metricLabels = { scaffolding: "Appropriate Scaffolding", rigor: "Appropriate Rigor", avoids_over: "Avoids Over-Scaffolding" };
    var prompts = ["plain", "eval_aware"];

    // best value per (prompt, metric) column
    var best = {};
    prompts.forEach(function (p) {
      metrics.forEach(function (m) {
        best[p + "." + m] = Math.max.apply(null, data.models.map(function (d) { return d[p][m]; }));
      });
    });

    var html = "<thead>";
    html += '<tr class="group-row"><th></th><th></th>' +
      '<th colspan="3" class="group-plain">Plain prompt</th>' +
      '<th colspan="3" class="group-aware">Evaluation-aware prompt</th></tr>';
    html += '<tr><th>Model</th><th class="reasoning">Reasoning</th>';
    prompts.forEach(function (p) {
      metrics.forEach(function (m, i) {
        html += "<th" + (i === 0 ? ' class="table-divider"' : "") + ">" + metricLabels[m] + "</th>";
      });
    });
    html += "</tr></thead><tbody>";

    rows.forEach(function (d) {
      // One provider parameter per line, as the config lists them.
      html += "<tr><td>" + d.name + '</td><td class="reasoning">' +
        (d.reasoning || "").split(", ").join("<br>") + "</td>";
      prompts.forEach(function (p) {
        metrics.forEach(function (m, i) {
          var v = d[p][m];
          var cls = [];
          if (i === 0) cls.push("table-divider");
          if (v === best[p + "." + m]) cls.push("best");
          html += "<td" + (cls.length ? ' class="' + cls.join(" ") + '"' : "") + ">" + v.toFixed(3) + "</td>";
        });
      });
      html += "</tr>";
    });
    html += "</tbody>";
    table.innerHTML = html;
  }

  /* ---------- time-to-first-answer-token vs performance scatter ----------
     x is the probe's TTFT p50 (`ttft_s`) from `tutormoments latency`, a
     strictly serial probe: the only latency figure that is comparable across
     models. The site calls it TTFAT, time to first *answer* token: the probe
     times the first visible token, so a reasoning model's thinking counts
     toward it. Elsewhere "TTFT" usually means the first token of any kind,
     reasoning included, which would flatter a model that thinks out loud. The paper's Fig. 7
     plotted end-to-end seconds per turn, which benchmark runs gather under
     concurrency, so it moves with each model's rate-limit tier as well as with
     the model. That figure is kept in the tooltip rather than on the axis.
     See docs/latency.md in allenai/tutormoments. */

  /* Direct labels for a scatter: each point's label takes the first side
     (right, left, above, below; `prefer[id]` goes first) whose box clears
     the plot's edges, any line segments, every other marker and the labels
     already placed. Returns {place(point) -> box, clear(box)}. */
  function makeLabeler(points, segments, bounds, prefer) {
    var placed = [];

    function labelBox(side, cx, cy, w) {
      if (side === "right") return { x0: cx + 14, x1: cx + 14 + w, y0: cy - 7, y1: cy + 7, lx: cx + 14, ly: cy + 4, anchor: "start" };
      if (side === "left") return { x0: cx - 14 - w, x1: cx - 14, y0: cy - 7, y1: cy + 7, lx: cx - 14, ly: cy + 4, anchor: "end" };
      if (side === "above") return { x0: cx - w / 2, x1: cx + w / 2, y0: cy - 24, y1: cy - 10, lx: cx, ly: cy - 14, anchor: "middle" };
      return { x0: cx - w / 2, x1: cx + w / 2, y0: cy + 12, y1: cy + 26, lx: cx, ly: cy + 22, anchor: "middle" };
    }

    function clear(b, self) {
      if (b.x0 < bounds.left || b.x1 > bounds.right || b.y0 < bounds.top || b.y1 > bounds.bottom) return false;
      var hitsSegment = segments.some(function (g) {
        for (var t = 0; t <= 1; t += 0.02) {
          var px = g[0] + (g[2] - g[0]) * t, py = g[1] + (g[3] - g[1]) * t;
          if (px > b.x0 - 2 && px < b.x1 + 2 && py > b.y0 - 2 && py < b.y1 + 2) return true;
        }
        return false;
      });
      if (hitsSegment) return false;
      var hitsMarker = points.some(function (p) {
        return p !== self && p.cx + 9 > b.x0 && p.cx - 9 < b.x1 && p.cy + 9 > b.y0 && p.cy - 9 < b.y1;
      });
      if (hitsMarker) return false;
      return !placed.some(function (o) { return o.x0 < b.x1 && o.x1 > b.x0 && o.y0 < b.y1 && o.y1 > b.y0; });
    }

    function place(p) {
      var w = p.d.name.length * 7;
      var sides = ["right", "left", "above", "below"];
      var first = (prefer || {})[p.d.id];
      if (first) sides = [first].concat(sides.filter(function (v) { return v !== first; }));
      var box = null;
      for (var si = 0; si < sides.length && !box; si++) {
        var cand = labelBox(sides[si], p.cx, p.cy, w);
        if (clear(cand, p)) box = cand;
      }
      box = box || labelBox(sides[0], p.cx, p.cy, w); // nothing clear: the first choice
      placed.push(box);
      return box;
    }

    return { place: place, clear: clear };
  }

  // Seconds on a log axis: "0.5", "1", "2", "5", "10", "20".
  function fmtSec(v) { return v < 1 ? v.toFixed(1) : String(Math.round(v)); }

  function renderLatency(data) {
    var block = document.getElementById("latency-block");
    var mount = document.getElementById("latency-chart");
    var tabs = block.querySelectorAll(".chart-tabs button[data-stat]");
    var STAT = {
      p50: { key: "ttft_s", ttlt: "ttlt_s", axis: "median" },
      p95: { key: "ttft_p95_s", ttlt: "ttlt_p95_s", axis: "95th percentile" }
    };
    var current = "p50";
    var has = function (d, k) { return typeof d[k] === "number" && d[k] > 0; };
    if (!data.models.some(function (d) { return has(d, "ttft_s"); })) {
      block.hidden = true;
      return;
    }
    if (!data.models.some(function (d) { return has(d, "ttft_p95_s"); })) {
      block.querySelector(".chart-tabs[data-stats]").hidden = true;
    }

    // One log x-axis for both views, so switching to p95 shows each point
    // moving into its tail rather than the axis rescaling. Log: TTFAT spans
    // more than an order of magnitude, and a linear axis piles the
    // sub-second tutors on top of each other.
    var allSecs = [];
    data.models.forEach(function (d) {
      ["ttft_s", "ttft_p95_s"].forEach(function (k) { if (has(d, k)) allSecs.push(d[k]); });
    });
    var ticks = logTicks(Math.min.apply(null, allSecs) / 1.2, Math.max.apply(null, allSecs) * 1.2);

    function draw() {
      mount.innerHTML = "";
      var st = STAT[current];
      var xv = function (d) { return d[st.key]; };
      // A model with no figure for this statistic cannot be placed. Omitted
      // rather than zero-filled: "not measured" is not "fast".
      var models = data.models.filter(function (d) { return has(d, st.key); });
      var missing = data.models.filter(function (d) { return models.indexOf(d) < 0; })
        .map(function (d) { return d.name; });

      var W = 920, H = 480;
      var m = { top: 24, right: 120, bottom: 58, left: 74 };
      var iw = W - m.left - m.right, ih = H - m.top - m.bottom;
      var lMin = Math.log10(ticks[0]), lMax = Math.log10(ticks[ticks.length - 1]);
      var yMin = SCORE_MIN, yMax = SCORE_MAX;
      var x = function (v) { return m.left + ((Math.log10(v) - lMin) / (lMax - lMin)) * iw; };
      var y = function (v) { return m.top + (1 - (v - yMin) / (yMax - yMin)) * ih; };

      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
        "aria-label": "Scatter plot of tutoring performance against " + st.axis +
          " time to first answer token, log scale, for " + models.length + " language models" });

      var yi;
      ticks.forEach(function (t, i) {
        if (i > 0) el("line", { x1: x(t), y1: m.top, x2: x(t), y2: m.top + ih, stroke: GRID, "stroke-width": 1 }, svg);
        el("text", { x: x(t), y: m.top + ih + 22, "text-anchor": "middle", "font-size": 12, fill: INK_MUTED }, svg)
          .textContent = fmtSec(t);
      });
      for (yi = yMin; yi <= yMax + 0.001; yi += 0.1) {
        el("line", { x1: m.left, y1: y(yi), x2: m.left + iw, y2: y(yi), stroke: GRID, "stroke-width": 1 }, svg);
        el("text", { x: m.left - 10, y: y(yi) + 4, "text-anchor": "end", "font-size": 12, fill: INK_MUTED }, svg)
          .textContent = yi.toFixed(1);
      }
      el("line", { x1: m.left, y1: m.top + ih, x2: m.left + iw, y2: m.top + ih, stroke: INK_MUTED, "stroke-width": 1 }, svg);

      el("text", { x: m.left + iw / 2, y: H - 12, "text-anchor": "middle", "font-size": 13, fill: INK }, svg)
        .textContent = "Time to first answer token (TTFAT), " + st.axis + " seconds (log scale)";
      var yl = el("text", { x: 18, y: m.top + ih / 2, "text-anchor": "middle", "font-size": 13, fill: INK,
        transform: "rotate(-90 18 " + (m.top + ih / 2) + ")" }, svg);
      yl.textContent = "Appropriate scaffolding & rigor (mean)";

      // Latency-performance frontier: the models nothing faster outscores.
      var frontier = scoreFrontier(models, xv);
      var drawn = drawFrontier(svg, frontier, function (d) { return x(xv(d)); }, function (d) { return y(d.score); });

      var points = models.map(function (d) { return { d: d, cx: x(xv(d)), cy: y(d.score) }; });
      var labeler = makeLabeler(points, drawn.segments, { left: m.left, right: W - 4, top: 0, bottom: m.top + ih }, {});

      var pair = function (d, a, b) {
        return (has(d, a) ? d[a].toFixed(2) : "–") + " / " + (has(d, b) ? d[b].toFixed(2) : "–") + " s";
      };

      points.forEach(function (p) {
        var d = p.d, cx = p.cx, cy = p.cy;
        var s = MODEL_STYLE[d.id] || { color: INK, marker: "square" };
        markerNode(s.marker, cx, cy, 8, s.color, svg);
        var box = labeler.place(p);
        el("text", { x: box.lx, y: box.ly, "font-size": 12.5, "font-weight": 600, fill: INK,
          "text-anchor": box.anchor }, svg).textContent = d.name;

        // oversized invisible hit target for hover
        var hit = el("circle", { cx: cx, cy: cy, r: 17, fill: "transparent", cursor: "pointer" }, svg);
        attachHover(hit, function () {
          // Two decimals: the figures are stored to 0.01 s, and one decimal
          // hid real gaps (0.80 vs 0.82 s decides the p50 frontier).
          var html = '<div class="tt-title">' + d.name + "</div>" +
            reasoningRow(d.id) +
            ttRow("Score", d.score.toFixed(3)) +
            ttRow("First answer token, p50 / p95", pair(d, "ttft_s", "ttft_p95_s"));
          // Split on turn position (turn 1 vs turns 3 and 5); probe rows only.
          if (has(d, "ttft_first_s") && has(d, "ttft_later_s")) {
            html += ttRow("First / later messages (p50)", pair(d, "ttft_first_s", "ttft_later_s"));
          }
          // TTLT: when the student can reply.
          if (has(d, "ttlt_s")) html += ttRow("Full turn, p50 / p95", pair(d, "ttlt_s", "ttlt_p95_s"));
          if (drawn.on[d.id]) html += ttRow("Frontier", "nothing faster scores higher");
          html += ttRow("Measured", d.ttft_source === "run"
            ? "benchmark run, " + (d.ttft_concurrency || "several") + " conversations at a time"
            : "serial probe, 112 moments");
          return html;
        });
      });

      nameFrontier(svg, drawn.segments, labeler.clear);
      mount.appendChild(svg);

      var notes = [];
      if (missing.length) notes.push("Not measured, so not shown: " + missing.join(", ") + ".");
      if (current === "p95") notes.push("p95: the slowest one turn in twenty is at least this slow.");
      if (frontier.length > 1) {
        notes.push("Dashed line: the latency-performance frontier, joining the models that nothing faster outscores.");
      }
      if (models.some(function (d) { return d.ttft_source === "run"; })) {
        notes.push("The working paper's models were timed by a serial probe over 112 moments (August 2026); " +
          "models added since are timed from their benchmark runs (September–October 2026), four conversations " +
          "at a time over all 520 moments. Measured both ways, four of the newer models were between 26% faster " +
          "and 1% slower in their runs than in a serial probe days later: a latency figure is a snapshot of the " +
          "provider on the day it was taken.");
      }
      document.getElementById("latency-footnote").textContent = notes.join(" ");
    }

    tabs.forEach(function (btn) {
      btn.addEventListener("click", function () {
        current = btn.getAttribute("data-stat");
        tabs.forEach(function (b) { b.setAttribute("aria-selected", String(b === btn)); });
        draw();
      });
    });
    draw();
  }

  /* ---------- cost vs performance scatter ----------
     x is the uncached list cost per tutor response: every prompt token at
     the model's input rate, output and reasoning at its output rate, over
     the counted tutor calls, priced at the registry rates in force when the
     data was refreshed. Measured from the model's full benchmark run, or
     from the latency probe where that run predates usage capture; each
     row's `source` says which, and the tooltip shows it. Uncached because the as-run cost moves with the cache mix
     our harness happened to get, which describes the harness, not the
     model. Shown per 1,000 responses: single responses are fractions of a
     cent. Log scale: costs across models span orders of magnitude. See
     docs/cost.md in allenai/tutormoments. */

  function fmtUSD(v) {
    if (v >= 100) return "$" + v.toFixed(0);
    if (v >= 1) return "$" + v.toFixed(2);
    return "$" + v.toPrecision(2);
  }

  // 1-2-5 log ticks: the largest at or below lo through the smallest at or
  // above hi, so the axis starts and ends on a labelled value.
  function logTicks(lo, hi) {
    var all = [];
    for (var e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)); e++) {
      [1, 2, 5].forEach(function (f) { all.push(f * Math.pow(10, e)); });
    }
    var first = all.filter(function (v) { return v <= lo * 1.0001; }).pop();
    var last = all.filter(function (v) { return v >= hi * 0.9999; })[0];
    return all.filter(function (v) { return v >= first && v <= last; });
  }

  // Frontier on a lower-is-better x (cost, latency): the models no other
  // model beats on both x and score. Walking up the x axis, a model is on it
  // when it scores above every model with a lower x.
  function scoreFrontier(models, xOf) {
    var best = -Infinity;
    return models.slice()
      .sort(function (a, b) { return xOf(a) - xOf(b) || b.score - a.score; })
      .filter(function (d) {
        if (d.score <= best) return false;
        best = d.score;
        return true;
      });
  }

  // Draw a frontier as straight segments between its models, the way
  // Artificial Analysis draws it, ending at the top frontier model. Returns
  // the segments (for label placement) and the set of frontier ids.
  function drawFrontier(svg, frontier, px, py) {
    var on = {}, segments = [];
    frontier.forEach(function (d) { on[d.id] = true; });
    for (var fi = 1; fi < frontier.length; fi++) {
      segments.push([px(frontier[fi - 1]), py(frontier[fi - 1]), px(frontier[fi]), py(frontier[fi])]);
    }
    if (segments.length) {
      el("path", {
        d: "M" + segments[0][0].toFixed(1) + " " + segments[0][1].toFixed(1) + " " +
          segments.map(function (g) { return "L" + g[2].toFixed(1) + " " + g[3].toFixed(1); }).join(" "),
        fill: "none", stroke: INK_MUTED, "stroke-width": 1.5,
        "stroke-dasharray": "5 4", "stroke-linejoin": "round"
      }, svg);
    }
    return { segments: segments, on: on };
  }

  // Name a frontier once, beside its longest segment, where `clear` finds room.
  function nameFrontier(svg, segments, clear) {
    if (!segments.length) return;
    var longest = segments.slice().sort(function (g, h) {
      return Math.hypot(h[2] - h[0], h[3] - h[1]) - Math.hypot(g[2] - g[0], g[3] - g[1]);
    })[0];
    var mx = (longest[0] + longest[2]) / 2, my = (longest[1] + longest[3]) / 2;
    [[0, -12], [0, 16], [-30, -12], [30, 16]].some(function (o) {
      var b = { x0: mx + o[0] - 22, x1: mx + o[0] + 22, y0: my + o[1] - 10, y1: my + o[1] + 3 };
      if (!clear(b, null)) return false;
      el("text", { x: mx + o[0], y: my + o[1], "text-anchor": "middle", "font-size": 11.5,
        "font-style": "italic", fill: INK_MUTED }, svg).textContent = "frontier";
      return true;
    });
  }

  function renderCost(data) {
    var block = document.getElementById("cost-block");
    var mount = document.getElementById("cost-chart");
    var models = data.models.filter(function (d) {
      return typeof d.uncached_cost_per_response_usd === "number" && d.uncached_cost_per_response_usd > 0;
    });
    if (!models.length) return; // block stays hidden

    var per1k = function (d) { return d.uncached_cost_per_response_usd * 1000; };

    var W = 920, H = 480;
    var m = { top: 24, right: 120, bottom: 58, left: 74 };
    var iw = W - m.left - m.right, ih = H - m.top - m.bottom;

    var costs = models.map(per1k);
    // A little padding so no marker sits on the axis ends.
    var ticks = logTicks(Math.min.apply(null, costs) / 1.2, Math.max.apply(null, costs) * 1.2);
    var cMin = ticks[0], cMax = ticks[ticks.length - 1];

    var yMin = SCORE_MIN, yMax = SCORE_MAX;
    var lMin = Math.log10(cMin), lMax = Math.log10(cMax);
    var x = function (v) { return m.left + ((Math.log10(v) - lMin) / (lMax - lMin)) * iw; };
    var y = function (v) { return m.top + (1 - (v - yMin) / (yMax - yMin)) * ih; };

    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": "Scatter plot of tutoring performance against uncached cost per 1,000 tutor responses, log scale, for "
        + models.length + " language models" });

    var yi;
    ticks.forEach(function (t, i) {
      if (i > 0) {
        el("line", { x1: x(t), y1: m.top, x2: x(t), y2: m.top + ih, stroke: GRID, "stroke-width": 1 }, svg);
      }
      el("text", { x: x(t), y: m.top + ih + 22, "text-anchor": "middle", "font-size": 12, fill: INK_MUTED }, svg)
        .textContent = fmtUSD(t).replace(/\.00$/, "");
    });
    for (yi = yMin; yi <= yMax + 0.001; yi += 0.1) {
      el("line", { x1: m.left, y1: y(yi), x2: m.left + iw, y2: y(yi), stroke: GRID, "stroke-width": 1 }, svg);
      el("text", { x: m.left - 10, y: y(yi) + 4, "text-anchor": "end", "font-size": 12, fill: INK_MUTED }, svg)
        .textContent = yi.toFixed(1);
    }
    el("line", { x1: m.left, y1: m.top + ih, x2: m.left + iw, y2: m.top + ih, stroke: INK_MUTED, "stroke-width": 1 }, svg);

    el("text", { x: m.left + iw / 2, y: H - 12, "text-anchor": "middle", "font-size": 13, fill: INK }, svg)
      .textContent = "Uncached cost per 1,000 tutor responses (US$, log scale)";
    var yl = el("text", { x: 18, y: m.top + ih / 2, "text-anchor": "middle", "font-size": 13, fill: INK,
      transform: "rotate(-90 18 " + (m.top + ih / 2) + ")" }, svg);
    yl.textContent = "Appropriate scaffolding & rigor (mean)";

    var frontier = scoreFrontier(models, per1k);
    var drawn = drawFrontier(svg, frontier, function (d) { return x(per1k(d)); }, function (d) { return y(d.score); });
    var segments = drawn.segments, onFrontier = drawn.on;

    // Labels: each model takes the first placement whose box clears the
    // frontier line, every marker and the labels already placed. A model in
    // `prefer` tries that side first.
    var prefer = { "claude-opus-5-5": "below", "claude-sonnet-5-5": "above" };
    var points = models.map(function (d) { return { d: d, cx: x(per1k(d)), cy: y(d.score) }; });
    var labeler = makeLabeler(points, segments, { left: m.left, right: W - 4, top: 0, bottom: m.top + ih }, prefer);
    var boxClear = labeler.clear;

    points.forEach(function (p) {
      var d = p.d, cx = p.cx, cy = p.cy;
      var s = MODEL_STYLE[d.id] || { color: INK, marker: "square" };
      markerNode(s.marker, cx, cy, 8, s.color, svg);
      var box = labeler.place(p);
      el("text", {
        x: box.lx, y: box.ly, "font-size": 12.5, "font-weight": 600, fill: INK,
        "text-anchor": box.anchor
      }, svg).textContent = d.name;

      var hit = el("circle", { cx: cx, cy: cy, r: 17, fill: "transparent", cursor: "pointer" }, svg);
      attachHover(hit, function () {
        var html = '<div class="tt-title">' + d.name + "</div>" +
          reasoningRow(d.id) +
          ttRow("Score", d.score.toFixed(3)) +
          ttRow("Cost per 1,000 responses", fmtUSD(per1k(d)));
        if (typeof d.prompt_tokens_per_response === "number" && typeof d.output_tokens_per_response === "number") {
          html += ttRow("Tokens per response, in / out",
            d.prompt_tokens_per_response.toLocaleString() + " / " + d.output_tokens_per_response.toLocaleString());
        }
        if (d.rates) {
          html += ttRow("List price per MTok, in / out",
            "$" + d.rates.input_per_mtok + " / $" + d.rates.output_per_mtok + " (" + d.rates.as_of + ")");
        }
        if (onFrontier[d.id]) html += ttRow("Frontier", "nothing cheaper scores higher");
        if (d.source && typeof d.n_calls === "number") {
          html += ttRow("Measured over", d.n_calls.toLocaleString() + " responses, " +
            (d.source.kind === "run" ? "benchmark run" : "latency probe"));
        }
        return html;
      });
    });

    nameFrontier(svg, segments, boxClear);

    mount.appendChild(svg);

    var notes = [];
    var omitted = data.omitted || [];
    if (omitted.length) {
      notes.push("Not shown, no run with recorded token usage: " + omitted.join(", ") + ".");
    }
    if (frontier.length > 1) {
      notes.push("Dashed line: the cost-performance frontier, joining the models that nothing cheaper outscores.");
    }
    notes.push("Uncached list prices, a ceiling: provider prompt caching can cut the input share substantially.");
    document.getElementById("cost-footnote").textContent = notes.join(" ");
    block.hidden = false;
  }

  /* ---------- action distribution strip plot (paper Fig. 4) ---------- */

  function fmtPct(v) {
    var s = v.pct.toFixed(1) + "%";
    if (v.ci) s += " (95% CI " + v.ci[0].toFixed(1) + "–" + v.ci[1].toFixed(1) + ")";
    return s;
  }

  function renderActions(data) {
    var block = document.getElementById("actions-block");
    var mount = document.getElementById("actions-chart");
    var legend = document.getElementById("actions-legend");
    var promptTabs = block.querySelectorAll(".chart-tabs button[data-prompt]");
    var providerTabs = block.querySelectorAll(".chart-tabs button[data-provider]");
    var current = "plain";
    // "all" (the default) or one provider, so a crowded strip can be read
    // a few series at a time.
    var currentProvider = "all";
    function providerOf(d) { return (MODEL_STYLE[d.id] || {}).provider; }

    var chartHighlight = null; // reassigned by draw(); legend hovers call the current one

    function draw() {
      mount.innerHTML = "";
      var models = data.models.filter(function (d) {
        return currentProvider === "all" || providerOf(d) === currentProvider;
      });
      var offsets = models.map(function (_, i) {
        return models.length > 1 ? -0.4 + (0.8 * i) / (models.length - 1) : 0;
      });
      var r = models.length > 10 ? 5 : 6;
      var cats = data.categories;
      var W = 960, H = 460;
      var m = { top: 18, right: 12, bottom: 64, left: 62 };
      var iw = W - m.left - m.right, ih = H - m.top - m.bottom;

      // One y-axis for every tab: the max over both prompts and all models,
      // so switching prompt or provider never rescales the chart.
      var yMax = 0;
      data.models.forEach(function (d) {
        ["plain", "eval_aware"].forEach(function (p) {
          cats.forEach(function (c) {
            var v = (d[p] && d[p][c.key] && d[p][c.key].pct) || 0;
            if (v > yMax) yMax = v;
          });
        });
      });
      cats.forEach(function (c) { if (c.human.pct > yMax) yMax = c.human.pct; });
      yMax = Math.ceil((yMax + 2) / 5) * 5;

      var colW = iw / cats.length;
      var cx = function (ci, off) { return m.left + colW * (ci + 0.5 + (off || 0) * 0.9); };
      var y = function (v) { return m.top + (1 - v / yMax) * ih; };

      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
        "aria-label": "Strip plot of the share of tutor actions per pedagogical move, for each model, with a dashed human-tutor baseline per move" });

      var yi;
      for (yi = 0; yi <= yMax; yi += 5) {
        el("line", { x1: m.left, y1: y(yi), x2: m.left + iw, y2: y(yi), stroke: GRID, "stroke-width": 1 }, svg);
        el("text", { x: m.left - 8, y: y(yi) + 4, "text-anchor": "end", "font-size": 11.5, fill: INK_MUTED }, svg)
          .textContent = yi;
      }

      cats.forEach(function (c, ci) {
        if (ci > 0) {
          el("line", { x1: m.left + colW * ci, y1: m.top, x2: m.left + colW * ci, y2: m.top + ih,
            stroke: GRID, "stroke-width": 1 }, svg);
        }
        // category label, wrapped on spaces to at most 2 lines
        var words = c.label.split(" ");
        var lines = words.length > 1
          ? [words.slice(0, Math.ceil(words.length / 2)).join(" "), words.slice(Math.ceil(words.length / 2)).join(" ")]
          : [c.label];
        lines.forEach(function (ln, li) {
          el("text", { x: cx(ci, 0), y: m.top + ih + 18 + li * 13, "text-anchor": "middle",
            "font-size": 11, fill: INK }, svg).textContent = ln;
        });

        // human baseline
        var hb = el("line", {
          x1: cx(ci, 0) - colW * 0.42, x2: cx(ci, 0) + colW * 0.42,
          y1: y(c.human.pct), y2: y(c.human.pct),
          stroke: INK, "stroke-width": 1.6, "stroke-dasharray": "5 3"
        }, svg);
        attachHover(hb, function () {
          return '<div class="tt-title">' + c.label + "</div>" +
            ttRow("Human tutors", fmtPct(c.human));
        });
      });

      // one <g> per model so hovering any point can highlight the whole series
      var seriesGroups = {};
      models.forEach(function (d) {
        seriesGroups[d.id] = el("g", { style: "transition: opacity 0.12s ease" }, svg);
      });
      var hitLayer = el("g", {}, svg);

      function highlight(id) {
        models.forEach(function (d, i) {
          var dim = id && d.id !== id;
          seriesGroups[d.id].setAttribute("opacity", dim ? 0.15 : 1);
          var item = legend.children[i + 1]; // children[0] is the human entry
          if (item) item.style.opacity = dim ? 0.35 : 1;
        });
      }
      chartHighlight = highlight;

      models.forEach(function (d, di) {
        var n = d.n_moments && d.n_moments[current];
        cats.forEach(function (c, ci) {
          var v = (d[current] && d[current][c.key]) || { pct: 0 };
          var s = MODEL_STYLE[d.id] || { color: INK, marker: "square" };
          var px = cx(ci, offsets[di]), py = y(v.pct);
          markerNode(s.marker, px, py, r, s.color, seriesGroups[d.id]);
          var hit = el("circle", { cx: px, cy: py, r: 12, fill: "transparent", cursor: "pointer" }, hitLayer);
          attachHover(hit, function () {
            return '<div class="tt-title">' + c.label + "</div>" +
              ttRow(d.name, fmtPct(v)) +
              reasoningRow(d.id) +
              (n ? ttRow("Moments classified", n.toLocaleString()) : "") +
              ttRow("Human tutors", fmtPct(c.human));
          });
          hit.addEventListener("mouseenter", function () { highlight(d.id); });
          hit.addEventListener("mouseleave", function () { highlight(null); });
        });
      });

      el("text", { x: 16, y: m.top + ih / 2, "text-anchor": "middle", "font-size": 12.5, fill: INK,
        transform: "rotate(-90 16 " + (m.top + ih / 2) + ")" }, svg).textContent = "Share of tutor actions (%)";

      mount.appendChild(svg);
      drawLegend(models);
    }

    // legend (human baseline + the visible models)
    function drawLegend(models) {
      legend.innerHTML = "";
      var humanItem = document.createElement("span");
      humanItem.className = "item";
      humanItem.innerHTML = '<svg viewBox="0 0 16 16"><line x1="1" y1="8" x2="15" y2="8" stroke="' + INK +
        '" stroke-width="2" stroke-dasharray="4 2.5"/></svg>Human tutors';
      legend.appendChild(humanItem);
      models.forEach(function (d) {
        var item = document.createElement("span");
        item.className = "item";
        item.appendChild(legendSwatch(d.id));
        item.appendChild(document.createTextNode(d.name));
        item.addEventListener("mouseenter", function () { if (chartHighlight) chartHighlight(d.id); });
        item.addEventListener("mouseleave", function () { if (chartHighlight) chartHighlight(null); });
        legend.appendChild(item);
      });
    }

    function bindTabs(tabs, attr, set) {
      tabs.forEach(function (btn) {
        btn.addEventListener("click", function () {
          set(btn.getAttribute(attr));
          tabs.forEach(function (b) { b.setAttribute("aria-selected", String(b === btn)); });
          draw();
        });
      });
    }
    bindTabs(promptTabs, "data-prompt", function (v) { current = v; });
    bindTabs(providerTabs, "data-provider", function (v) { currentProvider = v; });

    block.hidden = false;
    draw();
  }

  /* ---------- scaffolding-vs-rigor KL dot plot ----------
     One row per model, x = the KL divergence between its action
     distributions in scaffolding and in rigor moments (mean of KL(S||R) and
     KL(R||S), nats): how much its choice of move depends on what the moment
     calls for. The dotted line is the human tutors at the same moments.
     Every series is at full sample; see kl.json's source. */

  function renderKL(data) {
    var block = document.getElementById("kl-block");
    var mount = document.getElementById("kl-chart");
    var tabs = block.querySelectorAll(".chart-tabs button[data-prompt]");
    var current = "plain";
    if (!data.models.length || !data.human) return; // block stays hidden

    // One x-axis for both prompts, so switching never rescales.
    var vmax = data.human.mean;
    data.models.forEach(function (d) {
      ["plain", "eval_aware"].forEach(function (p) { if (d[p] && d[p].mean > vmax) vmax = d[p].mean; });
    });
    var xMax = Math.ceil((vmax * 1.08) / 0.1) * 0.1;

    function draw() {
      mount.innerHTML = "";
      var rows = data.models.filter(function (d) { return d[current]; })
        .sort(function (a, b) { return b[current].mean - a[current].mean || a.name.localeCompare(b.name); });
      var rowH = 26;
      var W = 920;
      var m = { top: 30, right: 40, bottom: 56, left: 190 };
      var iw = W - m.left - m.right, ih = rows.length * rowH;
      var H = m.top + ih + m.bottom;
      var x = function (v) { return m.left + (v / xMax) * iw; };
      var rowY = function (i) { return m.top + rowH * (i + 0.5); };

      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
        "aria-label": "Dot plot of each model's scaffolding-versus-rigor KL divergence, with a dotted line for human tutors" });

      for (var t = 0; t <= xMax + 1e-9; t += 0.1) {
        el("line", { x1: x(t), y1: m.top, x2: x(t), y2: m.top + ih, stroke: GRID, "stroke-width": 1 }, svg);
        el("text", { x: x(t), y: m.top + ih + 20, "text-anchor": "middle", "font-size": 12, fill: INK_MUTED }, svg)
          .textContent = t.toFixed(1);
      }
      el("line", { x1: m.left, y1: m.top + ih, x2: m.left + iw, y2: m.top + ih, stroke: INK_MUTED, "stroke-width": 1 }, svg);
      el("text", { x: m.left + iw / 2, y: H - 12, "text-anchor": "middle", "font-size": 13, fill: INK }, svg)
        .textContent = "KL divergence, scaffolding vs rigor moments (mean of both directions, nats)";

      // human reference: dotted line down the chart
      var hx = x(data.human.mean);
      var hl = el("line", { x1: hx, y1: m.top - 8, x2: hx, y2: m.top + ih, stroke: INK,
        "stroke-width": 1.6, "stroke-dasharray": "2 4", "stroke-linecap": "round" }, svg);
      el("text", { x: hx, y: m.top - 14, "text-anchor": "middle", "font-size": 12, "font-weight": 600, fill: INK }, svg)
        .textContent = "Human tutors " + data.human.mean.toFixed(2);
      var hhit = el("rect", { x: hx - 8, y: m.top, width: 16, height: ih, fill: "transparent" }, svg);
      attachHover(hhit, function () {
        var h = data.human;
        return '<div class="tt-title">Human tutors</div>' +
          ttRow("KL, mean", h.mean.toFixed(3)) +
          ttRow("KL(S‖R) / KL(R‖S)", h.s_r.toFixed(3) + " / " + h.r_s.toFixed(3)) +
          ttRow("Moments, S / R", h.n_scaffolding + " / " + h.n_rigor);
      });

      rows.forEach(function (d, i) {
        var v = d[current], cy = rowY(i), cx = x(v.mean);
        var s = MODEL_STYLE[d.id] || { color: INK, marker: "square" };
        el("line", { x1: m.left, y1: cy, x2: cx, y2: cy, stroke: GRID, "stroke-width": 1.5 }, svg);
        el("text", { x: m.left - 12, y: cy + 4, "text-anchor": "end", "font-size": 12.5, "font-weight": 600, fill: INK }, svg)
          .textContent = d.name;
        markerNode(s.marker, cx, cy, 7, s.color, svg);
        var hit = el("rect", { x: m.left - 180, y: cy - rowH / 2, width: cx - m.left + 196, height: rowH,
          fill: "transparent", cursor: "pointer" }, svg);
        attachHover(hit, function () {
          return '<div class="tt-title">' + d.name + "</div>" +
            reasoningRow(d.id) +
            ttRow("KL, mean", v.mean.toFixed(3)) +
            ttRow("KL(S‖R) / KL(R‖S)", v.s_r.toFixed(3) + " / " + v.r_s.toFixed(3)) +
            ttRow("Moments, S / R", v.n_scaffolding + " / " + v.n_rigor) +
            ttRow("Human tutors", data.human.mean.toFixed(3));
        });
      });

      mount.appendChild(svg);
    }

    tabs.forEach(function (btn) {
      btn.addEventListener("click", function () {
        current = btn.getAttribute("data-prompt");
        tabs.forEach(function (b) { b.setAttribute("aria-selected", String(b === btn)); });
        draw();
      });
    });

    document.getElementById("kl-footnote").textContent =
      "Every series at full sample, about 260 moments per situation, so these values are not comparable to " +
      "the working paper's KL table, whose add-one smoothing over ~50 moments per situation pulls every " +
      "value toward zero (human tutors 0.18 there). Paper models from the action-taxonomy release's " +
      "classifications of their full replays; later models from their own runs.";
    block.hidden = false;
    draw();
  }

  /* ---------- moment explorer ----------
     Ten curated moments: the session summary and the turns just before the
     cut, then the human tutor's real continuation beside one model's replay.
     The human side is labelled by the expert annotators, the model side by
     the benchmark's scorer (`right` = the leaderboard's rule). All text is
     set with textContent: transcripts are data, never markup. */

  var ACTION_TEXT = {
    scaffolding: "Scaffolded",
    rigor: "Pushed for rigor",
    both: "Scaffolded and pushed for rigor",
    neither: "Neither scaffolded nor pushed"
  };

  function h(tag, cls, text, parent) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }

  function renderTurns(turns, parent, studentLabel) {
    var box = h("div", "ex-turns", null, parent);
    turns.forEach(function (t) {
      // A bracketed line on its own is a screen or pause annotation from the
      // transcript, not speech; show it as an event.
      var isEvent = /^\[[A-Z_ :0-9]+/.test(t.text);
      var div = h("div", "ex-turn " + (isEvent ? "event" : t.role), null, box);
      if (!isEvent) h("span", "who", t.role === "student" ? studentLabel : "Tutor", div);
      div.appendChild(document.createTextNode(t.text));
    });
    return box;
  }

  function renderExplorer(data) {
    var section = document.getElementById("explorer");
    var card = document.getElementById("ex-card");
    var momentSel = document.getElementById("ex-moment");
    var modelSel = document.getElementById("ex-model");
    var prev = document.getElementById("ex-prev"), next = document.getElementById("ex-next");
    var promptTabs = document.querySelectorAll("#ex-prompts button[data-prompt]");
    var moments = data.moments || [];
    if (!moments.length) return;

    var state = { i: 0, prompt: "eval_aware", model: null };
    var order = RANKED.length ? RANKED : Object.keys(moments[0].models);
    state.model = order.filter(function (id) { return moments[0].models[id]; })[0];

    moments.forEach(function (mo, i) {
      var opt = h("option", null, (i + 1) + " of " + moments.length + " · " + mo.title, momentSel);
      opt.value = i;
    });
    order.forEach(function (id) {
      if (!moments.some(function (mo) { return mo.models[id]; })) return;
      var opt = h("option", null, NAMES[id] || id, modelSel);
      opt.value = id;
    });

    function draw() {
      var mo = moments[state.i], key = state.prompt;
      momentSel.value = state.i;
      modelSel.value = state.model;
      prev.disabled = state.i === 0;
      next.disabled = state.i === moments.length - 1;
      card.innerHTML = "";

      var meta = h("div", "ex-meta", null, card);
      h("span", "ex-chip", mo.dimension === "rigor" ? "Rigor moment" : "Scaffolding moment", meta);
      h("span", null, mo.n_right[key] + " of " + mo.n_models[key] + " models responded appropriately with the " +
        (key === "plain" ? "plain" : "evaluation-aware") + " prompt", meta);

      h("h4", null, "The session so far", card);
      h("p", "ex-summary", mo.summary, card);

      h("h4", null, "Just before the cut", card);
      renderTurns(mo.context, card, "Student");
      if (mo.hint) h("p", "ex-hint", "Annotator's note on the moment: " + mo.hint.trim(), card);

      h("div", "ex-cut", "Cut point", card);

      var cols = h("div", "ex-columns", null, card);

      var hc = h("div", "ex-col", null, cols);
      h("h4", null, "Human tutor", hc);
      var hv = h("div", "ex-verdict", null, hc);
      if (mo.human.direction) h("span", "ex-chip", ACTION_TEXT[mo.human.direction] || mo.human.direction, hv);
      var eff = mo.human.effectiveness || {}, effN = 0;
      Object.keys(eff).forEach(function (k) { effN += eff[k]; });
      if (effN) {
        h("span", null, "per the annotators; " + (eff.effective || 0) + " of " + effN + " rated it effective", hv);
      }
      renderTurns(mo.human.turns, hc, "Student");

      var mc = h("div", "ex-col", null, cols);
      var cell = (mo.models[state.model] || {})[key];
      var mh = h("h4", null, NAMES[state.model] || state.model, mc);
      if (REASONING[state.model]) h("small", null, REASONING[state.model], mh);
      if (!cell) {
        h("p", "ex-hint", "No replay of this moment for this model and prompt.", mc);
      } else {
        var mv = h("div", "ex-verdict", null, mc);
        // A miss for over-scaffolding is named as that alone: the action
        // label beside it would read as a second, contradictory verdict.
        var overMiss = !cell.right && cell.overscaffold;
        h("span", "ex-chip " + (cell.right ? "ok" : "miss"),
          (cell.right ? "✓ " : "✗ ") +
            (overMiss ? "Over-scaffolded" : (ACTION_TEXT[cell.label] || cell.label || "No call")), mv);
        var why = cell.right ? "fits the moment, per the scorer"
          : (overMiss ? "per the scorer"
            : (mo.dimension === "rigor" ? "rigor was called for, per the scorer" : "scaffolding was called for, per the scorer"));
        h("span", null, why, mv);
        renderTurns(cell.turns, mc, "Student (simulated)");
        if (cell.action) h("p", "ex-scorer", "Scorer: " + cell.action, mc);
      }

      var strip = h("div", "ex-strip", null, card);
      h("h4", null, "Every model on this moment", strip);
      var items = h("div", "ex-strip-items", null, strip);
      order.forEach(function (id) {
        var c = (mo.models[id] || {})[key];
        if (!c) return;
        var b = h("button", c.right ? "ok" : "miss", null, items);
        b.type = "button";
        b.setAttribute("aria-pressed", String(id === state.model));
        b.title = (c.right ? "Fits the moment" : "Missed it") + " — show this model's replay";
        b.appendChild(legendSwatch(id));
        h("span", null, NAMES[id] || id, b);
        h("span", "mark", c.right ? "✓" : "✗", b);
        b.addEventListener("click", function () { state.model = id; draw(); });
      });
    }

    momentSel.addEventListener("change", function () { state.i = +momentSel.value; draw(); });
    modelSel.addEventListener("change", function () { state.model = modelSel.value; draw(); });
    prev.addEventListener("click", function () { if (state.i > 0) { state.i--; draw(); } });
    next.addEventListener("click", function () { if (state.i < moments.length - 1) { state.i++; draw(); } });
    promptTabs.forEach(function (btn) {
      btn.addEventListener("click", function () {
        state.prompt = btn.getAttribute("data-prompt");
        promptTabs.forEach(function (b) { b.setAttribute("aria-selected", String(b === btn)); });
        draw();
      });
    });

    section.hidden = false;
    draw();
  }

  /* ---------- animation embed ----------
     The animation page is a fixed 1280x720 stage; scale the iframe to the
     card's width (the card's CSS aspect-ratio keeps the height in step). */

  function fitAnimation() {
    document.querySelectorAll(".animation-card iframe").forEach(function (ifr) {
      var w = ifr.parentElement.clientWidth;
      ifr.style.transform = "scale(" + w / 1280 + ")";
    });
  }
  window.addEventListener("resize", fitAnimation);
  fitAnimation();

  // The animation autoplays as soon as the iframe loads, which is well before
  // the reader scrolls to it. Restart it from the top the first time the card
  // actually comes into view (same-origin, so we can click its restart button).
  document.querySelectorAll(".animation-card iframe").forEach(function (ifr) {
    new IntersectionObserver(function (entries, observer) {
      if (!entries.some(function (e) { return e.isIntersecting; })) return;
      observer.disconnect();
      var tries = 0;
      (function restart() {
        var doc = ifr.contentDocument;
        var btn = doc && doc.getElementById("btnReplay");
        if (btn) return btn.click();
        if (++tries < 40) setTimeout(restart, 250); // animation may still be booting
      })();
    }, { threshold: 0.4 }).observe(ifr.parentElement);
  });

  /* ---------- boot ---------- */

  // The charts wait for the leaderboard so their tooltips have REASONING;
  // if it fails they still render, just without that row.
  var leaderboard = fetchJSON("./static/data/leaderboard.json").then(function (data) {
    data.models.forEach(function (d) {
      if (d.reasoning) REASONING[d.id] = d.reasoning;
      NAMES[d.id] = d.name;
    });
    RANKED = data.models.slice().sort(function (a, b) {
      return overallScore(b) - overallScore(a) || a.name.localeCompare(b.name);
    }).map(function (d) { return d.id; });
    renderLeaderboard(data);
  }).catch(function (e) { console.error("leaderboard:", e); });

  function afterLeaderboard(url) {
    return leaderboard.then(function () { return fetchJSON(url); });
  }

  afterLeaderboard("./static/data/latency.json").then(renderLatency)
    .catch(function (e) { console.error("latency chart:", e); });

  // Generated by scripts/refresh-data.py; the section stays hidden until it exists.
  afterLeaderboard("./static/data/cost.json").then(renderCost)
    .catch(function () { /* data pending — leave #cost-block hidden */ });

  // Generated by scripts/refresh-data.py; the section stays hidden until it exists.
  afterLeaderboard("./static/data/moments.json").then(renderExplorer)
    .catch(function () { /* data pending — leave #explorer hidden */ });

  // Generated by scripts/refresh-data.py; the section stays hidden until it exists.
  afterLeaderboard("./static/data/kl.json").then(renderKL)
    .catch(function () { /* data pending — leave #kl-block hidden */ });

  // Generated by scripts/refresh-data.py; the section stays hidden until it exists.
  afterLeaderboard("./static/data/action_distribution.json").then(renderActions)
    .catch(function () { /* data pending — leave #actions-block hidden */ });
})();
