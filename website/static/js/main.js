/* TutorMoments-Preview site
   Renders the leaderboard table and the two findings charts from static/data/*.json.
   Charts are hand-rolled SVG; per-model colors and marker shapes match the paper's
   figures (Okabe-Ito palette; see analysis/working-paper-20260630 in allenai/tutormoments)
   so a model reads the same on the site and in the paper. */

(function () {
  "use strict";

  var MODEL_STYLE = {
    "claude-opus-4-8":             { color: "#D55E00", marker: "triangle-down" },
    "claude-sonnet-4-6":           { color: "#E69F00", marker: "square" },
    "deepseek-ai_DeepSeek-V4-Pro": { color: "#CC79A7", marker: "triangle-up" },
    "gemini-2.5-pro":              { color: "#0072B2", marker: "diamond" },
    "gemini-3.5-flash":            { color: "#56B4E9", marker: "pentagon" },
    "gpt-5.5-2026-04-23":          { color: "#009E73", marker: "plus" },
    "gpt-5.4-mini-2026-03-17":     { color: "#F0529C", marker: "cross" }
  };

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

  function fetchJSON(url) {
    return fetch(url).then(function (r) {
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

  function markerNode(shape, cx, cy, r, color, parent) {
    var points;
    switch (shape) {
      case "triangle-down": points = polygonPoints(cx, cy, r * 1.1, 3, 180); break;
      case "triangle-up":   points = polygonPoints(cx, cy, r * 1.1, 3, 0); break;
      case "square":        points = polygonPoints(cx, cy, r, 4, 45); break;
      case "diamond":       points = polygonPoints(cx, cy, r * 1.1, 4, 0); break;
      case "pentagon":      points = polygonPoints(cx, cy, r * 1.1, 5, 0); break;
      case "plus":          points = crossPoints(cx, cy, r * 1.15, 0); break;
      case "cross":         points = crossPoints(cx, cy, r * 1.15, 45); break;
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

  /* ---------- leaderboard ---------- */

  function renderLeaderboard(data) {
    var table = document.getElementById("leaderboard-table");
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
    html += '<tr class="group-row"><th></th>' +
      '<th colspan="3" class="group-plain">Plain prompt</th>' +
      '<th colspan="3" class="group-aware">Evaluation-aware prompt</th></tr>';
    html += "<tr><th>Model</th>";
    prompts.forEach(function (p) {
      metrics.forEach(function (m, i) {
        html += "<th" + (i === 0 ? ' class="table-divider"' : "") + ">" + metricLabels[m] + "</th>";
      });
    });
    html += "</tr></thead><tbody>";

    data.models.forEach(function (d) {
      html += "<tr><td>" + d.name + "</td>";
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

  /* ---------- latency vs performance scatter (paper Fig. 7) ---------- */

  function renderLatency(data) {
    var mount = document.getElementById("latency-chart");
    var W = 920, H = 480;
    var m = { top: 24, right: 120, bottom: 58, left: 74 };
    var iw = W - m.left - m.right, ih = H - m.top - m.bottom;

    var xMin = 2, xMax = 20, yMin = 0.5, yMax = 0.9;
    var x = function (v) { return m.left + ((v - xMin) / (xMax - xMin)) * iw; };
    var y = function (v) { return m.top + (1 - (v - yMin) / (yMax - yMin)) * ih; };

    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": "Scatter plot of tutoring performance against mean response latency for seven language models" });

    // gridlines + ticks
    var xi, yi;
    for (xi = 4; xi <= 18; xi += 2) {
      el("line", { x1: x(xi), y1: m.top, x2: x(xi), y2: m.top + ih, stroke: GRID, "stroke-width": 1 }, svg);
      el("text", { x: x(xi), y: m.top + ih + 22, "text-anchor": "middle", "font-size": 12, fill: INK_MUTED }, svg)
        .textContent = xi;
    }
    for (yi = 0.5; yi <= 0.901; yi += 0.1) {
      el("line", { x1: m.left, y1: y(yi), x2: m.left + iw, y2: y(yi), stroke: GRID, "stroke-width": 1 }, svg);
      el("text", { x: m.left - 10, y: y(yi) + 4, "text-anchor": "end", "font-size": 12, fill: INK_MUTED }, svg)
        .textContent = yi.toFixed(1);
    }
    el("line", { x1: m.left, y1: m.top + ih, x2: m.left + iw, y2: m.top + ih, stroke: INK_MUTED, "stroke-width": 1 }, svg);

    // axis titles
    el("text", { x: m.left + iw / 2, y: H - 12, "text-anchor": "middle", "font-size": 13, fill: INK }, svg)
      .textContent = "Mean tutor latency per turn (seconds)";
    var yl = el("text", { x: 18, y: m.top + ih / 2, "text-anchor": "middle", "font-size": 13, fill: INK,
      transform: "rotate(-90 18 " + (m.top + ih / 2) + ")" }, svg);
    yl.textContent = "Appropriate scaffolding & rigor (mean)";

    // points + direct labels
    var labelLeft = { "gemini-2.5-pro": true };
    var labelBelow = { "gpt-5.5-2026-04-23": true };
    data.models.forEach(function (d) {
      var s = MODEL_STYLE[d.id] || { color: INK, marker: "square" };
      var cx = x(d.latency_s), cy = y(d.score);
      markerNode(s.marker, cx, cy, 8, s.color, svg);

      var lx = labelLeft[d.id] ? cx - 14 : cx + 14;
      var ly = labelBelow[d.id] ? cy + 22 : cy + 4;
      if (labelBelow[d.id]) lx = cx;
      el("text", {
        x: lx, y: ly, "font-size": 12.5, "font-weight": 600, fill: INK,
        "text-anchor": labelBelow[d.id] ? "middle" : (labelLeft[d.id] ? "end" : "start")
      }, svg).textContent = d.name;

      // oversized invisible hit target for hover
      var hit = el("circle", { cx: cx, cy: cy, r: 17, fill: "transparent", cursor: "pointer" }, svg);
      attachHover(hit, function () {
        return '<div class="tt-title">' + d.name + "</div>" +
          ttRow("Score", d.score.toFixed(3)) +
          ttRow("Latency / turn", d.latency_s.toFixed(1) + " s" + (d.latency_estimated ? " (approx.)" : ""));
      });
    });

    mount.appendChild(svg);

    if (data.models.some(function (d) { return d.latency_estimated; })) {
      document.getElementById("latency-footnote").textContent =
        "Scores are exact (Table 8 of the paper).";
    }
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
    var tabs = block.querySelectorAll(".chart-tabs button");
    var current = "plain";

    var modelIds = data.models.map(function (d) { return d.id; });
    var offsets = modelIds.map(function (_, i) {
      return -0.33 + (0.66 * i) / (modelIds.length - 1);
    });
    var chartHighlight = null; // reassigned by draw(); legend hovers call the current one

    function draw() {
      mount.innerHTML = "";
      var cats = data.categories;
      var W = 960, H = 460;
      var m = { top: 18, right: 12, bottom: 64, left: 62 };
      var iw = W - m.left - m.right, ih = H - m.top - m.bottom;

      var yMax = 0;
      data.models.forEach(function (d) {
        cats.forEach(function (c) {
          var v = (d[current] && d[current][c.key] && d[current][c.key].pct) || 0;
          if (v > yMax) yMax = v;
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
      data.models.forEach(function (d) {
        seriesGroups[d.id] = el("g", { style: "transition: opacity 0.12s ease" }, svg);
      });
      var hitLayer = el("g", {}, svg);

      function highlight(id) {
        data.models.forEach(function (d, i) {
          var dim = id && d.id !== id;
          seriesGroups[d.id].setAttribute("opacity", dim ? 0.15 : 1);
          var item = legend.children[i + 1]; // children[0] is the human entry
          if (item) item.style.opacity = dim ? 0.35 : 1;
        });
      }
      chartHighlight = highlight;

      data.models.forEach(function (d, di) {
        cats.forEach(function (c, ci) {
          var v = (d[current] && d[current][c.key]) || { pct: 0 };
          var s = MODEL_STYLE[d.id] || { color: INK, marker: "square" };
          var px = cx(ci, offsets[di]), py = y(v.pct);
          markerNode(s.marker, px, py, 6, s.color, seriesGroups[d.id]);
          var hit = el("circle", { cx: px, cy: py, r: 12, fill: "transparent", cursor: "pointer" }, hitLayer);
          attachHover(hit, function () {
            return '<div class="tt-title">' + c.label + "</div>" +
              ttRow(d.name, fmtPct(v)) +
              ttRow("Human tutors", fmtPct(c.human));
          });
          hit.addEventListener("mouseenter", function () { highlight(d.id); });
          hit.addEventListener("mouseleave", function () { highlight(null); });
        });
      });

      el("text", { x: 16, y: m.top + ih / 2, "text-anchor": "middle", "font-size": 12.5, fill: INK,
        transform: "rotate(-90 16 " + (m.top + ih / 2) + ")" }, svg).textContent = "Share of tutor actions (%)";

      mount.appendChild(svg);
    }

    // legend (models + human baseline)
    legend.innerHTML = "";
    var humanItem = document.createElement("span");
    humanItem.className = "item";
    humanItem.innerHTML = '<svg viewBox="0 0 16 16"><line x1="1" y1="8" x2="15" y2="8" stroke="' + INK +
      '" stroke-width="2" stroke-dasharray="4 2.5"/></svg>Human tutors';
    legend.appendChild(humanItem);
    data.models.forEach(function (d) {
      var item = document.createElement("span");
      item.className = "item";
      item.appendChild(legendSwatch(d.id));
      item.appendChild(document.createTextNode(d.name));
      item.addEventListener("mouseenter", function () { if (chartHighlight) chartHighlight(d.id); });
      item.addEventListener("mouseleave", function () { if (chartHighlight) chartHighlight(null); });
      legend.appendChild(item);
    });

    tabs.forEach(function (btn) {
      btn.addEventListener("click", function () {
        current = btn.getAttribute("data-prompt");
        tabs.forEach(function (b) { b.setAttribute("aria-selected", String(b === btn)); });
        draw();
      });
    });

    block.hidden = false;
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

  fetchJSON("./static/data/leaderboard.json").then(renderLeaderboard)
    .catch(function (e) { console.error("leaderboard:", e); });

  fetchJSON("./static/data/latency.json").then(renderLatency)
    .catch(function (e) { console.error("latency chart:", e); });

  // Generated by scripts/refresh-data.py; the section stays hidden until it exists.
  fetchJSON("./static/data/action_distribution.json").then(renderActions)
    .catch(function () { /* data pending — leave #actions-block hidden */ });
})();
