const DATA_DIR = "./assets/data/";

const OPTIMIZERS = ["Adam", "AdamW", "RMSProp", "AdaGrad", "SGD"];
const PERTURBATIONS = ["clean", "feature_noise", "edge_removal", "fake_edge_addition"];
const PERTURBATION_LABELS = {
  clean: "Clean graph",
  feature_noise: "Feature noise",
  edge_removal: "Edge removal",
  fake_edge_addition: "Fake edge addition",
};
const PERTURBATION_AXIS_LABELS = {
  clean: "Clean",
  feature_noise: "Feature noise",
  edge_removal: "Edge removal",
  fake_edge_addition: "Fake edges",
};
const OPTIMIZER_COLORS = {
  Adam: "#65ffd4",
  AdamW: "#f6c445",
  RMSProp: "#ff7f6e",
  AdaGrad: "#a78bfa",
  SGD: "#9fc2df",
};
const CLASS_COLORS = ["#65ffd4", "#f6c445", "#ff7f6e", "#a78bfa", "#9be37d", "#f18fc3", "#9fc2df"];

const state = {
  selectedOptimizer: "Adam",
  selectedPerturbation: "feature_noise",
  allResults: [],
  summary: [],
  lossHistory: [],
  dataset: {},
  graph: { nodes: [], edges: [] },
};

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines.shift().split(",");
  return lines.map((line) => {
    const values = line.split(",");
    return headers.reduce((row, header, index) => {
      const value = values[index];
      const numeric = Number(value);
      row[header] = value !== "" && Number.isFinite(numeric) ? numeric : value;
      return row;
    }, {});
  });
}

async function fetchText(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load ${path}`);
  }
  return response.text();
}

async function loadData() {
  const [allResults, summary, lossHistory, dataset, graph] = await Promise.all([
    fetchText(`${DATA_DIR}all_results.csv`).then(parseCsv),
    fetchText(`${DATA_DIR}final_optimizer_summary.csv`).then(parseCsv),
    fetchText(`${DATA_DIR}ossama_loss_history.csv`).then(parseCsv),
    fetchText(`${DATA_DIR}cora_dataset_summary.csv`).then(parseCsv).then((rows) => rows[0]),
    fetch(`${DATA_DIR}cora_graph_sample.json`).then((response) => {
      if (!response.ok) {
        throw new Error("Could not load graph sample");
      }
      return response.json();
    }),
  ]);

  state.allResults = allResults;
  state.summary = summary;
  state.lossHistory = lossHistory;
  state.dataset = dataset;
  state.graph = graph;
}

function formatPct(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function formatNumber(value) {
  return Number(value).toLocaleString("en-US");
}

function getOptimizerAverage(optimizer) {
  const rows = state.allResults.filter((row) => row.optimizer === optimizer);
  const value = rows.reduce((sum, row) => sum + row.test_accuracy, 0) / rows.length;
  return Number.isFinite(value) ? value : 0;
}

function getCleanAccuracy(optimizer) {
  const row = state.summary.find((item) => item.optimizer === optimizer && item.perturbation_type === "clean");
  return row ? row.mean_test_accuracy : 0;
}

function getBestOptimizer() {
  return OPTIMIZERS.map((optimizer) => ({
    optimizer,
    accuracy: getOptimizerAverage(optimizer),
  })).sort((a, b) => b.accuracy - a.accuracy)[0];
}

function setText(selector, value) {
  document.querySelectorAll(selector).forEach((element) => {
    element.textContent = value;
  });
}

function renderStats() {
  const best = getBestOptimizer();
  const bestClean = getCleanAccuracy(best.optimizer);
  setText('[data-stat="bestOptimizer"]', best.optimizer);
  setText('[data-stat="overallAccuracy"]', formatPct(best.accuracy));
  setText('[data-stat="cleanAccuracy"]', formatPct(bestClean));
  setText('[data-stat="nodes"]', formatNumber(state.dataset.num_nodes));
  setText('[data-stat="edges"]', formatNumber(state.dataset.num_edges));
  setText('[data-stat="features"]', formatNumber(state.dataset.num_features));
  setText('[data-stat="classes"]', formatNumber(state.dataset.num_classes));
  setText('[data-stat="rows"]', formatNumber(state.allResults.length));
}

function svgElement(tag, attrs = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function clearSvg(svg, width = 900, height = 420) {
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  return { width, height };
}

function drawText(svg, text, x, y, attrs = {}) {
  const element = svgElement("text", { x, y, ...attrs });
  element.textContent = text;
  svg.appendChild(element);
  return element;
}

function linePath(points) {
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(" ");
}

function drawChartFrame(svg, bounds, yTicks, yScale, xLabels = []) {
  yTicks.forEach((tick) => {
    const y = yScale(tick);
    svg.appendChild(
      svgElement("line", {
        class: "grid-line",
        x1: bounds.left,
        x2: bounds.right,
        y1: y,
        y2: y,
      }),
    );
    drawText(svg, `${Math.round(tick * 100)}%`, 10, y + 4, { "font-size": 12 });
  });

  svg.appendChild(
    svgElement("line", {
      class: "axis-line",
      x1: bounds.left,
      x2: bounds.right,
      y1: bounds.bottom,
      y2: bounds.bottom,
    }),
  );
  svg.appendChild(
    svgElement("line", {
      class: "axis-line",
      x1: bounds.left,
      x2: bounds.left,
      y1: bounds.top,
      y2: bounds.bottom,
    }),
  );

  xLabels.forEach(({ label, x }) => {
    drawText(svg, label, x, bounds.bottom + 28, {
      "font-size": 12,
      "text-anchor": "middle",
    });
  });
}

function drawLegend(svg, x, y) {
  OPTIMIZERS.forEach((optimizer, index) => {
    const group = svgElement("g", { transform: `translate(${x + index * 116}, ${y})` });
    group.appendChild(
      svgElement("circle", {
        cx: 0,
        cy: 0,
        r: 5,
        fill: OPTIMIZER_COLORS[optimizer],
      }),
    );
    const label = svgElement("text", {
      x: 12,
      y: 4,
      class: "legend-item",
    });
    label.textContent = optimizer;
    group.appendChild(label);
    svg.appendChild(group);
  });
}

function drawSummaryChart() {
  const svg = document.getElementById("summaryChart");
  const { width, height } = clearSvg(svg, 940, 420);
  const bounds = { left: 58, right: width - 24, top: 28, bottom: height - 72 };
  const xStep = (bounds.right - bounds.left) / (PERTURBATIONS.length - 1);
  const xByPerturbation = Object.fromEntries(
    PERTURBATIONS.map((perturbation, index) => [perturbation, bounds.left + xStep * index]),
  );
  const yMin = 0.05;
  const yMax = 0.84;
  const yScale = (value) => bounds.bottom - ((value - yMin) / (yMax - yMin)) * (bounds.bottom - bounds.top);

  drawChartFrame(
    svg,
    bounds,
    [0.2, 0.4, 0.6, 0.8],
    yScale,
    PERTURBATIONS.map((perturbation) => ({ label: PERTURBATION_AXIS_LABELS[perturbation], x: xByPerturbation[perturbation] })),
  );

  OPTIMIZERS.forEach((optimizer) => {
    const points = PERTURBATIONS.map((perturbation) => {
      const row = state.summary.find((item) => item.optimizer === optimizer && item.perturbation_type === perturbation);
      return { x: xByPerturbation[perturbation], y: yScale(row.mean_test_accuracy), value: row.mean_test_accuracy };
    });

    svg.appendChild(
      svgElement("path", {
        d: linePath(points),
        fill: "none",
        stroke: OPTIMIZER_COLORS[optimizer],
        "stroke-width": optimizer === state.selectedOptimizer ? 4 : 2.2,
        "stroke-opacity": optimizer === state.selectedOptimizer ? 1 : 0.5,
      }),
    );

    points.forEach((point) => {
      svg.appendChild(
        svgElement("circle", {
          cx: point.x,
          cy: point.y,
          r: optimizer === state.selectedOptimizer ? 5.5 : 4,
          fill: OPTIMIZER_COLORS[optimizer],
          "fill-opacity": optimizer === state.selectedOptimizer ? 1 : 0.72,
        }),
      );
    });
  });

  drawLegend(svg, bounds.left, height - 24);
}

function drawRobustnessChart() {
  const svg = document.getElementById("robustnessChart");
  const { width, height } = clearSvg(svg, 940, 470);
  const rows = state.allResults.filter((row) => row.perturbation_type === state.selectedPerturbation);
  const severities = [...new Set(rows.map((row) => row.severity))].sort((a, b) => a - b);
  const bounds = { left: 58, right: width - 24, top: 28, bottom: height - 72 };
  const xStep = (bounds.right - bounds.left) / Math.max(1, severities.length - 1);
  const xBySeverity = Object.fromEntries(severities.map((severity, index) => [severity, bounds.left + xStep * index]));
  const yMin = 0.05;
  const yMax = 0.84;
  const yScale = (value) => bounds.bottom - ((value - yMin) / (yMax - yMin)) * (bounds.bottom - bounds.top);

  document.querySelector("[data-perturbation-title]").textContent =
    `${PERTURBATION_LABELS[state.selectedPerturbation]} robustness`;

  drawChartFrame(
    svg,
    bounds,
    [0.2, 0.4, 0.6, 0.8],
    yScale,
    severities.map((severity) => ({ label: `${Math.round(severity * 100)}%`, x: xBySeverity[severity] })),
  );

  OPTIMIZERS.forEach((optimizer) => {
    const points = severities.map((severity) => {
      const row = rows.find((item) => item.optimizer === optimizer && item.severity === severity);
      return { x: xBySeverity[severity], y: yScale(row.test_accuracy), value: row.test_accuracy };
    });

    svg.appendChild(
      svgElement("path", {
        d: linePath(points),
        fill: "none",
        stroke: OPTIMIZER_COLORS[optimizer],
        "stroke-width": optimizer === state.selectedOptimizer ? 4 : 2.2,
        "stroke-opacity": optimizer === state.selectedOptimizer ? 1 : 0.48,
      }),
    );

    points.forEach((point) => {
      svg.appendChild(
        svgElement("circle", {
          cx: point.x,
          cy: point.y,
          r: optimizer === state.selectedOptimizer ? 5.5 : 4,
          fill: OPTIMIZER_COLORS[optimizer],
          "fill-opacity": optimizer === state.selectedOptimizer ? 1 : 0.72,
        }),
      );
    });

    const last = points[points.length - 1];
    if (optimizer === state.selectedOptimizer) {
      drawText(svg, `${optimizer} ${formatPct(last.value, 1)}`, last.x - 6, last.y - 12, {
        "font-size": 13,
        "text-anchor": "end",
        fill: OPTIMIZER_COLORS[optimizer],
      });
    }
  });

  drawLegend(svg, bounds.left, height - 24);
}

function drawLossChart() {
  const svg = document.getElementById("lossChart");
  const { width, height } = clearSvg(svg, 940, 470);
  const bounds = { left: 58, right: width - 24, top: 28, bottom: height - 72 };
  const yMin = 0;
  const yMax = 2.05;
  const xScale = (epoch) => bounds.left + ((epoch - 1) / 199) * (bounds.right - bounds.left);
  const yScale = (value) => bounds.bottom - ((value - yMin) / (yMax - yMin)) * (bounds.bottom - bounds.top);

  [0, 0.5, 1, 1.5, 2].forEach((tick) => {
    const y = yScale(tick);
    svg.appendChild(svgElement("line", { class: "grid-line", x1: bounds.left, x2: bounds.right, y1: y, y2: y }));
    drawText(svg, tick.toFixed(1), 16, y + 4, { "font-size": 12 });
  });

  [1, 50, 100, 150, 200].forEach((epoch) => {
    drawText(svg, String(epoch), xScale(epoch), bounds.bottom + 28, {
      "font-size": 12,
      "text-anchor": "middle",
    });
  });

  svg.appendChild(svgElement("line", { class: "axis-line", x1: bounds.left, x2: bounds.right, y1: bounds.bottom, y2: bounds.bottom }));
  svg.appendChild(svgElement("line", { class: "axis-line", x1: bounds.left, x2: bounds.left, y1: bounds.top, y2: bounds.bottom }));

  OPTIMIZERS.forEach((optimizer) => {
    const points = state.lossHistory
      .filter((row) => row.optimizer === optimizer && row.perturbation_type === "clean")
      .filter((row) => row.epoch === 1 || row.epoch % 2 === 0)
      .map((row) => ({ x: xScale(row.epoch), y: yScale(row.validation_loss), value: row.validation_loss }));

    svg.appendChild(
      svgElement("path", {
        d: linePath(points),
        fill: "none",
        stroke: OPTIMIZER_COLORS[optimizer],
        "stroke-width": optimizer === state.selectedOptimizer ? 3.8 : 2,
        "stroke-opacity": optimizer === state.selectedOptimizer ? 1 : 0.45,
      }),
    );
  });

  drawLegend(svg, bounds.left, height - 24);
}

function renderProfile() {
  const optimizer = state.selectedOptimizer;
  const rows = state.summary.filter((row) => row.optimizer === optimizer);
  const meanAccuracy = getOptimizerAverage(optimizer);
  document.querySelector('[data-profile="name"]').textContent = optimizer;
  document.querySelector('[data-profile="score"]').textContent = formatPct(meanAccuracy);
  document.querySelector('[data-profile="dot"]').style.background = OPTIMIZER_COLORS[optimizer];
  document.querySelector('[data-profile="dot"]').style.boxShadow = `0 0 24px ${OPTIMIZER_COLORS[optimizer]}99`;

  const container = document.getElementById("profileBars");
  container.replaceChildren();
  PERTURBATIONS.forEach((perturbation) => {
    const row = rows.find((item) => item.perturbation_type === perturbation);
    const wrapper = document.createElement("div");
    wrapper.className = "bar-row";
    wrapper.innerHTML = `
      <div class="bar-row-head">
        <span>${PERTURBATION_LABELS[perturbation]}</span>
        <strong>${formatPct(row.mean_test_accuracy)}</strong>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width: ${Math.max(4, row.mean_test_accuracy * 100)}%; background: ${OPTIMIZER_COLORS[optimizer]}"></div>
      </div>
    `;
    container.appendChild(wrapper);
  });
}

function renderLeaderboard() {
  const container = document.getElementById("leaderboard");
  container.replaceChildren();
  OPTIMIZERS.map((optimizer) => ({ optimizer, accuracy: getOptimizerAverage(optimizer) }))
    .sort((a, b) => b.accuracy - a.accuracy)
    .forEach((row, index) => {
      const item = document.createElement("div");
      item.className = "leader-row";
      item.innerHTML = `
        <span class="leader-rank">${index + 1}</span>
        <span class="leader-name" style="color:${OPTIMIZER_COLORS[row.optimizer]}">${row.optimizer}</span>
        <span class="leader-score">${formatPct(row.accuracy)}</span>
      `;
      container.appendChild(item);
    });
}

function renderAllCharts() {
  drawSummaryChart();
  drawRobustnessChart();
  drawLossChart();
  renderProfile();
  renderLeaderboard();
}

function bindControls() {
  document.querySelectorAll(".optimizer-button").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedOptimizer = button.dataset.optimizer;
      document.querySelectorAll(".optimizer-button").forEach((item) => {
        item.classList.toggle("active", item === button);
      });
      renderAllCharts();
    });
  });

  document.querySelectorAll(".segment").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedPerturbation = button.dataset.perturbation;
      document.querySelectorAll(".segment").forEach((item) => {
        const isActive = item === button;
        item.classList.toggle("active", isActive);
        item.setAttribute("aria-selected", String(isActive));
      });
      drawRobustnessChart();
    });
  });

  window.addEventListener("resize", () => {
    renderAllCharts();
  });
}

function setupNetworkCanvas() {
  const canvas = document.getElementById("networkCanvas");
  const ctx = canvas.getContext("2d");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let width = 0;
  let height = 0;
  let pixelRatio = 1;
  let pointer = { x: -9999, y: -9999 };

  function resize() {
    pixelRatio = Math.min(2, window.devicePixelRatio || 1);
    width = canvas.clientWidth;
    height = canvas.clientHeight;
    canvas.width = Math.floor(width * pixelRatio);
    canvas.height = Math.floor(height * pixelRatio);
    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  }

  function pointFor(node, time) {
    const orbit = node.degree_rank * 10 + 4;
    return {
      x: node.x * width + Math.sin(time * 0.0004 + node.phase) * orbit,
      y: node.y * height + Math.cos(time * 0.00035 + node.phase * 1.7) * orbit * 0.62,
    };
  }

  function draw(time = 0) {
    if (!width || !height) {
      resize();
    }

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#05070a";
    ctx.fillRect(0, 0, width, height);

    const positions = new Map();
    state.graph.nodes.forEach((node) => {
      positions.set(node.id, pointFor(node, reducedMotion ? 0 : time));
    });

    const nearest = state.graph.nodes
      .map((node) => {
        const position = positions.get(node.id);
        const distance = Math.hypot(position.x - pointer.x, position.y - pointer.y);
        return { node, distance };
      })
      .sort((a, b) => a.distance - b.distance)[0];
    const highlightedId = nearest && nearest.distance < 44 ? nearest.node.id : null;

    ctx.lineWidth = 1;
    state.graph.edges.forEach((edge) => {
      const source = positions.get(edge.source);
      const target = positions.get(edge.target);
      if (!source || !target) return;
      const active = highlightedId && (edge.source === highlightedId || edge.target === highlightedId);
      ctx.strokeStyle = active ? "rgba(246, 196, 69, 0.58)" : "rgba(155, 194, 223, 0.09)";
      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.stroke();
    });

    state.graph.nodes.forEach((node) => {
      const position = positions.get(node.id);
      const active = node.id === highlightedId;
      const radius = active ? 6.2 : 2.1 + node.degree_rank * 3.2;
      ctx.fillStyle = CLASS_COLORS[node.class_id % CLASS_COLORS.length];
      ctx.globalAlpha = active ? 1 : 0.68;
      ctx.beginPath();
      ctx.arc(position.x, position.y, radius, 0, Math.PI * 2);
      ctx.fill();

      if (active) {
        ctx.globalAlpha = 1;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.86)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.fillStyle = "#f4f0e6";
        ctx.font = "12px Inter, system-ui, sans-serif";
        ctx.fillText(`Cora node ${node.original_id} | class ${node.class_id}`, position.x + 10, position.y - 10);
      }
    });
    ctx.globalAlpha = 1;

    if (!reducedMotion) {
      requestAnimationFrame(draw);
    }
  }

  canvas.addEventListener("pointermove", (event) => {
    const rect = canvas.getBoundingClientRect();
    pointer = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  });

  canvas.addEventListener("pointerleave", () => {
    pointer = { x: -9999, y: -9999 };
  });

  window.addEventListener("resize", resize);
  resize();
  draw();
}

function showError(error) {
  document.body.innerHTML = `
    <main class="loading-error">
      <h1>Frontend data could not load.</h1>
      <p>${error.message}</p>
      <p>Run <code>python scripts/build_frontend_data.py</code>, then serve the <code>docs/</code> folder again.</p>
    </main>
  `;
}

loadData()
  .then(() => {
    renderStats();
    bindControls();
    renderAllCharts();
    setupNetworkCanvas();
  })
  .catch(showError);
