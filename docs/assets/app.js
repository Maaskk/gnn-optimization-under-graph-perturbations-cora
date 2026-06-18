const DATA_DIR = "./assets/data/";

const OPTIMIZERS = ["Adam", "AdamW", "RMSProp", "AdaGrad", "SGD"];
const PERTURBATIONS = ["clean", "feature_noise", "edge_removal", "fake_edge_addition"];
const SEVERITIES = [0.05, 0.1, 0.2, 0.3];

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

const PERTURBATION_CAPTIONS = {
  feature_noise: "Gaussian feature disturbance",
  edge_removal: "Random citation links removed",
  fake_edge_addition: "Non-existing citations injected",
};

const OPTIMIZER_COLORS = {
  Adam: "#65ffd4",
  AdamW: "#f5c84b",
  RMSProp: "#ff7b6e",
  AdaGrad: "#a98cff",
  SGD: "#8fc7ff",
};

const CLASS_COLORS = ["#65ffd4", "#f5c84b", "#ff7b6e", "#a98cff", "#a4e86f", "#ff9ac8", "#8fc7ff"];

const state = {
  selectedOptimizer: "Adam",
  selectedPerturbation: "feature_noise",
  selectedSeverityIndex: 0,
  allResults: [],
  summary: [],
  lossHistory: [],
  dataset: {},
  graph: { nodes: [], edges: [] },
  pointer: { x: -9999, y: -9999 },
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
    throw new Error(`Missing frontend data file: ${path}`);
  }
  return response.text();
}

async function loadData() {
  const [allResults, summary, lossHistory, dataset, graph] = await Promise.all([
    fetchText(`${DATA_DIR}all_results.csv`).then(parseCsv),
    fetchText(`${DATA_DIR}final_optimizer_summary.csv`).then(parseCsv),
    fetchText(`${DATA_DIR}clean_loss_history.csv`).then(parseCsv),
    fetchText(`${DATA_DIR}cora_dataset_summary.csv`).then(parseCsv).then((rows) => rows[0]),
    fetch(`${DATA_DIR}cora_graph_sample.json`).then((response) => {
      if (!response.ok) throw new Error("Missing Cora graph sample");
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
  setText('[data-stat="bestOptimizer"]', best.optimizer);
  setText('[data-stat="overallAccuracy"]', formatPct(best.accuracy));
  setText('[data-stat="cleanAccuracy"]', formatPct(getCleanAccuracy(best.optimizer)));
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

function drawText(svg, text, x, y, attrs = {}) {
  const element = svgElement("text", { x, y, ...attrs });
  element.textContent = text;
  svg.appendChild(element);
  return element;
}

function clearSvg(svg, width = 940, height = 450) {
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  return { width, height };
}

function linePath(points) {
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(" ");
}

function drawChartFrame(svg, bounds, yTicks, yScale, xLabels = []) {
  yTicks.forEach((tick) => {
    const y = yScale(tick);
    svg.appendChild(svgElement("line", { class: "grid-line", x1: bounds.left, x2: bounds.right, y1: y, y2: y }));
    drawText(svg, `${Math.round(tick * 100)}%`, bounds.left - 18, y + 4, {
      "font-size": 12,
      "text-anchor": "end",
    });
  });

  svg.appendChild(svgElement("line", { class: "axis-line", x1: bounds.left, x2: bounds.right, y1: bounds.bottom, y2: bounds.bottom }));
  svg.appendChild(svgElement("line", { class: "axis-line", x1: bounds.left, x2: bounds.left, y1: bounds.top, y2: bounds.bottom }));

  xLabels.forEach(({ label, x }) => {
    drawText(svg, label, x, bounds.bottom + 30, {
      "font-size": 12,
      "text-anchor": "middle",
    });
  });
}

function drawLegend(svg, x, y) {
  OPTIMIZERS.forEach((optimizer, index) => {
    const group = svgElement("g", { transform: `translate(${x + index * 118}, ${y})` });
    group.appendChild(svgElement("circle", { cx: 0, cy: 0, r: 5, fill: OPTIMIZER_COLORS[optimizer] }));
    const label = svgElement("text", { x: 12, y: 4, "font-size": 12 });
    label.textContent = optimizer;
    group.appendChild(label);
    svg.appendChild(group);
  });
}

function drawSummaryChart() {
  const svg = document.getElementById("summaryChart");
  const { width, height } = clearSvg(svg, 940, 450);
  const bounds = { left: 66, right: width - 28, top: 36, bottom: height - 78 };
  const xStep = (bounds.right - bounds.left) / (PERTURBATIONS.length - 1);
  const xByPerturbation = Object.fromEntries(PERTURBATIONS.map((perturbation, index) => [perturbation, bounds.left + xStep * index]));
  const yMin = 0.05;
  const yMax = 0.84;
  const yScale = (value) => bounds.bottom - ((value - yMin) / (yMax - yMin)) * (bounds.bottom - bounds.top);

  PERTURBATIONS.forEach((perturbation, index) => {
    if (index % 2 === 0) {
      const x = xByPerturbation[perturbation] - xStep / 2;
      svg.appendChild(svgElement("rect", { class: "chart-band", x, y: bounds.top, width: xStep, height: bounds.bottom - bounds.top }));
    }
  });

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
    const selected = optimizer === state.selectedOptimizer;
    svg.appendChild(svgElement("path", {
      d: linePath(points),
      fill: "none",
      stroke: OPTIMIZER_COLORS[optimizer],
      "stroke-width": selected ? 4.5 : 2.1,
      "stroke-opacity": selected ? 1 : 0.38,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
    }));

    points.forEach((point) => {
      svg.appendChild(svgElement("circle", {
        cx: point.x,
        cy: point.y,
        r: selected ? 6 : 4,
        fill: OPTIMIZER_COLORS[optimizer],
        "fill-opacity": selected ? 1 : 0.64,
      }));
    });

    if (selected) {
      const last = points[points.length - 1];
      drawText(svg, `${optimizer} ${formatPct(last.value)}`, last.x - 8, last.y - 14, {
        "font-size": 13,
        "font-weight": 800,
        "text-anchor": "end",
        fill: OPTIMIZER_COLORS[optimizer],
      });
    }
  });

  drawLegend(svg, bounds.left, height - 28);
}

function drawRobustnessChart() {
  const svg = document.getElementById("robustnessChart");
  const { width, height } = clearSvg(svg, 940, 470);
  const rows = state.allResults.filter((row) => row.perturbation_type === state.selectedPerturbation);
  const severities = SEVERITIES;
  const bounds = { left: 66, right: width - 28, top: 34, bottom: height - 76 };
  const xStep = (bounds.right - bounds.left) / (severities.length - 1);
  const xBySeverity = Object.fromEntries(severities.map((severity, index) => [severity, bounds.left + xStep * index]));
  const yMin = 0.05;
  const yMax = 0.84;
  const yScale = (value) => bounds.bottom - ((value - yMin) / (yMax - yMin)) * (bounds.bottom - bounds.top);
  const selectedSeverity = SEVERITIES[state.selectedSeverityIndex];

  document.querySelector("[data-perturbation-title]").textContent =
    `${PERTURBATION_LABELS[state.selectedPerturbation]} robustness`;

  drawChartFrame(
    svg,
    bounds,
    [0.2, 0.4, 0.6, 0.8],
    yScale,
    severities.map((severity) => ({ label: `${Math.round(severity * 100)}%`, x: xBySeverity[severity] })),
  );

  svg.appendChild(svgElement("line", {
    x1: xBySeverity[selectedSeverity],
    x2: xBySeverity[selectedSeverity],
    y1: bounds.top,
    y2: bounds.bottom,
    stroke: "rgba(245, 200, 75, 0.48)",
    "stroke-width": 1.5,
    "stroke-dasharray": "6 8",
  }));

  OPTIMIZERS.forEach((optimizer) => {
    const points = severities.map((severity) => {
      const row = rows.find((item) => item.optimizer === optimizer && item.severity === severity);
      return { x: xBySeverity[severity], y: yScale(row.test_accuracy), value: row.test_accuracy };
    });
    const selected = optimizer === state.selectedOptimizer;
    svg.appendChild(svgElement("path", {
      d: linePath(points),
      fill: "none",
      stroke: OPTIMIZER_COLORS[optimizer],
      "stroke-width": selected ? 4.2 : 2,
      "stroke-opacity": selected ? 1 : 0.38,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
    }));
    points.forEach((point) => {
      const atSelectedSeverity = Math.abs(point.x - xBySeverity[selectedSeverity]) < 0.5;
      svg.appendChild(svgElement("circle", {
        cx: point.x,
        cy: point.y,
        r: selected || atSelectedSeverity ? 5.5 : 3.8,
        fill: OPTIMIZER_COLORS[optimizer],
        "fill-opacity": selected || atSelectedSeverity ? 1 : 0.58,
      }));
    });
  });

  drawLegend(svg, bounds.left, height - 28);
}

function drawLossChart() {
  const svg = document.getElementById("lossChart");
  const { width, height } = clearSvg(svg, 940, 470);
  const bounds = { left: 66, right: width - 28, top: 34, bottom: height - 76 };
  const yScale = (value) => bounds.bottom - (value / 2.05) * (bounds.bottom - bounds.top);
  const xScale = (epoch) => bounds.left + ((epoch - 1) / 199) * (bounds.right - bounds.left);

  [0, 0.5, 1, 1.5, 2].forEach((tick) => {
    const y = yScale(tick);
    svg.appendChild(svgElement("line", { class: "grid-line", x1: bounds.left, x2: bounds.right, y1: y, y2: y }));
    drawText(svg, tick.toFixed(1), bounds.left - 18, y + 4, { "font-size": 12, "text-anchor": "end" });
  });

  [1, 50, 100, 150, 200].forEach((epoch) => {
    drawText(svg, String(epoch), xScale(epoch), bounds.bottom + 30, { "font-size": 12, "text-anchor": "middle" });
  });

  svg.appendChild(svgElement("line", { class: "axis-line", x1: bounds.left, x2: bounds.right, y1: bounds.bottom, y2: bounds.bottom }));
  svg.appendChild(svgElement("line", { class: "axis-line", x1: bounds.left, x2: bounds.left, y1: bounds.top, y2: bounds.bottom }));

  OPTIMIZERS.forEach((optimizer) => {
    const points = state.lossHistory
      .filter((row) => row.optimizer === optimizer && row.perturbation_type === "clean")
      .filter((row) => row.epoch === 1 || row.epoch % 2 === 0)
      .map((row) => ({ x: xScale(row.epoch), y: yScale(row.validation_loss) }));
    const selected = optimizer === state.selectedOptimizer;
    svg.appendChild(svgElement("path", {
      d: linePath(points),
      fill: "none",
      stroke: OPTIMIZER_COLORS[optimizer],
      "stroke-width": selected ? 3.8 : 2,
      "stroke-opacity": selected ? 1 : 0.38,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
    }));
  });

  drawLegend(svg, bounds.left, height - 28);
}

function renderProfile() {
  const optimizer = state.selectedOptimizer;
  const rows = state.summary.filter((row) => row.optimizer === optimizer);
  const meanAccuracy = getOptimizerAverage(optimizer);
  document.querySelector('[data-profile="name"]').textContent = optimizer;
  document.querySelector('[data-profile="score"]').textContent = formatPct(meanAccuracy);
  const dot = document.querySelector('[data-profile="dot"]');
  dot.style.background = OPTIMIZER_COLORS[optimizer];
  dot.style.boxShadow = `0 0 24px ${OPTIMIZER_COLORS[optimizer]}aa`;

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

function renderSeverityMetrics() {
  const severity = SEVERITIES[state.selectedSeverityIndex];
  const rows = state.allResults
    .filter((row) => row.perturbation_type === state.selectedPerturbation && row.severity === severity)
    .sort((a, b) => b.test_accuracy - a.test_accuracy);
  const container = document.getElementById("severityMetrics");
  container.replaceChildren();
  rows.slice(0, 5).forEach((row) => {
    const item = document.createElement("div");
    item.className = "metric-row";
    item.innerHTML = `
      <span style="color:${OPTIMIZER_COLORS[row.optimizer]}">${row.optimizer}</span>
      <strong>${formatPct(row.test_accuracy)}</strong>
    `;
    container.appendChild(item);
  });
}

function renderHud() {
  const severity = SEVERITIES[state.selectedSeverityIndex];
  document.querySelector("[data-hud='mode']").textContent = PERTURBATION_LABELS[state.selectedPerturbation];
  document.querySelector("[data-hud='severity']").textContent = `${Math.round(severity * 100)}%`;
  document.querySelector("[data-hud='caption']").textContent = PERTURBATION_CAPTIONS[state.selectedPerturbation];
  document.querySelector("[data-severity-label]").textContent = `${Math.round(severity * 100)}%`;
}

function renderAllCharts() {
  drawSummaryChart();
  drawRobustnessChart();
  drawLossChart();
  renderProfile();
  renderLeaderboard();
  renderSeverityMetrics();
  renderHud();
}

function bindControls() {
  document.querySelectorAll(".optimizer-button").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedOptimizer = button.dataset.optimizer;
      document.querySelectorAll(".optimizer-button").forEach((item) => item.classList.toggle("active", item === button));
      renderAllCharts();
    });
  });

  document.querySelectorAll(".segment").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedPerturbation = button.dataset.perturbation;
      document.querySelectorAll(".segment").forEach((item) => {
        const active = item === button;
        item.classList.toggle("active", active);
        item.setAttribute("aria-selected", String(active));
      });
      renderAllCharts();
    });
  });

  document.getElementById("severityRange").addEventListener("input", (event) => {
    state.selectedSeverityIndex = Number(event.target.value);
    drawRobustnessChart();
    renderSeverityMetrics();
    renderHud();
  });
}

function setupCanvas(canvas) {
  const ctx = canvas.getContext("2d");
  let previousWidth = 0;
  let previousHeight = 0;
  let previousRatio = 0;
  const resize = () => {
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (width !== previousWidth || height !== previousHeight || ratio !== previousRatio) {
      canvas.width = Math.max(1, Math.floor(width * ratio));
      canvas.height = Math.max(1, Math.floor(height * ratio));
      previousWidth = width;
      previousHeight = height;
      previousRatio = ratio;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { width, height };
  };
  return { ctx, resize };
}

function nodePosition(node, width, height, time, mode = "hero") {
  const centerX = mode === "hero" ? width * 0.56 : width * 0.5;
  const centerY = mode === "hero" ? height * 0.48 : height * 0.48;
  const spreadX = mode === "hero" ? width * 0.78 : width * 0.76;
  const spreadY = mode === "hero" ? height * 0.72 : height * 0.68;
  const severity = SEVERITIES[state.selectedSeverityIndex];
  const noisePhase = node.phase + time * 0.00045;
  const chamberJitter = mode === "chamber" && state.selectedPerturbation === "feature_noise" ? severity * 82 : 0;
  const orbit = mode === "hero" ? node.degree_rank * 18 : node.degree_rank * 10;
  return {
    x: centerX + (node.x - 0.5) * spreadX + Math.sin(noisePhase) * (orbit + chamberJitter),
    y: centerY + (node.y - 0.5) * spreadY + Math.cos(noisePhase * 1.35) * (orbit * 0.7 + chamberJitter * 0.6),
  };
}

function edgeHash(source, target) {
  const value = Math.sin(source * 12.9898 + target * 78.233) * 43758.5453;
  return value - Math.floor(value);
}

function paintBackground(ctx, width, height, time) {
  ctx.clearRect(0, 0, width, height);
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "#050609");
  gradient.addColorStop(0.48, "#071019");
  gradient.addColorStop(1, "#050507");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  ctx.save();
  ctx.globalAlpha = 0.16;
  ctx.strokeStyle = "#f5efe2";
  ctx.lineWidth = 1;
  for (let x = ((time * 0.006) % 90) - 90; x < width; x += 90) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x + height * 0.2, height);
    ctx.stroke();
  }
  for (let y = 0; y < height; y += 90) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawHeroScene(ctx, width, height, time) {
  paintBackground(ctx, width, height, time);
  const positions = new Map();
  state.graph.nodes.forEach((node) => positions.set(node.id, nodePosition(node, width, height, time, "hero")));

  const sweep = (time * 0.00018) % (Math.PI * 2);
  const center = { x: width * 0.58, y: height * 0.47 };
  ctx.save();
  ctx.translate(center.x, center.y);
  ctx.rotate(sweep);
  const sweepGradient = ctx.createLinearGradient(0, 0, width * 0.5, 0);
  sweepGradient.addColorStop(0, "rgba(101,255,212,0)");
  sweepGradient.addColorStop(1, "rgba(101,255,212,0.18)");
  ctx.fillStyle = sweepGradient;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.arc(0, 0, Math.max(width, height) * 0.62, -0.08, 0.08);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  state.graph.edges.forEach((edge) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) return;
    ctx.strokeStyle = "rgba(143,199,255,0.095)";
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.moveTo(source.x, source.y);
    ctx.lineTo(target.x, target.y);
    ctx.stroke();
  });

  let nearest = null;
  let nearestDistance = Infinity;
  state.graph.nodes.forEach((node) => {
    const point = positions.get(node.id);
    const distance = Math.hypot(point.x - state.pointer.x, point.y - state.pointer.y);
    if (distance < nearestDistance) {
      nearest = node;
      nearestDistance = distance;
    }
  });
  const highlightedId = nearestDistance < 42 ? nearest.id : null;

  state.graph.nodes.forEach((node) => {
    const point = positions.get(node.id);
    const active = node.id === highlightedId;
    ctx.globalAlpha = active ? 1 : 0.78;
    ctx.fillStyle = CLASS_COLORS[node.class_id % CLASS_COLORS.length];
    ctx.beginPath();
    ctx.arc(point.x, point.y, active ? 6 : 2.1 + node.degree_rank * 3.4, 0, Math.PI * 2);
    ctx.fill();
    if (active) {
      ctx.globalAlpha = 1;
      ctx.strokeStyle = "rgba(245,239,226,0.9)";
      ctx.lineWidth = 1.4;
      ctx.stroke();
      ctx.fillStyle = "#f5efe2";
      ctx.font = "12px Inter, system-ui, sans-serif";
      ctx.fillText(`Cora node ${node.original_id} | class ${node.class_id}`, point.x + 11, point.y - 12);
    }
  });
  ctx.globalAlpha = 1;
}

function drawPerturbationScene(ctx, width, height, time) {
  paintBackground(ctx, width, height, time * 0.5);
  const severity = SEVERITIES[state.selectedSeverityIndex];
  const positions = new Map();
  state.graph.nodes.forEach((node) => positions.set(node.id, nodePosition(node, width, height, time, "chamber")));

  ctx.save();
  ctx.translate(width * 0.5, height * 0.48);
  ctx.strokeStyle = "rgba(245,200,75,0.18)";
  ctx.lineWidth = 1;
  for (let r = 80; r < Math.min(width, height) * 0.52; r += 70) {
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();

  state.graph.edges.forEach((edge) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) return;
    const h = edgeHash(edge.source, edge.target);
    const removed = state.selectedPerturbation === "edge_removal" && h < severity;
    ctx.strokeStyle = removed ? "rgba(255,123,110,0.13)" : "rgba(143,199,255,0.13)";
    ctx.lineWidth = removed ? 0.6 : 0.9;
    ctx.setLineDash(removed ? [2, 8] : []);
    ctx.beginPath();
    ctx.moveTo(source.x, source.y);
    ctx.lineTo(target.x, target.y);
    ctx.stroke();
  });
  ctx.setLineDash([]);

  if (state.selectedPerturbation === "fake_edge_addition") {
    const fakeCount = Math.floor(45 + severity * 260);
    ctx.strokeStyle = "rgba(255,123,110,0.36)";
    ctx.lineWidth = 1.1;
    ctx.setLineDash([6, 9]);
    for (let i = 0; i < fakeCount; i += 1) {
      const a = state.graph.nodes[(i * 17) % state.graph.nodes.length];
      const b = state.graph.nodes[(i * 43 + 7) % state.graph.nodes.length];
      const source = positions.get(a.id);
      const target = positions.get(b.id);
      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  state.graph.nodes.forEach((node) => {
    const point = positions.get(node.id);
    const pulse = state.selectedPerturbation === "feature_noise" ? Math.sin(time * 0.004 + node.phase) * severity * 4 : 0;
    ctx.fillStyle = CLASS_COLORS[node.class_id % CLASS_COLORS.length];
    ctx.globalAlpha = 0.72;
    ctx.beginPath();
    ctx.arc(point.x, point.y, 2.2 + node.degree_rank * 3.4 + pulse, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.globalAlpha = 1;
}

function startCanvasAnimation() {
  const heroCanvas = document.getElementById("heroCanvas");
  const perturbationCanvas = document.getElementById("perturbationCanvas");
  const hero = setupCanvas(heroCanvas);
  const chamber = setupCanvas(perturbationCanvas);
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  heroCanvas.addEventListener("pointermove", (event) => {
    const rect = heroCanvas.getBoundingClientRect();
    state.pointer = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  });
  heroCanvas.addEventListener("pointerleave", () => {
    state.pointer = { x: -9999, y: -9999 };
  });

  function frame(time = 0) {
    const heroSize = hero.resize();
    const chamberSize = chamber.resize();
    drawHeroScene(hero.ctx, heroSize.width, heroSize.height, reducedMotion ? 0 : time);
    drawPerturbationScene(chamber.ctx, chamberSize.width, chamberSize.height, reducedMotion ? 0 : time);
    if (!reducedMotion) requestAnimationFrame(frame);
  }

  window.addEventListener("resize", () => {
    renderAllCharts();
  });
  frame();
}

function showError(error) {
  document.body.innerHTML = `
    <main class="loading-error">
      <h1>Frontend data is incomplete.</h1>
      <p>${error.message}</p>
      <p>The static data bundle in <code>docs/assets/data/</code> must include the final result tables and graph sample.</p>
    </main>
  `;
}

loadData()
  .then(() => {
    renderStats();
    bindControls();
    renderAllCharts();
    startCanvasAnimation();
  })
  .catch(showError);
