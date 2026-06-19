const DATA_DIR = "./assets/data/";

const OPTIMIZERS = ["Adam", "AdamW", "RMSProp", "AdaGrad", "SGD"];
const PERTURBATIONS = ["clean", "feature_noise", "edge_removal", "fake_edge_addition"];
const SEVERITIES = [0.05, 0.1, 0.2, 0.3];

const PERTURBATION_LABELS = {
  clean: "Graphe propre",
  feature_noise: "Bruit sur attributs",
  edge_removal: "Suppression d'arêtes",
  fake_edge_addition: "Ajout de fausses arêtes",
};

const PERTURBATION_AXIS_LABELS = {
  clean: "Propre",
  feature_noise: "Bruit attributs",
  edge_removal: "Arêtes supprimées",
  fake_edge_addition: "Fausses arêtes",
};

const PERTURBATION_CAPTIONS = {
  feature_noise: "Perturbation gaussienne des attributs",
  edge_removal: "Citations retirées aléatoirement",
  fake_edge_addition: "Citations artificielles injectées",
};

const OPTIMIZER_COLORS = {
  Adam: "#8f1d2c",
  AdamW: "#0f7c80",
  RMSProp: "#4b5c96",
  AdaGrad: "#b7791f",
  SGD: "#6b7280",
};

const CLASS_COLORS = ["#8f1d2c", "#0f7c80", "#4b5c96", "#b7791f", "#7357a6", "#243043", "#47796b"];

const state = {
  selectedOptimizer: "Adam",
  selectedPerturbation: "feature_noise",
  selectedSeverityIndex: 0,
  allResults: [],
  summary: [],
  lossHistory: [],
  dataset: {},
  graph: { nodes: [], edges: [] },
  pointers: {
    hero: { x: -9999, y: -9999 },
    chamber: { x: -9999, y: -9999 },
  },
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
    throw new Error(`Fichier de données introuvable : ${path}`);
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
      if (!response.ok) throw new Error("Échantillon du graphe Cora introuvable");
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
  return Number(value).toLocaleString("fr-FR");
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

function drawLegend(svg, x, y, gap = 118) {
  OPTIMIZERS.forEach((optimizer, index) => {
    const group = svgElement("g", { transform: `translate(${x + index * gap}, ${y})` });
    group.appendChild(svgElement("circle", { cx: 0, cy: 0, r: 5, fill: OPTIMIZER_COLORS[optimizer] }));
    const label = svgElement("text", { x: 12, y: 4, "font-size": 12 });
    label.textContent = optimizer;
    group.appendChild(label);
    svg.appendChild(group);
  });
}

function drawSummaryChart() {
  const svg = document.getElementById("summaryChart");
  const { width, height } = clearSvg(svg, 960, 450);
  const bounds = { left: 70, right: width - 30, top: 34, bottom: height - 78 };
  const xStep = (bounds.right - bounds.left) / (PERTURBATIONS.length - 1);
  const xByPerturbation = Object.fromEntries(PERTURBATIONS.map((perturbation, index) => [perturbation, bounds.left + xStep * index]));
  const yMin = 0.05;
  const yMax = 0.84;
  const yScale = (value) => bounds.bottom - ((value - yMin) / (yMax - yMin)) * (bounds.bottom - bounds.top);

  PERTURBATIONS.forEach((perturbation, index) => {
    if (index % 2 === 0) {
      const x = Math.max(bounds.left, xByPerturbation[perturbation] - xStep / 2);
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
      "stroke-width": selected ? 4.6 : 2.1,
      "stroke-opacity": selected ? 1 : 0.34,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
    }));

    points.forEach((point) => {
      svg.appendChild(svgElement("circle", {
        cx: point.x,
        cy: point.y,
        r: selected ? 6 : 4,
        fill: OPTIMIZER_COLORS[optimizer],
        "fill-opacity": selected ? 1 : 0.6,
      }));
    });

    if (selected) {
      const last = points[points.length - 1];
      drawText(svg, `${optimizer} ${formatPct(last.value)}`, last.x - 8, last.y - 14, {
        "font-size": 13,
        "font-weight": 900,
        "text-anchor": "end",
        fill: OPTIMIZER_COLORS[optimizer],
      });
    }
  });

  drawLegend(svg, bounds.left, height - 28);
}

function drawRobustnessChart() {
  const svg = document.getElementById("robustnessChart");
  const { width, height } = clearSvg(svg, 960, 470);
  const rows = state.allResults.filter((row) => row.perturbation_type === state.selectedPerturbation);
  const bounds = { left: 70, right: width - 30, top: 34, bottom: height - 76 };
  const xStep = (bounds.right - bounds.left) / (SEVERITIES.length - 1);
  const xBySeverity = Object.fromEntries(SEVERITIES.map((severity, index) => [severity, bounds.left + xStep * index]));
  const yMin = 0.05;
  const yMax = 0.84;
  const yScale = (value) => bounds.bottom - ((value - yMin) / (yMax - yMin)) * (bounds.bottom - bounds.top);
  const selectedSeverity = SEVERITIES[state.selectedSeverityIndex];

  document.querySelector("[data-perturbation-title]").textContent =
    `Robustesse : ${PERTURBATION_LABELS[state.selectedPerturbation]}`;

  drawChartFrame(
    svg,
    bounds,
    [0.2, 0.4, 0.6, 0.8],
    yScale,
    SEVERITIES.map((severity) => ({ label: `${Math.round(severity * 100)}%`, x: xBySeverity[severity] })),
  );

  svg.appendChild(svgElement("line", {
    x1: xBySeverity[selectedSeverity],
    x2: xBySeverity[selectedSeverity],
    y1: bounds.top,
    y2: bounds.bottom,
    stroke: "rgba(15, 124, 128, 0.5)",
    "stroke-width": 1.5,
    "stroke-dasharray": "6 8",
  }));

  OPTIMIZERS.forEach((optimizer) => {
    const points = SEVERITIES.map((severity) => {
      const row = rows.find((item) => item.optimizer === optimizer && item.severity === severity);
      return { x: xBySeverity[severity], y: yScale(row.test_accuracy), value: row.test_accuracy };
    });
    const selected = optimizer === state.selectedOptimizer;
    svg.appendChild(svgElement("path", {
      d: linePath(points),
      fill: "none",
      stroke: OPTIMIZER_COLORS[optimizer],
      "stroke-width": selected ? 4.2 : 2,
      "stroke-opacity": selected ? 1 : 0.34,
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
  const { width, height } = clearSvg(svg, 960, 470);
  const bounds = { left: 70, right: width - 30, top: 34, bottom: height - 76 };
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
      "stroke-opacity": selected ? 1 : 0.34,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
    }));
  });

  drawLegend(svg, bounds.left, height - 28);
}

function drawDropMatrixChart() {
  const svg = document.getElementById("dropMatrixChart");
  const { width, height } = clearSvg(svg, 1020, 520);
  const columns = ["feature_noise", "edge_removal", "fake_edge_addition"].flatMap((perturbation) =>
    SEVERITIES.map((severity) => ({ perturbation, severity })),
  );
  const bounds = { left: 116, right: width - 34, top: 88, bottom: height - 74 };
  const cellW = (bounds.right - bounds.left) / columns.length;
  const cellH = (bounds.bottom - bounds.top) / OPTIMIZERS.length;
  const maxDrop = 0.32;

  const groups = [
    { label: "Bruit attributs", start: 0, end: 3 },
    { label: "Arêtes supprimées", start: 4, end: 7 },
    { label: "Fausses arêtes", start: 8, end: 11 },
  ];

  groups.forEach((group) => {
    const x = bounds.left + group.start * cellW;
    const w = (group.end - group.start + 1) * cellW;
    svg.appendChild(svgElement("rect", {
      x,
      y: 38,
      width: w - 6,
      height: 28,
      rx: 4,
      fill: group.label === "Arêtes supprimées" ? "rgba(15,124,128,0.08)" : "rgba(143,29,44,0.08)",
      stroke: "rgba(36,48,67,0.12)",
    }));
    drawText(svg, group.label, x + w / 2 - 3, 57, {
      "font-size": 12,
      "font-weight": 900,
      "text-anchor": "middle",
      fill: "#253044",
    });
  });

  columns.forEach((column, index) => {
    const x = bounds.left + index * cellW + cellW / 2;
    drawText(svg, `${Math.round(column.severity * 100)}%`, x, bounds.top - 12, {
      "font-size": 11,
      "text-anchor": "middle",
    });
  });

  OPTIMIZERS.forEach((optimizer, rowIndex) => {
    const y = bounds.top + rowIndex * cellH;
    drawText(svg, optimizer, bounds.left - 18, y + cellH / 2 + 5, {
      "font-size": 13,
      "font-weight": 900,
      "text-anchor": "end",
      fill: OPTIMIZER_COLORS[optimizer],
    });

    columns.forEach((column, colIndex) => {
      const result = state.allResults.find(
        (item) =>
          item.optimizer === optimizer &&
          item.perturbation_type === column.perturbation &&
          item.severity === column.severity,
      );
      const drop = getCleanAccuracy(optimizer) - result.test_accuracy;
      const positive = Math.max(0, drop);
      const intensity = Math.min(1, positive / maxDrop);
      const negative = Math.max(0, -drop);
      const x = bounds.left + colIndex * cellW;
      const fill = negative > 0.002
        ? `rgba(15, 124, 128, ${0.1 + Math.min(0.75, negative * 8)})`
        : `rgba(143, 29, 44, ${0.04 + intensity * 0.82})`;
      const textColor = intensity > 0.55 ? "#ffffff" : "#253044";

      svg.appendChild(svgElement("rect", {
        x: x + 2,
        y: y + 2,
        width: cellW - 4,
        height: cellH - 4,
        rx: 5,
        fill,
        stroke: optimizer === state.selectedOptimizer ? OPTIMIZER_COLORS[optimizer] : "rgba(36,48,67,0.12)",
        "stroke-width": optimizer === state.selectedOptimizer ? 2 : 1,
      }));
      drawText(svg, `${(drop * 100).toFixed(1)}`, x + cellW / 2, y + cellH / 2 + 4, {
        "font-size": 11,
        "font-weight": 850,
        "text-anchor": "middle",
        fill: textColor,
      });
    });
  });

  drawText(svg, "points perdus vs précision propre", bounds.left, height - 30, {
    "font-size": 12,
    "font-weight": 800,
    fill: "#617083",
  });
  drawText(svg, "les cellules teal indiquent un léger gain vs propre", bounds.right, height - 30, {
    "font-size": 12,
    "text-anchor": "end",
    fill: "#617083",
  });
}

function renderProfile() {
  const optimizer = state.selectedOptimizer;
  const rows = state.summary.filter((row) => row.optimizer === optimizer);
  const meanAccuracy = getOptimizerAverage(optimizer);
  document.querySelector('[data-profile="name"]').textContent = optimizer;
  document.querySelector('[data-profile="score"]').textContent = formatPct(meanAccuracy);
  const dot = document.querySelector('[data-profile="dot"]');
  dot.style.background = OPTIMIZER_COLORS[optimizer];

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
  drawDropMatrixChart();
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

function bindCanvasPointer(canvas, key, target = canvas) {
  target.addEventListener("pointermove", (event) => {
    const rect = canvas.getBoundingClientRect();
    state.pointers[key] = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  });
  target.addEventListener("pointerleave", () => {
    state.pointers[key] = { x: -9999, y: -9999 };
  });
}

function nodePosition(node, width, height, time, mode = "hero") {
  const centerX = mode === "hero" ? width * 0.58 : width * 0.5;
  const centerY = mode === "hero" ? height * 0.52 : height * 0.48;
  const spreadX = mode === "hero" ? width * 0.72 : width * 0.76;
  const spreadY = mode === "hero" ? height * 0.62 : height * 0.68;
  const severity = SEVERITIES[state.selectedSeverityIndex];
  const noisePhase = node.phase + time * 0.00045;
  const chamberJitter = mode === "chamber" && state.selectedPerturbation === "feature_noise" ? severity * 82 : 0;
  const orbit = mode === "hero" ? node.degree_rank * 15 : node.degree_rank * 10;
  return {
    x: centerX + (node.x - 0.5) * spreadX + Math.sin(noisePhase) * (orbit + chamberJitter),
    y: centerY + (node.y - 0.5) * spreadY + Math.cos(noisePhase * 1.35) * (orbit * 0.7 + chamberJitter * 0.6),
  };
}

function applyPointerField(point, pointer, radius, power) {
  const dx = point.x - pointer.x;
  const dy = point.y - pointer.y;
  const distance = Math.hypot(dx, dy);
  if (distance <= 0 || distance > radius) return point;
  const force = (1 - distance / radius) * power;
  return {
    x: point.x + (dx / distance) * force,
    y: point.y + (dy / distance) * force,
  };
}

function pointerIsActive(pointer, width, height) {
  return pointer.x > -80 && pointer.y > -80 && pointer.x < width + 80 && pointer.y < height + 80;
}

function applyTangleField(point, pointer, width, height, time, radius, power, twist = 1) {
  if (!pointerIsActive(pointer, width, height)) return point;
  const dx = point.x - pointer.x;
  const dy = point.y - pointer.y;
  const distance = Math.hypot(dx, dy);
  if (distance <= 0.001 || distance > radius) return point;

  const influence = 1 - distance / radius;
  const eased = influence * influence * (3 - 2 * influence);
  const unitX = dx / distance;
  const unitY = dy / distance;
  const tangentX = -unitY;
  const tangentY = unitX;
  const pulse = Math.sin(time * 0.003 + distance * 0.045) * 0.22 + 0.88;
  const spin = power * eased * pulse * twist;
  const lensPull = -power * 0.28 * eased;

  return {
    x: point.x + tangentX * spin + unitX * lensPull,
    y: point.y + tangentY * spin + unitY * lensPull,
  };
}

function edgeHash(source, target) {
  const value = Math.sin(source * 12.9898 + target * 78.233) * 43758.5453;
  return value - Math.floor(value);
}

function paintBackground(ctx, width, height, time) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  ctx.save();
  ctx.lineWidth = 1;
  for (let x = 0; x < width; x += 44) {
    ctx.strokeStyle = x % 220 === 0 ? "rgba(15,124,128,0.14)" : "rgba(36,48,67,0.075)";
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y < height; y += 44) {
    ctx.strokeStyle = y % 220 === 0 ? "rgba(143,29,44,0.11)" : "rgba(36,48,67,0.065)";
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  const scanX = (time * 0.045) % (width + 180) - 90;
  ctx.strokeStyle = "rgba(15,124,128,0.34)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(scanX, 0);
  ctx.lineTo(scanX + height * 0.18, height);
  ctx.stroke();
  ctx.restore();
}

function drawTangleField(ctx, pointer, width, height, time, intensity = 1) {
  if (!pointerIsActive(pointer, width, height)) return;

  ctx.save();
  const radius = 190 * intensity;
  const gradient = ctx.createRadialGradient(pointer.x, pointer.y, 8, pointer.x, pointer.y, radius);
  gradient.addColorStop(0, "rgba(15,124,128,0.28)");
  gradient.addColorStop(0.42, "rgba(143,29,44,0.12)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(pointer.x, pointer.y, radius, 0, Math.PI * 2);
  ctx.fill();

  for (let i = 0; i < 5; i += 1) {
    const ring = 34 + i * 34 + Math.sin(time * 0.003 + i) * 5;
    ctx.strokeStyle = i % 2 === 0 ? "rgba(15,124,128,0.42)" : "rgba(143,29,44,0.36)";
    ctx.lineWidth = 1.45;
    ctx.setLineDash(i % 2 === 0 ? [10, 12] : [2, 10]);
    ctx.beginPath();
    ctx.arc(pointer.x, pointer.y, ring * intensity, time * 0.001 + i, Math.PI * 1.62 + time * 0.001 + i);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  for (let i = 0; i < 11; i += 1) {
    const angle = time * 0.0016 + i * ((Math.PI * 2) / 11);
    const inner = 28 * intensity;
    const outer = (130 + (i % 3) * 22) * intensity;
    const startX = pointer.x + Math.cos(angle) * inner;
    const startY = pointer.y + Math.sin(angle) * inner;
    const endX = pointer.x + Math.cos(angle + 0.78) * outer;
    const endY = pointer.y + Math.sin(angle + 0.78) * outer;
    const controlX = pointer.x + Math.cos(angle + 1.8) * outer * 0.62;
    const controlY = pointer.y + Math.sin(angle + 1.8) * outer * 0.62;
    ctx.strokeStyle = i % 2 === 0 ? "rgba(15,124,128,0.5)" : "rgba(143,29,44,0.42)";
    ctx.lineWidth = 1.55;
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.quadraticCurveTo(controlX, controlY, endX, endY);
    ctx.stroke();
  }
  ctx.restore();
}

function drawTangledEdge(ctx, source, target, pointer, width, height, time, options = {}) {
  const {
    baseColor = "rgba(36,48,67,0.16)",
    activeColor = "rgba(15,124,128,0.46)",
    lineWidth = 0.8,
    dash = [],
  } = options;
  const midpointX = (source.x + target.x) / 2;
  const midpointY = (source.y + target.y) / 2;
  const pointerActive = pointerIsActive(pointer, width, height);
  const distance = pointerActive ? Math.hypot(midpointX - pointer.x, midpointY - pointer.y) : Infinity;
  const influence = Math.max(0, 1 - distance / 270);
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const length = Math.max(1, Math.hypot(dx, dy));
  const normalX = -dy / length;
  const normalY = dx / length;
  const curve = influence * influence * (42 + Math.sin(time * 0.004 + length) * 13);
  const controlX = midpointX + normalX * curve + (pointer.x - midpointX) * influence * 0.16;
  const controlY = midpointY + normalY * curve + (pointer.y - midpointY) * influence * 0.16;

  ctx.setLineDash(dash);
  ctx.strokeStyle = influence > 0.02 ? activeColor : baseColor;
  ctx.lineWidth = lineWidth + influence * 1.2;
  ctx.beginPath();
  ctx.moveTo(source.x, source.y);
  if (influence > 0.02) {
    ctx.quadraticCurveTo(controlX, controlY, target.x, target.y);
  } else {
    ctx.lineTo(target.x, target.y);
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

function nearestNode(positions, pointer) {
  let nearest = null;
  let nearestDistance = Infinity;
  state.graph.nodes.forEach((node) => {
    const point = positions.get(node.id);
    const distance = Math.hypot(point.x - pointer.x, point.y - pointer.y);
    if (distance < nearestDistance) {
      nearest = node;
      nearestDistance = distance;
    }
  });
  return nearestDistance < 48 ? nearest : null;
}

function drawNodeLabel(ctx, node, point, color, widthLimit) {
  const label = `Nœud Cora ${node.original_id} | classe ${node.class_id} | degré ${node.degree}`;
  ctx.save();
  ctx.font = "12px Inter, system-ui, sans-serif";
  const width = ctx.measureText(label).width + 20;
  const x = Math.min(point.x + 14, widthLimit - width - 16);
  const y = Math.max(18, point.y - 28);
  ctx.fillStyle = "rgba(255,255,255,0.92)";
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, width, 28, 6);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#14213d";
  ctx.fillText(label, x + 10, y + 18);
  ctx.restore();
}

function drawHeroScene(ctx, width, height, time) {
  paintBackground(ctx, width, height, time);
  const pointer = state.pointers.hero;
  const positions = new Map();
  state.graph.nodes.forEach((node) => {
    const base = nodePosition(node, width, height, time, "hero");
    const repelled = applyPointerField(base, pointer, 210, 18);
    positions.set(node.id, applyTangleField(repelled, pointer, width, height, time, 330, 88, 1.05));
  });
  drawTangleField(ctx, pointer, width, height, time, 1.08);

  ctx.save();
  ctx.globalAlpha = 0.68;
  state.graph.edges.forEach((edge) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) return;
    drawTangledEdge(ctx, source, target, pointer, width, height, time, {
      baseColor: "rgba(36,48,67,0.14)",
      activeColor: "rgba(15,124,128,0.62)",
      lineWidth: 0.75,
    });
  });
  ctx.restore();

  const highlighted = nearestNode(positions, pointer);
  state.graph.nodes.forEach((node) => {
    const point = positions.get(node.id);
    const active = highlighted && node.id === highlighted.id;
    const color = CLASS_COLORS[node.class_id % CLASS_COLORS.length];
    ctx.globalAlpha = active ? 1 : 0.86;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(point.x, point.y, active ? 6.5 : 2 + node.degree_rank * 3.6, 0, Math.PI * 2);
    ctx.fill();
    if (active) {
      ctx.globalAlpha = 1;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();
      drawNodeLabel(ctx, node, point, color, width);
    }
  });
  ctx.globalAlpha = 1;

  if (pointer.x > -100) {
    ctx.save();
    ctx.strokeStyle = "rgba(15,124,128,0.38)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pointer.x - 18, pointer.y);
    ctx.lineTo(pointer.x + 18, pointer.y);
    ctx.moveTo(pointer.x, pointer.y - 18);
    ctx.lineTo(pointer.x, pointer.y + 18);
    ctx.stroke();
    ctx.restore();
  }
}

function drawPerturbationScene(ctx, width, height, time) {
  paintBackground(ctx, width, height, time * 0.5);
  const pointer = state.pointers.chamber;
  const severity = SEVERITIES[state.selectedSeverityIndex];
  const positions = new Map();
  state.graph.nodes.forEach((node) => {
    const base = nodePosition(node, width, height, time, "chamber");
    const repelled = applyPointerField(base, pointer, 180, 16 + severity * 22);
    positions.set(node.id, applyTangleField(repelled, pointer, width, height, time, 260, 48 + severity * 80, 0.88));
  });
  drawTangleField(ctx, pointer, width, height, time, 0.88 + severity * 0.8);

  ctx.save();
  ctx.translate(width * 0.5, height * 0.48);
  ctx.strokeStyle = "rgba(143,29,44,0.16)";
  ctx.lineWidth = 1;
  for (let r = 82; r < Math.min(width, height) * 0.52; r += 70) {
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
    drawTangledEdge(ctx, source, target, pointer, width, height, time, {
      baseColor: removed ? "rgba(143,29,44,0.25)" : "rgba(36,48,67,0.18)",
      activeColor: removed ? "rgba(143,29,44,0.48)" : "rgba(15,124,128,0.44)",
      lineWidth: removed ? 0.7 : 0.9,
      dash: removed ? [2, 8] : [],
    });
  });
  ctx.setLineDash([]);

  if (state.selectedPerturbation === "fake_edge_addition") {
    const fakeCount = Math.floor(45 + severity * 260);
    ctx.strokeStyle = "rgba(143,29,44,0.46)";
    ctx.lineWidth = 1.05;
    ctx.setLineDash([6, 9]);
    for (let i = 0; i < fakeCount; i += 1) {
      const a = state.graph.nodes[(i * 17) % state.graph.nodes.length];
      const b = state.graph.nodes[(i * 43 + 7) % state.graph.nodes.length];
      const source = positions.get(a.id);
      const target = positions.get(b.id);
      drawTangledEdge(ctx, source, target, pointer, width, height, time + i * 13, {
        baseColor: "rgba(143,29,44,0.42)",
        activeColor: "rgba(143,29,44,0.62)",
        lineWidth: 1.05,
        dash: [6, 9],
      });
    }
    ctx.setLineDash([]);
  }

  const highlighted = nearestNode(positions, pointer);
  state.graph.nodes.forEach((node) => {
    const point = positions.get(node.id);
    const color = CLASS_COLORS[node.class_id % CLASS_COLORS.length];
    const active = highlighted && node.id === highlighted.id;
    const pulse = state.selectedPerturbation === "feature_noise" ? Math.sin(time * 0.004 + node.phase) * severity * 4 : 0;
    ctx.fillStyle = color;
    ctx.globalAlpha = active ? 1 : 0.74;
    ctx.beginPath();
    ctx.arc(point.x, point.y, (active ? 6 : 2.2 + node.degree_rank * 3.4) + pulse, 0, Math.PI * 2);
    ctx.fill();
    if (active) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();
      drawNodeLabel(ctx, node, point, color, width);
    }
  });
  ctx.globalAlpha = 1;
}

function startCanvasAnimation() {
  const heroCanvas = document.getElementById("heroCanvas");
  const perturbationCanvas = document.getElementById("perturbationCanvas");
  const hero = setupCanvas(heroCanvas);
  const chamber = setupCanvas(perturbationCanvas);
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  bindCanvasPointer(heroCanvas, "hero", heroCanvas.closest(".hero"));
  bindCanvasPointer(perturbationCanvas, "chamber", perturbationCanvas.closest(".chamber-visual"));

  window.addEventListener("pointermove", (event) => {
    document.documentElement.style.setProperty("--mouse-x", `${event.clientX}px`);
    document.documentElement.style.setProperty("--mouse-y", `${event.clientY}px`);
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
      <h1>Les données du site sont incomplètes.</h1>
      <p>${error.message}</p>
      <p>Le dossier statique <code>docs/assets/data/</code> doit contenir les tableaux finaux et l'échantillon du graphe.</p>
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
