#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const artifactImport =
  process.env.ARTIFACT_TOOL_IMPORT ?? "@oai/artifact-tool";
const { Presentation, PresentationFile } = await import(artifactImport);

const __filename = fileURLToPath(import.meta.url);
const ROOT = path.resolve(path.dirname(__filename), "..");
const OUT = path.join(ROOT, "deliverables", "GCN_Robustness_Final_Defense.pptx");
const AGG = path.join(ROOT, "results", "v2", "aggregated", "v2_aggregated_summary.csv");
const STATS = path.join(ROOT, "results", "v2", "statistics", "optimizer_paired_comparisons.csv");
const QR_GITHUB = path.join(ROOT, "deliverables", "qr_github.png");
const QR_SITE = path.join(ROOT, "deliverables", "qr_website.png");

const W = 960;
const H = 540;
const COLORS = {
  bg: "#11161F",
  panel: "#171E2B",
  panel2: "#202938",
  ink: "#F5F7FB",
  muted: "#AEB8C8",
  red: "#D1394A",
  redDark: "#8F1D2C",
  teal: "#16A0A3",
  gold: "#D59A35",
  line: "#354052",
};
const OPTIMIZERS = ["Adam", "AdamW", "RMSProp", "AdaGrad", "SGD"];
const OPT_COLORS = {
  Adam: "#D1394A",
  AdamW: "#16A0A3",
  RMSProp: "#7C8ED8",
  AdaGrad: "#D59A35",
  SGD: "#9AA3B2",
};
const PERT_LABELS = {
  clean: "Propre",
  feature_masking: "Masquage",
  edge_removal: "Arêtes supprimées",
  fake_edge_addition: "Fausses arêtes",
};

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (ch === '"' && quoted && next === '"') {
      cell += '"';
      i += 1;
    } else if (ch === '"') {
      quoted = !quoted;
    } else if (ch === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((ch === "\n" || ch === "\r") && !quoted) {
      if (ch === "\r" && next === "\n") i += 1;
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += ch;
    }
  }
  if (cell || row.length) {
    row.push(cell);
    rows.push(row);
  }
  const headers = rows.shift() ?? [];
  return rows
    .filter((values) => values.length === headers.length)
    .map((values) =>
      Object.fromEntries(
        headers.map((header, index) => {
          const value = values[index];
          const numeric = Number(value);
          return [header, value !== "" && Number.isFinite(numeric) ? numeric : value];
        }),
      ),
    );
}

async function loadCsv(file) {
  try {
    return parseCsv(await fs.readFile(file, "utf8"));
  } catch {
    return [];
  }
}

function pct(value, digits = 1) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "--";
  return `${(numeric * 100).toFixed(digits)}%`;
}

function coreRows(rows) {
  return rows.filter(
    (row) =>
      row.dataset === "Cora" &&
      row.protocol === "fixed" &&
      row.robustness_setting === "training_time",
  );
}

function meanFor(rows, optimizer, perturbation, severity = null) {
  const filtered = rows.filter(
    (row) =>
      row.optimizer === optimizer &&
      row.perturbation_type === perturbation &&
      (severity === null || Number(row.requested_severity) === Number(severity)),
  );
  if (!filtered.length) return 0;
  const total = filtered.reduce((sum, row) => sum + Number(row.n_seeds || 0), 0);
  return (
    filtered.reduce(
      (sum, row) => sum + Number(row.mean_test_accuracy || 0) * Number(row.n_seeds || 0),
      0,
    ) / Math.max(1, total)
  );
}

function addText(slide, text, x, y, w, h, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: style.fontSize ?? 22,
    bold: style.bold ?? false,
    color: style.color ?? COLORS.ink,
    alignment: style.alignment ?? "left",
  };
  return shape;
}

function addPanel(slide, x, y, w, h, fill = COLORS.panel) {
  return slide.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: COLORS.line, width: 1 },
    borderRadius: 10,
  });
}

function decorate(slide, section = "Projet 13 / Option 4") {
  slide.background.fill = COLORS.bg;
  slide.shapes.add({
    geometry: "rect",
    position: { left: 0, top: 0, width: W, height: 10 },
    fill: COLORS.redDark,
    line: { style: "solid", fill: "none", width: 0 },
  });
  addText(slide, section, 42, 24, 360, 28, {
    fontSize: 11,
    bold: true,
    color: COLORS.teal,
  });
}

function title(slide, text, subtitle = "") {
  addText(slide, text, 42, 58, 760, 96, { fontSize: 34, bold: true });
  if (subtitle) addText(slide, subtitle, 44, 148, 760, 48, { fontSize: 16, color: COLORS.muted });
}

function bullets(slide, items, x, y, w, size = 18) {
  items.forEach((item, index) => {
    slide.shapes.add({
      geometry: "ellipse",
      position: { left: x, top: y + index * 48 + 9, width: 10, height: 10 },
      fill: index % 2 ? COLORS.teal : COLORS.red,
      line: { style: "solid", fill: "none", width: 0 },
    });
    addText(slide, item, x + 24, y + index * 48, w - 24, 36, {
      fontSize: size,
      color: COLORS.ink,
    });
  });
}

function metric(slide, label, value, x, y, w = 180) {
  addPanel(slide, x, y, w, 92, COLORS.panel2);
  addText(slide, label, x + 16, y + 14, w - 32, 22, {
    fontSize: 11,
    bold: true,
    color: COLORS.muted,
  });
  addText(slide, value, x + 16, y + 40, w - 32, 36, {
    fontSize: 26,
    bold: true,
    color: COLORS.ink,
  });
}

function addBarChart(slide, position, categories, values, fills, titleText) {
  slide.charts.add("bar", {
    position,
    title: titleText,
    titleTextStyle: { fontSize: 14, fill: COLORS.ink },
    categories,
    series: [
      {
        name: "Accuracy",
        values,
        fill: COLORS.red,
        points: fills.map((fill, idx) => ({ idx, fill })),
      },
    ],
    hasLegend: false,
    barOptions: { direction: "column", grouping: "clustered" },
    yAxis: { min: 0, max: 1, numberFormatCode: "0%" },
    dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.0%" },
    chartFill: COLORS.panel,
    plotAreaFill: COLORS.panel,
  });
}

function addLineChart(slide, position, categories, series, titleText) {
  slide.charts.add("line", {
    position,
    title: titleText,
    titleTextStyle: { fontSize: 14, fill: COLORS.ink },
    categories,
    series,
    hasLegend: true,
    legend: { position: "bottom" },
    yAxis: { min: 0, max: 1, numberFormatCode: "0%" },
    dataLabels: { showValue: false },
    chartFill: COLORS.panel,
    plotAreaFill: COLORS.panel,
  });
}

async function addQr(slide, file, label, x, y) {
  const bytes = await fs.readFile(file);
  slide.images.add({
    blob: bytes,
    contentType: "image/png",
    alt: label,
    fit: "contain",
    position: { left: x, top: y, width: 116, height: 116 },
  });
  addText(slide, label, x - 10, y + 122, 136, 24, {
    fontSize: 12,
    bold: true,
    alignment: "center",
    color: COLORS.muted,
  });
}

async function main() {
  const aggregate = await loadCsv(AGG);
  const stats = await loadCsv(STATS);
  const core = coreRows(aggregate);
  const clean = OPTIMIZERS.map((optimizer) => ({
    optimizer,
    value: meanFor(core, optimizer, "clean"),
  })).sort((a, b) => b.value - a.value);
  const aggregateRanking = OPTIMIZERS.map((optimizer) => ({
    optimizer,
    value:
      ["clean", "feature_masking", "edge_removal", "fake_edge_addition"].reduce(
        (sum, perturbation) => sum + meanFor(core, optimizer, perturbation),
        0,
      ) / 4,
  })).sort((a, b) => b.value - a.value);

  const presentation = Presentation.create({ slideSize: { width: W, height: H } });

  let slide = presentation.slides.add();
  decorate(slide);
  addText(slide, "Robustesse des GNN sous perturbations aléatoires", 42, 62, 740, 128, {
    fontSize: 40,
    bold: true,
  });
  addText(
    slide,
    "Adam vs AdamW vs RMSProp vs AdaGrad vs SGD sur Cora Citation Network",
    46,
    198,
    760,
    34,
    { fontSize: 18, color: COLORS.muted },
  );
  metric(slide, "Runs principaux", "650", 46, 310);
  metric(slide, "Graines", "10", 250, 310);
  metric(slide, "Époques/run", "200", 454, 310);
  await addQr(slide, QR_SITE, "Tableau de bord", 776, 300);

  slide = presentation.slides.add();
  decorate(slide);
  title(slide, "Contexte et question de recherche");
  bullets(slide, [
    "Les GNN dépendent fortement de la structure du graphe.",
    "Cora peut perdre des attributs actifs, des arêtes, ou recevoir de fausses arêtes.",
    "Question: quel optimiseur reste le plus stable sous un protocole fixe?",
  ], 58, 190, 760);

  slide = presentation.slides.add();
  decorate(slide);
  title(slide, "Cora et architecture GCN");
  metric(slide, "Noeuds", "2708", 58, 190);
  metric(slide, "Attributs", "1433", 272, 190);
  metric(slide, "Classes", "7", 486, 190);
  bullets(slide, [
    "GCN à deux couches, 16 canaux cachés, ReLU et dropout 0.5.",
    "Perte cross-entropy sur le masque d'entraînement.",
    "Même split et même architecture pour tous les optimiseurs.",
  ], 58, 330, 800, 16);

  slide = presentation.slides.add();
  decorate(slide);
  title(slide, "Optimiseurs comparés");
  bullets(slide, [
    "Adam et AdamW: méthodes adaptatives très utilisées en deep learning.",
    "RMSProp et AdaGrad: adaptation du pas par historique de gradients.",
    "SGD: référence non adaptative, interprétée uniquement dans ce protocole.",
  ], 58, 190, 820);

  slide = presentation.slides.add();
  decorate(slide);
  title(slide, "Protocole expérimental final");
  metric(slide, "Optimiseurs", "5", 58, 188);
  metric(slide, "Conditions", "13", 272, 188);
  metric(slide, "Graines", "10", 486, 188);
  metric(slide, "Runs", "650", 700, 188);
  bullets(slide, [
    "Budget fixe: 200 époques par entraînement.",
    "Les métriques sont moyennées avec écart-type et IC95.",
    "Le test n'est jamais utilisé pour choisir les hyperparamètres.",
  ], 58, 326, 820, 16);

  slide = presentation.slides.add();
  decorate(slide);
  title(slide, "Perturbations aléatoires");
  bullets(slide, [
    "Masquage aléatoire d’une proportion de caractéristiques actives non nulles.",
    "Edge removal: suppression de connexions non orientées uniques.",
    "Fake edge addition: ajout de paires non connectées, sans self-loop ni duplicat.",
  ], 58, 178, 820);

  slide = presentation.slides.add();
  decorate(slide);
  title(slide, "Garanties scientifiques");
  bullets(slide, [
    "Graines appariées pour comparer les optimiseurs.",
    "IC95 et bootstrap apparié pour les différences clés.",
    "Test de Wilcoxon et correction de Holm quand les données le permettent.",
    "Conclusion limitée au dataset, au GCN et au protocole étudiés.",
  ], 58, 170, 820, 17);

  slide = presentation.slides.add();
  decorate(slide);
  title(slide, "Performances sur graphe propre");
  addBarChart(
    slide,
    { left: 72, top: 170, width: 620, height: 290 },
    clean.map((row) => row.optimizer),
    clean.map((row) => row.value),
    clean.map((row) => OPT_COLORS[row.optimizer]),
    "Accuracy moyenne sur Cora propre",
  );
  addText(slide, `Meilleur score propre: ${clean[0]?.optimizer ?? "--"} (${pct(clean[0]?.value)})`, 720, 220, 190, 70, {
    fontSize: 22,
    bold: true,
  });
  addText(slide, "Les barres doivent être lues avec les intervalles du rapport.", 720, 306, 180, 70, {
    fontSize: 15,
    color: COLORS.muted,
  });

  slide = presentation.slides.add();
  decorate(slide);
  title(slide, "Robustesse agrégée");
  addBarChart(
    slide,
    { left: 72, top: 170, width: 640, height: 300 },
    aggregateRanking.map((row) => row.optimizer),
    aggregateRanking.map((row) => row.value),
    aggregateRanking.map((row) => OPT_COLORS[row.optimizer]),
    "Moyenne sur propre + perturbations",
  );
  addText(slide, "Adam et RMSProp restent proches: les petites différences ne sont pas appelées dominance.", 728, 220, 178, 120, {
    fontSize: 16,
    color: COLORS.muted,
  });

  slide = presentation.slides.add();
  decorate(slide);
  title(slide, "Robustesse au masquage d'attributs");
  addLineChart(
    slide,
    { left: 72, top: 162, width: 760, height: 310 },
    ["5%", "10%", "20%", "30%"],
    OPTIMIZERS.map((optimizer) => ({
      name: optimizer,
      values: [0.05, 0.1, 0.2, 0.3].map((severity) =>
        meanFor(core, optimizer, "feature_masking", severity),
      ),
      line: { fill: OPT_COLORS[optimizer], width: 2 },
      marker: { symbol: "circle", size: 5 },
    })),
    "Accuracy moyenne par taux de masquage",
  );

  slide = presentation.slides.add();
  decorate(slide);
  title(slide, "Robustesse structurelle");
  addLineChart(
    slide,
    { left: 56, top: 150, width: 410, height: 310 },
    ["5%", "10%", "20%", "30%"],
    OPTIMIZERS.map((optimizer) => ({
      name: optimizer,
      values: [0.05, 0.1, 0.2, 0.3].map((severity) =>
        meanFor(core, optimizer, "edge_removal", severity),
      ),
      line: { fill: OPT_COLORS[optimizer], width: 2 },
    })),
    "Suppression d'arêtes",
  );
  addLineChart(
    slide,
    { left: 500, top: 150, width: 410, height: 310 },
    ["5%", "10%", "20%", "30%"],
    OPTIMIZERS.map((optimizer) => ({
      name: optimizer,
      values: [0.05, 0.1, 0.2, 0.3].map((severity) =>
        meanFor(core, optimizer, "fake_edge_addition", severity),
      ),
      line: { fill: OPT_COLORS[optimizer], width: 2 },
    })),
    "Ajout de fausses arêtes",
  );

  slide = presentation.slides.add();
  decorate(slide);
  title(slide, "Validation complémentaire");
  metric(slide, "Cross-dataset", String(aggregate.filter((r) => r.dataset !== "Cora").length ? "300" : "en attente"), 62, 180, 220);
  metric(slide, "Protocole tuné", String(aggregate.some((r) => r.protocol === "tuned") ? "100" : "en attente"), 370, 180, 220);
  metric(slide, "Inférence", String(aggregate.some((r) => r.robustness_setting === "inference_time") ? "75" : "en attente"), 678, 180, 220);
  bullets(slide, [
    "Ces validations ne remplacent pas le coeur Cora: elles testent la stabilité des conclusions.",
    "Les résultats absents ne sont pas inventés; ils restent marqués comme en attente.",
  ], 74, 340, 780, 16);

  slide = presentation.slides.add();
  decorate(slide);
  title(slide, "Limites et perspectives");
  bullets(slide, [
    "Perturbations aléatoires, pas d'attaques optimisées.",
    "Benchmark contrôlé: Cora, GCN à deux couches, hyperparamètres fixes.",
    "Les temps CPU dépendent du matériel.",
    "Suite: autres architectures, attaques réelles, datasets plus variés.",
  ], 58, 170, 820, 17);

  slide = presentation.slides.add();
  decorate(slide);
  title(slide, "Conclusion + démonstration dashboard");
  addText(slide, "Conclusion prudente", 58, 170, 300, 34, { fontSize: 22, bold: true, color: COLORS.teal });
  bullets(slide, [
    "Les optimiseurs adaptatifs sont les plus stables dans ce protocole.",
    "Adam et RMSProp doivent être comparés avec les intervalles.",
    "Le dashboard donne accès aux résultats bruts et aux artefacts.",
  ], 58, 220, 520, 16);
  await addQr(slide, QR_SITE, "Tableau de bord", 660, 190);
  await addQr(slide, QR_GITHUB, "GitHub", 800, 190);
  if (stats.length) {
    addText(slide, "Les comparaisons statistiques restent prudentes et doivent être lues avec les IC95.", 660, 350, 240, 74, {
      fontSize: 14,
      color: COLORS.muted,
    });
  }

  await fs.mkdir(path.dirname(OUT), { recursive: true });
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUT);
  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(
    path.join(ROOT, "deliverables", "GCN_Robustness_Final_Defense_montage.webp"),
    new Uint8Array(await montage.arrayBuffer()),
  );
  console.log(`Wrote ${OUT}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
