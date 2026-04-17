import React, { useMemo, useState } from "react";
import Plot from "./Plot";

/**
 * Protein-style COT Folding Visualization (v3 — refined aesthetics)
 *
 * Split into two vertically stacked plots:
 *   Top (70%):  Folded structure (MDS 2D) with equal aspect ratio
 *   Bottom (30%): Sequence annotation tracks (HMM state, entropy, confidence)
 */

const EXPLORE = "#5B8DEF";  // softer blue
const EXPLOIT = "#E05A47";  // softer red
const BOND_COLOR = "rgba(220,180,60,0.25)";
const BG = "transparent";

const FLOW_COLORS = {
  arterial: "#FF8A65",
  venous: "#7E57C2",
  capillary: "#FFD54F",
  shunt: "#9E9E9E",
};

const FUNC_COLORS = {
  core: "#4CAF50",
  closure: "#FF9800",
  drift: "#F44336",
  return_site: "#AB47BC",
  productive: "#90A4AE",
  catalytic_bond: "#FFD700",
};

function FoldingArcDiagram({ data, onColorModeChange, colorMode = "entropy", decodedSimilarity = null, flowData = null, functionalData = null, onSliceClick, compact = false, miniature = false, focusMode = "global", answerIsland = null, hoveredSlice = null }) {
  const { mds_coords, similarity_shape, hmm_states, entropy, confidence, metrics, mds_stress, effectiveness } = data;
  const n = similarity_shape[0];
  // Use pre-decoded similarity passed from parent (decoded once in useFoldingState)
  const similarity = decodedSimilarity || data.similarity || null;
  const threshold = metrics.contact_threshold;
  const longRange = Math.floor(n / 4);
  const unitLabel = data.unit_label || "slice";
  const unitTitle = unitLabel.charAt(0).toUpperCase() + unitLabel.slice(1);

  const hasEntropy = entropy.some((e) => e !== 0);
  const hasConf = confidence.some((c) => c !== 0);
  const hasFunctional = false;  // functional overlay disabled
  const hasFlow = false;        // flow overlay disabled
  const hasEffectiveness = effectiveness && effectiveness.scores;

  // Compute tail range directly from hmm_states — last contiguous exploit run
  const tailRange = useMemo(() => {
    if (focusMode !== "answer-tail") return null;
    if (!hmm_states || hmm_states.length < 2) return null;
    const len = hmm_states.length;
    if (hmm_states[len - 1] !== 1) return null;
    let start = len - 1;
    while (start > 0 && hmm_states[start - 1] === 1) start--;
    if (len - 1 - start < 1) return null;
    return { tailStart: start, tailEnd: len - 1 };
  }, [hmm_states, focusMode]);

  const isTailFocus = tailRange !== null;
  const isExFocus = false;

  // ── Normalize ──
  const entMin = Math.min(...entropy), entMax = Math.max(...entropy);
  const confMin = Math.min(...confidence), confMax = Math.max(...confidence);
  const entRange = entMax - entMin || 1e-8;
  const confRange = confMax - confMin || 1e-8;
  const entNorm = entropy.map((e) => (e - entMin) / entRange);
  const confNorm = confidence.map((c) => (c - confMin) / confRange);

  // ── All geometry computed once ──
  const geo = useMemo(() => {
    const xs = mds_coords.map((p) => p[0]);
    const ys = mds_coords.map((p) => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) || 1;
    const s = 1.0 / span;
    const mx = xs.map((x) => (x - cx) * s);
    const my = ys.map((y) => (y - cy) * s);

    // Backbone segments by HMM state
    const segs = [];
    let start = 0;
    for (let i = 1; i <= n; i++) {
      if (i === n || hmm_states[i] !== hmm_states[start]) {
        const end = Math.min(i - 1, n - 1);
        const sx = [], sy = [];
        for (let j = start; j <= end; j++) { sx.push(mx[j]); sy.push(my[j]); }
        segs.push({ xs: sx, ys: sy, state: hmm_states[start], start, end });
        start = i;
      }
    }

    // Long-range contacts (skip if similarity not yet loaded)
    const contacts = [];
    if (similarity) {
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          if (j - i <= longRange) continue;
          const sim = similarity[i * n + j];
          if (sim > threshold) contacts.push({ i, j, sim, gap: j - i });
        }
      }
      contacts.sort((a, b) => b.sim - a.sim);
    }

    return { mx, my, segs, contacts: contacts.slice(0, 50) };
  }, [mds_coords, similarity, hmm_states, n, threshold, longRange]);

  const { mx, my, segs, contacts } = geo;

  // ═══════════════════════════════════════════════
  //  TOP PLOT: Folded Structure
  // ═══════════════════════════════════════════════

  const structTraces = useMemo(() => {
    const traces = [];

    if (hasFunctional) {
      // --- Functional mode: replace bonds with catalytic/structural classification ---
      const returnEdges = functionalData.return_edges || [];

      // Catalytic contacts (cross-state: explore-exploit or exploit-explore) — gold
      const catX = [], catY = [];
      // Structural contacts (same-state) — light gray
      const strX = [], strY = [];

      for (const e of returnEdges) {
        const etype = e.type || "";
        const isCatalytic = etype.includes("explore") && etype.includes("exploit");
        if (e.i < n && e.j < n) {
          if (isCatalytic) {
            catX.push(mx[e.i], mx[e.j], null);
            catY.push(my[e.i], my[e.j], null);
          } else {
            strX.push(mx[e.i], mx[e.j], null);
            strY.push(my[e.i], my[e.j], null);
          }
        }
      }

      // Structural bonds (behind, subtle)
      if (strX.length > 0) {
        traces.push({
          x: strX, y: strY, mode: "lines",
          line: { color: "rgba(100,100,100,0.12)", width: 0.5 },
          hoverinfo: "skip", showlegend: false,
        });
      }
      // Catalytic bonds (prominent gold)
      if (catX.length > 0) {
        traces.push({
          x: catX, y: catY, mode: "lines",
          line: { color: FUNC_COLORS.catalytic_bond, width: 1.5 },
          opacity: 0.6,
          hoverinfo: "skip", showlegend: false,
        });
      }
    } else {
      // --- Default: original gold bonds ---
      const bx = [], by = [];
      for (const c of contacts) {
        bx.push(mx[c.i], mx[c.j], null);
        by.push(my[c.i], my[c.j], null);
      }
      traces.push({
        x: bx, y: by, mode: "lines",
        line: { color: BOND_COLOR, width: miniature ? 0.5 : 1 },
        hoverinfo: "skip", showlegend: false,
      });
    }

    const lw = miniature ? 0.45 : 1;  // miniature: thinner lines for Compare thumbnails

    if (colorMode === "effectiveness" && hasEffectiveness) {
      // Effectiveness mode: keep explore(blue)/exploit(red) backbone color
      // but vary line width by effectiveness + dash circling regions
      const scores = effectiveness.scores;
      const labels = effectiveness.labels;
      for (let i = 0; i < n - 1; i++) {
        const score = (scores[i] + scores[i + 1]) / 2;
        const isCircling = labels[i] === "circling";
        const isExplore = hmm_states[i] === 0;
        const baseColor = isExplore ? EXPLORE : EXPLOIT;
        // Low effectiveness = thin faded line; high = thick bold line
        const width = (isCircling ? 1.5 : 1.5 + score * 4) * lw;
        const alpha = isCircling ? 0.35 : 0.5 + score * 0.5;
        // Glow — explore/exploit color, width reflects effectiveness
        traces.push({
          x: [mx[i], mx[i + 1]], y: [my[i], my[i + 1]], mode: "lines",
          line: { color: isCircling ? "rgba(211,47,47,0.10)" : (isExplore ? `rgba(91,141,239,${alpha * 0.25})` : `rgba(224,90,71,${alpha * 0.25})`), width: (isCircling ? 18 : 6 + score * 6) * lw, shape: "spline" },
          hoverinfo: "skip", showlegend: false,
        });
        // Core line
        traces.push({
          x: [mx[i], mx[i + 1]], y: [my[i], my[i + 1]], mode: "lines",
          line: {
            color: isCircling ? "rgba(211,47,47,0.6)" : baseColor,
            width, shape: "spline",
            dash: isCircling ? "dot" : "solid",
          },
          hoverinfo: "skip", showlegend: false,
        });
      }
    } else {
      // Default backbone glow
      const glowWidth = (isExFocus ? 12 : 8) * lw;
      const coreWidth = (isExFocus ? 4 : 2.5) * lw;

      for (const seg of segs) {
        let glowAlpha = 0.2;
        if (isTailFocus) {
          const segOverlapsTail = seg.end >= tailRange.tailStart && seg.start <= tailRange.tailEnd;
          glowAlpha = segOverlapsTail ? 0.3 : 0.05;
        }
        traces.push({
          x: seg.xs, y: seg.ys, mode: "lines",
          line: { color: seg.state === 0 ? `rgba(91,141,239,${glowAlpha})` : `rgba(224,90,71,${glowAlpha})`, width: glowWidth, shape: "spline" },
          hoverinfo: "skip", showlegend: false,
        });
      }

      // Backbone core
      let shE = false, shX = false;
      for (const seg of segs) {
        const isE = seg.state === 0;
        let lineAlpha = 1;
        if (isTailFocus) {
          const segOverlapsTail = seg.end >= tailRange.tailStart && seg.start <= tailRange.tailEnd;
          lineAlpha = segOverlapsTail ? 1 : 0.15;
        }
        const baseColor = isE ? EXPLORE : EXPLOIT;
        const lineColor = lineAlpha < 1 ? (isE ? `rgba(91,141,239,${lineAlpha})` : `rgba(224,90,71,${lineAlpha})`) : baseColor;
        traces.push({
          x: seg.xs, y: seg.ys, mode: "lines",
          line: { color: lineColor, width: coreWidth, shape: "spline" },
          showlegend: isE ? !shE : !shX,
          name: isE ? "Explore" : "Exploit",
          legendgroup: isE ? "E" : "X",
          hoverinfo: "skip",
        });
        if (isE) shE = true; else shX = true;
      }
    }

    return traces;
  }, [mx, my, segs, contacts, hasFunctional, functionalData, n, colorMode, hasEffectiveness, effectiveness, isTailFocus, tailRange, miniature]);

  // Node trace (rebuilt when colorMode changes)
  const nodeTrace = useMemo(() => {
    const sz = miniature ? 0.45 : 1;  // miniature mode: shrink nodes for Compare thumbnails
    let sizes = hasEntropy ? entNorm.map((e) => (4 + 13 * e) * sz) : 6 * sz;
    let color, cscale, cbar, opacity;
    let markerLine = { color: "rgba(128,128,128,0.3)", width: 0.8 };

    if (colorMode === "effectiveness" && hasEffectiveness) {
      // Node size driven by effectiveness (bigger = more productive)
      sizes = effectiveness.scores.map((s) => (4 + 18 * s) * sz);
      color = effectiveness.scores;
      cscale = [[0, "#D32F2F"], [0.25, "#FF5722"], [0.5, "#FF9800"], [0.75, "#8BC34A"], [1, "#2E7D32"]];
      cbar = { title: { text: "Effective", font: { size: 10 } }, thickness: 10, len: 0.5, y: 0.5, outlinewidth: 0 };
      opacity = 0.9;
      // Circling = bold red border + X symbol; productive = green border
      const labels = effectiveness.labels || [];
      markerLine = {
        color: labels.map((l) => l === "circling" ? "#D32F2F" : l.includes("productive") ? "#2E7D32" : "rgba(200,200,200,0.6)"),
        width: labels.map((l) => l === "circling" ? 3 : l.includes("productive") ? 2 : 0.8),
      };
    } else if (colorMode === "entropy" && hasEntropy) {
      color = entropy;
      cscale = [[0, "#FFF7EC"], [0.3, "#FDD49E"], [0.6, "#F16913"], [1, "#8C2D04"]];
      cbar = { title: { text: "Entropy", font: { size: 10 } }, thickness: 10, len: 0.5, y: 0.5, outlinewidth: 0 };
      opacity = hasConf ? confNorm.map((c) => 0.4 + 0.6 * c) : 0.85;
    } else if (colorMode === "confidence" && hasConf) {
      color = confidence;
      cscale = [[0, "#EFF3FF"], [0.3, "#BDD7E7"], [0.6, "#3182BD"], [1, "#08519C"]];
      cbar = { title: { text: "Confidence", font: { size: 10 } }, thickness: 10, len: 0.5, y: 0.5, outlinewidth: 0 };
      opacity = 0.85;
    } else {
      color = hmm_states.map((s) => s === 0 ? EXPLORE : EXPLOIT);
      cscale = undefined;
      cbar = undefined;
      opacity = 0.85;
    }

    // Functional mode: override marker border to show role
    if (hasFunctional && colorMode !== "effectiveness") {
      const roles = functionalData.slice_roles || [];
      const borderColors = roles.map((r) => FUNC_COLORS[r] || "rgba(128,128,128,0.3)");
      const borderWidths = roles.map((r) => r === "core" || r === "closure" ? 2 : (r === "drift" ? 1.5 : 0.8));
      markerLine = { color: borderColors, width: borderWidths };
    }

    // Answer-tail focus: override opacity + add border to tail nodes
    if (isTailFocus) {
      opacity = Array.from({ length: n }, (_, i) =>
        i >= tailRange.tailStart && i <= tailRange.tailEnd ? 0.95 : 0.15
      );
      markerLine = {
        color: Array.from({ length: n }, (_, i) =>
          i >= tailRange.tailStart && i <= tailRange.tailEnd ? "#FF9800" : "rgba(128,128,128,0.1)"
        ),
        width: Array.from({ length: n }, (_, i) =>
          i >= tailRange.tailStart && i <= tailRange.tailEnd ? 2 : 0.5
        ),
      };
    }

    return {
      x: mx, y: my, mode: "markers",
      marker: { size: sizes, color, colorscale: cscale, colorbar: cbar, opacity, line: markerLine },
      customdata: Array.from({ length: n }, (_, i) => i),
      text: Array.from({ length: n }, (_, i) => {
        let txt = `<b>${unitTitle} ${i}</b><br>` +
          `${hmm_states[i] === 0 ? "Explore" : "Exploit"}<br>` +
          `Entropy: ${entropy[i].toFixed(3)}<br>` +
          `Confidence: ${confidence[i].toFixed(2)}`;
        if (hasEffectiveness) {
          txt += `<br>Effectiveness: ${(effectiveness.scores[i] * 100).toFixed(0)}%`;
          txt += `<br>Label: ${effectiveness.labels[i]}`;
        }
        if (hasFunctional && functionalData.slice_roles) {
          txt += `<br>Role: ${functionalData.slice_roles[i] || "unknown"}`;
        }
        return txt;
      }),
      hoverinfo: "text", showlegend: false,
    };
  }, [mx, my, entropy, confidence, hmm_states, entNorm, confNorm, colorMode, hasEntropy, hasConf, n, hasFunctional, functionalData, hasEffectiveness, effectiveness, isTailFocus, tailRange]);

  // ── Functional marker overlays (top plot) ──
  const funcMarkerTraces = useMemo(() => {
    if (!hasFunctional) return [];
    const traces = [];
    const roles = functionalData.slice_roles || [];

    // Drift slices: red × markers
    const driftX = [], driftY = [], driftText = [];
    // Closure slices: orange square border
    const closureX = [], closureY = [], closureText = [];
    // Return site slices: purple diamond
    const retX = [], retY = [], retText = [];

    for (let i = 0; i < n; i++) {
      if (roles[i] === "drift") {
        driftX.push(mx[i]); driftY.push(my[i]);
        driftText.push(`${unitTitle} ${i}: Drift`);
      } else if (roles[i] === "closure") {
        closureX.push(mx[i]); closureY.push(my[i]);
        closureText.push(`${unitTitle} ${i}: Closure`);
      } else if (roles[i] === "return_site") {
        retX.push(mx[i]); retY.push(my[i]);
        retText.push(`${unitTitle} ${i}: Return Site`);
      }
    }

    if (driftX.length > 0) {
      traces.push({
        x: driftX, y: driftY, mode: "markers",
        marker: { size: 8, color: "rgba(244,67,54,0.15)", symbol: "x", line: { color: FUNC_COLORS.drift, width: 1.5 } },
        text: driftText, hoverinfo: "text", showlegend: false,
      });
    }
    if (closureX.length > 0) {
      traces.push({
        x: closureX, y: closureY, mode: "markers",
        marker: { size: 10, color: "rgba(255,152,0,0.1)", symbol: "square-open", line: { color: FUNC_COLORS.closure, width: 1.5 } },
        text: closureText, hoverinfo: "text", showlegend: false,
      });
    }
    if (retX.length > 0) {
      traces.push({
        x: retX, y: retY, mode: "markers",
        marker: { size: 7, color: "rgba(171,71,188,0.2)", symbol: "diamond", line: { color: FUNC_COLORS.return_site, width: 1 } },
        text: retText, hoverinfo: "text", showlegend: false,
      });
    }

    return traces;
  }, [hasFunctional, functionalData, mx, my, n]);

  // ── Flow overlay traces (top plot) ──
  const { flowTraces, flowAnnotations } = useMemo(() => {
    if (!flowData) return { flowTraces: [], flowAnnotations: [] };

    const traces = [];
    const annotations = [];
    const ft = flowData.flow_type;
    const fm = flowData.flow_magnitude;
    const fv = flowData.flux_vectors;
    const nEdges = ft.length;

    // 5b-1: Flow backbone overlay — group consecutive same-type segments
    let segStart = 0;
    for (let i = 1; i <= nEdges; i++) {
      if (i === nEdges || ft[i] !== ft[segStart]) {
        const type = ft[segStart];
        const sx = [], sy = [];
        for (let j = segStart; j <= Math.min(i, n - 1); j++) {
          sx.push(mx[j]); sy.push(my[j]);
        }
        // Average magnitude for line width
        let avgMag = 0;
        for (let j = segStart; j < i; j++) avgMag += fm[j];
        avgMag /= (i - segStart);

        traces.push({
          x: sx, y: sy, mode: "lines",
          line: {
            color: FLOW_COLORS[type],
            width: 2 + avgMag * 12,
            shape: "spline",
            dash: type === "shunt" ? "dash" : "solid",
          },
          opacity: 0.7,
          hoverinfo: "skip", showlegend: false,
        });
        segStart = i;
      }
    }

    // 5b-2: Direction arrows — every step slices along backbone
    const step = Math.max(5, Math.floor(n / 12));
    for (let i = 0; i + step < n; i += step) {
      const j = i + 1; // arrow points forward
      if (i >= nEdges) break;
      annotations.push({
        x: mx[j], y: my[j],
        ax: mx[i], ay: my[i],
        xref: "x", yref: "y", axref: "x", ayref: "y",
        showarrow: true,
        arrowhead: 3, arrowsize: 1.2, arrowwidth: 1.8,
        arrowcolor: FLOW_COLORS[ft[i]] || "#999",
        opacity: 0.8,
      });
    }

    // 5b-3: Flux vectors — every 8 slices, gray arrows
    const fvStep = Math.max(8, Math.floor(n / 10));
    // flux_vectors is flat array of [dx, dy] pairs
    for (let i = 0; i < n; i += fvStep) {
      const fvi = i < fv.length / 2 ? [fv[i * 2], fv[i * 2 + 1]] : (Array.isArray(fv[i]) ? fv[i] : [0, 0]);
      // Handle both flat and nested array formats
      let dx, dy;
      if (Array.isArray(fv[0])) {
        // nested: [[dx, dy], ...]
        dx = fv[i] ? fv[i][0] : 0;
        dy = fv[i] ? fv[i][1] : 0;
      } else {
        // flat: [dx0, dy0, dx1, dy1, ...]
        dx = fv[i * 2] || 0;
        dy = fv[i * 2 + 1] || 0;
      }
      if (Math.abs(dx) < 0.001 && Math.abs(dy) < 0.001) continue;

      // Scale flux to MDS normalized coords
      const cx_ = (Math.min(...mx) + Math.max(...mx)) / 2;
      const cy_ = (Math.min(...my) + Math.max(...my)) / 2;
      const span_ = Math.max(Math.max(...mx) - Math.min(...mx), Math.max(...my) - Math.min(...my)) || 1;
      const scale = 1.0 / span_ * 0.8;

      annotations.push({
        x: mx[i] + dx * scale, y: my[i] + dy * scale,
        ax: mx[i], ay: my[i],
        xref: "x", yref: "y", axref: "x", ayref: "y",
        showarrow: true,
        arrowhead: 2, arrowsize: 0.8, arrowwidth: 1,
        arrowcolor: "rgba(120,120,120,0.4)",
      });
    }

    return { flowTraces: traces, flowAnnotations: annotations };
  }, [flowData, mx, my, n]);

  // Start / End markers
  const markerTraces = [
    {
      x: [mx[0]], y: [my[0]], mode: "markers+text",
      marker: { size: 11, color: "#2E7D32", symbol: "diamond", line: { color: "rgba(128,128,128,0.3)", width: 1.5 } },
      text: ["N"], textposition: "top center", textfont: { size: 9, color: "#2E7D32", family: "Arial Black" },
      hovertext: [`START (${unitLabel} 0)`], hoverinfo: "text", showlegend: false,
    },
    {
      x: [mx[n - 1]], y: [my[n - 1]], mode: "markers+text",
      marker: { size: 11, color: "#E65100", symbol: "square", line: { color: "rgba(128,128,128,0.3)", width: 1.5 } },
      text: ["C"], textposition: "top center", textfont: { size: 9, color: "#E65100", family: "Arial Black" },
      hovertext: [`END (${unitLabel} ${n - 1})`], hoverinfo: "text", showlegend: false,
    },
  ];

  // Hovered node highlight (from text panel hover) — bright glow + crosshair
  const hoverHighlightTraces = [];
  if (hoveredSlice != null && hoveredSlice >= 0 && hoveredSlice < n) {
    const hx = mx[hoveredSlice], hy = my[hoveredSlice];
    const hColor = "#FFD600";
    // Large glow disc
    hoverHighlightTraces.push({
      x: [hx], y: [hy], mode: "markers",
      marker: { size: 32, color: "rgba(255,214,0,0.25)", line: { color: hColor, width: 2.5 }, symbol: "circle" },
      hoverinfo: "skip", showlegend: false,
    });
    // Inner bright dot
    hoverHighlightTraces.push({
      x: [hx], y: [hy], mode: "markers",
      marker: { size: 12, color: hColor, line: { color: "#FF6F00", width: 2 }, symbol: "circle" },
      hoverinfo: "skip", showlegend: false,
    });
    // Crosshair lines
    const arm = 0.06;
    hoverHighlightTraces.push({
      x: [hx - arm, hx + arm, null, hx, hx], y: [hy, hy, null, hy - arm, hy + arm], mode: "lines",
      line: { color: hColor, width: 1.5, dash: "dot" },
      hoverinfo: "skip", showlegend: false,
    });
  }

  // Circling region highlight traces + annotations
  const { circlingTraces, circlingAnnotations } = useMemo(() => {
    if (!hasEffectiveness || colorMode !== "effectiveness") return { circlingTraces: [], circlingAnnotations: [] };
    const regions = effectiveness.circling_regions || [];
    const traces = [];
    const annotations = [];

    regions.forEach((region, idx) => {
      const [start, end] = region;
      const regionX = [];
      const regionY = [];
      for (let i = start; i < Math.min(end, n); i++) {
        regionX.push(mx[i]);
        regionY.push(my[i]);
      }
      if (regionX.length < 2) return;

      // Bold red glow behind circling region
      traces.push({
        x: regionX, y: regionY, mode: "lines",
        line: { color: "rgba(211,47,47,0.20)", width: 30, shape: "spline" },
        hoverinfo: "skip", showlegend: false,
      });
      // Inner dashed outline
      traces.push({
        x: regionX, y: regionY, mode: "lines",
        line: { color: "rgba(211,47,47,0.35)", width: 5, shape: "spline", dash: "dot" },
        hoverinfo: "skip", showlegend: false,
      });

      // Label annotation at the midpoint
      const midIdx = Math.floor(regionX.length / 2);
      annotations.push({
        x: regionX[midIdx], y: regionY[midIdx] + 0.06,
        text: `<b>Circling</b><br>${end - start} slices`,
        showarrow: true, arrowhead: 0, arrowcolor: "#D32F2F",
        ax: 0, ay: -30,
        font: { size: 10, color: "#D32F2F", family: "Arial" },
        bgcolor: "rgba(128,128,128,0.15)", bordercolor: "#D32F2F", borderwidth: 1,
        borderpad: 3,
      });
    });

    // Summary: productive fraction
    if (effectiveness.productive_fraction != null) {
      annotations.push({
        x: 1, y: 1, xref: "paper", yref: "paper", xanchor: "right", yanchor: "top",
        text: `<b>Productive: ${(effectiveness.productive_fraction * 100).toFixed(0)}%</b>  |  Circling: ${regions.length} regions`,
        showarrow: false,
        font: { size: 11, color: effectiveness.productive_fraction > 0.5 ? "#2E7D32" : "#D32F2F" },
        bgcolor: "rgba(128,128,128,0.15)", borderpad: 4,
      });
    }

    return { circlingTraces: traces, circlingAnnotations: annotations };
  }, [hasEffectiveness, effectiveness, colorMode, mx, my, n]);

  // Axes range with padding
  const pad = 0.12;
  const xr = [Math.min(...mx) - pad, Math.max(...mx) + pad];
  const yr = [Math.min(...my) - pad, Math.max(...my) + pad];

  // ═══════════════════════════════════════════════
  //  BOTTOM PLOT: Annotation Tracks
  // ═══════════════════════════════════════════════

  const annTraces = useMemo(() => {
    const traces = [];
    const idx = Array.from({ length: n }, (_, i) => i);

    // Track 1: HMM state as colored bar segments
    for (const seg of segs) {
      const rx = [];
      for (let j = seg.start; j <= seg.end; j++) rx.push(j);
      traces.push({
        x: rx, y: rx.map(() => 2.5), type: "bar",
        marker: { color: seg.state === 0 ? EXPLORE : EXPLOIT, line: { width: 0 } },
        width: 1.05, base: rx.map(() => 2),
        hovertemplate: rx.map((j) => `${unitTitle} ${j}: ${hmm_states[j] === 0 ? "Explore" : "Exploit"}<extra></extra>`),
        showlegend: false, xaxis: "x", yaxis: "y",
      });
    }

    // Track 2: Entropy
    if (hasEntropy) {
      traces.push({
        x: idx, y: idx.map(() => 0.95), mode: "lines",
        line: { color: "transparent", width: 0 }, hoverinfo: "skip", showlegend: false,
      });
      traces.push({
        x: idx, y: entNorm.map((e) => 0.95 + e * 0.85), mode: "lines", fill: "tonexty",
        line: { color: "#F16913", width: 1.2 },
        fillcolor: "rgba(241,105,19,0.2)",
        hovertemplate: idx.map((i) => `${unitTitle} ${i}<br>Entropy: ${entropy[i].toFixed(3)}<extra></extra>`),
        showlegend: false,
      });
    }

    // Track 3: Confidence
    if (hasConf) {
      traces.push({
        x: idx, y: idx.map(() => -0.1), mode: "lines",
        line: { color: "transparent", width: 0 }, hoverinfo: "skip", showlegend: false,
      });
      traces.push({
        x: idx, y: confNorm.map((c) => -0.1 + c * 0.85), mode: "lines", fill: "tonexty",
        line: { color: "#3182BD", width: 1.2 },
        fillcolor: "rgba(49,130,189,0.2)",
        hovertemplate: idx.map((i) => `${unitTitle} ${i}<br>Confidence: ${confidence[i].toFixed(2)}<extra></extra>`),
        showlegend: false,
      });
    }

    // Track 4: Flow type (colored bars, below confidence)
    if (hasFlow) {
      const ft = flowData.flow_type;
      // Group consecutive same-type segments
      let fStart = 0;
      for (let i = 1; i <= ft.length; i++) {
        if (i === ft.length || ft[i] !== ft[fStart]) {
          const type = ft[fStart];
          const rx = [];
          for (let j = fStart; j < i; j++) rx.push(j);
          traces.push({
            x: rx, y: rx.map(() => -0.9), type: "bar",
            marker: { color: FLOW_COLORS[type], line: { width: 0 } },
            width: 1.05, base: rx.map(() => -1.4),
            hovertemplate: rx.map((j) => `${unitTitle} ${j}: ${type}<extra></extra>`),
            showlegend: false,
          });
          fStart = i;
        }
      }
    }

    // Track 5: Functional Role (colored bars, below flow or confidence)
    if (hasFunctional) {
      const roles = functionalData.slice_roles || [];
      const funcBaseY = hasFlow ? -2.1 : -1.3;
      // Group consecutive same-role segments
      let rStart = 0;
      for (let i = 1; i <= roles.length; i++) {
        if (i === roles.length || roles[i] !== roles[rStart]) {
          const role = roles[rStart];
          const rx = [];
          for (let j = rStart; j < Math.min(i, n); j++) rx.push(j);
          traces.push({
            x: rx, y: rx.map(() => funcBaseY + 0.25), type: "bar",
            marker: { color: FUNC_COLORS[role] || FUNC_COLORS.productive, line: { width: 0 } },
            width: 1.05, base: rx.map(() => funcBaseY - 0.25),
            hovertemplate: rx.map((j) => `${unitTitle} ${j}: ${roles[j] || "unknown"}<extra></extra>`),
            showlegend: false,
          });
          rStart = i;
        }
      }
    }

    // Track 6: Effectiveness (colored bars, red=circling, green=productive)
    if (hasEffectiveness && colorMode === "effectiveness") {
      const labels = effectiveness.labels || [];
      const effColors = {
        core: "#2E7D32",
        closure: "#FF9800",
        drift: "#D32F2F",
        return_site: "#AB47BC",
        productive_exploit: "#388E3C",
        productive_explore: "#43A047",
        exploit: "#FFAB91",
        explore: "#90CAF9",
        circling: "#D32F2F",
      };
      const effBaseY = hasFunctional ? (hasFlow ? -3.0 : -2.2) : (hasFlow ? -2.1 : -1.3);
      let eStart = 0;
      for (let i = 1; i <= labels.length; i++) {
        if (i === labels.length || labels[i] !== labels[eStart]) {
          const label = labels[eStart];
          const rx = [];
          for (let j = eStart; j < Math.min(i, n); j++) rx.push(j);
          traces.push({
            x: rx, y: rx.map(() => effBaseY + 0.25), type: "bar",
            marker: { color: effColors[label] || "#90A4AE", line: { width: 0 } },
            width: 1.05, base: rx.map(() => effBaseY - 0.25),
            hovertemplate: rx.map((j) => `${unitTitle} ${j}: ${labels[j]}<br>Score: ${(effectiveness.scores[j]*100).toFixed(0)}%<extra></extra>`),
            showlegend: false,
          });
          eStart = i;
        }
      }
    }

    return traces;
  }, [segs, hmm_states, entNorm, confNorm, entropy, confidence, hasEntropy, hasConf, n, hasFlow, flowData, hasFunctional, functionalData, hasEffectiveness, effectiveness, colorMode]);

  const annAnnotations = [
    { x: -0.01, y: 2.25, xref: "paper", text: "State", showarrow: false, font: { size: 9, color: "#888", family: "Arial" }, xanchor: "right" },
  ];
  if (hasEntropy) annAnnotations.push({ x: -0.01, y: 1.35, xref: "paper", text: "Entropy", showarrow: false, font: { size: 9, color: "#F16913", family: "Arial" }, xanchor: "right" });
  if (hasConf) annAnnotations.push({ x: -0.01, y: 0.35, xref: "paper", text: "Conf.", showarrow: false, font: { size: 9, color: "#3182BD", family: "Arial" }, xanchor: "right" });
  if (hasFlow) annAnnotations.push({ x: -0.01, y: -1.15, xref: "paper", text: "Flow", showarrow: false, font: { size: 9, color: "#888", family: "Arial" }, xanchor: "right" });
  if (hasFunctional) {
    const funcLabelY = hasFlow ? -2.1 : -1.3;
    annAnnotations.push({ x: -0.01, y: funcLabelY, xref: "paper", text: "Func.", showarrow: false, font: { size: 9, color: "#4CAF50", family: "Arial" }, xanchor: "right" });
  }
  if (hasEffectiveness && colorMode === "effectiveness") {
    const effLabelY = hasFunctional ? (hasFlow ? -3.0 : -2.2) : (hasFlow ? -2.1 : -1.3);
    annAnnotations.push({ x: -0.01, y: effLabelY, xref: "paper", text: "Effect.", showarrow: false, font: { size: 9, color: "#2E7D32", family: "Arial" }, xanchor: "right" });
  }

  // Dynamic y-range for bottom plot
  let bottomYMin = -0.3;
  if (hasFlow) bottomYMin = -1.5;
  if (hasFunctional && !hasFlow) bottomYMin = -1.8;
  if (hasFunctional && hasFlow) bottomYMin = -2.7;
  if (hasEffectiveness && colorMode === "effectiveness") bottomYMin -= 1.0;

  return (
    <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", background: BG, borderRadius: 6, overflow: "hidden" }}>

      {/* ── Folded Structure ── */}
      <div style={{ flex: compact ? 1 : 7, minHeight: 0 }}>
        <Plot
          data={[...circlingTraces, ...structTraces, ...flowTraces, nodeTrace, ...funcMarkerTraces, ...markerTraces, ...hoverHighlightTraces]}
          layout={{
            xaxis: { visible: false, range: xr },
            yaxis: { visible: false, range: yr, scaleanchor: "x" },
            legend: { x: 1, y: 1, xanchor: "right", yanchor: "top", bgcolor: "rgba(128,128,128,0.15)", font: { size: 10 } },
            margin: { l: 5, r: 5, t: 5, b: 5 },
            annotations: [
              ...flowAnnotations, ...circlingAnnotations,
              ...(isTailFocus ? [{
                x: 0, y: 1, xref: "paper", yref: "paper", xanchor: "left", yanchor: "top",
                text: `<b>Answer tail</b> (${unitLabel}s ${tailRange.tailStart}\u2013${tailRange.tailEnd})`,
                showarrow: false,
                font: { size: 10, color: "#FF9800" },
                bgcolor: "rgba(128,128,128,0.15)", borderpad: 3,
              }] : []),
            ],
            paper_bgcolor: BG, plot_bgcolor: BG,
          }}
          useResizeHandler style={{ width: "100%", height: "100%" }}
          config={{ responsive: true, displayModeBar: false }}
          onClick={(event) => {
            if (onSliceClick && event.points && event.points.length > 0) {
              const pt = event.points[0];
              if (pt.customdata != null) onSliceClick(pt.customdata);
            }
          }}
        />
      </div>

      {!compact && <>
        {/* ── Divider ── */}
        <div style={{ height: 1, background: "var(--color-border)", margin: "0 16px" }} />

        {/* ── Annotation Tracks (bottom) ── */}
        <div style={{ flex: 3, minHeight: 0 }}>
          <Plot
            data={annTraces}
            layout={{
              xaxis: { range: [-1, n], showticklabels: true, tickfont: { size: 8, color: "#aaa" }, dtick: Math.ceil(n / 8), zeroline: false, showgrid: false },
              yaxis: { visible: false, range: [bottomYMin, 3.2] },
              annotations: annAnnotations,
              bargap: 0,
              margin: { l: 50, r: 15, t: 5, b: 22 },
              paper_bgcolor: BG, plot_bgcolor: BG,
            }}
            useResizeHandler style={{ width: "100%", height: "100%" }}
            config={{ responsive: true, displayModeBar: false }}
          />
        </div>
      </>}
    </div>
  );
}

export default React.memo(FoldingArcDiagram);
