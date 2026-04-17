/**
 * Export utilities for charts and data.
 */

/**
 * Export a Plotly chart as PNG or SVG.
 * @param {HTMLElement} plotDiv - The Plotly div element
 * @param {string} format - "png" or "svg"
 * @param {string} filename - Filename without extension
 */
export function exportPlotlyChart(plotDiv, format = "png", filename = "chart") {
  if (!plotDiv || !window.Plotly) return;
  window.Plotly.downloadImage(plotDiv, {
    format,
    filename,
    width: 1200,
    height: 800,
  });
}

/**
 * Export data as CSV and trigger download.
 * @param {Array<Object>} data - Array of row objects
 * @param {string} filename - Filename without extension
 */
export function exportCSV(data, filename = "data") {
  if (!data || data.length === 0) return;
  const headers = Object.keys(data[0]);
  const rows = data.map((row) =>
    headers.map((h) => {
      const val = row[h];
      if (val == null) return "";
      const str = String(val);
      return str.includes(",") || str.includes('"') || str.includes("\n")
        ? `"${str.replace(/"/g, '""')}"`
        : str;
    }).join(",")
  );
  const csv = [headers.join(","), ...rows].join("\n");
  downloadBlob(csv, `${filename}.csv`, "text/csv");
}

/**
 * Export metrics data for a folding sample.
 * @param {Object} foldingData - The folding data object
 * @param {string} filename - Filename prefix
 */
export function exportMetrics(foldingData, filename = "metrics") {
  if (!foldingData) return;
  const m = foldingData.metrics;
  const rows = [
    { metric: "n_slices", value: m.n_slices },
    { metric: "n_explore", value: m.n_explore },
    { metric: "n_exploit", value: m.n_exploit },
    { metric: "n_transitions", value: m.n_transitions },
    { metric: "folding_degree", value: m.folding_degree },
    { metric: "contact_order", value: m.contact_order },
    { metric: "radius_of_gyration", value: m.radius_of_gyration },
    { metric: "mds_stress", value: foldingData.mds_stress },
    { metric: "contact_threshold", value: m.contact_threshold },
    { metric: "total_contacts", value: m.total_contacts },
    { metric: "long_range_contacts", value: m.long_range_contacts },
  ];
  exportCSV(rows, filename);
}

/**
 * Export similarity matrix as CSV.
 */
export function exportSimilarityMatrix(foldingData, filename = "similarity") {
  if (!foldingData) return;
  const n = foldingData.similarity_shape[0];
  const sim = foldingData.similarity;
  const rows = [];
  for (let i = 0; i < n; i++) {
    const row = {};
    row["slice"] = i;
    for (let j = 0; j < n; j++) {
      row[`s${j}`] = sim[i * n + j];
    }
    rows.push(row);
  }
  exportCSV(rows, filename);
}

function downloadBlob(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
