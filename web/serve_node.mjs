#!/usr/bin/env node
/**
 * Concurrent static file server for COT Folding Map.
 * Handles gzip, CORS, proper MIME types, and SPA fallback.
 *
 * Usage: node serve_node.mjs [port]
 */

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { join, extname } from "node:path";
import { fileURLToPath } from "node:url";
import { createReadStream } from "node:fs";

const PORT = parseInt(process.argv[2] || "8080", 10);
const __dirname = fileURLToPath(new URL(".", import.meta.url));
const DIST = join(__dirname, "dist");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js":   "application/javascript; charset=utf-8",
  ".css":  "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png":  "image/png",
  ".svg":  "image/svg+xml",
  ".ico":  "image/x-icon",
  ".woff": "font/woff",
  ".woff2":"font/woff2",
};

const server = createServer(async (req, res) => {
  // CORS
  res.setHeader("Access-Control-Allow-Origin", "*");

  let urlPath = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
  if (urlPath === "/") urlPath = "/index.html";

  const filePath = join(DIST, urlPath);

  // Prevent path traversal
  if (!filePath.startsWith(DIST)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  try {
    const fileStat = await stat(filePath);
    if (!fileStat.isFile()) throw new Error("Not a file");

    const ext = extname(filePath).toLowerCase();
    const mime = MIME[ext] || "application/octet-stream";

    res.writeHead(200, {
      "Content-Type": mime,
      "Content-Length": fileStat.size,
      // Cache JSON data aggressively (immutable exports), assets too
      "Cache-Control": ext === ".json"
        ? "public, max-age=86400, immutable"
        : ext === ".html"
          ? "no-cache"
          : "public, max-age=31536000, immutable",
    });

    createReadStream(filePath).pipe(res);
  } catch {
    // SPA fallback: serve index.html for non-file routes
    try {
      const indexPath = join(DIST, "index.html");
      const html = await readFile(indexPath);
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(html);
    } catch {
      res.writeHead(404);
      res.end("Not Found");
    }
  }
});

server.listen(PORT, "0.0.0.0", () => {
  console.log("============================================");
  console.log("  COT Folding Map - AIME24 (Static)");
  console.log(`  URL:  http://0.0.0.0:${PORT}`);
  console.log("  Ctrl+C to stop");
  console.log("============================================");
});
