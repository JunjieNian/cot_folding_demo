// ─── Base URL resolution ───
function resolveBase(dataset, checkpoint) {
  if (dataset === "rl" && checkpoint) return `./data/rl/${checkpoint}`;
  return `./data/${dataset}`;
}

let currentBase = "./data/aime24";

export function setActiveBase(dataset, checkpoint) {
  currentBase = resolveBase(dataset, checkpoint);
}

// ─── LRU Cache ───
// Keeps recently loaded bundles in memory so back-navigation is instant.
const CACHE_MAX = 24;  // ~24 bundles × avg 2.4MB parsed ≈ 60MB in memory (acceptable)
const cache = new Map();
const inflight = new Map();  // URL → Promise (dedup concurrent requests)

function cacheGet(key) {
  if (!cache.has(key)) return undefined;
  const val = cache.get(key);
  // Move to end (most recently used)
  cache.delete(key);
  cache.set(key, val);
  return val;
}

function cacheSet(key, val) {
  cache.delete(key);  // refresh position
  cache.set(key, val);
  // Evict oldest if over limit
  if (cache.size > CACHE_MAX) {
    const oldest = cache.keys().next().value;
    cache.delete(oldest);
  }
}

async function fetchJSON(url, signal) {
  const init = { cache: "default" };
  if (signal) init.signal = signal;
  const res = await fetch(url, init);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} loading ${url}`);
  }
  return res.json();
}

async function fetchText(url, signal) {
  const init = { cache: "default" };
  if (signal) init.signal = signal;
  const res = await fetch(url, init);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} loading ${url}`);
  }
  return res.text();
}

async function cachedFetch(url, signal) {
  const hit = cacheGet(url);
  if (hit) return hit;
  const existing = inflight.get(url);
  if (existing) return existing;
  const promise = fetchJSON(url, signal).then((data) => {
    cacheSet(url, data);
    inflight.delete(url);
    return data;
  }).catch((e) => {
    inflight.delete(url);
    throw e;
  });
  inflight.set(url, promise);
  return promise;
}

async function cachedFetchText(url, signal) {
  const hit = cacheGet(url);
  if (hit) return hit;
  const existing = inflight.get(url);
  if (existing) return existing;
  const promise = fetchText(url, signal).then((data) => {
    cacheSet(url, data);
    inflight.delete(url);
    return data;
  }).catch((e) => {
    inflight.delete(url);
    throw e;
  });
  inflight.set(url, promise);
  return promise;
}

// ─── Public API (per-checkpoint) ───

export const getAppConfig = (signal, base) =>
  cachedFetch(`${base || currentBase}/app.json`, signal);

export const getProblemsIndex = (signal, base) =>
  cachedFetch(`${base || currentBase}/problems.index.json`, signal);

export const getOverview = (signal, base) =>
  cachedFetch(`${base || currentBase}/overview.json`, signal);

export const getSampleBundle = (pid, sid, signal, base) =>
  cachedFetch(`${base || currentBase}/samples/p${pid}/s${sid}.bundle.json`, signal);

export const getSampleText = (pid, sid, signal, base) =>
  cachedFetch(`${base || currentBase}/samples/p${pid}/s${sid}.text.json`, signal);

export const getProblemCompare = (pid, signal, base) =>
  cachedFetch(`${base || currentBase}/compare/p${pid}.json`, signal);

// Similarity sidecar (lazy-loaded for ContactMap)
export const getSimilarity = (pid, sid, signal, base) =>
  cachedFetchText(`${base || currentBase}/samples/p${pid}/s${sid}.sim.b64`, signal);

// Prefetch — fire-and-forget, no error propagation, dedup with inflight
export const prefetchSampleBundle = (pid, sid, base) => {
  const url = `${base || currentBase}/samples/p${pid}/s${sid}.bundle.json`;
  if (cache.has(url) || inflight.has(url)) return;
  const promise = fetchJSON(url).then((data) => {
    cacheSet(url, data);
    inflight.delete(url);
    return data;
  }).catch(() => { inflight.delete(url); });
  inflight.set(url, promise);
};

// ─── Semantic Validation ───

export const getSemanticValidation = (signal, base) =>
  cachedFetch(`${base || currentBase}/semantic_validation.json`, signal);

export const getSliceNeighbors = (pid, sid, signal, base) =>
  cachedFetch(`${base || currentBase}/samples/p${pid}/s${sid}.neighbors.json`, signal);

export const getComparePresets = (signal, base) =>
  cachedFetch(`${base || currentBase}/compare_presets.json`, signal);

// ─── RL-specific API (cross-checkpoint, at rl root level) ───

export const getCheckpoints = (signal) =>
  cachedFetch(`./data/rl/checkpoints.json`, signal);

export const getTrajectory = (signal) =>
  cachedFetch(`./data/rl/trajectory.json`, signal);

export const getProblemsMeta = (signal) =>
  cachedFetch(`./data/rl/problems.meta.json`, signal);

export const getRanking = (signal) =>
  cachedFetch(`./data/rl/model_ranking.json`, signal);

// Compatibility shims
export const getFolding = (pid, sid, signal, base) => getSampleBundle(pid, sid, signal, base);
export const getStructuralComparison = (pid, signal, base) => getProblemCompare(pid, signal, base);
export const getBatchOverview = (signal, base) => getOverview(signal, base);
