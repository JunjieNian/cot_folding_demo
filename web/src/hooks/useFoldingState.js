import { useEffect, useState, useRef, useCallback } from "react";
import {
  getAppConfig, getProblemsIndex, getSampleBundle, getSampleText,
  prefetchSampleBundle, setActiveBase, getCheckpoints, getTrajectory,
  getProblemsMeta, getRanking, getSemanticValidation,
} from "../api";

// Stable reference — prevents useURLState from re-running on every render
const DATASETS = ["aime24", "rl"];

function resolveBase(dataset, checkpoint) {
  if (dataset === "rl" && checkpoint) return `./data/rl/${checkpoint}`;
  return `./data/${dataset}`;
}

// Decode similarity_b64 — upper-triangle packed uint8 → full n×n Float32 symmetric matrix
function decodeSimilarityB64(b64, n) {
  const bin = atob(b64);
  const arr = new Float32Array(n * n);
  let k = 0;
  for (let i = 0; i < n; i++) {
    arr[i * n + i] = 1;  // diagonal = self-similarity
    for (let j = i + 1; j < n; j++, k++) {
      const v = bin.charCodeAt(k) / 255;
      arr[i * n + j] = v;
      arr[j * n + i] = v;
    }
  }
  return arr;
}

export default function useFoldingState() {
  // Dataset & checkpoint state
  const [activeDataset, setActiveDataset] = useState("aime24");
  const [checkpoints, setCheckpoints] = useState([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState(null);
  const [trajectoryData, setTrajectoryData] = useState(null);
  const [rankingData, setRankingData] = useState(null);
  const [problemsMeta, setProblemsMeta] = useState(null);

  // Index data (loaded once per dataset/checkpoint)
  const [problemsIndex, setProblemsIndex] = useState(null);
  const [problems, setProblems] = useState([]);
  const [selectedProblem, setSelectedProblem] = useState(null);
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);

  // Sample bundle (folding + clustering + flow + functional in one)
  const [foldingData, setFoldingData] = useState(null);
  const [clustering, setClustering] = useState(null);
  const [flowData, setFlowData] = useState(null);
  const [functionalData, setFunctionalData] = useState(null);

  // Decoded similarity matrix (from bundle inline data)
  const [decodedSimilarity, setDecodedSimilarity] = useState(null);

  // Text bundle (loaded once per sample, then sliced client-side)
  const [textBundle, setTextBundle] = useState(null);
  const [selectedSlice, setSelectedSlice] = useState(null);
  const [sliceTextData, setSliceTextData] = useState(null);
  const [sliceTextLoading, setSliceTextLoading] = useState(false);

  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showOverview, setShowOverview] = useState(false);
  const [view, setView] = useState("arc");
  const [colorMode, setColorMode] = useState("entropy");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [lensMode, setLensMode] = useState("2d");          // "2d" | "3d" | "split"
  const [focusMode, setFocusMode] = useState("global");    // "global" | "explore-exploit" | "answer-tail"
  const [darkMode, setDarkMode] = useState(() => {
    try { return localStorage.getItem("cot-theme") === "dark"; } catch { return false; }
  });

  // Per-checkpoint sample correctness for the selected sample (RL only)
  const [checkpointSampleCorrectness, setCheckpointSampleCorrectness] = useState({});
  const [allCheckpointIndexes, setAllCheckpointIndexes] = useState({});  // { checkpointName: indexData }

  // Semantic validation (global summary, loaded once for aime24)
  const [semanticValidation, setSemanticValidation] = useState(null);

  const highlightRef = useRef(null);
  const bundleAbort = useRef(null);

  // --- Helper: load problems index for current base ---
  const loadProblemsForBase = useCallback((base) => {
    return getProblemsIndex(undefined, base).then((idx) => {
      setProblemsIndex(idx);
      const probs = idx.problems.map((p) => ({
        problem_id: p.problem_id,
        n_samples: p.n_samples,
        accuracy: p.accuracy,
        processing_time_s: p.processing_time_s,
        short_prompt: p.short_prompt || "",
        ground_truth: p.ground_truth || "",
      }));
      setProblems(probs);
      return probs;
    });
  }, []);

  // --- Initial load: fetch problems index for aime24 + app.json defaults ---
  useEffect(() => {
    const base = resolveBase("aime24", null);
    setActiveBase("aime24", null);
    // Load semantic validation summary (non-blocking)
    getSemanticValidation(undefined, base).then(setSemanticValidation).catch(() => {});
    Promise.all([
      loadProblemsForBase(base),
      getAppConfig(undefined, base).catch(() => null),
    ])
      .then(([probs, config]) => {
        if (probs.length > 0) {
          const defaultPid = config?.defaultProblemId;
          const found = defaultPid != null && probs.find((p) => p.problem_id === defaultPid);
          setSelectedProblem(found ? defaultPid : probs[0].problem_id);
        }
      })
      .catch((e) => setError(e.message));
  }, [loadProblemsForBase]);

  // --- Dataset switching ---
  const handleDatasetSwitch = useCallback((name) => {
    if (name === activeDataset) return;
    setLoading(true);
    setError(null);

    // Clear existing data
    setProblems([]);
    setSelectedProblem(null);
    setSamples([]);
    setSelectedSample(null);
    setFoldingData(null);
    setClustering(null);
    setFlowData(null);
    setFunctionalData(null);
    setTextBundle(null);
    setSelectedSlice(null);
    setSliceTextData(null);
    setDecodedSimilarity(null);

    if (name === "rl") {
      // Load RL metadata
      Promise.all([
        getCheckpoints(),
        getProblemsMeta(),
      ]).then(([ckpts, meta]) => {
        setCheckpoints(ckpts);
        setProblemsMeta(meta);
        const defaultCkpt = "base";
        setSelectedCheckpoint(defaultCkpt);
        setActiveDataset("rl");

        const base = resolveBase("rl", defaultCkpt);
        setActiveBase("rl", defaultCkpt);

        // Delay prefetch of all checkpoint indexes — not needed for first render
        const schedulePreload = typeof requestIdleCallback === "function" ? requestIdleCallback : (cb) => setTimeout(cb, 2000);
        schedulePreload(() => {
          Promise.all(
            ckpts.map((ckpt) =>
              getProblemsIndex(undefined, resolveBase("rl", ckpt.name))
                .then((idx) => ({ name: ckpt.name, idx }))
                .catch(() => null)
            )
          ).then((results) => {
            const map = {};
            for (const r of results) {
              if (r) map[r.name] = r.idx;
            }
            setAllCheckpointIndexes(map);
          });
        });

        return Promise.all([
          loadProblemsForBase(base),
          getAppConfig(undefined, base).catch(() => null),
        ]);
      }).then(([probs, config]) => {
        if (probs.length > 0) {
          const defaultPid = config?.defaultProblemId;
          const found = defaultPid != null && probs.find((p) => p.problem_id === defaultPid);
          setSelectedProblem(found ? defaultPid : probs[0].problem_id);
        }
      }).catch((e) => setError(e.message))
        .finally(() => setLoading(false));

      // Also load trajectory + ranking (non-blocking)
      getTrajectory().then(setTrajectoryData).catch(() => {});
      getRanking().then(setRankingData).catch(() => {});
    } else {
      // Switch to non-RL dataset
      setSelectedCheckpoint(null);
      setCheckpoints([]);
      setTrajectoryData(null);
      setRankingData(null);
      setProblemsMeta(null);
      setActiveDataset(name);

      const base = resolveBase(name, null);
      setActiveBase(name, null);
      Promise.all([
        loadProblemsForBase(base),
        getAppConfig(undefined, base).catch(() => null),
      ])
        .then(([probs, config]) => {
          if (probs.length > 0) {
            const defaultPid = config?.defaultProblemId;
            const found = defaultPid != null && probs.find((p) => p.problem_id === defaultPid);
            setSelectedProblem(found ? defaultPid : probs[0].problem_id);
          }
        })
        .catch((e) => setError(e.message))
        .finally(() => setLoading(false));
    }
  }, [activeDataset, loadProblemsForBase]);

  // --- Checkpoint change (RL only) ---
  const handleCheckpointChange = useCallback((checkpoint) => {
    if (checkpoint === selectedCheckpoint) return;
    setLoading(true);
    setError(null);

    // Preserve selected problem and sample across checkpoints
    const preservedProblem = selectedProblem;
    const preservedSample = selectedSample;

    // Clear text/slice state but keep foldingData visible until new bundle arrives
    setTextBundle(null);
    setSelectedSlice(null);
    setSliceTextData(null);

    setSelectedCheckpoint(checkpoint);
    const base = resolveBase("rl", checkpoint);
    setActiveBase("rl", checkpoint);

    loadProblemsForBase(base)
      .then((probs) => {
        // Preserve problem selection if it exists in the new checkpoint
        const found = probs.find((p) => p.problem_id === preservedProblem);
        if (found) {
          setSelectedProblem(found.problem_id);
        } else if (probs.length > 0) {
          setSelectedProblem(probs[0].problem_id);
        }

        // Prefetch adjacent checkpoint bundles for current problem+sample
        if (preservedProblem != null && preservedSample != null) {
          const ckptIdx = checkpoints.findIndex((c) => c.name === checkpoint);
          for (const offset of [-1, 1]) {
            const adj = checkpoints[ckptIdx + offset];
            if (adj) {
              prefetchSampleBundle(
                preservedProblem, preservedSample,
                resolveBase("rl", adj.name),
              );
            }
          }
        }
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [selectedCheckpoint, selectedProblem, selectedSample, checkpoints, loadProblemsForBase]);

  // --- On problem change: extract samples from index ---
  useEffect(() => {
    if (selectedProblem == null || !problemsIndex) return;
    const prob = problemsIndex.problems.find((p) => p.problem_id === selectedProblem);
    if (!prob) { setSamples([]); setSelectedSample(null); return; }
    setSamples(prob.samples);
    // Clear text/slice state but keep foldingData visible until new bundle arrives
    setTextBundle(null);
    setSelectedSlice(null);
    setSliceTextData(null);
    if (prob.samples.length > 0) setSelectedSample(prob.samples[0].sample_id);
  }, [selectedProblem, problemsIndex]);

  // --- Compute per-checkpoint sample correctness for selected sample (RL) ---
  useEffect(() => {
    if (activeDataset !== "rl" || selectedProblem == null || selectedSample == null) {
      setCheckpointSampleCorrectness({});
      return;
    }
    const corr = {};
    for (const [ckptName, idx] of Object.entries(allCheckpointIndexes)) {
      const prob = idx.problems.find((p) => p.problem_id === selectedProblem);
      if (prob && prob.samples) {
        const sample = prob.samples.find((s) => s.sample_id === selectedSample);
        if (sample && sample.is_correct != null) {
          corr[ckptName] = sample.is_correct;
        }
      }
    }
    setCheckpointSampleCorrectness(corr);
  }, [activeDataset, selectedProblem, selectedSample, allCheckpointIndexes]);

  // --- On sample change: fetch bundle ---
  useEffect(() => {
    if (selectedProblem == null || selectedSample == null) return;
    setLoading(true);
    setError(null);
    setSelectedSlice(null);
    setSliceTextData(null);
    setTextBundle(null);
    setDecodedSimilarity(null);
    setFocusMode("global");

    if (bundleAbort.current) bundleAbort.current.abort();
    const controller = new AbortController();
    bundleAbort.current = controller;

    const base = resolveBase(activeDataset, selectedCheckpoint);
    getSampleBundle(selectedProblem, selectedSample, controller.signal, base)
      .then((bundle) => {
        if (controller.signal.aborted) return;

        const folding = bundle.folding;
        if (folding.similarity_b64) {
          const n = folding.similarity_shape[0];
          setDecodedSimilarity(decodeSimilarityB64(folding.similarity_b64, n));
        }

        setFoldingData(folding);
        setClustering(bundle.clustering);
        setFlowData(bundle.flow || null);
        setFunctionalData(bundle.functional || null);

        // Preload text data in parallel (always, not just on click)
        getSampleText(selectedProblem, selectedSample, controller.signal, base)
          .then((tb) => {
            if (controller.signal.aborted) return;
            tb._pid = selectedProblem;
            tb._sid = selectedSample;
            setTextBundle(tb);
          })
          .catch(() => {});  // non-critical — will retry on slice click

        // Prefetch adjacent samples
        const idx = samples.findIndex((s) => s.sample_id === selectedSample);
        if (idx >= 0 && idx < samples.length - 1) {
          prefetchSampleBundle(selectedProblem, samples[idx + 1].sample_id, base);
        }
        if (idx > 0) {
          prefetchSampleBundle(selectedProblem, samples[idx - 1].sample_id, base);
        }
      })
      .catch((e) => {
        if (e.name === "AbortError") return;
        setError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [selectedProblem, selectedSample, samples, activeDataset, selectedCheckpoint]);

  // --- Slice text: fetch text bundle once per sample, then derive client-side ---
  const handleSliceClick = useCallback((sliceIdxOrPhase) => {
    if (selectedProblem == null || selectedSample == null) return;
    setSelectedSlice(sliceIdxOrPhase);

    if (textBundle && textBundle._pid === selectedProblem && textBundle._sid === selectedSample) {
      deriveSliceText(textBundle, sliceIdxOrPhase);
      return;
    }

    setSliceTextLoading(true);
    const base = resolveBase(activeDataset, selectedCheckpoint);
    getSampleText(selectedProblem, selectedSample, undefined, base)
      .then((tb) => {
        tb._pid = selectedProblem;
        tb._sid = selectedSample;
        setTextBundle(tb);
        deriveSliceText(tb, sliceIdxOrPhase);
      })
      .catch((err) => {
        setSliceTextData({ success: false, error: err.message });
      })
      .finally(() => setSliceTextLoading(false));
  }, [selectedProblem, selectedSample, textBundle, activeDataset, selectedCheckpoint]);

  function deriveSliceText(tb, sliceIdxOrPhase) {
    if (!tb || !tb.items) {
      setSliceTextData({ success: false, error: "No text data" });
      return;
    }
    const fullText = tb.full_text;
    const totalTokens = tb.items.length > 0 ? Math.max(...tb.items.map((i) => i.token_end)) : 0;

    // Phase range click
    if (typeof sliceIdxOrPhase === "object" && sliceIdxOrPhase !== null) {
      const { start, end, phaseIdx, stateName } = sliceIdxOrPhase;
      const phaseItems = tb.items.filter((it) => it.slice_idx >= start && it.slice_idx <= end);
      if (phaseItems.length === 0) {
        setSliceTextData({ success: false, error: `Phase ${phaseIdx + 1} slices not found` });
        return;
      }
      const charStart = Math.min(...phaseItems.map((it) => it.char_start));
      const charEnd = Math.max(...phaseItems.map((it) => it.char_end));
      const tokenStart = Math.min(...phaseItems.map((it) => it.token_start));
      const tokenEnd = Math.max(...phaseItems.map((it) => it.token_end));
      setSliceTextData({
        success: true,
        slice_idx: start,
        is_phase: true,
        phase_idx: phaseIdx,
        phase_start: start,
        phase_end: end,
        phase_state: stateName,
        unit_label: tb.unit_label || "slice",
        before_text: fullText.slice(0, charStart),
        current_text: fullText.slice(charStart, charEnd),
        after_text: fullText.slice(charEnd),
        token_start: tokenStart,
        token_end: tokenEnd,
        total_tokens: totalTokens,
      });
      setTimeout(() => highlightRef.current?.scrollIntoView({ behavior: "smooth", block: "center" }), 50);
      return;
    }

    // Single slice click
    const sliceIdx = sliceIdxOrPhase;
    const item = tb.items.find((it) => it.slice_idx === sliceIdx);
    if (!item) {
      setSliceTextData({ success: false, error: `Slice ${sliceIdx} not found` });
      return;
    }
    setSliceTextData({
      success: true,
      slice_idx: sliceIdx,
      unit_label: tb.unit_label || "slice",
      before_text: fullText.slice(0, item.char_start),
      current_text: fullText.slice(item.char_start, item.char_end),
      after_text: fullText.slice(item.char_end),
      token_start: item.token_start,
      token_end: item.token_end,
      total_tokens: totalTokens,
    });
    setTimeout(() => highlightRef.current?.scrollIntoView({ behavior: "smooth", block: "center" }), 50);
  }

  // Dark mode persistence
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", darkMode ? "dark" : "light");
    try { localStorage.setItem("cot-theme", darkMode ? "dark" : "light"); } catch {}
  }, [darkMode]);

  const toggleDarkMode = useCallback(() => setDarkMode((v) => !v), []);
  const toggleSidebar = useCallback(() => setSidebarCollapsed((v) => !v), []);

  return {
    // Data state
    datasets: DATASETS, activeDataset, semanticValidation,
    checkpoints, selectedCheckpoint, trajectoryData, rankingData, problemsMeta, checkpointSampleCorrectness,
    problems, selectedProblem, samples, selectedSample,
    foldingData, clustering, loading, error,
    decodedSimilarity,
    flowData, functionalData,
    textBundle,
    selectedSlice, sliceTextData, sliceTextLoading,
    // UI state
    showOverview, view, colorMode, sidebarCollapsed, darkMode,
    lensMode, focusMode,
    highlightRef,
    // Actions
    setSelectedProblem, setSelectedSample,
    setShowOverview, setView, setColorMode,
    setSelectedSlice, setSliceTextData,
    setLensMode, setFocusMode,
    handleSliceClick, handleDatasetSwitch,
    handleCheckpointChange,
    toggleDarkMode, toggleSidebar,
  };
}
