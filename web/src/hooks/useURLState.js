import { useEffect, useRef } from "react";

const VALID_VIEWS = new Set(["arc", "structure", "trajectory"]);

// Compat: map removed view names to their replacements
const VIEW_COMPAT = {
  detail: "arc",
  phase: "arc",
  arc3d: "arc",
  phase3d: "arc",
  casestudy: "structure",
  ranking: "trajectory",
};

export default function useURLState({
  activeDataset, selectedProblem, selectedSample, view, colorMode,
  handleDatasetSwitch, setSelectedProblem, setSelectedSample, setView, setColorMode,
  datasets,
  // RL-specific
  selectedCheckpoint, handleCheckpointChange,
}) {
  const initialized = useRef(false);

  // Read URL params once on mount
  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;

    const params = new URLSearchParams(window.location.search);
    const ds = params.get("dataset");
    const ckpt = params.get("checkpoint");
    const prob = params.get("problem");
    const samp = params.get("sample");
    const v = params.get("view");
    const c = params.get("color");

    // Dataset switch (must happen first so checkpoint/problems load)
    if (ds && datasets.includes(ds) && ds !== activeDataset) {
      handleDatasetSwitch(ds);
    }

    // Checkpoint (RL)
    if (ckpt && handleCheckpointChange) {
      // Defer slightly to allow dataset switch to complete
      setTimeout(() => handleCheckpointChange(ckpt), 100);
    }

    if (prob != null) {
      const pid = isNaN(Number(prob)) ? prob : Number(prob);
      // Defer to allow problems to load
      setTimeout(() => setSelectedProblem(pid), ds ? 200 : 0);
    }
    if (samp != null) {
      setTimeout(() => setSelectedSample(Number(samp)), ds ? 300 : 0);
    }
    if (v) {
      if (VALID_VIEWS.has(v)) setView(v);
      else if (VIEW_COMPAT[v]) setView(VIEW_COMPAT[v]);
    }
    if (c === "entropy" || c === "confidence" || c === "state") setColorMode(c);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Write state to URL
  useEffect(() => {
    if (!initialized.current) return;
    const params = new URLSearchParams();
    if (activeDataset && activeDataset !== "aime24") params.set("dataset", activeDataset);
    if (selectedCheckpoint) params.set("checkpoint", selectedCheckpoint);
    if (selectedProblem != null) params.set("problem", String(selectedProblem));
    if (selectedSample != null) params.set("sample", String(selectedSample));
    if (view) params.set("view", view);
    if (colorMode) params.set("color", colorMode);

    const qs = params.toString();
    const newURL = qs ? `${window.location.pathname}?${qs}` : window.location.pathname;
    window.history.replaceState(null, "", newURL);
  }, [activeDataset, selectedCheckpoint, selectedProblem, selectedSample, view, colorMode]);
}
