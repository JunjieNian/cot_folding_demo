import { useEffect, useState } from "react";

export default function useKeyboardShortcuts({
  problems, selectedProblem, setSelectedProblem,
  samples, selectedSample, setSelectedSample,
  view, setView,
  setShowOverview,
  lensMode, setLensMode,
  focusMode, setFocusMode,
}) {
  const [showHelp, setShowHelp] = useState(false);

  useEffect(() => {
    const handler = (e) => {
      // Ignore shortcuts when typing in inputs
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;

      switch (e.key) {
        case "ArrowRight": {
          // Next sample
          if (!samples.length || selectedSample == null) return;
          const sIdx = samples.findIndex((s) => s.sample_id === selectedSample);
          if (sIdx < samples.length - 1) setSelectedSample(samples[sIdx + 1].sample_id);
          e.preventDefault();
          break;
        }
        case "ArrowLeft": {
          // Previous sample
          if (!samples.length || selectedSample == null) return;
          const sIdx2 = samples.findIndex((s) => s.sample_id === selectedSample);
          if (sIdx2 > 0) setSelectedSample(samples[sIdx2 - 1].sample_id);
          e.preventDefault();
          break;
        }
        case "ArrowDown": {
          // Next problem
          if (!problems.length || selectedProblem == null) return;
          const pIdx = problems.findIndex((p) => p.problem_id === selectedProblem);
          if (pIdx < problems.length - 1) setSelectedProblem(problems[pIdx + 1].problem_id);
          e.preventDefault();
          break;
        }
        case "ArrowUp": {
          // Previous problem
          if (!problems.length || selectedProblem == null) return;
          const pIdx2 = problems.findIndex((p) => p.problem_id === selectedProblem);
          if (pIdx2 > 0) setSelectedProblem(problems[pIdx2 - 1].problem_id);
          e.preventDefault();
          break;
        }
        // Page switching
        case "1":
          setView("arc");
          break;
        case "2":
          setView("structure");
          break;
        case "3":
          setView("trajectory");
          break;
        // Lens switching (Inspect only)
        case "d":
        case "D":
          if (view === "arc") setLensMode("2d");
          break;
        case "t":
        case "T":
          if (view === "arc") setLensMode("3d");
          break;
        case "s":
        case "S":
          if (view === "arc") setLensMode("split");
          break;
        // Focus switching (Inspect only)
        case "g":
        case "G":
          if (view === "arc") setFocusMode("global");
          break;
        case "a":
        case "A":
          if (view === "arc") setFocusMode("answer-tail");
          break;
        case "b":
        case "B":
          setShowOverview((v) => !v);
          break;
        case "?":
          setShowHelp((v) => !v);
          break;
        case "Escape":
          setShowHelp(false);
          break;
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [problems, selectedProblem, samples, selectedSample, view,
      setSelectedProblem, setSelectedSample, setView, setShowOverview,
      lensMode, setLensMode, focusMode, setFocusMode]);

  return { showHelp, setShowHelp };
}
