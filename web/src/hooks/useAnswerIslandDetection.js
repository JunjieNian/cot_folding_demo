import { useMemo } from "react";

/**
 * Detect "answer tail" — the last contiguous exploit segment in hmm_states.
 *
 * This is the final stretch where the model converges on its answer,
 * corresponding to the terminal exploit phase in the HMM state sequence.
 *
 * Returns { detected, tailStart, tailEnd, tailFraction } or null.
 */
function detectAnswerTailHMM(foldingData) {
  if (!foldingData?.hmm_states) return null;

  const states = foldingData.hmm_states;
  const n = states.length;
  if (n < 4) return null;

  // The last state must be exploit (1) to have an answer tail
  if (states[n - 1] !== 1) return null;

  // Walk backwards from the end to find where this exploit run starts
  let tailStart = n - 1;
  while (tailStart > 0 && states[tailStart - 1] === 1) {
    tailStart--;
  }

  const tailEnd = n - 1;
  const tailLen = tailEnd - tailStart + 1;

  // Must be at least 2 slices
  if (tailLen < 2) return null;

  const tailFraction = tailLen / n;

  return {
    detected: true,
    tailStart,
    tailEnd,
    tailFraction,
    source: "hmm",
  };
}

export default function useAnswerIslandDetection(foldingData, currentSample) {
  return useMemo(() => {
    // Prefer graph-based pre-computed result from answer island analysis
    if (currentSample?.answer_island?.detected) {
      const ai = currentSample.answer_island;
      const n = foldingData?.n_slices || foldingData?.similarity_shape?.[0] || 0;
      return {
        detected: true,
        tailStart: ai.t_start,
        tailEnd: n > 0 ? n - 1 : ai.t_start + ai.tail_length - 1,
        tailFraction: n > 0 ? ai.tail_length / n : 0,
        source: "graph",
        attachmentScore: ai.attachment_score,
        containsAnswer: ai.contains_answer,
      };
    }

    // Fallback: HMM-based detection (original logic)
    return detectAnswerTailHMM(foldingData);
  }, [foldingData, currentSample]);
}
