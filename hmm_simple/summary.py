#!/usr/bin/env python3
"""
Human-readable summary for HMM-Simple scoring.
"""

from .core import compute_hmm_score


def summarize_hmm(H, C, p_stay=0.9, transition_weight=2.0):
    """Produce a human-readable summary of HMM-Simple analysis.

    Parameters
    ----------
    H : array-like
        Entropy time-series.
    C : array-like
        Confidence time-series.
    p_stay : float
        HMM self-transition probability (default 0.9).
    transition_weight : float
        Weight for the transition-quality term (default 2.0).

    Returns
    -------
    dict
        Keys: ``text``, ``verdict``, ``score``, ``transition_quality``,
        ``exploit_stability``, ``explore_fraction``, ``n_transitions``.
    """
    result = compute_hmm_score(H, C, p_stay=p_stay,
                               transition_weight=transition_weight)
    score = result["score"]

    # Verdict
    if score >= 0.7:
        verdict = "good"
    elif score >= 0.5:
        verdict = "borderline"
    else:
        verdict = "poor"

    # Build text summary
    lines = [f"HMM-Simple score: {score:.4f}  [{verdict}]"]

    if "transition_quality" in result:
        tq = result["transition_quality"]
        es = result["exploit_stability"]
        ef = result["explore_fraction"]
        nt = result["n_transitions"]
        lines.append(f"  transition_quality : {tq:+.4f}")
        lines.append(f"  exploit_stability  : {es:.4f}")
        lines.append(f"  explore_fraction   : {ef:.2%}")
        lines.append(f"  n_transitions      : {nt}")
    else:
        lines.append("  (too short for full analysis)")

    text = "\n".join(lines)

    summary = {
        "text": text,
        "verdict": verdict,
        "score": score,
    }

    # Copy selected features into summary when available
    for key in ("transition_quality", "exploit_stability",
                "explore_fraction", "n_transitions"):
        if key in result:
            summary[key] = result[key]

    return summary
