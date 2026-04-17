#!/usr/bin/env python3
"""
HMM-Simple-3 — compressed 3-metric scoring via geometric mean.

Replaces the legacy 6/7 small metrics with exactly 3 macro metrics:
  1. transition_strength  — Explore→Exploit shift quality
  2. exploit_quality      — Exploit phase goodness vs global baseline
  3. exploit_stability    — per-step penalty for instability in Exploit

Final score = geometric mean (no tuneable weights).
"""

import numpy as np

from .core import _hmm_viterbi_2state


def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def compute_hmmsimple3_metrics(entropy, confidence, eps=1e-12, hmm_kwargs=None):
    """Compute 3 macro metrics and geometric-mean score.

    Parameters
    ----------
    entropy : array-like
        Entropy time-series (H).
    confidence : array-like
        Confidence time-series (C), values ideally in [0, 1].
    eps : float
        Small constant to prevent division by zero.
    hmm_kwargs : dict or None
        Extra keyword arguments forwarded to ``_hmm_viterbi_2state``.

    Returns
    -------
    dict
        Keys: transition_strength, exploit_quality, exploit_stability,
        hmm3_score, n_explore, n_exploit, T, plus diagnostics.

    Raises
    ------
    ValueError
        If inputs are empty, contain NaN, or contain inf.
    """
    # --- 1a. Input validation ---
    H = np.asarray(entropy, dtype=np.float64)
    C = np.asarray(confidence, dtype=np.float64)

    if H.size == 0 or C.size == 0:
        raise ValueError("H and C must be non-empty arrays")
    if np.any(np.isnan(H)) or np.any(np.isnan(C)):
        raise ValueError("H and C must not contain NaN values")
    if np.any(np.isinf(H)) or np.any(np.isinf(C)):
        raise ValueError("H and C must not contain inf values")

    # Silently clip confidence to [0, 1]
    C = np.clip(C, 0.0, 1.0)

    T = len(H)

    # --- 1b. T == 1 early return ---
    if T == 1:
        score = float((0.5 * 0.5 * 1.0) ** (1.0 / 3.0))
        return {
            "transition_strength": 0.5,
            "exploit_quality": 0.5,
            "exploit_stability": 1.0,
            "hmm3_score": score,
            "n_explore": 0,
            "n_exploit": 1,
            "T": T,
            "med_H_E": float("nan"),
            "med_H_X": float(H[0]),
            "med_C_E": float("nan"),
            "med_C_X": float(C[0]),
            "mean_tau_E": float("nan"),
            "mean_tau_X": 0.0,
            "s_H": float("nan"),
            "s_C": float("nan"),
            "J_X": 0.0,
        }

    # --- 1c. HMM segmentation ---
    kw = dict(p_stay=0.9)
    if hmm_kwargs:
        kw.update(hmm_kwargs)
    states = _hmm_viterbi_2state(H, **kw)

    # --- 1d. Canonicalize labels: state 0 = Explore (higher mean H) ---
    mask0 = states == 0
    mask1 = states == 1
    n0 = int(np.sum(mask0))
    n1 = int(np.sum(mask1))

    if n0 > 0 and n1 > 0:
        if np.mean(H[mask0]) < np.mean(H[mask1]):
            states = 1 - states

    explore_mask = states == 0
    exploit_mask = states == 1
    n_explore = int(np.sum(explore_mask))
    n_exploit = int(np.sum(exploit_mask))

    # --- 1e. One-state fallback ---
    if n_explore == 0 or n_exploit == 0:
        K = max(1, T // 4)
        explore_mask = np.zeros(T, dtype=bool)
        explore_mask[:K] = True
        exploit_mask = ~explore_mask
        n_explore = int(np.sum(explore_mask))
        n_exploit = int(np.sum(exploit_mask))

    # --- 1f. Global robust statistics ---
    med_H_all = float(np.median(H))
    med_C_all = float(np.median(C))
    s_H = float(np.median(np.abs(H - med_H_all)) + eps)
    s_C = float(np.median(np.abs(C - med_C_all)) + eps)

    # --- 1g. Temporal position ---
    tau = np.arange(T, dtype=np.float64) / max(T - 1, 1)

    # --- 1h. Per-state statistics ---
    E = explore_mask
    X = exploit_mask

    med_H_E = float(np.median(H[E]))
    med_H_X = float(np.median(H[X]))
    med_C_E = float(np.median(C[E]))
    med_C_X = float(np.median(C[X]))
    mean_tau_E = float(np.mean(tau[E]))
    mean_tau_X = float(np.mean(tau[X]))

    # --- 1i. Metric 1 — transition_strength ---
    ts_raw = (
        (med_H_E - med_H_X) / s_H
        + (med_C_X - med_C_E) / s_C
        + (mean_tau_X - mean_tau_E)
    )
    transition_strength = float(_sigmoid(ts_raw))

    # --- 1j. Metric 2 — exploit_quality ---
    eq_raw = (
        (med_C_X - med_C_all) / s_C
        + (med_H_all - med_H_X) / s_H
    )
    exploit_quality = float(_sigmoid(eq_raw))

    # --- 1k. Metric 3 — exploit_stability ---
    # Exploit steps t where t >= 1 (have a predecessor in the raw series)
    exploit_indices = np.where(X)[0]
    exploit_with_pred = exploit_indices[exploit_indices >= 1]

    if len(exploit_with_pred) == 0:
        J_X = 0.0
    else:
        h_diff = np.maximum(H[exploit_with_pred] - H[exploit_with_pred - 1], 0.0)
        c_diff = np.maximum(C[exploit_with_pred - 1] - C[exploit_with_pred], 0.0)
        # Squash each per-step contribution with tanh to prevent blow-up
        # when s_H or s_C is very small (near-constant segments)
        per_step = np.tanh(h_diff / s_H) + np.tanh(c_diff / s_C)
        J_X = float(np.mean(per_step))

    exploit_stability = float(np.exp(-J_X))

    # --- 1l. Final score ---
    hmm3_score = float(
        (transition_strength * exploit_quality * exploit_stability) ** (1.0 / 3.0)
    )

    # --- 1m. Return dict ---
    return {
        "transition_strength": transition_strength,
        "exploit_quality": exploit_quality,
        "exploit_stability": exploit_stability,
        "hmm3_score": hmm3_score,
        "n_explore": n_explore,
        "n_exploit": n_exploit,
        "T": T,
        "med_H_E": med_H_E,
        "med_H_X": med_H_X,
        "med_C_E": med_C_E,
        "med_C_X": med_C_X,
        "mean_tau_E": mean_tau_E,
        "mean_tau_X": mean_tau_X,
        "s_H": s_H,
        "s_C": s_C,
        "J_X": J_X,
    }


def summarize_hmmsimple3(entropy, confidence, eps=1e-12, hmm_kwargs=None):
    """Produce a human-readable summary of HMM-Simple-3 analysis.

    Parameters
    ----------
    entropy : array-like
        Entropy time-series.
    confidence : array-like
        Confidence time-series.
    eps : float
        Small constant to prevent division by zero.
    hmm_kwargs : dict or None
        Extra keyword arguments forwarded to ``_hmm_viterbi_2state``.

    Returns
    -------
    dict
        Keys: text, verdict, score, transition_strength, exploit_quality,
        exploit_stability.
    """
    result = compute_hmmsimple3_metrics(
        entropy, confidence, eps=eps, hmm_kwargs=hmm_kwargs,
    )
    score = result["hmm3_score"]

    if score >= 0.7:
        verdict = "good"
    elif score >= 0.5:
        verdict = "borderline"
    else:
        verdict = "poor"

    lines = [
        f"HMM-Simple-3 score: {score:.4f}  [{verdict}]",
        f"  transition_strength : {result['transition_strength']:.4f}",
        f"  exploit_quality     : {result['exploit_quality']:.4f}",
        f"  exploit_stability   : {result['exploit_stability']:.4f}",
    ]
    text = "\n".join(lines)

    return {
        "text": text,
        "verdict": verdict,
        "score": score,
        "transition_strength": result["transition_strength"],
        "exploit_quality": result["exploit_quality"],
        "exploit_stability": result["exploit_stability"],
    }
