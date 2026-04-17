#!/usr/bin/env python3
"""
Fusion score: F = log(1 + I) * Q

I = intensity kernel  — positive-progress terms from HMM-Simple (T1+T2+T4+T5)
Q = credibility kernel — geometric mean of HMM-Simple3's three primitives

The fusion decouples "how much progress" (I) from "how structurally credible"
(Q), avoiding the double-counting of stability and transition penalties that
would occur if the full Simple composite were multiplied by Q.

No new hyper-parameters are introduced: all coefficients are inherited from
the original Simple and Simple3 implementations.
"""

import numpy as np

EPS = 1e-9


def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _softplus(x):
    """Numerically stable softplus: log(1 + exp(x))."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -500, 20))))


def compute_fusion_score(H, C, explore_mask, exploit_mask, eps=1e-12):
    """Compute Fusion score F = log(1+I) * Q.

    Parameters
    ----------
    H : array-like
        Entropy time-series.
    C : array-like
        Confidence time-series.
    explore_mask : array-like of bool
        Boolean mask for explore states.
    exploit_mask : array-like of bool
        Boolean mask for exploit states.
    eps : float
        Small constant to prevent division by zero in Q kernel.

    Returns
    -------
    dict
        Keys include all intermediate quantities and the final fusion_score:
        - I kernel: transition_quality, efficiency, exploit_improvement,
          overall_H_drop, overall_C_gain, T1, T2, T4, T5, I_raw, I
        - Q kernel: transition_strength, exploit_quality, exploit_stability,
          Q, s_H, s_C, J_X
        - Fusion: fusion_score
        - Diagnostics: n_explore, n_exploit, T

    Raises
    ------
    ValueError
        If inputs contain NaN/inf or are empty.
    """
    H = np.asarray(H, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    explore_mask = np.asarray(explore_mask, dtype=bool)
    exploit_mask = np.asarray(exploit_mask, dtype=bool)

    # Input validation
    if H.size == 0 or C.size == 0:
        raise ValueError("H and C must be non-empty arrays")
    if np.any(np.isnan(H)) or np.any(np.isnan(C)):
        raise ValueError("H and C must not contain NaN values")
    if np.any(np.isinf(H)) or np.any(np.isinf(C)):
        raise ValueError("H and C must not contain inf values")

    T = len(H)
    C_clipped = np.clip(C, 0.0, 1.0)

    n_explore = int(np.sum(explore_mask))
    n_exploit = int(np.sum(exploit_mask))

    # Degenerate fallback: if masks are all-one-state, split heuristically
    if n_explore == 0 or n_exploit == 0:
        K = max(1, T // 4)
        explore_mask = np.zeros(T, dtype=bool)
        explore_mask[:K] = True
        exploit_mask = ~explore_mask
        n_explore = int(np.sum(explore_mask))
        n_exploit = int(np.sum(exploit_mask))

    # =====================================================================
    # I kernel — intensity (positive-progress terms from Simple)
    # =====================================================================
    H_explore = H[explore_mask]
    C_explore = C[explore_mask]
    H_exploit = H[exploit_mask]
    C_exploit = C[exploit_mask]

    explore_H_mean = float(np.mean(H_explore))
    explore_C_mean = float(np.mean(C_explore))
    exploit_H_mean = float(np.mean(H_exploit))
    exploit_C_mean = float(np.mean(C_exploit))

    # transition_quality (same as Simple)
    entropy_reduction = explore_H_mean - exploit_H_mean
    confidence_gain = exploit_C_mean - explore_C_mean
    transition_quality = float(
        np.tanh(entropy_reduction) + np.tanh(confidence_gain / 10.0)
    )

    # efficiency (same as Simple)
    efficiency = transition_quality / max(np.log2(n_explore + 1), 1.0)

    # exploit_improvement (same as Simple)
    if n_exploit > 0:
        exploit_improvement = float(C_exploit[-1] - C_exploit[0])
    else:
        exploit_improvement = 0.0

    # overall gains (same as Simple)
    overall_H_drop = float(H[0] - H[-1])
    overall_C_gain = float(C[-1] - C[0])

    # T1: transition term (coefficient 2.0, same as Simple)
    T1 = 2.0 * float(np.tanh(2.0 * transition_quality))

    # T2: efficiency term
    T2 = float(np.tanh(3.0 * efficiency))

    # T4: improvement term
    T4 = float(np.tanh(exploit_improvement / 10.0))

    # T5: gain term
    T5 = 0.5 * float(np.tanh(overall_H_drop) + np.tanh(overall_C_gain / 10.0))

    I_raw = T1 + T2 + T4 + T5
    I = max(I_raw, 0.0)

    # =====================================================================
    # Q kernel — credibility (Simple3's three primitives, geometric mean)
    # =====================================================================
    E = explore_mask
    X = exploit_mask

    # Global robust statistics
    med_H_all = float(np.median(H))
    med_C_all = float(np.median(C_clipped))
    s_H = float(np.median(np.abs(H - med_H_all)) + eps)
    s_C = float(np.median(np.abs(C_clipped - med_C_all)) + eps)

    # Temporal position
    tau = np.arange(T, dtype=np.float64) / max(T - 1, 1)

    # Per-state statistics
    med_H_E = float(np.median(H[E]))
    med_H_X = float(np.median(H[X]))
    med_C_E = float(np.median(C_clipped[E]))
    med_C_X = float(np.median(C_clipped[X]))
    mean_tau_E = float(np.mean(tau[E]))
    mean_tau_X = float(np.mean(tau[X]))

    # Metric 1 — transition_strength
    ts_raw = (
        (med_H_E - med_H_X) / s_H
        + (med_C_X - med_C_E) / s_C
        + (mean_tau_X - mean_tau_E)
    )
    transition_strength = float(_sigmoid(ts_raw))

    # Metric 2 — exploit_quality
    eq_raw = (
        (med_C_X - med_C_all) / s_C
        + (med_H_all - med_H_X) / s_H
    )
    exploit_quality = float(_sigmoid(eq_raw))

    # Metric 3 — exploit_stability
    exploit_indices = np.where(X)[0]
    exploit_with_pred = exploit_indices[exploit_indices >= 1]

    if len(exploit_with_pred) == 0:
        J_X = 0.0
    else:
        h_diff = np.maximum(H[exploit_with_pred] - H[exploit_with_pred - 1], 0.0)
        c_diff = np.maximum(
            C_clipped[exploit_with_pred - 1] - C_clipped[exploit_with_pred], 0.0
        )
        per_step = np.tanh(h_diff / s_H) + np.tanh(c_diff / s_C)
        J_X = float(np.mean(per_step))

    exploit_stability = float(np.exp(-J_X))

    Q = float(
        (transition_strength * exploit_quality * exploit_stability) ** (1.0 / 3.0)
    )

    # =====================================================================
    # Fusion: F = log(1 + I) * Q
    # =====================================================================
    fusion_score = float(np.log1p(I) * Q)

    return {
        # I kernel intermediates
        "transition_quality": transition_quality,
        "efficiency": efficiency,
        "exploit_improvement": exploit_improvement,
        "overall_H_drop": overall_H_drop,
        "overall_C_gain": overall_C_gain,
        "T1": T1,
        "T2": T2,
        "T4": T4,
        "T5": T5,
        "I_raw": I_raw,
        "I": I,
        # Q kernel intermediates
        "transition_strength": transition_strength,
        "exploit_quality": exploit_quality,
        "exploit_stability": exploit_stability,
        "Q": Q,
        "s_H": s_H,
        "s_C": s_C,
        "J_X": J_X,
        # Fusion
        "fusion_score": fusion_score,
        # Diagnostics
        "n_explore": n_explore,
        "n_exploit": n_exploit,
        "T": T,
    }


def compute_fusion_v2_score(H, C, explore_mask, exploit_mask, eps=1e-12):
    """Compute Fusion v2 score F = log(1 + softplus(I_raw) * sqrt(eq * es)).

    Also returns ablation variants:
      - I_smooth:        log(1 + softplus(I_raw))          — pure intensity
      - fusion_all_score: log(1 + softplus(I_raw) * Q)     — ts included

    Parameters
    ----------
    H, C, explore_mask, exploit_mask, eps :
        Same as compute_fusion_score.

    Returns
    -------
    dict
        All keys from compute_fusion_score plus:
        - A            : softplus(I_raw)
        - C_cred       : sqrt(exploit_quality * exploit_stability)
        - fusion_v2_score : log(1 + A * C_cred)
        - I_smooth     : log(1 + A)
        - fusion_all_score : log(1 + A * Q)
    """
    v1 = compute_fusion_score(H, C, explore_mask, exploit_mask, eps)

    A = float(_softplus(np.float64(v1["I_raw"])))
    C_cred = float(np.sqrt(v1["exploit_quality"] * v1["exploit_stability"]))

    fusion_v2_score = float(np.log1p(A * C_cred))
    I_smooth = float(np.log1p(A))
    fusion_all_score = float(np.log1p(A * v1["Q"]))

    return {
        **v1,
        "A": A,
        "C_cred": C_cred,
        "fusion_v2_score": fusion_v2_score,
        "I_smooth": I_smooth,
        "fusion_all_score": fusion_all_score,
    }
