#!/usr/bin/env python3
"""
HMM-Simple core — 2-state HMM segmented features and scoring.

The only tunable hyper-parameter is ``transition_weight`` (default 2.0);
all other weights in the composite score are fixed at 1.0.
"""

import numpy as np

EPS = 1e-9


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gaussian_logpdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    """Gaussian log-probability density."""
    x = np.asarray(x, dtype=np.float64)
    var = float(max(var, 1e-6))
    return -0.5 * np.log(2 * np.pi * var) - 0.5 * ((x - mean) ** 2) / var


def _hmm_viterbi_2state(
    obs: np.ndarray,
    p_stay: float = 0.9,
    state0_mean: float = None,
    state1_mean: float = None,
    state0_var: float = None,
    state1_var: float = None,
) -> np.ndarray:
    """
    2-state HMM Viterbi decoder.

    State 0: Exploration (high entropy)
    State 1: Exploitation (low entropy)

    Returns an int32 array of state labels (0 or 1) with length ``len(obs)``.
    """
    obs = np.asarray(obs, dtype=np.float64)
    T = len(obs)

    if T == 0:
        return np.array([], dtype=np.int32)

    # Auto-estimate emission parameters
    if state0_mean is None:
        state0_mean = np.percentile(obs, 75)
    if state1_mean is None:
        state1_mean = np.percentile(obs, 25)
    if state0_var is None:
        state0_var = np.var(obs) * 0.5
    if state1_var is None:
        state1_var = np.var(obs) * 0.5

    # Transition log-probabilities
    p_switch = 1.0 - p_stay
    log_p_stay = np.log(max(p_stay, 1e-9))
    log_p_switch = np.log(max(p_switch, 1e-9))

    # Emission log-probabilities
    logprob0 = _gaussian_logpdf(obs, state0_mean, state0_var)
    logprob1 = _gaussian_logpdf(obs, state1_mean, state1_var)

    # Viterbi forward pass
    log_delta = np.zeros((T, 2), dtype=np.float64)
    psi = np.zeros((T, 2), dtype=np.int32)

    log_delta[0, 0] = logprob0[0]
    log_delta[0, 1] = logprob1[0]

    for t in range(1, T):
        for s in (0, 1):
            val0 = log_delta[t - 1, 0] + (log_p_stay if s == 0 else log_p_switch)
            val1 = log_delta[t - 1, 1] + (log_p_switch if s == 0 else log_p_stay)

            if val0 > val1:
                log_delta[t, s] = val0 + (logprob0[t] if s == 0 else logprob1[t])
                psi[t, s] = 0
            else:
                log_delta[t, s] = val1 + (logprob0[t] if s == 0 else logprob1[t])
                psi[t, s] = 1

    # Backtrace
    states = np.zeros(T, dtype=np.int32)
    states[T - 1] = int(np.argmax(log_delta[T - 1]))

    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_hmm_features(H, C, p_stay=0.9):
    """Compute 22 HMM-segmented explore/exploit features.

    Parameters
    ----------
    H : array-like
        Entropy time-series.
    C : array-like
        Confidence time-series.
    p_stay : float
        HMM self-transition probability (default 0.9).

    Returns
    -------
    dict
        22-key feature dictionary.

    Raises
    ------
    ValueError
        If inputs contain NaN/inf or are empty.
    """
    H = np.asarray(H, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    # Input validation
    if H.size == 0 or C.size == 0:
        raise ValueError("H and C must be non-empty arrays")
    if np.any(np.isnan(H)) or np.any(np.isnan(C)):
        raise ValueError("H and C must not contain NaN values")
    if np.any(np.isinf(H)) or np.any(np.isinf(C)):
        raise ValueError("H and C must not contain inf values")

    T = len(H)
    if T < 3:
        return None

    # HMM segmentation
    states = _hmm_viterbi_2state(H, p_stay=p_stay)

    explore_mask = (states == 0)
    exploit_mask = (states == 1)

    n_explore = int(np.sum(explore_mask))
    n_exploit = int(np.sum(exploit_mask))

    # Fallback if HMM assigns everything to one state
    if n_explore == 0 or n_exploit == 0:
        K = max(1, T // 4)
        explore_mask = np.zeros(T, dtype=bool)
        explore_mask[:K] = True
        exploit_mask = ~explore_mask
        n_explore = int(np.sum(explore_mask))
        n_exploit = int(np.sum(exploit_mask))

    # Segment statistics
    H_explore = H[explore_mask]
    C_explore = C[explore_mask]
    explore_H_mean = float(np.mean(H_explore))
    explore_C_mean = float(np.mean(C_explore))
    explore_H_std = float(np.std(H_explore))
    explore_C_std = float(np.std(C_explore))

    H_exploit = H[exploit_mask]
    C_exploit = C[exploit_mask]
    exploit_H_mean = float(np.mean(H_exploit))
    exploit_C_mean = float(np.mean(C_exploit))
    exploit_H_std = float(np.std(H_exploit))
    exploit_C_std = float(np.std(C_exploit))

    # Transition quality
    entropy_reduction = explore_H_mean - exploit_H_mean
    confidence_gain = exploit_C_mean - explore_C_mean
    transition_quality = np.tanh(entropy_reduction) + np.tanh(confidence_gain / 10.0)

    # Exploration efficiency
    explore_fraction = n_explore / T
    explore_intensity = explore_H_std / max(explore_H_mean, EPS)
    efficiency = transition_quality / max(np.log2(n_explore + 1), 1.0)

    # Exploitation stability
    exploit_stability = 1.0 / (1.0 + exploit_H_std)

    if n_exploit > 0:
        exploit_C_start = C_exploit[0]
        exploit_C_end = C_exploit[-1]
        exploit_improvement = exploit_C_end - exploit_C_start
    else:
        exploit_improvement = 0.0

    # Transition count
    n_transitions = int(np.sum(np.diff(states) != 0))
    transition_penalty = np.tanh(n_transitions / 5.0)

    # Overall gain
    overall_H_drop = H[0] - H[-1]
    overall_C_gain = C[-1] - C[0]

    return {
        "n_explore": n_explore,
        "n_exploit": n_exploit,
        "explore_fraction": explore_fraction,
        "n_transitions": n_transitions,
        "explore_H_mean": explore_H_mean,
        "explore_C_mean": explore_C_mean,
        "explore_H_std": explore_H_std,
        "explore_C_std": explore_C_std,
        "explore_intensity": explore_intensity,
        "exploit_H_mean": exploit_H_mean,
        "exploit_C_mean": exploit_C_mean,
        "exploit_H_std": exploit_H_std,
        "exploit_C_std": exploit_C_std,
        "exploit_stability": exploit_stability,
        "exploit_improvement": exploit_improvement,
        "entropy_reduction": entropy_reduction,
        "confidence_gain": confidence_gain,
        "transition_quality": transition_quality,
        "efficiency": efficiency,
        "transition_penalty": transition_penalty,
        "overall_H_drop": overall_H_drop,
        "overall_C_gain": overall_C_gain,
        "T": T,
    }


def compute_hmm_score(H, C, p_stay=0.9, transition_weight=2.0):
    """One-shot HMM-Simple scoring: features + composite score.

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
        All 22 features plus a ``score`` key (float in [0, 1]).
        For T < 3, returns ``{"score": 0.5}``.

    Raises
    ------
    ValueError
        If inputs contain NaN/inf or are empty.
    """
    features = compute_hmm_features(H, C, p_stay=p_stay)

    if features is None:
        return {"score": 0.5}

    # Composite score (only transition_weight is tuneable)
    transition_term = np.tanh(transition_weight * features["transition_quality"])
    efficiency_term = np.tanh(3.0 * features["efficiency"])
    stability_term = np.tanh(features["exploit_stability"])
    improvement_term = np.tanh(features["exploit_improvement"] / 10.0)
    penalty_term = features["transition_penalty"]
    gain_term = (
        np.tanh(features["overall_H_drop"])
        + np.tanh(features["overall_C_gain"] / 10.0)
    ) * 0.5

    composite_raw = (
        transition_weight * transition_term
        + 1.0 * efficiency_term
        + 1.0 * stability_term
        + 1.0 * improvement_term
        + 1.0 * gain_term
        - 1.0 * penalty_term
    )

    score = float(1.0 / (1.0 + np.exp(-3.0 * composite_raw)))

    result = dict(features)
    result["score"] = score
    return result
