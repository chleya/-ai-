#!/usr/bin/env python3
"""
Cicada Protocol - Core Module
=============================
A simple, readable implementation of periodic reset for system stability.

Key concepts:
- lambda_max: Spectral radius (core stability metric)
- Point attractor: Stable state (healthy, lambda_max < 1.8)
- Reset: Periodic restart that prevents instability

Usage:
    from cicada.core import cicada_protocol, analyze_spectrum
    
    W, s, history = cicada_protocol(N=200, reset_interval=300)
    spectrum = analyze_spectrum(W)

Author: Chen Leiyang
Date: 2026-02-08
"""

import numpy as np
from typing import List, Dict, Tuple


# ═══════════════════════════════════════════════════════════════════════
# Core Protocol
# ═══════════════════════════════════════════════════════════════════════

def cicada_protocol(
    N: int = 200,
    reset_interval: int = 300,
    steps: int = 800,
    seed: int = None,
    lr: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Run Cicada Protocol with periodic reset.
    
    The protocol works by periodically resetting the weight matrix W
    to prevent spectral radius (lambda_max) from growing too large.
    
    Parameters
    ----------
    N : int
        System size (number of nodes/neurons)
    reset_interval : int
        Reset every N steps. Smaller = more frequent resets.
    steps : int
        Total evolution steps
    seed : int, optional
        Random seed for reproducibility
    lr : float
        Learning rate (how fast weights change)
    
    Returns
    -------
    tuple
        (W, s, lambda_history)
        W: Final weight matrix (N x N)
        s: Final system state (N,)
        lambda_history: List of lambda_max over time
    
    Example
    -------
    >>> W, s, history = cicada_protocol(N=200, reset_interval=300)
    >>> print(f"Final lambda_max: {history[-1]:.4f}")
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize weight matrix (Xavier initialization)
    W = np.random.randn(N, N) * np.sqrt(2.0 / N)
    s = np.random.randn(N)
    
    # Track lambda_max over time
    lambda_history: List[float] = []
    
    for t in range(steps):
        # ── Step 1: Generate input with noise ───────────────────────
        s = 0.5 * s + 0.5 * np.random.randn(N)
        
        # ── Step 2: Apply weights ──────────────────────────────────
        s = np.tanh(W @ s + 0.2 * s)
        
        # ── Step 3: Hebbian update ─────────────────────────────────
        W = W + lr * np.outer(s, s)
        
        # ── Step 4: Normalize to prevent explosion ───────────────────
        if np.linalg.norm(W) > 10:
            W = W / np.linalg.norm(W) * 10
        
        # ── Step 5: Record lambda_max ─────────────────────────────────
        eigenvals = np.linalg.eigvalsh(W)
        lambda_max = np.max(np.abs(eigenvals))
        lambda_history.append(lambda_max)
        
        # ── Step 6: Reset (Cicada moment) ──────────────────────────────
        # This is the KEY mechanism that maintains stability!
        if (t + 1) % reset_interval == 0:
            W = np.random.randn(N, N) * np.sqrt(2.0 / N)
            s = np.random.randn(N)
    
    return W, s, lambda_history


def event_triggered_protocol(
    N: int = 200,
    steps: int = 800,
    seed: int = None,
    alpha: float = 1.5,
    window: int = 20,
    lr: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[int]]:
    """
    Run Event-Triggered Cicada Protocol.
    
    Reset is triggered when system jitter exceeds a threshold,
    rather than at fixed intervals.
    
    Parameters
    ----------
    N : int
        System size
    steps : int
        Total steps
    seed : int, optional
        Random seed
    alpha : float
        Trigger threshold multiplier. Optimal ≈ 1.6.
    window : int
        Window size for jitter calculation
    lr : float
        Learning rate
    
    Returns
    -------
    tuple
        (W, s, lambda_history, reset_points)
    """
    if seed is not None:
        np.random.seed(seed)
    
    W = np.random.randn(N, N) * np.sqrt(2.0 / N)
    s = np.random.randn(N)
    
    lambda_history: List[float] = []
    jitter_history: List[float] = []
    reset_points: List[int] = []
    
    for t in range(steps):
        # Evolution (same as basic protocol)
        s = 0.5 * s + 0.5 * np.random.randn(N)
        s = np.tanh(W @ s + 0.2 * s)
        W = W + lr * np.outer(s, s)
        
        if np.linalg.norm(W) > 10:
            W = W / np.linalg.norm(W) * 10
        
        # Record
        eigenvals = np.linalg.eigvalsh(W)
        lambda_max = np.max(np.abs(eigenvals))
        lambda_history.append(lambda_max)
        
        jitter = np.std(s)
        jitter_history.append(jitter)
        
        # Event-triggered reset
        if len(jitter_history) > window:
            current_jitter = jitter_history[-1]
            mean_jitter = np.mean(jitter_history[-window:])
            
            if current_jitter > alpha * mean_jitter:
                W = np.random.randn(N, N) * np.sqrt(2.0 / N)
                s = np.random.randn(N)
                reset_points.append(t)
                jitter_history = []
    
    return W, s, lambda_history, reset_points


# ═══════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze_spectrum(W: np.ndarray) -> Dict[str, float]:
    """
    Analyze weight matrix spectral properties.
    
    Parameters
    ----------
    W : np.ndarray
        Weight matrix (N x N)
    
    Returns
    -------
    dict
        - max: Largest eigenvalue (lambda_max)
        - min: Smallest eigenvalue
        - ratio: max / |min|
        - spread: max - min
        - mean: Mean eigenvalue
        - std: Std of eigenvalues
    
    Note
    ----
    Key thresholds:
    - lambda_max < 1.8: Healthy (stable)
    - lambda_max > 2.0: Warning (unstable)
    """
    eigenvals = np.linalg.eigvalsh(W)
    
    return {
        'max': float(np.max(eigenvals)),
        'min': float(np.min(eigenvals)),
        'ratio': float(eigenvals[-1] / (abs(eigenvals[0]) + 1e-6)),
        'spread': float(eigenvals[-1] - eigenvals[0]),
        'mean': float(np.mean(eigenvals)),
        'std': float(np.std(eigenvals))
    }


def calculate_survival_rate(lambda_history: List[float], threshold: float = 1.8) -> Dict[str, float]:
    """
    Calculate survival rate - proportion of healthy steps.
    
    Parameters
    ----------
    lambda_history : List[float]
        History of lambda_max over time
    threshold : float
        Health threshold (default: 1.8)
    
    Returns
    -------
    dict
        - survival_rate: Proportion of healthy steps
        - healthy_steps: Count of healthy steps
        - total_steps: Total steps
    """
    healthy = sum(1 for l in lambda_history if l < threshold)
    total = len(lambda_history)
    
    return {
        'survival_rate': healthy / total,
        'healthy_steps': healthy,
        'total_steps': total
    }


# ═══════════════════════════════════════════════════════════════════════
# Experiments
# ═══════════════════════════════════════════════════════════════════════

def run_experiment(
    N: int = 200,
    reset_interval: int = 300,
    steps: int = 500,
    trials: int = 10,
    protocol: str = 'fixed',
    **kwargs
) -> Dict:
    """
    Run multiple experimental trials.
    
    Parameters
    ----------
    N : int
        System size
    reset_interval : int
        Reset interval (for fixed protocol)
    steps : int
        Steps per trial
    trials : int
        Number of trials
    protocol : str
        'fixed' or 'event'
    **kwargs
        Additional protocol parameters
    
    Returns
    -------
    dict
        Experiment results
    """
    successes = 0
    lambda_finals = []
    lambda_maxes = []
    
    for trial in range(trials):
        if protocol == 'fixed':
            _, _, history = cicada_protocol(
                N=N, reset_interval=reset_interval, steps=steps, seed=trial
            )
            resets = steps // reset_interval
        else:  # event
            _, _, history, _ = event_triggered_protocol(
                N=N, steps=steps, seed=trial, **kwargs
            )
            resets = 0  # Not tracked for event protocol
        
        if np.mean(history[-10:]) > 0:  # Consensus achieved
            successes += 1
        
        lambda_finals.append(history[-1])
        lambda_maxes.append(max(history))
    
    return {
        'success_rate': successes / trials,
        'mean_lambda_final': np.mean(lambda_finals),
        'mean_lambda_max': np.mean(lambda_maxes),
        'mean_resets': resets if protocol == 'fixed' else None
    }


# ═══════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Cicada Protocol - Core Demo")
    print("=" * 60)
    print()
    
    # Run protocol
    print("Running protocol (N=200, interval=300, steps=800)...")
    W, s, history = cicada_protocol(N=200, reset_interval=300, steps=800)
    
    # Analyze
    spectrum = analyze_spectrum(W)
    survival = calculate_survival_rate(history)
    
    print()
    print("Results:")
    print("  lambda_max: {:.4f}".format(spectrum['max']))
    print("  survival_rate: {:.1%}".format(survival['survival_rate']))
    
    print()
    print("=" * 60)
