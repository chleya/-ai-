#!/usr/bin/env python3
"""
Cicada Protocol - Simple Demo
============================
Shows WHY periodic reset is necessary for system stability.

Key insight:
- Without reset: spectral radius (lambda_max) grows -> instability
- With reset: lambda_max stays healthy -> stability

Run: python demo.py

Author: Chen Leiyang
Date: 2026-02-08
"""

import numpy as np
import matplotlib.pyplot as plt
import sys


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

CONFIG = {
    'N': 100,              # System size (nodes)
    'steps': 500,           # Evolution steps
    'learning_rate': 0.05,  # Weight change rate
    'seed': 42,            # Random seed
}


# ═══════════════════════════════════════════════════════════════════════
# Core Simulation
# ═══════════════════════════════════════════════════════════════════════

def simulate(N, steps, lr, reset_interval=None, seed=42):
    """
    Run one simulation with optional reset.
    
    Parameters
    ----------
    N : int
        System size (number of nodes)
    steps : int
        Total simulation steps
    lr : float
        Learning rate
    reset_interval : int, optional
        Reset every N steps. None = no reset.
    seed : int
        Random seed
        
    Returns
    -------
    list
        History of spectral radius over time
    """
    np.random.seed(seed)
    W = np.random.randn(N, N) * 0.01  # Start small
    history = []
    
    for t in range(steps):
        # 1. Random input
        s = np.random.randn(N)
        s = s / np.linalg.norm(s)
        
        # 2. Hebbian update (causes growth without reset!)
        W += lr * np.outer(s, s)
        
        # 3. Record spectral radius
        eigenvals = np.linalg.eigvalsh(W)
        lambda_max = np.max(np.abs(eigenvals))
        history.append(lambda_max)
        
        # 4. Reset (Cicada moment)
        if reset_interval and (t + 1) % reset_interval == 0:
            W = np.random.randn(N, N) * 0.01
    
    return history


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot(no_reset, reset_100, reset_200, save_path):
    """Create comparison plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ── Left: Time series ───────────────────────────────────────
    ax1.plot(no_reset, 'r-', lw=2, label='No Reset')
    ax1.plot(reset_100, 'b-', lw=2, label='Reset 100')
    ax1.plot(reset_200, 'g-', lw=2, label='Reset 200')
    ax1.axhline(y=1.8, color='orange', ls='--', lw=2, label='Healthy (1.8)')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Spectral Radius (λ_max)')
    ax1.set_title('Spectral Radius Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ── Right: Bar chart ─────────────────────────────────────────
    names = ['No Reset', 'Reset 100', 'Reset 200']
    values = [no_reset[-1], reset_100[-1], reset_200[-1]]
    colors = ['red', 'blue', 'green']
    
    bars = ax2.bar(names, values, color=colors, edgecolor='black', alpha=0.8)
    ax2.axhline(y=1.8, color='orange', ls='--', lw=2)
    ax2.set_ylabel('Final λ_max')
    ax2.set_title('Final λ Comparison')
    
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Run the demo."""
    print("=" * 60)
    print("Cicada Protocol Demo")
    print("=" * 60)
    print()
    print("Config: N={}, steps={}, lr={}".format(
        CONFIG['N'], CONFIG['steps'], CONFIG['learning_rate']))
    print()
    
    # Run simulations
    print("Running simulations...")
    print("  1/3: No Reset...")
    no_reset = simulate(CONFIG['N'], CONFIG['steps'], CONFIG['learning_rate'], None)
    
    print("  2/3: Reset every 100...")
    reset_100 = simulate(CONFIG['N'], CONFIG['steps'], CONFIG['learning_rate'], 100)
    
    print("  3/3: Reset every 200...")
    reset_200 = simulate(CONFIG['N'], CONFIG['steps'], CONFIG['learning_rate'], 200)
    
    # Results
    print()
    print("-" * 60)
    print("RESULTS:")
    print("-" * 60)
    print("  No Reset     -> λ = {:.2f}".format(no_reset[-1]))
    print("  Reset 100    -> λ = {:.2f} (↓{:.0f}%)".format(
        reset_100[-1], (1 - reset_100[-1]/no_reset[-1]) * 100))
    print("  Reset 200    -> λ = {:.2f} (↓{:.0f}%)".format(
        reset_200[-1], (1 - reset_200[-1]/no_reset[-1]) * 100))
    
    # Plot
    print()
    print("Creating plot...")
    save_path = plot(no_reset, reset_100, reset_200, 'cicada_demo_output.png')
    print("  Saved: {}".format(save_path))
    
    # Key insight
    print()
    print("=" * 60)
    print("KEY INSIGHT:")
    print("=" * 60)
    print()
    print("Without reset: λ grows to {:.2f}".format(no_reset[-1]))
    print("With reset:    λ stays at {:.2f}".format(reset_100[-1]))
    print()
    print("This demonstrates WHY periodic reset is necessary!")
    print("=" * 60)


if __name__ == "__main__":
    # Command-line overrides: python demo.py N steps
    if len(sys.argv) > 1:
        CONFIG['N'] = int(sys.argv[1])
    if len(sys.argv) > 2:
        CONFIG['steps'] = int(sys.argv[2])
    
    main()
