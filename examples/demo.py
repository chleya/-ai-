#!/usr/bin/env python3
"""
Cicada Protocol - Demo
=====================
Shows WHY periodic reset is necessary for system stability.

Key insight:
- Without reset: spectral radius (lambda_max) grows -> instability
- With reset: lambda_max stays controlled -> stability

Parameters:
- N: System size (200)
- lr: Learning rate (0.001)
- input: s(t) ~ N(0, 1) Gaussian
- weight_init: W(0) ~ N(0, 0.01)

Run: python demo.py

Author: Chen Leiyang
Date: 2026-02-08
"""

import numpy as np
import matplotlib.pyplot as plt
import sys


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION (Key parameters documented!)
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    'N': 200,              # System size (nodes/neurons)
    'steps': 1000,         # Evolution steps
    'lr': 0.001,           # Learning rate (η)
    'seed': 42,            # Random seed for reproducibility
    'reset_interval': 300, # Reset every 300 steps
}

# ═══════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════

def simulate(N, steps, lr, reset_interval=None, seed=42):
    """
    Run one simulation.
    
    Parameters
    ----------
    N : int
        System size
    steps : int
        Total steps
    lr : float
        Learning rate (η)
    reset_interval : int, optional
        Reset every N steps. None = no reset.
    seed : int
        Random seed
        
    Returns
    -------
    list
        History of λ_max over time
    """
    np.random.seed(seed)
    W = np.random.randn(N, N) * 0.01  # W(0) ~ N(0, 0.01)
    history = []
    
    for t in range(steps):
        # 1. Input: s(t) ~ N(0, 1) Gaussian
        s = np.random.randn(N)
        
        # 2. Hebbian update: W += η × s × s^T
        W += lr * np.outer(s, s)
        
        # 3. Record spectral radius
        eigenvals = np.linalg.eigvalsh(W)
        lambda_max = np.max(np.abs(eigenvals))
        history.append(lambda_max)
        
        # 4. Reset (Cicada moment)
        if reset_interval and (t + 1) % reset_interval == 0:
            W = np.random.randn(N, N) * 0.01
    
    return history


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def plot(no_reset, reset_300, save_path):
    """Create comparison plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Time series
    ax1.plot(no_reset, 'r-', lw=2, label='No Reset')
    ax1.plot(reset_300, 'b-', lw=2, label='Reset 300')
    ax1.axhline(y=1.8, color='orange', ls='--', lw=2, label='Healthy (1.8)')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Spectral Radius (λ_max)')
    ax1.set_title('Spectral Radius Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Bar chart
    names = ['No Reset', 'Reset 300']
    values = [no_reset[-1], reset_300[-1]]
    colors = ['red', 'blue']
    
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


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    """Run the demo."""
    print("=" * 60)
    print("Cicada Protocol Demo")
    print("=" * 60)
    print()
    print("Parameters:")
    print(f"  N = {CONFIG['N']} (system size)")
    print(f"  lr = {CONFIG['lr']} (learning rate)")
    print(f"  steps = {CONFIG['steps']}")
    print(f"  reset_interval = {CONFIG['reset_interval']}")
    print()
    
    # Run simulations
    print("Running simulations...")
    print("  1/2: No Reset...")
    no_reset = simulate(CONFIG['N'], CONFIG['steps'], CONFIG['lr'], None)
    
    print("  2/2: Reset every 300...")
    reset_300 = simulate(CONFIG['N'], CONFIG['steps'], CONFIG['lr'], CONFIG['reset_interval'])
    
    # Results
    print()
    print("-" * 60)
    print("RESULTS:")
    print("-" * 60)
    print("  No Reset     -> λ = {:.2f} (max: {:.2f})".format(
        no_reset[-1], max(no_reset)))
    print("  Reset 300   -> λ = {:.2f} (max: {:.2f})".format(
        reset_300[-1], max(reset_300)))
    print("  Reduction: {:.0f}%".format((1 - reset_300[-1]/no_reset[-1]) * 100))
    
    # Plot
    print()
    print("Creating plot...")
    save_path = plot(no_reset, reset_300, 'cicada_demo_output.png')
    print("  Saved: {}".format(save_path))
    
    # Key insight
    print()
    print("=" * 60)
    print("KEY INSIGHT:")
    print("=" * 60)
    print()
    print("Without reset: λ grows to {:.2f}".format(no_reset[-1]))
    print("With reset:    λ stays at {:.2f}".format(reset_300[-1]))
    print()
    print("This demonstrates WHY periodic reset is necessary!")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        CONFIG['N'] = int(sys.argv[1])
    if len(sys.argv) > 2:
        CONFIG['steps'] = int(sys.argv[2])
    
    main()
