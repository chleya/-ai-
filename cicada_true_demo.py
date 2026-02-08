#!/usr/bin/env python3
"""
Cicada Protocol - TRUE Demo (no normalization = real explosion)
================================================================
This properly demonstrates the problem and solution.

Without normalization: spectral radius explodes
With periodic reset: prevents explosion
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List
import warnings
warnings.filterwarnings('ignore')
plt.switch_backend('Agg')


def run_demo(N: int = 200, steps: int = 500, learning_rate: float = 0.1,
             reset_interval: int = None, seed: int = 42) -> List[float]:
    """
    Run demo without normalization - this shows real explosion
    """
    np.random.seed(seed)
    
    # Initialize small random weights
    W = np.random.randn(N, N) * 0.01
    history = []
    
    for t in range(steps):
        # Random input
        s = np.random.randn(N)
        s = s / (np.linalg.norm(s) + 1e-6)
        
        # Hebbian update (causes explosion without normalization!)
        W += learning_rate * np.outer(s, s)
        
        # NO normalization - this is the key!
        
        # Record spectral radius
        eigenvals = np.linalg.eigvalsh(W)
        history.append(np.max(np.abs(eigenvals)))
        
        # Reset if interval specified
        if reset_interval and (t + 1) % reset_interval == 0:
            W = np.random.randn(N, N) * 0.01
    
    return history


def main():
    print("=" * 60)
    print("Cicada Protocol - TRUE Demo (No Normalization)")
    print("=" * 60)
    
    # Config
    N = 100  # Smaller for faster computation
    steps = 500
    lr = 0.05  # Larger learning rate = faster explosion
    
    print(f"\nConfig: N={N}, steps={steps}, lr={lr}")
    print("Note: NO normalization - spectral radius will explode!")
    print("\nRunning...")
    
    # Run three scenarios
    history_no_reset = run_demo(N=N, steps=steps, learning_rate=lr, reset_interval=None)
    history_reset_100 = run_demo(N=N, steps=steps, learning_rate=lr, reset_interval=100)
    history_reset_200 = run_demo(N=N, steps=steps, learning_rate=lr, reset_interval=200)
    
    # Results
    print("\n" + "-" * 60)
    print("RESULTS:")
    print(f"  No Reset - Final lambda: {history_no_reset[-1]:.2f}")
    print(f"  Reset 100 - Final lambda: {history_reset_100[-1]:.2f}")
    print(f"  Reset 200 - Final lambda: {history_reset_200[-1]:.2f}")
    
    print(f"\n  Reduction (100 reset): {history_no_reset[-1] - history_reset_100[-1]:.2f} ({(1-history_reset_100[-1]/history_no_reset[-1])*100:.1f}%)")
    print(f"  Reduction (200 reset): {history_no_reset[-1] - history_reset_200[-1]:.2f} ({(1-history_reset_200[-1]/history_no_reset[-1])*100:.1f}%)")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Full history
    ax1 = axes[0]
    ax1.plot(history_no_reset, 'r-', label='No Reset', linewidth=2)
    ax1.plot(history_reset_100, 'b-', label='Reset every 100', linewidth=2)
    ax1.plot(history_reset_200, 'g-', label='Reset every 200', linewidth=2)
    ax1.axhline(y=5, color='orange', linestyle='--', label='Explosion threshold', alpha=0.7)
    ax1.set_xlabel('Steps', fontsize=12)
    ax1.set_ylabel('Spectral Radius (lambda_max)', fontsize=12)
    ax1.set_title('Spectral Radius Over Time', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final comparison
    ax2 = axes[1]
    finals = [history_no_reset[-1], history_reset_100[-1], history_reset_200[-1]]
    labels = ['No Reset', 'Reset 100', 'Reset 200']
    colors = ['red', 'blue', 'green']
    
    bars = ax2.bar(labels, finals, color=colors, alpha=0.8, edgecolor='black')
    ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.7)
    ax2.set_ylabel('Final Spectral Radius', fontsize=12)
    ax2.set_title('Comparison: Reset Prevents Explosion', fontsize=14)
    
    for bar, val in zip(bars, finals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('cicada_true_demo.png', dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: cicada_true_demo.png")
    
    # Key takeaway
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY:")
    print("=" * 60)
    print("Without reset: spectral radius explodes to {:.1f}".format(history_no_reset[-1]))
    print("With reset (every 100): only {:.1f}".format(history_reset_100[-1]))
    print("\nThis demonstrates WHY periodic reset is necessary!")
    print("=" * 60)


if __name__ == "__main__":
    main()
