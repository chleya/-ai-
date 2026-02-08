#!/usr/bin/env python3
"""
Cicada Protocol - Standalone Demo
=================================
This demo runs WITHOUT requiring pip install or complex setup.
Just run: python demo.py

Shows how periodic reset prevents spectral radius explosion.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Avoid tkinter warnings
import warnings
warnings.filterwarnings('ignore')
plt.switch_backend('Agg')

print("=" * 60)
print("Cicada Protocol - Standalone Demo")
print("=" * 60)
print()

# =============================================================================
# Minimal Implementation (inline to ensure it works)
# =============================================================================

def run_demo(N=100, steps=500, lr=0.05, reset_interval=None, seed=42):
    """
    Run a demo without normalization - shows real spectral explosion.
    
    Args:
        N: System size
        steps: Evolution steps
        lr: Learning rate
        reset_interval: Reset interval (None = no reset)
        seed: Random seed
    """
    np.random.seed(seed)
    
    # Initialize small weights
    W = np.random.randn(N, N) * 0.01
    history = []
    
    for t in range(steps):
        # Random input
        s = np.random.randn(N)
        s = s / (np.linalg.norm(s) + 1e-6)
        
        # Hebbian update (causes explosion without normalization!)
        W += lr * np.outer(s, s)
        
        # NO normalization - this is the key!
        
        # Record spectral radius
        eigenvals = np.linalg.eigvalsh(W)
        history.append(np.max(np.abs(eigenvals)))
        
        # Reset if interval specified
        if reset_interval and (t + 1) % reset_interval == 0:
            W = np.random.randn(N, N) * 0.01
    
    return history


# =============================================================================
# Run Demo
# =============================================================================

print("Configuration:")
print(f"  N = 100")
print(f"  steps = 500")
print(f"  learning_rate = 0.05")
print()
print("Running experiments...")
print()

# Run three scenarios
print("  1/3: No Reset...")
history_no_reset = run_demo(N=100, steps=500, lr=0.05, reset_interval=None)

print("  2/3: Reset every 100 steps...")
history_reset_100 = run_demo(N=100, steps=500, lr=0.05, reset_interval=100)

print("  3/3: Reset every 200 steps...")
history_reset_200 = run_demo(N=100, steps=500, lr=0.05, reset_interval=200)

print()
print("-" * 60)
print("RESULTS:")
print(f"  No Reset     -> Final λ: {history_no_reset[-1]:.2f}")
print(f"  Reset 100    -> Final λ: {history_reset_100[-1]:.2f} (↓{(1-history_reset_100[-1]/history_no_reset[-1])*100:.0f}%)")
print(f"  Reset 200    -> Final λ: {history_reset_200[-1]:.2f} (↓{(1-history_reset_200[-1]/history_no_reset[-1])*100:.0f}%)")
print()

# =============================================================================
# Visualization
# =============================================================================

print("Generating plot: cicada_demo_output.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Evolution
ax1.plot(history_no_reset, 'r-', label='No Reset', linewidth=2)
ax1.plot(history_reset_100, 'b-', label='Reset every 100', linewidth=2)
ax1.plot(history_reset_200, 'g-', label='Reset every 200', linewidth=2)
ax1.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Explosion threshold')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Spectral Radius (λ_max)')
ax1.set_title('Spectral Radius Evolution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Comparison
strategies = ['No Reset', 'Reset 100', 'Reset 200']
finals = [history_no_reset[-1], history_reset_100[-1], history_reset_200[-1]]
colors = ['red', 'blue', 'green']

bars = ax2.bar(strategies, finals, color=colors, alpha=0.8, edgecolor='black')
ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.7)
ax2.set_ylabel('Final Spectral Radius')
ax2.set_title('Final λ Comparison')

for bar, val in zip(bars, finals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val:.2f}', ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('cicada_demo_output.png', dpi=150, bbox_inches='tight')

print()
print("=" * 60)
print("KEY INSIGHT:")
print("=" * 60)
print(f"Without reset: λ grows to {history_no_reset[-1]:.2f}")
print(f"With reset:    λ stays at {history_reset_100[-1]:.2f}")
print()
print("This demonstrates WHY periodic reset is necessary!")
print("=" * 60)
print()
print(f"Plot saved: {os.path.abspath('cicada_demo_output.png')}")
print()
print("Next steps:")
print("  - Try different parameters: python demo.py --N 200 --steps 1000")
print("  - See cicada_minimal.py for more options")
print("  - Read papers/CICADA_PAPER.md for theory")
