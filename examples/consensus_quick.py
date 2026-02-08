#!/usr/bin/env python3
"""
Distributed Consensus - Quick Test
==================================
"""

import numpy as np
import matplotlib.pyplot as plt


def consensus(N=100, steps=200, lr=0.001, reset=None):
    """Quick consensus test."""
    np.random.seed(42)
    values = np.random.randn(N)
    W = np.random.randn(N, N) * 0.01
    
    errors = []
    lambdas = []
    
    for t in range(steps):
        s = np.random.randn(N)
        W += lr * np.outer(s, s)
        values = 0.9 * values + 0.1 * np.mean(W @ values)
        
        errors.append(np.std(values))
        lambdas.append(np.max(np.abs(np.linalg.eigvalsh(W))))
        
        if reset and (t + 1) % reset == 0:
            W = np.random.randn(N, N) * 0.01
    
    return errors, lambdas


print("=" * 60)
print("Cicada Protocol - Consensus Experiment")
print("=" * 60)
print()

# Quick test
print("Running N=100 experiment...")
errors_no, lambda_no = consensus(N=100, steps=200, reset=None)
errors_reset, lambda_reset = consensus(N=100, steps=200, reset=100)

print(f"No reset:  final error={errors_no[-1]:.4f}, λ_max={lambda_no[-1]:.3f}")
print(f"Reset:     final error={errors_reset[-1]:.4f}, λ_max={lambda_reset[-1]:.3f}")
print()

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(errors_no, 'r-', lw=1.5, label='No reset')
axes[0].plot(errors_reset, 'g-', lw=1.5, label='Reset')
axes[0].set_xlabel('Steps')
axes[0].set_ylabel('Consensus Error')
axes[0].set_title('Convergence')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(lambda_no, 'r-', lw=1.5, label='No reset')
axes[1].plot(lambda_reset, 'g-', lw=1.5, label='Reset')
axes[1].axhline(y=1.8, color='orange', ls='--', label='Threshold')
axes[1].set_xlabel('Steps')
axes[1].set_ylabel('λ_max')
axes[1].set_title('Spectral Radius')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualization/consensus_quick.png', dpi=150)
print("Plot saved to visualization/consensus_quick.png")
