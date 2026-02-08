#!/usr/bin/env python3
"""Alpha Grid Search - Simple Version"""

import numpy as np
import matplotlib.pyplot as plt


def cicada(N=200, steps=500, lr=0.001, reset_interval=None):
    """Quick experiment."""
    np.random.seed(42)
    W = np.random.randn(N, N) * 0.01
    history = []
    
    for t in range(steps):
        s = np.random.randn(N)
        W += lr * np.outer(s, s)
        history.append(np.max(np.abs(np.linalg.eigvalsh(W))))
        if reset_interval and (t + 1) % reset_interval == 0:
            W = np.random.randn(N, N) * 0.01
    
    return history


print("=" * 70)
print("Alpha Grid Search")
print("=" * 70)
print()

N = 200
steps = 500
lr = 0.001
alphas = [1.2, 1.4, 1.6, 1.8, 2.0]

print("-" * 70)
print(f"{'Alpha':<8} {'Final λ':<12} {'Max λ':<12} {'Resets':<10}")
print("-" * 70)

results = []

for alpha in alphas:
    np.random.seed(42)
    W = np.random.randn(N, N) * 0.01
    history = []
    resets = 0
    threshold = 1.8
    
    for t in range(steps):
        s = np.random.randn(N)
        W += lr * np.outer(s, s)
        lam = np.max(np.abs(np.linalg.eigvalsh(W)))
        history.append(lam)
        
        if lam > threshold:
            W = np.random.randn(N, N) * 0.01
            resets += 1
    
    print(f"{alpha:<8} {history[-1]:<12.3f} {max(history):<12.3f} {resets:<10}")
    results.append((alpha, history, resets, max(history)))

# Baselines
print("-" * 70)
no_reset_hist = cicada()
fixed_hist = cicada(reset_interval=200)
print(f"{'No reset':<8} {no_reset_hist[-1]:<12.3f} {max(no_reset_hist):<12.3f} {'0':<10}")
print(f"{'Fixed 200':<8} {fixed_hist[-1]:<12.3f} {max(fixed_hist):<12.3f} {steps//200:<10}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(no_reset_hist, 'k-', lw=2, label='No reset', alpha=0.5)
axes[0].plot(fixed_hist, 'g--', lw=2, label='Fixed 200')
for alpha, history, resets, max_lam in results:
    axes[0].plot(history, lw=1.5, alpha=0.7, label=f'α={alpha}')
axes[0].axhline(y=1.8, color='orange', ls='--', label='Threshold')
axes[0].set_xlabel('Steps')
axes[0].set_ylabel('λ_max')
axes[0].set_title('λ_max Evolution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

max_lams = [r[3] for r in results]
axes[1].bar([f'α={a}' for a in alphas], max_lams, color='blue', alpha=0.8)
axes[1].axhline(y=1.8, color='orange', ls='--', label='Threshold')
axes[1].set_ylabel('Max λ_max')
axes[1].set_title('Max λ_max vs Alpha')
axes[1].legend()

plt.tight_layout()
plt.savefig('visualization/alpha_grid_search.png', dpi=150)
print()
print("Plot saved to visualization/alpha_grid_search.png")

# Find optimal
print()
print("=" * 70)
print("OPTIMAL ALPHA ANALYSIS")
print("=" * 70)

# Best: lowest max_lambda
best = min(results, key=lambda x: x[3])
print(f"Best alpha: {best[0]} (max λ = {best[3]:.3f})")
