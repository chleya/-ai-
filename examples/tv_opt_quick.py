#!/usr/bin/env python3
"""Time-Varying Optimization - Quick Test"""

import numpy as np
import matplotlib.pyplot as plt
import json


def hebbian_update(W, s, eta):
    return W + eta * np.outer(s, s)


def lambda_max_approx(W, iters=5):
    v = np.random.rand(W.shape[0])
    for _ in range(iters):
        v = W @ v
        v /= np.linalg.norm(v)
    return np.abs(v @ (W @ v))


def time_varying_opt(K=3, N=300, steps=100, eta=0.01, theta=1.2, cicada=True, seed=42):
    np.random.seed(seed)
    W_list = [np.random.randn(N, N) * 0.1 for _ in range(K)]
    errors, lams = [], []
    
    for t in range(steps):
        opt = np.sin(t/10) * np.ones(N)
        for k in range(K):
            s = np.random.randn(N) + 0.1 * opt
            W_list[k] = hebbian_update(W_list[k], s, eta)
        
        state = np.mean([W @ np.random.randn(N) for W in W_list], axis=0)
        err = np.linalg.norm(state - opt)
        errors.append(err)
        
        lam = np.mean([lambda_max_approx(W) for W in W_list])
        lams.append(lam)
        
        if cicada and lam > theta:
            for k in range(K):
                W_list[k] = np.random.randn(N, N) * 0.1
    
    return errors, lams


print("=" * 60)
print("TIME-VARYING OPTIMIZATION - QUICK TEST")
print("=" * 60)

# Test
w_err, w_lam = time_varying_opt(K=3, N=300, cicada=True)
wo_err, wo_lam = time_varying_opt(K=3, N=300, cicada=False)

imp = (np.mean(wo_err[-20:]) - np.mean(w_err[-20:])) / np.mean(wo_err[-20:]) * 100

print(f"With Cicada:    avg_error={np.mean(w_err[-20:]):.4f}, avg_lambda={np.mean(w_lam[-20:]):.4f}")
print(f"Without Cicada: avg_error={np.mean(wo_err[-20:]):.4f}, avg_lambda={np.mean(wo_lam[-20:]):.4f}")
print(f"Improvement: {imp:.1f}%")

# Save
data = {
    'with': {'error': w_err, 'lambda': w_lam, 'avg_error': np.mean(w_err[-20:])},
    'without': {'error': wo_err, 'lambda': wo_lam, 'avg_error': np.mean(wo_err[-20:])},
    'improvement': imp
}
with open('results/tv_opt_quick.json', 'w') as f:
    json.dump(data, f, indent=2)
print("Saved: results/tv_opt_quick.json")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(w_err, 'g-', label='With')
axes[0].plot(wo_err, 'r-', label='Without')
axes[0].set_ylabel('Error'); axes[0].set_title('Error over Time')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(w_lam, 'g-', label='With')
axes[1].plot(wo_lam, 'r-', label='Without')
axes[1].axhline(y=1.2, ls='--', alpha=0.5)
axes[1].set_ylabel('λ_max'); axes[1].set_title('Lambda over Time')
axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].axis('off')
axes[2].text(0.1, 0.6, f"""
RESULTS
=======
Improvement: {imp:.1f}%
With λ: {np.mean(w_lam[-20:]):.3f}
Without λ: {np.mean(wo_lam[-20:]):.3f}
""", fontsize=12, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        transform=axes[2].transAxes)

plt.tight_layout()
plt.savefig('visualization/tv_opt_quick.png')
print("Saved: visualization/tv_opt_quick.png")

print("\n" + "=" * 60)
print(f"RESULT: {imp:.1f}% IMPROVEMENT")
print("=" * 60)
