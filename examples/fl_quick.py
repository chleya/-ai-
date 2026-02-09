#!/usr/bin/env python3
"""Federated Learning - Quick Test"""

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


def fl_simulation(K=5, N=200, rounds=30, eta=0.01, hetero=2, cicada=True, seed=42):
    np.random.seed(seed)
    
    # Non-IID data
    client_means = [np.random.randn(N) * (1 + k*0.1) for k in range(K)]
    
    W_global = np.random.randn(N, N) * 0.1
    losses, lams = [], []
    
    for r in range(rounds):
        # Local updates
        W_list = []
        for k in range(K):
            W_k = W_global.copy()
            for _ in range(3):
                s = client_means[k] + np.random.randn(N) * 0.5
                W_k = hebbian_update(W_k, s, eta)
            W_list.append(W_k)
        
        # Aggregate
        W_global = np.mean(W_list, axis=0)
        
        # Loss
        loss = np.mean([np.linalg.norm(W_global @ m) for m in client_means])
        losses.append(loss)
        
        # Lambda
        lam = np.mean([lambda_max_approx(W) for W in W_list])
        lams.append(lam)
        
        # Cicada
        if cicada and lam > 1.2:
            W_global = np.random.randn(N, N) * 0.1
    
    return losses, lams


print("=" * 60)
print("FEDERATED LEARNING - QUICK TEST")
print("=" * 60)

# Basic test
w_loss, w_lam = fl_simulation(K=5, N=200, cicada=True)
wo_loss, wo_lam = fl_simulation(K=5, N=200, cicada=False)

imp = (np.mean(wo_loss[-10:]) - np.mean(w_loss[-10:])) / np.mean(wo_loss[-10:]) * 100

print(f"\nWith Cicada:    avg_loss={np.mean(w_loss[-10:]):.4f}, avg_lambda={np.mean(w_lam[-10:]):.4f}")
print(f"Without Cicada: avg_loss={np.mean(wo_loss[-10:]):.4f}, avg_lambda={np.mean(wo_lam[-10:]):.4f}")
print(f"\nImprovement: {imp:.1f}%")

# Scale test
print("\n[Scale Test]")
scale_ims = []
for N in [100, 200, 400]:
    w_l, w_la = fl_simulation(K=3, N=N, cicada=True)
    wo_l, wo_la = fl_simulation(K=3, N=N, cicada=False)
    im = (np.mean(wo_l[-10:]) - np.mean(w_l[-10:])) / np.mean(wo_l[-10:]) * 100
    scale_ims.append({'N': N, 'imp': im})
    print(f"N={N}: {im:.1f}%")

# Save
data = {'improvement': imp, 'scale': scale_ims}
with open('results/fl_quick.json', 'w') as f:
    json.dump(data, f, indent=2)
print("\nSaved: results/fl_quick.json")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(w_loss, 'g-', label='With')
axes[0].plot(wo_loss, 'r-', label='Without')
axes[0].set_ylabel('Loss'); axes[0].set_title('Loss over Rounds')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(w_lam, 'g-', label='With')
axes[1].plot(wo_lam, 'r-', label='Without')
axes[1].axhline(y=1.2, ls='--', alpha=0.5)
axes[1].set_ylabel('λ_max'); axes[1].set_title('Lambda over Rounds')
axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].axis('off')
axes[2].text(0.1, 0.6, f"""
RESULTS
=======
Improvement: {imp:.1f}%
With λ: {np.mean(w_lam[-10:]):.3f}
Without λ: {np.mean(wo_lam[-10:]):.3f}
""", fontsize=12, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        transform=axes[2].transAxes)

plt.tight_layout()
plt.savefig('visualization/fl_quick.png')
print("Saved: visualization/fl_quick.png")

print("\n" + "=" * 60)
print(f"RESULT: {imp:.1f}% IMPROVEMENT")
print("=" * 60)
