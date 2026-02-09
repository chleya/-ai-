#!/usr/bin/env python3
"""Pattern Classification - Hard Version"""

import numpy as np
import matplotlib.pyplot as plt
import json


def hebbian_update(W, x, target_onehot, eta, anti_onehot=None):
    W_new = W + eta * np.outer(target_onehot, x)
    if anti_onehot is not None:
        W_new -= eta * 0.5 * np.outer(anti_onehot, x)
    return W_new


def lambda_max(W, iters=5):
    M = W @ W.T
    v = np.random.rand(M.shape[0])
    for _ in range(iters):
        v = M @ v
        v /= np.linalg.norm(v)
    return np.abs(v @ (M @ v))


def create_hard_data(n_samples=200, n_classes=5, n_features=50, similarity=0.8, seed=42):
    np.random.seed(seed)
    base_patterns = np.random.randn(n_features)
    class_patterns = {}
    for c in range(n_classes):
        noise = np.random.randn(n_features) * (1 - similarity)
        class_patterns[c] = base_patterns + noise
    
    X, y = [], []
    for i in range(n_samples):
        c = np.random.randint(0, n_classes)
        x = class_patterns[c] + np.random.randn(n_features) * 0.8
        X.append(x)
        y.append(c)
    return np.array(X), np.array(y), class_patterns


def train(X, y, W_init, epochs=40, eta=0.01, theta=1.2, cicada=True, seed=42):
    np.random.seed(seed)
    W = W_init.copy()
    accuracies, lams, losses = [], [], []
    n_classes = W.shape[0]
    
    for epoch in range(epochs):
        correct, total_loss = 0, 0
        for i in range(len(X)):
            x = X[i]
            target = y[i]
            target_onehot = np.zeros(n_classes)
            target_onehot[target] = 1.0
            anti_onehot = np.ones(n_classes) - target_onehot
            W = hebbian_update(W, x, target_onehot, eta, anti_onehot)
            
            scores = W @ x
            pred = np.argmax(scores)
            if pred == target:
                correct += 1
            
            exp_s = np.exp(scores - np.max(scores))
            probs = exp_s / np.sum(exp_s)
            loss = -np.log(probs[target] + 1e-10)
            total_loss += loss
        
        acc = correct / len(X) * 100
        accuracies.append(acc)
        lams.append(lambda_max(W))
        losses.append(total_loss / len(X))
        
        if cicada and lams[-1] > theta:
            W = np.random.randn(n_classes, W.shape[1]) * 0.01
    
    return {
        'accuracies': accuracies,
        'lambdas': lams,
        'losses': losses,
        'avg_acc': np.mean(accuracies[-10:]),
        'avg_loss': np.mean(losses[-10:])
    }


print("=" * 60)
print("PATTERN CLASSIFICATION - HARD VERSION")
print("=" * 60)

# Similarity test
print("\n[Similarity Test]")
sim_results = []
for similarity in [0.5, 0.7, 0.9]:
    X, y, _ = create_hard_data(n_samples=200, similarity=similarity)
    W_init = np.random.randn(5, 50) * 0.1
    
    r_with = train(X, y, W_init, epochs=50, cicada=True)
    r_without = train(X, y, W_init, epochs=50, cicada=False)
    
    imp = r_with['avg_acc'] - r_without['avg_acc']
    sim_results.append({'sim': similarity, 'imp': imp, 'with': r_with['avg_acc'], 'without': r_without['avg_acc']})
    print(f"sim={similarity:.1f}: With={r_with['avg_acc']:.1f}%, Without={r_without['avg_acc']:.1f}%, Imp={imp:+.1f}%")

# Basic test
print("\n[Basic Test]")
X, y, _ = create_hard_data(n_samples=200, similarity=0.7)
W_init = np.random.randn(5, 50) * 0.1

r_with = train(X, y, W_init, epochs=50, cicada=True)
r_without = train(X, y, W_init, epochs=50, cicada=False)

imp = r_with['avg_acc'] - r_without['avg_acc']
print(f"With Cicada:    acc={r_with['avg_acc']:.1f}%, lambda={np.mean(r_with['lambdas'][-10:]):.1f}")
print(f"Without Cicada: acc={r_without['avg_acc']:.1f}%, lambda={np.mean(r_without['lambdas'][-10:]):.1f}")
print(f"Improvement: {imp:+.1f}%")

# Save
data = {'improvement': imp, 'similarity': sim_results}
with open('results/pattern_hard.json', 'w') as f:
    json.dump(data, f, indent=2)
print("\nSaved: results/pattern_hard.json")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(r_with['accuracies'], 'g-', lw=2, label='With')
axes[0].plot(r_without['accuracies'], 'r-', lw=2, label='Without')
axes[0].set_ylabel('Accuracy (%)'); axes[0].set_title('Classification')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(r_with['lambdas'], 'g-', lw=2, label='With')
axes[1].plot(r_without['lambdas'], 'r-', lw=2, label='Without')
axes[1].set_ylabel('lambda (log)'); axes[1].set_title('Spectral Radius')
axes[1].set_yscale('log'); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].axis('off')
axes[2].text(0.05, 0.7, f"RESULT\n=====\nImp: {imp:.1f}%\n\nSim Effect:", 
             fontsize=11, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             transform=axes[2].transAxes)
for r in sim_results:
    axes[2].text(0.05, 0.35 - sim_results.index(r)*0.08,
                f"sim={r['sim']:.1f}: {r['imp']:+.1f}%",
                transform=axes[2].transAxes, fontsize=10)

plt.tight_layout()
plt.savefig('visualization/pattern_hard.png')
print("Saved: visualization/pattern_hard.png")

print("\n" + "=" * 60)
print(f"RESULT: {imp:.1f}% IMPROVEMENT")
print("=" * 60)
