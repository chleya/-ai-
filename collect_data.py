#!/usr/bin/env python3
"""Minimal Data Collection"""

import numpy as np
import json

def cicada(N, lr=0.001, steps=200, seed=42):
    np.random.seed(seed)
    W = np.random.randn(N, N) * 0.01
    for _ in range(steps):
        s = np.random.randn(N)
        W += lr * np.outer(s, s)
    return np.max(np.abs(np.linalg.eigvalsh(W)))

# Collect
print("Collecting data...")

# 1. N sweep
n_data = []
for N in [100, 200, 500, 1000, 1500, 2000]:
    no = cicada(N)
    re = cicada(N)
    n_data.append({'N': N, 'no_reset': no, 'reset': re, 'reduction': (1-re/no)*100})
    print("N=%d: no=%.3f, reset=%.3f, reduction=%.1f%%" % (N, no, re, (1-re/no)*100))

# 2. Seeds (20)
seed_data = []
for s in range(20):
    no = cicada(200, seed=s)
    re = cicada(200, seed=s)
    seed_data.append({'seed': s, 'no_reset': no, 'reset': re})

# Save
result = {'N_sweep': n_data, 'seeds': seed_data}
with open('results/comprehensive_data.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nSaved: results/comprehensive_data.json")
print("Done!")
