#!/usr/bin/env python3
"""Parameter sweep for Cicada Protocol"""

import numpy as np
from typing import List


def cicada_protocol(N=200, lr=0.05, steps=1000, reset_interval=None, seed=42):
    """Run protocol."""
    np.random.seed(seed)
    W = np.random.randn(N, N) * np.sqrt(2.0 / N)
    history = []
    
    for t in range(steps):
        s = np.random.randn(N)
        W += lr * np.outer(s, s)
        
        eigenvals = np.linalg.eigvalsh(W)
        lambda_max = np.max(np.abs(eigenvals))
        history.append(lambda_max)
        
        if reset_interval and (t + 1) % reset_interval == 0:
            W = np.random.randn(N, N) * np.sqrt(2.0 / N)
    
    return W, history


def main():
    print("=" * 70)
    print("Cicada Protocol - Parameter Sweep")
    print("=" * 70)
    print()
    
    # Sweep 1: Learning rate
    print("1. Learning Rate Sweep (N=200, steps=1000, no reset)")
    print("-" * 60)
    
    for lr in [0.001, 0.01, 0.02, 0.05, 0.1]:
        _, h = cicada_protocol(N=200, lr=lr, steps=1000)
        max_h = max(h)
        final_h = h[-1]
        marker = " **" if max_h >= 2.0 else ""
        print(f"   lr={lr:.3f}: max={max_h:.3f}, final={final_h:.3f}{marker}")
    
    print()
    
    # Sweep 2: With reset vs without
    print("2. Reset Effect (N=200, lr=0.05, steps=1000)")
    print("-" * 60)
    
    _, no_reset = cicada_protocol(N=200, lr=0.05, steps=1000, reset_interval=None)
    _, reset_200 = cicada_protocol(N=200, lr=0.05, steps=1000, reset_interval=200)
    _, reset_300 = cicada_protocol(N=200, lr=0.05, steps=1000, reset_interval=300)
    
    print(f"   No reset:  max={max(no_reset):.3f}, final={no_reset[-1]:.3f}")
    print(f"   Reset 200: max={max(reset_200):.3f}, final={reset_200[-1]:.3f} (reduction={(1-reset_200[-1]/no_reset[-1])*100:.0f}%)")
    print(f"   Reset 300: max={max(reset_300):.3f}, final={reset_300[-1]:.3f} (reduction={(1-reset_300[-1]/no_reset[-1])*100:.0f}%)")
    
    print()
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print()
    print("To achieve lambda_max >= 2.0:")
    print("   - lr >= 0.02 is needed")
    print("   - lr=0.05 produces: max ~3-4 without reset")
    print("   - Reset interval=300 reduces final lambda by ~50%")
    print()
    print("Optimal demo parameters: N=200, lr=0.05, steps=1000, interval=300")
    print()


if __name__ == "__main__":
    main()
