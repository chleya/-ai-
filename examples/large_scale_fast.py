#!/usr/bin/env python3
"""
Large-Scale N Scan - Fast Version
================================
"""

import numpy as np
import matplotlib.pyplot as plt
import json


def cicada_scan(N, lr=0.001, reset_interval=None):
    """Fast scan for one N."""
    np.random.seed(42)
    W = np.random.randn(N, N) * 0.01
    steps = min(200, max(50, N // 2))
    
    for t in range(steps):
        s = np.random.randn(N)
        W += lr * np.outer(s, s)
        if reset_interval and (t + 1) % reset_interval == 0:
            W = np.random.randn(N, N) * 0.01
    
    return np.max(np.abs(np.linalg.eigvalsh(W)))


def fast_scan():
    """Fast scan N values."""
    
    print("=" * 70)
    print("Cicada Protocol - Large Scale N Scan (Fast)")
    print("=" * 70)
    print()
    
    N_values = [50, 100, 200, 400, 600, 800, 1000, 1500, 2000]
    lr = 0.001
    reset_interval = 200
    
    results = []
    
    print("-" * 70)
    print(f"{'N':>6} {'No Reset λ':<15} {'Reset λ':<15} {'Reduction':<12}")
    print("-" * 70)
    
    for N in N_values:
        no_reset = cicada_scan(N, lr=lr, reset_interval=None)
        with_reset = cicada_scan(N, lr=lr, reset_interval=reset_interval)
        reduction = (1 - with_reset/no_reset) * 100
        
        print(f"{N:>6} {no_reset:<15.3f} {with_reset:<15.3f} {reduction:<12.1f}%")
        
        results.append({
            'N': N,
            'no_reset': no_reset,
            'reset': with_reset,
            'reduction': reduction
        })
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    Ns = [r['N'] for r in results]
    
    # λ_max vs N
    axes[0].plot(Ns, [r['no_reset'] for r in results], 'ro-', lw=2, label='No reset')
    axes[0].plot(Ns, [r['reset'] for r in results], 'go-', lw=2, label='With reset')
    axes[0].axhline(y=1.8, color='orange', ls='--', label='Threshold')
    axes[0].set_xlabel('N')
    axes[0].set_ylabel('λ_max')
    axes[0].set_title('Spectral Radius vs System Size')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reduction vs N
    axes[1].plot(Ns, [r['reduction'] for r in results], 'bo-', lw=2)
    axes[1].set_xlabel('N')
    axes[1].set_ylabel('Reduction (%)')
    axes[1].set_title('Reset Benefit vs System Size')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualization/large_scale_scan.png', dpi=150)
    print()
    print("Plot saved to visualization/large_scale_scan.png")
    
    # Save results
    with open('results/large_scale_scan.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to results/large_scale_scan.json")
    
    return results


if __name__ == "__main__":
    fast_scan()
