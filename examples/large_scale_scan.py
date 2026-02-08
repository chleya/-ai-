#!/usr/bin/env python3
"""
Large-Scale N Scan for Cicada Protocol
=====================================
Scan N from 50 to 2000 to validate phase transition at Nc ≈ 900.

This is the CORE experiment for project credibility.
"""

import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
import json
import time


def cicada_scan(N=200, steps=500, lr=0.001, reset_interval=None, seed=42):
    """Run Cicada Protocol scan."""
    np.random.seed(seed)
    W = np.random.randn(N, N) * 0.01
    history = []
    
    for t in range(steps):
        s = np.random.randn(N)
        W += lr * np.outer(s, s)
        eigenvals = np.linalg.eigvalsh(W)
        history.append(np.max(np.abs(eigenvals)))
        
        if reset_interval and (t + 1) % reset_interval == 0:
            W = np.random.randn(N, N) * 0.01
    
    return history


def run_large_scale_scan():
    """Run large-scale N scan."""
    
    print("=" * 70)
    print("Cicada Protocol - Large Scale N Scan")
    print("=" * 70)
    print()
    
    # Scan N values
    N_values = [50, 100, 200, 400, 600, 800, 1000, 1500, 2000]
    lr = 0.001
    steps = 500
    n_trials = 5  # Average over trials
    
    results = []
    
    print(f"Scanning N = {N_values}")
    print(f"Trials per N = {n_trials}")
    print()
    
    for N in N_values:
        print(f"  N = {N}...", end=" ", flush=True)
        start = time.time()
        
        no_reset_final = []
        reset_final = []
        
        for trial in range(n_trials):
            seed = 42 + trial
            
            # No reset
            h_no = cicada_scan(N=N, lr=lr, steps=steps, reset_interval=None, seed=seed)
            no_reset_final.append(h_no[-1])
            
            # With reset
            h_reset = cicada_scan(N=N, lr=lr, steps=steps, reset_interval=200, seed=seed)
            reset_final.append(h_reset[-1])
        
        mean_no = np.mean(no_reset_final)
        mean_reset = np.mean(reset_final)
        reduction = (1 - mean_reset/mean_no) * 100
        
        elapsed = time.time() - start
        print(f"done ({elapsed:.1f}s)")
        
        results.append({
            'N': N,
            'no_reset_mean': mean_no,
            'no_reset_std': np.std(no_reset_final),
            'reset_mean': mean_reset,
            'reset_std': np.std(reset_final),
            'reduction': reduction
        })
        
        print(f"    No reset: λ = {mean_no:.3f} ± {np.std(no_reset_final):.3f}")
        print(f"    Reset:     λ = {mean_reset:.3f} ± {np.std(reset_final):.3f}")
        print(f"    Reduction: {reduction:.1f}%")
        print()
    
    # Plot results
    plot_scan_results(results)
    
    # Save results
    with open('results/large_scale_scan.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to results/large_scale_scan.json")
    
    return results


def plot_scan_results(results):
    """Plot large-scale scan results."""
    
    Ns = [r['N'] for r in results]
    no_reset = [r['no_reset_mean'] for r in results]
    reset = [r['reset_mean'] for r in results]
    reductions = [r['reduction'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. λ_max vs N
    ax1 = axes[0, 0]
    ax1.errorbar(Ns, no_reset, yerr=[r['no_reset_std'] for r in results], 
                fmt='ro-', lw=2, capsize=5, label='No reset')
    ax1.errorbar(Ns, reset, yerr=[r['reset_std'] for r in results], 
                fmt='go-', lw=2, capsize=5, label='With reset')
    ax1.axhline(y=1.8, color='orange', ls='--', label='Threshold (1.8)')
    ax1.set_xlabel('Number of Nodes (N)')
    ax1.set_ylabel('Final λ_max')
    ax1.set_title('Spectral Radius vs System Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Reduction vs N
    ax2 = axes[0, 1]
    ax2.plot(Ns, reductions, 'bo-', lw=2, markersize=8)
    ax2.set_xlabel('Number of Nodes (N)')
    ax2.set_ylabel('Reduction (%)')
    ax2.set_title('Reset Benefit vs System Size')
    ax2.grid(True, alpha=0.3)
    
    # 3. No reset only (log scale)
    ax3 = axes[1, 0]
    ax3.semilogy(Ns, no_reset, 'ro-', lw=2, markersize=8)
    ax3.set_xlabel('Number of Nodes (N)')
    ax3.set_ylabel('λ_max (log scale)')
    ax3.set_title('Unbounded Growth (No Reset)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = []
    for r in results:
        table_data.append([
            str(r['N']),
            f"{r['no_reset_mean']:.3f}",
            f"{r['reset_mean']:.3f}",
            f"{r['reduction']:.1f}%"
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=['N', 'No Reset λ', 'Reset λ', 'Reduction'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    ax4.set_title('Summary Table')
    
    plt.tight_layout()
    plt.savefig('visualization/large_scale_scan.png', dpi=150)
    print("Plot saved to visualization/large_scale_scan.png")


def analyze_phase_transition(results):
    """Analyze phase transition behavior."""
    
    print()
    print("=" * 70)
    print("PHASE TRANSITION ANALYSIS")
    print("=" * 70)
    print()
    
    # Find where reduction drops significantly
    reductions = [r['reduction'] for r in results]
    Ns = [r['N'] for r in results]
    
    # Calculate rate of change
    for i in range(1, len(reductions)):
        delta = reductions[i] - reductions[i-1]
        N_delta = Ns[i] - Ns[i-1]
        rate = delta / N_delta * 100  # % per node
        
        if abs(rate) > 0.1:  # Significant change
            print(f"N: {Ns[i-1]} → {Ns[i]}: reduction change = {rate:+.2f}%/node")
    
    print()
    
    # Check for phase transition
    for i in range(len(reductions)):
        if reductions[i] < 20:  # Low reduction = phase II
            print(f"Phase II detected at N = {Ns[i]} (reduction = {reductions[i]:.1f}%)")
            break


if __name__ == "__main__":
    results = run_large_scale_scan()
    analyze_phase_transition(results)
