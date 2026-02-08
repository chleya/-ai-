#!/usr/bin/env python3
"""
Distributed Consensus Experiment - Simplified
============================================
Tests: Does periodic reset improve consensus convergence?

Task: N nodes must converge to global mean.
"""

import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt


def consensus_experiment(N=200, steps=500, lr=0.001, reset_interval=None, seed=42):
    """Run consensus experiment."""
    np.random.seed(seed)
    
    # Initialize node values
    values = np.random.randn(N)
    global_mean = np.mean(values)
    
    # Weight matrix (full connectivity for simplicity)
    W = np.random.randn(N, N) * 0.01
    
    errors = []
    lambda_maxes = []
    
    for t in range(steps):
        # Hebbian update
        s = np.random.randn(N)
        W += lr * np.outer(s, s)
        
        # Update values (average toward neighbors)
        values = 0.9 * values + 0.1 * np.mean(W @ values)
        
        # Consensus error
        error = np.std(values - global_mean)
        errors.append(error)
        
        # λ_max
        eigenvals = np.linalg.eigvalsh(W)
        lambda_max = np.max(np.abs(eigenvals))
        lambda_maxes.append(lambda_max)
        
        # Reset
        if reset_interval and (t + 1) % reset_interval == 0:
            W = np.random.randn(N, N) * 0.01
    
    return errors, lambda_maxes


def run_scalability():
    """Test different N values."""
    
    print("=" * 70)
    print("Cicada Protocol - Scalability Experiment")
    print("=" * 70)
    print()
    
    N_values = [50, 100, 200, 500]
    lr = 0.001
    steps = 500
    reset_interval = 200
    
    print("-" * 70)
    print(f"{'N':>5} {'Strategy':<12} {'Final Err':<12} {'λ_max':<10}")
    print("-" * 70)
    
    results = {}
    
    for N in N_values:
        # No reset
        errors_no, lambda_no = consensus_experiment(N=N, lr=lr, steps=steps)
        print(f"{N:>5} {'No reset':<12} {errors_no[-1]:<12.4f} {lambda_no[-1]:<10.3f}")
        results[(N, 'no_reset')] = (errors_no, lambda_no)
        
        # With reset
        errors_reset, lambda_reset = consensus_experiment(
            N=N, lr=lr, steps=steps, reset_interval=reset_interval
        )
        print(f"{N:>5} {'Reset 200':<12} {errors_reset[-1]:<12.4f} {lambda_reset[-1]:<10.3f}")
        results[(N, 'reset')] = (errors_reset, lambda_reset)
        
        # Improvement
        improvement = (errors_no[-1] - errors_reset[-1]) / (errors_no[-1] + 1e-6) * 100
        print(f"       Improvement: {improvement:+.1f}%")
        print()
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Error vs N
    ax1 = axes[0, 0]
    Ns = N_values
    ax1.plot(Ns, [results[(n, 'no_reset')][0][-1] for n in Ns], 'ro-', lw=2, label='No reset')
    ax1.plot(Ns, [results[(n, 'reset')][0][-1] for n in Ns], 'go-', lw=2, label='With reset')
    ax1.set_xlabel('Number of Nodes (N)')
    ax1.set_ylabel('Final Consensus Error')
    ax1.set_title('Consensus Error vs System Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. λ_max vs N
    ax2 = axes[0, 1]
    ax2.plot(Ns, [results[(n, 'no_reset')][1][-1] for n in Ns], 'ro-', lw=2, label='No reset')
    ax2.plot(Ns, [results[(n, 'reset')][1][-1] for n in Ns], 'go-', lw=2, label='With reset')
    ax2.axhline(y=1.8, color='orange', ls='--', label='Threshold')
    ax2.set_xlabel('Number of Nodes (N)')
    ax2.set_ylabel('Final λ_max')
    ax2.set_title('Spectral Radius vs System Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Convergence curve (N=200)
    ax3 = axes[1, 0]
    errors_no, lambda_no = results[(200, 'no_reset')]
    errors_reset, lambda_reset = results[(200, 'reset')]
    ax3.plot(errors_no, 'r-', lw=1.5, alpha=0.7, label='No reset')
    ax3.plot(errors_reset, 'g-', lw=1.5, alpha=0.7, label='Reset 200')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Consensus Error')
    ax3.set_title('Convergence (N=200)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. λ_max curve (N=200)
    ax4 = axes[1, 1]
    ax4.plot(lambda_no, 'r-', lw=1.5, alpha=0.7, label='No reset')
    ax4.plot(lambda_reset, 'g-', lw=1.5, alpha=0.7, label='Reset 200')
    ax4.axhline(y=1.8, color='orange', ls='--')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('λ_max')
    ax4.set_title('Spectral Radius (N=200)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualization/consensus_scalability.png', dpi=150)
    print()
    print("Plot saved to visualization/consensus_scalability.png")


if __name__ == "__main__":
    run_scalability()
