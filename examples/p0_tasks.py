#!/usr/bin/env python3
"""
P0 Tasks: Seed Average + Eta Sweep
=================================
"""

import numpy as np
import matplotlib.pyplot as plt
import json


def cicada(N=200, steps=500, lr=0.001, reset_interval=None, seed=42):
    """Run experiment."""
    np.random.seed(seed)
    W = np.random.randn(N, N) * 0.01
    history = []
    
    for t in range(steps):
        s = np.random.randn(N)
        W += lr * np.outer(s, s)
        lam = np.max(np.abs(np.linalg.eigvalsh(W)))
        history.append(lam)
        
        if reset_interval and (t + 1) % reset_interval == 0:
            W = np.random.randn(N, N) * 0.01
    
    return history


def task1_seed_average():
    """P0 Task 1: Multi-seed average (10 trials)."""
    
    print("=" * 70)
    print("P0-Task 1: Multi-Seed Average (10 trials)")
    print("=" * 70)
    print()
    
    N = 200
    steps = 500
    lr = 0.001
    
    # Test different seeds
    seeds = [42, 123, 456, 789, 1011, 1314, 1617, 1920, 2223, 2526]
    
    print(f"Testing {len(seeds)} seeds...")
    print()
    
    results_no = []
    results_reset = []
    
    for seed in seeds:
        h_no = cicada(N=N, steps=steps, lr=lr, reset_interval=None, seed=seed)
        h_reset = cicada(N=N, steps=steps, lr=lr, reset_interval=200, seed=seed)
        
        results_no.append(h_no[-1])
        results_reset.append(h_reset[-1])
    
    mean_no = np.mean(results_no)
    std_no = np.std(results_no)
    mean_reset = np.mean(results_reset)
    std_reset = np.std(results_reset)
    
    print("-" * 70)
    print(f"No Reset:  λ = {mean_no:.4f} ± {std_no:.4f}")
    print(f"Reset:     λ = {mean_reset:.4f} ± {std_reset:.4f}")
    print(f"Reduction: {(1 - mean_reset/mean_no) * 100:.1f}%")
    print()
    
    # Save
    with open('results/seed_average.json', 'w') as f:
        json.dump({
            'seeds': seeds,
            'no_reset': {'mean': mean_no, 'std': std_no, 'values': results_no},
            'reset': {'mean': mean_reset, 'std': std_reset, 'values': results_reset}
        }, f, indent=2)
    print("Saved: results/seed_average.json")
    
    return mean_no, std_no, mean_reset, std_reset


def task2_eta_sweep():
    """P0 Task 2: Different learning rates."""
    
    print()
    print("=" * 70)
    print("P0-Task 2: Eta Sweep")
    print("=" * 70)
    print()
    
    N = 200
    steps = 500
    
    eta_values = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    
    print("-" * 70)
    print(f"{'η':<10} {'No Reset λ':<15} {'Reset λ':<15} {'Reduction':<12}")
    print("-" * 70)
    
    results = []
    
    for eta in eta_values:
        h_no = cicada(N=N, steps=steps, lr=eta, reset_interval=None)
        h_reset = cicada(N=N, steps=steps, lr=eta, reset_interval=200)
        
        final_no = h_no[-1]
        final_reset = h_reset[-1]
        reduction = (1 - final_reset/final_no) * 100
        
        print(f"{eta:<10} {final_no:<15.4f} {final_reset:<15.4f} {reduction:<12.1f}%")
        results.append({'eta': eta, 'no_reset': final_no, 'reset': final_reset, 'reduction': reduction})
    
    # Save
    with open('results/eta_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)
    print()
    print("Saved: results/eta_sweep.json")
    
    return results


def plot_p0_results(seed_results, eta_results):
    """P0: Generate all plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Seed average bar chart
    ax1 = axes[0, 0]
    ax1.bar(['No Reset', 'Reset'], 
            [seed_results[0], seed_results[2]],
            yerr=[seed_results[1], seed_results[3]],
            color=['red', 'green'], alpha=0.8, capsize=5)
    ax1.set_ylabel('λ_max')
    ax1.set_title('Seed Average (10 trials)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Eta sweep
    ax2 = axes[0, 1]
    etas = [r['eta'] for r in eta_results]
    no_reset = [r['no_reset'] for r in eta_results]
    reset = [r['reset'] for r in eta_results]
    ax2.semilogx(etas, no_reset, 'ro-', lw=2, label='No reset')
    ax2.semilogx(etas, reset, 'go-', lw=2, label='Reset')
    ax2.set_xlabel('Learning Rate (η)')
    ax2.set_ylabel('Final λ_max')
    ax2.set_title('Eta Sweep')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Reduction vs Eta
    ax3 = axes[1, 0]
    reductions = [r['reduction'] for r in eta_results]
    ax3.semilogx(etas, reductions, 'bo-', lw=2, markersize=8)
    ax3.set_xlabel('Learning Rate (η)')
    ax3.set_ylabel('Reduction (%)')
    ax3.set_title('Reset Benefit vs Eta')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary = f"""
    P0 SUMMARY
    ==========
    
    Seed Average (10 trials):
      No Reset: λ = {seed_results[0]:.4f} ± {seed_results[1]:.4f}
      Reset:    λ = {seed_results[2]:.4f} ± {seed_results[3]:.4f}
      Reduction: {(1 - seed_results[2]/seed_results[0]) * 100:.1f}%
    
    Eta Sweep:
      Best η: {min(eta_results, key=lambda x: x['reduction'])['eta']}
      Worst η: {max(eta_results, key=lambda x: x['reduction'])['eta']}
    """
    ax4.text(0.1, 0.5, summary, transform=ax4.transAxes,
            fontsize=12, fontfamily='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('visualization/p0_results.png', dpi=150)
    print()
    print("Saved: visualization/p0_results.png")


def main():
    """Run all P0 tasks."""
    
    # Task 1: Seed average
    seed_results = task1_seed_average()
    
    # Task 2: Eta sweep
    eta_results = task2_eta_sweep()
    
    # Generate plots
    plot_p0_results(seed_results, eta_results)
    
    print()
    print("=" * 70)
    print("P0 Tasks Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
