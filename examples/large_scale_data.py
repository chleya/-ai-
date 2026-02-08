#!/usr/bin/env python3
"""
Large-Scale P0 Tasks: More Data!
================================
- Seeds: 20 trials
- N: [100, 200, 500, 1000, 2000]
- Eta: [0.0001, 0.0005, 0.001, 0.005, 0.01]
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
    
    return history[-1]


def run_large_scale():
    """Run large-scale experiments."""
    
    print("=" * 70)
    print("Large-Scale P0 Tasks (More Data)")
    print("=" * 70)
    print()
    
    # ═══════════════════════════════════════════════════════
    # Task 1: More seeds (20 trials)
    # ═══════════════════════════════════════════════════════
    print("Task 1: 20-Seed Average")
    print("-" * 70)
    
    N = 200
    lr = 0.001
    seeds = list(range(20))  # 20 seeds
    
    no_reset_vals = []
    reset_vals = []
    
    for seed in seeds:
        no_reset_vals.append(cicada(N=N, lr=lr, reset_interval=None, seed=seed))
        reset_vals.append(cicada(N=N, lr=lr, reset_interval=200, seed=seed))
    
    mean_no = np.mean(no_reset_vals)
    std_no = np.std(no_reset_vals)
    mean_reset = np.mean(reset_vals)
    std_reset = np.std(reset_vals)
    
    print(f"No Reset: λ = {mean_no:.4f} ± {std_no:.4f} (n={len(seeds)})")
    print(f"Reset:     λ = {mean_reset:.4f} ± {std_reset:.4f} (n={len(seeds)})")
    print(f"Reduction: {(1 - mean_reset/mean_no) * 100:.1f}%")
    
    # Save
    with open('results/20seed_average.json', 'w') as f:
        json.dump({
            'n_trials': len(seeds),
            'no_reset': {'mean': mean_no, 'std': std_no, 'values': no_reset_vals},
            'reset': {'mean': mean_reset, 'std': std_reset, 'values': reset_vals}
        }, f, indent=2)
    print("Saved: results/20seed_average.json")
    print()
    
    # ═══════════════════════════════════════════════════════
    # Task 2: Large N sweep
    # ═══════════════════════════════════════════════════════
    print("Task 2: Large N Sweep (N=100~2000)")
    print("-" * 70)
    
    N_values = [100, 200, 500, 1000, 1500, 2000]
    eta = 0.001
    
    n_results = []
    
    for N in N_values:
        h_no = cicada(N=N, lr=eta, reset_interval=None, seed=42)
        h_reset = cicada(N=N, lr=eta, reset_interval=200, seed=42)
        reduction = (1 - h_reset/h_no) * 100
        
        print(f"N={N:4d}: No Reset={h_no:.4f}, Reset={h_reset:.4f}, Reduction={reduction:.1f}%")
        n_results.append({'N': N, 'no_reset': h_no, 'reset': h_reset, 'reduction': reduction})
    
    with open('results/large_n_sweep.json', 'w') as f:
        json.dump(n_results, f, indent=2)
    print("Saved: results/large_n_sweep.json")
    print()
    
    # ═══════════════════════════════════════════════════════
    # Task 3: Fine Eta sweep
    # ═══════════════════════════════════════════════════════
    print("Task 3: Fine Eta Sweep")
    print("-" * 70)
    
    eta_values = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
    N = 500
    
    eta_results = []
    
    for eta in eta_values:
        h_no = cicada(N=N, lr=eta, reset_interval=None, seed=42)
        h_reset = cicada(N=N, lr=eta, reset_interval=200, seed=42)
        reduction = (1 - h_reset/h_no) * 100
        
        print(f"η={eta:.4f}: No Reset={h_no:.4f}, Reset={h_reset:.4f}, Reduction={reduction:.1f}%")
        eta_results.append({'eta': eta, 'no_reset': h_no, 'reset': h_reset, 'reduction': reduction})
    
    with open('results/fine_eta_sweep.json', 'w') as f:
        json.dump(eta_results, f, indent=2)
    print("Saved: results/fine_eta_sweep.json")
    print()
    
    # ═══════════════════════════════════════════════════════
    # Generate comprehensive plots
    # ═══════════════════════════════════════════════════════
    print("Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 20-seed comparison
    ax1 = axes[0, 0]
    ax1.bar(['No Reset', 'Reset'], [mean_no, mean_reset],
            yerr=[std_no, std_reset], color=['red', 'green'], alpha=0.8, capsize=10)
    ax1.set_ylabel('λ_max')
    ax1.set_title(f'20-Seed Average (N={N})')
    ax1.grid(True, alpha=0.3)
    
    # 2. Large N sweep
    ax2 = axes[0, 1]
    ax2.plot([r['N'] for r in n_results], [r['no_reset'] for r in n_results], 'ro-', lw=2, label='No reset')
    ax2.plot([r['N'] for r in n_results], [r['reset'] for r in n_results], 'go-', lw=2, label='Reset')
    ax2.set_xlabel('N')
    ax2.set_ylabel('λ_max')
    ax2.set_title('Large N Sweep')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Eta sweep
    ax3 = axes[1, 0]
    ax3.semilogx([r['eta'] for r in eta_results], [r['no_reset'] for r in eta_results], 'ro-', lw=2, label='No reset')
    ax3.semilogx([r['eta'] for r in eta_results], [r['reset'] for r in eta_results], 'go-', lw=2, label='Reset')
    ax3.set_xlabel('η (log scale)')
    ax3.set_ylabel('λ_max')
    ax3.set_title('Fine Eta Sweep (N=500)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Reduction curves
    ax4 = axes[1, 1]
    ax4.plot([r['N'] for r in n_results], [r['reduction'] for r in n_results], 'bo-', lw=2, label='Large N')
    ax4_twin = ax4.twinx()
    ax4_twin.plot([r['eta'] for r in eta_results], [r['reduction'] for r in eta_results], 'rs-', lw=2, label='Fine Eta')
    ax4.set_xlabel('N / η')
    ax4.set_ylabel('Reduction (%) - N', color='blue')
    ax4_twin.set_ylabel('Reduction (%) - η', color='red')
    ax4.set_title('Reduction vs Parameters')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualization/large_scale_data.png', dpi=150)
    print("Saved: visualization/large_scale_data.png")
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"20-Seed Average: {(1 - mean_reset/mean_no) * 100:.1f}% reduction")
    print(f"Large N Range: {n_results[0]['N']} ~ {n_results[-1]['N']}")
    print(f"Eta Range: {eta_results[0]['eta']} ~ {eta_results[-1]['eta']}")
    print()
    print("Data saved:")
    print("  - results/20seed_average.json")
    print("  - results/large_n_sweep.json")
    print("  - results/fine_eta_sweep.json")
    print("  - visualization/large_scale_data.png")


if __name__ == "__main__":
    run_large_scale()
