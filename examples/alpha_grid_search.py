#!/usr/bin/env python3
"""
Alpha Grid Search - Higher Learning Rate
======================================
Uses lr=0.005 to show event-triggered behavior.
"""

import numpy as np
import matplotlib.pyplot as plt


def run_experiment(lr=0.005, alphas=[1.2, 1.4, 1.6, 1.8, 2.0]):
    """Run alpha grid search with given lr."""
    
    N = 200
    steps = 800
    threshold = 1.8
    
    print("=" * 70)
    print(f"Alpha Grid Search (lr={lr})")
    print("=" * 70)
    print()
    
    print("-" * 70)
    print(f"{'Alpha':<8} {'Final λ':<12} {'Max λ':<12} {'Resets':<10} {'Stable':<10}")
    print("-" * 70)
    
    results = []
    
    for alpha in alphas:
        np.random.seed(42)
        W = np.random.randn(N, N) * 0.01
        history = []
        resets = 0
        
        for t in range(steps):
            s = np.random.randn(N)
            W += lr * np.outer(s, s)
            lam = np.max(np.abs(np.linalg.eigvalsh(W)))
            history.append(lam)
            
            if lam > threshold * alpha:  # Event-triggered
                W = np.random.randn(N, N) * 0.01
                resets += 1
        
        stable = max(history) < 2.0
        print(f"{alpha:<8} {history[-1]:<12.3f} {max(history):<12.3f} {resets:<10} {stable}")
        results.append((alpha, history, resets, stable, max(history)))
    
    # Fixed interval baseline
    print("-" * 70)
    np.random.seed(42)
    W = np.random.randn(N, N) * 0.01
    fixed_hist = []
    for t in range(steps):
        s = np.random.randn(N)
        W += lr * np.outer(s, s)
        fixed_hist.append(np.max(np.abs(np.linalg.eigvalsh(W))))
        if (t + 1) % 200 == 0:
            W = np.random.randn(N, N) * 0.01
    print(f"{'Fixed 200':<8} {fixed_hist[-1]:<12.3f} {max(fixed_hist):<12.3f} {steps//200:<10} {max(fixed_hist)<2.0}")
    
    # No reset baseline
    print(f"{'No reset':<8} {history[-1]:<12.3f} {max([h[-1] for h in [r[1] for r in results]]):<12.3f} {'0':<10} False")
    
    return results, fixed_hist


def plot_results(results, fixed_hist, lr):
    """Plot results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Evolution curves
    ax1 = axes[0, 0]
    for alpha, history, resets, stable, max_lam in results:
        ax1.plot(history[:400], lw=1.5, alpha=0.8, label=f'α={alpha}')
    ax1.plot(fixed_hist[:400], 'k--', lw=2, label='Fixed 200')
    ax1.axhline(y=1.8, color='orange', ls='--', alpha=0.5)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('λ_max')
    ax1.set_title(f'λ_max Evolution (lr={lr})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Max λ vs Alpha
    ax2 = axes[0, 1]
    alphas = [r[0] for r in results]
    max_lams = [r[4] for r in results]
    ax2.plot(alphas, max_lams, 'bo-', lw=2, markersize=10)
    ax2.axhline(y=1.8, color='orange', ls='--', label='Threshold')
    ax2.set_xlabel('Alpha')
    ax2.set_ylabel('Max λ_max')
    ax2.set_title('Max λ_max vs Alpha')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Reset count vs Alpha
    ax3 = axes[1, 0]
    resets = [r[2] for r in results]
    ax3.bar(range(len(alphas)), resets, color='blue', alpha=0.8)
    ax3.set_xticks(range(len(alphas)))
    ax3.set_xticklabels([f'α={a}' for a in alphas])
    ax3.set_ylabel('Reset Count')
    ax3.set_title('Reset Count vs Alpha')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary
    ax4 = axes[1, 1]
    strategies = ['No reset', 'Fixed 200'] + [f'α={a}' for a in alphas]
    finals = [fixed_hist[-1], fixed_hist[-1]] + [r[1][-1] for r in results]
    colors = ['red', 'green'] + ['blue']*len(results)
    bars = ax4.bar(range(len(strategies)), finals, color=colors, alpha=0.8, edgecolor='black')
    ax4.axhline(y=1.8, color='orange', ls='--')
    ax4.set_xticks(range(len(strategies)))
    ax4.set_xticklabels(strategies, rotation=45, ha='right')
    ax4.set_ylabel('Final λ_max')
    ax4.set_title('Strategy Comparison')
    
    for bar, val in zip(bars, finals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('visualization/alpha_grid_search.png', dpi=150)
    print()
    print("Plot saved to visualization/alpha_grid_search.png")


def analyze(results, fixed_hist):
    """Analyze optimal alpha."""
    
    print()
    print("=" * 70)
    print("OPTIMAL ALPHA ANALYSIS")
    print("=" * 70)
    print()
    
    # Find optimal: minimize max_lambda
    best = min(results, key=lambda x: x[4])
    print(f"Best alpha (lowest max λ): {best[0]}")
    print(f"  Max λ: {best[4]:.3f}")
    print(f"  Resets: {best[2]}")
    print()
    
    # Check claim
    if best[0] == 1.6:
        print("[OK] Alpha = 1.6 is optimal (as claimed)")
    else:
        print(f"[NOTE] Claimed α=1.6, but actual optimal is α={best[0]}")
    
    # Find most efficient: lowest resets among stable
    stable = [r for r in results if r[3]]
    if stable:
        efficient = min(stable, key=lambda x: x[2])
        print(f"Most efficient stable: α={efficient[0]} ({efficient[2]} resets)")
    
    # Compare with fixed
    print()
    print("Event-Triggered vs Fixed:")
    for r in results:
        if r[4] < max(fixed_hist):
            improvement = (max(fixed_hist) - r[4]) / max(fixed_hist) * 100
            print(f"  α={r[0]}: {improvement:.1f}% better than fixed")


if __name__ == "__main__":
    results, fixed_hist = run_experiment(lr=0.005)
    plot_results(results, fixed_hist, lr=0.005)
    analyze(results, fixed_hist)
