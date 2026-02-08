#!/usr/bin/env python3
"""
Cicada Protocol - Demo that actually shows reset effect
======================================================
This version properly demonstrates why reset helps.

Key: Larger learning rate + longer evolution = spectral explosion
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import warnings
warnings.filterwarnings('ignore')
plt.switch_backend('Agg')


@dataclass
class Result:
    name: str
    history: List[float]
    final_lambda: float


def run_experiment(N: int = 200, steps: int = 1000, reset_interval: int = 200,
                   learning_rate: float = 0.01, seed: int = 42) -> dict:
    """Run experiment with different strategies"""
    np.random.seed(seed)
    
    # Strategy 1: No reset
    W1 = np.random.randn(N, N) * 0.1
    history1 = []
    
    for t in range(steps):
        s = np.random.randn(N)
        s = s / (np.linalg.norm(s) + 1e-6)
        W1 += learning_rate * np.outer(s, s)
        
        # Normalize only when really needed
        if np.linalg.norm(W1) > 10:
            W1 = W1 / np.linalg.norm(W1) * 10
        
        eigenvals = np.linalg.eigvalsh(W1)
        history1.append(np.max(np.abs(eigenvals)))
    
    # Strategy 2: With reset
    W2 = np.random.randn(N, N) * 0.1
    history2 = []
    reset_count = 0
    
    for t in range(steps):
        s = np.random.randn(N)
        s = s / (np.linalg.norm(s) + 1e-6)
        W2 += learning_rate * np.outer(s, s)
        
        if np.linalg.norm(W2) > 10:
            W2 = W2 / np.linalg.norm(W2) * 10
        
        eigenvals = np.linalg.eigvalsh(W2)
        history2.append(np.max(np.abs(eigenvals)))
        
        # Reset at interval
        if (t + 1) % reset_interval == 0:
            W2 = np.random.randn(N, N) * 0.1
            reset_count += 1
    
    # Strategy 3: More frequent reset
    W3 = np.random.randn(N, N) * 0.1
    history3 = []
    reset_count3 = 0
    
    for t in range(steps):
        s = np.random.randn(N)
        s = s / (np.linalg.norm(s) + 1e-6)
        W3 += learning_rate * np.outer(s, s)
        
        if np.linalg.norm(W3) > 10:
            W3 = W3 / np.linalg.norm(W3) * 10
        
        eigenvals = np.linalg.eigvalsh(W3)
        history3.append(np.max(np.abs(eigenvals)))
        
        # Reset every 100 steps
        if (t + 1) % 100 == 0:
            W3 = np.random.randn(N, N) * 0.1
            reset_count3 += 1
    
    return {
        'none': Result('No Reset', history1, history1[-1]),
        'fixed_200': Result(f'Reset every {reset_interval}', history2, history2[-1]),
        'fixed_100': Result('Reset every 100', history3, history3[-1]),
        'resets_200': reset_count,
        'resets_100': reset_count3
    }


def main():
    print("=" * 60)
    print("Cicada Protocol - Demonstrating Reset Effect")
    print("=" * 60)
    
    # Config
    N = 200
    steps = 1000
    learning_rate = 0.01
    
    print(f"\nConfig: N={N}, steps={steps}, lr={learning_rate}")
    print("\nRunning experiment...")
    
    results = run_experiment(N=N, steps=steps, learning_rate=learning_rate)
    
    # Calculate average lambda (excluding spikes)
    avg_none = np.mean(results['none'].history[:300])  # Before explosion
    avg_fixed = np.mean(results['fixed_200'].history)
    
    print("\n" + "-" * 60)
    print("Results:")
    print(f"  No Reset (early avg): {avg_none:.3f}")
    print(f"  No Reset (final):    {results['none'].final_lambda:.3f}")
    print(f"  Reset every 200 (final): {results['fixed_200'].final_lambda:.3f}")
    print(f"  Reset every 100 (final): {results['fixed_100'].final_lambda:.3f}")
    print(f"\n  Reset count (200 interval): {results['resets_200']}")
    print(f"  Reset count (100 interval): {results['resets_100']}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Full evolution
    ax1.plot(results['none'].history, 'r-', label='No Reset', linewidth=1.5, alpha=0.8)
    ax1.plot(results['fixed_200'].history, 'b-', label='Reset every 200', linewidth=1.5)
    ax1.plot(results['fixed_100'].history, 'g-', label='Reset every 100', linewidth=1.5)
    ax1.axhline(y=2.0, color='orange', linestyle='--', label='Threshold', alpha=0.7)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Spectral Radius (lambda_max)')
    ax1.set_title('Spectral Radius Evolution Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final comparison
    strategies = ['No Reset', 'Reset 200', 'Reset 100']
    finals = [results['none'].final_lambda, 
              results['fixed_200'].final_lambda,
              results['fixed_100'].final_lambda]
    colors = ['red', 'blue', 'green']
    
    bars = ax2.bar(strategies, finals, color=colors, alpha=0.8, edgecolor='black')
    ax2.axhline(y=2.0, color='orange', linestyle='--', label='Threshold', alpha=0.7)
    ax2.set_ylabel('Final Spectral Radius')
    ax2.set_title('Final Lambda Comparison')
    ax2.legend()
    
    for bar, val in zip(bars, finals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('cicada_demo.png', dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: cicada_demo.png")
    
    # Key insight
    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("=" * 60)
    improvement = results['none'].final_lambda - results['fixed_200'].final_lambda
    if improvement > 0:
        print(f"Reset reduces final lambda by {improvement:.2f} ({improvement/results['none'].final_lambda*100:.1f}%)")
    else:
        print(f"Reset doesn't help in this configuration.")
        print("Try adjusting: lower learning rate or shorter steps")
    print("=" * 60)


if __name__ == "__main__":
    main()
