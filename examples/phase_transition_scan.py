#!/usr/bin/env python3
"""
Phase Transition Scan for Cicada Protocol
======================================
Scan N from 100 to 2000 to verify Nc ≈ 900.

Key question: At what system size N does the survival rate drop dramatically?
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import json


def cicada_protocol(N=200, lr=0.001, steps=500, reset_interval=300, seed=42):
    """Run protocol."""
    np.random.seed(seed)
    W = np.random.randn(N, N) * 0.01
    history = []
    
    for t in range(steps):
        s = np.random.randn(N)
        W += lr * np.outer(s, s)
        
        eigenvals = np.linalg.eigvalsh(W)
        lambda_max = np.max(np.abs(eigenvals))
        history.append(lambda_max)
        
        if reset_interval and (t + 1) % reset_interval == 0:
            W = np.random.randn(N, N) * 0.01
    
    return history


def calculate_survival_rate(history, threshold=1.8):
    """Calculate proportion of steps where λ_max < threshold."""
    healthy = sum(1 for x in history if x < threshold)
    return healthy / len(history)


def run_phase_transition_scan():
    """Scan N values to find phase transition."""
    
    print("=" * 70)
    print("Cicada Protocol - Phase Transition Scan")
    print("=" * 70)
    print()
    
    N_values = [50, 100, 200, 400, 600, 800, 900, 1000, 1200, 1500, 2000]
    results = []
    
    print("Scanning N values...")
    print("-" * 70)
    print(f"{'N':>6} {'Survival':>10} {'Final λ':>10} {'Max λ':>10} {'Phase':>15}")
    print("-" * 70)
    
    for N in N_values:
        # Run multiple trials
        survival_rates = []
        final_lambdas = []
        max_lambdas = []
        
        for trial in range(5):
            history = cicada_protocol(N=N, lr=0.001, steps=500)
            
            survival = calculate_survival_rate(history)
            survival_rates.append(survival)
            final_lambdas.append(history[-1])
            max_lambdas.append(max(history))
        
        mean_survival = np.mean(survival_rates)
        mean_final = np.mean(final_lambdas)
        mean_max = np.mean(max_lambdas)
        
        # Determine phase
        if mean_survival > 0.8:
            phase = "I (stable)"
        elif mean_survival > 0.4:
            phase = "Transition"
        else:
            phase = "II (unstable)"
        
        print(f"{N:>6} {mean_survival:>10.1%} {mean_final:>10.3f} {mean_max:>10.3f} {phase:>15}")
        
        results.append({
            'N': N,
            'survival_rate': mean_survival,
            'final_lambda': mean_final,
            'max_lambda': mean_max,
            'phase': phase
        })
    
    print("-" * 70)
    print()
    
    # Find critical N (phase transition point)
    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]
        
        if prev['survival_rate'] > 0.5 and curr['survival_rate'] < 0.5:
            print(f"PHASE TRANSITION DETECTED!")
            print(f"  Between N={prev['N']} (survival={prev['survival_rate']:.1%})")
            print(f"  and N={curr['N']} (survival={curr['survival_rate']:.1%})")
            print(f"  Critical Nc ≈ {curr['N']}")
            break
    
    # Plot
    plot_results(results)
    
    # Save results
    with open('results/phase_transition_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print()
    print("Results saved to results/phase_transition_results.json")
    
    return results


def plot_results(results):
    """Plot phase transition curve."""
    
    Ns = [r['N'] for r in results]
    survivals = [r['survival_rate'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Survival rate vs N
    ax1.plot(Ns, survivals, 'bo-', lw=2, markersize=8)
    ax1.axhline(y=0.5, color='red', ls='--', label='50% threshold')
    ax1.axvline(x=900, color='green', ls='--', alpha=0.7, label='Nc=900')
    ax1.set_xlabel('System Size N')
    ax1.set_ylabel('Survival Rate')
    ax1.set_title('Phase Transition: Survival Rate vs N')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 2000])
    ax1.set_ylim([0, 1.05])
    
    # Right: λ_max vs N
    finals = [r['final_lambda'] for r in results]
    maxes = [r['max_lambda'] for r in results]
    
    ax2.plot(Ns, finals, 'go-', lw=2, label='Final λ_max')
    ax2.plot(Ns, maxes, 'r^--', lw=2, label='Max λ_max')
    ax2.axhline(y=1.8, color='orange', ls=':', label='Threshold (1.8)')
    ax2.set_xlabel('System Size N')
    ax2.set_ylabel('Spectral Radius λ_max')
    ax2.set_title('λ_max vs N')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 2000])
    
    plt.tight_layout()
    plt.savefig('visualization/phase_transition.png', dpi=150, bbox_inches='tight')
    print("Plot saved to visualization/phase_transition.png")


if __name__ == "__main__":
    run_phase_transition_scan()
