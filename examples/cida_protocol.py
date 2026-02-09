#!/usr/bin/env python3
"""
Cida Protocol - Core Implementation
==============================
Based on: Attractor + H-Theorem + RMT Framework
"""

import numpy as np
import matplotlib.pyplot as plt
import json


def hebbian_update(W, s, eta):
    """Hebbian Update: W += eta * s * s^T"""
    return W + eta * np.outer(s, s)


def compute_lambda_max_approx(W, iterations=10):
    """Power iteration approximation for λ_max"""
    v = np.random.rand(W.shape[0])
    for _ in range(iterations):
        v = W @ v
        v /= np.linalg.norm(v)
    return np.abs(v @ (W @ v))  # Rayleigh quotient


def approximate_entropy(W, epsilon=1e-6):
    """Approximate von Neumann entropy: log det(W + eps*I)"""
    N = W.shape[0]
    try:
        return np.log(np.abs(np.linalg.det(W + np.eye(N) * epsilon))) / N
    except:
        return 0.0


def cida_protocol_simulation(N=787, steps=500, eta=0.01, theta=1.2, alpha=1.0, reset_prob=0.05, seed=42):
    """Cida Protocol Simulation"""
    np.random.seed(seed)
    
    # Initial random matrix (RMT-style)
    W = np.random.normal(0, 1/np.sqrt(N), (N, N))
    
    lambda_history = []
    entropy_history = []
    reset_count = 0
    
    print(f"Simulating: N={N}, steps={steps}, eta={eta}, theta={theta}, alpha={alpha}")
    print("-" * 50)
    
    for t in range(steps):
        # Random input pattern
        s = np.random.normal(0, 1, N)
        
        # Hebbian update
        W = hebbian_update(W, s, eta)
        
        # Compute metrics
        lambda_max = compute_lambda_max_approx(W)
        entropy = approximate_entropy(W)
        
        lambda_history.append(lambda_max)
        entropy_history.append(entropy)
        
        # Reset condition: λ_max > θ × α
        threshold = theta * alpha
        
        if lambda_max > threshold:
            # Partial reset: inject negative entropy
            mask = np.random.rand(N, N) < reset_prob
            W[mask] = np.random.normal(0, 1/np.sqrt(N))
            reset_count += 1
            print(f"Step {t}: Reset! λ={lambda_max:.2f} > {threshold:.2f}")
    
    return {
        'lambda_history': lambda_history,
        'entropy_history': entropy_history,
        'reset_count': reset_count
    }


def run_scaling_experiment():
    """Verify scaling law: λ_max ∝ N^β"""
    print("\n" + "=" * 70)
    print("SCALING LAW VERIFICATION")
    print("=" * 70)
    
    N_values = [100, 200, 400, 600, 787, 1000]
    lambda_means = []
    
    for N in N_values:
        result = cida_protocol_simulation(N=N, steps=200, eta=0.01)
        lambda_means.append(np.mean(result['lambda_history']))
        print(f"N={N}: <λ_max> = {lambda_means[-1]:.4f}")
    
    # Fit power law
    log_N = np.log(N_values)
    log_lambda = np.log(lambda_means)
    beta, log_a = np.polyfit(log_N, log_lambda, 1)
    a = np.exp(log_a)
    
    print("\n" + "-" * 50)
    print(f"Power Law: λ = {a:.4f} × N^{beta:.4f}")
    print(f"Expected: λ ∝ N^0.72")
    print(f"Difference: {abs(beta - 0.72)/0.72 * 100:.1f}%")
    
    return {'N_values': N_values, 'lambda_means': lambda_means, 'a': a, 'beta': beta}


def visualize_results():
    """Generate visualization"""
    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    
    # Run simulations
    result = cida_protocol_simulation(N=787, steps=300, eta=0.01, theta=1.2, alpha=1.0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: λ_max over time
    ax = axes[0, 0]
    ax.plot(result['lambda_history'], 'b-', lw=1)
    ax.axhline(y=1.2, color='red', ls='--', label='Threshold (θ=1.2)')
    ax.set_xlabel('Steps')
    ax.set_ylabel('λ_max')
    ax.set_title('λ_max over Time (with Resets)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Entropy over time
    ax = axes[0, 1]
    ax.plot(result['entropy_history'], 'g-', lw=1)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Approximate Entropy')
    ax.set_title('Entropy over Time')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Scaling law
    ax = axes[1, 0]
    scaling = run_scaling_experiment()
    ax.loglog(scaling['N_values'], scaling['lambda_means'], 'bo-', lw=2, markersize=10)
    N_fit = np.logspace(2, 3, 100)
    ax.loglog(N_fit, scaling['a'] * N_fit ** scaling['beta'], 'r--', lw=2,
              label=f'Fit: λ ∝ N^{scaling["beta"]:.2f}')
    ax.set_xlabel('N (log scale)')
    ax.set_ylabel('λ_max (log scale)')
    ax.set_title('Scaling Law Verification')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║           CIDA PROTOCOL SIMULATION SUMMARY           ║
╠══════════════════════════════════════════════════════════════╣
║                                                          ║
║  Parameters:                                             ║
║    N = 787 (phase transition point)                      ║
║    Steps = 300                                         ║
║    Learning rate = 0.01                                 ║
║    Reset threshold = 1.2                                 ║
║                                                          ║
║  Results:                                               ║
║    Resets triggered: {result['reset_count']}                                ║
║    Final λ_max: {result['lambda_history'][-1]:.4f}                              ║
║    Final entropy: {result['entropy_history'][-1]:.4f}                            ║
║                                                          ║
╠══════════════════════════════════════════════════════════════╣
║                                                          ║
║  Scaling Law:                                           ║
║    λ ∝ N^{scaling['beta']:.3f}                                     ║
║    (Expected: N^0.72)                                      ║
║                                                          ║
║  Core Insight:                                          ║
║    Reset combats continuous entropy increase.               ║
║    H-Theorem: dH/dt > 0 → chaos                        ║
║    Reset: injects negative entropy.                      ║
║                                                          ║
╚══════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.02, 0.98, summary, transform=ax.transAxes,
            fontsize=10, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('visualization/cida_simulation.png', dpi=200, bbox_inches='tight')
    print("\nSaved: visualization/cida_simulation.png")


if __name__ == "__main__":
    print("=" * 70)
    print("CIDA PROTOCOL SIMULATION")
    print("Framework: Attractor + H-Theorem + RMT")
    print("=" * 70)
    
    visualize_results()
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE!")
    print("=" * 70)
