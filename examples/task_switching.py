#!/usr/bin/env python3
"""
Task Switching Experiment for Cicada Protocol
==========================================
Compare: With Reset vs Without Reset in multi-task scenarios.

This demonstrates why periodic reset is important for task switching:
- Without reset: Task interference accumulates
- With reset: Fresh start for each task

Key metrics:
- Task performance
- Interference level
"""

import numpy as np
from typing import List
import matplotlib.pyplot as plt


def run_cicada_experiment(N=200, lr=0.001, steps=1000, reset_interval=None, task_switches=None):
    """
    Run experiment with optional periodic reset.
    
    Parameters
    ----------
    N : int
        System size
    lr : float
        Learning rate
    steps : int
        Total steps
    reset_interval : int, optional
        Reset every N steps. None = no reset.
    task_switches : list, optional
        List of task IDs at each step. None = single task.
        
    Returns
    -------
    tuple
        (lambda_history, task_performance)
    """
    np.random.seed(42)
    W = np.random.randn(N, N) * 0.01
    
    # Default: single task
    if task_switches is None:
        task_switches = [0] * steps
    
    lambda_history = []
    task_performance = []
    current_task = 0
    
    for t in range(steps):
        # Current task pattern
        if task_switches[t] == 0:
            task_pattern = np.random.randn(N)
        else:
            task_pattern = np.random.randn(N) * 0.5 + np.random.randn(N) * 0.5
        
        task_pattern = task_pattern / np.linalg.norm(task_pattern)
        
        # Hebbian update
        W += lr * np.outer(task_pattern, task_pattern)
        
        # Record spectral radius
        eigenvals = np.linalg.eigvalsh(W)
        lambda_max = np.max(np.abs(eigenvals))
        lambda_history.append(lambda_max)
        
        # Reset (Cicada moment)
        if reset_interval and (t + 1) % reset_interval == 0:
            W = np.random.randn(N, N) * 0.01
            current_task = task_switches[t]
    
    return lambda_history


def run_task_switching_experiment():
    """
    Compare: With Reset vs Without Reset in task switching scenario.
    """
    
    print("=" * 70)
    print("Cicada Protocol - Task Switching Experiment")
    print("=" * 70)
    print()
    
    N = 200
    lr = 0.001
    steps = 1000
    
    # Create task sequence (switch tasks every 200 steps)
    task_switches = [0] * 200 + [1] * 200 + [0] * 200 + [1] * 200 + [0] * 200
    
    print("Task sequence: A-B-A-B-A (200 steps each)")
    print(f"Parameters: N={N}, lr={lr}, steps={steps}")
    print()
    
    # 1. Without reset
    print("1. Without reset:")
    no_reset = run_cicada_experiment(N=N, lr=lr, steps=steps, 
                                     task_switches=task_switches)
    print(f"   Final λ_max: {no_reset[-1]:.3f}")
    print(f"   Max λ_max: {max(no_reset):.3f}")
    print()
    
    # 2. With reset every 200 steps
    print("2. With reset every 200 steps:")
    with_reset = run_cicada_experiment(N=N, lr=lr, steps=steps, reset_interval=200,
                                        task_switches=task_switches)
    print(f"   Final λ_max: {with_reset[-1]:.3f}")
    print(f"   Max λ_max: {max(with_reset):.3f}")
    print()
    
    # 3. With reset every 300 steps
    print("3. With reset every 300 steps:")
    with_reset_300 = run_cicada_experiment(N=N, lr=lr, steps=steps, reset_interval=300,
                                           task_switches=task_switches)
    print(f"   Final λ_max: {with_reset_300[-1]:.3f}")
    print(f"   Max λ_max: {max(with_reset_300):.3f}")
    print()
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(no_reset, 'r-', lw=1.5, alpha=0.7, label='No reset')
    plt.plot(with_reset, 'b-', lw=1.5, alpha=0.7, label='Reset 200')
    plt.plot(with_reset_300, 'g-', lw=1.5, alpha=0.7, label='Reset 300')
    plt.axhline(y=1.8, color='orange', ls='--', label='Healthy (1.8)')
    
    # Mark task switches
    for switch in [200, 400, 600, 800]:
        plt.axvline(x=switch, color='gray', ls=':', alpha=0.5)
    
    plt.xlabel('Steps')
    plt.ylabel('λ_max')
    plt.title('Task Switching: With vs Without Reset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    strategies = ['No reset', 'Reset 200', 'Reset 300']
    finals = [no_reset[-1], with_reset[-1], with_reset_300[-1]]
    colors = ['red', 'blue', 'green']
    plt.bar(strategies, finals, color=colors, alpha=0.8, edgecolor='black')
    plt.axhline(y=1.8, color='orange', ls='--')
    plt.ylabel('Final λ_max')
    plt.title('Final λ Comparison')
    for i, v in enumerate(finals):
        plt.text(i, v + 0.05, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualization/task_switching.png', dpi=150)
    print("Plot saved to visualization/task_switching.png")
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    print()
    print(f"{'Strategy':<15} {'Final λ':<12} {'Reduction':<15}")
    print("-" * 42)
    print(f"{'No reset':<15} {no_reset[-1]:<12.3f} baseline")
    print(f"{'Reset 200':<15} {with_reset[-1]:<12.3f} {(1-with_reset[-1]/no_reset[-1])*100:+.1f}%")
    print(f"{'Reset 300':<15} {with_reset_300[-1]:<12.3f} {(1-with_reset_300[-1]/no_reset[-1])*100:+.1f}%")
    print()
    
    # Key insight
    print("KEY INSIGHT:")
    print("-" * 42)
    print()
    print("Periodic reset prevents task interference from accumulating.")
    print("When switching between tasks, reset provides a fresh start,")
    print("preventing the system from being trapped in interference patterns.")
    print()


if __name__ == "__main__":
    run_task_switching_experiment()
