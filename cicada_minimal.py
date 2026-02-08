#!/usr/bin/env python3
"""
Cicada Protocol - Minimal Working Prototype
==========================================
Demonstrates how periodic reset prevents spectral radius explosion
in Hebbian weight evolution.

Core mechanism:
- Weight matrix W evolves via Hebbian learning: W += eta*s*s^T
- Periodic reset of W prevents spectral radius growth
- Shows how "cicada reset" maintains long-term stability

Author: Chen Leiyang
Date: 2026-02-08
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import warnings
warnings.filterwarnings('ignore')

# Use non-interactive backend
plt.switch_backend('Agg')


@dataclass
class ExperimentResult:
    """Experiment result data class"""
    strategy: str
    lambda_max_history: List[float]
    survival_rate: float
    reset_count: int
    final_lambda: float


class SimpleCicada:
    """
    Simplified Cicada Protocol implementation
    
    Core parameters:
        N: System size (number of nodes)
        reset_interval: Reset interval in steps
        learning_rate: Learning rate
    """
    
    def __init__(self, N: int = 200, reset_interval: int = 200,
                 learning_rate: float = 0.01):
        self.N = N
        self.reset_interval = reset_interval
        self.learning_rate = learning_rate
        
        # Initialize weight matrix (Xavier initialization)
        self.W = np.random.randn(N, N) * np.sqrt(2.0 / N)
        
        # History
        self.lambda_max_history: List[float] = []
        
    def step(self) -> float:
        """Execute one evolution step"""
        # Generate input
        s = np.random.randn(self.N)
        s = s / (np.linalg.norm(s) + 1e-6)
        
        # Hebbian weight update
        self.W += self.learning_rate * np.outer(s, s)
        
        # Spectral radius
        eigenvals = np.linalg.eigvalsh(self.W)
        lambda_max = np.max(np.abs(eigenvals))
        
        self.lambda_max_history.append(lambda_max)
        return lambda_max
    
    def reset(self):
        """Cicada: reset weight matrix"""
        self.W = np.random.randn(self.N, self.N) * np.sqrt(2.0 / self.N)
    
    def run(self, steps: int, strategy: str = 'fixed') -> ExperimentResult:
        """
        Run experiment
        
        Args:
            steps: Total steps
            strategy: 'none', 'fixed', or 'event'
        """
        reset_count = 0
        jitters: List[float] = []
        
        for t in range(steps):
            lambda_max = self.step()
            
            # Compute jitter for event-triggered strategy
            if len(self.lambda_max_history) > 10:
                jitter = np.std(self.lambda_max_history[-10:])
                jitters.append(jitter)
            
            # Reset decision
            if strategy == 'fixed':
                if (t + 1) % self.reset_interval == 0:
                    self.reset()
                    reset_count += 1
                    
            elif strategy == 'event':
                if len(jitters) > 20:
                    current_jitter = jitters[-1]
                    mean_jitter = np.mean(jitters[-20:])
                    if current_jitter > 1.5 * mean_jitter:
                        self.reset()
                        reset_count += 1
                        jitters = []
        
        # Survival rate (steps where lambda < 1.8)
        healthy_threshold = 1.8
        healthy_steps = sum(1 for l in self.lambda_max_history if l < healthy_threshold)
        survival_rate = healthy_steps / steps
        
        return ExperimentResult(
            strategy=strategy,
            lambda_max_history=self.lambda_max_history,
            survival_rate=survival_rate,
            reset_count=reset_count,
            final_lambda=self.lambda_max_history[-1]
        )


def run_comparison():
    """Run strategy comparison experiment"""
    print("=" * 60)
    print("Cicada Protocol Experiment: Strategy Comparison")
    print("=" * 60)
    
    # Config
    N = 200
    steps = 800
    reset_interval = 200
    
    strategies = ['none', 'fixed', 'event']
    results = {}
    
    for strategy in strategies:
        print(f"\nRunning strategy: {strategy}...")
        
        trials = 3
        all_results = []
        
        for trial in range(trials):
            np.random.seed(trial * 42)
            cicada = SimpleCicada(N=N, reset_interval=reset_interval)
            result = cicada.run(steps, strategy)
            all_results.append(result)
        
        results[strategy] = {
            'survival_rate': np.mean([r.survival_rate for r in all_results]),
            'reset_count': np.mean([r.reset_count for r in all_results]),
            'final_lambda': np.mean([r.final_lambda for r in all_results]),
            'lambda_history': all_results[0].lambda_max_history
        }
        
        print(f"  Survival rate: {results[strategy]['survival_rate']:.1%}")
        print(f"  Reset count: {results[strategy]['reset_count']:.1f}")
        print(f"  Final lambda: {results[strategy]['final_lambda']:.3f}")
    
    # Comparison table
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Strategy':<12} {'Survival':<12} {'Resets':<10} {'Final Lambda':<12}")
    print("-" * 60)
    
    for strategy in strategies:
        r = results[strategy]
        print(f"{strategy:<12} {r['survival_rate']:<12.1%} {r['reset_count']:<10.1f} {r['final_lambda']:<12.3f}")
    
    return results


def plot_results(results: dict, save_path: str = 'cicada_results.png'):
    """Visualize experiment results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Spectral radius evolution
    ax1 = axes[0]
    colors = {'none': 'red', 'fixed': 'blue', 'event': 'green'}
    labels = {'none': 'No Reset', 'fixed': 'Fixed (200 steps)', 'event': 'Event-triggered'}
    
    for strategy in ['none', 'fixed', 'event']:
        history = results[strategy]['lambda_history']
        ax1.plot(history, color=colors[strategy], label=labels[strategy], linewidth=1.5)
    
    ax1.axhline(y=1.8, color='orange', linestyle='--', linewidth=2, label='Critical (1.8)')
    ax1.set_xlabel('Steps', fontsize=12)
    ax1.set_ylabel('Spectral Radius (lambda_max)', fontsize=12)
    ax1.set_title('Spectral Radius Evolution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 5)
    
    # Plot 2: Comparison bar chart
    ax2 = axes[1]
    strategies = ['none', 'fixed', 'event']
    survival_rates = [results[s]['survival_rate'] for s in strategies]
    colors_bar = [colors[s] for s in strategies]
    
    bars = ax2.bar(strategies, survival_rates, color=colors_bar, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Survival Rate', fontsize=12)
    ax2.set_title('Survival Rate Comparison', fontsize=14)
    ax2.set_ylim(0, 1.1)
    
    # Add labels
    for bar, rate in zip(bars, survival_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.1%}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {save_path}")
    plt.close()


def main():
    """Main function"""
    # Run experiment
    results = run_comparison()
    
    # Visualize
    plot_results(results, 'cicada_results.png')
    
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
