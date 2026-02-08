#!/usr/bin/env python3
"""
Cicada Protocol - Minimal Working Prototype v2
=============================================
Fixed parameters to properly demonstrate reset effect.

Key fixes:
- Lower learning rate (0.001)
- Shorter reset interval (100 steps)  
- Stronger normalization
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import warnings
warnings.filterwarnings('ignore')
plt.switch_backend('Agg')


@dataclass
class ExperimentResult:
    strategy: str
    lambda_max_history: List[float]
    survival_rate: float
    reset_count: int
    final_lambda: float


class SimpleCicada:
    """Simplified Cicada Protocol with proper parameters"""
    
    def __init__(self, N: int = 200, reset_interval: int = 100,
                 learning_rate: float = 0.001):
        self.N = N
        self.reset_interval = reset_interval
        self.learning_rate = learning_rate
        
        # Initialize
        self.W = np.random.randn(N, N) * 0.1  # Smaller init
        self.lambda_max_history: List[float] = []
        
    def step(self) -> float:
        """One evolution step"""
        s = np.random.randn(self.N)
        s = s / (np.linalg.norm(s) + 1e-6)
        
        # Hebbian update
        self.W += self.learning_rate * np.outer(s, s)
        
        # Normalize to prevent explosion
        if np.linalg.norm(self.W) > 5:
            self.W = self.W / np.linalg.norm(self.W) * 5
        
        # Spectral radius
        eigenvals = np.linalg.eigvalsh(self.W)
        lambda_max = np.max(np.abs(eigenvals))
        
        self.lambda_max_history.append(lambda_max)
        return lambda_max
    
    def reset(self):
        """Reset weight matrix"""
        self.W = np.random.randn(self.N, self.N) * 0.1
    
    def run(self, steps: int, strategy: str = 'fixed') -> ExperimentResult:
        """Run experiment"""
        reset_count = 0
        
        for t in range(steps):
            self.step()
            
            if strategy == 'fixed':
                if (t + 1) % self.reset_interval == 0:
                    self.reset()
                    reset_count += 1
        
        # Survival: lambda < 2.0 (adjusted threshold)
        healthy_threshold = 2.0
        healthy_steps = sum(1 for l in self.lambda_max_history if l < healthy_threshold)
        survival_rate = healthy_steps / steps
        
        return ExperimentResult(
            strategy=strategy,
            lambda_max_history=self.lambda_max_history.copy(),
            survival_rate=survival_rate,
            reset_count=reset_count,
            final_lambda=self.lambda_max_history[-1]
        )


def main():
    print("=" * 60)
    print("Cicada Protocol - Fixed Parameters")
    print("=" * 60)
    
    N = 200
    steps = 800
    reset_interval = 100
    
    strategies = ['none', 'fixed']
    results = {}
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        
        trials = 3
        all_survival = []
        all_final = []
        
        for trial in range(trials):
            np.random.seed(trial * 42)
            cicada = SimpleCicada(N=N, reset_interval=reset_interval)
            result = cicada.run(steps, strategy)
            all_survival.append(result.survival_rate)
            all_final.append(result.final_lambda)
        
        results[strategy] = {
            'survival_rate': np.mean(all_survival),
            'final_lambda': np.mean(all_final),
            'lambda_history': cicada.lambda_max_history
        }
        
        print(f"  Survival: {np.mean(all_survival):.1%}")
        print(f"  Final lambda: {np.mean(all_final):.3f}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot evolution
    for strategy in strategies:
        history = results[strategy]['lambda_history']
        color = 'red' if strategy == 'none' else 'blue'
        label = 'No Reset' if strategy == 'none' else f'Fixed ({reset_interval} steps)'
        ax1.plot(history, color=color, label=label, linewidth=1.5)
    
    ax1.axhline(y=2.0, color='orange', linestyle='--', label='Threshold (2.0)')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Spectral Radius')
    ax1.set_title('Spectral Radius Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot comparison
    surv = [results[s]['survival_rate'] for s in strategies]
    colors = ['red', 'blue']
    bars = ax2.bar(strategies, surv, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Survival Rate')
    ax2.set_title('Survival Rate Comparison')
    ax2.set_ylim(0, 1.1)
    
    for bar, s in zip(bars, surv):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{s:.1%}', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('cicada_results.png', dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: cicada_results.png")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"No Reset survival: {results['none']['survival_rate']:.1%}")
    print(f"Fixed reset survival: {results['fixed']['survival_rate']:.1%}")
    
    improvement = results['fixed']['survival_rate'] - results['none']['survival_rate']
    print(f"\nImprovement: {improvement:+.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
