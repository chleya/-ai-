#!/usr/bin/env python3
"""
Cicada Protocol - Production-Ready Implementation
=================================================

A Python library for maintaining long-term stability in distributed
consensus systems through periodic system reset.

Based on the research paper:
"Cicada Protocol: Long-term Stability for Edge Computing Consensus"

Author: Chen Leiyang
Email: chleiyang@example.com
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Chen Leiyang"

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
from abc import ABC, abstractmethod
import json


# =============================================================================
# Types and Data Classes
# =============================================================================

@dataclass
class ExperimentResult:
    """Result of a cicada protocol experiment."""
    strategy: str
    lambda_max_history: List[float]
    survival_rate: float
    reset_count: int
    final_lambda: float
    config: Dict = field(default_factory=dict)


@dataclass
class ProtocolConfig:
    """Configuration for cicada protocol."""
    N: int = 200                      # System size
    reset_interval: int = 300         # Reset interval in steps
    learning_rate: float = 0.001     # Learning rate
    noise_level: float = 0.5          # Noise level
    seed: Optional[int] = None        # Random seed


# =============================================================================
# Reset Strategies
# =============================================================================

class ResetStrategy(ABC):
    """Abstract base class for reset strategies."""
    
    @abstractmethod
    def should_reset(self, step: int, history: List[float], 
                    config: ProtocolConfig) -> bool:
        """Determine if reset should be triggered."""
        pass


class FixedIntervalStrategy(ResetStrategy):
    """Reset at fixed intervals."""
    
    def should_reset(self, step: int, history: List[float],
                    config: ProtocolConfig) -> bool:
        return (step + 1) % config.reset_interval == 0


class EventTriggeredStrategy(ResetStrategy):
    """Reset when jitter exceeds threshold."""
    
    def __init__(self, alpha: float = 1.5, window: int = 20):
        self.alpha = alpha
        self.window = window
    
    def should_reset(self, step: int, history: List[float],
                    config: ProtocolConfig) -> bool:
        if len(history) < self.window + 1:
            return False
        
        current = history[-1]
        mean = np.mean(history[-self.window:])
        return current > self.alpha * mean


class AdaptiveStrategy(ResetStrategy):
    """Adaptive reset based on spectral radius."""
    
    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold
    
    def should_reset(self, step: int, history: List[float],
                    config: ProtocolConfig) -> bool:
        if not history:
            return False
        return history[-1] > self.threshold


# =============================================================================
# Core Protocol
# =============================================================================

class CicadaProtocol:
    """
    Cicada Protocol: Periodic reset for consensus stability.
    
    A framework for maintaining long-term stability in distributed
    consensus systems through periodic system reset.
    
    Parameters
    ----------
    N : int
        System size (number of nodes/neurons).
    reset_interval : int
        Reset interval in steps.
    learning_rate : float
        Learning rate for weight updates.
    seed : int, optional
        Random seed for reproducibility.
    
    Examples
    --------
    >>> from cicada import CicadaProtocol
    >>> protocol = CicadaProtocol(N=200, reset_interval=300)
    >>> W, s = protocol.evolve(steps=800)
    >>> stats = protocol.analyze()
    >>> print(f"Survival rate: {stats['survival_rate']:.1%}")
    """
    
    def __init__(
        self,
        N: int = 200,
        reset_interval: int = 300,
        learning_rate: float = 0.001,
        noise_level: float = 0.5,
        seed: Optional[int] = None,
        strategy: Optional[ResetStrategy] = None
    ):
        self.config = ProtocolConfig(
            N=N,
            reset_interval=reset_interval,
            learning_rate=learning_rate,
            noise_level=noise_level,
            seed=seed
        )
        self.strategy = strategy or FixedIntervalStrategy()
        
        # Initialize state
        self.W: Optional[np.ndarray] = None
        self.s: Optional[np.ndarray] = None
        self.lambda_history: List[float] = []
        self.reset_count: int = 0
        
        self._initialize()
    
    def _initialize(self):
        """Initialize weight matrix and state."""
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        # Xavier initialization
        self.W = np.random.randn(self.config.N, self.config.N) * np.sqrt(
            2.0 / self.config.N
        )
        self.s = np.random.randn(self.config.N)
        self.lambda_history = []
        self.reset_count = 0
    
    def _step(self, step: int) -> float:
        """Execute one evolution step."""
        # Generate input with noise
        s = np.random.randn(self.config.N)
        s = s / (np.linalg.norm(s) + 1e-6)
        
        # Hebbian weight update
        self.W += self.config.learning_rate * np.outer(s, s)
        
        # Record spectral radius
        eigenvals = np.linalg.eigvalsh(self.W)
        lambda_max = np.max(np.abs(eigenvals))
        self.lambda_history.append(lambda_max)
        
        return lambda_max
    
    def _reset(self):
        """Reset weight matrix and state."""
        self.W = np.random.randn(self.config.N, self.config.N) * np.sqrt(
            2.0 / self.config.N
        )
        self.s = np.random.randn(self.config.N)
        self.reset_count += 1
    
    def evolve(self, steps: int, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run protocol evolution.
        
        Parameters
        ----------
        steps : int
            Number of evolution steps.
        verbose : bool
            Print progress.
        
        Returns
        -------
        tuple
            (final_weight_matrix, final_state)
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        for step in range(steps):
            lambda_max = self._step(step)
            
            if verbose and (step + 1) % 100 == 0:
                print(f"Step {step + 1}/{steps}: λ_max = {lambda_max:.4f}")
            
            # Check reset condition
            if self.strategy.should_reset(step, self.lambda_history, self.config):
                self._reset()
        
        return self.W, self.s
    
    def analyze(self) -> Dict:
        """
        Analyze experiment results.
        
        Returns
        -------
        dict
            Analysis results including survival rate, spectral metrics.
        """
        if not self.lambda_history:
            return {"error": "No data. Run evolve() first."}
        
        healthy_threshold = 1.8
        healthy_steps = sum(1 for l in self.lambda_history if l < healthy_threshold)
        survival_rate = healthy_steps / len(self.lambda_history)
        
        eigenvals = np.linalg.eigvalsh(self.W)
        
        return {
            "survival_rate": survival_rate,
            "final_lambda": self.lambda_history[-1],
            "mean_lambda": np.mean(self.lambda_history),
            "max_lambda": np.max(self.lambda_history),
            "reset_count": self.reset_count,
            "spectral_ratio": eigenvals[-1] / (abs(eigenvals[0]) + 1e-6),
        }
    
    def run_experiment(
        self,
        steps: int = 800,
        trials: int = 10,
        verbose: bool = False
    ) -> ExperimentResult:
        """
        Run multiple trials.
        
        Parameters
        ----------
        steps : int
            Steps per trial.
        trials : int
            Number of trials.
        verbose : bool
            Print progress.
        
        Returns
        -------
        ExperimentResult
            Aggregated results.
        """
        all_survival = []
        all_final = []
        
        for trial in range(trials):
            if verbose:
                print(f"Trial {trial + 1}/{trials}")
            
            # Reset state for each trial
            self._initialize()
            self.evolve(steps)
            
            stats = self.analyze()
            all_survival.append(stats["survival_rate"])
            all_final.append(stats["final_lambda"])
        
        return ExperimentResult(
            strategy=self.strategy.__class__.__name__,
            lambda_max_history=self.lambda_history,
            survival_rate=np.mean(all_survival),
            reset_count=self.reset_count,
            final_lambda=np.mean(all_final),
            config={
                "N": self.config.N,
                "reset_interval": self.config.reset_interval,
                "learning_rate": self.config.learning_rate,
            }
        )
    
    def plot_history(self, save_path: Optional[str] = None):
        """
        Plot spectral radius history.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save figure.
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.lambda_history, linewidth=1.5)
        plt.axhline(y=1.8, color='r', linestyle='--', label='Threshold (1.8)')
        plt.xlabel('Steps')
        plt.ylabel('Spectral Radius (λ_max)')
        plt.title(f'Spectral Radius Evolution - {self.strategy.__class__.__name__}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


# =============================================================================
# Convenience Functions
# =============================================================================

def compare_strategies(
    N: int = 200,
    steps: int = 800,
    trials: int = 5
) -> Dict[str, ExperimentResult]:
    """
    Compare different reset strategies.
    
    Parameters
    ----------
    N : int
        System size.
    steps : int
        Evolution steps.
    trials : int
        Trials per strategy.
    
    Returns
    -------
    dict
        Results for each strategy.
    """
    strategies = {
        "FixedInterval": FixedIntervalStrategy(),
        "EventTriggered": EventTriggeredStrategy(alpha=1.5),
        "Adaptive": AdaptiveStrategy(threshold=2.0),
    }
    
    results = {}
    for name, strategy in strategies.items():
        print(f"\nRunning {name}...")
        protocol = CicadaProtocol(N=N, strategy=strategy)
        result = protocol.run_experiment(steps=steps, trials=trials)
        results[name] = result
        
        print(f"  Survival: {result.survival_rate:.1%}")
        print(f"  Final λ: {result.final_lambda:.3f}")
        print(f"  Resets: {result.reset_count}")
    
    return results


def quick_demo():
    """Run a quick demonstration."""
    print("=" * 60)
    print("Cicada Protocol - Quick Demo")
    print("=" * 60)
    
    # Create protocol
    protocol = CicadaProtocol(N=200, reset_interval=300, seed=42)
    
    # Run evolution
    print("\nRunning evolution (800 steps)...")
    W, s = protocol.evolve(steps=800)
    
    # Analyze
    stats = protocol.analyze()
    
    print("\nResults:")
    print(f"  Survival rate: {stats['survival_rate']:.1%}")
    print(f"  Final λ_max: {stats['final_lambda']:.4f}")
    print(f"  Reset count: {stats['reset_count']}")
    
    # Plot
    print("\nGenerating plot...")
    protocol.plot_history('cicada_demo.png')
    
    print("\nDone! Plot saved to cicada_demo.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    quick_demo()
