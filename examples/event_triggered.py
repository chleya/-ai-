#!/usr/bin/env python3
"""
Event-Triggered Strategy for Cicada Protocol
==========================================
Adaptive reset using ABSOLUTE THRESHOLD.

This is more reliable for Hebbian learning where λ_max grows monotonically.
Reset when: λ_max(t) > threshold
"""

import numpy as np
from typing import List
import matplotlib.pyplot as plt


class EventTriggeredStrategy:
    """
    Event-triggered reset with absolute threshold.
    
    Reset triggered when:
        λ_max(t) > threshold
    
    Parameters
    ----------
    threshold : float
        Absolute threshold for λ_max. Default 1.8 (healthy boundary).
    min_interval : int
        Minimum steps between resets.
    """
    
    def __init__(self, threshold: float = 1.8, min_interval: int = 50):
        self.threshold = threshold
        self.min_interval = min_interval
        self.reset_count = 0
        self.last_reset = -min_interval
    
    def should_reset(self, step: int, history: List[float]) -> bool:
        if step - self.last_reset < self.min_interval:
            return False
        if not history:
            return False
        
        current = history[-1]
        
        if current > self.threshold:
            self.last_reset = step
            self.reset_count += 1
            return True
        return False


def run_experiment():
    """Run experiment with event-triggered strategy."""
    
    print("=" * 70)
    print("Cicada Protocol - Event-Triggered Strategy")
    print("=" * 70)
    print()
    
    N = 200
    lr = 0.001  # Demo parameter
    steps = 1000
    threshold = 1.8
    
    # 1. No reset
    print("1. No reset (baseline):")
    np.random.seed(42)
    W = np.random.randn(N, N) * 0.01
    no_reset = []
    for t in range(steps):
        s = np.random.randn(N)
        W += lr * np.outer(s, s)
        eigenvals = np.linalg.eigvalsh(W)
        no_reset.append(np.max(np.abs(eigenvals)))
    print(f"   Final λ_max: {no_reset[-1]:.3f}")
    print(f"   Steps above threshold: {sum(1 for x in no_reset if x > threshold)}")
    print()
    
    # 2. Fixed interval
    print("2. Fixed interval (300 steps):")
    np.random.seed(42)
    W = np.random.randn(N, N) * 0.01
    fixed = []
    fixed_resets = 0
    for t in range(steps):
        s = np.random.randn(N)
        W += lr * np.outer(s, s)
        eigenvals = np.linalg.eigvalsh(W)
        fixed.append(np.max(np.abs(eigenvals)))
        if (t + 1) % 300 == 0:
            W = np.random.randn(N, N) * 0.01
            fixed_resets += 1
    print(f"   Final λ_max: {fixed[-1]:.3f}")
    print(f"   Resets: {fixed_resets}")
    print()
    
    # 3. Event-triggered (α=1.6 adaptation)
    print(f"3. Event-triggered (threshold={threshold}, α adaptation):")
    np.random.seed(42)
    W = np.random.randn(N, N) * 0.01
    event = []
    event_strat = EventTriggeredStrategy(threshold=threshold, min_interval=50)
    for t in range(steps):
        s = np.random.randn(N)
        W += lr * np.outer(s, s)
        eigenvals = np.linalg.eigvalsh(W)
        lam = np.max(np.abs(eigenvals))
        event.append(lam)
        event_strat.should_reset(t, event)
    print(f"   Final λ_max: {event[-1]:.3f}")
    print(f"   Resets: {event_strat.reset_count}")
    print()
    
    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(no_reset, 'r-', lw=1.5, alpha=0.7, label='No reset')
    plt.plot(fixed, 'b-', lw=1.5, alpha=0.7, label='Fixed (300)')
    plt.plot(event, 'g-', lw=1.5, alpha=0.7, label='Event-triggered')
    plt.axhline(y=threshold, color='orange', ls='--', label=f'Threshold ({threshold})')
    plt.xlabel('Steps')
    plt.ylabel('λ_max')
    plt.title('Strategy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    strategies = ['No reset', 'Fixed', 'Event']
    finals = [no_reset[-1], fixed[-1], event[-1]]
    colors = ['red', 'blue', 'green']
    plt.bar(strategies, finals, color=colors, alpha=0.8)
    plt.axhline(y=threshold, color='orange', ls='--')
    plt.ylabel('Final λ_max')
    plt.title('Final λ Comparison')
    for i, v in enumerate(finals):
        plt.text(i, v + 0.1, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualization/event_triggered_comparison.png', dpi=150)
    print("Plot saved to visualization/event_triggered_comparison.png")
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Final λ':<12} {'Resets':<10} {'Efficiency':<15}")
    print("-" * 57)
    print(f"{'No reset':<20} {no_reset[-1]:<12.3f} {'0':<10} {'N/A':<15}")
    print(f"{'Fixed (300)':<20} {fixed[-1]:<12.3f} {fixed_resets:<10} {'Fixed schedule':<15}")
    print(f"{'Event (threshold)':<20} {event[-1]:<12.3f} {event_strat.reset_count:<10} {'Adaptive!':<15}")
    print()
    
    if event_strat.reset_count < fixed_resets:
        pct = (1 - event_strat.reset_count/fixed_resets) * 100
        print(f"Event-triggered uses {pct:.0f}% FEWER resets than fixed interval!")
    print()


if __name__ == "__main__":
    run_experiment()
